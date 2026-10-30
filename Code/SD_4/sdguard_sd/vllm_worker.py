"""Isolated vLLM worker for one or two cells of the SD_4 matrix."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .io_utils import JsonlWriter, read_json, write_json
from .inline_guard import aggregate_inline_records
from .text_split import split_completion
from .timing import extract_request_timing, summarize_ms


PINNED_VLLM_BASE = "0.31.0"
EXTENSION_CLASS = "sdguard_sd.vllm_extension.SD4WorkerExtension"


def _rpc_one(llm: Any, method: str, **kwargs: Any) -> Any:
    values = llm.collective_rpc(method, kwargs=kwargs)
    if len(values) != 1:
        raise RuntimeError(
            f"SD_4 requires tensor_parallel_size=1; {method} returned {len(values)} values"
        )
    return values[0]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _completion_output(request_output: Any) -> Any:
    outputs = getattr(request_output, "outputs", None)
    if not outputs or len(outputs) != 1:
        raise RuntimeError("Expected exactly one completion from vLLM")
    return outputs[0]


def _token_ids(request_output: Any) -> list[int]:
    token_ids = getattr(_completion_output(request_output), "token_ids", None)
    if token_ids is None:
        raise RuntimeError("vLLM CompletionOutput has no token_ids")
    return [int(value) for value in token_ids]


def _spec_metrics(request_output: Any) -> dict[str, Any]:
    completion = _completion_output(request_output)
    raw = getattr(completion, "spec_decode_metrics", None)
    if raw is None:
        raw = getattr(request_output, "spec_decode_metrics", None)
    if raw is None:
        # vLLM 0.31's public RequestOutput contract exposes this under
        # request_output.metrics.speculative_decoding.  Keep the two aliases
        # above for local 0.31 builds that surfaced the accumulator directly.
        request_metrics = getattr(request_output, "metrics", None)
        raw = getattr(request_metrics, "speculative_decoding", None)
    if raw is None:
        raise RuntimeError(
            "vLLM did not return per-request speculative metrics; SD_4 requires "
            "per_request_spec_decode_metrics='detailed'"
        )
    payload = _jsonable(raw)
    accepted = payload.get("per_step_accepted")
    drafted = payload.get("per_step_drafted")
    if accepted is None or drafted is None or len(accepted) != len(drafted):
        raise RuntimeError("Detailed per-step speculative metrics are missing")
    proposed_count = int(payload.get("num_draft_tokens", sum(drafted)))
    accepted_count = int(payload.get("num_accepted_draft_tokens", sum(accepted)))
    return {
        **payload,
        "round_count": len(drafted),
        "proposed_token_count": proposed_count,
        "accepted_draft_token_count": accepted_count,
        "acceptance_rate": (
            None if proposed_count == 0 else accepted_count / proposed_count
        ),
    }


def _sampling_params(SamplingParams: Any, settings: dict[str, Any], seed: int, *, max_tokens: int | None = None) -> Any:
    return SamplingParams(
        max_tokens=int(max_tokens or settings["max_new_tokens"]),
        temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]),
        top_k=int(settings["top_k"]),
        min_p=0.0,
        seed=int(seed),
        detokenize=False,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )


def _generate(
    llm: Any,
    SamplingParams: Any,
    record: dict[str, Any],
    settings: dict[str, Any],
    *,
    max_tokens: int | None = None,
) -> tuple[Any, float]:
    params = _sampling_params(
        SamplingParams, settings, int(record["sample_seed"]), max_tokens=max_tokens
    )
    started_ns = time.perf_counter_ns()
    outputs = llm.generate(
        [{"prompt_token_ids": [int(value) for value in record["prompt_token_ids"]]}],
        sampling_params=[params],
        use_tqdm=False,
    )
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one vLLM output, received {len(outputs)}")
    return outputs[0], elapsed_ms


def _posthoc_guard(
    llm: Any,
    *,
    prompt_ids: list[int],
    generated_ids: list[int],
    answer_start: int | None,
) -> tuple[dict[str, Any], float]:
    started_ns = time.perf_counter_ns()
    payload = _rpc_one(
        llm,
        "sd4_run_posthoc_guard",
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        answer_start_generated_index=answer_start,
    )
    rpc_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return payload, rpc_ms


def _guard_record(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_risk_level": payload["prompt_risk_level"],
        "prompt_risk_probability": payload["prompt_risk_probability"],
        "latency_ms": float(payload["latency_ms"]),
        "input_packaging_ms": float(payload.get("input_packaging_ms", 0.0)),
        "input_h2d_ms": float(payload.get("input_h2d_ms", 0.0)),
        "input_preparation_ms": float(payload.get("input_preparation_ms", 0.0)),
        "output_decode_d2h_ms": float(payload.get("output_decode_d2h_ms", 0.0)),
        "scopes": payload["scopes"],
    }


def _attach_inline_consistency(
    inline: dict[str, Any], posthoc: dict[str, Any]
) -> None:
    committed = [
        record
        for record in inline.get("token_records", [])
        if record.get("phase") == "committed"
    ]
    posthoc_tokens = posthoc.get("token_records", [])
    online_labels = [str(record["risk_level"]) for record in committed]
    posthoc_labels = [str(record["risk_level"]) for record in posthoc_tokens]
    mismatch_indices = [
        index
        for index in range(max(len(online_labels), len(posthoc_labels)))
        if index >= len(online_labels)
        or index >= len(posthoc_labels)
        or online_labels[index] != posthoc_labels[index]
    ]

    def project(scopes: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for scope in ("full_trace", "answer_only"):
            value = scopes[scope]
            if not value["available"]:
                result[scope] = None
                continue
            result[scope] = {
                policy: {
                    key: value["decisions"][policy].get(key)
                    for key in (
                        "prediction",
                        "triggered",
                        "trigger_generated_index",
                        "trigger_pair",
                    )
                }
                for policy in ("strict", "loose")
            }
        return result

    decision_match = project(inline["verified_commit"]["scopes"]) == project(
        posthoc["scopes"]
    )
    label_match = not mismatch_indices
    inline["consistency"].update(
        {
            "verified_token_labels_match_posthoc": label_match,
            "verified_scope_decisions_match_posthoc": decision_match,
            "verified_posthoc_match": label_match and decision_match,
            "verified_posthoc_exact_match": label_match and decision_match,
            "verified_posthoc_decision_consistent": decision_match,
            "first_label_mismatch_generated_index": (
                mismatch_indices[0] if mismatch_indices else None
            ),
            "token_label_mismatch_count": len(mismatch_indices),
        }
    )


def _runtime_metadata(torch: Any, installed_vllm: str) -> dict[str, Any]:
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "logical_device": f"cuda:{index}",
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    try:
        topology = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except FileNotFoundError:
        topology = None
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "vllm_version": installed_vllm,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu_count": torch.cuda.device_count(),
        "gpus": devices,
        "placement": {
            "target": "cuda:0",
            "draft": "cuda:0",
            "guard": "cuda:1",
        },
        "peer_access": {
            "cuda:0_to_cuda:1": bool(torch.cuda.can_device_access_peer(0, 1)),
            "cuda:1_to_cuda:0": bool(torch.cuda.can_device_access_peer(1, 0)),
        },
        "nvidia_smi_topology": topology,
        "topology_note": (
            "SYS traverses PCIe plus the inter-NUMA CPU interconnect; it is not NVLink"
            if topology is not None and "SYS" in topology
            else None
        ),
    }


class CellSink:
    def __init__(self, root: Path, run_config: dict[str, Any]) -> None:
        self.root = root
        self.raw = root / "raw"
        self.blind = root / "blind"
        self.private = root / "private"
        for directory in (self.raw, self.blind, self.private):
            directory.mkdir(parents=True, exist_ok=True)
        write_json(root / "run_config.json", run_config)
        self.stack = ExitStack()
        self.generation = self.stack.enter_context(
            JsonlWriter(self.raw / "generation_records.jsonl")
        )
        self.guard_tokens = self.stack.enter_context(
            JsonlWriter(self.raw / "guard_token_records.jsonl")
        )
        self.rounds = self.stack.enter_context(
            JsonlWriter(self.raw / "sd_round_records.jsonl")
        )
        self.inline_tokens = self.stack.enter_context(
            JsonlWriter(self.raw / "inline_guard_token_records.jsonl")
        )
        self.inline_rounds = self.stack.enter_context(
            JsonlWriter(self.raw / "inline_guard_round_records.jsonl")
        )
        self.blind_outputs = self.stack.enter_context(
            JsonlWriter(self.blind / "blind_outputs.jsonl")
        )
        self.audit = self.stack.enter_context(
            JsonlWriter(self.private / "original_dataset_audit.jsonl")
        )
        self.timings: list[dict[str, float | None]] = []
        self.inline_results: list[dict[str, Any]] = []

    def write(
        self,
        *,
        source: dict[str, Any],
        row: dict[str, Any],
        guard_payload: dict[str, Any],
        rounds: Iterable[dict[str, Any]],
        inline: dict[str, Any] | None,
    ) -> None:
        self.generation.write(row)
        for token in guard_payload.get("token_records", []):
            self.guard_tokens.write(
                {
                    "sample_id": source["sample_id"],
                    "sample_order": source["sample_order"],
                    **token,
                }
            )
        for round_record in rounds:
            self.rounds.write(
                {
                    "sample_id": source["sample_id"],
                    "sample_order": source["sample_order"],
                    **round_record,
                }
            )
        if inline is not None:
            for token in inline.get("token_records", []):
                self.inline_tokens.write(
                    {
                        "sample_id": source["sample_id"],
                        "sample_order": source["sample_order"],
                        **token,
                    }
                )
            for round_record in inline.get("round_records", []):
                self.inline_rounds.write(
                    {
                        "sample_id": source["sample_id"],
                        "sample_order": source["sample_order"],
                        **round_record,
                    }
                )
            self.inline_results.append(
                {
                    key: value
                    for key, value in inline.items()
                    if key not in {"token_records", "round_records"}
                }
            )
        completion = row["completion"]
        self.blind_outputs.write(
            {
                "schema_version": 3,
                "sample_id": source["sample_id"],
                "sample_order": source["sample_order"],
                "prompt": source["prompt"],
                "thinking": completion["thinking"],
                "final_answer": completion["final_answer"],
                "full_completion": completion["full_completion"],
                "parse_status": completion["parse_status"],
                "truncated": row["generation"]["truncated"],
            }
        )
        self.audit.write(
            {
                "sample_id": source["sample_id"],
                "sample_order": source["sample_order"],
                "dataset_index": source["dataset_index"],
                "original_label": source.get("original_label"),
                "original_source": source.get("original_source"),
                "original_unsafe_type": source.get("original_unsafe_type"),
            }
        )
        self.timings.append(row["timing"])

    def close(self) -> None:
        self.stack.close()
        keys = sorted(
            {
                key
                for row in self.timings
                for key, value in row.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        write_json(
            self.raw / "latency_summary.json",
            {
                "schema_version": 1,
                "sample_count": len(self.timings),
                "statistics": {
                    key: summarize_ms(
                        float(row[key])
                        for row in self.timings
                        if isinstance(row.get(key), (int, float))
                    )
                    for key in keys
                },
                "formal_timing": "unprofiled",
                "model_load_excluded": True,
            },
        )
        if self.inline_results:
            write_json(
                self.raw / "inline_guard_summary.json",
                aggregate_inline_records(self.inline_results),
            )


def _make_row(
    *,
    source: dict[str, Any],
    mode: str,
    thinking: bool,
    request_output: Any,
    generated_ids: list[int],
    completion: Any,
    generation_ms: float,
    pipeline_ms: float,
    timing: dict[str, Any],
    guard: dict[str, Any],
    speculative: dict[str, Any] | None,
    inline: dict[str, Any] | None,
) -> dict[str, Any]:
    completion_output = _completion_output(request_output)
    finish_reason = str(getattr(completion_output, "finish_reason", ""))
    return {
        "schema_version": 3,
        "sample_id": source["sample_id"],
        "sample_order": int(source["sample_order"]),
        "dataset_index": int(source["dataset_index"]),
        "sample_seed": int(source["sample_seed"]),
        "generation_mode": mode,
        "thinking_enabled": bool(thinking),
        "prompt": source["prompt"],
        "completion": asdict(completion),
        "generation": {
            "prompt_token_count": len(source["prompt_token_ids"]),
            "generated_token_count": len(generated_ids),
            "generated_ids": generated_ids,
            "latency_ms": float(generation_ms),
            "pipeline_latency_ms": float(pipeline_ms),
            "tokens_per_second": (
                None if generation_ms <= 0 else len(generated_ids) / (generation_ms / 1000.0)
            ),
            "finish_reason": finish_reason,
            "stopped_on_eos": finish_reason == "stop",
            "truncated": finish_reason == "length",
            "request_timing": extract_request_timing(request_output),
        },
        "timing": timing,
        "speculative_decoding": speculative,
        "guard": _guard_record(guard),
        "inline_guard": inline,
    }


def _cell_config(config: dict[str, Any], thinking_name: str, mode: str) -> dict[str, Any]:
    return {
        **config["public_config"],
        "generation_mode": mode,
        "enable_thinking": thinking_name == "think_on",
        "cell": f"{thinking_name}/{mode}",
    }


def _cell_root(config: dict[str, Any], thinking_name: str, mode: str) -> Path:
    return Path(config["matrix_dir"]) / thinking_name / mode


def _records(config: dict[str, Any], thinking_name: str) -> list[dict[str, Any]]:
    result = []
    for raw in config["samples"]:
        row = dict(raw)
        row["prompt_token_ids"] = raw["prompt_token_ids"][thinking_name]
        result.append(row)
    return result


def _warm_direct(llm: Any, SamplingParams: Any, record: dict[str, Any], settings: dict[str, Any]) -> None:
    output, _ = _generate(llm, SamplingParams, record, settings)
    ids = _token_ids(output)
    completion = split_completion(
        llm.get_tokenizer(), ids, thinking_expected=bool(record["thinking_enabled"])
    )
    _posthoc_guard(
        llm,
        prompt_ids=record["prompt_token_ids"],
        generated_ids=ids,
        answer_start=completion.answer_start_generated_index,
    )


def _run_direct_cells(llm: Any, SamplingParams: Any, config: dict[str, Any], mode: str) -> None:
    settings = config["settings"]
    for thinking_name in ("think_on", "think_off"):
        records = _records(config, thinking_name)
        for record in records:
            record["thinking_enabled"] = thinking_name == "think_on"
        if int(config["benchmark"]["warmup_samples"]) > 0:
            _warm_direct(llm, SamplingParams, records[0], settings)
        sink = CellSink(
            _cell_root(config, thinking_name, mode),
            _cell_config(config, thinking_name, mode),
        )
        try:
            for source in records:
                pipeline_started = time.perf_counter_ns()
                output, generation_ms = _generate(
                    llm, SamplingParams, source, settings
                )
                ids = _token_ids(output)
                completion = split_completion(
                    llm.get_tokenizer(),
                    ids,
                    thinking_expected=bool(source["thinking_enabled"]),
                )
                guard, guard_rpc_ms = _posthoc_guard(
                    llm,
                    prompt_ids=source["prompt_token_ids"],
                    generated_ids=ids,
                    answer_start=completion.answer_start_generated_index,
                )
                pipeline_ms = (
                    time.perf_counter_ns() - pipeline_started
                ) / 1_000_000.0
                stage_name = (
                    "target_model_generation_ms"
                    if mode == "target_only"
                    else "draft_model_generation_ms"
                )
                timing = {
                    "primary_model_generation_ms": generation_ms,
                    stage_name: generation_ms,
                    "posthoc_guard_model_ms": float(guard["latency_ms"]),
                    "posthoc_guard_input_packaging_ms": float(
                        guard.get("input_packaging_ms", 0.0)
                    ),
                    "posthoc_guard_input_h2d_ms": float(
                        guard.get("input_h2d_ms", 0.0)
                    ),
                    "posthoc_guard_input_preparation_ms": float(
                        guard.get("input_preparation_ms", 0.0)
                    ),
                    "posthoc_guard_output_decode_d2h_ms": float(
                        guard.get("output_decode_d2h_ms", 0.0)
                    ),
                    "posthoc_guard_rpc_ms": guard_rpc_ms,
                    "pipeline_total_ms": pipeline_ms,
                    "pipeline_transfer_excluded_ms": (
                        pipeline_ms - float(guard.get("input_h2d_ms", 0.0))
                    ),
                }
                row = _make_row(
                    source=source,
                    mode=mode,
                    thinking=bool(source["thinking_enabled"]),
                    request_output=output,
                    generated_ids=ids,
                    completion=completion,
                    generation_ms=generation_ms,
                    pipeline_ms=pipeline_ms,
                    timing=timing,
                    guard=guard,
                    speculative=None,
                    inline=None,
                )
                sink.write(source=source, row=row, guard_payload=guard, rounds=(), inline=None)
        finally:
            sink.close()


def _spec_round_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "round_index": index,
            "proposed_count": int(proposed),
            "accepted_count": int(accepted),
            "source": "vllm_per_request_spec_decode_metrics_detailed",
        }
        for index, (accepted, proposed) in enumerate(
            zip(metrics["per_step_accepted"], metrics["per_step_drafted"])
        )
    ]


def _validate_inline_acceptance_trace(
    round_records: list[dict[str, Any]], metrics: dict[str, Any]
) -> dict[str, int]:
    """Compare scheduler-visible verify steps, retaining post-stop GPU diagnostics.

    With async scheduling the worker can finish another verification after the
    scheduler has stopped the request.  The adapter records that executed work
    as a terminal_output_clipped diagnostic, while vLLM's per-request metrics
    cannot include it because the Request has already finished.
    """
    observed = [
        row
        for row in round_records
        if int(row["round_index"]) >= 0
        and row.get("commit_kind") != "unverified_after_stop"
    ]
    accepted = [int(value) for value in metrics["per_step_accepted"]]
    drafted = [int(value) for value in metrics["per_step_drafted"]]
    if len(accepted) != len(drafted):
        raise RuntimeError("vLLM detailed acceptance arrays have different lengths")
    visible = observed[: len(accepted)]
    excluded = observed[len(accepted) :]
    observed_accepted = [
        int(row.get("target_verified_accepted_count", row["accepted_count"]))
        for row in visible
    ]
    observed_drafted = [
        int(row.get("target_verified_proposed_count", len(row["proposed_ids"])))
        for row in visible
    ]
    if (
        len(visible) != len(accepted)
        or observed_accepted != accepted
        or observed_drafted != drafted
        or any(row.get("commit_kind") == "terminal_output_clipped" for row in visible)
        or any(row.get("commit_kind") != "terminal_output_clipped" for row in excluded)
    ):
        raise RuntimeError(
            "Inline adapter trace does not match vLLM detailed acceptance metrics: "
            f"scheduler-visible observed accepted/drafted="
            f"{observed_accepted}/{observed_drafted}, "
            f"vLLM={accepted}/{drafted}, "
            f"post-stop commit kinds={[row.get('commit_kind') for row in excluded]}"
        )
    return {
        "scheduler_metric_round_count": len(accepted),
        "observed_verified_round_count": len(observed),
        "excluded_post_stop_verified_round_count": len(excluded),
    }


def _run_inline_once(
    llm: Any,
    SamplingParams: Any,
    source: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[Any, list[int], Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    operational_started = time.perf_counter_ns()
    begin_started = time.perf_counter_ns()
    begin = _rpc_one(
        llm,
        "sd4_begin_inline",
        sample_id=source["sample_id"],
        prompt_ids=source["prompt_token_ids"],
        enable_thinking=bool(source["thinking_enabled"]),
    )
    begin_rpc_ms = (time.perf_counter_ns() - begin_started) / 1_000_000.0
    output, engine_generate_ms = _generate(llm, SamplingParams, source, settings)
    ids = _token_ids(output)
    metrics = _spec_metrics(output)
    completion = split_completion(
        llm.get_tokenizer(), ids, thinking_expected=bool(source["thinking_enabled"])
    )
    finish_started = time.perf_counter_ns()
    inline = _rpc_one(
        llm,
        "sd4_finish_inline",
        sample_id=source["sample_id"],
        generated_ids=ids,
        answer_start_generated_index=completion.answer_start_generated_index,
        per_step_accepted=[int(value) for value in metrics["per_step_accepted"]],
        per_step_drafted=[int(value) for value in metrics["per_step_drafted"]],
    )
    finish_rpc_ms = (time.perf_counter_ns() - finish_started) / 1_000_000.0
    operational_ms = (time.perf_counter_ns() - operational_started) / 1_000_000.0
    guard, audit_rpc_ms = _posthoc_guard(
        llm,
        prompt_ids=source["prompt_token_ids"],
        generated_ids=ids,
        answer_start=completion.answer_start_generated_index,
    )
    inline["adapter"]["acceptance_trace_reconciliation"] = (
        _validate_inline_acceptance_trace(inline.get("round_records", []), metrics)
    )
    inline["adapter"]["trace_matches_vllm_detailed_metrics"] = True
    _attach_inline_consistency(inline, guard)
    guard_timing = inline["timing"]
    timing = {
        "engine_generate_with_inline_guard_ms": engine_generate_ms,
        "inline_begin_rpc_ms": begin_rpc_ms,
        "inline_finish_rpc_ms": finish_rpc_ms,
        "inline_guard_prefill_ms": float(guard_timing["prefill_latency_ms"]),
        "inline_guard_proposal_ms": float(guard_timing["proposal_latency_ms"]),
        "inline_guard_commit_ms": float(guard_timing["commit_latency_ms"]),
        "inline_guard_total_model_ms": float(
            guard_timing["total_inline_guard_latency_ms"]
        ),
        "inline_guard_prefill_input_h2d_ms": float(
            guard_timing.get("prefill_input_h2d_ms", 0.0)
        ),
        "inline_guard_proposal_output_d2h_ms": float(
            guard_timing.get("proposal_output_d2h_ms", 0.0)
        ),
        "inline_guard_proposal_input_h2d_ms": float(
            guard_timing.get("proposal_input_h2d_ms", 0.0)
        ),
        "inline_guard_commit_output_d2h_ms": float(
            guard_timing.get("commit_output_d2h_ms", 0.0)
        ),
        "inline_guard_commit_input_h2d_ms": float(
            guard_timing.get("commit_input_h2d_ms", 0.0)
        ),
        "inline_guard_staged_cross_device_handoff_ms": float(
            guard_timing.get("staged_cross_device_handoff_ms", 0.0)
        ),
        "operational_pipeline_total_ms": operational_ms,
        "operational_pipeline_transfer_excluded_ms": (
            operational_ms
            - float(guard_timing.get("staged_cross_device_handoff_ms", 0.0))
        ),
        "posthoc_audit_guard_model_ms_excluded": float(guard["latency_ms"]),
        "posthoc_audit_guard_input_h2d_ms_excluded": float(
            guard.get("input_h2d_ms", 0.0)
        ),
        "posthoc_audit_guard_rpc_ms_excluded": audit_rpc_ms,
    }
    del begin
    return output, ids, completion, inline, guard, {"metrics": metrics, "timing": timing}


def _warm_spec_pair(llm: Any, SamplingParams: Any, source: dict[str, Any], settings: dict[str, Any]) -> None:
    standard, _ = _generate(llm, SamplingParams, source, settings)
    inline, ids, completion, _, _, _ = _run_inline_once(
        llm, SamplingParams, source, settings
    )
    if _token_ids(standard) != ids:
        raise RuntimeError("Warm-up standard/inline outputs differ")
    del inline, completion


def _run_spec_cells(llm: Any, SamplingParams: Any, config: dict[str, Any]) -> None:
    settings = config["settings"]
    for thinking_name in ("think_on", "think_off"):
        records = _records(config, thinking_name)
        for record in records:
            record["thinking_enabled"] = thinking_name == "think_on"
        if int(config["benchmark"]["warmup_samples"]) > 0:
            _warm_spec_pair(llm, SamplingParams, records[0], settings)
        standard_sink = CellSink(
            _cell_root(config, thinking_name, "speculative"),
            _cell_config(config, thinking_name, "speculative"),
        )
        inline_sink = CellSink(
            _cell_root(config, thinking_name, "inline_guard"),
            _cell_config(config, thinking_name, "inline_guard"),
        )
        try:
            for source in records:
                standard_pipeline_started = time.perf_counter_ns()
                standard_output, standard_ms = _generate(
                    llm, SamplingParams, source, settings
                )
                standard_ids = _token_ids(standard_output)
                standard_completion = split_completion(
                    llm.get_tokenizer(),
                    standard_ids,
                    thinking_expected=bool(source["thinking_enabled"]),
                )
                standard_guard, standard_guard_rpc_ms = _posthoc_guard(
                    llm,
                    prompt_ids=source["prompt_token_ids"],
                    generated_ids=standard_ids,
                    answer_start=standard_completion.answer_start_generated_index,
                )
                standard_pipeline_ms = (
                    time.perf_counter_ns() - standard_pipeline_started
                ) / 1_000_000.0
                standard_metrics = _spec_metrics(standard_output)
                standard_timing = {
                    "engine_generate_ms": standard_ms,
                    "posthoc_guard_model_ms": float(standard_guard["latency_ms"]),
                    "posthoc_guard_input_packaging_ms": float(
                        standard_guard.get("input_packaging_ms", 0.0)
                    ),
                    "posthoc_guard_input_h2d_ms": float(
                        standard_guard.get("input_h2d_ms", 0.0)
                    ),
                    "posthoc_guard_input_preparation_ms": float(
                        standard_guard.get("input_preparation_ms", 0.0)
                    ),
                    "posthoc_guard_output_decode_d2h_ms": float(
                        standard_guard.get("output_decode_d2h_ms", 0.0)
                    ),
                    "posthoc_guard_rpc_ms": standard_guard_rpc_ms,
                    "operational_pipeline_total_ms": standard_pipeline_ms,
                    "operational_pipeline_transfer_excluded_ms": (
                        standard_pipeline_ms
                        - float(standard_guard.get("input_h2d_ms", 0.0))
                    ),
                }
                standard_row = _make_row(
                    source=source,
                    mode="speculative",
                    thinking=bool(source["thinking_enabled"]),
                    request_output=standard_output,
                    generated_ids=standard_ids,
                    completion=standard_completion,
                    generation_ms=standard_ms,
                    pipeline_ms=standard_pipeline_ms,
                    timing=standard_timing,
                    guard=standard_guard,
                    speculative=standard_metrics,
                    inline=None,
                )
                standard_sink.write(
                    source=source,
                    row=standard_row,
                    guard_payload=standard_guard,
                    rounds=_spec_round_rows(standard_metrics),
                    inline=None,
                )

                (
                    inline_output,
                    inline_ids,
                    inline_completion,
                    inline,
                    inline_guard,
                    extra,
                ) = _run_inline_once(llm, SamplingParams, source, settings)
                if inline_ids != standard_ids:
                    mismatch = next(
                        (
                            index
                            for index, pair in enumerate(zip(standard_ids, inline_ids))
                            if pair[0] != pair[1]
                        ),
                        min(len(standard_ids), len(inline_ids)),
                    )
                    raise RuntimeError(
                        f"Passive Guard changed output for {source['sample_id']}; "
                        f"first mismatch at generated token {mismatch}"
                    )
                if asdict(inline_completion) != asdict(standard_completion):
                    raise RuntimeError(
                        f"Completion parsing differs for {source['sample_id']}"
                    )
                inline_metrics = extra["metrics"]
                if (
                    inline_metrics["per_step_accepted"]
                    != standard_metrics["per_step_accepted"]
                    or inline_metrics["per_step_drafted"]
                    != standard_metrics["per_step_drafted"]
                ):
                    raise RuntimeError(
                        f"Passive Guard changed acceptance trace for {source['sample_id']}"
                    )
                timing = extra["timing"]
                timing.update(
                    {
                        "reference_speculative_engine_ms": standard_ms,
                        "reference_speculative_pipeline_ms": standard_pipeline_ms,
                        "paired_engine_overhead_ms": (
                            float(timing["engine_generate_with_inline_guard_ms"])
                            - standard_ms
                        ),
                        "paired_operational_pipeline_overhead_ms": (
                            float(timing["operational_pipeline_total_ms"])
                            - standard_pipeline_ms
                        ),
                        "paired_transfer_excluded_pipeline_overhead_ms": (
                            float(
                                timing[
                                    "operational_pipeline_transfer_excluded_ms"
                                ]
                            )
                            - float(
                                standard_timing[
                                    "operational_pipeline_transfer_excluded_ms"
                                ]
                            )
                        ),
                    }
                )
                inline["reference_equivalence"] = {
                    "generated_ids_match": True,
                    "completion_match": True,
                    "acceptance_trace_match": True,
                    "human_labels_reusable": True,
                    "reference_cell": f"{thinking_name}/speculative",
                }
                timing["inline_end_to_end_generation_latency_ms"] = float(
                    timing["operational_pipeline_total_ms"]
                )
                timing["observed_generation_overhead_ms"] = float(
                    timing["paired_engine_overhead_ms"]
                )
                inline["timing"].update(timing)
                inline_pipeline_ms = float(timing["operational_pipeline_total_ms"])
                inline_row = _make_row(
                    source=source,
                    mode="inline_guard",
                    thinking=bool(source["thinking_enabled"]),
                    request_output=inline_output,
                    generated_ids=inline_ids,
                    completion=inline_completion,
                    generation_ms=inline_pipeline_ms,
                    pipeline_ms=inline_pipeline_ms,
                    timing=timing,
                    guard=inline_guard,
                    speculative=inline_metrics,
                    inline=inline,
                )
                inline_sink.write(
                    source=source,
                    row=inline_row,
                    guard_payload=inline_guard,
                    rounds=inline.get("round_records", ()),
                    inline=inline,
                )
        finally:
            standard_sink.close()
            inline_sink.close()


def _guard_timing_only(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_forward_ms": float(payload.get("latency_ms", 0.0)),
        "input_packaging_ms": float(payload.get("input_packaging_ms", 0.0)),
        "input_h2d_ms": float(payload.get("input_h2d_ms", 0.0)),
        "input_preparation_ms": float(payload.get("input_preparation_ms", 0.0)),
        "output_decode_d2h_ms": float(payload.get("output_decode_d2h_ms", 0.0)),
    }


def _stage_timing_call(llm: Any, label: str, operation: Any) -> dict[str, Any]:
    begin_started_ns = time.perf_counter_ns()
    begin = _rpc_one(llm, "sd4_begin_stage_timing", label=label)
    begin_rpc_ms = (time.perf_counter_ns() - begin_started_ns) / 1_000_000.0
    operation_started_ns = time.perf_counter_ns()
    try:
        result = operation()
    finally:
        operation_wall_ms = (
            time.perf_counter_ns() - operation_started_ns
        ) / 1_000_000.0
        finish_started_ns = time.perf_counter_ns()
        measured = _rpc_one(llm, "sd4_finish_stage_timing", label=label)
        finish_rpc_ms = (
            time.perf_counter_ns() - finish_started_ns
        ) / 1_000_000.0
    measured.update(
        {
            "operation_wall_ms": operation_wall_ms,
            "begin_control_rpc_ms": begin_rpc_ms,
            "finish_control_rpc_ms": finish_rpc_ms,
            "control_rpc_total_ms": begin_rpc_ms + finish_rpc_ms,
            "begin_contract": begin,
            "result": result,
        }
    )
    return measured


def _paired_sd_equivalence(
    standard_result: dict[str, Any], inline_result: dict[str, Any]
) -> dict[str, bool]:
    standard_metrics = standard_result["speculative_decoding"]
    inline_metrics = inline_result["speculative_decoding"]
    return {
        "generated_ids_match": (
            standard_result["generated_ids"] == inline_result["generated_ids"]
        ),
        "acceptance_trace_match": (
            standard_metrics["per_step_accepted"]
            == inline_metrics["per_step_accepted"]
            and standard_metrics["per_step_drafted"]
            == inline_metrics["per_step_drafted"]
        ),
    }


def _require_paired_sd_equivalence(
    standard_result: dict[str, Any],
    inline_result: dict[str, Any],
    *,
    context: str,
) -> dict[str, bool]:
    equivalence = _paired_sd_equivalence(standard_result, inline_result)
    if not all(equivalence.values()):
        raise RuntimeError(
            f"Lightweight timing changed passive SD behavior for {context}: "
            f"outputs_match={equivalence['generated_ids_match']}, "
            f"acceptance_match={equivalence['acceptance_trace_match']}"
        )
    return equivalence


def _run_stage_timing_diagnostics(
    llm: Any, SamplingParams: Any, config: dict[str, Any]
) -> None:
    diagnostic = config["stage_timing"]
    if not diagnostic["enabled"]:
        return
    directory = Path(diagnostic["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    cap = min(
        int(diagnostic["max_new_tokens"]),
        int(config["settings"]["max_new_tokens"]),
    )
    warmup_runs = int(config["benchmark"].get("warmup_samples", 1))
    index: dict[str, Any] = {}
    for thinking_name, thinking_enabled in (("think_on", True), ("think_off", False)):
        source = _records(config, thinking_name)[0]
        source["thinking_enabled"] = thinking_enabled

        def standard() -> dict[str, Any]:
            output, generation_ms = _generate(
                llm, SamplingParams, source, config["settings"], max_tokens=cap
            )
            ids = _token_ids(output)
            metrics = _spec_metrics(output)
            completion = split_completion(
                llm.get_tokenizer(), ids, thinking_expected=thinking_enabled
            )
            guard, guard_rpc_ms = _posthoc_guard(
                llm,
                prompt_ids=source["prompt_token_ids"],
                generated_ids=ids,
                answer_start=completion.answer_start_generated_index,
            )
            return {
                "generated_ids": ids,
                "generated_token_count": len(ids),
                "engine_generate_ms_under_stage_sync": generation_ms,
                "finish_reason": getattr(_completion_output(output), "finish_reason", None),
                "speculative_decoding": metrics,
                "guard": _guard_timing_only(guard),
                "guard_rpc_ms": guard_rpc_ms,
            }

        def inline() -> dict[str, Any]:
            _rpc_one(
                llm,
                "sd4_begin_inline",
                sample_id=source["sample_id"],
                prompt_ids=source["prompt_token_ids"],
                enable_thinking=thinking_enabled,
            )
            output, generation_ms = _generate(
                llm, SamplingParams, source, config["settings"], max_tokens=cap
            )
            ids = _token_ids(output)
            metrics = _spec_metrics(output)
            completion = split_completion(
                llm.get_tokenizer(), ids, thinking_expected=thinking_enabled
            )
            inline_result = _rpc_one(
                llm,
                "sd4_finish_inline",
                sample_id=source["sample_id"],
                generated_ids=ids,
                answer_start_generated_index=completion.answer_start_generated_index,
                per_step_accepted=[
                    int(value) for value in metrics["per_step_accepted"]
                ],
                per_step_drafted=[
                    int(value) for value in metrics["per_step_drafted"]
                ],
            )
            guard_timing = inline_result["timing"]
            return {
                "generated_ids": ids,
                "generated_token_count": len(ids),
                "engine_generate_ms_under_stage_sync": generation_ms,
                "finish_reason": getattr(_completion_output(output), "finish_reason", None),
                "speculative_decoding": metrics,
                "guard": {
                    "prefill_model_ms": float(guard_timing["prefill_latency_ms"]),
                    "proposal_model_ms": float(guard_timing["proposal_latency_ms"]),
                    "commit_model_ms": float(guard_timing["commit_latency_ms"]),
                    "total_model_ms": float(
                        guard_timing["total_inline_guard_latency_ms"]
                    ),
                    "staged_cross_device_handoff_ms": float(
                        guard_timing.get("staged_cross_device_handoff_ms", 0.0)
                    ),
                    "proposal_forward_count": int(
                        guard_timing.get("proposal_forward_count", 0)
                    ),
                    "commit_forward_count": int(
                        guard_timing.get("commit_forward_count", 0)
                    ),
                },
            }

        warmup_pairs: list[dict[str, Any]] = []
        for warmup_index in range(warmup_runs):
            print(
                f"[stage-timing] {thinking_name}: warm-up "
                f"{warmup_index + 1}/{warmup_runs} standard+inline "
                "(excluded from statistics)",
                flush=True,
            )
            # No stage-timing RPC is active here. These calls exercise the same
            # Target, Draft, rejection sampler, Guard, and transfer paths as the
            # measured pair, but none of their latency samples enter the report.
            warmup_standard = standard()
            warmup_inline = inline()
            warmup_equivalence = _require_paired_sd_equivalence(
                warmup_standard,
                warmup_inline,
                context=f"{thinking_name} warm-up {warmup_index + 1}",
            )
            warmup_pairs.append(
                {
                    "run_index": warmup_index + 1,
                    "generated_token_count": int(
                        warmup_standard["generated_token_count"]
                    ),
                    "paired_equivalence": warmup_equivalence,
                }
            )

        print(
            f"[stage-timing] {thinking_name}: measuring standard SD",
            flush=True,
        )
        standard_summary = _stage_timing_call(
            llm, f"sd4_{thinking_name}_standard_sd", standard
        )
        print(
            f"[stage-timing] {thinking_name}: measuring inline SD",
            flush=True,
        )
        inline_summary = _stage_timing_call(
            llm, f"sd4_{thinking_name}_inline_sd", inline
        )
        standard_result = standard_summary["result"]
        inline_result = inline_summary["result"]
        equivalence = _require_paired_sd_equivalence(
            standard_result, inline_result, context=thinking_name
        )
        for result in (standard_result, inline_result):
            metrics = result["speculative_decoding"]
            result["speculative_decoding"] = {
                "round_count": int(metrics["round_count"]),
                "proposed_token_count": int(metrics["proposed_token_count"]),
                "accepted_draft_token_count": int(
                    metrics["accepted_draft_token_count"]
                ),
                "acceptance_rate": metrics["acceptance_rate"],
            }
            del result["generated_ids"]
        standard_summary["paired_equivalence"] = equivalence
        inline_summary["paired_equivalence"] = equivalence
        write_json(
            _cell_root(config, thinking_name, "speculative")
            / "raw"
            / "stage_timing_summary.json",
            standard_summary,
        )
        write_json(
            _cell_root(config, thinking_name, "inline_guard")
            / "raw"
            / "stage_timing_summary.json",
            inline_summary,
        )
        index[thinking_name] = {
            "warmup": {
                "runs_per_path": warmup_runs,
                "max_new_tokens": cap,
                "excluded_from_stage_statistics": True,
                "stage_timing_session_active": False,
                "pairs": warmup_pairs,
            },
            "standard_sd": standard_summary,
            "inline_sd": inline_summary,
        }
    write_json(directory / "stage_timing_index.json", index)


def _build_engine(config: dict[str, Any]) -> tuple[Any, Any, float, float, dict[str, Any]]:
    if not (3, 10) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError(
            "vLLM 0.31.0 requires Python >=3.10,<3.15; "
            f"found Python {sys.version.split()[0]} at {sys.executable}"
        )
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("Install requirements-vllm.txt on the Linux GPU host") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(
            "SD_4 dual-GPU mode requires exactly two visible GPUs ordered as "
            "engine,guard in CUDA_VISIBLE_DEVICES"
        )
    installed = importlib.metadata.version("vllm")
    if installed.split("+", 1)[0] != PINNED_VLLM_BASE:
        raise RuntimeError(
            f"SD_4 adapter requires vLLM base version {PINNED_VLLM_BASE}; found {installed}"
        )
    installed_torch = importlib.metadata.version("torch")
    installed_transformers = importlib.metadata.version("transformers")
    if installed_torch.split("+", 1)[0] != "2.13.0" or torch.version.cuda != "12.9":
        raise RuntimeError(
            "The two-L20 SD_4 path requires torch 2.13.0+cu129; found "
            f"torch={installed_torch}, torch.version.cuda={torch.version.cuda}"
        )
    if installed_transformers != "5.17.0":
        raise RuntimeError(
            "The two-L20 SD_4 path requires transformers==5.17.0; found "
            f"{installed_transformers}"
        )
    role = config["role"]
    engine = config["engine"]
    model_key = "draft" if role == "draft" else "target"
    kwargs: dict[str, Any] = {
        "model": config["models"][model_key],
        "tokenizer": config["models"][model_key],
        "dtype": engine["dtype"],
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": float(engine["gpu_memory_utilization"]),
        "max_model_len": int(engine["max_model_len"]),
        "max_num_seqs": 1,
        "enable_prefix_caching": False,
        "enforce_eager": bool(engine["enforce_eager"]),
        "seed": int(config["settings"]["seed"]),
        "trust_remote_code": bool(engine["trust_remote_code"]),
        "disable_log_stats": False,
        "worker_extension_cls": EXTENSION_CLASS,
    }
    revision = config["revisions"].get(model_key)
    if revision is not None:
        kwargs["revision"] = revision
        kwargs["tokenizer_revision"] = revision
    if role == "speculative":
        kwargs["per_request_spec_decode_metrics"] = "detailed"
        spec: dict[str, Any] = {
            "method": "draft_model",
            "model": config["models"]["draft"],
            "num_speculative_tokens": int(config["settings"]["num_speculative_tokens"]),
            "draft_sample_method": "greedy",
            "rejection_sample_method": "standard",
            "max_model_len": int(engine["max_model_len"]),
        }
        if config["revisions"].get("draft") is not None:
            spec["revision"] = config["revisions"]["draft"]
        kwargs["speculative_config"] = spec
    load_started = time.perf_counter_ns()
    llm = LLM(**kwargs)
    engine_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000.0
    guard_info = _rpc_one(llm, "sd4_initialize_guard", **config["guard"])
    if (
        str(guard_info.get("engine_device")) != "cuda:0"
        or str(guard_info.get("guard_device")) != "cuda:1"
    ):
        raise RuntimeError(
            "SD_4 device placement mismatch: expected Target+Draft on cuda:0 "
            f"and Guard on cuda:1, received {guard_info}"
        )
    if (
        role == "speculative"
        and guard_info.get("hook") == "none (non-speculative engine)"
    ):
        raise RuntimeError("Speculative worker did not install an inline Guard hook")
    guard_load_ms = float(guard_info["guard_load_ms"])
    runtime = _runtime_metadata(torch, installed)
    runtime["guard_adapter"] = _jsonable(guard_info)
    runtime["device_transfer_diagnostic"] = _jsonable(
        _rpc_one(
            llm,
            "sd4_measure_device_transfer",
            token_counts=[
                int(config["settings"]["num_speculative_tokens"]),
                int(config["settings"]["max_new_tokens"]),
            ],
            repetitions=100,
            warmup=10,
        )
    )
    return llm, SamplingParams, engine_load_ms, guard_load_ms, runtime


def run(config: dict[str, Any]) -> dict[str, Any]:
    llm, SamplingParams, engine_load_ms, guard_load_ms, runtime = _build_engine(config)
    try:
        role = config["role"]
        execution_mode = config.get("execution_mode", "formal")
        if execution_mode == "stage_timing_only":
            if role != "speculative":
                raise RuntimeError("Stage-timing-only execution requires speculative role")
            _run_stage_timing_diagnostics(llm, SamplingParams, config)
        elif role == "target":
            _run_direct_cells(llm, SamplingParams, config, "target_only")
        elif role == "draft":
            _run_direct_cells(llm, SamplingParams, config, "draft_only")
        elif role == "speculative":
            _run_spec_cells(llm, SamplingParams, config)
        else:
            raise ValueError(f"Unknown worker role: {role!r}")
        return {
            "schema_version": 1,
            "status": "complete",
            "role": role,
            "engine_load_ms": engine_load_ms,
            "guard_load_ms": guard_load_ms,
            "runtime": runtime,
            "model_load_time_in_formal_latency": False,
        }
    finally:
        if hasattr(llm, "shutdown"):
            llm.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(read_json(args.config))
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
