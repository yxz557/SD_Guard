"""Run one isolated vLLM engine for the latency benchmark.

The parent process starts this module once for Target-only and once for
speculative decoding.  A fresh process is intentional: vLLM owns the CUDA
allocator, KV cache, workers, and CUDA graphs, and those resources are not
reliably reusable by a differently configured engine in the same process.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from sd_latency.vllm_profile import summarize_profile_directory


PINNED_VLLM = "0.31.0"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _as_number_list(value: Any) -> list[float] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return None


def _merge_metric_value(
    destination: dict[str, float | list[float]],
    name: str,
    value: float | list[float],
) -> None:
    previous = destination.get(name)
    if previous is None:
        destination[name] = value
        return
    if isinstance(previous, list) and isinstance(value, list):
        size = max(len(previous), len(value))
        merged = [0.0] * size
        for index in range(size):
            merged[index] = (
                (previous[index] if index < len(previous) else 0.0)
                + (value[index] if index < len(value) else 0.0)
            )
        destination[name] = merged
        return
    if not isinstance(previous, list) and not isinstance(value, list):
        destination[name] = float(previous) + float(value)


def _metric_snapshot(llm: Any) -> dict[str, float | list[float]]:
    """Read cumulative speculative-decoding counters from vLLM if available."""

    if not hasattr(llm, "get_metrics"):
        return {}
    metrics: Iterable[Any] = llm.get_metrics()

    snapshot: dict[str, float | list[float]] = {}
    for metric in metrics:
        name = str(getattr(metric, "name", ""))
        if "spec_decode" not in name:
            continue
        if hasattr(metric, "values"):
            values = _as_number_list(getattr(metric, "values"))
            if values is not None:
                _merge_metric_value(snapshot, name, values)
                continue
        raw_value = getattr(metric, "value", None)
        if isinstance(raw_value, (int, float)):
            _merge_metric_value(snapshot, name, float(raw_value))
    return snapshot


def _subtract_metric_snapshots(
    after: dict[str, float | list[float]],
    before: dict[str, float | list[float]],
) -> dict[str, float | list[float]]:
    result: dict[str, float | list[float]] = {}
    for name, after_value in after.items():
        if name.startswith("__"):
            continue
        before_value = before.get(name)
        if isinstance(after_value, list):
            previous = before_value if isinstance(before_value, list) else []
            result[name] = [
                value - (previous[index] if index < len(previous) else 0.0)
                for index, value in enumerate(after_value)
            ]
        else:
            previous_scalar = (
                float(before_value)
                if before_value is not None and not isinstance(before_value, list)
                else 0.0
            )
            result[name] = float(after_value) - previous_scalar
    return result


def _normalized_metric_name(name: str) -> str:
    return name.removesuffix("_total")


def _find_scalar_metric(
    metrics: dict[str, float | list[float]], suffix: str
) -> float | None:
    for name, value in metrics.items():
        if _normalized_metric_name(name).endswith(suffix) and not isinstance(value, list):
            return float(value)
    return None


def _find_vector_metric(
    metrics: dict[str, float | list[float]], suffix: str
) -> list[float] | None:
    for name, value in metrics.items():
        if _normalized_metric_name(name).endswith(suffix) and isinstance(value, list):
            return [float(item) for item in value]
    return None


def derive_speculative_metrics(
    raw_delta: dict[str, float | list[float]],
) -> dict[str, Any]:
    """Convert vLLM's cumulative counters into stable benchmark fields."""

    drafts = _find_scalar_metric(raw_delta, "spec_decode_num_drafts")
    draft_tokens = _find_scalar_metric(raw_delta, "spec_decode_num_draft_tokens")
    accepted = _find_scalar_metric(raw_delta, "spec_decode_num_accepted_tokens")
    accepted_per_position = _find_vector_metric(
        raw_delta, "spec_decode_num_accepted_tokens_per_pos"
    )
    available = drafts is not None or draft_tokens is not None or accepted is not None
    acceptance_per_position = None
    if accepted_per_position is not None and drafts is not None and drafts > 0:
        acceptance_per_position = [value / drafts for value in accepted_per_position]
    return {
        "available": available,
        "num_spec_steps": None if drafts is None else int(round(drafts)),
        "num_draft_tokens": None if draft_tokens is None else int(round(draft_tokens)),
        "num_accepted_draft_tokens": (
            None if accepted is None else int(round(accepted))
        ),
        "draft_acceptance_rate": (
            None
            if accepted is None or draft_tokens is None or draft_tokens <= 0
            else accepted / draft_tokens
        ),
        "mean_acceptance_length": (
            None
            if accepted is None or drafts is None or drafts <= 0
            else 1.0 + accepted / drafts
        ),
        "accepted_tokens_per_position": accepted_per_position,
        "acceptance_rate_per_position": acceptance_per_position,
        "raw_counter_delta": raw_delta,
        "note": (
            "Counters exclude warm-up because the worker snapshots metrics immediately "
            "before and after measured requests. Mean acceptance length includes the "
            "always-committed target token."
        ),
    }


def _collect_runtime_metadata(torch: Any) -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "device": f"cuda:{index}",
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "vllm_version": importlib.metadata.version("vllm"),
        "cuda_runtime_version": torch.version.cuda,
        "gpus": gpus,
    }


def _build_sampling_params(
    SamplingParams: Any,
    *,
    mode: str,
    settings: dict[str, Any],
    seed: int,
) -> Any:
    if mode == "greedy":
        temperature = 0.0
        top_p = 1.0
        top_k = 0
    elif mode == "sample":
        temperature = float(settings["temperature"])
        top_p = float(settings["top_p"])
        top_k = int(settings["top_k"])
    else:  # pragma: no cover - parent validates this
        raise ValueError(f"Unsupported mode: {mode}")
    return SamplingParams(
        max_tokens=int(settings["max_new_tokens"]),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=0.0,
        seed=seed,
        detokenize=False,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )


def _partition(records: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    return [records[index : index + size] for index in range(0, len(records), size)]


def _generate_batch(
    llm: Any,
    SamplingParams: Any,
    batch: list[dict[str, Any]],
    *,
    mode: str,
    settings: dict[str, Any],
) -> tuple[list[Any], float]:
    prompts = [
        {"prompt_token_ids": [int(token) for token in record["prompt_token_ids"]]}
        for record in batch
    ]
    sampling_params = [
        _build_sampling_params(
            SamplingParams,
            mode=mode,
            settings=settings,
            seed=int(record["seed"]),
        )
        for record in batch
    ]
    started_ns = time.perf_counter_ns()
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return list(outputs), elapsed_ms


def _output_token_ids(request_output: Any) -> list[int]:
    candidates = getattr(request_output, "outputs", None)
    if not candidates:
        raise RuntimeError("vLLM returned no completion candidates")
    token_ids = getattr(candidates[0], "token_ids", None)
    if token_ids is None:
        raise RuntimeError("vLLM completion did not expose token_ids")
    return [int(value) for value in token_ids]


def _run_mode(
    llm: Any,
    SamplingParams: Any,
    prompts: list[dict[str, Any]],
    *,
    mode: str,
    settings: dict[str, Any],
    warmup: int,
    repeats: int,
    batch_size: int,
) -> dict[str, Any]:
    batches = _partition(prompts, batch_size)
    for _ in range(warmup):
        for batch in batches:
            _generate_batch(
                llm,
                SamplingParams,
                batch,
                mode=mode,
                settings=settings,
            )

    metrics_before = _metric_snapshot(llm)
    rows: list[dict[str, Any]] = []
    first_outputs: dict[str, list[int]] = {}
    for repeat_index in range(repeats):
        for batch_index, batch in enumerate(batches):
            outputs, elapsed_ms = _generate_batch(
                llm,
                SamplingParams,
                batch,
                mode=mode,
                settings=settings,
            )
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} vLLM outputs, received {len(outputs)}"
                )
            output_ids = {
                str(record["id"]): _output_token_ids(output)
                for record, output in zip(batch, outputs)
            }
            if repeat_index == 0:
                first_outputs.update(output_ids)
            generated_token_count = sum(len(ids) for ids in output_ids.values())
            rows.append(
                {
                    "run_key": f"batch{batch_index}:repeat{repeat_index}",
                    "batch_index": batch_index,
                    "repeat_index": repeat_index,
                    "prompt_ids": [str(record["id"]) for record in batch],
                    "request_count": len(batch),
                    "preparation_ms": 0.0,
                    "generate_ms": elapsed_ms,
                    "output_transfer_ms": 0.0,
                    "total_ms": elapsed_ms,
                    "generated_token_count": generated_token_count,
                    "tokens_per_second": (
                        None
                        if elapsed_ms <= 0
                        else generated_token_count / (elapsed_ms / 1000.0)
                    ),
                    "output_token_ids": output_ids,
                }
            )
    metrics_after = _metric_snapshot(llm)
    metrics_delta = _subtract_metric_snapshots(metrics_after, metrics_before)
    return {
        "runs": rows,
        "first_outputs": first_outputs,
        "speculative_decoding_metrics": derive_speculative_metrics(metrics_delta),
    }


def _inspect_plain_draft_implementation() -> dict[str, Any]:
    """Record version-specific facts that materially affect latency."""

    try:
        from vllm.v1.worker.gpu.spec_decode.draft_model.speculator import (
            PlainDraftModelSpeculator,
        )

        init_source = inspect.getsource(PlainDraftModelSpeculator.init_cudagraph_manager)
        capture_source = inspect.getsource(PlainDraftModelSpeculator.capture)
        source_file = inspect.getsourcefile(PlainDraftModelSpeculator)
        combined = f"{init_source}\n{capture_source}"
        unsupported = "CUDA graph not yet supported" in combined
        return {
            "available": True,
            "plain_draft_cuda_graph_supported": not unsupported,
            "source_file": source_file,
            "evidence": (
                "PlainDraftModelSpeculator.init_cudagraph_manager() and capture() "
                "explicitly state that CUDA graph is not yet supported."
                if unsupported
                else "The expected no-CUDA-graph marker was not found; inspect the source."
            ),
            "interpretation": (
                "The target runner may use CUDA graphs when enforce_eager=false, but the "
                "plain draft-model proposal loop still runs without CUDA graphs in this "
                "installed vLLM build. Small batch-1 draft steps can therefore be launch- "
                "and framework-overhead dominated."
                if unsupported
                else None
            ),
        }
    except (ImportError, OSError, TypeError) as exc:
        return {
            "available": False,
            "plain_draft_cuda_graph_supported": None,
            "source_file": None,
            "evidence": f"Could not inspect PlainDraftModelSpeculator: {exc}",
            "interpretation": None,
        }


def _wait_for_profile_files(directory: Path, timeout_seconds: float = 30.0) -> None:
    """Allow the engine subprocess to finish flushing profiler output."""

    deadline = time.monotonic() + timeout_seconds
    previous: tuple[tuple[str, int], ...] | None = None
    stable_observations = 0
    while time.monotonic() < deadline:
        trace_paths = [*directory.rglob("*.pt.trace.json")]
        trace_paths.extend(directory.rglob("*.pt.trace.json.gz"))
        files = tuple(
            sorted(
                (str(path), path.stat().st_size)
                for path in trace_paths
                if path.is_file()
            )
        )
        if files and files == previous:
            stable_observations += 1
            if stable_observations >= 2:
                return
        else:
            stable_observations = 0
        previous = files
        time.sleep(0.25)


def _diagnose_profile(
    trace_summary: dict[str, Any],
    profile_metrics: dict[str, Any],
    implementation: dict[str, Any],
) -> dict[str, Any]:
    stages = trace_summary.get("stages", {})
    classified = {
        name: value
        for name, value in stages.items()
        if name != "other_or_unattributed" and isinstance(value, dict)
    }
    dominant_stage = None
    if classified:
        dominant_stage = max(
            classified,
            key=lambda name: float(classified[name].get("gpu_kernel_ms") or 0.0),
        )
    coverage = trace_summary.get("attribution_coverage")
    sufficient_coverage = isinstance(coverage, (int, float)) and coverage >= 0.6
    signals: list[dict[str, Any]] = []

    acceptance = profile_metrics.get("draft_acceptance_rate")
    if isinstance(acceptance, (int, float)):
        signals.append(
            {
                "id": "draft_acceptance",
                "observed": acceptance,
                "status": "low" if acceptance < 0.5 else "acceptable",
                "interpretation": (
                    "Fewer than half of proposed draft tokens were accepted, so each "
                    "accepted token amortizes relatively many draft-model steps."
                    if acceptance < 0.5
                    else "At least half of proposed draft tokens were accepted in the "
                    "profile request."
                ),
            }
        )

    cuda_graph_supported = implementation.get("plain_draft_cuda_graph_supported")
    if cuda_graph_supported is False:
        signals.append(
            {
                "id": "plain_draft_cuda_graph",
                "observed": False,
                "status": "unsupported",
                "interpretation": (
                    "This installed vLLM implementation cannot CUDA-graph the plain "
                    "draft proposal loop; batch-1 small-model steps remain exposed to "
                    "kernel-launch and Python/framework overhead."
                ),
            }
        )

    if sufficient_coverage and dominant_stage is not None:
        dominant_value = classified[dominant_stage]
        signals.append(
            {
                "id": "dominant_attributed_gpu_stage",
                "observed": dominant_stage,
                "status": "measured",
                "gpu_kernel_ms": dominant_value.get("gpu_kernel_ms"),
                "share_of_total_gpu_kernel_time": dominant_value.get(
                    "share_of_total_gpu_kernel_time"
                ),
                "interpretation": (
                    f"{dominant_stage} has the largest classified GPU kernel time in "
                    "the diagnostic trace."
                ),
            }
        )
    else:
        signals.append(
            {
                "id": "stage_attribution_coverage",
                "observed": coverage,
                "status": "inconclusive",
                "interpretation": (
                    "Less than 60% of GPU kernel time was attributed to a named stage; "
                    "use the raw trace instead of treating the stage shares as exact."
                ),
            }
        )

    return {
        "dominant_attributed_gpu_stage": (
            dominant_stage if sufficient_coverage else None
        ),
        "stage_conclusion_supported": sufficient_coverage,
        "signals": signals,
    }


def _run_profile(
    llm: Any,
    SamplingParams: Any,
    prompts: list[dict[str, Any]],
    *,
    mode: str,
    settings: dict[str, Any],
    profile_config: dict[str, Any],
    batch_size: int,
    untraced_mode_result: dict[str, Any],
) -> dict[str, Any]:
    """Profile one extra request after the uninstrumented benchmark is complete."""

    directory = Path(profile_config["output_dir"]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    profile_settings = dict(settings)
    profile_settings["max_new_tokens"] = min(
        int(settings["max_new_tokens"]),
        int(profile_config["max_new_tokens"]),
    )
    batch = _partition(prompts, batch_size)[0]
    # Measure an unprofiled request with exactly the same token cap and batch so
    # profiler overhead is not confused with the shorter diagnostic generation.
    _, same_shape_unprofiled_ms = _generate_batch(
        llm,
        SamplingParams,
        batch,
        mode=mode,
        settings=profile_settings,
    )
    metrics_before = _metric_snapshot(llm)
    profile_started = False
    stop_flush_ms: float | None = None
    try:
        llm.start_profile(profile_prefix=f"sd_{mode}")
        profile_started = True
        outputs, elapsed_ms = _generate_batch(
            llm,
            SamplingParams,
            batch,
            mode=mode,
            settings=profile_settings,
        )
    finally:
        if profile_started:
            stop_started_ns = time.perf_counter_ns()
            llm.stop_profile()
            stop_flush_ms = (time.perf_counter_ns() - stop_started_ns) / 1_000_000.0

    _wait_for_profile_files(directory)
    metrics_after = _metric_snapshot(llm)
    profile_metrics = derive_speculative_metrics(
        _subtract_metric_snapshots(metrics_after, metrics_before)
    )
    generated_token_count = sum(len(_output_token_ids(output)) for output in outputs)
    untraced_ms = [float(row["total_ms"]) for row in untraced_mode_result["runs"]]
    untraced_median_ms = statistics.median(untraced_ms)
    trace_summary = summarize_profile_directory(directory)
    implementation = _inspect_plain_draft_implementation()
    return {
        "enabled": True,
        "mode": mode,
        "request_count": len(batch),
        "max_new_tokens": profile_settings["max_new_tokens"],
        "generated_token_count": generated_token_count,
        "profiled_generate_wall_ms": elapsed_ms,
        "same_shape_unprofiled_generate_ms": same_shape_unprofiled_ms,
        "untraced_benchmark_median_ms": untraced_median_ms,
        "profiler_overhead_ratio": (
            None
            if same_shape_unprofiled_ms <= 0
            else elapsed_ms / same_shape_unprofiled_ms
        ),
        "profiler_stop_and_flush_ms": stop_flush_ms,
        "speculative_decoding_metrics": profile_metrics,
        "gpu_kernel_time_breakdown": trace_summary,
        "implementation_findings": implementation,
        "diagnosis": _diagnose_profile(
            trace_summary,
            profile_metrics,
            implementation,
        ),
        "notes": [
            "The same-shape baseline and profiled request execute only after all "
            "formal untraced latency measurements; both are excluded from speedup.",
            "profiler_overhead_ratio compares the profiled request with the immediately "
            "preceding unprofiled request using the same batch and output-token cap.",
            "Use the formal untraced measurements for performance conclusions.",
            "The stage table is GPU-kernel-time attribution, not synchronized "
            "end-to-end phase time. The raw trace is retained for Perfetto inspection.",
        ],
    }


def _build_engine(config: dict[str, Any]) -> tuple[Any, Any, float]:
    if not (3, 10) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError(
            "vllm==0.31.0 requires Python >=3.10,<3.15; "
            f"found Python {sys.version.split()[0]} at {sys.executable}"
        )
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - requires the GPU environment
        raise RuntimeError(
            "Install requirements-vllm.txt in a dedicated environment before "
            "running the vLLM benchmark"
        ) from exc

    installed = importlib.metadata.version("vllm")
    required = str(config.get("required_vllm_version", PINNED_VLLM))
    if installed != required:
        raise RuntimeError(f"This benchmark requires vllm=={required}; found {installed}")

    engine = config["engine"]
    kwargs: dict[str, Any] = {
        "model": config["models"]["target"],
        "tokenizer": config["models"]["target"],
        "dtype": engine["dtype"],
        "tensor_parallel_size": int(engine["tensor_parallel_size"]),
        "gpu_memory_utilization": float(engine["gpu_memory_utilization"]),
        "max_model_len": int(engine["max_model_len"]),
        "enable_prefix_caching": bool(engine["enable_prefix_caching"]),
        "enforce_eager": bool(engine["enforce_eager"]),
        "seed": int(engine["seed"]),
        "trust_remote_code": bool(engine["trust_remote_code"]),
        "disable_log_stats": False,
    }
    profiling = config.get("profiling", {})
    if bool(profiling.get("enabled")):
        profile_directory = Path(profiling["output_dir"]).expanduser().resolve()
        profile_directory.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_directory),
            "torch_profiler_with_stack": True,
            "torch_profiler_with_flops": False,
            "torch_profiler_use_gzip": True,
            "torch_profiler_dump_cuda_time_total": True,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
        }
    if engine.get("max_num_seqs") is not None:
        kwargs["max_num_seqs"] = int(engine["max_num_seqs"])
    target_revision = config["revisions"].get("target")
    if target_revision is not None:
        kwargs["revision"] = target_revision
        kwargs["tokenizer_revision"] = target_revision

    if config["engine_mode"] == "speculative":
        speculative_config: dict[str, Any] = {
            "method": "draft_model",
            "model": config["models"]["draft"],
            "num_speculative_tokens": int(
                config["settings"]["num_speculative_tokens"]
            ),
            "draft_sample_method": config["settings"]["draft_sample_method"],
            "rejection_sample_method": "standard",
            "max_model_len": int(engine["max_model_len"]),
        }
        draft_revision = config["revisions"].get("draft")
        if draft_revision is not None:
            speculative_config["revision"] = draft_revision
        kwargs["speculative_config"] = speculative_config
    elif config["engine_mode"] != "target_only":
        raise ValueError(f"Unknown engine_mode: {config['engine_mode']!r}")

    load_started_ns = time.perf_counter_ns()
    llm = LLM(**kwargs)
    load_ms = (time.perf_counter_ns() - load_started_ns) / 1_000_000.0
    return llm, SamplingParams, load_ms


def run_worker(config: dict[str, Any]) -> dict[str, Any]:
    llm, SamplingParams, engine_load_ms = _build_engine(config)
    try:
        modes: dict[str, Any] = {}
        benchmark = config["benchmark"]
        for mode in benchmark["modes"]:
            modes[mode] = _run_mode(
                llm,
                SamplingParams,
                config["prompts"],
                mode=mode,
                settings=config["settings"],
                warmup=int(benchmark["warmup_runs_per_batch"]),
                repeats=int(benchmark["measured_runs_per_batch"]),
                batch_size=int(benchmark["batch_size"]),
            )
        profile_config = config.get("profiling", {})
        if bool(profile_config.get("enabled")):
            profile_mode = str(profile_config["mode"])
            profiling = _run_profile(
                llm,
                SamplingParams,
                config["prompts"],
                mode=profile_mode,
                settings=config["settings"],
                profile_config=profile_config,
                batch_size=int(benchmark["batch_size"]),
                untraced_mode_result=modes[profile_mode],
            )
        else:
            profiling = {"enabled": False}
        import torch

        runtime = _collect_runtime_metadata(torch)
    finally:
        # Process exit is the actual isolation boundary.  Do not rely on private
        # vLLM teardown APIs whose names change between releases.
        del llm

    return {
        "schema_version": 1,
        "engine_mode": config["engine_mode"],
        "runtime": runtime,
        "engine_load_ms": engine_load_ms,
        "engine": config["engine"],
        "models": config["models"],
        "revisions": config["revisions"],
        "settings": config["settings"],
        "benchmark": config["benchmark"],
        "modes": modes,
        "profiling": profiling,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal isolated vLLM benchmark worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_worker(_read_json(args.config))
    _write_json(args.output, result)
    print(
        json.dumps(
            {
                "engine_mode": result["engine_mode"],
                "engine_load_ms": result["engine_load_ms"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
