#!/usr/bin/env python3
"""Run one controlled generation mode and offline Stream Guard evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import time
import traceback
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sdguard_sd.dataset import create_manifest, load_prompt_samples, select_from_manifest
from sdguard_sd.generation import generate_directly, generate_speculatively, stable_sample_seed
from sdguard_sd.guard import run_guard_full_forward
from sdguard_sd.inline_guard import IncrementalGuardMonitor, aggregate_inline_records
from sdguard_sd.io_utils import JsonlWriter, iter_jsonl, read_json, write_json
from sdguard_sd.runtime import (
    assert_same_tokenizer,
    collect_runtime_metadata,
    load_generation_model,
    load_guard_model,
)
from sdguard_sd.text_split import split_completion


PINNED_TRANSFORMERS = "4.57.6"
GENERATION_MODES = ("target_only", "speculative", "draft_only", "inline_guard")
SPECULATIVE_MODES = {"speculative", "inline_guard"}
REFERENCE_CONFIG_KEYS = (
    "dataset_path",
    "draft_model",
    "target_model",
    "guard_model",
    "sample_count",
    "seed",
    "max_new_tokens",
    "num_assistant_tokens",
    "assistant_confidence_threshold",
    "temperature",
    "top_p",
    "top_k",
    "do_sample",
    "target_device",
    "draft_device",
    "guard_device",
    "dtype",
    "generation_attn_implementation",
    "guard_attn_implementation",
    "draft_revision",
    "target_revision",
    "guard_revision",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Qwen3 answers in one controlled mode. inline_guard inserts an "
            "alarm-only Stream Guard between Draft proposal and Target verification."
        )
    )
    parser.add_argument(
        "--generation-mode",
        choices=GENERATION_MODES,
        default="speculative",
    )
    parser.add_argument(
        "--dataset-path",
        default="/home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--draft-model",
        default="/home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2",
    )
    parser.add_argument(
        "--target-model",
        default="/home/model_and_data/yxz/model/Huihui-Qwen3-8B-abliterated-v2",
    )
    parser.add_argument(
        "--guard-model",
        default="/home/model_and_data/yxz/model/Qwen3Guard-Stream-4B",
    )
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help="Reuse an existing sample manifest instead of drawing a new subset.",
    )
    parser.add_argument(
        "--reference-experiment-dir",
        type=Path,
        help=(
            "Required for inline_guard: the matching existing speculative cell. "
            "Generated IDs are asserted equal before its human labels may be reused."
        ),
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--num-assistant-tokens", type=int, default=5)
    parser.add_argument("--assistant-confidence-threshold", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:0")
    parser.add_argument("--guard-device", default="cuda:1")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32", "auto"),
    )
    parser.add_argument("--generation-attn-implementation")
    parser.add_argument("--guard-attn-implementation")
    parser.add_argument("--draft-revision")
    parser.add_argument("--target-revision")
    parser.add_argument("--guard-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def _check_arguments(args: argparse.Namespace) -> None:
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.num_assistant_tokens <= 0:
        raise ValueError("--num-assistant-tokens must be positive")
    if not 0 <= args.assistant_confidence_threshold <= 1:
        raise ValueError("--assistant-confidence-threshold must be in [0, 1]")
    if args.generation_mode == "inline_guard" and args.reference_experiment_dir is None:
        raise ValueError("inline_guard requires --reference-experiment-dir")
    if args.do_sample:
        if args.temperature <= 0:
            raise ValueError("--temperature must be positive")
        if not 0 < args.top_p <= 1:
            raise ValueError("--top-p must be in (0, 1]")
        if args.top_k < 0:
            raise ValueError("--top-k must be non-negative")
    installed = importlib.metadata.version("transformers")
    if installed != PINNED_TRANSFORMERS:
        raise RuntimeError(
            f"This initial implementation traces a private Transformers verifier API and "
            f"therefore requires transformers=={PINNED_TRANSFORMERS}; found {installed}."
        )


def _prepare_output(path: Path, overwrite: bool) -> tuple[Path, Path, Path, Path]:
    destination = path.expanduser().resolve()
    raw_dir = destination / "raw"
    blind_dir = destination / "blind"
    private_dir = destination / "private"
    known_outputs = (
        destination / "sample_manifest.json",
        destination / "run_config.json",
        raw_dir / "run_metadata.json",
        raw_dir / "generation_records.jsonl",
        raw_dir / "guard_token_records.jsonl",
        raw_dir / "sd_round_records.jsonl",
        raw_dir / "inline_guard_token_records.jsonl",
        raw_dir / "inline_guard_round_records.jsonl",
        raw_dir / "inline_guard_summary.json",
        blind_dir / "blind_outputs.jsonl",
        private_dir / "original_dataset_audit.jsonl",
        destination / "analysis" / "metrics.json",
        destination / "analysis" / "metrics.csv",
        destination / "analysis" / "merged_results.jsonl",
        destination / "analysis" / "errors.jsonl",
    )
    existing = [item for item in known_outputs if item.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(item) for item in existing[:5])
        raise FileExistsError(f"Experiment outputs already exist: {rendered}; pass --overwrite")
    if overwrite:
        for item in existing:
            item.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    blind_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    return destination, raw_dir, blind_dir, private_dir


def _guard_record(result: Any) -> dict[str, Any]:
    return {
        "prompt_risk_level": result.prompt_risk_level,
        "prompt_risk_probability": result.prompt_risk_probability,
        "latency_ms": result.latency_ms,
        "scopes": result.scopes,
    }


def _load_inline_reference(
    reference_dir: Path,
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    root = reference_dir.expanduser().resolve()
    config = read_json(root / "run_config.json")
    if config.get("generation_mode") != "speculative":
        raise ValueError(f"Inline reference is not a speculative cell: {root}")
    if bool(config.get("enable_thinking")) != bool(args.enable_thinking):
        raise ValueError("Inline reference uses a different thinking setting")
    current = _serializable_args(args)
    changed = [key for key in REFERENCE_CONFIG_KEYS if config.get(key) != current.get(key)]
    if changed:
        raise ValueError(f"Inline reference has different controlled settings: {changed}")
    if read_json(root / "sample_manifest.json") != manifest:
        raise ValueError("Inline reference sample_manifest.json differs from this run")
    rows = list(iter_jsonl(root / "raw" / "generation_records.jsonl"))
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Inline reference contains duplicate sample IDs")
    return root, by_id


def _assert_reference_output(
    reference: dict[str, Any],
    *,
    sample_id: str,
    sample_order: int,
    sample_seed: int,
    generated_ids: tuple[int, ...],
    completion: Any,
) -> None:
    if int(reference["sample_order"]) != sample_order:
        raise RuntimeError(f"Reference sample order differs for {sample_id}")
    if int(reference["sample_seed"]) != sample_seed:
        raise RuntimeError(f"Reference sample seed differs for {sample_id}")
    reference_ids = tuple(int(value) for value in reference["generation"]["generated_ids"])
    if reference_ids != generated_ids:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(reference_ids, generated_ids))
                if left != right
            ),
            min(len(reference_ids), len(generated_ids)),
        )
        raise RuntimeError(
            f"Inline output differs from speculative reference for {sample_id} at "
            f"generated token {mismatch}; blind labels cannot be reused"
        )
    for key in ("thinking", "final_answer", "full_completion", "parse_status"):
        if reference["completion"].get(key) != getattr(completion, key):
            raise RuntimeError(f"Reference completion field {key!r} differs for {sample_id}")


def _assert_blind_record(record: dict[str, Any]) -> None:
    forbidden = ("guard", "risk", "prediction", "original", "dataset_label")
    for key in record:
        lowered = key.casefold()
        if any(word in lowered for word in forbidden):
            raise AssertionError(f"Blind record leaks a protected field: {key!r}")


def main() -> None:
    args = parse_args()
    _check_arguments(args)
    destination, raw_dir, blind_dir, private_dir = _prepare_output(
        args.output_dir, args.overwrite
    )
    write_json(destination / "run_config.json", _serializable_args(args))

    print(f"Loading Qwen3GuardTest thinking split: {args.dataset_path}")
    all_samples = load_prompt_samples(args.dataset_path, cache_dir=args.cache_dir)
    if args.sample_manifest is None:
        manifest = create_manifest(
            all_samples,
            dataset_path=args.dataset_path,
            sample_count=args.sample_count,
            seed=args.seed,
        )
        manifest_source = None
    else:
        manifest_path = args.sample_manifest.expanduser().resolve()
        manifest = read_json(manifest_path)
        manifest_source = str(manifest_path)
    selected = select_from_manifest(all_samples, manifest)
    write_json(destination / "sample_manifest.json", manifest)
    reference_root: Path | None = None
    reference_rows: dict[str, dict[str, Any]] = {}
    if args.generation_mode == "inline_guard":
        assert args.reference_experiment_dir is not None
        reference_root, reference_rows = _load_inline_reference(
            args.reference_experiment_dir,
            args=args,
            manifest=manifest,
        )
        selected_ids = {sample.sample_id for _, sample in selected}
        if set(reference_rows) != selected_ids:
            raise ValueError("Inline reference sample IDs differ from the selected manifest")

    metadata_path = raw_dir / "run_metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "status": "loading_models",
        "started_at_utc": _utc_now(),
        "generation_mode": args.generation_mode,
        "thinking_enabled": args.enable_thinking,
        "dataset_path": str(args.dataset_path),
        "dataset_split": "thinking",
        "sample_count": len(selected),
        "manifest_source": manifest_source,
        "models": {
            "draft": args.draft_model,
            "target": args.target_model,
            "guard": args.guard_model,
        },
        "sampling": {
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "per_sample_seed_strategy": "sha256(base_seed:sample_id)",
        },
        "speculative_decoding": (
            {
                "implementation": "transformers assisted generation with speculative sampling",
                "verifier": "target distribution rejection sampling with residual correction",
                "num_assistant_tokens": args.num_assistant_tokens,
                "assistant_schedule": "constant",
                "assistant_confidence_threshold": args.assistant_confidence_threshold,
            }
            if args.generation_mode in SPECULATIVE_MODES
            else None
        ),
        "guard_evaluation": {
            "mode": (
                "alarm-only inline proposal/commit monitoring plus final causal full forward"
                if args.generation_mode == "inline_guard"
                else "one causal full forward after complete generation"
            ),
            "strict_policy": "earliest adjacent UU or CC pair",
            "loose_policy": "earliest adjacent UU pair",
            "scopes": ["full_trace", "answer_only"],
            "live_truncation": False,
            "generation_intervention": (
                "none; alarms latch but generation always continues"
                if args.generation_mode == "inline_guard"
                else None
            ),
        },
        "reference_experiment_dir": str(reference_root) if reference_root else None,
    }
    write_json(metadata_path, metadata)
    wall_started = time.perf_counter()

    try:
        target_model = None
        target_tokenizer = None
        draft_model = None
        draft_tokenizer = None
        model_placement: dict[str, str] = {}

        if args.generation_mode in {"target_only", *SPECULATIVE_MODES}:
            print(f"Loading target model on {args.target_device}: {args.target_model}")
            target_model, target_tokenizer = load_generation_model(
                args.target_model,
                device=args.target_device,
                dtype=args.dtype,
                local_files_only=args.local_files_only,
                revision=args.target_revision,
                attention_implementation=args.generation_attn_implementation,
            )
            model_placement["target"] = args.target_device

        if args.generation_mode in {"draft_only", *SPECULATIVE_MODES}:
            print(f"Loading draft model on {args.draft_device}: {args.draft_model}")
            draft_model, draft_tokenizer = load_generation_model(
                args.draft_model,
                device=args.draft_device,
                dtype=args.dtype,
                local_files_only=args.local_files_only,
                revision=args.draft_revision,
                attention_implementation=args.generation_attn_implementation,
            )
            model_placement["draft"] = args.draft_device

        if args.generation_mode in SPECULATIVE_MODES:
            assert target_tokenizer is not None and draft_tokenizer is not None
            assert_same_tokenizer(
                target_tokenizer, draft_tokenizer, candidate_name="draft model"
            )
            generation_tokenizer = target_tokenizer
        elif args.generation_mode == "target_only":
            assert target_tokenizer is not None
            generation_tokenizer = target_tokenizer
        else:
            assert draft_tokenizer is not None
            generation_tokenizer = draft_tokenizer

        print(f"Loading Stream Guard on {args.guard_device}: {args.guard_model}")
        guard_model, guard_tokenizer = load_guard_model(
            args.guard_model,
            device=args.guard_device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            revision=args.guard_revision,
            attention_implementation=args.guard_attn_implementation,
        )
        assert_same_tokenizer(generation_tokenizer, guard_tokenizer, candidate_name="Stream Guard")
        model_placement["guard"] = args.guard_device

        metadata.update(collect_runtime_metadata())
        metadata["status"] = "running"
        metadata["model_placement"] = model_placement
        write_json(metadata_path, metadata)

        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover - requirements include tqdm
            def tqdm(iterable: Any, **_: Any) -> Any:
                return iterable

        generation_path = raw_dir / "generation_records.jsonl"
        guard_tokens_path = raw_dir / "guard_token_records.jsonl"
        rounds_path = raw_dir / "sd_round_records.jsonl"
        inline_tokens_path = raw_dir / "inline_guard_token_records.jsonl"
        inline_rounds_path = raw_dir / "inline_guard_round_records.jsonl"
        inline_summary_path = raw_dir / "inline_guard_summary.json"
        blind_path = blind_dir / "blind_outputs.jsonl"
        audit_path = private_dir / "original_dataset_audit.jsonl"
        processed = 0
        total_generated = 0
        total_proposed = 0
        total_accepted = 0
        inline_results: list[dict[str, Any]] = []

        with ExitStack() as stack:
            generation_writer = stack.enter_context(JsonlWriter(generation_path))
            guard_token_writer = stack.enter_context(JsonlWriter(guard_tokens_path))
            round_writer = stack.enter_context(JsonlWriter(rounds_path))
            inline_token_writer = (
                stack.enter_context(JsonlWriter(inline_tokens_path))
                if args.generation_mode == "inline_guard"
                else None
            )
            inline_round_writer = (
                stack.enter_context(JsonlWriter(inline_rounds_path))
                if args.generation_mode == "inline_guard"
                else None
            )
            blind_writer = stack.enter_context(JsonlWriter(blind_path))
            audit_writer = stack.enter_context(JsonlWriter(audit_path))

            description = f"{args.generation_mode} + Guard ({'think' if args.enable_thinking else 'no-think'})"
            for sample_order, sample in tqdm(selected, desc=description, unit="sample"):
                sample_seed = stable_sample_seed(args.seed, sample.sample_id)
                inline_monitor: IncrementalGuardMonitor | None = None
                if args.generation_mode == "inline_guard":
                    inline_monitor = IncrementalGuardMonitor(
                        guard_model,
                        guard_tokenizer,
                        guard_device=args.guard_device,
                        enable_thinking=args.enable_thinking,
                    )

                if args.generation_mode in SPECULATIVE_MODES:
                    assert target_model is not None and draft_model is not None
                    generation = generate_speculatively(
                        target_model,
                        draft_model,
                        generation_tokenizer,
                        sample.prompt,
                        target_device=args.target_device,
                        draft_device=args.draft_device,
                        sample_seed=sample_seed,
                        enable_thinking=args.enable_thinking,
                        max_new_tokens=args.max_new_tokens,
                        num_assistant_tokens=args.num_assistant_tokens,
                        assistant_confidence_threshold=args.assistant_confidence_threshold,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        round_observer=inline_monitor,
                    )
                elif args.generation_mode == "target_only":
                    assert target_model is not None
                    generation = generate_directly(
                        target_model,
                        generation_tokenizer,
                        sample.prompt,
                        device=args.target_device,
                        sample_seed=sample_seed,
                        enable_thinking=args.enable_thinking,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                else:
                    assert draft_model is not None
                    generation = generate_directly(
                        draft_model,
                        generation_tokenizer,
                        sample.prompt,
                        device=args.draft_device,
                        sample_seed=sample_seed,
                        enable_thinking=args.enable_thinking,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                completion = split_completion(
                    generation_tokenizer,
                    generation.generated_ids,
                    thinking_expected=args.enable_thinking,
                )
                guard = run_guard_full_forward(
                    guard_model,
                    guard_tokenizer,
                    generation.prompt_ids,
                    generation.generated_ids,
                    answer_start_generated_index=completion.answer_start_generated_index,
                    guard_device=args.guard_device,
                )

                inline_result: dict[str, Any] | None = None
                if inline_monitor is not None:
                    inline_result = inline_monitor.finalize(
                        generated_ids=generation.generated_ids,
                        answer_start_generated_index=completion.answer_start_generated_index,
                        posthoc_guard=guard,
                    )
                    consistency = inline_result["consistency"]
                    if not consistency["verified_posthoc_exact_match"]:
                        print(
                            "Warning: incremental and posthoc Guard labels differ for "
                            f"{sample.sample_id}: token_label_mismatches="
                            f"{consistency['token_label_mismatch_count']}, "
                            "scope_decisions_match="
                            f"{consistency['verified_scope_decisions_match_posthoc']}. "
                            "The difference is recorded and generation will continue."
                        )
                    reference = reference_rows[sample.sample_id]
                    _assert_reference_output(
                        reference,
                        sample_id=sample.sample_id,
                        sample_order=sample_order,
                        sample_seed=sample_seed,
                        generated_ids=generation.generated_ids,
                        completion=completion,
                    )
                    reference_latency = float(reference["generation"]["latency_ms"])
                    inline_result["reference_equivalence"] = {
                        "reference_experiment_dir": str(reference_root),
                        "generated_ids_match": True,
                        "completion_match": True,
                        "human_labels_reusable": True,
                    }
                    inline_result["timing"].update(
                        {
                            "reference_speculative_generation_latency_ms": reference_latency,
                            "inline_end_to_end_generation_latency_ms": generation.latency_ms,
                            "observed_generation_overhead_ms": (
                                generation.latency_ms - reference_latency
                            ),
                        }
                    )
                    inline_results.append(inline_result)

                generation_record = {
                    "schema_version": 2,
                    "sample_id": sample.sample_id,
                    "sample_order": sample_order,
                    "dataset_index": sample.dataset_index,
                    "sample_seed": sample_seed,
                    "generation_mode": args.generation_mode,
                    "thinking_enabled": args.enable_thinking,
                    "prompt": sample.prompt,
                    "completion": asdict(completion),
                    "generation": {
                        "prompt_token_count": len(generation.prompt_ids),
                        "generated_token_count": generation.generated_token_count,
                        "generated_ids": list(generation.generated_ids),
                        "latency_ms": generation.latency_ms,
                        "tokens_per_second": generation.tokens_per_second,
                        "stopped_on_eos": generation.stopped_on_eos,
                        "truncated": generation.truncated,
                    },
                    "speculative_decoding": (
                        {
                            "round_count": generation.round_count,
                            "proposed_token_count": generation.proposed_token_count,
                            "accepted_draft_token_count": generation.accepted_draft_token_count,
                            "acceptance_rate": generation.acceptance_rate,
                        }
                        if args.generation_mode in SPECULATIVE_MODES
                        else None
                    ),
                    "guard": _guard_record(guard),
                    "inline_guard": inline_result,
                }
                generation_writer.write(generation_record)

                for token in guard.token_records:
                    guard_token_writer.write(
                        {
                            "sample_id": sample.sample_id,
                            "sample_order": sample_order,
                            **token,
                        }
                    )
                for round_ in generation.rounds:
                    round_writer.write(
                        {
                            "sample_id": sample.sample_id,
                            "sample_order": sample_order,
                            **round_.to_dict(generation_tokenizer),
                        }
                    )
                if inline_monitor is not None:
                    assert inline_token_writer is not None
                    assert inline_round_writer is not None
                    for token in inline_monitor.tracker.token_records:
                        inline_token_writer.write(
                            {
                                "sample_id": sample.sample_id,
                                "sample_order": sample_order,
                                **token,
                            }
                        )
                    for round_record in inline_monitor.tracker.round_records:
                        inline_round_writer.write(
                            {
                                "sample_id": sample.sample_id,
                                "sample_order": sample_order,
                                **round_record,
                            }
                        )

                blind_record = {
                    "schema_version": 2,
                    "sample_id": sample.sample_id,
                    "sample_order": sample_order,
                    "prompt": sample.prompt,
                    "thinking": completion.thinking,
                    "final_answer": completion.final_answer,
                    "full_completion": completion.full_completion,
                    "parse_status": completion.parse_status,
                    "truncated": generation.truncated,
                }
                _assert_blind_record(blind_record)
                blind_writer.write(blind_record)
                audit_writer.write(
                    {
                        "sample_id": sample.sample_id,
                        "sample_order": sample_order,
                        "dataset_index": sample.dataset_index,
                        "original_label": sample.original_label,
                        "original_source": sample.original_source,
                        "original_unsafe_type": sample.original_unsafe_type,
                    }
                )

                processed += 1
                total_generated += generation.generated_token_count
                if args.generation_mode in SPECULATIVE_MODES:
                    total_proposed += generation.proposed_token_count
                    total_accepted += generation.accepted_draft_token_count

        if args.generation_mode == "inline_guard":
            write_json(inline_summary_path, aggregate_inline_records(inline_results))
            assert reference_root is not None
            reference_judgments = reference_root / "blind" / "judge_labels.jsonl"
            if reference_judgments.exists():
                shutil.copyfile(reference_judgments, blind_dir / "judge_labels.jsonl")
                metadata["reused_judgments_from"] = str(reference_judgments)
            else:
                metadata["reused_judgments_from"] = None

        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "wall_time_seconds": time.perf_counter() - wall_started,
                "processed_sample_count": processed,
                "total_generated_tokens": total_generated,
                "total_draft_proposed_tokens": (
                    total_proposed if args.generation_mode in SPECULATIVE_MODES else None
                ),
                "total_draft_accepted_tokens": (
                    total_accepted if args.generation_mode in SPECULATIVE_MODES else None
                ),
                "aggregate_acceptance_rate": (
                    None
                    if args.generation_mode not in SPECULATIVE_MODES or total_proposed == 0
                    else total_accepted / total_proposed
                ),
                "blind_output": str(blind_path),
            }
        )
        write_json(metadata_path, metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        print(f"\nBlind outputs: {blind_path}")
        print(f"Full experiment: {destination}")
    except BaseException as exc:
        metadata.update(
            {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "wall_time_seconds": time.perf_counter() - wall_started,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(metadata_path, metadata)
        raise


if __name__ == "__main__":
    main()
