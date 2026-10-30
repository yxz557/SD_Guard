#!/usr/bin/env python3
"""Quickly compare several Qwen3Guard-Stream checkpoints on shared samples.

This is intentionally a small streaming-only benchmark.  Every checkpoint uses
the same random manifest and the existing ``guard_eval`` implementation for
chat templating, true token-by-token moderation, latency timing, policies, and
binary safety metrics.
"""

from __future__ import annotations

import argparse
import gc
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from guard_eval.dataset import load_guard_samples
from guard_eval.latency_metrics import summarize_latencies
from guard_eval.model_loader import (
    collect_runtime_metadata,
    load_stream_guard,
    require_single_cuda_model,
)
from guard_eval.policies import evaluate_all_policies
from guard_eval.result_io import (
    JsonlWriter,
    read_json,
    write_csv,
    write_json,
)
from guard_eval.safety_metrics import evaluate_policy_metrics
from guard_eval.sampling import create_sample_manifest, select_samples_from_manifest
from guard_eval.sequence import prepare_sequence
from guard_eval.stream_runner import run_stream_sample, warmup_stream_sample


DEFAULT_CHECKPOINT_NAMES = (
    "Qwen3Guard-Stream-0.6B",
    "Qwen3Guard-Stream-4B",
    "Qwen3Guard-Stream-8B",
)
OUTPUT_FILENAMES = (
    "sample_manifest.json",
    "summary.json",
    "metrics.csv",
    "sample_results.jsonl",
    "token_records.jsonl",
)
POLICY_NAMES = ("strict", "loose")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Qwen3Guard-Stream checkpoints with the same small random "
            "thinking subset and true token-by-token inference."
        )
    )
    parser.add_argument("--dataset-path", required=True)
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument(
        "--model-root",
        type=Path,
        help=(
            "Directory containing Qwen3Guard-Stream-0.6B, 4B, and 8B. "
            "All three are tested in that order."
        ),
    )
    model_source.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="[NAME=]PATH",
        help=(
            "Checkpoint to test. Repeat this option for multiple models, for "
            "example --model 0.6B=/path/to/model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/quick-stream-compare"),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
        help="Random thinking samples when no manifest is supplied (default: 8).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help="Reuse an existing manifest instead of drawing another subset.",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--warmup-response-tokens", type=int, default=8)
    parser.add_argument(
        "--max-response-tokens",
        type=int,
        help=(
            "Optional speed limit per response. FNR then describes only the "
            "evaluated prefix and can overestimate full-response misses."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32", "auto"),
    )
    parser.add_argument("--attn-implementation")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _display_name(model_path: str) -> str:
    name = Path(model_path).expanduser().name
    return name or model_path.rstrip("/").rsplit("/", 1)[-1]


def _parse_model_spec(raw: str) -> ModelSpec:
    if "=" in raw:
        name, path = raw.split("=", 1)
        name, path = name.strip(), path.strip()
    else:
        path = raw.strip()
        name = _display_name(path)
    if not name or not path:
        raise ValueError(f"Invalid --model value {raw!r}; expected [NAME=]PATH")
    return ModelSpec(name=name, path=path)


def resolve_models(args: argparse.Namespace) -> list[ModelSpec]:
    if args.model_root is not None:
        root = args.model_root.expanduser()
        models = [
            ModelSpec(name=name, path=str(root / name))
            for name in DEFAULT_CHECKPOINT_NAMES
        ]
    else:
        models = [_parse_model_spec(value) for value in args.models]

    names = [model.name for model in models]
    if len(names) != len(set(names)):
        raise ValueError(f"Model names must be unique, received: {names}")
    return models


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. "
            "Pass --overwrite to replace this script's output files."
        )
    destination.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for filename in OUTPUT_FILENAMES:
            target = destination / filename
            if target.is_file():
                target.unlink()
    return destination


def _policy_pair(
    decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "strict": dict(decisions["online_strict"]),
        "loose": dict(decisions["online_loose"]),
    }


def _classification_outcome(prediction: str, target: str) -> str:
    if prediction == "Unsafe":
        return "TP" if target == "Unsafe" else "FP"
    return "FN" if target == "Unsafe" else "TN"


def _prompt_latency_summary(values: list[float]) -> dict[str, int | float | None]:
    summary = summarize_latencies(values)
    summary["sample_count"] = summary.pop("token_count")
    summary.pop("tokens_per_second")
    return summary


def _metric_rows(
    model_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    latency = model_summary["streaming_token_latency_ms"]
    common = {
        "model": model_summary["model"],
        "model_path": model_summary["model_path"],
        "sample_count": model_summary["sample_count"],
        "evaluated_response_tokens": latency["token_count"],
        "truncated_sample_count": model_summary["truncated_sample_count"],
        "mean_token_latency_ms": latency["mean_ms"],
        "p50_token_latency_ms": latency["p50_ms"],
        "p95_token_latency_ms": latency["p95_ms"],
        "p99_token_latency_ms": latency["p99_ms"],
        "guard_tokens_per_second": latency["tokens_per_second"],
        "model_load_seconds": model_summary["model_load_seconds"],
        "evaluation_wall_seconds": model_summary["evaluation_wall_seconds"],
    }
    return [
        {**common, "policy": policy_name, **model_summary["metrics"][policy_name]}
        for policy_name in POLICY_NAMES
    ]


def _format_rate(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.1f}%"


def print_model_summary(summary: Mapping[str, Any]) -> None:
    latency = summary["streaming_token_latency_ms"]
    print(
        f"[{summary['model']}] token latency: mean={latency['mean_ms']:.3f} ms, "
        f"p50={latency['p50_ms']:.3f} ms, p95={latency['p95_ms']:.3f} ms, "
        f"p99={latency['p99_ms']:.3f} ms"
    )
    for policy_name in POLICY_NAMES:
        metrics = summary["metrics"][policy_name]
        print(
            f"[{summary['model']}] {policy_name}: "
            f"TP={metrics['tp']} TN={metrics['tn']} "
            f"FP={metrics['fp']} FN={metrics['fn']} "
            f"FPR={_format_rate(metrics['fpr'])} "
            f"FNR={_format_rate(metrics['fnr'])} "
            f"F1={metrics['f1']:.3f}"
        )


def release_cuda_memory() -> None:
    """Release cached allocations before loading the next checkpoint."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.warmup_repeats < 0 or args.warmup_response_tokens < 0:
        raise ValueError("Warmup counts must be non-negative")
    if args.max_response_tokens is not None and args.max_response_tokens <= 0:
        raise ValueError("--max-response-tokens must be positive when supplied")

    models = resolve_models(args)
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)

    print(f"Loading thinking split: {args.dataset_path}")
    all_samples = load_guard_samples(
        args.dataset_path,
        "thinking",
        cache_dir=args.cache_dir,
    )
    if args.sample_manifest is None:
        manifest = create_sample_manifest(
            all_samples,
            sample_count=args.sample_count,
            seed=args.seed,
            dataset_path=args.dataset_path,
        )
        manifest_source = None
    else:
        manifest_path = args.sample_manifest.expanduser().resolve()
        manifest = read_json(manifest_path)
        manifest_source = str(manifest_path)
    selected = select_samples_from_manifest(all_samples, manifest)
    write_json(output_dir / "sample_manifest.json", manifest)

    label_counts = Counter(sample.label for _, _, sample in selected)
    print(
        f"Selected {len(selected)} shared samples with seed={manifest.get('seed')}: "
        f"Safe={label_counts.get('Safe', 0)}, Unsafe={label_counts.get('Unsafe', 0)}"
    )
    if not label_counts.get("Safe") or not label_counts.get("Unsafe"):
        print(
            "Warning: one target class is absent; either FPR or FNR will be N/A. "
            "Choose another seed for a two-class smoke test."
        )
    if args.max_response_tokens is not None:
        print(
            "Warning: response truncation is enabled. Accuracy and FNR describe "
            "only each evaluated prefix, not the full response."
        )

    started_at = datetime.now(timezone.utc).isoformat()
    run_summary: dict[str, Any] = {
        "status": "running",
        "started_at_utc": started_at,
        "dataset_path": str(args.dataset_path),
        "split": "thinking",
        "sample_count": len(selected),
        "label_counts": dict(label_counts),
        "seed": manifest.get("seed"),
        "manifest_source": manifest_source,
        "manifest_path": str(output_dir / "sample_manifest.json"),
        "model_order": [model.name for model in models],
        "dtype_argument": args.dtype,
        "attention_implementation_argument": args.attn_implementation,
        "warmup_repeats": args.warmup_repeats,
        "warmup_response_tokens": args.warmup_response_tokens,
        "max_response_tokens": args.max_response_tokens,
        "evaluation_scope": (
            "truncated response prefix"
            if args.max_response_tokens is not None
            else "complete response"
        ),
        "latency_scope": (
            "CUDA sync -> stream_moderate_from_ids -> CUDA sync -> risk label extraction"
        ),
        "models": [],
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, run_summary)

    metric_rows: list[dict[str, Any]] = []
    sample_path = output_dir / "sample_results.jsonl"
    token_path = output_dir / "token_records.jsonl"

    try:
        with JsonlWriter(sample_path) as sample_writer, JsonlWriter(
            token_path
        ) as token_writer:
            for model_index, spec in enumerate(models, start=1):
                print(f"\n[{model_index}/{len(models)}] Loading {spec.name}: {spec.path}")
                loaded = None
                try:
                    load_started = time.perf_counter()
                    loaded = load_stream_guard(
                        spec.path,
                        dtype=args.dtype,
                        device_map={"": 0},
                        attention_implementation=args.attn_implementation,
                        local_files_only=args.local_files_only,
                        revision=args.revision,
                    )
                    model_load_seconds = time.perf_counter() - load_started
                    model_device = require_single_cuda_model(loaded)
                    runtime_metadata = collect_runtime_metadata(loaded)
                    print(
                        f"Verified {spec.name} on {model_device}; "
                        f"load time={model_load_seconds:.1f}s"
                    )

                    model_sample_rows: list[dict[str, Any]] = []
                    token_latencies: list[float] = []
                    prompt_latencies: list[float] = []
                    warmed_up = False
                    evaluation_started = time.perf_counter()

                    try:
                        from tqdm.auto import tqdm
                    except ImportError:  # pragma: no cover - requirements include tqdm
                        def tqdm(iterable: Any, **_: Any) -> Any:
                            return iterable

                    for sample_order, dataset_index, sample in tqdm(
                        selected,
                        desc=f"{spec.name} · token stream",
                    ):
                        sequence = prepare_sequence(
                            loaded.tokenizer,
                            sample.messages,
                            "thinking",
                            annotated_input_ids=sample.input_ids,
                            require_alignment=False,
                        )
                        if not warmed_up and args.warmup_repeats:
                            warmup_stream_sample(
                                loaded,
                                sequence,
                                repeats=args.warmup_repeats,
                                response_tokens=args.warmup_response_tokens,
                            )
                            warmed_up = True

                        stream = run_stream_sample(
                            loaded,
                            sequence,
                            sample.unique_id,
                            model_name=spec.name,
                            max_response_tokens=args.max_response_tokens,
                        )
                        all_decisions = evaluate_all_policies(
                            stream.risk_levels,
                            full_index_offset=sequence.eval_start_index,
                        )
                        policies = _policy_pair(all_decisions)
                        outcomes = {
                            policy_name: _classification_outcome(
                                policies[policy_name]["prediction"], sample.label
                            )
                            for policy_name in POLICY_NAMES
                        }

                        sample_row = {
                            "model": spec.name,
                            "model_path": spec.path,
                            "split": "thinking",
                            "sample_id": sample.unique_id,
                            "sample_order": sample_order,
                            "dataset_index": dataset_index,
                            "label": sample.label,
                            "source": sample.source,
                            "unsafe_type": sample.unsafe_type,
                            "total_sequence_token_count": len(sequence.input_ids),
                            "prompt_token_count": sequence.prompt_token_count,
                            "bridge_token_count": sequence.bridge_token_count,
                            "available_response_token_count": (
                                stream.available_response_token_count
                            ),
                            "evaluated_response_token_count": stream.response_token_count,
                            "truncated": stream.truncated,
                            "mean_context_length": stream.mean_context_length,
                            "mean_token_latency_ms": stream.mean_token_latency_ms,
                            "prompt_prefill_latency_ms": (
                                stream.prompt.prompt_prefill_latency_ms
                            ),
                            "policy_decisions": policies,
                            "classification_outcomes": outcomes,
                            "alignment": {
                                "checked": sequence.alignment.checked,
                                "matches": sequence.alignment.matches,
                                "first_mismatch_index": (
                                    sequence.alignment.first_mismatch_index
                                ),
                            },
                        }
                        model_sample_rows.append(sample_row)
                        sample_writer.write(sample_row)

                        prompt_latencies.append(
                            stream.prompt.prompt_prefill_latency_ms
                        )
                        for token_record in stream.token_records:
                            token_latencies.append(token_record.latency_ms)
                            token_writer.write(
                                {
                                    **token_record.to_dict(),
                                    "sample_order": sample_order,
                                    "dataset_index": dataset_index,
                                }
                            )
                        sample_writer.flush()
                        token_writer.flush()

                    evaluation_wall_seconds = time.perf_counter() - evaluation_started
                    metrics = evaluate_policy_metrics(
                        model_sample_rows,
                        policy_names=POLICY_NAMES,
                        decisions_field="policy_decisions",
                    )
                    model_summary = {
                        "status": "completed",
                        "model": spec.name,
                        "model_path": spec.path,
                        "device": str(model_device),
                        "sample_count": len(model_sample_rows),
                        "truncated_sample_count": sum(
                            bool(row["truncated"]) for row in model_sample_rows
                        ),
                        "model_load_seconds": model_load_seconds,
                        "evaluation_wall_seconds": evaluation_wall_seconds,
                        "runtime": runtime_metadata,
                        "streaming_token_latency_ms": summarize_latencies(
                            token_latencies
                        ),
                        "prompt_prefill_latency_ms": _prompt_latency_summary(
                            prompt_latencies
                        ),
                        "metrics": metrics,
                    }
                    run_summary["models"].append(model_summary)
                    metric_rows.extend(_metric_rows(model_summary))
                    write_csv(output_dir / "metrics.csv", metric_rows)
                    write_json(summary_path, run_summary)
                    print_model_summary(model_summary)
                except BaseException as exc:
                    run_summary["status"] = "failed"
                    run_summary["failed_model"] = spec.name
                    run_summary["error_type"] = type(exc).__name__
                    run_summary["error_message"] = str(exc)
                    run_summary["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
                    write_json(summary_path, run_summary)
                    raise
                finally:
                    loaded = None
                    release_cuda_memory()

        run_summary["status"] = "completed"
        run_summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(summary_path, run_summary)
    except BaseException:
        raise

    print(f"\nCompleted {len(models)} model(s) on {len(selected)} shared samples")
    print(f"Summary: {summary_path}")
    print(f"Flat comparison: {output_dir / 'metrics.csv'}")
    print(f"Per-sample outcomes: {sample_path}")
    print(f"Per-token latency: {token_path}")


if __name__ == "__main__":
    main()
