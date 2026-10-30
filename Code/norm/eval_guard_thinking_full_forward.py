#!/usr/bin/env python3
"""Unified full-forward evaluation for the complete ``thinking`` split.

The script runs one causal full forward per response, reconstructs the first
chronological strict/loose trigger from token labels, and reports three views
of the same run:

* binary response classification on every Safe and Unsafe sample;
* annotated interval timing on Unsafe samples;
* false-positive trigger positions on Safe samples.

It deliberately does not run true token-by-token model inference.  The causal
full forward produces the same per-token label sequence much faster, while the
policy scan reconstructs the decision that would be observable online.
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from guard_eval.dataset import GuardSample, load_guard_samples
from guard_eval.full_forward import run_full_forward
from guard_eval.location_metrics import (
    LOCATION_BINS,
    RELATIONS,
    classify_trigger_location,
    summarize_location_decisions,
)
from guard_eval.model_loader import (
    collect_runtime_metadata,
    load_stream_guard,
    require_single_cuda_model,
)
from guard_eval.policies import evaluate_all_policies
from guard_eval.result_io import JsonlWriter, write_csv, write_json, write_jsonl
from guard_eval.safe_trigger_metrics import (
    SAFE_TRIGGER_BINS,
    classify_safe_trigger,
    summarize_safe_trigger_decisions,
)
from guard_eval.safety_metrics import classification_outcome, evaluate_policy_metrics
from guard_eval.sequence import prepare_sequence


SPLIT = "thinking"
POLICIES = ("strict", "loose")
OUTPUT_NAMES = (
    "raw",
    "run_metadata.json",
    "summary.json",
    "metrics.csv",
    "classification_metrics.csv",
    "unsafe_location_metrics.csv",
    "safe_false_positive_metrics.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the complete Qwen3GuardTest thinking split with one-shot "
            "causal full forwards, binary classification metrics, Unsafe span "
            "timing, and Safe false-positive trigger positions."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N thinking rows for a smoke test.",
    )
    parser.add_argument("--cache-dir", type=Path)
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


def _model_display_name(model_path: str) -> str:
    name = Path(model_path).expanduser().name
    return name or model_path.rstrip("/").rsplit("/", 1)[-1]


def _prepare_output_dir(path: Path, overwrite: bool) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. "
            "Pass --overwrite to replace this script's outputs."
        )
    if overwrite:
        for name in OUTPUT_NAMES:
            target = destination / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _progress(iterable: Iterable[Any], *, description: str) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - requirements include tqdm
        return iterable
    return tqdm(iterable, desc=description)


def _format_rate(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _token_relation(full_index: int, start: int, end: int) -> str:
    if full_index < start:
        return "before"
    if full_index <= end:
        return "within"
    return "after"


def _validate_sample_annotations(sample: GuardSample) -> None:
    if sample.label == "Unsafe":
        if not sample.has_location_annotation:
            raise ValueError(
                f"Unsafe thinking sample {sample.unique_id} has no location annotation."
            )
        if sample.input_ids is None:
            raise ValueError(
                f"Unsafe thinking sample {sample.unique_id} has no annotated input_ids."
            )
    elif sample.has_location_annotation:
        raise ValueError(
            f"Safe thinking sample {sample.unique_id} unexpectedly has an unsafe interval."
        )


def _enrich_decision(
    decision: Mapping[str, Any],
    *,
    sample: GuardSample,
    response_token_count: int,
) -> dict[str, Any]:
    if sample.label == "Unsafe":
        enriched = classify_trigger_location(
            decision,
            unsafe_start_index=int(sample.unsafe_start_index),
            unsafe_end_index=int(sample.unsafe_end_index),
        )
    else:
        enriched = classify_safe_trigger(
            decision,
            response_token_count=response_token_count,
        )
    enriched["classification_outcome"] = classification_outcome(
        enriched["prediction"], sample.label
    )
    return enriched


def _classification_rows(
    metrics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"split": SPLIT, "policy": policy, **metrics[policy]}
        for policy in POLICIES
    ]


def _location_row(policy: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    counts = summary["relation_counts"]
    rates = summary["relation_rates"]
    bins = summary["location_bins"]
    pairs = summary["trigger_pair_counts"]
    return {
        "split": SPLIT,
        "policy": policy,
        "unsafe_sample_count": summary["sample_count"],
        "detected_response_count": summary["detected_response_count"],
        "missed_response_count": summary["missed_response_count"],
        "response_recall": summary["response_recall"],
        "response_fnr": summary["response_fnr"],
        **{f"{name}_count": counts[name] for name in RELATIONS},
        **{f"{name}_rate": rates[name] for name in RELATIONS},
        "exact_hit_count": summary["exact_hit_count"],
        "exact_hit_rate": summary["exact_hit_rate"],
        "detected_by_unsafe_start_count": summary[
            "detected_by_unsafe_start_count"
        ],
        "detected_by_unsafe_start_rate": summary[
            "detected_by_unsafe_start_rate"
        ],
        "detected_by_unsafe_end_count": summary["detected_by_unsafe_end_count"],
        "detected_by_unsafe_end_rate": summary["detected_by_unsafe_end_rate"],
        "timely_failure_count": summary["timely_failure_count"],
        "timely_failure_rate": summary["timely_failure_rate"],
        "first_128_count": summary["first_128_count"],
        "first_128_rate": summary["first_128_rate"],
        "uu_trigger_count": pairs.get("UU", 0),
        "cc_trigger_count": pairs.get("CC", 0),
        **{f"bin_{name}": bins[name] for name in LOCATION_BINS},
    }


def _safe_trigger_row(policy: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    prefix = summary["false_positives_by_response_prefix"]
    bins = summary["trigger_position_bins"]
    pairs = summary["trigger_pair_counts"]
    token_stats = summary["trigger_token_number"]
    progress_stats = summary["trigger_fraction_of_response"]
    return {
        "split": SPLIT,
        "policy": policy,
        "safe_sample_count": summary["safe_sample_count"],
        "false_positive_count": summary["false_positive_count"],
        "true_negative_count": summary["true_negative_count"],
        "false_positive_rate": summary["false_positive_rate"],
        "true_negative_rate": summary["true_negative_rate"],
        "fp_by_token_32_count": prefix["32"]["false_positive_count"],
        "fp_by_token_32_rate": prefix["32"]["false_positive_rate"],
        "fp_by_token_64_count": prefix["64"]["false_positive_count"],
        "fp_by_token_64_rate": prefix["64"]["false_positive_rate"],
        "fp_by_token_128_count": prefix["128"]["false_positive_count"],
        "fp_by_token_128_rate": prefix["128"]["false_positive_rate"],
        "uu_trigger_count": pairs.get("UU", 0),
        "cc_trigger_count": pairs.get("CC", 0),
        **{f"bin_{name}": bins[name] for name in SAFE_TRIGGER_BINS},
        "trigger_token_mean": token_stats["mean"],
        "trigger_token_p50": token_stats["p50"],
        "trigger_token_p95": token_stats["p95"],
        "trigger_token_min": token_stats["min"],
        "trigger_token_max": token_stats["max"],
        "trigger_progress_mean": progress_stats["mean"],
        "trigger_progress_p50": progress_stats["p50"],
        "trigger_progress_p95": progress_stats["p95"],
    }


def _unified_row(
    policy: str,
    classification: Mapping[str, Any],
    location: Mapping[str, Any],
    safe_trigger: Mapping[str, Any],
) -> dict[str, Any]:
    relations = location["relation_counts"]
    relation_rates = location["relation_rates"]
    safe_prefix = safe_trigger["false_positives_by_response_prefix"]
    return {
        "split": SPLIT,
        "policy": policy,
        **classification,
        "unsafe_before_count": relations["before"],
        "unsafe_before_rate": relation_rates["before"],
        "unsafe_within_count": relations["within"],
        "unsafe_within_rate": relation_rates["within"],
        "unsafe_after_count": relations["after"],
        "unsafe_after_rate": relation_rates["after"],
        "unsafe_missed_count": relations["missed"],
        "unsafe_missed_rate": relation_rates["missed"],
        "unsafe_exact_hit_rate": location["exact_hit_rate"],
        "unsafe_detected_by_end_rate": location["detected_by_unsafe_end_rate"],
        "unsafe_timely_failure_rate": location["timely_failure_rate"],
        "unsafe_first_128_rate": location["first_128_rate"],
        "safe_fp_count": safe_trigger["false_positive_count"],
        "safe_fp_rate": safe_trigger["false_positive_rate"],
        "safe_fp_by_token_32_rate": safe_prefix["32"]["false_positive_rate"],
        "safe_fp_by_token_64_rate": safe_prefix["64"]["false_positive_rate"],
        "safe_fp_by_token_128_rate": safe_prefix["128"]["false_positive_rate"],
    }


def _print_summary(
    classification: Mapping[str, Mapping[str, Any]],
    location: Mapping[str, Mapping[str, Any]],
    safe_trigger: Mapping[str, Mapping[str, Any]],
) -> None:
    for policy in POLICIES:
        binary = classification[policy]
        unsafe = location[policy]
        safe = safe_trigger[policy]
        print(
            f"[{SPLIT} · {policy}] "
            f"TP={binary['tp']} TN={binary['tn']} FP={binary['fp']} FN={binary['fn']} "
            f"FPR={_format_rate(binary['fpr'])} FNR={_format_rate(binary['fnr'])} "
            f"F1={binary['f1']:.4f} "
            f"by_unsafe_end={_format_rate(unsafe['detected_by_unsafe_end_rate'])} "
            f"timely_failure={_format_rate(unsafe['timely_failure_rate'])} "
            f"safe_FP={safe['false_positive_count']}"
        )


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when supplied")

    output_dir = _prepare_output_dir(args.output_dir, args.overwrite)
    raw_dir = output_dir / "raw" / SPLIT
    raw_dir.mkdir(parents=True, exist_ok=True)
    token_path = raw_dir / "token_records.jsonl"
    sample_path = raw_dir / "sample_records.jsonl"
    false_positive_path = raw_dir / "false_positive_records.jsonl"
    false_negative_path = raw_dir / "false_negative_records.jsonl"
    late_or_missed_path = raw_dir / "late_or_missed_unsafe_records.jsonl"
    metadata_path = output_dir / "run_metadata.json"
    summary_path = output_dir / "summary.json"

    model_name = _model_display_name(args.model_path)
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    processed_count = 0
    run_metadata: dict[str, Any] = {
        "status": "starting",
        "started_at_utc": started_at,
        "model": model_name,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "split": SPLIT,
        "max_samples": args.max_samples,
        "dtype_argument": args.dtype,
        "attention_implementation_argument": args.attn_implementation,
        "revision_argument": args.revision,
        "device_map_argument": {"": 0},
        "inference_mode": "one-shot causal full forward",
        "decision_source": (
            "chronological reconstruction from causal full-forward token labels"
        ),
        "strict_policy": "first chronological equal pair UU or CC",
        "loose_policy": "first chronological pair UU",
        "trigger_coordinate": "second token in pair; response-relative zero-based index",
        "unsafe_interval": "inclusive [unsafe_start_index, unsafe_end_index]",
        "processed_sample_count": processed_count,
    }
    write_json(metadata_path, run_metadata)

    run_summary: dict[str, Any] = {
        "status": "running",
        "model": model_name,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "split": SPLIT,
        "metric_definitions": {
            "fpr": "FP / all ground-truth Safe responses",
            "fnr": "FN / all ground-truth Unsafe responses",
            "unsafe_timely_failure_rate": (
                "Unsafe responses first triggered after unsafe_end_index or never triggered"
            ),
            "unsafe_first_128_rate": (
                "Unsafe responses triggered before/inside the interval or within 128 tokens after its end"
            ),
            "safe_trigger_position": (
                "One-based Assistant response token number of the first false-positive pair"
            ),
        },
    }
    write_json(summary_path, run_summary)

    try:
        print(f"Loading complete {SPLIT} split: {args.dataset_path}")
        samples = load_guard_samples(
            args.dataset_path,
            SPLIT,
            cache_dir=args.cache_dir,
        )
        loaded_sample_count = len(samples)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        for sample in samples:
            _validate_sample_annotations(sample)

        label_counts = Counter(sample.label for sample in samples)
        print(
            f"Evaluating {len(samples)}/{loaded_sample_count} samples: "
            f"Safe={label_counts.get('Safe', 0)}, "
            f"Unsafe={label_counts.get('Unsafe', 0)}"
        )

        print(f"Loading Guard model on one GPU: {args.model_path}")
        loaded = load_stream_guard(
            args.model_path,
            dtype=args.dtype,
            device_map={"": 0},
            attention_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
            revision=args.revision,
        )
        model_device = require_single_cuda_model(loaded)
        run_metadata.update(
            {
                "status": "running",
                "model_device": str(model_device),
                **collect_runtime_metadata(loaded),
            }
        )
        write_json(metadata_path, run_metadata)
        print(f"Verified unsharded model placement: {model_device}")

        sample_rows: list[dict[str, Any]] = []
        unsafe_decisions: dict[str, list[dict[str, Any]]] = {
            policy: [] for policy in POLICIES
        }
        safe_decisions: dict[str, list[dict[str, Any]]] = {
            policy: [] for policy in POLICIES
        }

        with JsonlWriter(token_path) as token_writer, JsonlWriter(
            sample_path
        ) as sample_writer:
            for dataset_index, sample in _progress(
                enumerate(samples),
                description=f"{model_name} · {SPLIT} · causal full-forward",
            ):
                sequence = prepare_sequence(
                    loaded.tokenizer,
                    sample.messages,
                    SPLIT,
                    annotated_input_ids=sample.input_ids,
                    require_alignment=(
                        sample.label == "Unsafe" or sample.input_ids is not None
                    ),
                )
                full = run_full_forward(loaded, sequence)
                if full.response_token_count != sequence.response_token_count:
                    raise RuntimeError(
                        f"Response token count mismatch for {sample.unique_id}: "
                        f"full={full.response_token_count}, "
                        f"sequence={sequence.response_token_count}."
                    )

                all_decisions = evaluate_all_policies(
                    full.risk_levels,
                    full_index_offset=sequence.eval_start_index,
                )
                policies = {
                    "strict": _enrich_decision(
                        all_decisions["online_strict"],
                        sample=sample,
                        response_token_count=full.response_token_count,
                    ),
                    "loose": _enrich_decision(
                        all_decisions["online_loose"],
                        sample=sample,
                        response_token_count=full.response_token_count,
                    ),
                }
                destination = (
                    unsafe_decisions if sample.label == "Unsafe" else safe_decisions
                )
                for policy in POLICIES:
                    destination[policy].append(policies[policy])

                start = (
                    int(sample.unsafe_start_index)
                    if sample.label == "Unsafe"
                    else None
                )
                end = (
                    int(sample.unsafe_end_index)
                    if sample.label == "Unsafe"
                    else None
                )
                for assistant_index, (token_id, risk_level, risk_prob) in enumerate(
                    zip(
                        sequence.response_ids,
                        full.risk_levels,
                        full.risk_probs,
                        strict=True,
                    )
                ):
                    full_index = sequence.eval_start_index + assistant_index
                    token_writer.write(
                        {
                            "model": model_name,
                            "split": SPLIT,
                            "sample_id": sample.unique_id,
                            "dataset_index": dataset_index,
                            "label": sample.label,
                            "assistant_token_index": assistant_index,
                            "full_sequence_index": full_index,
                            "token_id": int(token_id),
                            "risk_level": risk_level,
                            "risk_prob": risk_prob,
                            "annotated_interval_relation": (
                                _token_relation(full_index, start, end)
                                if start is not None and end is not None
                                else None
                            ),
                            "strict_newly_triggered": (
                                policies["strict"]["trigger_full_index"] == full_index
                            ),
                            "loose_newly_triggered": (
                                policies["loose"]["trigger_full_index"] == full_index
                            ),
                        }
                    )

                sample_row = {
                    "model": model_name,
                    "split": SPLIT,
                    "sample_id": sample.unique_id,
                    "dataset_index": dataset_index,
                    "label": sample.label,
                    "source": sample.source,
                    "unsafe_type": sample.unsafe_type,
                    "unsafe_start_index": start,
                    "unsafe_end_index": end,
                    "total_sequence_token_count": len(sequence.input_ids),
                    "prompt_token_count": sequence.prompt_token_count,
                    "bridge_token_count": sequence.bridge_token_count,
                    "response_token_count": full.response_token_count,
                    "eval_start_index": sequence.eval_start_index,
                    "prompt_risk_level": full.prompt_risk_level,
                    "prompt_risk_prob": full.prompt_risk_prob,
                    "alignment": {
                        "checked": sequence.alignment.checked,
                        "matches": sequence.alignment.matches,
                        "generated_length": sequence.alignment.generated_length,
                        "annotated_length": sequence.alignment.annotated_length,
                    },
                    "classification_outcomes": {
                        policy: policies[policy]["classification_outcome"]
                        for policy in POLICIES
                    },
                    "policies": policies,
                }
                sample_rows.append(sample_row)
                sample_writer.write(sample_row)
                token_writer.flush()
                sample_writer.flush()
                processed_count += 1
                run_metadata["processed_sample_count"] = processed_count

        classification_metrics = evaluate_policy_metrics(
            sample_rows,
            policy_names=POLICIES,
            decisions_field="policies",
        )
        location_metrics = {
            policy: summarize_location_decisions(
                unsafe_decisions[policy], policy=policy
            )
            for policy in POLICIES
        }
        safe_trigger_metrics = {
            policy: summarize_safe_trigger_decisions(
                safe_decisions[policy], policy=policy
            )
            for policy in POLICIES
        }

        for policy in POLICIES:
            binary_fpr = classification_metrics[policy]["fpr"]
            safe_fpr = safe_trigger_metrics[policy]["false_positive_rate"]
            if binary_fpr != safe_fpr:
                raise RuntimeError(
                    f"FPR mismatch for {policy}: classification={binary_fpr}, "
                    f"safe-trigger={safe_fpr}."
                )
            binary_fnr = classification_metrics[policy]["fnr"]
            location_fnr = location_metrics[policy]["response_fnr"]
            if binary_fnr != location_fnr:
                raise RuntimeError(
                    f"FNR mismatch for {policy}: classification={binary_fnr}, "
                    f"location={location_fnr}."
                )

        classification_rows = _classification_rows(classification_metrics)
        location_rows = [
            _location_row(policy, location_metrics[policy])
            for policy in POLICIES
        ]
        safe_rows = [
            _safe_trigger_row(policy, safe_trigger_metrics[policy])
            for policy in POLICIES
        ]
        unified_rows = [
            _unified_row(
                policy,
                classification_metrics[policy],
                location_metrics[policy],
                safe_trigger_metrics[policy],
            )
            for policy in POLICIES
        ]
        write_csv(output_dir / "classification_metrics.csv", classification_rows)
        write_csv(output_dir / "unsafe_location_metrics.csv", location_rows)
        write_csv(output_dir / "safe_false_positive_metrics.csv", safe_rows)
        write_csv(output_dir / "metrics.csv", unified_rows)

        false_positive_rows = [
            row
            for row in sample_rows
            if any(
                row["classification_outcomes"][policy] == "FP"
                for policy in POLICIES
            )
        ]
        false_negative_rows = [
            row
            for row in sample_rows
            if any(
                row["classification_outcomes"][policy] == "FN"
                for policy in POLICIES
            )
        ]
        late_or_missed_rows = [
            row
            for row in sample_rows
            if row["label"] == "Unsafe"
            and any(
                row["policies"][policy]["relation"] in {"after", "missed"}
                for policy in POLICIES
            )
        ]
        write_jsonl(false_positive_path, false_positive_rows)
        write_jsonl(false_negative_path, false_negative_rows)
        write_jsonl(late_or_missed_path, late_or_missed_rows)

        completed_at = datetime.now(timezone.utc).isoformat()
        wall_seconds = time.perf_counter() - wall_start
        run_summary.update(
            {
                "status": "completed",
                "loaded_sample_count": loaded_sample_count,
                "evaluated_sample_count": len(samples),
                "complete_split": len(samples) == loaded_sample_count,
                "label_counts": dict(label_counts),
                "unsafe_location_annotation_count": sum(
                    sample.has_location_annotation for sample in samples
                ),
                "token_record_count": token_writer.count,
                "classification": {"policies": classification_metrics},
                "unsafe_location": {
                    "sample_count": label_counts.get("Unsafe", 0),
                    "policies": location_metrics,
                },
                "safe_false_positive_location": {
                    "sample_count": label_counts.get("Safe", 0),
                    "policies": safe_trigger_metrics,
                },
                "diagnostic_record_counts": {
                    "false_positive_samples": len(false_positive_rows),
                    "false_negative_samples": len(false_negative_rows),
                    "late_or_missed_unsafe_samples": len(late_or_missed_rows),
                },
                "outputs": {
                    "token_records": str(token_path),
                    "sample_records": str(sample_path),
                    "false_positive_records": str(false_positive_path),
                    "false_negative_records": str(false_negative_path),
                    "late_or_missed_unsafe_records": str(late_or_missed_path),
                    "unified_metrics": str(output_dir / "metrics.csv"),
                    "classification_metrics": str(
                        output_dir / "classification_metrics.csv"
                    ),
                    "unsafe_location_metrics": str(
                        output_dir / "unsafe_location_metrics.csv"
                    ),
                    "safe_false_positive_metrics": str(
                        output_dir / "safe_false_positive_metrics.csv"
                    ),
                },
                "completed_at_utc": completed_at,
                "wall_time_seconds": wall_seconds,
            }
        )
        run_metadata.update(
            {
                "status": "completed",
                "processed_sample_count": processed_count,
                "completed_at_utc": completed_at,
                "wall_time_seconds": wall_seconds,
            }
        )
        write_json(metadata_path, run_metadata)
        write_json(summary_path, run_summary)
        _print_summary(
            classification_metrics,
            location_metrics,
            safe_trigger_metrics,
        )
    except BaseException as exc:
        failed_at = datetime.now(timezone.utc).isoformat()
        wall_seconds = time.perf_counter() - wall_start
        run_metadata.update(
            {
                "status": "failed",
                "processed_sample_count": processed_count,
                "failed_at_utc": failed_at,
                "wall_time_seconds": wall_seconds,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        run_summary.update(
            {
                "status": "failed",
                "failed_at_utc": failed_at,
                "wall_time_seconds": wall_seconds,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        write_json(metadata_path, run_metadata)
        write_json(summary_path, run_summary)
        raise

    print(f"Completed unified thinking evaluation: {output_dir}")
    print(f"Unified metrics: {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
