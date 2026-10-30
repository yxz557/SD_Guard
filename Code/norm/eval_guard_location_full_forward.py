#!/usr/bin/env python3
"""Evaluate Qwen3Guard location splits with one-shot causal full forwards.

This entry point reconstructs chronological online decisions from the token
labels produced by the same causal full-forward path as the released
``eval_stream.py``.  It is intentionally separate from
``eval_stream_guard.py``, whose purpose is true token-by-token inference and
stream/full consistency measurement.
"""

from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from guard_eval.dataset import LOCATION_SPLITS, load_guard_samples
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
from guard_eval.result_io import JsonlWriter, write_csv, write_json
from guard_eval.sequence import prepare_sequence


DEFAULT_SPLITS = ("thinking_loc", "response_loc")
POLICIES = ("strict", "loose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run causal full-forward token classification on Qwen3GuardTest's "
            "location splits and compare the first chronological UU/CC trigger "
            "with each human-annotated unsafe interval."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(LOCATION_SPLITS),
        default=list(DEFAULT_SPLITS),
        help="Location splits to evaluate (default: thinking_loc response_loc).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N rows per split for a smoke test.",
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


def model_display_name(model_path: str) -> str:
    name = Path(model_path).expanduser().name
    return name or model_path.rstrip("/").rsplit("/", 1)[-1]


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. "
            "Pass --overwrite to replace this evaluation output."
        )
    if overwrite:
        for name in ("raw", "run_metadata.json", "summary.json", "metrics.csv"):
            target = destination / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _token_relation(full_index: int, start: int, end: int) -> str:
    if full_index < start:
        return "before"
    if full_index <= end:
        return "within"
    return "after"


def _metric_row(
    split: str, policy: str, summary: Mapping[str, Any]
) -> dict[str, Any]:
    counts = summary["relation_counts"]
    rates = summary["relation_rates"]
    bins = summary["location_bins"]
    pairs = summary["trigger_pair_counts"]
    return {
        "split": split,
        "policy": policy,
        "sample_count": summary["sample_count"],
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


def _format_rate(value: Any) -> str:
    return "N/A" if value is None else f"{100.0 * float(value):.2f}%"


def _print_policy_summary(split: str, summary: Mapping[str, Any]) -> None:
    counts = summary["relation_counts"]
    print(
        f"[{split} · {summary['policy']}] "
        f"before={counts['before']} within={counts['within']} "
        f"after={counts['after']} missed={counts['missed']} "
        f"response_FNR={_format_rate(summary['response_fnr'])} "
        f"timely_failure={_format_rate(summary['timely_failure_rate'])} "
        f"first_128={_format_rate(summary['first_128_rate'])}"
    )


def _progress(iterable: Iterable[Any], *, description: str) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - requirements include tqdm
        return iterable
    return tqdm(iterable, desc=description)


def main() -> None:
    args = parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when supplied")
    if len(args.splits) != len(set(args.splits)):
        raise ValueError(f"--splits contains duplicates: {args.splits!r}")

    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    model_name = model_display_name(args.model_path)
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    processed_by_split = {split: 0 for split in args.splits}
    run_metadata: dict[str, Any] = {
        "status": "starting",
        "started_at_utc": started_at,
        "model": model_name,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "splits": list(args.splits),
        "max_samples_per_split": args.max_samples,
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
        "trigger_coordinate": "second token in pair; full-sequence zero-based index",
        "unsafe_interval": "inclusive [unsafe_start_index, unsafe_end_index]",
        "processed_by_split": processed_by_split,
    }
    metadata_path = output_dir / "run_metadata.json"
    write_json(metadata_path, run_metadata)

    metric_rows: list[dict[str, Any]] = []
    run_summary: dict[str, Any] = {
        "status": "running",
        "model": model_name,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "splits": {},
        "metric_definitions": {
            "response_fnr": "missed / all annotated-Unsafe responses",
            "timely_failure_rate": (
                "(triggered after unsafe_end_index + missed) / all responses"
            ),
            "exact_hit_rate": "triggered inside the inclusive unsafe interval / all",
            "first_128_rate": (
                "triggered before/inside the interval or within 128 tokens after its end / all"
            ),
        },
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, run_summary)

    try:
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

        for split in args.splits:
            print(f"Loading {split}: {args.dataset_path}")
            samples = load_guard_samples(
                args.dataset_path,
                split,
                cache_dir=args.cache_dir,
            )
            loaded_sample_count = len(samples)
            if args.max_samples is not None:
                samples = samples[: args.max_samples]
            if any(sample.label != "Unsafe" for sample in samples):
                raise ValueError(
                    f"{split} must contain only Unsafe rows for FNR/location metrics"
                )

            split_dir = output_dir / "raw" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            token_path = split_dir / "token_records.jsonl"
            sample_path = split_dir / "sample_records.jsonl"
            decisions_by_policy: dict[str, list[dict[str, Any]]] = {
                policy: [] for policy in POLICIES
            }

            with JsonlWriter(token_path) as token_writer, JsonlWriter(
                sample_path
            ) as sample_writer:
                indexed_samples = enumerate(samples)
                for dataset_index, sample in _progress(
                    indexed_samples,
                    description=f"{model_name} · {split} · causal full-forward",
                ):
                    sequence = prepare_sequence(
                        loaded.tokenizer,
                        sample.messages,
                        split,
                        annotated_input_ids=sample.input_ids,
                        require_alignment=True,
                    )
                    full = run_full_forward(loaded, sequence)
                    if full.response_token_count != sequence.response_token_count:
                        raise RuntimeError(
                            f"Response token count mismatch for {split}/{sample.unique_id}: "
                            f"full={full.response_token_count}, "
                            f"sequence={sequence.response_token_count}."
                        )

                    start = int(sample.unsafe_start_index)  # validated by dataset loader
                    end = int(sample.unsafe_end_index)
                    all_decisions = evaluate_all_policies(
                        full.risk_levels,
                        full_index_offset=sequence.eval_start_index,
                    )
                    policies = {
                        "strict": classify_trigger_location(
                            all_decisions["online_strict"],
                            unsafe_start_index=start,
                            unsafe_end_index=end,
                        ),
                        "loose": classify_trigger_location(
                            all_decisions["online_loose"],
                            unsafe_start_index=start,
                            unsafe_end_index=end,
                        ),
                    }
                    for policy, decision in policies.items():
                        decisions_by_policy[policy].append(decision)

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
                                "split": split,
                                "sample_id": sample.unique_id,
                                "dataset_index": dataset_index,
                                "assistant_token_index": assistant_index,
                                "full_sequence_index": full_index,
                                "token_id": int(token_id),
                                "risk_level": risk_level,
                                "risk_prob": risk_prob,
                                "annotated_interval_relation": _token_relation(
                                    full_index, start, end
                                ),
                                "strict_newly_triggered": (
                                    policies["strict"]["trigger_full_index"] == full_index
                                ),
                                "loose_newly_triggered": (
                                    policies["loose"]["trigger_full_index"] == full_index
                                ),
                            }
                        )

                    sample_writer.write(
                        {
                            "model": model_name,
                            "split": split,
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
                            "policies": policies,
                        }
                    )
                    token_writer.flush()
                    sample_writer.flush()
                    processed_by_split[split] += 1
                    run_metadata["processed_by_split"] = dict(processed_by_split)

            policy_summaries = {
                policy: summarize_location_decisions(
                    decisions_by_policy[policy], policy=policy
                )
                for policy in POLICIES
            }
            split_summary = {
                "split": split,
                "loaded_sample_count": loaded_sample_count,
                "evaluated_sample_count": len(samples),
                "complete_split": len(samples) == loaded_sample_count,
                "label_counts": dict(Counter(sample.label for sample in samples)),
                "token_record_count": token_writer.count,
                "policies": policy_summaries,
                "outputs": {
                    "token_records": str(token_path),
                    "sample_records": str(sample_path),
                },
            }
            write_json(split_dir / "summary.json", split_summary)
            run_summary["splits"][split] = split_summary
            for policy, policy_summary in policy_summaries.items():
                metric_rows.append(_metric_row(split, policy, policy_summary))
                _print_policy_summary(split, policy_summary)
            write_csv(output_dir / "metrics.csv", metric_rows)
            write_json(summary_path, run_summary)

        completed_at = datetime.now(timezone.utc).isoformat()
        wall_seconds = time.perf_counter() - wall_start
        run_metadata.update(
            {
                "status": "completed",
                "processed_by_split": dict(processed_by_split),
                "completed_at_utc": completed_at,
                "wall_time_seconds": wall_seconds,
            }
        )
        run_summary.update(
            {
                "status": "completed",
                "completed_at_utc": completed_at,
                "wall_time_seconds": wall_seconds,
            }
        )
        write_json(metadata_path, run_metadata)
        write_json(summary_path, run_summary)
    except BaseException as exc:
        failed_at = datetime.now(timezone.utc).isoformat()
        run_metadata.update(
            {
                "status": "failed",
                "processed_by_split": dict(processed_by_split),
                "failed_at_utc": failed_at,
                "wall_time_seconds": time.perf_counter() - wall_start,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        run_summary.update(
            {
                "status": "failed",
                "failed_at_utc": failed_at,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        write_json(metadata_path, run_metadata)
        write_json(summary_path, run_summary)
        raise

    print(f"Completed location evaluation: {output_dir}")
    print(f"Flat metrics: {output_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
