#!/usr/bin/env python3
"""Analyze the shared thinking subset and compare all three evaluation paths."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from guard_eval.latency_metrics import (
    build_sample_latency_rows,
    spearman_context_latency,
    summarize_latencies,
)
from guard_eval.policies import evaluate_all_policies, normalize_risk_label
from guard_eval.result_io import read_jsonl, write_csv, write_json, write_jsonl
from guard_eval.safety_metrics import binary_safety_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate strict/loose safety metrics, streaming latency, response-length "
            "correlation, and agreement with sampled official predictions."
        )
    )
    parser.add_argument(
        "--custom-dir",
        type=Path,
        required=True,
        help="Output directory created by eval_stream_guard.py.",
    )
    parser.add_argument(
        "--official-output",
        type=Path,
        help="JSONL created by official/eval_stream_sampled.py.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Pass --overwrite to replace it."
        )
    if overwrite:
        for name in (
            "summary.json",
            "metrics.csv",
            "latency_summary.csv",
            "sample_latency.csv",
            "disagreements.jsonl",
        ):
            target = destination / name
            if target.exists():
                target.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file does not exist: {path}")
    return path


def metric_row(
    method: str,
    policy: str,
    predictions: Iterable[str],
    targets: Iterable[str],
) -> dict[str, Any]:
    return {
        "method": method,
        "policy": policy,
        **binary_safety_metrics(predictions, targets),
    }


def custom_predictions(
    records: list[dict[str, Any]], field: str, policy: str
) -> list[str]:
    return [str(record[field][policy]["prediction"]) for record in records]


def official_decisions(record: Mapping[str, Any]) -> dict[str, str]:
    labels = [normalize_risk_label(value) for value in record["pred_risk_levels"]]
    policies = evaluate_all_policies(labels)
    return {
        "strict": str(policies["official_strict"]["prediction"]),
        "loose": str(policies["official_loose"]["prediction"]),
    }


def agreement_count(left: Iterable[str], right: Iterable[str]) -> dict[str, Any]:
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError("Agreement inputs have different lengths")
    matched = sum(a == b for a, b in zip(left_values, right_values))
    return {
        "sample_count": len(left_values),
        "matched_sample_count": matched,
        "mismatched_sample_count": len(left_values) - matched,
        "agreement": matched / len(left_values) if left_values else None,
    }


def main() -> None:
    args = parse_args()
    custom_dir = args.custom_dir.expanduser().resolve()
    raw_dir = custom_dir / "raw"
    sample_records = read_jsonl(require_file(raw_dir / "sample_records.jsonl"))
    token_records = read_jsonl(require_file(raw_dir / "token_records.jsonl"))
    full_records = read_jsonl(require_file(raw_dir / "full_forward_records.jsonl"))
    if not sample_records:
        raise ValueError("Custom evaluation contains no completed samples")

    sample_records.sort(key=lambda row: int(row["sample_order"]))
    full_records.sort(key=lambda row: int(row["sample_order"]))
    sample_ids = [str(row["sample_id"]) for row in sample_records]
    full_ids = [str(row["sample_id"]) for row in full_records]
    if sample_ids != full_ids:
        raise ValueError("Custom streaming and full-forward sample orders differ")

    targets = [str(record["label"]) for record in sample_records]
    metrics: list[dict[str, Any]] = []
    custom_by_method: dict[str, dict[str, list[str]]] = {
        "custom_stream": {},
        "custom_full_forward": {},
    }
    for policy in ("strict", "loose"):
        stream_values = custom_predictions(sample_records, "stream_policies", policy)
        full_values = custom_predictions(
            sample_records, "full_forward_policies", policy
        )
        custom_by_method["custom_stream"][policy] = stream_values
        custom_by_method["custom_full_forward"][policy] = full_values
        metrics.append(metric_row("custom_stream", policy, stream_values, targets))
        metrics.append(
            metric_row("custom_full_forward", policy, full_values, targets)
        )

    disagreements: list[dict[str, Any]] = []
    stream_full_agreement: dict[str, Any] = {}
    for policy in ("strict", "loose"):
        stream_values = custom_by_method["custom_stream"][policy]
        full_values = custom_by_method["custom_full_forward"][policy]
        stream_full_agreement[policy] = agreement_count(stream_values, full_values)
        for sample_id, stream_prediction, full_prediction in zip(
            sample_ids, stream_values, full_values
        ):
            if stream_prediction != full_prediction:
                disagreements.append(
                    {
                        "comparison": "custom_stream_vs_custom_full_forward",
                        "policy": policy,
                        "sample_id": sample_id,
                        "left_prediction": stream_prediction,
                        "right_prediction": full_prediction,
                    }
                )

    stream_full_token_count = sum(
        int(row["agreement"]["token_count"]) for row in sample_records
    )
    stream_full_token_mismatches = sum(
        int(row["agreement"]["token_label_mismatch_count"])
        for row in sample_records
    )
    stream_full_token_summary = {
        "token_count": stream_full_token_count,
        "mismatched_token_count": stream_full_token_mismatches,
        "token_label_agreement": (
            1.0 - stream_full_token_mismatches / stream_full_token_count
            if stream_full_token_count
            else None
        ),
    }

    official_summary: dict[str, Any] | None = None
    if args.official_output is not None:
        official_records = read_jsonl(require_file(args.official_output.expanduser().resolve()))
        official_records.sort(key=lambda row: int(row["sample_order"]))
        official_ids = [
            str(row.get("unique_id", row.get("sample_id")))
            for row in official_records
        ]
        if official_ids != sample_ids:
            raise ValueError(
                "Official output does not contain the same samples in the same order "
                "as the custom evaluation manifest"
            )

        official_by_policy = {"strict": [], "loose": []}
        for record in official_records:
            decisions = official_decisions(record)
            for policy in ("strict", "loose"):
                official_by_policy[policy].append(decisions[policy])
        for policy in ("strict", "loose"):
            metrics.append(
                metric_row(
                    "official_script",
                    policy,
                    official_by_policy[policy],
                    targets,
                )
            )

        full_official_agreement: dict[str, Any] = {}
        for policy in ("strict", "loose"):
            full_values = custom_by_method["custom_full_forward"][policy]
            official_values = official_by_policy[policy]
            full_official_agreement[policy] = agreement_count(
                full_values, official_values
            )
            for sample_id, full_prediction, official_prediction in zip(
                sample_ids, full_values, official_values
            ):
                if full_prediction != official_prediction:
                    disagreements.append(
                        {
                            "comparison": "custom_full_forward_vs_official_script",
                            "policy": policy,
                            "sample_id": sample_id,
                            "left_prediction": full_prediction,
                            "right_prediction": official_prediction,
                        }
                    )

        full_by_id = {str(row["sample_id"]): row for row in full_records}
        official_token_count = 0
        official_token_mismatches = 0
        tokenization_mismatch_samples: list[str] = []
        for official_record in official_records:
            sample_id = str(
                official_record.get("unique_id", official_record.get("sample_id"))
            )
            full_record = full_by_id[sample_id]
            same_ids = list(full_record["input_ids"]) == list(
                official_record["input_ids"]
            )
            same_boundary = int(full_record["eval_start_index"]) == int(
                official_record["split_idx_eval"]
            )
            if not (same_ids and same_boundary):
                tokenization_mismatch_samples.append(sample_id)
                disagreements.append(
                    {
                        "comparison": "custom_full_forward_vs_official_script",
                        "policy": "tokenization",
                        "sample_id": sample_id,
                        "input_ids_match": same_ids,
                        "response_boundary_matches": same_boundary,
                    }
                )
                continue

            full_labels = [str(value) for value in full_record["risk_levels"]]
            official_labels = [
                normalize_risk_label(value)
                for value in official_record["pred_risk_levels"]
            ]
            paired = min(len(full_labels), len(official_labels))
            mismatches = sum(
                full_labels[index] != official_labels[index]
                for index in range(paired)
            ) + abs(len(full_labels) - len(official_labels))
            official_token_count += max(len(full_labels), len(official_labels))
            official_token_mismatches += mismatches
            if mismatches:
                disagreements.append(
                    {
                        "comparison": "custom_full_forward_vs_official_script",
                        "policy": "token_labels",
                        "sample_id": sample_id,
                        "token_count": max(len(full_labels), len(official_labels)),
                        "mismatched_token_count": mismatches,
                    }
                )

        official_summary = {
            "sample_agreement": full_official_agreement,
            "tokenization_mismatch_sample_count": len(
                tokenization_mismatch_samples
            ),
            "tokenization_mismatch_sample_ids": tokenization_mismatch_samples,
            "token_label_comparison": {
                "token_count": official_token_count,
                "mismatched_token_count": official_token_mismatches,
                "token_label_agreement": (
                    1.0 - official_token_mismatches / official_token_count
                    if official_token_count
                    else None
                ),
            },
        }

    latency = summarize_latencies(record.get("latency_ms") for record in token_records)
    sample_latency = build_sample_latency_rows(token_records)
    sample_metadata = {str(row["sample_id"]): row for row in sample_records}
    for row in sample_latency:
        metadata = sample_metadata[str(row["sample_id"])]
        row["sample_order"] = metadata["sample_order"]
        row["dataset_index"] = metadata["dataset_index"]
        row["label"] = metadata["label"]
    sample_latency.sort(key=lambda row: int(row["sample_order"]))
    spearman = spearman_context_latency(sample_latency)

    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    write_csv(output_dir / "metrics.csv", metrics)
    write_csv(output_dir / "latency_summary.csv", [latency])
    write_csv(output_dir / "sample_latency.csv", sample_latency)
    write_jsonl(output_dir / "disagreements.jsonl", disagreements)

    summary = {
        "sample_count": len(sample_records),
        "sample_ids": sample_ids,
        "metrics": metrics,
        "streaming_latency": latency,
        "response_length_latency_correlation": spearman,
        "custom_stream_vs_full_forward": {
            "sample_agreement": stream_full_agreement,
            "token_label_comparison": stream_full_token_summary,
        },
        "custom_full_forward_vs_official_script": official_summary,
        "disagreement_record_count": len(disagreements),
        "notes": {
            "strict": "UU or CC; Controversial pairs are Unsafe at sample level",
            "loose": "UU only; Controversial pairs remain Safe at sample level",
            "latency_scope": (
                "Assistant token already exists through Stream Guard risk label extraction; "
                "Assistant generation, prompt prefill, full forward, tokenization, and I/O excluded"
            ),
            "official_metric_handling": (
                "Metrics are recomputed from official pred_risk_levels because the released "
                "script prints strict/loose lists under swapped names"
            ),
        },
    }
    write_json(output_dir / "summary.json", summary)

    print(f"Analyzed {len(sample_records)} shared thinking samples")
    for row in metrics:
        print(
            f"[{row['method']} · {row['policy']}] "
            f"FPR={row['fpr']!r} FNR={row['fnr']!r} F1={row['f1']!r}"
        )
    print(
        "[stream latency] "
        f"tokens={latency['token_count']} mean={latency['mean_ms']!r} ms "
        f"p95={latency['p95_ms']!r} ms p99={latency['p99_ms']!r} ms"
    )
    print(f"Analysis written to: {output_dir}")


if __name__ == "__main__":
    main()
