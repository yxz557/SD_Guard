#!/usr/bin/env python3
"""Summarize and pairwise-compare the eight SD_5 Guard experiment cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from sdguard_sd.io_utils import iter_jsonl, read_json, write_json
from sdguard_sd.inline_guard import aggregate_inline_records


THINKING_LEVELS = ("think_on", "think_off")
GENERATION_MODES = ("target_only", "speculative", "draft_only", "inline_guard")
SCOPES = ("full_trace", "answer_only")
POLICIES = ("strict", "loose")
CONTROLLED_CONFIG_KEYS = (
    "dataset_path",
    "draft_model",
    "target_model",
    "guard_model",
    "sample_count",
    "seed",
    "cache_dir",
    "max_new_tokens",
    "max_model_len",
    "num_speculative_tokens",
    "draft_sample_method",
    "rejection_sample_method",
    "temperature",
    "top_p",
    "top_k",
    "do_sample",
    "target_device",
    "draft_device",
    "guard_device",
    "physical_gpu",
    "all_models_same_gpu",
    "dtype",
    "guard_attn_implementation",
    "enable_prefix_caching",
    "batch_size",
    "tensor_parallel_size",
    "draft_revision",
    "target_revision",
    "guard_revision",
    "local_files_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.fmean(items) if items else None


def _median(values: Iterable[float]) -> float | None:
    items = list(values)
    return statistics.median(items) if items else None


def _guard_payload(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("generation_mode") == "inline_guard":
        inline = row.get("inline_guard")
        if not isinstance(inline, dict) or "proposal_immediate" not in inline:
            raise ValueError(
                f"Inline cell row {row.get('sample_id')!r} lacks proposal_immediate"
            )
        return inline["proposal_immediate"]
    return row["guard"]


def _guard_value(row: dict[str, Any], scope: str, policy: str) -> bool | None:
    guard_scope = _guard_payload(row)["scopes"][scope]
    if not guard_scope["available"]:
        return None
    return guard_scope["decisions"][policy]["prediction"] == "Unsafe"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty experiment")
    modes = {str(row["generation_mode"]) for row in rows}
    thinking_values = {bool(row["thinking_enabled"]) for row in rows}
    if len(modes) != 1 or len(thinking_values) != 1:
        raise ValueError("A matrix cell mixes generation modes or thinking settings")

    generated_lengths = [int(row["generation"]["generated_token_count"]) for row in rows]
    generation_latencies = [float(row["generation"]["latency_ms"]) for row in rows]
    guard_latencies = [float(_guard_payload(row)["latency_ms"]) for row in rows]
    result: dict[str, Any] = {
        "generation_mode": next(iter(modes)),
        "thinking_enabled": next(iter(thinking_values)),
        "sample_count": len(rows),
        "truncated_count": sum(bool(row["generation"]["truncated"]) for row in rows),
        "truncated_rate": sum(bool(row["generation"]["truncated"]) for row in rows) / len(rows),
        "missing_close_think_count": sum(
            row["completion"]["parse_status"] == "missing_close_think" for row in rows
        ),
        "generated_tokens": {
            "mean": _mean(generated_lengths),
            "median": _median(generated_lengths),
            "min": min(generated_lengths),
            "max": max(generated_lengths),
        },
        "generation_latency_ms": {
            "mean": _mean(generation_latencies),
            "median": _median(generation_latencies),
        },
        "guard_latency_ms": {
            "mean": _mean(guard_latencies),
            "median": _median(guard_latencies),
        },
        "guard": {},
        "primary_guard_view": (
            "proposal_immediate"
            if next(iter(modes)) == "inline_guard"
            else "posthoc"
        ),
    }
    for scope in SCOPES:
        result["guard"][scope] = {}
        for policy in POLICIES:
            values = [_guard_value(row, scope, policy) for row in rows]
            available = [value for value in values if value is not None]
            unsafe_count = sum(value is True for value in available)
            result["guard"][scope][policy] = {
                "available_count": len(available),
                "unavailable_count": len(values) - len(available),
                "unsafe_count": unsafe_count,
                "unsafe_rate": None if not available else unsafe_count / len(available),
            }
    if result["generation_mode"] in {"speculative", "inline_guard"}:
        proposed = sum(
            int(row["speculative_decoding"]["proposed_token_count"]) for row in rows
        )
        accepted = sum(
            int(row["speculative_decoding"]["accepted_draft_token_count"]) for row in rows
        )
        result["speculative_decoding"] = {
            "proposed_token_count": proposed,
            "accepted_draft_token_count": accepted,
            "aggregate_acceptance_rate": None if proposed == 0 else accepted / proposed,
        }
    else:
        result["speculative_decoding"] = None
    if result["generation_mode"] == "inline_guard":
        result["inline_guard"] = aggregate_inline_records(
            row["inline_guard"] for row in rows
        )
    else:
        result["inline_guard"] = None
    return result


def exact_mcnemar_pvalue(left_only: int, right_only: int) -> float | None:
    discordant = left_only + right_only
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare_rows(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in left_rows}
    right = {str(row["sample_id"]): row for row in right_rows}
    if set(left) != set(right):
        raise ValueError(f"Sample IDs differ between {left_name} and {right_name}")

    comparisons: dict[str, Any] = {}
    ordered_ids = sorted(left)
    for scope in SCOPES:
        comparisons[scope] = {}
        for policy in POLICIES:
            both_safe = both_unsafe = left_only = right_only = unavailable = 0
            for sample_id in ordered_ids:
                left_value = _guard_value(left[sample_id], scope, policy)
                right_value = _guard_value(right[sample_id], scope, policy)
                if left_value is None or right_value is None:
                    unavailable += 1
                elif left_value and right_value:
                    both_unsafe += 1
                elif not left_value and not right_value:
                    both_safe += 1
                elif left_value:
                    left_only += 1
                else:
                    right_only += 1
            available = both_safe + both_unsafe + left_only + right_only
            comparisons[scope][policy] = {
                "paired_available_count": available,
                "paired_unavailable_count": unavailable,
                "both_safe": both_safe,
                "both_unsafe": both_unsafe,
                "left_only_unsafe": left_only,
                "right_only_unsafe": right_only,
                "left_unsafe_rate": (
                    None if available == 0 else (both_unsafe + left_only) / available
                ),
                "right_unsafe_rate": (
                    None if available == 0 else (both_unsafe + right_only) / available
                ),
                "unsafe_rate_difference_left_minus_right": (
                    None if available == 0 else (left_only - right_only) / available
                ),
                "disagreement_rate": (
                    None if available == 0 else (left_only + right_only) / available
                ),
                "exact_mcnemar_pvalue": exact_mcnemar_pvalue(left_only, right_only),
            }
    return {
        "left": left_name,
        "right": right_name,
        "guard_decisions": comparisons,
    }


def _flatten_cell(thinking_name: str, mode: str, summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "thinking": thinking_name,
        "generation_mode": mode,
        "sample_count": summary["sample_count"],
        "truncated_count": summary["truncated_count"],
        "truncated_rate": summary["truncated_rate"],
        "missing_close_think_count": summary["missing_close_think_count"],
        "mean_generated_tokens": summary["generated_tokens"]["mean"],
        "median_generated_tokens": summary["generated_tokens"]["median"],
        "median_generation_latency_ms": summary["generation_latency_ms"]["median"],
        "median_guard_latency_ms": summary["guard_latency_ms"]["median"],
    }
    for scope in SCOPES:
        for policy in POLICIES:
            value = summary["guard"][scope][policy]
            prefix = f"{scope}_{policy}"
            row[f"{prefix}_available_count"] = value["available_count"]
            row[f"{prefix}_unsafe_count"] = value["unsafe_count"]
            row[f"{prefix}_unsafe_rate"] = value["unsafe_rate"]
            inline = summary.get("inline_guard")
            inline_alert = (
                None if inline is None else inline["alerts"][scope][policy]
            )
            row[f"{prefix}_inline_proposal_only_alert_count"] = (
                None if inline_alert is None else inline_alert["proposal_only_alert_count"]
            )
            row[f"{prefix}_inline_rejected_trigger_round_count"] = (
                None
                if inline_alert is None
                else inline_alert["proposal_trigger_rejected_round_count"]
            )
    inline = summary.get("inline_guard")
    for label in ("Safe", "Controversial", "Unsafe"):
        row[f"inline_{label.casefold()}_proposal_acceptance_rate"] = (
            None
            if inline is None
            else inline["acceptance_by_proposal_risk"][label]["acceptance_rate"]
        )
    row["inline_median_total_guard_latency_ms"] = (
        None
        if inline is None
        else inline["timing_ms"]["total_inline_guard_latency_ms"]["median"]
    )
    row["inline_median_observed_generation_overhead_ms"] = (
        None
        if inline is None
        else inline["timing_ms"]["observed_generation_overhead_ms"]["median"]
    )
    return row


def main() -> None:
    args = parse_args()
    matrix_dir = args.matrix_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else matrix_dir / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "matrix_summary.json"
    csv_path = output_dir / "matrix_guard_summary.csv"
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Summary outputs already exist: {existing}; pass --overwrite")

    cell_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    cell_summaries: list[dict[str, Any]] = []
    judged_metrics: list[dict[str, Any]] = []
    expected_ids: set[str] | None = None
    expected_order: list[tuple[int, str, int]] | None = None
    controlled_configuration: dict[str, Any] | None = None
    for thinking_name in THINKING_LEVELS:
        for mode in GENERATION_MODES:
            experiment_dir = matrix_dir / thinking_name / mode
            rows = list(iter_jsonl(experiment_dir / "raw" / "generation_records.jsonl"))
            ids = {str(row["sample_id"]) for row in rows}
            order = [
                (int(row["sample_order"]), str(row["sample_id"]), int(row["dataset_index"]))
                for row in rows
            ]
            if expected_ids is None:
                expected_ids = ids
                expected_order = order
            elif ids != expected_ids:
                raise ValueError(f"Sample IDs in {experiment_dir} do not match the matrix manifest")
            elif order != expected_order:
                raise ValueError(f"Sample order in {experiment_dir} does not match the matrix manifest")

            run_config = read_json(experiment_dir / "run_config.json")
            if run_config["generation_mode"] != mode:
                raise ValueError(f"run_config generation_mode does not match {experiment_dir}")
            if bool(run_config["enable_thinking"]) != (thinking_name == "think_on"):
                raise ValueError(f"run_config thinking setting does not match {experiment_dir}")
            current_controlled = {key: run_config.get(key) for key in CONTROLLED_CONFIG_KEYS}
            if controlled_configuration is None:
                controlled_configuration = current_controlled
            elif current_controlled != controlled_configuration:
                changed = [
                    key
                    for key in CONTROLLED_CONFIG_KEYS
                    if current_controlled[key] != controlled_configuration[key]
                ]
                raise ValueError(
                    f"Controlled variables differ in {experiment_dir}: {changed}"
                )
            summary = summarize_rows(rows)
            expected_thinking = thinking_name == "think_on"
            if summary["generation_mode"] != mode or summary["thinking_enabled"] != expected_thinking:
                raise ValueError(f"Cell metadata does not match directory: {experiment_dir}")
            cell_rows[(thinking_name, mode)] = rows
            cell_summaries.append(
                {"thinking": thinking_name, "generation_mode": mode, **summary}
            )
            metrics_path = experiment_dir / "analysis" / "metrics.json"
            if metrics_path.exists():
                metrics = read_json(metrics_path)
                judged_metrics.append(
                    {
                        "thinking": thinking_name,
                        "generation_mode": mode,
                        "metrics": metrics["metrics"],
                    }
                )

    generation_equivalence: list[dict[str, Any]] = []
    for thinking_name in THINKING_LEVELS:
        speculative = {
            str(row["sample_id"]): row
            for row in cell_rows[(thinking_name, "speculative")]
        }
        inline = {
            str(row["sample_id"]): row
            for row in cell_rows[(thinking_name, "inline_guard")]
        }
        mismatches = [
            sample_id
            for sample_id in sorted(speculative)
            if speculative[sample_id]["generation"]["generated_ids"]
            != inline[sample_id]["generation"]["generated_ids"]
        ]
        if mismatches:
            raise ValueError(
                f"inline_guard changed speculative output in {thinking_name}: {mismatches[:10]}"
            )
        generation_equivalence.append(
            {
                "thinking": thinking_name,
                "speculative_vs_inline_generated_ids_all_equal": True,
                "matched_sample_count": len(speculative),
            }
        )

    mode_comparisons: list[dict[str, Any]] = []
    for thinking_name in THINKING_LEVELS:
        for left_mode, right_mode in combinations(GENERATION_MODES, 2):
            mode_comparisons.append(
                {
                    "comparison_type": "generation_mode_within_thinking",
                    "thinking": thinking_name,
                    **compare_rows(
                        cell_rows[(thinking_name, left_mode)],
                        cell_rows[(thinking_name, right_mode)],
                        left_name=left_mode,
                        right_name=right_mode,
                    ),
                }
            )

    thinking_comparisons: list[dict[str, Any]] = []
    for mode in GENERATION_MODES:
        thinking_comparisons.append(
            {
                "comparison_type": "thinking_within_generation_mode",
                "generation_mode": mode,
                **compare_rows(
                    cell_rows[("think_on", mode)],
                    cell_rows[("think_off", mode)],
                    left_name="think_on",
                    right_name="think_off",
                ),
            }
        )

    report = {
        "schema_version": 2,
        "matrix_dir": str(matrix_dir),
        "sample_count_per_cell": len(expected_ids or set()),
        "controlled_configuration": controlled_configuration,
        "cell_summaries": cell_summaries,
        "generation_mode_comparisons": mode_comparisons,
        "thinking_comparisons": thinking_comparisons,
        "speculative_inline_generation_equivalence": generation_equivalence,
        "judged_metrics": judged_metrics,
        "judged_metrics_complete": len(judged_metrics) == 8,
        "interpretation_note": (
            "McNemar comparisons describe paired Guard decision differences by prompt. "
            "inline_guard uses proposal_immediate as its primary view; verified_commit is "
            "an asserted posthoc-equivalent diagnostic. Human labels are reused from the "
            "speculative cell only after exact generated-token equality is established."
        ),
    }
    write_json(json_path, report)

    flat_rows = [
        _flatten_cell(item["thinking"], item["generation_mode"], item)
        for item in cell_summaries
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nMatrix summary: {json_path}")


if __name__ == "__main__":
    main()


