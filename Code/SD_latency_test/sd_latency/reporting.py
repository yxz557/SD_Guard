"""Pure reporting helpers, kept separate so they can be unit-tested without CUDA."""

from __future__ import annotations

import difflib
import statistics
from typing import Any, Sequence


def first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def compare_outputs(
    target_ids: Sequence[int],
    assisted_ids: Sequence[int],
    target_text: str,
    assisted_text: str,
) -> dict[str, Any]:
    mismatch = first_mismatch(target_ids, assisted_ids)
    common_prefix = min(len(target_ids), len(assisted_ids)) if mismatch is None else mismatch
    return {
        "exact_token_match": tuple(target_ids) == tuple(assisted_ids),
        "first_mismatch_index": mismatch,
        "common_prefix_token_count": common_prefix,
        "target_token_count": len(target_ids),
        "sd_token_count": len(assisted_ids),
        "token_count_difference": len(assisted_ids) - len(target_ids),
        "text_similarity_ratio": difflib.SequenceMatcher(
            None, target_text, assisted_text, autojunk=False
        ).ratio(),
    }


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty run list")
    latencies = [float(row["total_ms"]) for row in rows]
    generate_latencies = [float(row["generate_ms"]) for row in rows]
    total_tokens = sum(int(row["generated_token_count"]) for row in rows)
    total_seconds = sum(latencies) / 1000.0
    return {
        "runs": rows,
        "median_total_ms": statistics.median(latencies),
        "mean_total_ms": statistics.fmean(latencies),
        "median_generate_ms": statistics.median(generate_latencies),
        "aggregate_generated_tokens": total_tokens,
        "aggregate_tokens_per_second": (
            None if total_seconds <= 0 else total_tokens / total_seconds
        ),
    }


def summarize_pair(
    target_rows: list[dict[str, Any]], assisted_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    target = summarize_runs(target_rows)
    assisted = summarize_runs(assisted_rows)
    target_tps = target["aggregate_tokens_per_second"]
    assisted_tps = assisted["aggregate_tokens_per_second"]
    return {
        "target_only": target,
        "sd_untraced": assisted,
        "latency_speedup": target["median_total_ms"] / assisted["median_total_ms"],
        "throughput_speedup": (
            None
            if target_tps is None or assisted_tps is None
            else assisted_tps / target_tps
        ),
        "speedup_definition": "values greater than 1 mean SD is faster",
        "paired_token_counts_match": [
            row["generated_token_count"] for row in target_rows
        ]
        == [row["generated_token_count"] for row in assisted_rows],
    }
