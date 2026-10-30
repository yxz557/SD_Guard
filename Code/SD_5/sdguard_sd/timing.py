"""Latency statistics and robust extraction of vLLM request metrics."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


def percentile(values: Iterable[float], probability: float) -> float | None:
    items = sorted(float(value) for value in values)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    position = (len(items) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    fraction = position - lower
    return items[lower] * (1.0 - fraction) + items[upper] * fraction


def summarize_ms(values: Iterable[float]) -> dict[str, float | int | None]:
    items = [float(value) for value in values]
    return {
        "count": len(items),
        "mean": statistics.fmean(items) if items else None,
        "median": statistics.median(items) if items else None,
        "p90": percentile(items, 0.90),
        "p95": percentile(items, 0.95),
        "min": min(items) if items else None,
        "max": max(items) if items else None,
    }


def _seconds_delta(end: Any, start: Any) -> float | None:
    if end is None or start is None:
        return None
    try:
        return max(0.0, (float(end) - float(start)) * 1000.0)
    except (TypeError, ValueError):
        return None


def extract_request_timing(request_output: Any) -> dict[str, float | None]:
    """Extract vLLM timestamps without depending on one private metrics class."""

    metrics = getattr(request_output, "metrics", None)
    if metrics is None:
        return {
            "queue_ms": None,
            "prefill_to_first_token_ms": None,
            "decode_after_first_token_ms": None,
            "scheduled_to_last_token_ms": None,
        }
    arrival = getattr(metrics, "arrival_time", None)
    scheduled = getattr(metrics, "first_scheduled_time", None)
    first_token = getattr(metrics, "first_token_time", None)
    last_token = getattr(metrics, "last_token_time", None)
    if last_token is None:
        last_token = getattr(metrics, "finished_time", None)
    return {
        "queue_ms": _seconds_delta(scheduled, arrival),
        "prefill_to_first_token_ms": _seconds_delta(first_token, scheduled),
        "decode_after_first_token_ms": _seconds_delta(last_token, first_token),
        "scheduled_to_last_token_ms": _seconds_delta(last_token, scheduled),
    }


def subtract_known(total_ms: float, *parts: float | None) -> float:
    return max(0.0, float(total_ms) - sum(float(value or 0.0) for value in parts))


__all__ = ["extract_request_timing", "percentile", "subtract_known", "summarize_ms"]
