"""Token- and sample-level Stream Guard latency statistics."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: Iterable[Any], percentile_value: float) -> float | None:
    """Linear-interpolated percentile, equivalent to NumPy's default method."""

    if not 0 <= percentile_value <= 100:
        raise ValueError("Percentile must be in [0, 100]")
    numbers = sorted(
        number for value in values if (number := _finite_float(value)) is not None
    )
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0]
    position = (len(numbers) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numbers[lower]
    fraction = position - lower
    return numbers[lower] + fraction * (numbers[upper] - numbers[lower])


def summarize_latencies(values: Iterable[Any]) -> dict[str, int | float | None]:
    """Return the required overall per-token latency statistics."""

    numbers = [
        number for value in values if (number := _finite_float(value)) is not None
    ]
    total_ms = sum(numbers)
    return {
        "token_count": len(numbers),
        "mean_ms": fmean(numbers) if numbers else None,
        "p50_ms": percentile(numbers, 50),
        "p95_ms": percentile(numbers, 95),
        "p99_ms": percentile(numbers, 99),
        "min_ms": min(numbers) if numbers else None,
        "max_ms": max(numbers) if numbers else None,
        "total_ms": total_ms,
        "tokens_per_second": (
            1000.0 * len(numbers) / total_ms if total_ms > 0 else None
        ),
    }


def _identity(record: Mapping[str, Any], identity_fields: Sequence[str]) -> tuple[Any, ...]:
    try:
        return tuple(record[field] for field in identity_fields)
    except KeyError as exc:
        raise KeyError(f"Missing sample identity field: {exc.args[0]!r}") from exc


def build_sample_latency_rows(
    token_records: Iterable[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str] = ("sample_id",),
    context_field: str = "context_length",
    latency_field: str = "latency_ms",
) -> list[dict[str, Any]]:
    """Collapse raw token rows into one latency row per sample.

    Call this on one model/split at a time, or include ``model`` and ``split``
    in ``identity_fields`` when passing mixed records.
    """

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in token_records:
        groups[_identity(record, identity_fields)].append(record)

    rows: list[dict[str, Any]] = []
    for identity, records in groups.items():
        context_lengths = [
            number
            for record in records
            if (number := _finite_float(record.get(context_field))) is not None
        ]
        latencies = [
            number
            for record in records
            if (number := _finite_float(record.get(latency_field))) is not None
        ]
        row = {field: value for field, value in zip(identity_fields, identity)}
        row.update(
            {
                "response_token_count": len(records),
                "mean_context_length": (
                    fmean(context_lengths) if context_lengths else None
                ),
                "mean_token_latency_ms": fmean(latencies) if latencies else None,
                "measured_token_count": len(latencies),
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: tuple(str(row[field]) for field in identity_fields))
    return rows


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        # Ranks are conventionally one-based. Ties receive their mean rank.
        average_rank = ((position + 1) + end) / 2.0
        for sorted_position in range(position, end):
            ranks[indexed[sorted_position][0]] = average_rank
        position = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Both inputs must have equal length")
    if len(left) < 2:
        return None
    mean_left, mean_right = fmean(left), fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        return None
    return sum(
        first * second for first, second in zip(centered_left, centered_right)
    ) / denominator


def spearman_context_latency(
    sample_rows: Iterable[Mapping[str, Any]],
    *,
    context_field: str = "mean_context_length",
    latency_field: str = "mean_token_latency_ms",
) -> dict[str, int | float | None]:
    """Measure sample-level monotonic association between context and latency."""

    pairs: list[tuple[float, float]] = []
    for row in sample_rows:
        context = _finite_float(row.get(context_field))
        latency = _finite_float(row.get(latency_field))
        if context is not None and latency is not None:
            pairs.append((context, latency))

    sample_count = len(pairs)
    if sample_count < 2:
        return {"sample_count": sample_count, "spearman_rho": None, "p_value": None}

    contexts, latencies = zip(*pairs)
    rho = _pearson(_average_ranks(contexts), _average_ranks(latencies))
    if rho is None:
        return {"sample_count": sample_count, "spearman_rho": None, "p_value": None}

    p_value: float | None = None
    try:
        from scipy.stats import spearmanr  # type: ignore

        scipy_result = spearmanr(contexts, latencies)
        scipy_statistic = getattr(scipy_result, "statistic", None)
        if scipy_statistic is None:
            scipy_statistic = getattr(scipy_result, "correlation")
        scipy_rho = float(scipy_statistic)
        scipy_p = float(scipy_result.pvalue)
        if math.isfinite(scipy_rho):
            rho = scipy_rho
        if math.isfinite(scipy_p):
            p_value = scipy_p
    except (ImportError, AttributeError, TypeError, ValueError):
        # Rho remains available without making SciPy mandatory. P-values need
        # the distribution implementation supplied by SciPy.
        pass

    return {"sample_count": sample_count, "spearman_rho": rho, "p_value": p_value}


# Concise alias used by entry-point code.
overall_token_latency = summarize_latencies

