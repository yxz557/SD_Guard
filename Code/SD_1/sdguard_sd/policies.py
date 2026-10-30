"""Chronological strict/loose policies for token-level Guard labels."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


SAFE = "Safe"
UNSAFE = "Unsafe"
CONTROVERSIAL = "Controversial"


def normalize_label(value: Any) -> str:
    text = str(value).strip().casefold()
    mapping = {
        "safe": SAFE,
        "unsafe": UNSAFE,
        "controversial": CONTROVERSIAL,
        "0": SAFE,
        "1": UNSAFE,
        "2": CONTROVERSIAL,
    }
    try:
        return mapping[text]
    except KeyError as exc:
        raise ValueError(f"Unknown risk label: {value!r}") from exc


def _decision(
    policy: str,
    labels: Sequence[str],
    predicate: Any,
    *,
    offset: int,
) -> dict[str, Any]:
    for index in range(1, len(labels)):
        if predicate(labels[index - 1], labels[index]):
            return {
                "policy": policy,
                "prediction": UNSAFE,
                "triggered": True,
                "trigger_scope_index": index,
                "trigger_generated_index": offset + index,
                "trigger_pair": [labels[index - 1], labels[index]],
            }
    return {
        "policy": policy,
        "prediction": SAFE,
        "triggered": False,
        "trigger_scope_index": None,
        "trigger_generated_index": None,
        "trigger_pair": None,
    }


def evaluate_policies(
    risk_levels: Iterable[Any],
    *,
    generated_index_offset: int = 0,
) -> dict[str, dict[str, Any]]:
    labels = [normalize_label(value) for value in risk_levels]
    return {
        "strict": _decision(
            "strict",
            labels,
            lambda left, right: left == right and left in {UNSAFE, CONTROVERSIAL},
            offset=generated_index_offset,
        ),
        "loose": _decision(
            "loose",
            labels,
            lambda left, right: left == right == UNSAFE,
            offset=generated_index_offset,
        ),
    }


__all__ = [
    "CONTROVERSIAL",
    "SAFE",
    "UNSAFE",
    "evaluate_policies",
    "normalize_label",
]

