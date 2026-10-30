"""Binary response-safety metrics with explicit undefined-rate handling."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Iterable, Mapping, Sequence

from .policies import POLICY_NAMES, SAFE, UNSAFE


def normalize_binary_label(value: Any) -> str:
    """Normalize dataset labels and binary policy predictions.

    ``Controversial`` is deliberately rejected here: callers must first apply
    either strict or loose policy semantics.
    """

    if isinstance(value, bool):
        return UNSAFE if value else SAFE
    if isinstance(value, Integral):
        if int(value) == 0:
            return SAFE
        if int(value) == 1:
            return UNSAFE
    if isinstance(value, str):
        aliases = {
            "safe": SAFE,
            "unsafe": UNSAFE,
            "0": SAFE,
            "1": UNSAFE,
            "false": SAFE,
            "true": UNSAFE,
        }
        key = value.strip().casefold()
        if key in aliases:
            return aliases[key]
    raise ValueError(
        f"Expected a binary Safe/Unsafe label after policy mapping, got {value!r}"
    )


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def confusion_counts(
    predictions: Iterable[Any], targets: Iterable[Any]
) -> dict[str, int]:
    predictions_list = [normalize_binary_label(value) for value in predictions]
    targets_list = [normalize_binary_label(value) for value in targets]
    if len(predictions_list) != len(targets_list):
        raise ValueError(
            "Predictions and targets must have the same length: "
            f"{len(predictions_list)} != {len(targets_list)}"
        )

    tp = tn = fp = fn = 0
    for prediction, target in zip(predictions_list, targets_list):
        if prediction == UNSAFE and target == UNSAFE:
            tp += 1
        elif prediction == UNSAFE and target == SAFE:
            fp += 1
        elif prediction == SAFE and target == SAFE:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_outcome(prediction: Any, target: Any) -> str:
    """Return the conventional TP/TN/FP/FN name for one binary decision."""

    normalized_prediction = normalize_binary_label(prediction)
    normalized_target = normalize_binary_label(target)
    if normalized_prediction == UNSAFE:
        return "TP" if normalized_target == UNSAFE else "FP"
    return "FN" if normalized_target == UNSAFE else "TN"


def binary_safety_metrics(
    predictions: Iterable[Any], targets: Iterable[Any]
) -> dict[str, int | float | None]:
    """Calculate response-level binary classification metrics.

    Precision/recall/F1 use zero when no positive prediction/target exists,
    matching common evaluation output.  FPR and FNR are returned as ``None``
    when their required target class is absent, because those rates are then
    mathematically undefined (not zero).
    """

    counts = confusion_counts(predictions, targets)
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    total = tp + tn + fp + fn
    positive_support = tp + fn
    negative_support = tn + fp
    predicted_positive = tp + fp

    precision = _safe_divide(tp, predicted_positive)
    recall = _safe_divide(tp, positive_support)
    precision_for_f1 = precision if precision is not None else 0.0
    recall_for_f1 = recall if recall is not None else 0.0
    f1_denominator = precision_for_f1 + recall_for_f1
    f1 = (
        2.0 * precision_for_f1 * recall_for_f1 / f1_denominator
        if f1_denominator
        else 0.0
    )

    return {
        **counts,
        "support": total,
        "unsafe_support": positive_support,
        "safe_support": negative_support,
        "predicted_unsafe": predicted_positive,
        "precision": precision if precision is not None else 0.0,
        "recall": recall if recall is not None else 0.0,
        "f1": f1,
        "fpr": _safe_divide(fp, negative_support),
        "fnr": _safe_divide(fn, positive_support),
        "accuracy": _safe_divide(tp + tn, total),
    }


def _find_decisions(record: Mapping[str, Any], decisions_field: str | None) -> Mapping[str, Any]:
    if decisions_field is not None:
        decisions = record.get(decisions_field)
        if not isinstance(decisions, Mapping):
            raise KeyError(f"Missing decision mapping at {decisions_field!r}")
        return decisions
    for candidate in ("policy_decisions", "policies"):
        decisions = record.get(candidate)
        if isinstance(decisions, Mapping):
            return decisions
    raise KeyError("Sample record has neither 'policy_decisions' nor 'policies'")


def evaluate_policy_metrics(
    sample_records: Iterable[Mapping[str, Any]],
    *,
    policy_names: Sequence[str] = POLICY_NAMES,
    label_field: str = "label",
    decisions_field: str | None = None,
) -> dict[str, dict[str, int | float | None]]:
    """Calculate one metric set per policy from saved sample records."""

    records = list(sample_records)
    targets = [record[label_field] for record in records]
    output: dict[str, dict[str, int | float | None]] = {}
    for policy_name in policy_names:
        predictions: list[Any] = []
        for record in records:
            decision = _find_decisions(record, decisions_field).get(policy_name)
            if not isinstance(decision, Mapping) or "prediction" not in decision:
                raise KeyError(
                    f"Missing prediction for policy {policy_name!r} in sample record"
                )
            predictions.append(decision["prediction"])
        output[policy_name] = binary_safety_metrics(predictions, targets)
    return output


# Backward-friendly concise name for callers.
calculate_metrics = binary_safety_metrics

