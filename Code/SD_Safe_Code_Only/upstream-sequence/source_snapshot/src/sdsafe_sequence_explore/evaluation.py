"""No-oracle episode metrics for fine_sequence_explore_v1 (Lane T).

The primary score for an episode is the maximum raw logit over label-blind, real CT
inference positions (unknown positions included).  Padded/E positions are excluded by
mask, never by truth.  Empty bags score ``-inf`` (no alarm) and stay in every
denominator.  AUC is tie-correct and operating points use a deterministic ``>=``
threshold rule.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BUDGETS = (0.05, 0.02, 0.01)
HIGHER_IS_POSITIVE = True


def _as_arrays(scores: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    label = np.asarray(labels).reshape(-1)
    if score.shape != label.shape:
        raise ValueError("scores and labels must have the same length")
    if np.isnan(score).any():
        raise ValueError("scores contain NaN")
    if np.isposinf(score).any():
        raise ValueError("+inf episode scores are invalid; only -inf missing-bag sentinel is allowed")
    if not np.isin(label, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return score, label.astype(np.int64)


def roc_auc(scores: Any, labels: Any) -> float:
    """Tie-correct rank AUC (Mann-Whitney U with average ranks)."""
    score, label = _as_arrays(scores, labels)
    positive = label == 1
    negative = ~positive
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        stop = start + 1
        while stop < len(sorted_scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        # 1-based average rank for this tie group.
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    auc = (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def confusion_at(scores: Any, labels: Any, threshold: float) -> dict[str, Any]:
    score, label = _as_arrays(scores, labels)
    # A missing-CT bag is -inf and must never alarm, even at threshold=-inf.
    predicted = np.isfinite(score) & (score >= float(threshold))
    tp = int(np.sum(predicted & (label == 1)))
    fp = int(np.sum(predicted & (label == 0)))
    tn = int(np.sum(~predicted & (label == 0)))
    fn = int(np.sum(~predicted & (label == 1)))
    n_pos = tp + fn
    n_neg = fp + tn
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "fpr": float(fp / n_neg) if n_neg else float("nan"),
        "fnr": float(fn / n_pos) if n_pos else float("nan"),
        "tpr": float(tp / n_pos) if n_pos else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
    }


def empirical_operating_point(
    scores: Any,
    labels: Any,
    target_fpr: float = 0.05,
) -> dict[str, Any]:
    """Deterministic empirical ROC point with FPR <= target, using score >= threshold.

    Tie handling: thresholds are unique observed scores (plus +inf for "predict none"),
    and among feasible points we minimise ``(FNR, actual FPR, -threshold)``.  This is
    tie-safe and does not use labels to filter scores.
    """
    score, label = _as_arrays(scores, labels)
    n_pos = int((label == 1).sum())
    n_neg = int((label == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {
            **confusion_at(score, label, float("inf")),
            "target_fpr": float(target_fpr),
            "feasible": bool(n_neg == 0),
            "achieved_fpr": 0.0 if n_neg == 0 else float("nan"),
            "achieved_fnr": float("nan"),
            "auc": float("nan"),
        }
    candidates: list[tuple[tuple[float, float, float], dict[str, Any]]] = []
    thresholds = np.concatenate(([np.inf], np.unique(score)[::-1]))
    for threshold in thresholds:
        point = confusion_at(score, label, float(threshold))
        fpr = float(point["fpr"])
        if fpr <= float(target_fpr) + 1e-12:
            key = (float(point["fnr"]), fpr, -float(threshold))
            candidates.append((key, point))
    if not candidates:  # target_fpr < 0
        candidates.append(((1.0, 0.0, -float("inf")), confusion_at(score, label, float("inf"))))
    candidates.sort(key=lambda item: item[0])
    chosen = dict(candidates[0][1])
    chosen.update(
        {
            "target_fpr": float(target_fpr),
            "feasible": True,
            "achieved_fpr": float(chosen["fpr"]),
            "achieved_fnr": float(chosen["fnr"]),
            "auc": roc_auc(score, label),
        }
    )
    return chosen


def evaluate_scores(
    scores: Any,
    labels: Any,
    *,
    budgets: Sequence[float] = BUDGETS,
) -> dict[str, Any]:
    score, label = _as_arrays(scores, labels)
    auc = roc_auc(score, label)
    points = {f"{100.0 * float(b):g}%": empirical_operating_point(score, label, float(b)) for b in budgets}
    empty = int(np.sum(~np.isfinite(score)))
    return {
        "auc": auc,
        "n_episodes": int(len(score)),
        "n_positives": int((label == 1).sum()),
        "n_negatives": int((label == 0).sum()),
        "empty_bag_scores": empty,
        "operating_points": points,
        "primary": points[f"{100.0 * float(budgets[0]):g}%"],
    }


def episode_scores_from_logits(
    logits: Any,
    valid_mask: Any,
    ct_mask: Any | None = None,
) -> np.ndarray:
    """Max raw logit over real CT positions; -inf for an empty bag."""
    import torch

    logits_t = logits.detach().to(torch.float32)
    valid = valid_mask.detach().to(torch.bool)
    eligible = valid if ct_mask is None else valid & ct_mask.detach().to(torch.bool)
    masked = logits_t.masked_fill(~eligible, float("-inf"))
    scores = masked.max(dim=1).values
    if bool(torch.isnan(scores).any()) or bool(torch.isposinf(scores).any()):
        raise ValueError("episode scores contain NaN or +inf; real logits must be finite")
    return scores.cpu().numpy().astype(np.float64)


def evaluate_episode_logits(
    logits: Any,
    valid_mask: Any,
    ct_mask: Any,
    bag_label: Any,
    *,
    budgets: Sequence[float] = BUDGETS,
) -> dict[str, Any]:
    scores = episode_scores_from_logits(logits, valid_mask, ct_mask)
    labels = np.asarray(bag_label.detach().cpu() if hasattr(bag_label, "detach") else bag_label).reshape(-1)
    return evaluate_scores(scores, labels.astype(np.int64), budgets=budgets)


def frozen_threshold(
    scores: Any,
    labels: Any,
    *,
    target_fpr: float = 0.05,
) -> float:
    """CAL-only empirical threshold; apply frozen elsewhere and name it as such."""
    return float(empirical_operating_point(scores, labels, target_fpr)["threshold"])


def apply_frozen_threshold(scores: Any, labels: Any, threshold: float) -> dict[str, Any]:
    return confusion_at(scores, labels, float(threshold))


def summarise_by(
    scores: Any,
    labels: Any,
    keys: Iterable[Any],
    *,
    budgets: Sequence[float] = BUDGETS,
    global_threshold: float | None = None,
) -> dict[str, Any]:
    """Per-source/group summaries.

    ``own_roc_*`` fields are explicitly recalibrated within each subgroup.  If a common
    ``global_threshold`` is supplied, an additional frozen-threshold confusion is added;
    no subgroup-specific threshold is mislabelled as a global operating point.
    """
    score, label = _as_arrays(scores, labels)
    key_array = np.asarray(list(keys), dtype=object)
    if key_array.shape[0] != score.shape[0]:
        raise ValueError("keys length mismatch")
    out: dict[str, Any] = {}
    for key in sorted({str(value) for value in key_array.tolist()}):
        mask = np.asarray([str(value) == key for value in key_array.tolist()])
        sub_scores, sub_labels = score[mask], label[mask]
        n_pos = int((sub_labels == 1).sum())
        n_neg = int((sub_labels == 0).sum())
        item: dict[str, Any] = {
            "n_episodes": int(mask.sum()),
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "roc_scope": "own_subgroup_roc",
        }
        if n_pos and n_neg:
            item["own_roc_auc"] = roc_auc(sub_scores, sub_labels)
            item["own_roc_primary"] = empirical_operating_point(sub_scores, sub_labels, budgets[0])
        if global_threshold is not None:
            item["global_threshold_confusion"] = confusion_at(sub_scores, sub_labels, global_threshold)
        out[key] = item
    return out


def predictions_payload(
    *,
    logits: Any,
    valid_mask: Any,
    ct_mask: Any,
    prefix_state: Any,
    exact: Any,
    bag_label: Any,
    episode_ids: Sequence[str],
    group_ids: Sequence[str],
    scopes: Sequence[str],
    lengths: Any,
    row_start: Any,
    row_end: Any,
    scores: Any,
    split_name: str,
    is_agentlab: Any | None = None,
) -> dict[str, np.ndarray]:
    """Plain arrays for independent no-scorer replay; no model is needed to read them."""
    import torch

    def array(value: Any, dtype: Any) -> np.ndarray:
        if value is None:
            return np.zeros((0,), dtype=dtype)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=dtype)

    logits_array = np.ascontiguousarray(array(logits, np.float32))
    # Actual CPU torch.float32 sigmoid; this is a compatibility diagnostic only,
    # never a calibrated risk or the raw-logit primary score.
    sigmoid_float32 = torch.sigmoid(torch.from_numpy(logits_array)).numpy().astype(np.float32, copy=False)
    payload = {
        "logits": logits_array,
        "sigmoid_float32_diagnostic": sigmoid_float32,
        "valid_mask": array(valid_mask, np.bool_),
        "ct_mask": array(ct_mask, np.bool_),
        "prefix_state": array(prefix_state, np.int8),
        "exact": array(exact, np.bool_),
        "bag_label": array(bag_label, np.float32),
        "episode_scores": array(scores, np.float64),
        "lengths": array(lengths, np.int64),
        "row_start": array(row_start, np.int64),
        "row_end": array(row_end, np.int64),
        "episode_ids": np.asarray([str(x) for x in episode_ids]),
        "group_ids": np.asarray([str(x) for x in group_ids]),
        "scopes": np.asarray([str(x) for x in scopes]),
        "split": np.asarray(str(split_name)),
    }
    payload["is_agentlab"] = (
        array(is_agentlab, np.bool_)
        if is_agentlab is not None
        else np.asarray([str(scope) == "agentlab" for scope in scopes], dtype=np.bool_)
    )
    payload["is_common_dev"] = ~payload["is_agentlab"]
    return payload


__all__ = [
    "BUDGETS",
    "roc_auc",
    "confusion_at",
    "empirical_operating_point",
    "evaluate_scores",
    "episode_scores_from_logits",
    "evaluate_episode_logits",
    "frozen_threshold",
    "apply_frozen_threshold",
    "summarise_by",
    "predictions_payload",
]
