from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def choose_fbeta_threshold(labels: np.ndarray, scores: np.ndarray, beta: float = 2.0) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    beta_sq = beta * beta
    fbeta = (1.0 + beta_sq) * precision[:-1] * recall[:-1] / (
        beta_sq * precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[int(np.nanargmax(fbeta))])


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    result: dict[str, Any] = {
        "samples": int(labels.size),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if labels.size else 0.0,
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1.mean()),
        "negative_precision": float(precision[0]),
        "negative_recall": float(recall[0]),
        "negative_f1": float(f1[0]),
        "positive_precision": float(precision[1]),
        "positive_recall": float(recall[1]),
        "positive_f1": float(f1[1]),
        "support": [int(value) for value in support],
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    if len(np.unique(labels)) == 2:
        result["average_precision"] = float(average_precision_score(labels, scores))
        result["roc_auc"] = float(roc_auc_score(labels, scores))
    else:
        result["average_precision"] = None
        result["roc_auc"] = None
    return result

