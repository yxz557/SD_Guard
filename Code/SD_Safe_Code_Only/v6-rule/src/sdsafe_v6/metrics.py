"""Metrics, thresholds, bootstrap -- independent of the training code's own bookkeeping."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, budget: float = 0.05) -> float:
    """Highest threshold with FPR <= budget (``score >= threshold``); ties handled by sorting.

    Same rule as the accepted stage: candidate thresholds are the observed scores, taken from the
    high end so that the constraint holds on the full sample; no interpolation, no in-draw
    re-selection during bootstrap.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    neg = np.sort(scores[labels == 0])[::-1]
    n_neg = neg.size
    allowed = int(np.floor(budget * n_neg + 1e-12))
    if allowed >= n_neg:
        return float(scores.min())
    if allowed < 0:
        return float(np.inf)
    # keep at most `allowed` negatives at or above the threshold
    if allowed == 0:
        return float(np.nextafter(neg[0], np.inf))
    return float(neg[allowed - 1])


def confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = np.asarray(scores, dtype=np.float64) >= float(threshold)
    labels = np.asarray(labels, dtype=np.int64)
    tp = int(((labels == 1) & pred).sum())
    fp = int(((labels == 0) & pred).sum())
    fn = int(((labels == 1) & ~pred).sum())
    tn = int(((labels == 0) & ~pred).sum())
    pos, neg = tp + fn, fp + tn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "positives": pos, "negatives": neg,
            "fnr": fn / pos if pos else None, "fpr": fp / neg if neg else None,
            "threshold": float(threshold), "comparator": ">="}


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.float64)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def episode_scores(row_scores: np.ndarray, episode_index: np.ndarray, n_episodes: int) -> np.ndarray:
    out = np.full(n_episodes, -np.inf, dtype=np.float32)
    np.maximum.at(out, episode_index, np.asarray(row_scores, dtype=np.float32))
    return out


def group_bootstrap(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray, groups: Sequence[str],
    thr_a: float, thr_b: float, *, replicates: int = 2000, seed: int = 20261114,
) -> dict[str, Any]:
    """Paired whole-group bootstrap with FIXED thresholds (no in-draw ROC re-selection)."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray([str(g) for g in groups])
    uniq = np.asarray(sorted(set(groups.tolist())))
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    d_fn, d_fp = [], []
    for _ in range(replicates):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([members[g] for g in pick])
        y = labels[idx]
        pa = np.asarray(scores_a)[idx] >= thr_a
        pb = np.asarray(scores_b)[idx] >= thr_b
        # signed change candidate - baseline: negative is an improvement for both error types
        d_fn.append(int(((y == 1) & (~pb)).sum() - ((y == 1) & (~pa)).sum()))
        d_fp.append(int(((y == 0) & pb).sum() - ((y == 0) & pa).sum()))
    d_fn = np.asarray(d_fn, dtype=np.float64)
    d_fp = np.asarray(d_fp, dtype=np.float64)
    return {
        "replicates": replicates,
        "delta_fn_point": int(((labels == 1) & (np.asarray(scores_b) < thr_b)).sum()
                              - ((labels == 1) & (np.asarray(scores_a) < thr_a)).sum()),
        "delta_fp_point": int(((labels == 0) & (np.asarray(scores_b) >= thr_b)).sum()
                              - ((labels == 0) & (np.asarray(scores_a) >= thr_a)).sum()),
        "delta_fn_mean": float(d_fn.mean()),
        "delta_fn_ci95": [float(np.percentile(d_fn, 2.5)), float(np.percentile(d_fn, 97.5))],
        "delta_fp_mean": float(d_fp.mean()),
        "delta_fp_ci95": [float(np.percentile(d_fp, 2.5)), float(np.percentile(d_fp, 97.5))],
    }


def slices(scores: np.ndarray, labels: np.ndarray, threshold: float,
           meta: dict[str, np.ndarray]) -> dict[str, Any]:
    pred = np.asarray(scores, dtype=np.float64) >= float(threshold)
    labels = np.asarray(labels, dtype=np.int64)
    out: dict[str, Any] = {}
    for name, values in meta.items():
        table: dict[str, Any] = {}
        values = np.asarray([str(v) for v in values])
        for value in sorted(set(values.tolist())):
            m = values == value
            tp = int((pred[m] & (labels[m] == 1)).sum())
            fp = int((pred[m] & (labels[m] == 0)).sum())
            fn = int((~pred[m] & (labels[m] == 1)).sum())
            tn = int((~pred[m] & (labels[m] == 0)).sum())
            table[value] = {"episodes": int(m.sum()), "positives": tp + fn, "tp": tp, "fp": fp,
                            "fn": fn, "tn": tn}
        out[name] = table
    return out
