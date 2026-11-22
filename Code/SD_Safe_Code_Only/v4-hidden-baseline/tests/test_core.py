"""Fast unit tests for the metric/threshold/fold primitives (no GPU, no capture)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v4.train_eval import auc, confusions, fpr_threshold, group_folds  # noqa: E402
from sdsafe_v4.validate import _auc as indep_auc  # noqa: E402
from sdsafe_v4.validate import _metrics as indep_metrics  # noqa: E402
from sdsafe_v4.validate import _threshold_at_fpr as indep_threshold  # noqa: E402


def test_auc_matches_two_implementations_on_random_data() -> None:
    rng = np.random.default_rng(7)
    for _ in range(5):
        y = rng.integers(0, 2, size=500)
        if y.sum() in (0, 500):
            continue
        s = rng.normal(size=500) + 1.2 * y
        a, b = auc(y, s), indep_auc(y, s)
        assert abs(a - b) < 1e-12, (a, b)


def test_auc_known_value() -> None:
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    assert abs(auc(y, s) - 1.0) < 1e-12
    assert abs(auc(y, -s) - 0.0) < 1e-12
    assert abs(auc(y, np.array([1.0, 1.0, 1.0, 1.0])) - 0.5) < 1e-12


def test_threshold_hits_fpr_budget() -> None:
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, size=4000)
    s = rng.normal(size=4000) + 0.5 * y
    thr = fpr_threshold(y, s, 0.05)
    cm = confusions(y, s, thr)
    fpr = cm["fp"] / (cm["fp"] + cm["tn"])
    assert fpr <= 0.05 + 1e-12, fpr
    assert fpr > 0.04, fpr
    assert abs(thr - indep_threshold(y, s, 0.05)) < 1e-9


def test_confusions_consistency() -> None:
    y = np.array([1, 0, 1, 1, 0])
    s = np.array([2.0, 0.5, -1.0, 3.0, -3.0])
    cm = confusions(y, s, 1.0)
    assert cm == {"tp": 2, "fp": 0, "fn": 1, "tn": 2}, cm
    mine = indep_metrics(y, s, 1.0)
    assert (mine["tp"], mine["fp"], mine["fn"], mine["tn"]) == (2, 0, 1, 2)


def test_group_folds_never_split_a_group() -> None:
    groups = [f"g{i//3}" for i in range(300)]
    folds = group_folds(groups)
    seen = {}
    for g, f in zip(groups, folds.tolist()):
        seen.setdefault(g, set()).add(f)
    assert all(len(v) == 1 for v in seen.values())
    assert len(set(folds.tolist())) == 4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} tests passed")
