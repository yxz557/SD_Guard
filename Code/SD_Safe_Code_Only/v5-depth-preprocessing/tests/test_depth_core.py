"""CPU unit tests for the v5 depth pipeline pieces that do not need the GPU."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdsafe_v5 import eval_depth as E
from sdsafe_v5.validate_depth import check_report_metrics
from sdsafe_v5.train_depth import _fit, group_folds


def test_metrics_and_threshold():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.2
    s = rng.normal(size=500) + y * 1.5
    thr = E.fpr_threshold(y, s)
    m = E.metrics(y, s, thr)
    assert m["fp"] <= int(np.floor(0.05 * (~y).sum())), m
    assert 0.5 < m["auc"] < 1.0
    # threshold is deterministic
    assert E.fpr_threshold(y, s) == thr
    # a perfect separation gives FN=0
    s2 = y.astype(float) * 2 + rng.normal(scale=1e-3, size=500)
    assert E.metrics(y, s2, E.fpr_threshold(y, s2))["fn"] == 0


def test_bootstrap_paired():
    rng = np.random.default_rng(1)
    n = 200
    groups = np.asarray([f"g{i%20}" for i in range(n)], dtype=object)
    y = rng.random(n) < 0.3
    base = rng.normal(size=n) + y
    cand = base + 0.5 * y
    b = E.bootstrap(y, base, cand, groups, E.fpr_threshold(y, base), E.fpr_threshold(y, cand), reps=200)
    assert b["replicates_valid"] > 150
    assert b["d_fn"]["mean"] <= 0  # candidate shifts positives up


def test_group_folds_deterministic():
    g = [f"g{i}" for i in range(50)]
    a = group_folds(g)
    b = group_folds(g)
    assert np.array_equal(a, b)


def test_forged_metric_guard():
    indep = {"checks": [{"arm": "H", "tag": "full", "kind": "dev",
                         "stored": {"tp": 393, "fp": 80, "fn": 21, "tn": 1534, "auc": 0.98},
                         "independent": {"tp": 393, "fp": 80, "fn": 21, "tn": 1534, "auc": 0.98}}]}
    clean = {"dev_variants": {"H:full": {"fn": 21}}}
    assert check_report_metrics(clean, indep) == []
    forged = {"dev_variants": {"H:full": {"fn": 18}}}
    bad = check_report_metrics(forged, indep)
    assert len(bad) == 1 and "fn" in bad[0]


def test_fit_runs_on_cpu():
    rng = np.random.default_rng(2)
    n, d = 240, 32
    x = rng.normal(size=(n, d)).astype(np.float32)
    tr = {"in_loss": np.ones(n, dtype=bool),
          "row_label": (rng.random(n) < 0.25).astype(np.int8)}
    folds = group_folds([f"g{i%6}" for i in range(n)])
    import torch
    xg = torch.from_numpy(x).to("cpu")
    rec = _fit("H", 0, "test", (1, 2, 3), xg, tr, folds, epochs=1)
    assert rec["fit_rows"] > 0 and np.isfinite(rec["final_loss"])


if __name__ == "__main__":
    for fn in [test_metrics_and_threshold, test_bootstrap_paired, test_group_folds_deterministic,
               test_forged_metric_guard, test_fit_runs_on_cpu]:
        fn()
        print("PASS", fn.__name__)
    print("ALL DEPTH CORE TESTS PASSED")
