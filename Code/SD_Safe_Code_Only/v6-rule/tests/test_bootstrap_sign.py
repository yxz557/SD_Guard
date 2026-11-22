from __future__ import annotations

import unittest

import numpy as np

from sdsafe_v6 import metrics


class BootstrapSignTest(unittest.TestCase):
    def test_negative_delta_fn_is_improvement(self):
        labels = np.array([1, 1, 0, 0], dtype=np.int64)
        groups = np.array(["g1", "g1", "g2", "g2"])
        # baseline threshold: both positives below -> 2 FN; candidate: one positive above
        base = np.array([-2.0, -2.0, 2.0, 2.0])
        cand = np.array([2.0, -2.0, 2.0, 2.0])
        got = metrics.group_bootstrap(base, cand, labels, groups, 0.0, 0.0, replicates=20)
        self.assertEqual(got["delta_fn_point"], -1)
        self.assertLess(got["delta_fn_mean"], 0)
        self.assertGreaterEqual(got["delta_fp_point"], 0)


if __name__ == "__main__":
    unittest.main()
