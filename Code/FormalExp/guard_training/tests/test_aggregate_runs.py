from __future__ import annotations

import unittest

from guard_training.frozen_hidden_mlp.aggregate_runs import aggregate_dicts


class AggregateRunsTest(unittest.TestCase):
    def test_numeric_metrics_are_aggregated_recursively(self) -> None:
        result = aggregate_dicts(
            [
                {"overall": {"average_precision": 0.6, "samples": 10}},
                {"overall": {"average_precision": 0.8, "samples": 10}},
            ]
        )
        self.assertAlmostEqual(result["overall"]["average_precision"]["mean"], 0.7)
        self.assertAlmostEqual(
            result["overall"]["average_precision"]["std"], 0.1414213562
        )
        self.assertEqual(result["overall"]["samples"]["values"], [10.0, 10.0])


if __name__ == "__main__":
    unittest.main()

