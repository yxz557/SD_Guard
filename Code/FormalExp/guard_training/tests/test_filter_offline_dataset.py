from __future__ import annotations

import unittest

from guard_training.frozen_hidden_mlp.filter_offline_dataset import (
    normalize_suites,
    select_indices,
    summarize_rows,
)


class FilterOfflineDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "attack_type": "direct",
                "data_source": "local",
                "group_id": "g1",
                "label": 1,
                "suite_name": "travel",
            },
            {
                "attack_type": "tool_knowledge",
                "data_source": "local",
                "group_id": "g1",
                "label": 0,
                "suite_name": "travel",
            },
            {
                "attack_type": "direct",
                "data_source": "public",
                "group_id": "g2",
                "label": 0,
                "suite_name": "banking",
            },
        ]

    def test_normalize_suites_deduplicates_and_sorts(self) -> None:
        self.assertEqual(
            normalize_suites([" travel ", "banking", "travel"]),
            ("banking", "travel"),
        )

    def test_select_indices_filters_only_requested_suite(self) -> None:
        self.assertEqual(select_indices(self.rows, ("travel",)), [0, 1])

    def test_summarize_rows_reports_labels_sources_attacks_and_groups(self) -> None:
        summary = summarize_rows(self.rows[:2])
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["positives"], 1)
        self.assertEqual(summary["negatives"], 1)
        self.assertEqual(summary["groups"], 1)
        self.assertEqual(summary["labels"], {"0": 1, "1": 1})
        self.assertEqual(summary["sources"], {"local": 2})
        self.assertEqual(
            summary["attacks"], {"direct": 1, "tool_knowledge": 1}
        )
        self.assertEqual(summary["suites"], {"travel": 2})

    def test_summarize_rows_rejects_non_binary_labels(self) -> None:
        row = dict(self.rows[0])
        row["label"] = 2
        with self.assertRaisesRegex(ValueError, "Expected binary labels"):
            summarize_rows([row])


if __name__ == "__main__":
    unittest.main()
