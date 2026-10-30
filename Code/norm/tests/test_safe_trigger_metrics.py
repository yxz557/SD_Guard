from __future__ import annotations

import unittest

from guard_eval.safe_trigger_metrics import (
    classify_safe_trigger,
    summarize_safe_trigger_decisions,
)
from guard_eval.safety_metrics import classification_outcome


def decision(
    trigger_index: int | None,
    *,
    pair: tuple[str, str] = ("Unsafe", "Unsafe"),
) -> dict[str, object]:
    return {
        "policy": "online_strict",
        "prediction": "Safe" if trigger_index is None else "Unsafe",
        "triggered": trigger_index is not None,
        "trigger_index": trigger_index,
        "trigger_pair": None if trigger_index is None else list(pair),
    }


class SafeTriggerMetricTests(unittest.TestCase):
    def test_classifies_response_relative_false_positive_position(self) -> None:
        row = classify_safe_trigger(
            decision(31, pair=("Controversial", "Controversial")),
            response_token_count=100,
        )
        self.assertTrue(row["false_positive"])
        self.assertEqual(row["trigger_token_number"], 32)
        self.assertEqual(row["trigger_pair_code"], "CC")
        self.assertEqual(row["safe_trigger_bin"], "1-32")
        self.assertEqual(row["tokens_remaining_after_trigger"], 68)
        self.assertAlmostEqual(row["trigger_fraction_of_response"], 0.32)

        no_trigger = classify_safe_trigger(
            decision(None), response_token_count=40
        )
        self.assertFalse(no_trigger["false_positive"])
        self.assertEqual(no_trigger["safe_trigger_bin"], "No trigger")

    def test_summary_reports_fpr_prefixes_pairs_and_positions(self) -> None:
        rows = [
            classify_safe_trigger(decision(1), response_token_count=200),
            classify_safe_trigger(
                decision(32, pair=("Controversial", "Controversial")),
                response_token_count=100,
            ),
            classify_safe_trigger(decision(127), response_token_count=128),
            classify_safe_trigger(decision(128), response_token_count=256),
            classify_safe_trigger(decision(None), response_token_count=80),
        ]
        summary = summarize_safe_trigger_decisions(rows, policy="strict")

        self.assertEqual(summary["safe_sample_count"], 5)
        self.assertEqual(summary["false_positive_count"], 4)
        self.assertEqual(summary["true_negative_count"], 1)
        self.assertEqual(summary["false_positive_rate"], 0.8)
        self.assertEqual(summary["trigger_pair_counts"], {"CC": 1, "UU": 3})
        self.assertEqual(
            summary["trigger_position_bins"],
            {
                "1-32": 1,
                "33-64": 1,
                "65-128": 1,
                ">128": 1,
                "No trigger": 1,
            },
        )
        prefixes = summary["false_positives_by_response_prefix"]
        self.assertEqual(prefixes["32"]["false_positive_count"], 1)
        self.assertEqual(prefixes["64"]["false_positive_count"], 2)
        self.assertEqual(prefixes["128"]["false_positive_count"], 3)
        self.assertEqual(prefixes["full_response"]["false_positive_count"], 4)

    def test_rejects_trigger_outside_response(self) -> None:
        with self.assertRaises(ValueError):
            classify_safe_trigger(decision(20), response_token_count=20)

    def test_classification_outcome_uses_binary_semantics(self) -> None:
        self.assertEqual(classification_outcome("Unsafe", "Unsafe"), "TP")
        self.assertEqual(classification_outcome("Safe", "Unsafe"), "FN")
        self.assertEqual(classification_outcome("Unsafe", "Safe"), "FP")
        self.assertEqual(classification_outcome("Safe", "Safe"), "TN")


if __name__ == "__main__":
    unittest.main()
