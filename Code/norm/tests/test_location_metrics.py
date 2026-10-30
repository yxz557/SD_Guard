from __future__ import annotations

import unittest

from guard_eval.location_metrics import (
    classify_trigger_location,
    summarize_location_decisions,
)


def decision(
    trigger_full_index: int | None,
    *,
    pair: tuple[str, str] | None = ("Unsafe", "Unsafe"),
) -> dict[str, object]:
    return {
        "policy": "online_strict",
        "prediction": "Safe" if trigger_full_index is None else "Unsafe",
        "triggered": trigger_full_index is not None,
        "trigger_index": None if trigger_full_index is None else trigger_full_index - 5,
        "trigger_full_index": trigger_full_index,
        "trigger_pair": None if trigger_full_index is None else list(pair or ()),
    }


class TriggerLocationTests(unittest.TestCase):
    def test_classifies_inclusive_interval_and_signed_distance(self) -> None:
        before = classify_trigger_location(
            decision(8), unsafe_start_index=10, unsafe_end_index=20
        )
        self.assertEqual(before["relation"], "before")
        self.assertEqual(before["distance_to_unsafe_interval"], -2)
        self.assertEqual(before["tokens_before_unsafe_start"], 2)

        at_start = classify_trigger_location(
            decision(10), unsafe_start_index=10, unsafe_end_index=20
        )
        at_end = classify_trigger_location(
            decision(20), unsafe_start_index=10, unsafe_end_index=20
        )
        self.assertEqual(at_start["relation"], "within")
        self.assertEqual(at_end["relation"], "within")
        self.assertEqual(at_end["distance_to_unsafe_interval"], 0)

        after = classify_trigger_location(
            decision(23), unsafe_start_index=10, unsafe_end_index=20
        )
        self.assertEqual(after["relation"], "after")
        self.assertEqual(after["distance_to_unsafe_interval"], 3)
        self.assertEqual(after["tokens_after_unsafe_end"], 3)

        missed = classify_trigger_location(
            decision(None), unsafe_start_index=10, unsafe_end_index=20
        )
        self.assertEqual(missed["relation"], "missed")
        self.assertIsNone(missed["distance_to_unsafe_interval"])

    def test_summary_reports_response_and_timeliness_failures(self) -> None:
        raw = [
            decision(8),
            decision(10, pair=("Controversial", "Controversial")),
            decision(15),
            decision(20),
            decision(21),
            decision(100),
            decision(200),
            decision(None),
        ]
        rows = [
            classify_trigger_location(
                item, unsafe_start_index=10, unsafe_end_index=20
            )
            for item in raw
        ]
        summary = summarize_location_decisions(rows, policy="strict")

        self.assertEqual(summary["sample_count"], 8)
        self.assertEqual(
            summary["relation_counts"],
            {"before": 1, "within": 3, "after": 3, "missed": 1},
        )
        self.assertEqual(summary["response_fnr"], 1 / 8)
        self.assertEqual(summary["response_recall"], 7 / 8)
        self.assertEqual(summary["detected_by_unsafe_start_count"], 2)
        self.assertEqual(summary["detected_by_unsafe_end_count"], 4)
        self.assertEqual(summary["timely_failure_count"], 4)
        self.assertEqual(summary["timely_failure_rate"], 0.5)
        self.assertEqual(summary["first_128_count"], 6)
        self.assertEqual(summary["first_128_rate"], 0.75)
        self.assertEqual(summary["trigger_pair_counts"], {"CC": 1, "UU": 6})
        self.assertEqual(
            summary["location_bins"],
            {
                "Ahead": 1,
                "Hit": 3,
                "1-32": 1,
                "33-64": 0,
                "65-128": 1,
                "129-256": 1,
                ">256": 0,
                "Missed": 1,
            },
        )

    def test_rejects_invalid_annotation_interval(self) -> None:
        with self.assertRaises(ValueError):
            classify_trigger_location(
                decision(5), unsafe_start_index=10, unsafe_end_index=9
            )


if __name__ == "__main__":
    unittest.main()
