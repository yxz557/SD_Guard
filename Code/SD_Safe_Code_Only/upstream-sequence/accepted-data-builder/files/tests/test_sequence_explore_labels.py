"""Focused synthetic + accepted-provenance tests for sequence label semantics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from sdsafe_sequence_explore.labels import (
    MISSING,
    build_dev_label_arrays,
    build_train_label_arrays,
    checker_state,
    episode_label_support,
    load_policy_assignments,
    recompute_policy_targets,
)
from sdsafe_sequence_explore import (
    EXPECTED_POLICY_COUNTS,
    EXPECTED_TRAIN_CT_EXACT,
    EXPECTED_TRAIN_CT_ROWS,
    EXPECTED_TRAIN_CT_UNKNOWN,
    LabelContractError,
)

ROOT = Path(__file__).resolve().parents[1]
REPAIR_DATA = ROOT / "artifacts/fine_label_repair_v1/data"


def event(
    episode: str,
    index: int,
    token_index: int,
    *,
    final: bool = True,
    status: str = "exact",
    prefix: bool | None = False,
    previous: bool | None = False,
    round_id: int = 1,
    message_index: int = 1,
    scope: str = "cot",
    event_type: str = "content_segment",
) -> dict:
    exact = status == "exact"
    if not exact:
        prefix = None
        previous = None
    return {
        "event_id": f"{episode}::r{round_id}::{event_type}::{index}",
        "episode_id": episode,
        "namespaced_episode_id": episode,
        "scope": scope,
        "source_tag": {"cot": "cot32k", "nothink": "nothink_ctx40k", "ctx40k": "ctx40k", "agentlab": "ramp_ablation"}[scope],
        "split": "train",
        "group_id": f"{scope}|u|i",
        "event_type": event_type,
        "round_id": round_id,
        "message_index": message_index,
        "index": index,
        "call_index": index if event_type == "tool_call" else None,
        "segment_index": index if event_type == "content_segment" else None,
        "feature_token_index": token_index,
        "strict_final": final,
        "label_status": status,
        "prefix_security": bool(prefix) if exact else None,
        "previous_security": bool(previous) if exact else None,
        "security_flip": bool(prefix and not previous) if exact else None,
    }


def assignments_for(rows: list[dict]) -> dict[str, dict[str, dict]]:
    semantic, _ = recompute_policy_targets(rows)
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        event_id = str(row["event_id"])
        out[event_id] = {policy: dict(semantic[policy][event_id]) for policy in ("L0", "L1", "L2", "L3")}
        # L4 is a control; for synthetic fixtures mirror L1 so the population is valid.
        out[event_id]["L4"] = dict(semantic["L1"][event_id])
    return out


class AcceptedProvenanceTests(unittest.TestCase):
    def test_accepted_assignment_counts(self) -> None:
        assignments = load_policy_assignments(ROOT)
        self.assertEqual(len(assignments), EXPECTED_TRAIN_CT_ROWS)

    def test_actual_population_and_prefix_state(self) -> None:
        rows = [json.loads(line) for line in (REPAIR_DATA / "repaired_train_rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(rows), EXPECTED_TRAIN_CT_ROWS)
        assignments = load_policy_assignments(ROOT)
        arrays, summary = build_train_label_arrays(rows, assignments)
        self.assertEqual(summary["exact_rows"], EXPECTED_TRAIN_CT_EXACT)
        self.assertEqual(summary["unknown_rows"], EXPECTED_TRAIN_CT_UNKNOWN)
        self.assertTrue(np.array_equal(arrays["prefix_state"], arrays["label_L3"]))
        self.assertTrue(np.all(arrays["label_L0"][~arrays["exact"]] == MISSING))
        for policy, expected in EXPECTED_POLICY_COUNTS.items():
            self.assertEqual(summary["per_policy"][policy]["positive"], expected["positive"])
            self.assertEqual(summary["per_policy"][policy]["negative"], expected["negative"])
            self.assertEqual(summary["per_policy"][policy]["masked_exact"], expected["masked_exact"])
            self.assertEqual(summary["per_policy"][policy]["unknown"], expected["unknown"])
        self.assertEqual(summary["semantic_recomputation"]["positive_episodes"], 1722)
        self.assertEqual(summary["semantic_recomputation"]["l3_positive_episodes"], 1740)


class SyntheticSemanticsTests(unittest.TestCase):
    def fixture(self) -> list[dict]:
        return [
            event("pos", 0, 10, prefix=False, previous=False),
            event("pos", 1, 20, prefix=True, previous=False),   # onset anchor
            event("pos", 2, 30, prefix=True, previous=True),    # same-turn post-hit -> L2 1
            event("pos", 3, 40, prefix=True, previous=True, round_id=2, message_index=2),
            event("pos", 4, 50, prefix=True, previous=False, round_id=2, message_index=2),  # second flip, not onset
            event("transient", 0, 5, final=False, prefix=True, previous=False),
            event("neg", 0, 7, final=False, prefix=False, previous=False),
            event("unknown", 0, 9, status="unknown"),
            event("other", 0, 15, final=False, scope="agentlab", event_type="tool_call", prefix=True, previous=True),
        ]

    def test_state_onset_local_masks(self) -> None:
        rows = self.fixture()
        targets, _ = recompute_policy_targets(rows)
        by_id = {row["event_id"]: row for row in rows}

        def label(policy: str, event_id: str):
            return targets[policy][event_id]["label"]

        anchor = "pos::r1::content_segment::1"
        same_turn = "pos::r1::content_segment::2"
        other_turn = "pos::r2::content_segment::3"
        second_flip = "pos::r2::content_segment::4"
        self.assertEqual(label("L0", anchor), 1)
        self.assertEqual(label("L0", same_turn), 0)
        self.assertEqual(label("L0", other_turn), 0)
        self.assertEqual(label("L0", second_flip), 0)
        self.assertEqual(label("L1", same_turn), None)
        self.assertEqual(label("L1", other_turn), None)
        self.assertEqual(label("L2", same_turn), 1)
        self.assertEqual(label("L2", other_turn), None)
        self.assertEqual(label("L3", same_turn), 1)
        self.assertEqual(label("L3", other_turn), 1)
        self.assertEqual(label("L3", second_flip), 1)
        transient = "transient::r1::content_segment::0"
        self.assertEqual(label("L0", transient), 0)
        self.assertEqual(label("L1", transient), None)
        self.assertEqual(label("L2", transient), None)
        self.assertEqual(label("L3", transient), 1)
        self.assertEqual(label("L0", "unknown::r1::content_segment::0"), None)
        self.assertEqual(by_id["unknown::r1::content_segment::0"]["prefix_security"], None)
        self.assertEqual(checker_state(by_id["unknown::r1::content_segment::0"]), (MISSING, MISSING, MISSING))

    def test_local_two_is_same_episode_round_message_only(self) -> None:
        rows = [
            event("ep", 0, 10, prefix=True, previous=False),
            event("ep", 1, 20, prefix=True, previous=True, round_id=1, message_index=2),
            event("ep", 2, 30, prefix=True, previous=True, round_id=2, message_index=2),
        ]
        targets, _ = recompute_policy_targets(rows)
        self.assertEqual(targets["L2"]["ep::r1::content_segment::0"]["label"], 1)
        self.assertEqual(targets["L2"]["ep::r1::content_segment::1"]["label"], None)
        self.assertEqual(targets["L2"]["ep::r2::content_segment::2"]["label"], None)
        self.assertEqual(targets["L1"]["ep::r1::content_segment::1"]["label"], None)
        self.assertEqual(targets["L3"]["ep::r2::content_segment::2"]["label"], 1)

    def test_onset_is_first_flip_not_every_flip(self) -> None:
        rows = [
            event("ep", 0, 10, prefix=True, previous=False),
            event("ep", 1, 20, prefix=True, previous=True),
            event("ep", 2, 30, prefix=True, previous=False),
        ]
        targets, _ = recompute_policy_targets(rows)
        labels = [targets["L0"][row["event_id"]]["label"] for row in rows]
        self.assertEqual(labels, [1, 0, 0])

    def test_conservative_masks_prefix_positive_non_anchor(self) -> None:
        rows = [
            event("ep", 0, 10, prefix=False, previous=False),
            event("ep", 1, 20, prefix=True, previous=False),
            event("ep", 2, 30, prefix=True, previous=True),
        ]
        targets, _ = recompute_policy_targets(rows)
        self.assertEqual(targets["L0"]["ep::r1::content_segment::2"]["label"], 0)
        self.assertEqual(targets["L1"]["ep::r1::content_segment::2"]["label"], None)
        self.assertEqual(targets["L1"]["ep::r1::content_segment::2"]["mask_reason"], "prefix_positive_non_anchor_censored")

    def test_train_arrays_unknown_never_zero(self) -> None:
        rows = self.fixture()
        assignments = assignments_for(rows)
        arrays, summary = build_train_label_arrays(rows, assignments, enforce_expected_counts=False)
        self.assertTrue(np.all(arrays["label_L0"][~arrays["exact"]] == MISSING))
        self.assertTrue(np.all(arrays["mask_L0"][~arrays["exact"]]))
        self.assertTrue(summary["unknown_never_zero"])
        self.assertTrue(np.array_equal(arrays["prefix_state"], arrays["label_L3"]))

    def test_assignment_tamper_is_rejected(self) -> None:
        rows = [
            event("ep", 0, 10, prefix=False, previous=False),
            event("ep", 1, 20, prefix=True, previous=False),
        ]
        assignments = assignments_for(rows)
        assignments["ep::r1::content_segment::0"]["L0"] = {"label": 1, "mask": False, "mask_reason": None}
        with self.assertRaises(LabelContractError):
            build_train_label_arrays(rows, assignments, enforce_expected_counts=False)

    def test_dev_truth_has_no_training_policy(self) -> None:
        rows = [
            event("dev", 0, 10, prefix=True, previous=False),
            event("dev", 1, 20, status="unknown"),
        ]
        for row in rows:
            row["split"] = "dev"
        arrays, summary = build_dev_label_arrays(rows)
        self.assertEqual(set(arrays), {"prefix_state", "previous_state", "security_flip", "exact"})
        self.assertNotIn("label_L0", arrays)
        self.assertEqual(arrays["prefix_state"].tolist(), [1, MISSING])
        self.assertEqual(summary["unknown_rows"], 1)

    def test_episode_support_conflict_and_transient(self) -> None:
        support = episode_label_support(
            n_positions=3,
            episode_ct_mask=np.array([True, True, False]),
            episode_prefix_state=np.array([1, MISSING, 0], dtype=np.int8),
            episode_security_flip=np.array([1, MISSING, 0], dtype=np.int8),
            episode_bag_label=False,
            policy_labels={"L0": np.array([0, MISSING, MISSING], dtype=np.int8)},
            policy_masks={"L0": np.array([False, True, True])},
        )
        self.assertTrue(support.transient_prefix_true)
        self.assertEqual(support.n_unknown, 1)
        self.assertEqual(support.n_prefix_true_ct, 1)
        self.assertEqual(support.n_prefix_true_e, 0)
        self.assertEqual(support.policy_positive["L0"], 0)
        self.assertEqual(support.policy_negative["L0"], 1)


if __name__ == "__main__":
    unittest.main()
