from __future__ import annotations

import tempfile
import unittest
import random
from pathlib import Path

from analyze_judgments import load_judgments
from run_experiment import _assert_blind_record
from sanity_check import first_mismatch, summarize_latency_runs
from sdguard_sd.generation import stable_sample_seed
from sdguard_sd.io_utils import write_jsonl
from sdguard_sd.inline_guard import InlineDecisionTracker
from sdguard_sd.metrics import confusion_metrics, is_unsafe
from sdguard_sd.policies import evaluate_policies
from sdguard_sd.text_split import split_completion


class FakeTokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"</think>": 9}.get(token, 0)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        pieces = {1: "想", 2: "法", 3: " ", 4: "答", 5: "案", 9: "</think>"}
        return "".join(
            "" if skip_special_tokens and value == 9 else pieces[value] for value in ids
        )


class PolicyTests(unittest.TestCase):
    def test_strict_and_loose_choose_earliest_pair(self) -> None:
        result = evaluate_policies(
            ["Safe", "Controversial", "Controversial", "Unsafe", "Unsafe"]
        )
        self.assertEqual(result["strict"]["trigger_generated_index"], 2)
        self.assertEqual(result["strict"]["trigger_pair"], ["Controversial"] * 2)
        self.assertEqual(result["loose"]["trigger_generated_index"], 4)

    def test_generated_offset_is_preserved(self) -> None:
        result = evaluate_policies(["Unsafe", "Unsafe"], generated_index_offset=17)
        self.assertEqual(result["loose"]["trigger_generated_index"], 18)


def guard_record(label: str, token_id: int) -> dict:
    return {
        "token_id": token_id,
        "token_text": str(token_id),
        "risk_level": label,
        "risk_probability": 1.0,
        "category": "test",
        "category_probability": 1.0,
    }


class InlineGuardPolicyTests(unittest.TestCase):
    def test_rejected_proposal_latches_noise_without_leaking_to_next_round(self) -> None:
        tracker = InlineDecisionTracker(enable_thinking=False, close_think_token_id=99)
        tracker.observe_proposal(
            round_index=0,
            prefix_generated_count=0,
            token_ids=[1, 2],
            token_records=[guard_record("Unsafe", 1), guard_record("Unsafe", 2)],
        )
        tracker.observe_commit(
            round_index=0,
            accepted_count=0,
            committed_ids=[3],
            commit_kind="target_correction",
            token_records=[guard_record("Safe", 3)],
        )
        tracker.observe_proposal(
            round_index=1,
            prefix_generated_count=1,
            token_ids=[4],
            token_records=[guard_record("Unsafe", 4)],
        )
        tracker.observe_commit(
            round_index=1,
            accepted_count=0,
            committed_ids=[5],
            commit_kind="target_correction",
            token_records=[guard_record("Safe", 5)],
        )
        result = tracker.finalize(answer_start_generated_index=0)
        self.assertTrue(
            result["proposal_immediate"]["scopes"]["full_trace"]["decisions"][
                "loose"
            ]["triggered"]
        )
        self.assertFalse(
            result["verified_commit"]["scopes"]["full_trace"]["decisions"][
                "loose"
            ]["triggered"]
        )
        self.assertTrue(result["noise"]["full_trace"]["loose"]["proposal_only_alert"])
        self.assertFalse(
            tracker.round_records[1]["proposal_scopes"]["full_trace"]["decisions"][
                "loose"
            ]["triggered"]
        )

    def test_accepted_proposal_trigger_survives_verification(self) -> None:
        tracker = InlineDecisionTracker(enable_thinking=False, close_think_token_id=99)
        records = [guard_record("Unsafe", 1), guard_record("Unsafe", 2)]
        tracker.observe_proposal(
            round_index=0,
            prefix_generated_count=0,
            token_ids=[1, 2],
            token_records=records,
        )
        tracker.observe_commit(
            round_index=0,
            accepted_count=2,
            committed_ids=[1, 2],
            commit_kind="terminal_candidate",
            token_records=records,
        )
        result = tracker.finalize(answer_start_generated_index=0)
        self.assertTrue(result["noise"]["full_trace"]["loose"]["verified_commit_alert"])
        self.assertTrue(
            result["noise"]["full_trace"]["loose"][
                "first_alert_survived_verification"
            ]
        )

    def test_target_corrections_are_included_in_both_final_views(self) -> None:
        tracker = InlineDecisionTracker(enable_thinking=False, close_think_token_id=99)
        for round_index, committed_id in enumerate((10, 11)):
            tracker.observe_proposal(
                round_index=round_index,
                prefix_generated_count=round_index,
                token_ids=[20 + round_index],
                token_records=[guard_record("Safe", 20 + round_index)],
            )
            tracker.observe_commit(
                round_index=round_index,
                accepted_count=0,
                committed_ids=[committed_id],
                commit_kind="target_correction",
                token_records=[guard_record("Unsafe", committed_id)],
            )
        result = tracker.finalize(answer_start_generated_index=0)
        self.assertTrue(
            result["verified_commit"]["scopes"]["full_trace"]["decisions"][
                "loose"
            ]["triggered"]
        )
        self.assertTrue(
            result["proposal_immediate"]["scopes"]["full_trace"]["decisions"][
                "loose"
            ]["triggered"]
        )

    def test_verified_chunking_matches_one_shot_policy(self) -> None:
        labels = ("Safe", "Controversial", "Unsafe")
        for seed in range(50):
            rng = random.Random(seed)
            tracker = InlineDecisionTracker(enable_thinking=False, close_think_token_id=99)
            final_labels: list[str] = []
            next_token = 1
            for round_index in range(8):
                proposal_count = rng.randint(1, 5)
                proposed_ids = list(range(next_token, next_token + proposal_count))
                next_token += proposal_count
                proposal_labels = [rng.choice(labels) for _ in proposed_ids]
                accepted = rng.randint(0, proposal_count)
                correction_id = next_token
                next_token += 1
                correction_label = rng.choice(labels)
                committed_ids = proposed_ids[:accepted] + [correction_id]
                committed_labels = proposal_labels[:accepted] + [correction_label]
                tracker.observe_proposal(
                    round_index=round_index,
                    prefix_generated_count=len(final_labels),
                    token_ids=proposed_ids,
                    token_records=[
                        guard_record(label, token_id)
                        for label, token_id in zip(proposal_labels, proposed_ids)
                    ],
                )
                tracker.observe_commit(
                    round_index=round_index,
                    accepted_count=accepted,
                    committed_ids=committed_ids,
                    commit_kind="target_correction",
                    token_records=[
                        guard_record(label, token_id)
                        for label, token_id in zip(committed_labels, committed_ids)
                    ],
                )
                final_labels.extend(committed_labels)
            result = tracker.finalize(answer_start_generated_index=0)
            expected = evaluate_policies(final_labels)
            actual = result["verified_commit"]["scopes"]["full_trace"]["decisions"]
            for policy in ("strict", "loose"):
                self.assertEqual(actual[policy]["prediction"], expected[policy]["prediction"])
                self.assertEqual(
                    actual[policy]["trigger_generated_index"],
                    expected[policy]["trigger_generated_index"],
                )


class CompletionTests(unittest.TestCase):
    def test_thinking_and_answer_split_by_token_id(self) -> None:
        result = split_completion(FakeTokenizer(), [1, 2, 9, 3, 4, 5])
        self.assertEqual(result.thinking, "想法")
        self.assertEqual(result.final_answer, "答案")
        self.assertEqual(result.answer_start_generated_index, 3)
        self.assertEqual(result.parse_status, "ok")

    def test_missing_close_think_marks_answer_unavailable(self) -> None:
        result = split_completion(FakeTokenizer(), [1, 2])
        self.assertIsNone(result.answer_start_generated_index)
        self.assertEqual(result.parse_status, "missing_close_think")

    def test_thinking_disabled_treats_all_tokens_as_answer(self) -> None:
        result = split_completion(FakeTokenizer(), [4, 5], thinking_expected=False)
        self.assertEqual(result.thinking, "")
        self.assertEqual(result.final_answer, "答案")
        self.assertEqual(result.answer_start_generated_index, 0)
        self.assertEqual(result.parse_status, "thinking_disabled")


class MetricTests(unittest.TestCase):
    def test_binary_mapping_is_shared_by_both_guard_policies(self) -> None:
        self.assertTrue(is_unsafe("Unsafe", "strict"))
        self.assertTrue(is_unsafe("Unsafe", "loose"))
        self.assertFalse(is_unsafe("Safe", "strict"))
        self.assertFalse(is_unsafe("Safe", "loose"))
        with self.assertRaises(ValueError):
            is_unsafe("Controversial", "strict")

    def test_confusion_metrics(self) -> None:
        result = confusion_metrics([(True, True), (False, False), (False, True), (True, False)])
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]), (1, 1, 1, 1))
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["false_positive_rate"], 0.5)
        self.assertEqual(result["false_negative_rate"], 0.5)


class BlindnessTests(unittest.TestCase):
    def test_blind_schema_rejects_guard_fields(self) -> None:
        _assert_blind_record({"sample_id": "x", "prompt": "p", "thinking": "t"})
        with self.assertRaises(AssertionError):
            _assert_blind_record({"sample_id": "x", "guard_prediction": "Safe"})

    def test_judgment_loader_accepts_nested_schema(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "labels.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "sample_id": "x",
                        "full_trace": {"label": "Unsafe", "categories": ["Unethical Acts"]},
                        "answer_only": {"label": "Safe", "categories": []},
                    }
                ],
            )
            loaded = load_judgments(path)
            self.assertEqual(loaded["x"]["full_trace"]["label"], "Unsafe")
            self.assertEqual(loaded["x"]["answer_only"]["label"], "Safe")

    def test_judgment_loader_rejects_controversial(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "labels.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "sample_id": "x",
                        "full_trace": {"label": "Controversial", "categories": []},
                        "answer_only": {"label": "Safe", "categories": []},
                    }
                ],
            )
            with self.assertRaises(ValueError):
                load_judgments(path)


class ReproducibilityTests(unittest.TestCase):
    def test_per_sample_seed_is_stable_and_order_independent(self) -> None:
        self.assertEqual(stable_sample_seed(42, "abc"), stable_sample_seed(42, "abc"))
        self.assertNotEqual(stable_sample_seed(42, "abc"), stable_sample_seed(42, "def"))

    def test_same_sample_seed_can_be_reused_across_all_matrix_cells(self) -> None:
        expected = stable_sample_seed(42, "sample-17")
        matrix_seeds = [
            stable_sample_seed(42, "sample-17")
            for _thinking in (True, False)
            for _mode in ("target_only", "speculative", "draft_only", "inline_guard")
        ]
        self.assertEqual(matrix_seeds, [expected] * 8)

    def test_first_mismatch_reports_content_and_length_changes(self) -> None:
        self.assertIsNone(first_mismatch((1, 2), (1, 2)))
        self.assertEqual(first_mismatch((1, 2), (1, 3)), 1)
        self.assertEqual(first_mismatch((1, 2), (1, 2, 3)), 2)


class LatencySummaryTests(unittest.TestCase):
    def test_speedup_greater_than_one_means_sd_is_faster(self) -> None:
        target = [
            {"latency_ms": 200.0, "generated_token_count": 20},
            {"latency_ms": 220.0, "generated_token_count": 20},
        ]
        assisted = [
            {"latency_ms": 100.0, "generated_token_count": 20},
            {"latency_ms": 110.0, "generated_token_count": 20},
        ]
        result = summarize_latency_runs(target, assisted)
        self.assertEqual(result["median_latency_speedup"], 2.0)
        self.assertEqual(result["throughput_speedup"], 2.0)
        self.assertTrue(result["all_generated_token_counts_match"])


if __name__ == "__main__":
    unittest.main()
