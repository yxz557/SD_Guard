from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analyze_judgments import load_judgments
from run_experiment import _assert_blind_record
from sanity_check import first_mismatch, summarize_latency_runs
from sdguard_sd.generation import stable_sample_seed
from sdguard_sd.io_utils import write_jsonl
from sdguard_sd.metrics import confusion_metrics, is_unsafe
from sdguard_sd.policies import evaluate_policies
from sdguard_sd.prompts import STYLE_CONTROL_SYSTEM_PROMPT, build_generation_messages
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
    def test_three_class_policy_mapping(self) -> None:
        self.assertTrue(is_unsafe("Controversial", "strict"))
        self.assertFalse(is_unsafe("Controversial", "loose"))
        self.assertTrue(is_unsafe("Unsafe", "loose"))

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
                        "full_trace": {"label": "Controversial", "categories": ["边界"]},
                        "answer_only": {"label": "Safe", "categories": []},
                    }
                ],
            )
            loaded = load_judgments(path)
            self.assertEqual(loaded["x"]["full_trace"]["label"], "Controversial")
            self.assertEqual(loaded["x"]["answer_only"]["label"], "Safe")


class ReproducibilityTests(unittest.TestCase):
    def test_per_sample_seed_is_stable_and_order_independent(self) -> None:
        self.assertEqual(stable_sample_seed(42, "abc"), stable_sample_seed(42, "abc"))
        self.assertNotEqual(stable_sample_seed(42, "abc"), stable_sample_seed(42, "def"))

    def test_first_mismatch_reports_content_and_length_changes(self) -> None:
        self.assertIsNone(first_mismatch((1, 2), (1, 2)))
        self.assertEqual(first_mismatch((1, 2), (1, 3)), 1)
        self.assertEqual(first_mismatch((1, 2), (1, 2, 3)), 2)


class StyleControlTests(unittest.TestCase):
    def test_generation_messages_include_control_for_draft_and_target(self) -> None:
        messages = build_generation_messages(
            "question", system_prompt=STYLE_CONTROL_SYSTEM_PROMPT
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[-1]["content"], "question")

    def test_guard_messages_can_be_rebuilt_without_control_prompt(self) -> None:
        messages = build_generation_messages("question", system_prompt=None)
        self.assertEqual(messages, [{"role": "user", "content": "question"}])

    def test_empty_control_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_generation_messages("question", system_prompt="   ")


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
