from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from analyze_stream_results import main as analyze_main
from analyze_stream_results import official_decisions
from guard_eval.dataset import GuardSample
from guard_eval.latency_metrics import (
    build_sample_latency_rows,
    spearman_context_latency,
    summarize_latencies,
)
from guard_eval.policies import OnlinePolicyTracker, evaluate_all_policies
from guard_eval.safety_metrics import binary_safety_metrics
from guard_eval.sampling import create_sample_manifest, select_samples_from_manifest
from guard_eval.sequence import prepare_sequence
from guard_eval.result_io import read_json, write_jsonl


class FakeTokenizer:
    token_to_id = {
        "<|im_start|>": 100,
        "<|im_end|>": 101,
        "<think>": 102,
        "</think>": 103,
        "user": 200,
        "assistant": 201,
        "\n": 10,
        "\n\n": 11,
        "prompt": 300,
        "reasoning": 301,
        "answer": 302,
    }
    id_to_token = {value: key for key, value in token_to_id.items()}

    def __init__(self) -> None:
        self.ids = [
            100,
            200,
            10,
            300,
            101,
            10,
            100,
            201,
            10,
            102,
            10,
            301,
            103,
            11,
            302,
            101,
        ]

    def apply_chat_template(self, *_args, **_kwargs):
        return "rendered"

    def __call__(self, *_args, **_kwargs):
        return {"input_ids": list(self.ids)}

    def convert_tokens_to_ids(self, token):
        return self.token_to_id[token]

    def decode(self, token_ids, **_kwargs):
        return self.id_to_token[token_ids[0]]


def fake_samples(count: int) -> list[GuardSample]:
    return [
        GuardSample(
            split="thinking",
            unique_id=f"sample-{index}",
            messages=(
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "answer"},
            ),
            label="Safe" if index % 2 == 0 else "Unsafe",
        )
        for index in range(count)
    ]


class PolicyTests(unittest.TestCase):
    def test_online_strict_uses_equal_non_safe_pairs(self) -> None:
        decisions = evaluate_all_policies(
            ["Controversial", "Controversial", "Safe", "Unsafe", "Unsafe"],
            full_index_offset=20,
        )
        self.assertEqual(decisions["online_strict"]["trigger_index"], 1)
        self.assertEqual(decisions["online_strict"]["trigger_full_index"], 21)
        self.assertEqual(decisions["online_loose"]["trigger_index"], 4)
        self.assertEqual(decisions["official_strict"]["trigger_index"], 4)
        self.assertEqual(decisions["official_loose"]["trigger_index"], 4)

        mixed = evaluate_all_policies(["Unsafe", "Controversial"])
        self.assertFalse(mixed["online_strict"]["triggered"])
        self.assertFalse(mixed["online_loose"]["triggered"])
        self.assertFalse(mixed["official_strict"]["triggered"])

        controversial = evaluate_all_policies(
            ["Safe", "Controversial", "Controversial"]
        )
        self.assertEqual(
            controversial["online_strict"]["prediction"], "Unsafe"
        )
        self.assertEqual(controversial["online_loose"]["prediction"], "Safe")
        self.assertEqual(
            controversial["official_strict"]["prediction"], "Unsafe"
        )
        self.assertEqual(controversial["official_loose"]["prediction"], "Safe")

    def test_streaming_tracker_uses_the_same_online_rules(self) -> None:
        tracker = OnlinePolicyTracker()
        tracker.update("Unsafe")
        mixed = tracker.update("Controversial")
        self.assertEqual(mixed["newly_triggered"], [])
        repeated = tracker.update("Controversial")
        self.assertEqual(repeated["newly_triggered"], ["online_strict"])
        decisions = tracker.decisions()
        self.assertEqual(decisions["online_strict"]["prediction"], "Unsafe")
        self.assertEqual(decisions["online_loose"]["prediction"], "Safe")

    def test_official_analysis_recomputes_both_policies_from_raw_ids(self) -> None:
        decisions = official_decisions({"pred_risk_levels": [0, 2, 2]})
        self.assertEqual(decisions["strict"], "Unsafe")
        self.assertEqual(decisions["loose"], "Safe")

    def test_binary_metrics_include_fpr_fnr_and_f1(self) -> None:
        metrics = binary_safety_metrics(
            ["Unsafe", "Unsafe", "Safe", "Safe"],
            ["Unsafe", "Safe", "Unsafe", "Safe"],
        )
        self.assertEqual((metrics["tp"], metrics["tn"]), (1, 1))
        self.assertEqual((metrics["fp"], metrics["fn"]), (1, 1))
        self.assertEqual(metrics["fpr"], 0.5)
        self.assertEqual(metrics["fnr"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)


class SamplingTests(unittest.TestCase):
    def test_manifest_is_reproducible_and_resolves_exact_samples(self) -> None:
        samples = fake_samples(20)
        first = create_sample_manifest(
            samples,
            sample_count=10,
            seed=42,
            dataset_path="dataset",
        )
        second = create_sample_manifest(
            samples,
            sample_count=10,
            seed=42,
            dataset_path="dataset",
        )
        self.assertEqual(first, second)

        selected = select_samples_from_manifest(samples, first)
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            [sample.unique_id for _, _, sample in selected],
            [entry["sample_id"] for entry in first["samples"]],
        )


class LatencyTests(unittest.TestCase):
    def test_token_and_sample_latency_statistics(self) -> None:
        tokens = []
        for sample_number in range(1, 6):
            for token_index in range(sample_number):
                tokens.append(
                    {
                        "sample_id": f"s{sample_number}",
                        "context_length": sample_number * 10 + token_index,
                        "latency_ms": float(sample_number),
                    }
                )

        summary = summarize_latencies(row["latency_ms"] for row in tokens)
        self.assertEqual(summary["token_count"], 15)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 5.0)
        self.assertIsNotNone(summary["tokens_per_second"])

        samples = build_sample_latency_rows(tokens)
        correlation = spearman_context_latency(samples)
        self.assertAlmostEqual(correlation["spearman_rho"], 1.0)


class SequenceTests(unittest.TestCase):
    def test_thinking_boundary_matches_official_rule(self) -> None:
        tokenizer = FakeTokenizer()
        messages = (
            {"role": "user", "content": "prompt"},
            {"role": "assistant", "content": "reasoning answer"},
        )
        sequence = prepare_sequence(
            tokenizer,
            messages,
            "thinking",
            annotated_input_ids=tokenizer.ids,
            require_alignment=True,
        )
        self.assertEqual(sequence.user_end_index, 4)
        self.assertEqual(sequence.eval_start_index, 11)
        self.assertEqual(sequence.prompt_ids[-1], 101)
        self.assertEqual(sequence.response_ids[0], 301)


class AnalysisTests(unittest.TestCase):
    def test_analysis_compares_all_three_methods_and_both_policies(self) -> None:
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            raw = root / "custom" / "raw"
            raw.mkdir(parents=True)

            safe_decision = {"prediction": "Safe"}
            unsafe_decision = {"prediction": "Unsafe"}
            sample_records = [
                {
                    "sample_id": "safe",
                    "sample_order": 0,
                    "dataset_index": 2,
                    "label": "Safe",
                    "stream_policies": {
                        "strict": safe_decision,
                        "loose": safe_decision,
                    },
                    "full_forward_policies": {
                        "strict": safe_decision,
                        "loose": safe_decision,
                    },
                    "agreement": {
                        "token_count": 1,
                        "token_label_mismatch_count": 0,
                    },
                },
                {
                    "sample_id": "unsafe",
                    "sample_order": 1,
                    "dataset_index": 7,
                    "label": "Unsafe",
                    "stream_policies": {
                        "strict": unsafe_decision,
                        "loose": unsafe_decision,
                    },
                    "full_forward_policies": {
                        "strict": unsafe_decision,
                        "loose": unsafe_decision,
                    },
                    "agreement": {
                        "token_count": 2,
                        "token_label_mismatch_count": 0,
                    },
                },
            ]
            token_records = [
                {"sample_id": "safe", "context_length": 10, "latency_ms": 2.0},
                {"sample_id": "unsafe", "context_length": 20, "latency_ms": 3.0},
                {"sample_id": "unsafe", "context_length": 21, "latency_ms": 3.0},
            ]
            full_records = [
                {
                    "sample_id": "safe",
                    "sample_order": 0,
                    "input_ids": [1, 2],
                    "eval_start_index": 1,
                    "risk_levels": ["Safe"],
                },
                {
                    "sample_id": "unsafe",
                    "sample_order": 1,
                    "input_ids": [1, 3, 4],
                    "eval_start_index": 1,
                    "risk_levels": ["Unsafe", "Unsafe"],
                },
            ]
            official_records = [
                {
                    "unique_id": "safe",
                    "sample_order": 0,
                    "input_ids": [1, 2],
                    "split_idx_eval": 1,
                    "pred_risk_levels": [0],
                },
                {
                    "unique_id": "unsafe",
                    "sample_order": 1,
                    "input_ids": [1, 3, 4],
                    "split_idx_eval": 1,
                    "pred_risk_levels": [1, 1],
                },
            ]
            write_jsonl(raw / "sample_records.jsonl", sample_records)
            write_jsonl(raw / "token_records.jsonl", token_records)
            write_jsonl(raw / "full_forward_records.jsonl", full_records)
            official_path = root / "official.jsonl"
            write_jsonl(official_path, official_records)
            analysis = root / "analysis"

            argv = [
                "analyze_stream_results.py",
                "--custom-dir",
                str(root / "custom"),
                "--official-output",
                str(official_path),
                "--output-dir",
                str(analysis),
            ]
            with patch("sys.argv", argv):
                analyze_main()

            summary = read_json(analysis / "summary.json")
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(len(summary["metrics"]), 6)
            self.assertEqual(
                summary["custom_full_forward_vs_official_script"][
                    "token_label_comparison"
                ]["token_label_agreement"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
