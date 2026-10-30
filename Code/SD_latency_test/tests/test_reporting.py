from __future__ import annotations

import unittest
from types import SimpleNamespace

from sd_latency.generation import (
    GenerationResult,
    GenerationSettings,
    SpeculativeRound,
    _configure_draft,
    _generation_config,
    _generation_overrides,
)
from sd_latency.reporting import compare_outputs, first_mismatch, summarize_pair


class DifferenceTests(unittest.TestCase):
    def test_first_mismatch_covers_content_and_length(self) -> None:
        self.assertIsNone(first_mismatch((1, 2), (1, 2)))
        self.assertEqual(first_mismatch((1, 2), (1, 3)), 1)
        self.assertEqual(first_mismatch((1, 2), (1, 2, 3)), 2)

    def test_compare_outputs_reports_prefix_and_similarity(self) -> None:
        result = compare_outputs((1, 2, 3), (1, 2, 4), "abc", "abd")
        self.assertFalse(result["exact_token_match"])
        self.assertEqual(result["first_mismatch_index"], 2)
        self.assertEqual(result["common_prefix_token_count"], 2)
        self.assertGreater(result["text_similarity_ratio"], 0.5)


class LatencyTests(unittest.TestCase):
    def test_speedup_greater_than_one_means_sd_is_faster(self) -> None:
        target = [
            {"total_ms": 200.0, "generate_ms": 190.0, "generated_token_count": 20},
            {"total_ms": 220.0, "generate_ms": 210.0, "generated_token_count": 20},
        ]
        assisted = [
            {"total_ms": 100.0, "generate_ms": 90.0, "generated_token_count": 20},
            {"total_ms": 110.0, "generate_ms": 100.0, "generated_token_count": 20},
        ]
        result = summarize_pair(target, assisted)
        self.assertEqual(result["latency_speedup"], 2.0)
        self.assertEqual(result["throughput_speedup"], 2.0)
        self.assertTrue(result["paired_token_counts_match"])

    def test_phase_summary_does_not_double_count_target_cuda_subset(self) -> None:
        round_ = SpeculativeRound(
            round_index=0,
            prefix_length=10,
            proposed_ids=(1, 2),
            accepted_count=1,
            committed_ids=(1, 3),
            commit_kind="target_correction",
            draft_proposal_ms=2.0,
            proposed_trace_ms=0.5,
            verify_and_decide_ms=4.0,
            target_forward_cuda_ms=3.0,
            result_trace_ms=0.25,
            strategy_update_ms=0.25,
        )
        result = GenerationResult(
            prompt_ids=(9,),
            generated_ids=(1, 3),
            preparation_ms=1.0,
            generate_ms=10.0,
            output_transfer_ms=1.0,
            total_ms=12.0,
            rounds=(round_,),
            traced=True,
            stopped_on_eos=False,
            truncated=False,
        )
        phases = result.phase_summary()
        self.assertEqual(phases["profiled_round_phases_ms"], 7.0)
        self.assertEqual(phases["target_forward_cuda_ms"], 3.0)
        self.assertEqual(phases["generate_framework_other_estimate_ms"], 3.0)


class GenerationModeTests(unittest.TestCase):
    def test_copied_config_contains_explicit_assisted_generation_controls(self) -> None:
        model = SimpleNamespace(
            generation_config=SimpleNamespace(
                max_new_tokens=20,
                do_sample=True,
                num_beams=1,
                use_cache=True,
                num_assistant_tokens=20.0,
                num_assistant_tokens_schedule="heuristic",
                assistant_confidence_threshold=0.0,
                eos_token_id=1,
            )
        )

        config = _generation_config(model, GenerationSettings(do_sample=False))

        self.assertFalse(config.do_sample)
        self.assertEqual(config.num_assistant_tokens, 5.0)
        self.assertEqual(config.num_assistant_tokens_schedule, "constant")
        self.assertEqual(config.assistant_confidence_threshold, 0.4)

    def test_greedy_is_passed_as_an_explicit_generate_override(self) -> None:
        overrides = _generation_overrides(GenerationSettings(do_sample=False))

        self.assertEqual(overrides, {"do_sample": False})

    def test_sampling_parameters_are_passed_as_generate_overrides(self) -> None:
        settings = GenerationSettings(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=12,
        )

        self.assertEqual(
            _generation_overrides(settings),
            {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 12,
                "min_p": None,
            },
        )

    def test_draft_model_default_cannot_restore_sampling_in_greedy_mode(self) -> None:
        draft = SimpleNamespace(
            generation_config=SimpleNamespace(
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.1,
                num_assistant_tokens=20.0,
                num_assistant_tokens_schedule="heuristic",
                assistant_confidence_threshold=0.0,
            )
        )

        _configure_draft(draft, GenerationSettings(do_sample=False))

        self.assertFalse(draft.generation_config.do_sample)
        self.assertEqual(draft.generation_config.num_assistant_tokens, 5.0)
        self.assertEqual(
            draft.generation_config.num_assistant_tokens_schedule, "constant"
        )
        self.assertEqual(draft.generation_config.assistant_confidence_threshold, 0.4)


if __name__ == "__main__":
    unittest.main()
