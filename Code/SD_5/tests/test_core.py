from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from sdguard_sd.inline_guard import InlineDecisionTracker  # noqa: E402
from sdguard_sd.runtime import (  # noqa: E402
    ensure_legacy_default_rope_initializer,
    ensure_legacy_masking_api,
    ensure_pad_token_id,
)
from sdguard_sd.seeding import stable_sample_seed  # noqa: E402
from sdguard_sd.timing import summarize_ms  # noqa: E402
from sdguard_sd.vllm_profile import classify_profile_event  # noqa: E402
from sdguard_sd.vllm_extension import SD4WorkerExtension, _v2_commit_plan  # noqa: E402
from sdguard_sd.vllm_worker import _spec_metrics  # noqa: E402


def guard_record(label: str, token_id: int) -> dict[str, object]:
    return {
        "token_id": token_id,
        "token_text": str(token_id),
        "risk_level": label,
        "risk_probability": 0.9,
        "category": "x",
        "category_probability": 0.8,
    }


class CoreTests(unittest.TestCase):
    def test_missing_pad_token_id_is_backfilled_for_qwen3guard(self) -> None:
        config = SimpleNamespace()
        tokenizer = SimpleNamespace(pad_token_id=151643, eos_token_id=151645)
        returned = ensure_pad_token_id(config, tokenizer)
        self.assertIs(returned, config)
        self.assertEqual(config.pad_token_id, 151643)

    def test_existing_pad_token_id_is_preserved(self) -> None:
        config = SimpleNamespace(pad_token_id=7)
        tokenizer = SimpleNamespace(pad_token_id=151643, eos_token_id=151645)
        ensure_pad_token_id(config, tokenizer)
        self.assertEqual(config.pad_token_id, 7)

    def test_missing_default_rope_initializer_is_restored(self) -> None:
        try:
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        except ModuleNotFoundError:
            self.skipTest("transformers is not installed in this test environment")

        sentinel = object()
        previous = ROPE_INIT_FUNCTIONS.pop("default", sentinel)
        try:
            self.assertTrue(ensure_legacy_default_rope_initializer())
            initializer = ROPE_INIT_FUNCTIONS["default"]
            self.assertFalse(ensure_legacy_default_rope_initializer())

            config = SimpleNamespace(
                rope_theta=10000.0,
                head_dim=8,
                hidden_size=8,
                num_attention_heads=1,
            )
            inv_freq, attention_factor = initializer(config, device="cpu")
            self.assertEqual(tuple(inv_freq.shape), (4,))
            self.assertAlmostEqual(float(inv_freq[0]), 1.0)
            self.assertEqual(attention_factor, 1.0)
        finally:
            ROPE_INIT_FUNCTIONS.pop("default", None)
            if previous is not sentinel:
                ROPE_INIT_FUNCTIONS["default"] = previous

    def test_transformers_five_mask_api_is_adapted_for_qwen3guard(self) -> None:
        try:
            from transformers import masking_utils
        except ModuleNotFoundError:
            self.skipTest("transformers is not installed in this test environment")

        calls: list[dict[str, object]] = []

        def modern_mask_factory(
            *,
            config: object,
            inputs_embeds: object,
            attention_mask: object,
            past_key_values: object,
            position_ids: object,
        ) -> tuple[object, object]:
            calls.append(
                {
                    "config": config,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                }
            )
            return inputs_embeds, past_key_values

        with (
            patch.object(masking_utils, "create_causal_mask", modern_mask_factory),
            patch.object(
                masking_utils,
                "create_sliding_window_causal_mask",
                modern_mask_factory,
            ),
        ):
            self.assertTrue(ensure_legacy_masking_api())
            self.assertFalse(ensure_legacy_masking_api())
            marker = object()
            cache = object()
            result = masking_utils.create_causal_mask(
                config=object(),
                input_embeds=marker,
                attention_mask=None,
                cache_position=object(),
                past_key_values=cache,
                position_ids=object(),
            )

        self.assertEqual(result, (marker, cache))
        self.assertIs(calls[0]["inputs_embeds"], marker)
        self.assertNotIn("cache_position", calls[0])

    def test_seed_matches_sd3_contract(self) -> None:
        expected = int.from_bytes(
            hashlib.sha256(b"42:sample-a").digest()[:8], "big"
        ) % (2**31 - 1)
        self.assertEqual(stable_sample_seed(42, "sample-a"), expected)
        self.assertEqual(
            stable_sample_seed(42, "sample-a"),
            stable_sample_seed(42, "sample-a"),
        )
        self.assertNotEqual(
            stable_sample_seed(42, "sample-a"),
            stable_sample_seed(42, "sample-b"),
        )

    def test_latency_summary_has_required_quantiles(self) -> None:
        summary = summarize_ms([1, 2, 3, 4, 5])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["median"], 3)
        self.assertIn("p90", summary)
        self.assertIn("p95", summary)
        self.assertEqual(summary["min"], 1)
        self.assertEqual(summary["max"], 5)

    def test_tracker_records_accepted_prefix_without_changing_tokens(self) -> None:
        tracker = InlineDecisionTracker(enable_thinking=False, close_think_token_id=99)
        tracker.observe_proposal(
            round_index=0,
            prefix_generated_count=0,
            token_ids=[10, 11, 12],
            token_records=[
                guard_record("Safe", 10),
                guard_record("Unsafe", 11),
                guard_record("Unsafe", 12),
            ],
        )
        tracker.observe_commit(
            round_index=0,
            accepted_count=2,
            committed_ids=[10, 11, 20],
            commit_kind="target_correction",
            token_records=[
                guard_record("Safe", 10),
                guard_record("Unsafe", 11),
                guard_record("Safe", 20),
            ],
        )
        result = tracker.finalize(answer_start_generated_index=0)
        self.assertEqual(tracker.committed_ids, [10, 11, 20])
        self.assertEqual(result["acceptance_by_proposal_risk"]["Safe"]["accepted"], 1)
        self.assertEqual(
            result["acceptance_by_proposal_risk"]["Unsafe"]["accepted"], 1
        )

    def test_terminal_unverified_proposal_can_be_discarded(self) -> None:
        tracker = InlineDecisionTracker(enable_thinking=False, close_think_token_id=99)
        tracker.observe_proposal(
            round_index=0,
            prefix_generated_count=0,
            token_ids=[1, 2],
            token_records=[guard_record("Safe", 1), guard_record("Safe", 2)],
        )
        tracker.discard_pending(reason="unverified_after_stop")
        result = tracker.finalize(answer_start_generated_index=0)
        self.assertEqual(tracker.committed_ids, [])
        self.assertEqual(tracker.round_records[0]["commit_kind"], "unverified_after_stop")
        self.assertEqual(result["acceptance_by_proposal_risk"]["Safe"]["rejected"], 2)

    def test_vllm_detailed_metrics_are_required_and_normalized(self) -> None:
        metrics = SimpleNamespace(
            per_step_accepted=[2, 0],
            per_step_drafted=[5, 5],
            num_draft_tokens=10,
            num_accepted_draft_tokens=2,
        )
        completion = SimpleNamespace(spec_decode_metrics=metrics)
        request = SimpleNamespace(outputs=[completion])
        normalized = _spec_metrics(request)
        self.assertEqual(normalized["round_count"], 2)
        self.assertEqual(normalized["proposed_token_count"], 10)
        self.assertEqual(normalized["accepted_draft_token_count"], 2)

    def test_vllm_public_metrics_location_is_supported(self) -> None:
        metrics = SimpleNamespace(
            per_step_accepted=[1],
            per_step_drafted=[5],
            num_draft_tokens=5,
            num_accepted_draft_tokens=1,
        )
        completion = SimpleNamespace(spec_decode_metrics=None)
        request = SimpleNamespace(
            outputs=[completion],
            metrics=SimpleNamespace(speculative_decoding=metrics),
        )
        normalized = _spec_metrics(request)
        self.assertEqual(normalized["per_step_accepted"], [1])
        self.assertEqual(normalized["per_step_drafted"], [5])

    def test_profile_ranges_include_guard_and_draft(self) -> None:
        guard_event = {"name": "sd4_guard:proposal", "args": {}}
        draft_event = {"name": "gpu_model_runner: draft", "args": {}}
        self.assertEqual(classify_profile_event(guard_event), "guard_proposal")
        self.assertEqual(classify_profile_event(draft_event), "draft_proposal")

    def test_v2_commit_plan_handles_target_correction(self) -> None:
        accepted, committed, kind = _v2_commit_plan(
            [10, 11, 12, 13, 14],
            sampled_count=3,
            rejected_count=3,
            last_sampled_token=99,
        )
        self.assertEqual(accepted, 2)
        self.assertEqual(committed, [10, 11, 99])
        self.assertEqual(kind, "target_correction")

    def test_v2_commit_plan_handles_target_bonus(self) -> None:
        accepted, committed, kind = _v2_commit_plan(
            [10, 11, 12, 13, 14],
            sampled_count=6,
            rejected_count=0,
            last_sampled_token=20,
        )
        self.assertEqual(accepted, 5)
        self.assertEqual(committed, [10, 11, 12, 13, 14, 20])
        self.assertEqual(kind, "target_bonus")

    def test_v2_commit_plan_handles_scheduler_shortened_proposal(self) -> None:
        accepted, committed, kind = _v2_commit_plan(
            [10, 11, 12, 13, 14],
            sampled_count=3,
            rejected_count=0,
            last_sampled_token=20,
            drafted_count=2,
        )
        self.assertEqual(accepted, 2)
        self.assertEqual(committed, [10, 11, 20])
        self.assertEqual(kind, "target_bonus")

    def test_v2_hook_observes_but_returns_original_proposal(self) -> None:
        proposal = [[10, 11]]

        class PlainDraftModelSpeculator:
            def propose(self, *args: object, **kwargs: object) -> list[list[int]]:
                return proposal

        speculator = PlainDraftModelSpeculator()
        worker = SD4WorkerExtension()
        worker.rank = 0
        worker.device = "cuda:0"
        worker.model_runner = SimpleNamespace(
            speculator=speculator,
            speculative_config=object(),
        )
        worker.vllm_config = SimpleNamespace(speculative_config=object())
        with patch(
            "sdguard_sd.vllm_extension.load_guard_model",
            return_value=(object(), object()),
        ):
            status = worker.sd4_initialize_guard(
                model="guard",
                dtype="bfloat16",
                local_files_only=True,
                revision=None,
                attention_implementation=None,
            )

        class FakeMonitor:
            def __init__(self) -> None:
                self.prompt_ids = (1, 2)
                self.tracker = SimpleNamespace(committed_ids=[])
                self._pending_proposal_ids = None
                self.proposal = None

            def observe_unproposed_commit(self, **kwargs: object) -> None:
                self.tracker.committed_ids.extend(kwargs["committed_ids"])

            def observe_proposal(self, **kwargs: object) -> None:
                self.proposal = kwargs
                self._pending_proposal_ids = tuple(kwargs["proposed_ids"])

        monitor = FakeMonitor()
        worker._sd4_monitor = monitor
        input_batch = SimpleNamespace(
            num_reqs=1,
            idx_mapping=[0],
            num_draft_tokens=0,
        )
        returned = speculator.propose(
            input_batch,
            None,
            None,
            None,
            None,
            [1],
            [0],
            [7],
            None,
            None,
            None,
        )
        self.assertIs(returned, proposal)
        self.assertEqual(status["hook"], "PlainDraftModelSpeculator.propose")
        self.assertEqual(monitor.tracker.committed_ids, [7])
        self.assertEqual(monitor.proposal["proposed_ids"], [10, 11])
        self.assertEqual(monitor.proposal["prefix_length"], 3)


if __name__ == "__main__":
    unittest.main()
