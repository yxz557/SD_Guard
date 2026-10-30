from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from sdguard_sd.inline_guard import (  # noqa: E402
    IncrementalGuardMonitor,
    InlineDecisionTracker,
)
from sdguard_sd.runtime import (  # noqa: E402
    _validate_guard_outputs,
    crop_cache_to_length,
    ensure_guard_config,
    legacy_masking_api_adapter,
)
from sdguard_sd.seeding import stable_sample_seed  # noqa: E402
from sdguard_sd.timing import summarize_ms  # noqa: E402
from sdguard_sd.vllm_extension import (  # noqa: E402
    SD4WorkerExtension,
    _guard_payload,
    _one_row_timed,
    _summarize_stage_values,
    _v2_commit_plan,
)
from sdguard_sd.vllm_worker import (  # noqa: E402
    _paired_sd_equivalence,
    _spec_metrics,
    _validate_inline_acceptance_trace,
)
from run_matrix import _validate  # noqa: E402


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
    @staticmethod
    def _validation_args(**overrides: object) -> SimpleNamespace:
        values: dict[str, object] = {
            "sample_count": 1,
            "max_new_tokens": 32,
            "max_model_len": 128,
            "num_speculative_tokens": 3,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "gpu_memory_utilization": 0.8,
            "warmup_samples": 1,
            "stage_timing_only": False,
            "stage_timing_max_new_tokens": 32,
            "profile_vllm": None,
            "profile_max_new_tokens": None,
            "gpu_id": None,
            "engine_gpu_id": "0",
            "guard_gpu_id": "1",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_dual_gpu_validation_requires_distinct_devices(self) -> None:
        _validate(self._validation_args())
        with self.assertRaisesRegex(ValueError, "two different physical GPUs"):
            _validate(self._validation_args(guard_gpu_id="0"))

    def test_legacy_profile_flag_selects_trace_free_stage_timing(self) -> None:
        args = self._validation_args(profile_vllm=True)
        _validate(args)
        self.assertTrue(args.stage_timing_only)
        self.assertEqual(args.stage_timing_max_new_tokens, 32)

    def test_stage_timing_pair_equivalence_checks_output_and_acceptance(self) -> None:
        standard = {
            "generated_ids": [1, 2],
            "speculative_decoding": {
                "per_step_accepted": [2],
                "per_step_drafted": [5],
            },
        }
        inline = {
            "generated_ids": [1, 2],
            "speculative_decoding": {
                "per_step_accepted": [2],
                "per_step_drafted": [5],
            },
        }
        self.assertEqual(
            _paired_sd_equivalence(standard, inline),
            {"generated_ids_match": True, "acceptance_trace_match": True},
        )
        inline["speculative_decoding"]["per_step_accepted"] = [1]
        self.assertEqual(
            _paired_sd_equivalence(standard, inline),
            {"generated_ids_match": True, "acceptance_trace_match": False},
        )

    def test_guard_config_gets_transformers5_missing_pad_token_id(self) -> None:
        config = SimpleNamespace(
            bos_token_id=1,
            num_hidden_layers=2,
            use_sliding_window=False,
            sliding_window=None,
        )
        tokenizer = SimpleNamespace(pad_token_id=151643, eos_token_id=151645)
        returned = ensure_guard_config(config, tokenizer)
        self.assertIs(returned, config)
        self.assertEqual(returned.pad_token_id, 151643)
        self.assertEqual(returned.layer_types, ["full_attention", "full_attention"])

    def test_guard_config_preserves_checkpoint_pad_token_id(self) -> None:
        config = SimpleNamespace(
            pad_token_id=7,
            num_hidden_layers=1,
            layer_types=["full_attention"],
        )
        tokenizer = SimpleNamespace(pad_token_id=151643, eos_token_id=151645)
        returned = ensure_guard_config(config, tokenizer)
        self.assertEqual(returned.pad_token_id, 7)

    def test_guard_config_rejects_ambiguous_sliding_layers(self) -> None:
        config = SimpleNamespace(
            num_hidden_layers=2,
            use_sliding_window=True,
            sliding_window=1024,
        )
        tokenizer = SimpleNamespace(pad_token_id=1, eos_token_id=2)
        with self.assertRaisesRegex(RuntimeError, "missing layer_types"):
            ensure_guard_config(config, tokenizer)

    def test_transformers5_mask_arguments_are_translated(self) -> None:
        calls: list[dict[str, object]] = []

        def modern_mask_factory(**kwargs: object) -> object:
            calls.append(kwargs)
            return kwargs["inputs_embeds"]

        marker = object()
        wrapped = legacy_masking_api_adapter(modern_mask_factory)
        self.assertIs(
            wrapped(input_embeds=marker, cache_position=object()), marker
        )
        self.assertIs(calls[0]["inputs_embeds"], marker)
        self.assertNotIn("input_embeds", calls[0])
        self.assertNotIn("cache_position", calls[0])

    def test_guard_cache_rollback_uses_negative_remove_count(self) -> None:
        class FakeCache:
            def __init__(self) -> None:
                self.length = 8
                self.arguments: list[int] = []

            def get_seq_length(self) -> int:
                return self.length

            def crop(self, value: int) -> None:
                self.arguments.append(value)
                self.length -= abs(value)

        cache = FakeCache()
        crop_cache_to_length(cache, 5)
        self.assertEqual(cache.arguments, [-3])
        self.assertEqual(cache.length, 5)

    def test_guard_output_contract_is_checked_before_experiment(self) -> None:
        class FakeTensor:
            def __init__(self, shape: tuple[int, ...]) -> None:
                self.shape = shape

        cache = SimpleNamespace(get_seq_length=lambda: 2, crop=lambda _: None)
        outputs = SimpleNamespace(
            risk_level_logits=FakeTensor((1, 2, 3)),
            category_logits=FakeTensor((1, 2, 8)),
            query_risk_level_logits=FakeTensor((1, 2, 3)),
            query_category_logits=FakeTensor((1, 2, 9)),
            past_key_values=cache,
        )
        config = SimpleNamespace(
            num_risk_level=3,
            num_category=8,
            num_query_risk_level=3,
            num_query_category=9,
        )
        shapes = _validate_guard_outputs(outputs, token_count=2, config=config)
        self.assertEqual(shapes["risk_level_logits"], [1, 2, 3])

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

    def test_lightweight_stage_summary_needs_no_trace(self) -> None:
        summary = _summarize_stage_values([1.0, 2.0, 3.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["total_ms"], 6.0)
        self.assertEqual(summary["per_call_ms"], [1.0, 2.0, 3.0])
        self.assertEqual(_summarize_stage_values([])["count"], 0)

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

    def test_terminal_gpu_commit_is_clipped_to_scheduler_output(self) -> None:
        class FakeCache:
            def __init__(self, length: int) -> None:
                self.length = length

            def get_seq_length(self) -> int:
                return self.length

            def crop(self, amount: int) -> None:
                self.length += amount

        tracker = InlineDecisionTracker(enable_thinking=False, close_think_token_id=99)
        proposal_records = [
            guard_record("Safe", 10),
            guard_record("Safe", 11),
            guard_record("Unsafe", 12),
        ]
        tracker.observe_proposal(
            round_index=0,
            prefix_generated_count=0,
            token_ids=[10, 11, 12],
            token_records=proposal_records,
        )
        snapshot = tracker.capture_commit_state()
        commit_records = proposal_records + [guard_record("Unsafe", 20)]
        tracker.observe_commit(
            round_index=0,
            accepted_count=3,
            committed_ids=[10, 11, 12, 20],
            commit_kind="target_bonus",
            token_records=commit_records,
        )
        # vLLM can already have produced another proposal before its scheduler
        # applies EOS to the public output.
        tracker.observe_proposal(
            round_index=1,
            prefix_generated_count=4,
            token_ids=[30, 31],
            token_records=[guard_record("Safe", 30), guard_record("Safe", 31)],
        )

        monitor = IncrementalGuardMonitor.__new__(IncrementalGuardMonitor)
        monitor.prompt_ids = (1,)
        monitor.cache = FakeCache(7)
        monitor.tracker = tracker
        monitor._pending_cache_length = 5
        monitor._pending_proposal_ids = (30, 31)
        monitor._pending_proposal_records = [
            guard_record("Safe", 30),
            guard_record("Safe", 31),
        ]
        transaction = {
            "tracker_snapshot": snapshot,
            "round_index": 0,
            "proposed_ids": (10, 11, 12),
            "accepted_count": 3,
            "committed_ids": (10, 11, 12, 20),
            "commit_kind": "target_bonus",
            "token_records": commit_records,
            "target_verified_proposed_count": 3,
            "target_verified_accepted_count": 3,
        }
        monitor._commit_transactions = [transaction]
        monitor._last_commit_transaction = transaction
        result = monitor.clip_commits_to_output(
            [10, 11],
            target_verified_accepted_count=3,
            target_verified_proposed_count=3,
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["clipped_token_count"], 2)
        self.assertEqual(monitor.cache.get_seq_length(), 3)
        self.assertEqual(monitor.tracker.committed_ids, [10, 11])
        self.assertIsNone(monitor._pending_proposal_ids)
        round_record = monitor.tracker.round_records[-1]
        self.assertEqual(round_record["accepted_count"], 2)
        self.assertEqual(round_record["target_verified_accepted_count"], 3)
        self.assertEqual(round_record["target_verified_proposed_count"], 3)

    def test_terminal_output_can_cross_multiple_internal_commits(self) -> None:
        class FakeCache:
            def __init__(self, length: int) -> None:
                self.length = length

            def get_seq_length(self) -> int:
                return self.length

            def crop(self, amount: int) -> None:
                self.length += amount

        tracker = InlineDecisionTracker(enable_thinking=False, close_think_token_id=99)

        first_proposal_records = [
            guard_record("Safe", 10),
            guard_record("Safe", 11),
            guard_record("Unsafe", 12),
        ]
        tracker.observe_proposal(
            round_index=0,
            prefix_generated_count=0,
            token_ids=[10, 11, 12],
            token_records=first_proposal_records,
        )
        first_snapshot = tracker.capture_commit_state()
        first_commit_records = first_proposal_records + [guard_record("Unsafe", 20)]
        tracker.observe_commit(
            round_index=0,
            accepted_count=3,
            committed_ids=[10, 11, 12, 20],
            commit_kind="target_bonus",
            token_records=first_commit_records,
        )
        first_transaction = {
            "tracker_snapshot": first_snapshot,
            "round_index": 0,
            "proposed_ids": (10, 11, 12),
            "accepted_count": 3,
            "committed_ids": (10, 11, 12, 20),
            "commit_kind": "target_bonus",
            "token_records": first_commit_records,
            "target_verified_proposed_count": 3,
            "target_verified_accepted_count": 3,
        }

        second_proposal_records = [
            guard_record("Safe", 30),
            guard_record("Safe", 31),
        ]
        tracker.observe_proposal(
            round_index=1,
            prefix_generated_count=4,
            token_ids=[30, 31],
            token_records=second_proposal_records,
        )
        second_snapshot = tracker.capture_commit_state()
        second_commit_records = [
            guard_record("Safe", 30),
            guard_record("Unsafe", 40),
        ]
        tracker.observe_commit(
            round_index=1,
            accepted_count=1,
            committed_ids=[30, 40],
            commit_kind="target_correction",
            token_records=second_commit_records,
        )
        second_transaction = {
            "tracker_snapshot": second_snapshot,
            "round_index": 1,
            "proposed_ids": (30, 31),
            "accepted_count": 1,
            "committed_ids": (30, 40),
            "commit_kind": "target_correction",
            "token_records": second_commit_records,
            "target_verified_proposed_count": 2,
            "target_verified_accepted_count": 1,
        }

        tracker.observe_proposal(
            round_index=2,
            prefix_generated_count=6,
            token_ids=[50],
            token_records=[guard_record("Safe", 50)],
        )
        monitor = IncrementalGuardMonitor.__new__(IncrementalGuardMonitor)
        monitor.prompt_ids = (1,)
        monitor.cache = FakeCache(8)
        monitor.tracker = tracker
        monitor._pending_cache_length = 7
        monitor._pending_proposal_ids = (50,)
        monitor._pending_proposal_records = [guard_record("Safe", 50)]
        monitor._commit_transactions = [first_transaction, second_transaction]
        monitor._last_commit_transaction = second_transaction

        result = monitor.clip_commits_to_output([10, 11])
        self.assertTrue(result["applied"])
        self.assertEqual(result["clipped_token_count"], 4)
        self.assertEqual(result["discarded_verified_round_count"], 1)
        self.assertEqual(monitor.cache.get_seq_length(), 3)
        self.assertEqual(monitor.tracker.committed_ids, [10, 11])
        self.assertEqual(len(monitor.tracker.round_records), 2)
        self.assertEqual(
            monitor.tracker.round_records[1]["commit_kind"],
            "terminal_output_clipped",
        )
        self.assertEqual(
            monitor.tracker.round_records[1]["target_verified_accepted_count"], 1
        )
        finalized = monitor.tracker.finalize(answer_start_generated_index=0)
        self.assertEqual(
            finalized["verified_commit"]["scopes"]["full_trace"]["token_count"], 2
        )

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

    def test_acceptance_trace_excludes_only_post_stop_verified_suffix(self) -> None:
        # Regression for a 20th GPU verify step after vLLM finalized a
        # 19-step request.  The first 19 steps must still match exactly.
        accepted = [5, 5, 5, 1, 5, 2, 5, 2, 5, 4, 0, 0, 1, 1, 1, 0, 0, 0, 5]
        rows = [
            {
                "round_index": index,
                "commit_kind": "target_correction",
                "accepted_count": count,
                "proposed_ids": [0] * 5,
                "target_verified_accepted_count": count,
                "target_verified_proposed_count": 5,
            }
            for index, count in enumerate(accepted)
        ]
        rows.append(
            {
                "round_index": 19,
                "commit_kind": "terminal_output_clipped",
                "accepted_count": 0,
                "proposed_ids": [0] * 5,
                "target_verified_accepted_count": 4,
                "target_verified_proposed_count": 5,
            }
        )
        metrics = {
            "per_step_accepted": accepted,
            "per_step_drafted": [5] * len(accepted),
        }
        result = _validate_inline_acceptance_trace(rows, metrics)
        self.assertEqual(result["scheduler_metric_round_count"], 19)
        self.assertEqual(result["observed_verified_round_count"], 20)
        self.assertEqual(result["excluded_post_stop_verified_round_count"], 1)

        # Genuine parity errors or unmarked extra rounds must never be hidden.
        bad_metrics = dict(metrics, per_step_accepted=[4] + accepted[1:])
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _validate_inline_acceptance_trace(rows, bad_metrics)
        rows[-1]["commit_kind"] = "target_correction"
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _validate_inline_acceptance_trace(rows, metrics)

    def test_acceptance_trace_rejects_missing_or_early_clipped_round(self) -> None:
        metrics = {"per_step_accepted": [1], "per_step_drafted": [5]}
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _validate_inline_acceptance_trace([], metrics)
        clipped = {
            "round_index": 0,
            "commit_kind": "terminal_output_clipped",
            "accepted_count": 0,
            "proposed_ids": [0] * 5,
            "target_verified_accepted_count": 1,
            "target_verified_proposed_count": 5,
        }
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            _validate_inline_acceptance_trace([clipped], metrics)
        # The terminal verify step itself is still in scheduler metrics even
        # if the scheduler trims some tokens from that step's output.
        terminal_step = dict(
            clipped,
            commit_kind="target_correction",
            terminal_output_clipped=True,
        )
        result = _validate_inline_acceptance_trace([terminal_step], metrics)
        self.assertEqual(result["excluded_post_stop_verified_round_count"], 0)

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
            execute_model=lambda *args, **kwargs: None,
            sample=lambda *args, **kwargs: None,
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
                device="cuda:1",
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
        self.assertEqual(status["engine_device"], "cuda:0")
        self.assertEqual(status["guard_device"], "cuda:1")
        self.assertEqual(
            status.get("hook"), "PlainDraftModelSpeculator.propose"
        )
        self.assertEqual(monitor.tracker.committed_ids, [7])
        self.assertEqual(monitor.proposal["proposed_ids"], [10, 11])
        self.assertEqual(monitor.proposal["prefix_length"], 3)

    def test_python_proposal_has_no_gpu_transfer_charge(self) -> None:
        values, elapsed_ms = _one_row_timed([[10, 11, -1]], limit=2)
        self.assertEqual(values, [10, 11])
        self.assertEqual(elapsed_ms, 0.0)

    def test_guard_payload_exposes_transfer_breakdown(self) -> None:
        payload = _guard_payload(
            SimpleNamespace(
                prompt_risk_level="Safe",
                prompt_risk_probability=0.9,
                latency_ms=3.0,
                input_packaging_ms=0.1,
                input_h2d_ms=0.2,
                input_preparation_ms=0.3,
                output_decode_d2h_ms=0.4,
                scopes={},
                token_records=(),
            )
        )
        self.assertEqual(payload["input_h2d_ms"], 0.2)
        self.assertEqual(payload["output_decode_d2h_ms"], 0.4)


if __name__ == "__main__":
    unittest.main()
