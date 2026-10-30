"""vLLM 0.31 worker extension for passive Guard-before-Target observation.

The primary V2 adapter wraps ``PlainDraftModelSpeculator.propose``.  A V1
``GPUModelRunner.propose_draft_token_ids`` fallback is retained for 0.31 builds
that select the older runner.  Neither adapter edits site-packages, Draft IDs,
sampling state, or Target acceptance.
"""

from __future__ import annotations

import time
import types
from typing import Any, Sequence

from .guard import run_guard_full_forward
from .inline_guard import IncrementalGuardMonitor
from .runtime import load_guard_model


def _one_row(value: Any, *, limit: int | None = None) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise RuntimeError("SD_5 inline adapter requires vLLM max_num_seqs=1")
        value = list(value[0])
    if not isinstance(value, list):
        raise TypeError(f"Expected token tensor/list, received {type(value).__name__}")
    result = [int(token) for token in value if int(token) >= 0]
    return result if limit is None else result[: int(limit)]


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for first, second in zip(left, right):
        if int(first) != int(second):
            break
        count += 1
    return count


def _scalar(value: Any, index: int | None = None) -> int:
    """Read one integer from a torch/numpy/Python scalar without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if index is not None:
        value = value[int(index)]
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"Expected one scalar value, received {len(value)}")
        value = value[0]
    return int(value)


def _v2_commit_plan(
    pending_ids: Sequence[int],
    *,
    sampled_count: int,
    rejected_count: int,
    last_sampled_token: int,
    drafted_count: int | None = None,
) -> tuple[int, list[int], str]:
    """Convert V2 rejection counters into one verified speculative commit."""

    pending = [int(value) for value in pending_ids]
    sampled = int(sampled_count)
    rejected = int(rejected_count)
    drafted = len(pending) if drafted_count is None else int(drafted_count)
    if sampled <= 0:
        raise RuntimeError("Cannot reconcile a speculative round with no sampled token")
    if drafted < 0 or drafted > len(pending):
        raise RuntimeError(
            f"vLLM verified {drafted} tokens from a {len(pending)}-token proposal"
        )
    if rejected < 0 or rejected > drafted:
        raise RuntimeError(
            f"vLLM reported invalid rejected count {rejected} for {drafted} "
            "verified Draft tokens"
        )
    accepted_from_sampled = sampled - 1
    accepted_from_rejected = drafted - rejected
    if accepted_from_sampled != accepted_from_rejected:
        raise RuntimeError(
            "vLLM V2 speculative counters disagree: "
            f"num_sampled-1={accepted_from_sampled}, "
            f"drafted-num_rejected={accepted_from_rejected}"
        )
    if accepted_from_sampled < 0 or accepted_from_sampled > len(pending):
        raise RuntimeError("vLLM reported an invalid accepted Draft token count")
    committed = pending[:accepted_from_sampled] + [int(last_sampled_token)]
    kind = "target_correction" if rejected else "target_bonus"
    return accepted_from_sampled, committed, kind


def _guard_payload(result: Any) -> dict[str, Any]:
    return {
        "prompt_risk_level": result.prompt_risk_level,
        "prompt_risk_probability": result.prompt_risk_probability,
        "latency_ms": float(result.latency_ms),
        "scopes": result.scopes,
        "token_records": list(result.token_records),
    }


class SD4WorkerExtension:
    """Methods injected into vLLM's GPU worker through ``worker_extension_cls``."""

    def sd4_initialize_guard(
        self,
        *,
        model: str,
        dtype: str,
        local_files_only: bool,
        revision: str | None = None,
        attention_implementation: str | None = None,
    ) -> dict[str, Any]:
        if int(getattr(self, "rank", 0)) != 0:
            raise RuntimeError("SD_5 supports tensor_parallel_size=1 only")
        runner = getattr(self, "model_runner", None)
        if runner is None:
            raise RuntimeError("vLLM 0.31 adapter could not find the GPU model runner")
        if getattr(self, "_sd4_guard_model", None) is not None:
            raise RuntimeError("SD_5 Guard was initialized twice")

        started_ns = time.perf_counter_ns()
        guard_model, guard_tokenizer = load_guard_model(
            model,
            device=str(getattr(self, "device", "cuda:0")),
            dtype=str(dtype),
            local_files_only=bool(local_files_only),
            revision=revision,
            attention_implementation=attention_implementation,
        )
        self._sd4_guard_model = guard_model
        self._sd4_guard_tokenizer = guard_tokenizer
        self._sd4_monitor = None
        self._sd4_sample_id = None
        self._sd4_next_round_index = 0
        self._sd4_original_propose = None
        self._sd4_hook_kind = "none (non-speculative engine)"

        # V2: reconcile the preceding Target result, invoke the untouched Draft
        # method, then observe its return before it reaches Target verification.
        speculator = getattr(runner, "speculator", None)
        if speculator is not None:
            speculator_name = type(speculator).__name__
            if speculator_name != "PlainDraftModelSpeculator":
                raise RuntimeError(
                    "SD_5 requires vLLM PlainDraftModelSpeculator, received "
                    f"{speculator_name}"
                )
            original_v2_propose = speculator.propose
            self._sd4_original_propose = original_v2_propose
            self._sd4_hook_kind = "PlainDraftModelSpeculator.propose"

            def wrapped_v2_propose(
                speculator_self: Any, *args: Any, **kwargs: Any
            ) -> Any:
                extension = self
                monitor = getattr(extension, "_sd4_monitor", None)
                observe_this_proposal = False
                dummy_run = bool(
                    kwargs.get("dummy_run", args[12] if len(args) > 12 else False)
                )
                if monitor is not None and not dummy_run:
                    input_batch = kwargs.get(
                        "input_batch", args[0] if len(args) > 0 else None
                    )
                    num_sampled = kwargs.get(
                        "num_sampled", args[5] if len(args) > 5 else None
                    )
                    num_rejected = kwargs.get(
                        "num_rejected", args[6] if len(args) > 6 else None
                    )
                    last_sampled = kwargs.get(
                        "last_sampled", args[7] if len(args) > 7 else None
                    )
                    if any(
                        value is None
                        for value in (
                            input_batch,
                            num_sampled,
                            num_rejected,
                            last_sampled,
                        )
                    ):
                        raise RuntimeError(
                            "vLLM PlainDraftModelSpeculator.propose signature changed"
                        )
                    if int(input_batch.num_reqs) != 1:
                        raise RuntimeError(
                            "Inline Guard requires one active vLLM request (batch_size=1)"
                        )
                    observe_this_proposal = _scalar(num_sampled, 0) > 0
                    if observe_this_proposal:
                        extension._sd4_reconcile_v2(
                            input_batch=input_batch,
                            num_sampled=num_sampled,
                            num_rejected=num_rejected,
                            last_sampled=last_sampled,
                        )

                draft_ids = original_v2_propose(*args, **kwargs)
                monitor = getattr(extension, "_sd4_monitor", None)
                if monitor is not None and not dummy_run and observe_this_proposal:
                    proposed = _one_row(draft_ids)
                    if proposed:
                        round_index = int(extension._sd4_next_round_index)
                        prefix_length = len(monitor.prompt_ids or ()) + len(
                            monitor.tracker.committed_ids
                        )
                        monitor.observe_proposal(
                            round_index=round_index,
                            prefix_length=prefix_length,
                            proposed_ids=proposed,
                        )
                        extension._sd4_next_round_index = round_index + 1
                return draft_ids

            speculator.propose = types.MethodType(wrapped_v2_propose, speculator)

        # A 0.31 build can explicitly select V1.  Install the fallback only for
        # speculative decoding; direct target/draft engines keep Guard RPC only.
        vllm_config = getattr(self, "vllm_config", None)
        speculative_config = getattr(runner, "speculative_config", None)
        if speculative_config is None and vllm_config is not None:
            speculative_config = getattr(vllm_config, "speculative_config", None)
        v1_propose = getattr(runner, "propose_draft_token_ids", None)
        if speculator is None and speculative_config is not None and callable(v1_propose):
            self._sd4_original_propose = v1_propose
            self._sd4_hook_kind = "GPUModelRunner.propose_draft_token_ids"

            def wrapped_v1_propose(
                runner_self: Any, *args: Any, **kwargs: Any
            ) -> Any:
                extension = self
                monitor = getattr(extension, "_sd4_monitor", None)
                observe_this_proposal = False
                if monitor is not None:
                    if len(runner_self.input_batch.req_ids) != 1:
                        raise RuntimeError(
                            "Inline Guard requires one active vLLM request (batch_size=1)"
                        )
                    sampled = kwargs.get("sampled_token_ids")
                    if sampled is None:
                        if len(args) < 2:
                            raise RuntimeError("vLLM V1 propose signature changed")
                        sampled = args[1]
                    sampled_ids = _one_row(sampled)
                    observe_this_proposal = bool(sampled_ids)
                    extension._sd4_reconcile_sampled(sampled_ids)

                draft_ids = extension._sd4_original_propose(*args, **kwargs)
                monitor = getattr(extension, "_sd4_monitor", None)
                if monitor is not None and observe_this_proposal:
                    scheduler_output = kwargs.get("scheduler_output")
                    if scheduler_output is None:
                        if not args:
                            raise RuntimeError("vLLM V1 propose signature changed")
                        scheduler_output = args[0]
                    limit = int(scheduler_output.num_spec_tokens_to_schedule)
                    proposed = _one_row(draft_ids, limit=limit)
                    if proposed:
                        round_index = int(extension._sd4_next_round_index)
                        prefix_length = len(monitor.prompt_ids or ()) + len(
                            monitor.tracker.committed_ids
                        )
                        monitor.observe_proposal(
                            round_index=round_index,
                            prefix_length=prefix_length,
                            proposed_ids=proposed,
                        )
                        extension._sd4_next_round_index = round_index + 1
                return draft_ids

            runner.propose_draft_token_ids = types.MethodType(wrapped_v1_propose, runner)

        return {
            "guard_load_ms": (time.perf_counter_ns() - started_ns) / 1_000_000.0,
            "device": str(getattr(self, "device", "cuda:0")),
            "hook": self._sd4_hook_kind,
            "passive": True,
        }

    def _sd4_reconcile_v2(
        self,
        *,
        input_batch: Any,
        num_sampled: Any,
        num_rejected: Any,
        last_sampled: Any,
    ) -> None:
        monitor = self._sd4_monitor
        if monitor is None:
            return
        sampled_count = _scalar(num_sampled, 0)
        if sampled_count == 0:
            return
        state_index = _scalar(input_batch.idx_mapping, 0)
        last_token = _scalar(last_sampled, state_index)
        pending = monitor._pending_proposal_ids
        if pending is None:
            if sampled_count != 1:
                raise RuntimeError(
                    "vLLM bootstrap unexpectedly sampled more than one Target token"
                )
            monitor.observe_unproposed_commit(
                round_index=-1,
                committed_ids=[last_token],
                commit_kind="target_bootstrap",
            )
            return
        rejected_count = _scalar(num_rejected, 0)
        drafted_count = int(input_batch.num_draft_tokens)
        accepted, committed, kind = _v2_commit_plan(
            pending,
            sampled_count=sampled_count,
            rejected_count=rejected_count,
            last_sampled_token=last_token,
            drafted_count=drafted_count,
        )
        round_value = types.SimpleNamespace(
            round_index=monitor.tracker._pending["round_index"],
            proposed_ids=tuple(pending),
            accepted_count=accepted,
            committed_ids=tuple(committed),
            commit_kind=kind,
        )
        monitor.observe_commit(round_value)
        # PlainDraftModelSpeculator returns fixed K, while the scheduler can
        # verify a shorter prefix near a length boundary.  Keep both facts:
        # raw proposal diagnostics use all observed IDs and vLLM trace parity
        # uses the effective count that actually reached Target.
        monitor.tracker.round_records[-1][
            "target_verified_proposed_count"
        ] = drafted_count

    def _sd4_reconcile_sampled(self, sampled_ids: Sequence[int]) -> None:
        """Reconcile V1 sampled IDs against the previously guarded proposal."""

        monitor = self._sd4_monitor
        if monitor is None or not sampled_ids:
            return
        pending = monitor._pending_proposal_ids
        if pending is None:
            monitor.observe_unproposed_commit(
                round_index=-1,
                committed_ids=sampled_ids,
                commit_kind="target_bootstrap",
            )
            return
        accepted = _common_prefix(pending, sampled_ids)
        if accepted < len(pending):
            kind = "target_correction"
        elif len(sampled_ids) > accepted:
            kind = "target_bonus"
        else:
            kind = "accepted_draft_terminal"
        round_value = types.SimpleNamespace(
            round_index=monitor.tracker._pending["round_index"],
            proposed_ids=tuple(pending),
            accepted_count=accepted,
            committed_ids=tuple(int(value) for value in sampled_ids),
            commit_kind=kind,
        )
        monitor.observe_commit(round_value)

    def sd4_begin_inline(
        self,
        *,
        sample_id: str,
        prompt_ids: list[int],
        enable_thinking: bool,
    ) -> dict[str, Any]:
        if self._sd4_monitor is not None:
            raise RuntimeError("Previous inline request was not finalized")
        if self._sd4_original_propose is None:
            raise RuntimeError(
                "Inline Guard was requested on an engine without speculative decoding"
            )
        monitor = IncrementalGuardMonitor(
            self._sd4_guard_model,
            self._sd4_guard_tokenizer,
            guard_device=str(getattr(self, "device", "cuda:0")),
            enable_thinking=bool(enable_thinking),
        )
        monitor.start(prompt_ids)
        self._sd4_monitor = monitor
        self._sd4_sample_id = str(sample_id)
        self._sd4_next_round_index = 0
        return {
            "sample_id": self._sd4_sample_id,
            "prefill_latency_ms": monitor.prefill_latency_ms,
        }

    def sd4_finish_inline(
        self,
        *,
        sample_id: str,
        generated_ids: list[int],
        answer_start_generated_index: int | None,
    ) -> dict[str, Any]:
        if str(sample_id) != self._sd4_sample_id or self._sd4_monitor is None:
            raise RuntimeError("Inline finalization does not match the active sample")
        monitor = self._sd4_monitor
        expected = [int(value) for value in generated_ids]
        committed = monitor.tracker.committed_ids
        if expected[: len(committed)] != committed:
            raise RuntimeError("vLLM output diverged from the observed committed stream")
        remaining = expected[len(committed) :]
        if remaining:
            self._sd4_reconcile_sampled(remaining)
        if monitor._pending_proposal_ids is not None:
            monitor.discard_pending(reason="unverified_after_stop")
        result = monitor.finalize(
            generated_ids=expected,
            answer_start_generated_index=answer_start_generated_index,
        )
        result["token_records"] = list(monitor.tracker.token_records)
        result["round_records"] = list(monitor.tracker.round_records)
        result["adapter"] = {
            "hook": self._sd4_hook_kind,
            "placement": "after Draft proposal and before next Target verification",
            "passive": True,
            "sample_id": str(sample_id),
        }
        self._sd4_monitor = None
        self._sd4_sample_id = None
        return result

    def sd4_run_posthoc_guard(
        self,
        *,
        prompt_ids: list[int],
        generated_ids: list[int],
        answer_start_generated_index: int | None,
    ) -> dict[str, Any]:
        result = run_guard_full_forward(
            self._sd4_guard_model,
            self._sd4_guard_tokenizer,
            prompt_ids,
            generated_ids,
            answer_start_generated_index=answer_start_generated_index,
            guard_device=str(getattr(self, "device", "cuda:0")),
        )
        return _guard_payload(result)

    def sd4_adapter_status(self) -> dict[str, Any]:
        return {
            "guard_loaded": getattr(self, "_sd4_guard_model", None) is not None,
            "active_sample_id": getattr(self, "_sd4_sample_id", None),
            "hook_installed": getattr(self, "_sd4_original_propose", None) is not None,
            "hook": getattr(self, "_sd4_hook_kind", None),
            "device": str(getattr(self, "device", "unknown")),
        }


__all__ = ["SD4WorkerExtension"]
