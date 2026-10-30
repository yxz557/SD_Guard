"""vLLM 0.31 worker extension for passive Guard-before-Target observation.

The primary V2 adapter wraps ``PlainDraftModelSpeculator.propose``.  A V1
``GPUModelRunner.propose_draft_token_ids`` fallback is retained for 0.31 builds
that select the older runner.  Neither adapter edits site-packages, Draft IDs,
sampling state, or Target acceptance.
"""

from __future__ import annotations

import statistics
import time
import types
from typing import Any, Sequence

from .guard import run_guard_full_forward
from .inline_guard import IncrementalGuardMonitor
from .runtime import import_runtime, load_guard_model


STAGE_TIMING_NAMES = (
    "target_forward",
    "target_sampling",
    "rejection_sampling",
    "draft_proposal",
)


def _summarize_stage_values(values: Sequence[float]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if not normalized:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "min_ms": None,
            "max_ms": None,
            "per_call_ms": [],
        }
    ordered = sorted(normalized)
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return {
        "count": len(normalized),
        "total_ms": sum(normalized),
        "mean_ms": statistics.fmean(normalized),
        "median_ms": statistics.median(normalized),
        "p90_ms": ordered[p90_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "per_call_ms": normalized,
    }


def _timed_engine_call(extension: Any, stage: str, operation: Any) -> Any:
    """Measure one pinned-vLLM stage without collecting a profiler trace.

    The synchronizations intentionally perturb scheduling. This helper is only
    active during a separate short diagnostic request and is never enabled for
    the formal latency matrix.
    """

    session = getattr(extension, "_sd4_stage_timing_session", None)
    if session is None:
        return operation()
    torch, _, _, _ = import_runtime()
    device = str(getattr(extension, "device", "cuda:0"))
    torch.cuda.synchronize(device)
    started_ns = time.perf_counter_ns()
    try:
        return operation()
    finally:
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        session["stages"].setdefault(str(stage), []).append(elapsed_ms)


def _input_batch_has_drafts(input_batch: Any) -> bool:
    if input_batch is None:
        return False
    value = getattr(input_batch, "num_draft_tokens", 0)
    if hasattr(value, "item"):
        value = value.item()
    return int(value) > 0


def _one_row(value: Any, *, limit: int | None = None) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise RuntimeError("SD_4 inline adapter requires vLLM max_num_seqs=1")
        value = list(value[0])
    if not isinstance(value, list):
        raise TypeError(f"Expected token tensor/list, received {type(value).__name__}")
    result = [int(token) for token in value if int(token) >= 0]
    return result if limit is None else result[: int(limit)]


def _synchronize_value_device(value: Any) -> None:
    device = getattr(value, "device", None)
    if device is None or getattr(device, "type", str(device).split(":", 1)[0]) != "cuda":
        return
    torch, _, _, _ = import_runtime()
    torch.cuda.synchronize(device)


def _one_row_timed(value: Any, *, limit: int | None = None) -> tuple[list[int], float]:
    if not hasattr(value, "detach"):
        return _one_row(value, limit=limit), 0.0
    _synchronize_value_device(value)
    started_ns = time.perf_counter_ns()
    result = _one_row(value, limit=limit)
    return result, (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _add_monitor_timing(monitor: Any, name: str, value: float) -> None:
    setattr(monitor, name, float(getattr(monitor, name, 0.0)) + float(value))


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
        "input_packaging_ms": float(result.input_packaging_ms),
        "input_h2d_ms": float(result.input_h2d_ms),
        "input_preparation_ms": float(result.input_preparation_ms),
        "output_decode_d2h_ms": float(result.output_decode_d2h_ms),
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
        device: str = "cuda:1",
        revision: str | None = None,
        attention_implementation: str | None = None,
        enable_stage_timing_hooks: bool = False,
    ) -> dict[str, Any]:
        if int(getattr(self, "rank", 0)) != 0:
            raise RuntimeError("SD_4 supports tensor_parallel_size=1 only")
        runner = getattr(self, "model_runner", None)
        if runner is None:
            raise RuntimeError("vLLM 0.31 adapter could not find the GPU model runner")
        if getattr(self, "_sd4_guard_model", None) is not None:
            raise RuntimeError("SD_4 Guard was initialized twice")

        engine_device = str(getattr(self, "device", "cuda:0"))
        guard_device = str(device)
        if engine_device == guard_device:
            raise RuntimeError(
                "SD_4 dual-GPU mode requires Guard on a different logical device "
                f"from the vLLM engine; both resolved to {engine_device}"
            )

        started_ns = time.perf_counter_ns()
        guard_model, guard_tokenizer = load_guard_model(
            model,
            device=guard_device,
            dtype=str(dtype),
            local_files_only=bool(local_files_only),
            revision=revision,
            attention_implementation=attention_implementation,
        )
        self._sd4_guard_model = guard_model
        self._sd4_guard_tokenizer = guard_tokenizer
        self._sd4_guard_device = guard_device
        self._sd4_monitor = None
        self._sd4_sample_id = None
        self._sd4_next_round_index = 0
        self._sd4_original_propose = None
        self._sd4_original_execute_model = None
        self._sd4_original_sample = None
        self._sd4_stage_timing_session = None
        self._sd4_stage_timing_hooks: list[str] = []
        self._sd4_hook_kind = "none (non-speculative engine)"

        # V2: reconcile the preceding Target result, invoke the untouched Draft
        # method, then observe its return before it reaches Target verification.
        speculator = getattr(runner, "speculator", None)
        if speculator is not None:
            speculator_name = type(speculator).__name__
            if speculator_name != "PlainDraftModelSpeculator":
                raise RuntimeError(
                    "SD_4 requires vLLM PlainDraftModelSpeculator, received "
                    f"{speculator_name}"
                )
            original_v2_propose = speculator.propose
            self._sd4_original_propose = original_v2_propose
            self._sd4_hook_kind = "PlainDraftModelSpeculator.propose"

            if enable_stage_timing_hooks:
                original_execute_model = getattr(runner, "execute_model", None)
                original_sample = getattr(runner, "sample", None)
                if not callable(original_execute_model) or not callable(original_sample):
                    raise RuntimeError(
                        "SD_4 lightweight timing requires vLLM 0.31 V2 "
                        "GPUModelRunner.execute_model() and sample()"
                    )
                self._sd4_original_execute_model = original_execute_model
                self._sd4_original_sample = original_sample

                def wrapped_execute_model(
                    runner_self: Any, *args: Any, **kwargs: Any
                ) -> Any:
                    del runner_self
                    return _timed_engine_call(
                        self,
                        "target_forward",
                        lambda: original_execute_model(*args, **kwargs),
                    )

                def wrapped_sample(
                    runner_self: Any, *args: Any, **kwargs: Any
                ) -> Any:
                    del runner_self
                    input_batch = kwargs.get(
                        "input_batch", args[1] if len(args) > 1 else None
                    )
                    stage = (
                        "rejection_sampling"
                        if _input_batch_has_drafts(input_batch)
                        else "target_sampling"
                    )
                    return _timed_engine_call(
                        self,
                        stage,
                        lambda: original_sample(*args, **kwargs),
                    )

                runner.execute_model = types.MethodType(wrapped_execute_model, runner)
                runner.sample = types.MethodType(wrapped_sample, runner)
                self._sd4_stage_timing_hooks.extend(
                    ["GPUModelRunner.execute_model", "GPUModelRunner.sample"]
                )

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

                draft_ids = _timed_engine_call(
                    extension,
                    "draft_proposal",
                    lambda: original_v2_propose(*args, **kwargs),
                )
                monitor = getattr(extension, "_sd4_monitor", None)
                if monitor is not None and not dummy_run and observe_this_proposal:
                    proposed, transfer_ms = _one_row_timed(draft_ids)
                    if proposed:
                        _add_monitor_timing(
                            monitor, "proposal_output_d2h_ms", transfer_ms
                        )
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
            if enable_stage_timing_hooks:
                self._sd4_stage_timing_hooks.append(
                    "PlainDraftModelSpeculator.propose"
                )

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
                    sampled_ids, transfer_ms = _one_row_timed(sampled)
                    _add_monitor_timing(
                        monitor, "commit_output_d2h_ms", transfer_ms
                    )
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
                    proposed, transfer_ms = _one_row_timed(draft_ids, limit=limit)
                    if proposed:
                        _add_monitor_timing(
                            monitor, "proposal_output_d2h_ms", transfer_ms
                        )
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
            "engine_device": engine_device,
            "guard_device": guard_device,
            "hook": self._sd4_hook_kind,
            "passive": True,
            "compatibility": getattr(
                guard_model, "_sd4_compatibility_report", {"status": "unknown"}
            ),
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
        for value in (num_sampled, num_rejected, last_sampled):
            _synchronize_value_device(value)
        transfer_started_ns = time.perf_counter_ns()
        sampled_count = _scalar(num_sampled, 0)
        if sampled_count == 0:
            return
        state_index = _scalar(input_batch.idx_mapping, 0)
        last_token = _scalar(last_sampled, state_index)
        rejected_count = _scalar(num_rejected, 0)
        _add_monitor_timing(
            monitor,
            "commit_output_d2h_ms",
            (time.perf_counter_ns() - transfer_started_ns) / 1_000_000.0,
        )
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
        monitor.set_last_commit_verification_counts(
            accepted_count=accepted,
            proposed_count=drafted_count,
        )

    def _sd4_reconcile_sampled(
        self,
        sampled_ids: Sequence[int],
        *,
        target_verified_accepted_count: int | None = None,
        target_verified_proposed_count: int | None = None,
    ) -> None:
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
        monitor.set_last_commit_verification_counts(
            accepted_count=(
                accepted
                if target_verified_accepted_count is None
                else int(target_verified_accepted_count)
            ),
            proposed_count=(
                len(pending)
                if target_verified_proposed_count is None
                else int(target_verified_proposed_count)
            ),
        )

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
            guard_device=str(self._sd4_guard_device),
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

    def sd4_begin_stage_timing(self, *, label: str) -> dict[str, Any]:
        """Start a short synchronized timing diagnostic with no trace output."""

        if self._sd4_stage_timing_session is not None:
            raise RuntimeError("A lightweight stage-timing session is already active")
        if self._sd4_hook_kind != "PlainDraftModelSpeculator.propose":
            raise RuntimeError(
                "Lightweight stage timing requires the pinned vLLM 0.31 V2 adapter"
            )
        required = {
            "GPUModelRunner.execute_model",
            "GPUModelRunner.sample",
            "PlainDraftModelSpeculator.propose",
        }
        installed = set(self._sd4_stage_timing_hooks)
        if not required.issubset(installed):
            raise RuntimeError(
                "Lightweight stage-timing hooks are incomplete: "
                f"required={sorted(required)}, installed={sorted(installed)}"
            )
        self._sd4_stage_timing_session = {
            "label": str(label),
            "stages": {name: [] for name in STAGE_TIMING_NAMES},
        }
        return {
            "label": str(label),
            "started": True,
            "full_trace_collected": False,
            "hooks": list(self._sd4_stage_timing_hooks),
        }

    def sd4_finish_stage_timing(self, *, label: str) -> dict[str, Any]:
        session = self._sd4_stage_timing_session
        if session is None or str(session["label"]) != str(label):
            raise RuntimeError("Stage-timing finalization does not match its session")
        torch, _, _, _ = import_runtime()
        torch.cuda.synchronize(str(getattr(self, "device", "cuda:0")))
        self._sd4_stage_timing_session = None
        stages = {
            name: _summarize_stage_values(session["stages"].get(name, ()))
            for name in STAGE_TIMING_NAMES
        }
        return {
            "label": str(label),
            "measurement": "synchronized wall time around pinned vLLM call boundaries",
            "diagnostic_only": True,
            "formal_latency_included": False,
            "synchronization_added": True,
            "full_trace_collected": False,
            "trace_files": [],
            "hooks": list(self._sd4_stage_timing_hooks),
            "stages": stages,
            "total_measured_stage_ms": sum(
                float(stage["total_ms"]) for stage in stages.values()
            ),
            "note": (
                "Synchronization makes stage boundaries observable but perturbs "
                "normal overlap; use the uninstrumented formal matrix for end-to-end "
                "latency."
            ),
        }

    def sd4_finish_inline(
        self,
        *,
        sample_id: str,
        generated_ids: list[int],
        answer_start_generated_index: int | None,
        per_step_accepted: list[int] | None = None,
        per_step_drafted: list[int] | None = None,
    ) -> dict[str, Any]:
        if str(sample_id) != self._sd4_sample_id or self._sd4_monitor is None:
            raise RuntimeError("Inline finalization does not match the active sample")
        monitor = self._sd4_monitor
        expected = [int(value) for value in generated_ids]
        committed = monitor.tracker.committed_ids
        terminal_reconciliation = {"applied": False, "clipped_token_count": 0}
        if expected[: len(committed)] != committed:
            if len(expected) < len(committed) and committed[: len(expected)] == expected:
                accepted_metrics = [int(value) for value in (per_step_accepted or [])]
                drafted_metrics = [int(value) for value in (per_step_drafted or [])]
                committed_round_count = sum(
                    1
                    for row in monitor.tracker.round_records
                    if int(row["round_index"]) >= 0
                    and row.get("commit_kind") != "unverified_after_stop"
                )
                metric_index = committed_round_count - 1
                metric_accepted = (
                    accepted_metrics[metric_index]
                    if 0 <= metric_index < len(accepted_metrics)
                    else None
                )
                metric_drafted = (
                    drafted_metrics[metric_index]
                    if 0 <= metric_index < len(drafted_metrics)
                    else None
                )
                terminal_reconciliation = monitor.clip_commits_to_output(
                    expected,
                    target_verified_accepted_count=metric_accepted,
                    target_verified_proposed_count=metric_drafted,
                )
                committed = monitor.tracker.committed_ids
            else:
                common = _common_prefix(expected, committed)
                left = max(0, common - 3)
                right = common + 4
                raise RuntimeError(
                    "vLLM output truly diverged from the observed committed stream: "
                    f"common_prefix={common}, output_len={len(expected)}, "
                    f"observed_len={len(committed)}, "
                    f"output_window={expected[left:right]}, "
                    f"observed_window={committed[left:right]}"
                )
        remaining = expected[len(committed) :]
        if remaining:
            accepted_metrics = [int(value) for value in (per_step_accepted or [])]
            drafted_metrics = [int(value) for value in (per_step_drafted or [])]
            metric_index = sum(
                1
                for row in monitor.tracker.round_records
                if int(row["round_index"]) >= 0
                and row.get("commit_kind") != "unverified_after_stop"
            )
            self._sd4_reconcile_sampled(
                remaining,
                target_verified_accepted_count=(
                    accepted_metrics[metric_index]
                    if metric_index < len(accepted_metrics)
                    else None
                ),
                target_verified_proposed_count=(
                    drafted_metrics[metric_index]
                    if metric_index < len(drafted_metrics)
                    else None
                ),
            )
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
            "terminal_reconciliation": terminal_reconciliation,
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
            guard_device=str(self._sd4_guard_device),
        )
        return _guard_payload(result)

    def sd4_measure_device_transfer(
        self,
        *,
        token_counts: list[int],
        repetitions: int = 100,
        warmup: int = 10,
    ) -> dict[str, Any]:
        """Diagnostic direct GPU-to-GPU copy latency; excluded from formal timing."""

        from .runtime import import_runtime, synchronize_devices

        torch, _, _, _ = import_runtime()
        source = str(getattr(self, "device", "cuda:0"))
        destination = str(self._sd4_guard_device)
        result: dict[str, Any] = {
            "source": source,
            "destination": destination,
            "path": "torch Tensor.to direct device-to-device diagnostic",
            "formal_latency_included": False,
            "peer_access": bool(torch.cuda.can_device_access_peer(0, 1)),
            "measurements": {},
        }
        for raw_count in token_counts:
            count = max(1, int(raw_count))
            try:
                source_tensor = torch.arange(count, dtype=torch.int64, device=source)
                for _ in range(max(0, int(warmup))):
                    copied = source_tensor.to(destination, non_blocking=False)
                    synchronize_devices(source, destination)
                    del copied
                values: list[float] = []
                for _ in range(max(1, int(repetitions))):
                    synchronize_devices(source, destination)
                    started_ns = time.perf_counter_ns()
                    copied = source_tensor.to(destination, non_blocking=False)
                    synchronize_devices(source, destination)
                    values.append(
                        (time.perf_counter_ns() - started_ns) / 1_000_000.0
                    )
                    del copied
                ordered = sorted(values)
                result["measurements"][str(count)] = {
                    "token_count": count,
                    "bytes": count * 8,
                    "count": len(values),
                    "mean_ms": statistics.fmean(values),
                    "median_ms": statistics.median(values),
                    "p90_ms": ordered[
                        min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
                    ],
                    "min_ms": min(values),
                    "max_ms": max(values),
                }
            except Exception as exc:  # diagnostic failure must not abort the matrix
                result["measurements"][str(count)] = {
                    "token_count": count,
                    "bytes": count * 8,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return result

    def sd4_adapter_status(self) -> dict[str, Any]:
        return {
            "guard_loaded": getattr(self, "_sd4_guard_model", None) is not None,
            "active_sample_id": getattr(self, "_sd4_sample_id", None),
            "hook_installed": getattr(self, "_sd4_original_propose", None) is not None,
            "hook": getattr(self, "_sd4_hook_kind", None),
            "stage_timing_active": getattr(
                self, "_sd4_stage_timing_session", None
            )
            is not None,
            "stage_timing_hooks": list(
                getattr(self, "_sd4_stage_timing_hooks", ())
            ),
            "engine_device": str(getattr(self, "device", "unknown")),
            "guard_device": str(getattr(self, "_sd4_guard_device", "unknown")),
        }


__all__ = ["SD4WorkerExtension", "_summarize_stage_values"]
