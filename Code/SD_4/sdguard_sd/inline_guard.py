"""Online Guard monitoring for speculative proposals and verified commits.

The monitor is deliberately alarm-only: it never changes candidate tokens, Target
verification, sampling, or stopping.  A proposal is evaluated on a temporary KV-cache
branch and that branch is rolled back before the verified tokens are committed.
"""

from __future__ import annotations

from copy import deepcopy
import time
from collections import Counter
import statistics
from typing import Any, Iterable, Sequence

from .guard import _decode_head, _mapping
from .policies import CONTROVERSIAL, SAFE, UNSAFE, normalize_label
from .runtime import crop_cache_to_length, import_runtime, synchronize_devices


POLICIES = ("strict", "loose")
SCOPES = ("full_trace", "answer_only")


def _policy_matches(policy: str, left: str, right: str) -> bool:
    if policy == "strict":
        return left == right and left in {UNSAFE, CONTROVERSIAL}
    if policy == "loose":
        return left == right == UNSAFE
    raise ValueError(f"Unknown policy: {policy!r}")


def _safe_decision(policy: str) -> dict[str, Any]:
    return {
        "policy": policy,
        "prediction": SAFE,
        "triggered": False,
        "trigger_scope_index": None,
        "trigger_generated_index": None,
        "trigger_pair": None,
        "trigger_token_text": None,
        "trigger_category": None,
        "trigger_source": None,
        "trigger_round_index": None,
        "trigger_proposal_index": None,
        "trigger_survived_verification": None,
    }


def _first_trigger(
    records: Sequence[dict[str, Any]],
    *,
    policy: str,
    scope_start: int,
    source: str,
    round_index: int,
) -> dict[str, Any] | None:
    scoped = [record for record in records if int(record["generated_token_index"]) >= scope_start]
    for index in range(1, len(scoped)):
        left = normalize_label(scoped[index - 1]["risk_level"])
        right = normalize_label(scoped[index]["risk_level"])
        if not _policy_matches(policy, left, right):
            continue
        right_record = scoped[index]
        return {
            "policy": policy,
            "prediction": UNSAFE,
            "triggered": True,
            "trigger_scope_index": int(right_record["generated_token_index"]) - scope_start,
            "trigger_generated_index": int(right_record["generated_token_index"]),
            "trigger_pair": [left, right],
            "trigger_token_text": right_record.get("token_text"),
            "trigger_category": right_record.get("category"),
            "trigger_source": source,
            "trigger_round_index": round_index,
            "trigger_proposal_index": right_record.get("proposal_token_index"),
            "trigger_survived_verification": None,
        }
    return None


class InlineDecisionTracker:
    """Pure decision/state logic shared by offline replay and live insertion."""

    def __init__(self, *, enable_thinking: bool, close_think_token_id: int) -> None:
        self.enable_thinking = bool(enable_thinking)
        self.close_think_token_id = int(close_think_token_id)
        self.committed_ids: list[int] = []
        self.committed_records: list[dict[str, Any]] = []
        self.committed_answer_start: int | None = 0 if not enable_thinking else None
        self.immediate = {
            scope: {policy: _safe_decision(policy) for policy in POLICIES}
            for scope in SCOPES
        }
        self.verified = {
            scope: {policy: _safe_decision(policy) for policy in POLICIES}
            for scope in SCOPES
        }
        self.round_records: list[dict[str, Any]] = []
        self.token_records: list[dict[str, Any]] = []
        self.acceptance_by_risk: dict[str, Counter[str]] = {
            SAFE: Counter(),
            CONTROVERSIAL: Counter(),
            UNSAFE: Counter(),
        }
        self._pending: dict[str, Any] | None = None

    def capture_commit_state(self) -> dict[str, Any]:
        """Capture the bounded mutable state needed to redo the latest commit."""

        return {
            "committed_ids_len": len(self.committed_ids),
            "committed_records_len": len(self.committed_records),
            "round_records_len": len(self.round_records),
            "token_records_len": len(self.token_records),
            "committed_answer_start": self.committed_answer_start,
            "immediate": deepcopy(self.immediate),
            "verified": deepcopy(self.verified),
            "acceptance_by_risk": deepcopy(self.acceptance_by_risk),
            "pending": deepcopy(self._pending),
        }

    def restore_commit_state(self, snapshot: dict[str, Any]) -> None:
        """Restore a state returned by :meth:`capture_commit_state`."""

        del self.committed_ids[int(snapshot["committed_ids_len"]) :]
        del self.committed_records[int(snapshot["committed_records_len"]) :]
        del self.round_records[int(snapshot["round_records_len"]) :]
        del self.token_records[int(snapshot["token_records_len"]) :]
        self.committed_answer_start = snapshot["committed_answer_start"]
        self.immediate = deepcopy(snapshot["immediate"])
        self.verified = deepcopy(snapshot["verified"])
        self.acceptance_by_risk = deepcopy(snapshot["acceptance_by_risk"])
        self._pending = deepcopy(snapshot["pending"])

    def _latch(
        self,
        destination: dict[str, dict[str, dict[str, Any]]],
        scope: str,
        policy: str,
        trigger: dict[str, Any] | None,
    ) -> None:
        if trigger is not None and not destination[scope][policy]["triggered"]:
            destination[scope][policy] = trigger

    def observe_proposal(
        self,
        *,
        round_index: int,
        prefix_generated_count: int,
        token_ids: Sequence[int],
        token_records: Sequence[dict[str, Any]],
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("A proposal is already pending Target verification")
        if prefix_generated_count != len(self.committed_ids):
            raise RuntimeError(
                "Inline Guard prefix drift: "
                f"trace={prefix_generated_count}, committed={len(self.committed_ids)}"
            )
        if len(token_ids) != len(token_records):
            raise RuntimeError("Proposal token IDs and Guard records have different lengths")

        proposal_records: list[dict[str, Any]] = []
        for proposal_index, (token_id, raw) in enumerate(zip(token_ids, token_records)):
            record = {
                **raw,
                "phase": "proposal",
                "round_index": int(round_index),
                "proposal_token_index": proposal_index,
                "generated_token_index": prefix_generated_count + proposal_index,
                "token_id": int(token_id),
            }
            proposal_records.append(record)
            self.token_records.append(record)

        branch_records = (
            ([self.committed_records[-1]] if self.committed_records else [])
            + proposal_records
        )
        proposal_answer_start = self.committed_answer_start
        if proposal_answer_start is None:
            for proposal_index, token_id in enumerate(token_ids):
                if int(token_id) == self.close_think_token_id:
                    proposal_answer_start = prefix_generated_count + proposal_index + 1
                    break

        decisions: dict[str, Any] = {}
        for scope in SCOPES:
            available = scope == "full_trace" or proposal_answer_start is not None
            scope_start = 0 if scope == "full_trace" else proposal_answer_start
            scope_decisions: dict[str, Any] | None = None
            if available:
                assert scope_start is not None
                scope_decisions = {}
                for policy in POLICIES:
                    trigger = _first_trigger(
                        branch_records,
                        policy=policy,
                        scope_start=scope_start,
                        source="proposal",
                        round_index=round_index,
                    )
                    scope_decisions[policy] = trigger or _safe_decision(policy)
                    self._latch(self.immediate, scope, policy, trigger)
            decisions[scope] = {
                "available": available,
                "start_generated_index": scope_start,
                "decisions": scope_decisions,
            }

        self._pending = {
            "round_index": int(round_index),
            "prefix_generated_count": int(prefix_generated_count),
            "token_ids": [int(value) for value in token_ids],
            "token_records": proposal_records,
            "decisions": decisions,
        }

    def observe_commit(
        self,
        *,
        round_index: int,
        accepted_count: int,
        committed_ids: Sequence[int],
        commit_kind: str,
        token_records: Sequence[dict[str, Any]],
    ) -> None:
        pending = self._pending
        if pending is None or int(pending["round_index"]) != int(round_index):
            raise RuntimeError("Inline Guard commit does not match its pending proposal")
        if len(committed_ids) != len(token_records):
            raise RuntimeError("Committed token IDs and Guard records have different lengths")
        proposed_ids = pending["token_ids"]
        if accepted_count < 0 or accepted_count > len(proposed_ids):
            raise RuntimeError("Invalid accepted Draft token count")
        if list(committed_ids[:accepted_count]) != proposed_ids[:accepted_count]:
            raise RuntimeError("Accepted Draft prefix differs from committed token IDs")

        for scope in SCOPES:
            scope_data = pending["decisions"][scope]
            if not scope_data["available"]:
                continue
            for policy in POLICIES:
                decision = scope_data["decisions"][policy]
                if decision["triggered"]:
                    proposal_index = decision["trigger_proposal_index"]
                    decision["trigger_survived_verification"] = (
                        proposal_index is not None and int(proposal_index) < accepted_count
                    )
                    latched = self.immediate[scope][policy]
                    if (
                        latched["triggered"]
                        and latched["trigger_source"] == "proposal"
                        and latched["trigger_round_index"] == round_index
                        and latched["trigger_generated_index"]
                        == decision["trigger_generated_index"]
                    ):
                        latched["trigger_survived_verification"] = decision[
                            "trigger_survived_verification"
                        ]

        committed_records: list[dict[str, Any]] = []
        start = len(self.committed_ids)
        for commit_index, (token_id, raw) in enumerate(zip(committed_ids, token_records)):
            record = {
                **raw,
                "phase": "committed",
                "round_index": int(round_index),
                "commit_token_index": commit_index,
                "generated_token_index": start + commit_index,
                "token_id": int(token_id),
                "from_accepted_draft": commit_index < accepted_count,
            }
            committed_records.append(record)
            self.token_records.append(record)

        boundary_records = (
            ([self.committed_records[-1]] if self.committed_records else [])
            + committed_records
        )
        old_answer_start = self.committed_answer_start
        if self.committed_answer_start is None:
            for commit_index, token_id in enumerate(committed_ids):
                if int(token_id) == self.close_think_token_id:
                    self.committed_answer_start = start + commit_index + 1
                    break

        for scope in SCOPES:
            scope_start = 0 if scope == "full_trace" else self.committed_answer_start
            if scope_start is None:
                continue
            # If answer scope begins inside this commit, no pre-answer boundary may leak in.
            records = boundary_records
            if scope == "answer_only" and old_answer_start is None:
                records = [
                    record
                    for record in boundary_records
                    if int(record["generated_token_index"]) >= scope_start
                ]
            for policy in POLICIES:
                trigger = _first_trigger(
                    records,
                    policy=policy,
                    scope_start=scope_start,
                    source="committed",
                    round_index=round_index,
                )
                self._latch(self.verified, scope, policy, trigger)
                # proposal_immediate is a superset: it also observes Target correction/
                # bonus tokens so that proposal noise is the only semantic difference.
                self._latch(self.immediate, scope, policy, trigger)

        for proposal_index, record in enumerate(pending["token_records"]):
            label = normalize_label(record["risk_level"])
            bucket = self.acceptance_by_risk[label]
            bucket["proposed"] += 1
            if proposal_index < accepted_count:
                bucket["accepted"] += 1
            else:
                bucket["rejected"] += 1

        round_record = {
            "round_index": int(round_index),
            "prefix_generated_count": int(pending["prefix_generated_count"]),
            "proposed_ids": proposed_ids,
            "accepted_count": int(accepted_count),
            "committed_ids": [int(value) for value in committed_ids],
            "commit_kind": str(commit_kind),
            "proposal_scopes": pending["decisions"],
        }
        self.round_records.append(round_record)
        self.committed_ids.extend(int(value) for value in committed_ids)
        self.committed_records.extend(committed_records)
        self._pending = None

    def discard_pending(self, *, reason: str) -> None:
        """Close a proposal that vLLM produced after the request had already ended."""

        pending = self._pending
        if pending is None:
            return
        for scope in SCOPES:
            scope_data = pending["decisions"][scope]
            if not scope_data["available"]:
                continue
            for policy in POLICIES:
                decision = scope_data["decisions"][policy]
                if decision["triggered"]:
                    decision["trigger_survived_verification"] = False
        for record in pending["token_records"]:
            label = normalize_label(record["risk_level"])
            self.acceptance_by_risk[label]["proposed"] += 1
            self.acceptance_by_risk[label]["rejected"] += 1
        self.round_records.append(
            {
                "round_index": int(pending["round_index"]),
                "prefix_generated_count": int(pending["prefix_generated_count"]),
                "proposed_ids": list(pending["token_ids"]),
                "accepted_count": 0,
                "committed_ids": [],
                "commit_kind": str(reason),
                "proposal_scopes": pending["decisions"],
            }
        )
        self._pending = None

    @staticmethod
    def _scope_payload(
        decisions: dict[str, dict[str, dict[str, Any]]],
        scope: str,
        *,
        answer_start_generated_index: int | None,
        generated_token_count: int,
    ) -> dict[str, Any]:
        if scope == "answer_only" and answer_start_generated_index is None:
            return {
                "available": False,
                "start_generated_index": None,
                "token_count": 0,
                "decisions": None,
                "unavailable_reason": "missing_close_think",
            }
        start = 0 if scope == "full_trace" else int(answer_start_generated_index)
        return {
            "available": True,
            "start_generated_index": start,
            "token_count": generated_token_count - start,
            "decisions": decisions[scope],
            "unavailable_reason": None,
        }

    def finalize(self, *, answer_start_generated_index: int | None) -> dict[str, Any]:
        if self._pending is not None:
            raise RuntimeError("Generation ended with an uncommitted speculative proposal")
        if self.committed_answer_start != answer_start_generated_index:
            raise RuntimeError(
                "Inline answer boundary differs from final completion parser: "
                f"{self.committed_answer_start!r} != {answer_start_generated_index!r}"
            )
        count = len(self.committed_ids)
        immediate_scopes = {
            scope: self._scope_payload(
                self.immediate,
                scope,
                answer_start_generated_index=answer_start_generated_index,
                generated_token_count=count,
            )
            for scope in SCOPES
        }
        verified_scopes = {
            scope: self._scope_payload(
                self.verified,
                scope,
                answer_start_generated_index=answer_start_generated_index,
                generated_token_count=count,
            )
            for scope in SCOPES
        }
        acceptance = {}
        for label in (SAFE, CONTROVERSIAL, UNSAFE):
            values = self.acceptance_by_risk[label]
            proposed = int(values["proposed"])
            accepted = int(values["accepted"])
            acceptance[label] = {
                "proposed": proposed,
                "accepted": accepted,
                "rejected": int(values["rejected"]),
                "acceptance_rate": None if proposed == 0 else accepted / proposed,
            }
        noise: dict[str, Any] = {}
        proposal_trigger_outcomes: dict[str, Any] = {}
        for scope in SCOPES:
            noise[scope] = {}
            proposal_trigger_outcomes[scope] = {}
            if not immediate_scopes[scope]["available"]:
                for policy in POLICIES:
                    noise[scope][policy] = {
                        "available": False,
                        "proposal_immediate_alert": None,
                        "verified_commit_alert": None,
                        "proposal_only_alert": None,
                        "first_alert_survived_verification": None,
                    }
                    proposal_trigger_outcomes[scope][policy] = {
                        "triggered_round_count": 0,
                        "survived_round_count": 0,
                        "rejected_round_count": 0,
                    }
                continue
            for policy in POLICIES:
                immediate = self.immediate[scope][policy]
                verified = self.verified[scope][policy]
                noise[scope][policy] = {
                    "available": True,
                    "proposal_immediate_alert": bool(immediate["triggered"]),
                    "verified_commit_alert": bool(verified["triggered"]),
                    "proposal_only_alert": bool(
                        immediate["triggered"] and not verified["triggered"]
                    ),
                    "first_alert_survived_verification": immediate[
                        "trigger_survived_verification"
                    ],
                }
                round_decisions = [
                    record["proposal_scopes"][scope]["decisions"][policy]
                    for record in self.round_records
                    if record["proposal_scopes"][scope]["available"]
                ]
                triggered = [value for value in round_decisions if value["triggered"]]
                proposal_trigger_outcomes[scope][policy] = {
                    "triggered_round_count": len(triggered),
                    "survived_round_count": sum(
                        value["trigger_survived_verification"] is True
                        for value in triggered
                    ),
                    "rejected_round_count": sum(
                        value["trigger_survived_verification"] is False
                        for value in triggered
                    ),
                }
        return {
            "proposal_immediate": {"scopes": immediate_scopes},
            "verified_commit": {"scopes": verified_scopes},
            "noise": noise,
            "proposal_trigger_outcomes": proposal_trigger_outcomes,
            "acceptance_by_proposal_risk": acceptance,
        }


class IncrementalGuardMonitor:
    """Guard KV-cache engine plus :class:`InlineDecisionTracker`."""

    def __init__(
        self,
        guard_model: Any,
        tokenizer: Any,
        *,
        guard_device: str,
        enable_thinking: bool,
    ) -> None:
        self.guard_model = guard_model
        self.tokenizer = tokenizer
        self.guard_device = guard_device
        close_id = tokenizer.convert_tokens_to_ids("</think>")
        if close_id is None or int(close_id) < 0:
            raise RuntimeError("Guard tokenizer does not define </think>")
        self.tracker = InlineDecisionTracker(
            enable_thinking=enable_thinking,
            close_think_token_id=int(close_id),
        )
        self.prompt_ids: tuple[int, ...] | None = None
        self.prompt_risk_level: str | None = None
        self.prompt_risk_probability: float | None = None
        self.cache: Any = None
        self.prefill_latency_ms = 0.0
        self.proposal_latency_ms = 0.0
        self.commit_latency_ms = 0.0
        self.prefill_input_packaging_ms = 0.0
        self.prefill_input_h2d_ms = 0.0
        self.proposal_input_packaging_ms = 0.0
        self.proposal_input_h2d_ms = 0.0
        self.commit_input_packaging_ms = 0.0
        self.commit_input_h2d_ms = 0.0
        self.proposal_output_d2h_ms = 0.0
        self.commit_output_d2h_ms = 0.0
        self.proposal_forward_count = 0
        self.commit_forward_count = 0
        self._pending_cache_length: int | None = None
        self._pending_proposal_ids: tuple[int, ...] | None = None
        self._pending_proposal_records: list[dict[str, Any]] | None = None
        self._commit_transactions: list[dict[str, Any]] = []
        self._last_commit_transaction: dict[str, Any] | None = None
        self.response_risk_map = _mapping(
            guard_model,
            "response_risk_level_map",
            {0: SAFE, 1: UNSAFE, 2: CONTROVERSIAL},
        )
        self.response_category_map = _mapping(
            guard_model, "response_category_map", {}
        )
        self.query_risk_map = _mapping(
            guard_model,
            "query_risk_level_map",
            {0: SAFE, 1: UNSAFE, 2: CONTROVERSIAL},
        )

    def _forward(
        self, token_ids: Sequence[int], *, profile_stage: str
    ) -> tuple[list[dict[str, Any]], float, Any | None, float, float]:
        if not token_ids:
            return [], 0.0, None, 0.0, 0.0
        torch, _, _, _ = import_runtime()
        packaging_started_ns = time.perf_counter_ns()
        host_input_ids = torch.tensor(
            [int(value) for value in token_ids],
            dtype=torch.long,
        ).unsqueeze(0)
        packaging_ms = (
            time.perf_counter_ns() - packaging_started_ns
        ) / 1_000_000.0
        synchronize_devices(self.guard_device)
        transfer_started_ns = time.perf_counter_ns()
        input_ids = host_input_ids.to(device=self.guard_device, non_blocking=False)
        synchronize_devices(self.guard_device)
        input_h2d_ms = (
            time.perf_counter_ns() - transfer_started_ns
        ) / 1_000_000.0
        started_ns = time.perf_counter_ns()
        with torch.profiler.record_function(f"sd4_guard:{profile_stage}"):
            with torch.inference_mode():
                outputs = self.guard_model.forward(
                    input_ids=input_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                    logits_to_keep=0,
                    return_dict=True,
                )
                risk_labels, risk_probabilities = _decode_head(
                    torch, outputs.risk_level_logits[0], self.response_risk_map
                )
                categories, category_probabilities = _decode_head(
                    torch, outputs.category_logits[0], self.response_category_map
                )
        # The CPU decoding above synchronizes the Guard device, so this is real
        # observed wall latency for the inserted forward rather than enqueue time.
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
        self.cache = outputs.past_key_values
        if self.cache is None or not hasattr(self.cache, "crop"):
            raise RuntimeError("Qwen3Guard did not return a croppable DynamicCache")
        records = []
        for index, token_id in enumerate(token_ids):
            records.append(
                {
                    "token_id": int(token_id),
                    "token_text": self.tokenizer.decode(
                        [int(token_id)], skip_special_tokens=False
                    ),
                    "risk_level": normalize_label(risk_labels[index]),
                    "risk_probability": risk_probabilities[index],
                    "category": categories[index],
                    "category_probability": category_probabilities[index],
                }
            )
        return records, elapsed_ms, outputs, packaging_ms, input_h2d_ms

    def start(self, prompt_ids: Sequence[int]) -> None:
        if self.prompt_ids is not None:
            raise RuntimeError("Inline Guard monitor was started twice")
        full = tuple(int(value) for value in prompt_ids)
        max_positions = int(
            getattr(self.guard_model.config, "max_position_embeddings", len(full))
        )
        if len(full) > max_positions:
            raise RuntimeError("Generation prompt exceeds the Guard context window")
        records, elapsed_ms, outputs, packaging_ms, input_h2d_ms = self._forward(
            full, profile_stage="prefill"
        )
        self.prompt_ids = full
        self.prefill_latency_ms = elapsed_ms
        self.prefill_input_packaging_ms = packaging_ms
        self.prefill_input_h2d_ms = input_h2d_ms
        user_end_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        matches = [index for index, value in enumerate(full) if value == user_end_id]
        if not matches:
            raise RuntimeError("Could not find user <|im_end|> in the generation prompt")
        user_end = matches[-1]

        torch, _, _, _ = import_runtime()
        started_ns = time.perf_counter_ns()
        assert outputs is not None
        labels, probabilities = _decode_head(
            torch,
            outputs.query_risk_level_logits[0, user_end : user_end + 1, :],
            self.query_risk_map,
        )
        self.prefill_latency_ms += (time.perf_counter_ns() - started_ns) / 1_000_000.0
        self.prompt_risk_level = normalize_label(labels[-1])
        self.prompt_risk_probability = probabilities[-1]
        del records

    def _check_context(self, extra_count: int) -> None:
        if self.prompt_ids is None:
            raise RuntimeError("Inline Guard monitor has not been started")
        total = len(self.prompt_ids) + len(self.tracker.committed_ids) + extra_count
        maximum = int(getattr(self.guard_model.config, "max_position_embeddings", total))
        if total > maximum:
            raise RuntimeError(
                f"Inline Guard path length {total} exceeds max_position_embeddings={maximum}"
            )

    def observe_proposal(
        self,
        *,
        round_index: int,
        prefix_length: int,
        proposed_ids: Sequence[int],
    ) -> None:
        if self.prompt_ids is None:
            raise RuntimeError("Inline Guard monitor has not been started")
        prefix_generated = int(prefix_length) - len(self.prompt_ids)
        self._check_context(len(proposed_ids))
        if self._pending_cache_length is not None:
            raise RuntimeError("Guard cache still contains an unverified proposal branch")
        before = int(self.cache.get_seq_length())
        if before != int(prefix_length):
            raise RuntimeError(
                f"Guard KV-cache length differs from SD prefix: {before} != {prefix_length}"
            )
        records, elapsed_ms, _, packaging_ms, input_h2d_ms = self._forward(
            proposed_ids, profile_stage="proposal"
        )
        self.proposal_latency_ms += elapsed_ms
        self.proposal_input_packaging_ms += packaging_ms
        self.proposal_input_h2d_ms += input_h2d_ms
        self.proposal_forward_count += 1
        self._pending_cache_length = before
        self._pending_proposal_ids = tuple(int(value) for value in proposed_ids)
        self._pending_proposal_records = records
        self.tracker.observe_proposal(
            round_index=round_index,
            prefix_generated_count=prefix_generated,
            token_ids=proposed_ids,
            token_records=records,
        )

    def observe_commit(self, round_: Any) -> None:
        self._check_context(len(round_.committed_ids))
        if (
            self._pending_cache_length is None
            or self._pending_proposal_ids is None
            or self._pending_proposal_records is None
        ):
            raise RuntimeError("Target commit arrived without a Guarded proposal branch")
        if tuple(round_.proposed_ids) != self._pending_proposal_ids:
            raise RuntimeError("Target commit proposal differs from the Guarded branch")
        accepted = int(round_.accepted_count)
        if accepted < 0 or accepted > len(self._pending_proposal_ids):
            raise RuntimeError("Target returned an invalid accepted Draft token count")
        if tuple(round_.committed_ids[:accepted]) != self._pending_proposal_ids[:accepted]:
            raise RuntimeError("Target commit does not begin with its accepted Draft prefix")
        retained_length = self._pending_cache_length + accepted
        # Keep accepted Draft states and remove only the rejected suffix.  The
        # negative tokens-to-remove form is the non-deprecated 5.17 cache API.
        crop_cache_to_length(self.cache, retained_length)
        target_tail = tuple(int(value) for value in round_.committed_ids[accepted:])
        tail_records, elapsed_ms, _, packaging_ms, input_h2d_ms = self._forward(
            target_tail, profile_stage="commit"
        )
        records = self._pending_proposal_records[:accepted] + tail_records
        tracker_snapshot = self.tracker.capture_commit_state()
        self.commit_latency_ms += elapsed_ms
        self.commit_input_packaging_ms += packaging_ms
        self.commit_input_h2d_ms += input_h2d_ms
        if target_tail:
            self.commit_forward_count += 1
        self.tracker.observe_commit(
            round_index=int(round_.round_index),
            accepted_count=int(round_.accepted_count),
            committed_ids=round_.committed_ids,
            commit_kind=str(round_.commit_kind),
            token_records=records,
        )
        transaction = {
            "tracker_snapshot": tracker_snapshot,
            "round_index": int(round_.round_index),
            "proposed_ids": tuple(int(value) for value in round_.proposed_ids),
            "accepted_count": accepted,
            "committed_ids": tuple(int(value) for value in round_.committed_ids),
            "commit_kind": str(round_.commit_kind),
            "token_records": list(records),
            "target_verified_proposed_count": len(round_.proposed_ids),
            "target_verified_accepted_count": accepted,
        }
        self._commit_transactions.append(transaction)
        self._last_commit_transaction = transaction
        self._pending_cache_length = None
        self._pending_proposal_ids = None
        self._pending_proposal_records = None

    def set_last_commit_verification_counts(
        self, *, accepted_count: int, proposed_count: int
    ) -> None:
        """Attach vLLM's uncropped rejection-sampling counts to the last commit."""

        transaction = self._last_commit_transaction
        if transaction is None:
            raise RuntimeError("No inline commit is available for vLLM metric annotation")
        transaction["target_verified_accepted_count"] = int(accepted_count)
        transaction["target_verified_proposed_count"] = int(proposed_count)
        if not self.tracker.round_records:
            raise RuntimeError("Inline commit did not create a round record")
        self.tracker.round_records[-1]["target_verified_accepted_count"] = int(
            accepted_count
        )
        self.tracker.round_records[-1]["target_verified_proposed_count"] = int(
            proposed_count
        )

    def clip_commits_to_output(
        self,
        generated_ids: Sequence[int],
        *,
        target_verified_accepted_count: int | None = None,
        target_verified_proposed_count: int | None = None,
    ) -> dict[str, Any]:
        """Clip the last GPU commit to the scheduler-visible terminal output.

        vLLM can verify tokens beyond EOS before its scheduler applies stop
        handling.  This only reconciles our observation record; it never changes
        Draft proposals, Target sampling, or vLLM request state.
        """

        expected = [int(value) for value in generated_ids]
        committed = list(self.tracker.committed_ids)
        if committed == expected:
            return {"applied": False, "clipped_token_count": 0}
        if len(expected) >= len(committed) or committed[: len(expected)] != expected:
            raise RuntimeError(
                "Terminal clipping requires generated output to be a strict committed prefix"
            )
        transactions = list(self._commit_transactions)
        if not transactions:
            raise RuntimeError("Terminal clipping has no reversible commit transactions")
        transaction_index = next(
            (
                index
                for index, item in enumerate(transactions)
                if int(item["tracker_snapshot"]["committed_ids_len"])
                <= len(expected)
                <= int(item["tracker_snapshot"]["committed_ids_len"])
                + len(item["committed_ids"])
            ),
            None,
        )
        if transaction_index is None:
            commit_ranges = [
                (
                    int(item["tracker_snapshot"]["committed_ids_len"]),
                    int(item["tracker_snapshot"]["committed_ids_len"])
                    + len(item["committed_ids"]),
                )
                for item in transactions
            ]
            raise RuntimeError(
                "Terminal output does not end inside any observed commit: "
                f"output_len={len(expected)}, commit_ranges="
                f"{commit_ranges}"
            )
        transaction = transactions[transaction_index]
        snapshot = transaction["tracker_snapshot"]
        prefix_count = int(snapshot["committed_ids_len"])
        visible_tail = expected[prefix_count:]
        original_tail = list(transaction["committed_ids"])
        if original_tail[: len(visible_tail)] != visible_tail:
            raise RuntimeError(
                "Scheduler-visible terminal tokens are not a prefix of the latest commit"
            )

        # The cache can also contain a new proposal produced before the scheduler
        # noticed EOS.  Since all retained tokens are a prefix, one crop removes
        # that unused proposal and the post-stop commit suffix.
        crop_cache_to_length(
            self.cache, len(self.prompt_ids or ()) + len(expected)
        )
        self.tracker.restore_commit_state(snapshot)
        semantic_accepted = min(
            int(transaction["accepted_count"]), len(visible_tail)
        )
        self.tracker.observe_commit(
            round_index=int(transaction["round_index"]),
            accepted_count=semantic_accepted,
            committed_ids=visible_tail,
            commit_kind=str(transaction["commit_kind"]),
            token_records=list(transaction["token_records"][: len(visible_tail)]),
        )
        record = self.tracker.round_records[-1]
        # Finish-time metric overrides refer to the latest round.  They are safe
        # only when the scheduler stop lies in that same transaction; otherwise
        # every earlier transaction already contains the exact live V2 counters.
        is_latest = transaction_index == len(transactions) - 1
        verified_accepted = int(
            target_verified_accepted_count
            if is_latest and target_verified_accepted_count is not None
            else transaction["target_verified_accepted_count"]
        )
        verified_proposed = int(
            target_verified_proposed_count
            if is_latest and target_verified_proposed_count is not None
            else transaction["target_verified_proposed_count"]
        )
        record.update(
            {
                "target_verified_accepted_count": verified_accepted,
                "target_verified_proposed_count": verified_proposed,
                "terminal_output_clipped": True,
                "terminal_clipped_token_count": len(committed) - len(expected),
            }
        )
        # Later GPU verification rounds happened before vLLM's scheduler noticed
        # the stop token.  Retain their Target counters as executed-work
        # diagnostics, but exclude them from scheduler-visible metric parity and
        # do not replay their tokens or Guard decisions into the completion.
        for discarded in transactions[transaction_index + 1 :]:
            discarded_snapshot = discarded["tracker_snapshot"]
            pending = discarded_snapshot.get("pending") or {}
            self.tracker.round_records.append(
                {
                    "round_index": int(discarded["round_index"]),
                    "prefix_generated_count": int(
                        pending.get(
                            "prefix_generated_count",
                            discarded_snapshot["committed_ids_len"],
                        )
                    ),
                    "proposed_ids": [int(value) for value in discarded["proposed_ids"]],
                    "accepted_count": 0,
                    "committed_ids": [],
                    "commit_kind": "terminal_output_clipped",
                    "proposal_scopes": {
                        scope: {
                            "available": False,
                            "start_generated_index": None,
                            "decisions": None,
                        }
                        for scope in SCOPES
                    },
                    "target_verified_accepted_count": int(
                        discarded["target_verified_accepted_count"]
                    ),
                    "target_verified_proposed_count": int(
                        discarded["target_verified_proposed_count"]
                    ),
                    "scheduler_metric_included": False,
                    "terminal_output_clipped": True,
                    "terminal_clipped_token_count": len(discarded["committed_ids"]),
                }
            )
        self._pending_cache_length = None
        self._pending_proposal_ids = None
        self._pending_proposal_records = None
        self._commit_transactions = transactions[: transaction_index + 1]
        self._last_commit_transaction = None
        return {
            "applied": True,
            "clipped_token_count": len(committed) - len(expected),
            "target_verified_accepted_count": verified_accepted,
            "target_verified_proposed_count": verified_proposed,
            "scheduler_visible_accepted_count": semantic_accepted,
            "discarded_verified_round_count": (
                len(transactions) - transaction_index - 1
            ),
        }

    def observe_unproposed_commit(
        self,
        *,
        round_index: int,
        committed_ids: Sequence[int],
        commit_kind: str,
    ) -> None:
        """Commit bootstrap/final Target tokens that have no preceding Draft branch."""

        prefix_length = len(self.prompt_ids or ()) + len(self.tracker.committed_ids)
        self.observe_proposal(
            round_index=round_index,
            prefix_length=prefix_length,
            proposed_ids=(),
        )
        round_value = type(
            "SD4Commit",
            (),
            {
                "round_index": round_index,
                "proposed_ids": (),
                "accepted_count": 0,
                "committed_ids": tuple(int(value) for value in committed_ids),
                "commit_kind": commit_kind,
            },
        )()
        self.observe_commit(round_value)

    def discard_pending(self, *, reason: str = "unverified_after_stop") -> None:
        """Roll back a terminal proposal that vLLM will never send to Target."""

        if self._pending_cache_length is None:
            return
        crop_cache_to_length(self.cache, self._pending_cache_length)
        self.tracker.discard_pending(reason=reason)
        self._pending_cache_length = None
        self._pending_proposal_ids = None
        self._pending_proposal_records = None

    def finalize(
        self,
        *,
        generated_ids: Sequence[int],
        answer_start_generated_index: int | None,
        posthoc_guard: Any | None = None,
    ) -> dict[str, Any]:
        expected = [int(value) for value in generated_ids]
        if self.tracker.committed_ids != expected:
            raise RuntimeError("Inline committed token stream differs from generated output")
        result = self.tracker.finalize(
            answer_start_generated_index=answer_start_generated_index
        )
        total_latency = (
            self.prefill_latency_ms + self.proposal_latency_ms + self.commit_latency_ms
        )
        staged_handoff_ms = (
            self.proposal_output_d2h_ms
            + self.proposal_input_h2d_ms
            + self.commit_output_d2h_ms
            + self.commit_input_h2d_ms
        )
        result["proposal_immediate"].update(
            {
                "prompt_risk_level": self.prompt_risk_level,
                "prompt_risk_probability": self.prompt_risk_probability,
                "latency_ms": total_latency,
            }
        )
        result["verified_commit"].update(
            {
                "prompt_risk_level": self.prompt_risk_level,
                "prompt_risk_probability": self.prompt_risk_probability,
                # Accepted-token labels are produced by the proposal forward and
                # reused after verification, so there is no disjoint verified-only
                # wall-clock path.  Report the actual total inserted Guard time.
                "latency_ms": total_latency,
            }
        )
        result["timing"] = {
            "prefill_latency_ms": self.prefill_latency_ms,
            "proposal_latency_ms": self.proposal_latency_ms,
            "commit_latency_ms": self.commit_latency_ms,
            "total_inline_guard_latency_ms": total_latency,
            "prefill_input_packaging_ms": self.prefill_input_packaging_ms,
            "prefill_input_h2d_ms": self.prefill_input_h2d_ms,
            "proposal_input_packaging_ms": self.proposal_input_packaging_ms,
            "proposal_input_h2d_ms": self.proposal_input_h2d_ms,
            "commit_input_packaging_ms": self.commit_input_packaging_ms,
            "commit_input_h2d_ms": self.commit_input_h2d_ms,
            "proposal_output_d2h_ms": self.proposal_output_d2h_ms,
            "commit_output_d2h_ms": self.commit_output_d2h_ms,
            "staged_cross_device_handoff_ms": staged_handoff_ms,
            "proposal_forward_count": self.proposal_forward_count,
            "commit_forward_count": self.commit_forward_count,
        }

        consistency = {
            "verified_token_labels_match_posthoc": None,
            "verified_scope_decisions_match_posthoc": None,
            "verified_posthoc_match": None,
            "verified_posthoc_exact_match": None,
            "verified_posthoc_decision_consistent": None,
            "first_label_mismatch_generated_index": None,
            "token_label_mismatch_count": None,
            "token_label_mismatch_examples": [],
            "token_label_transition_counts": {},
        }
        if posthoc_guard is not None:
            online_labels = [
                normalize_label(record["risk_level"])
                for record in self.tracker.committed_records
            ]
            posthoc_labels = [
                normalize_label(record["risk_level"])
                for record in posthoc_guard.token_records
            ]
            labels_match = online_labels == posthoc_labels
            mismatch_indices = [
                index
                for index, (left, right) in enumerate(zip(online_labels, posthoc_labels))
                if left != right
            ]
            if len(online_labels) != len(posthoc_labels):
                mismatch_indices.extend(
                    range(min(len(online_labels), len(posthoc_labels)), max(len(online_labels), len(posthoc_labels)))
                )
            first_mismatch = mismatch_indices[0] if mismatch_indices else None
            transitions: Counter[str] = Counter()
            mismatch_examples: list[dict[str, Any]] = []
            for index in mismatch_indices:
                online_record = (
                    self.tracker.committed_records[index]
                    if index < len(self.tracker.committed_records)
                    else None
                )
                posthoc_record = (
                    posthoc_guard.token_records[index]
                    if index < len(posthoc_guard.token_records)
                    else None
                )
                online_label = None if online_record is None else online_record["risk_level"]
                posthoc_label = None if posthoc_record is None else posthoc_record["risk_level"]
                transitions[f"{online_label}->{posthoc_label}"] += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        {
                            "generated_token_index": index,
                            "token_id": (
                                None if online_record is None else online_record.get("token_id")
                            ),
                            "token_text": (
                                None if online_record is None else online_record.get("token_text")
                            ),
                            "online_risk_level": online_label,
                            "online_risk_probability": (
                                None
                                if online_record is None
                                else online_record.get("risk_probability")
                            ),
                            "posthoc_risk_level": posthoc_label,
                            "posthoc_risk_probability": (
                                None
                                if posthoc_record is None
                                else posthoc_record.get("risk_probability")
                            ),
                        }
                    )

            def projection(scopes: dict[str, Any]) -> dict[str, Any]:
                projected: dict[str, Any] = {}
                for scope in SCOPES:
                    value = scopes[scope]
                    if not value["available"]:
                        projected[scope] = None
                        continue
                    projected[scope] = {
                        policy: {
                            key: value["decisions"][policy].get(key)
                            for key in (
                                "prediction",
                                "triggered",
                                "trigger_generated_index",
                                "trigger_pair",
                            )
                        }
                        for policy in POLICIES
                    }
                return projected

            decisions_match = projection(result["verified_commit"]["scopes"]) == projection(
                posthoc_guard.scopes
            )
            consistency.update(
                {
                    "verified_token_labels_match_posthoc": labels_match,
                    "verified_scope_decisions_match_posthoc": decisions_match,
                    "verified_posthoc_match": labels_match and decisions_match,
                    "verified_posthoc_exact_match": labels_match and decisions_match,
                    "verified_posthoc_decision_consistent": decisions_match,
                    "first_label_mismatch_generated_index": first_mismatch,
                    "token_label_mismatch_count": len(mismatch_indices),
                    "token_label_mismatch_examples": mismatch_examples,
                    "token_label_transition_counts": dict(transitions),
                }
            )
        result["consistency"] = consistency
        return result


def aggregate_inline_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the diagnostics used by both replay and the final 2x4 report."""

    rows = list(records)
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "verified_posthoc_decision_match_count": sum(
            row["consistency"].get("verified_posthoc_decision_consistent") is True
            for row in rows
        ),
        "verified_posthoc_decision_mismatch_count": sum(
            row["consistency"].get("verified_posthoc_decision_consistent") is False
            for row in rows
        ),
        "verified_posthoc_exact_match_count": sum(
            row["consistency"].get("verified_posthoc_exact_match") is True
            for row in rows
        ),
        "verified_posthoc_exact_mismatch_count": sum(
            row["consistency"].get("verified_posthoc_exact_match") is False
            for row in rows
        ),
        "verified_posthoc_match_count": sum(
            row["consistency"].get("verified_posthoc_match") is True for row in rows
        ),
        "verified_posthoc_mismatch_count": sum(
            row["consistency"].get("verified_posthoc_match") is False for row in rows
        ),
        "alerts": {},
        "acceptance_by_proposal_risk": {},
        "timing_ms": {},
    }
    label_transitions: Counter[str] = Counter()
    for row in rows:
        label_transitions.update(
            row["consistency"].get("token_label_transition_counts", {})
        )
    result["token_label_consistency"] = {
        "sample_mismatch_count": sum(
            int(row["consistency"].get("token_label_mismatch_count") or 0) > 0
            for row in rows
        ),
        "total_mismatched_token_count": sum(
            int(row["consistency"].get("token_label_mismatch_count") or 0)
            for row in rows
        ),
        "transition_counts_online_to_posthoc": dict(label_transitions),
        "note": (
            "Incremental KV-cache and one-shot full-forward can differ at isolated "
            "bf16 decision boundaries; response policy consistency is reported separately."
        ),
    }
    for scope in SCOPES:
        result["alerts"][scope] = {}
        for policy in POLICIES:
            values = [row["noise"][scope][policy] for row in rows]
            available = [value for value in values if value["available"]]
            result["alerts"][scope][policy] = {
                "available_count": len(available),
                "proposal_immediate_alert_count": sum(
                    value["proposal_immediate_alert"] is True for value in available
                ),
                "verified_commit_alert_count": sum(
                    value["verified_commit_alert"] is True for value in available
                ),
                "proposal_only_alert_count": sum(
                    value["proposal_only_alert"] is True for value in available
                ),
                "first_alert_rejected_count": sum(
                    value["proposal_immediate_alert"] is True
                    and value["first_alert_survived_verification"] is False
                    for value in available
                ),
                "first_alert_survived_count": sum(
                    value["proposal_immediate_alert"] is True
                    and value["first_alert_survived_verification"] is True
                    for value in available
                ),
                "proposal_triggered_round_count": sum(
                    int(
                        row["proposal_trigger_outcomes"][scope][policy][
                            "triggered_round_count"
                        ]
                    )
                    for row in rows
                ),
                "proposal_trigger_survived_round_count": sum(
                    int(
                        row["proposal_trigger_outcomes"][scope][policy][
                            "survived_round_count"
                        ]
                    )
                    for row in rows
                ),
                "proposal_trigger_rejected_round_count": sum(
                    int(
                        row["proposal_trigger_outcomes"][scope][policy][
                            "rejected_round_count"
                        ]
                    )
                    for row in rows
                ),
            }
    for label in (SAFE, CONTROVERSIAL, UNSAFE):
        proposed = sum(
            int(row["acceptance_by_proposal_risk"][label]["proposed"]) for row in rows
        )
        accepted = sum(
            int(row["acceptance_by_proposal_risk"][label]["accepted"]) for row in rows
        )
        result["acceptance_by_proposal_risk"][label] = {
            "proposed": proposed,
            "accepted": accepted,
            "rejected": proposed - accepted,
            "acceptance_rate": None if proposed == 0 else accepted / proposed,
        }
    timing_keys = (
        "prefill_latency_ms",
        "proposal_latency_ms",
        "commit_latency_ms",
        "total_inline_guard_latency_ms",
        "prefill_input_packaging_ms",
        "prefill_input_h2d_ms",
        "proposal_input_packaging_ms",
        "proposal_input_h2d_ms",
        "commit_input_packaging_ms",
        "commit_input_h2d_ms",
        "proposal_output_d2h_ms",
        "commit_output_d2h_ms",
        "staged_cross_device_handoff_ms",
        "inline_end_to_end_generation_latency_ms",
        "observed_generation_overhead_ms",
    )
    for key in timing_keys:
        values = [float(row["timing"][key]) for row in rows if key in row.get("timing", {})]
        result["timing_ms"][key] = {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "median": statistics.median(values) if values else None,
        }
    return result


__all__ = [
    "IncrementalGuardMonitor",
    "InlineDecisionTracker",
    "aggregate_inline_records",
]

