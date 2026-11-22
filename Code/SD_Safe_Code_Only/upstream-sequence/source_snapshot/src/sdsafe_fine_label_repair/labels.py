"""Official-checker labels for repaired positions and the fixed L0..L4 policies.

New candidate positions are replayed only on the affected episode with the current
official AgentDojo adapter/checker, using the original archived serializer and the
exact causal visible prefix.  Old labels are reused for a UTF-8 endpoint correction only
when the API-visible text prefix and environment/task binding are exactly the same label
point.  A divergence is ``unknown`` with a machine-readable reason, never ``0``.

Policy semantics (TRAIN only, exact rows only):

``L0``  first exact security flip of each strict-final-positive episode -> 1; every
        other exact row -> 0.  This is the old first_hit rule, recomputed on repaired
        rows (not a silently overlaid legacy vector view).
``L1``  keep L0 positives, label exact prefix-false rows 0, and mask every other exact
        prefix-positive row instead of calling it a negative.
``L2``  L1 plus exact prefix-positive rows in the same episode/assistant round/message
        as the L0 first-hit anchor -> 1; other prefix-positive rows stay masked.  This
        is a local prefix-risk target, not a claim that every post-hit text is malicious.
``L3``  every exact row's label is its exact prefix state, including transient positives
        in strict-final-negative episodes.
``L4``  random L0-negative drop control: keep L0 positives; drop the same number of L0
        negatives as L1 per ``(scope,event_type)`` stratum with fixed RNG 20261109.
        This is an imperfect-label control, not a candidate accurate policy.

Unknown labels are always masked out of loss and retained as inference positions.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sdsafe_features.common import canonical_sha256
from sdsafe_fine_confirm import train_data

from . import (
    EVENT_CONTENT_SEGMENT,
    EVENT_TOOL_CALL,
    LABEL_EXACT,
    LABEL_UNKNOWN,
    L4_MASK_RNG_SEED,
    POLICIES,
    SPLIT_TRAIN,
    FineLabelRepairError,
    LabelRepairError,
)

MASK_UNKNOWN = "unknown_label_never_loss"
MASK_PREFIX_NON_ANCHOR = "prefix_positive_non_anchor_censored"
MASK_L4_DROP = "random_drop_matched_l0_negative"
UNKNOWN_MIXED_ORDER = "mixed_generation_execution_order"
UNKNOWN_CONTENT_PREFIX = "content_prefix_closure_mismatch"
UNKNOWN_ROUND_END = "round_end_checker_divergence"
UNKNOWN_ENVIRONMENT = "environment_control_divergence"
UNKNOWN_TOOL_RESULT = "tool_result_divergence"
UNKNOWN_CONTROL_RESULT = "paired_control_result_divergence"
UNKNOWN_CALL_SNAPSHOT = "call_snapshot_mismatch"
UNKNOWN_CHECKER_MUTATION = "checker_mutated_environment"


def _event_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return train_data.event_order_key(row) + (str(row["event_id"]),)


def _key(row: Mapping[str, Any]) -> str:
    return str(row["event_id"])


def _exact(row: Mapping[str, Any]) -> bool:
    return str(row.get("label_status")) == LABEL_EXACT


def _prefix_positive(row: Mapping[str, Any]) -> bool:
    return row.get("prefix_security") is True


def _final_positive(row: Mapping[str, Any]) -> bool:
    if row.get("strict_final") is None:
        raise LabelRepairError(f"{row.get('event_id')!r}: missing strict_final")
    return bool(row["strict_final"])


def _validate_label_state(row: Mapping[str, Any]) -> None:
    event_id = row.get("event_id")
    if not _exact(row):
        if row.get("prefix_security") is not None or row.get("previous_security") is not None:
            raise LabelRepairError(f"{event_id!r}: unknown row carries checker state")
        if row.get("security_flip") is not None:
            raise LabelRepairError(f"{event_id!r}: unknown row carries security_flip")
        return
    prefix = row.get("prefix_security")
    previous = row.get("previous_security")
    flip = row.get("security_flip")
    if not isinstance(prefix, bool) or not isinstance(previous, bool):
        raise LabelRepairError(f"{event_id!r}: exact row lacks boolean checker state")
    expected = bool(prefix and not previous)
    if flip is not None and bool(flip) is not expected:
        raise LabelRepairError(f"{event_id!r}: security_flip is incoherent with previous/prefix state")


# --------------------------------------------------------------------------- candidate replay


def _material_reasons(round_row: Any) -> list[str]:
    from sdsafe_event_data.events import MATERIAL_ROUND_ISSUES

    return [str(issue["reason"]) for issue in round_row.issues if str(issue["reason"]) in MATERIAL_ROUND_ISSUES]


def _visible_prefixes_for_episode(episode: Any, tokenizer: Any, candidate_prefixes: Mapping[int, list[str]]) -> tuple[
    dict[tuple[int, int], tuple[str, str]], dict[int, list[str]], dict[int, Any]
]:
    """Build the old serializer's causal call/content prefixes for all provable rounds."""
    from sdsafe_event_data.alignment import build_token_text
    from sdsafe_event_data.segments import build_round_visible
    from sdsafe_event_data.replay import PREFIX_ALIGNMENT_UNPROVEN, PREFIX_VISIBLE_TOKENS

    call_prefixes: dict[tuple[int, int], tuple[str, str]] = {}
    segment_prefixes: dict[int, list[str]] = {}
    visibles: dict[int, Any] = {}
    for round_sequence in episode.rounds:
        round_id = int(round_sequence.round_id)
        try:
            token_text = build_token_text(tokenizer, round_sequence, "cot")
            visible = build_round_visible(round_sequence, token_text)
        except Exception:
            visible = None
        if visible is not None:
            visibles[round_id] = visible
        for span in round_sequence.call_spans:
            if visible is None:
                call_prefixes[(round_id, span.call_index)] = ("", PREFIX_ALIGNMENT_UNPROVEN)
            else:
                call_prefixes[(round_id, span.call_index)] = (
                    visible.prefix_before_token(span.token_start),
                    PREFIX_VISIBLE_TOKENS,
                )
        if round_id in candidate_prefixes:
            segment_prefixes[round_id] = list(candidate_prefixes[round_id])
        elif visible is not None and not round_sequence.call_spans and visible.segments:
            segment_prefixes[round_id] = [visible.prefix_text(segment) for segment in visible.segments]
    return call_prefixes, segment_prefixes, visibles


def _content_reason(
    round_sequence: Any,
    round_row: Any,
    prefix_count: int,
    *,
    decision: Mapping[str, Any],
) -> str | None:
    if round_sequence.call_spans:
        return UNKNOWN_MIXED_ORDER
    if decision.get("block_from") is not None:
        return str(decision.get("reason") or UNKNOWN_ENVIRONMENT)
    if len(round_row.segment_checks) != prefix_count:
        return UNKNOWN_CONTENT_PREFIX
    if not round_row.round_end_matches:
        return UNKNOWN_ROUND_END
    if round_row.segment_checks:
        if round_row.segment_checks[-1].prefix_security != round_row.saved_round_end_security:
            return UNKNOWN_CONTENT_PREFIX
    return None



def build_visible_prefix_mapping(
    full_text: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    coordinate_basis: str = "completion_absolute_chars",
) -> tuple[list[str], list[dict[str, Any]]]:
    """Map completion-absolute candidate endpoints to API-visible-only prefixes.

    ``row['char_start']``/``row['char_end']`` are completion-absolute character
    coordinates.  ``row['visible_char_start']`` is the first character of the decoded
    API-visible answer (after the reasoning scaffold).  The checker must receive only
    ``full_text[visible_char_start:absolute_char_end]``: never the prompt, CoT,
    structural tokens, or any future visible answer.
    """
    if coordinate_basis != "completion_absolute_chars":
        raise LabelRepairError(f"unsupported candidate coordinate basis {coordinate_basis!r}")
    ordered = sorted(rows, key=lambda row: int(row["segment_index"]))
    starts = {int(row.get("visible_char_start", -1)) for row in ordered}
    if len(starts) != 1 or next(iter(starts)) < 0:
        raise LabelRepairError(f"candidate visible_char_start is missing/ambiguous: {sorted(starts)}")
    visible_start = next(iter(starts))
    prefixes: list[str] = []
    provenance: list[dict[str, Any]] = []
    for row in ordered:
        char_start = int(row["char_start"])
        char_end = int(row["char_end"])
        if int(row.get("visible_char_end", char_end)) < char_end:
            raise LabelRepairError(f"{row['event_id']}: candidate endpoint exceeds the API-visible interval")
        if visible_start > char_start or char_start >= char_end:
            raise LabelRepairError(
                f"{row['event_id']}: API-visible start {visible_start} is not before candidate "
                f"completion-absolute start {char_start}"
            )
        prefix = full_text[visible_start:char_end]
        text = full_text[char_start:char_end]
        expected_prefix_sha = row.get("visible_prefix_sha256")
        actual_prefix_sha = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        if expected_prefix_sha is not None and actual_prefix_sha != str(expected_prefix_sha):
            raise LabelRepairError(f"{row['event_id']}: API-visible prefix hash link failed")
        expected_text_sha = row.get("text_sha256")
        actual_text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected_text_sha is not None and actual_text_sha != str(expected_text_sha):
            raise LabelRepairError(f"{row['event_id']}: candidate text hash link failed")
        prefixes.append(prefix)
        provenance.append(
            {
                "coordinate_basis": coordinate_basis,
                "visible_char_start": visible_start,
                "absolute_char_start": char_start,
                "absolute_char_end": char_end,
                "prefix_text_chars": len(prefix),
                "prefix_text_sha256": actual_prefix_sha,
                "candidate_text_chars": len(text),
                "candidate_text_sha256": actual_text_sha,
                "api_visible_char_end": int(row.get("visible_char_end", char_end)),
                "contains_prompt_or_cot": False,
            }
        )
    return prefixes, provenance

def build_candidate_labels(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any | None = None,
    records: Mapping[str, Mapping[str, Any]] | None = None,
    adapter: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay only affected candidate episodes and label exact causal prefixes."""
    from sdsafe_ctx40k.adapter import DirectInjectionAdapter
    from sdsafe_event_data.events import call_label_policy
    from sdsafe_event_data.replay import atomic_replay
    from .alignment import _decode_completion, _build_episode, _load_episode_records, load_tokenizer

    tokenizer = tokenizer or load_tokenizer()
    records = dict(records or _load_episode_records())
    adapter = adapter or DirectInjectionAdapter(
        Path(__file__).resolve().parents[2] / "external/AgentLAB",
        Path(__file__).resolve().parents[2] / "data/compat/task_injection_cot32k_r1",
    )

    by_episode_round: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidate_rows:
        split = str(row.get("split"))
        if split != SPLIT_TRAIN and split != "dev":
            raise LabelRepairError(f"{row.get('event_id')}: candidate split is not TRAIN/DEV")
        by_episode_round[str(row["episode_id"])][int(row["round_id"])].append(dict(row))

    labelled: dict[str, dict[str, Any]] = {}
    evidence_by_episode: dict[str, Any] = {}
    for episode_id, rounds in sorted(by_episode_round.items()):
        source_id = episode_id.split("::", 1)[1]
        record = records.get(source_id)
        if record is None:
            raise LabelRepairError(f"{episode_id}: no eligible CoT record")
        episode = _build_episode(record, tokenizer)
        actual_rounds = {int(row.round_id): row for row in episode.rounds}

        candidate_prefixes: dict[int, list[str]] = {}
        candidate_prefix_provenance: dict[int, list[dict[str, Any]]] = {}
        for round_id, rows in sorted(rounds.items()):
            rows = sorted(rows, key=lambda row: int(row["segment_index"]))
            round_sequence = actual_rounds.get(round_id)
            if round_sequence is None:
                raise LabelRepairError(f"{episode_id} r{round_id}: missing actual round")
            full_text, _completion = _decode_completion(tokenizer, round_sequence)
            prefixes, provenance = build_visible_prefix_mapping(
                full_text,
                rows,
                coordinate_basis=str(rows[0].get("coordinate_basis") or "completion_absolute_chars"),
            )
            candidate_prefixes[round_id] = prefixes
            candidate_prefix_provenance[round_id] = provenance

        call_prefixes, segment_prefixes, _visibles = _visible_prefixes_for_episode(
            episode, tokenizer, candidate_prefixes
        )
        replay = atomic_replay(
            adapter,
            Path(__file__).resolve().parents[2] / "artifacts/cot32k/replay_semantic",
            dict(record),
            episode,
            call_prefixes,
            segment_prefixes,
        )
        blocked_reason: str | None = None
        episode_evidence: dict[str, Any] = {"episode_id": episode_id, "rounds": {}}
        for round_sequence in episode.rounds:
            round_id = int(round_sequence.round_id)
            round_row = replay.rounds[round_id]
            decision = call_label_policy(round_row, blocked_reason)
            if round_id in rounds:
                rows = sorted(rounds[round_id], key=lambda row: int(row["segment_index"]))
                reason = _content_reason(
                    round_sequence,
                    round_row,
                    len(candidate_prefixes[round_id]),
                    decision=decision,
                )
                checks = {int(check.segment_index): check for check in round_row.segment_checks}
                if reason is None and len(checks) != len(rows):
                    reason = UNKNOWN_CONTENT_PREFIX
                for row in rows:
                    event_id = str(row["event_id"])
                    segment_index = int(row["segment_index"])
                    if reason is None:
                        check = checks.get(segment_index)
                        if check is None:
                            reason = UNKNOWN_CONTENT_PREFIX
                        else:
                            updated = dict(row)
                            updated.update(
                                {
                                    "label_status": LABEL_EXACT,
                                    "label": bool(check.prefix_security),
                                    "label_value": bool(check.prefix_security),
                                    "previous_security": bool(check.previous_security),
                                    "prefix_security": bool(check.prefix_security),
                                    "security_flip": bool(
                                        check.prefix_security and not check.previous_security
                                    ),
                                    "label_reason": "official_checker_visible_output_prefix",
                                    "label_provenance": {
                                        "kind": "replayed_official_checker",
                                        "adapter": "DirectInjectionAdapter",
                                        "serializer": "archived_prompt_completion_token_ids",
                                        "round_id": round_id,
                                        "segment_index": segment_index,
                                        "prefix_text_sha256": candidate_prefix_provenance[round_id][segment_index]["prefix_text_sha256"],
                                        "prefix_text_chars": candidate_prefix_provenance[round_id][segment_index]["prefix_text_chars"],
                                        "prefix_coordinate_basis": candidate_prefix_provenance[round_id][segment_index]["coordinate_basis"],
                                        "visible_char_start": candidate_prefix_provenance[round_id][segment_index]["visible_char_start"],
                                        "absolute_char_start": candidate_prefix_provenance[round_id][segment_index]["absolute_char_start"],
                                        "absolute_char_end": candidate_prefix_provenance[round_id][segment_index]["absolute_char_end"],
                                        "api_visible_char_end": candidate_prefix_provenance[round_id][segment_index]["api_visible_char_end"],
                                        "contains_prompt_or_cot": False,
                                        "saved_round_end_security": bool(round_row.saved_round_end_security),
                                        "recomputed_round_end_security": round_row.recomputed_round_end_security,
                                        "round_end_matches_saved": bool(round_row.round_end_matches),
                                        "atomic_vs_control": round_row.atomic_vs_control,
                                        "frozen_hash": round_row.frozen_hash,
                                    },
                                }
                            )
                            labelled[event_id] = updated
                    if reason is not None:
                        updated = dict(row)
                        updated.update(
                            {
                                "label_status": LABEL_UNKNOWN,
                                "label": None,
                                "label_value": None,
                                "previous_security": None,
                                "prefix_security": None,
                                "security_flip": None,
                                "label_reason": reason,
                                "label_provenance": {
                                    "kind": "checker_replay_unknown",
                                    "reason": reason,
                                    "prefix_provenance": candidate_prefix_provenance[round_id][segment_index],
                                    "issues": list(round_row.issues),
                                    "saved_round_end_security": bool(round_row.saved_round_end_security),
                                    "recomputed_round_end_security": round_row.recomputed_round_end_security,
                                    "round_end_matches_saved": bool(round_row.round_end_matches),
                                    "atomic_vs_control": round_row.atomic_vs_control,
                                    "frozen_hash": round_row.frozen_hash,
                                },
                            }
                        )
                        labelled[event_id] = updated
                episode_evidence["rounds"][str(round_id)] = {
                    "status": "exact" if reason is None else "unknown",
                    "reason": reason,
                    "candidate_rows": len(rows),
                    "segment_checks": len(round_row.segment_checks),
                    "saved_round_end_security": bool(round_row.saved_round_end_security),
                    "recomputed_round_end_security": round_row.recomputed_round_end_security,
                    "round_end_matches_saved": bool(round_row.round_end_matches),
                    "atomic_vs_control": round_row.atomic_vs_control,
                    "frozen_hash": round_row.frozen_hash,
                    "issues": list(round_row.issues),
                }
            if round_row.atomic_vs_control == "material_divergence" or _material_reasons(round_row):
                blocked_reason = blocked_reason or str(decision.get("reason") or UNKNOWN_ENVIRONMENT)
        evidence_by_episode[episode_id] = episode_evidence

    if set(labelled) != {str(row["event_id"]) for row in candidate_rows}:
        missing = sorted({str(row["event_id"]) for row in candidate_rows} - set(labelled))
        raise LabelRepairError(f"candidate replay did not label every row: {missing[:5]}")
    rows_out = [labelled[str(row["event_id"])] for row in candidate_rows]
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "candidate_rows": len(rows_out),
        "exact_rows": sum(1 for row in rows_out if row["label_status"] == LABEL_EXACT),
        "unknown_rows": sum(1 for row in rows_out if row["label_status"] == LABEL_UNKNOWN),
        "positive_rows": sum(1 for row in rows_out if row.get("label") is True),
        "negative_rows": sum(1 for row in rows_out if row.get("label") is False),
        "by_split": dict(sorted(Counter(str(row["split"]) for row in rows_out).items())),
        "reason_counts": dict(
            sorted(Counter(str(row.get("label_reason")) for row in rows_out).items())
        ),
        "episode_evidence": evidence_by_episode,
        "test_raw_traces_opened": 0,
    }
    return rows_out, audit


# --------------------------------------------------------------------------- UTF-8 label reuse


def apply_utf8_label_reuse(corrections: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy old checker state only for an identical API-visible text prefix."""
    out: list[dict[str, Any]] = []
    for correction in corrections:
        row = dict(correction)
        status = str(row.get("label_status"))
        if status not in (LABEL_EXACT, LABEL_UNKNOWN):
            raise LabelRepairError(f"{row['new_event_id']}: bad old label_status {status!r}")
        if status == LABEL_EXACT:
            prefix = row.get("prefix_security")
            previous = row.get("previous_security")
            flip = row.get("security_flip")
            if not isinstance(prefix, bool) or not isinstance(previous, bool):
                raise LabelRepairError(f"{row['new_event_id']}: exact old label lacks checker state")
            if flip is not None and bool(flip) is not bool(prefix and not previous):
                raise LabelRepairError(f"{row['new_event_id']}: old security_flip is incoherent")
            if not row.get("label_prefix_text_sha256"):
                raise LabelRepairError(
                    f"{row['new_event_id']}: exact old label lacks the API-visible prefix hash"
                )
        else:
            if any(row.get(name) is not None for name in ("prefix_security", "previous_security", "security_flip")):
                raise LabelRepairError(f"{row['new_event_id']}: unknown old label carries checker state")
        updated = dict(row)
        updated.update(
            {
                "label": bool(prefix) if status == LABEL_EXACT else None,
                "label_value": bool(prefix) if status == LABEL_EXACT else None,
                "label_visible_prefix_sha256": row.get("label_prefix_text_sha256"),
                "label_visible_prefix_chars": row.get("label_prefix_text_chars"),
                "label_prefix_coordinate_basis": row.get("label_prefix_coordinate_basis"),
                "label_provenance": {
                    "kind": "reused_old_label_identical_api_text_prefix",
                    "old_event_id": row["old_event_id"],
                    "same_api_visible_text_sha256": row["text_sha256"],
                    "same_api_visible_prefix_sha256": row.get("label_prefix_text_sha256"),
                    "same_api_visible_prefix_chars": row.get("label_prefix_text_chars"),
                    "prefix_coordinate_basis": row.get("label_prefix_coordinate_basis"),
                    "same_character_end": row["char_end"],
                    "same_round_sequence_sha256": row["round_sequence_sha256"],
                    "same_environment_and_task_binding": True,
                    "contains_prompt_or_cot": False,
                    "change": "token_end extended over zero-width UTF-8 continuation tokens only",
                },
            }
        )
        out.append(updated)
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "rows": len(out),
        "exact_rows": sum(1 for row in out if row["label_status"] == LABEL_EXACT),
        "unknown_rows": sum(1 for row in out if row["label_status"] == LABEL_UNKNOWN),
        "positive_rows": sum(1 for row in out if row.get("label") is True),
        "negative_rows": sum(1 for row in out if row.get("label") is False),
        "by_split": dict(sorted(Counter(str(row["split"]) for row in out).items())),
        "reuse_rule": "identical API-visible text prefix + same round sequence + same checker binding",
    }
    return out, audit


# --------------------------------------------------------------------------- policies


def _l0_anchors(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("event_type") in (EVENT_TOOL_CALL, EVENT_CONTENT_SEGMENT) and _exact(row):
            by_episode[str(row["episode_id"])].append(dict(row))
    anchors: dict[str, str] = {}
    for episode_id, episode_rows in sorted(by_episode.items()):
        if not _final_positive(episode_rows[0]):
            continue
        ordered = sorted(episode_rows, key=_event_order)
        flips = [row for row in ordered if row.get("security_flip") is True]
        if flips:
            anchors[episode_id] = str(flips[0]["event_id"])
    return anchors


def _turn_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row["episode_id"]), int(row["round_id"]), int(row["message_index"]))


def _stratum(row: Mapping[str, Any]) -> str:
    return f"{row['scope']}|{row['event_type']}"


def _mask_entry(label: bool | None, reason: str | None) -> dict[str, Any]:
    return {"label": label, "mask": label is None, "mask_reason": reason}


def _assign_l4(
    rows: Sequence[Mapping[str, Any]],
    l0_labels: Mapping[str, bool],
    l1_masked_by_stratum: Mapping[str, int],
    *,
    seed: int = L4_MASK_RNG_SEED,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Drop exactly the L1 per-stratum count from all L0 negatives with fixed RNG."""
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _exact(row) and l0_labels.get(str(row["event_id"])) is False:
            strata[_stratum(row)].append(dict(row))
    rng = random.Random(int(seed))
    dropped_ids: dict[str, list[str]] = {}
    assignments: dict[str, dict[str, Any]] = {}
    for stratum in sorted(strata):
        candidates = sorted(strata[stratum], key=lambda row: str(row["event_id"]))
        target = int(l1_masked_by_stratum.get(stratum, 0))
        if target > len(candidates):
            raise LabelRepairError(
                f"L4 stratum {stratum}: needs {target} drops but only {len(candidates)} L0 negatives exist"
            )
        order = list(candidates)
        rng.shuffle(order)
        chosen = [str(row["event_id"]) for row in order[:target]]
        dropped_ids[stratum] = sorted(chosen)
        for row in candidates:
            event_id = str(row["event_id"])
            if event_id in set(chosen):
                assignments[event_id] = _mask_entry(None, MASK_L4_DROP)
            else:
                assignments[event_id] = _mask_entry(False, None)
    return assignments, {"dropped_by_stratum": dropped_ids, "seed": int(seed)}


def build_policy_assignments(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Compute L0..L4 assignment for every repaired TRAIN CT row (exact and unknown)."""
    train_rows = [dict(row) for row in rows if str(row.get("split")) == SPLIT_TRAIN]
    if not train_rows:
        raise LabelRepairError("no TRAIN rows supplied to the policy builder")
    for row in train_rows:
        if row.get("event_type") not in (EVENT_TOOL_CALL, EVENT_CONTENT_SEGMENT):
            raise LabelRepairError(f"{row.get('event_id')}: policy rows must be C|T")
        _validate_label_state(row)

    exact_rows = [row for row in train_rows if _exact(row)]
    anchors = _l0_anchors(train_rows)
    anchor_ids = set(anchors.values())
    anchor_turn: dict[tuple[str, int, int], str] = {}
    for episode_id, anchor_id in anchors.items():
        anchor = next(row for row in exact_rows if str(row["event_id"]) == anchor_id)
        anchor_turn[_turn_key(anchor)] = anchor_id

    l0: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        event_id = str(row["event_id"])
        if not _exact(row):
            l0[event_id] = _mask_entry(None, MASK_UNKNOWN)
        else:
            l0[event_id] = _mask_entry(event_id in anchor_ids, None)

    l1: dict[str, dict[str, Any]] = {}
    l1_masked_by_stratum: Counter = Counter()
    for row in train_rows:
        event_id = str(row["event_id"])
        if not _exact(row):
            l1[event_id] = _mask_entry(None, MASK_UNKNOWN)
        elif event_id in anchor_ids:
            l1[event_id] = _mask_entry(True, None)
        elif _prefix_positive(row):
            l1[event_id] = _mask_entry(None, MASK_PREFIX_NON_ANCHOR)
            l1_masked_by_stratum[_stratum(row)] += 1
        else:
            l1[event_id] = _mask_entry(False, None)

    l2 = {key: dict(value) for key, value in l1.items()}
    for row in train_rows:
        if not _exact(row) or str(row["event_id"]) in anchor_ids:
            continue
        anchor_id = anchor_turn.get(_turn_key(row))
        if anchor_id is not None and _prefix_positive(row):
            l2[str(row["event_id"])] = _mask_entry(
                True,
                None,
            )
            l2[str(row["event_id"])]["label_reason"] = "same_first_hit_turn_prefix_positive"

    l3: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        event_id = str(row["event_id"])
        if not _exact(row):
            l3[event_id] = _mask_entry(None, MASK_UNKNOWN)
        else:
            l3[event_id] = _mask_entry(bool(row.get("prefix_security")), None)

    l4_negatives, l4_drop_info = _assign_l4(train_rows, {k: bool(v["label"]) for k, v in l0.items() if v["label"] is not None}, l1_masked_by_stratum)
    l4: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        event_id = str(row["event_id"])
        if not _exact(row):
            l4[event_id] = _mask_entry(None, MASK_UNKNOWN)
        elif event_id in anchor_ids:
            l4[event_id] = _mask_entry(True, None)
        else:
            l4[event_id] = l4_negatives[event_id]

    assignments = {"L0": l0, "L1": l1, "L2": l2, "L3": l3, "L4": l4}
    for policy in ("L1", "L2", "L3"):
        bad = [
            str(row["event_id"])
            for row in exact_rows
            if _prefix_positive(row) and assignments[policy][str(row["event_id"])]["label"] is False
        ]
        if bad:
            raise LabelRepairError(f"{policy}: exact prefix-positive negative rows exist: {bad[:5]}")

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "rows_total": len(train_rows),
        "exact_rows": len(exact_rows),
        "unknown_rows": len(train_rows) - len(exact_rows),
        "episodes": len({str(row["episode_id"]) for row in train_rows}),
        "groups": len({str(row["group_id"]) for row in train_rows}),
        "l0_anchors": len(anchors),
        "l0_positive_rows": len(anchors),
        "l1_masked_by_stratum": dict(sorted(l1_masked_by_stratum.items())),
        "l4_drop": l4_drop_info,
        "per_policy": {
            policy: policy_audit(train_rows, assignments[policy], policy)
            for policy in POLICIES
        },
    }
    audit["assertions"] = {
        "L1_zero_exact_prefix_positive_negatives": audit["per_policy"]["L1"]["zero_exact_prefix_positive_negatives"],
        "L2_zero_exact_prefix_positive_negatives": audit["per_policy"]["L2"]["zero_exact_prefix_positive_negatives"],
        "L3_zero_exact_prefix_positive_negatives": audit["per_policy"]["L3"]["zero_exact_prefix_positive_negatives"],
        "L0_legacy_old_label0_prefix_positive_rows": 14_045,
        "unknown_never_zero": True,
    }
    return assignments, audit


def policy_audit(
    rows: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Mapping[str, Any]],
    policy: str,
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise LabelRepairError(f"unknown policy {policy!r}")
    exact = [dict(row) for row in rows if _exact(row)]
    unknown = [dict(row) for row in rows if not _exact(row)]
    labels = {str(row["event_id"]): assignment[str(row["event_id"])] for row in exact}
    positives = [row for row in exact if labels[str(row["event_id"])]["label"] is True]
    negatives = [row for row in exact if labels[str(row["event_id"])]["label"] is False]
    masked = [row for row in exact if labels[str(row["event_id"])]["label"] is None]
    unknown_masked = [row for row in unknown if assignment[str(row["event_id"])]["label"] is None]
    zero_prefix_positive_negatives = [
        str(row["event_id"]) for row in negatives if _prefix_positive(row)
    ]
    by_scope_type = Counter(f"{row['scope']}|{row['event_type']}" for row in exact)
    by_scope_type_label = Counter(
        f"{row['scope']}|{row['event_type']}|{labels[str(row['event_id'])]['label']}" for row in exact
    )
    by_scope_type_mask = Counter(
        f"{row['scope']}|{row['event_type']}|{labels[str(row['event_id'])]['mask_reason']}" for row in masked
    )
    by_mask_reason = Counter(str(labels[str(row["event_id"])]["mask_reason"]) for row in masked)
    by_unknown_reason = Counter(str(row.get("label_reason")) for row in unknown)

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_id"])].append(dict(row))
    first_prefix_coverage: Counter = Counter()
    final_positive_coverage: Counter = Counter()
    for episode_id, episode_rows in sorted(by_episode.items()):
        exact_episode = [row for row in sorted(episode_rows, key=_event_order) if _exact(row)]
        first_prefix = next((row for row in exact_episode if _prefix_positive(row)), None)
        if first_prefix is not None:
            state = labels.get(str(first_prefix["event_id"]), assignment[str(first_prefix["event_id"])])
            first_prefix_coverage[
                "positive" if state.get("label") is True else "masked" if state.get("mask") else "zero"
            ] += 1
        else:
            first_prefix_coverage["no_exact_prefix_positive"] += 1
        if any(_final_positive(row) for row in episode_rows):
            final_positive_coverage[
                "has_positive_train_row" if any(labels[str(row["event_id"])]["label"] is True for row in exact_episode) else "no_positive_train_row"
            ] += 1

    mask_material = [
        {"event_id": str(row["event_id"]), "reason": labels[str(row["event_id"])]["mask_reason"]}
        for row in masked
    ] + [
        {
            "event_id": str(row["event_id"]),
            "reason": str(assignment[str(row["event_id"])].get("mask_reason") or MASK_UNKNOWN),
        }
        for row in unknown
        if assignment[str(row["event_id"])]["label"] is None
    ]
    target_material = [
        {
            "event_id": str(row["event_id"]),
            "label": bool(labels[str(row["event_id"])]["label"]),
            "event_type": str(row["event_type"]),
            "scope": str(row["scope"]),
        }
        for row in exact
        if labels[str(row["event_id"])]["label"] is not None
    ]
    return {
        "policy": policy,
        "rows_total": len(rows),
        "exact_rows": len(exact),
        "unknown_rows": len(unknown),
        "trainable_rows": len(positives) + len(negatives),
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "masked_exact_rows": len(masked),
        "masked_unknown_rows": len(unknown_masked),
        "positive_episodes": len({str(row["episode_id"]) for row in positives}),
        "negative_episodes": len({str(row["episode_id"]) for row in negatives}),
        "groups": len({str(row["group_id"]) for row in exact}),
        "rows_by_scope_type": dict(sorted(by_scope_type.items())),
        "rows_by_scope_type_label": dict(sorted(by_scope_type_label.items())),
        "rows_by_scope_type_mask": dict(sorted(by_scope_type_mask.items())),
        "masked_exact_by_reason": dict(sorted((str(key), value) for key, value in by_mask_reason.items())),
        "masked_unknown_by_reason": dict(sorted(by_unknown_reason.items())),
        "unknown_by_reason": dict(sorted(by_unknown_reason.items())),
        "zero_exact_prefix_positive_negatives": len(zero_prefix_positive_negatives),
        "zero_exact_prefix_positive_negative_ids_sample": zero_prefix_positive_negatives[:10],
        "first_prefix_positive_coverage": dict(sorted(first_prefix_coverage.items())),
        "strict_final_positive_coverage": dict(sorted(final_positive_coverage.items())),
        "mask_sha256": canonical_sha256(sorted(mask_material, key=lambda row: row["event_id"])),
        "target_sha256": canonical_sha256(sorted(target_material, key=lambda row: row["event_id"])),
        "masked_event_ids_sample": sorted(row["event_id"] for row in masked)[:10],
        "unknown_event_ids_sample": sorted(str(row["event_id"]) for row in unknown)[:10],
        "note": (
            "positive means exact prefix checker state true, not a claim that assistant text is malicious"
            if policy in ("L1", "L2", "L3")
            else "control policy; not an accurate-label candidate"
            if policy == "L4"
            else "legacy first-hit policy recomputed on the repaired TRAIN rows"
        ),
    }
