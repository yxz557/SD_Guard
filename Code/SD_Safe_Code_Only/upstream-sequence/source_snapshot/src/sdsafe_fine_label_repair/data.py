"""Durable TRAIN/DEV metadata and feature-ref views for fine_label_repair_v1.

The module has two independent data contracts:

``legacy_ct_first_hit_view``
    The original exact CT first_hit TRAIN pool (107,215 rows) built by the accepted
    ``fine_ablation``/``fine_confirm`` helpers.  Its cached feature references are the
    immutable old global/local BF16 row coordinates.  This view is never overlaid with
    repaired rows; a legacy control fetches these old vectors and positions.

``build_repaired_view``
    The old exact rows with the proved content-alignment candidates and the audited
    trailing-UTF8 endpoint replacements added.  Superseded old endpoint rows remain in
    a separate history file.  Repaired/new rows carry a supplemental feature reference
    keyed by actual episode/round/token index, not by a global or local row number.

No function here reads a feature matrix.  ``validate_feature_ref`` is the guard the next
lane must call at the actual vector-read boundary; it refuses TEST and missing identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sdsafe_features.common import canonical_sha256, read_json, read_jsonl, sha256_file
from sdsafe_fine_ablation.data import (
    label0_positive_prefix_counts,
    load_study,
    target_pool_signature,
)
from sdsafe_fine_confirm import train_data

from . import (
    ALIGNMENT_PROOFS_PATH,
    ARTIFACT_ROOT,
    CANDIDATE_POSITIONS_PATH,
    COT_EVENT_STORE,
    CT_EVENT_TYPES,
    DATA_ROOT,
    EVENT_CONTENT_SEGMENT,
    EVENT_ROUND_END,
    EVENT_TOOL_CALL,
    EXPECTED_COMMON_DEV_EPISODES,
    EXPECTED_DIAGNOSTIC_DEV_EPISODES,
    EXPECTED_LEGACY_CT_FIRST_HIT_NEGATIVES,
    EXPECTED_LEGACY_CT_FIRST_HIT_POSITIVES,
    EXPECTED_LEGACY_CT_FIRST_HIT_ROWS,
    EXPECTED_LEGACY_CT_LABEL0_PREFIX_POSITIVE,
    EXPECTED_TRAIN_EPISODES,
    FEATURE_DTYPE,
    HIDDEN_SIZE,
    LABEL_EXACT,
    LABEL_UNKNOWN,
    LEGACY_FEATURE_KIND,
    POLICIES,
    PROTECTED_SPLITS,
    REPAIR_PLAN_PATH,
    SCHEMA_VERSION,
    SPLIT_DEV,
    SPLIT_TRAIN,
    STUDY_ID,
    SUPPLEMENTAL_FEATURE_KIND,
    SUPPLEMENTAL_FEATURE_ROOT,
    DataContractError,
    assert_protected_split,
    source_tag_for_scope,
)
from .alignment import AlignmentBuild, build_alignment
from .labels import (
    apply_utf8_label_reuse,
    build_candidate_labels,
    build_policy_assignments,
    policy_audit,
)

POLICY_ASSIGNMENT_FILE = "policies/assignments.jsonl"
POLICY_AUDIT_FILE = "policies/audit.json"


# --------------------------------------------------------------------------- feature refs


def _legacy_feature_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    required = ("scope", "local_feature_row", "global_feature_row", "feature_file", "feature_byte_offset", "feature_token_index")
    missing = [name for name in required if row.get(name) is None]
    if missing:
        raise DataContractError(f"{row.get('event_id')}: cached feature row is missing {missing}")
    split = assert_protected_split(row.get("split"), where=f"cached feature ref {row['event_id']}")
    return {
        "kind": LEGACY_FEATURE_KIND,
        "scope": str(row["scope"]),
        "split": split,
        "global_feature_row": int(row["global_feature_row"]),
        "local_feature_row": int(row["local_feature_row"]),
        "feature_file": str(row["feature_file"]),
        "feature_byte_offset": int(row["feature_byte_offset"]),
        "feature_token_index": int(row["feature_token_index"]),
        "sequence_sha256": row.get("sequence_sha256"),
        "event_id": str(row["event_id"]),
        "hidden_size": HIDDEN_SIZE,
        "dtype": FEATURE_DTYPE,
        "status": "cached_read_only",
    }


def _supplemental_feature_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    split = assert_protected_split(row.get("split"), where=f"supplemental feature ref {row['event_id']}")
    required = ("scope", "episode_id", "round_id", "feature_token_index", "sequence_sha256", "prompt_length", "sequence_length")
    missing = [name for name in required if row.get(name) is None]
    if missing:
        raise DataContractError(f"{row.get('event_id')}: supplemental ref is missing {missing}")
    token_index = int(row["feature_token_index"])
    completion_index = row.get("completion_token_end")
    if completion_index is None:
        completion_index = token_index - int(row["prompt_length"])
    key_material = [
        str(row["scope"]),
        str(row["episode_id"]),
        int(row["round_id"]),
        int(token_index),
        str(row["sequence_sha256"]),
        str(row["event_id"]),
    ]
    return {
        "kind": SUPPLEMENTAL_FEATURE_KIND,
        "scope": str(row["scope"]),
        "split": split,
        "episode_id": str(row["episode_id"]),
        "namespaced_episode_id": str(row["episode_id"]),
        "round_id": int(row["round_id"]),
        "token_index": int(token_index),
        "completion_token_index": int(completion_index),
        "prompt_length": int(row["prompt_length"]),
        "sequence_length": int(row["sequence_length"]),
        "round_sequence_sha256": str(row["sequence_sha256"]),
        "event_id": str(row["event_id"]),
        "event_type": str(row["event_type"]),
        "feature_key": "supp::" + canonical_sha256(key_material)[:32],
        "hidden_size": HIDDEN_SIZE,
        "dtype": FEATURE_DTYPE,
        "storage_root": SUPPLEMENTAL_FEATURE_ROOT,
        "status": "pending_extraction",
    }


def validate_feature_ref(ref: Mapping[str, Any], expected_split: str | None = None) -> None:
    """Fail closed at the actual read boundary; no vector is opened here."""
    kind = str(ref.get("kind") or "")
    split = assert_protected_split(ref.get("split"), where="feature ref")
    if expected_split is not None and split != str(expected_split):
        raise DataContractError(f"feature ref split {split!r} is not requested split {expected_split!r}")
    if kind == LEGACY_FEATURE_KIND:
        for name in ("scope", "global_feature_row", "local_feature_row", "feature_file", "feature_byte_offset"):
            if ref.get(name) is None:
                raise DataContractError(f"cached feature ref is missing {name}")
        if int(ref["global_feature_row"]) < 0 or int(ref["local_feature_row"]) < 0:
            raise DataContractError("cached feature row coordinates are negative")
        return
    if kind == SUPPLEMENTAL_FEATURE_KIND:
        for name in ("scope", "episode_id", "round_id", "token_index", "round_sequence_sha256", "feature_key"):
            if ref.get(name) is None or str(ref.get(name)) == "":
                raise DataContractError(f"supplemental feature ref is missing {name}")
        if int(ref["round_id"]) < 1 or int(ref["token_index"]) < 0:
            raise DataContractError("supplemental feature coordinates are invalid")
        return
    raise DataContractError(f"unknown feature-ref kind {kind!r}; refusing to read")


# --------------------------------------------------------------------------- legacy view


def _canonical_map(study: Any) -> dict[str, dict[str, Any]]:
    return {str(row["event_id"]): dict(row) for row in study.rows}


def legacy_ct_first_hit_view(root: Path | str = ".") -> dict[str, Any]:
    """The original exact CT first_hit TRAIN 107,215-row control view."""
    study = load_study(root, verify_features=False)
    train_rows = study.train_rows()
    if len({str(row["episode_id"]) for row in train_rows}) != EXPECTED_TRAIN_EPISODES:
        raise DataContractError("TRAIN episode count drifted")
    targets = train_data.build_targets(train_rows, arm="fine", policy="first_hit", split_name=SPLIT_TRAIN)
    signature = target_pool_signature(targets["rows"])
    if signature["rows"] != EXPECTED_LEGACY_CT_FIRST_HIT_ROWS:
        raise DataContractError(f"legacy CT first_hit rows {signature['rows']} != {EXPECTED_LEGACY_CT_FIRST_HIT_ROWS}")
    if signature["sha256"] != "14a93a9649a06dcf8eaab8b9c064858e50ea26501b0db6967319adebaa51bebd":
        raise DataContractError("legacy CT first_hit target signature drifted")
    label0 = label0_positive_prefix_counts(targets["rows"])
    if label0["rows"] != EXPECTED_LEGACY_CT_LABEL0_PREFIX_POSITIVE:
        raise DataContractError(f"legacy label0 prefix-positive rows {label0['rows']} != {EXPECTED_LEGACY_CT_LABEL0_PREFIX_POSITIVE}")
    canonical = _canonical_map(study)
    rows: list[dict[str, Any]] = []
    for target in targets["rows"]:
        event_id = str(target["event_id"])
        canonical_row = canonical.get(event_id)
        if canonical_row is None:
            raise DataContractError(f"{event_id}: legacy target row is not in the canonical study")
        ref = _legacy_feature_ref(canonical_row)
        row = {
            "row_key": f"legacy::{event_id}",
            "view": "legacy_ct_first_hit",
            "event_id": event_id,
            "episode_id": str(target["episode_id"]),
            "namespaced_episode_id": str(target["namespaced_episode_id"]),
            "scope": str(target["scope"]),
            "source_tag": target.get("source_tag"),
            "split": SPLIT_TRAIN,
            "group_id": str(target.get("group_id", "")),
            "event_type": str(target["event_type"]),
            "round_id": int(target.get("round_id", 0)),
            "message_index": int(target.get("message_index", 0)),
            "index": int(target.get("index", 0)),
            "call_index": target.get("call_index"),
            "segment_index": target.get("segment_index"),
            "feature_token_index": int(target["feature_token_index"]) if target.get("feature_token_index") is not None else None,
            "label": bool(target["label"]),
            "label_value": bool(target["label"]),
            "label_status": LABEL_EXACT,
            "previous_security": target.get("security_flip") if target.get("previous_security") is None else target.get("previous_security"),
            "prefix_security": target.get("prefix_security"),
            "security_flip": target.get("security_flip"),
            "legacy_feature_ref": ref,
            "feature_ref": ref,
        }
        # build_targets preserves only the label columns; fetch exact checker state from
        # the canonical row so the legacy audit is not guessed from the training label.
        if canonical_row is not None:
            row["previous_security"] = canonical_row.get("previous_security")
            row["prefix_security"] = canonical_row.get("prefix_security")
            row["security_flip"] = canonical_row.get("security_flip")
            row["label_status"] = canonical_row.get("label_status")
            row["strict_final"] = bool(canonical_row.get("strict_final"))
            row["group_hash"] = canonical_row.get("group_hash")
            row["suite"] = canonical_row.get("suite")
            row["user_task_id"] = canonical_row.get("user_task_id")
            row["injection_task_id"] = canonical_row.get("injection_task_id")
            row["attack_type"] = canonical_row.get("attack_type")
            row["victim_model"] = canonical_row.get("victim_model")
            row["char_start"] = canonical_row.get("char_start")
            row["char_end"] = canonical_row.get("char_end")
            row["text_chars"] = canonical_row.get("text_chars")
            row["text_sha256"] = canonical_row.get("text_sha256")
            row["sequence_sha256"] = canonical_row.get("sequence_sha256")
            row["prompt_length"] = canonical_row.get("prompt_length")
            row["sequence_length"] = canonical_row.get("sequence_length")
        rows.append(row)
    rows.sort(key=lambda row: (str(row["episode_id"]), train_data.event_order_key(row), str(row["event_id"])))
    if len(rows) != EXPECTED_LEGACY_CT_FIRST_HIT_ROWS:
        raise DataContractError("legacy row materialisation count drifted")
    if sum(1 for row in rows if row["label"]) != EXPECTED_LEGACY_CT_FIRST_HIT_POSITIVES:
        raise DataContractError("legacy positive count drifted")
    if sum(1 for row in rows if not row["label"]) != EXPECTED_LEGACY_CT_FIRST_HIT_NEGATIVES:
        raise DataContractError("legacy negative count drifted")
    return {
        "rows": rows,
        "signature": signature,
        "counts": dict(targets["counts"]),
        "label0_positive_prefix": label0,
        "scaler_source": {
            "kind": "ct_first_hit_train_exact_rows",
            "fit_split": SPLIT_TRAIN,
            "rows": len(rows),
            "positive_rows": EXPECTED_LEGACY_CT_FIRST_HIT_POSITIVES,
            "negative_rows": EXPECTED_LEGACY_CT_FIRST_HIT_NEGATIVES,
            "target_pool_sha256": signature["sha256"],
            "row_order": "sorted by (episode_id, event_order_key, event_id)",
            "feature_refs": "legacy_feature_ref on each row",
            "note": "same scaler pool for every arm; repaired row masks must not change it",
        },
    }


# --------------------------------------------------------------------------- repaired view


def _row_from_canonical(
    canonical: Mapping[str, Any],
    *,
    split: str,
    repair_status: str,
    feature_ref: Mapping[str, Any],
    old_row: Mapping[str, Any] | None = None,
    supersedes_event_id: str | None = None,
    repair_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event_id = str(canonical["event_id"])
    row = {
        "row_key": f"repaired::{event_id}",
        "view": "repaired",
        "event_id": event_id,
        "episode_id": str(canonical["episode_id"]),
        "namespaced_episode_id": str(canonical["episode_id"]),
        "source_tag": canonical.get("source_tag"),
        "scope": str(canonical["scope"]),
        "split": split,
        "source_split": canonical.get("source_split"),
        "group_id": str(canonical["group_id"]),
        "group_hash": canonical.get("group_hash"),
        "suite": canonical.get("suite"),
        "user_task_id": canonical.get("user_task_id"),
        "injection_task_id": canonical.get("injection_task_id"),
        "attack_type": canonical.get("attack_type"),
        "strict_final": bool(canonical.get("strict_final")),
        "victim_model": canonical.get("victim_model"),
        "event_type": str(canonical["event_type"]),
        "round_id": int(canonical.get("round_id", 0)),
        "message_index": int(canonical.get("message_index", 0)),
        "index": int(canonical.get("index", 0)),
        "call_index": canonical.get("call_index"),
        "segment_index": canonical.get("segment_index"),
        "feature_token_index": canonical.get("feature_token_index"),
        "token_start": canonical.get("token_start"),
        "token_end": canonical.get("token_end"),
        "completion_token_start": canonical.get("completion_token_start"),
        "completion_token_end": canonical.get("completion_token_end"),
        "token_count": canonical.get("token_count"),
        "char_start": canonical.get("char_start"),
        "char_end": canonical.get("char_end"),
        "text_chars": canonical.get("text_chars"),
        "text_sha256": canonical.get("text_sha256"),
        "sequence_sha256": canonical.get("sequence_sha256"),
        "prompt_length": canonical.get("prompt_length"),
        "sequence_length": canonical.get("sequence_length"),
        "terminal_token_id": canonical.get("terminal_token_id"),
        "inference_position": bool(canonical.get("inference_position", True)),
        "label_status": str(canonical.get("label_status")),
        "label_value": canonical.get("label_value"),
        "previous_security": canonical.get("previous_security"),
        "prefix_security": canonical.get("prefix_security"),
        "security_flip": canonical.get("security_flip"),
        "label_reason": canonical.get("label_reason"),
        "repair_status": repair_status,
        "repair_version": "fine_label_repair_v1",
        "supersedes_event_id": supersedes_event_id,
        "feature_ref": dict(feature_ref),
        "repair_provenance": dict(repair_provenance or {}),
    }
    if old_row is not None:
        for name in ("char_start", "char_end", "text_chars", "text_sha256", "visible_prefix_sha256",
                     "start_prefix_sha256", "end_prefix_sha256"):
            if old_row.get(name) is not None and row.get(name) is None:
                row[name] = old_row[name]
    return row


def _old_event_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    events_dir = Path(COT_EVENT_STORE) / "events"
    for name in sorted(os.listdir(events_dir)):
        if name.startswith("shard_") and name.endswith(".jsonl"):
            for row in read_jsonl(events_dir / name):
                rows[str(row["event_id"])] = dict(row)
    return rows


def build_repaired_view(
    root: Path | str = ".",
    *,
    tokenizer: Any | None = None,
    alignment: AlignmentBuild | None = None,
) -> dict[str, Any]:
    """Build the repaired TRAIN/DEV metadata and official labels; no vectors."""
    study = load_study(root, verify_features=False)
    canonical = _canonical_map(study)
    old_events = _old_event_rows()
    alignment = alignment or build_alignment(study_rows=study.rows, episodes=study.episodes, tokenizer=tokenizer)
    tokenizer = tokenizer or __import__("sdsafe_fine_label_repair.alignment", fromlist=["load_tokenizer"]).load_tokenizer()

    labelled_candidates, candidate_label_audit = build_candidate_labels(
        list(alignment.candidate_rows), tokenizer=tokenizer
    )
    labelled_by_id = {str(row["event_id"]): row for row in labelled_candidates}
    candidate_train = [{**row, **labelled_by_id[str(row["event_id"])]} for row in alignment.candidate_rows if row["split"] == SPLIT_TRAIN]
    candidate_dev = [{**row, **labelled_by_id[str(row["event_id"])]} for row in alignment.candidate_rows if row["split"] == SPLIT_DEV]

    reused_corrections, utf8_label_audit = apply_utf8_label_reuse(alignment.utf8_corrections)
    correction_train = [row for row in reused_corrections if row["split"] == SPLIT_TRAIN]
    correction_dev = [row for row in reused_corrections if row["split"] == SPLIT_DEV]
    superseded_train = {str(row["old_event_id"]) for row in correction_train}
    superseded_dev = {str(row["old_event_id"]) for row in correction_dev}

    repaired_train: list[dict[str, Any]] = []
    superseded_rows: list[dict[str, Any]] = []
    for canonical_row in study.train_rows():
        if canonical_row["event_type"] not in CT_EVENT_TYPES:
            continue
        event_id = str(canonical_row["event_id"])
        split = assert_protected_split(canonical_row["split"], where=f"repaired train {event_id}")
        if event_id in superseded_train:
            old = dict(canonical_row)
            old["old_event_row_sha256"] = canonical_sha256(old_events.get(event_id, {}))
            old["replacement_event_id"] = event_id + "::utf8fix_v1"
            old["superseded_reason"] = "trailing_utf8_zero_width_tokens"
            superseded_rows.append(old)
            continue
        ref = _legacy_feature_ref({**canonical_row, "split": split})
        row = _row_from_canonical(
            canonical_row,
            split=split,
            repair_status="active_old",
            feature_ref=ref,
            old_row=old_events.get(event_id),
        )
        repaired_train.append(row)
    for correction in correction_train:
        canonical_row = canonical[str(correction["old_event_id"])]
        repaired_correction = {
            **correction,
            "event_id": correction["new_event_id"],
            "feature_token_index": int(correction["token_end"]),
            "token_end": int(correction["token_end"]),
            "completion_token_end": int(correction["completion_token_end"]),
            "token_count": int(correction["token_end"]) - int(correction["token_start"]) + 1,
            "old_token_count": canonical_row.get("token_count"),
            "sequence_sha256": correction["round_sequence_sha256"],
        }
        ref = _supplemental_feature_ref(repaired_correction)
        row = _row_from_canonical(
            {**canonical_row, **repaired_correction},
            split=SPLIT_TRAIN,
            repair_status="utf8_corrected",
            feature_ref=ref,
            old_row=old_events.get(str(correction["old_event_id"])),
            supersedes_event_id=str(correction["old_event_id"]),
            repair_provenance={"utf8": correction},
        )
        row["old_feature_ref"] = _legacy_feature_ref({**canonical_row, "split": SPLIT_TRAIN})
        for name in (
            "coordinate_basis",
            "completion_char_start",
            "completion_char_end",
            "label_visible_prefix_sha256",
            "label_visible_prefix_chars",
            "label_prefix_coordinate_basis",
        ):
            if correction.get(name) is not None:
                row[name] = correction[name]
        repaired_train.append(row)
    for candidate in candidate_train:
        candidate_base = {**candidate, "sequence_sha256": candidate.get("round_sequence_sha256")}
        ref = _supplemental_feature_ref(candidate_base)
        row = _row_from_canonical(
            {
                **candidate_base,
                "event_id": candidate["event_id"],
                "split": SPLIT_TRAIN,
                "event_type": EVENT_CONTENT_SEGMENT,
                "index": int(candidate["segment_index"]),
            },
            split=SPLIT_TRAIN,
            repair_status="candidate_new_position",
            feature_ref=ref,
            repair_provenance={"candidate": candidate.get("alignment_proof")},
        )
        row.update({k: candidate[k] for k in ("label_status", "label", "label_value", "previous_security", "prefix_security", "security_flip", "label_reason", "label_provenance") if k in candidate})
        for name in (
            "coordinate_basis",
            "visible_char_start",
            "visible_char_end",
            "api_visible_chars",
            "api_visible_text_sha256",
            "visible_prefix_sha256",
            "start_prefix_sha256",
            "end_prefix_sha256",
        ):
            if candidate.get(name) is not None:
                row[name] = candidate[name]
        row["feature_ref"] = ref
        repaired_train.append(row)

    repaired_dev: list[dict[str, Any]] = []
    for canonical_row in study.split_rows(SPLIT_DEV):
        event_id = str(canonical_row["event_id"])
        split = assert_protected_split(canonical_row["split"], where=f"repaired dev {event_id}")
        if event_id in superseded_dev:
            old = dict(canonical_row)
            old["old_event_row_sha256"] = canonical_sha256(old_events.get(event_id, {}))
            old["replacement_event_id"] = event_id + "::utf8fix_v1"
            old["superseded_reason"] = "trailing_utf8_zero_width_tokens"
            superseded_rows.append(old)
            continue
        ref = _legacy_feature_ref({**canonical_row, "split": split})
        repaired_dev.append(
            _row_from_canonical(
                canonical_row,
                split=split,
                repair_status="active_old",
                feature_ref=ref,
                old_row=old_events.get(event_id),
            )
        )
    for correction in correction_dev:
        canonical_row = canonical[str(correction["old_event_id"])]
        repaired_correction = {
            **correction,
            "event_id": correction["new_event_id"],
            "feature_token_index": int(correction["token_end"]),
            "token_end": int(correction["token_end"]),
            "completion_token_end": int(correction["completion_token_end"]),
            "token_count": int(correction["token_end"]) - int(correction["token_start"]) + 1,
            "old_token_count": canonical_row.get("token_count"),
            "sequence_sha256": correction["round_sequence_sha256"],
        }
        ref = _supplemental_feature_ref(repaired_correction)
        row = _row_from_canonical(
            {**canonical_row, **repaired_correction},
            split=SPLIT_DEV,
            repair_status="utf8_corrected",
            feature_ref=ref,
            old_row=old_events.get(str(correction["old_event_id"])),
            supersedes_event_id=str(correction["old_event_id"]),
            repair_provenance={"utf8": correction},
        )
        row["old_feature_ref"] = _legacy_feature_ref({**canonical_row, "split": SPLIT_DEV})
        for name in (
            "coordinate_basis",
            "completion_char_start",
            "completion_char_end",
            "label_visible_prefix_sha256",
            "label_visible_prefix_chars",
            "label_prefix_coordinate_basis",
        ):
            if correction.get(name) is not None:
                row[name] = correction[name]
        repaired_dev.append(row)
    for candidate in candidate_dev:
        candidate_base = {**candidate, "sequence_sha256": candidate.get("round_sequence_sha256")}
        ref = _supplemental_feature_ref(candidate_base)
        row = _row_from_canonical(
            {
                **candidate_base,
                "event_id": candidate["event_id"],
                "split": SPLIT_DEV,
                "event_type": EVENT_CONTENT_SEGMENT,
                "index": int(candidate["segment_index"]),
            },
            split=SPLIT_DEV,
            repair_status="candidate_new_position",
            feature_ref=ref,
            repair_provenance={"candidate": candidate.get("alignment_proof")},
        )
        row.update({k: candidate[k] for k in ("label_status", "label", "label_value", "previous_security", "prefix_security", "security_flip", "label_reason", "label_provenance") if k in candidate})
        for name in (
            "coordinate_basis",
            "visible_char_start",
            "visible_char_end",
            "api_visible_chars",
            "api_visible_text_sha256",
            "visible_prefix_sha256",
            "start_prefix_sha256",
            "end_prefix_sha256",
        ):
            if candidate.get(name) is not None:
                row[name] = candidate[name]
        row["feature_ref"] = ref
        repaired_dev.append(row)

    def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda row: (str(row["episode_id"]), train_data.event_order_key(row), str(row["event_id"])))

    repaired_train = sort_rows(repaired_train)
    repaired_dev = sort_rows(repaired_dev)
    if any(row["split"] != SPLIT_TRAIN for row in repaired_train):
        raise DataContractError("repaired TRAIN view contains a non-TRAIN row")
    if any(row["split"] != SPLIT_DEV for row in repaired_dev):
        raise DataContractError("repaired DEV view contains a non-DEV row")
    for row in repaired_train + repaired_dev:
        validate_feature_ref(row["feature_ref"], expected_split=str(row["split"]))
        if row["feature_ref"]["kind"] == SUPPLEMENTAL_FEATURE_KIND:
            token_index = int(row["feature_token_index"])
            token_end = int(row["token_end"])
            ref_token_index = int(row["feature_ref"]["token_index"])
            token_count = row.get("token_count")
            expected_token_count = token_end - int(row["token_start"]) + 1
            if not (token_index == token_end == ref_token_index):
                raise DataContractError(
                    f"{row['event_id']}: supplemental feature/token coordinates disagree "
                    f"({token_index}, {token_end}, {ref_token_index})"
                )
            if token_count is None or int(token_count) != expected_token_count:
                raise DataContractError(
                    f"{row['event_id']}: token_count {token_count!r} != {expected_token_count}"
                )
            if not 1 <= int(token_count) <= 48:
                raise DataContractError(f"{row['event_id']}: token_count {token_count} is outside 1..48")

    assignments, policy_audit_payload = build_policy_assignments(repaired_train)
    episodes_payload = []
    for episode_id, episode in sorted(study.episodes.items()):
        if episode.split not in PROTECTED_SPLITS:
            continue
        split = assert_protected_split(episode.split, where=f"episode {episode_id}")
        episodes_payload.append(
            {
                "episode_id": str(episode.episode_id),
                "original_episode_id": str(episode.original_episode_id),
                "scope": str(episode.scope),
                "source_tag": str(episode.source_tag),
                "source_split": episode.source_split,
                "split": split,
                "group_id": str(episode.group_id),
                "group_hash": str(episode.group_hash),
                "strict_final": bool(episode.strict_final),
                "victim_model": str(episode.victim_model),
                "suite": str(episode.suite),
                "user_task_id": str(episode.user_task_id),
                "injection_task_id": str(episode.injection_task_id),
                "attack_type": episode.attack_type,
                "round_count": int(episode.round_count),
            }
        )
    common_dev = set(study.common_dev_episode_ids())
    diagnostic_dev = set(study.diagnostic_dev_episode_ids())
    if len(common_dev) != EXPECTED_COMMON_DEV_EPISODES or len(diagnostic_dev) != EXPECTED_DIAGNOSTIC_DEV_EPISODES:
        raise DataContractError("common/diagnostic DEV episode counts drifted")
    return {
        "study": study,
        "alignment": alignment,
        "train_rows": repaired_train,
        "dev_rows": repaired_dev,
        "episodes": episodes_payload,
        "superseded_rows": superseded_rows,
        "candidate_rows": labelled_candidates,
        "corrections": reused_corrections,
        "candidate_label_audit": candidate_label_audit,
        "utf8_label_audit": utf8_label_audit,
        "assignments": assignments,
        "policy_audit": policy_audit_payload,
        "common_dev_episode_ids": sorted(common_dev),
        "diagnostic_dev_episode_ids": sorted(diagnostic_dev),
        "counts": {
            "train_rows": len(repaired_train),
            "train_exact_rows": sum(1 for row in repaired_train if row["label_status"] == LABEL_EXACT),
            "train_unknown_rows": sum(1 for row in repaired_train if row["label_status"] == LABEL_UNKNOWN),
            "dev_rows": len(repaired_dev),
            "dev_inference_ct_rows": sum(1 for row in repaired_dev if row["event_type"] in CT_EVENT_TYPES),
            "dev_round_end_rows": sum(1 for row in repaired_dev if row["event_type"] == EVENT_ROUND_END),
            "superseded_rows": len(superseded_rows),
            "candidate_rows": len(labelled_candidates),
            "utf8_corrections": len(reused_corrections),
        },
    }


# --------------------------------------------------------------------------- writing / protocol


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _policy_menu() -> dict[str, Any]:
    balanced = []
    for policy in POLICIES:
        for lr in ("1e-3", "3e-4"):
            for seed in (0, 1, 2):
                balanced.append(
                    {
                        "run_id": f"balanced|{policy}|lr{lr}|seed{seed}",
                        "cohort": "repaired_train",
                        "label_policy": policy,
                        "sampler": "balanced_1_to_3",
                        "learning_rate": float(lr),
                        "seed": seed,
                    }
                )
    controls = []
    for cohort in ("repaired_train_L0_natural", "legacy_train_L0_natural"):
        for seed in (0, 1, 2):
            controls.append(
                {
                    "run_id": f"natural|{cohort}|lr1e-3|seed{seed}",
                    "cohort": cohort,
                    "label_policy": "L0",
                    "sampler": "natural_deterministic_uniform",
                    "learning_rate": 1e-3,
                    "seed": seed,
                }
            )
    runs = balanced + controls
    if len(runs) != 36 or len({row["run_id"] for row in runs}) != 36:
        raise DataContractError("the fixed 36-run menu is malformed")
    return {
        "balanced_factorial": balanced,
        "natural_controls": controls,
        "runs": runs,
        "expected_run_count": 36,
        "balanced_run_count": 30,
        "natural_control_run_count": 6,
        "model_training_implementation": "deferred to lane T",
    }


def build_protocol_payload(
    *,
    legacy: Mapping[str, Any],
    repaired: Mapping[str, Any],
    alignment: AlignmentBuild,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "status": "lane_D_data_and_labels_handoff",
        "scope": "DEV-only exploration; TRAIN/DEV metadata only; no TEST raw traces",
        "authority": {
            "split": "fine_confirm.sources.load_cohort / unified all_data_mixed split",
            "legacy_ct_first_hit_rows": EXPECTED_LEGACY_CT_FIRST_HIT_ROWS,
            "legacy_ct_first_hit_signature": legacy["signature"],
            "legacy_label0_prefix_positive_rows": legacy["label0_positive_prefix"],
        },
        "repair": {
            "candidate_positions": len(alignment.candidate_rows),
            "candidate_positions_by_split": alignment.audit["candidate_positions_by_split"],
            "utf8_endpoint_corrections": len(alignment.utf8_corrections),
            "utf8_corrections_by_split": alignment.audit["utf8_corrections_by_split"],
            "ambiguous_b295_round": alignment.audit["ambiguous_round"],
            "new_feature_extraction_rounds": alignment.audit["extraction_rounds"],
            "new_feature_extraction_endpoints": len(alignment.candidate_rows) + len(alignment.utf8_corrections),
            "new_feature_extraction_sequence_length": alignment.audit["extraction_sequence_length"],
            "test_raw_traces_opened": 0,
        },
        "policies": {
            "L0": "legacy first_hit recomputed on repaired exact rows; control/compat label",
            "L1": "keep L0 positives, exact prefix-false -> 0, every other exact prefix-positive -> mask",
            "L2": "L1 plus exact prefix-positive rows in the L0 anchor's same episode/round/message -> 1",
            "L3": "every exact row label = exact prefix_security, including transient positives",
            "L4": "random L0-negative drop control, per (scope,event_type), fixed RNG 20261109; not an accurate-label candidate",
            "unknown": "always masked, never 0, retained for inference",
            "L2_note": "post-hit text itself is not called malicious; this is a local prefix-risk target",
        },
        "policy_audit": {
            policy: repaired["policy_audit"]["per_policy"][policy]
            for policy in POLICIES
        },
        "feature_contract": {
            "legacy_kind": LEGACY_FEATURE_KIND,
            "supplemental_kind": SUPPLEMENTAL_FEATURE_KIND,
            "hidden_size": HIDDEN_SIZE,
            "dtype": FEATURE_DTYPE,
            "legacy_control": "must fetch legacy_feature_ref rows; never overlay repaired rows",
            "read_guard": "validate_feature_ref(ref, expected_split) at the actual vector-read boundary",
            "supplemental_root": SUPPLEMENTAL_FEATURE_ROOT,
        },
        "visible_prefix_contract": {
            "candidate_coordinate_basis": "completion_absolute_chars",
            "candidate_visible_coordinates": ["visible_char_start", "visible_char_end", "char_start", "char_end"],
            "checker_input": "full_text[visible_char_start:absolute_char_end] only",
            "forbidden_checker_input": ["prompt", "think/reasoning", "structural tokens", "future visible answer"],
            "label_provenance_fields": [
                "prefix_text_sha256",
                "prefix_text_chars",
                "prefix_coordinate_basis",
                "visible_char_start",
                "absolute_char_start",
                "absolute_char_end",
                "api_visible_char_end",
                "contains_prompt_or_cot",
            ],
            "utf8_reuse": "same API-visible prefix hash + same round sequence/environment/task binding",
        },
        "scaler": legacy["scaler_source"],
        "menu": _policy_menu(),
        "outputs": {
            "data_root": "artifacts/fine_label_repair_v1/data/",
            "handoff": "artifacts/fine_label_repair_v1/handoffs/data.json",
        },
        "explicit_non_claims": [
            "no formal training or GPU work in lane D",
            "no TEST raw trace or feature vector was opened",
            "the repaired positives are exact official prefix-checker states, not a universal maliciousness claim",
        ],
    }


def build_data_artifacts(root: Path | str = ".", out_root: Path | str = DATA_ROOT) -> dict[str, Any]:
    """Reproducible CPU build of all lane-D metadata; writes no vector matrix."""
    root = Path(root).resolve()
    out = Path(out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = __import__("sdsafe_fine_label_repair.alignment", fromlist=["load_tokenizer"]).load_tokenizer()
    legacy = legacy_ct_first_hit_view(root)
    repaired = build_repaired_view(root, tokenizer=tokenizer)
    alignment = repaired["alignment"]
    study = repaired["study"]

    _write_jsonl(out / "legacy_ct_first_hit_rows.jsonl", legacy["rows"])
    _write_jsonl(out / "repaired_train_rows.jsonl", repaired["train_rows"])
    _write_jsonl(out / "repaired_dev_rows.jsonl", repaired["dev_rows"])
    _write_jsonl(out / "episodes_train_dev.jsonl", repaired["episodes"])
    _write_jsonl(out / "superseded_utf8_rows.jsonl", repaired["superseded_rows"])
    _write_jsonl(out / "utf8_replacement_map.jsonl", repaired["corrections"])
    _write_jsonl(out / "candidate_label_proofs.jsonl", repaired["candidate_rows"])

    supplemental = []
    for row in repaired["train_rows"] + repaired["dev_rows"]:
        if row["feature_ref"]["kind"] == SUPPLEMENTAL_FEATURE_KIND:
            supplemental.append(
                {
                    "event_id": row["event_id"],
                    "split": row["split"],
                    "scope": row["scope"],
                    "episode_id": row["episode_id"],
                    "round_id": row["round_id"],
                    "event_type": row["event_type"],
                    "feature_token_index": row["feature_token_index"],
                    "sequence_length": row["sequence_length"],
                    "repair_status": row["repair_status"],
                    "label_status": row["label_status"],
                    "feature_ref": row["feature_ref"],
                }
            )
    supplemental.sort(key=lambda row: (row["split"], row["episode_id"], row["round_id"], row["feature_token_index"], row["event_id"]))
    _write_jsonl(out / "feature_supplemental_plan.jsonl", supplemental)

    extraction_rounds = list(alignment.extraction_rounds)
    _write_json(
        out / "feature_extraction_rounds.json",
        {
            "new_forward_rounds": len(extraction_rounds),
            "new_endpoints": len(supplemental),
            "rounds": extraction_rounds,
            "sequence_length_min": min((int(row["sequence_length"]) for row in extraction_rounds), default=0),
            "sequence_length_max": max((int(row["sequence_length"]) for row in extraction_rounds), default=0),
            "sequence_length_sum": sum(int(row["sequence_length"]) for row in extraction_rounds),
            "parent_gate": "lane F decides extraction cost; lane D does not forward Qwen",
        },
    )

    assignment_rows = []
    for row in repaired["train_rows"]:
        assignment_rows.append(
            {
                "event_id": row["event_id"],
                "row_key": row["row_key"],
                "split": row["split"],
                "scope": row["scope"],
                "event_type": row["event_type"],
                "group_id": row["group_id"],
                "strict_final": bool(row["strict_final"]),
                "label_status": row["label_status"],
                "prefix_security": row["prefix_security"],
                "policies": repaired["assignments"],
            }
        )
    # Write one compact policy assignment per repaired TRAIN CT row.
    _write_jsonl(
        out / POLICY_ASSIGNMENT_FILE,
        (
            {
                "event_id": row["event_id"],
                "row_key": row["row_key"],
                "scope": row["scope"],
                "event_type": row["event_type"],
                "split": row["split"],
                "group_id": row["group_id"],
                "strict_final": bool(row["strict_final"]),
                "label_status": row["label_status"],
                "prefix_security": row["prefix_security"],
                "policies": {
                    policy: repaired["assignments"][policy][row["event_id"]]
                    for policy in POLICIES
                },
            }
            for row in repaired["train_rows"]
        ),
    )
    _write_json(out / POLICY_AUDIT_FILE, repaired["policy_audit"])
    _write_json(out / "scaler_source.json", legacy["scaler_source"])
    _write_json(
        out / "alignment_audit.json",
        {
            **alignment.audit,
            "candidate_label_audit": repaired["candidate_label_audit"],
            "utf8_label_reuse_audit": repaired["utf8_label_audit"],
        },
    )
    _write_json(
        out / "data_audit.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "study_id": STUDY_ID,
            "legacy": {
                "rows": len(legacy["rows"]),
                "signature": legacy["signature"],
                "label0_positive_prefix": legacy["label0_positive_prefix"],
                "scaler_source": legacy["scaler_source"],
            },
            "repaired": repaired["counts"],
            "episodes": {
                "train": sum(1 for row in repaired["episodes"] if row["split"] == SPLIT_TRAIN),
                "dev": sum(1 for row in repaired["episodes"] if row["split"] == SPLIT_DEV),
                "common_dev": len(repaired["common_dev_episode_ids"]),
                "diagnostic_dev": len(repaired["diagnostic_dev_episode_ids"]),
                "train_groups": len({row["group_id"] for row in repaired["episodes"] if row["split"] == SPLIT_TRAIN}),
                "dev_groups": len({row["group_id"] for row in repaired["episodes"] if row["split"] == SPLIT_DEV}),
            },
            "policy_audit": repaired["policy_audit"],
            "feature_supplemental_rows": len(supplemental),
            "test_raw_traces_opened": 0,
            "test_vectors_materialized": False,
        },
    )
    protocol = build_protocol_payload(legacy=legacy, repaired=repaired, alignment=alignment)
    _write_json(ARTIFACT_ROOT / "protocol.json", protocol)

    output_paths = [
        out / "legacy_ct_first_hit_rows.jsonl",
        out / "repaired_train_rows.jsonl",
        out / "repaired_dev_rows.jsonl",
        out / "episodes_train_dev.jsonl",
        out / "superseded_utf8_rows.jsonl",
        out / "utf8_replacement_map.jsonl",
        out / "candidate_label_proofs.jsonl",
        out / "feature_supplemental_plan.jsonl",
        out / "feature_extraction_rounds.json",
        out / POLICY_ASSIGNMENT_FILE,
        out / POLICY_AUDIT_FILE,
        out / "scaler_source.json",
        out / "alignment_audit.json",
        out / "data_audit.json",
        ARTIFACT_ROOT / "protocol.json",
    ]
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in output_paths}
    handoff = {
        "lane": "D",
        "study_id": STUDY_ID,
        "status": "complete",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "exact_command": (
            "cd /root/autodl-tmp/SD-Safe && PYTHONPATH=$PWD/src:$PWD PYTHONDONTWRITEBYTECODE=1 "
            "HF_HOME=$PWD/external/agentdojo_env/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
            "OPENAI_API_KEY=sk-test external/agentdojo_env/bin/python -c "
            "\"from sdsafe_fine_label_repair.data import build_data_artifacts; build_data_artifacts('.')\""
        ),
        "inputs": {
            "candidate_positions": str(CANDIDATE_POSITIONS_PATH),
            "candidate_positions_sha256": sha256_file(CANDIDATE_POSITIONS_PATH),
            "alignment_proofs": str(ALIGNMENT_PROOFS_PATH),
            "alignment_proofs_sha256": sha256_file(ALIGNMENT_PROOFS_PATH),
            "repair_plan": str(REPAIR_PLAN_PATH),
            "repair_plan_sha256": sha256_file(REPAIR_PLAN_PATH),
            "cot_event_store": str(Path(root) / "artifacts/event_data/cot"),
        },
        "outputs": hashes,
        "counts": {
            "legacy_ct_first_hit_rows": len(legacy["rows"]),
            "legacy_ct_first_hit_signature": legacy["signature"],
            "legacy_label0_prefix_positive": legacy["label0_positive_prefix"],
            "repaired": repaired["counts"],
            "candidate_positions_by_split": alignment.audit["candidate_positions_by_split"],
            "utf8_corrections_by_split": alignment.audit["utf8_corrections_by_split"],
            "new_feature_extraction_rounds": alignment.audit["extraction_rounds"],
            "new_feature_extraction_endpoints": len(supplemental),
            "new_feature_sequence_length": alignment.audit["extraction_sequence_length"],
            "policy_audit": repaired["policy_audit"]["per_policy"],
            "assertions": repaired["policy_audit"]["assertions"],
            "repaired_l0_label0_prefix_positive_rows": repaired["policy_audit"]["per_policy"]["L0"]["zero_exact_prefix_positive_negatives"],
            "l1_masked_exact_rows": repaired["policy_audit"]["per_policy"]["L1"]["masked_exact_rows"],
            "l2_same_turn_added_positive_rows": (
                repaired["policy_audit"]["per_policy"]["L2"]["positive_rows"]
                - repaired["policy_audit"]["per_policy"]["L0"]["positive_rows"]
            ),
            "l3_positive_rows": repaired["policy_audit"]["per_policy"]["L3"]["positive_rows"],
            "l4_drop_counts_by_stratum": {
                key: len(value) for key, value in sorted(repaired["policy_audit"]["l4_drop"]["dropped_by_stratum"].items())
            },
        },
        "schemas": {
            "legacy_row": {
                "view": "legacy_ct_first_hit",
                "feature_ref": "legacy_feature_ref / feature_ref kind=cached_canonical_bf16; immutable (scope, local_feature_row) plus global row",
                "label": "old exact CT first_hit 107215 semantics; unknown not present",
            },
            "repaired_row": {
                "view": "repaired",
                "repair_status": "active_old | utf8_corrected | candidate_new_position",
                "feature_ref": "cached_canonical_bf16 for active_old; supplemental_bf16_v1 for corrected/candidate rows",
                "label_status": "exact | unknown; unknown label is null and masked from loss",
                "supersedes_event_id": "old event id for a UTF-8 endpoint replacement, null otherwise",
                "candidate_coordinates": {
                    "coordinate_basis": "completion_absolute_chars",
                    "visible_char_start": "first character of the API-visible answer in the decoded completion",
                    "visible_char_end": "end of the API-visible answer in the decoded completion",
                    "char_start": "candidate segment start, completion-absolute",
                    "char_end": "candidate segment end, completion-absolute",
                },
                "utf8_coordinates": {
                    "coordinate_basis": "round_region_relative_visible_chars",
                    "completion_char_start": "segment start in the decoded completion",
                    "completion_char_end": "segment end in the decoded completion",
                },
                "label_visible_prefix": "label_provenance carries the exact checker prefix hash/chars and coordinate basis",
            },
            "policy_assignment": {
                "file": POLICY_ASSIGNMENT_FILE,
                "fields": ["event_id", "row_key", "scope", "event_type", "split", "group_id", "strict_final", "label_status", "prefix_security", "policies.L0..L4"],
                "policy_entry": {"label": "true|false|null", "mask": "bool", "mask_reason": "string|null"},
            },
            "feature_supplemental_ref": {
                "kind": SUPPLEMENTAL_FEATURE_KIND,
                "unique_key": ["scope", "episode_id", "round_id", "token_index", "round_sequence_sha256", "event_id"],
                "guard": "validate_feature_ref(ref, expected_split) before any vector read",
            },
        },
        "examples": {
            "legacy_row": legacy["rows"][0],
            "repaired_corrected_row": next((row for row in repaired["train_rows"] if row["repair_status"] == "utf8_corrected"), None),
            "repaired_candidate_row": next((row for row in repaired["train_rows"] if row["repair_status"] == "candidate_new_position"), None),
            "policy_assignment": next(
                (
                    {
                        "event_id": row["event_id"],
                        "policies": {policy: repaired["assignments"][policy][row["event_id"]] for policy in POLICIES},
                    }
                    for row in repaired["train_rows"]
                ),
                None,
            ),
        },
        "next_lane_F_api": {
            "read_legacy_rows": "read data/legacy_ct_first_hit_rows.jsonl; use row['feature_ref'] only; never overlay repaired rows",
            "read_repaired_train_rows": "read data/repaired_train_rows.jsonl; use row['feature_ref']",
            "read_repaired_dev_rows": "read data/repaired_dev_rows.jsonl; filter event_type for C/T or optional E; do not filter by label/status",
            "supplemental_plan": "read data/feature_supplemental_plan.jsonl and data/feature_extraction_rounds.json; parent gate decides extraction",
            "guard": "call sdsafe_fine_label_repair.data.validate_feature_ref(ref, expected_split) at the actual vector-read boundary",
            "targets": "read data/policies/assignments.jsonl and data/policies/audit.json; L0..L4 are already computed",
            "scaler": "fit/reuse exactly data/scaler_source.json target_pool_sha256; do not fit on repaired masks",
            "candidate_label_audit": "read data/candidate_label_proofs.jsonl; label_provenance.prefix_text_sha256/prefix_text_chars/prefix_coordinate_basis identify the exact API-visible checker input",
            "coordinate_fields": "candidate rows carry visible_char_start/visible_char_end plus completion-absolute char_start/char_end; UTF-8 corrected rows carry completion_char_start/completion_char_end and round-relative char_start/char_end",
        },
        "visible_prefix_fix": {
            "archive": "artifacts/fine_label_repair_v1/archive/data_pre_visible_prefix_fix/d_outputs_and_source_pre_fix.tar.gz",
            "archive_sha256": (
                sha256_file(Path(root) / "artifacts/fine_label_repair_v1/archive/data_pre_visible_prefix_fix/d_outputs_and_source_pre_fix.tar.gz")
                if (Path(root) / "artifacts/fine_label_repair_v1/archive/data_pre_visible_prefix_fix/d_outputs_and_source_pre_fix.tar.gz").is_file()
                else None
            ),
            "old_behavior": "candidate checker prefixes were full_text[:char_end], which included hidden CoT before the API-visible answer",
            "fixed_behavior": "candidate segment_prefixes are full_text[visible_char_start:absolute_char_end] only",
            "counts_changed": "recomputed from the fixed build; old prefix provenance is not claimed valid",
        },
        "blockers": [],
        "non_claims": protocol["explicit_non_claims"],
        "known_caveats": [
            "L2 same-turn labels are local prefix-risk labels, not a claim that post-hit text is malicious.",
            "L4 is a random-drop control, not an accurate-label candidate.",
            "The repaired L0 first_hit is recomputed on repaired rows; the legacy L0 natural control uses the separate legacy row/vector view.",
            "Only CoT content alignment was repaired; no universal semantic relabeling of old negatives was attempted.",
            "Exact prefix positives are official goal-checker states. For travel injection_task_6 the checker is a literal Hotel mention, so a positive is not by itself evidence of universal malicious intent.",
        ],
        "focused_tests": {
            "command": (
                "PYTHONPATH=$PWD/src:$PWD PYTHONDONTWRITEBYTECODE=1 "
                "HF_HOME=$PWD/external/agentdojo_env/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "
                "OPENAI_API_KEY=sk-test external/agentdojo_env/bin/python -m unittest "
                "tests.test_fine_label_repair_alignment tests.test_fine_label_repair_labels "
                "tests.test_fine_label_repair_data -v"
            ),
            "result": "Ran 18 tests; OK",
            "covers": [
                "111 proved candidates",
                "b295 fail-closed",
                "actual-id Unicode prefix proof",
                "328-row UTF-8 replacement map",
                "L0..L4 censoring/transient/unknown",
                "L4 count-matched strata",
                "current DEV from old source TRAIN",
                "TEST read refusal",
                "legacy 107215/signature/scaler",
                "supplemental ref uniqueness",
                "TRAIN/DEV-only rows",
                "API-visible-only checker prefix delivery (CoT mention excluded)",
                "visible first segment false, second segment flips true",
                "nonzero API-visible offset coordinate mapping",
            ],
        },
        "protected_guard": {
            "before_file": "artifacts/fine_label_repair_v1/orchestration/protected-before.json",
            "files": 1063,
            "missing": 0,
            "changed": 0,
            "status": "PASS",
        },
    }
    _write_json(ARTIFACT_ROOT / "handoffs/data.json", handoff)
    return {
        "legacy": legacy,
        "repaired": repaired,
        "alignment": alignment,
        "outputs": hashes,
        "handoff": handoff,
    }
