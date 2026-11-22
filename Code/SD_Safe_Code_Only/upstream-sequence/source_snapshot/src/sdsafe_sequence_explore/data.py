"""Materialise and open the NEW read-only sequence data version.

The materialised version lives under
``artifacts/fine_sequence_explore_v1/data/sequence_v1`` and contains:

* one shared BF16 matrix of deduplicated physical feature rows;
* per-position metadata preserving the original producer feature ref and the resolved
  physical ``(matrix, row)`` identity plus token geometry;
* per-episode chronological E/C/T offsets, lengths, CT mask, groups, authoritative truth
  and TRAIN FIT/CAL/SELECT partition;
* per-position label arrays (L0..L4 plus prefix/previous/flip/exact) with ``-1`` masks;
* FIT-only and FULL-TRAIN scalers fitted on deterministic label-blind CT populations.

The builder is the only writer.  :class:`SequenceDataset` is strictly read-only and never
rewrites manifests/scalers.  Every feature read goes through a provider whose real
implementation is ``sdsafe_fine_label_repair.features.FeatureStore``; that reader resolves
the physical row and re-checks the authoritative split *before* opening a vector.  TEST is
not an accepted split anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence

import numpy as np

from sdsafe_features.common import (
    atomic_write_json,
    canonical_sha256,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)
from sdsafe_fine_confirm import SCOPES, SPLIT_AUTHORITY
from sdsafe_fine_confirm.sources import load_cohort

from . import (
    CT_EVENT_TYPES,
    DATASET_VERSION,
    DEFAULT_DATASET_DIR,
    DEPENDENCY_SOURCE_FILES,
    E_EVENT_TYPES,
    EVENT_CONTENT_SEGMENT,
    EVENT_ROUND_END,
    EVENT_TOOL_CALL,
    EXPECTED_DEV_CT_EXACT,
    EXPECTED_DEV_CT_ROWS,
    EXPECTED_DEV_CT_UNKNOWN,
    EXPECTED_DEV_E_ROWS,
    EXPECTED_DEV_EPISODES,
    EXPECTED_DEV_GROUPS,
    EXPECTED_DEV_POSITIVES,
    EXPECTED_DEV_ROWS,
    EXPECTED_COMMON_DEV_EPISODES,
    EXPECTED_DIAGNOSTIC_DEV_EPISODES,
    EXPECTED_EXACT_PREFIX_POSITIVE_ROWS,
    EXPECTED_POLICY_COUNTS,
    EXPECTED_TRAIN_CT_EXACT,
    EXPECTED_TRAIN_CT_ROWS,
    EXPECTED_TRAIN_CT_UNKNOWN,
    EXPECTED_TRAIN_E_ROWS,
    EXPECTED_TRAIN_EPISODES,
    EXPECTED_TRAIN_GROUPS,
    EXPECTED_TRAIN_POSITIVES,
    FAMILY_CT,
    FAMILY_E,
    FEATURE_DTYPE,
    FINE_CONFIRM_FEATURES_ROOT,
    FUTURE_CONSUMER_MODULES,
    HIDDEN_SIZE,
    LABEL_EXACT,
    LABEL_UNKNOWN,
    MATERIALIZED_SPLITS,
    MATRIX_FILE,
    PARTITION_FRACTIONS,
    PARTITIONS,
    POLICIES,
    PRODUCER_FILES,
    REPAIR_DATA_ROOT,
    REPAIR_ROOT,
    ROW_BYTES,
    SCHEMA_VERSION,
    SPLIT_DEV,
    SPLIT_SEED,
    SPLIT_TRAIN,
    STD_FLOOR,
    STUDY_ID,
    DataContractError,
    DatasetError,
    assert_materialized_split,
)
from .labels import (
    MISSING,
    build_dev_label_arrays,
    build_train_label_arrays,
    checker_state,
    episode_label_support,
    event_order_key,
    is_exact,
    load_policy_assignments,
    summarise_policy_support,
)

MANIFEST_FILE = "manifest.json"
DATASET_SPEC_FILE = "dataset_spec.json"
SPLITS_FILE = "splits.json"
EPISODES_TRAIN_FILE = "episodes_train.jsonl"
EPISODES_DEV_FILE = "episodes_dev.jsonl"
ROWS_TRAIN_FILE = "rows_train.jsonl"
ROWS_DEV_FILE = "rows_dev.jsonl"
LABELS_TRAIN_FILE = "labels_train.npz"
LABELS_DEV_FILE = "labels_dev.npz"
GROUPS_FILE = "groups.jsonl"
AUDIT_FILE = "data_audit.json"
SOURCE_LEDGER_FILE = "source_ledger.json"
SCALER_DIR = "scalers"
FIT_SCALER_FILE = "fit_scaler.npz"
FULL_TRAIN_SCALER_FILE = "full_train_scaler.npz"
SCALER_AUDIT_FILE = "scaler_audit.json"
MATRIX_SHA256_FILE = "features_bf16.sha256"

INPUT_PATHS = {
    "repaired_train_rows": "artifacts/fine_label_repair_v1/data/repaired_train_rows.jsonl",
    "repaired_dev_rows": "artifacts/fine_label_repair_v1/data/repaired_dev_rows.jsonl",
    "episodes_train_dev": "artifacts/fine_label_repair_v1/data/episodes_train_dev.jsonl",
    "policy_assignments": "artifacts/fine_label_repair_v1/data/policies/assignments.jsonl",
    "policy_audit": "artifacts/fine_label_repair_v1/data/policies/audit.json",
    "repair_data_audit": "artifacts/fine_label_repair_v1/data/data_audit.json",
    "repair_scaler_source": "artifacts/fine_label_repair_v1/data/scaler_source.json",
}
FEATURE_INPUT_PATHS = {
    scope: {
        "manifest": f"artifacts/fine_confirm/features/{scope}/manifest.json",
        "index": f"artifacts/fine_confirm/features/{scope}/index.jsonl",
        "matrix": f"artifacts/fine_confirm/features/{scope}/matrix.bf16",
    }
    for scope in SCOPES
}

#: Explicit source/test ledger.  No future T/I module is included.
SOURCE_TEST_FILES = tuple(dict.fromkeys(PRODUCER_FILES + DEPENDENCY_SOURCE_FILES))


class FeatureProvider(Protocol):
    """The actual read boundary used by the builder."""

    def resolve_ref(self, ref: Mapping[str, Any], expected_split: str) -> "ResolvedFeature": ...

    def read_refs(self, refs: Sequence[Mapping[str, Any]], expected_split: str) -> np.ndarray: ...

    def round_end_metadata(self, split: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ResolvedFeature:
    physical_key: tuple[Any, ...]
    kind: str
    version: str
    matrix_path: Path
    matrix_total_rows: int
    feature_row: int
    scope: str
    event_id: str
    sequence_sha256: str
    feature_token_index: int


class FeatureStoreProvider:
    """Read-only adapter over the accepted FeatureStore."""

    def __init__(self, root: Path | str = ".") -> None:
        from sdsafe_fine_label_repair.features import FeatureStore

        self.root = Path(root)
        self.store = FeatureStore.open(self.root, require_complete=True)

    def resolve_ref(self, ref: Mapping[str, Any], expected_split: str) -> ResolvedFeature:
        physical = self.store.resolve_ref(ref, expected_split)
        return ResolvedFeature(
            physical_key=(
                str(physical.kind),
                str(physical.matrix_path),
                int(physical.total_rows),
                int(physical.feature_row),
            ),
            kind=str(physical.kind),
            version=str(physical.version),
            matrix_path=Path(physical.matrix_path),
            matrix_total_rows=int(physical.total_rows),
            feature_row=int(physical.feature_row),
            scope=str(physical.scope),
            event_id=str(physical.event_id),
            sequence_sha256=str(physical.sequence_sha256),
            feature_token_index=int(physical.feature_token_index),
        )

    def read_refs(self, refs: Sequence[Mapping[str, Any]], expected_split: str) -> np.ndarray:
        import torch

        tensor = self.store.read(refs, expected_split)
        return tensor.detach().cpu().to(torch.float32).numpy()

    def round_end_metadata(self, split: str) -> list[dict[str, Any]]:
        split = assert_materialized_split(split, where="round_end_metadata")
        rows: list[dict[str, Any]] = []
        for (_, _), old in sorted(self.store.old.by_round_end.items()):
            old_split = str(old.get("mixed_split") or old.get("split"))
            if old_split != split:
                continue
            rows.append(_old_round_end_row(old, split))
        return rows


# --------------------------------------------------------------------------- metadata normalisation


def _old_round_end_row(old: Mapping[str, Any], split: str) -> dict[str, Any]:
    event_type = str(old.get("event_type"))
    if event_type != EVENT_ROUND_END:
        raise DataContractError(f"{old.get('event_id')}: expected round_end, got {event_type!r}")
    namespaced = str(old.get("namespaced_episode_id"))
    event_id = str(old["event_id"])
    sequence_sha = str(old.get("round_sequence_sha256") or old.get("sequence_sha256"))
    feature_ref = {
        "kind": "cached_canonical_bf16",
        "scope": str(old["_scope"]),
        "split": split,
        "global_feature_row": int(old["_global_feature_row"]),
        "local_feature_row": int(old["_local_feature_row"]),
        "feature_file": str(old.get("feature_file") or "matrix.bf16"),
        "feature_byte_offset": int(old["feature_byte_offset"]),
        "feature_token_index": int(old["feature_token_index"]),
        "sequence_sha256": sequence_sha,
        "event_id": event_id,
        "hidden_size": HIDDEN_SIZE,
        "dtype": FEATURE_DTYPE,
        "status": "cached_read_only",
    }
    status = str(old.get("label_status"))
    if status not in (LABEL_EXACT, LABEL_UNKNOWN):
        raise DataContractError(f"{event_id}: bad round_end label_status {status!r}")
    return {
        "episode_id": namespaced,
        "namespaced_episode_id": namespaced,
        "event_id": event_id,
        "source_tag": str(old.get("source_tag")),
        "scope": str(old["_scope"]),
        "split": split,
        "source_split": old.get("source_split"),
        "group_id": str(old.get("group_id")),
        "group_hash": old.get("group_hash"),
        "strict_final": bool(old.get("strict_final")),
        "event_type": EVENT_ROUND_END,
        "round_id": int(old["round_id"]),
        "message_index": int(old.get("message_index", 0)),
        "index": int(old.get("index", 0)),
        "call_index": old.get("call_index"),
        "segment_index": old.get("segment_index"),
        "feature_token_index": int(old["feature_token_index"]),
        "token_start": old.get("token_start"),
        "token_end": old.get("token_end"),
        "prompt_length": old.get("prompt_length"),
        "sequence_length": old.get("sequence_length"),
        "sequence_sha256": sequence_sha,
        "round_sequence_sha256": sequence_sha,
        "terminal_token_id": old.get("terminal_token_id"),
        "inference_position": bool(old.get("inference_position", True)),
        "label_status": status,
        "label_value": old.get("label_value"),
        "prefix_security": old.get("prefix_security"),
        "previous_security": old.get("previous_security"),
        "security_flip": old.get("security_flip"),
        "repair_status": "old_round_end_reference",
        "repair_version": "fine_label_repair_v1",
        "supersedes_event_id": None,
        "feature_ref": feature_ref,
    }


def _normalise_source_row(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    event_id = str(row.get("event_id") or "")
    if not event_id:
        raise DataContractError("source row has no event_id")
    namespaced = str(row.get("namespaced_episode_id") or row.get("episode_id"))
    event_type = str(row.get("event_type"))
    if event_type not in CT_EVENT_TYPES + E_EVENT_TYPES:
        raise DataContractError(f"{event_id}: unsupported event_type {event_type!r}")
    if str(row.get("split")) != split:
        raise DataContractError(f"{event_id}: source split {row.get('split')!r} != requested {split!r}")
    status = str(row.get("label_status"))
    if status not in (LABEL_EXACT, LABEL_UNKNOWN):
        raise DataContractError(f"{event_id}: bad label_status {status!r}")
    ref = row.get("feature_ref")
    if not isinstance(ref, Mapping):
        raise DataContractError(f"{event_id}: source row has no feature_ref")
    token_index = row.get("feature_token_index")
    if token_index is None:
        raise DataContractError(f"{event_id}: source row has no feature_token_index")
    sequence_sha = row.get("sequence_sha256")
    if sequence_sha is None:
        raise DataContractError(f"{event_id}: source row has no sequence_sha256")
    return {
        "episode_id": namespaced,
        "namespaced_episode_id": namespaced,
        "event_id": event_id,
        "source_tag": row.get("source_tag"),
        "scope": str(row.get("scope")),
        "split": split,
        "source_split": row.get("source_split"),
        "group_id": str(row.get("group_id")),
        "group_hash": row.get("group_hash"),
        "strict_final": bool(row.get("strict_final")),
        "event_type": event_type,
        "round_id": int(row.get("round_id", 0)),
        "message_index": int(row.get("message_index", 0)),
        "index": int(row.get("index", 0)),
        "call_index": row.get("call_index"),
        "segment_index": row.get("segment_index"),
        "feature_token_index": int(token_index),
        "token_start": row.get("token_start"),
        "token_end": row.get("token_end"),
        "prompt_length": row.get("prompt_length"),
        "sequence_length": row.get("sequence_length"),
        "sequence_sha256": str(sequence_sha),
        "round_sequence_sha256": str(sequence_sha),
        "terminal_token_id": row.get("terminal_token_id"),
        "inference_position": bool(row.get("inference_position", True)),
        "label_status": status,
        "label_value": row.get("label_value"),
        "prefix_security": row.get("prefix_security"),
        "previous_security": row.get("previous_security"),
        "security_flip": row.get("security_flip"),
        "repair_status": row.get("repair_status"),
        "repair_version": row.get("repair_version"),
        "supersedes_event_id": row.get("supersedes_event_id"),
        "feature_ref": dict(ref),
    }


def _assert_episode_identity(row: Mapping[str, Any], episode: Any, split: str) -> None:
    event_id = str(row.get("event_id"))
    if str(row.get("split")) != split:
        raise DataContractError(f"{event_id}: row split is not {split!r}")
    if str(episode.split) != split:
        raise DataContractError(f"{event_id}: authoritative episode split is {episode.split!r}, not {split!r}")
    if str(row.get("namespaced_episode_id")) != episode.namespaced_episode_id:
        raise DataContractError(f"{event_id}: row episode identity differs from the authoritative episode")
    if str(row.get("scope")) != episode.scope:
        raise DataContractError(f"{event_id}: row scope differs from authoritative scope")
    if str(row.get("source_tag")) != episode.source_tag:
        raise DataContractError(f"{event_id}: row source_tag differs from authoritative source")
    if str(row.get("group_id")) != episode.group_id:
        raise DataContractError(f"{event_id}: row group_id differs from the authoritative group")
    if row.get("group_hash") is not None and episode.group_hash is not None:
        if str(row["group_hash"]) != str(episode.group_hash):
            raise DataContractError(f"{event_id}: row group_hash differs from authoritative group")
    if bool(row.get("strict_final")) is not bool(episode.strict_final):
        raise DataContractError(f"{event_id}: row strict_final differs from authoritative strict_final")


def _validate_resolved_ref(row: Mapping[str, Any], resolved: ResolvedFeature, split: str) -> None:
    event_id = str(row["event_id"])
    if resolved.event_id != event_id:
        raise DataContractError(f"{event_id}: physical ref resolved to another event {resolved.event_id!r}")
    if resolved.scope != str(row["scope"]):
        raise DataContractError(f"{event_id}: physical ref scope differs")
    if resolved.feature_token_index != int(row["feature_token_index"]):
        raise DataContractError(f"{event_id}: physical ref token index differs")
    if str(row["sequence_sha256"]) != str(resolved.sequence_sha256):
        raise DataContractError(f"{event_id}: physical ref sequence hash differs")
    if split not in MATERIALIZED_SPLITS:
        raise DataContractError(f"{event_id}: refusing non-TRAIN/DEV split {split!r}")


# --------------------------------------------------------------------------- group partition


def _group_seed_key(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{group_id}".encode("utf-8")).hexdigest()


def assign_group_partitions(
    episodes: Sequence[Any],
    *,
    seed: int = SPLIT_SEED,
    enforce_expected_counts: bool = True,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Deterministic whole-group FIT/CAL/SELECT allocation on TRAIN only.

    Rule (fixed before materialisation): groups are never split.  Targets are 80/10/10 of
    TRAIN episodes *and* TRAIN strict-final positives.  Groups are processed by descending
    episode count with ``sha256(seed:group_id)`` as the deterministic tie-break; each group
    goes to the partition minimising the worst relative episode/positive fill.  Only TRAIN
    group truth is used; DEV contributes nothing.
    """
    episode_list = [episode for episode in episodes if str(episode.split) == SPLIT_TRAIN]
    if enforce_expected_counts and len(episode_list) != EXPECTED_TRAIN_EPISODES:
        raise DataContractError(f"TRAIN episodes {len(episode_list)} != {EXPECTED_TRAIN_EPISODES}")
    by_group: dict[str, list[Any]] = defaultdict(list)
    for episode in episode_list:
        by_group[episode.group_id].append(episode)
    if enforce_expected_counts and len(by_group) != EXPECTED_TRAIN_GROUPS:
        raise DataContractError(f"TRAIN groups {len(by_group)} != {EXPECTED_TRAIN_GROUPS}")
    total_episodes = len(episode_list)
    total_positives = sum(1 for episode in episode_list if episode.strict_final)
    if enforce_expected_counts and total_positives != EXPECTED_TRAIN_POSITIVES:
        raise DataContractError(f"TRAIN positives {total_positives} != {EXPECTED_TRAIN_POSITIVES}")

    targets = {
        partition: [
            int(round(PARTITION_FRACTIONS[partition] * total_episodes)),
            int(round(PARTITION_FRACTIONS[partition] * total_positives)),
        ]
        for partition in PARTITIONS
    }
    targets["select"][0] = total_episodes - targets["fit"][0] - targets["cal"][0]
    targets["select"][1] = total_positives - targets["fit"][1] - targets["cal"][1]
    current = {partition: [0, 0] for partition in PARTITIONS}
    assigned: dict[str, list[str]] = {partition: [] for partition in PARTITIONS}

    ordered_groups = sorted(
        by_group,
        key=lambda group_id: (
            -len(by_group[group_id]),
            _group_seed_key(seed, group_id),
            group_id,
        ),
    )
    order_index = {partition: index for index, partition in enumerate(PARTITIONS)}
    for group_id in ordered_groups:
        group_episodes = by_group[group_id]
        size = len(group_episodes)
        positives = sum(1 for episode in group_episodes if episode.strict_final)

        def score(partition: str) -> tuple[float, int]:
            target_episodes, target_positives = targets[partition]
            episode_fill = (current[partition][0] + size) / max(target_episodes, 1)
            positive_fill = (current[partition][1] + positives) / max(target_positives, 1)
            return max(episode_fill, positive_fill), order_index[partition]

        chosen = min(PARTITIONS, key=score)
        assigned[chosen].append(group_id)
        current[chosen][0] += size
        current[chosen][1] += positives

    group_sets = [set(assigned[partition]) for partition in PARTITIONS]
    for left in range(len(PARTITIONS)):
        for right in range(left + 1, len(PARTITIONS)):
            if group_sets[left] & group_sets[right]:
                raise DataContractError("FIT/CAL/SELECT group sets overlap")
    if set().union(*group_sets) != set(by_group):
        raise DataContractError("FIT/CAL/SELECT does not cover every TRAIN group")
    for partition in PARTITIONS:
        assigned[partition] = sorted(assigned[partition])

    audit = {
        "seed": int(seed),
        "rule": (
            "whole-group greedy allocation; targets are 80/10/10 of TRAIN episodes and TRAIN "
            "strict-final positives; groups ordered by descending episode count then "
            "sha256(seed:group_id); partition chosen by minimising the worst relative fill; "
            "only TRAIN group truth is used"
        ),
        "targets": {partition: {"episodes": targets[partition][0], "positives": targets[partition][1]} for partition in PARTITIONS},
        "actual": {
            partition: {
                "groups": len(assigned[partition]),
                "episodes": sum(len(by_group[group]) for group in assigned[partition]),
                "positives": sum(
                    1 for group in assigned[partition] for episode in by_group[group] if episode.strict_final
                ),
                "negatives": sum(
                    1 for group in assigned[partition] for episode in by_group[group] if not episode.strict_final
                ),
            }
            for partition in PARTITIONS
        },
        "group_disjoint": True,
        "group_union_complete": True,
        "dev_used": False,
    }
    return assigned, audit


# --------------------------------------------------------------------------- split materialisation


@dataclass
class SplitBuild:
    split: str
    rows: list[dict[str, Any]]
    episodes: list[dict[str, Any]]
    unique_refs: list[tuple[dict[str, Any], ResolvedFeature]]
    cache_offset: int
    label_arrays: dict[str, np.ndarray]
    label_summary: dict[str, Any]


def _build_split(
    *,
    split: str,
    cohort_episodes: Sequence[Any],
    repaired_by_episode: Mapping[str, list[dict[str, Any]]],
    e_by_episode: Mapping[str, list[dict[str, Any]]],
    provider: FeatureProvider,
    cache_offset: int,
    row_offset: int,
    partition_by_episode: Mapping[str, str] | None,
    enforce_expected_counts: bool = True,
) -> SplitBuild:
    split = assert_materialized_split(split, where="build_split")
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    unique_refs: list[tuple[dict[str, Any], ResolvedFeature]] = []
    cache_by_key: dict[tuple[Any, ...], int] = {}

    for episode_index, episode in enumerate(sorted(cohort_episodes, key=lambda item: item.namespaced_episode_id)):
        source_rows = list(repaired_by_episode.get(episode.namespaced_episode_id, []))
        if split == SPLIT_TRAIN:
            source_rows = [row for row in source_rows if str(row.get("event_type")) in CT_EVENT_TYPES]
            extra_e = list(e_by_episode.get(episode.namespaced_episode_id, []))
        else:
            extra_e = []
            if any(str(row.get("event_type")) == EVENT_ROUND_END for row in source_rows):
                # Repair already includes E for DEV; never add a second copy.
                pass
        normalised: list[dict[str, Any]] = []
        for row in source_rows:
            item = _normalise_source_row(row, split)
            _assert_episode_identity(item, episode, split)
            normalised.append(item)
        for row in extra_e:
            item = _normalise_source_row(row, split)
            _assert_episode_identity(item, episode, split)
            normalised.append(item)
        if len({str(item["event_id"]) for item in normalised}) != len(normalised):
            raise DataContractError(f"{episode.namespaced_episode_id}: duplicate event_id in episode")
        normalised.sort(key=event_order_key)

        previous_token: int | None = None
        previous_round: int | None = None
        start = row_offset + len(rows)
        for item in normalised:
            resolved = provider.resolve_ref(item["feature_ref"], split)
            _validate_resolved_ref(item, resolved, split)
            key = resolved.physical_key
            if key not in cache_by_key:
                cache_by_key[key] = cache_offset + len(unique_refs)
                unique_refs.append((item["feature_ref"], resolved))
            cache_row = cache_by_key[key]
            token_index = int(item["feature_token_index"])
            round_id = int(item["round_id"])
            item["row_index"] = row_offset + len(rows)
            item["cache_row"] = int(cache_row)
            item["event_family"] = FAMILY_CT if item["event_type"] in CT_EVENT_TYPES else FAMILY_E
            item["ct_mask"] = bool(item["event_family"] == FAMILY_CT)
            item["token_gap"] = 0 if previous_token is None else max(0, token_index - previous_token)
            item["round_gap"] = 0 if previous_round is None else max(0, round_id - previous_round)
            previous_token = token_index
            previous_round = round_id
            matrix_rel = _relative_to(resolved.matrix_path, _ROOT_FOR_BUILD)
            item["physical_ref"] = {
                "kind": resolved.kind,
                "version": resolved.version,
                "scope": resolved.scope,
                "event_id": resolved.event_id,
                "matrix_path": matrix_rel,
                "matrix_total_rows": int(resolved.matrix_total_rows),
                "feature_row": int(resolved.feature_row),
                "feature_token_index": int(resolved.feature_token_index),
                "sequence_sha256": resolved.sequence_sha256,
                "global_feature_row": item["feature_ref"].get("global_feature_row"),
                "local_feature_row": item["feature_ref"].get("local_feature_row"),
                "feature_file": item["feature_ref"].get("feature_file"),
                "feature_byte_offset": item["feature_ref"].get("feature_byte_offset"),
            }
            rows.append(item)
        end = row_offset + len(rows)
        if start == end:
            # Empty-position episodes are retained explicitly; padding never becomes real.
            pass
        episode_record = {
            "episode_index": int(episode_index),
            "episode_id": episode.namespaced_episode_id,
            "scope": episode.scope,
            "source_tag": episode.source_tag,
            "group_id": episode.group_id,
            "group_hash": episode.group_hash,
            "split": split,
            "source_split": episode.source_split,
            "strict_final": bool(episode.strict_final),
            "bag_label": int(bool(episode.strict_final)),
            "partition": None if partition_by_episode is None else partition_by_episode.get(episode.namespaced_episode_id),
            "row_start": int(start),
            "row_end": int(end),
            "length": int(end - start),
            "ct_count": int(sum(1 for item in normalised if item["ct_mask"])),
            "e_count": int(sum(1 for item in normalised if not item["ct_mask"])),
            "zero_ct": bool(all(not item["ct_mask"] for item in normalised)),
        }
        episodes.append(episode_record)

    if enforce_expected_counts and len(episodes) != (
        EXPECTED_TRAIN_EPISODES if split == SPLIT_TRAIN else EXPECTED_DEV_EPISODES
    ):
        raise DataContractError(f"{split}: episode count {len(episodes)} is not the authoritative count")
    if len({episode["episode_id"] for episode in episodes}) != len(episodes):
        raise DataContractError(f"{split}: duplicate episode id")

    label_arrays, label_summary = _build_labels_for_split(split, rows)
    # Fill per-episode support counts.
    for episode in episodes:
        start, end = int(episode["row_start"]) - row_offset, int(episode["row_end"]) - row_offset
        support = _episode_support_from_arrays(
            rows=rows[start:end],
            label_arrays=label_arrays,
            start=start,
            end=end,
            bag_label=bool(episode["bag_label"]),
        )
        episode.update(support)
    return SplitBuild(
        split=split,
        rows=rows,
        episodes=episodes,
        unique_refs=unique_refs,
        cache_offset=cache_offset,
        label_arrays=label_arrays,
        label_summary=label_summary,
    )


def _build_labels_for_split(split: str, rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if split == SPLIT_TRAIN:
        ct_positions = [index for index, row in enumerate(rows) if bool(row["ct_mask"])]
        ct_rows = [rows[index] for index in ct_positions]
        assignments = load_policy_assignments(_ROOT_FOR_BUILD)
        ct_arrays, summary = build_train_label_arrays(ct_rows, assignments)
        arrays: dict[str, np.ndarray] = {}
        n = len(rows)
        for policy in POLICIES:
            labels = np.full(n, MISSING, dtype=np.int8)
            masks = np.ones(n, dtype=np.bool_)
            labels[ct_positions] = ct_arrays[f"label_{policy}"]
            masks[ct_positions] = ct_arrays[f"mask_{policy}"]
            arrays[f"label_{policy}"] = labels
            arrays[f"mask_{policy}"] = masks
        arrays["prefix_state"] = np.asarray([checker_state(row)[0] for row in rows], dtype=np.int8)
        arrays["previous_state"] = np.asarray([checker_state(row)[1] for row in rows], dtype=np.int8)
        arrays["security_flip"] = np.asarray([checker_state(row)[2] for row in rows], dtype=np.int8)
        arrays["exact"] = np.asarray([is_exact(row) for row in rows], dtype=np.bool_)
        # CT state arrays must equal the accepted semantic arrays; E state stays separate.
        for key in ("prefix_state", "previous_state", "security_flip", "exact"):
            if not np.array_equal(arrays[key][ct_positions], ct_arrays[key]):
                raise DataContractError(f"TRAIN CT {key} differs between population and full arrays")
        summary["kind"] = "train_policy_labels"
        summary["e_rows_context_only"] = True
        summary["e_policy_masked"] = int(sum(1 for row in rows if not row["ct_mask"]))
        summary["support"] = summarise_policy_support(arrays, arrays, arrays["exact"])
        return arrays, summary
    if split == SPLIT_DEV:
        arrays, summary = build_dev_label_arrays(rows)
        summary["kind"] = "dev_truth_no_policy"
        return arrays, summary
    raise DataContractError(f"unsupported split {split!r}")


def _episode_support_from_arrays(
    *,
    rows: Sequence[Mapping[str, Any]],
    label_arrays: Mapping[str, np.ndarray],
    start: int,
    end: int,
    bag_label: bool,
) -> dict[str, Any]:
    ct_mask = np.asarray([bool(row["ct_mask"]) for row in rows], dtype=np.bool_)
    prefix = np.asarray(label_arrays["prefix_state"][start:end], dtype=np.int8)
    flips = np.asarray(label_arrays["security_flip"][start:end], dtype=np.int8)
    policy_labels = {
        policy: label_arrays[f"label_{policy}"][start:end]
        for policy in POLICIES
        if f"label_{policy}" in label_arrays
    }
    policy_masks = {
        policy: label_arrays[f"mask_{policy}"][start:end]
        for policy in POLICIES
        if f"mask_{policy}" in label_arrays
    }
    support = episode_label_support(
        n_positions=len(rows),
        episode_ct_mask=ct_mask,
        episode_prefix_state=prefix,
        episode_security_flip=flips,
        episode_bag_label=bag_label,
        policy_labels=policy_labels or None,
        policy_masks=policy_masks or None,
    )
    out = support.to_dict()
    out["has_exact_prefix_positive"] = bool((prefix == 1).any())
    out["has_unknown"] = bool((prefix == MISSING).any())
    out["label_status_unknown_positions"] = int((prefix == MISSING).sum())
    out["positive_support"] = bool(bag_label and (prefix == 1).any())
    out["transient_or_conflict"] = bool((not bag_label) and (prefix == 1).any())
    return out


# --------------------------------------------------------------------------- matrix/scaler helpers


def _relative_to(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _float32_to_bf16_bytes(array: np.ndarray) -> bytes:
    import torch

    if array.ndim != 2 or array.shape[1] != HIDDEN_SIZE:
        raise DataContractError(f"feature array shape {array.shape} is not (N,{HIDDEN_SIZE})")
    values = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(values).all():
        raise DataContractError("feature array contains non-finite values")
    tensor = torch.from_numpy(values).to(torch.bfloat16).contiguous()
    return tensor.view(torch.uint16).numpy().tobytes()


def _bf16_uint16_to_float32(array: np.ndarray) -> np.ndarray:
    import torch

    values = np.ascontiguousarray(array, dtype=np.uint16)
    return torch.from_numpy(values).view(torch.bfloat16).float().numpy()


def _write_unique_refs(
    handle: Any,
    *,
    provider: FeatureProvider,
    split: str,
    unique_refs: Sequence[tuple[dict[str, Any], ResolvedFeature]],
    hasher: Any,
    chunk_rows: int,
    total_written_rows: int,
) -> int:
    for start in range(0, len(unique_refs), chunk_rows):
        chunk = unique_refs[start : start + chunk_rows]
        refs = [ref for ref, _ in chunk]
        matrix = provider.read_refs(refs, split)
        if matrix.shape != (len(refs), HIDDEN_SIZE):
            raise DataContractError(f"{split}: feature read shape {matrix.shape} is wrong")
        payload = _float32_to_bf16_bytes(matrix)
        if len(payload) != len(refs) * ROW_BYTES:
            raise DataContractError(f"{split}: BF16 payload length is wrong")
        handle.write(payload)
        hasher.update(payload)
        total_written_rows += len(refs)
    return total_written_rows


def _read_matrix_rows(matrix: np.memmap, cache_rows: Sequence[int]) -> np.ndarray:
    indices = np.asarray(cache_rows, dtype=np.int64)
    if indices.size == 0:
        return np.empty((0, HIDDEN_SIZE), dtype=np.float32)
    return _bf16_uint16_to_float32(np.asarray(matrix[indices]))


def _fit_scaler_from_matrix(
    matrix_path: Path,
    *,
    total_rows: int,
    cache_rows: Sequence[int],
    chunk_rows: int = 8192,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    rows = np.asarray(cache_rows, dtype=np.int64)
    if rows.size == 0:
        raise DatasetError("scaler population is empty")
    matrix = np.memmap(matrix_path, dtype=np.uint16, mode="r", shape=(int(total_rows), HIDDEN_SIZE))
    total = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    total_sq = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    nonfinite = 0
    for start in range(0, rows.size, chunk_rows):
        chunk = rows[start : start + chunk_rows]
        values = np.asarray(matrix[chunk])
        floats = _bf16_uint16_to_float32(values).astype(np.float64)
        finite = np.isfinite(floats)
        nonfinite += int((~finite).sum())
        if not finite.all():
            raise DatasetError("scaler population contains non-finite feature values")
        total += floats.sum(axis=0)
        total_sq += (floats * floats).sum(axis=0)
    count = float(rows.size)
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), STD_FLOOR)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise DatasetError("scaler statistics are non-finite")
    mean32 = mean.astype(np.float32)
    std32 = std.astype(np.float32)
    audit = {
        "rows": int(rows.size),
        "dtype": "float32",
        "accumulation": "float64 streaming sum/sumsq over BF16->float32 rows",
        "eps": STD_FLOOR,
        "nonfinite_values": nonfinite,
        "mean_sha256": sha256_bytes(mean32.tobytes()),
        "std_sha256": sha256_bytes(std32.tobytes()),
        "min_std": float(std32.min()),
        "max_std": float(std32.max()),
        "mean_abs_max": float(np.abs(mean32).max()),
    }
    return mean32, std32, audit


def _save_scaler(path: Path, mean: np.ndarray, std: np.ndarray, rows: int, meta: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            mean=np.ascontiguousarray(mean, dtype=np.float32),
            std=np.ascontiguousarray(std, dtype=np.float32),
            fit_rows=np.asarray(int(rows), dtype=np.int64),
            eps=np.asarray(float(STD_FLOOR), dtype=np.float64),
        )
    os.replace(tmp, path)
    record = dict(meta)
    record.update(
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
            "mean_sha256": sha256_bytes(mean.tobytes()),
            "std_sha256": sha256_bytes(std.tobytes()),
            "fit_rows": int(rows),
            "eps": STD_FLOOR,
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        }
    )
    return record


def _scaler_population_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(str(row["event_id"]).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


# --------------------------------------------------------------------------- builder


_ROOT_FOR_BUILD: Path = Path(".")


def build_sequence_dataset(
    root: Path | str = ".",
    *,
    dataset_dir: Path | str | None = None,
    force: bool = False,
    chunk_rows: int = 4096,
    provider: FeatureProvider | None = None,
    verify: bool = True,
) -> dict[str, Any]:
    """Materialise the immutable sequence data version once; safe no-op when complete."""
    global _ROOT_FOR_BUILD
    root_path = Path(root).resolve()
    _ROOT_FOR_BUILD = root_path
    out_dir = (
        Path(dataset_dir)
        if dataset_dir is not None
        else root_path / "artifacts/fine_sequence_explore_v1/data/sequence_v1"
    )
    if not out_dir.is_absolute():
        out_dir = root_path / out_dir
    manifest_path = out_dir / MANIFEST_FILE
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        audit_record = existing.get("audit") or {}
        complete_accepted = (
            existing.get("status") == "complete"
            and audit_record.get("status") in ("PASS", "SKIPPED")
        )
        if complete_accepted:
            if force:
                raise DataContractError(
                    "refusing to overwrite a complete accepted sequence dataset; "
                    "use a new dataset_dir/version"
                )
            # The no-op path must prove actual artifact integrity and write nothing.
            open_sequence_dataset(root_path, dataset_dir=out_dir, verify_integrity=True)
            return {
                "status": "complete",
                "noop": True,
                "dataset_dir": str(out_dir),
                "manifest": existing,
                "note": "complete dataset verified and left byte-for-byte unchanged",
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    input_hashes = _input_hashes(root_path)
    source_ledger = _source_ledger(root_path)

    cohort = _load_authoritative_cohort(root_path)
    train_episodes = [episode for episode in cohort.values() if episode.split == SPLIT_TRAIN]
    dev_episodes = [episode for episode in cohort.values() if episode.split == SPLIT_DEV]
    if len(train_episodes) != EXPECTED_TRAIN_EPISODES or len(dev_episodes) != EXPECTED_DEV_EPISODES:
        raise DataContractError("authoritative cohort episode counts drifted")
    partition_groups, partition_audit = assign_group_partitions(train_episodes)
    partition_by_episode: dict[str, str] = {}
    for partition, group_ids in partition_groups.items():
        group_set = set(group_ids)
        for episode in train_episodes:
            if episode.group_id in group_set:
                partition_by_episode[episode.namespaced_episode_id] = partition
    if len(partition_by_episode) != EXPECTED_TRAIN_EPISODES:
        raise DataContractError("partition assignment did not cover every TRAIN episode")

    repaired_train = read_jsonl(root_path / INPUT_PATHS["repaired_train_rows"])
    repaired_dev = read_jsonl(root_path / INPUT_PATHS["repaired_dev_rows"])
    repaired_train_by_episode = _group_rows_by_episode(repaired_train, SPLIT_TRAIN)
    repaired_dev_by_episode = _group_rows_by_episode(repaired_dev, SPLIT_DEV)
    _assert_repaired_population(repaired_train_by_episode, train_episodes, SPLIT_TRAIN)
    _assert_repaired_population(repaired_dev_by_episode, dev_episodes, SPLIT_DEV)

    owns_provider = provider is None
    provider = provider or FeatureStoreProvider(root_path)
    try:
        train_e_rows = provider.round_end_metadata(SPLIT_TRAIN)
        if len(train_e_rows) != EXPECTED_TRAIN_E_ROWS:
            raise DataContractError(f"TRAIN E rows {len(train_e_rows)} != {EXPECTED_TRAIN_E_ROWS}")
        e_by_episode = _group_rows_by_episode(train_e_rows, SPLIT_TRAIN)
        if set(e_by_episode) - {episode.namespaced_episode_id for episode in train_episodes}:
            raise DataContractError("TRAIN E rows reference an episode outside the authoritative TRAIN cohort")

        matrix_tmp = out_dir / (MATRIX_FILE + ".tmp")
        matrix_hasher = hashlib.sha256()
        with matrix_tmp.open("wb") as handle:
            train_build = _build_split(
                split=SPLIT_TRAIN,
                cohort_episodes=train_episodes,
                repaired_by_episode=repaired_train_by_episode,
                e_by_episode=e_by_episode,
                provider=provider,
                cache_offset=0,
                row_offset=0,
                partition_by_episode=partition_by_episode,
            )
            written = _write_unique_refs(
                handle,
                provider=provider,
                split=SPLIT_TRAIN,
                unique_refs=train_build.unique_refs,
                hasher=matrix_hasher,
                chunk_rows=chunk_rows,
                total_written_rows=0,
            )
            if written != len(train_build.unique_refs):
                raise DataContractError("TRAIN matrix row count mismatch")
            dev_build = _build_split(
                split=SPLIT_DEV,
                cohort_episodes=dev_episodes,
                repaired_by_episode=repaired_dev_by_episode,
                e_by_episode={},
                provider=provider,
                cache_offset=len(train_build.unique_refs),
                row_offset=len(train_build.rows),
                partition_by_episode=None,
            )
            written = _write_unique_refs(
                handle,
                provider=provider,
                split=SPLIT_DEV,
                unique_refs=dev_build.unique_refs,
                hasher=matrix_hasher,
                chunk_rows=chunk_rows,
                total_written_rows=written,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(matrix_tmp, out_dir / MATRIX_FILE)
    except Exception:
        raise

    total_cache_rows = len(train_build.unique_refs) + len(dev_build.unique_refs)
    if total_cache_rows * ROW_BYTES != (out_dir / MATRIX_FILE).stat().st_size:
        raise DataContractError("written matrix size does not match the cache row count")

    all_rows = train_build.rows + dev_build.rows
    _assert_row_populations(train_build, dev_build)
    _write_jsonl_stream(out_dir / ROWS_TRAIN_FILE, (_row_to_json(row) for row in train_build.rows))
    _write_jsonl_stream(out_dir / ROWS_DEV_FILE, (_row_to_json(row) for row in dev_build.rows))
    _write_jsonl_stream(out_dir / EPISODES_TRAIN_FILE, train_build.episodes)
    _write_jsonl_stream(out_dir / EPISODES_DEV_FILE, dev_build.episodes)
    _save_npz(out_dir / LABELS_TRAIN_FILE, train_build.label_arrays)
    _save_npz(out_dir / LABELS_DEV_FILE, dev_build.label_arrays)
    _write_jsonl_stream(out_dir / GROUPS_FILE, _group_rows(train_build.episodes, dev_build.episodes, partition_by_episode))
    splits = _splits_payload(train_build, dev_build, partition_groups, partition_audit, cohort)
    atomic_write_json(out_dir / SPLITS_FILE, splits)

    fit_rows = [
        int(row["cache_row"])
        for row in train_build.rows
        if bool(row["ct_mask"])
        and partition_by_episode.get(str(row["episode_id"])) == "fit"
    ]
    full_train_rows = [int(row["cache_row"]) for row in train_build.rows if bool(row["ct_mask"])]
    if len(fit_rows) == 0 or len(full_train_rows) != EXPECTED_TRAIN_CT_ROWS:
        raise DataContractError("scaler populations are empty or the wrong size")
    fit_population_rows = [
        row
        for row in train_build.rows
        if bool(row["ct_mask"]) and partition_by_episode.get(str(row["episode_id"])) == "fit"
    ]
    full_population_rows = [row for row in train_build.rows if bool(row["ct_mask"])]

    fit_mean, fit_std, fit_audit = _fit_scaler_from_matrix(
        out_dir / MATRIX_FILE, total_rows=total_cache_rows, cache_rows=fit_rows, chunk_rows=chunk_rows
    )
    fit_mean_2, fit_std_2, _ = _fit_scaler_from_matrix(
        out_dir / MATRIX_FILE, total_rows=total_cache_rows, cache_rows=fit_rows, chunk_rows=chunk_rows
    )
    full_mean, full_std, full_audit = _fit_scaler_from_matrix(
        out_dir / MATRIX_FILE, total_rows=total_cache_rows, cache_rows=full_train_rows, chunk_rows=chunk_rows
    )
    full_mean_2, full_std_2, _ = _fit_scaler_from_matrix(
        out_dir / MATRIX_FILE, total_rows=total_cache_rows, cache_rows=full_train_rows, chunk_rows=chunk_rows
    )
    if not (np.array_equal(fit_mean, fit_mean_2) and np.array_equal(fit_std, fit_std_2)):
        raise DataContractError("FIT scaler is not reproducible")
    if not (np.array_equal(full_mean, full_mean_2) and np.array_equal(full_std, full_std_2)):
        raise DataContractError("FULL-TRAIN scaler is not reproducible")
    fit_audit.update(
        {
            "name": "fit_scaler",
            "fit_split": SPLIT_TRAIN,
            "population": "FIT partition CT rows including unknown",
            "population_hash": _scaler_population_hash(fit_population_rows),
            "population_rows": len(fit_population_rows),
            "population_unknown_rows": int(sum(1 for row in fit_population_rows if not is_exact(row))),
            "reproducible_refit": True,
        }
    )
    full_audit.update(
        {
            "name": "full_train_scaler",
            "fit_split": SPLIT_TRAIN,
            "population": "all TRAIN CT rows including unknown",
            "population_hash": _scaler_population_hash(full_population_rows),
            "population_rows": len(full_population_rows),
            "population_unknown_rows": int(sum(1 for row in full_population_rows if not is_exact(row))),
            "reproducible_refit": True,
        }
    )
    fit_record = _save_scaler(out_dir / SCALER_DIR / FIT_SCALER_FILE, fit_mean, fit_std, len(fit_rows), fit_audit)
    full_record = _save_scaler(
        out_dir / SCALER_DIR / FULL_TRAIN_SCALER_FILE, full_mean, full_std, len(full_train_rows), full_audit
    )
    scaler_audit = {
        "status": "PASS",
        "fit": fit_record,
        "full_train": full_record,
        "fit_only_for_internal_selection": True,
        "dev_in_fit_scaler": False,
        "cal_or_select_in_fit_scaler": False,
        "std_floor": STD_FLOOR,
        "rule": "fit/validation scalers use only their named TRAIN population; DEV is never fitted",
    }
    atomic_write_json(out_dir / SCALER_DIR / SCALER_AUDIT_FILE, scaler_audit)

    matrix_sha = matrix_hasher.hexdigest()
    (out_dir / MATRIX_SHA256_FILE).write_text(matrix_sha + "\n", encoding="utf-8")
    dataset_spec = _dataset_spec()
    atomic_write_json(out_dir / DATASET_SPEC_FILE, dataset_spec)
    atomic_write_json(out_dir / SOURCE_LEDGER_FILE, source_ledger)

    counts = _counts_payload(train_build, dev_build, partition_by_episode, total_cache_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "dataset_version": DATASET_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": {
            "split": SPLIT_AUTHORITY,
            "episodes": "sdsafe_fine_confirm.sources.load_cohort",
            "features": "sdsafe_fine_label_repair.features.FeatureStore.open(require_complete=True)",
            "labels": "accepted repaired rows + policies/assignments.jsonl",
        },
        "split_seed": SPLIT_SEED,
        "matrix": {
            "file": MATRIX_FILE,
            "dtype": FEATURE_DTYPE,
            "shape": [int(total_cache_rows), HIDDEN_SIZE],
            "row_bytes": ROW_BYTES,
            "sha256": matrix_sha,
            "row_order": "TRAIN unique physical rows then DEV unique physical rows, first-occurrence order",
        },
        "counts": counts,
        "partition": partition_audit,
        "inputs": input_hashes,
        "source_ledger": source_ledger,
        "files": {},
    }
    manifest["files"] = _file_hashes(
        out_dir,
        [
            MATRIX_SHA256_FILE,
            DATASET_SPEC_FILE,
            SPLITS_FILE,
            EPISODES_TRAIN_FILE,
            EPISODES_DEV_FILE,
            ROWS_TRAIN_FILE,
            ROWS_DEV_FILE,
            LABELS_TRAIN_FILE,
            LABELS_DEV_FILE,
            GROUPS_FILE,
            SOURCE_LEDGER_FILE,
            f"{SCALER_DIR}/{FIT_SCALER_FILE}",
            f"{SCALER_DIR}/{FULL_TRAIN_SCALER_FILE}",
            f"{SCALER_DIR}/{SCALER_AUDIT_FILE}",
        ],
    )
    atomic_write_json(manifest_path, manifest)

    audit = None
    if verify:
        try:
            audit = verify_dataset(root_path, dataset_dir=out_dir, provider=provider)
        except Exception:
            manifest_path.unlink(missing_ok=True)
            (out_dir / AUDIT_FILE).unlink(missing_ok=True)
            raise
        atomic_write_json(out_dir / AUDIT_FILE, audit)
        # Recompute the manifest file map so data_audit.json is covered too.
        manifest["files"] = _file_hashes(
            out_dir,
            list(manifest["files"].keys()) + [AUDIT_FILE],
        )
        manifest["audit"] = {"status": audit["status"], "file": AUDIT_FILE, "sha256": manifest["files"][AUDIT_FILE]["sha256"]}
        atomic_write_json(manifest_path, manifest)
    else:
        manifest["audit"] = {"status": "SKIPPED"}
        atomic_write_json(manifest_path, manifest)
    return {
        "status": "complete",
        "noop": False,
        "dataset_dir": str(out_dir),
        "manifest": manifest,
        "audit": audit,
        "elapsed_seconds": round(time.time() - started, 2),
    }


# --------------------------------------------------------------------------- builder helpers


def _load_authoritative_cohort(root: Path) -> dict[str, Any]:
    episodes: dict[str, Any] = {}
    seen_test = 0
    for scope in SCOPES:
        cohort = load_cohort(root, scope)
        for episode in cohort.episodes:
            key = episode.namespaced_episode_id
            if key in episodes:
                raise DataContractError(f"duplicate namespaced episode in authoritative cohort: {key}")
            episodes[key] = episode
            if episode.split not in MATERIALIZED_SPLITS:
                seen_test += 1
    if len(episodes) != EXPECTED_TRAIN_EPISODES + EXPECTED_DEV_EPISODES + seen_test:
        raise DataContractError("authoritative cohort total is inconsistent")
    train = [episode for episode in episodes.values() if episode.split == SPLIT_TRAIN]
    dev = [episode for episode in episodes.values() if episode.split == SPLIT_DEV]
    common_dev = [episode for episode in dev if episode.scope in ("cot", "nothink", "ctx40k")]
    diagnostic_dev = [episode for episode in dev if episode.scope == "agentlab"]
    if len(common_dev) != EXPECTED_COMMON_DEV_EPISODES or len(diagnostic_dev) != EXPECTED_DIAGNOSTIC_DEV_EPISODES:
        raise DataContractError("DEV common/diagnostic cohort counts drifted")
    if sum(1 for episode in dev if episode.strict_final) != EXPECTED_DEV_POSITIVES:
        raise DataContractError("DEV positive count drifted")
    if len({episode.group_id for episode in dev}) != EXPECTED_DEV_GROUPS:
        raise DataContractError("DEV group count drifted")
    train_groups = {episode.group_id for episode in train}
    dev_groups = {episode.group_id for episode in dev}
    if train_groups & dev_groups:
        raise DataContractError("TRAIN and DEV groups overlap")
    return episodes


def _group_rows_by_episode(rows: Sequence[Mapping[str, Any]], split: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("split")) != split:
            raise DataContractError(f"{row.get('event_id')}: row split is not {split!r}")
        event_type = str(row.get("event_type"))
        if event_type not in CT_EVENT_TYPES + E_EVENT_TYPES:
            raise DataContractError(f"{row.get('event_id')}: unsupported event_type {event_type!r}")
        key = str(row.get("namespaced_episode_id") or row.get("episode_id"))
        grouped[key].append(dict(row))
    return grouped


def _assert_repaired_population(
    grouped: Mapping[str, list[dict[str, Any]]],
    cohort_episodes: Sequence[Any],
    split: str,
) -> None:
    expected = {episode.namespaced_episode_id for episode in cohort_episodes}
    if set(grouped) != expected:
        missing = len(expected - set(grouped))
        extra = len(set(grouped) - expected)
        raise DataContractError(f"{split}: repaired rows episode set differs from authority: missing={missing} extra={extra}")
    for episode_id, rows in grouped.items():
        event_ids = [str(row["event_id"]) for row in rows]
        if len(event_ids) != len(set(event_ids)):
            raise DataContractError(f"{episode_id}: duplicate event_id")
        if split == SPLIT_TRAIN and any(str(row["event_type"]) == EVENT_ROUND_END for row in rows):
            raise DataContractError(f"{episode_id}: repaired TRAIN unexpectedly contains round_end")
        if split == SPLIT_DEV and not any(str(row["event_type"]) == EVENT_ROUND_END for row in rows):
            raise DataContractError(f"{episode_id}: repaired DEV is missing the old round_end E context")


def _assert_row_populations(train_build: SplitBuild, dev_build: SplitBuild) -> None:
    train_ct = sum(1 for row in train_build.rows if row["ct_mask"])
    train_e = sum(1 for row in train_build.rows if not row["ct_mask"])
    dev_ct = sum(1 for row in dev_build.rows if row["ct_mask"])
    dev_e = sum(1 for row in dev_build.rows if not row["ct_mask"])
    assert_counts = {
        "train_ct": (train_ct, EXPECTED_TRAIN_CT_ROWS),
        "train_e": (train_e, EXPECTED_TRAIN_E_ROWS),
        "dev_ct": (dev_ct, EXPECTED_DEV_CT_ROWS),
        "dev_e": (dev_e, EXPECTED_DEV_E_ROWS),
        "dev_rows": (len(dev_build.rows), EXPECTED_DEV_ROWS),
    }
    for name, (actual, expected) in assert_counts.items():
        if int(actual) != int(expected):
            raise DataContractError(f"{name} {actual} != {expected}")
    train_exact = sum(1 for row in train_build.rows if row["ct_mask"] and is_exact(row))
    train_unknown = sum(1 for row in train_build.rows if row["ct_mask"] and not is_exact(row))
    dev_exact = sum(1 for row in dev_build.rows if row["ct_mask"] and is_exact(row))
    dev_unknown = sum(1 for row in dev_build.rows if row["ct_mask"] and not is_exact(row))
    if train_exact != EXPECTED_TRAIN_CT_EXACT or train_unknown != EXPECTED_TRAIN_CT_UNKNOWN:
        raise DataContractError("TRAIN CT exact/unknown population drifted")
    if dev_exact != EXPECTED_DEV_CT_EXACT or dev_unknown != EXPECTED_DEV_CT_UNKNOWN:
        raise DataContractError("DEV CT exact/unknown population drifted")


def _row_to_json(row: Mapping[str, Any]) -> dict[str, Any]:
    # Explicitly drop helper keys that are not part of the public row schema.
    return dict(row)


def _write_jsonl_stream(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    tmp = path.with_suffix(path.suffix + ".tmp")
    hasher = hashlib.sha256()
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write(payload + "\n")
            hasher.update(payload.encode("utf-8"))
            hasher.update(b"\n")
            count += 1
    os.replace(tmp, path)
    return {"sha256": hasher.hexdigest(), "bytes": path.stat().st_size, "rows": count}


def _save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **{key: np.ascontiguousarray(value) for key, value in sorted(arrays.items())})
    os.replace(tmp, path)
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _group_rows(
    train_episodes: Sequence[Mapping[str, Any]],
    dev_episodes: Sequence[Mapping[str, Any]],
    partition_by_episode: Mapping[str, str],
) -> Iterator[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for episode in list(train_episodes) + list(dev_episodes):
        grouped[(str(episode["split"]), str(episode["group_id"]))].append(episode)
    for (split, group_id), episodes in sorted(grouped.items()):
        partitions = {str(episode.get("partition")) for episode in episodes if episode.get("partition")}
        yield {
            "split": split,
            "group_id": group_id,
            "group_hash": episodes[0].get("group_hash"),
            "partition": next(iter(partitions)) if len(partitions) == 1 else None,
            "episodes": len(episodes),
            "positives": sum(1 for episode in episodes if episode["bag_label"] == 1),
            "negatives": sum(1 for episode in episodes if episode["bag_label"] == 0),
            "by_scope": dict(sorted(Counter(str(episode["scope"]) for episode in episodes).items())),
            "episode_ids": sorted(str(episode["episode_id"]) for episode in episodes),
        }


def _splits_payload(
    train_build: SplitBuild,
    dev_build: SplitBuild,
    partition_groups: Mapping[str, Sequence[str]],
    partition_audit: Mapping[str, Any],
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    def summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "episodes": len(episodes),
            "positives": sum(1 for episode in episodes if episode["bag_label"] == 1),
            "negatives": sum(1 for episode in episodes if episode["bag_label"] == 0),
            "groups": len({str(episode["group_id"]) for episode in episodes}),
            "zero_ct_episodes": sum(1 for episode in episodes if episode.get("zero_ct")),
            "by_scope": dict(sorted(Counter(str(episode["scope"]) for episode in episodes).items())),
        }

    train_episode_ids = sorted(str(episode["episode_id"]) for episode in train_build.episodes)
    dev_episode_ids = sorted(str(episode["episode_id"]) for episode in dev_build.episodes)
    partitions = {}
    for partition in PARTITIONS:
        episodes = [episode for episode in train_build.episodes if episode.get("partition") == partition]
        partitions[partition] = {
            "groups": sorted(partition_groups[partition]),
            "episode_ids": sorted(str(episode["episode_id"]) for episode in episodes),
            "summary": summary(episodes),
        }
    partition_group_sets = {partition: set(partition_groups[partition]) for partition in PARTITIONS}
    group_disjoint = {}
    for left in range(len(PARTITIONS)):
        for right in range(left + 1, len(PARTITIONS)):
            a, b = PARTITIONS[left], PARTITIONS[right]
            group_disjoint[f"{a}_{b}"] = len(partition_group_sets[a] & partition_group_sets[b])
    test_episodes = [episode for episode in cohort.values() if episode.split == "test"]
    dev_groups = {str(episode["group_id"]) for episode in dev_build.episodes}
    test_groups = {episode.group_id for episode in test_episodes}
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": SPLIT_SEED,
        "authority": SPLIT_AUTHORITY,
        "rule": partition_audit["rule"],
        "partition_targets": partition_audit["targets"],
        "partition_actual": partition_audit["actual"],
        "train": {
            "episode_ids": train_episode_ids,
            "groups": sorted({str(episode["group_id"]) for episode in train_build.episodes}),
            "summary": summary(train_build.episodes),
        },
        "dev": {
            "episode_ids": dev_episode_ids,
            "groups": sorted({str(episode["group_id"]) for episode in dev_build.episodes}),
            "summary": summary(dev_build.episodes),
        },
        "partitions": partitions,
        "group_disjoint": group_disjoint,
        "traindev_group_overlap": len({str(episode["group_id"]) for episode in train_build.episodes} & dev_groups),
        "test": {
            "materialized_episodes": 0,
            "materialized_rows": 0,
            "vectors_opened": 0,
            "group_overlap_with_dev": len(dev_groups & test_groups),
            "note": "TEST was used only as authoritative cohort metadata for group-overlap accounting; no raw trace/vector/prediction was opened",
        },
        "empty_position_episodes_retained": sum(1 for episode in train_build.episodes + dev_build.episodes if episode["length"] == 0),
    }


def _counts_payload(
    train_build: SplitBuild,
    dev_build: SplitBuild,
    partition_by_episode: Mapping[str, str],
    total_cache_rows: int,
) -> dict[str, Any]:
    def split_counts(build: SplitBuild) -> dict[str, Any]:
        by_scope = Counter(str(row["scope"]) for row in build.rows)
        by_type = Counter(str(row["event_type"]) for row in build.rows)
        by_partition = Counter(
            partition_by_episode.get(str(row["episode_id"]), "none") for row in build.rows
        )
        return {
            "episodes": len(build.episodes),
            "rows": len(build.rows),
            "ct_rows": sum(1 for row in build.rows if row["ct_mask"]),
            "e_rows": sum(1 for row in build.rows if not row["ct_mask"]),
            "unique_cache_rows": len(build.unique_refs),
            "rows_by_scope": dict(sorted(by_scope.items())),
            "rows_by_event_type": dict(sorted(by_type.items())),
            "rows_by_partition": dict(sorted(by_partition.items())),
            "episode_bag_positives": sum(1 for episode in build.episodes if episode["bag_label"] == 1),
            "episode_zero_ct": sum(1 for episode in build.episodes if episode["zero_ct"]),
            "episodes_with_unknown": sum(1 for episode in build.episodes if episode.get("has_unknown")),
            "episodes_with_prefix_true": sum(1 for episode in build.episodes if episode.get("has_exact_prefix_positive")),
            "episodes_transient_prefix_true": sum(1 for episode in build.episodes if episode.get("transient_prefix_true")),
            "label_summary": build.label_summary,
        }

    train_counts = split_counts(train_build)
    dev_counts = split_counts(dev_build)
    train_counts["policy_expected"] = EXPECTED_POLICY_COUNTS
    train_counts["policy_actual_from_assignments"] = train_build.label_summary.get("per_policy", {})
    return {
        "train": train_counts,
        "dev": dev_counts,
        "total_rows": len(train_build.rows) + len(dev_build.rows),
        "total_unique_cache_rows": int(total_cache_rows),
        "matrix_shape": [int(total_cache_rows), HIDDEN_SIZE],
        "test_materialized": 0,
    }


def _dataset_spec() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "matrix": {
            "file": MATRIX_FILE,
            "dtype": "bfloat16 stored as little-endian uint16",
            "shape": ["unique_cache_rows", HIDDEN_SIZE],
            "row_bytes": ROW_BYTES,
            "standardize": "read BF16 rows on demand; scaler transforms to float32 when requested",
        },
        "episode_metadata": {
            "files": [EPISODES_TRAIN_FILE, EPISODES_DEV_FILE],
            "fields": {
                "episode_id": "namespaced authoritative episode id",
                "split": "train|dev",
                "group_id/group_hash": "authoritative whole-group identity",
                "strict_final": "official strict-final episode truth (bag label)",
                "bag_label": "1/0 strict-final MIL bag target; not a per-position label",
                "partition": "fit|cal|select for TRAIN, null for DEV",
                "row_start/row_end": "global row offsets in the split row files",
                "ct_count/e_count": "position family counts; empty-position episodes retained",
                "policy_positive/policy_negative/policy_masked": "per-policy support counts",
                "transient_prefix_true": "final-negative episode with any exact prefix-true row",
            },
        },
        "row_metadata": {
            "files": [ROWS_TRAIN_FILE, ROWS_DEV_FILE],
            "fields": {
                "row_index": "global sequence row index",
                "cache_row": "row in features_bf16.bin",
                "event_id/episode_id/scope/split": "namespaced identity",
                "event_family": "ct|e",
                "ct_mask": "True for tool_call/content_segment, False for round_end E context",
                "round_id/message_index/index/token geometry": "chronological identity",
                "token_gap/round_gap": "bounded causal metadata only",
                "feature_ref": "original accepted producer reference",
                "physical_ref": "resolved physical matrix path/row/token/hash",
            },
        },
        "labels": {
            "train": {
                "file": LABELS_TRAIN_FILE,
                "arrays": {
                    "label_L0..L4": "int8, -1=masked/unknown, 0/1 exact target",
                    "mask_L0..L4": "bool, True=not in loss",
                    "prefix_state": "int8 exact checker bool or -1 unknown",
                    "previous_state": "int8 exact checker bool or -1 unknown",
                    "security_flip": "int8 exact flip bool or -1 unknown",
                    "exact": "bool label_status==exact",
                },
                "e_rows": "context-only: policy labels/masks are -1/True",
            },
            "dev": {
                "file": LABELS_DEV_FILE,
                "arrays": ["prefix_state", "previous_state", "security_flip", "exact"],
                "note": "no training policy on DEV",
            },
        },
        "scalers": {
            "fit": f"{SCALER_DIR}/{FIT_SCALER_FILE}",
            "full_train": f"{SCALER_DIR}/{FULL_TRAIN_SCALER_FILE}",
            "arrays": ["mean[4096] float32", "std[4096] float32", "fit_rows", "eps"],
            "fit_population": "FIT partition CT rows including unknown, label-blind",
            "full_train_population": "all TRAIN CT rows including unknown",
        },
        "read_only": True,
        "no_test_data": True,
    }


def _file_hashes(root: Path, relatives: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            raise DatasetError(f"expected materialised file is missing: {path}")
        out[str(relative)] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return out


def _input_hashes(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, relative in INPUT_PATHS.items():
        path = root / relative
        if not path.is_file():
            raise DataContractError(f"required producer input is missing: {path}")
        out[name] = {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
    for scope, paths in FEATURE_INPUT_PATHS.items():
        for name, relative in paths.items():
            path = root / relative
            if not path.is_file():
                raise DataContractError(f"required feature input is missing: {path}")
            out[f"{scope}_{name}"] = {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
    return out


def _source_ledger(root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative in SOURCE_TEST_FILES:
        path = root / relative
        if not path.is_file():
            raise DataContractError(f"producer/test source file is missing: {path}")
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "producer_and_dependencies_only": True,
        "future_consumer_modules_excluded": list(FUTURE_CONSUMER_MODULES),
        "files": files,
        "ledger_sha256": canonical_sha256(files),
    }


# --------------------------------------------------------------------------- verification


def verify_dataset(
    root: Path | str = ".",
    *,
    dataset_dir: Path | str | None = None,
    provider: FeatureProvider | None = None,
    sample_rows: int = 256,
) -> dict[str, Any]:
    """Read-only materialised-data audit; no writer side effects."""
    root_path = Path(root).resolve()
    dataset = open_sequence_dataset(root_path, dataset_dir=dataset_dir, verify_integrity=True)
    manifest = dataset.manifest
    errors: list[str] = []
    counts = manifest["counts"]
    if counts["train"]["episodes"] != EXPECTED_TRAIN_EPISODES:
        errors.append("TRAIN episode count")
    if counts["dev"]["episodes"] != EXPECTED_DEV_EPISODES:
        errors.append("DEV episode count")
    if counts["train"]["ct_rows"] != EXPECTED_TRAIN_CT_ROWS:
        errors.append("TRAIN CT count")
    if counts["train"]["e_rows"] != EXPECTED_TRAIN_E_ROWS:
        errors.append("TRAIN E count")
    if counts["dev"]["ct_rows"] != EXPECTED_DEV_CT_ROWS:
        errors.append("DEV CT count")
    if counts["dev"]["e_rows"] != EXPECTED_DEV_E_ROWS:
        errors.append("DEV E count")
    if errors:
        raise DatasetError(f"dataset population audit failed: {errors}")

    # Re-resolve every physical ref through the actual provider boundary and compare.
    owns_provider = provider is None
    provider = provider or FeatureStoreProvider(root_path)
    try:
        mismatches: list[dict[str, Any]] = []
        for split in MATERIALIZED_SPLITS:
            for row in dataset.rows_for_split(split):
                resolved = provider.resolve_ref(row["feature_ref"], split)
                physical = row["physical_ref"]
                expected = {
                    "kind": resolved.kind,
                    "version": resolved.version,
                    "scope": resolved.scope,
                    "event_id": resolved.event_id,
                    "matrix_path": _relative_to(resolved.matrix_path, root_path),
                    "matrix_total_rows": int(resolved.matrix_total_rows),
                    "feature_row": int(resolved.feature_row),
                    "feature_token_index": int(resolved.feature_token_index),
                    "sequence_sha256": resolved.sequence_sha256,
                }
                for key, value in expected.items():
                    if str(physical.get(key)) != str(value):
                        mismatches.append({"event_id": row["event_id"], "field": key})
                        if len(mismatches) >= 10:
                            break
                if len(mismatches) >= 10:
                    break
            if len(mismatches) >= 10:
                break
        if mismatches:
            raise DatasetError(f"physical ref identity audit failed: {mismatches[:3]}")

        # Bitwise sample read-back from the immutable source through the same boundary.
        all_rows = dataset.rows
        if sample_rows > 0 and all_rows:
            stride = max(1, len(all_rows) // int(sample_rows))
            sampled = all_rows[::stride][: int(sample_rows)]
            sample_checks = []
            by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in MATERIALIZED_SPLITS}
            for row in sampled:
                by_split[row["split"]].append(row)
            for split, rows in by_split.items():
                if not rows:
                    continue
                refs = [row["feature_ref"] for row in rows]
                source = provider.read_refs(refs, split)
                cache_rows = [int(row["cache_row"]) for row in rows]
                new_values = _read_matrix_rows(dataset._matrix, cache_rows)
                source_bf16 = _float32_to_bf16_bytes(source)
                new_bf16 = _float32_to_bf16_bytes(new_values)
                if source_bf16 != new_bf16:
                    raise DatasetError(f"{split}: sampled matrix rows are not bitwise equal to the source")
                sample_checks.append({"split": split, "rows": len(rows), "bitwise_equal": True})
        else:
            sample_checks = []

        # Deterministic norm sample from the materialised matrix.
        norm_sample = dataset.rows[:: max(1, len(dataset.rows) // 512)][:512]
        norm_values = _read_matrix_rows(dataset._matrix, [int(row["cache_row"]) for row in norm_sample])
        norms = np.linalg.norm(norm_values.astype(np.float64), axis=1)
        audit = {
            "status": "PASS",
            "dataset_dir": str(dataset.dataset_dir),
            "manifest_at_audit_input_sha256": sha256_file(dataset.dataset_dir / MANIFEST_FILE),
            "matrix_sha256": manifest["matrix"]["sha256"],
            "counts": counts,
            "policy_counts": manifest["counts"]["train"]["policy_actual_from_assignments"],
            "expected_policy_counts": EXPECTED_POLICY_COUNTS,
            "physical_ref_rows_checked": len(dataset.rows),
            "physical_ref_mismatches": 0,
            "matrix_sample_checks": sample_checks,
            "norm_sample": {
                "rows": int(norms.size),
                "min": float(norms.min()) if norms.size else None,
                "max": float(norms.max()) if norms.size else None,
                "mean": float(norms.mean()) if norms.size else None,
            },
            "partition": manifest["partition"],
            "test_materialized": 0,
            "read_only": True,
        }
        if not np.isfinite(norms).all():
            raise DatasetError("norm sample contains non-finite values")
        return audit
    finally:
        pass


# --------------------------------------------------------------------------- read-only dataset API


@dataclass
class EpisodeBatch:
    features: Any
    padding_mask: Any
    ct_mask: Any
    event_family: Any
    round_id: Any
    token_index: Any
    token_gap: Any
    round_gap: Any
    lengths: Any
    policy_labels: dict[str, Any]
    policy_masks: dict[str, Any]
    prefix_state: Any
    exact: Any
    bag_label: Any
    episode_ids: list[str]
    split: str
    partition: str | None
    scaler_name: str | None


def _validate_collate_partitions(
    episodes: Sequence[Mapping[str, Any]],
    *,
    allow_mixed_train_partitions: bool,
) -> str | None:
    """Validate partition mixing; mixed TRAIN is explicit opt-in only."""
    split = str(episodes[0]["split"])
    partitions = {episode.get("partition") for episode in episodes}
    if allow_mixed_train_partitions:
        if split != SPLIT_TRAIN:
            raise DatasetError("allow_mixed_train_partitions is only valid for authoritative TRAIN batches")
        if len(partitions) > 1:
            return "mixed"
        value = next(iter(partitions), None)
        return None if value is None else str(value)
    if len(partitions) != 1:
        raise DatasetError(
            "cannot mix FIT/CAL/SELECT partitions in one batch; "
            "pass allow_mixed_train_partitions=True only for authoritative TRAIN refits"
        )
    value = next(iter(partitions))
    return None if value is None else str(value)


class SequenceDataset:
    """Strictly read-only view over a completed sequence data version."""

    def __init__(
        self,
        root: Path | str,
        dataset_dir: Path | str | None = None,
        *,
        verify_integrity: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.dataset_dir = _resolve_dataset_dir(self.root, dataset_dir)
        self.manifest = _load_manifest(self.dataset_dir, verify_integrity=verify_integrity)
        self.episodes_train = read_jsonl(self.dataset_dir / EPISODES_TRAIN_FILE)
        self.episodes_dev = read_jsonl(self.dataset_dir / EPISODES_DEV_FILE)
        self.episodes = self.episodes_train + self.episodes_dev
        self.rows_train = read_jsonl(self.dataset_dir / ROWS_TRAIN_FILE)
        self.rows_dev = read_jsonl(self.dataset_dir / ROWS_DEV_FILE)
        self.rows = self.rows_train + self.rows_dev
        self._split_row_offsets = {SPLIT_TRAIN: 0, SPLIT_DEV: len(self.rows_train)}
        self.labels_train = _load_npz(self.dataset_dir / LABELS_TRAIN_FILE)
        self.labels_dev = _load_npz(self.dataset_dir / LABELS_DEV_FILE)
        self.splits = read_json(self.dataset_dir / SPLITS_FILE)
        self._episodes_by_split = {SPLIT_TRAIN: self.episodes_train, SPLIT_DEV: self.episodes_dev}
        self._rows_by_split = {SPLIT_TRAIN: self.rows_train, SPLIT_DEV: self.rows_dev}
        self._validate_metadata()
        self._scalers = {
            "fit": _load_scaler(self.dataset_dir / SCALER_DIR / FIT_SCALER_FILE),
            "full_train": _load_scaler(self.dataset_dir / SCALER_DIR / FULL_TRAIN_SCALER_FILE),
        }
        matrix_path = self.dataset_dir / MATRIX_FILE
        expected_rows = int(self.manifest["matrix"]["shape"][0])
        if matrix_path.stat().st_size != expected_rows * ROW_BYTES:
            raise DatasetError("matrix byte size does not match the manifest shape")
        self._matrix = np.memmap(matrix_path, dtype=np.uint16, mode="r", shape=(expected_rows, HIDDEN_SIZE))
        self._partition_indices = self._build_partition_indices()
        self._episode_index_by_split = {
            split: [index for index, episode in enumerate(self.episodes) if episode["split"] == split]
            for split in MATERIALIZED_SPLITS
        }

    # -- identity / read-only metadata -------------------------------------

    def _unique_cache_rows(self) -> set[int]:
        return {int(row["cache_row"]) for row in self.rows}

    def _validate_metadata(self) -> None:
        if len(self.episodes_train) != EXPECTED_TRAIN_EPISODES:
            raise DatasetError("TRAIN episode metadata count mismatch")
        if len(self.episodes_dev) != EXPECTED_DEV_EPISODES:
            raise DatasetError("DEV episode metadata count mismatch")
        if len(self.rows_train) != EXPECTED_TRAIN_CT_ROWS + EXPECTED_TRAIN_E_ROWS:
            raise DatasetError("TRAIN row metadata count mismatch")
        if len(self.rows_dev) != EXPECTED_DEV_ROWS:
            raise DatasetError("DEV row metadata count mismatch")
        expected_index = 0
        for row in self.rows:
            if int(row["row_index"]) != expected_index:
                raise DatasetError("row_index is not contiguous")
            if str(row["split"]) not in MATERIALIZED_SPLITS:
                raise DatasetError(f"row {row['event_id']} has a non-TRAIN/DEV split")
            if bool(row["ct_mask"]) != (str(row["event_family"]) == FAMILY_CT):
                raise DatasetError(f"row {row['event_id']} CT mask/family mismatch")
            if int(row["cache_row"]) < 0:
                raise DatasetError(f"row {row['event_id']} has a negative cache row")
            expected_index += 1
        for episode in self.episodes:
            start, end = int(episode["row_start"]), int(episode["row_end"])
            if start < 0 or end < start or end > len(self.rows):
                raise DatasetError(f"{episode['episode_id']}: bad row offsets")
            if int(episode["length"]) != end - start:
                raise DatasetError(f"{episode['episode_id']}: length does not match offsets")
        partition_sets = {
            partition: {episode["group_id"] for episode in self.episodes_train if episode.get("partition") == partition}
            for partition in PARTITIONS
        }
        for left in range(len(PARTITIONS)):
            for right in range(left + 1, len(PARTITIONS)):
                if partition_sets[PARTITIONS[left]] & partition_sets[PARTITIONS[right]]:
                    raise DatasetError("FIT/CAL/SELECT group sets overlap")
        self._labels = {SPLIT_TRAIN: self.labels_train, SPLIT_DEV: self.labels_dev}
        for split, labels in self._labels.items():
            n = len(self._rows_by_split[split])
            for key, array in labels.items():
                if int(array.shape[0]) != n:
                    raise DatasetError(f"{split} label array {key} length mismatch")
        for policy in POLICIES:
            if f"label_{policy}" not in self.labels_train or f"mask_{policy}" not in self.labels_train:
                raise DatasetError(f"TRAIN labels missing policy {policy}")
            labels = self.labels_train[f"label_{policy}"]
            masks = self.labels_train[f"mask_{policy}"]
            if np.any((labels < 0) & ~masks):
                raise DatasetError(f"{policy}: -1 label is not masked")
            if np.any((labels >= 0) & masks):
                raise DatasetError(f"{policy}: labelled row is masked")

    def _build_partition_indices(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {partition: [] for partition in PARTITIONS}
        for index, episode in enumerate(self.episodes_train):
            partition = episode.get("partition")
            if partition in out:
                out[partition].append(index)
        return out

    # -- public metadata ----------------------------------------------------

    @property
    def matrix(self) -> np.memmap:
        return self._matrix

    def row(self, row_index: int) -> dict[str, Any]:
        return self.rows[int(row_index)]

    def episode(self, episode_index: int) -> dict[str, Any]:
        return self.episodes[int(episode_index)]

    def rows_for_split(self, split: str) -> list[dict[str, Any]]:
        split = assert_materialized_split(split, where="rows_for_split")
        return self._rows_by_split[split]

    def split_episode_indices(self, name: str) -> list[int]:
        if name in MATERIALIZED_SPLITS:
            return list(self._episode_index_by_split[name])
        if name in PARTITIONS:
            return list(self._partition_indices[name])
        raise DatasetError(f"unknown split/partition {name!r}")

    def split_episodes(self, name: str) -> list[dict[str, Any]]:
        return [self.episodes[index] for index in self.split_episode_indices(name)]

    def scaler(self, name: str = "fit") -> "Scaler":
        if name not in self._scalers:
            raise DatasetError(f"unknown scaler {name!r}; choose {sorted(self._scalers)}")
        return self._scalers[name]

    def label_arrays(self, split: str) -> dict[str, np.ndarray]:
        split = assert_materialized_split(split, where="label_arrays")
        return self._labels[split]

    def label_summary(self) -> dict[str, Any]:
        return self.manifest["counts"]["train"]["label_summary"]

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "manifest": self.manifest,
            "splits": self.splits,
            "episode_counts": {
                split: len(self._episodes_by_split[split]) for split in MATERIALIZED_SPLITS
            },
            "row_counts": {split: len(self._rows_by_split[split]) for split in MATERIALIZED_SPLITS},
            "partitions": {partition: len(indices) for partition, indices in self._partition_indices.items()},
        }

    # -- tensor API ---------------------------------------------------------

    def read_rows(self, row_indices: Sequence[int], *, dtype: str = "float32") -> Any:
        import torch

        rows = [self.rows[int(index)] for index in row_indices]
        values = _read_matrix_rows(self._matrix, [int(row["cache_row"]) for row in rows])
        tensor = torch.from_numpy(values)
        if dtype == "float32":
            return tensor
        if dtype == "bfloat16":
            return tensor.to(torch.bfloat16)
        raise DatasetError(f"unsupported dtype {dtype!r}")

    def read_episode(self, episode_index: int, *, dtype: str = "float32") -> Any:
        episode = self.episodes[int(episode_index)]
        return self.read_rows(range(int(episode["row_start"]), int(episode["row_end"])), dtype=dtype)

    def collate_episodes(
        self,
        episode_indices: Sequence[int],
        *,
        dtype: str = "bfloat16",
        scaled: bool = False,
        scaler: str = "fit",
        include_labels: bool = True,
        include_causal_meta: bool = True,
        allow_mixed_train_partitions: bool = False,
    ) -> EpisodeBatch:
        """Collate a batch of whole episodes; padding is never a real position.

        ``padding_mask`` is TRUE for a REAL valid token, the opposite of PyTorch's
        ``key_padding_mask`` convention.  ``include_labels=False`` makes the batch
        gold-free: policy labels/masks are empty and bag/prefix/exact are ``None``.
        Mixed FIT/CAL/SELECT batches are refused unless
        ``allow_mixed_train_partitions=True`` and every episode is authoritative TRAIN.
        An all-empty episode batch returns one structural PAD slot with no valid
        positions, preserving zero-position episodes in the denominator.

        Returns unscaled raw BF16 by default.  Set ``scaled=True`` to apply a named
        scaler to float32 values.  Causal metadata is returned separately and contains
        no future length/end tag.
        """
        import torch

        indices = [int(index) for index in episode_indices]
        if not indices:
            raise DatasetError("cannot collate an empty episode batch")
        episodes = [self.episodes[index] for index in indices]
        split = str(episodes[0]["split"])
        if any(str(episode["split"]) != split for episode in episodes):
            raise DatasetError("cannot mix TRAIN and DEV episodes in one batch")
        partition = _validate_collate_partitions(
            episodes,
            allow_mixed_train_partitions=bool(allow_mixed_train_partitions),
        )
        lengths = [int(episode["length"]) for episode in episodes]
        max_len = max(lengths)
        if max_len == 0:
            # Zero-position episodes stay in the denominator.  One structural PAD slot
            # gives a valid tensor while padding_mask stays all False.
            max_len = 1
        feature_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32 if dtype == "float32" else None
        if feature_dtype is None:
            raise DatasetError(f"unsupported dtype {dtype!r}")
        features = torch.zeros((len(episodes), max_len, HIDDEN_SIZE), dtype=feature_dtype)
        padding_mask = torch.zeros((len(episodes), max_len), dtype=torch.bool)
        ct_mask = torch.zeros((len(episodes), max_len), dtype=torch.bool)
        event_family = torch.zeros((len(episodes), max_len), dtype=torch.int64)
        round_id = torch.zeros((len(episodes), max_len), dtype=torch.int64)
        token_index = torch.zeros((len(episodes), max_len), dtype=torch.int64)
        token_gap = torch.zeros((len(episodes), max_len), dtype=torch.int64)
        round_gap = torch.zeros((len(episodes), max_len), dtype=torch.int64)
        policy_labels = {
            policy: torch.full((len(episodes), max_len), MISSING, dtype=torch.int8)
            for policy in POLICIES
            if split == SPLIT_TRAIN and f"label_{policy}" in self.labels_train
        }
        policy_masks = {
            policy: torch.ones((len(episodes), max_len), dtype=torch.bool)
            for policy in POLICIES
            if split == SPLIT_TRAIN and f"mask_{policy}" in self.labels_train
        }
        prefix_state = torch.full((len(episodes), max_len), MISSING, dtype=torch.int8)
        exact = torch.zeros((len(episodes), max_len), dtype=torch.bool)
        for batch_index, episode in enumerate(episodes):
            start, end = int(episode["row_start"]), int(episode["row_end"])
            length = end - start
            if length == 0:
                continue
            row_indices = list(range(start, end))
            values = self.read_rows(row_indices, dtype=dtype)
            features[batch_index, :length] = values
            padding_mask[batch_index, :length] = True
            for offset, row in enumerate(self.rows[start:end]):
                ct_mask[batch_index, offset] = bool(row["ct_mask"])
                event_family[batch_index, offset] = 0 if row["event_family"] == FAMILY_CT else 1
                round_id[batch_index, offset] = int(row["round_id"])
                token_index[batch_index, offset] = int(row["feature_token_index"])
                token_gap[batch_index, offset] = int(row["token_gap"])
                round_gap[batch_index, offset] = int(row["round_gap"])
            labels = self._labels[split]
            label_start = start - self._split_row_offsets[split]
            label_end = end - self._split_row_offsets[split]
            if label_start < 0 or label_end > len(labels["prefix_state"]):
                raise DatasetError(f"{episode['episode_id']}: label slice is outside the split label arrays")
            prefix_state[batch_index, :length] = torch.from_numpy(labels["prefix_state"][label_start:label_end].copy())
            exact[batch_index, :length] = torch.from_numpy(labels["exact"][label_start:label_end].copy())
            if include_labels:
                for policy in policy_labels:
                    policy_labels[policy][batch_index, :length] = torch.from_numpy(
                        labels[f"label_{policy}"][label_start:label_end].copy()
                    )
                    policy_masks[policy][batch_index, :length] = torch.from_numpy(
                        labels[f"mask_{policy}"][label_start:label_end].copy()
                    )
        bag_label = torch.tensor([float(episode["bag_label"]) for episode in episodes], dtype=torch.float32)
        scaler_name = None
        if scaled:
            scaler_obj = self.scaler(scaler)
            scaler_name = scaler
            features = scaler_obj.transform(features.float()).to(feature_dtype)
            features = features * padding_mask.unsqueeze(-1).to(feature_dtype)
        result = EpisodeBatch(
            features=features,
            padding_mask=padding_mask,
            ct_mask=ct_mask,
            event_family=event_family if include_causal_meta else None,
            round_id=round_id if include_causal_meta else None,
            token_index=token_index if include_causal_meta else None,
            token_gap=token_gap if include_causal_meta else None,
            round_gap=round_gap if include_causal_meta else None,
            lengths=torch.tensor(lengths, dtype=torch.int64),
            policy_labels=policy_labels if include_labels else {},
            policy_masks=policy_masks if include_labels else {},
            prefix_state=prefix_state if include_labels else None,
            exact=exact if include_labels else None,
            bag_label=bag_label if include_labels else None,
            episode_ids=[str(episode["episode_id"]) for episode in episodes],
            split=split,
            partition=partition,
            scaler_name=scaler_name,
        )
        return result

    def iter_batches(
        self,
        name: str,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 0,
        **collate_kwargs: Any,
    ) -> Iterator[EpisodeBatch]:
        """Yield episode-batch minibatches.

        Collate options pass through, including ``allow_mixed_train_partitions=True`` for
        an intentional all-TRAIN refit.  Episodes are never split across batches.
        """
        if int(batch_size) <= 0:
            raise DatasetError("batch_size must be positive")
        indices = self.split_episode_indices(name)
        if shuffle:
            order = np.random.default_rng(int(seed)).permutation(len(indices))
            indices = [indices[int(position)] for position in order]
        for start in range(0, len(indices), int(batch_size)):
            yield self.collate_episodes(indices[start : start + int(batch_size)], **collate_kwargs)


class Scaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray, fit_rows: int, eps: float) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.fit_rows = int(fit_rows)
        self.eps = float(eps)
        if self.mean.shape != (HIDDEN_SIZE,) or self.std.shape != (HIDDEN_SIZE,):
            raise DatasetError("scaler arrays have the wrong shape")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise DatasetError("scaler arrays are non-finite")
        if np.any(self.std < self.eps):
            raise DatasetError("scaler std is below the declared floor")

    def transform(self, features: Any) -> Any:
        import torch

        if isinstance(features, np.ndarray):
            return (features.astype(np.float32) - self.mean) / self.std
        tensor = features.to(torch.float32)
        mean = torch.from_numpy(self.mean).to(tensor.device)
        std = torch.from_numpy(self.std).to(tensor.device)
        return (tensor - mean) / std

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "std": self.std.copy(),
            "fit_rows": self.fit_rows,
            "eps": self.eps,
            "mean_sha256": sha256_bytes(self.mean.tobytes()),
            "std_sha256": sha256_bytes(self.std.tobytes()),
        }


def _load_scaler(path: Path) -> Scaler:
    if not path.is_file():
        raise DatasetError(f"scaler file is missing: {path}")
    payload = np.load(path, allow_pickle=False)
    return Scaler(
        mean=payload["mean"],
        std=payload["std"],
        fit_rows=int(payload["fit_rows"]),
        eps=float(payload["eps"]),
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise DatasetError(f"label file is missing: {path}")
    payload = np.load(path, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


def _resolve_dataset_dir(root: Path, dataset_dir: Path | str | None) -> Path:
    if dataset_dir is None:
        return root / "artifacts/fine_sequence_explore_v1/data/sequence_v1"
    path = Path(dataset_dir)
    return path if path.is_absolute() else root / path


def _load_manifest(dataset_dir: Path, *, verify_integrity: bool) -> dict[str, Any]:
    path = dataset_dir / MANIFEST_FILE
    if not path.is_file():
        raise DatasetError(f"sequence dataset manifest is missing: {path}")
    manifest = read_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError("sequence dataset schema version mismatch")
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise DatasetError("sequence dataset version mismatch")
    if manifest.get("status") != "complete":
        raise DatasetError(f"sequence dataset is not complete: {manifest.get('status')!r}")
    if verify_integrity:
        for relative, record in manifest.get("files", {}).items():
            file_path = dataset_dir / relative
            if not file_path.is_file():
                raise DatasetError(f"manifest file is missing: {relative}")
            if int(file_path.stat().st_size) != int(record["bytes"]):
                raise DatasetError(f"manifest byte size mismatch: {relative}")
            if sha256_file(file_path) != str(record["sha256"]):
                raise DatasetError(f"manifest sha256 mismatch: {relative}")
        matrix_record = manifest.get("matrix") or {}
        matrix_path = dataset_dir / str(matrix_record.get("file"))
        if not matrix_path.is_file():
            raise DatasetError("feature matrix is missing")
        if sha256_file(matrix_path) != str(matrix_record.get("sha256")):
            raise DatasetError("feature matrix sha256 mismatch")
    return manifest


def open_sequence_dataset(
    root: Path | str = ".",
    *,
    dataset_dir: Path | str | None = None,
    verify_integrity: bool = True,
) -> SequenceDataset:
    """Open the completed sequence dataset read-only.  Writes nothing."""
    return SequenceDataset(root, dataset_dir, verify_integrity=verify_integrity)


__all__ = [
    "FeatureProvider",
    "FeatureStoreProvider",
    "ResolvedFeature",
    "Scaler",
    "SequenceDataset",
    "EpisodeBatch",
    "SplitBuild",
    "assign_group_partitions",
    "build_sequence_dataset",
    "open_sequence_dataset",
    "verify_dataset",
]
