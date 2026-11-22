"""Lane F: read-only feature store plus the authorized 439-endpoint supplement.

Two physical sources are kept strictly apart:

``cached_canonical_bf16``
    The immutable historical ``artifacts/fine_confirm/features/{scope}/matrix.bf16``
    aggregates.  A legacy reference is always resolved through the old per-scope index,
    after the old index metadata has itself been filtered through the accepted D TRAIN/DEV
    authority.  A TEST row can never be materialised, even if a forged reference claims
    ``split=train``.

``supplemental_bf16_v1``
    The single new BF16 matrix in ``artifacts/fine_label_repair_v1/features/cot``.  It
    contains exactly the 439 accepted D plan endpoints plus one fresh ``round_end``
    reference row per forwarded round (338), used only for old-vs-new geometry.  The
    supplemental reference key is the D ``feature_key``; there is no implicit replacement
    of an old row by a new row with the same logical event id.

The reader is CPU-only and returns ``torch.float32`` rows.  The producer forwards one
archived CoT round once (batch 1, BF16, causal, no generation) and writes one append-only
BF16 matrix plus atomic per-round journals.  It never writes the historical files and it
does not store all-token or all-layer activations.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from sdsafe_features.common import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_sha256,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
)
from sdsafe_fine_label_repair import (
    ARTIFACT_ROOT,
    DATA_ROOT,
    FEATURE_DTYPE,
    HIDDEN_SIZE,
    LEGACY_FEATURE_KIND,
    SPLIT_DEV,
    SPLIT_TRAIN,
    STUDY_ID,
    SUPPLEMENTAL_FEATURE_KIND,
    DataContractError,
    assert_protected_split,
)
from sdsafe_fine_label_repair.data import validate_feature_ref

STAGE = "fine_label_repair.features"
SCHEMA_VERSION = 1
SCOPE = "cot"
SOURCE_TAG = "cot32k"
SCOPES = ("cot", "nothink", "ctx40k", "agentlab")

MATRIX_FILE = "matrix.bf16"
INDEX_FILE = "index.jsonl"
JOURNAL_DIR = "journal"
CONFIG_FILE = "config.json"
MANIFEST_FILE = "manifest.json"
PROGRESS_FILE = "progress.json"
VALIDATION_FILE = "validation.json"
GEOMETRY_FILE = "geometry.json"
FAILURES_FILE = "failures.jsonl"

ROW_BYTES = HIDDEN_SIZE * 2
ENDPOINT_ROLE = "endpoint"
REFERENCE_ROLE = "round_end_reference"

EXPECTED_ENDPOINTS = 439
EXPECTED_ROUNDS = 338
EXPECTED_SPLIT_ENDPOINTS = {"train": 350, "dev": 89}
EXPECTED_REPAIR_STATUS = {"utf8_corrected": 328, "candidate_new_position": 111}
EXPECTED_SEQUENCE_MIN = 1843
EXPECTED_SEQUENCE_MAX = 14139
EXPECTED_SEQUENCE_SUM = 1_924_746
EXPECTED_TRAIN_EPISODES = 8229
EXPECTED_DEV_EPISODES = 2386

#: Pre-declared *before* looking at any old/new vector comparison.  The user protocol
#: says: report bitwise equality if true; otherwise require cosine >= .999 and relative
#: L2 <= .04, and disclose the mixed producer/numerics.  Never loosen after results.
COMPATIBILITY_COSINE_MIN = 0.999
COMPATIBILITY_RELATIVE_L2_MAX = 0.04
PREFIX_COSINE_MIN = 0.999
PREFIX_RELATIVE_L2_MAX = 0.04

#: Explicit producer/D dependencies.  Do not glob the package: lanes T/I will add
#: train/sampler/evaluation/cli/comparison/report modules that must not invalidate a
#: completed feature root.
PRODUCER_FILES = (
    "src/sdsafe_fine_label_repair/__init__.py",
    "src/sdsafe_fine_label_repair/data.py",
    "src/sdsafe_fine_label_repair/features.py",
    "src/sdsafe_fine_confirm/feature_model.py",
    "src/sdsafe_fine_confirm/sequences.py",
    "src/sdsafe_features/model.py",
    "src/sdsafe_features/common.py",
    "src/sdsafe_cot/rounds.py",
)
SOURCE_INPUT_FILES = (
    "artifacts/event_data/cot/config.json",
    "artifacts/event_data/cot/manifest.json",
    "artifacts/event_data/cot/episodes.jsonl",
    "artifacts/cot32k/replay_semantic/eligible_episodes.jsonl",
)
MODEL_INPUT_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")
FUTURE_CONSUMER_MODULES = (
    "src/sdsafe_fine_label_repair/train.py",
    "src/sdsafe_fine_label_repair/sampler.py",
    "src/sdsafe_fine_label_repair/evaluation.py",
    "src/sdsafe_fine_label_repair/cli.py",
    "src/sdsafe_fine_label_repair/comparisons.py",
    "src/sdsafe_fine_label_repair/report.py",
    "tests/test_fine_label_repair_train.py",
    "tests/test_fine_label_repair_sampler.py",
    "tests/test_fine_label_repair_evaluation.py",
    "tests/test_fine_label_repair_integration.py",
)


class FeatureStoreError(RuntimeError):
    """A feature identity/coverage/split contract was violated."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- paths


@dataclass(frozen=True)
class FeaturePaths:
    authority: Path
    plan: Path
    rounds: Path
    repaired_train: Path
    repaired_dev: Path
    old_features_root: Path
    supplement_root: Path
    scaler_source: Path
    model_root: Path


def default_paths(root: Path | str = ".") -> FeaturePaths:
    base = Path(root)
    data = base / "artifacts/fine_label_repair_v1/data"
    return FeaturePaths(
        authority=data / "episodes_train_dev.jsonl",
        plan=data / "feature_supplemental_plan.jsonl",
        rounds=data / "feature_extraction_rounds.json",
        repaired_train=data / "repaired_train_rows.jsonl",
        repaired_dev=data / "repaired_dev_rows.jsonl",
        old_features_root=base / "artifacts/fine_confirm/features",
        supplement_root=base / "artifacts/fine_label_repair_v1/features",
        scaler_source=data / "scaler_source.json",
        model_root=base / "models/Qwen3-8B",
    )


# --------------------------------------------------------------------------- authority


@dataclass(frozen=True)
class Authority:
    rows: dict[str, dict[str, Any]]
    split: dict[str, str]
    sha256: str
    counts: dict[str, int]


def load_authority(path: Path | str) -> Authority:
    path = Path(path)
    rows: dict[str, dict[str, Any]] = {}
    split: dict[str, str] = {}
    for row in read_jsonl(path):
        episode_id = str(row["episode_id"])
        if episode_id in rows:
            raise FeatureStoreError(f"duplicate authority episode {episode_id}")
        value = assert_protected_split(row.get("split"), where=f"authority {episode_id}")
        rows[episode_id] = dict(row)
        split[episode_id] = value
    counts = Counter(split.values())
    if counts.get(SPLIT_TRAIN) != EXPECTED_TRAIN_EPISODES:
        raise FeatureStoreError(f"authority TRAIN episodes {counts.get(SPLIT_TRAIN)} != {EXPECTED_TRAIN_EPISODES}")
    if counts.get(SPLIT_DEV) != EXPECTED_DEV_EPISODES:
        raise FeatureStoreError(f"authority DEV episodes {counts.get(SPLIT_DEV)} != {EXPECTED_DEV_EPISODES}")
    if set(counts) - {SPLIT_TRAIN, SPLIT_DEV}:
        raise FeatureStoreError("authority exposes a non TRAIN/DEV split")
    return Authority(rows=rows, split=split, sha256=sha256_file(path), counts=dict(counts))


# --------------------------------------------------------------------------- D plan


@dataclass(frozen=True)
class PlanRound:
    episode_id: str
    round_id: int
    split: str
    sequence_length: int
    prompt_length: int
    endpoints: tuple[dict[str, Any], ...]
    meta: dict[str, Any]
    reference: dict[str, Any] | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.episode_id, int(self.round_id))

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoints)

    @property
    def row_count(self) -> int:
        return len(self.endpoints) + 1


@dataclass
class SupplementPlan:
    rows: tuple[dict[str, Any], ...]
    rounds: tuple[PlanRound, ...]
    counts: dict[str, Any]
    plan_sha256: str
    rounds_sha256: str
    by_feature_key: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_event_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_round: dict[tuple[str, int], PlanRound] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_feature_key = {str(row["feature_key"]): row for row in self.rows}
        self.by_event_id = {str(row["event_id"]): row for row in self.rows}
        self.by_round = {row.key: row for row in self.rounds}
        if len(self.by_feature_key) != len(self.rows):
            raise FeatureStoreError("duplicate supplemental feature_key")
        if len(self.by_event_id) != len(self.rows):
            raise FeatureStoreError("duplicate supplemental event_id")
        if len(self.by_round) != len(self.rounds):
            raise FeatureStoreError("duplicate supplemental plan round")

    def select_rounds(self, keys: Sequence[tuple[str, int]] | None) -> tuple[PlanRound, ...]:
        if keys is None:
            return self.rounds
        wanted = {(str(episode), int(round_id)) for episode, round_id in keys}
        selected = tuple(row for row in self.rounds if row.key in wanted)
        if len(selected) != len(wanted):
            missing = sorted(wanted - {row.key for row in selected})
            raise FeatureStoreError(f"selection contains unknown rounds: {missing[:5]}")
        return selected

    def attach_old_references(self, old: "OldIndex") -> None:
        attached: list[PlanRound] = []
        for plan_round in self.rounds:
            row = old.round_end(plan_round.episode_id, plan_round.round_id)
            if int(row["feature_token_index"]) != plan_round.sequence_length - 1:
                raise FeatureStoreError(
                    f"{plan_round.key}: old round_end token {row['feature_token_index']} "
                    f"!= sequence_length-1 {plan_round.sequence_length - 1}"
                )
            hashes = {str(endpoint["round_sequence_sha256"]) for endpoint in plan_round.endpoints}
            if len(hashes) != 1:
                raise FeatureStoreError(f"{plan_round.key}: endpoints disagree on the round sequence hash")
            if str(row["sequence_sha256"]) != hashes.pop():
                raise FeatureStoreError(f"{plan_round.key}: old round_end sequence hash differs from the D plan")
            if int(row["sequence_length"]) != plan_round.sequence_length:
                raise FeatureStoreError(f"{plan_round.key}: old round_end sequence_length differs")
            reference = {
                "role": REFERENCE_ROLE,
                "feature_key": "suppref::"
                + canonical_sha256(
                    [SCOPE, plan_round.episode_id, int(plan_round.round_id), int(row["feature_token_index"])]
                )[:32],
                "event_id": str(row["event_id"]),
                "event_type": "round_end",
                "token_index": int(row["feature_token_index"]),
                "completion_token_index": int(row["feature_token_index"]) - int(plan_round.prompt_length),
                "old_scope": str(row["_scope"]),
                "old_local_feature_row": int(row["_local_feature_row"]),
                "old_global_feature_row": int(row["_global_feature_row"]),
                "old_sequence_sha256": str(row["sequence_sha256"]),
                "index": int(row.get("index", 0) or 0),
                "segment_index": None,
            }
            attached.append(
                PlanRound(
                    episode_id=plan_round.episode_id,
                    round_id=plan_round.round_id,
                    split=plan_round.split,
                    sequence_length=plan_round.sequence_length,
                    prompt_length=plan_round.prompt_length,
                    endpoints=plan_round.endpoints,
                    meta=plan_round.meta,
                    reference=reference,
                )
            )
        self.rounds = tuple(attached)
        self.by_round = {row.key: row for row in self.rounds}


def load_plan(paths: FeaturePaths, authority: Authority, *, verify_repaired: bool = True) -> SupplementPlan:
    plan_path = Path(paths.plan)
    rounds_path = Path(paths.rounds)
    raw_rows = read_jsonl(plan_path)
    rounds_payload = read_json(rounds_path)
    if len(raw_rows) != EXPECTED_ENDPOINTS:
        raise FeatureStoreError(f"supplemental plan has {len(raw_rows)} rows, expected {EXPECTED_ENDPOINTS}")
    if rounds_payload.get("new_endpoints") != EXPECTED_ENDPOINTS:
        raise FeatureStoreError("rounds payload endpoint count disagrees with the D plan")
    if rounds_payload.get("new_forward_rounds") != EXPECTED_ROUNDS:
        raise FeatureStoreError("rounds payload forward-round count disagrees with the D plan")
    round_meta = rounds_payload.get("rounds") or []
    if len(round_meta) != EXPECTED_ROUNDS:
        raise FeatureStoreError(f"rounds payload has {len(round_meta)} rounds, expected {EXPECTED_ROUNDS}")

    split_counts: Counter = Counter()
    repair_counts: Counter = Counter()
    rows: list[dict[str, Any]] = []
    by_event: dict[str, dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        event_id = str(raw.get("event_id") or "")
        if not event_id:
            raise FeatureStoreError("supplemental plan row without event_id")
        if event_id in by_event:
            raise FeatureStoreError(f"duplicate supplemental plan event {event_id}")
        ref = raw.get("feature_ref")
        if not isinstance(ref, Mapping):
            raise FeatureStoreError(f"{event_id}: missing supplemental feature_ref")
        validate_feature_ref(ref)
        if ref.get("kind") != SUPPLEMENTAL_FEATURE_KIND:
            raise FeatureStoreError(f"{event_id}: plan ref kind is {ref.get('kind')!r}")
        if str(ref.get("scope")) != SCOPE:
            raise FeatureStoreError(f"{event_id}: supplemental scope {ref.get('scope')!r} is not {SCOPE!r}")
        episode_id = str(raw.get("episode_id") or "")
        if not episode_id or str(ref.get("episode_id")) != episode_id:
            raise FeatureStoreError(f"{event_id}: episode identity is missing or disagrees with the ref")
        split = assert_protected_split(raw.get("split"), where=f"plan {event_id}")
        if authority.split.get(episode_id) != split:
            raise FeatureStoreError(
                f"{event_id}: D plan split {split!r} is not the accepted authority split "
                f"{authority.split.get(episode_id)!r}"
            )
        if str(ref.get("split")) != split:
            raise FeatureStoreError(f"{event_id}: ref split disagrees with the plan row")
        round_id = int(raw.get("round_id"))
        if int(ref.get("round_id")) != round_id:
            raise FeatureStoreError(f"{event_id}: ref round_id disagrees with the plan row")
        token_index = int(ref.get("token_index"))
        if int(raw.get("feature_token_index")) != token_index:
            raise FeatureStoreError(f"{event_id}: plan feature_token_index disagrees with the ref")
        prompt_length = int(ref.get("prompt_length"))
        sequence_length = int(ref.get("sequence_length"))
        completion_index = ref.get("completion_token_index")
        if completion_index is None:
            completion_index = token_index - prompt_length
        completion_index = int(completion_index)
        if token_index != prompt_length + completion_index:
            raise FeatureStoreError(f"{event_id}: token_index is not prompt_length + completion_token_index")
        if not (0 <= token_index < sequence_length):
            raise FeatureStoreError(f"{event_id}: token_index is outside the sequence")
        if int(raw.get("sequence_length")) != sequence_length:
            raise FeatureStoreError(f"{event_id}: plan sequence_length disagrees with the ref")
        if str(raw.get("scope")) != SCOPE:
            raise FeatureStoreError(f"{event_id}: plan scope is not {SCOPE!r}")
        feature_key = str(ref.get("feature_key") or "")
        if not feature_key.startswith("supp::"):
            raise FeatureStoreError(f"{event_id}: unexpected supplemental feature_key {feature_key!r}")
        if feature_key in by_key:
            raise FeatureStoreError(f"duplicate supplemental feature_key {feature_key}")
        entry = {
            "event_id": event_id,
            "feature_key": feature_key,
            "episode_id": episode_id,
            "namespaced_episode_id": str(ref.get("namespaced_episode_id") or episode_id),
            "scope": SCOPE,
            "split": split,
            "round_id": round_id,
            "event_type": str(raw.get("event_type") or ref.get("event_type") or ""),
            "token_index": token_index,
            "completion_token_index": completion_index,
            "prompt_length": prompt_length,
            "sequence_length": sequence_length,
            "round_sequence_sha256": str(ref.get("round_sequence_sha256")),
            "label_status": str(raw.get("label_status") or ""),
            "repair_status": str(raw.get("repair_status") or ""),
            "plan": dict(raw),
            "ref": dict(ref),
        }
        if entry["namespaced_episode_id"] != episode_id:
            raise FeatureStoreError(f"{event_id}: namespaced episode identity disagrees")
        rows.append(entry)
        by_event[event_id] = entry
        by_key[feature_key] = entry
        split_counts[split] += 1
        repair_counts[entry["repair_status"]] += 1

    if dict(split_counts) != EXPECTED_SPLIT_ENDPOINTS:
        raise FeatureStoreError(f"plan split endpoints {dict(split_counts)} != {EXPECTED_SPLIT_ENDPOINTS}")
    if dict(repair_counts) != EXPECTED_REPAIR_STATUS:
        raise FeatureStoreError(f"plan repair-status counts {dict(repair_counts)} != {EXPECTED_REPAIR_STATUS}")
    if int(rounds_payload.get("sequence_length_sum", -1)) != EXPECTED_SEQUENCE_SUM:
        raise FeatureStoreError("rounds payload sequence_length_sum drifted")
    if int(rounds_payload.get("sequence_length_min", -1)) != EXPECTED_SEQUENCE_MIN:
        raise FeatureStoreError("rounds payload sequence_length_min drifted")
    if int(rounds_payload.get("sequence_length_max", -1)) != EXPECTED_SEQUENCE_MAX:
        raise FeatureStoreError("rounds payload sequence_length_max drifted")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["episode_id"], int(row["round_id"]))].append(row)
    plan_rounds: list[PlanRound] = []
    seen: set[tuple[str, int]] = set()
    for meta in round_meta:
        episode_id = str(meta["episode_id"])
        round_id = int(meta["round_id"])
        key = (episode_id, round_id)
        if key in seen:
            raise FeatureStoreError(f"duplicate plan round {key}")
        seen.add(key)
        endpoints = grouped.get(key)
        if not endpoints:
            raise FeatureStoreError(f"{key}: round has no planned endpoints")
        if int(meta.get("endpoints", -1)) != len(endpoints):
            raise FeatureStoreError(f"{key}: round endpoint count disagrees with the plan")
        sequence_length = int(meta["sequence_length"])
        if any(int(row["sequence_length"]) != sequence_length for row in endpoints):
            raise FeatureStoreError(f"{key}: endpoint sequence lengths disagree")
        hashes = {str(row["round_sequence_sha256"]) for row in endpoints}
        if len(hashes) != 1:
            raise FeatureStoreError(f"{key}: endpoints disagree on the round sequence hash")
        prompt_lengths = {int(row["prompt_length"]) for row in endpoints}
        if len(prompt_lengths) != 1:
            raise FeatureStoreError(f"{key}: endpoints disagree on the prompt length")
        split = authority.split.get(episode_id)
        if split not in (SPLIT_TRAIN, SPLIT_DEV):
            raise FeatureStoreError(f"{key}: episode is not in the TRAIN/DEV authority")
        if any(row["split"] != split for row in endpoints):
            raise FeatureStoreError(f"{key}: endpoint split disagrees with authority")
        endpoints.sort(key=lambda row: (int(row["token_index"]), str(row["event_id"])))
        plan_rounds.append(
            PlanRound(
                episode_id=episode_id,
                round_id=round_id,
                split=str(split),
                sequence_length=sequence_length,
                prompt_length=next(iter(prompt_lengths)),
                endpoints=tuple(endpoints),
                meta=dict(meta),
            )
        )
    if len(plan_rounds) != EXPECTED_ROUNDS:
        raise FeatureStoreError(f"plan rounds {len(plan_rounds)} != {EXPECTED_ROUNDS}")
    if set(grouped) != seen:
        extra = sorted(set(grouped) - seen)[:5]
        raise FeatureStoreError(f"planned endpoints have rounds absent from the rounds payload: {extra}")

    if verify_repaired:
        repaired: dict[str, dict[str, Any]] = {}
        for path in (paths.repaired_train, paths.repaired_dev):
            for row in read_jsonl(path):
                event_id = str(row["event_id"])
                if event_id in repaired:
                    raise FeatureStoreError(f"duplicate D repaired row {event_id}")
                repaired[event_id] = dict(row)
        supplemental = {
            event_id: row
            for event_id, row in repaired.items()
            if isinstance(row.get("feature_ref"), Mapping)
            and row["feature_ref"].get("kind") == SUPPLEMENTAL_FEATURE_KIND
        }
        if set(supplemental) != set(by_event):
            missing = sorted(set(by_event) - set(supplemental))[:5]
            extra = sorted(set(supplemental) - set(by_event))[:5]
            raise FeatureStoreError(f"D repaired supplemental rows disagree with plan: missing={missing} extra={extra}")
        for event_id, entry in by_event.items():
            drow = supplemental[event_id]
            if dict(drow["feature_ref"]) != dict(entry["ref"]):
                raise FeatureStoreError(f"{event_id}: D repaired feature_ref differs from the accepted plan")
            if str(drow.get("split")) != entry["split"]:
                raise FeatureStoreError(f"{event_id}: D repaired split differs from the accepted plan")
            if str(drow.get("repair_status")) != entry["repair_status"]:
                raise FeatureStoreError(f"{event_id}: D repaired status differs from the accepted plan")

    counts = {
        "endpoints": len(rows),
        "rounds": len(plan_rounds),
        "splits": dict(split_counts),
        "repair_status": dict(repair_counts),
        "label_status": dict(Counter(row["label_status"] for row in rows)),
        "sequence_length_sum": EXPECTED_SEQUENCE_SUM,
        "sequence_length_min": EXPECTED_SEQUENCE_MIN,
        "sequence_length_max": EXPECTED_SEQUENCE_MAX,
    }
    return SupplementPlan(
        rows=tuple(rows),
        rounds=tuple(plan_rounds),
        counts=counts,
        plan_sha256=sha256_file(plan_path),
        rounds_sha256=sha256_file(rounds_path),
    )


# --------------------------------------------------------------------------- old cached index


class OldIndex:
    """Read-only metadata reader over the immutable historical per-scope feature stores."""

    def __init__(
        self,
        features_root: Path | str,
        authority: Authority,
        *,
        scopes: Sequence[str] = SCOPES,
    ) -> None:
        self.features_root = Path(features_root)
        self.authority = authority
        self.scopes = tuple(scopes)
        self.manifests: dict[str, dict[str, Any]] = {}
        self.matrix_paths: dict[str, Path] = {}
        self.rows_by_scope: dict[str, int] = {}
        self.scope_offset: dict[str, int] = {}
        self.by_event: dict[str, dict[str, Any]] = {}
        self.by_local: dict[tuple[str, int], dict[str, Any]] = {}
        self.by_round_end: dict[tuple[str, int], dict[str, Any]] = {}
        self.excluded_test_events: set[str] = set()
        offset = 0
        for scope in self.scopes:
            base = self.features_root / scope
            manifest_path = base / "manifest.json"
            if not manifest_path.is_file():
                raise FeatureStoreError(f"{scope}: old feature manifest is missing")
            manifest = read_json(manifest_path)
            if manifest.get("status") != "complete":
                raise FeatureStoreError(f"{scope}: old feature store is not complete")
            index_path = base / INDEX_FILE
            if not index_path.is_file():
                raise FeatureStoreError(f"{scope}: old feature index is missing")
            matrix_path = base / MATRIX_FILE
            if not matrix_path.is_file():
                raise FeatureStoreError(f"{scope}: old feature matrix is missing")
            index_rows = read_jsonl(index_path)
            rows = int(manifest.get("counts", {}).get("events", -1))
            if rows != len(index_rows):
                raise FeatureStoreError(f"{scope}: old index rows {len(index_rows)} != manifest {rows}")
            if matrix_path.stat().st_size != rows * ROW_BYTES:
                raise FeatureStoreError(f"{scope}: old matrix size is inconsistent with its manifest")
            self.manifests[scope] = manifest
            self.matrix_paths[scope] = matrix_path
            self.rows_by_scope[scope] = rows
            self.scope_offset[scope] = offset
            for row in index_rows:
                local = int(row["feature_row"])
                if local < 0 or local >= rows:
                    raise FeatureStoreError(f"{scope}: old local row {local} is outside the aggregate")
                event_id = str(row["event_id"])
                namespaced = str(row.get("namespaced_episode_id") or "")
                authority_split = self.authority.split.get(namespaced)
                row_split = str(row.get("mixed_split") or row.get("split") or "")
                if row_split not in (SPLIT_TRAIN, SPLIT_DEV):
                    self.excluded_test_events.add(event_id)
                    continue
                if authority_split != row_split:
                    # Metadata that claims a non-authority split is never materialised.
                    self.excluded_test_events.add(event_id)
                    continue
                if int(row.get("feature_byte_offset", local * ROW_BYTES)) != local * ROW_BYTES:
                    raise FeatureStoreError(f"{scope}/{event_id}: old byte offset is not derived from the true row")
                wrapper = dict(row)
                wrapper["_scope"] = scope
                wrapper["_local_feature_row"] = local
                wrapper["_global_feature_row"] = offset + local
                wrapper["_matrix_path"] = matrix_path
                if event_id in self.by_event:
                    raise FeatureStoreError(f"duplicate old event_id {event_id}")
                self.by_event[event_id] = wrapper
                self.by_local[(scope, local)] = wrapper
                if str(row.get("event_type")) == "round_end":
                    key = (namespaced, int(row["round_id"]))
                    if key in self.by_round_end:
                        raise FeatureStoreError(f"duplicate old round_end for {key}")
                    self.by_round_end[key] = wrapper
            offset += rows
        self.total_rows = offset

    def round_end(self, episode_id: str, round_id: int) -> dict[str, Any]:
        row = self.by_round_end.get((str(episode_id), int(round_id)))
        if row is None:
            raise FeatureStoreError(f"{episode_id} r{round_id}: no authoritative old round_end row")
        return row

    def resolve_legacy(self, ref: Mapping[str, Any], expected_split: str) -> "PhysicalRef":
        event_id = str(ref.get("event_id") or "")
        if not event_id:
            raise FeatureStoreError("cached feature ref has no event_id")
        if event_id in self.excluded_test_events:
            raise FeatureStoreError(f"{event_id}: refusing a non-authority/TEST old index row")
        row = self.by_event.get(event_id)
        if row is None:
            raise FeatureStoreError(f"{event_id}: cached event is not in the accepted TRAIN/DEV old index")
        scope = str(ref.get("scope") or "")
        if scope != str(row["_scope"]):
            raise FeatureStoreError(f"{event_id}: cached ref scope {scope!r} != old index scope {row['_scope']!r}")
        local = int(ref.get("local_feature_row"))
        if local != int(row["_local_feature_row"]):
            raise FeatureStoreError(f"{event_id}: forged local row {local} != true old row {row['_local_feature_row']}")
        global_row = int(ref.get("global_feature_row"))
        if global_row != int(row["_global_feature_row"]):
            raise FeatureStoreError(
                f"{event_id}: forged global row {global_row} != true old global row {row['_global_feature_row']}"
            )
        if int(ref.get("feature_token_index", -1)) != int(row["feature_token_index"]):
            raise FeatureStoreError(f"{event_id}: cached token index differs from the old index")
        if str(ref.get("sequence_sha256")) != str(row.get("sequence_sha256")):
            raise FeatureStoreError(f"{event_id}: cached sequence hash differs from the old index")
        if str(ref.get("feature_file") or "") != str(row.get("feature_file") or MATRIX_FILE):
            raise FeatureStoreError(f"{event_id}: cached feature file differs from the old index")
        namespaced = str(row["namespaced_episode_id"])
        authority_split = self.authority.split.get(namespaced)
        if authority_split != expected_split:
            raise FeatureStoreError(
                f"{event_id}: old row authority split {authority_split!r} is not requested {expected_split!r}"
            )
        row_split = str(row.get("mixed_split") or row.get("split") or "")
        if row_split != expected_split:
            raise FeatureStoreError(f"{event_id}: old index split {row_split!r} is not requested {expected_split!r}")
        derived_byte = local * ROW_BYTES
        if ref.get("feature_byte_offset") is not None and int(ref["feature_byte_offset"]) != derived_byte:
            raise FeatureStoreError(f"{event_id}: cached byte offset is not the true index offset")
        return PhysicalRef(
            kind=LEGACY_FEATURE_KIND,
            version="cached_canonical_bf16",
            key=(LEGACY_FEATURE_KIND, scope, local),
            matrix_path=self.matrix_paths[scope],
            total_rows=self.rows_by_scope[scope],
            feature_row=local,
            scope=scope,
            event_id=event_id,
            sequence_sha256=str(row.get("sequence_sha256")),
            feature_token_index=int(row["feature_token_index"]),
        )

    def read_local(self, scope: str, local: int):
        row = self.by_local.get((str(scope), int(local)))
        if row is None:
            raise FeatureStoreError(f"{scope}: local row {local} is not an authoritative TRAIN/DEV old row")
        return _read_matrix_rows(self.matrix_paths[scope], self.rows_by_scope[scope], [int(local)])[0]

    def read_raw_local(self, scope: str, local: int) -> bytes:
        row = self.by_local.get((str(scope), int(local)))
        if row is None:
            raise FeatureStoreError(f"{scope}: local row {local} is not an authoritative TRAIN/DEV old row")
        return _read_matrix_raw(self.matrix_paths[scope], self.rows_by_scope[scope], int(local))


# --------------------------------------------------------------------------- new supplemental index


@dataclass(frozen=True)
class PhysicalRef:
    kind: str
    version: str
    key: tuple[Any, ...]
    matrix_path: Path
    total_rows: int
    feature_row: int
    scope: str
    event_id: str
    sequence_sha256: str
    feature_token_index: int


@dataclass
class SupplementIndex:
    out_dir: Path
    manifest: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    matrix_path: Path
    matrix_rows: int
    by_feature_key: dict[str, dict[str, Any]]
    by_event_id: dict[str, dict[str, Any]]

    @classmethod
    def open(cls, out_dir: Path | str, plan: SupplementPlan, *, require_complete: bool) -> "SupplementIndex | None":
        out_dir = Path(out_dir)
        manifest_path = out_dir / MANIFEST_FILE
        if not manifest_path.is_file():
            if require_complete:
                raise FeatureStoreError(f"{out_dir}: supplemental feature manifest is missing")
            return None
        manifest = read_json(manifest_path)
        if require_complete and manifest.get("status") != "complete":
            raise FeatureStoreError(f"{out_dir}: supplemental feature store is not complete ({manifest.get('status')})")
        index_path = out_dir / INDEX_FILE
        if not index_path.is_file():
            if require_complete:
                raise FeatureStoreError(f"{out_dir}: supplemental feature index is missing")
            return None
        rows = tuple(read_jsonl(index_path))
        storage = manifest.get("storage", {}).get("matrix", {})
        matrix_rows = int(manifest.get("counts", {}).get("matrix_rows", storage.get("shape", [0, 0])[0]))
        if len(rows) != matrix_rows:
            raise FeatureStoreError(f"{out_dir}: supplemental index rows {len(rows)} != manifest {matrix_rows}")
        matrix_path = out_dir / str(storage.get("file") or MATRIX_FILE)
        if not matrix_path.is_file():
            raise FeatureStoreError(f"{out_dir}: supplemental matrix is missing")
        if matrix_path.stat().st_size != matrix_rows * ROW_BYTES:
            raise FeatureStoreError(f"{out_dir}: supplemental matrix size is inconsistent")
        by_key: dict[str, dict[str, Any]] = {}
        by_event: dict[str, dict[str, Any]] = {}
        endpoint_keys: set[str] = set()
        reference_rows = 0
        for row in rows:
            role = str(row.get("role"))
            if role == ENDPOINT_ROLE:
                key = str(row.get("feature_key"))
                if key in by_key:
                    raise FeatureStoreError(f"{out_dir}: duplicate supplemental index feature_key {key}")
                by_key[key] = row
                by_event[str(row.get("event_id"))] = row
                endpoint_keys.add(key)
            elif role == REFERENCE_ROLE:
                reference_rows += 1
            else:
                raise FeatureStoreError(f"{out_dir}: unknown supplemental index role {role!r}")
        if require_complete:
            expected_keys = set(plan.by_feature_key)
            if endpoint_keys != expected_keys:
                missing = sorted(expected_keys - endpoint_keys)[:5]
                extra = sorted(endpoint_keys - expected_keys)[:5]
                raise FeatureStoreError(f"{out_dir}: supplemental endpoint coverage mismatch: missing={missing} extra={extra}")
            if reference_rows != len(plan.rounds):
                raise FeatureStoreError(f"{out_dir}: {reference_rows} reference rows != {len(plan.rounds)} rounds")
        return cls(
            out_dir=out_dir,
            manifest=manifest,
            rows=rows,
            matrix_path=matrix_path,
            matrix_rows=matrix_rows,
            by_feature_key=by_key,
            by_event_id=by_event,
        )

    def resolve_supplement(self, ref: Mapping[str, Any], plan: SupplementPlan, expected_split: str) -> PhysicalRef:
        feature_key = str(ref.get("feature_key") or "")
        plan_row = plan.by_feature_key.get(feature_key)
        if plan_row is None:
            raise FeatureStoreError(f"{feature_key!r}: supplemental feature_key is not in the accepted D plan")
        for name in (
            "event_id",
            "episode_id",
            "round_id",
            "token_index",
            "round_sequence_sha256",
            "split",
            "prompt_length",
            "sequence_length",
            "event_type",
        ):
            ref_value = ref.get(name)
            plan_value = plan_row["ref"].get(name)
            if ref_value is not None and str(ref_value) != str(plan_value):
                raise FeatureStoreError(f"{feature_key}: supplemental ref {name} disagrees with the accepted plan")
        if plan_row["event_id"] != str(ref.get("event_id")):
            raise FeatureStoreError(f"{feature_key}: logical event id disagrees with the accepted plan")
        if plan_row["split"] != expected_split:
            raise FeatureStoreError(f"{feature_key}: plan split {plan_row['split']!r} is not requested {expected_split!r}")
        index_row = self.by_feature_key.get(feature_key)
        if index_row is None:
            raise FeatureStoreError(f"{feature_key}: supplemental vector is missing (extraction incomplete)")
        if str(index_row.get("event_id")) != plan_row["event_id"]:
            raise FeatureStoreError(f"{feature_key}: supplemental index event id disagrees with the plan")
        if str(index_row.get("sequence_sha256")) != plan_row["round_sequence_sha256"]:
            raise FeatureStoreError(f"{feature_key}: supplemental index sequence hash disagrees with the plan")
        if int(index_row.get("token_index", -1)) != int(plan_row["token_index"]):
            raise FeatureStoreError(f"{feature_key}: supplemental index token index disagrees with the plan")
        feature_row = int(index_row["feature_row"])
        derived_byte = feature_row * ROW_BYTES
        if int(index_row.get("feature_byte_offset", derived_byte)) != derived_byte:
            raise FeatureStoreError(f"{feature_key}: supplemental byte offset is not derived from the true index")
        return PhysicalRef(
            kind=SUPPLEMENTAL_FEATURE_KIND,
            version="supplemental_bf16_v1",
            key=(SUPPLEMENTAL_FEATURE_KIND, str(index_row.get("feature_key")), feature_row),
            matrix_path=self.matrix_path,
            total_rows=self.matrix_rows,
            feature_row=feature_row,
            scope=SCOPE,
            event_id=plan_row["event_id"],
            sequence_sha256=plan_row["round_sequence_sha256"],
            feature_token_index=int(plan_row["token_index"]),
        )


# --------------------------------------------------------------------------- raw matrix helpers


def _matrix_u2(path: Path, total_rows: int):
    import numpy as np

    if total_rows <= 0:
        raise FeatureStoreError(f"{path}: empty matrix")
    if path.stat().st_size != total_rows * ROW_BYTES:
        raise FeatureStoreError(f"{path}: matrix size is inconsistent")
    return np.memmap(path, dtype=np.dtype("<u2"), mode="r", shape=(total_rows, HIDDEN_SIZE))


def _u2_to_float(array: Any):
    import numpy as np
    import torch

    copy = np.ascontiguousarray(array, dtype=np.uint16)
    return torch.from_numpy(copy).view(torch.bfloat16).reshape(len(copy), HIDDEN_SIZE).to(torch.float32)


def _read_matrix_rows(path: Path | str, total_rows: int, rows: Sequence[int]):
    import numpy as np

    path = Path(path)
    requested = [int(row) for row in rows]
    if not requested:
        import torch

        return torch.empty((0, HIDDEN_SIZE), dtype=torch.float32)
    if min(requested) < 0 or max(requested) >= total_rows:
        raise FeatureStoreError(f"{path}: requested row outside the aggregate")
    matrix = _matrix_u2(path, total_rows)
    return _u2_to_float(np.asarray(matrix[requested]))


def _read_matrix_raw(path: Path | str, total_rows: int, row: int) -> bytes:
    path = Path(path)
    row = int(row)
    if row < 0 or row >= total_rows:
        raise FeatureStoreError(f"{path}: row {row} outside the aggregate")
    if path.stat().st_size != total_rows * ROW_BYTES:
        raise FeatureStoreError(f"{path}: matrix size is inconsistent")
    with path.open("rb") as handle:
        handle.seek(row * ROW_BYTES)
        payload = handle.read(ROW_BYTES)
    if len(payload) != ROW_BYTES:
        raise FeatureStoreError(f"{path}: short matrix read at row {row}")
    return payload


# --------------------------------------------------------------------------- read-only store


class FeatureStore:
    """Read-only TRAIN/DEV feature store for lanes T/I.

    A reference is always resolved to its *physical* kind/version/row before any vector is
    opened.  Legacy references always read the immutable old vector; supplemental
    references always read the new supplement; the two are never silently substituted.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        paths: FeaturePaths | None = None,
        scopes: Sequence[str] = SCOPES,
        require_complete: bool = True,
        verify_repaired: bool = True,
        authority: Authority | None = None,
        plan: SupplementPlan | None = None,
        old: OldIndex | None = None,
        attach_references: bool = True,
    ) -> None:
        self.root = Path(root)
        self.paths = paths or default_paths(self.root)
        self.authority = authority or load_authority(self.paths.authority)
        self.plan = plan or load_plan(self.paths, self.authority, verify_repaired=verify_repaired)
        self.old = old or OldIndex(self.paths.old_features_root, self.authority, scopes=scopes)
        if attach_references and any(row.reference is None for row in self.plan.rounds):
            self.plan.attach_old_references(self.old)
        self.supplement = SupplementIndex.open(
            self.paths.supplement_root / SCOPE,
            self.plan,
            require_complete=require_complete,
        )
        self.scaler = read_json(self.paths.scaler_source) if self.paths.scaler_source.is_file() else None

    @classmethod
    def open(
        cls,
        root: Path | str = ".",
        *,
        paths: FeaturePaths | None = None,
        scopes: Sequence[str] = SCOPES,
        require_complete: bool = True,
        verify_repaired: bool = True,
        authority: Authority | None = None,
        plan: SupplementPlan | None = None,
        old: OldIndex | None = None,
        attach_references: bool = True,
    ) -> "FeatureStore":
        return cls(
            root,
            paths=paths,
            scopes=scopes,
            require_complete=require_complete,
            verify_repaired=verify_repaired,
            authority=authority,
            plan=plan,
            old=old,
            attach_references=attach_references,
        )

    def resolve_ref(self, ref: Mapping[str, Any], expected_split: str) -> PhysicalRef:
        if expected_split not in (SPLIT_TRAIN, SPLIT_DEV):
            raise FeatureStoreError(f"feature access is TRAIN/DEV-only, got {expected_split!r}")
        validate_feature_ref(ref, expected_split)
        kind = str(ref.get("kind"))
        if kind == LEGACY_FEATURE_KIND:
            return self.old.resolve_legacy(ref, expected_split)
        if kind == SUPPLEMENTAL_FEATURE_KIND:
            if self.supplement is None:
                raise FeatureStoreError("supplemental feature store is not complete")
            return self.supplement.resolve_supplement(ref, self.plan, expected_split)
        raise FeatureStoreError(f"unknown feature kind {kind!r}")

    def read(self, refs: Sequence[Mapping[str, Any]], expected_split: str):
        import torch

        resolved = [self.resolve_ref(ref, expected_split) for ref in refs]
        output = torch.empty((len(resolved), HIDDEN_SIZE), dtype=torch.float32)
        by_physical: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for position, physical in enumerate(resolved):
            by_physical[(str(physical.matrix_path), int(physical.total_rows), int(physical.feature_row))].append(position)
        grouped: dict[tuple[str, int], list[tuple[int, list[int]]]] = defaultdict(list)
        for (path, total, row), positions in by_physical.items():
            grouped[(path, total)].append((row, positions))
        for (path, total), entries in grouped.items():
            entries.sort(key=lambda item: item[0])
            for start in range(0, len(entries), 4096):
                chunk = entries[start : start + 4096]
                rows = [row for row, _ in chunk]
                vectors = _read_matrix_rows(Path(path), int(total), rows)
                for vector, (_, positions) in zip(vectors, chunk):
                    output[torch.tensor(positions, dtype=torch.long)] = vector
        return output

    __call__ = read

    def read_rows(self, rows: Sequence[Mapping[str, Any]], expected_split: str):
        refs: list[Mapping[str, Any]] = []
        for row in rows:
            if row.get("split") is not None and str(row.get("split")) != expected_split:
                raise FeatureStoreError(f"{row.get('event_id')}: row split is not requested {expected_split!r}")
            ref = row.get("feature_ref")
            if not isinstance(ref, Mapping):
                raise FeatureStoreError(f"{row.get('event_id')}: row has no feature_ref")
            refs.append(ref)
        return self.read(refs, expected_split)

    def scaler_refs(self, legacy_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """The old 107,215-row scaler pool is always materialised from legacy refs."""
        refs: list[Mapping[str, Any]] = []
        for row in legacy_rows:
            ref = row.get("feature_ref")
            if not isinstance(ref, Mapping) or ref.get("kind") != LEGACY_FEATURE_KIND:
                raise FeatureStoreError("the scaler pool must be the immutable legacy feature view")
            refs.append(ref)
        return refs

    def scaler_source(self) -> dict[str, Any] | None:
        return self.scaler


# --------------------------------------------------------------------------- journal storage


def journal_path(out_dir: Path | str, episode_id: str, round_id: int) -> Path:
    return Path(out_dir) / JOURNAL_DIR / str(episode_id) / f"r{int(round_id)}.json"


def load_journals(out_dir: Path | str) -> dict[tuple[str, int], dict[str, Any]]:
    root = Path(out_dir) / JOURNAL_DIR
    journals: dict[tuple[str, int], dict[str, Any]] = {}
    if not root.is_dir():
        return journals
    for stale in root.glob("*/*.tmp"):
        stale.unlink()
    for path in sorted(root.glob("*/*.json")):
        journal = read_json(path)
        identity = journal.get("identity") or {}
        key = (str(identity.get("episode_id")), int(identity.get("round_id")))
        if key in journals:
            raise FeatureStoreError(f"{out_dir}: duplicate round journal {key}")
        journals[key] = journal
    return journals


def verify_journal_prefix(journals: Iterable[Mapping[str, Any]]) -> int:
    ordered = sorted(
        journals,
        key=lambda journal: (str(journal["identity"]["episode_id"]), int(journal["identity"]["round_id"])),
    )
    expected = 0
    for journal in ordered:
        identity = journal["identity"]
        start = int(identity["row_start"])
        if start != expected:
            raise FeatureStoreError(
                f"{identity['episode_id']} r{identity['round_id']}: journal starts at row {start}, expected {expected}"
            )
        expected += int(identity["event_count"])
    return expected


class SupplementAppendStore:
    """Append-only single BF16 matrix plus atomic per-round journals."""

    def __init__(self, out_dir: Path | str, config_digest: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config_digest = str(config_digest)
        self.matrix_path = self.out_dir / MATRIX_FILE
        self.journals = load_journals(self.out_dir)
        for key, journal in self.journals.items():
            if journal.get("config_sha256") != self.config_digest:
                raise FeatureStoreError(
                    f"{self.out_dir}: journal {key} belongs to another feature config; refusing stale inputs"
                )
            if canonical_sha256(journal["identity"]) != journal.get("identity_sha256"):
                raise FeatureStoreError(f"{self.out_dir}: journal {key} has a bad identity hash")
            if int(journal["identity"]["event_count"]) * ROW_BYTES != int(journal.get("bytes", -1)):
                raise FeatureStoreError(f"{self.out_dir}: journal {key} has an inconsistent byte count")
        self.committed_rows = verify_journal_prefix(self.journals.values())
        self.committed_bytes = self.committed_rows * ROW_BYTES
        if not self.matrix_path.exists():
            if self.committed_rows:
                raise FeatureStoreError(f"{self.matrix_path}: missing aggregate with committed rows")
            self.matrix_path.write_bytes(b"")
        size = self.matrix_path.stat().st_size
        if size < self.committed_bytes:
            raise FeatureStoreError(f"{self.matrix_path}: shorter than committed journals")
        self.truncated_tail_bytes = size - self.committed_bytes
        if self.truncated_tail_bytes:
            with self.matrix_path.open("r+b") as handle:
                handle.truncate(self.committed_bytes)
        self._handle = self.matrix_path.open("r+b")

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def read_range(self, start_row: int, rows: int) -> bytes:
        payload = os.pread(self._handle.fileno(), rows * ROW_BYTES, start_row * ROW_BYTES)
        if len(payload) != rows * ROW_BYTES:
            raise FeatureStoreError(f"short aggregate read at row {start_row} (+{rows})")
        return payload

    def reuse(self, key: tuple[str, int], identity: dict[str, Any]) -> dict[str, Any]:
        journal = self.journals.get(key)
        if journal is None:
            raise FeatureStoreError(f"{self.out_dir}: no committed round {key}")
        if journal.get("identity_sha256") != canonical_sha256(identity) or journal.get("identity") != identity:
            raise FeatureStoreError(
                f"{self.out_dir}: committed round {key} no longer matches its code/model/data/position identity"
            )
        rows = int(identity["event_count"])
        start = int(identity["row_start"])
        if int(journal.get("byte_start", -1)) != start * ROW_BYTES:
            raise FeatureStoreError(f"{self.out_dir}: round {key} journal start byte is inconsistent")
        payload = self.read_range(start, rows)
        if sha256_bytes(payload) != journal.get("features_sha256"):
            raise FeatureStoreError(f"{self.out_dir}: committed feature bytes of round {key} changed")
        return journal

    def append(self, identity: dict[str, Any], payload: bytes) -> dict[str, Any]:
        key = (str(identity["episode_id"]), int(identity["round_id"]))
        if key in self.journals:
            raise FeatureStoreError(f"{self.out_dir}: round {key} is already committed")
        rows = int(identity["event_count"])
        start = int(identity["row_start"])
        if start != self.committed_rows:
            raise FeatureStoreError(f"{self.out_dir}: round {key} starts at {start}, expected {self.committed_rows}")
        if len(payload) != rows * ROW_BYTES:
            raise FeatureStoreError(f"{self.out_dir}: round {key} payload length is wrong")
        self._handle.seek(start * ROW_BYTES)
        self._handle.write(payload)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        journal = {
            "schema_version": SCHEMA_VERSION,
            "stage": STAGE,
            "config_sha256": self.config_digest,
            "identity": identity,
            "identity_sha256": canonical_sha256(identity),
            "features_sha256": sha256_bytes(payload),
            "byte_start": start * ROW_BYTES,
            "bytes": len(payload),
        }
        atomic_write_json(journal_path(self.out_dir, *key), journal)
        self.journals[key] = journal
        self.committed_rows += rows
        return journal


# --------------------------------------------------------------------------- producer identity


def producer_fingerprint(root: Path | str, model_identity: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root)
    files: dict[str, Any] = {}
    for name in PRODUCER_FILES:
        path = root / name
        if not path.is_file():
            raise FeatureStoreError(f"producer dependency is missing: {name}")
        files[name] = sha256_file(path)
    source_inputs: dict[str, Any] = {}
    for name in SOURCE_INPUT_FILES:
        path = root / name
        if not path.is_file():
            raise FeatureStoreError(f"source input is missing: {name}")
        source_inputs[name] = sha256_file(path)
    model_inputs = dict(model_identity.get("model_source_sha256") or {})
    missing = [name for name in MODEL_INPUT_FILES if name not in model_inputs]
    if missing:
        raise FeatureStoreError(f"model identity is missing input hashes: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "files": files,
        "source_inputs": source_inputs,
        "model_inputs": model_inputs,
        "model_repository": model_identity.get("model_repository"),
        "model_path": model_identity.get("model_path"),
        "resolved_commit": model_identity.get("resolved_commit"),
        "sequence_serializer": "archived_prompt_completion_token_ids",
        "explicitly_not_included": list(FUTURE_CONSUMER_MODULES),
        "note": "explicit producer/D/model/serializer list; never a package glob",
    }


def _config_digest(config: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "config_sha256"}
    return canonical_sha256(payload)


def build_config(
    root: Path | str,
    paths: FeaturePaths,
    *,
    model_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
    plan: SupplementPlan,
    selection: Sequence[tuple[str, int]] | None,
    out_root: Path,
) -> dict[str, Any]:
    root = Path(root)
    fingerprint = producer_fingerprint(root, model_identity)
    selection_payload = (
        {"mode": "full", "round_keys": None}
        if selection is None
        else {"mode": "subset", "round_keys": [[str(ep), int(rid)] for ep, rid in sorted(selection)]}
    )
    payload = {
        "stage": STAGE + ".config",
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "scope": SCOPE,
        "source_tag": SOURCE_TAG,
        "storage": {
            "matrix_file": MATRIX_FILE,
            "index_file": INDEX_FILE,
            "journal_dir": JOURNAL_DIR,
            "dtype": FEATURE_DTYPE,
            "hidden_size": HIDDEN_SIZE,
            "row_bytes": ROW_BYTES,
            "layout": [ENDPOINT_ROLE, REFERENCE_ROLE],
            "all_token_or_all_layer_activations": 0,
            "per_round_vector_copies": 0,
        },
        "selection": selection_payload,
        "plan": {
            "path": _rel(root, paths.plan),
            "sha256": plan.plan_sha256,
            "rounds_path": _rel(root, paths.rounds),
            "rounds_sha256": plan.rounds_sha256,
            "counts": plan.counts,
        },
        "data": {
            "authority": _rel(root, paths.authority),
            "authority_sha256": sha256_file(paths.authority),
            "repaired_train": _rel(root, paths.repaired_train),
            "repaired_train_sha256": sha256_file(paths.repaired_train),
            "repaired_dev": _rel(root, paths.repaired_dev),
            "repaired_dev_sha256": sha256_file(paths.repaired_dev),
            "old_features_root": _rel(root, paths.old_features_root),
            "old_cot_index_sha256": sha256_file(paths.old_features_root / SCOPE / INDEX_FILE),
            "old_cot_manifest_sha256": sha256_file(paths.old_features_root / SCOPE / MANIFEST_FILE),
        },
        "producer_fingerprint": fingerprint,
        "producer_fingerprint_sha256": canonical_sha256(fingerprint),
        "model_identity": dict(model_identity),
        "model_identity_sha256": canonical_sha256(model_identity),
        "runtime_summary": {
            "transformers": runtime.get("transformers"),
            "torch": runtime.get("torch"),
            "tokenizers": runtime.get("tokenizers"),
            "attention_implementation": runtime.get("attention_implementation"),
            "backbone_output": runtime.get("backbone_output"),
            "feature_dtype": runtime.get("feature_dtype"),
            "inference_dtype": runtime.get("inference_dtype"),
        },
        "declarations": {
            "one_assistant_round_per_forward": True,
            "max_batch_size": 1,
            "causal": True,
            "logits_used": False,
            "text_generated": False,
            "victim_regenerated": False,
            "weights_modified": False,
            "truncation": False,
            "history_cot_replayed": False,
            "token_ids_encoded_or_decoded": False,
            "positions_selected_by_label": False,
            "test_raw_or_vector_read": False,
            "all_439_endpoints_regardless_of_label_status": True,
            "reference_rows_used_only_for_geometry": True,
        },
    }
    digest = canonical_sha256(payload)
    config = dict(payload)
    config["config_sha256"] = digest
    out_dir = Path(out_root) / SCOPE
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / CONFIG_FILE
    if config_path.is_file():
        existing = read_json(config_path)
        if _config_digest(existing) != digest:
            raise FeatureStoreError(
                f"{out_dir}: existing feature config is stale (plan/model/code/data changed); refusing to reset"
            )
    else:
        atomic_write_json(config_path, config)
    return config


# --------------------------------------------------------------------------- extraction helpers


def _sequence_source_hashes(sequence_episode: Any, sequence: Any) -> dict[str, Any]:
    record = dict(getattr(sequence_episode.episode, "record", {}) or {})
    keys = (
        "raw_trace_sha256",
        "content_sha256",
        "trajectory_sha256",
        "messages_sha256",
        "source_path",
        "replay_final",
        "replay_strict_hijacked",
    )
    hashes = {name: record[name] for name in keys if record.get(name) is not None}
    hashes.update(
        {
            "prompt_token_sha256": getattr(sequence, "prompt_token_sha256", None),
            "completion_token_sha256": getattr(sequence, "completion_token_sha256", None),
            "sequence_sha256": sequence.sequence_sha256,
            "sequence_length": int(sequence.sequence_length),
        }
    )
    return hashes


def _position_rows(plan_round: PlanRound) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for endpoint in plan_round.endpoints:
        rows.append(
            {
                "role": ENDPOINT_ROLE,
                "feature_key": endpoint["feature_key"],
                "event_id": endpoint["event_id"],
                "event_type": endpoint["event_type"],
                "token_index": int(endpoint["token_index"]),
                "completion_token_index": int(endpoint["completion_token_index"]),
                "index": int(endpoint.get("index", 0) or 0),
                "segment_index": endpoint.get("segment_index"),
                "split": endpoint["split"],
                "episode_id": endpoint["episode_id"],
                "round_id": int(endpoint["round_id"]),
                "prompt_length": int(endpoint["prompt_length"]),
                "sequence_length": int(endpoint["sequence_length"]),
                "sequence_sha256": endpoint["round_sequence_sha256"],
                "label_status": endpoint["label_status"],
                "repair_status": endpoint["repair_status"],
            }
        )
    if plan_round.reference is None:
        raise FeatureStoreError(f"{plan_round.key}: old round_end reference is not attached")
    reference = dict(plan_round.reference)
    reference.update(
        {
            "split": plan_round.split,
            "episode_id": plan_round.episode_id,
            "round_id": int(plan_round.round_id),
            "prompt_length": int(plan_round.prompt_length),
            "sequence_length": int(plan_round.sequence_length),
            "sequence_sha256": plan_round.reference["old_sequence_sha256"],
        }
    )
    rows.append(reference)
    return rows


def _unique_offsets(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted({int(row["token_index"]) for row in rows})


def build_round_identity(
    config: Mapping[str, Any],
    plan_round: PlanRound,
    sequence_episode: Any,
    sequence: Any,
    *,
    row_start: int,
) -> dict[str, Any]:
    if str(sequence.sequence_sha256) != str(plan_round.endpoints[0]["round_sequence_sha256"]):
        raise FeatureStoreError(f"{plan_round.key}: archived sequence hash differs from the D plan")
    if int(sequence.sequence_length) != plan_round.sequence_length:
        raise FeatureStoreError(f"{plan_round.key}: archived sequence length differs from the D plan")
    if int(sequence.prompt_length) != plan_round.prompt_length:
        raise FeatureStoreError(f"{plan_round.key}: archived prompt length differs from the D plan")
    rows = _position_rows(plan_round)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "config_sha256": config["config_sha256"],
        "producer_fingerprint_sha256": config["producer_fingerprint_sha256"],
        "model_identity_sha256": config["model_identity_sha256"],
        "scope": SCOPE,
        "source_tag": SOURCE_TAG,
        "episode_id": plan_round.episode_id,
        "namespaced_episode_id": plan_round.episode_id,
        "split": plan_round.split,
        "round_id": int(plan_round.round_id),
        "message_index": int(getattr(sequence, "message_index", 0)),
        "sequence_sha256": str(sequence.sequence_sha256),
        "sequence_length": int(sequence.sequence_length),
        "prompt_length": int(sequence.prompt_length),
        "completion_length": int(sequence.completion_length),
        "prompt_token_sha256": getattr(sequence, "prompt_token_sha256", None),
        "completion_token_sha256": getattr(sequence, "completion_token_sha256", None),
        "terminal_token_id": int(sequence.terminal_token_id),
        "sequence_serializer": "archived_prompt_completion_token_ids",
        "sequence_provenance": str(getattr(sequence_episode, "provenance", "")),
        "history_cot_replayed": False,
        "source_hashes": _sequence_source_hashes(sequence_episode, sequence),
        "row_start": int(row_start),
        "event_count": len(rows),
        "rows": rows,
        "offsets": _unique_offsets(rows),
    }
    return identity


def round_payload(vectors: Any, rows: Sequence[Mapping[str, Any]], offsets: Sequence[int]) -> bytes:
    import torch

    if not isinstance(vectors, torch.Tensor):
        raise FeatureStoreError("the extractor did not return a torch tensor")
    if vectors.dtype is not torch.bfloat16:
        raise FeatureStoreError(f"the extractor returned {vectors.dtype}, not bfloat16")
    if tuple(vectors.shape) != (len(offsets), HIDDEN_SIZE):
        raise FeatureStoreError(f"unexpected hidden-state shape {tuple(vectors.shape)}")
    values = vectors.float()
    if not bool(torch.isfinite(values).all()):
        raise FeatureStoreError("the extractor returned a non-finite hidden state")
    index = {int(offset): position for position, offset in enumerate(offsets)}
    selected = vectors[[index[int(row["token_index"])] for row in rows]]
    selected_values = selected.float()
    if not bool(torch.isfinite(selected_values).all()):
        raise FeatureStoreError("a selected feature vector is non-finite")
    if bool((torch.linalg.vector_norm(selected_values, dim=1) == 0).any()):
        raise FeatureStoreError("a selected feature vector is all zeros")
    return selected.contiguous().view(torch.uint8).numpy().tobytes()


def prefix_comparison(extractor: Any, sequence: Any, token_index: int, full_vector: Any) -> dict[str, Any]:
    import torch

    token_index = int(token_index)
    prefix_ids = list(sequence.input_ids[: token_index + 1])
    prefix_vector = extractor.extract_prefix(prefix_ids)
    if not isinstance(prefix_vector, torch.Tensor):
        raise FeatureStoreError("extract_prefix did not return a torch tensor")
    if tuple(prefix_vector.shape) == (1, HIDDEN_SIZE):
        prefix_vector = prefix_vector[0]
    if tuple(prefix_vector.shape) != (HIDDEN_SIZE,):
        raise FeatureStoreError(f"unexpected prefix vector shape {tuple(prefix_vector.shape)}")
    full = full_vector.float()
    prefix = prefix_vector.float()
    difference = full - prefix
    relative_l2 = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(prefix))
    cosine = float(torch.nn.functional.cosine_similarity(full, prefix, dim=0))
    max_abs = float(difference.abs().max())
    within = cosine >= PREFIX_COSINE_MIN and relative_l2 <= PREFIX_RELATIVE_L2_MAX
    return {
        "round_id": int(sequence.round_id),
        "sequence_sha256": str(sequence.sequence_sha256),
        "feature_token_index": token_index,
        "full_sequence_length": int(sequence.sequence_length),
        "prefix_length": len(prefix_ids),
        "bitwise_equal": bool(torch.equal(full_vector.view(torch.uint16), prefix_vector.view(torch.uint16))),
        "max_abs_error": max_abs,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "within_bf16_tolerance": within,
        "tolerance": {
            "cosine_min": PREFIX_COSINE_MIN,
            "relative_l2_max": PREFIX_RELATIVE_L2_MAX,
            "note": "predeclared BF16 length-dependent reduction tolerance; not an oracle future-token test",
        },
    }


def _write_progress(path: Path, progress: Mapping[str, Any]) -> None:
    payload = dict(progress)
    payload["updated_utc"] = _now()
    atomic_write_json(path, payload)


def _append_failure(out_dir: Path, entry: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(entry), ensure_ascii=False, sort_keys=True) + "\n"
    with (out_dir / FAILURES_FILE).open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _select_smoke_rounds(plan: SupplementPlan) -> tuple[tuple[str, int], ...]:
    keys: list[tuple[str, int]] = []
    wanted_episode_prefixes = (
        "cot32k::agentlab-d40c",
        "cot32k::agentlab-a72d",
    )
    for prefix in wanted_episode_prefixes:
        match = next((row for row in plan.rounds if row.episode_id.startswith(prefix)), None)
        if match is None:
            raise FeatureStoreError(f"smoke frame {prefix!r} is not in the plan")
        keys.append(match.key)
    unicode_round = next(
        (
            row
            for row in plan.rounds
            if any(endpoint["repair_status"] == "utf8_corrected" for endpoint in row.endpoints)
        ),
        None,
    )
    if unicode_round is None:
        raise FeatureStoreError("no UTF-8 corrected smoke frame in the plan")
    keys.append(unicode_round.key)
    return tuple(keys)


# --------------------------------------------------------------------------- extraction


def extract_rounds(
    root: Path | str = ".",
    *,
    out_root: Path | str | None = None,
    paths: FeaturePaths | None = None,
    selection: Sequence[tuple[str, int]] | None = None,
    model_path: Path | str | None = None,
    device: str = "cuda",
    extractor: Any | None = None,
    finalize: bool = True,
    prefix_check: bool = False,
    progress_every: int = 1,
    scopes: Sequence[str] = SCOPES,
) -> dict[str, Any]:
    root = Path(root)
    paths = paths or default_paths(root)
    out_root = Path(out_root) if out_root is not None else paths.supplement_root
    authority = load_authority(paths.authority)
    plan = load_plan(paths, authority, verify_repaired=True)
    old = OldIndex(paths.old_features_root, authority, scopes=scopes)
    plan.attach_old_references(old)
    selected_rounds = plan.select_rounds(selection)
    selected_keys = {row.key for row in selected_rounds}
    expected_endpoints = sum(row.endpoint_count for row in selected_rounds)
    expected_rows = sum(row.row_count for row in selected_rounds)

    if extractor is None:
        from sdsafe_fine_confirm.feature_model import load_extractor, extractor_runtime, model_identity

        extractor = load_extractor(Path(model_path) if model_path is not None else paths.model_root, device=device)
        runtime = extractor_runtime(extractor)
        model_identity_payload = model_identity(runtime)
    else:
        from sdsafe_fine_confirm.feature_model import extractor_runtime, model_identity

        runtime = extractor_runtime(extractor)
        model_identity_payload = model_identity(runtime)

    config = build_config(
        root,
        paths,
        model_identity=model_identity_payload,
        runtime=runtime,
        plan=plan,
        selection=None if len(selected_rounds) == len(plan.rounds) else tuple(selected_keys),
        out_root=out_root,
    )
    out_dir = Path(out_root) / SCOPE
    manifest_path = out_dir / MANIFEST_FILE
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("status") == "complete":
            if manifest.get("config_sha256") != config["config_sha256"]:
                raise FeatureStoreError(f"{out_dir}: complete feature manifest belongs to another config")
            return {
                "status": "complete",
                "reused_complete": True,
                "out_dir": str(out_dir),
                "config_sha256": config["config_sha256"],
                "counts": manifest.get("counts"),
                "manifest": manifest,
            }

    from sdsafe_fine_confirm.sequences import iter_sequences

    episode_ids = sorted({row.episode_id.split("::", 1)[1] for row in selected_rounds})
    sequences = {
        episode.episode.namespaced_episode_id: episode
        for episode in iter_sequences(root, SCOPE, episode_ids=episode_ids, tokenizer=extractor.tokenizer)
    }
    missing = sorted({row.episode_id for row in selected_rounds} - set(sequences))
    if missing:
        raise FeatureStoreError(f"archived sequence builder did not return selected episodes: {missing[:5]}")

    started = time.time()
    progress_path = out_dir / PROGRESS_FILE
    progress: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "scope": SCOPE,
        "status": "extracting",
        "complete": False,
        "config_sha256": config["config_sha256"],
        "selection": config["selection"],
        "planned": {
            "rounds": len(selected_rounds),
            "endpoints": expected_endpoints,
            "matrix_rows": expected_rows,
        },
        "completed": {"rounds": 0, "endpoints": 0, "reference_rows": 0, "matrix_rows": 0},
        "resumed_this_process": {"rounds": 0, "endpoints": 0},
        "forwarded_this_process": {"rounds": 0, "endpoints": 0},
        "qwen_forwards_this_process": 0,
        "prefix_checks": {"performed": 0, "passed": 0, "failed": 0} if prefix_check else None,
        "last_episode_id": None,
        "last_round_id": None,
        "started_utc": _now(),
    }
    _write_progress(progress_path, progress)
    writer: SupplementAppendStore | None = None
    prefix_results: list[dict[str, Any]] = []
    prefix_target: tuple[str, int, int] | None = None
    if prefix_check:
        for row in selected_rounds:
            if row.split == SPLIT_DEV:
                for endpoint in row.endpoints:
                    if endpoint["repair_status"] == "candidate_new_position":
                        prefix_target = (row.episode_id, row.round_id, int(endpoint["token_index"]))
                        break
            if prefix_target is not None:
                break
        if prefix_target is None:
            raise FeatureStoreError("no early DEV new-text endpoint is available for the causal prefix check")

    completed = {"rounds": 0, "endpoints": 0, "reference_rows": 0, "matrix_rows": 0}
    resumed = {"rounds": 0, "endpoints": 0}
    forwarded = {"rounds": 0, "endpoints": 0}
    running_row = 0
    try:
        writer = SupplementAppendStore(out_dir, config["config_sha256"])
        progress["recovered_partial_tail_bytes"] = writer.truncated_tail_bytes
        for plan_round in selected_rounds:
            sequence_episode = sequences[plan_round.episode_id]
            sequence = sequence_episode.rounds[plan_round.round_id]
            identity = build_round_identity(config, plan_round, sequence_episode, sequence, row_start=running_row)
            key = plan_round.key
            if key in writer.journals:
                writer.reuse(key, identity)
                resumed["rounds"] += 1
                resumed["endpoints"] += plan_round.endpoint_count
            else:
                offsets = identity["offsets"]
                vectors = extractor.extract(sequence.input_ids, offsets)
                if prefix_check and prefix_target is not None and prefix_target[:2] == key:
                    target_offset = prefix_target[2]
                    target_row = next(row for row in identity["rows"] if int(row["token_index"]) == target_offset)
                    position = offsets.index(int(target_offset))
                    comparison = prefix_comparison(extractor, sequence, target_offset, vectors[position])
                    comparison.update(
                        {
                            "episode_id": plan_round.episode_id,
                            "event_id": target_row["event_id"],
                            "event_type": target_row["event_type"],
                        }
                    )
                    prefix_results.append(comparison)
                    progress["qwen_forwards_this_process"] += 1
                    progress["prefix_checks"]["performed"] += 1
                    if comparison["within_bf16_tolerance"]:
                        progress["prefix_checks"]["passed"] += 1
                    else:
                        progress["prefix_checks"]["failed"] += 1
                payload = round_payload(vectors, identity["rows"], offsets)
                writer.append(identity, payload)
                forwarded["rounds"] += 1
                forwarded["endpoints"] += plan_round.endpoint_count
            running_row += plan_round.row_count
            completed["rounds"] += 1
            completed["endpoints"] += plan_round.endpoint_count
            completed["reference_rows"] += 1
            completed["matrix_rows"] = running_row
            progress["completed"] = dict(completed)
            progress["resumed_this_process"] = dict(resumed)
            progress["forwarded_this_process"] = dict(forwarded)
            progress["qwen_forwards_this_process"] = forwarded["rounds"] + (len(prefix_results) if prefix_check else 0)
            progress["last_episode_id"] = plan_round.episode_id
            progress["last_round_id"] = plan_round.round_id
            if progress_every <= 1 or completed["rounds"] % progress_every == 0:
                _write_progress(progress_path, progress)
        if writer is not None:
            writer.close()
            writer = None
        committed = load_journals(out_dir)
        if set(committed) != selected_keys:
            missing_keys = sorted(selected_keys - set(committed))[:5]
            extra_keys = sorted(set(committed) - selected_keys)[:5]
            raise FeatureStoreError(f"{out_dir}: journal coverage mismatch missing={missing_keys} extra={extra_keys}")
        if completed["rounds"] != len(selected_rounds) or completed["endpoints"] != expected_endpoints:
            raise FeatureStoreError(f"{out_dir}: extraction coverage gap {completed}")
        if completed["matrix_rows"] != expected_rows:
            raise FeatureStoreError(f"{out_dir}: matrix rows {completed['matrix_rows']} != {expected_rows}")
        progress["status"] = "extracted" if finalize else "partial"
        progress["complete"] = False
        _write_progress(progress_path, progress)
        if not finalize:
            return {
                "status": "partial",
                "out_dir": str(out_dir),
                "config_sha256": config["config_sha256"],
                "counts": dict(completed),
                "forwarded_this_process": dict(forwarded),
                "resumed_this_process": dict(resumed),
                "progress": progress,
            }
        summary = finalize_supplement(
            root,
            out_dir,
            plan,
            selected_rounds,
            config,
            runtime,
            old,
            prefix_results=prefix_results,
            elapsed_seconds=time.time() - started,
            forwarded=forwarded,
            resumed=resumed,
        )
        progress["status"] = "complete"
        progress["complete"] = True
        progress["completed"] = dict(summary["counts"])
        progress["elapsed_seconds_this_process"] = summary["extraction"]["elapsed_seconds_this_process"]
        _write_progress(progress_path, progress)
        return summary
    except Exception as error:  # noqa: BLE001 - every failure stays explicit
        progress["status"] = "failed"
        progress["complete"] = False
        progress["failure_reason"] = f"{type(error).__name__}: {error}"
        _write_progress(progress_path, progress)
        _append_failure(
            out_dir,
            {
                "stage": STAGE,
                "last_episode_id": progress.get("last_episode_id"),
                "last_round_id": progress.get("last_round_id"),
                "reason": type(error).__name__,
                "error": str(error),
                "occurred_utc": _now(),
            },
        )
        raise
    finally:
        if writer is not None:
            writer.close()


# --------------------------------------------------------------------------- finalization / validation


def _matrix_statistics(path: Path, rows: int) -> dict[str, Any]:
    import torch

    if rows == 0:
        raise FeatureStoreError(f"{path}: empty feature matrix")
    norms: list[Any] = []
    value_min = float("inf")
    value_max = float("-inf")
    with path.open("rb") as handle:
        remaining = rows
        while remaining:
            take = min(4096, remaining)
            raw = bytearray(handle.read(take * ROW_BYTES))
            if len(raw) != take * ROW_BYTES:
                raise FeatureStoreError(f"{path}: short aggregate read")
            values = (
                torch.frombuffer(raw, dtype=torch.uint16)
                .view(torch.bfloat16)
                .reshape(take, HIDDEN_SIZE)
                .float()
            )
            if not bool(torch.isfinite(values).all()):
                raise FeatureStoreError(f"{path}: non-finite hidden state in the aggregate")
            norms.append(torch.linalg.vector_norm(values, dim=1))
            value_min = min(value_min, float(values.min()))
            value_max = max(value_max, float(values.max()))
            remaining -= take
    norm = torch.cat(norms)
    zero = int((norm == 0).sum())
    if zero:
        raise FeatureStoreError(f"{path}: {zero} all-zero feature vectors")
    return {
        "rows": rows,
        "finite": True,
        "dimensions": HIDDEN_SIZE,
        "zero_vectors": zero,
        "norm_min": float(norm.min()),
        "norm_mean": float(norm.mean()),
        "norm_max": float(norm.max()),
        "value_min": value_min,
        "value_max": value_max,
    }


def _geometry_comparison(
    out_dir: Path,
    journals: Mapping[tuple[str, int], Mapping[str, Any]],
    old: OldIndex,
    plan: SupplementPlan,
    config: Mapping[str, Any],
    matrix_rows: int,
) -> dict[str, Any]:
    import torch

    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    ordered = sorted(journals.values(), key=lambda journal: (journal["identity"]["episode_id"], journal["identity"]["round_id"]))
    for journal in ordered:
        identity = journal["identity"]
        key = (str(identity["episode_id"]), int(identity["round_id"]))
        plan_round = plan.by_round[key]
        reference = plan_round.reference
        if reference is None:
            raise FeatureStoreError(f"{key}: no old round_end reference")
        reference_row = int(identity["row_start"]) + int(identity["event_count"]) - 1
        fresh_raw = _read_matrix_raw(out_dir / MATRIX_FILE, int(matrix_rows), reference_row)
        old_raw = old.read_raw_local(reference["old_scope"], int(reference["old_local_feature_row"]))
        fresh = torch.frombuffer(bytearray(fresh_raw), dtype=torch.uint16).view(torch.bfloat16).float()
        previous = torch.frombuffer(bytearray(old_raw), dtype=torch.uint16).view(torch.bfloat16).float()
        difference = fresh - previous
        relative_l2 = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(previous))
        cosine = float(torch.nn.functional.cosine_similarity(fresh, previous, dim=0))
        max_abs = float(difference.abs().max())
        bitwise = fresh_raw == old_raw
        within = cosine >= COMPATIBILITY_COSINE_MIN and relative_l2 <= COMPATIBILITY_RELATIVE_L2_MAX
        row = {
            "episode_id": key[0],
            "round_id": key[1],
            "split": plan_round.split,
            "fresh_feature_row": reference_row,
            "old_scope": reference["old_scope"],
            "old_local_feature_row": int(reference["old_local_feature_row"]),
            "old_global_feature_row": int(reference["old_global_feature_row"]),
            "old_event_id": reference["event_id"],
            "sequence_sha256": str(identity["sequence_sha256"]),
            "bitwise_equal": bool(bitwise),
            "cosine_similarity": cosine,
            "relative_l2_error": relative_l2,
            "max_abs_error": max_abs,
            "within_predeclared_tolerance": bool(within),
        }
        comparisons.append(row)
        if not within:
            failures.append(row)
    all_bitwise = bool(comparisons) and all(row["bitwise_equal"] for row in comparisons)
    status = "bitwise_exact" if all_bitwise else ("compatible_tolerance" if not failures else "FAIL")
    report = {
        "stage": STAGE + ".geometry",
        "status": status,
        "rows_compared": len(comparisons),
        "all_bitwise_equal": all_bitwise,
        "predeclared_tolerance": {
            "cosine_min": COMPATIBILITY_COSINE_MIN,
            "relative_l2_max": COMPATIBILITY_RELATIVE_L2_MAX,
            "note": "predeclared before results; never loosened",
        },
        "min_cosine_similarity": min((row["cosine_similarity"] for row in comparisons), default=None),
        "max_relative_l2_error": max((row["relative_l2_error"] for row in comparisons), default=None),
        "max_abs_error": max((row["max_abs_error"] for row in comparisons), default=None),
        "failures": failures,
        "mixed_producer_disclosure": {
            "old_feature_store": "artifacts/fine_confirm/features (immutable cached_canonical_bf16)",
            "new_feature_store": str(out_dir),
            "old_model_identity": old.manifests.get(SCOPE, {}).get("model"),
            "new_model_identity_sha256": config.get("model_identity_sha256"),
            "note": "same frozen Qwen3-8B BF16 extractor contract; old/new producer code differs, so bitwise equality is reported only if observed",
        },
        "comparisons": comparisons,
    }
    return report


def finalize_supplement(
    root: Path | str,
    out_dir: Path | str,
    plan: SupplementPlan,
    selected_rounds: Sequence[PlanRound],
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    old: OldIndex,
    *,
    prefix_results: Sequence[Mapping[str, Any]] = (),
    elapsed_seconds: float = 0.0,
    forwarded: Mapping[str, Any] | None = None,
    resumed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    out_dir = Path(out_dir)
    journals = load_journals(out_dir)
    expected_keys = {row.key for row in selected_rounds}
    if set(journals) != expected_keys:
        missing = sorted(expected_keys - set(journals))[:5]
        extra = sorted(set(journals) - expected_keys)[:5]
        raise FeatureStoreError(f"{out_dir}: finalize coverage mismatch missing={missing} extra={extra}")
    committed_rows = verify_journal_prefix(journals.values())
    if committed_rows != sum(row.row_count for row in selected_rounds):
        raise FeatureStoreError(f"{out_dir}: committed matrix rows are inconsistent with the plan")
    matrix_path = out_dir / MATRIX_FILE
    if matrix_path.stat().st_size != committed_rows * ROW_BYTES:
        raise FeatureStoreError(f"{out_dir}: matrix size is inconsistent before finalize")

    index_rows: list[dict[str, Any]] = []
    endpoint_count = 0
    split_counts: Counter = Counter()
    repair_counts: Counter = Counter()
    label_counts: Counter = Counter()
    for plan_round in selected_rounds:
        journal = journals[plan_round.key]
        identity = journal["identity"]
        if identity["episode_id"] != plan_round.episode_id or int(identity["round_id"]) != plan_round.round_id:
            raise FeatureStoreError(f"{plan_round.key}: journal identity disagrees with the plan")
        if int(identity["event_count"]) != plan_round.row_count:
            raise FeatureStoreError(f"{plan_round.key}: journal row count disagrees with the plan")
        for position, row in enumerate(identity["rows"]):
            feature_row = int(identity["row_start"]) + position
            index_row = {
                "schema_version": SCHEMA_VERSION,
                "stage": STAGE + ".index",
                "role": row["role"],
                "feature_key": row["feature_key"],
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "scope": SCOPE,
                "source_tag": SOURCE_TAG,
                "split": plan_round.split,
                "episode_id": plan_round.episode_id,
                "namespaced_episode_id": plan_round.episode_id,
                "round_id": int(plan_round.round_id),
                "token_index": int(row["token_index"]),
                "completion_token_index": int(row.get("completion_token_index")) if row.get("completion_token_index") is not None else None,
                "feature_token_index": int(row["token_index"]),
                "prompt_length": int(plan_round.prompt_length),
                "sequence_length": int(plan_round.sequence_length),
                "sequence_sha256": str(row.get("sequence_sha256") or identity["sequence_sha256"]),
                "round_sequence_sha256": str(row.get("sequence_sha256") or identity["sequence_sha256"]),
                "feature_file": MATRIX_FILE,
                "feature_row": feature_row,
                "feature_byte_offset": feature_row * ROW_BYTES,
                "dtype": FEATURE_DTYPE,
                "hidden_size": HIDDEN_SIZE,
                "kind": SUPPLEMENTAL_FEATURE_KIND,
                "status": "extracted",
                "journal": str(journal_path(out_dir, plan_round.episode_id, plan_round.round_id).relative_to(out_dir)),
                "journal_identity_sha256": journal["identity_sha256"],
                "config_sha256": config["config_sha256"],
            }
            if row["role"] == ENDPOINT_ROLE:
                endpoint_count += 1
                split_counts[plan_round.split] += 1
                repair_counts[str(row.get("repair_status"))] += 1
                label_counts[str(row.get("label_status"))] += 1
                index_row["label_status"] = row.get("label_status")
                index_row["repair_status"] = row.get("repair_status")
            elif row["role"] == REFERENCE_ROLE:
                index_row["old_scope"] = row.get("old_scope")
                index_row["old_local_feature_row"] = row.get("old_local_feature_row")
                index_row["old_global_feature_row"] = row.get("old_global_feature_row")
                index_row["old_event_id"] = row.get("event_id")
            else:
                raise FeatureStoreError(f"{plan_round.key}: unknown row role {row['role']!r}")
            index_rows.append(index_row)
    endpoint_keys = {str(row["feature_key"]) for row in index_rows if row["role"] == ENDPOINT_ROLE}
    expected_endpoint_keys = {
        endpoint["feature_key"] for plan_round in selected_rounds for endpoint in plan_round.endpoints
    }
    if endpoint_keys != expected_endpoint_keys:
        raise FeatureStoreError("final index does not cover exactly the selected accepted D feature_keys")
    if endpoint_count != sum(row.endpoint_count for row in selected_rounds):
        raise FeatureStoreError("final endpoint count is inconsistent")
    if len([row for row in index_rows if row["role"] == REFERENCE_ROLE]) != len(selected_rounds):
        raise FeatureStoreError("final reference row count is inconsistent")
    expected_split_counts: Counter = Counter()
    expected_repair_counts: Counter = Counter()
    for plan_round in selected_rounds:
        for endpoint in plan_round.endpoints:
            expected_split_counts[plan_round.split] += 1
            expected_repair_counts[str(endpoint["repair_status"])] += 1
    if dict(split_counts) != dict(expected_split_counts):
        raise FeatureStoreError(f"final split counts are inconsistent: {dict(split_counts)} != {dict(expected_split_counts)}")
    if dict(repair_counts) != dict(expected_repair_counts):
        raise FeatureStoreError(
            f"final repair-status counts are inconsistent: {dict(repair_counts)} != {dict(expected_repair_counts)}"
        )
    for row in index_rows:
        if int(row["feature_byte_offset"]) != int(row["feature_row"]) * ROW_BYTES:
            raise FeatureStoreError(f"{row['feature_key']}: final byte offset is not derived from the true index")
    index_rows.sort(key=lambda row: int(row["feature_row"]))
    if [int(row["feature_row"]) for row in index_rows] != list(range(len(index_rows))):
        raise FeatureStoreError("final index rows are not contiguous from zero")

    statistics = _matrix_statistics(matrix_path, committed_rows)
    geometry = _geometry_comparison(out_dir, journals, old, plan, config, committed_rows)
    if geometry["status"] == "FAIL":
        raise FeatureStoreError(
            "old/new round_end geometry is outside the predeclared compatibility tolerance; "
            f"failures={geometry['failures'][:3]}"
        )
    if prefix_results and not all(row["within_bf16_tolerance"] for row in prefix_results):
        raise FeatureStoreError(f"prefix causal check failed: {[row for row in prefix_results if not row['within_bf16_tolerance']][:3]}")

    atomic_write_jsonl(out_dir / INDEX_FILE, index_rows)
    atomic_write_json(out_dir / GEOMETRY_FILE, geometry)
    validation = {
        "stage": STAGE + ".validation",
        "status": "PASS",
        "config_sha256": config["config_sha256"],
        "counts": {
            "endpoints": endpoint_count,
            "rounds": len(selected_rounds),
            "reference_rows": len(selected_rounds),
            "matrix_rows": committed_rows,
            "splits": dict(split_counts),
            "repair_status": dict(repair_counts),
            "label_status": dict(label_counts),
        },
        "matrix_statistics": statistics,
        "geometry": {
            "status": geometry["status"],
            "rows_compared": geometry["rows_compared"],
            "all_bitwise_equal": geometry["all_bitwise_equal"],
            "min_cosine_similarity": geometry["min_cosine_similarity"],
            "max_relative_l2_error": geometry["max_relative_l2_error"],
        },
        "prefix_checks": {
            "performed": len(prefix_results),
            "status": "PASS" if all(row["within_bf16_tolerance"] for row in prefix_results) else "FAIL",
            "comparisons": list(prefix_results),
        },
        "checks": {
            "accepted_plan_outside_basic_schema_crosscheck": "PASS",
            "endpoint_feature_key_coverage_exact": "PASS",
            "journal_identity_and_byte_ranges": "PASS",
            "finite_4096_dimensions": "PASS",
            "all_439_endpoints_regardless_of_label_status": "PASS",
            "test_vector_read": False,
        },
        "completed_utc": _now(),
    }
    atomic_write_json(out_dir / VALIDATION_FILE, validation)

    manifest = {
        "stage": STAGE,
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "status": "complete",
        "scope": SCOPE,
        "source_tag": SOURCE_TAG,
        "completed_utc": _now(),
        "config_sha256": config["config_sha256"],
        "producer_fingerprint_sha256": config["producer_fingerprint_sha256"],
        "model_identity_sha256": config["model_identity_sha256"],
        "model_identity": config["model_identity"],
        "selection": config["selection"],
        "plan": config["plan"],
        "data": config["data"],
        "counts": {
            "rounds": len(selected_rounds),
            "endpoints": endpoint_count,
            "reference_rows": len(selected_rounds),
            "matrix_rows": committed_rows,
            "splits": dict(split_counts),
            "repair_status": dict(repair_counts),
            "label_status": dict(label_counts),
            "qwen_forwards": int((forwarded or {}).get("rounds", len(selected_rounds))),
        },
        "storage": {
            "matrix": {
                "file": MATRIX_FILE,
                "dtype": FEATURE_DTYPE,
                "hidden_size": HIDDEN_SIZE,
                "shape": [committed_rows, HIDDEN_SIZE],
                "bytes": matrix_path.stat().st_size,
                "sha256": sha256_file(matrix_path),
            },
            "index": {"file": INDEX_FILE, "sha256": sha256_file(out_dir / INDEX_FILE), "rows": len(index_rows)},
            "journals": {"dir": JOURNAL_DIR, "rounds": len(journals)},
        },
        "matrix_statistics": statistics,
        "geometry_status": geometry["status"],
        "validation": {"file": VALIDATION_FILE, "status": "PASS"},
        "extraction": {
            "started_utc": None,
            "elapsed_seconds_this_process": float(elapsed_seconds),
            "forwarded_this_process": dict(forwarded or {}),
            "resumed_this_process": dict(resumed or {}),
            "runtime_summary": config["runtime_summary"],
            "declarations": config["declarations"],
        },
        "declarations": config["declarations"],
    }
    atomic_write_json(out_dir / MANIFEST_FILE, manifest)
    return {
        "status": "complete",
        "out_dir": str(out_dir),
        "config_sha256": config["config_sha256"],
        "counts": manifest["counts"],
        "storage": manifest["storage"],
        "geometry": geometry,
        "prefix_checks": validation["prefix_checks"],
        "manifest": manifest,
        "extraction": manifest["extraction"],
    }


# --------------------------------------------------------------------------- independent validation


def validate_sequences(
    root: Path | str,
    plan: SupplementPlan,
    *,
    tokenizer: Any | None = None,
    selection: Sequence[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    from sdsafe_fine_confirm.sequences import iter_sequences

    root = Path(root)
    selected = plan.select_rounds(selection)
    episode_ids = sorted({row.episode_id.split("::", 1)[1] for row in selected})
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(Path(root) / "models/Qwen3-8B", local_files_only=True)
    sequences = {
        episode.episode.namespaced_episode_id: episode
        for episode in iter_sequences(root, SCOPE, episode_ids=episode_ids, tokenizer=tokenizer)
    }
    verified = 0
    for plan_round in selected:
        sequence_episode = sequences.get(plan_round.episode_id)
        if sequence_episode is None:
            raise FeatureStoreError(f"{plan_round.episode_id}: archived episode is missing")
        sequence = sequence_episode.rounds.get(plan_round.round_id)
        if sequence is None:
            raise FeatureStoreError(f"{plan_round.key}: archived round is missing")
        hashes = {str(endpoint["round_sequence_sha256"]) for endpoint in plan_round.endpoints}
        if len(hashes) != 1 or sequence.sequence_sha256 not in hashes:
            raise FeatureStoreError(f"{plan_round.key}: archived sequence hash differs from the D plan")
        if int(sequence.sequence_length) != plan_round.sequence_length:
            raise FeatureStoreError(f"{plan_round.key}: archived sequence length differs")
        if int(sequence.prompt_length) != plan_round.prompt_length:
            raise FeatureStoreError(f"{plan_round.key}: archived prompt length differs")
        if int(sequence.terminal_token_id) != int(sequence.input_ids[-1]):
            raise FeatureStoreError(f"{plan_round.key}: archived terminal token is inconsistent")
        verified += 1
    return {"status": "PASS", "rounds_verified": verified, "sequences": sequences}


def validate_complete_store(
    root: Path | str = ".",
    *,
    paths: FeaturePaths | None = None,
    scopes: Sequence[str] = SCOPES,
    check_sequences: bool = True,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    root = Path(root)
    store = FeatureStore.open(root, paths=paths, scopes=scopes, require_complete=True, verify_repaired=True)
    assert store.supplement is not None
    out_dir = store.paths.supplement_root / SCOPE
    manifest = store.supplement.manifest
    if manifest.get("status") != "complete":
        raise FeatureStoreError(f"{out_dir}: feature store is not complete")
    if manifest.get("config_sha256") != read_json(out_dir / CONFIG_FILE).get("config_sha256"):
        raise FeatureStoreError(f"{out_dir}: manifest/config digest mismatch")
    config = read_json(out_dir / CONFIG_FILE)
    if _config_digest(config) != config.get("config_sha256"):
        raise FeatureStoreError(f"{out_dir}: config digest is not canonical")
    if config.get("producer_fingerprint_sha256") != canonical_sha256(config.get("producer_fingerprint")):
        raise FeatureStoreError(f"{out_dir}: producer fingerprint digest is inconsistent")
    if config.get("model_identity_sha256") != canonical_sha256(config.get("model_identity")):
        raise FeatureStoreError(f"{out_dir}: model identity digest is inconsistent")
    if manifest.get("producer_fingerprint_sha256") != config.get("producer_fingerprint_sha256"):
        raise FeatureStoreError(f"{out_dir}: manifest producer fingerprint differs from config")

    journals = load_journals(out_dir)
    if set(journals) != set(store.plan.by_round):
        missing = sorted(set(store.plan.by_round) - set(journals))[:5]
        extra = sorted(set(journals) - set(store.plan.by_round))[:5]
        raise FeatureStoreError(f"{out_dir}: journal coverage mismatch missing={missing} extra={extra}")
    committed_rows = verify_journal_prefix(journals.values())
    if committed_rows != int(manifest["counts"]["matrix_rows"]):
        raise FeatureStoreError(f"{out_dir}: journal rows != manifest matrix rows")
    matrix_path = out_dir / MATRIX_FILE
    if matrix_path.stat().st_size != committed_rows * ROW_BYTES:
        raise FeatureStoreError(f"{out_dir}: matrix size is inconsistent")
    if sha256_file(matrix_path) != manifest["storage"]["matrix"]["sha256"]:
        raise FeatureStoreError(f"{out_dir}: matrix sha256 differs from the manifest")
    statistics = _matrix_statistics(matrix_path, committed_rows)

    sequence_report = None
    sequences: dict[str, Any] = {}
    if check_sequences:
        sequence_report = validate_sequences(root, store.plan, tokenizer=tokenizer)
        sequences = sequence_report.pop("sequences")
    running_row = 0
    for plan_round in store.plan.rounds:
        journal = journals[plan_round.key]
        if sequence_report is not None:
            sequence_episode = sequences[plan_round.episode_id]
            sequence = sequence_episode.rounds[plan_round.round_id]
            expected = build_round_identity(config, plan_round, sequence_episode, sequence, row_start=running_row)
        else:
            expected = journal["identity"]
            expected["row_start"] = running_row
        if journal["identity"] != expected:
            raise FeatureStoreError(f"{plan_round.key}: committed journal identity is stale or forged")
        if journal.get("identity_sha256") != canonical_sha256(journal["identity"]):
            raise FeatureStoreError(f"{plan_round.key}: journal identity hash is not canonical")
        with matrix_path.open("rb") as handle:
            handle.seek(int(journal["identity"]["row_start"]) * ROW_BYTES)
            payload = handle.read(int(journal["identity"]["event_count"]) * ROW_BYTES)
        if len(payload) != int(journal["identity"]["event_count"]) * ROW_BYTES:
            raise FeatureStoreError(f"{plan_round.key}: short matrix read")
        if sha256_bytes(payload) != journal.get("features_sha256"):
            raise FeatureStoreError(f"{plan_round.key}: journal feature bytes changed")
        running_row += plan_round.row_count

    geometry = _geometry_comparison(out_dir, journals, store.old, store.plan, config, committed_rows)
    if geometry["status"] == "FAIL":
        raise FeatureStoreError(f"{out_dir}: geometry failed {geometry['failures'][:3]}")
    validation = {
        "stage": STAGE + ".validation",
        "status": "PASS",
        "config_sha256": config["config_sha256"],
        "counts": manifest["counts"],
        "matrix_statistics": statistics,
        "geometry": {
            "status": geometry["status"],
            "rows_compared": geometry["rows_compared"],
            "all_bitwise_equal": geometry["all_bitwise_equal"],
            "min_cosine_similarity": geometry["min_cosine_similarity"],
            "max_relative_l2_error": geometry["max_relative_l2_error"],
        },
        "sequence_validation": sequence_report,
        "checks": {
            "plan_authority_crosscheck": "PASS",
            "accepted_plan_crosscheck_against_D_repaired_rows": "PASS",
            "all_439_endpoint_refs_readable": "PASS",
            "journals_exact_338": "PASS",
            "all_rows_finite_4096": "PASS",
            "old_new_geometry_predeclared": geometry["status"],
            "test_vector_read": False,
        },
        "completed_utc": _now(),
    }
    atomic_write_json(out_dir / GEOMETRY_FILE, geometry)
    atomic_write_json(out_dir / VALIDATION_FILE, validation)
    return {
        "status": "PASS",
        "out_dir": str(out_dir),
        "config_sha256": config["config_sha256"],
        "counts": manifest["counts"],
        "matrix_statistics": statistics,
        "geometry": geometry,
        "sequence_validation": sequence_report,
        "validation": validation,
        "store": store,
    }


def read_all_plan_refs(store: FeatureStore) -> dict[str, Any]:
    """Read every accepted endpoint ref (never filtered by label status) for a final guard."""
    import torch

    refs = [row["ref"] for row in store.plan.rows]
    splits: Counter = Counter(row["split"] for row in store.plan.rows)
    outputs: dict[str, int] = {}
    for split in (SPLIT_TRAIN, SPLIT_DEV):
        selected = [row["ref"] for row in store.plan.rows if row["split"] == split]
        vectors = store.read(selected, split)
        if tuple(vectors.shape) != (len(selected), HIDDEN_SIZE):
            raise FeatureStoreError(f"{split}: reader returned {tuple(vectors.shape)}")
        if vectors.dtype is not torch.float32:
            raise FeatureStoreError(f"{split}: reader returned {vectors.dtype}, not float32")
        if not bool(torch.isfinite(vectors).all()):
            raise FeatureStoreError(f"{split}: reader returned non-finite vectors")
        outputs[split] = int(vectors.shape[0])
    if outputs.get(SPLIT_TRAIN) != 350 or outputs.get(SPLIT_DEV) != 89:
        raise FeatureStoreError(f"reader endpoint counts are wrong: {outputs}")
    if sum(outputs.values()) != EXPECTED_ENDPOINTS:
        raise FeatureStoreError("reader did not materialize all 439 endpoints")
    return {"status": "PASS", "refs": len(refs), "read_by_split": outputs, "splits": dict(splits)}
