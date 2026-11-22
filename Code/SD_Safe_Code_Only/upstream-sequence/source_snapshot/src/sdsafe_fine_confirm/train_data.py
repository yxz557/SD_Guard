"""Non-oracle training/evaluation data views for the fine-confirm probe seam.

This module never selects inference positions using labels, goal type, injection task,
source or final status.  The *only* inference selector is the event family:

``fine``     every ``tool_call`` and every visible ``content_segment`` event;
``control``  every ``round_end`` event (the matched whole-round control).

Training targets are separate and are the only place labels are used:

``first_hit``
    one positive per final-positive episode: the first *exact* ``security_flip`` event in
    that arm's trajectory.  Every later exact event is a negative (including an exact
    positive state after the first flip).  Exact rows of final-negative episodes are 0.
``state``
    exact ``prefix_security`` rows of final-positive episodes are 1; exact rows of
    final-negative episodes are 0.  Transient positive state rows of final-negative
    episodes are excluded from the positive class without being relabelled and are
    reported as such.

Unknown labels never enter the supervised loss (they remain null, never 0).  Safe visible
content from tool-target episodes is retained exactly as labelled.  Missing inference
positions do not delete an episode: its score is ``None``/``-inf`` and it stays in the
episode denominator with an explicit coverage record.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from sdsafe_event_data import (
    EVENT_CONTENT_SEGMENT,
    EVENT_ROUND_END,
    EVENT_TOOL_CALL,
    EVENT_TYPE_ORDER,
    LABEL_EXACT,
    LABEL_UNKNOWN,
)
from sdsafe_features.common import read_json, read_jsonl

from . import SCOPE_BY_SOURCE_TAG, SCOPES

FINE_EVENT_TYPES = (EVENT_TOOL_CALL, EVENT_CONTENT_SEGMENT)
CONTROL_EVENT_TYPES = (EVENT_ROUND_END,)
ARMS = ("fine", "control")
POLICIES = ("first_hit", "state")
COHORT_SOURCES = {
    "A": ("cot", "nothink", "ctx40k"),
    "B": ("cot", "nothink", "ctx40k", "agentlab"),
}
QWEN_SOURCES = ("cot", "nothink", "ctx40k")
EXPECTED_SEEDS = (0, 1, 2)
PRIMARY_BUDGET = 0.05
THRESHOLD_BUDGETS = (0.05, 0.02, 0.01)
HIDDEN_SIZE = 4096
MATRIX_FILE = "matrix.bf16"
INDEX_FILE = "index.jsonl"
DTYPE_BYTES = 2


class TrainingDataError(ValueError):
    """Raised for a non-oracle data-view contract violation."""


def arm_event_types(arm: str) -> tuple[str, ...]:
    if arm == "fine":
        return FINE_EVENT_TYPES
    if arm == "control":
        return CONTROL_EVENT_TYPES
    raise TrainingDataError(f"Unknown arm: {arm!r}")


def select_inference_rows(rows: Iterable[Mapping[str, Any]], arm: str) -> list[dict[str, Any]]:
    """The universal inference selector: event family only, never metadata or labels."""
    wanted = set(arm_event_types(arm))
    return [dict(row) for row in rows if str(row.get("event_type")) in wanted]


def event_order_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int, int, int]:
    """Actual token-boundary trajectory order, with a deterministic synthetic fallback.

    Real event/index rows carry ``feature_token_index``; ordering by
    ``(round_id, feature_token_index)`` is the physical generation order and therefore a
    visible content segment before a tool call in the same round sorts before the call.
    Only fixtures lacking that offset fall back to ``(round_id, message_index, event family,
    index, ...)``.  This fallback is never label/goal/source dependent.
    """
    offset = row.get("feature_token_index")
    if offset is not None:
        primary = int(offset)
        has_offset = 0
    else:
        primary = int(row.get("message_index", 0))
        has_offset = 1
    return (
        int(row.get("round_id", 0)),
        primary,
        has_offset,
        int(EVENT_TYPE_ORDER.get(str(row.get("event_type")), 99)),
        int(row.get("index", 0)),
        int(row.get("call_index", 0) or 0),
        int(row.get("segment_index", 0) or 0),
    )


def normalize_scope(row: Mapping[str, Any]) -> str:
    """Normalize a real scope name or a ``source_tag`` through ``SCOPE_BY_SOURCE_TAG``."""
    raw_scope = row.get("scope")
    raw_tag = row.get("source_tag")
    if raw_scope in SCOPES:
        scope = str(raw_scope)
    elif raw_scope in SCOPE_BY_SOURCE_TAG:
        scope = SCOPE_BY_SOURCE_TAG[str(raw_scope)]
    elif raw_tag in SCOPE_BY_SOURCE_TAG:
        scope = SCOPE_BY_SOURCE_TAG[str(raw_tag)]
    else:
        raise TrainingDataError(f"Unknown scope/source_tag: scope={raw_scope!r}, source_tag={raw_tag!r}")
    if raw_tag is not None:
        if raw_tag not in SCOPE_BY_SOURCE_TAG:
            raise TrainingDataError(f"Unknown source_tag: {raw_tag!r}")
        if SCOPE_BY_SOURCE_TAG[str(raw_tag)] != scope:
            raise TrainingDataError(
                f"Scope/source_tag disagree: scope={raw_scope!r}, source_tag={raw_tag!r}"
            )
    return scope


def assert_canonical_ids(rows: Iterable[Mapping[str, Any]]) -> None:
    """Reject local/mixed identities before they can silently merge across sources.

    Real cross-source rows must name a known scope and must already use the namespaced
    episode id as ``episode_id`` (``episode_id == namespaced_episode_id``).  Integrations
    normalize and globally remap per-scope feature rows before calling this seam.  A
    synthetic fixture must obey the same identity contract; there is no production bypass.
    """
    seen_events: set[str] = set()
    seen_episodes: set[str] = set()
    for row in rows:
        scope = normalize_scope(row)
        namespaced = row.get("namespaced_episode_id")
        episode_id = row.get("episode_id")
        if namespaced is None or str(namespaced) == "":
            raise TrainingDataError(
                f"Row {row.get('event_id')!r}: missing namespaced_episode_id for scope {scope}"
            )
        if episode_id is None or str(episode_id) != str(namespaced):
            raise TrainingDataError(
                f"Row {row.get('event_id')!r}: non-canonical local/mixed episode id "
                f"{episode_id!r} != namespaced {namespaced!r}"
            )
        event_id = row.get("event_id")
        if event_id is None or str(event_id) == "":
            raise TrainingDataError(f"Row for {episode_id}: missing event_id")
        if str(event_id) in seen_events:
            raise TrainingDataError(f"Duplicate event_id in rows: {event_id!r}")
        seen_events.add(str(event_id))
        seen_episodes.add(str(episode_id))


def _episode_final(row: Mapping[str, Any]) -> bool:
    if row.get("strict_final") is not None:
        return bool(row["strict_final"])
    if row.get("episode_final") is not None:
        return bool(row["episode_final"])
    raise TrainingDataError(f"Row {row.get('event_id')!r} has no frozen final label")


def episode_final_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, bool]:
    """Return one consistent frozen final label per episode; conflicting rows are fatal."""
    finals: dict[str, bool] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        value = _episode_final(row)
        if episode_id in finals and finals[episode_id] != value:
            raise TrainingDataError(f"{episode_id}: conflicting final labels in event rows")
        finals[episode_id] = value
    return finals


def episode_labels(episode_rows: Mapping[str, Mapping[str, Any]] | Iterable[Mapping[str, Any]]) -> dict[str, bool]:
    """Frozen episode labels (``strict_final``) from episode or event rows."""
    if isinstance(episode_rows, Mapping):
        source = episode_rows.values()
    else:
        source = episode_rows
    labels: dict[str, bool] = {}
    for row in source:
        episode_id = str(row["episode_id"])
        value = _episode_final(row)
        if episode_id in labels and labels[episode_id] != value:
            raise TrainingDataError(f"{episode_id}: conflicting final labels")
        labels[episode_id] = value
    return labels


def episode_group_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    groups: dict[str, str] = {}
    for row in rows:
        episode_id = str(row["episode_id"])
        group = str(row["group_id"])
        if episode_id in groups and groups[episode_id] != group:
            raise TrainingDataError(f"{episode_id}: conflicting group_id")
        groups[episode_id] = group
    return groups


def _status(row: Mapping[str, Any]) -> str:
    status = row.get("label_status")
    if status not in (LABEL_EXACT, LABEL_UNKNOWN):
        raise TrainingDataError(f"{row.get('event_id')!r}: bad label_status {status!r}")
    if status == LABEL_UNKNOWN:
        if row.get("label_value") is not None or row.get("prefix_security") is not None:
            raise TrainingDataError(f"{row.get('event_id')!r}: unknown label is not null")
    return str(status)


def build_targets(
    rows: Iterable[Mapping[str, Any]],
    *,
    arm: str,
    policy: str,
    finals: Mapping[str, bool] | None = None,
    split_name: str = "train",
) -> dict[str, Any]:
    """Build exact-label supervision for one arm/policy and the honest audit counts.

    The returned ``rows`` contain only exact labels and are the rows that may enter loss.
    Unknown rows are counted in ``counts`` but never returned as trainable rows.
    """
    if policy not in POLICIES:
        raise TrainingDataError(f"Unknown target policy: {policy!r}")
    source_rows = select_inference_rows(rows, arm)
    assert_canonical_ids(source_rows)
    split_counts: Counter = Counter()
    for row in source_rows:
        split = _row_split(row)
        split_counts[split or "missing"] += 1
        if split != split_name:
            raise TrainingDataError(
                f"{row.get('event_id')!r}: supervised row split {split!r} is not {split_name!r}"
            )
    finals = episode_final_map(source_rows) if finals is None else {str(key): bool(value) for key, value in finals.items()}
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_episode[str(row["episode_id"])].append(row)

    trainable: list[dict[str, Any]] = []
    counts: Counter = Counter()
    counts["arm_rows"] = len(source_rows)
    counts["episodes_with_arm_rows"] = len(by_episode)
    counts["unknown_rows_excluded"] = 0
    counts["exact_rows"] = 0
    counts["positive_rows"] = 0
    counts["negative_rows"] = 0
    counts["positive_episodes"] = 0
    counts["positive_episodes_without_trainable_positive"] = 0
    counts["episodes_with_unknown_flip"] = 0
    counts["repeated_flip_episodes"] = 0
    counts["later_exact_flip_rows_negative"] = 0
    counts["transient_positive_rows_total"] = 0
    counts["transient_positive_rows_excluded"] = 0
    counts["transient_positive_rows_kept_negative"] = 0
    counts["row_splits"] = dict(sorted(split_counts.items()))

    for episode_id in sorted(by_episode):
        final_positive = bool(finals.get(episode_id, False))
        ordered = sorted(by_episode[episode_id], key=event_order_key)
        exact_rows: list[dict[str, Any]] = []
        exact_flips: list[dict[str, Any]] = []
        unknown_flips = 0
        transient_positive = 0
        for row in ordered:
            status = _status(row)
            if status == LABEL_UNKNOWN:
                counts["unknown_rows_excluded"] += 1
                if row.get("security_flip") is True:
                    unknown_flips += 1
                continue
            counts["exact_rows"] += 1
            exact_rows.append(row)
            if row.get("security_flip") is True:
                exact_flips.append(row)
            if not final_positive and row.get("prefix_security") is True:
                transient_positive += 1
        if unknown_flips:
            counts["episodes_with_unknown_flip"] += 1
        if len(exact_flips) > 1:
            counts["repeated_flip_episodes"] += 1

        positive_label_count = 0
        if policy == "first_hit":
            first_flip_id = str(exact_flips[0]["event_id"]) if (final_positive and exact_flips) else None
            if first_flip_id is not None:
                counts["later_exact_flip_rows_negative"] += len(exact_flips) - 1
            for row in exact_rows:
                label = first_flip_id is not None and str(row["event_id"]) == first_flip_id
                if label:
                    positive_label_count += 1
                entry = _training_row(row, episode_id, label)
                trainable.append(entry)
                counts["positive_rows" if label else "negative_rows"] += 1
        else:
            for row in exact_rows:
                is_transient_positive = (not final_positive) and row.get("prefix_security") is True
                if is_transient_positive:
                    # Do not relabel a transient positive as a negative: exclude it from
                    # this target's supervised loss.  Inference still scores the position.
                    continue
                label = final_positive and row.get("prefix_security") is True
                if label:
                    positive_label_count += 1
                entry = _training_row(row, episode_id, label)
                trainable.append(entry)
                counts["positive_rows" if label else "negative_rows"] += 1
        counts["transient_positive_rows_total"] += transient_positive
        if policy == "first_hit":
            counts["transient_positive_rows_kept_negative"] += transient_positive
        else:
            counts["transient_positive_rows_excluded"] += transient_positive
        if positive_label_count:
            counts["positive_episodes"] += 1
        elif final_positive:
            counts["positive_episodes_without_trainable_positive"] += 1

    trainable.sort(key=lambda row: (row["episode_id"], row["order_key"], row["event_id"]))
    counts["trainable_rows"] = len(trainable)
    counts["unknown_never_zero"] = True
    return {
        "arm": arm,
        "policy": policy,
        "rows": trainable,
        "counts": dict(sorted(counts.items())),
        "rules": {
            "first_hit": (
                "positive only for the first exact security_flip of a final-positive episode; "
                "later exact rows (including later flips and positive states) are 0; every "
                "exact row of a final-negative episode is 0"
            ),
            "state": (
                "exact prefix_security of a final-positive episode is 1; exact prefix-negative "
                "rows of a final-negative episode are 0; transient prefix-positive rows of a "
                "final-negative episode are excluded from loss and are not relabelled"
            ),
            "unknown": "label_status=unknown is excluded from loss, never converted to 0",
        },
        "final_labels": {episode_id: bool(value) for episode_id, value in sorted(finals.items())},
    }


def _training_row(row: Mapping[str, Any], episode_id: str, label: bool) -> dict[str, Any]:
    return {
        "event_id": str(row["event_id"]),
        "episode_id": episode_id,
        "namespaced_episode_id": str(row["namespaced_episode_id"]),
        "scope": normalize_scope(row),
        "source_tag": row.get("source_tag"),
        "split": _row_split(row),
        "group_id": str(row.get("group_id", "")),
        "event_type": str(row["event_type"]),
        "feature_row": int(row["feature_row"]),
        "label": bool(label),
        "order_key": list(event_order_key(row)),
        "round_id": int(row.get("round_id", 0)),
        "message_index": int(row.get("message_index", 0)),
        "index": int(row.get("index", 0)),
        "feature_token_index": row.get("feature_token_index"),
        "security_flip": bool(row.get("security_flip")) if row.get("security_flip") is not None else None,
        "prefix_security": bool(row.get("prefix_security")) if row.get("prefix_security") is not None else None,
    }


def build_training_arrays(
    targets: Mapping[str, Any], feature_reader: Callable[[Sequence[int]], Any]
) -> dict[str, Any]:
    """Gather features for exactly the exact-label rows, in deterministic row order."""
    rows = list(targets["rows"])
    if not rows:
        raise TrainingDataError(f"{targets['arm']}/{targets['policy']}: no trainable rows")
    assert_canonical_ids(rows)
    for row in rows:
        if row.get("split") != "train":
            raise TrainingDataError(
                f"{row.get('event_id')!r}: target row split {row.get('split')!r} is not 'train'"
            )
    feature_rows = [int(row["feature_row"]) for row in rows]
    if len(set(feature_rows)) != len(feature_rows):
        raise TrainingDataError("Duplicate feature_row in training rows")
    import torch

    features = feature_reader(feature_rows)
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 2 or features.shape[0] != len(rows):
        raise TrainingDataError("Feature reader returned the wrong shape")
    labels = torch.tensor([1.0 if row["label"] else 0.0 for row in rows], dtype=torch.float32)
    positives = int(labels.sum().item())
    negatives = int(labels.numel() - positives)
    if positives <= 0:
        raise TrainingDataError(f"{targets['arm']}/{targets['policy']}: no positive train row")
    return {
        "rows": rows,
        "features": features,
        "labels": labels,
        "positives": positives,
        "negatives": negatives,
        "pos_weight": negatives / positives,
        "counts": dict(targets["counts"]),
    }


def episode_scores(
    rows: Iterable[Mapping[str, Any]],
    episode_ids: Sequence[str],
    *,
    arm: str,
    score_by_event: Mapping[str, float],
) -> dict[str, Any]:
    """Aggregate arm scores to episodes; missing positions stay in the denominator."""
    frozen_ids = [str(episode_id) for episode_id in episode_ids]
    if len(set(frozen_ids)) != len(frozen_ids):
        raise TrainingDataError("Duplicate episode id in the frozen episode set")
    frozen = set(frozen_ids)
    selected = select_inference_rows(rows, arm)
    assert_canonical_ids(selected)
    by_episode: dict[str, list[float]] = defaultdict(list)
    event_counts: Counter = Counter()
    for row in selected:
        event_id = str(row["event_id"])
        episode_id = str(row["episode_id"])
        if episode_id not in frozen:
            raise TrainingDataError(f"{event_id}: episode {episode_id} is outside the frozen set")
        if event_id not in score_by_event:
            raise TrainingDataError(f"{event_id}: missing prediction score")
        score = float(score_by_event[event_id])
        if math.isnan(score) or score == float("inf"):
            raise TrainingDataError(f"{event_id}: non-finite score {score!r}")
        by_episode[episode_id].append(score)
        event_counts[str(row["event_type"])] += 1
    scores: dict[str, float | None] = {}
    coverage: dict[str, Any] = {}
    missing: list[str] = []
    for episode_id in frozen_ids:
        values = by_episode.get(episode_id, [])
        if values:
            scores[episode_id] = max(values)
            coverage[episode_id] = {"positions": len(values), "no_positions": False}
        else:
            scores[episode_id] = None
            coverage[episode_id] = {"positions": 0, "no_positions": True}
            missing.append(episode_id)
    extra = sorted(set(by_episode) - frozen)
    if extra:
        raise TrainingDataError(f"Scored episodes outside the frozen set: {extra[:5]}")
    return {
        "scores": scores,
        "coverage": {
            "episodes": len(frozen_ids),
            "episodes_with_positions": len(frozen_ids) - len(missing),
            "episodes_without_positions": len(missing),
            "no_position_episode_ids": missing,
            "position_events": sum(event_counts.values()),
            "positions_by_type": dict(sorted(event_counts.items())),
            "missing_position_behavior": "episode retained with score=-inf and alarm=false",
        },
    }


def position_matrix_rows(
    rows: Iterable[Mapping[str, Any]], episode_ids: Sequence[str], *, arm: str
) -> list[dict[str, Any]]:
    """Every selected inference row, deterministically ordered, with frozen episode scope."""
    frozen = set(str(episode_id) for episode_id in episode_ids)
    selected = select_inference_rows(rows, arm)
    assert_canonical_ids(selected)
    for row in selected:
        if str(row["episode_id"]) not in frozen:
            raise TrainingDataError(f"{row.get('event_id')}: outside the frozen episode set")
    return sorted(
        selected,
        key=lambda row: (str(row["episode_id"]), event_order_key(row), str(row["event_id"])),
    )


def assert_disjoint_splits(split_episode_ids: Mapping[str, Iterable[str]]) -> None:
    """Train/dev/test episode sets must be pairwise disjoint (no silent leakage)."""
    normalised = {name: set(str(value) for value in values) for name, values in split_episode_ids.items()}
    names = sorted(normalised)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = normalised[left] & normalised[right]
            if overlap:
                raise TrainingDataError(
                    f"Split overlap between {left} and {right}: {sorted(overlap)[:5]}"
                )


def _row_scope(row: Mapping[str, Any]) -> str:
    return normalize_scope(row)


def _row_split(row: Mapping[str, Any]) -> str:
    return str(row.get("split") or row.get("mixed_split") or "")


def cohort_episode_ids(rows: Iterable[Mapping[str, Any]], cohort: str) -> set[str]:
    if cohort not in COHORT_SOURCES:
        raise TrainingDataError(f"Unknown cohort: {cohort!r}")
    sources = set(COHORT_SOURCES[cohort])
    return {str(row["episode_id"]) for row in rows if _row_scope(row) in sources}


def cohort_split_episode_ids(
    rows: Iterable[Mapping[str, Any]], cohort: str, split: str
) -> set[str]:
    """Frozen unified-split episodes of one cohort; AgentLAB never enters cohort A train."""
    if cohort not in COHORT_SOURCES:
        raise TrainingDataError(f"Unknown cohort: {cohort!r}")
    if split not in ("train", "dev", "test"):
        raise TrainingDataError(f"Unknown split: {split!r}")
    sources = set(COHORT_SOURCES[cohort])
    selected = {
        str(row["episode_id"])
        for row in rows
        if _row_scope(row) in sources and _row_split(row) == split
    }
    return selected


def assert_disjoint_group_splits(split_group_ids: Mapping[str, Iterable[str]]) -> None:
    """Task-group disjointness alongside episode disjointness.

    Integration calls this on authoritative episode metadata so a group never spans two
    splits even when the local episode ids look different.
    """
    normalised = {name: set(str(value) for value in values) for name, values in split_group_ids.items()}
    names = sorted(normalised)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = normalised[left] & normalised[right]
            if overlap:
                raise TrainingDataError(
                    f"Group overlap between {left} and {right}: {sorted(overlap)[:5]}"
                )


def assert_agentlab_excluded_from_train(rows: Iterable[Mapping[str, Any]], cohort: str) -> None:
    """Call on the actual cohort train rows; cohort A must contain zero AgentLAB rows."""
    if cohort != "A":
        return
    leaked = {
        str(row["episode_id"])
        for row in rows
        if _row_scope(row) == "agentlab" and _row_split(row) == "train"
    }
    if leaked:
        raise TrainingDataError(f"AgentLAB leaked into cohort A train: {sorted(leaked)[:5]}")


def selection_dev_episode_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """The common calibration/selection dev: unified dev, Qwen3-8B sources only."""
    return {
        str(row["episode_id"])
        for row in rows
        if _row_split(row) == "dev" and _row_scope(row) in set(QWEN_SOURCES)
    }


@dataclass
class ScopeFeatureReader:
    """Lazy read-only memmap over one completed lane-F ``matrix.bf16`` aggregate."""

    out_dir: Path
    scope: str
    rows: int
    hidden_size: int
    matrix_path: Path
    manifest: dict[str, Any]

    @classmethod
    def open(cls, root: Path | str, scope: str, *, verify_manifest: bool = True) -> "ScopeFeatureReader":
        base = Path(root) / "artifacts" / "fine_confirm" / "features" / scope
        manifest_path = base / "manifest.json"
        if not manifest_path.is_file():
            raise TrainingDataError(f"{scope}: feature manifest is missing (extraction incomplete)")
        manifest = read_json(manifest_path)
        if verify_manifest and manifest.get("status") != "complete":
            raise TrainingDataError(f"{scope}: feature extraction is not complete ({manifest.get('status')})")
        storage = manifest.get("storage", {}).get("matrix", {})
        rows = int(manifest.get("counts", {}).get("events", storage.get("shape", [0, 0])[0]))
        hidden_size = int(storage.get("shape", [0, HIDDEN_SIZE])[1])
        matrix_path = base / str(storage.get("file", MATRIX_FILE))
        if not matrix_path.is_file():
            raise TrainingDataError(f"{scope}: feature matrix is missing")
        expected = rows * hidden_size * DTYPE_BYTES
        if matrix_path.stat().st_size != expected:
            raise TrainingDataError(
                f"{scope}: matrix size {matrix_path.stat().st_size} != expected {expected} (incomplete)"
            )
        return cls(base, scope, rows, hidden_size, matrix_path, manifest)

    def index_rows(self) -> list[dict[str, Any]]:
        path = self.out_dir / INDEX_FILE
        if not path.is_file():
            raise TrainingDataError(f"{self.scope}: feature index is missing")
        rows = read_jsonl(path)
        if len(rows) != self.rows:
            raise TrainingDataError(f"{self.scope}: index rows {len(rows)} != manifest rows {self.rows}")
        return rows

    def read(self, feature_rows: Sequence[int]):
        import numpy as np
        import torch

        requested = [int(row) for row in feature_rows]
        if not requested:
            return torch.empty((0, self.hidden_size), dtype=torch.float32)
        if min(requested) < 0 or max(requested) >= self.rows:
            raise TrainingDataError("feature_row outside the aggregate")
        matrix = np.memmap(self.matrix_path, dtype=np.dtype("<u2"), mode="r", shape=(self.rows, self.hidden_size))
        array = np.asarray(matrix[requested])
        return torch.from_numpy(array).view(torch.bfloat16).reshape(len(requested), self.hidden_size).to(torch.float32)

    __call__ = read
