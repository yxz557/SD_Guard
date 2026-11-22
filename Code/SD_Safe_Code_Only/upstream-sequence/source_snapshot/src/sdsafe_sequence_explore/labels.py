"""Label semantics for the NEW sequence data version.

The accepted TRAIN L0..L4 assignments are *provenance*, not rebuilt here.  This module
recomputes the semantic L0..L3 rules from the already-accepted exact checker states and
fails closed if they disagree with ``artifacts/fine_label_repair_v1/data/policies``.
L4 is a random-drop control and is loaded from the accepted assignment file.

Semantics (TRAIN CT only; unknown is always ``-1``/masked, never 0):

``L0`` onset/first exact hit: one positive per strict-final-positive episode, at the
       first exact ``security_flip``; every other exact CT row is 0.
``L1`` conservative: keep L0 anchors; exact prefix-false rows 0; every other exact
       prefix-true row is masked, not called negative.
``L2`` local: L1 plus exact prefix-true rows in the same episode/assistant round/message
       as the L0 anchor -> 1.  This is a local prefix-risk label, not a claim that every
       post-hit text is malicious.
``L3`` exact prefix state: the official checker bool for every exact row, including
       transient positives in strict-final-negative episodes.
``L4`` random-drop negative control from the accepted assignment file.

E/``round_end`` rows are context-only by default.  Their exact checker state is exported
as separate prefix-state columns, but no policy target is assigned to them here.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from sdsafe_features.common import read_json, read_jsonl
from sdsafe_fine_confirm.train_data import event_order_key as _canonical_event_order_key

from . import (
    CT_EVENT_TYPES,
    EVENT_CONTENT_SEGMENT,
    EVENT_TOOL_CALL,
    EXPECTED_POLICY_COUNTS,
    EXPECTED_TRAIN_CT_EXACT,
    EXPECTED_TRAIN_CT_ROWS,
    EXPECTED_TRAIN_CT_UNKNOWN,
    EXPECTED_EXACT_PREFIX_POSITIVE_ROWS,
    LABEL_EXACT,
    LABEL_UNKNOWN,
    POLICIES,
    SPLIT_TRAIN,
    LabelContractError,
)

MASK_UNKNOWN = "unknown_label_never_loss"
MASK_PREFIX_NON_ANCHOR = "prefix_positive_non_anchor_censored"
MASK_L4_DROP = "random_drop_matched_l0_negative"
MISSING = -1


def event_order_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Physical trajectory order with a deterministic total-order tie-break."""
    return _canonical_event_order_key(row) + (str(row.get("event_id") or ""),)


def is_exact(row: Mapping[str, Any]) -> bool:
    return str(row.get("label_status")) == LABEL_EXACT


def is_unknown(row: Mapping[str, Any]) -> bool:
    return str(row.get("label_status")) == LABEL_UNKNOWN


def _require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise LabelContractError(f"{where}: expected bool checker state, got {value!r}")
    return value


def checker_state(row: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return ``(prefix_state, previous_state, security_flip)`` as ints; unknown -> -1."""
    event_id = row.get("event_id")
    if is_unknown(row):
        if any(row.get(name) is not None for name in ("prefix_security", "previous_security", "security_flip")):
            raise LabelContractError(f"{event_id}: unknown row carries checker state")
        return MISSING, MISSING, MISSING
    if not is_exact(row):
        raise LabelContractError(f"{event_id}: bad label_status {row.get('label_status')!r}")
    prefix = _require_bool(row.get("prefix_security"), f"{event_id} prefix_security")
    previous = _require_bool(row.get("previous_security"), f"{event_id} previous_security")
    flip = row.get("security_flip")
    if flip is not None:
        flip = _require_bool(flip, f"{event_id} security_flip")
        if flip is not bool(prefix and not previous):
            raise LabelContractError(f"{event_id}: security_flip is incoherent")
    return int(prefix), int(previous), int(prefix and not previous)


def _ct_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if str(row.get("event_type")) in CT_EVENT_TYPES]
    for row in selected:
        if str(row.get("split")) != SPLIT_TRAIN:
            raise LabelContractError(f"{row.get('event_id')}: policy rows must be TRAIN")
    return selected


def _l0_anchors(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if is_exact(row):
            by_episode[str(row["episode_id"])].append(dict(row))
    anchors: dict[str, str] = {}
    for episode_id, episode_rows in sorted(by_episode.items()):
        if not bool(episode_rows[0].get("strict_final")):
            continue
        ordered = sorted(episode_rows, key=event_order_key)
        flips = [row for row in ordered if row.get("security_flip") is True]
        if len(flips) == 0:
            raise LabelContractError(f"{episode_id}: strict-final positive episode has no exact security flip")
        anchors[episode_id] = str(flips[0]["event_id"])
    return anchors


def _turn_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (str(row["episode_id"]), int(row["round_id"]), int(row["message_index"]))


def _entry(label: int | None, reason: str | None) -> dict[str, Any]:
    return {"label": label, "mask": label is None, "mask_reason": reason}


def recompute_policy_targets(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Recompute semantic L0..L3 on TRAIN CT rows; L4 is not semantic and is not returned.

    The result is used only to validate the accepted assignment file and to exercise the
    label semantics in synthetic tests.  Official checker labels themselves are loaded
    from the accepted repaired rows and are never relabelled here.
    """
    ct = _ct_rows(rows)
    if len(ct) != len({str(row["event_id"]) for row in ct}):
        raise LabelContractError("duplicate CT event_id in label population")
    anchors = _l0_anchors(ct)
    anchor_ids = set(anchors.values())
    anchor_turn: dict[tuple[str, int, int], str] = {}
    by_id = {str(row["event_id"]): row for row in ct}
    for anchor_id in anchor_ids:
        anchor = by_id[anchor_id]
        anchor_turn[_turn_key(anchor)] = anchor_id

    l0: dict[str, dict[str, Any]] = {}
    l1: dict[str, dict[str, Any]] = {}
    l2: dict[str, dict[str, Any]] = {}
    l3: dict[str, dict[str, Any]] = {}
    for row in ct:
        event_id = str(row["event_id"])
        if not is_exact(row):
            l0[event_id] = _entry(None, MASK_UNKNOWN)
            l1[event_id] = _entry(None, MASK_UNKNOWN)
            l2[event_id] = _entry(None, MASK_UNKNOWN)
            l3[event_id] = _entry(None, MASK_UNKNOWN)
            continue
        prefix = bool(row.get("prefix_security"))
        l3[event_id] = _entry(int(prefix), None)
        if event_id in anchor_ids:
            l0[event_id] = _entry(1, None)
            l1[event_id] = _entry(1, None)
            l2[event_id] = _entry(1, None)
        else:
            l0[event_id] = _entry(0, None)
            if prefix:
                l1[event_id] = _entry(None, MASK_PREFIX_NON_ANCHOR)
                l2[event_id] = _entry(None, MASK_PREFIX_NON_ANCHOR)
            else:
                l1[event_id] = _entry(0, None)
                l2[event_id] = _entry(0, None)

    for row in ct:
        event_id = str(row["event_id"])
        if not is_exact(row) or event_id in anchor_ids:
            continue
        anchor_id = anchor_turn.get(_turn_key(row))
        if anchor_id is not None and bool(row.get("prefix_security")):
            l2[event_id] = {
                "label": 1,
                "mask": False,
                "mask_reason": None,
                "label_reason": "same_first_hit_turn_prefix_positive",
            }

    targets = {"L0": l0, "L1": l1, "L2": l2, "L3": l3}
    for policy in ("L1", "L2", "L3"):
        bad = [
            str(row["event_id"])
            for row in ct
            if is_exact(row)
            and bool(row.get("prefix_security"))
            and targets[policy][str(row["event_id"])]["label"] == 0
        ]
        if bad:
            raise LabelContractError(f"{policy}: exact prefix-positive negative rows exist: {bad[:5]}")
    summary = {
        "rows": len(ct),
        "exact_rows": sum(1 for row in ct if is_exact(row)),
        "unknown_rows": sum(1 for row in ct if not is_exact(row)),
        "anchors": len(anchors),
        "positive_episodes": len(anchors),
        "l3_positive_episodes": len(
            {str(row["episode_id"]) for row in ct if is_exact(row) and bool(row.get("prefix_security"))}
        ),
    }
    return targets, summary


def load_policy_assignments(root: Path | str = ".") -> dict[str, dict[str, dict[str, Any]]]:
    path = Path(root) / "artifacts/fine_label_repair_v1/data/policies/assignments.jsonl"
    if not path.is_file():
        raise LabelContractError(f"accepted policy assignments are missing: {path}")
    assignments: dict[str, dict[str, dict[str, Any]]] = {}
    for row in read_jsonl(path):
        event_id = str(row.get("event_id") or "")
        if not event_id:
            raise LabelContractError(f"{path}: assignment row has no event_id")
        if event_id in assignments:
            raise LabelContractError(f"{path}: duplicate assignment {event_id}")
        policies = row.get("policies")
        if not isinstance(policies, Mapping) or set(policies) != set(POLICIES):
            raise LabelContractError(f"{event_id}: assignment does not cover exactly {POLICIES}")
        if str(row.get("split")) != SPLIT_TRAIN:
            raise LabelContractError(f"{event_id}: accepted assignment is not TRAIN")
        assignments[event_id] = {
            policy: {
                "label": policies[policy].get("label"),
                "mask": bool(policies[policy].get("mask")),
                "mask_reason": policies[policy].get("mask_reason"),
            }
            for policy in POLICIES
        }
    if len(assignments) != EXPECTED_TRAIN_CT_ROWS:
        raise LabelContractError(f"accepted assignment rows {len(assignments)} != {EXPECTED_TRAIN_CT_ROWS}")
    counts = assignment_counts(assignments)
    for policy, expected in EXPECTED_POLICY_COUNTS.items():
        actual = counts[policy]
        for key in ("positive", "negative", "masked_exact", "unknown"):
            if int(actual[key]) != int(expected[key]):
                raise LabelContractError(f"{policy} {key} {actual[key]} != accepted {expected[key]}")
    return assignments


def assignment_counts(assignments: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for policy in POLICIES:
        positive = negative = masked_exact = unknown = 0
        reasons: Counter = Counter()
        for entry in assignments.values():
            item = entry[policy]
            label = item.get("label")
            reason = item.get("mask_reason")
            if reason == MASK_UNKNOWN:
                unknown += 1
                continue
            if label is True:
                positive += 1
            elif label is False:
                negative += 1
            else:
                masked_exact += 1
                reasons[str(reason)] += 1
        counts[policy] = {
            "positive": positive,
            "negative": negative,
            "masked_exact": masked_exact,
            "unknown": unknown,
            "masked_exact_by_reason": dict(sorted(reasons.items())),
        }
    return counts


def _validate_assignments_against_semantics(
    ct: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    semantic, summary = recompute_policy_targets(ct)
    mismatches: list[dict[str, Any]] = []
    for policy in ("L0", "L1", "L2", "L3"):
        for row in ct:
            event_id = str(row["event_id"])
            accepted = assignments[event_id][policy]
            expected = semantic[policy][event_id]
            accepted_label = accepted.get("label")
            expected_label = expected.get("label")
            # L2 accepted rows may carry an extra label_reason; only target/mask matters.
            if accepted_label != expected_label or bool(accepted.get("mask")) != bool(expected.get("mask")):
                mismatches.append(
                    {
                        "policy": policy,
                        "event_id": event_id,
                        "accepted": {"label": accepted_label, "mask": accepted.get("mask")},
                        "recomputed": {"label": expected_label, "mask": expected.get("mask")},
                    }
                )
                if len(mismatches) >= 10:
                    break
        if len(mismatches) >= 10:
            break
    if mismatches:
        raise LabelContractError(f"accepted L0..L3 assignments disagree with semantic recomputation: {mismatches[:3]}")
    return {
        "checked": True,
        "policies": ["L0", "L1", "L2", "L3"],
        "rows": len(ct),
        "anchors": summary["anchors"],
        "positive_episodes": summary["positive_episodes"],
        "l3_positive_episodes": summary["l3_positive_episodes"],
    }


def _array_from_entries(
    entries: Mapping[str, Mapping[str, Any]],
    event_ids: Sequence[str],
    *,
    policy: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(len(event_ids), MISSING, dtype=np.int8)
    masks = np.ones(len(event_ids), dtype=np.bool_)
    for index, event_id in enumerate(event_ids):
        item = entries[event_id]
        label = item.get("label")
        mask = bool(item.get("mask"))
        if label is None:
            if not mask:
                raise LabelContractError(f"{policy}/{event_id}: null label is not masked")
            labels[index] = MISSING
            masks[index] = True
        else:
            if mask:
                raise LabelContractError(f"{policy}/{event_id}: labelled row is masked")
            if label not in (True, False):
                raise LabelContractError(f"{policy}/{event_id}: label must be bool, got {label!r}")
            labels[index] = int(label)
            masks[index] = False
    return labels, masks


def build_train_label_arrays(
    ct_rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    enforce_expected_counts: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build per-position TRAIN CT arrays in the supplied row order.

    ``ct_rows`` must be the exact TRAIN CT population in sequence order.  The accepted
    assignments are validated against semantic L0..L3 and exact accepted counts; unknown
    is stored as ``-1`` with ``mask=True`` and is never coerced to 0.
    """
    ct = [dict(row) for row in ct_rows]
    event_ids = [str(row["event_id"]) for row in ct]
    if len(event_ids) != len(set(event_ids)):
        raise LabelContractError("duplicate event_id in TRAIN CT label arrays")
    if set(event_ids) != set(assignments):
        missing = len(set(event_ids) - set(assignments))
        extra = len(set(assignments) - set(event_ids))
        raise LabelContractError(f"assignment population mismatch: missing={missing} extra={extra}")
    semantic_audit = _validate_assignments_against_semantics(ct, assignments)

    arrays: dict[str, np.ndarray] = {}
    for policy in POLICIES:
        labels, masks = _array_from_entries(
            {event_id: assignments[event_id][policy] for event_id in event_ids},
            event_ids,
            policy=policy,
        )
        arrays[f"label_{policy}"] = labels
        arrays[f"mask_{policy}"] = masks
    arrays["prefix_state"] = np.asarray(
        [checker_state(row)[0] for row in ct], dtype=np.int8
    )
    arrays["previous_state"] = np.asarray(
        [checker_state(row)[1] for row in ct], dtype=np.int8
    )
    arrays["security_flip"] = np.asarray(
        [checker_state(row)[2] for row in ct], dtype=np.int8
    )
    arrays["exact"] = np.asarray([is_exact(row) for row in ct], dtype=np.bool_)

    counts = assignment_counts(assignments)
    summary = {
        "rows": len(ct),
        "exact_rows": int(arrays["exact"].sum()),
        "unknown_rows": int((~arrays["exact"]).sum()),
        "per_policy": counts,
        "semantic_recomputation": semantic_audit,
        "unknown_never_zero": bool(
            all(np.all(arrays[f"label_{policy}"][~arrays["exact"]] == MISSING) for policy in POLICIES)
        ),
        "prefix_state_matches_L3": bool(
            np.array_equal(arrays["prefix_state"], arrays["label_L3"])
        ),
    }
    if enforce_expected_counts and summary["exact_rows"] != EXPECTED_TRAIN_CT_EXACT:
        raise LabelContractError(f"TRAIN CT exact rows {summary['exact_rows']} != {EXPECTED_TRAIN_CT_EXACT}")
    if enforce_expected_counts and summary["unknown_rows"] != EXPECTED_TRAIN_CT_UNKNOWN:
        raise LabelContractError(f"TRAIN CT unknown rows {summary['unknown_rows']} != {EXPECTED_TRAIN_CT_UNKNOWN}")
    if not summary["unknown_never_zero"]:
        raise LabelContractError("an unknown row was assigned a training label")
    if not summary["prefix_state_matches_L3"]:
        raise LabelContractError("prefix_state does not equal L3 exact checker state")
    exact_prefix_positive = int(np.sum(arrays["prefix_state"] == 1))
    if enforce_expected_counts and exact_prefix_positive != EXPECTED_EXACT_PREFIX_POSITIVE_ROWS:
        raise LabelContractError(
            f"exact prefix-positive rows {exact_prefix_positive} != {EXPECTED_EXACT_PREFIX_POSITIVE_ROWS}"
        )
    summary["exact_prefix_positive_rows"] = exact_prefix_positive
    return arrays, summary


def build_dev_label_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """DEV truth arrays for evaluation/diagnostics; no training policy is assigned."""
    rows = [dict(row) for row in rows]
    event_ids = [str(row["event_id"]) for row in rows]
    if len(event_ids) != len(set(event_ids)):
        raise LabelContractError("duplicate event_id in DEV truth arrays")
    prefix = []
    previous = []
    flip = []
    exact = []
    for row in rows:
        p, q, f = checker_state(row)
        prefix.append(p)
        previous.append(q)
        flip.append(f)
        exact.append(is_exact(row))
    arrays = {
        "prefix_state": np.asarray(prefix, dtype=np.int8),
        "previous_state": np.asarray(previous, dtype=np.int8),
        "security_flip": np.asarray(flip, dtype=np.int8),
        "exact": np.asarray(exact, dtype=np.bool_),
    }
    summary = {
        "rows": len(rows),
        "exact_rows": int(arrays["exact"].sum()),
        "unknown_rows": int((~arrays["exact"]).sum()),
        "exact_prefix_positive_rows": int(np.sum(arrays["prefix_state"] == 1)),
        "note": "DEV truth is evaluation/diagnostic only; L0..L4 are TRAIN supervision policies",
    }
    return arrays, summary


@dataclass(frozen=True)
class EpisodeLabelSupport:
    n_positions: int
    n_ct: int
    n_e: int
    n_exact: int
    n_unknown: int
    n_prefix_true: int
    n_prefix_true_ct: int
    n_prefix_true_e: int
    n_security_flip: int
    transient_prefix_true: bool
    policy_positive: dict[str, int]
    policy_negative: dict[str, int]
    policy_masked: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_positions": self.n_positions,
            "n_ct": self.n_ct,
            "n_e": self.n_e,
            "n_exact": self.n_exact,
            "n_unknown": self.n_unknown,
            "n_prefix_true": self.n_prefix_true,
            "n_prefix_true_ct": self.n_prefix_true_ct,
            "n_prefix_true_e": self.n_prefix_true_e,
            "n_security_flip": self.n_security_flip,
            "transient_prefix_true": self.transient_prefix_true,
            "policy_positive": dict(self.policy_positive),
            "policy_negative": dict(self.policy_negative),
            "policy_masked": dict(self.policy_masked),
        }


def episode_label_support(
    *,
    n_positions: int,
    episode_ct_mask: np.ndarray,
    episode_prefix_state: np.ndarray,
    episode_security_flip: np.ndarray,
    episode_bag_label: bool,
    policy_labels: Mapping[str, np.ndarray] | None = None,
    policy_masks: Mapping[str, np.ndarray] | None = None,
) -> EpisodeLabelSupport:
    """Summarise support/conflict/unknown counts for one episode."""
    ct_mask = np.asarray(episode_ct_mask, dtype=np.bool_)
    prefix = np.asarray(episode_prefix_state, dtype=np.int8)
    flips = np.asarray(episode_security_flip, dtype=np.int8)
    prefix_true = prefix == 1
    policy_positive: dict[str, int] = {}
    policy_negative: dict[str, int] = {}
    policy_masked: dict[str, int] = {}
    for policy in POLICIES:
        if policy_labels is None or policy_masks is None or policy not in policy_labels:
            continue
        labels = np.asarray(policy_labels[policy])
        masks = np.asarray(policy_masks[policy], dtype=np.bool_)
        policy_positive[policy] = int(np.sum((labels == 1) & ~masks))
        policy_negative[policy] = int(np.sum((labels == 0) & ~masks))
        policy_masked[policy] = int(np.sum(masks))
    return EpisodeLabelSupport(
        n_positions=int(n_positions),
        n_ct=int(ct_mask.sum()),
        n_e=int((~ct_mask).sum()),
        n_exact=int(np.sum(prefix != MISSING)),
        n_unknown=int(np.sum(prefix == MISSING)),
        n_prefix_true=int(prefix_true.sum()),
        n_prefix_true_ct=int((prefix_true & ct_mask).sum()),
        n_prefix_true_e=int((prefix_true & ~ct_mask).sum()),
        n_security_flip=int(np.sum(flips == 1)),
        transient_prefix_true=bool((not episode_bag_label) and bool(prefix_true.any())),
        policy_positive=policy_positive,
        policy_negative=policy_negative,
        policy_masked=policy_masked,
    )


def summarise_policy_support(
    labels: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
    exact: np.ndarray,
) -> dict[str, Any]:
    """Dataset-level support summary with unknown counted separately."""
    exact = np.asarray(exact, dtype=np.bool_)
    out: dict[str, Any] = {}
    for policy in POLICIES:
        if f"label_{policy}" not in labels:
            continue
        label = np.asarray(labels[f"label_{policy}"])
        mask = np.asarray(masks[f"mask_{policy}"], dtype=np.bool_)
        out[policy] = {
            "positive": int(np.sum((label == 1) & ~mask)),
            "negative": int(np.sum((label == 0) & ~mask)),
            "masked": int(np.sum(mask)),
            "masked_exact": int(np.sum(mask & exact)),
            "unknown": int(np.sum(mask & ~exact)),
            "trainable_exact": int(np.sum(~mask)),
        }
    return out


__all__ = [
    "MASK_UNKNOWN",
    "MASK_PREFIX_NON_ANCHOR",
    "MASK_L4_DROP",
    "MISSING",
    "EpisodeLabelSupport",
    "assignment_counts",
    "build_dev_label_arrays",
    "build_train_label_arrays",
    "checker_state",
    "episode_label_support",
    "event_order_key",
    "is_exact",
    "is_unknown",
    "load_policy_assignments",
    "recompute_policy_targets",
    "summarise_policy_support",
]
