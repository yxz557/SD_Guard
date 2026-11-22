"""Evaluate corrected-r3 FUSION_S2 plus general causal-flow evidence.

Feature weights are selected only from TRAIN OOF scores using group-wise folds. The common DEV
set is used to calibrate the final 5% FPR threshold and report development performance. No TEST
artifact is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdsafe_v6 import metrics, rule_taint  # noqa: E402


DEFAULT_SOURCE = Path("/root/sdsafe-compute/v6-param/artifacts")
BUDGET = 0.05


def load_views(path: Path) -> dict[str, list[dict[str, Any]]]:
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            episodes[str(rec["episode_id"])].append(rec)
    for records in episodes.values():
        records.sort(key=lambda rec: int(rec["row_index"]))
    return dict(episodes)


def score_rule(episodes: dict[str, list[dict[str, Any]]], ids: list[str]) -> dict[str, Any]:
    features = np.zeros((len(ids), 2), dtype=np.float64)
    labels = np.zeros(len(ids), dtype=np.int64)
    groups: list[str] = []
    evidence = []
    for index, episode in enumerate(ids):
        records = episodes[episode]
        result = rule_taint.score_episode_features(records)
        features[index] = [result["text_object_flow"], result["tool_identifier_flow"]]
        labels[index] = int(bool(records[0]["episode_label"]))
        groups.append(str(records[0]["group_id"]))
        for ev in result["evidence"]:
            item = ev.to_dict()
            item.update({"episode_label": int(labels[index]), "group_id": groups[-1],
                         "scope": str(records[0]["scope"])})
            evidence.append(item)
    return {"features": features, "labels": labels,
            "groups": np.asarray(groups, dtype=object), "evidence": evidence}


def episode_max(row_scores: np.ndarray, row_episode: np.ndarray,
                episode_ids: list[str]) -> np.ndarray:
    index = {episode: position for position, episode in enumerate(episode_ids)}
    out = np.full(len(episode_ids), -np.inf, dtype=np.float64)
    rows = np.asarray([index[str(episode)] for episode in row_episode], dtype=np.int64)
    np.maximum.at(out, rows, np.asarray(row_scores, dtype=np.float64))
    return out


def align(mapping: dict[str, float], ids: list[str]) -> np.ndarray:
    missing = [episode for episode in ids if episode not in mapping]
    if missing:
        raise RuntimeError(f"missing {len(missing)} scores; first={missing[0]}")
    return np.asarray([mapping[episode] for episode in ids], dtype=np.float64)


def train_fusion(source: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    base = np.load(source / "train_base_episodes.npz", allow_pickle=True)
    base_ids = [str(value) for value in base["episode_id"]]
    base_of = {episode: float(value) for episode, value in zip(base_ids, base["base"])}
    labels_of = {episode: int(value) for episode, value in zip(base_ids, base["labels"])}
    corrected = source / "corrected-r3"
    oof = np.load(corrected / "train_oof_S2.npz", allow_pickle=True)
    oof_ids = [str(value) for value in oof["episode_id"]]
    student_ids = sorted(set(oof_ids))
    student = episode_max(oof["row_scores"], np.asarray(oof_ids, dtype=object), student_ids)
    student_of = {episode: float(value) for episode, value in zip(student_ids, student)}
    frozen = json.loads((corrected / "fusion_frozen_corrected_r3_S2.json").read_text())
    fused = (
        0.5 * ((align(base_of, ids) - float(frozen["base_mean_TRAIN"]))
               / float(frozen["base_std_TRAIN"]))
        + 0.5 * ((align(student_of, ids) - float(frozen["student_mean_TRAIN_OOF"]))
                 / float(frozen["student_std_TRAIN_OOF"]))
    )
    return fused, np.asarray([labels_of[episode] for episode in ids], dtype=np.int64)


def group_folds(groups: np.ndarray, count: int = 5) -> np.ndarray:
    return np.asarray([
        int.from_bytes(hashlib.sha256(str(group).encode()).digest()[:4], "big") % count
        for group in groups
    ], dtype=np.int64)


def select_weights(base: np.ndarray, rule: np.ndarray, labels: np.ndarray,
                   groups: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(base)
    folds = group_folds(groups[finite])
    best = None
    for text_weight in np.linspace(0.0, 3.0, 31):
        for tool_weight in np.linspace(0.0, 2.0, 21):
            score = (base[finite] + float(text_weight) * rule[finite, 0]
                     + float(tool_weight) * rule[finite, 1])
            fold_auc = [metrics.auc(score[folds == fold], labels[finite][folds == fold])
                        for fold in range(5)]
            mean_auc = float(np.mean(fold_auc))
            key = (-mean_auc, text_weight + tool_weight, text_weight, tool_weight)
            record = {"text_weight": float(text_weight), "tool_weight": float(tool_weight),
                      "group_cv_mean_auc": mean_auc, "group_cv_fold_auc": fold_auc,
                      "finite_train_episodes": int(finite.sum())}
            if best is None or key < best[0]:
                best = (key, record)
    assert best is not None
    return best[1]


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    return {**metrics.confusion(scores, labels, threshold),
            "auc": metrics.auc(scores, labels), "threshold": float(threshold),
            "threshold_hex": float(threshold).hex()}


def feature_slices(features: np.ndarray, labels: np.ndarray,
                   groups: np.ndarray) -> dict[str, Any]:
    table = {}
    suites = sorted({str(group).split("|")[0] for group in groups})
    names = {"text_object_flow": features[:, 0] > 0,
             "tool_identifier_flow": features[:, 1] > 0,
             "either": np.max(features, axis=1) > 0}
    for feature_name, flagged in names.items():
        table[feature_name] = {}
        for suite in ["all", *suites]:
            mask = np.ones(len(labels), dtype=bool) if suite == "all" else np.asarray(
                [str(group).split("|")[0] == suite for group in groups], dtype=bool)
            count = int((mask & flagged).sum())
            positive = int((mask & flagged & (labels == 1)).sum())
            negative = int((mask & flagged & (labels == 0)).sum())
            table[feature_name][suite] = {
                "flagged": count, "positive": positive, "negative": negative,
                "precision": None if count == 0 else positive / count,
            }
    return table


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/rule-v2")
    args = ap.parse_args(argv)
    corrected = args.source / "corrected-r3"
    args.out.mkdir(parents=True, exist_ok=True)

    train_episodes = load_views(corrected / "views_train.jsonl")
    dev_episodes = load_views(corrected / "views_dev.jsonl")
    train_ids = sorted(train_episodes)
    dev_ids = sorted(dev_episodes)
    train_rule = score_rule(train_episodes, train_ids)
    dev_rule = score_rule(dev_episodes, dev_ids)
    train_base, train_y = train_fusion(args.source, train_ids)
    if not np.array_equal(train_y, train_rule["labels"]):
        raise RuntimeError("TRAIN label mismatch")

    replay = np.load(corrected / "replay_S2.npz", allow_pickle=True)
    if [str(value) for value in replay["episode_id"]] != dev_ids:
        raise RuntimeError("DEV episode order mismatch")
    dev_base = np.asarray(replay["fused"], dtype=np.float64)
    dev_y = np.asarray(replay["labels"], dtype=np.int64)
    if not np.array_equal(dev_y, dev_rule["labels"]):
        raise RuntimeError("DEV label mismatch")
    base_tau = float(replay["threshold"])

    selected = select_weights(
        train_base, train_rule["features"], train_y, train_rule["groups"])
    text_weight = float(selected["text_weight"])
    tool_weight = float(selected["tool_weight"])
    dev_hybrid = (dev_base + text_weight * dev_rule["features"][:, 0]
                  + tool_weight * dev_rule["features"][:, 1])
    dev_tau = metrics.threshold_at_fpr(dev_hybrid, dev_y, BUDGET)
    base_pred = dev_base >= base_tau
    hybrid_pred = dev_hybrid >= dev_tau

    report = {
        "status": "development result; weights selected on TRAIN OOF groups; DEV threshold calibrated; no TEST opened",
        "rule": {
            "name": "causal directive-to-sink flow v2",
            "text_feature": "imperative object copied into later assistant text",
            "tool_feature": "typed identifier copied into action/target-role aligned tool arguments",
            "fixed_entity_word_count": False,
            "uses_title_case": False,
            "uses_suite_or_task_ids": False,
            "uses_known_payload_values": False,
            "event_stateful": True,
        },
        "universe": {"train_episodes": len(train_ids), "dev_episodes": len(dev_ids),
                     "dev_positives": int(dev_y.sum()),
                     "dev_negatives": int((dev_y == 0).sum())},
        "weight_selection_train_only": selected,
        "raw_feature_audit": {
            "train": feature_slices(train_rule["features"], train_y, train_rule["groups"]),
            "dev": feature_slices(dev_rule["features"], dev_y, dev_rule["groups"]),
        },
        "dev": {
            "base_frozen": evaluate(dev_base, dev_y, base_tau),
            "causal_fusion_at_5pct": evaluate(dev_hybrid, dev_y, dev_tau),
            "recovered_base_fn_ids": [
                dev_ids[index] for index in np.flatnonzero((dev_y == 1) & ~base_pred & hybrid_pred)
            ],
            "new_fn_ids": [
                dev_ids[index] for index in np.flatnonzero((dev_y == 1) & base_pred & ~hybrid_pred)
            ],
            "remaining_fn_ids": [
                dev_ids[index] for index in np.flatnonzero((dev_y == 1) & ~hybrid_pred)
            ],
        },
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    for split, items in (("train", train_rule["evidence"]), ("dev", dev_rule["evidence"])):
        with (args.out / f"evidence_{split}.jsonl").open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    np.savez(
        args.out / "scores_dev.npz", episode_id=np.asarray(dev_ids, dtype=object),
        labels=dev_y, base=dev_base, rule_features=dev_rule["features"],
        hybrid=dev_hybrid, base_threshold=np.float64(base_tau),
        hybrid_threshold=np.float64(dev_tau), text_weight=np.float64(text_weight),
        tool_weight=np.float64(tool_weight),
    )
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
