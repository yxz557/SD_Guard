"""Freeze the deployable equal-logit fusion mapping using TRAIN material only.

The earlier PROTOCOL §6 rank fusion ranked the whole DEV batch, which uses other (future) DEV
episodes' score distribution.  That is not a per-episode online score.  This script replaces it
with a fixed, label-free TRAIN mapping:

    z_base    = (base_ensemble - base_mean_TRAIN) / base_std_TRAIN
    z_student = (student_OOF  - stud_mean_TRAIN) / stud_std_TRAIN
    fused     = 0.5 * z_base + 0.5 * z_student

No weight is fitted on any label.  The student side uses outer-group-heldout OOF scores from
folds 1-3; fold 0 is excluded because the shared epoch count was selected on fold 0.  The base
probes are in-sample on TRAIN (AUC ~0.9998), so their TRAIN mean/std is used only as a fixed
scale; this caveat is recorded in the frozen manifest and in the report.  DEV is not read here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ART = Path("/root/sdsafe-compute/v6-param/artifacts")
POOL = Path("/dev/shm/sdsafe-gpu-v4/upstream/sdsafe-lowcost-authorization-v2/artifacts/tensors/train_pool_arrays.npz")
CACHE = ART / "cache_train_v3.npz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_max(row_scores: np.ndarray, row_episode: np.ndarray, episodes: list[str]) -> np.ndarray:
    index = {e: i for i, e in enumerate(episodes)}
    out = np.full(len(episodes), -np.inf, dtype=np.float32)
    idx = np.asarray([index[str(e)] for e in row_episode], dtype=np.int64)
    np.maximum.at(out, idx, np.asarray(row_scores, dtype=np.float32))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="arm selected on TRAIN fold-0 only")
    ap.add_argument("--out", default=str(ART / "fusion_frozen_v3.json"))
    args = ap.parse_args(argv)

    cache = np.load(CACHE, allow_pickle=True)
    row_index = np.asarray(cache["row_index"], dtype=np.int64)
    episode_id = np.asarray([str(x) for x in cache["episode_id"].tolist()])
    episode_label = np.asarray(cache["episode_label"], dtype=np.int64)
    group_id = np.asarray([str(x) for x in cache["group_id"].tolist()])
    episodes = sorted(set(episode_id.tolist()))
    ep_pos = {e: i for i, e in enumerate(episodes)}
    # one label per episode (consistency check)
    labels = np.zeros(len(episodes), dtype=np.int64)
    seen: dict[str, int] = {}
    for e, y in zip(episode_id, episode_label):
        if e in seen and seen[e] != int(y):
            raise RuntimeError(f"episode {e} has inconsistent labels")
        seen[e] = int(y)
        labels[ep_pos[e]] = int(y)

    pool = np.load(POOL, allow_pickle=True)
    pool_rows = np.asarray(pool["row_index"], dtype=np.int64)
    base_logits = np.asarray(pool["base_logits"], dtype=np.float32)
    row_to_ep = {int(r): e for r, e in zip(row_index, episode_id)}
    keep = np.asarray([int(r) in row_to_ep for r in pool_rows], dtype=bool)
    pool_rows = pool_rows[keep]
    base_logits = base_logits[:, keep]
    pool_ep = np.asarray([row_to_ep[int(r)] for r in pool_rows], dtype=object)
    per_seed = np.stack([episode_max(base_logits[s], pool_ep, episodes) for s in range(3)], axis=0)
    base_train = per_seed.mean(axis=0, dtype=np.float32)

    oof = np.load(ART / f"train_oof_{args.primary}_v3.npz", allow_pickle=True)
    oof_scores = np.asarray(oof["row_scores"], dtype=np.float32)
    oof_ep = np.asarray([str(x) for x in oof["episode_id"].tolist()])
    oof_fold = np.asarray(oof["fold"], dtype=np.int64)
    use = np.isfinite(oof_scores) & (oof_fold != 0)
    if not use.any():
        raise RuntimeError("no held-out OOF rows outside fold 0")
    stud_episodes = sorted(set(oof_ep[use].tolist()))
    stud_oof = episode_max(oof_scores[use], oof_ep[use], stud_episodes)
    stud_labels = np.asarray([int(labels[ep_pos[e]]) for e in stud_episodes], dtype=np.int64)

    base_finite = np.isfinite(base_train)
    stud_finite = np.isfinite(stud_oof)
    base_mean = float(base_train[base_finite].mean(dtype=np.float64))
    base_std = float(base_train[base_finite].std(dtype=np.float64) + 1e-9)
    stud_mean = float(stud_oof[stud_finite].mean(dtype=np.float64))
    stud_std = float(stud_oof[stud_finite].std(dtype=np.float64) + 1e-9)
    payload: dict[str, Any] = {
        "formula": "0.5*((base-base_mean)/base_std) + 0.5*((student-stud_mean)/stud_std)",
        "weights": [0.5, 0.5],
        "base_mean_TRAIN": base_mean, "base_std_TRAIN": base_std,
        "student_mean_TRAIN_OOF": stud_mean, "student_std_TRAIN_OOF": stud_std,
        "primary_arm": args.primary,
        "student_oof_folds_used": sorted(set((oof_fold[use]).tolist())),
        "student_oof_fold0_excluded_reason": "epoch count selected on fold 0; fold 0 scores are in-sample for selection",
        "base_TRAIN_in_sample_caveat": "frozen base probes score ~0.9998 AUC on TRAIN; their mean/std is a fixed label-free scale, not a fitted weight",
        "train_episodes": len(episodes),
        "train_base_episodes_finite": int(base_finite.sum()),
        "train_base_episodes_missing": int((~base_finite).sum()),
        "train_base_rows": int(pool_rows.size),
        "train_oof_rows_used": int(use.sum()),
        "train_oof_episodes": len(stud_episodes),
        "train_base_auc_diagnostic_only": float(_auc(base_train, labels)),
        "train_student_oof_auc_diagnostic_only": float(_auc(stud_oof, stud_labels)),
        "created_utc": "2026-11-14T17:50:00Z",
        "dev_read": False,
        "inputs_sha256": {str(POOL): sha(POOL), str(CACHE): sha(CACHE),
                          str(ART / f"train_oof_{args.primary}_v3.npz"): sha(ART / f"train_oof_{args.primary}_v3.npz")},
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    payload["frozen_sha256"] = sha(out)
    print(json.dumps(payload, indent=1))
    return 0


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    s = scores[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    raise SystemExit(main())
