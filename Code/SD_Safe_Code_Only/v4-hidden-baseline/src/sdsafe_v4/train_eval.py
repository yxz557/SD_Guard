"""Build the row-level dataset from captured episode shards and train the small heads.

Arms (pre-registered, matched width/depth/epochs/recipe; only the input block differs):

``C``  terminal-only control  : accepted final-layer hidden (4096)
``E``  early only             : block4 || block8 (8192)
``F``  fused                  : final || block4 || block8 (12288)

Every arm is a ``Linear(d, 256) -> GELU -> Dropout(0.1) -> Linear(256, 1)`` row
classifier trained with the accepted balanced recipe, then aggregated per episode with
``max`` over CT rows and averaged over 3 seeds -- exactly the accepted operating point.

Two fits per arm/seed:
``ho``  fit groups = folds 1..3 (fold 0's 154 groups never seen) for TRAIN-only selection
``full`` fit all TRAIN groups, used for the exploratory DEV evaluation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import (
    ACCEPTED_FEATURE_BIN, ART, CAPTURE_BLOCKS, FOLD_SEED, FOLDS, HIDDEN, TRAIN_SEEDS,
    V2_BASELINE, V2_DEV_ARRAYS, V2_TRAIN_ARRAYS,
)
from .common import StudyError, atomic_write_json, json_load, read_jsonl
from .seqsrc import RowIndex, dev_common_universe

DS = ART / "dataset"
DEVICE = "cuda"  # tests set this to "cpu"
BASE_EPOCHS = 16
BATCH = 64
N_POS = 16
N_NEG = 48
LR = 3e-4
HIDDEN_W = 256


# ------------------------------------------------------------------ dataset build
def episode_dirs(split: str) -> list[Path]:
    """New shards plus, optionally, the read-only old-GPU shard dirs (union view).

    SDSAFE_EPISODE_DIRS is a colon-separated list; when set it replaces the default
    single new-capture directory.  Old shards are never modified, only read.
    """
    env = os.environ.get("SDSAFE_EPISODE_DIRS")
    if env:
        return [Path(p.replace("{split}", split)) for p in env.split(":") if p]
    return [ART / "capture" / split / "episodes"]


def _episode_files(split: str) -> list[Path]:
    files: list[Path] = []
    for directory in episode_dirs(split):
        files.extend(sorted(directory.glob("*.npz")))
    return files


def build(split: str) -> dict[str, Any]:
    out = DS / split
    out.mkdir(parents=True, exist_ok=True)
    files = _episode_files(split)
    if not files:
        raise StudyError(f"no captured episodes for {split}")
    row_index = RowIndex(split)
    accepted = np.memmap(ACCEPTED_FEATURE_BIN, dtype="<u2", mode="r")
    n_total = 0
    rows: list[dict[str, Any]] = []
    chunk_b4: list[np.ndarray] = []
    chunk_b8: list[np.ndarray] = []
    chunk_cache: list[np.ndarray] = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        n = int(z["row_index"].shape[0])
        n_total += n
        rows.append(
            {
                "episode_id": str(z["episode_id"]),
                "scope": str(z["scope"]),
                "group_id": str(z["group_id"]),
                "label": bool(z["label"]),
                "n": n,
            }
        )
        chunk_b4.append(np.asarray(z["block4"]))
        chunk_b8.append(np.asarray(z["block8"]))
        chunk_cache.append(np.asarray(z["cache_row"], dtype=np.int64))
        z.close()
    b4 = np.concatenate(chunk_b4, axis=0)
    b8 = np.concatenate(chunk_b8, axis=0)
    cache = np.concatenate(chunk_cache)
    if b4.shape != (n_total, HIDDEN) or b8.shape != (n_total, HIDDEN):
        raise StudyError("captured block shapes are inconsistent")

    # row-level metadata and labels, joined by the accepted global row index
    meta_rows: list[np.ndarray] = []
    ep_ids: list[str] = []
    for path, info in zip(files, rows):
        z = np.load(path, allow_pickle=True)
        meta_rows.append(
            np.stack(
                [
                    np.asarray(z["row_index"], dtype=np.int64),
                    np.asarray(z["cache_row"], dtype=np.int64),
                    np.asarray(z["round_id"], dtype=np.int32),
                    np.asarray(z["message_index"], dtype=np.int32),
                    np.asarray(z["call_index"], dtype=np.int32),
                    np.asarray(z["segment_index"], dtype=np.int32),
                    np.asarray(z["token_start"], dtype=np.int32),
                    np.asarray(z["token_end"], dtype=np.int32),
                    np.asarray(z["positions"], dtype=np.int32),
                ]
            )
        )
        ep_ids.extend([info["episode_id"]] * info["n"])
        z.close()
    meta = np.concatenate(meta_rows, axis=1)
    row_ids = meta[0]
    if len(set(row_ids.tolist())) != row_ids.shape[0]:
        raise StudyError("duplicate captured row ids")

    groups = np.asarray([row_index.episode_group[e] for e in ep_ids], dtype=object)
    ep_labels = np.asarray([row_index.episode_label[e] for e in ep_ids], dtype=bool)
    scopes = np.asarray([row_index.episode_scope[e] for e in ep_ids], dtype=object)

    # final-layer hidden of the *accepted* matrix at the same row
    flat = accepted.reshape(accepted.shape[0] // HIDDEN, HIDDEN)
    final = np.ascontiguousarray(flat[cache])

    order = np.argsort(row_ids, kind="stable")
    arrays = {
        "row_index": row_ids[order],
        "cache_row": cache[order],
        "round_id": meta[2][order],
        "message_index": meta[3][order],
        "call_index": meta[4][order],
        "segment_index": meta[5][order],
        "token_start": meta[6][order],
        "token_end": meta[7][order],
        "positions": meta[8][order],
        "episode_id": np.asarray(ep_ids, dtype=object)[order],
        "group_id": groups[order],
        "episode_label": ep_labels[order],
        "scope": scopes[order],
    }
    b4 = b4.astype(np.uint16)[order]
    b8 = b8.astype(np.uint16)[order]
    final = final.astype(np.uint16)[order]

    early = np.empty((n_total, 2 * HIDDEN), dtype=np.uint16)
    early[:, :HIDDEN] = b4
    early[:, HIDDEN:] = b8
    np.save(out / "early_bf16.npy", early)
    np.save(out / "final_bf16.npy", final)
    del early, b4, b8, final

    # row labels: TRAIN uses the accepted v2 pool; DEV gets no training label
    label = np.full(n_total, -1, dtype=np.int8)
    in_loss = np.zeros(n_total, dtype=bool)
    if split == "train":
        pool = np.load(V2_TRAIN_ARRAYS, allow_pickle=True)
        pool_rows = np.asarray(pool["row_index"], dtype=np.int64)
        pool_label = np.asarray(pool["security_label"], dtype=np.int64)
        pool_mask = np.asarray(pool["security_mask"], dtype=bool)
        slot = {int(r): i for i, r in enumerate(pool_rows.tolist())}
        for i, r in enumerate(arrays["row_index"].tolist()):
            j = slot.get(int(r))
            if j is None:
                continue
            label[i] = int(pool_label[j])
            in_loss[i] = not bool(pool_mask[j])
    arrays["row_label"] = label
    arrays["in_loss"] = in_loss
    np.savez(out / "meta.npz", **arrays)
    report = {
        "split": split,
        "rows": int(n_total),
        "episodes": len(rows),
        "episodes_by_scope": {},
        "rows_in_loss": int(in_loss.sum()),
        "rows_with_label": int((label >= 0).sum()),
        "positives_in_loss": int(((label == 1) & in_loss).sum()),
        "negatives_in_loss": int(((label == 0) & in_loss).sum()),
        "bytes": int((out / "early_bf16.npy").stat().st_size + (out / "final_bf16.npy").stat().st_size),
    }
    from collections import Counter

    report["episodes_by_scope"] = dict(Counter(r["scope"] for r in rows))
    atomic_write_json(out / "dataset_report.json", report)
    return report


# ------------------------------------------------------------------ model / training
def _torch():
    import torch

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return torch


class Arm:
    """Row classifier over one input block combination."""

    def __init__(self, arm: str, seed: int, device: str | None = None) -> None:
        torch = _torch()
        torch.manual_seed(seed)
        dims = {"C": HIDDEN, "E": 2 * HIDDEN, "F": 3 * HIDDEN}
        if arm not in dims:
            raise StudyError(f"unknown arm {arm}")
        device = DEVICE if device is None else device
        self.arm = arm
        self.dim = dims[arm]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.dim, HIDDEN_W),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(HIDDEN_W, 1),
        ).to(device)
        self.device = device
        self.torch = torch

    def parameters(self):
        return self.net.parameters()


def load_split(split: str) -> dict[str, Any]:
    d = DS / split
    meta = np.load(d / "meta.npz", allow_pickle=True)
    out = {k: meta[k] for k in meta.files}
    out["early"] = np.load(d / "early_bf16.npy", mmap_mode="r")
    out["final"] = np.load(d / "final_bf16.npy", mmap_mode="r")
    return out


MAX_RESIDENT_BYTES = 7 << 30


def full_matrix(data: dict[str, Any], arm: str) -> np.ndarray:
    """The whole arm input as one float32 array (bf16 bit patterns -> float32)."""
    parts = []
    if arm in ("C", "F"):
        parts.append(np.asarray(data["final"]))
    if arm in ("E", "F"):
        parts.append(np.asarray(data["early"]))
    raw = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    flat = ((raw.astype(np.uint32)) << 16).view(np.float32).reshape(raw.shape[0], -1)
    return np.ascontiguousarray(flat)


class Resident:
    """Arm inputs resident on the compute device: one gather instead of a memmap concat."""

    def __init__(self, data: dict[str, Any], arm: str, torch) -> None:
        self.torch = torch
        self.arm = arm
        self.dim = {"C": HIDDEN, "E": 2 * HIDDEN, "F": 3 * HIDDEN}[arm]
        self.matrix = None
        self.source = data
        n = int(data["row_index"].shape[0])
        if n * self.dim * 4 <= MAX_RESIDENT_BYTES:
            host = full_matrix(data, arm)
            self.matrix = torch.from_numpy(host).to(DEVICE)

    def take(self, idx: np.ndarray):
        if self.matrix is not None:
            return self.matrix[self.torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(DEVICE)]
        return _batch_matrix(self.source, self.arm, idx, self.torch)


def _batch_matrix(data: dict[str, Any], arm: str, idx: np.ndarray, torch) -> Any:
    parts = []
    if arm in ("C", "F"):
        parts.append(np.asarray(data["final"][idx]))
    if arm in ("E", "F"):
        parts.append(np.asarray(data["early"][idx]))
    raw = np.concatenate(parts, axis=1)
    return torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16).to(DEVICE).float()


def group_folds(groups: Sequence[str]) -> np.ndarray:
    uniq = sorted(set(map(str, groups)))
    rng = np.random.default_rng(FOLD_SEED)
    order = rng.permutation(len(uniq))
    assign = {uniq[j]: int(i % FOLDS) for i, j in enumerate(order.tolist())}
    return np.asarray([assign[str(g)] for g in groups], dtype=np.int64)


def train_arm(
    arm: str,
    seed: int,
    fit_folds: Sequence[int],
    tag: str,
    epochs: int = BASE_EPOCHS,
) -> dict[str, Any]:
    torch = _torch()
    data = load_split("train")
    folds = group_folds(data["group_id"].tolist())
    base_mask = data["in_loss"] & (data["row_label"] >= 0)
    fit = base_mask & np.isin(folds, np.asarray(fit_folds))
    rows = np.flatnonzero(fit)
    labels = data["row_label"][rows].astype(np.int64)
    pos = rows[labels == 1]
    neg = rows[labels == 0]
    if pos.size == 0 or neg.size == 0:
        raise StudyError("both classes are required")
    model = Arm(arm, seed)
    resident = Resident(data, arm, torch)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(seed + 1000 * max(fit_folds))
    steps = max(1, len(rows) // BATCH)
    history = []
    for epoch in range(epochs):
        order = []
        for _ in range(steps):
            order.append(np.concatenate([rng.choice(pos, N_POS), rng.choice(neg, N_NEG)]))
        perm = rng.permutation(len(order))
        model.net.train()
        total, count = 0.0, 0
        for step in perm:
            idx = order[int(step)]
            x = resident.take(idx)
            y = torch.from_numpy(data["row_label"][idx].astype(np.float32)).to(DEVICE).unsqueeze(1)
            logit = model.net(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
            loss = 2.0 * loss.sum() / max(loss.numel(), 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * loss.numel()
            count += loss.numel()
        history.append(total / max(count, 1))
    model.net.eval()
    out_dir = ART / "models" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm}_s{seed}.pt"
    torch.save({"arm": arm, "seed": seed, "state": model.net.state_dict(), "dim": model.dim}, path)
    return {
        "tag": tag,
        "arm": arm,
        "seed": seed,
        "fit_folds": list(map(int, fit_folds)),
        "fit_rows": int(len(rows)),
        "fit_positives": int(len(pos)),
        "epochs": epochs,
        "params": int(sum(p.numel() for p in model.parameters())),
        "final_loss": history[-1],
        "first_loss": history[0],
        "checkpoint": str(path.relative_to(ART.parent)),
    }


def score_rows(arm: str, seed: int, tag: str, split: str) -> np.ndarray:
    torch = _torch()
    data = load_split(split)
    model = Arm(arm, seed)
    state = torch.load(ART / "models" / tag / f"{arm}_s{seed}.pt", weights_only=False)
    model.net.load_state_dict(state["state"])
    model.net.eval()
    out = np.empty(data["row_index"].shape[0], dtype=np.float32)
    resident = Resident(data, arm, torch)
    with torch.inference_mode():
        for start in range(0, out.shape[0], 4096):
            idx = np.arange(start, min(start + 4096, out.shape[0]))
            x = resident.take(idx)
            out[idx] = model.net(x).squeeze(1).float().cpu().numpy()
    return out


def episode_scores(row_logits: np.ndarray, episodes: np.ndarray) -> dict[str, np.ndarray]:
    ids = np.asarray(episodes, dtype=object)
    uniq, inv = np.unique(ids, return_inverse=True)
    out = np.full(uniq.shape[0], -np.inf, dtype=np.float64)
    np.maximum.at(out, inv, row_logits.astype(np.float64))
    return {"episode_id": uniq, "score": out}


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels).astype(np.int64)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fpr_threshold(labels: np.ndarray, scores: np.ndarray, fpr: float = 0.05) -> float:
    """Highest threshold whose FPR stays within budget (accepted operating-point rule)."""
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    neg = np.sort(s[~y])[::-1]
    k = int(math.floor(fpr * neg.size))
    if k <= 0:
        return float(neg[0] + 1e-9)
    return float((neg[k - 1] + neg[k]) / 2.0)


def confusions(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    y = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(arms: Sequence[str], tags: Sequence[str], split: str, extra_rows: dict[str, np.ndarray] | None = None):
    """Episode-level ensemble metrics for each (arm, tag) on ``split``."""
    data = load_split(split)
    episodes = data["episode_id"]
    uniq, inv = np.unique(np.asarray(episodes, dtype=object), return_inverse=True)
    ep_label = np.zeros(uniq.shape[0], dtype=bool)
    for e, lab in zip(data["episode_id"], data["episode_label"]):
        ep_label[np.searchsorted(uniq, e)] = bool(lab)
    results = {}
    for arm, tag in zip(arms, tags):
        per_seed = []
        for seed in TRAIN_SEEDS:
            logits = score_rows(arm, seed, tag, split)
            ep = np.full(uniq.shape[0], -np.inf)
            np.maximum.at(ep, inv, logits.astype(np.float64))
            per_seed.append(ep)
        score = np.mean(np.stack(per_seed), axis=0)
        thr = fpr_threshold(ep_label, score)
        cm = confusions(ep_label, score, thr)
        results[f"{arm}:{tag}"] = {
            "episodes": int(uniq.shape[0]),
            "positives": int(ep_label.sum()),
            **cm,
            "fnr": cm["fn"] / max(cm["tp"] + cm["fn"], 1),
            "fpr": cm["fp"] / max(cm["fp"] + cm["tn"], 1),
            "auc": auc(ep_label, score),
            "threshold": thr,
            "per_seed_score_sha256": None,
            "score": score,
            "episode_id": uniq,
            "labels": ep_label,
        }
    return results
