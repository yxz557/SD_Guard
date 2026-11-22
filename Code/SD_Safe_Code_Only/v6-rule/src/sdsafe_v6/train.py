"""Pre-registered TRAIN-only fitting, fold-0 selection, 3-seed final fits, DEV scoring.

Arms (declared before any DEV number is produced):

  S0  main security head only (text/facts features)
  S1  main + parameter-level aux heads, lambda 0.3
  S2  main + parameter-level aux heads, lambda 1.0
  S3  main + *label-shuffled* aux heads, lambda 1.0  (capacity / supervision control)

Fixed schedule: batch 64 (16 positive / 48 negative main rows), 600 steps per epoch, at most 8
epochs, AdamW lr 1e-3, wd 1e-2, dropout 0.2, hashed bags 2^17 x 16, hidden 256.  Epoch count is
selected on TRAIN fold 0 held-out episodes only (AUC), then the chosen epoch count is used for the
3-seed full-TRAIN fits that score common DEV.  DEV is never read during fitting or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import feats, metrics, model

ART = Path("/root/sdsafe-compute/v6-param/artifacts")
TRAIN_VIEWS = [ART / "views/train_all.jsonl", ART / "views/train_all_fixup.jsonl"]
DEV_VIEWS = [ART / "views/dev_common.jsonl", ART / "views/dev_common_fixup.jsonl"]
POOL = Path("/dev/shm/sdsafe-gpu-v4/upstream/sdsafe-lowcost-authorization-v2/artifacts/tensors/train_pool_arrays.npz")
AUX_CACHE = ART / "teacher/aux_calls_v2.jsonl"
AUX_VIEWS = ART / "views/aux.jsonl"

ACTION_MAP = {"authorized": 0, "unauthorized": 1, "unclear": 2}
REL_MAP = {"direct": 0, "prerequisite": 1, "extra": 2, "conflicting": 3, "unclear": 4}
ALIGN_MAP = {"full": 0, "partial": 1, "none": 2, "unclear": 3}

DEFAULTS = dict(batch=64, n_pos=16, steps_per_epoch=600, max_epochs=8, lr=1e-3, wd=1e-2,
                dropout=0.2, seeds=(0, 1, 2), folds=4)


def fold_of(group_id: str, folds: int = DEFAULTS["folds"]) -> int:
    return int(hashlib.blake2b(str(group_id).encode(), digest_size=4).hexdigest(), 16) % folds


class Tensors:
    """Ragged feature cache with fast per-batch assembly."""

    def __init__(self, cache: dict[str, Any], mean: np.ndarray, std: np.ndarray, device: str):
        self.device = device
        self.idx: dict[str, list[np.ndarray]] = {}
        self.off: dict[str, np.ndarray] = {}
        n = cache["row_index"].shape[0]
        for channel in ("cur", "vis", "user"):
            flat = cache[f"{channel}_idx"]
            off = cache[f"{channel}_off"]
            self.idx[channel] = [flat[off[i]:off[i + 1]].astype(np.int64) for i in range(n)]
            self.off[channel] = off
        # the source channel is indexed by (event, source) bags
        src_flat = cache["src_idx"]
        src_off = cache["src_off"]
        src_counts = cache["src_counts"].astype(np.int64)
        bags = [src_flat[src_off[j]:src_off[j + 1]].astype(np.int64) for j in range(src_off.size - 1)]
        self.idx["src"] = []
        cursor = 0
        for i in range(n):
            k = int(src_counts[i])
            self.idx["src"].append(bags[cursor:cursor + k])
            cursor += k
        self.src_counts = cache["src_counts"].astype(np.int64)
        self.dense = ((cache["dense"] - mean) / std).astype(np.float32)
        self.n = n
        self.row_index = cache["row_index"]
        self.episode_id = cache["episode_id"]
        self.group_id = cache["group_id"]
        self.event_type = cache["event_type"]
        self.row_label = cache["row_label"]
        self.episode_label = cache["episode_label"]
        self.main_label = cache["main_label"] if "main_label" in cache else np.full(n, -1, np.int64)
        self.in_pool = cache["in_pool"] if "in_pool" in cache else np.zeros(n, bool)

    def batch(self, rows: np.ndarray) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        dev = self.device
        for channel in ("cur", "vis", "user"):
            parts = [self.idx[channel][i] for i in rows]
            lens = [p.size for p in parts]
            flat = np.concatenate(parts) if parts else np.zeros(0, np.int64)
            off = np.zeros(len(rows), dtype=np.int64)
            np.cumsum(lens[:-1], out=off[1:])
            out[f"{channel}_idx"] = torch.from_numpy(flat).to(dev)
            out[f"{channel}_off"] = torch.from_numpy(off).to(dev)
        flat_parts = []
        offs = []
        counts = []
        total = 0
        for i in rows:
            bags = self.idx["src"][i]
            counts.append(len(bags))
            for bag in bags:
                offs.append(total)
                total += bag.size
                flat_parts.append(bag)
        flat = np.concatenate(flat_parts) if flat_parts else np.zeros(0, np.int64)
        out["src_idx"] = torch.from_numpy(flat).to(dev)
        out["src_off"] = torch.tensor(offs, dtype=torch.long, device=dev)
        out["src_counts"] = torch.tensor(counts, dtype=torch.long, device=dev)
        out["dense"] = torch.from_numpy(self.dense[rows]).to(dev)
        return out


def forward(model_: model.Student, tensors: Tensors, rows: np.ndarray) -> dict[str, torch.Tensor]:
    batch = tensors.batch(rows)
    cur = model_.emb(batch["cur_idx"], batch["cur_off"])
    vis = model_.emb(batch["vis_idx"], batch["vis_off"])
    user = model_.emb(batch["user_idx"], batch["user_off"])
    src_rows = model_.emb(batch["src_idx"], batch["src_off"])
    repeat = torch.repeat_interleave(torch.arange(batch["src_counts"].numel(), device=cur.device),
                                     batch["src_counts"])
    src = torch.zeros_like(cur)
    bag_counts = torch.repeat_interleave(batch["src_counts"], batch["src_counts"]).float()
    src.index_add_(0, repeat, src_rows / bag_counts.unsqueeze(-1))
    x = torch.cat([cur, vis, user, src, batch["dense"]], dim=-1)
    h = model_.trunk(model_.norm(x))
    return {"main": model_.head_main(h).squeeze(-1), "action": model_.head_action(h),
            "rel": model_.head_rel(h), "align": model_.head_align(h), "arg": model_.head_arg(h)}


def aux_arrays(aux_labels: dict[int, dict[str, int]], row_index: np.ndarray) -> dict[str, np.ndarray]:
    n = row_index.shape[0]
    action = np.full(n, -1, np.int64)
    rel = np.full(n, -1, np.int64)
    align = np.full(n, -1, np.int64)
    arg = np.full(n, -1, np.int64)
    for i, r in enumerate(row_index):
        lab = aux_labels.get(int(r))
        if not lab:
            continue
        action[i] = lab["action"]
        rel[i] = lab["rel"]
        align[i] = lab["align"]
        arg[i] = lab["arg"]
    return {"action": action, "rel": rel, "align": align, "arg": arg}


def train_one(tensors: Tensors, *, rows_fit: np.ndarray, aux: dict[str, np.ndarray], epochs: int,
              lam: float, seed: int, device: str, shuffle_aux: bool = False,
              steps_per_epoch: int = DEFAULTS["steps_per_epoch"],
              net: model.Student | None = None, rng_offset: int = 0) -> model.Student:
    """Train (or continue training) one student.  ``epochs`` epochs of ``steps_per_epoch`` steps."""
    rng = np.random.default_rng(seed * 100000 + rng_offset)
    if net is None:
        torch.manual_seed(seed)
        net = model.Student(model.ModelConfig(dropout=DEFAULTS["dropout"])).to(device)
    else:
        net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=DEFAULTS["lr"], weight_decay=DEFAULTS["wd"])
    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss()
    fit = rows_fit[tensors.main_label[rows_fit] >= 0]
    pos = fit[tensors.main_label[fit] == 1]
    neg = fit[tensors.main_label[fit] == 0]
    aux_rows = rows_fit[aux["action"][rows_fit] >= 0]
    if shuffle_aux and aux_rows.size:
        perm = rng.permutation(aux_rows.size)
        for name in ("action", "rel", "align", "arg"):
            aux[name][aux_rows] = aux[name][aux_rows][perm]
    for _epoch in range(epochs):
        net.train()
        for _step in range(steps_per_epoch):
            bp = rng.choice(pos, size=DEFAULTS["n_pos"], replace=pos.size < DEFAULTS["n_pos"])
            bn = rng.choice(neg, size=DEFAULTS["batch"] - DEFAULTS["n_pos"],
                            replace=neg.size < (DEFAULTS["batch"] - DEFAULTS["n_pos"]))
            rows = np.concatenate([bp, bn])
            out = forward(net, tensors, rows)
            y = torch.from_numpy(tensors.main_label[rows].astype(np.float32)).to(device)
            loss = bce(out["main"], y)
            if lam > 0 and aux_rows.size:
                ab = rng.choice(aux_rows, size=min(64, aux_rows.size), replace=aux_rows.size < 64)
                aout = forward(net, tensors, ab)
                aux_loss = 0.0
                for name in ("action", "rel", "align", "arg"):
                    target = aux[name][ab]
                    mask = target >= 0
                    if mask.sum() == 0:
                        continue
                    aux_loss = aux_loss + ce(aout[name][mask],
                                             torch.from_numpy(target[mask]).to(device))
                loss = loss + lam * aux_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net


def score_rows(net: model.Student, tensors: Tensors, rows: np.ndarray, device: str,
               batch: int = 1024) -> np.ndarray:
    net.eval()
    out = np.zeros(rows.size, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, rows.size, batch):
            chunk = rows[start:start + batch]
            res = forward(net, tensors, chunk)
            out[start:start + chunk.size] = res["main"].detach().float().cpu().numpy()
    return out


def episode_scores_for(tensors: Tensors, rows: np.ndarray, row_scores: np.ndarray,
                       episodes: list[str]) -> np.ndarray:
    index = {e: i for i, e in enumerate(episodes)}
    out = np.full(len(episodes), -np.inf, dtype=np.float32)
    eps = np.asarray([index[str(e)] for e in tensors.episode_id[rows]], dtype=np.int64)
    np.maximum.at(out, eps, row_scores.astype(np.float32))
    return out
