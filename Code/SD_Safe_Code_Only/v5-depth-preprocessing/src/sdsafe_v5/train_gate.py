"""Conditional second-wave arm K: gated low-rank fusion of [final | block16 | block24].

Pre-declared in PROTOCOL.md §6 and only run if the first-wave TRAIN fold-0 table shows a
depth signal without reaching FN ≤ 14.  Same TRAIN-only scaler, recipe, epochs, seeds and
folds as the first wave; parameter count within 0.02 % of the J head.

Architecture:
    z = Σ_b softplus(g_b) · (x_b W_b + c_b)      W_b: 4096→256, one scalar gate per block
    h = GELU(z);  logit = w2·h + b2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import ART, HIDDEN, TRAIN_SEEDS
from .common import atomic_write_json
from .train_depth import (
    BATCH, DEVICE, HIDDEN_W, LR, N_NEG, N_POS, SCORES, block_stats, group_folds,
    load_split, torch_mod,
)

OUT = ART / "models_depth"
ARM = "K"
EPOCHS = 16


class GatedLowRank:
    """Thin wrapper so the checkpoint format is explicit and validators can reconstruct it."""

    @staticmethod
    def build(torch):
        d, h, nb = HIDDEN, HIDDEN_W, 3
        W = torch.nn.Parameter(torch.empty(nb, d, h))
        b = torch.nn.Parameter(torch.zeros(nb, h))
        gates = torch.nn.Parameter(torch.full((nb,), 0.541324854612918))  # softplus(.) = 1
        for i in range(nb):
            torch.nn.init.kaiming_uniform_(W[i], a=5 ** 0.5)
        w2 = torch.nn.Linear(h, 1)
        return torch.nn.ParameterDict({"W": W, "b": b, "gates": gates}), w2

    @staticmethod
    def forward(torch, params, w2, x):
        # x: [B, 3*4096] standardized
        B = x.shape[0]
        parts = x.view(B, 3, HIDDEN)
        g = torch.nn.functional.softplus(params["gates"])
        z = None
        for i in range(3):
            zi = parts[:, i, :] @ params["W"][i] + params["b"][i]
            z = g[i] * zi if z is None else z + g[i] * zi
        h = torch.nn.functional.gelu(z)
        return w2(h)


def _fit(seed, fitf, x_gpu, tr, folds, epochs):
    torch = torch_mod()
    torch.manual_seed(seed)
    params, w2 = GatedLowRank.build(torch)
    params = params.to(DEVICE)
    w2 = w2.to(DEVICE)
    allp = list(params.parameters()) + list(w2.parameters())
    fit = (np.asarray(tr["in_loss"]) & (np.asarray(tr["row_label"]) >= 0)
           & np.isin(folds, np.asarray(fitf)))
    rows = np.flatnonzero(fit)
    labels = np.asarray(tr["row_label"])[rows].astype(np.int64)
    pos, neg = rows[labels == 1], rows[labels == 0]
    opt = torch.optim.AdamW(allp, lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(seed + 1000 * int(sum(fitf)))
    steps = max(1, len(rows) // BATCH)
    y_all = torch.from_numpy(np.asarray(tr["row_label"], dtype=np.float32)).to(DEVICE)
    hist = []
    for _epoch in range(epochs):
        order = [np.concatenate([rng.choice(pos, N_POS), rng.choice(neg, N_NEG)]) for _ in range(steps)]
        for step in rng.permutation(len(order)):
            idx = order[int(step)]
            t = torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(DEVICE)
            logit = GatedLowRank.forward(torch, params, w2, x_gpu[t])
            yb = y_all[t].unsqueeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb, reduction="none")
            loss = 2.0 * loss.sum() / max(loss.numel(), 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0)
            opt.step()
            hist.append(float(loss.detach()))
    return params, w2, hist


def _score(x_gpu, params, w2):
    torch = torch_mod()
    n = x_gpu.shape[0]
    out = np.empty(n, dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n, 4096):
            e = min(s + 4096, n)
            out[s:e] = GatedLowRank.forward(torch, params, w2, x_gpu[s:e]).squeeze(1).float().cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tags", default="ho,full")
    args = ap.parse_args()
    torch = torch_mod()
    tr = load_split("train")
    dv = load_split("dev")
    stats = block_stats(tr)
    from .train_depth import arm_matrix
    t0 = time.time()
    x_tr = arm_matrix(tr, "J", stats)
    x_dv = arm_matrix(dv, "J", stats)
    fold_path = ART / "dataset" / "folds.npy"
    if fold_path.is_file():
        folds = np.load(fold_path)
    else:
        folds = group_folds(np.asarray(tr["group_id"]).tolist())
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(fold_path, folds)
    xg = torch.from_numpy(x_tr).to(DEVICE)
    xd = torch.from_numpy(x_dv).to(DEVICE)
    runs = []
    for tag in args.tags.split(","):
        fitf = (1, 2, 3) if tag == "ho" else (0, 1, 2, 3)
        for seed in TRAIN_SEEDS:
            p = OUT / tag / f"{ARM}_s{seed}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.is_file():
                params, w2, hist = _fit(seed, fitf, xg, tr, folds, args.epochs)
                torch.save({"arch": "gated_lowrank", "seed": seed, "tag": tag, "arm": ARM,
                            "dim": 3 * HIDDEN, "fit_folds": list(fitf),
                            "params": {k: v.detach().cpu() for k, v in params.items()},
                            "w2": {k: v.detach().cpu() for k, v in w2.state_dict().items()},
                            "first_loss": hist[0], "final_loss": hist[-1]}, p)
            ck = torch.load(p, weights_only=False)
            params = torch.nn.ParameterDict({k: v.to(DEVICE) for k, v in ck["params"].items()})
            w2 = torch.nn.Linear(HIDDEN_W, 1).to(DEVICE)
            w2.load_state_dict(ck["w2"])
            SCORES.mkdir(parents=True, exist_ok=True)
            np.save(SCORES / f"{ARM}_{tag}_s{seed}_train.npy", _score(xg, params, w2))
            np.save(SCORES / f"{ARM}_{tag}_s{seed}_dev.npy", _score(xd, params, w2))
            runs.append({"arm": ARM, "tag": tag, "seed": seed, "fit_folds": list(fitf),
                         "final_loss": ck.get("final_loss")})
            print(json.dumps({"fitted": f"K:{tag}:s{seed}"}), flush=True)
    atomic_write_json(ART / "runs_gate.json", {"runs": runs, "build_seconds": round(time.time() - t0, 1)})
    print(json.dumps({"runs": len(runs)}))


if __name__ == "__main__":
    main()
