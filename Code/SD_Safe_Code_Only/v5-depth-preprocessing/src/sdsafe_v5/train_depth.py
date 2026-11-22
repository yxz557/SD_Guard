"""Train the pre-registered depth arms on the new block16/24 dataset.

Arms (pre-registered):
  ``C``  final only (accepted feature matrix) -- matched capacity control
  ``H``  final || block16
  ``I``  final || block24
  ``J``  final || block16 || block24  (capacity-matched to the imported G48 control)
  ``Craw`` final only without standardization (pipeline reproduction of the v4 C result)

All except ``Craw`` are standardized with TRAIN-only per-feature mu/sigma (the G recipe);
this is pre-declared, not selected on DEV.  Recipe is byte-for-byte the accepted one:
Linear(d,256) -> GELU -> Dropout(.1) -> Linear(256,1), AdamW lr 3e-4 wd .01, 16 epochs,
balanced 16 pos + 48 neg, grad clip 1.0, seeds 0/1/2, 4 group folds (seed 20261113).

Checkpoints are per (arm, tag, seed); resume skips only when the checkpoint exists.
Row logits are saved per (arm, tag, seed, split) so evaluation never re-runs a fit.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import ART, FOLD_SEED, FOLDS, HIDDEN, TRAIN_SEEDS
from .common import atomic_write_json, json_load
from .loader import bf16_bits_to_f32

DS = Path(os.environ.get("SDSAFE_DEPTH_DATASET", "/dev/shm/sdsafe-depth-v5/dataset"))
OUT = ART / "models_depth"
SCORES = ART / "scores"
DEVICE = os.environ.get("SDSAFE_DEVICE", "cuda")
BASE_EPOCHS = 16
BATCH = 64
N_POS = 16
N_NEG = 48
LR = 3e-4
HIDDEN_W = 256
ARMS = ("C", "Craw", "H", "I", "J")
BLOCKS = {"C": ("final",), "Craw": ("final",), "H": ("final", "b16"),
          "I": ("final", "b24"), "J": ("final", "b16", "b24")}


def torch_mod():
    import torch

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return torch


def load_split(split: str) -> dict[str, Any]:
    d = DS / split
    if not (d / "meta.npz").is_file():
        raise FileNotFoundError(f"dataset missing: {d} (run dataset_depth first)")
    meta = np.load(d / "meta.npz", allow_pickle=True)
    out = {k: meta[k] for k in meta.files}
    out["final"] = np.load(d / "final_bf16.npy", mmap_mode="r")
    out["b16"] = np.load(d / "block16_bf16.npy", mmap_mode="r")
    out["b24"] = np.load(d / "block24_bf16.npy", mmap_mode="r")
    return out


def block_stats(data: dict[str, Any], chunk: int = 8192) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """TRAIN-only per-feature mu/sigma over in-loss rows (chunked, float64 accumulators)."""
    mask = np.asarray(data["in_loss"]) & (np.asarray(data["row_label"]) >= 0)
    idx = np.flatnonzero(mask)
    stats = {}
    for name in ("final", "b16", "b24"):
        mat = data[name]
        s = np.zeros(HIDDEN, dtype=np.float64)
        ss = np.zeros(HIDDEN, dtype=np.float64)
        n = 0
        for start in range(0, idx.size, chunk):
            sel = idx[start:start + chunk]
            x = bf16_bits_to_f32(np.asarray(mat[sel])).astype(np.float64)
            s += x.sum(axis=0)
            ss += (x * x).sum(axis=0)
            n += x.shape[0]
        mu = s / n
        var = np.maximum(ss / n - mu * mu, 0.0)
        sigma = np.maximum(np.sqrt(var), 1e-6)
        stats[name] = (mu.astype(np.float32), sigma.astype(np.float32))
    return stats


def arm_matrix(data: dict[str, Any], arm: str, stats: dict[str, tuple[np.ndarray, np.ndarray]] | None,
               chunk: int = 8192) -> np.ndarray:
    names = BLOCKS[arm]
    n = int(data["row_index"].shape[0])
    dim = len(names) * HIDDEN
    out = np.empty((n, dim), dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        for j, name in enumerate(names):
            raw = bf16_bits_to_f32(np.asarray(data[name][start:stop]))
            if stats is not None:
                mu, sigma = stats[name]
                raw = (raw - mu) / sigma
            out[start:stop, j * HIDDEN:(j + 1) * HIDDEN] = raw
    return out


def group_folds(groups: Sequence[str]) -> np.ndarray:
    uniq = sorted(set(map(str, groups)))
    rng = np.random.default_rng(FOLD_SEED)
    order = rng.permutation(len(uniq))
    assign = {uniq[j]: int(i % FOLDS) for i, j in enumerate(order.tolist())}
    return np.asarray([assign[str(g)] for g in groups], dtype=np.int64)


def _fit(arm: str, seed: int, tag: str, fitf: Sequence[int], x_gpu, data: dict[str, Any],
         folds: np.ndarray, epochs: int) -> dict[str, Any]:
    torch = torch_mod()
    device = DEVICE
    torch.manual_seed(seed)
    dim = x_gpu.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(dim, HIDDEN_W), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(HIDDEN_W, 1),
    ).to(device)
    fit = (np.asarray(data["in_loss"]) & (np.asarray(data["row_label"]) >= 0)
           & np.isin(folds, np.asarray(fitf)))
    rows = np.flatnonzero(fit)
    labels = np.asarray(data["row_label"])[rows].astype(np.int64)
    pos, neg = rows[labels == 1], rows[labels == 0]
    if pos.size == 0 or neg.size == 0:
        raise RuntimeError("both classes required")
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(seed + 1000 * int(sum(fitf)))
    steps = max(1, len(rows) // BATCH)
    history = []
    y_all = torch.from_numpy(np.asarray(data["row_label"], dtype=np.float32)).to(device)
    for _epoch in range(epochs):
        order = [np.concatenate([rng.choice(pos, N_POS), rng.choice(neg, N_NEG)]) for _ in range(steps)]
        net.train()
        tot = cnt = 0.0
        for step in rng.permutation(len(order)):
            idx = order[int(step)]
            xb = x_gpu[torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device)]
            yb = y_all[torch.from_numpy(np.asarray(idx, dtype=np.int64)).to(device)].unsqueeze(1)
            logit = net(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb, reduction="none")
            loss = 2.0 * loss.sum() / max(loss.numel(), 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()) * loss.numel()
            cnt += loss.numel()
        history.append(tot / max(cnt, 1))
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm}_s{seed}.pt"
    torch.save({"arm": arm, "seed": seed, "state": net.state_dict(), "dim": dim,
                "tag": tag, "fit_folds": list(map(int, fitf))}, path)
    return {"arm": arm, "tag": tag, "seed": seed, "fit_folds": list(map(int, fitf)),
            "fit_rows": int(len(rows)), "fit_positives": int(len(pos)),
            "epochs": epochs, "final_loss": history[-1], "first_loss": history[0],
            "checkpoint": str(path.relative_to(ART.parent))}


def _load_net(ckpt: Path):
    torch = torch_mod()
    state = torch.load(ckpt, weights_only=False)
    net = torch.nn.Sequential(
        torch.nn.Linear(state["dim"], HIDDEN_W), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(HIDDEN_W, 1),
    ).to(DEVICE)
    net.load_state_dict(state["state"])
    net.eval()
    return net


def _score(net, x_gpu) -> np.ndarray:
    torch = torch_mod()
    n = x_gpu.shape[0]
    out = np.empty(n, dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, n, 4096):
            e = min(s + 4096, n)
            out[s:e] = net(x_gpu[s:e]).squeeze(1).float().cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=BASE_EPOCHS)
    ap.add_argument("--arms", default="C,Craw,H,I,J")
    ap.add_argument("--tags", default="ho,full")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    torch = torch_mod()
    arms = args.arms.split(",")
    tags = args.tags.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    tr = load_split("train")
    dv = load_split("dev")
    t_stats = block_stats(tr)
    stats_payload = {k: {"mu": v[0].tolist(), "sigma": v[1].tolist()} for k, v in t_stats.items()}
    (ART / "dataset").mkdir(parents=True, exist_ok=True)
    atomic_write_json(ART / "dataset" / "train_stats.json", {
        "fit_rows": int((np.asarray(tr["in_loss"]) & (np.asarray(tr["row_label"]) >= 0)).sum()),
        "blocks": stats_payload,
        "note": "TRAIN-only in-loss rows; DEV never used",
    })
    folds = group_folds(np.asarray(tr["group_id"]).tolist())
    np.save(ART / "dataset" / "folds.npy", folds)
    SCORES.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for arm in arms:
        stats = None if arm == "Craw" else t_stats
        t0 = time.time()
        x_tr = arm_matrix(tr, arm, stats)
        x_dv = arm_matrix(dv, arm, stats)
        print(json.dumps({"arm": arm, "dim": int(x_tr.shape[1]),
                          "build_seconds": round(time.time() - t0, 1)}), flush=True)
        x_tr_gpu = torch.from_numpy(x_tr).to(DEVICE)
        x_dv_gpu = torch.from_numpy(x_dv).to(DEVICE)
        for tag in tags:
            fitf = (1, 2, 3) if tag == "ho" else (0, 1, 2, 3)
            for seed in seeds:
                ckpt = OUT / tag / f"{arm}_s{seed}.pt"
                t0 = time.time()
                if not ckpt.is_file():
                    rec = _fit(arm, seed, tag, fitf, x_tr_gpu, tr, folds, args.epochs)
                else:
                    rec = {"arm": arm, "tag": tag, "seed": seed, "fit_folds": list(fitf),
                           "checkpoint": str(ckpt.relative_to(ART.parent)), "resumed": True}
                net = _load_net(ckpt)
                sc_tr = _score(net, x_tr_gpu)
                sc_dv = _score(net, x_dv_gpu)
                np.save(SCORES / f"{arm}_{tag}_s{seed}_train.npy", sc_tr)
                np.save(SCORES / f"{arm}_{tag}_s{seed}_dev.npy", sc_dv)
                rec["score_seconds"] = round(time.time() - t0, 1)
                runs.append(rec)
                print(json.dumps({"fitted": f"{arm}:{tag}:s{seed}",
                                  "loss": rec.get("final_loss"), "fit_rows": rec.get("fit_rows"),
                                  "seconds": rec.get("score_seconds")}), flush=True)
                del net
        del x_tr, x_dv, x_tr_gpu, x_dv_gpu
        torch.cuda.empty_cache()
    atomic_write_json(ART / "runs_depth.json", {"runs": runs, "arms": arms, "tags": tags,
                                                "seeds": seeds, "device": DEVICE})
    print(json.dumps({"runs": len(runs)}), flush=True)


if __name__ == "__main__":
    main()
