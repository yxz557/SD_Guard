"""Train/evaluate the S4 per-argument field arm (cache-only; no new labels).

S4 = event trunk from S0 + a shared per-field encoder over (path, value, source window, user goal)
with a path-level authorization BCE, plus the event full/partial alignment head.  The event logit
gets a learned max/mean aggregation of field logits.  Event action/relation and the old aggregate
argument head are *not* used, so this is a clean test of fine-grained supervision.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sdsafe_v6 import argfeats, argmodel, aux_data, feats, metrics, model, train

ART = train.ART


def score_rows(net: argmodel.ArgStudent, tr: train.Tensors, ar: argmodel.ArgTensors,
               rows: np.ndarray, device: str, batch: int = 512) -> np.ndarray:
    net.eval()
    out = np.zeros(rows.size, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, rows.size, batch):
            chunk = rows[start:start + batch]
            res = net(tr.batch(chunk), ar.batch(chunk))
            out[start:start + chunk.size] = res["main"].float().cpu().numpy()
    return out


def train_arg_one(tr: train.Tensors, ar: argmodel.ArgTensors, *,
                  rows_fit: np.ndarray, aux: dict[str, np.ndarray],
                  per_arg: dict[int, list[dict]], epochs: int, seed: int, device: str,
                  lam_align: float = 0.3, lam_arg: float = 1.0,
                  steps_per_epoch: int = 600,
                  net: argmodel.ArgStudent | None = None, rng_offset: int = 0) -> argmodel.ArgStudent:
    rng = np.random.default_rng(seed * 100000 + rng_offset)
    if net is None:
        torch.manual_seed(seed)
        net = argmodel.ArgStudent(model.ModelConfig(dropout=train.DEFAULTS["dropout"])).to(device)
    else:
        net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=train.DEFAULTS["lr"], weight_decay=train.DEFAULTS["wd"])
    bce = torch.nn.BCEWithLogitsLoss()
    ce = torch.nn.CrossEntropyLoss()
    fit = rows_fit[tr.main_label[rows_fit] >= 0]
    pos = fit[tr.main_label[fit] == 1]
    neg = fit[tr.main_label[fit] == 0]
    align_rows = rows_fit[aux["align"][rows_fit] >= 0]
    arg_rows = np.asarray([r for r in rows_fit if per_arg.get(int(tr.row_index[r]))], dtype=np.int64)
    for _epoch in range(epochs):
        net.train()
        for _step in range(steps_per_epoch):
            bp = rng.choice(pos, size=train.DEFAULTS["n_pos"], replace=pos.size < train.DEFAULTS["n_pos"])
            bn = rng.choice(neg, size=train.DEFAULTS["batch"] - train.DEFAULTS["n_pos"],
                            replace=neg.size < (train.DEFAULTS["batch"] - train.DEFAULTS["n_pos"]))
            rows = np.concatenate([bp, bn])
            out = net(tr.batch(rows), ar.batch(rows))
            y = torch.from_numpy(tr.main_label[rows].astype(np.float32)).to(device)
            loss = bce(out["main"], y)
            if lam_align > 0 and align_rows.size:
                ab = rng.choice(align_rows, size=min(64, align_rows.size),
                                replace=align_rows.size < 64)
                aout = net(tr.batch(ab), ar.batch(ab))
                target = aux["align"][ab]
                mask = target >= 0
                if mask.sum():
                    loss = loss + lam_align * ce(aout["align"][mask],
                                                 torch.from_numpy(target[mask]).to(device))
            if lam_arg > 0 and arg_rows.size:
                ab = rng.choice(arg_rows, size=min(64, arg_rows.size),
                                replace=arg_rows.size < 64)
                aout = net(tr.batch(ab), ar.batch(ab))
                paths = aout.get("paths") if isinstance(aout, dict) else None
                # rebuild the path strings exactly in the same order as the arg batch
                argb = ar.batch(ab)
                target = np.full(len(argb["paths"]), -1, dtype=np.int64)
                for i, (path, local) in enumerate(zip(argb["paths"], argb["row"].tolist())):
                    labels = {e["path"]: int(e["authorization"]) for e in
                              per_arg.get(int(tr.row_index[ab[local]]), [])}
                    if path in labels:
                        target[i] = labels[path]
                mask = target >= 0
                if mask.sum():
                    tt = torch.from_numpy(target[mask]).float().to(device)
                    loss = loss + lam_arg * bce(aout["arg_logits"][mask], tt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ART / "study_arg_v3.json"))
    args = ap.parse_args(argv)
    t0 = time.time()
    device = args.device if torch.cuda.is_available() else "cpu"

    cache = {k: v for k, v in np.load(ART / "cache_train_v3.npz", allow_pickle=True).items()}
    cache = feats.attach_main_labels(cache, train.POOL)
    scaler = np.load(ART / "scaler_v6_v3.npz", allow_pickle=True)
    dcache = {k: v for k, v in np.load(ART / "cache_dev_v3.npz", allow_pickle=True).items()}
    tr = train.Tensors(cache, scaler["mean"], scaler["std"], device)
    dv = train.Tensors(dcache, scaler["mean"], scaler["std"], device)
    ar = argmodel.ArgTensors(argfeats.load(ART / "arg_cache_train_v3.npz"), device)
    ad = argmodel.ArgTensors(argfeats.load(ART / "arg_cache_dev_v3.npz"), device)

    aux_labels, aux_manifest, per_arg = aux_data.build(write=False)
    aux = train.aux_arrays(aux_labels, tr.row_index)
    report: dict = {"device": device, "aux": aux_manifest["stats"], "epoch_scan": {}, "params": None}

    fold = np.asarray([train.fold_of(g) for g in tr.group_id], dtype=np.int64)
    episodes = sorted(set(tr.episode_id.tolist()))
    ep_index = {e: i for i, e in enumerate(episodes)}
    ep_fold = np.zeros(len(episodes), np.int64)
    ep_label = np.zeros(len(episodes), np.int64)
    for i, e in enumerate(tr.episode_id):
        j = ep_index[e]
        ep_fold[j] = fold[i]
        ep_label[j] = int(tr.episode_label[i])
    rows_fit = np.flatnonzero(fold != 0)
    rows_ho = np.flatnonzero(fold == 0)
    net = None
    best = None
    for epochs in range(1, args.max_epochs + 1):
        net = train_arg_one(tr, ar, rows_fit=rows_fit, aux=aux, per_arg=per_arg, epochs=1,
                            seed=100, device=device, net=net, rng_offset=epochs)
        sc = score_rows(net, tr, ar, rows_ho, device)
        ep = train.episode_scores_for(tr, rows_ho, sc, episodes)
        held = ep_fold == 0
        auc = metrics.auc(ep[held], ep_label[held])
        thr = metrics.threshold_at_fpr(ep[held], ep_label[held], 0.05)
        conf = metrics.confusion(ep[held], ep_label[held], thr)
        report["epoch_scan"][str(epochs)] = {"auc": auc, "fn": conf["fn"], "fp": conf["fp"],
                                             "threshold": thr}
        key = (auc, -conf["fn"])
        if best is None or key > best[0]:
            best = (key, epochs, auc, conf)
    report["selected_epoch"] = best[1]
    report["selected_fold0"] = {"auc": best[2], **{k: best[3][k] for k in ("fn", "fp", "tp", "tn")}}
    print(f"[S4] selected epoch={best[1]} auc={best[2]:.6f} fn={best[3]['fn']} fp={best[3]['fp']}", flush=True)

    seed_scores = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"[S4] final fit seed {seed} epochs {best[1]}", flush=True)
        net = train_arg_one(tr, ar, rows_fit=np.arange(tr.n), aux=aux, per_arg=per_arg,
                            epochs=best[1], seed=seed, device=device)
        sc = score_rows(net, dv, ad, np.arange(dv.n), device)
        seed_scores.append(sc)
        torch.save({"state_dict": net.state_dict(), "config": vars(net.base.cfg),
                    "arm": "S4", "seed": seed, "epochs": best[1], "params": net.n_params()},
                   ART / f"ckpt_S4_v3_seed{seed}.pt")
        report["params"] = net.n_params()
    seed_scores = np.stack(seed_scores, axis=0)
    np.savez(ART / "dev_scores_S4_v3.npz", row_scores=seed_scores, row_index=dv.row_index,
             episode_id=dv.episode_id)
    dev_eps = sorted(set(dv.episode_id.tolist()))
    per_seed_ep = np.stack([train.episode_scores_for(dv, np.arange(dv.n), seed_scores[s], dev_eps)
                            for s in range(seed_scores.shape[0])], axis=0)
    ens = per_seed_ep.mean(axis=0, dtype=np.float32)
    np.savez(ART / "dev_episode_S4_v3.npz", episode_id=np.asarray(dev_eps, dtype=object),
             per_seed=per_seed_ep, ensemble=ens)
    report["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("selected_epoch", "selected_fold0", "params", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
