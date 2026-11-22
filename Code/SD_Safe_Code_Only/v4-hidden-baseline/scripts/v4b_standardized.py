"""v4b arm G — pre-declared after the v4 negative result, before running.

Question: was the pre-registered F head limited by the raw bf16 scale (final absmean ~1.5,
early ~0.25), rather than by missing information?  G = F's exact input
``[accepted final ‖ block4 ‖ block8]`` (12288) with per-feature standardization fit on
authority-TRAIN in-loss rows only (mu/sigma), same MLP/recipe/epochs/seeds/folds.
Selection is TRAIN held-out fold 0; DEV is reported as exploratory reuse.  No new capture,
no new layer, no label/denominator change, no DEV selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from sdsafe_v4 import ART, FOLD_SEED, FOLDS, HIDDEN, TRAIN_SEEDS
from sdsafe_v4.common import atomic_write_json
from sdsafe_v4.train_eval import (
    BATCH, HIDDEN_W, LR, N_NEG, N_POS, auc, confusions, fpr_threshold, group_folds,
    load_split,
)
from sdsafe_v4.run_all import _bootstrap

DEVICE = "cuda"
EPOCHS = 16
DIM = 3 * HIDDEN
OUT = ART / "eval" / "v4b_report.json"
MODELS = ART / "models_v4b"


def matrix(data, mu, sigma):
    parts = [np.asarray(data["final"]), np.asarray(data["early"])]
    raw = np.concatenate(parts, axis=1)
    x = ((raw.astype(np.uint32)) << 16).view(np.float32)
    return (x - mu) / sigma


def train(seed, fitf, x, row_label, in_loss, fold_assign):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(DIM, HIDDEN_W), torch.nn.GELU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(HIDDEN_W, 1),
    ).to(DEVICE)
    fit = in_loss & (row_label >= 0) & np.isin(fold_assign, np.asarray(fitf))
    rows = np.flatnonzero(fit)
    labels = row_label[rows].astype(np.int64)
    pos, neg = rows[labels == 1], rows[labels == 0]
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    rng = np.random.default_rng(seed + 1000 * int(sum(fitf)))
    steps = max(1, len(rows) // BATCH)
    for _epoch in range(EPOCHS):
        order = [np.concatenate([rng.choice(pos, N_POS), rng.choice(neg, N_NEG)]) for _ in range(steps)]
        net.train()
        for step in rng.permutation(len(order)):
            idx = order[int(step)]
            xb = torch.from_numpy(x[idx]).to(DEVICE)
            yb = torch.from_numpy(row_label[idx].astype(np.float32)).to(DEVICE).unsqueeze(1)
            logit = net(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb, reduction="none")
            loss = 2.0 * loss.sum() / max(loss.numel(), 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return net


def score(net, x):
    out = np.empty(x.shape[0], dtype=np.float32)
    net.eval()
    with torch.inference_mode():
        for s in range(0, x.shape[0], 4096):
            e = min(s + 4096, x.shape[0])
            out[s:e] = net(torch.from_numpy(x[s:e]).to(DEVICE)).squeeze(1).float().cpu().numpy()
    return out


def episode_eval(data, row_logits):
    ep = np.asarray(data["episode_id"], dtype=object)
    uniq, inv = np.unique(ep, return_inverse=True)
    ep_sc = np.full(uniq.shape[0], -np.inf)
    np.maximum.at(ep_sc, inv, row_logits.astype(np.float64))
    lab = np.zeros(uniq.shape[0], dtype=bool)
    for e, l in zip(ep, data["episode_label"]):
        lab[np.searchsorted(uniq, e)] = bool(l)
    grp = np.asarray([data["group_id"][np.searchsorted(ep, e)] for e in uniq], dtype=object)
    thr = fpr_threshold(lab, ep_sc)
    cm = confusions(lab, ep_sc, thr)
    return {"episode_id": uniq, "score": ep_sc, "labels": lab, "groups": grp,
            "threshold": thr, **cm, "auc": auc(lab, ep_sc),
            "fpr": cm["fp"] / max(cm["fp"] + cm["tn"], 1),
            "fnr": cm["fn"] / max(cm["tp"] + cm["fn"], 1)}


def main():
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tr = load_split("train")
    dv = load_split("dev")
    Xtr = matrix(tr, 0.0, 1.0)  # temp to get raw; recompute properly below
    # TRAIN-only statistics
    sel = tr["in_loss"]
    raw_tr = ((np.concatenate([np.asarray(tr["final"]), np.asarray(tr["early"])], axis=1).astype(np.uint32)) << 16).view(np.float32)
    mu = raw_tr[sel].mean(axis=0).astype(np.float32)
    sigma = raw_tr[sel].std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, 1e-6)
    del Xtr, raw_tr
    Xtr = (matrix(tr, mu, sigma)).astype(np.float32)
    Xdv = (matrix(dv, mu, sigma)).astype(np.float32)

    folds = group_folds(tr["group_id"].tolist())
    results = {}
    for seed in TRAIN_SEEDS:
        for tag, fitf in (("ho", (1, 2, 3)), ("full", (0, 1, 2, 3))):
            p = MODELS / tag / f"G_s{seed}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.is_file():
                net = train(seed, fitf, Xtr, tr["row_label"], tr["in_loss"], folds)
                torch.save({"state": net.state_dict()}, p)
            else:
                print("resume skip", seed, tag, flush=True)
            results[f"{seed}:{tag}"] = {
                "train_rows": int((tr["in_loss"] & (tr["row_label"] >= 0) & np.isin(folds, fitf)).sum()),
            }
            print("fitted", seed, tag, results[f"{seed}:{tag}"], flush=True)

    # heldout TRAIN fold0 (never fitted for ho)
    ep_ids = np.asarray(dv["episode_id"], dtype=object)
    d_uniq, d_inv = np.unique(ep_ids, return_inverse=True)

    def epmax(s):
        out = np.full(d_uniq.size, -np.inf)
        np.maximum.at(out, d_inv, np.asarray(s, dtype=np.float64))
        return out

    dev_out = {}
    for tag in ("ho", "full"):
        per = []
        for seed in TRAIN_SEEDS:
            st = torch.load(MODELS / tag / f"G_s{seed}.pt", weights_only=False)
            net = torch.nn.Sequential(
                torch.nn.Linear(DIM, HIDDEN_W), torch.nn.GELU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(HIDDEN_W, 1),
            ).to(DEVICE)
            net.load_state_dict(st["state"])
            per.append(score(net, Xdv))
        ens = np.mean(np.stack(per), axis=0)
        m = episode_eval(dv, ens)
        m["per_seed_auc"] = [auc(m["labels"], epmax(s)) for s in per]
        m["score"] = m["score"].tolist()
        m["labels"] = m["labels"].tolist()
        m["episode_id"] = m["episode_id"].tolist()
        m["groups"] = m["groups"].tolist()
        dev_out[f"G:{tag}"] = m
        print("DEV", tag, {k: m[k] for k in ("tp", "fp", "fn", "tn", "auc", "threshold")}, flush=True)

    # heldout TRAIN fold0 metrics
    held = np.isin(folds, [0])
    ho = {}
    for tag in ("ho",):
        per = []
        for seed in TRAIN_SEEDS:
            st = torch.load(MODELS / tag / f"G_s{seed}.pt", weights_only=False)
            net = torch.nn.Sequential(
                torch.nn.Linear(DIM, HIDDEN_W), torch.nn.GELU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(HIDDEN_W, 1),
            ).to(DEVICE)
            net.load_state_dict(st["state"])
            per.append(score(net, Xtr[held]))
        ens = np.mean(np.stack(per), axis=0)
        n_rows = tr["row_index"].shape[0]
        sub = {k: (v[held] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_rows else v) for k, v in tr.items()}
        m = episode_eval(sub, ens)
        ho["G:ho"] = {k: m[k] for k in ("episodes" if "episodes" in m else "auc", "tp", "fp", "fn", "tn", "auc", "threshold") if k in m}
        ho["G:ho"] = {k: m[k] for k in ("tp", "fp", "fn", "tn", "auc", "threshold")}
        ho["G:ho"]["episodes"] = int(len(np.unique(sub["episode_id"])))
        ho["G:ho"]["positives"] = int(m["labels"].sum())

    # bootstrap G:full vs v4 F:full and vs accepted baseline on DEV
    v4 = json.loads((ART / "eval" / "report.json").read_text())
    g = dev_out["G:full"]
    y = np.asarray(g["labels"], dtype=bool)
    grp = np.asarray(g["groups"], dtype=object)
    base_npz = np.load("/dev/shm/sdsafe-gpu-v4/upstream/sdsafe-lowcost-authorization-v2/artifacts/baseline/baseline_scores_v2.npz", allow_pickle=True)
    base_map = {str(e): float(s) for e, s in zip(base_npz["episode_id"], base_npz["ensemble_score"])}
    base_vec = np.asarray([base_map[str(e)] for e in g["episode_id"]], dtype=np.float64)
    thr_b = fpr_threshold(y, base_vec)
    bootstrap = {"G_vs_F": _bootstrap(y, np.asarray(v4["dev"]["variants"]["F:full"]["score"]), np.asarray(g["score"]), grp,
                                     float(v4["dev"]["variants"]["F:full"]["threshold"]), float(g["threshold"])),
                 "G_vs_accepted_base": _bootstrap(y, base_vec, np.asarray(g["score"]), grp, float(thr_b), float(g["threshold"]))}

    report = {"arm": "G = standardized [final|block4|block8], TRAIN-only mu/sigma",
              "protocol": "pre-declared before running; TRAIN heldout fold0 selection; DEV exploratory",
              "mu_sigma_fit_rows": int(sel.sum()), "heldout": ho, "dev": dev_out, "bootstrap": bootstrap,
              "runs": results}
    atomic_write_json(OUT, report)
    print(json.dumps({"dev": {k: {x: v[x] for x in ("tp", "fp", "fn", "tn", "auc", "threshold")} for k, v in dev_out.items()},
                      "heldout": ho, "bootstrap": {k: {"d_fn": v["d_fn"], "d_fp": v["d_fp"]} for k, v in bootstrap.items()}}, indent=2))


if __name__ == "__main__":
    main()
