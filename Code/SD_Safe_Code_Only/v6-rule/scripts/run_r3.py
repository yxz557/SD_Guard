"""corrected-r3 fixed-menu fits: S2 lambda1, matched S0 and S3, 5 epochs, 3 seeds.

Plus TRAIN-only outer-held-group OOF fits for S0/S2 (frozen fusion scale).  No DEV label is
read during fitting or scaling; DEV rows are only scored after the fits.  No epoch scan,
no hyper-parameter search.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v6 import aux_data, feats, metrics, model, train  # noqa: E402

OUT = train.ART / "corrected-r3"
EPOCHS = 5
ARMS = {"S0": 0.0, "S2": 1.0, "S3": 1.0}
OOF_ARMS = ("S0", "S2")


def load_aux_labels() -> dict[int, dict[str, int]]:
    path = OUT / "aux/aux_labels_corrected_r3.json"
    raw = json.loads(path.read_text())
    return {int(k): {kk: int(vv) for kk, vv in v.items()} for k, v in raw.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="S0,S2,S3")
    ap.add_argument("--out", default=str(OUT / "study_arms_corrected_r3.json"))
    args = ap.parse_args(argv)
    dev = args.device if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    report: dict = {"device": dev, "epochs": EPOCHS,
                    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "arms": {}}

    cache = feats.load_cache(OUT / "cache_train.npz")
    cache = feats.attach_main_labels(cache, train.POOL)
    dcache = feats.load_cache(OUT / "cache_dev.npz")
    mask = cache["in_pool"]
    mean = cache["dense"][mask].mean(axis=0)
    std = cache["dense"][mask].std(axis=0) + 1e-6
    np.savez(OUT / "scaler_corrected_r3.npz", mean=mean, std=std)
    report["scaler_rows"] = int(mask.sum())
    report["train_rows"] = int(cache["row_index"].shape[0])
    report["dev_rows"] = int(dcache["row_index"].shape[0])

    tr = train.Tensors(cache, mean, std, dev)
    dv = train.Tensors(dcache, mean, std, dev)
    print(f"train rows {tr.n} in_pool {int(tr.in_pool.sum())} pos {int((tr.main_label==1).sum())}",
          flush=True)
    print(f"dev rows {dv.n}", flush=True)

    labels = load_aux_labels()
    aux = train.aux_arrays(labels, tr.row_index)
    aux = {k: v.copy() for k, v in aux.items()}
    report["aux_trainable_rows"] = {k: int((v >= 0).sum()) for k, v in aux.items()}
    report["aux_labels_rows"] = len(labels)

    fold = np.asarray([train.fold_of(g) for g in tr.group_id], dtype=np.int64)
    report["fold_sizes"] = {int(f): int((fold == f).sum()) for f in range(4)}
    episodes = sorted(set(tr.episode_id.tolist()))
    ep_index = {e: i for i, e in enumerate(episodes)}
    ep_fold = np.zeros(len(episodes), dtype=np.int64)
    ep_label = np.zeros(len(episodes), dtype=np.int64)
    ep_group = ["" for _ in episodes]
    for i, e in enumerate(tr.episode_id):
        j = ep_index[e]
        ep_fold[j] = fold[i]
        ep_label[j] = int(tr.episode_label[i])
        ep_group[j] = str(tr.group_id[i])
    seen: dict[str, int] = {}
    for g, f in zip(ep_group, ep_fold.tolist()):
        if g in seen and seen[g] != f:
            raise RuntimeError(f"group {g} spans folds")
        seen[g] = f
    report["episode_folds"] = {int(f): int((ep_fold == f).sum()) for f in range(4)}
    report["episode_positives_per_fold"] = {int(f): int(ep_label[ep_fold == f].sum())
                                            for f in range(4)}
    report["groups"] = len(seen)

    dev_eps = sorted(set(dv.episode_id.tolist()))
    for arm in [a.strip() for a in args.arms.split(",")]:
        lam = ARMS[arm]
        shuffle_aux = arm == "S3"
        arm_rep: dict = {"lambda": lam, "shuffle_aux": shuffle_aux, "epochs": EPOCHS,
                         "seeds": [int(s) for s in args.seeds.split(",")]}
        # ---- OOF (TRAIN only) for the frozen fusion scale
        if arm in OOF_ARMS:
            oof = np.full(tr.n, np.nan, dtype=np.float32)
            for f in range(4):
                net = train.train_one(
                    tr, rows_fit=np.flatnonzero(fold != f),
                    aux={k: v.copy() for k, v in aux.items()},
                    epochs=EPOCHS, lam=lam, seed=1000 + f, device=dev,
                    shuffle_aux=shuffle_aux)
                rows_f = np.flatnonzero(fold == f)
                oof[rows_f] = train.score_rows(net, tr, rows_f, dev)
                torch.save({"state_dict": net.state_dict(), "config": vars(net.cfg),
                            "arm": arm, "fold": f, "epochs": EPOCHS, "lam": lam,
                            "role": "oof"}, OUT / f"ckpt_{arm}_oof_fold{f}.pt")
            np.savez(OUT / f"train_oof_{arm}.npz", row_scores=oof, row_index=tr.row_index,
                     episode_id=tr.episode_id, fold=fold, main_label=tr.main_label)
            arm_rep["oof_scored_rows"] = int(np.isfinite(oof).sum())
            print(f"[{arm}] OOF done ({arm_rep['oof_scored_rows']} rows)", flush=True)
        # ---- final 3 seeds on all TRAIN
        rows_all = np.arange(tr.n)
        seed_scores = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            net = train.train_one(tr, rows_fit=rows_all,
                                  aux={k: v.copy() for k, v in aux.items()},
                                  epochs=EPOCHS, lam=lam, seed=seed, device=dev,
                                  shuffle_aux=shuffle_aux)
            sc = train.score_rows(net, dv, np.arange(dv.n), dev)
            seed_scores.append(sc)
            torch.save({"state_dict": net.state_dict(), "config": vars(net.cfg),
                        "arm": arm, "seed": seed, "epochs": EPOCHS, "lam": lam,
                        "role": "formal"}, OUT / f"ckpt_{arm}_seed{seed}.pt")
            print(f"[{arm}] seed {seed} scored {sc.size} dev rows", flush=True)
        seed_scores = np.stack(seed_scores, axis=0).astype(np.float32)
        np.savez(OUT / f"dev_scores_{arm}.npz", row_scores=seed_scores,
                 row_index=dv.row_index, episode_id=dv.episode_id)
        per_seed_ep = np.stack([
            train.episode_scores_for(dv, np.arange(dv.n), seed_scores[s], dev_eps)
            for s in range(seed_scores.shape[0])], axis=0).astype(np.float32)
        ens = per_seed_ep.mean(axis=0, dtype=np.float32)
        np.savez(OUT / f"dev_episode_{arm}.npz",
                 episode_id=np.asarray(dev_eps, dtype=object),
                 per_seed=per_seed_ep, ensemble=ens)
        arm_rep.update({"n_seeds": int(seed_scores.shape[0]),
                        "dev_scored_rows": int(np.isfinite(seed_scores).all(axis=0).sum())})
        report["arms"][arm] = arm_rep
        (OUT / "study_arms_corrected_r3.json").write_text(json.dumps(report, indent=1) + "\n")
    report["seconds"] = round(time.time() - t0, 1)
    (OUT / "study_arms_corrected_r3.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("aux_trainable_rows", "fold_sizes", "seconds")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
