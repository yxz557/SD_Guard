"""End-to-end driver: caches, aux labels, fold-0 selection, final fits, DEV row scores."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import aux_data, feats, metrics, model, train

ART = train.ART


def aux_labels_from_cache(path: Path) -> tuple[dict[int, dict[str, int]], dict[str, object]]:
    """Two-pass collation: keep a field only when both passes agree (else mask it).

    ``unclear`` answers are kept as their own class, not silently dropped.  Disagreement between
    the two independent passes is counted and reported; it is never called human gold.
    """
    best: dict[str, dict] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("ok"):
            continue
        parsed = teacher.parse_label(item.get("content") or "")
        if not parsed.get("parse_ok"):
            continue
        best[item["digest"]] = {"row": int(item["row_index"]), "pass": item["pass"], "parsed": parsed}
    by_row: dict[int, dict[str, dict]] = {}
    for item in best.values():
        by_row.setdefault(item["row"], {})[item["pass"]] = item["parsed"]
    labels: dict[int, dict[str, int]] = {}
    stats = {"rows_seen": len(by_row), "both": 0, "single": 0, "field_agree": {}, "field_n": {}}
    for row, passes in by_row.items():
        a, b = passes.get("A"), passes.get("B")
        rec: dict[str, int] = {}
        if a and b:
            stats["both"] += 1
            pairs = (("action", "action_authorization", train.ACTION_MAP),
                     ("rel", "task_relation", train.REL_MAP),
                     ("align", "external_directive_alignment", train.ALIGN_MAP))
            for name, field, mapping in pairs:
                stats["field_n"][name] = stats["field_n"].get(name, 0) + 1
                if a.get(field) == b.get(field):
                    stats["field_agree"][name] = stats["field_agree"].get(name, 0) + 1
                    rec[name] = mapping[a[field]] if a.get(field) in mapping else -1
                else:
                    rec[name] = -1     # masked: the two passes disagree
            a_args = {x["path"]: x for x in a.get("arguments") or []}
            b_args = {x["path"]: x for x in b.get("arguments") or []}
            arg_votes = []
            for path in set(a_args) & set(b_args):
                av = a_args[path].get("authorization")
                bv = b_args[path].get("authorization")
                if av == bv and av in ("authorized", "unauthorized"):
                    arg_votes.append(1 if av == "unauthorized" else 0)
                elif av != bv:
                    arg_votes.append(-1)
            if arg_votes and all(v >= 0 for v in arg_votes):
                rec["arg"] = int(any(v == 1 for v in arg_votes))
            else:
                rec["arg"] = -1
        else:
            stats["single"] += 1
            one = a or b
            rec = {
                "action": train.ACTION_MAP.get(one.get("action_authorization"), -1),
                "rel": train.REL_MAP.get(one.get("task_relation"), -1),
                "align": train.ALIGN_MAP.get(one.get("external_directive_alignment"), -1),
                "arg": -1,
            }
        labels[row] = rec
    stats["rows_labelled"] = len(labels)
    stats["field_agree_rate"] = {k: round(stats["field_agree"][k] / stats["field_n"][k], 4)
                                 for k in stats["field_agree"]}
    return labels, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="S0,S1,S2,S3")
    ap.add_argument("--max-epochs", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default=str(ART / "study_arms.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="v3")
    args = ap.parse_args(argv)

    t0 = time.time()
    dev = args.device if torch_available() else "cpu"
    report: dict = {"device": dev, "arms": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ---- feature caches (v3: provenance-fixed views, view_version v6param-view-3)
    train_cache_path = ART / "cache_train_v3.npz"
    if train_cache_path.is_file():
        cache = feats.load_cache(train_cache_path)
    else:
        records = feats.load_views(train.TRAIN_VIEWS)
        feats.build_cache(records, train_cache_path)
        cache = feats.load_cache(train_cache_path)
    cache = feats.attach_main_labels(cache, train.POOL)
    dev_cache_path = ART / "cache_dev_v3.npz"
    if dev_cache_path.is_file():
        dcache = feats.load_cache(dev_cache_path)
    else:
        feats.build_cache(feats.load_views(train.DEV_VIEWS), dev_cache_path)
        dcache = feats.load_cache(dev_cache_path)

    # ---- standardisation from TRAIN in-pool rows only
    mask = cache["in_pool"]
    mean = cache["dense"][mask].mean(axis=0)
    std = cache["dense"][mask].std(axis=0) + 1e-6
    np.savez(ART / "scaler_v6_v3.npz", mean=mean, std=std)
    report["scaler_rows"] = int(mask.sum())

    tr = train.Tensors(cache, mean, std, dev)
    dv = train.Tensors(dcache, mean, std, dev)
    print(f"train rows {tr.n} in_pool {int(tr.in_pool.sum())} pos {int((tr.main_label==1).sum())}", flush=True)
    print(f"dev rows {dv.n}", flush=True)

    aux_labels, aux_manifest, _aux_per_arg = aux_data.build(write=False)
    aux_stats = aux_manifest["stats"]
    report["aux"] = aux_stats
    aux = train.aux_arrays(aux_labels, tr.row_index)
    for name in aux:
        aux[name] = aux[name].copy()
    report["aux"]["trainable_rows"] = {k: int((v >= 0).sum()) for k, v in aux.items()}

    # ---- group-disjoint folds
    fold = np.asarray([train.fold_of(g) for g in tr.group_id], dtype=np.int64)
    report["fold_sizes"] = {int(f): int((fold == f).sum()) for f in range(train.DEFAULTS["folds"])}
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
    # verify group disjointness across folds
    seen: dict[str, int] = {}
    for g, f in zip(ep_group, ep_fold.tolist()):
        if g in seen and seen[g] != f:
            raise RuntimeError(f"group {g} spans folds")
        seen[g] = f
    report["episode_folds"] = {int(f): int((ep_fold == f).sum()) for f in range(train.DEFAULTS["folds"])}
    report["episode_positives_per_fold"] = {int(f): int(ep_label[ep_fold == f].sum())
                                            for f in range(train.DEFAULTS["folds"])}

    lam_of = {"S0": 0.0, "S1": 0.3, "S2": 1.0, "S3": 1.0}
    for arm in args.arms.split(","):
        arm = arm.strip()
        lam = lam_of[arm]
        shuffle_aux = arm == "S3"
        arm_report: dict = {"lambda": lam, "shuffle_aux": shuffle_aux, "epoch_scan": {}, "fold0": {}}
        rows_fit = np.flatnonzero(fold != 0)
        rows_ho = np.flatnonzero(fold == 0)
        aux_fit = {k: v.copy() for k, v in aux.items()}
        best = None
        net = None
        for epochs in range(1, args.max_epochs + 1):
            # one continuous run, evaluated after each epoch (the scan is a prefix of it)
            net = train.train_one(tr, rows_fit=rows_fit, aux=aux_fit, epochs=1, lam=lam,
                                  seed=100, device=dev, shuffle_aux=shuffle_aux, net=net,
                                  rng_offset=epochs)
            sc = train.score_rows(net, tr, rows_ho, dev)
            ep_sc = train.episode_scores_for(tr, rows_ho, sc, episodes)
            y = ep_label
            held = ep_fold == 0
            auc = metrics.auc(ep_sc[held], y[held])
            thr = metrics.threshold_at_fpr(ep_sc[held], y[held], 0.05)
            conf = metrics.confusion(ep_sc[held], y[held], thr)
            arm_report["epoch_scan"][str(epochs)] = {"auc": auc, "fn": conf["fn"], "fp": conf["fp"],
                                                     "threshold": thr}
            key = (auc, -conf["fn"])
            if best is None or key > best[0]:
                best = (key, epochs, auc, conf)
        arm_report["selected_epoch"] = best[1]
        arm_report["selected_fold0"] = {"auc": best[2], **{k: best[3][k] for k in ("fn", "fp", "tp", "tn")}}
        print(f"[{arm}] fold0 epoch scan {arm_report['epoch_scan']}", flush=True)
        print(f"[{arm}] selected epochs={best[1]} auc={best[2]:.6f} fn={best[3]['fn']} fp={best[3]['fp']}", flush=True)

        # group-disjoint OOF row scores (one model per held-out fold) for TRAIN-only stacking
        oof = np.full(tr.n, np.nan, dtype=np.float32)
        for f in range(train.DEFAULTS["folds"]):
            net_f = train.train_one(tr, rows_fit=np.flatnonzero(fold != f),
                                    aux={k: v.copy() for k, v in aux.items()},
                                    epochs=best[1], lam=lam, seed=1000 + f, device=dev,
                                    shuffle_aux=shuffle_aux)
            rows_f = np.flatnonzero(fold == f)
            oof[rows_f] = train.score_rows(net_f, tr, rows_f, dev)
        np.savez(ART / f"train_oof_{arm}_{args.tag}.npz", row_scores=oof, row_index=tr.row_index,
                 episode_id=tr.episode_id, fold=fold, main_label=tr.main_label)
        arm_report["oof_scored_rows"] = int(np.isfinite(oof).sum())

        # final fits on ALL TRAIN with the selected epoch count, 3 seeds
        rows_all = np.arange(tr.n)
        seed_scores = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            net = train.train_one(tr, rows_fit=rows_all, aux={k: v.copy() for k, v in aux.items()},
                                  epochs=best[1], lam=lam, seed=seed, device=dev,
                                  shuffle_aux=shuffle_aux)
            sc = train.score_rows(net, dv, np.arange(dv.n), dev)
            seed_scores.append(sc)
            torch.save({"state_dict": net.state_dict(), "config": vars(net.cfg),
                        "arm": arm, "seed": seed, "epochs": best[1], "lam": lam},
                       ART / f"ckpt_{arm}_{args.tag}_seed{seed}.pt")
        seed_scores = np.stack(seed_scores, axis=0)
        np.savez(ART / f"dev_scores_{arm}_{args.tag}.npz", row_scores=seed_scores,
                 row_index=dv.row_index, episode_id=dv.episode_id)
        # per-seed episode max then mean (float32), matching the baseline composition
        dev_eps = sorted(set(dv.episode_id.tolist()))
        per_seed_ep = np.stack([train.episode_scores_for(dv, np.arange(dv.n), seed_scores[s], dev_eps)
                                for s in range(seed_scores.shape[0])], axis=0)
        ens = per_seed_ep.mean(axis=0, dtype=np.float32)
        np.savez(ART / f"dev_episode_{arm}_{args.tag}.npz", episode_id=np.asarray(dev_eps, dtype=object),
                 per_seed=per_seed_ep, ensemble=ens)
        arm_report["seeds"] = int(seed_scores.shape[0])
        report["arms"][arm] = arm_report
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
    report["seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: report[k] for k in ("aux", "fold_sizes", "episode_folds", "seconds")}, indent=1))
    return 0


def torch_available() -> bool:
    return torch.cuda.is_available()


if __name__ == "__main__":
    raise SystemExit(main())
