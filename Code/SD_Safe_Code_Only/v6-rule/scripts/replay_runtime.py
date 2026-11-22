"""End-to-end replay of S2/FUSION_S2 through runtime_fast (no threshold change)."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v6 import baseline, data, evaluate as ev, metrics, runtime_fast  # noqa: E402

ART = runtime_fast.ART


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "runtime-r2" / "replay_s2_fusion.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)
    views = []
    for name in ("dev_common.jsonl", "dev_common_fixup.jsonl"):
        views += [json.loads(l) for l in (ART / "views" / name).read_text().splitlines()]
    by_ep = collections.defaultdict(list)
    for v in views:
        by_ep[v["episode_id"]].append(v)
    for vs in by_ep.values():
        vs.sort(key=lambda r: (int(r["round_id"]), int(r["token_end"]), int(r["row_index"])))

    scorer = runtime_fast.FrozenScorer(arm="S2", device=args.device)
    tc = runtime_fast.TextCache()
    bp = runtime_fast.BlockProcessor(tc)
    row_logits: dict[int, np.ndarray] = {}
    row_episode: dict[int, str] = {}
    ep_rows = collections.defaultdict(list)
    n = 0
    for ep in sorted(by_ep):
        rec0 = data.load_prepared("dev", ep)
        st = runtime_fast.EpisodeStream(
            episode_id=ep, ids=rec0["ids_flat"],
            offsets=[int(x) for x in rec0["ids_offsets"].tolist()],
            round_ids=[int(x) for x in rec0["round_id"].tolist()],
            prompt_len=[int(x) for x in rec0["prompt_length"].tolist()],
            seq_len=[int(x) for x in rec0["seq_len"].tolist()],
            tok=tok, tc=tc, bp=bp)
        for v in by_ep[ep]:
            meta = {"row_index": v["row_index"], "event_id": v["event_id"],
                    "scope": v["scope"], "group_id": v["group_id"],
                    "row_label": v["row_label"], "episode_label": v["episode_label"]}
            got = st.event(rec_meta=meta, t_start=v["token_start"], t_end=v["token_end"],
                           event_type=v["event_type"], round_id=v["round_id"],
                           token_start=v["token_start"], token_end=v["token_end"], fast=True)
            logits = scorer.row_logits(got["feats"])
            row_logits[int(v["row_index"])] = logits
            row_episode[int(v["row_index"])] = ep
            ep_rows[ep].append((int(v["row_index"]), logits))
            n += 1
    print("scored rows", n, flush=True)

    saved = np.load(ART / "dev_scores_S2_v3.npz", allow_pickle=True)
    saved_rows = [int(x) for x in saved["row_index"].tolist()]
    saved_scores = np.asarray(saved["row_scores"], dtype=np.float32)
    maxdiff = 0.0
    diffs = []
    for j, r in enumerate(saved_rows):
        got = row_logits[r]
        d = np.abs(got.astype(np.float64) - saved_scores[:, j].astype(np.float64)).max()
        maxdiff = max(maxdiff, float(d))
        diffs.append(float(d))
    diffs = np.asarray(diffs)
    print("row maxdiff", maxdiff, "p99", float(np.percentile(diffs, 99)), flush=True)

    # per-seed episode max from stream
    base = ev.load_dev_base()
    ids = [str(x) for x in base["episode_id"].tolist()]
    labels = np.asarray(base["labels"], dtype=np.int64)
    per_seed = np.full((3, len(ids)), -np.inf, dtype=np.float32)
    idx = {e: i for i, e in enumerate(ids)}
    for ep, rows in ep_rows.items():
        i = idx[ep]
        for s in range(3):
            per_seed[s, i] = max(float(np.float32(x[1][s])) for x in rows)
    ens = per_seed.mean(axis=0, dtype=np.float32)
    zep = np.load(ART / "dev_episode_S2_v3.npz", allow_pickle=True)
    saved_ps = np.asarray(zep["per_seed"], dtype=np.float32)
    saved_ens = np.asarray(zep["ensemble"], dtype=np.float32)
    print("per_seed maxdiff", float(np.abs(per_seed - saved_ps).max()),
          "ensemble maxdiff", float(np.abs(ens - saved_ens).max()), flush=True)

    # independent reference forward from the frozen scored cache
    from sdsafe_v6 import train
    scaler = np.load(ART / "scaler_v6_v3.npz", allow_pickle=True)
    dv = train.Tensors(np.load(ART / "cache_dev_v3.npz", allow_pickle=True),
                       scaler["mean"], scaler["std"], args.device)
    import torch
    ref_net = runtime_fast.TriStudent([ART / f"ckpt_S2_v3_seed{s}.pt" for s in (0, 1, 2)],
                                      device=args.device)
    # batch reference through the same merged net (batch 1024) to separate model-batch rounding
    ref_rows = {}
    rows_all = np.arange(dv.n)
    for start in range(0, rows_all.size, 1024):
        chunk = rows_all[start:start + 1024]
        feats = {
            "bags": {
                "cur": np.concatenate([dv.idx["cur"][i] for i in chunk]),
                "vis": np.concatenate([dv.idx["vis"][i] for i in chunk]),
                "user": np.concatenate([dv.idx["user"][i] for i in chunk]),
                "src": None,
            },
            "dense": (dv.dense[chunk] * scaler["std"] + scaler["mean"]),
            "_chunk": chunk,
        }
        # build per-event ragged src via tensors.batch + a tiny adapter is easier: loop rows
        # (reference cache batch uses the same maths; here only a subset is checked b1)
        del ref_rows  # placeholder
        break
    # b1 merged-net reference on the scored cache for 300 rows
    sample_rows = np.arange(0, min(300, dv.n))
    b1_maxdiff = 0.0
    for i in sample_rows:
        feats = {"bags": {
            "cur": dv.idx["cur"][i],
            "vis": dv.idx["vis"][i],
            "user": dv.idx["user"][i],
            "src": dv.idx["src"][i],
        }, "dense": (dv.dense[i].astype(np.float32) * scaler["std"] + scaler["mean"])}
        got = ref_net.forward(feats)
        r = int(dv.row_index[i])
        if r in row_logits:
            b1_maxdiff = max(b1_maxdiff, float(np.abs(got - row_logits[r]).max()))
    print("merged-net b1 cache-feature maxdiff (300 rows)", b1_maxdiff, flush=True)

    # formal metrics with frozen thresholds / mapping
    thr_s2 = metrics.threshold_at_fpr(ens.astype(np.float64), labels, 0.05)
    conf_s2 = metrics.confusion(ens.astype(np.float64), labels, thr_s2)
    fused = scorer.fused(np.asarray(base["ensemble_f32"], dtype=np.float64), ens)
    thr_f = metrics.threshold_at_fpr(fused, labels, 0.05)
    conf_f = metrics.confusion(fused, labels, thr_f)
    # reuse the frozen saved thresholds and compare predictions directly
    frozen_thr_s2 = float(zep["per_seed"].dtype == np.float32) and None
    res = json.loads((ART / "results_v3.json").read_text())
    s2_pred_ref = np.asarray(res["predictions"]["S2_pred"], dtype=np.int64)
    s2_pred_got = (ens.astype(np.float64) >= res["predictions"]["S2_threshold"]).astype(np.int64)
    fus_pred_ref = (np.asarray(res["fusion"]["S2"]["fn_ids"]).size * 0)  # placeholder
    ref_fused = 0.5 * ((np.asarray(base["ensemble_f32"], dtype=np.float64) - scorer.routing.base_mean)
                       / scorer.routing.base_std) + 0.5 * ((saved_ens.astype(np.float64)
                                                            - scorer.routing.stud_mean)
                                                           / scorer.routing.stud_std)
    ref_thr_f = res["fusion"]["S2"]["threshold"]
    pred_ref_f = (ref_fused >= ref_thr_f)
    pred_got_f = (fused >= ref_thr_f)
    frozen_s2 = metrics.confusion(ens.astype(np.float64), labels, res["predictions"]["S2_threshold"])
    frozen_f = metrics.confusion(fused, labels, float(ref_thr_f))
    changed = np.flatnonzero(pred_got_f != pred_ref_f)
    boundary = []
    for j in changed:
        boundary.append({"episode_id": ids[int(j)], "label": int(labels[int(j)]),
                         "stream_fused": float(fused[int(j)]), "saved_fused": float(ref_fused[int(j)]),
                         "frozen_threshold": float(ref_thr_f),
                         "delta_stream_minus_threshold": float(fused[int(j)] - ref_thr_f)})
    np.savez(ART / "runtime-r2" / "replay_stream_scores.npz",
             episode_id=np.asarray(ids, dtype=object), label=labels,
             per_seed=per_seed, ensemble=ens, fused=fused)
    out = {
        "rows_scored": n,
        "frozen_threshold_confusion": {"s2": frozen_s2, "fusion": frozen_f,
                                       "s2_threshold": res["predictions"]["S2_threshold"],
                                       "fusion_threshold": float(ref_thr_f)},
        "fusion_boundary_changes": boundary,
        "row_logit_maxdiff_vs_saved": maxdiff,
        "row_logit_diff_p99": float(np.percentile(diffs, 99)),
        "per_seed_episode_maxdiff": float(np.abs(per_seed - saved_ps).max()),
        "ensemble_episode_maxdiff": float(np.abs(ens - saved_ens).max()),
        "s2": {"fn": conf_s2["fn"], "fp": conf_s2["fp"], "auc": metrics.auc(ens.astype(np.float64), labels),
               "threshold_refit": float(thr_s2),
               "threshold_saved": res["predictions"]["S2_threshold"],
               "prediction_changes_vs_saved": int((s2_pred_got != s2_pred_ref).sum())},
        "fusion": {"fn": conf_f["fn"], "fp": conf_f["fp"], "auc": metrics.auc(fused, labels),
                   "threshold_refit": float(thr_f),
                   "threshold_saved": float(ref_thr_f),
                   "prediction_changes_vs_saved": int((pred_got_f != pred_ref_f).sum())},
        "frozen_preds_saved": {"s2_fn": int(((labels == 1) & ~s2_pred_ref.astype(bool)).sum()),
                               "s2_fp": int(((labels == 0) & s2_pred_ref.astype(bool)).sum()),
                               "fusion_fn": int(((labels == 1) & ~pred_ref_f).sum()),
                               "fusion_fp": int(((labels == 0) & pred_ref_f).sum())},
        "merged_net_b1_cache_feature_maxdiff_300": b1_maxdiff,
    }
    print(json.dumps(out, indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
