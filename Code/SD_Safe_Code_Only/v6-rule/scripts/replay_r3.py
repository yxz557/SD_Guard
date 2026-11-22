"""corrected-r3 causal stream replay + one-shot deployment fusion-tau calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v6 import data, metrics, runtime_fast  # noqa: E402

ART = runtime_fast.ART
OUT = ART / "corrected-r3"


def sha_bytes(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def fusion_scale(arm: str) -> dict:
    base = np.load(ART / "train_base_episodes.npz", allow_pickle=True)
    be = np.asarray(base["base"], dtype=np.float32)
    fin = np.isfinite(be)
    oof = np.load(OUT / f"train_oof_{arm}.npz", allow_pickle=True)
    rs = np.asarray(oof["row_scores"], dtype=np.float32)
    rep = np.asarray([str(x) for x in oof["episode_id"].tolist()])
    ok = np.isfinite(rs)
    eps = sorted(set(rep[ok].tolist()))
    idx = {e: i for i, e in enumerate(eps)}
    ep = np.full(len(eps), -np.inf, dtype=np.float32)
    np.maximum.at(ep, np.asarray([idx[e] for e in rep[ok]], dtype=np.int64), rs[ok])
    fin2 = np.isfinite(ep)
    return {"base_mean_TRAIN": float(be[fin].mean(dtype=np.float64)),
            "base_std_TRAIN": float(be[fin].std(dtype=np.float64)),
            "student_mean_TRAIN_OOF": float(ep[fin2].mean(dtype=np.float64)),
            "student_std_TRAIN_OOF": float(ep[fin2].std(dtype=np.float64)),
            "base_episodes": int(fin.sum()), "oof_episodes": len(eps),
            "oof_rows": int(ok.sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="S2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)
    base = np.load(ART / "baseline_dev.npz", allow_pickle=True)
    ids = [str(x) for x in base["episode_id"].tolist()]
    labels = np.asarray(base["labels"], dtype=np.int64)
    base_ens = np.asarray(base["ensemble_f32"], dtype=np.float32).astype(np.float64)
    ep_pos = {e: i for i, e in enumerate(ids)}

    views = [json.loads(l) for l in (OUT / "views_dev.jsonl").read_text().splitlines() if l.strip()]
    by_ep: dict[str, list[dict]] = {}
    for v in views:
        by_ep.setdefault(v["episode_id"], []).append(v)
    for vs in by_ep.values():
        vs.sort(key=lambda r: (int(r["round_id"]), int(r["token_end"]), int(r["row_index"])))

    has_oof = (OUT / f"train_oof_{args.arm}.npz").is_file()
    scale = fusion_scale(args.arm) if has_oof else None
    fusion_payload = {"formula": "0.5*((base-base_mean)/base_std)+0.5*((student-stud_mean)/stud_std)",
                      "weights": [0.5, 0.5], "primary_arm": args.arm,
                      "student_oof_folds_used": [0, 1, 2, 3],
                      "fold0_note": "no epoch/arm selection in r3; all group-disjoint OOF folds usable",
                      "dtype": "base/student float32 values, float64 fusion arithmetic", **(scale or {})}
    fusion_path = OUT / f"fusion_frozen_corrected_r3_{args.arm}.json"
    fusion_path.write_text(json.dumps(fusion_payload, indent=1) + "\n")

    if has_oof:
        score_fusion_path = fusion_path
    else:
        score_fusion_path = OUT / f"_dummy_fusion_{args.arm}.json"
        score_fusion_path.write_text(json.dumps({
            "base_mean_TRAIN": 0.0, "base_std_TRAIN": 1.0,
            "student_mean_TRAIN_OOF": 0.0, "student_std_TRAIN_OOF": 1.0}))
    scorer = runtime_fast.FrozenScorer(arm=args.arm, device=args.device, artdir=OUT,
                                       scaler_path=OUT / "scaler_corrected_r3.npz",
                                       fusion_path=score_fusion_path)
    tc = runtime_fast.TextCache()
    bp = runtime_fast.BlockProcessor(tc)
    stud = np.full((3, len(ids)), -np.inf, dtype=np.float32)
    n_scored = 0
    for ep in sorted(by_ep):
        rec0 = data.load_prepared("dev", ep)
        st = runtime_fast.EpisodeStream(
            episode_id=ep, ids=rec0["ids_flat"],
            offsets=[int(x) for x in rec0["ids_offsets"].tolist()],
            round_ids=[int(x) for x in rec0["round_id"].tolist()],
            prompt_len=[int(x) for x in rec0["prompt_length"].tolist()],
            seq_len=[int(x) for x in rec0["seq_len"].tolist()],
            tok=tok, tc=tc, bp=bp)
        j = ep_pos[ep]
        for v in by_ep[ep]:
            meta = {"row_index": v["row_index"], "event_id": v["event_id"],
                    "scope": v["scope"], "group_id": v["group_id"],
                    "row_label": v["row_label"], "episode_label": v["episode_label"]}
            got = st.event(rec_meta=meta, t_start=v["token_start"], t_end=v["token_end"],
                           event_type=v["event_type"], round_id=v["round_id"],
                           token_start=v["token_start"], token_end=v["token_end"],
                           fast=True, call_index=v["current"].get("call_index"))
            if got["rec"]["no_new_emission"] or got["rec"]["unknown_adapter"]:
                continue
            logits = scorer.row_logits(got["feats"]).astype(np.float32)
            np.maximum(stud[:, j], logits, out=stud[:, j])
            n_scored += 1
    ens = stud.mean(axis=0, dtype=np.float32)
    saved = np.load(OUT / f"dev_episode_{args.arm}.npz", allow_pickle=True)
    saved_ens = np.asarray(saved["ensemble"], dtype=np.float32)
    s_thr = metrics.threshold_at_fpr(ens.astype(np.float64), labels, 0.05)
    s_conf = metrics.confusion(ens.astype(np.float64), labels, s_thr)
    if has_oof:
        fused = (0.5 * ((base_ens - scale["base_mean_TRAIN"]) / scale["base_std_TRAIN"])
                 + 0.5 * ((ens.astype(np.float64) - scale["student_mean_TRAIN_OOF"])
                          / scale["student_std_TRAIN_OOF"]))
        tau = float(metrics.threshold_at_fpr(fused, labels, 0.05))
        f_conf = metrics.confusion(fused, labels, tau)
    else:
        fused = np.full(len(ids), np.nan, dtype=np.float64)
        tau = float("nan")
        f_conf = {"fn": None, "fp": None}
    payload = {
        "arm": args.arm, "events_scored": n_scored, "episodes": len(ids),
        "student_episode_sha256": sha_bytes(ens),
        "student_ensemble_maxdiff_vs_saved": float(np.abs(ens - saved_ens).max()),
        "student_threshold": s_thr, "student_threshold_hex": float(s_thr).hex(),
        "student_fn": s_conf["fn"], "student_fp": s_conf["fp"],
        "student_auc": metrics.auc(ens.astype(np.float64), labels),
        "fused_sha256": sha_bytes(fused) if has_oof else None, "fused_dtype": "float64",
        "fusion_threshold": tau, "fusion_threshold_hex": tau.hex(),
        "fusion_fn": f_conf["fn"], "fusion_fp": f_conf["fp"],
        "fusion_auc": metrics.auc(fused, labels) if has_oof else None,
        "fusion_margin_min_abs": float(np.min(np.abs(fused - tau))) if has_oof else None,
        "calibration": "one-shot own full-sample 5%-FPR on DEV, exact float64, frozen",
        "fusion_json_sha256": hashlib.sha256(fusion_path.read_bytes()).hexdigest(),
    }
    if has_oof:
        fusion_payload.update({"threshold": tau, "threshold_hex": tau.hex(),
                               "fused_sha256": sha_bytes(fused), "fn": f_conf["fn"],
                               "fp": f_conf["fp"], "auc": metrics.auc(fused, labels),
                               "student_ensemble_sha256": sha_bytes(ens),
                               "calibrated_on": "DEV full sample, once, deployment batch-1 stream"})
        fusion_path.write_text(json.dumps(fusion_payload, indent=1) + "\n")
        payload["fusion_json_sha256"] = hashlib.sha256(fusion_path.read_bytes()).hexdigest()
    else:
        payload["fusion_json_sha256"] = None
        try:
            fusion_path.unlink()
        except FileNotFoundError:
            pass
        try:
            score_fusion_path.unlink()
        except FileNotFoundError:
            pass
    payload_npz = {"episode_id": np.asarray(ids, dtype=object), "per_seed": stud,
                   "ensemble": ens, "labels": labels}
    if has_oof:
        payload_npz.update({"fused": fused, "threshold": np.float64(tau)})
    np.savez(OUT / f"replay_{args.arm}.npz", **payload_npz)
    out = Path(args.out or (OUT / f"replay_{args.arm}.json"))
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
