"""Full common-DEV parity for runtime-r4 against the frozen corrected-r3 reference.

For every one of the 25 003 CT events the r4 stream is run twice:

R  rawtext interface  (detector parses the full prompt itself)
T  typed interface    (source adapter supplies role-tagged blocks; adapter charged separately)

and compared layer by layer with the *frozen* corrected-r3 artifacts:

  views_dev.jsonl       roles, call binding, retrieval, value sources, flags, hashes
  cache_dev.npz         bag indices and dense features (bitwise)
  dev_scores_S2.npz     per-seed CT logits (float32)
  replay_S2.npz/json    per-seed episode max, float32 seed mean, float64 fusion, frozen tau

Reference features are used only as a parity oracle; they are never fed to the cost path.
No TEST material, no training, no threshold change.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v6 import data, evaluate as ev, runtime_fast as rf, runtime_r4 as r4  # noqa: E402

ART = rf.ART
OUT = ART / "corrected-r3"
VIEW_KEYS = ("episode_id", "row_index", "event_id", "event_type", "scope", "group_id",
             "round_id", "token_start", "token_end", "row_label", "episode_label",
             "user_request", "prior", "relevant_tool_blocks", "current", "empty_segment",
             "no_new_emission", "unknown_adapter", "block_fallback", "visible_chars",
             "value_sources", "prefix_chars", "counts", "has_reasoning_before",
             "prefix_sha256", "current_sha256")


def compare_view(got, ref):
    bad = []
    for k in VIEW_KEYS:
        if k not in ref:
            continue
        g = got.get(k)
        r = ref[k]
        if k == "current":
            for ck in r:
                if g.get(ck) != r[ck]:
                    bad.append(f"current.{ck}")
        elif g != r:
            bad.append(k)
    return bad


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--profile", type=int, default=0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)
    views = [json.loads(l) for l in (OUT / "views_dev.jsonl").read_text().splitlines() if l.strip()]
    ref_view = {int(v["row_index"]): v for v in views}
    _cz = np.load(OUT / "cache_dev.npz", allow_pickle=True)
    cache = {k: np.asarray(_cz[k]) for k in _cz.files}
    cache_pos = {int(r): i for i, r in enumerate(cache["row_index"].tolist())}
    src_counts = cache["src_counts"].astype(np.int64)
    src_bag_start = np.concatenate([[0], np.cumsum(src_counts)])
    _sz = np.load(OUT / "dev_scores_S2.npz", allow_pickle=True)
    sc = {k: np.asarray(_sz[k]) for k in _sz.files}
    score_pos = {int(r): i for i, r in enumerate(sc["row_index"].tolist())}
    _rz = np.load(OUT / "replay_S2.npz", allow_pickle=True)
    z = {k: np.asarray(_rz[k]) for k in _rz.files}
    ref_ids = [str(x) for x in z["episode_id"].tolist()]
    ep_pos = {e: i for i, e in enumerate(ref_ids)}
    s2_thr = float(json.loads((OUT / "replay_S2.json").read_text())["student_threshold"])
    fused_thr = float(json.loads((OUT / "fusion_frozen_corrected_r3_S2.json").read_text())["threshold"])
    base = ev.load_dev_base()
    ep_ids = [str(x) for x in base["episode_id"].tolist()]
    if args.episodes:
        ep_ids = ep_ids[: args.episodes]
    rows = data.load_rows("dev")
    by_ep = {}
    for r in rows.values():
        by_ep.setdefault(r["episode_id"], []).append(r)
    for vs in by_ep.values():
        vs.sort(key=lambda r: (int(r["round_id"]), int(r["token_end"]), int(r["row_index"])))
    scorer = r4.FrozenScorerR4(arm="S2", backend="graph", artdir=OUT,
                               scaler_path=OUT / "scaler_corrected_r3.npz",
                               fusion_path=OUT / "fusion_frozen_corrected_r3_S2.json")
    report = {"view_version": rf.view.VIEW_VERSION, "events": 25003, "episodes": len(ep_ids),
              "modes": {}, "tau_hex": fused_thr.hex(), "tau": fused_thr,
              "s2_thr_hex": s2_thr.hex()}
    all_scores = {}
    all_eps = {}
    for mode in ("R", "T"):
        tc = rf.TextCache()
        bp = rf.BlockProcessor(tc)
        bad = {}
        n = 0
        dense_max = 0.0
        logit_max = 0.0
        bag_bad = 0
        row_logits = np.full((3, len(sc["row_index"])), np.nan, dtype=np.float32)
        row_index = sc["row_index"].astype(np.int64).copy()
        ep_ps = np.full((3, len(ep_ids)), -np.inf, dtype=np.float32)
        for ei, ep in enumerate(ep_ids):
            if ei and ei % 100 == 0:
                print(f"  {mode} ep {ei}/{len(ep_ids)} {time.perf_counter()-t0:.1f}s", flush=True)
            rec0 = data.load_prepared("dev", ep)
            ids = rec0["ids_flat"]
            offsets = [int(x) for x in rec0["ids_offsets"].tolist()]
            rids = [int(x) for x in rec0["round_id"].tolist()]
            plen = [int(x) for x in rec0["prompt_length"].tolist()]
            seq = [int(x) for x in rec0["seq_len"].tolist()]
            st = r4.EpisodeStreamR4(episode_id=ep, ids=ids, offsets=offsets, round_ids=rids,
                                    prompt_len=plen, seq_len=seq, tok=tok, tc=tc, bp=bp)
            last = None
            for r in by_ep[ep]:
                rid = int(r["round_id"])
                first = rid != last
                last = rid
                override = body = delta = blocks = None
                if first:
                    ri = rids.index(rid)
                    off = offsets[ri]
                    pl = plen[ri]
                    override = rf.decode_ids(tok, ids[off:off + pl])
                    body = rf.decode_ids(tok, ids[off + pl:off + int(r["token_end"])])
                    if mode == "T":
                        blocks = r4.typed_source_blocks(override)
                else:
                    cur = int(st.cursor)
                    end = int(r["token_end"])
                    delta = rf.decode_ids(tok, ids[off + cur:off + end]) if end > cur else ""
                meta = {"row_index": int(r["row_index"]), "event_id": r["event_id"],
                        "scope": r["scope"], "group_id": r["group_id"],
                        "row_label": bool(r["label_value"]),
                        "episode_label": bool(r["strict_final"])}
                got = st.event(rec_meta=meta, t_start=int(r["token_start"]),
                               t_end=int(r["token_end"]), event_type=r["event_type"],
                               round_id=rid, token_start=int(r["token_start"]),
                               token_end=int(r["token_end"]), fast=True, with_prefix_sha=True,
                               call_index=(None if r["call_index"] is None else int(r["call_index"])),
                               prompt_text_override=override, body_text_override=body,
                               text_delta_override=delta, blocks_override=blocks)
                n += 1
                row = int(r["row_index"])
                b = compare_view(got["rec"], ref_view[row])
                if b:
                    bad.setdefault(row, b)
                if got["rec"]["no_new_emission"] or got["rec"]["unknown_adapter"]:
                    continue
                ci = cache_pos[row]
                for ch in ("cur", "vis", "user"):
                    gb = tuple(int(x) for x in got["feats"]["bags"][ch])
                    rb0 = int(cache[ch + "_off"][ci]); rb1 = int(cache[ch + "_off"][ci + 1])
                    rb = tuple(int(x) for x in cache[ch + "_idx"][rb0:rb1])
                    if gb != rb:
                        bag_bad += 1
                        bad.setdefault(row, []).append(f"bag.{ch}")
                gs = [tuple(int(x) for x in b) for b in got["feats"]["bags"]["src"]]
                so0 = int(src_bag_start[ci]); nn = int(src_counts[ci])
                rs = []
                for j in range(nn):
                    a = int(cache["src_off"][so0 + j]); e2 = int(cache["src_off"][so0 + j + 1])
                    rs.append(tuple(int(x) for x in cache["src_idx"][a:e2]))
                if not rs:
                    rs = [tuple()]
                if gs != rs:
                    bag_bad += 1
                    bad.setdefault(row, []).append("bag.src")
                dg = np.asarray(got["feats"]["dense"], dtype=np.float32)
                dr = np.asarray(cache["dense"][ci], dtype=np.float32)
                d = float(np.abs(dg - dr).max())
                dense_max = max(dense_max, d)
                if d != 0.0:
                    bad.setdefault(row, []).append("dense")
                lg = scorer.row_logits(got["feats"]).astype(np.float32)
                si = score_pos[row]
                lr = sc["row_scores"][:, si].astype(np.float32)
                dl = float(np.abs(lg - lr).max())
                logit_max = max(logit_max, dl)
                row_logits[:, si] = lg
                np.maximum(ep_ps[:, ei], lg, out=ep_ps[:, ei])
        ens = ep_ps.mean(axis=0, dtype=np.float32)
        fused = scorer.fused(base["ensemble_f32"].astype(np.float64), ens)
        st_ep = ep_ps[:, [ep_pos[e] for e in ep_ids]]
        ens_ref = z["ensemble"][[ep_pos[e] for e in ep_ids]]
        fused_ref = z["fused"][[ep_pos[e] for e in ep_ids]]
        labels = base["labels"].astype(np.int64)
        fn = int(((labels == 1) & (fused < fused_thr)).sum())
        fp = int(((labels == 0) & (fused >= fused_thr)).sum())
        s2_fn = int(((labels == 1) & (ens.astype(np.float64) < s2_thr)).sum())
        s2_fp = int(((labels == 0) & (ens.astype(np.float64) >= s2_thr)).sum())
        report["modes"][mode] = {
            "events": n, "bad_rows": len(bad), "bad_examples": dict(list(bad.items())[:10]),
            "bag_bad": bag_bad, "dense_maxdiff": dense_max, "logit_maxdiff": logit_max,
            "per_seed_ep_maxdiff": float(np.abs(st_ep - z["per_seed"]).max()),
            "ensemble_maxdiff": float(np.abs(ens - ens_ref).max()),
            "fused_maxdiff": float(np.abs(fused - fused_ref).max()),
            "s2_fn": s2_fn, "s2_fp": s2_fp, "fusion_fn": fn, "fusion_fp": fp,
            "graph_fallback": scorer.net.n_fallback,
            "cap_counts": {str(k): v for k, v in scorer.net.cap_counts.items()},
            "reference_s2_fn": 13, "reference_s2_fp": 80,
            "reference_fusion_fn": 12, "reference_fusion_fp": 80,
        }
        all_scores[mode] = row_logits
        all_eps[mode] = {"per_seed": st_ep.astype(np.float32),
                         "ensemble": ens.astype(np.float32),
                         "fused": fused.astype(np.float64),
                         "s2_fn": s2_fn, "s2_fp": s2_fp,
                         "fusion_fn": fn, "fusion_fp": fp}
    out = OUT / "runtime-r4"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "parity_r4_row_logits.npz",
                        row_index=row_index,
                        R_row_logits=all_scores["R"], T_row_logits=all_scores["T"],
                        dtype_note=np.asarray("row logits float32; episode per_seed float32; "
                                              "ensemble float32; fused float64"),
                        episode_ids=np.asarray(ep_ids),
                        ref_per_seed=np.asarray(z["per_seed"], dtype=np.float32),
                        ref_ensemble=np.asarray(z["ensemble"], dtype=np.float32),
                        ref_fused=np.asarray(z["fused"], dtype=np.float64),
                        R_per_seed=all_eps["R"]["per_seed"],
                        R_ensemble=all_eps["R"]["ensemble"],
                        R_fused=all_eps["R"]["fused"],
                        T_per_seed=all_eps["T"]["per_seed"],
                        T_ensemble=all_eps["T"]["ensemble"],
                        T_fused=all_eps["T"]["fused"],
                        labels=labels,
                        tau_hex=np.asarray(fused_thr.hex()),
                        s2_thr_hex=np.asarray(s2_thr.hex()))
    near = []
    for mode in ("R", "T"):
        f = all_eps[mode]["fused"]
        for i in np.where(np.abs(f - fused_thr) < 1e-5)[0]:
            near.append({"mode": mode, "episode_id": ep_ids[i],
                         "label": int(labels[i]), "fused": repr(float(f[i])),
                         "fused_hex": float(f[i]).hex(),
                         "margin": float(f[i] - fused_thr)})
    report["near_tau"] = near
    report["seconds"] = time.perf_counter() - t0
    (out / "parity_r4.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
