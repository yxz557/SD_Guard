"""Honest batch-1 streaming cost for the frozen S2+FUSION_S2 runtime (runtime-r2).

Per event, inside the timer: token decode of the newly emitted span, incremental history/source
index update, current-block finalisation, generic retrieval / value->source matching, hashed
features, the merged three-seed student forward, frozen base-cache lookup, per-seed running max,
float32 ensemble and the frozen FUSION_S2 mapping.  No view/cache lookup shortcut.

All 2028 common-DEV episodes / 25003 CT rows are streamed in causal episode order.  The 20 rows
without a frozen student view are no-new-emission events: decode/index advance only, no score.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v6 import baseline, data, evaluate as ev, metrics, runtime_fast  # noqa: E402

ART = runtime_fast.ART
FROZEN_S2_THR = -0.7444994449615479
FROZEN_FUSION_THR = 0.6624409698441314


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--process-id", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-episodes", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)

    views = []
    for name in ("dev_common.jsonl", "dev_common_fixup.jsonl"):
        views += [json.loads(l) for l in (ART / "views" / name).read_text().splitlines()]
    view_rows = {int(v["row_index"]) for v in views}
    by_ep_views = {}
    for v in views:
        by_ep_views.setdefault(v["episode_id"], []).append(v)

    base = ev.load_dev_base()
    ep_ids = [str(x) for x in base["episode_id"].tolist()]
    labels = np.asarray(base["labels"], dtype=np.int64)
    base_saved_ep = np.asarray(base["per_seed_episode"], dtype=np.float32)
    base_ens_saved = np.asarray(base["ensemble_f32"], dtype=np.float32)
    label_of = {str(base["episode_id"][i]): int(labels[i]) for i in range(len(labels))}
    if args.episodes:
        ep_ids = ep_ids[: args.episodes]
    labels = np.asarray([label_of[e] for e in ep_ids], dtype=np.int64)
    ep_set = set(ep_ids)
    rows = data.load_rows("dev")
    by_ep_rows = {}
    for r in rows.values():
        if r["episode_id"] in ep_set:
            by_ep_rows.setdefault(r["episode_id"], []).append(r)
    for vs in by_ep_rows.values():
        vs.sort(key=lambda r: (int(r["round_id"]), int(r["token_end"]), int(r["row_index"])))

    dev = baseline.dev_from_packs()
    grow = dev["rows"][0]["global_row_id"].astype(np.int64)
    base_logits_all = np.stack([dev["rows"][s]["logits"] for s in range(3)], axis=0).astype(np.float32)
    base_of_row = {int(r): i for i, r in enumerate(grow.tolist())}

    saved_stud = np.asarray(np.load(ART / "dev_episode_S2_v3.npz", allow_pickle=True)["ensemble"],
                            np.float32)
    scorer = runtime_fast.FrozenScorer(arm="S2", device=args.device)
    fused_thr = FROZEN_FUSION_THR
    s2_thr = FROZEN_S2_THR

    def one_pass(repeat: int, collect: bool) -> dict:
        tc = runtime_fast.TextCache()
        bp = runtime_fast.BlockProcessor(tc)
        times = {"total": [], "view_feat": [], "model": [], "fuse": [], "skip": []}
        kinds, scopes, lens, firsts = [], [], [], []
        student_ep = np.full((3, len(ep_ids)), -np.inf, dtype=np.float32)
        base_ep = np.full((3, len(ep_ids)), -np.inf, dtype=np.float32)
        ep_index = {e: i for i, e in enumerate(ep_ids)}
        n_events = n_scored = n_skip = 0
        load_s = 0.0
        t_pass = time.perf_counter()
        for ep in ep_ids:
            t_load = time.perf_counter()
            rec0 = data.load_prepared("dev", ep)
            load_s += time.perf_counter() - t_load
            st = runtime_fast.EpisodeStream(
                episode_id=ep, ids=rec0["ids_flat"],
                offsets=[int(x) for x in rec0["ids_offsets"].tolist()],
                round_ids=[int(x) for x in rec0["round_id"].tolist()],
                prompt_len=[int(x) for x in rec0["prompt_length"].tolist()],
                seq_len=[int(x) for x in rec0["seq_len"].tolist()],
                tok=tok, tc=tc, bp=bp)
            i_ep = ep_index[ep]
            last_round = None
            for r in by_ep_rows[ep]:
                rid = int(r["round_id"])
                first = rid != last_round
                last_round = rid
                t0 = time.perf_counter()
                if int(r["row_index"]) not in view_rows:
                    st.advance_only(rid, int(r["token_end"]))
                    t1 = time.perf_counter()
                    if collect:
                        times["skip"].append((t1 - t0) * 1e3)
                    n_skip += 1
                    n_events += 1
                    continue
                meta = {"row_index": int(r["row_index"]), "event_id": r["event_id"],
                        "scope": r["scope"], "group_id": r["group_id"],
                        "row_label": bool(r["label_value"]),
                        "episode_label": bool(r["strict_final"])}
                got = st.event(rec_meta=meta, t_start=int(r["token_start"]),
                               t_end=int(r["token_end"]), event_type=r["event_type"],
                               round_id=rid, token_start=int(r["token_start"]),
                               token_end=int(r["token_end"]), fast=True, timed=True)
                t1 = time.perf_counter()
                logits = scorer.row_logits(got["feats"])
                t2 = time.perf_counter()
                bi = base_of_row[int(r["row_index"])]
                np.maximum(student_ep[:, i_ep], logits.astype(np.float32), out=student_ep[:, i_ep])
                np.maximum(base_ep[:, i_ep], base_logits_all[:, bi], out=base_ep[:, i_ep])
                # frozen online score update (float32 student mean -> float64 mapping)
                stud_cur = student_ep[:, i_ep].mean(axis=0, dtype=np.float32)
                base_cur = base_ep[:, i_ep].mean(axis=0, dtype=np.float32)
                _ = scorer.fused(np.asarray([base_cur]), np.asarray([stud_cur]))
                t3 = time.perf_counter()
                n_scored += 1
                n_events += 1
                if collect:
                    times["total"].append((t3 - t0) * 1e3)
                    times["view_feat"].append((t1 - t0) * 1e3)
                    times["model"].append((t2 - t1) * 1e3)
                    times["fuse"].append((t3 - t2) * 1e3)
                    kinds.append(str(r["event_type"]))
                    scopes.append(str(r["scope"]))
                    lens.append(int(got["rec"]["prefix_chars"]))
                    firsts.append(first)
        stud_ens = student_ep.mean(axis=0, dtype=np.float32)
        base_ens = base_ep.mean(axis=0, dtype=np.float32)
        fused = scorer.fused(base_ens.astype(np.float64), stud_ens)
        out = {"repeat": repeat, "events": n_events, "scored": n_scored, "skipped": n_skip,
               "episodes": len(ep_ids), "load_s": load_s,
               "pass_s": time.perf_counter() - t_pass,
               "s2_fn": int(((labels == 1) & (stud_ens.astype(np.float64) < s2_thr)).sum()),
               "s2_fp": int(((labels == 0) & (stud_ens.astype(np.float64) >= s2_thr)).sum()),
               "fusion_fn": int(((labels == 1) & (fused < fused_thr)).sum()),
               "fusion_fp": int(((labels == 0) & (fused >= fused_thr)).sum()),
               "stud_ep_maxdiff_vs_saved": float(np.abs(stud_ens - saved_stud[idx_all_of(ep_ids)]).max()),
               "base_ep_maxdiff_vs_saved": float(np.abs(base_ens - base_ens_saved[idx_all_of(ep_ids)]).max())}
        if collect:
            tot = np.asarray(times["total"])
            out["p50_ms"] = pct(tot, 50); out["p95_ms"] = pct(tot, 95)
            out["p99_ms"] = pct(tot, 99); out["mean_ms"] = float(tot.mean())
            out["max_ms"] = float(tot.max())
            out["throughput_per_s"] = float(tot.size / tot.sum() * 1e3)
            out["components_p50_ms"] = {k: pct(np.asarray(v), 50) for k, v in times.items() if k != "total"}
            out["components_p95_ms"] = {k: pct(np.asarray(v), 95) for k, v in times.items() if k != "total"}
            kinds = np.asarray(kinds); scopes = np.asarray(scopes); firsts = np.asarray(firsts)
            lens = np.asarray(lens)
            tot = np.asarray(times["total"])
            out["slices"] = {}
            for name, mask in [("first_event", firsts), ("later_event", ~firsts)] + \
                    [(f"kind:{k}", kinds == k) for k in sorted(set(kinds.tolist()))] + \
                    [(f"scope:{s}", scopes == s) for s in sorted(set(scopes.tolist()))] + \
                    [(f"len:{lo}-{hi}", (lens >= lo) & (lens < hi)) for lo, hi in
                     ((0, 2000), (2000, 8000), (8000, 20000), (20000, 10**9))]:
                a = tot[mask]
                if a.size:
                    out["slices"][name] = {"n": int(a.size), "p50_ms": pct(a, 50),
                                           "p95_ms": pct(a, 95), "p99_ms": pct(a, 99)}
            out["warm"] = False
        return out

    def idx_all_of(eps):
        saved_ids = [str(x) for x in np.load(ART / "dev_episode_S2_v3.npz",
                                             allow_pickle=True)["episode_id"].tolist()]
        pos = {e: i for i, e in enumerate(saved_ids)}
        return np.asarray([pos[e] for e in eps], dtype=np.int64)

    # warm-up on the real path, then clear all data caches before measurement
    if args.warmup_episodes:
        one_pass(-1, False)
    import torch
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results = []
    for rep in range(args.repeats):
        results.append(one_pass(rep, True))
    out = {"process_id": args.process_id, "device": args.device, "repeats": results,
           "gpu_peak_alloc_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
           "gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if args.device == "cuda" else 0,
           "cpu_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    # medians across repeats
    for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "throughput_per_s"):
        out[f"{key}_median"] = float(statistics.median(r[key] for r in results))
    out["p95_ms_worst_repeat"] = float(max(r["p95_ms"] for r in results))
    out["cost_passed_1ms"] = bool(out["p95_ms_worst_repeat"] <= 1.0)
    path = Path(args.out or (ART / "runtime-r2" / f"cost_r2_p{args.process_id}.json"))
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "repeats"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
