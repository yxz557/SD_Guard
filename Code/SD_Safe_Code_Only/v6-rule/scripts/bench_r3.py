"""corrected-r3 batch-1 cost, mode A (tokens-only) and mode B (controller typed messages).

Mode A: every round/event re-establishes history from the causal token stream; prompt decode,
new-emission decode, index/retrieval/value-source, features, 3 students, base cache, per-seed
max, float32 mean and frozen fusion are all inside the per-event timer.

Mode B: the prompt text arrives from the controller (typed); only the newly emitted assistant
tokens are decoded by the adapter.  The controller-side prompt decode is measured separately and
charged to the first event of that round in the "full" variant.  All detector work remains
inside the timer.  Inputs are proofed by scripts/parity_r3.py --typed-prompt 1 (25 003/25 003
events, 0 mismatches).
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
OUT = ART / "corrected-r3"


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--process-id", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-episodes", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--modes", default="A,B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)

    views = [json.loads(l) for l in (OUT / "views_dev.jsonl").read_text().splitlines() if l.strip()]
    by_ep_views: dict[str, list[dict]] = {}
    for v in views:
        by_ep_views.setdefault(v["episode_id"], []).append(v)
    for vs in by_ep_views.values():
        vs.sort(key=lambda r: (int(r["round_id"]), int(r["token_end"]), int(r["row_index"])))

    base = ev.load_dev_base()
    ep_ids = [str(x) for x in base["episode_id"].tolist()]
    labels = np.asarray(base["labels"], dtype=np.int64)
    base_ens_saved = np.asarray(base["ensemble_f32"], dtype=np.float32)
    if args.episodes:
        ep_ids = ep_ids[: args.episodes]
    labels = labels[: len(ep_ids)]
    base_ens_saved = base_ens_saved[: len(ep_ids)]
    rows = data.load_rows("dev")
    ep_set = set(ep_ids)
    by_ep_rows: dict[str, list[dict]] = {}
    for r in rows.values():
        if r["episode_id"] in ep_set:
            by_ep_rows.setdefault(r["episode_id"], []).append(r)
    for vs in by_ep_rows.values():
        vs.sort(key=lambda r: (int(r["round_id"]), int(r["token_end"]), int(r["row_index"])))

    dev = baseline.dev_from_packs()
    grow = dev["rows"][0]["global_row_id"].astype(np.int64)
    base_logits_all = np.stack([dev["rows"][s]["logits"] for s in range(3)], axis=0).astype(np.float32)
    base_of_row = {int(r): i for i, r in enumerate(grow.tolist())}

    saved_stud = np.asarray(np.load(OUT / "replay_S2.npz", allow_pickle=True)["ensemble"], np.float32)
    saved_fused = np.asarray(np.load(OUT / "replay_S2.npz", allow_pickle=True)["fused"], np.float64)
    stud_ids = [str(x) for x in np.load(OUT / "replay_S2.npz", allow_pickle=True)["episode_id"].tolist()]
    pos = {e: i for i, e in enumerate(stud_ids)}
    idx_all = np.asarray([pos[e] for e in ep_ids], dtype=np.int64)
    s2_thr = float(json.loads((OUT / "replay_S2.json").read_text())["student_threshold"])
    fj = json.loads((OUT / "fusion_frozen_corrected_r3_S2.json").read_text())
    fused_thr = float(fj["threshold"])
    scorer = runtime_fast.FrozenScorer(arm="S2", device=args.device, artdir=OUT,
                                       scaler_path=OUT / "scaler_corrected_r3.npz",
                                       fusion_path=OUT / "fusion_frozen_corrected_r3_S2.json")

    def one_pass(mode: str, repeat: int, collect: bool) -> dict:
        tc = runtime_fast.TextCache()
        bp = runtime_fast.BlockProcessor(tc)
        times = {"detector": [], "full": [], "view_feat": [], "model": [], "fuse": [],
                 "prompt_shared": [], "segment_decode": []}
        kinds, scopes, lens, firsts = [], [], [], []
        student_ep = np.full((3, len(ep_ids)), -np.inf, dtype=np.float32)
        base_ep = np.full((3, len(ep_ids)), -np.inf, dtype=np.float32)
        ep_index = {e: i for i, e in enumerate(ep_ids)}
        n_events = n_scored = 0
        prompt_shared_total = 0.0
        load_s = 0.0
        t_pass = time.perf_counter()
        for ep in ep_ids:
            t_load = time.perf_counter()
            rec0 = data.load_prepared("dev", ep)
            load_s += time.perf_counter() - t_load
            ids = rec0["ids_flat"]
            offsets = [int(x) for x in rec0["ids_offsets"].tolist()]
            rids = [int(x) for x in rec0["round_id"].tolist()]
            plen = [int(x) for x in rec0["prompt_length"].tolist()]
            seq = [int(x) for x in rec0["seq_len"].tolist()]
            st = runtime_fast.EpisodeStream(episode_id=ep, ids=ids, offsets=offsets,
                                            round_ids=rids, prompt_len=plen, seq_len=seq,
                                            tok=tok, tc=tc, bp=bp)
            i_ep = ep_index[ep]
            last_round = None
            for r in by_ep_rows[ep]:
                rid = int(r["round_id"])
                first = rid != last_round
                last_round = rid
                override = None
                body_override = None
                delta_override = None
                blocks_override = None
                shared_ms = 0.0
                if mode == "B":
                    ri = rids.index(rid)
                    off = offsets[ri]
                    pl = plen[ri]
                    ts = time.perf_counter()
                    blocks_override = None
                    if first:
                        override = runtime_fast.decode_ids(tok, ids[off:off + pl])
                        body_override = runtime_fast.decode_ids(
                            tok, ids[off + pl:off + int(r["token_end"])])
                        # controller role formatter: message boundaries already known
                        blocks_override = runtime_fast.view.parse_blocks(override)
                    else:
                        cur = int(st.cursor)
                        end = int(r["token_end"])
                        delta_override = (runtime_fast.decode_ids(tok, ids[off + cur:off + end])
                                          if end > cur else "")
                    shared_ms = (time.perf_counter() - ts) * 1e3
                    prompt_shared_total += shared_ms
                t0 = time.perf_counter()
                meta = {"row_index": int(r["row_index"]), "event_id": r["event_id"],
                        "scope": r["scope"], "group_id": r["group_id"],
                        "row_label": bool(r["label_value"]),
                        "episode_label": bool(r["strict_final"])}
                got = st.event(rec_meta=meta, t_start=int(r["token_start"]),
                               t_end=int(r["token_end"]), event_type=r["event_type"],
                               round_id=rid, token_start=int(r["token_start"]),
                               token_end=int(r["token_end"]), fast=True, timed=True,
                               call_index=(None if r["call_index"] is None else int(r["call_index"])),
                               prompt_text_override=override,
                               body_text_override=body_override,
                               text_delta_override=delta_override,
                               blocks_override=blocks_override)
                t1 = time.perf_counter()
                if got["rec"]["no_new_emission"] or got["rec"]["unknown_adapter"]:
                    t3 = time.perf_counter()
                    n_events += 1
                    if collect:
                        times["detector"].append((t3 - t0) * 1e3)
                        times["full"].append((t3 - t0) * 1e3 + shared_ms)
                        firsts.append(first)
                        kinds.append("no_score")
                        scopes.append(str(r["scope"]))
                        lens.append(int(got["rec"]["prefix_chars"]))
                    continue
                logits = scorer.row_logits(got["feats"]).astype(np.float32)
                t2 = time.perf_counter()
                bi = base_of_row[int(r["row_index"])]
                np.maximum(student_ep[:, i_ep], logits, out=student_ep[:, i_ep])
                np.maximum(base_ep[:, i_ep], base_logits_all[:, bi], out=base_ep[:, i_ep])
                stud_cur = student_ep[:, i_ep].mean(axis=0, dtype=np.float32)
                base_cur = base_ep[:, i_ep].mean(axis=0, dtype=np.float32)
                _ = scorer.fused(np.asarray([base_cur]), np.asarray([stud_cur]))
                t3 = time.perf_counter()
                n_scored += 1
                n_events += 1
                if collect:
                    times["detector"].append((t3 - t0) * 1e3)
                    times["full"].append((t3 - t0) * 1e3 + shared_ms)
                    times["view_feat"].append((t1 - t0) * 1e3)
                    times["model"].append((t2 - t1) * 1e3)
                    times["fuse"].append((t3 - t2) * 1e3)
                    times["prompt_shared"].append(shared_ms)
                    times["segment_decode"].append(got["timings"].get("segment_decode", 0.0) * 1e3)
                    kinds.append(str(r["event_type"]))
                    scopes.append(str(r["scope"]))
                    lens.append(int(got["rec"]["prefix_chars"]))
                    firsts.append(first)
        stud_ens = student_ep.mean(axis=0, dtype=np.float32)
        base_ens = base_ep.mean(axis=0, dtype=np.float32)
        fused = scorer.fused(base_ens.astype(np.float64), stud_ens)
        out = {"mode": mode, "repeat": repeat, "events": n_events, "scored": n_scored,
               "episodes": len(ep_ids), "load_s": load_s,
               "pass_s": time.perf_counter() - t_pass,
               "prompt_shared_s": prompt_shared_total / 1e3,
               "s2_fn": int(((labels == 1) & (stud_ens.astype(np.float64) < s2_thr)).sum()),
               "s2_fp": int(((labels == 0) & (stud_ens.astype(np.float64) >= s2_thr)).sum()),
               "fusion_fn": int(((labels == 1) & (fused < fused_thr)).sum()),
               "fusion_fp": int(((labels == 0) & (fused >= fused_thr)).sum()),
               "stud_maxdiff_vs_saved": float(np.abs(stud_ens - saved_stud[idx_all]).max()),
               "fused_maxdiff_vs_saved": float(np.abs(fused - saved_fused[idx_all]).max()),
               "base_maxdiff_vs_saved": float(np.abs(base_ens - base_ens_saved[idx_all]).max())}
        if collect:
            for key in ("detector", "full", "view_feat", "model", "fuse", "prompt_shared",
                        "segment_decode"):
                a = np.asarray(times[key])
                out[f"{key}_p50_ms"] = pct(a, 50)
                out[f"{key}_p95_ms"] = pct(a, 95)
                out[f"{key}_p99_ms"] = pct(a, 99)
                out[f"{key}_mean_ms"] = float(a.mean()) if a.size else float("nan")
                out[f"{key}_max_ms"] = float(a.max()) if a.size else float("nan")
            tot = np.asarray(times["detector"])
            full = np.asarray(times["full"])
            kinds_a = np.asarray(kinds)
            scopes_a = np.asarray(scopes)
            firsts_a = np.asarray(firsts)
            lens_a = np.asarray(lens)
            out["throughput_per_s"] = float(tot.size / tot.sum() * 1e3)
            out["slices"] = {}
            for name, mask in [("first_event", firsts_a), ("later_event", ~firsts_a)] + \
                    [(f"kind:{k}", kinds_a == k) for k in sorted(set(kinds_a.tolist()))] + \
                    [(f"scope:{s}", scopes_a == s) for s in sorted(set(scopes_a.tolist()))] + \
                    [(f"len:{lo}-{hi}", (lens_a >= lo) & (lens_a < hi)) for lo, hi in
                     ((0, 2000), (2000, 8000), (8000, 20000), (20000, 10**9))]:
                a = full[mask]
                if a.size:
                    out["slices"][name] = {"n": int(a.size), "p50_ms": pct(a, 50),
                                           "p95_ms": pct(a, 95), "p99_ms": pct(a, 99)}
            out["warm"] = repeat > 0
        return out

    if args.warmup_episodes:
        one_pass("A", -1, False)
        one_pass("B", -1, False)
    import torch
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results = []
    for rep in range(args.repeats):
        for mode in [m.strip() for m in args.modes.split(",")]:
            results.append(one_pass(mode, rep, True))
    out = {"process_id": args.process_id, "device": args.device,
           "repeats": results,
           "gpu_peak_alloc_bytes": int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0,
           "gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if args.device == "cuda" else 0,
           "cpu_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    for mode in ("A", "B"):
        rs = [r for r in results if r["mode"] == mode]
        for key in ("detector_p50_ms", "detector_p95_ms", "detector_p99_ms",
                    "full_p50_ms", "full_p95_ms", "full_p99_ms", "throughput_per_s"):
            vals = [r[key] for r in rs if key in r]
            if vals:
                out[f"{mode}_{key}_median"] = float(statistics.median(vals))
        if rs:
            out[f"{mode}_full_p95_ms_worst_repeat"] = float(max(r["full_p95_ms"] for r in rs))
            out[f"{mode}_detector_p95_ms_worst_repeat"] = float(max(r["detector_p95_ms"] for r in rs))
    out["mode_A_cost_passed_1ms"] = bool(out.get("A_full_p95_ms_worst_repeat", 9e9) <= 1.0)
    out["mode_B_detector_cost_passed_1ms"] = bool(
        out.get("B_detector_p95_ms_worst_repeat", 9e9) <= 1.0)
    out["mode_B_full_cost_passed_1ms"] = bool(out.get("B_full_p95_ms_worst_repeat", 9e9) <= 1.0)
    path = Path(args.out or (OUT / f"runtime/cost_r3_p{args.process_id}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "repeats"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
