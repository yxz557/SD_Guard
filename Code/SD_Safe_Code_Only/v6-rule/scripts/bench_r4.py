"""Runtime-r4 batch-1 cost with the corrected timing scope.

Modes
-----
A  tokens-only: every round/event re-decodes the causal token stream; decode + parse + index +
   features + 3 students + base lookup + max/mean/fusion are inside the timer (same as r3 A).
R  rawtext interface: the controller already decoded the prompt and the assistant delta (shared
   controller decode measured separately); *all* detector work -- including
   ``view.parse_blocks`` of the full prompt, role/source index building, current JSON parse,
   retrieval and features -- is inside the detector timer.
T  typed interface: a source adapter supplies the controller's role-tagged message list.  The
   adapter (boundary recovery from the prompt text) is measured separately and charged in the
   ``T integrated`` column; the detector timer starts at typed-message ingest.  Pure T is shown
   only as an upper bound, because the prepared episode store does not itself carry a separate
   message list.

Two equivalent scoring variants:
sync  per-event logits are copied to CPU (D2H + sync) every event, as in r3.
gpu   per-event logits stay on the GPU; per-seed running max is accumulated on the GPU and the
      episode score is copied once at episode finalisation (all transfer still counted in the
      pass wall / throughput and in the separate ``episode_final`` timing).

Nothing is trained or re-thresholded; the frozen corrected-r3 S2 / FUSION_S2 arithmetic and
``>=`` tau are reused verbatim.  No TEST material is read.
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

from sdsafe_v6 import baseline, data, evaluate as ev, runtime_fast as rf, runtime_r4 as r4  # noqa: E402

ART = rf.ART
OUT = ART / "corrected-r3"


def pct(a, q):
    a = np.asarray(a)
    return float(np.percentile(a, q)) if a.size else float("nan")


def load_ref():
    z = np.load(OUT / "replay_S2.npz", allow_pickle=True)
    ids = [str(x) for x in z["episode_id"].tolist()]
    return z, {e: i for i, e in enumerate(ids)}


def one_pass(*, mode, variant, repeat, collect, ep_ids, by_ep_rows, tok, scorer, base_logits_all,
             base_of_row, labels, saved_stud, saved_fused, idx_all, s2_thr, fused_thr) -> dict:
    tc = rf.TextCache()
    bp = rf.BlockProcessor(tc)
    times = {k: [] for k in ("detector", "detector_int", "full", "view_feat", "model",
                              "fuse", "shared", "adapter", "segment_decode")}
    kinds, scopes, lens, firsts, empties = [], [], [], [], []
    student_ep = np.full((3, len(ep_ids)), -np.inf, dtype=np.float32)
    base_ep = np.full((3, len(ep_ids)), -np.inf, dtype=np.float32)
    ep_index = {e: i for i, e in enumerate(ep_ids)}
    n_events = n_scored = 0
    shared_total = 0.0
    adapter_total = 0.0
    final_ms = []
    load_s = 0.0
    t_pass = time.perf_counter()
    use_gpu = variant == "gpu"
    for ep in ep_ids:
        t_load = time.perf_counter()
        rec0 = data.load_prepared("dev", ep)
        load_s += time.perf_counter() - t_load
        ids = rec0["ids_flat"]
        offsets = [int(x) for x in rec0["ids_offsets"].tolist()]
        rids = [int(x) for x in rec0["round_id"].tolist()]
        plen = [int(x) for x in rec0["prompt_length"].tolist()]
        seq = [int(x) for x in rec0["seq_len"].tolist()]
        st = r4.EpisodeStreamR4(episode_id=ep, ids=ids, offsets=offsets, round_ids=rids,
                                prompt_len=plen, seq_len=seq, tok=tok, tc=tc, bp=bp)
        i_ep = ep_index[ep]
        if use_gpu:
            import torch
            dev_max = torch.full((3,), -float("inf"), dtype=torch.float32, device="cuda")
        last_round = None
        last_row = None
        for r in by_ep_rows[ep]:
            rid = int(r["round_id"])
            first = rid != last_round
            last_round = rid
            override = body_override = delta_override = blocks_override = None
            shared_ms = 0.0
            adapter_ms = 0.0
            if mode in ("R", "T"):
                ri = rids.index(rid)
                off = offsets[ri]
                pl = plen[ri]
                ts = time.perf_counter()
                if first:
                    override = rf.decode_ids(tok, ids[off:off + pl])
                    body_override = rf.decode_ids(tok, ids[off + pl:off + int(r["token_end"])])
                    if mode == "T":
                        ta = time.perf_counter()
                        blocks_override = r4.typed_source_blocks(override)
                        adapter_ms = (time.perf_counter() - ta) * 1e3
                        adapter_total += adapter_ms
                else:
                    cur = int(st.cursor)
                    end = int(r["token_end"])
                    delta_override = (rf.decode_ids(tok, ids[off + cur:off + end])
                                      if end > cur else "")
                shared_ms = (time.perf_counter() - ts) * 1e3 - adapter_ms
                shared_total += shared_ms
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
                           prompt_text_override=override, body_text_override=body_override,
                           text_delta_override=delta_override, blocks_override=blocks_override)
            t1 = time.perf_counter()
            if got["rec"]["no_new_emission"] or got["rec"]["unknown_adapter"]:
                t3 = time.perf_counter()
                n_events += 1
                if collect:
                    times["detector"].append((t3 - t0) * 1e3)
                    times["detector_int"].append((t3 - t0) * 1e3 + adapter_ms)
                    times["full"].append((t3 - t0) * 1e3 + shared_ms + adapter_ms)
                    times["view_feat"].append((t1 - t0) * 1e3)
                    times["model"].append(0.0)
                    times["fuse"].append(0.0)
                    times["shared"].append(shared_ms)
                    times["adapter"].append(adapter_ms)
                    times["segment_decode"].append(got["timings"].get("segment_decode", 0.0) * 1e3)
                    firsts.append(first); kinds.append("no_score")
                    scopes.append(str(r["scope"])); lens.append(int(got["rec"]["prefix_chars"]))
                    empties.append(bool(got["rec"]["empty_segment"]))
                last_row = (r, got, t0, t1)
                continue
            if use_gpu:
                logits_t = scorer.net.forward_gpu(got["feats"])
                t2 = time.perf_counter()
                import torch
                torch.maximum(dev_max, logits_t, out=dev_max)
            else:
                logits = scorer.row_logits(got["feats"]).astype(np.float32)
                t2 = time.perf_counter()
            bi = base_of_row[int(r["row_index"])]
            np.maximum(base_ep[:, i_ep], base_logits_all[:, bi], out=base_ep[:, i_ep])
            t3 = time.perf_counter()
            if not use_gpu:
                np.maximum(student_ep[:, i_ep], logits, out=student_ep[:, i_ep])
                stud_cur = student_ep[:, i_ep].mean(axis=0, dtype=np.float32)
                base_cur = base_ep[:, i_ep].mean(axis=0, dtype=np.float32)
                _ = scorer.fused(np.asarray([base_cur]), np.asarray([stud_cur]))
            n_scored += 1
            n_events += 1
            if collect:
                times["detector"].append((t3 - t0) * 1e3)
                times["detector_int"].append((t3 - t0) * 1e3 + adapter_ms)
                times["full"].append((t3 - t0) * 1e3 + shared_ms + adapter_ms)
                times["view_feat"].append((t1 - t0) * 1e3)
                times["model"].append((t2 - t1) * 1e3)
                times["fuse"].append((t3 - t2) * 1e3)
                times["shared"].append(shared_ms)
                times["adapter"].append(adapter_ms)
                times["segment_decode"].append(got["timings"].get("segment_decode", 0.0) * 1e3)
                firsts.append(first); kinds.append(str(r["event_type"]))
                scopes.append(str(r["scope"])); lens.append(int(got["rec"]["prefix_chars"]))
                empties.append(bool(got["rec"]["empty_segment"]))
            last_row = (r, got, t0, t1, logits if not use_gpu else None)
        if use_gpu:
            import torch
            tf = time.perf_counter()
            student_ep[:, i_ep] = dev_max.cpu().numpy()
            final_ms.append((time.perf_counter() - tf) * 1e3)
            # charge the episode-finalisation sync to the last event's full timer
            if collect and final_ms[-1] > 0 and times["full"]:
                times["full"][-1] += final_ms[-1]
                times["detector"][-1] += final_ms[-1]
                times["detector_int"][-1] += final_ms[-1]
    stud_ens = student_ep.mean(axis=0, dtype=np.float32)
    base_ens = base_ep.mean(axis=0, dtype=np.float32)
    fused = scorer.fused(base_ens.astype(np.float64), stud_ens)
    out = {"mode": mode, "variant": variant, "repeat": repeat, "events": n_events,
           "scored": n_scored, "episodes": len(ep_ids), "load_s": load_s,
           "pass_s": time.perf_counter() - t_pass,
           "shared_s": shared_total / 1e3, "adapter_s": adapter_total / 1e3,
           "s2_fn": int(((labels == 1) & (stud_ens.astype(np.float64) < s2_thr)).sum()),
           "s2_fp": int(((labels == 0) & (stud_ens.astype(np.float64) >= s2_thr)).sum()),
           "fusion_fn": int(((labels == 1) & (fused < fused_thr)).sum()),
           "fusion_fp": int(((labels == 0) & (fused >= fused_thr)).sum()),
           "stud_maxdiff_vs_saved": float(np.abs(stud_ens - saved_stud[idx_all]).max()),
           "fused_maxdiff_vs_saved": float(np.abs(fused - saved_fused[idx_all]).max())}
    if collect:
        for key in ("detector", "detector_int", "full", "view_feat", "model", "fuse",
                    "shared", "adapter", "segment_decode"):
            a = np.asarray(times[key], dtype=np.float64)
            for q in (50, 95, 99):
                out[f"{key}_p{q}_ms"] = pct(a, q)
            out[f"{key}_mean_ms"] = float(a.mean()) if a.size else float("nan")
            out[f"{key}_max_ms"] = float(a.max()) if a.size else float("nan")
        det = np.asarray(times["detector"], dtype=np.float64)
        full = np.asarray(times["full"], dtype=np.float64)
        kinds_a = np.asarray(kinds); scopes_a = np.asarray(scopes)
        firsts_a = np.asarray(firsts); lens_a = np.asarray(lens)
        empties_a = np.asarray(empties)
        if final_ms:
            out["episode_final_p50_ms"] = pct(final_ms, 50)
            out["episode_final_p95_ms"] = pct(final_ms, 95)
        out["throughput_per_s"] = float(det.size / det.sum() * 1e3) if det.size else float("nan")
        out["wall_throughput_per_s"] = float(n_events / out["pass_s"])
        out["slices"] = {}
        for name, mask in ([("first_event", firsts_a), ("later_event", ~firsts_a),
                            ("empty_segment", empties_a)] +
                           [(f"kind:{k}", kinds_a == k) for k in sorted(set(kinds_a.tolist()))] +
                           [(f"scope:{s}", scopes_a == s) for s in sorted(set(scopes_a.tolist()))] +
                           [(f"len:{lo}-{hi}", (lens_a >= lo) & (lens_a < hi)) for lo, hi in
                            ((0, 2000), (2000, 8000), (8000, 20000), (20000, 10**9))]):
            a = det[mask]
            if a.size:
                out["slices"][name] = {"n": int(a.size), "p50_ms": pct(a, 50),
                                       "p95_ms": pct(a, 95), "p99_ms": pct(a, 99)}
        out["episodes_per_seed_max"] = student_ep.tolist()
        out["episode_ensemble"] = stud_ens.tolist()
        out["episode_fused"] = fused.tolist()
        out["episode_base"] = base_ens.tolist()
        out["episode_ids"] = list(ep_ids)
        out["warm"] = repeat > 0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--process-id", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup-episodes", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--modes", default="A,R,T")
    ap.add_argument("--variants", default="gpu,sync")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-npz", type=int, default=1)
    ap.add_argument("--art-name", default="runtime-r4")
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)

    views = [json.loads(l) for l in (OUT / "views_dev.jsonl").read_text().splitlines() if l.strip()]
    by_ep_views = {}
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
    by_ep_rows = {}
    for r in rows.values():
        if r["episode_id"] in ep_set:
            by_ep_rows.setdefault(r["episode_id"], []).append(r)
    for vs in by_ep_rows.values():
        vs.sort(key=lambda r: (int(r["round_id"]), int(r["token_end"]), int(r["row_index"])))

    dev = baseline.dev_from_packs()
    grow = dev["rows"][0]["global_row_id"].astype(np.int64)
    base_logits_all = np.stack([dev["rows"][s]["logits"] for s in range(3)],
                               axis=0).astype(np.float32)
    base_of_row = {int(r): i for i, r in enumerate(grow.tolist())}

    z, pos = load_ref()
    saved_stud = np.asarray(z["ensemble"], np.float32)
    saved_fused = np.asarray(z["fused"], np.float64)
    idx_all = np.asarray([pos[e] for e in ep_ids], dtype=np.int64)
    s2_thr = float(json.loads((OUT / "replay_S2.json").read_text())["student_threshold"])
    fj = json.loads((OUT / "fusion_frozen_corrected_r3_S2.json").read_text())
    fused_thr = float(fj["threshold"])

    scorer = r4.FrozenScorerR4(arm="S2", backend="graph", artdir=OUT,
                               scaler_path=OUT / "scaler_corrected_r3.npz",
                               fusion_path=OUT / "fusion_frozen_corrected_r3_S2.json")
    torch.cuda.reset_peak_memory_stats()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    if args.warmup_episodes:
        for _ in range(max(1, args.warmup_episodes // 5)):
            for mode in modes:
                for variant in variants:
                    one_pass(mode=mode, variant=variant, repeat=-1, collect=False,
                             ep_ids=ep_ids, by_ep_rows=by_ep_rows, tok=tok, scorer=scorer,
                             base_logits_all=base_logits_all, base_of_row=base_of_row,
                             labels=labels, saved_stud=saved_stud, saved_fused=saved_fused,
                             idx_all=idx_all, s2_thr=s2_thr, fused_thr=fused_thr)
    results = []
    for rep in range(args.repeats):
        for variant in variants:
            for mode in modes:
                rec = one_pass(mode=mode, variant=variant, repeat=rep, collect=True,
                               ep_ids=ep_ids, by_ep_rows=by_ep_rows, tok=tok, scorer=scorer,
                               base_logits_all=base_logits_all, base_of_row=base_of_row,
                               labels=labels, saved_stud=saved_stud, saved_fused=saved_fused,
                               idx_all=idx_all, s2_thr=s2_thr, fused_thr=fused_thr)
                results.append(rec)
                if args.save_npz:
                    p = (OUT / args.art_name / f"scores_{mode}_{variant}_p{args.process_id}"
                         f"_r{rep}.npz")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        p, episode_ids=np.asarray(rec["episode_ids"]),
                        per_seed=np.asarray(rec["episodes_per_seed_max"], dtype=np.float32),
                        ensemble=np.asarray(rec["episode_ensemble"], dtype=np.float32),
                        fused=np.asarray(rec["episode_fused"], dtype=np.float64),
                        base=np.asarray(rec["episode_base"], dtype=np.float32),
                        threshold=np.float64(fused_thr),
                        s2_threshold=np.float64(s2_thr),
                        labels=labels)
                    rec.pop("episodes_per_seed_max", None)
                    rec.pop("episode_ensemble", None)
                    rec.pop("episode_fused", None)
                    rec.pop("episode_base", None)
                    rec.pop("episode_ids", None)
    out = {"process_id": args.process_id, "repeats": results,
           "threads": args.threads,
           "gpu_peak_alloc_bytes": int(torch.cuda.max_memory_allocated()),
           "gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
           "cpu_maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
           "graph_cap_counts": {str(k): v for k, v in scorer.net.cap_counts.items()},
           "graph_fallback": scorer.net.n_fallback}
    for mode in modes:
        for variant in variants:
            rs = [r for r in results if r["mode"] == mode and r["variant"] == variant]
            for key in ("detector_p50_ms", "detector_p95_ms", "detector_p99_ms",
                        "detector_int_p50_ms", "detector_int_p95_ms", "detector_int_p99_ms",
                        "full_p50_ms", "full_p95_ms", "full_p99_ms",
                        "throughput_per_s", "wall_throughput_per_s"):
                vals = [r[key] for r in rs if key in r]
                if vals:
                    out[f"{mode}_{variant}_{key}_median"] = float(statistics.median(vals))
            if rs:
                out[f"{mode}_{variant}_detector_p95_ms_worst"] = float(
                    max(r["detector_p95_ms"] for r in rs))
                out[f"{mode}_{variant}_detector_int_p95_ms_worst"] = float(
                    max(r["detector_int_p95_ms"] for r in rs))
                out[f"{mode}_{variant}_full_p95_ms_worst"] = float(
                    max(r["full_p95_ms"] for r in rs))
    for key in ("A_gpu", "A_sync", "R_gpu", "R_sync", "T_gpu", "T_sync"):
        v = out.get(f"{key}_detector_p95_ms_worst")
        out[f"{key}_cost_passed_1ms"] = bool(v is not None and v <= 1.0)
    path = Path(args.out or (OUT / args.art_name / f"cost_r4_p{args.process_id}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "repeats"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
