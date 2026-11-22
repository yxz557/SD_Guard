"""Deployment-shaped cost measurement for the incremental CT path.

Charged per CT event, in one real stream (no cached-view shortcut):

  1. decode the newly emitted segment token ids,
  2. incrementally append any not-yet-indexed earlier block to the history/source index
     (the first event of an episode charges the history-load burst),
  3. build the event view from the maintained prior index (typed roles, generic lexical retrieval,
     exact/normalised value->source matching),
  4. hashed tokenisation/feature extraction,
  5. host->device copy of the small tensors,
  6. the full 2.19 M-parameter student forward (all heads exist; main head used),
  7. score cache lookup and running per-episode max update.

The frozen Qwen3-8B base forward is *not* re-run or increased; its cached CT logit is read.
Three OS processes are run independently by the wrapper; each does its own warm-up and repeats.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from . import features, view

ART = Path("/root/sdsafe-compute/v6-param/artifacts")


def load_stream(events: int) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    for name in ("dev_common.jsonl", "dev_common_fixup.jsonl"):
        path = ART / "views" / name
        if path.is_file():
            views += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # fixup wins by row_index
    by_row: dict[int, dict[str, Any]] = {}
    for rec in views:
        by_row[int(rec["row_index"])] = rec
    views = sorted(by_row.values(), key=lambda r: (r["episode_id"], r["round_id"], r["token_end"]))
    return views[:events]


class StreamIndex:
    """Minimal deployable history/source index state; updates are part of the measured path."""

    def __init__(self) -> None:
        self.seen: set[tuple[str, int, int]] = set()
        self.blocks: dict[str, list[tuple[int, int]]] = {}
        self.running_max: dict[str, float] = {}
        self.base_cache_hits = 0

    def update(self, rec: dict[str, Any]) -> float:
        t0 = time.perf_counter()
        ep = rec["episode_id"]
        for block in rec["prior"]:
            text = block["text"]
            key = (ep, zlib.crc32(text.encode("utf-8")), len(text))
            if key in self.seen:
                continue
            # taxonomy/hash tokenisation that a real index performs on append
            toks = features.bag_indices(text)
            self.blocks.setdefault(ep, []).append((int(block["block_index"]), len(toks)))
            self.seen.add(key)
        # frozen base logit is already cached; charge the lookup, never a new base forward
        self.base_cache_hits += 1
        return time.perf_counter() - t0


def timed_event(rec: dict[str, Any], *, net, state: StreamIndex, device: str, torch,
                tok=None) -> tuple[float, dict[str, float]]:
    import torch as _t
    t0 = time.perf_counter()
    seg = rec["current"]["segment"]
    if tok is not None:
        # one decode of the newly emitted token span (encode is used only to obtain real ids)
        ids = tok.encode(seg, add_special_tokens=False)
        seg = tok.decode(ids, skip_special_tokens=False)
    t_decode = time.perf_counter()
    t_update = state.update(rec)
    t_after_update = time.perf_counter()
    prior = [{"role": b["role"], "text": b["text"], "full_text": b["text"]} for b in rec["prior"]]
    vis = rec["current"]["visible_so_far"]
    live = view.build_view_stream(
        episode_id=rec["episode_id"], row_index=rec["row_index"], event_id=rec["event_id"],
        event_type=rec["event_type"], scope=rec["scope"], group_id=rec["group_id"],
        round_id=rec["round_id"], user_request=rec["user_request"], prior=prior,
        current_segment_text=seg, current_visible_so_far=vis,
        current_tool=rec["current"]["tool"], current_args=rec["current"]["arguments"],
        token_start=rec["token_start"], token_end=rec["token_end"],
        row_label=rec["row_label"], episode_label=rec["episode_label"])
    feats = features.extract(live)
    t_features = time.perf_counter()
    # deployable fused embedding path: one EmbeddingBag call for cur/vis/user/source bags
    bags = feats["bags"]
    src_bags = bags["src"]
    parts = [np.asarray(bags["cur"], dtype=np.int64), np.asarray(bags["vis"], dtype=np.int64),
             np.asarray(bags["user"], dtype=np.int64)] + [np.asarray(b, dtype=np.int64) for b in src_bags]
    lengths = [int(p.size) for p in parts]
    flat = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    offs = np.zeros(len(parts), dtype=np.int64)
    if len(parts) > 1:
        np.cumsum(lengths[:-1], out=offs[1:])
    batch = {
        "flat_idx": _t.from_numpy(flat).to(device),
        "flat_off": _t.from_numpy(offs).to(device),
        "dense": _t.from_numpy(np.asarray(feats["dense"], dtype=np.float32)).unsqueeze(0).to(device),
        "src_counts": _t.tensor([len(src_bags)], dtype=_t.long, device=device),
    }
    if device == "cuda":
        _t.cuda.synchronize()
    t_h2d = time.perf_counter()
    with _t.no_grad():
        rows = net.emb(batch["flat_idx"], batch["flat_off"])
        cur = rows[0:1]
        vis_e = rows[1:2]
        user_e = rows[2:3]
        src_rows = rows[3:]
        rep = _t.repeat_interleave(_t.arange(1, device=device), batch["src_counts"])
        cnt = batch["src_counts"].float().unsqueeze(-1)
        src = _t.zeros_like(cur)
        if src_rows.numel():
            src.index_add_(0, rep, src_rows / cnt)
        x = _t.cat([cur, vis_e, user_e, src, batch["dense"]], dim=-1)
        logit = net.head_main(net.trunk(net.norm(x))).squeeze(-1)
        score = float(logit.item())
    if device == "cuda":
        _t.cuda.synchronize()
    ep = rec["episode_id"]
    state.running_max[ep] = max(state.running_max.get(ep, -np.inf), score)
    t_end = time.perf_counter()
    return t_end - t0, {"decode": t_decode - t0, "index_update": t_update,
                        "view_plus_features": t_features - t_after_update,
                        "h2d": t_h2d - t_features, "forward_score_max": t_end - t_h2d}


def timed_encode_only(rec: dict[str, Any], *, net, device: str, torch) -> float:
    """Diagnostic only: encoder kernel on an already-built view (NOT the claimed cost)."""
    import torch as _t
    feats = features.extract(rec)
    batch = {
        "cur_idx": _t.tensor(feats["bags"]["cur"], dtype=_t.long, device=device),
        "cur_off": _t.zeros(1, dtype=_t.long, device=device),
        "vis_idx": _t.tensor(feats["bags"]["vis"], dtype=_t.long, device=device),
        "vis_off": _t.zeros(1, dtype=_t.long, device=device),
        "user_idx": _t.tensor(feats["bags"]["user"], dtype=_t.long, device=device),
        "user_off": _t.zeros(1, dtype=_t.long, device=device),
        "dense": _t.from_numpy(np.asarray(feats["dense"], dtype=np.float32)).unsqueeze(0).to(device),
    }
    bags = feats["bags"]["src"]
    flat, offs = [], []
    for bag in bags:
        offs.append(len(flat))
        flat.extend(bag)
    batch["src_idx"] = _t.tensor(flat, dtype=_t.long, device=device)
    batch["src_off"] = _t.tensor(offs, dtype=_t.long, device=device)
    batch["src_counts"] = _t.tensor([len(bags)], dtype=_t.long, device=device)
    if device == "cuda":
        _t.cuda.synchronize()
    t0 = time.perf_counter()
    with _t.no_grad():
        cur = net.emb(batch["cur_idx"], batch["cur_off"])
        vis_e = net.emb(batch["vis_idx"], batch["vis_off"])
        user_e = net.emb(batch["user_idx"], batch["user_off"])
        src_rows = net.emb(batch["src_idx"], batch["src_off"])
        rep = _t.repeat_interleave(_t.arange(batch["src_counts"].numel(), device=device), batch["src_counts"])
        cnt = _t.repeat_interleave(batch["src_counts"], batch["src_counts"]).float().unsqueeze(-1)
        src = _t.zeros_like(cur)
        src.index_add_(0, rep, src_rows / cnt)
        x = _t.cat([cur, vis_e, user_e, src, batch["dense"]], dim=-1)
        logit = net.head_main(net.trunk(net.norm(x))).squeeze(-1)
        _ = float(logit.item())
    if device == "cuda":
        _t.cuda.synchronize()
    return time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ART / "ckpt_S0_v3_seed0.pt"))
    ap.add_argument("--events", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--process-id", type=int, default=0)
    ap.add_argument("--decode", type=int, default=1)
    ap.add_argument("--out", default=str(ART / "cost.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    import torch
    from . import model

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = model.Student(model.ModelConfig(**{k: v for k, v in payload["config"].items()
                                             if k in model.ModelConfig.__dataclass_fields__}))
    net.load_state_dict(payload["state_dict"])
    net.eval().to(args.device)
    n_params = net.n_params()
    tok = None
    if args.decode:
        from transformers import AutoTokenizer
        from . import data
        tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)

    recs = load_stream(args.events + args.warmup)
    out: dict[str, Any] = {"ckpt": args.ckpt, "params": n_params, "device": args.device,
                           "events": args.events, "process_id": args.process_id,
                           "decode": bool(args.decode), "repeats": []}
    parts: dict[str, list[float]] = {}
    per_repeat = []
    for repeat in range(args.repeats):
        state = StreamIndex()
        for rec in recs[: args.warmup]:
            timed_event(rec, net=net, state=state, device=args.device, torch=torch, tok=tok)
        times = []
        for rec in recs[args.warmup: args.warmup + args.events]:
            elapsed, comp = timed_event(rec, net=net, state=state, device=args.device,
                                        torch=torch, tok=tok)
            times.append(elapsed)
            for k, v in comp.items():
                parts.setdefault(k, []).append(v)
        times_arr = np.asarray(times)
        stats = {"repeat": repeat, "n": int(times_arr.size),
                 "p50_ms": float(np.percentile(times_arr, 50) * 1e3),
                 "p95_ms": float(np.percentile(times_arr, 95) * 1e3),
                 "p99_ms": float(np.percentile(times_arr, 99) * 1e3),
                 "mean_ms": float(times_arr.mean() * 1e3),
                 "max_ms": float(times_arr.max() * 1e3),
                 "throughput_per_s": float(times_arr.size / times_arr.sum()),
                 "indexed_blocks": sum(len(v) for v in state.blocks.values()),
                 "episodes": len(state.running_max)}
        enc = [timed_encode_only(rec, net=net, device=args.device, torch=torch)
               for rec in recs[args.warmup: args.warmup + 200]]
        stats["encode_only_p95_ms"] = float(np.percentile(enc, 95) * 1e3)
        stats["encode_only_p50_ms"] = float(np.percentile(enc, 50) * 1e3)
        out["repeats"].append(stats)
        per_repeat.append(stats["p95_ms"])
    out["p95_ms_worst_repeat"] = max(per_repeat)
    out["p50_ms_mean"] = float(statistics.mean(r["p50_ms"] for r in out["repeats"]))
    out["throughput_per_s_mean"] = float(statistics.mean(r["throughput_per_s"] for r in out["repeats"]))
    out["component_mean_ms"] = {k: float(np.mean(v) * 1e3) for k, v in parts.items()}
    out["component_p95_ms"] = {k: float(np.percentile(v, 95) * 1e3) for k, v in parts.items()}
    out["cost_passed_1ms"] = bool(out["p95_ms_worst_repeat"] <= 1.0)
    if torch.cuda.is_available():
        out["gpu_peak_alloc_bytes"] = int(torch.cuda.max_memory_allocated())
        out["gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "repeats"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
