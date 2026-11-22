"""Deployment increment of the depth arm, measured on the real 4090.

Claim boundary: the added cost is the increment on top of the main model's own forward --
the already-computed activation is read at the CT endpoint (hook/select), standardized,
scored by the three fused per-seed heads, and merged into the running episode max.  The
36-block backbone replay is measured separately (``cost_full36.py``); no end-to-end
main-model wall clock is invented here.

Host path = real bf16 memmap read + decode + H2D; device path = the same rows already
resident.  Batch 1, streaming, ``cuda.synchronize`` inside every timed step, 100 warm-ups,
600 CT events, one independent process per repetition.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import ACCEPTED_FEATURE_BIN, ART, HIDDEN, TRAIN_SEEDS
from .common import atomic_write_json, json_load
from .loader import bf16_bits_to_f32

DS = Path("/dev/shm/sdsafe-depth-v5/dataset")
OUT = ART / "cost"


def _torch():
    import torch

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return torch


def _bank(arm: str, tag: str) -> dict[str, Any]:
    torch = _torch()
    w1, b1, w2, b2 = [], [], [], []
    for seed in TRAIN_SEEDS:
        st = torch.load(ART / "models_depth" / tag / f"{arm}_s{seed}.pt", weights_only=False)["state"]
        w1.append(st["0.weight"])
        b1.append(st["0.bias"])
        w2.append(st["3.weight"].squeeze(0))
        b2.append(st["3.bias"].squeeze(0))
    return {"w1": torch.stack(w1).contiguous(), "b1": torch.stack(b1).contiguous(),
            "w2": torch.stack(w2).contiguous(), "b2": torch.stack(b2).contiguous()}


def _to_dev(bank, dev):
    return tuple(bank[k].to(dev).float() for k in ("w1", "b1", "w2", "b2"))


def _forward(x, w1, b1, w2, b2):
    import torch
    import torch.nn.functional as F

    xb = x.expand(w1.shape[0], -1).unsqueeze(-1)
    h = F.gelu(torch.bmm(w1, xb).squeeze(-1) + b1)
    s = (h * w2).sum(dim=1) + b2
    return s.max()


def run(rep: int, events: int = 600, warmup: int = 100) -> dict[str, Any]:
    torch = _torch()
    dev = "cuda"
    meta = np.load(DS / "dev" / "meta.npz", allow_pickle=True)
    final = np.load(DS / "dev" / "final_bf16.npy", mmap_mode="r")
    b16 = np.load(DS / "dev" / "block16_bf16.npy", mmap_mode="r")
    stats = json_load(ART / "dataset" / "train_stats.json")["blocks"]
    mu_f = torch.from_numpy(np.asarray(stats["final"]["mu"], dtype=np.float32)).to(dev)
    sg_f = torch.from_numpy(np.asarray(stats["final"]["sigma"], dtype=np.float32)).to(dev)
    mu_b = torch.from_numpy(np.asarray(stats["b16"]["mu"], dtype=np.float32)).to(dev)
    sg_b = torch.from_numpy(np.asarray(stats["b16"]["sigma"], dtype=np.float32)).to(dev)
    n = final.shape[0]
    rng = np.random.default_rng(20261114 + rep)
    idxs = rng.choice(n, size=events, replace=False)
    need = min(warmup + events, events)
    use = idxs[:need]
    bank_c = _to_dev(_bank("Craw", "full"), dev)
    bank_h = _to_dev(_bank("H", "full"), dev)

    # pre-decode host bytes (so the host path times only decode+scaler+H2D+head)
    host_c = [np.asarray(final[int(i)]).copy() for i in use]
    host_h = [(np.asarray(final[int(i)]).copy(), np.asarray(b16[int(i)]).copy()) for i in use]
    # device-resident float rows, pre-scaled
    dev_c = [torch.from_numpy(bf16_bits_to_f32(h)).to(dev) for h in host_c]
    dev_h = [torch.cat([torch.from_numpy(bf16_bits_to_f32(hf)), torch.from_numpy(bf16_bits_to_f32(hb))]).to(dev)
             for hf, hb in host_h]
    dev_h_std = [torch.cat([(x[:HIDDEN] - mu_f) / sg_f, (x[HIDDEN:] - mu_b) / sg_b]) for x in dev_h]

    def step_base_host(i, runmax):
        raw = bf16_bits_to_f32(host_c[i % need])
        x = torch.from_numpy(raw).to(dev).float().unsqueeze(0)
        s = _forward(x, *bank_c)
        return torch.maximum(runmax, s)

    def step_cand_host(i, runmax):
        hf, hb = host_h[i % need]
        xf = (torch.from_numpy(bf16_bits_to_f32(hf)).to(dev).float() - mu_f) / sg_f
        xb = (torch.from_numpy(bf16_bits_to_f32(hb)).to(dev).float() - mu_b) / sg_b
        x = torch.cat([xf, xb]).unsqueeze(0)
        s = _forward(x, *bank_h)
        return torch.maximum(runmax, s)

    def step_base_dev(i, runmax):
        s = _forward(dev_c[i % need].unsqueeze(0), *bank_c)
        return torch.maximum(runmax, s)

    def step_cand_dev(i, runmax, scaled=True):
        x = dev_h_std[i % need] if scaled else dev_h[i % need]
        s = _forward(x.unsqueeze(0), *bank_h)
        return torch.maximum(runmax, s)

    def timeit(fn, i, runmax):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(i, runmax)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3, r

    out: dict[str, Any] = {"rep": rep, "events": events, "warmup": warmup,
                           "device": torch.cuda.get_device_name(0)}
    for name, fn in (("base_host", step_base_host), ("cand_host", step_cand_host),
                     ("base_dev", step_base_dev), ("cand_dev", step_cand_dev)):
        runmax = torch.tensor(-1e9, device=dev)
        for i in range(warmup):
            runmax = fn(i, runmax)
        lat = []
        for i in range(events):
            dt, runmax = timeit(fn, i, runmax)
            lat.append(dt)
        a = np.asarray(lat)
        out[name] = {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
                     "mean_ms": float(a.mean()), "max_ms": float(a.max()),
                     "thrpt_per_s": float(1000.0 / a.mean())}
    for tag in ("host", "dev"):
        out[f"increment_{tag}_p95_ms"] = out[f"cand_{tag}"]["p95_ms"] - out[f"base_{tag}"]["p95_ms"]
        out[f"increment_{tag}_mean_ms"] = out[f"cand_{tag}"]["mean_ms"] - out[f"base_{tag}"]["mean_ms"]

    # ---- real hook/select/host-copy cost inside a real prefix forward
    from .prepared import load_episode
    from .prefix import PrefixModel

    prep = Path("/root/sdsafe-compute/v5-depth/prepared")
    index = json.loads((prep / "dev" / "index.json").read_text())
    # pick a DEV episode with several endpoints
    pick = None
    for eid, name in sorted(index.items()):
        ep = load_episode(prep / "dev" / name)
        if ep.rounds and len(ep.rounds[0][0].positions) >= 2:
            pick = (eid, ep)
            break
    prefix = PrefixModel(blocks=24, captures=(16, 24))
    torch.cuda.synchronize()
    hook = {}
    if pick is not None:
        eid, ep = pick
        r, ids = ep.rounds[0]
        cap = prefix.forward_positions(ids, r.positions)
        lat_sel, lat_scale, lat_head = [], [], []
        bank = _to_dev(_bank("H", "full"), dev)
        acc = np.memmap(ACCEPTED_FEATURE_BIN, dtype="<u2", mode="r")
        flat = acc.reshape(acc.shape[0] // HIDDEN, HIDDEN)
        fin_gpu = torch.from_numpy(bf16_bits_to_f32(np.asarray(
            flat[[int(e["cache_row"]) for e in r.events]]))).to(dev)
        for _ in range(5):
            for pos_i in range(len(r.positions)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                v16 = cap[16][pos_i]
                torch.cuda.synchronize()
                lat_sel.append((time.perf_counter() - t0) * 1e3)
                t0 = time.perf_counter()
                xf = (fin_gpu[pos_i] - mu_f) / sg_f
                xb = (v16.float() - mu_b) / sg_b
                x = torch.cat([xf, xb]).unsqueeze(0)
                torch.cuda.synchronize()
                lat_scale.append((time.perf_counter() - t0) * 1e3)
                t0 = time.perf_counter()
                _forward(x, *bank)
                torch.cuda.synchronize()
                lat_head.append((time.perf_counter() - t0) * 1e3)
        hook = {"episode": eid, "sequence_length": len(ids), "n_endpoints": len(r.positions),
                "candidate_arm": "H = [final|block16]",
                "select_p95_ms": float(np.percentile(lat_sel, 95)),
                "scaler_p95_ms": float(np.percentile(lat_scale, 95)),
                "head_p95_ms": float(np.percentile(lat_head, 95))}
        hook["total_p95_ms"] = hook["select_p95_ms"] + hook["scaler_p95_ms"] + hook["head_p95_ms"]
    out["real_hook"] = hook
    out["gate"] = {"added_p95_ms_max": 1.0,
                   "pass_host_p95": bool(out["increment_host_p95_ms"] <= 1.0),
                   "pass_dev_p95": bool(out["increment_dev_p95_ms"] <= 1.0)}
    prefix.close()
    del prefix
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--events", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = run(args.rep, args.events, args.warmup)
    atomic_write_json(OUT / f"depth_rep{args.rep}.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
