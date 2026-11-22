"""Paired full-36-block backbone replay on the 4090 (no lm_head, no token logits).

Measures the *shared-backbone* cost of the deployment hook: the same complete frozen
Qwen3Model forward run with and without the block-16/24 endpoint hook.  Everything is the
original graph; ``lm_head`` is never loaded.  Short and boundary-length rounds are attempted
honestly: an OOM is recorded as an OOM, not hidden.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
from . import ACCEPTED_FEATURE_BIN, ART, HIDDEN
from .common import atomic_write_json, json_load
from .prepared import load_episode
from .prefix import PrefixModel

PREP = Path("/root/sdsafe-compute/v5-depth/prepared")
OUT = ART / "cost"


def pick(shape: str):
    idx = json.loads((PREP / "dev" / "index.json").read_text())
    best = None
    for eid, name in sorted(idx.items()):
        ep = load_episode(PREP / "dev" / name)
        for r, ids in ep.rounds:
            L = len(ids)
            if shape == "short" and 200 <= L <= 600:
                return eid, r, ids
            if shape == "boundary" and (best is None or L > best[3]):
                best = (eid, r, ids, L)
    return None if best is None else best[:3]


def run(hook: bool, shape: str, reps: int = 8) -> dict:
    import torch
    picked = pick(shape)
    if picked is None:
        return {"shape": shape, "status": "no-round"}
    eid, r, ids = picked
    caps = (16, 24) if hook else ()
    out = {"shape": shape, "episode": eid, "round": int(r.round_id), "sequence_length": len(ids),
           "hook": hook, "captures": list(caps)}
    try:
        prefix = PrefixModel(blocks=36, captures=caps)
    except torch.cuda.OutOfMemoryError as exc:
        out.update({"status": "oom_on_load", "error": str(exc)})
        return out
    torch.cuda.synchronize()
    out["weights_allocated_bytes"] = int(torch.cuda.memory_allocated())
    out["lm_head_present"] = any("lm_head" in n for n, _ in prefix.model.named_modules())
    # warmup
    try:
        for _ in range(2):
            prefix.model(input_ids=torch.tensor([list(ids)], dtype=torch.long, device="cuda"),
                         use_cache=False, return_dict=True)
            torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        lat = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prefix.model(input_ids=torch.tensor([list(ids)], dtype=torch.long, device="cuda"),
                         use_cache=False, return_dict=True)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1e3)
        a = np.asarray(lat)
        out.update({"status": "ok", "p50_ms": float(np.percentile(a, 50)),
                    "p95_ms": float(np.percentile(a, 95)), "tokens_per_s": float(len(ids) * 1000 / a.mean()),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())})
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        out.update({"status": "oom_on_forward", "error": str(exc)})
    prefix.close()
    del prefix
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hook", type=int, required=True)
    ap.add_argument("--shape", required=True, choices=("short", "boundary"))
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = run(bool(args.hook), args.shape, args.reps)
    atomic_write_json(OUT / f"full36_{args.shape}_hook{int(args.hook)}.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
