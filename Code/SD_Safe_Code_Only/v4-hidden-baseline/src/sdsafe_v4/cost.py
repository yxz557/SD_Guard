"""Deployment-increment cost of the early-layer arm (batch 1, streaming, real rows).

Boundary of the claim
---------------------
The 16 GiB Qwen3-8B cannot be executed end-to-end on this 12 GiB GPU, so a full
"main-model + hook" wall-clock experiment is **impossible here** (v3 recorded the same
physical limit).  What is measured, and all that is claimed, is:

* the *added* operations on top of the main model's own forward -- the early-layer tensor
  it already produces must be copied out at the CT boundary, projected, scored by the three
  fused per-seed heads and merged into the running episode max;
* the accepted baseline path on the same rows for a paired increment;
* and, separately, the real hook cost inside the real prefix forward (index-select of the
  endpoint rows + host copy), measured on the actual capture pipeline.

Host path = includes the real memmap read; device path = the same ops with the row already
resident on the GPU (host read + H2D removed).  Batch 1, streaming, `cuda.synchronize`
inside every timed step, 100 warm-ups, 600 CT events, three independent processes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import ART, CAPTURE_BLOCKS, HIDDEN, TRAIN_SEEDS
from .common import atomic_write_json, json_load
from .train_eval import HIDDEN_W, load_split


def _torch():
    import torch

    return torch


def _fused_heads(arm: str, tag: str) -> "Any":
    """Stack the three per-seed first layers into one matmul (the v3 fused-batch idea)."""
    torch = _torch()
    dims = {"C": HIDDEN, "E": 2 * HIDDEN, "F": 3 * HIDDEN}
    w1 = []
    b1 = []
    w2 = []
    b2 = []
    for seed in TRAIN_SEEDS:
        state = torch.load(ART / "models" / tag / f"{arm}_s{seed}.pt", weights_only=False)["state"]
        w1.append(state["0.weight"])
        b1.append(state["0.bias"])
        w2.append(state["3.weight"])
        b2.append(state["3.bias"])
    return {
        "w1": torch.stack(w1).contiguous(),
        "b1": torch.stack(b1).contiguous(),
        "w2": torch.stack(w2).contiguous(),
        "b2": torch.stack(b2).contiguous(),
        "dim": dims[arm],
    }


def _arm_input(data: dict[str, Any], arm: str, idx: int, torch) -> Any:
    parts = []
    if arm in ("C", "F"):
        parts.append(np.asarray(data["final"][idx : idx + 1]))
    if arm in ("E", "F"):
        parts.append(np.asarray(data["early"][idx : idx + 1]))
    raw = np.concatenate(parts, axis=1)
    return torch.from_numpy(raw.view(np.uint16)).view(torch.bfloat16)


def run(rep: int, events: int = 600, warmup: int = 100, device_only: bool = True) -> dict[str, Any]:
    torch = _torch()
    data = load_split("dev")
    base_bank = _fused_heads("C", "full")
    cand_bank = _fused_heads("F", "full")
    n = data["row_index"].shape[0]
    rng = np.random.default_rng(20261113 + rep)
    idxs = rng.choice(n, size=events, replace=False)

    dev = "cuda"

    def bank(b):
        return (
            b["w1"].to(dev).float(),
            b["b1"].to(dev).float(),
            b["w2"].squeeze(1).to(dev).float(),
            b["b2"].squeeze(1).to(dev).float(),
        )

    w1b, b1b, w2b, b2b = bank(base_bank)
    w1c, b1c, w2c, b2c = bank(cand_bank)

    need = min(warmup + events, events)

    def prep(arm: str):
        rows = [_arm_input(data, arm, int(i), torch) for i in idxs[:need]]
        return [r.to(dev).contiguous() for r in rows]

    resident = {"C": prep("C"), "F": prep("F")}

    def forward(x, w1, b1, w2, b2):
        # x: [1, dim]; w1: [heads, 256, dim]; b1: [heads, 256]; w2: [heads, 256]
        xb = x.expand(w1.shape[0], -1).unsqueeze(-1)          # [heads, dim, 1]
        h = torch.nn.functional.gelu(torch.bmm(w1, xb).squeeze(-1) + b1)  # [heads, 256]
        s = (h * w2).sum(dim=1) + b2                          # [heads]
        return s.max()

    def step(arm: str, i: int, res: bool):
        if res:
            x = resident[arm][i % need].float()
        else:
            x = _arm_input(data, arm, int(idxs[i]), torch).to(dev).float()
        if arm == "C":
            return forward(x, w1b, b1b, w2b, b2b)
        return forward(x, w1c, b1c, w2c, b2c)

    out: dict[str, Any] = {"rep": rep, "events": events, "device_only": device_only}
    for name, arm in (("base", "C"), ("cand", "F")):
        for _ in range(warmup):
            step(arm, 0, device_only)
        torch.cuda.synchronize()
        times = []
        run_max = -1e30
        for i in range(events):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            s = step(arm, i, device_only)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
            run_max = max(run_max, float(s.max()))
        arr = np.asarray(times)
        out[name] = {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean()),
            "score_head_max": run_max,
        }
    out["increment_p95_ms"] = out["cand"]["p95_ms"] - out["base"]["p95_ms"]
    out["increment_p50_ms"] = out["cand"]["p50_ms"] - out["base"]["p50_ms"]
    out["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated())
    return out


def hook_cost(rep: int = 0, repeats: int = 5) -> dict[str, Any]:
    """Real cost of taking the early-layer vector out of a *live* prefix forward.

    Measured inside the real capture pipeline: index-select the CT endpoint rows out of the
    block-8 tensor that the forward just produced and copy them to host bf16.  The forward
    itself is timed for reference only (it is the main model's own work, not our increment).
    """
    import os
    import time as _t

    from .prefix import PrefixModel

    torch = _torch()
    model = PrefixModel(blocks=8, captures=CAPTURE_BLOCKS)
    prepared_dir = os.environ.get("SDSAFE_PREPARED_DIR")
    if prepared_dir:
        import json
        from pathlib import Path

        from .prepared import load_episode

        index = json.loads((Path(prepared_dir) / "index.json").read_text())
        rounds = []
        for e, name in index.items():
            ep = load_episode(Path(prepared_dir) / name)
            if ep.scope in ("ctx40k", "agentlab"):
                rounds.extend(ep.rounds)
            if len(rounds) >= 3:
                break
    else:
        from .seqsrc import RowIndex, iter_round_tasks

        row_index = RowIndex("dev")
        ids_wanted = [e for e in row_index.episode_ids if row_index.episode_scope[e] in ("ctx40k", "agentlab")][:3]
        rounds = [(task, ids) for task, ids in iter_round_tasks(row_index, ids_wanted)]
    model.forward_ids(rounds[0][1])
    torch.cuda.synchronize()
    fwd_ms, cut_ms = [], []
    for task, ids in rounds[:repeats]:
        torch.cuda.synchronize()
        t0 = _t.perf_counter()
        caps = model.forward_ids(ids)
        torch.cuda.synchronize()
        fwd_ms.append((_t.perf_counter() - t0) * 1e3)
        idx = torch.tensor(task.positions, dtype=torch.long, device="cuda")
        torch.cuda.synchronize()
        t0 = _t.perf_counter()
        block = caps[8]
        rows = block[0].index_select(0, idx)
        for b in CAPTURE_BLOCKS:
            _ = caps[b][0].index_select(0, idx).to(dtype=torch.bfloat16, device="cpu")
        torch.cuda.synchronize()
        cut_ms.append((_t.perf_counter() - t0) * 1e3)
    del model
    torch.cuda.empty_cache()
    return {
        "rep": rep,
        "rounds": len(fwd_ms),
        "sequence_lengths": [len(ids) for _task, ids in rounds[:repeats]],
        "ct_endpoints": [len(task.positions) for task, _ids in rounds[:repeats]],
        "prefix_forward_ms": fwd_ms,
        "hook_cutout_ms_per_event": [c / max(len(task.positions), 1) for (task, _i), c in
                                     zip(rounds[:repeats], cut_ms)],
        "note": "the prefix forward is the main model's own work offline; the increment is the cut-out",
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--events", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--hook", action="store_true")
    args = ap.parse_args()
    if args.hook:
        target = ART / "cost" / f"hook_rep{args.rep}.json"
        atomic_write_json(target, hook_cost(args.rep))
        print(json.dumps(json_load(target), indent=2))
        raise SystemExit(0)
    res = {"host": run(args.rep, args.events, device_only=False), "device": run(args.rep, args.events, device_only=True)}
    target = Path(args.out) if args.out else ART / "cost" / f"rep{args.rep}.json"
    atomic_write_json(target, res)
    print(json.dumps(res, indent=2))
