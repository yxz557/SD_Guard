"""Depth-prefix smoke: memory / throughput / exactness for block 16 and block 24.

One invocation measures ONE shape in a fresh process (the driver in
``scripts/run_smoke.py`` spawns these), so no previous model or peak counter can leak into
a number:

* ``short``    : a small multi-round TRAIN episode
* ``medium``   : a ~7k-token multi-call TRAIN episode
* ``boundary`` : the longest prepared TRAIN round (the real >35.7k worst case)
* ``equiv``    : monolithic vs KV-chunked prefill at fixed numerical tolerance, plus
                 suffix-invariance / wrong-endpoint / future-token mutations

Everything is read-only: the frozen model is loaded from the /dev/shm mirror, the token
ids come from the CPU-prepared jobs, and nothing is truncated or re-encoded.  ``lm_head``
is never materialised or executed (only decoder blocks 0..B-1), ``model.norm`` is never
applied.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import ART, CAPTURE_BLOCKS, HIDDEN, KV_CHUNK_COS_MIN, KV_CHUNK_MAX_ABS, MODEL_LAYERS
from .common import atomic_write_json, json_load
from .prepared import load_episode
from .prefix import PrefixModel

PREPARED = Path("/root/sdsafe-compute/v5-depth/prepared")
OUT = ART / "smoke_depth"


def _rel(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(b), 1e-6)
    return {
        "max_abs": float(np.abs(a - b).max()),
        "mean_abs": float(np.abs(a - b).mean()),
        "cosine_min": float(
            np.min((a * b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12))
        ),
        "rows": int(a.shape[0]),
    }


def _load_round(split: str, episode_id: str, round_id: int | None):
    index = json.loads((PREPARED / split / "index.json").read_text())
    ep = load_episode(PREPARED / split / index[episode_id])
    rounds = [(r, ids) for r, ids in ep.rounds if round_id is None or r.round_id == round_id]
    if round_id is None:
        r, ids = max(rounds, key=lambda p: len(p[1]))
    else:
        r, ids = rounds[0]
    positions = [int(e["feature_token_index"]) for e in r.events]
    return ep.episode_id, ep.scope, ep.label, r.round_id, ids, positions


def run(blocks: int, shape: str, picks: dict, chunk: int | None, out_path: Path) -> dict:
    import torch

    role = {"equiv": "medium", "equiv_boundary": "boundary"}.get(shape, shape)
    episode_id = picks[role]["episode_id"]
    round_id = picks[role]["round_id"]
    ep_id, scope, label, rid, ids, positions = _load_round("train", episode_id, round_id)
    caps = tuple(sorted(set(c for c in CAPTURE_BLOCKS if c <= blocks) | {8}))
    result: dict = {
        "stage": "depth-smoke",
        "blocks": blocks,
        "shape": shape,
        "episode_id": ep_id,
        "scope": scope,
        "label": bool(label),
        "round_id": rid,
        "sequence_length": len(ids),
        "n_endpoints": len(positions),
        "chunk": chunk,
        "model_layers_total": MODEL_LAYERS,
        "capture_blocks": list(caps),
    }
    t_load = time.time()
    prefix = PrefixModel(blocks=blocks, captures=caps)
    torch.cuda.synchronize()
    result["model_load_seconds"] = round(time.time() - t_load, 2)
    result["weights_allocated_bytes"] = int(torch.cuda.memory_allocated())
    result["weights_reserved_bytes"] = int(torch.cuda.memory_reserved())
    result["norm_is_meta"] = bool(prefix.model.norm.weight.device.type == "meta")
    result["lm_head_present"] = any("lm_head" in n for n, _ in prefix.model.named_modules())
    torch.cuda.reset_peak_memory_stats()
    # one throwaway forward so kernel selection is not inside the timed run
    prefix.forward_positions(ids[: min(len(ids), 96)], [min(len(ids), 96) - 1])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    mon = prefix.forward_positions(ids, positions)
    torch.cuda.synchronize()
    dt = time.time() - t0
    result["monolithic"] = {
        "seconds": round(dt, 4),
        "tokens_per_second": round(len(ids) / max(dt, 1e-9), 1),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "captured_shapes": {str(k): list(map(int, v.shape)) for k, v in mon.items()},
    }
    mon_np = {b: v.float().cpu().numpy() for b, v in mon.items()}
    del mon
    torch.cuda.empty_cache()
    # determinism of the monolithic path itself (identical input -> bitwise identical)
    rep = prefix.forward_positions(ids, positions)
    torch.cuda.synchronize()
    det = {str(b): float(np.abs(rep[b].float().cpu().numpy() - mon_np[b]).max()) for b in rep}
    result["determinism"] = det
    result["determinism_pass"] = all(v == 0.0 for v in det.values())
    del rep

    if chunk:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ch = prefix.forward_positions(ids, positions, chunk=chunk)
        torch.cuda.synchronize()
        dtc = time.time() - t0
        result["chunked"] = {
            "seconds": round(dtc, 4),
            "tokens_per_second": round(len(ids) / max(dtc, 1e-9), 1),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        eq = {}
        for b in ch:
            if b == 8:
                continue
            eq[str(b)] = _rel(ch[b].float().cpu().numpy(), mon_np[b])
        result["equivalence"] = eq
        result["equivalence_pass"] = all(
            v["cosine_min"] >= KV_CHUNK_COS_MIN and v["max_abs"] <= KV_CHUNK_MAX_ABS for v in eq.values()
        )
        del ch
        torch.cuda.empty_cache()

    if shape == "equiv":
        # ---- suffix invariance: full sequence endpoint == truncated forward endpoint
        p = positions[0]
        cut = prefix.forward_positions(ids[: p + 1], [p])
        suffix = {str(b): _rel(cut[b].float().cpu().numpy(), mon_np[b][:1]) for b in cut}
        result["suffix_invariance"] = suffix
        # bitwise equality is *not* expected: a shorter sequence selects a different SDPA
        # reduction order inside bf16.  The test is that a future token cannot change the
        # endpoint beyond the pre-set dtype tolerance, while a wrong endpoint is far outside it.
        result["suffix_invariance_pass"] = all(
            v["cosine_min"] >= KV_CHUNK_COS_MIN and v["max_abs"] <= KV_CHUNK_MAX_ABS
            for v in suffix.values())
        del cut
        # ---- wrong endpoint: p-1 must differ from p
        if p >= 1:
            two = prefix.forward_positions(ids[: p + 1], [p - 1, p])
            wrong = {}
            for b in two:
                v = two[b].float().cpu().numpy()
                wrong[str(b)] = {
                    "row0_vs_row1_max_abs": float(np.abs(v[0] - v[1]).max()),
                    "row1_matches_full_max_abs": float(np.abs(v[1] - mon_np[b][0]).max()),
                }
            result["wrong_endpoint"] = wrong
            result["wrong_endpoint_pass"] = all(
                v["row0_vs_row1_max_abs"] > 1.0 and v["row1_matches_full_max_abs"] <= KV_CHUNK_MAX_ABS
                for v in wrong.values()
            )
            del two
        # ---- capture block outside the executed prefix must be refused
        try:
            PrefixModel(blocks=blocks, captures=(blocks + 8,), verify_keys=True)
            result["outside_prefix_rejected"] = False
        except Exception as exc:  # noqa: BLE001
            result["outside_prefix_rejected"] = True
            result["outside_prefix_error"] = f"{type(exc).__name__}: {exc}"
        # ---- no future: truncating after a later token must not change the endpoint row
        if len(positions) >= 2 and positions[1] > p:
            later = prefix.forward_positions(ids[: positions[1] + 1], [p])
            fut = {str(b): _rel(later[b].float().cpu().numpy(), mon_np[b][:1]) for b in later}
            result["no_future_information"] = fut
            result["no_future_pass"] = all(
                v["cosine_min"] >= KV_CHUNK_COS_MIN and v["max_abs"] <= KV_CHUNK_MAX_ABS
                for v in fut.values())

    result["peak_allocated_bytes_overall"] = int(torch.cuda.max_memory_allocated())
    atomic_write_json(out_path, result)
    prefix.close()
    del prefix
    torch.cuda.empty_cache()
    print(json.dumps({k: v for k, v in result.items() if k not in ("wrong_endpoint",)}, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--shape", required=True, choices=("short", "medium", "boundary", "equiv", "equiv_boundary"))
    ap.add_argument("--picks", required=True)
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    picks = json_load(args.picks)
    run(args.blocks, args.shape, picks, args.chunk, Path(args.out))


if __name__ == "__main__":
    main()
