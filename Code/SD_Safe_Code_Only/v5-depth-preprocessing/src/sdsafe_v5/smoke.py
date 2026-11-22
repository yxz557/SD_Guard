"""Two-episode real smoke: prefix execution vs the complete original model.

Produces (all numbers are measured, none assumed):

* GPU peak memory and wall time for the prefix prefill of real TRAIN rounds,
* recorded-token counts,
* bitwise agreement between the early-exit hook path and an independent replay,
* agreement with the **complete 36-layer original model** run on CPU (same weights),
* end-to-end agreement between that complete CPU forward and the *accepted* final-layer
  feature matrix (proves ids + endpoint indices + weights + norm placement),
* CPU fp32 reference error bounds for the bf16 GPU values,
* suffix-invariance: capturing block 4/8 at an intermediate position of a longer round
  equals running the truncated prefix (this is why one forward per round suffices).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import ART, CAPTURE_BLOCKS, HIDDEN, PREFIX_LAYERS, SEQ
from .common import atomic_write_json, json_load, read_jsonl
from .seqsrc import RowIndex, iter_cot_round_tasks, iter_round_tasks
from .prefix import PrefixModel, ReplicatedPrefix, full_cpu_reference


def _accepted_feature_rows(cache_rows: Sequence[int]) -> np.ndarray:
    flat = np.memmap(SEQ / "features_bf16.bin", dtype="<u2", mode="r")
    m = flat.reshape(flat.shape[0] // HIDDEN, HIDDEN)
    raw = np.asarray(m[list(cache_rows)], dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _rel(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(b), 1e-6)
    return {
        "max_abs": float(np.abs(a - b).max()),
        "mean_abs": float(np.abs(a - b).mean()),
        "max_rel": float((np.abs(a - b) / denom).max()),
        "cosine_min": float(
            np.min(
                (a * b).sum(axis=1)
                / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
            )
        ),
    }


def pick_episodes(row_index: RowIndex) -> list[tuple[str, str]]:
    """Two authoritative TRAIN episodes: one short, one multi-call with a text segment."""
    lengths: dict[str, int] = {}
    multi: list[str] = []
    for (scope, ep, _round), rows in row_index.by_round.items():
        lengths[ep] = max(lengths.get(ep, 0), max(int(r["sequence_length"]) for r in rows))
        calls = {int(r["call_index"]) for r in rows if r.get("call_index") is not None}
        segs = [r for r in rows if r.get("segment_index") is not None]
        if len(calls) >= 2 and segs:
            multi.append(ep)
    train_eps = list(row_index.episode_ids)
    multi = [e for e in train_eps if e in set(multi)]
    short = sorted(train_eps, key=lambda e: (lengths[e], e))[3]
    multi_ep = multi[0] if multi else short
    if multi_ep == short:
        multi_ep = multi[min(1, len(multi) - 1)]
    return [(short, "short"), (multi_ep, "multi_call_text")]


def run(out_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"stage": "smoke", "blocks": list(CAPTURE_BLOCKS), "episodes": []}
    row_index = RowIndex("train")
    picks = pick_episodes(row_index)
    report["picks"] = [{"episode_id": e, "role": r} for e, r in picks]
    ids_wanted = [e for e, _ in picks]

    # ---- sequence sources (exact accepted binding is enforced inside the iterators)
    rounds: dict[str, list[tuple[Any, list[int]]]] = {e: [] for e in ids_wanted}
    t_seq = time.time()
    cot_ids = [e for e in ids_wanted if row_index.episode_scope[e] == "cot"]
    leg_ids = [e for e in ids_wanted if row_index.episode_scope[e] != "cot"]
    for source, ids in (("cot", cot_ids), ("legacy", leg_ids)):
        if not ids:
            continue
        iterator = (
            iter_cot_round_tasks(row_index, ids) if source == "cot" else iter_round_tasks(row_index, ids)
        )
        for task, seq_ids in iterator:
            rounds[task.episode_id].append((task, seq_ids))
    report["sequence_source_seconds"] = round(time.time() - t_seq, 3)

    prefix = PrefixModel(blocks=PREFIX_LAYERS, captures=CAPTURE_BLOCKS)
    report["gpu_weight_bytes"] = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()

    for ep, role in picks:
        entry: dict[str, Any] = {"episode_id": ep, "role": role, "scope": row_index.episode_scope[ep]}
        entry["rounds"] = []
        recorded_tokens = 0
        gpu_s = 0.0
        for task, seq_ids in rounds[ep]:
            torch.cuda.synchronize()
            t0 = time.time()
            captured = prefix.forward_ids(seq_ids)
            torch.cuda.synchronize()
            dt = time.time() - t0
            gpu_s += dt
            recorded_tokens += len(seq_ids)
            pos = task.positions
            idx = torch.tensor(pos, dtype=torch.long, device="cuda")
            entry["rounds"].append(
                {
                    "round_id": task.round_id,
                    "sequence_length": len(seq_ids),
                    "prompt_length": task.prompt_length,
                    "n_ct_endpoints": len(pos),
                    "gpu_seconds": round(dt, 4),
                    "token_layer_per_s": round(len(seq_ids) * PREFIX_LAYERS / dt / 1e3, 1),
                    "blocks": [b for b in captured],
                }
            )
            del idx
        entry["recorded_tokens"] = recorded_tokens
        entry["gpu_seconds"] = round(gpu_s, 3)
        entry["gpu_tokens_per_s"] = round(recorded_tokens / gpu_s, 1)
        entry["probe_first_round"] = True
        report["episodes"].append(entry)

    report["gpu_peak_bytes"] = int(torch.cuda.max_memory_allocated())

    # ---- path agreement on the short episode's longest round
    short = picks[0][0]
    task, seq_ids = max(rounds[short], key=lambda pair: len(pair[1]))
    pos = task.positions
    hook_caps = prefix.forward_ids(seq_ids)
    rep = ReplicatedPrefix(prefix)
    rep_caps = rep.forward_ids(seq_ids)
    idx = torch.tensor(pos, dtype=torch.long, device="cuda")
    path_cmp = {}
    for b in CAPTURE_BLOCKS:
        a = hook_caps[b][0].index_select(0, idx)
        c = rep_caps[b][0].index_select(0, idx)
        path_cmp[str(b)] = {
            "bitwise_equal": bool(torch.equal(a, c)),
            "max_abs_diff": float((a.float() - c.float()).abs().max()),
            "shape": list(a.shape),
        }
    report["hook_vs_replicated"] = path_cmp
    report["reference_episode"] = short
    report["reference_round"] = task.round_id
    report["reference_sequence_length"] = len(seq_ids)
    report["reference_positions"] = pos

    gpu_blocks = {b: hook_caps[b][0].index_select(0, idx).float().cpu().numpy() for b in CAPTURE_BLOCKS}

    # ---- suffix invariance: truncated prefix must give the same early-layer values
    cut = pos[0] + 1
    trunc_caps = prefix.forward_ids(seq_ids[:cut])
    trunc = trunc_caps[CAPTURE_BLOCKS[0]][0][cut - 1].float()
    full = hook_caps[CAPTURE_BLOCKS[0]][0][pos[0]].float()
    report["suffix_invariance"] = {
        "block": CAPTURE_BLOCKS[0],
        "position": pos[0],
        "max_abs_diff": float((trunc - full).abs().max()),
        "bitwise_equal": bool(torch.equal(trunc, full)),
    }

    # ---- CPU fp32 reference of the same prefix modules
    t0 = time.time()
    cpu_prefix = PrefixModel(
        blocks=PREFIX_LAYERS, captures=CAPTURE_BLOCKS, device="cpu", dtype=torch.float32
    )
    cpu_caps = cpu_prefix.forward_ids(seq_ids)
    cpu_pos = cpu_caps[CAPTURE_BLOCKS[0]][0].index_select(0, torch.tensor(pos)).float().numpy()
    cpu8_pos = cpu_caps[CAPTURE_BLOCKS[1]][0].index_select(0, torch.tensor(pos)).float().numpy()
    report["cpu_fp32_prefix_seconds"] = round(time.time() - t0, 2)
    del cpu_prefix
    report["gpu_bf16_vs_cpu_fp32"] = {
        str(CAPTURE_BLOCKS[0]): _rel(gpu_blocks[CAPTURE_BLOCKS[0]], cpu_pos),
        str(CAPTURE_BLOCKS[1]): _rel(gpu_blocks[CAPTURE_BLOCKS[1]], cpu8_pos),
    }

    # ---- complete original model on CPU: block values + end-to-end accepted-feature check
    t0 = time.time()
    ref = full_cpu_reference(seq_ids, capture_blocks=CAPTURE_BLOCKS, dtype=torch.bfloat16)
    report["cpu_full36_seconds"] = round(time.time() - t0, 2)
    ref4 = ref[CAPTURE_BLOCKS[0]][0].index_select(0, torch.tensor(pos)).float().numpy()
    ref8 = ref[CAPTURE_BLOCKS[1]][0].index_select(0, torch.tensor(pos)).float().numpy()
    report["gpu_prefix_vs_full36_cpu_bf16"] = {
        str(CAPTURE_BLOCKS[0]): _rel(gpu_blocks[CAPTURE_BLOCKS[0]], ref4),
        str(CAPTURE_BLOCKS[1]): _rel(gpu_blocks[CAPTURE_BLOCKS[1]], ref8),
    }
    accepted = _accepted_feature_rows([int(r["cache_row"]) for r in task.events])
    final = ref["final_norm"][0].index_select(0, torch.tensor(pos)).float().numpy()
    report["accepted_feature_end_to_end"] = {
        "rows": len(task.events),
        "cpu_full36_vs_accepted_bf16": _rel(final, accepted),
        "note": "complete 36-layer CPU bf16 forward through model.norm vs the accepted bf16 matrix",
    }
    del ref
    torch.cuda.empty_cache()

    atomic_write_json(out_path, report)
    return report


if __name__ == "__main__":
    import sys

    out = ART / "smoke" / "smoke_report.json"
    run(out)
    print(open(out).read())
