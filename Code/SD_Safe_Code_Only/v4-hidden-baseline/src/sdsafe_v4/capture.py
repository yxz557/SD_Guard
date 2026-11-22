"""Resumable, per-episode atomic capture of earlier-layer endpoint vectors.

For every accepted CT endpoint the capture stores the original model's residual-stream
vector after block 4 and after block 8 (HF ``output_hidden_states[4]`` /
``output_hidden_states[8]``), at exactly the accepted ``feature_token_index`` of the
accepted per-round token sequence.

One prefix forward per round; every CT endpoint of that round is read out of that single
forward.  Nothing is padded, truncated, re-encoded or regenerated.

Layout (per episode, atomic ``.npz`` rename):

    artifacts/capture/<split>/episodes/<sha1>.npz
        episode_id (str), scope, group_id, label (bool), sequence_sha256 per round (...)
        row_index  int64 [n]   global sequence row id (joins the accepted v1/v2 pools)
        cache_row  int64 [n]   row in the accepted features_bf16.bin
        message_index / call_index / segment_index / token_start / token_end  int32 [n]
        round_id   int32 [n]
        positions  int32 [n]   readout token index inside its round's sequence
        seq_len    int32 [r]   per-round executed sequence length
        block4     uint16 [n, 4096]  bf16 bit pattern
        block8     uint16 [n, 4096]

    artifacts/capture/<split>/progress.jsonl   one line per finished episode
    artifacts/capture/<split>/coverage.json    universe/coverage accounting
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import queue
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from . import ART, CAPTURE_BLOCKS, HIDDEN, PREFIX_LAYERS
from .common import StudyError, atomic_write_json, json_load, read_jsonl
from .prefix import PrefixModel
from .seqsrc import RowIndex, dev_common_universe, iter_cot_round_tasks, iter_round_tasks


CAPTURE_SCOPES = ("cot", "ctx40k", "agentlab", "nothink")
NOT_CAPTURED = {
    "nothink": (
        "the no-think scope can only be re-built with the frozen Qwen3 template *and* the "
        "AgentDojo tool schema; the only torch/transformers environment on this host has no "
        "agentdojo runtime dependencies (cohere/tenacity/google-genai/openai absent) and there "
        "is no network access, so its token ids cannot be proved against the accepted digest. "
        "Those rows stay explicit unknown (old-base fallback) rather than being re-encoded."
    )
}


def out_dir(split: str) -> Path:
    return ART / "capture" / split


def _episode_key(row_index: RowIndex, episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:24]


def _probe(model: PrefixModel) -> None:
    """One throwaway forward so kernel selection never lands in a measured round."""
    model.forward_ids(list(range(64, 320)))
    torch.cuda.synchronize()


def episode_order(row_index: RowIndex, universe: Sequence[str] | None, seed: int) -> list[str]:
    """Deterministic episode order; TRAIN is shuffled so partial coverage is unbiased."""
    eps = [e for e in row_index.episodes(universe) if row_index.episode_scope.get(e) in CAPTURE_SCOPES]
    if seed:
        import random

        rng = random.Random(seed)
        rng.shuffle(eps)
    return eps


def _producer(row_index: RowIndex, episodes: Sequence[str], q: "queue.Queue") -> None:
    """Sequence source thread: emits one whole episode's rounds at a time."""
    try:
        cot = [e for e in episodes if row_index.episode_scope.get(e) == "cot"]
        rest = [e for e in episodes if row_index.episode_scope.get(e) != "cot"]
        groups = [(cot, iter_cot_round_tasks(row_index, cot))]
        if rest:
            groups.append((rest, iter_round_tasks(row_index, rest)))
        for _group, iterator in groups:
            batch: list[tuple[Any, list[int]]] = []
            current = None
            for task, ids in iterator:
                if current is not None and task.episode_id != current:
                    q.put((current, batch))
                    batch = []
                current = task.episode_id
                batch.append((task, ids))
            if current is not None:
                q.put((current, batch))
    except BaseException as error:  # noqa: BLE001 - forward to the consumer
        q.put(error)
    finally:
        q.put(None)


def _producer_process(split: str, part: Sequence[str], q: Any) -> None:
    """Sequence source in its own OS process (no GIL contention with the GPU consumer)."""
    try:
        row_index = RowIndex(split)
        cot = [e for e in part if row_index.episode_scope.get(e) == "cot"]
        rest = [e for e in part if row_index.episode_scope.get(e) != "cot"]
        groups = []
        if cot:
            groups.append(iter_cot_round_tasks(row_index, cot))
        if rest:
            groups.append(iter_round_tasks(row_index, rest))
        for iterator in groups:
            batch: list[tuple[Any, list[int]]] = []
            current = None
            for task, ids in iterator:
                if current is not None and task.episode_id != current:
                    q.put(("episode", current, batch))
                    batch = []
                current = task.episode_id
                batch.append((task, ids))
            if current is not None:
                q.put(("episode", current, batch))
    except BaseException as error:  # noqa: BLE001 - surface in the parent
        q.put(("error", f"{type(error).__name__}: {error}", []))
    finally:
        q.put(("done", "", []))


def _producer_process_prepared(prepared_dir: str, part: Sequence[str], q: Any) -> None:
    """Token-id source from CPU-exported prepared jobs (no source access on this node)."""
    try:
        from .prepared import iter_prepared_episodes

        for ep in iter_prepared_episodes(Path(prepared_dir), part):
            q.put(("episode", ep.episode_id, ep.rounds))
    except BaseException as error:  # noqa: BLE001 - surface in the parent
        q.put(("error", f"{type(error).__name__}: {error}", []))
    finally:
        q.put(("done", "", []))


def _stack(named: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in named.items()}


def capture(
    split: str,
    *,
    universe: Sequence[str] | None = None,
    limit: int | None = None,
    seed: int = 0,
    device: str = "cuda",
    blocks: int = PREFIX_LAYERS,
) -> dict[str, Any]:
    directory = out_dir(split)
    ep_dir = directory / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.jsonl"
    row_index = RowIndex(split)
    order = episode_order(row_index, universe, seed)
    # resume: an episode is done iff its progress line carries the npz digest
    finished: set[str] = set()
    if progress_path.is_file():
        for row in read_jsonl(progress_path):
            if row.get("npz_sha256"):
                finished.add(row["episode_id"])
    # Node migration: the 3 612 TRAIN + 2 028 DEV shards completed on the old GPU are a
    # read-only dependency kept in the old v4 tree.  SDSAFE_PRESKIP_DONE names their
    # progress.jsonl so this run skips them without writing them into the new progress
    # (new progress therefore still matches exactly the new local files).
    preskip = os.environ.get("SDSAFE_PRESKIP_DONE")
    if preskip:
        skipped = 0
        for row in read_jsonl(preskip):
            if row.get("npz_sha256"):
                finished.add(row["episode_id"])
                skipped += 1
        print(f"[capture:{split}] preskipped {skipped} old-GPU episodes from {preskip}", flush=True)
    todo = [e for e in order if e not in finished]
    if limit is not None:
        todo = todo[:limit]

    # producers are forked before CUDA is initialised: children never touch the GPU
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    cot_part = [e for e in todo if row_index.episode_scope.get(e) == "cot"]
    rest_part = [e for e in todo if row_index.episode_scope.get(e) != "cot"]
    legacy_part = [e for e in rest_part if row_index.episode_scope.get(e) in ("ctx40k", "agentlab")]
    nothink_part = [e for e in rest_part if row_index.episode_scope.get(e) == "nothink"]
    def chunks(items: Sequence[str], n: int) -> list[list[str]]:
        if not items:
            return []
        size = (len(items) + n - 1) // n
        return [list(items[i : i + size]) for i in range(0, len(items), size)]

    prepared_dir = os.environ.get("SDSAFE_PREPARED_DIR")
    q = ctx.Queue(maxsize=12)
    if prepared_dir:
        # Node migration: token ids were serialised on the CPU/storage node by the frozen
        # iterators; this node only forwards them.  Several loaders keep the GPU fed.
        parts = chunks(list(todo), 6)
        procs = [
            ctx.Process(target=_producer_process_prepared, args=(prepared_dir, part, q), daemon=True)
            for part in parts
        ]
    else:
        # the frozen no-think render is by far the most expensive source; shard it so the GPU
        # is not starved (separate processes, so the GIL is not shared)
        parts = chunks(cot_part, 2) + chunks(legacy_part, 1) + chunks(nothink_part, 3)
        procs = [ctx.Process(target=_producer_process, args=(split, part, q), daemon=True) for part in parts]
    for proc in procs:
        proc.start()
    finished_sources = 0

    prefix = PrefixModel(blocks=blocks, captures=CAPTURE_BLOCKS, device=device)
    _probe(prefix)
    torch.cuda.reset_peak_memory_stats()

    log = open(progress_path, "a", encoding="utf-8")
    n_ep = n_rows = n_tok = 0
    t_start = time.time()
    gpu_time = 0.0
    peak = 0
    t_wait_total = 0.0
    t_post_total = 0.0
    while True:
        t_w = time.time()
        kind, episode_id, rounds = q.get()
        t_wait_total += time.time() - t_w
        if kind == "done":
            finished_sources += 1
            if finished_sources >= len(procs):
                break
            continue
        if kind == "error":
            raise StudyError(f"sequence producer failed: {episode_id}")
        if kind != "episode":
            raise StudyError(f"unexpected producer message {kind!r}")
        arrays: dict[str, list[Any]] = defaultdict(list)
        seq_lens: list[int] = []
        t0 = time.time()
        gpu_s = 0.0
        for task, ids in rounds:
            torch.cuda.synchronize()
            t_r = time.time()
            try:
                captured = prefix.forward_ids(ids)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                captured = prefix.forward_ids(ids)
            torch.cuda.synchronize()
            gpu_s += time.time() - t_r
            seq_lens.append(len(ids))
            idx = torch.tensor(task.positions, dtype=torch.long, device=prefix.device)
            for block in CAPTURE_BLOCKS:
                vec = captured[block][0].index_select(0, idx)
                arrays[f"block{block}"].append(vec.view(torch.uint16).cpu().numpy())
            for row in task.events:
                arrays["row_index"].append(int(row["row_index"]))
                arrays["cache_row"].append(int(row["cache_row"]))
                arrays["message_index"].append(int(row["message_index"]))
                arrays["call_index"].append(-1 if row.get("call_index") is None else int(row["call_index"]))
                arrays["segment_index"].append(-1 if row.get("segment_index") is None else int(row["segment_index"]))
                arrays["token_start"].append(int(row["token_start"]))
                arrays["token_end"].append(int(row["token_end"]))
                arrays["round_id"].append(int(task.round_id))
                arrays["positions"].append(int(row["feature_token_index"]))
                arrays["sequence_sha256"].append(task.sequence_sha256)
                arrays["event_id"].append(str(row["event_id"]))
            del captured, idx
        t_post = time.time()
        wall = time.time() - t0
        payload = {
            "episode_id": episode_id,
            "scope": row_index.episode_scope[episode_id],
            "group_id": row_index.episode_group[episode_id],
            "label": np.asarray(row_index.episode_label[episode_id]),
            "seq_len": np.asarray(seq_lens, dtype=np.int32),
            "row_index": np.asarray(arrays["row_index"], dtype=np.int64),
            "cache_row": np.asarray(arrays["cache_row"], dtype=np.int64),
            "message_index": np.asarray(arrays["message_index"], dtype=np.int32),
            "call_index": np.asarray(arrays["call_index"], dtype=np.int32),
            "segment_index": np.asarray(arrays["segment_index"], dtype=np.int32),
            "token_start": np.asarray(arrays["token_start"], dtype=np.int32),
            "token_end": np.asarray(arrays["token_end"], dtype=np.int32),
            "round_id": np.asarray(arrays["round_id"], dtype=np.int32),
            "positions": np.asarray(arrays["positions"], dtype=np.int32),
            "sequence_sha256": np.asarray(arrays["sequence_sha256"], dtype=object),
            "event_id": np.asarray(arrays["event_id"], dtype=object),
        }
        for block in CAPTURE_BLOCKS:
            payload[f"block{block}"] = np.concatenate(arrays[f"block{block}"], axis=0)
        key = _episode_key(row_index, episode_id)
        target = ep_dir / f"{key}.npz"
        tmp = ep_dir / f"{key}.npz.tmp"
        # serialise in memory, hash the exact bytes, then one atomic rename: no second read
        # and no per-episode fsync (the digest in progress.jsonl is the commit record; a
        # crash can only lose the episode being written, and resume re-captures it)
        buffer = io.BytesIO()
        np.savez(buffer, **payload)
        blob = buffer.getvalue()
        digest = hashlib.sha256(blob).hexdigest()
        with open(tmp, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, target)
        del blob, buffer
        n_tok_ep = int(sum(seq_lens))
        n_rows_ep = int(payload["row_index"].shape[0])
        n_ep += 1
        n_rows += n_rows_ep
        n_tok += n_tok_ep
        gpu_time += gpu_s
        peak = max(peak, int(torch.cuda.max_memory_allocated()))
        log.write(
            json.dumps(
                {
                    "episode_id": episode_id,
                    "scope": payload["scope"],
                    "rounds": len(seq_lens),
                    "tokens": n_tok_ep,
                    "rows": n_rows_ep,
                    "gpu_seconds": round(gpu_s, 4),
                    "wall_seconds": round(wall, 4),
                    "npz": target.name,
                    "npz_sha256": digest,
                    "blocks": list(CAPTURE_BLOCKS),
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        t_post_total += time.time() - t_post
        if n_ep % 25 == 0 or n_ep == len(todo):
            el = time.time() - t_start
            print(
                f"[capture:{split}] {n_ep}/{len(todo)} eps  rows={n_rows} tokens={n_tok} "
                f"gpu={gpu_time/60:.1f}min wall={el/60:.1f}min "
                f"tok/s={n_tok/max(el,1e-9):.0f} eta={(len(todo)-n_ep)*el/max(n_ep,1):.0f}s "
                f"wait%={100*t_wait_total/max(el,1e-9):.0f} gpu%={100*gpu_time/max(el,1e-9):.0f} "
                f"post%={100*t_post_total/max(el,1e-9):.0f}",
                flush=True,
            )
    log.close()
    for proc in procs:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
    total = time.time() - t_start
    coverage = {
        "split": split,
        "universe_episodes": len(order),
        "captured_episodes": len(finished) + n_ep,
        "this_run_episodes": n_ep,
        "ct_rows": n_rows,
        "tokens": n_tok,
        "gpu_seconds": round(gpu_time, 2),
        "wall_seconds": round(total, 2),
        "tokens_per_second_gpu": round(n_tok / max(gpu_time, 1e-9), 1),
        "peak_cuda_bytes": peak,
        "blocks": list(CAPTURE_BLOCKS),
        "rows_bytes_note": "block4+block8 stored as bf16 uint16; 8192 B per row per block",
    }
    atomic_write_json(directory / "coverage_run.json", coverage)
    return coverage


def coverage_report(split: str) -> dict[str, Any]:
    """Explicit coverage accounting for the captured split (including missing scopes)."""
    directory = out_dir(split)
    rows = [r for r in read_jsonl(directory / "progress.jsonl")] if (directory / "progress.jsonl").is_file() else []
    row_index = RowIndex(split)
    per_scope: dict[str, dict[str, int]] = defaultdict(lambda: {"episodes": 0, "rows": 0})
    for row in rows:
        per_scope[row["scope"]]["episodes"] += 1
        per_scope[row["scope"]]["rows"] += int(row["rows"])
    univ = defaultdict(int)
    for e in row_index.episode_ids:
        univ[row_index.episode_scope[e]] += 1
    out = {
        "split": split,
        "captured_episodes": len(rows),
        "captured_scope": dict(per_scope),
        "universe_scope_episodes": dict(univ),
    }
    if split == "dev":
        common = set(dev_common_universe())
        got = {r["episode_id"] for r in rows}
        out["common_dev_episodes"] = len(common)
        out["common_dev_covered"] = len(common & got)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "dev"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blocks", type=int, default=PREFIX_LAYERS)
    ap.add_argument("--coverage-only", action="store_true")
    args = ap.parse_args()
    if args.coverage_only:
        print(json.dumps(coverage_report(args.split), indent=2))
    else:
        universe = dev_common_universe() if args.split == "dev" else None
        print(json.dumps(capture(args.split, universe=universe, limit=args.limit, seed=args.seed, blocks=args.blocks), indent=2))
