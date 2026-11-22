"""Resumable, per-episode atomic capture of block-16 / block-24 endpoint vectors.

Runs the *original frozen* Qwen3-8B prefix (``embed_tokens + rotary + layers[0:24]``) on
the exact prepared token ids and reads only the accepted endpoint rows.  One forward per
round serves both depth endpoints.  ``lm_head`` is never built, ``model.norm`` never runs.

Layout (per episode, atomic rename; digest committed in ``progress.jsonl``):

    artifacts/capture/<split>/episodes/<sha1>.npz
        episode_id, scope, group_id, label, seq_len[r]
        row_index, cache_row, message_index, call_index, segment_index,
        token_start, token_end, round_id, positions, sequence_sha256, event_id
        block16 uint16[n,4096], block24 uint16[n,4096]

Resume is exactly "progress line carries npz_sha256 and the file exists".  A crash can
only lose the episode being written; everything committed is stable.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import ART, CAPTURE_BLOCKS, STUDY_ID
from .common import StudyError, atomic_write_json, read_jsonl, sha256_file
from .prefix import PrefixModel, shard_files

ROOT = Path(__file__).resolve().parents[2]
PROGRESS_FIELDS = (
    "row_index", "cache_row", "message_index", "call_index", "segment_index",
    "token_start", "token_end", "round_id", "positions", "event_id",
)


def out_dir(split: str) -> Path:
    return ART / "capture" / split


def _episode_key(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:24]


def _producer_prepared(prepared_split_dir: str, part: Sequence[str], q: Any) -> None:
    try:
        from .prepared import iter_prepared_episodes

        for ep in iter_prepared_episodes(Path(prepared_split_dir), part):
            q.put(("episode", ep.episode_id, ep.rounds))
    except BaseException as error:  # noqa: BLE001
        q.put(("error", f"{type(error).__name__}: {error}", []))
    finally:
        q.put(("done", "", []))


def _chunks(items: Sequence[str], n: int) -> list[list[str]]:
    if not items:
        return []
    size = (len(items) + n - 1) // n
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def _start_sync(split: str) -> subprocess.Popen | None:
    """Background CPU durable sync of the committed shards (one at a time)."""
    log = ROOT / "logs"
    log.mkdir(exist_ok=True)
    cmd = [sys.executable, str(ROOT / "scripts" / "sync_durable.py"),
           "--subdir", f"artifacts/capture/{split}", "--label", f"capture_{split}"]
    fh = open(log / f"sync_capture_{split}_{int(time.time())}.log", "w")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def capture(
    split: str,
    *,
    prepared_dir: str,
    blocks: int = 24,
    chunk: int | None = None,
    limit: int | None = None,
    sync_every: int = 400,
    device: str = "cuda",
) -> dict[str, Any]:
    directory = out_dir(split)
    ep_dir = directory / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    progress_path = directory / "progress.jsonl"
    index = json.loads((Path(prepared_dir) / split / "index.json").read_text())
    prepared_files = {e: n for e, n in index.items()}
    # ordering: prepared index is written with sort_keys -> deterministic episode order
    order = sorted(prepared_files)
    finished: set[str] = set()
    if progress_path.is_file():
        for row in read_jsonl(progress_path):
            if row.get("npz_sha256") and (ep_dir / str(row.get("npz", ""))).is_file():
                finished.add(row["episode_id"])
    todo = [e for e in order if e not in finished]
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        print(f"[capture:{split}] nothing to do", flush=True)

    import multiprocessing as mp

    ctx = mp.get_context("fork")
    q = ctx.Queue(maxsize=12)
    procs = [ctx.Process(target=_producer_prepared, args=(str(Path(prepared_dir) / split), part, q), daemon=True)
             for part in _chunks(todo, 6)] if todo else []
    for proc in procs:
        proc.start()

    t_load = time.time()
    prefix = PrefixModel(blocks=blocks, captures=tuple(sorted(set(CAPTURE_BLOCKS) | {8})))
    torch.cuda.synchronize()
    load_s = time.time() - t_load
    _probe(prefix)
    torch.cuda.reset_peak_memory_stats()
    provenance = {
        "study": STUDY_ID,
        "split": split,
        "blocks_executed": blocks,
        "capture_blocks": list(CAPTURE_BLOCKS),
        "crosscheck_block": 8,
        "backend": "monolithic" if not chunk else f"kv_chunked:{chunk}",
        "chunk": chunk,
        "model_path": str(prefix.model_path),
        "model_shards": {p.name: sha256_file(p) for p in shard_files(prefix.model_path)},
        "dtype": "bfloat16",
        "layers_executed": list(range(blocks)),
        "lm_head_loaded": False,
        "norm_applied": False,
        "weights_allocated_bytes": int(torch.cuda.memory_allocated()),
        "model_load_seconds": round(load_s, 2),
        "prepared_dir": prepared_dir,
        "prepared_index_files": len(index),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }
    atomic_write_json(directory / "capture_provenance.json", provenance)

    log = open(progress_path, "a", encoding="utf-8")
    n_ep = n_rows = n_tok = 0
    t_start = time.time()
    gpu_time = 0.0
    peak = 0
    syncer: subprocess.Popen | None = None
    sync_results: list[dict[str, Any]] = []

    def maybe_sync(force: bool = False) -> None:
        nonlocal syncer
        if sync_every <= 0 and not force:
            return
        if syncer is not None and syncer.poll() is None:
            return
        if syncer is not None:
            sync_results.append({"rc": syncer.returncode, "at": time.strftime("%H:%M:%S")})
        syncer = _start_sync(split)

    finished_sources = 0
    while True:
        kind, episode_id, rounds = q.get()
        if kind == "done":
            finished_sources += 1
            if finished_sources >= len(procs):
                break
            continue
        if kind == "error":
            raise StudyError(f"prepared producer failed: {episode_id}")
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
                captured = prefix.forward_positions(ids, task.positions, chunk=chunk)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise StudyError(
                    f"OOM on {episode_id} r{task.round_id} len={len(ids)} backend="
                    f"{provenance['backend']}; committed shards are intact, resume will retry"
                ) from exc
            torch.cuda.synchronize()
            gpu_s += time.time() - t_r
            seq_lens.append(len(ids))
            for block in CAPTURE_BLOCKS:
                vec = captured[block]
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
            del captured
        wall = time.time() - t0
        payload = {
            "episode_id": episode_id,
            "scope": task.scope,
            "group_id": task.group_id,
            "label": np.asarray(bool(task.label_value)),
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
        key = _episode_key(episode_id)
        target = ep_dir / f"{key}.npz"
        tmp = ep_dir / f"{key}.npz.tmp"
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
        log.write(json.dumps({
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
            "backend": provenance["backend"],
            "dataformat": "bf16-uint16-little-endian; row-major [n,4096]",
            "endpoint": "feature_token_index == token_end",
        }, sort_keys=True) + "\n")
        log.flush()
        if n_ep % 25 == 0 or n_ep == len(todo):
            el = time.time() - t_start
            print(f"[capture:{split}] {n_ep}/{len(todo)} eps rows={n_rows} tokens={n_tok} "
                  f"gpu={gpu_time/60:.1f}min wall={el/60:.1f}min tok/s={n_tok/max(el,1e-9):.0f} "
                  f"eta={(len(todo)-n_ep)*el/max(n_ep,1):.0f}s peak={peak/2**30:.2f}GiB",
                  flush=True)
        if n_ep % max(sync_every, 1) == 0:
            maybe_sync()
    log.close()
    for proc in procs:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
    # final background sync, then wait for it so the stage exits with a verified copy
    maybe_sync(force=True)
    if syncer is not None:
        rc = syncer.wait(timeout=1800)
        sync_results.append({"rc": rc, "at": time.strftime("%H:%M:%S"), "final": True})
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
        "backend": provenance["backend"],
        "sync": sync_results,
        "rows_bytes_note": "block16+block24 stored as bf16 uint16; 8192 B per row per block",
    }
    atomic_write_json(directory / "coverage_run.json", coverage)
    return coverage


def _probe(prefix: PrefixModel) -> None:
    ids = list(range(64, 320))
    prefix.forward_positions(ids, [200, len(ids) - 1])
    torch.cuda.synchronize()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "dev"))
    ap.add_argument("--prepared-dir", default=os.environ.get("SDSAFE_PREPARED_DIR"))
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("SDSAFE_CHUNK", "0")) or None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sync-every", type=int, default=400)
    args = ap.parse_args()
    if not args.prepared_dir:
        raise SystemExit("--prepared-dir or SDSAFE_PREPARED_DIR is required")
    print(json.dumps(capture(args.split, prepared_dir=args.prepared_dir, blocks=args.blocks,
                             chunk=args.chunk, limit=args.limit, sync_every=args.sync_every), indent=2))
