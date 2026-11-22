"""Build the v5 row dataset: accepted final hidden + new block16 / block24.

Joins captured shards to the accepted per-row universe by the accepted global
``row_index``; the final-layer vector comes from the accepted ``features_bf16.bin`` at
``cache_row``.  All bf16 stays as ``uint16`` bit patterns on disk (no lossy re-encode);
the loader shifts bits when a float view is needed.

Writes (default under /dev/shm, never the only copy of a captured shard):

    <out>/<split>/final_bf16.npy     uint16 [rows, 4096]
    <out>/<split>/block16_bf16.npy   uint16 [rows, 4096]
    <out>/<split>/block24_bf16.npy   uint16 [rows, 4096]
    <out>/<split>/meta.npz           row/episode metadata + labels
    <out>/<split>/dataset_report.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import ACCEPTED_FEATURE_BIN, ART, HIDDEN, V2_DEV_ARRAYS, V2_TRAIN_ARRAYS
from .common import StudyError, atomic_write_json, read_jsonl
from .loader import assert_known_bitpatterns

CAPTURE = ART / "capture"
DEFAULT_OUT = Path("/dev/shm/sdsafe-depth-v5/dataset")


def episode_dirs(split: str) -> list[Path]:
    env = os.environ.get("SDSAFE_DEPTH_EPISODE_DIRS")
    if env:
        return [Path(p.replace("{split}", split)) for p in env.split(":") if p]
    return [CAPTURE / split / "episodes"]


def build(split: str, out_root: Path = DEFAULT_OUT) -> dict[str, Any]:
    assert_known_bitpatterns()
    out = Path(out_root) / split
    out.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for d in episode_dirs(split):
        files.extend(sorted(d.glob("*.npz")))
    if not files:
        raise StudyError(f"no captured depth shards for {split}")
    n_total = 0
    meta_cols = {k: [] for k in ("row_index", "cache_row", "round_id", "message_index",
                                 "call_index", "segment_index", "token_start", "token_end",
                                 "positions")}
    ep_ids: list[str] = []
    groups: list[str] = []
    ep_labels: list[bool] = []
    scopes: list[str] = []
    b16: list[np.ndarray] = []
    b24: list[np.ndarray] = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        n = int(z["row_index"].shape[0])
        n_total += n
        if z["block16"].shape != (n, HIDDEN) or z["block24"].shape != (n, HIDDEN):
            raise StudyError(f"{path.name}: block shape mismatch")
        for k in meta_cols:
            meta_cols[k].append(np.asarray(z[k]))
        b16.append(np.asarray(z["block16"]))
        b24.append(np.asarray(z["block24"]))
        ep_ids.extend([str(z["episode_id"])] * n)
        groups.extend([str(z["group_id"])] * n)
        ep_labels.extend([bool(z["label"])] * n)
        scopes.extend([str(z["scope"])] * n)
        z.close()
    meta = {k: np.concatenate(v, axis=0) for k, v in meta_cols.items()}
    row_ids = meta["row_index"]
    if len(set(row_ids.tolist())) != row_ids.shape[0]:
        raise StudyError("duplicate captured row ids")
    accepted = np.memmap(ACCEPTED_FEATURE_BIN, dtype="<u2", mode="r")
    flat = accepted.reshape(accepted.shape[0] // HIDDEN, HIDDEN)
    final = np.ascontiguousarray(flat[meta["cache_row"]])
    order = np.argsort(row_ids, kind="stable")
    b16 = np.concatenate(b16, axis=0)[order]
    b24 = np.concatenate(b24, axis=0)[order]
    final = final[order]
    arrays = {k: v[order] for k, v in meta.items()}
    arrays["episode_id"] = np.asarray(ep_ids, dtype=object)[order]
    arrays["group_id"] = np.asarray(groups, dtype=object)[order]
    arrays["episode_label"] = np.asarray(ep_labels, dtype=bool)[order]
    arrays["scope"] = np.asarray(scopes, dtype=object)[order]

    # TRAIN row labels from the accepted v2 pool (DEV gets no training label)
    label = np.full(n_total, -1, dtype=np.int8)
    in_loss = np.zeros(n_total, dtype=bool)
    pool_info: dict[str, Any] = {}
    if split == "train":
        pool = np.load(V2_TRAIN_ARRAYS, allow_pickle=True)
        pool_rows = np.asarray(pool["row_index"], dtype=np.int64)
        pool_label = np.asarray(pool["security_label"], dtype=np.int64)
        pool_mask = np.asarray(pool["security_mask"], dtype=bool)
        slot = {int(r): i for i, r in enumerate(pool_rows.tolist())}
        matched = 0
        for i, r in enumerate(arrays["row_index"].tolist()):
            j = slot.get(int(r))
            if j is None:
                continue
            matched += 1
            label[i] = int(pool_label[j])
            in_loss[i] = not bool(pool_mask[j])
        pool_info = {
            "pool_rows": int(pool_rows.size),
            "pool_labeled": int((pool_label >= 0).sum()),
            "pool_in_loss": int(((pool_label >= 0) & ~pool_mask).sum()),
            "matched_capture_rows": matched,
        }
    arrays["row_label"] = label
    arrays["in_loss"] = in_loss
    np.savez(out / "meta.npz", **arrays)
    np.save(out / "final_bf16.npy", final.astype(np.uint16))
    np.save(out / "block16_bf16.npy", b16.astype(np.uint16))
    np.save(out / "block24_bf16.npy", b24.astype(np.uint16))
    from collections import Counter

    scope_of: dict[str, str] = {}
    for e, sc in zip(ep_ids, scopes):
        scope_of[e] = sc
    report = {
        "split": split,
        "rows": int(n_total),
        "episodes": int(len(scope_of)),
        "episodes_by_scope": dict(Counter(scope_of.values())),
        "rows_by_scope": dict(Counter(arrays["scope"].tolist())),
        "rows_with_label": int((label >= 0).sum()),
        "rows_in_loss": int(in_loss.sum()),
        "positives_in_loss": int(((label == 1) & in_loss).sum()),
        "negatives_in_loss": int(((label == 0) & in_loss).sum()),
        "label_pool": pool_info,
        "shards": len(files),
        "capture_dirs": [str(p) for p in episode_dirs(split)],
        "bytes": int((out / "block16_bf16.npy").stat().st_size * 2 + (out / "final_bf16.npy").stat().st_size),
        "dataformat": "uint16 bf16 bit patterns, row-major [n,4096]",
    }
    atomic_write_json(out / "dataset_report.json", report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train", "dev"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    print(json.dumps(build(args.split, Path(args.out)), indent=2))
