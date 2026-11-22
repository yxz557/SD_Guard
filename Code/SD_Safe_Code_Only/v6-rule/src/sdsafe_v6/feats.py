"""Precompute and cache the student's feature representation for every view record."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import features, view

ARTIFACTS = Path("/root/sdsafe-compute/v6-param/artifacts")


def load_views(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: dict[int, int] = {}
    for path in paths:
        if not Path(path).is_file():
            continue
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            r = int(rec["row_index"])
            if r in seen:      # fixup files win over the original record
                out[seen[r]] = rec
            else:
                seen[r] = len(out)
                out.append(rec)
    out.sort(key=lambda r: int(r["row_index"]))
    return out


def build_cache(records: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    flat: dict[str, list[int]] = {c: [] for c in features.CHANNELS if c != "src"}
    offsets: dict[str, list[int]] = {c: [0] for c in flat}
    src_flat: list[int] = []
    src_offsets: list[int] = [0]
    src_counts: list[int] = []
    dense = np.zeros((len(records), features.DENSE_DIM), dtype=np.float32)
    for i, rec in enumerate(records):
        feats = features.extract(rec)
        for channel in flat:
            idx = feats["bags"][channel]
            flat[channel].extend(idx)
            offsets[channel].append(len(flat[channel]))
        for bag in feats["bags"]["src"]:
            src_flat.extend(bag)
            src_offsets.append(len(src_flat))
        src_counts.append(len(feats["bags"]["src"]))
        dense[i] = np.asarray(feats["dense"], dtype=np.float32)
    payload: dict[str, Any] = {"dense": dense, "src_counts": np.asarray(src_counts, dtype=np.int64),
                               "row_index": np.asarray([int(r["row_index"]) for r in records], dtype=np.int64),
                               "episode_id": np.asarray([r["episode_id"] for r in records], dtype=object),
                               "group_id": np.asarray([r["group_id"] for r in records], dtype=object),
                               "event_type": np.asarray([r["event_type"] for r in records], dtype=object),
                               "row_label": np.asarray([bool(r["row_label"]) for r in records]),
                               "episode_label": np.asarray([bool(r["episode_label"]) for r in records])}
    for channel in flat:
        payload[f"{channel}_idx"] = np.asarray(flat[channel], dtype=np.int64)
        payload[f"{channel}_off"] = np.asarray(offsets[channel], dtype=np.int64)
    payload["src_idx"] = np.asarray(src_flat, dtype=np.int64)
    payload["src_off"] = np.asarray(src_offsets, dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    return {"records": len(records), "out": str(out),
            "idx_total": {c: int(len(flat[c])) for c in flat},
            "src_bags": int(len(src_counts))}


def load_cache(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def attach_main_labels(cache: dict[str, Any], pool_path: Path) -> dict[str, Any]:
    """Accepted TRAIN pool label for every row (``security_label`` in {-1,0,1})."""
    pool = np.load(pool_path, allow_pickle=True)
    label_of = {int(r): int(v) for r, v in zip(pool["row_index"], pool["security_label"])}
    rows = cache["row_index"]
    main = np.full(rows.shape[0], -1, dtype=np.int64)
    in_pool = np.zeros(rows.shape[0], dtype=bool)
    for i, r in enumerate(rows):
        if int(r) in label_of:
            main[i] = label_of[int(r)]
            in_pool[i] = True
    cache["main_label"] = main
    cache["in_pool"] = in_pool
    return cache
