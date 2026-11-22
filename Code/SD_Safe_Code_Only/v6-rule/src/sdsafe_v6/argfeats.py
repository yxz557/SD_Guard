"""Per-argument field features for the S4 fine-grained supervision arm.

Each event contributes zero or more argument fields ``(path, value, source window, user goal)``.
Only text visible in the causal view is used.  The teacher's per-argument authorization label is
*not* an input feature; it is a training target in ``run_arg_arm``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from . import features

ART = Path("/root/sdsafe-compute/v6-param/artifacts")
ARGB = 1 << 16
ARG_DENSE = 8
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def path_dense(item: dict[str, Any]) -> list[float]:
    u = item.get("user") or {}
    srcs = item.get("sources") or []
    value = str(item.get("value") or "")
    in_src = bool(srcs)
    numeric = bool(_NUM.fullmatch(value.strip()))
    digits = sum(c.isdigit() for c in value)
    return [float(bool(u.get("exact"))), float(bool(u.get("normalized"))),
            float(in_src), float(not in_src), float(numeric),
            float(digits >= 6), float("@" in value), float(len(value) >= 24)]


def _bag(text: str) -> list[int]:
    return [i % ARGB for i in features.bag_indices(text)]


def build_arg_cache(views: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    counts = np.zeros(len(views), dtype=np.int64)
    offsets = [0]
    paths: list[str] = []
    flat: dict[str, list[int]] = {c: [] for c in ("path", "value", "source", "user")}
    offs: dict[str, list[int]] = {c: [0] for c in flat}
    dense: list[list[float]] = []
    for i, rec in enumerate(views):
        items = rec.get("value_sources") or []
        counts[i] = len(items)
        by_index = {int(b["block_index"]): b for b in rec.get("prior") or []}
        user = rec.get("user_request") or ""
        for item in items:
            src_text = " ".join(str(by_index.get(int(s["block_index"]), {}).get("text") or "")
                                for s in (item.get("sources") or []))
            texts = {"path": str(item["path"]), "value": str(item["value"]),
                     "source": src_text, "user": user}
            for ch, text in texts.items():
                flat[ch].extend(_bag(text))
                offs[ch].append(len(flat[ch]))
            paths.append(str(item["path"]))
            dense.append(path_dense(item))
        offsets.append(len(paths))
    payload = {
        "arg_count": counts,
        "arg_off": np.asarray(offsets, dtype=np.int64),
        "arg_paths": np.asarray(paths, dtype=object),
        "arg_dense": np.asarray(dense, dtype=np.float32).reshape(-1, ARG_DENSE),
    }
    for ch in flat:
        payload[f"{ch}_idx"] = np.asarray(flat[ch], dtype=np.int64)
        payload[f"{ch}_off"] = np.asarray(offs[ch], dtype=np.int64)
    payload["row_index"] = np.asarray([int(r["row_index"]) for r in views], dtype=np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    return {"records": len(views), "paths": len(paths), "out": str(out),
            "bag_tokens": {k: len(v) for k, v in flat.items()}}


def load(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    from . import feats, train
    for split, views, out in (("train", train.TRAIN_VIEWS, ART / "arg_cache_train_v3.npz"),
                              ("dev", train.DEV_VIEWS, ART / "arg_cache_dev_v3.npz")):
        recs = feats.load_views(views)
        print(split, json.dumps(build_arg_cache(recs, out)))
