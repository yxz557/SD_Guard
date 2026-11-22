"""Read-only access to the accepted row space and the prepared exact token ids (GPU side)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

SEQ_DIR = Path("/root/autodl-tmp/SD-Safe/artifacts/fine_sequence_explore_v1/data/sequence_v1")
ROWS = {"train": SEQ_DIR / "rows_train.jsonl", "dev": SEQ_DIR / "rows_dev.jsonl"}
PREPARED = {"train": Path("/root/sdsafe-compute/v5-depth/prepared/train"),
            "dev": Path("/root/sdsafe-compute/v5-depth/prepared/dev")}
MODEL_DIR = Path("/dev/shm/sdsafe-gpu-v4/upstream/SD-Safe/models/Qwen3-8B")
V2_DEV_ARRAYS = Path("/dev/shm/sdsafe-gpu-v4/upstream/sdsafe-lowcost-authorization-v2/artifacts/tensors/dev_common_arrays.npz")
V2_BASELINE = Path("/dev/shm/sdsafe-gpu-v4/upstream/sdsafe-lowcost-authorization-v2/artifacts/baseline/baseline_scores_v2.npz")


def load_rows(split: str, *, ct_only: bool = True) -> dict[int, dict[str, Any]]:
    """row_index -> row (compact: only fields the study needs)."""
    keep = ("row_index", "episode_id", "scope", "group_id", "event_id", "event_type",
            "ct_mask", "label_value", "strict_final", "label_status", "round_id",
            "message_index", "call_index", "segment_index", "token_start", "token_end",
            "feature_token_index", "sequence_sha256", "prompt_length", "sequence_length")
    out: dict[int, dict[str, Any]] = {}
    with open(ROWS[split], "r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if ct_only and not row["ct_mask"]:
                continue
            out[int(row["row_index"])] = {k: row.get(k) for k in keep}
    return out


def episode_index(split: str) -> dict[str, str]:
    return json.loads((PREPARED[split] / "index.json").read_text())


def load_prepared(split: str, episode_id: str) -> dict[str, Any]:
    key = episode_index(split)[episode_id]
    return dict(np.load(PREPARED[split] / key, allow_pickle=True))


def rows_by_episode(rows: dict[int, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows.values():
        out.setdefault(row["episode_id"], []).append(row)
    for value in out.values():
        value.sort(key=lambda r: (int(r["round_id"]), int(r["feature_token_index"])))
    return out


def dev_common_ids() -> list[str]:
    z = np.load(V2_DEV_ARRAYS, allow_pickle=True)
    return [str(x) for x in z["episode_id"].tolist()]


def baseline_arrays() -> dict[str, np.ndarray]:
    z = np.load(V2_BASELINE, allow_pickle=True)
    return {k: z[k] for k in z.files}


def iter_prepared_episode(rec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield (round metadata, ids) pairs in round order."""
    round_ids = [int(x) for x in rec["round_id"].tolist()]
    offsets = [int(x) for x in rec["ids_offsets"].tolist()]
    seq = [int(x) for x in rec["seq_len"].tolist()]
    prompt = [int(x) for x in rec["prompt_length"].tolist()]
    order = sorted(range(len(round_ids)), key=lambda i: round_ids[i])
    ids = rec["ids_flat"]
    for i in order:
        yield {"round_id": round_ids[i], "seq_len": seq[i], "prompt_length": prompt[i],
               "offset": offsets[i]}, ids
