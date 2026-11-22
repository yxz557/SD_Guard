"""Accepted baseline reconstruction with the *pack* score dtype (audit + fusion inputs).

Audit finding (see REPORT.md section 1): the three registered ``dev_common.npz`` packs store
``position_logits`` and ``episode_score_logit`` as **float32**.  The published method aggregates
in float32 (max over CT positions per seed, arithmetic mean over the three seeds), so the frozen
threshold literal ``-0.6190447211265564`` is exactly the float32 image of the borderline
episode's score and replays 21 FN / 78 FP.  ``baseline_scores_v2.npz`` instead stored a float64
mean (``-0.6190447409947714``); pairing that float64 score vector with the float32-shaped literal
threshold flips that single borderline positive and yields 22/78.  Nothing is re-thresholded
here: we reproduce the float32 protocol and report the mismatched pairing explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACTS = Path("/root/sdsafe-compute/v6-param/artifacts")
PACKS = [
    ("s0", ARTIFACTS / "baseline_packs/w2a-current256-w2-tail5-s0.npz",
     "4a4ce6a7ba410089ae9bb3fa8bbbf0fa6199eb487745f22eae9e82ed25b487f0"),
    ("s1", ARTIFACTS / "baseline_packs/w3-current256-w2-tail5-s1.npz",
     "ffcc44433b633767f2ea9ec70acfa621013e1b7edd8188d67b58c20b2ab62984"),
    ("s2", ARTIFACTS / "baseline_packs/w3-current256-w2-tail5-s2.npz",
     "87273f8ce700f0a380aa61c9cdba17c9f68b810ebf9e4fe804995392090150ce"),
]
THRESHOLD_F32 = np.float32(-0.6190447211265564)
THRESHOLD_F64_FROM_ARTIFACT = -0.6190447409947714
V2_DEV = Path("/dev/shm/sdsafe-gpu-v4/upstream/sdsafe-lowcost-authorization-v2/artifacts/tensors/dev_common_arrays.npz")
V2_BASELINE = Path("/dev/shm/sdsafe-gpu-v4/upstream/sdsafe-lowcost-authorization-v2/artifacts/baseline/baseline_scores_v2.npz")
V2_TRAIN_POOL = Path("/dev/shm/sdsafe-gpu-v4/upstream/sdsafe-lowcost-authorization-v2/artifacts/tensors/train_pool_arrays.npz")


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dev_from_packs() -> dict[str, Any]:
    seeds = []
    for name, path, expected in PACKS:
        digest = _sha(path)
        if digest != expected:
            raise RuntimeError(f"{name}: pack sha mismatch {digest} != {expected}")
        z = np.load(path, allow_pickle=True)
        seeds.append(z)
    ids = np.asarray([str(x) for x in seeds[0]["episode_episode_id"].tolist()], dtype=object)
    labels = seeds[0]["episode_strict_final"].astype(np.int64)
    for z in seeds[1:]:
        if [str(x) for x in z["episode_episode_id"].tolist()] != ids.tolist():
            raise RuntimeError("episode order differs across packs")
        if not np.array_equal(z["episode_strict_final"].astype(np.int64), labels):
            raise RuntimeError("labels differ across packs")

    n = ids.size
    per_seed_episode = np.zeros((3, n), dtype=np.float32)
    row_parts = []
    for s, z in enumerate(seeds):
        per_seed_episode[s] = z["episode_score_logit"].astype(np.float32)
        row_parts.append({
            "episode_index": z["position_episode_index"].astype(np.int64),
            "episode_id_rows": np.asarray([str(x) for x in z["position_episode_id"].tolist()], dtype=object),
            "ct_mask": z["position_ct_mask"].astype(bool),
            "logits": z["position_logits"].astype(np.float32),
            "global_row_id": z["position_global_row_id"].astype(np.int64),
            "event_family": z["position_event_family"].astype(np.uint8),
            "is_unknown": z["position_is_unknown"].astype(bool),
        })
    local_of = {e: i for i, e in enumerate(ids.tolist())}
    rows_local = np.asarray([local_of[str(e)] for e in row_parts[0]["episode_id_rows"].tolist()], dtype=np.int64)
    for part in row_parts:
        part["episode_local"] = rows_local
    ens32 = per_seed_episode.mean(axis=0, dtype=np.float32)
    ens64 = per_seed_episode.astype(np.float64).mean(axis=0)
    pred32 = ens32 >= THRESHOLD_F32
    pred64_lit = ens64 >= float(THRESHOLD_F32)
    pred64_art = ens64 >= THRESHOLD_F64_FROM_ARTIFACT
    out = {
        "episode_id": ids, "labels": labels, "per_seed_episode": per_seed_episode,
        "ensemble_f32": ens32, "ensemble_f64": ens64, "rows": row_parts,
        "metrics": {
            "f32_threshold_literal": _counts(pred32, labels),
            "f64_threshold_literal": _counts(pred64_lit, labels),
            "f64_threshold_artifact": _counts(pred64_art, labels),
        },
        "threshold_f32_literal": float(THRESHOLD_F32),
        "threshold_f64_artifact": THRESHOLD_F64_FROM_ARTIFACT,
        "pack_sha256": {name: expected for name, _p, expected in PACKS},
        "dtype": {"position_logits": "float32", "episode_score_logit": "float32",
                  "ensemble_recompute": "float32", "stored_v2_ensemble": "float64"},
    }
    return out


def _counts(pred: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    return {"tp": int((pred & (labels == 1)).sum()), "fp": int((pred & (labels == 0)).sum()),
            "fn": int((~pred & (labels == 1)).sum()), "tn": int((~pred & (labels == 0)).sum())}


def dev_row_scores(dev: dict[str, Any]) -> dict[str, Any]:
    """Per-CT-row base logits for the common DEV universe (authoritative float32 pack values)."""
    rows = dev["rows"]
    local = np.asarray(rows[0]["episode_local"], dtype=np.int64)
    # all three packs share the position row space; verify and keep one
    for r in rows[1:]:
        if not np.array_equal(r["episode_index"], rows[0]["episode_index"]):
            raise RuntimeError("andev position row space differs across packs")
    logits = np.stack([r["logits"] for r in rows], axis=0)  # [3, n_rows]
    return {"episode_index": local, "ct_mask": rows[0]["ct_mask"], "logits": logits,
            "global_row_id": rows[0]["global_row_id"], "is_unknown": rows[0]["is_unknown"]}


def train_base_scores() -> dict[str, Any]:
    """TRAIN per-row base logits as stored by the accepted v2 pool (replay, 1e-5 drift noted)."""
    z = np.load(V2_TRAIN_POOL, allow_pickle=True)
    ep = np.load(ARTIFACTS / "train_row_episode.npz", allow_pickle=True)
    # episode index per pool row comes from the accepted rows file, resolved by row_index
    return {"row_index": z["row_index"].astype(np.int64),
            "base_logits": z["base_logits"].astype(np.float32),
            "security_label": z["security_label"].astype(np.int64)}


def pool_episode_scores(pool: dict[str, Any], row_episode_index: np.ndarray,
                        n_episodes: int) -> np.ndarray:
    """Per-seed episode score = max over pool rows (float32), then float32 mean over seeds."""
    per_seed = np.full((3, n_episodes), -np.inf, dtype=np.float32)
    for s in range(3):
        np.maximum.at(per_seed[s], row_episode_index, pool["base_logits"][s].astype(np.float32))
    return per_seed.mean(axis=0, dtype=np.float32)
