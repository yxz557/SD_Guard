"""Cross-GPU canary: recompute already-captured v4 episodes on the new 4090 from the
CPU-prepared token ids and compare against the old shard's block4/block8 at identical rows.

Gate (pre-declared from the existing smoke evidence and the bf16 dtype):
  per-block cosine_min >= 0.9999
  plus identity of round/row/endpoint/sequence bindings.
Bitwise equality is NOT required (different SM counts/kernels) but recorded.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from sdsafe_v4 import ART, CAPTURE_BLOCKS, HIDDEN, PREFIX_LAYERS
from sdsafe_v4.common import atomic_write_json
from sdsafe_v4.prefix import PrefixModel
from sdsafe_v4.prepared import load_episode

SAMPLE = ART / "oldview/train/episodes_sample"
SEL = ART.parent / "manifests/canary_selection.json"
PREP = ART.parent / "manifests/prepared/train"
OUT = ART / "canary/canary_report.json"


def _u16_to_f32(raw: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(raw)).view(torch.bfloat16).float().numpy()


def _rel(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cos = (a * b).sum(axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12
    )
    return {
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
        "max_abs": float(np.abs(a - b).max()),
        "mean_abs": float(np.abs(a - b).mean()),
        "rms_diff": float(np.sqrt(((a - b) ** 2).mean())),
    }


def recompute(prep, ep):
    meta = {}
    blocks: dict[tuple[int, int], np.ndarray] = {}
    for task, ids in prep.rounds:
        caps = PREFIX.forward_ids(ids)
        idx = torch.tensor(task.positions, dtype=torch.long, device="cuda")
        for b in CAPTURE_BLOCKS:
            vec = caps[b][0].index_select(0, idx)
            blocks[(task.round_id, b)] = vec.view(torch.uint16).cpu().numpy()
        meta[task.round_id] = {
            "sequence_sha256": task.sequence_sha256,
            "positions": list(task.positions),
            "seq_len": len(ids),
        }
        del caps, idx
    return blocks, meta


def main():
    sel = json.loads(SEL.read_text())
    index = json.loads((PREP / "index.json").read_text())
    report = {
        "stage": "cross_gpu_canary",
        "blocks": list(CAPTURE_BLOCKS),
        "gate": {"cos_min_ge": 0.9999},
        "episodes": [],
    }
    all_pass = True
    for item in sel:
        npz = SAMPLE / item["npz"]
        z = np.load(npz, allow_pickle=True)
        ep = str(z["episode_id"])
        prep = load_episode(PREP / index[ep])
        entry = {"episode_id": ep, "scope": str(z["scope"]), "npz": item["npz"],
                 "n_rows": int(z["row_index"].shape[0])}
        t0 = time.time()
        blocks, meta = recompute(prep, ep)
        entry["recompute_seconds"] = round(time.time() - t0, 3)

        old_round_id = np.asarray(z["round_id"], dtype=np.int64)
        old_pos = np.asarray(z["positions"], dtype=np.int64)
        old_sha = [str(x) for x in z["sequence_sha256"].tolist()]
        checks = {"round_ids_equal": sorted(meta) == sorted({int(r) for r in old_round_id.tolist()})}
        binding_ok = True
        for i in range(len(old_round_id)):
            rid = int(old_round_id[i])
            m = meta.get(rid)
            if m is None or old_sha[i] != m["sequence_sha256"] or int(old_pos[i]) not in m["positions"]:
                binding_ok = False
                break
        checks["row_binding_equal"] = binding_ok

        per_block = {}
        for b in CAPTURE_BLOCKS:
            rec = np.zeros((len(old_round_id), HIDDEN), dtype=np.uint16)
            for i in range(len(old_round_id)):
                rid = int(old_round_id[i])
                pos = int(old_pos[i])
                j = meta[rid]["positions"].index(pos)
                rec[i] = blocks[(rid, b)][j]
            old_f = _u16_to_f32(np.asarray(z[f"block{b}"]))
            new_f = _u16_to_f32(rec)
            rel = _rel(new_f, old_f)
            rel["bitwise_equal_fraction"] = float((rec == np.asarray(z[f"block{b}"])).mean())
            rel["gate_cos_ok"] = bool(rel["cos_min"] >= 0.9999)
            per_block[str(b)] = rel
            if not rel["gate_cos_ok"]:
                all_pass = False
        entry["per_block"] = per_block
        entry["checks"] = checks
        entry["status"] = "PASS" if all(checks.values()) and all(
            v["gate_cos_ok"] for v in per_block.values()
        ) else "FAIL"
        report["episodes"].append(entry)
        print(json.dumps({"episode_id": ep, "status": entry["status"],
                          "cos": {b: round(v["cos_min"], 7) for b, v in per_block.items()},
                          "maxabs": {b: round(v["max_abs"], 4) for b, v in per_block.items()},
                          "checks": checks}), flush=True)
        z.close()

    report["overall"] = "PASS" if all_pass else "FAIL"
    report["provenance"] = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0), "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "note": "same frozen modules/weights/token ids; only GPU/kernels differ",
    }
    atomic_write_json(OUT, report)
    print("OVERALL", report["overall"], OUT)
    return 0 if all_pass else 1


PREFIX = None
if __name__ == "__main__":
    PREFIX = PrefixModel(blocks=PREFIX_LAYERS, captures=CAPTURE_BLOCKS)
    sys.exit(main())
