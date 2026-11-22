"""CPU-only exporter: exact prepared token-id jobs for ALL 8229 TRAIN + 2028 common DEV.

New/versioned script: it does not modify the frozen v4 sources or the earlier bridge export.
Reuses the already validated 4623 TRAIN jobs (hard-linked into the new directory) and
exports only what is missing, in parallel worker processes.  Every produced job is re-read
through the frozen ``load_episode`` which re-proves each round's token digest against the
accepted per-row ``sequence_sha256``.

Run on the CPU/storage node with /root/vllm_env_backup/bin/python.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

V4 = Path("/root/autodl-tmp/sdsafe-lowcost-authorization-v4")
BRIDGE = Path("/root/autodl-tmp/sdsafe-gpu-migration-v1/bridge")
OLD_TRAIN = BRIDGE / "prepared" / "train"
OUT = BRIDGE / "prepared_all"
WORKERS = int(os.environ.get("EXPORT_WORKERS", "12"))

sys.path.insert(0, str(V4 / "src"))
sys.path.insert(0, str(BRIDGE))


def chunks(items, n):
    if not items:
        return []
    size = (len(items) + n - 1) // n
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def worker(job):
    split, scope, eps, outdir, part_id = job
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from sdsafe_v4.seqsrc import RowIndex, iter_cot_round_tasks, iter_round_tasks
    from prepared import load_episode, write_episode

    t0 = time.time()
    ri = RowIndex(split)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    index = {}
    manifest = {}
    tokens = 0
    bytes_ = 0
    buf = {}
    cur = None

    def flush():
        nonlocal tokens, bytes_
        if not buf:
            return
        eid = next(iter(buf))
        pairs = buf.pop(eid)
        p = write_episode(pairs, outdir)
        load_episode(p)  # independent digest re-proof of every round
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        index[eid] = p.name
        manifest[p.name] = {
            "episode_id": eid,
            "scope": scope,
            "split": split,
            "bytes": p.stat().st_size,
            "sha256": h,
            "rounds": len(pairs),
            "tokens": int(sum(len(ids) for _, ids in pairs)),
            "rows": int(sum(len(t.events) for t, _ in pairs)),
        }
        tokens += manifest[p.name]["tokens"]
        bytes_ += manifest[p.name]["bytes"]

    if scope == "cot":
        it = iter_cot_round_tasks(ri, eps)
    else:
        it = iter_round_tasks(ri, eps)
    for task, ids in it:
        eid = task.episode_id
        if cur is not None and eid != cur:
            flush()
        cur = eid
        buf.setdefault(eid, []).append((task, ids))
    flush()
    frag = outdir / f".fragment_{part_id}.json"
    frag.write_text(json.dumps({
        "split": split, "scope": scope, "part": part_id,
        "episodes": len(index), "tokens": tokens, "bytes": bytes_,
        "seconds": round(time.time() - t0, 1),
        "index": index, "manifest": manifest,
    }))
    return part_id, len(index), tokens


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from sdsafe_v4.capture import CAPTURE_SCOPES
    from sdsafe_v4.seqsrc import RowIndex, dev_common_universe
    # ---- reuse the validated 4623 TRAIN jobs by hard link (same filesystem)
    old_index = json.loads((OLD_TRAIN / "index.json").read_text())
    old_manifest = json.loads((OLD_TRAIN / "manifest.json").read_text())
    out_train = OUT / "train"
    out_dev = OUT / "dev"
    for d in (out_train, out_dev):
        d.mkdir(parents=True, exist_ok=True)
    linked = 0
    for eid, name in old_index.items():
        dst = out_train / name
        src = OLD_TRAIN / name
        if not dst.exists():
            os.link(src, dst)
            linked += 1
    reuse_index = dict(old_index)
    reuse_manifest = {k: v for k, v in old_manifest["files"].items() if k in set(old_index.values())}

    rtrain = RowIndex("train")
    all_train = [e for e in rtrain.episodes() if rtrain.episode_scope.get(e) in CAPTURE_SCOPES]
    remain_train = [e for e in all_train if e not in reuse_index]
    rdev = RowIndex("dev")
    common = dev_common_universe()
    dev_eps = [e for e in common if rdev.episode_scope.get(e) in CAPTURE_SCOPES]
    print(json.dumps({
        "train_universe": len(all_train), "train_reused": len(reuse_index),
        "train_remaining": len(remain_train), "dev_common": len(common),
        "dev_to_export": len(dev_eps), "hardlinked": linked,
        "train_scopes": {s: sum(1 for e in all_train if rtrain.episode_scope[e] == s)
                         for s in sorted(set(rtrain.episode_scope[e] for e in all_train))},
        "dev_scopes": {s: sum(1 for e in dev_eps if rdev.episode_scope[e] == s)
                       for s in sorted(set(rdev.episode_scope[e] for e in dev_eps))},
    }), flush=True)

    jobs = []
    pid = 0
    for split, outdir, ri_eps in (("train", out_train, remain_train), ("dev", out_dev, dev_eps)):
        by_scope = {}
        scope_of = rdev if split == "dev" else rtrain
        for e in ri_eps:
            by_scope.setdefault(scope_of.episode_scope[e], []).append(e)
        for scope, eps in sorted(by_scope.items()):
            nparts = max(1, min(WORKERS, (len(eps) + 199) // 200))
            for part in chunks(eps, nparts):
                jobs.append((split, scope, part, str(outdir), pid))
                pid += 1
    print(json.dumps({"jobs": len(jobs)}), flush=True)

    t0 = time.time()
    results = []
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for res in pool.imap_unordered(worker, jobs):
            results.append(res)
            print(json.dumps({"done_part": res, "elapsed": round(time.time() - t0, 1)}), flush=True)

    # ---- merge fragments
    idx = {"train": dict(reuse_index), "dev": {}}
    man = {"train": dict(reuse_manifest), "dev": {}}
    for frag in list(out_train.glob(".fragment_*.json")) + list(out_dev.glob(".fragment_*.json")):
        d = json.loads(frag.read_text())
        idx[d["split"]].update(d["index"])
        man[d["split"]].update(d["manifest"])
        frag.unlink()
    for split in ("train", "dev"):
        d = OUT / split
        if len(idx[split]) != len(set(idx[split].values())):
            raise SystemExit(f"duplicate job filenames in {split}")
        (d / "index.json").write_text(json.dumps(idx[split], indent=1, sort_keys=True) + "\n")
        (d / "manifest.json").write_text(json.dumps({
            "files": man[split],
            "episodes": len(idx[split]),
            "total_tokens": int(sum(v["tokens"] for v in man[split].values())),
            "total_bytes": int(sum(v["bytes"] for v in man[split].values())),
            "source_head": "ebed68372028654ca2d9bc33e29beafb915e858f",
            "note": "train jobs 4623 reused (hardlinked, SHA from original export); "
                    "remaining train + all common dev exported by export_prepared_all.py",
        }, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"split": split, "episodes": len(idx[split]),
                          "tokens": int(sum(v["tokens"] for v in man[split].values())),
                          "bytes": int(sum(v["bytes"] for v in man[split].values()))}), flush=True)
    print(json.dumps({"seconds": round(time.time() - t0, 1), "workers": WORKERS}), flush=True)


if __name__ == "__main__":
    main()
