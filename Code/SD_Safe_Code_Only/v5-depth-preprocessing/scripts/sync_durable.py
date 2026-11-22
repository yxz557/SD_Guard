"""Push a finished artifact sub-tree to the CPU durable store and verify SHA read-back.

GPU is the only writer of the GPU tree; the CPU node is passive storage.  This script:
1. finds files under ``--subdir`` not yet recorded in ``artifacts/_sync/state.json``,
2. hashes them locally, writes a POSIX sha256 manifest,
3. rsyncs exactly those files to the CPU v5 root (same relative layout),
4. runs ``sha256sum -c`` on the CPU over that manifest and records the result.

Only files listed in the verified manifest are considered durable.  A partial/interrupted
sync leaves them out of the state file, so the next call retries them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/root/sdsafe-compute/v5-depth")
CPU = "root@connect.bjb2.seetacloud.com"
DEST = "/root/autodl-tmp/sdsafe-lowcost-authorization-v5-depth"
SSH = [
    "ssh", "-p", "13661",
    "-i", "/root/sdsafe-runner-runtime/keys/id_ed25519_cpu_store",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "UserKnownHostsFile=/root/sdsafe-runner-runtime/known_hosts_cpu",
    "-o", "BatchMode=yes", CPU,
]
RSYNC_SSH = " ".join(SSH[:-1])


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", required=True, help="relative to the v5 study root")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    sub = Path(args.subdir)
    base = ROOT / sub
    sync_dir = ROOT / "artifacts" / "_sync"
    sync_dir.mkdir(parents=True, exist_ok=True)
    state_path = sync_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    MUTABLE = {"progress.jsonl", "coverage_run.json", "capture_provenance.json"}
    all_files = [p for p in sorted(base.rglob("*"))
                 if p.is_file() and not p.name.endswith(".tmp") and "_sync" not in p.parts]
    files = [p for p in all_files if p.name not in MUTABLE]
    mutable = [p for p in all_files if p.name in MUTABLE]
    todo = []
    for p in files:
        rel = str(p.relative_to(ROOT))
        rec = state.get(rel)
        if (rec is None or rec["bytes"] != p.stat().st_size
                or rec.get("mtime") != int(p.stat().st_mtime)):
            todo.append((rel, p))
    stamp = time.strftime("%Y%m%dT%H%M%S")
    label = args.label or sub.name
    manifest = sync_dir / f"batch_{label}_{stamp}.sha256"
    result_path = sync_dir / f"batch_{label}_{stamp}.result.json"
    if not todo:
        result_path.write_text(json.dumps({"label": label, "new_files": 0, "verified": True,
                                           "note": "nothing new"}, indent=2) + "\n")
        print(json.dumps({"label": label, "new_files": 0, "verified": True}))
        return
    lines = []
    for rel, p in todo:
        digest = sha256(p)
        lines.append(f"{digest}  {rel}")
        state[rel] = {"sha256": digest, "bytes": p.stat().st_size,
                      "mtime": int(p.stat().st_mtime), "synced_at": stamp}
    manifest.write_text("\n".join(lines) + "\n")
    # make sure the CPU destination dirs exist, then transfer only this batch
    subprocess.run(SSH + [f"mkdir -p {DEST}"], check=True)
    files_from = sync_dir / f"batch_{label}_{stamp}.files"
    files_from.write_text("\n".join(rel for rel, _ in todo) + "\n")
    t0 = time.time()
    subprocess.run(
        ["rsync", "-a", f"--files-from={files_from}", "-e", RSYNC_SSH,
         f"{ROOT}/", f"{CPU}:{DEST}/"],
        check=True,
    )
    transfer = round(time.time() - t0, 1)
    # ship the manifest, then read every transferred file back and check its SHA
    subprocess.run(SSH + [f"mkdir -p {DEST}/_sync"], check=True)
    subprocess.run(["rsync", "-a", "-e", RSYNC_SSH, str(manifest), f"{CPU}:{DEST}/_sync/"], check=True)
    remote = subprocess.run(
        SSH + [f"cd {DEST} && sha256sum -c {DEST}/_sync/{manifest.name} 2>&1 | tail -3; "
               f"cd {DEST} && sha256sum -c {DEST}/_sync/{manifest.name} >/dev/null 2>&1; echo RC=$?"],
        capture_output=True, text=True,
    )
    ok = "RC=0" in remote.stdout
    payload = {
        "label": label,
        "new_files": len(todo),
        "bytes": int(sum(p.stat().st_size for _, p in todo)),
        "transfer_seconds": transfer,
        "remote_check_stdout": remote.stdout.strip().splitlines()[-5:],
        "remote_check_stderr": remote.stderr.strip().splitlines()[-5:],
        "verified": bool(ok),
        "manifest": str(manifest.relative_to(ROOT)),
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    if not ok:
        # do not mark unverified files as durable
        for rel, _ in todo:
            state.pop(rel, None)
        state_path.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
        raise SystemExit(f"SHA read-back FAILED for {label}: {payload['remote_check_stdout']}")
    state_path.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    # mutable snapshots (progress.jsonl / coverage): exact bytes we hashed are the bytes sent
    mutable_results = []
    for p in mutable:
        data = p.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        rel = p.relative_to(ROOT)
        cmd = (f"mkdir -p {DEST}/{rel.parent} && cat > {DEST}/{rel}.tmp && "
               f"mv {DEST}/{rel}.tmp {DEST}/{rel} && sha256sum {DEST}/{rel}")
        rr = subprocess.run(SSH + [cmd], input=data, capture_output=True)
        got = (rr.stdout.decode().strip().split() or [""])[0]
        ok = rr.returncode == 0 and got == digest
        mutable_results.append({"file": str(rel), "sha256": digest, "verified": ok})
        if not ok:
            payload["verified"] = False
            raise SystemExit(f"mutable read-back FAILED for {rel}: rc={rr.returncode} {rr.stderr[-300:]}")
    payload["mutable"] = mutable_results
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
