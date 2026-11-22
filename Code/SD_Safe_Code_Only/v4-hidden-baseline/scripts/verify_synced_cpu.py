"""CPU-side durability verifier: every progress line's npz_sha256 must match the file."""
import hashlib, json, sys
from pathlib import Path
SPLIT = sys.argv[1]
D = Path("/root/autodl-tmp/sdsafe-lowcost-authorization-v4-gpu/capture") / SPLIT
EP = D / "episodes"
VER = D / "verified.jsonl"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
seen = set()
if VER.is_file():
    for line in VER.read_text().splitlines():
        if line.strip():
            seen.add(json.loads(line)["npz"])
rows = [json.loads(l) for l in (D / "progress.jsonl").read_text().splitlines() if l.strip()]
new_ok = new_bad = pending = 0
bad = []
out = open(VER, "a")
if True:
    for r in rows:
        npz = r.get("npz"); want = r.get("npz_sha256")
        if not npz or not want or npz in seen:
            continue
        f = EP / npz
        if not f.is_file():
            pending += 1; continue
        got = sha(f)
        if got != want:
            bad.append({"npz": npz, "error": "sha_mismatch", "want": want, "got": got}); new_bad += 1
        else:
            out.write(json.dumps({"npz": npz, "npz_sha256": got, "bytes": f.stat().st_size}) + "\n"); new_ok += 1
out.flush()
summary = {"split": SPLIT, "progress_lines": len(rows), "files": len(list(EP.glob("*.npz"))),
           "verified_total": len(seen) + new_ok, "new_verified": new_ok, "new_bad": new_bad,
           "pending_not_yet_synced": pending, "bad": bad[:5]}
(D / "verify_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
sys.exit(0 if new_bad == 0 else 2)
