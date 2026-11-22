"""Print a compact causal trace for selected corrected-r3 episode IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdsafe_v6.rule_taint import extract_directives  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="+")
    ap.add_argument("--split", choices=("train", "dev"), default="dev")
    args = ap.parse_args()
    wanted = set(args.episode)
    path = Path("/root/sdsafe-compute/v6-param/artifacts/corrected-r3") / f"views_{args.split}.jsonl"
    found = {episode: [] for episode in args.episode}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            episode = str(rec["episode_id"])
            if episode in wanted:
                found[episode].append(rec)
    for episode in args.episode:
        records = sorted(found[episode], key=lambda rec: int(rec["row_index"]))
        if not records:
            print(json.dumps({"episode": episode, "missing": True}))
            continue
        print(json.dumps({"episode": episode, "group": records[0]["group_id"],
                          "label": records[0]["episode_label"],
                          "user": records[0]["user_request"]}, ensure_ascii=False))
        seen = set()
        for rec in records:
            for block in rec.get("prior") or []:
                if block.get("role") != "tool":
                    continue
                key = str(block.get("text") or "")
                if key in seen:
                    continue
                seen.add(key)
                ds = extract_directives(key, block_index=int(block.get("block_index", -1)))
                if ds:
                    print(json.dumps({"source": key, "directives": [d.__dict__ for d in ds]},
                                     ensure_ascii=False, default=list))
            cur = rec.get("current") or {}
            if cur.get("kind") == "tool_call":
                event = {"row": rec["row_index"], "kind": "tool_call",
                         "tool": cur.get("tool"), "arguments": cur.get("arguments"),
                         "value_sources": rec.get("value_sources")}
            else:
                event = {"row": rec["row_index"], "kind": "text",
                         "visible": str(cur.get("visible_so_far") or "")}
            print(json.dumps(event, ensure_ascii=False))


if __name__ == "__main__":
    main()
