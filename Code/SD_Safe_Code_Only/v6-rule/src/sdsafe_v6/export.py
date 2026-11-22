"""Export strictly causal event views from the exact prepared token ids (GPU node).

Usage::

    python -m sdsafe_v6.export --split train --rows-file <json list of row_index> --out ...
    python -m sdsafe_v6.export --split dev --all --out ...

The output is a JSONL file of view records (``view.VIEW_VERSION``) plus a manifest that hashes
every record and pins the tokenizer files.  Nothing is written outside ``--out``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from . import data, view

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")


def tokenizer_hashes() -> dict[str, str]:
    out = {}
    for name in TOKENIZER_FILES:
        path = data.MODEL_DIR / name
        if path.is_file():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--rows-file", help="JSON file with a list of row_index (or dict of lists)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episode-file", help="JSON list of episode_id to export every CT row of")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)
    rows = data.load_rows(args.split)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted: list[int] = []
    if args.all:
        wanted = sorted(rows)
    elif args.episode_file:
        ids = set(json.loads(Path(args.episode_file).read_text()))
        wanted = sorted(r for r, row in rows.items() if row["episode_id"] in ids)
    else:
        payload = json.loads(Path(args.rows_file).read_text())
        if isinstance(payload, dict):
            for v in payload.values():
                wanted.extend(int(x) for x in v)
        else:
            wanted = [int(x) for x in payload]
        wanted = sorted(set(wanted))
    if args.limit:
        wanted = wanted[: args.limit]

    by_ep: dict[str, list[int]] = {}
    for r in wanted:
        by_ep.setdefault(rows[r]["episode_id"], []).append(r)

    t0 = time.time()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for episode_id, row_ids in sorted(by_ep.items()):
        rec = data.load_prepared(args.split, episode_id)
        round_ids = [int(x) for x in rec["round_id"].tolist()]
        seq = [int(x) for x in rec["seq_len"].tolist()]
        prompt = [int(x) for x in rec["prompt_length"].tolist()]
        offsets = [int(x) for x in rec["ids_offsets"].tolist()]
        ids = rec["ids_flat"]
        pos_of = {int(v): i for i, v in enumerate(rec["row_index"].tolist())}
        for r in sorted(row_ids):
            row = dict(rows[r])
            i = pos_of.get(int(r))
            if i is None:
                failures.append({"row_index": int(r), "episode_id": episode_id,
                                 "error": "row_index absent from prepared episode"})
                continue
            # authoritative span/round binding comes from the prepared payload
            if (int(rec["token_start"][i]), int(rec["token_end"][i])) != (
                    int(row["token_start"]), int(row["token_end"])):
                failures.append({"row_index": int(r), "episode_id": episode_id,
                                 "error": "prepared span disagrees with accepted row span"})
                continue
            row["token_start"] = int(rec["token_start"][i])
            row["token_end"] = int(rec["token_end"][i])
            row["row_round_id"] = int(rec["row_round_id"][i])
            row["event_id"] = str(rec["event_id"][i])
            try:
                records.append(view.build_view(
                    episode_id=episode_id, row=row, round_ids=round_ids,
                    round_seq_len=seq, round_prompt_len=prompt, offsets=offsets, ids=ids,
                    decode=tok.decode, row_label=bool(row["label_value"]),
                    episode_label=bool(row["strict_final"]), group_id=row["group_id"],
                    scope=row["scope"],
                ))
            except Exception as error:  # fail closed, keep going, report
                failures.append({"row_index": int(r), "episode_id": episode_id,
                                 "error": f"{type(error).__name__}: {error}"})
    records.sort(key=lambda r: r["row_index"])
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    manifest = {
        "view_version": view.VIEW_VERSION,
        "split": args.split,
        "records": len(records),
        "failures": len(failures),
        "failure_detail": failures[:20],
        "out": str(out_path),
        "out_sha256": digest,
        "rows_requested": len(wanted),
        "tokenizer": tokenizer_hashes(),
        "model_dir": str(data.MODEL_DIR),
        "seconds": round(time.time() - t0, 2),
        "truncation": {"user": view.MAX_USER_CHARS, "head": view.HEAD_CHARS, "tail": view.TAIL_CHARS,
                       "prior_blocks": view.MAX_PRIOR_BLOCKS, "current": view.MAX_CURRENT_CHARS},
    }
    Path(str(out_path) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "failure_detail"}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
