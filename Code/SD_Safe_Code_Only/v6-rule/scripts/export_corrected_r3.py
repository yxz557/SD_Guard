"""Export corrected-r3 views for all TRAIN/DEV CT rows (call-localisation + empty policy).

Writes only under ``artifacts/corrected-r3/``.  Old views are read (read-only) for a diff
report.  No teacher call, no network, no TEST material, no label enters a feature.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v6 import data, feats, view  # noqa: E402

ART = Path("/root/sdsafe-compute/v6-param/artifacts")
OUT = ART / "corrected-r3"
OLD_VIEWS = {
    "train": [ART / "views/train_all.jsonl", ART / "views/train_all_fixup.jsonl"],
    "dev": [ART / "views/dev_common.jsonl", ART / "views/dev_common_fixup.jsonl"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_old(split: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in OLD_VIEWS[split]:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[int(rec["row_index"])] = rec
    return out


def main() -> int:
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)
    summary: dict[str, Any] = {"view_version": view.VIEW_VERSION, "splits": {}}
    for split in ("train", "dev"):
        rows = data.load_rows(split)
        if split == "dev":
            common = set(data.dev_common_ids())
            rows = {r: v for r, v in rows.items() if v["episode_id"] in common}
        old = load_old(split)
        by_ep: dict[str, list[int]] = defaultdict(list)
        for r, row in rows.items():
            by_ep[row["episode_id"]].append(r)
        records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for episode_id, row_ids in sorted(by_ep.items()):
            rec = data.load_prepared(split, episode_id)
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
                if (int(rec["token_start"][i]), int(rec["token_end"][i])) != (
                        int(row["token_start"]), int(row["token_end"])):
                    failures.append({"row_index": int(r), "episode_id": episode_id,
                                     "error": "prepared span disagrees with accepted row span"})
                    continue
                row["token_start"] = int(rec["token_start"][i])
                row["token_end"] = int(rec["token_end"][i])
                row["row_round_id"] = int(rec["row_round_id"][i])
                row["event_id"] = str(rec["event_id"][i])
                row["call_index"] = (None if int(rec["call_index"][i]) < 0
                                     else int(rec["call_index"][i]))
                try:
                    records.append(view.build_view(
                        episode_id=episode_id, row=row, round_ids=round_ids,
                        round_seq_len=seq, round_prompt_len=prompt, offsets=offsets, ids=ids,
                        decode=tok.decode, row_label=bool(row["label_value"]),
                        episode_label=bool(row["strict_final"]), group_id=row["group_id"],
                        scope=row["scope"],
                    ))
                except Exception as error:
                    failures.append({"row_index": int(r), "episode_id": episode_id,
                                     "scope": row["scope"], "event_type": row["event_type"],
                                     "error": f"{type(error).__name__}: {error}"})
        records.sort(key=lambda r: r["row_index"])
        out_path = OUT / f"views_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

        stats: Counter = Counter()
        by_scope_changed: Counter = Counter()
        for rec in records:
            ri = int(rec["row_index"])
            stats["rows"] += 1
            stats[f"scope:{rec['scope']}"] += 1
            if rec["empty_segment"]:
                stats["empty_segment"] += 1
            if rec["no_new_emission"]:
                stats["no_new_emission"] += 1
            if rec["unknown_adapter"]:
                stats["unknown_adapter"] += 1
            if rec["block_fallback"]:
                stats["block_fallback"] += 1
            cur = rec["current"]
            if cur["call_binding_conflict"]:
                stats["call_binding_conflict"] += 1
            if cur["call_match"]:
                stats[f"call_match:{cur['call_match']}"] += 1
            prev = old.get(ri)
            if prev is None:
                stats["old_view_missing"] += 1
                continue
            if prev["current"].get("tool") != cur.get("tool") or \
                    prev["current"].get("arguments") != cur.get("arguments"):
                stats["changed_current_call"] += 1
                stats[f"changed_current_call_ci{cur.get('call_index')}"] += 1
                by_scope_changed[rec["scope"]] += 1
            if prev.get("value_sources") != rec.get("value_sources"):
                stats["changed_value_sources"] += 1
            if prev.get("user_request") != rec.get("user_request"):
                stats["changed_user_request"] += 1
            if [b.get("role") for b in prev.get("prior") or []] != \
                    [b.get("role") for b in rec.get("prior") or []]:
                stats["changed_prior_roles"] += 1
            if [b.get("text") for b in prev.get("prior") or []] != \
                    [b.get("text") for b in rec.get("prior") or []]:
                stats["changed_prior_text"] += 1
        manifest = {
            "view_version": view.VIEW_VERSION,
            "split": split,
            "records": len(records),
            "failures": len(failures),
            "failure_detail": failures[:20],
            "rows_requested": len(rows),
            "out_sha256": sha(out_path),
            "old_views": {p.name: sha(p) for p in OLD_VIEWS[split] if p.is_file()},
            "tokenizer": {name: sha(data.MODEL_DIR / name)
                          for name in ("tokenizer.json", "tokenizer_config.json",
                                       "vocab.json", "merges.txt")
                          if (data.MODEL_DIR / name).is_file()},
            "stats": dict(stats),
            "changed_current_call_by_scope": dict(by_scope_changed),
            "policy": "PROTOCOL_CORRECTED_R3 C1/C3",
        }
        (OUT / f"views_{split}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summary["splits"][split] = manifest
        print(json.dumps({k: v for k, v in manifest.items() if k != "failure_detail"}, indent=1))

        # feature cache: scoreable rows only (no_new_emission / unknown_adapter have no logit)
        scoreable = [r for r in records if not r["no_new_emission"] and not r["unknown_adapter"]]
        cache_path = OUT / f"cache_{split}.npz"
        info = feats.build_cache(scoreable, cache_path)
        (OUT / f"cache_{split}.manifest.json").write_text(json.dumps({
            "cache": cache_path.name, "sha256": sha(cache_path), "info": info,
            "scoreable_rows": len(scoreable),
            "excluded_no_new_emission": sum(1 for r in records if r["no_new_emission"]),
            "excluded_unknown_adapter": sum(1 for r in records if r["unknown_adapter"]),
            "view_sha256": sha(out_path),
        }, indent=1) + "\n")
        print(json.dumps(info))
    (OUT / "EXPORT_SUMMARY.json").write_text(json.dumps(summary, indent=1) + "\n")
    return 0 if all(s["failures"] == 0 for s in summary["splits"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
