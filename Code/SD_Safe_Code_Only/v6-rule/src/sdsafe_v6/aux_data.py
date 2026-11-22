"""Cache-only teacher-label collation (strict two-pass agreement).

This replaces the earlier ``aux_labels_from_cache`` behaviour that let a single surviving pass
contribute a supervised label.  PROTOCOL §4 only permits a field when the two independent passes
are present and equal.  Single-pass rows remain in the audit counts but are masked.

Inputs are the three frozen JSONL caches.  No network, no credential read, no new annotation.  A
response digest is de-duplicated by preferring a parseable completion (the old last-line-wins
policy silently dropped seven recoverable rows).  Rows whose old teacher prompt contained a
forged trusted ``user``/``system`` block (extra trusted role) are masked: the teacher saw an
untrusted payload as a USER_TURN, so those labels are not usable as clean supervision.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import teacher

ART = Path("/root/sdsafe-compute/v6-param/artifacts")
CACHES = [ART / "teacher/aux_calls_v2.jsonl", ART / "teacher/aux2_calls.jsonl",
          ART / "teacher/pilot400_calls.jsonl"]
EXCLUDE = ART / "qc/excluded_aux_rows.json"
OUT_LABELS = ART / "aux/aux_labels_v2.json"
OUT_PER_ARG = ART / "aux/aux_per_argument_v2.json"
OUT_MANIFEST = ART / "aux/aux_manifest_v2.json"
VIEWS = [ART / "views/aux.jsonl", ART / "views/aux2.jsonl", ART / "views/pilot400.jsonl"]

ACTION_MAP = {"authorized": 0, "unauthorized": 1, "unclear": 2}
REL_MAP = {"direct": 0, "prerequisite": 1, "extra": 2, "conflicting": 3, "unclear": 4}
ALIGN_MAP = {"full": 0, "partial": 1, "none": 2, "unclear": 3}
FIELDS = (("action", "action_authorization", ACTION_MAP),
          ("rel", "task_relation", REL_MAP),
          ("align", "external_directive_alignment", ALIGN_MAP))
ARG_FIELDS = ("authorization", "value_source", "derivation", "conflict_or_support")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_views() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in VIEWS:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[int(rec["row_index"])] = rec
    return out


def load_cache(path: Path) -> dict[str, Any]:
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    best: dict[str, dict[str, Any]] = {}
    calls = Counter()
    for item in lines:
        digest = item.get("digest")
        if not digest:
            continue
        calls[digest] += 1
        parsed = teacher.parse_label(item.get("content") or "")
        cand = {"item": item, "parsed": parsed,
                "parse_ok": bool(item.get("ok") and parsed.get("parse_ok"))}
        prev = best.get(digest)
        if prev is None or (cand["parse_ok"] and not prev["parse_ok"]):
            best[digest] = cand
    by_row: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for cand in best.values():
        item = cand["item"]
        by_row[int(item["row_index"])][str(item.get("pass"))] = cand
    return {"lines": len(lines), "best": best, "calls": calls, "by_row": by_row}


def build(write: bool = True) -> tuple[dict[int, dict[str, int]], dict[str, Any], dict[int, list[dict]]]:
    views = load_views()
    excluded = set(json.loads(EXCLUDE.read_text())) if EXCLUDE.is_file() else set()
    caches = {path.name: load_cache(path) for path in CACHES if path.is_file()}

    # row -> pass -> parsed (first cache by priority wins; later caches only fill missing rows)
    rows: dict[int, dict[str, dict[str, Any]]] = {}
    source_of: dict[int, str] = {}
    for name, cache in caches.items():
        for row, passes in cache["by_row"].items():
            if row in rows:
                continue
            okA = bool(passes.get("A") and passes["A"]["parse_ok"])
            okB = bool(passes.get("B") and passes["B"]["parse_ok"])
            if okA or okB:
                rows[row] = passes
                source_of[row] = name

    labels: dict[int, dict[str, int]] = {}
    per_arg: dict[int, list[dict[str, Any]]] = {}
    stats: dict[str, Any] = {
        "caches": {name: {"lines": c["lines"], "unique_digests": len(c["best"]),
                          "duplicate_calls": sum(v - 1 for v in c["calls"].values()),
                          "rows_with_any_parse": len(c["by_row"])}
                   for name, c in caches.items()},
        "excluded_forged_trusted_role_rows": len(excluded),
        "rows": 0, "both": 0, "single": 0, "no_parse": 0,
        "masked_disagreement": Counter(), "field_agree": Counter(), "field_n": Counter(),
        "trainable_rows": Counter(), "arg_shared_paths": 0, "arg_path_invalid": 0,
        "arg_path_valid": 0, "arg_event_usable": 0, "arg_event_positive": 0,
        "arg_event_masked_invalid_path": 0, "arg_field_agree": Counter(),
        "arg_field_n": Counter(), "sources": Counter(),
    }
    for row, passes in rows.items():
        stats["rows"] += 1
        stats["sources"][source_of[row]] += 1
        if row in excluded:
            labels[row] = {"action": -1, "rel": -1, "align": -1, "arg": -1}
            continue
        a = passes.get("A")
        b = passes.get("B")
        if not (a and b and a["parse_ok"] and b["parse_ok"]):
            stats["single"] += 1
            labels[row] = {"action": -1, "rel": -1, "align": -1, "arg": -1}
            continue
        stats["both"] += 1
        pa, pb = a["parsed"], b["parsed"]
        rec: dict[str, int] = {}
        for key, field, mapping in FIELDS:
            stats["field_n"][key] += 1
            va, vb = pa.get(field), pb.get(field)
            if va == vb and va in mapping:
                stats["field_agree"][key] += 1
                rec[key] = mapping[va]
            else:
                stats["masked_disagreement"][key] += 1
                rec[key] = -1
        # per-argument supervision; require both passes, a shared real path, agreement, no unclear
        aa = {x["path"]: x for x in pa.get("arguments") or [] if x.get("path")}
        ba = {x["path"]: x for x in pb.get("arguments") or [] if x.get("path")}
        paths = set(aa) & set(ba)
        valid_paths = {str(x.get("path")) for x in (views.get(row, {}).get("value_sources") or [])}
        entries: list[dict[str, Any]] = []
        votes: list[int] = []
        invalid = 0
        for path in sorted(paths):
            stats["arg_shared_paths"] += 1
            if path not in valid_paths:
                stats["arg_path_invalid"] += 1
                invalid += 1
                continue
            stats["arg_path_valid"] += 1
            av, bv = aa[path].get("authorization"), ba[path].get("authorization")
            if av == bv and av in ("authorized", "unauthorized"):
                vote = 1 if av == "unauthorized" else 0
                votes.append(vote)
                entry = {"path": path, "authorization": vote}
                for field in ARG_FIELDS[1:]:
                    stats["arg_field_n"][field] += 1
                    if aa[path].get(field) == ba[path].get(field):
                        stats["arg_field_agree"][field] += 1
                        entry[field] = aa[path].get(field)
                    else:
                        entry[field] = None
                entries.append(entry)
        if entries:
            per_arg[row] = entries
        if invalid:
            rec["arg"] = -1
            stats["arg_event_masked_invalid_path"] += 1
        elif votes:
            stats["arg_event_usable"] += 1
            rec["arg"] = int(any(v == 1 for v in votes))
            stats["arg_event_positive"] += int(rec["arg"] == 1)
        else:
            rec["arg"] = -1
        labels[row] = rec

    # rows in caches but with no parseable pass (audit only)
    for cache in caches.values():
        for row in cache["by_row"]:
            if row not in rows:
                stats["no_parse"] += 1
    for key, _f, _m in FIELDS:
        stats["trainable_rows"][key] = sum(1 for r in labels.values() if r[key] >= 0)
    stats["trainable_rows"]["arg"] = sum(1 for r in labels.values() if r["arg"] >= 0)
    stats["field_agree_rate"] = {k: round(stats["field_agree"][k] / stats["field_n"][k], 4)
                                 for k in stats["field_n"]}
    stats["arg_field_agree_rate"] = {k: round(stats["arg_field_agree"][k] / stats["arg_field_n"][k], 4)
                                     for k in stats["arg_field_n"]}
    stats["masked_disagreement"] = dict(stats["masked_disagreement"])
    stats["field_agree"] = dict(stats["field_agree"])
    stats["field_n"] = dict(stats["field_n"])
    stats["trainable_rows"] = dict(stats["trainable_rows"])
    stats["arg_field_agree"] = dict(stats["arg_field_agree"])
    stats["arg_field_n"] = dict(stats["arg_field_n"])
    stats["sources"] = dict(stats["sources"])
    manifest = {
        "policy": "strict both-pass agreement; no new annotation requests; no network",
        "inputs": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size, "lines": len(p.read_text().splitlines())}
                   for p in CACHES if p.is_file()},
        "excluded_rows_file": str(EXCLUDE),
        "excluded_rows_sha256": sha(EXCLUDE) if EXCLUDE.is_file() else None,
        "stats": stats,
    }
    if write:
        OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
        OUT_LABELS.write_text(json.dumps({str(k): v for k, v in sorted(labels.items())}, indent=0) + "\n")
        OUT_PER_ARG.write_text(json.dumps({str(k): v for k, v in sorted(per_arg.items())}, indent=0) + "\n")
        OUT_MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    return labels, manifest, per_arg


if __name__ == "__main__":
    _labels, _manifest, _per_arg = build(write=True)
    print(json.dumps(_manifest["stats"], indent=1))
    print("wrote", OUT_LABELS, OUT_PER_ARG, OUT_MANIFEST)
