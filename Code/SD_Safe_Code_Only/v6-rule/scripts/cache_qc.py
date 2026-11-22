"""Cache-only QC of the frozen teacher annotations (no network, no credentials, no new labels).

Reads only the three JSONL caches that were produced before the 402 block and the local view
files.  It never imports the network path in ``teacher`` and never touches the API key.

Outputs a compact JSON report plus a short stdout summary.  Definitions:

* a *response line* is one call attempt stored in a cache (duplicates/retries included);
* a *unique event* is a row with at least one parseable response;
* a *two-pass event* has both A and B parses; only those can contribute a supervised field;
* a *trainable field* follows PROTOCOL §4: both passes present and exactly equal, else masked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sdsafe_v6 import teacher

ROOT = Path("/root/sdsafe-compute/v6-param")
ART = ROOT / "artifacts"
CACHES = {
    "pilot400": ART / "teacher/pilot400_calls.jsonl",
    "aux_v2": ART / "teacher/aux_calls_v2.jsonl",
    "aux2": ART / "teacher/aux2_calls.jsonl",
}
VIEWS = {
    "pilot400": ART / "views/pilot400.jsonl",
    "aux": ART / "views/aux.jsonl",
    "aux2": ART / "views/aux2.jsonl",
}
FIELDS = (("action", "action_authorization", teacher.VOCAB["action_authorization"]),
          ("rel", "task_relation", teacher.VOCAB["task_relation"]),
          ("align", "external_directive_alignment", teacher.VOCAB["external_directive_alignment"]))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_views() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for name, path in VIEWS.items():
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[int(rec["row_index"])] = rec
    return out


def load_cache(path: Path) -> dict[str, Any]:
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    # best response per digest: prefer a parseable completion; later duplicate wins among equals
    best: dict[str, dict[str, Any]] = {}
    digest_calls = Counter()
    for item in lines:
        digest = item.get("digest")
        if not digest:
            continue
        digest_calls[digest] += 1
        parsed = teacher.parse_label(item.get("content") or "")
        cand = {"item": item, "parsed": parsed, "parse_ok": bool(item.get("ok") and parsed.get("parse_ok"))}
        prev = best.get(digest)
        if prev is None or (cand["parse_ok"] and not prev["parse_ok"]):
            best[digest] = cand
    by_row: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for cand in best.values():
        item = cand["item"]
        by_row[int(item["row_index"])][str(item.get("pass"))] = cand
    return {"lines": len(lines), "best": best, "digest_calls": digest_calls, "by_row": by_row}


def usage_of(item: dict[str, Any]) -> dict[str, int]:
    u = item.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    pd = u.get("prompt_tokens_details") or {}
    return {
        "prompt_tokens": int(u.get("prompt_tokens") or 0),
        "completion_tokens": int(u.get("completion_tokens") or 0),
        "total_tokens": int(u.get("total_tokens") or 0),
        "prompt_cache_hit_tokens": int(u.get("prompt_cache_hit_tokens") or 0),
        "prompt_cache_miss_tokens": int(u.get("prompt_cache_miss_tokens") or 0),
        "reasoning_tokens": int(det.get("reasoning_tokens") or 0),
        "prompt_cached_tokens_detail": int(pd.get("cached_tokens") or 0),
    }


def teacher_text(rec: dict[str, Any]) -> str:
    parts = [rec.get("user_request") or ""]
    for block in rec.get("prior") or []:
        parts.append(block.get("text") or "")
    cur = rec.get("current") or {}
    parts.append(json.dumps(cur.get("arguments"), ensure_ascii=False, sort_keys=True) or "")
    parts.append(cur.get("tool") or "")
    parts.append(cur.get("segment") or "")
    parts.append(cur.get("visible_so_far") or "")
    return "\n".join(parts)


def student_text(rec: dict[str, Any]) -> str:
    from sdsafe_v6 import view
    f = view.view_text_fields(rec)
    return "\n".join([f["user_request"], f["current"], f["visible_so_far"]] + f["sources"])


def arg_paths(rec: dict[str, Any]) -> set[str]:
    cur = rec.get("current") or {}
    return {str(x.get("path")) for x in (rec.get("value_sources") or [])}


def qc_file(name: str, cache: dict[str, Any], views: dict[int, dict[str, Any]]) -> dict[str, Any]:
    by_row = cache["by_row"]
    out: dict[str, Any] = {
        "file": str(CACHES[name]), "sha256": sha(CACHES[name]),
        "response_lines": cache["lines"], "unique_digests": len(cache["best"]),
        "digest_calls": {"calls": sum(cache["digest_calls"].values()),
                         "unique": len(cache["digest_calls"]),
                         "duplicate_calls": sum(v - 1 for v in cache["digest_calls"].values())},
        "unique_rows": len(by_row), "rows_in_views": sum(1 for r in by_row if r in views),
    }
    ok_lines = sum(1 for c in cache["best"].values() if c["item"].get("ok"))
    parse_lines = sum(1 for c in cache["best"].values() if c["parse_ok"])
    out["ok_digests"] = ok_lines
    out["parse_ok_digests"] = parse_lines
    out["nonparse_or_error_digests"] = len(cache["best"]) - parse_lines
    out["finish_reason"] = dict(Counter(str(c["item"].get("finish_reason")) for c in cache["best"].values()))
    passes = Counter()
    both = single = zero = 0
    for row, ps in by_row.items():
        pa = ps.get("A")
        pb = ps.get("B")
        okA = bool(pa and pa["parse_ok"])
        okB = bool(pb and pb["parse_ok"])
        passes["A" if okA else "noA"] += 1
        passes["B" if okB else "noB"] += 1
        if okA and okB:
            both += 1
        elif okA or okB:
            single += 1
        else:
            zero += 1
    out["events"] = {"both_parse": both, "single_parse": single, "no_parse": zero, "total_rows": len(by_row)}
    usage = Counter()
    usage_missing = 0
    for cand in cache["best"].values():
        u = usage_of(cand["item"])
        if not cand["item"].get("usage"):
            usage_missing += 1
        for k, v in u.items():
            usage[k] += v
    out["usage"] = dict(usage)
    out["usage_missing_digests"] = usage_missing
    # per-field agreement, unknown and invalid literals over two-pass events
    fstat = {k: Counter() for k, _f, _v in FIELDS}
    for row, ps in by_row.items():
        a, b = ps.get("A"), ps.get("B")
        if not (a and b and a["parse_ok"] and b["parse_ok"]):
            continue
        pa, pb = a["parsed"], b["parsed"]
        for key, field, allowed in FIELDS:
            fstat[key]["n"] += 1
            va, vb = pa.get(field), pb.get(field)
            fstat[key]["invalid_A"] += int(va not in allowed)
            fstat[key]["invalid_B"] += int(vb not in allowed)
            fstat[key]["unknown_A"] += int(va == "unclear")
            fstat[key]["unknown_B"] += int(vb == "unclear")
            fstat[key]["agree"] += int(va == vb)
            fstat[key]["agree_known"] += int(va == vb and va != "unclear")
            fstat[key]["disagree"] += int(va != vb)
    out["fields"] = {k: dict(v) for k, v in fstat.items()}

    # per-argument agreement on shared paths
    arg = Counter()
    per_field = {f: Counter() for f in ("authorization", "value_source", "derivation", "conflict_or_support")}
    path_valid = path_invalid = 0
    arg_rows_usable = 0
    arg_rows_positive = 0
    for row, ps in by_row.items():
        a, b = ps.get("A"), ps.get("B")
        rec = views.get(row)
        if not (a and b and a["parse_ok"] and b["parse_ok"] and rec):
            continue
        pa, pb = a["parsed"], b["parsed"]
        aa = {x["path"]: x for x in pa.get("arguments") or [] if x.get("path")}
        ba = {x["path"]: x for x in pb.get("arguments") or [] if x.get("path")}
        paths = set(aa) & set(ba)
        if paths:
            arg["events_with_shared_args"] += 1
        valid_paths = arg_paths(rec)
        for path in paths:
            arg["shared_paths"] += 1
            if path in valid_paths:
                path_valid += 1
            else:
                path_invalid += 1
            for field in per_field:
                per_field[field]["n"] += 1
                per_field[field]["agree"] += int(aa[path].get(field) == ba[path].get(field))
        # events usable for a fine-grained arg target: every shared path agrees on auth and none unclear
        votes = []
        for path in paths:
            av, bv = aa[path].get("authorization"), ba[path].get("authorization")
            if av == bv and av in ("authorized", "unauthorized"):
                votes.append(1 if av == "unauthorized" else 0)
            elif av != bv:
                votes.append(-1)
        if votes and all(v >= 0 for v in votes):
            arg_rows_usable += 1
            arg_rows_positive += int(any(v == 1 for v in votes))
    out["arguments"] = {
        **dict(arg), "path_valid": path_valid, "path_invalid": path_invalid,
        "usable_rows": arg_rows_usable, "usable_positive_rows": arg_rows_positive,
        "per_field": {k: dict(v) for k, v in per_field.items()},
    }

    # literal quotes and student visibility
    q = Counter()
    seen = 0
    for row, ps in by_row.items():
        rec = views.get(row)
        if rec is None:
            continue
        shown = teacher_text(rec)
        student = student_text(rec)
        for pid in ("A", "B"):
            cand = ps.get(pid)
            if not cand or not cand["parse_ok"]:
                continue
            parsed = cand["parsed"]
            seen += 1
            for field in ("user_authorization_quote", "external_requirement_quote"):
                val = parsed.get(field)
                if val is None:
                    q[f"{field}|null"] += 1
                elif val in shown:
                    q[f"{field}|literal"] += 1
                else:
                    q[f"{field}|not_literal"] += 1
                if val is not None:
                    q[f"{field}|visible_to_student"] += int(val in student)
            for a in parsed.get("arguments") or []:
                val = a.get("source_quote")
                if val is None:
                    q["arg_quote|null"] += 1
                elif val in shown:
                    q["arg_quote|literal"] += 1
                else:
                    q["arg_quote|not_literal"] += 1
                if val is not None:
                    q["arg_quote|visible_to_student"] += int(val in student)
    out["quotes"] = dict(q)
    out["quote_pass_parses"] = seen
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ART / "qc/teacher_cache_qc.json"))
    args = ap.parse_args()
    views = load_views()
    report: dict[str, Any] = {
        "study": "v6-param cache-only QC",
        "new_annotation_requests": 0,
        "network_used": False,
        "caches": {},
        "view_rows": {k: len({json.loads(l)["row_index"] for l in p.read_text().splitlines() if l.strip()})
                      for k, p in VIEWS.items() if p.is_file()},
    }
    caches = {name: load_cache(path) for name, path in CACHES.items() if path.is_file()}
    for name, cache in caches.items():
        report["caches"][name] = qc_file(name, cache, views)

    # union dataset: prefer aux_v2 then aux2 for a row; only both-parse fields
    union: dict[int, dict[str, Any]] = {}
    for name in ("aux_v2", "aux2", "pilot400"):
        cache = caches.get(name)
        if not cache:
            continue
        for row, ps in cache["by_row"].items():
            a, b = ps.get("A"), ps.get("B")
            if a and b and a["parse_ok"] and b["parse_ok"]:
                if row not in union:
                    union[row] = {"source": name, "A": a["parsed"], "B": b["parsed"],
                                  "view": row in views}
    train_rows = set()
    try:
        import numpy as np
        z = np.load(ART / "cache_train.npz", allow_pickle=True)
        train_rows = {int(x) for x in z["row_index"].tolist()}
    except Exception:
        pass
    ufields = {k: Counter() for k, _f, _v in FIELDS}
    for row, rec in union.items():
        for key, field, allowed in FIELDS:
            va, vb = rec["A"].get(field), rec["B"].get(field)
            ufields[key]["n"] += 1
            ufields[key]["agree"] += int(va == vb)
            ufields[key]["agree_known"] += int(va == vb and va != "unclear")
            ufields[key]["unknown"] += int(va == "unclear" and vb == "unclear")
            ufields[key]["in_train_cache"] += int(row in train_rows)
    uarg = Counter()
    for row, rec in union.items():
        aa = {x["path"]: x for x in rec["A"].get("arguments") or [] if x.get("path")}
        ba = {x["path"]: x for x in rec["B"].get("arguments") or [] if x.get("path")}
        paths = set(aa) & set(ba)
        if not paths:
            continue
        votes = []
        for path in paths:
            av, bv = aa[path].get("authorization"), ba[path].get("authorization")
            if av == bv and av in ("authorized", "unauthorized"):
                votes.append(1 if av == "unauthorized" else 0)
            elif av != bv:
                votes.append(-1)
        uarg["events_with_shared_args"] += 1
        if votes and all(v >= 0 for v in votes):
            uarg["usable_rows"] += 1
            uarg["usable_positive_rows"] += int(any(v == 1 for v in votes))
    report["union_two_pass"] = {
        "events": len(union), "in_train_cache": sum(1 for r in union if r in train_rows),
        "by_source": dict(Counter(r["source"] for r in union.values())),
        "fields": {k: dict(v) for k, v in ufields.items()},
        "arguments": dict(uarg),
    }
    # strict trainable counts reproduce run_arms' (buggy) collation and the corrected one
    policy = Counter()
    for row, rec in union.items():
        for key, field, allowed in FIELDS:
            if rec["A"].get(field) == rec["B"].get(field) and row in train_rows:
                policy[f"{key}|strict_both_agree"] += 1
    for name, cache in caches.items():
        for row, ps in cache["by_row"].items():
            if row not in train_rows:
                continue
            a, b = ps.get("A"), ps.get("B")
            has_any = bool((a and a["parse_ok"]) or (b and b["parse_ok"]))
            both = bool(a and b and a["parse_ok"] and b["parse_ok"])
            if has_any:
                policy[f"{name}|any_single_or_both"] += 1
            if both:
                policy[f"{name}|both"] += 1
    report["row_policy_counts"] = dict(policy)
    report["files_sha256"] = {name: sha(path) for name, path in CACHES.items() if path.is_file()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")

    # compact stdout (never dump annotation text)
    for name, r in report["caches"].items():
        print(f"[{name}] lines={r['response_lines']} digests={r['unique_digests']} "
              f"dups={r['digest_calls']['duplicate_calls']} rows={r['unique_rows']} "
              f"both={r['events']['both_parse']} single={r['events']['single_parse']} "
              f"noparse={r['events']['no_parse']} tokens={r['usage']['total_tokens']} "
              f"reasoning={r['usage']['reasoning_tokens']}")
        print("  fields", {k: (v['n'], v['agree'], v['unknown_A']) for k, v in r["fields"].items()})
        print("  args", r["arguments"])
        print("  quotes", r["quotes"])
    u = report["union_two_pass"]
    print(f"[union] events={u['events']} train={u['in_train_cache']} by_source={u['by_source']}")
    print("  fields", {k: (v['n'], v['agree'], v['unknown']) for k, v in u["fields"].items()})
    print("  args", u["arguments"])
    print("  policy", report["row_policy_counts"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
