"""Teacher-label quality audit: schema, quoting, two-pass agreement, student-view coverage.

Nothing here is called "human gold".  The numbers are: how often the two independent passes
agree per field, how often every quoted span literally occurs in the text the teacher was
shown, and how often the evidence a pass cites is even *visible* to the student encoder.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from . import teacher, view


def _sub(a: str | None, text: str) -> bool:
    if not a:
        return False
    return a in text


def _student_text(rec: dict[str, Any]) -> str:
    fields = view.view_text_fields(rec)
    return "\n".join([fields["user_request"], fields["current"], fields["visible_so_far"]] + fields["sources"])


def audit(views: list[dict[str, Any]], calls: list[dict[str, Any]]) -> dict[str, Any]:
    by_row: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    raw_ok = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    for item in calls:
        if not item.get("ok"):
            continue
        parsed = teacher.parse_label(item["content"])
        parsed["_finish"] = item.get("finish_reason")
        by_row[item["row_index"]][item["pass"]] = parsed
        raw_ok += 1
        u = item.get("usage") or {}
        for k in usage:
            usage[k] += int(u.get(k) or 0)
    view_by_row = {int(r["row_index"]): r for r in views}

    report: dict[str, Any] = {"calls_ok": raw_ok, "events": len(by_row), "usage": usage}
    strata = Counter()
    agree = Counter()
    total = Counter()
    quote_stats = Counter()
    coverage = defaultdict(Counter)
    mismatched = defaultdict(list)
    by_ep_label = defaultdict(Counter)
    conf_bins = defaultdict(Counter)
    arg_total = arg_agree = 0
    arg_field_agree = Counter()
    arg_field_total = Counter()

    for row, passes in by_row.items():
        rec = view_by_row.get(row)
        if rec is None:
            continue
        stratum = _stratum(rec)
        strata[stratum] += 1
        pa, pb = passes.get("A"), passes.get("B")
        if not pa or not pb or not pa.get("parse_ok") or not pb.get("parse_ok"):
            continue
        for field in ("action_authorization", "task_relation", "external_directive_alignment"):
            total[field] += 1
            if pa.get(field) is not None and pa.get(field) == pb.get(field):
                agree[field] += 1
            elif pa.get(field) != pb.get(field):
                mismatched[field].append({"row": row, "a": pa.get(field), "b": pb.get(field)})
        conf = min(pa.get("confidence") or 0.0, pb.get("confidence") or 0.0)
        bucket = "0.0-0.6" if conf < 0.6 else "0.6-0.8" if conf < 0.8 else "0.8-0.9" if conf < 0.9 else "0.9-1.0"
        for field in ("action_authorization", "external_directive_alignment"):
            conf_bins[(field, bucket)][pa.get(field) == pb.get(field)] += 1

        # per-argument agreement on the shared json paths
        a_args = {a["path"]: a for a in pa.get("arguments") or []}
        b_args = {a["path"]: a for a in pb.get("arguments") or []}
        for path in set(a_args) & set(b_args):
            arg_total += 1
            if a_args[path]["authorization"] == b_args[path]["authorization"]:
                arg_agree += 1
            for field in ("authorization", "value_source", "derivation", "conflict_or_support"):
                arg_field_total[field] += 1
                if a_args[path][field] == b_args[path][field]:
                    arg_field_agree[field] += 1

        # quote literalism against exactly what the teacher was shown
        shown = _teacher_text(rec)
        for field in ("user_authorization_quote", "external_requirement_quote"):
            for pass_id, parsed in (("A", pa), ("B", pb)):
                q = parsed.get(field)
                if q is None:
                    quote_stats[(field, "null")] += 1
                elif _sub(q, shown):
                    quote_stats[(field, "literal")] += 1
                else:
                    quote_stats[(field, "not_literal")] += 1
        for parsed, pass_id in ((pa, "A"), (pb, "B")):
            for arg in parsed.get("arguments") or []:
                q = arg.get("source_quote")
                if q is None:
                    quote_stats[("arg_quote", "null")] += 1
                    continue
                if _sub(q, shown):
                    quote_stats[("arg_quote", "literal")] += 1
                else:
                    quote_stats[("arg_quote", "not_literal")] += 1

        # student-view coverage of the evidence the teacher cites
        student = _student_text(rec)
        for pass_id, parsed in (("A", pa), ("B", pb)):
            flagged = [a for a in parsed.get("arguments") or [] if a.get("authorization") == "unauthorized"]
            for arg in flagged:
                q = arg.get("source_quote")
                if q is None:
                    continue
                coverage[(pass_id, "unauthorized_arg_quote_visible")][_sub(q, student)] += 1
            q = parsed.get("external_requirement_quote")
            if q:
                coverage[(pass_id, "external_quote_visible")][_sub(q, student)] += 1

        by_ep_label[(stratum, bool(rec["row_label"]))][pa.get("action_authorization")] += 1

    report["strata"] = dict(strata)
    report["agreement"] = {f: {"n": total[f], "agree": agree[f],
                               "rate": round(agree[f] / total[f], 4) if total[f] else None}
                           for f in total}
    report["arg_agreement"] = {"n": arg_total, "agree": arg_agree,
                               "rate": round(arg_agree / arg_total, 4) if arg_total else None,
                               "per_field": {f: {"n": arg_field_total[f], "agree": arg_field_agree[f],
                                                 "rate": round(arg_field_agree[f] / arg_field_total[f], 4)
                                                 if arg_field_total[f] else None}
                                             for f in arg_field_total}}
    report["quote_stats"] = {f"{k[0]}|{k[1]}": v for k, v in quote_stats.items()}
    report["student_view_coverage"] = {
        f"{k[0]}|{k[1]}": {"present": v.get(True, 0), "absent": v.get(False, 0),
                           "rate": round(v.get(True, 0) / max(1, v.get(True, 0) + v.get(False, 0)), 4)}
        for k, v in coverage.items()}
    report["confidence_agreement"] = {f"{k[0]}|{k[1]}": {"agree": v.get(True, 0), "disagree": v.get(False, 0)}
                                      for k, v in conf_bins.items()}
    report["stratum_x_rowlabel_x_action"] = {f"{k[0]}|rowlabel={k[1]}": dict(v)
                                             for k, v in by_ep_label.items()}
    report["mismatch_examples"] = {k: v[:6] for k, v in mismatched.items()}
    return report


def _stratum(rec: dict[str, Any]) -> str:
    row_label = bool(rec["row_label"])
    ep = bool(rec["episode_label"])
    kind = rec["event_type"]
    if row_label and kind == "tool_call":
        return "pos_call"
    if row_label and kind == "content_segment":
        return "pos_text"
    if (not row_label) and ep:
        return "pre_call"
    return "neg_%s" % ("call" if kind == "tool_call" else "text")


def _teacher_text(rec: dict[str, Any]) -> str:
    """Exactly the trajectory material the prompt contains (system/pass text excluded)."""
    parts = [rec.get("user_request") or ""]
    for block in rec.get("prior") or []:
        parts.append(block.get("text") or "")
    cur = rec.get("current") or {}
    parts.append(json.dumps(cur.get("arguments"), ensure_ascii=False, sort_keys=True) or "")
    parts.append(cur.get("tool") or "")
    parts.append(cur.get("segment") or "")
    parts.append(cur.get("visible_so_far") or "")
    return "\n".join(parts)
