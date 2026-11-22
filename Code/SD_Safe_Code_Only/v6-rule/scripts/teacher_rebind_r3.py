"""Corrected-r3 teacher-label collation: rebind the frozen cache to the corrected view.

Zero new annotation requests.  The three teacher caches are read-only.  A cached event is
supervised only when:

* its old rendered prompt used the corrected current call (name + arguments),
* the user request and prior block material are unchanged,
* both independent passes are present, parseable and exactly equal on the field,
* the teacher's cited quote and every argument path are literally valid in the corrected
  causal material.

Anything else is masked and counted with a reason.  No cache line is edited.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v6 import teacher  # noqa: E402

ART = Path("/root/sdsafe-compute/v6-param/artifacts")
OUT = ART / "corrected-r3" / "aux"
CACHES = [ART / "teacher/aux_calls_v2.jsonl", ART / "teacher/aux2_calls.jsonl",
          ART / "teacher/pilot400_calls.jsonl"]
OLD_VIEWS = [ART / "views/aux.jsonl", ART / "views/aux2.jsonl", ART / "views/pilot400.jsonl"]
EXCLUDE = ART / "qc/excluded_aux_rows.json"
FROZEN = ART / "teacher/FROZEN_MANIFEST.json"

ACTION_MAP = {"authorized": 0, "unauthorized": 1, "unclear": 2}
REL_MAP = {"direct": 0, "prerequisite": 1, "extra": 2, "conflicting": 3, "unclear": 4}
ALIGN_MAP = {"full": 0, "partial": 1, "none": 2, "unclear": 3}
FIELDS = (("action", "action_authorization", ACTION_MAP),
          ("rel", "task_relation", REL_MAP),
          ("align", "external_directive_alignment", ALIGN_MAP))
ARG_FIELDS = ("authorization", "value_source", "derivation", "conflict_or_support")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_old_views() -> dict[int, dict[str, Any]]:
    out = {}
    for path in OLD_VIEWS:
        for rec in load_jsonl(path):
            out[int(rec["row_index"])] = rec
    return out


def load_new_views() -> dict[int, dict[str, Any]]:
    return {int(r["row_index"]): r for r in load_jsonl(OUT.parent / "views_train.jsonl")}


def best_cache_rows() -> dict[int, dict[str, dict[str, Any]]]:
    """row -> pass -> best parseable candidate (same policy as the frozen collation)."""
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for path in CACHES:
        if not path.is_file():
            continue
        for item in load_jsonl(path):
            digest = item.get("digest")
            if not digest:
                continue
            parsed = teacher.parse_label(item.get("content") or "")
            cand = {"item": item, "parsed": parsed,
                    "parse_ok": bool(item.get("ok") and parsed.get("parse_ok"))}
            key = (str(digest), int(item["row_index"]))
            prev = best.get(key)
            if prev is None or (cand["parse_ok"] and not prev["parse_ok"]):
                best[key] = cand
    by_row: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for cand in best.values():
        item = cand["item"]
        by_row[int(item["row_index"])][str(item.get("pass"))] = cand
    return by_row


def _literal(quote: Any, haystacks: list[str]) -> bool:
    if quote is None:
        return True
    if not isinstance(quote, str) or not quote:
        return False
    return any(quote in text for text in haystacks)


def material(rec: dict[str, Any]) -> tuple[list[str], str, str, list[str]]:
    user = str(rec.get("user_request") or "")
    prior = rec.get("prior") or []
    texts = [str(b.get("text") or "") for b in prior]
    tool_texts = [str(b.get("text") or "") for b in prior if b.get("role") == "tool"]
    current = rec.get("current") or {}
    cur_json = json.dumps(current.get("arguments"), ensure_ascii=False, sort_keys=True)
    return tool_texts, user, cur_json, texts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    old_views = load_old_views()
    new_views = load_new_views()
    caches = best_cache_rows()
    excluded = set(json.loads(EXCLUDE.read_text())) if EXCLUDE.is_file() else set()
    frozen = json.loads(FROZEN.read_text()) if FROZEN.is_file() else {}
    frozen_shas = {Path(k).name: v.get("sha256") if isinstance(v, dict) else v
                   for k, v in (frozen.get("frozen_teacher_caches") or {}).items()}
    cache_shas = {p.name: sha(p) for p in CACHES if p.is_file()}

    stats: Counter = Counter()
    reasons: Counter = Counter()
    labels: dict[int, dict[str, int]] = {}
    per_arg: dict[int, list[dict[str, Any]]] = {}
    missing_new = []
    for row, passes in sorted(caches.items()):
        old = old_views.get(row)
        new = new_views.get(row)
        if old is None:
            stats["old_view_missing"] += 1
            reasons["old_view_missing"] += 1
            continue
        if new is None:
            stats["new_view_missing"] += 1
            reasons["new_view_missing"] += 1
            missing_new.append(row)
            continue
        # binding: did the cached prompt describe this row's actual call?
        old_cur = old.get("current") or {}
        new_cur = new.get("current") or {}
        binding_ok = (old_cur.get("tool") == new_cur.get("tool")
                      and old_cur.get("arguments") == new_cur.get("arguments"))
        material_ok = (str(old.get("user_request") or "") == str(new.get("user_request") or ""))
        if not material_ok:
            reasons["user_request_changed"] += 1
        old_prior = [(b.get("role"), b.get("text")) for b in (old.get("prior") or [])]
        new_prior = [(b.get("role"), b.get("text")) for b in (new.get("prior") or [])]
        if old_prior != new_prior:
            material_ok = False
            reasons["prior_material_changed"] += 1
        if not binding_ok:
            reasons["cached_prompt_bound_previous_call"] += 1
        if not (binding_ok and material_ok):
            labels[row] = {"action": -1, "rel": -1, "align": -1, "arg": -1}
            stats["masked_rebind"] += 1
            continue
        if row in excluded:
            labels[row] = {"action": -1, "rel": -1, "align": -1, "arg": -1}
            stats["masked_forged_role"] += 1
            continue
        a, b = passes.get("A"), passes.get("B")
        if not (a and b and a["parse_ok"] and b["parse_ok"]):
            labels[row] = {"action": -1, "rel": -1, "align": -1, "arg": -1}
            stats["masked_single_or_unparsed"] += 1
            continue
        stats["both_pass"] += 1
        pa, pb = a["parsed"], b["parsed"]
        tool_texts, user, cur_json, all_texts = material(new)
        user_hay = [user] + all_texts + [cur_json]
        ext_hay = tool_texts + [cur_json]
        arg_hay = [user] + all_texts + [cur_json]
        rec: dict[str, int] = {}
        for key, field, mapping in FIELDS:
            va, vb = pa.get(field), pb.get(field)
            valid = True
            if key == "action":
                valid = all(_literal(p.get("user_authorization_quote"), [user]) for p in (pa, pb))
            elif key == "align":
                valid = all(_literal(p.get("external_requirement_quote"), ext_hay) for p in (pa, pb))
                # a quote attributed to user authorization must also be literal
                valid = valid and all(_literal(p.get("user_authorization_quote"), [user])
                                      for p in (pa, pb))
            else:
                valid = all(_literal(p.get("user_authorization_quote"), [user])
                            for p in (pa, pb))
            if not valid:
                stats[f"masked_quote_{key}"] += 1
                rec[key] = -1
            elif va == vb and va in mapping:
                rec[key] = mapping[va]
                stats[f"trainable_{key}"] += 1
            else:
                stats[f"masked_disagree_{key}"] += 1
                rec[key] = -1
        aa = {x["path"]: x for x in pa.get("arguments") or [] if x.get("path")}
        ba = {x["path"]: x for x in pb.get("arguments") or [] if x.get("path")}
        valid_paths = {str(x.get("path")) for x in (new.get("value_sources") or [])}
        entries: list[dict[str, Any]] = []
        votes: list[int] = []
        invalid = invalid_quote = 0
        for path in sorted(set(aa) & set(ba)):
            stats["arg_shared_paths"] += 1
            if path not in valid_paths:
                invalid += 1
                stats["arg_path_invalid"] += 1
                continue
            av, bv = aa[path].get("authorization"), ba[path].get("authorization")
            if av != bv or av not in ("authorized", "unauthorized"):
                stats["arg_masked_disagree"] += 1
                continue
            if not all(_literal(x.get("source_quote"), arg_hay) for x in (aa[path], ba[path])):
                invalid_quote += 1
                stats["arg_quote_invalid"] += 1
                continue
            vote = 1 if av == "unauthorized" else 0
            votes.append(vote)
            entry: dict[str, Any] = {"path": path, "authorization": vote}
            for field in ARG_FIELDS[1:]:
                if aa[path].get(field) == ba[path].get(field):
                    entry[field] = aa[path].get(field)
                else:
                    entry[field] = None
            entries.append(entry)
        if entries:
            per_arg[row] = entries
        if invalid or invalid_quote:
            rec["arg"] = -1
            stats["arg_event_masked_invalid"] += 1
        elif votes:
            rec["arg"] = int(any(v == 1 for v in votes))
            stats["arg_event_usable"] += 1
            stats["arg_event_positive"] += int(rec["arg"] == 1)
        else:
            rec["arg"] = -1
        labels[row] = rec

    for key in ("action", "rel", "align", "arg"):
        stats[f"trainable_rows_{key}"] = sum(1 for r in labels.values() if r[key] >= 0)
    out_labels = OUT / "aux_labels_corrected_r3.json"
    out_per_arg = OUT / "aux_per_argument_corrected_r3.json"
    out_labels.write_text(json.dumps({str(k): v for k, v in sorted(labels.items())},
                                     indent=0) + "\n")
    out_per_arg.write_text(json.dumps({str(k): v for k, v in sorted(per_arg.items())},
                                      indent=0) + "\n")
    manifest = {
        "policy": "PROTOCOL_CORRECTED_R3 C2; strict both-pass + rebind + literal quote/path",
        "cached_teacher_rows": len(caches),
        "teacher_cache_sha256": cache_shas,
        "frozen_manifest_sha256": frozen_shas,
        "teacher_cache_sha_match": all(frozen_shas.get(k) == v for k, v in cache_shas.items()),
        "new_view_sha256": sha(OUT.parent / "views_train.jsonl"),
        "stats": dict(stats),
        "reasons": dict(reasons),
        "new_view_missing_rows": missing_new[:20],
        "labels_sha256": sha(out_labels),
        "per_arg_sha256": sha(out_per_arg),
    }
    (OUT / "aux_manifest_corrected_r3.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
