"""corrected-r3 validation: call localisation, teacher mask, empty policy, causality, tau."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v6 import data, features, metrics, runtime_fast, view  # noqa: E402

ART = runtime_fast.ART
OUT = ART / "corrected-r3"
CHECKS: list[dict] = []


def ck(name: str, ok: bool, detail) -> None:
    CHECKS.append({"check": name, "ok": bool(ok), "detail": detail})


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    import argparse as ap
    p = ap.ArgumentParser()
    p.add_argument("--out", default=str(OUT / "validation_corrected_r3.json"))
    args = p.parse_args()
    summary = json.loads((OUT / "EXPORT_SUMMARY.json").read_text())
    tr = summary["splits"]["train"]
    dv = summary["splits"]["dev"]

    ck("dev_records_25003", dv["records"] == 25003, dv["records"])
    ck("train_records_114799", tr["records"] == 114799, tr["records"])
    ck("no_export_failures", dv["failures"] == 0 and tr["failures"] == 0,
       [dv["failures"], tr["failures"]])
    ck("dev_tool_calls_all_ordinal_segment",
       dv["stats"].get("call_match:ordinal+segment") == 7813
       and dv["stats"].get("call_match:segment_only", 0) == 0,
       dv["stats"].get("call_match:ordinal+segment"))
    ck("dev_no_binding_conflict", dv["stats"].get("call_binding_conflict", 0) == 0,
       dv["stats"].get("call_binding_conflict", 0))
    ck("dev_changed_current_call_830", dv["stats"].get("changed_current_call") == 830,
       dv["stats"].get("changed_current_call"))
    ck("dev_empty_59_all_scored", dv["stats"].get("empty_segment") == 59
       and dv["stats"].get("no_new_emission", 0) == 0
       and dv["stats"].get("unknown_adapter", 0) == 0,
       {k: dv["stats"].get(k, 0) for k in ("empty_segment", "no_new_emission",
                                          "unknown_adapter")})
    ck("dev_cache_all_rows", json.loads((OUT / "cache_dev.manifest.json").read_text())["scoreable_rows"] == 25003,
       json.loads((OUT / "cache_dev.manifest.json").read_text())["scoreable_rows"])
    ck("train_segment_only_6_block_fallback_8",
       tr["stats"].get("call_match:segment_only") == 6 and tr["stats"].get("block_fallback") == 8,
       {k: tr["stats"].get(k, 0) for k in ("call_match:segment_only", "block_fallback")})
    ck("train_changed_current_call_3401", tr["stats"].get("changed_current_call") == 3401,
       tr["stats"].get("changed_current_call"))

    # universe / episode preservation
    views = [json.loads(l) for l in (OUT / "views_dev.jsonl").read_text().splitlines() if l.strip()]
    eps = {v["episode_id"] for v in views}
    groups = {v["group_id"] for v in views}
    ck("dev_2028_episodes_166_groups", len(eps) == 2028 and len(groups) == 166,
       [len(eps), len(groups)])

    # teacher rebind
    aux = json.loads((OUT / "aux/aux_manifest_corrected_r3.json").read_text())
    ck("teacher_cache_sha_unchanged", aux["teacher_cache_sha_match"] is True,
       aux["teacher_cache_sha256"])
    ck("teacher_both_pass_present", aux["stats"]["both_pass"] == 3793, aux["stats"]["both_pass"])
    ck("teacher_masked_rebind_1121", aux["stats"]["masked_rebind"] == 1121,
       aux["stats"]["masked_rebind"])
    old = {}
    for name in ("aux.jsonl", "aux2.jsonl", "pilot400.jsonl"):
        for line in (ART / "views" / name).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                old[int(r["row_index"])] = r
    train_views = [json.loads(l) for l in (OUT / "views_train.jsonl").read_text().splitlines() if l.strip()]
    new = {int(v["row_index"]): v for v in train_views}
    labels = {int(k): v for k, v in json.loads((OUT / "aux/aux_labels_corrected_r3.json").read_text()).items()}
    masked_example = clean_example = None
    for r, o in old.items():
        n = new.get(r)
        if n is None:
            continue
        if (o["current"].get("tool") != n["current"].get("tool")
                or o["current"].get("arguments") != n["current"].get("arguments")):
            if labels.get(r, {}).get("action") == -1:
                masked_example = r
        elif labels.get(r, {}).get("action", -1) >= 0:
            clean_example = r
        if masked_example and clean_example:
            break
    ck("teacher_rebind_example_masked", masked_example is not None, masked_example)
    ck("teacher_clean_example_retained", clean_example is not None, clean_example)
    if masked_example:
        o = old[masked_example]
        n = new[masked_example]
        ck("masked_example_really_bound_previous_call",
           o["current"].get("tool") != n["current"].get("tool")
           or o["current"].get("arguments") != n["current"].get("arguments"),
           {"row": masked_example, "old_tool": o["current"].get("tool"),
            "new_tool": n["current"].get("tool"), "label": labels.get(masked_example)})

    # F2: the 20 previously-missing rows are not student/base argmax
    d = np.load(OUT / "dev_scores_S2.npz", allow_pickle=True)
    rs = np.asarray(d["row_scores"], np.float32)
    ri = np.asarray(d["row_index"])
    rep = np.asarray([str(x) for x in d["episode_id"].tolist()])
    epi = {e: i for i, e in enumerate(sorted(set(rep.tolist())))}
    epmax = np.full((3, len(epi)), -np.inf, np.float32)
    np.maximum.at(epmax, (slice(None), np.asarray([epi[e] for e in rep])), rs)
    idx = {int(r): i for i, r in enumerate(ri)}
    twenty = {162426, 162448, 162534, 162829, 162900, 162975, 162998, 163079, 163989,
              164093, 164294, 164682, 164698, 164848, 164880, 165156, 165557, 165692,
              166072, 166232}
    argmax20 = 0
    for r in twenty:
        i = idx[r]
        j = epi[rep[i]]
        argmax20 += int(any(rs[s, i] == epmax[s, j] for s in range(3)))
    ck("f2_twenty_rows_not_student_argmax", argmax20 == 0, argmax20)
    empty_rows = [v["row_index"] for v in views if v["empty_segment"]]
    argmax_empty = 0
    for r in empty_rows:
        i = idx[r]
        j = epi[rep[i]]
        argmax_empty += int(any(rs[s, i] == epmax[s, j] for s in range(3)))
    ck("empty59_rows_with_any_seed_argmax", argmax_empty == 2, argmax_empty)

    # causality: suffix replacement at fixed endpoint
    random.seed(20261114)
    tc = runtime_fast.TextCache()
    bp = runtime_fast.BlockProcessor(tc)
    sample = [v for v in views if v["event_type"] == "tool_call" and (v["current"]["call_index"] or 0) >= 1][:10]
    sample += [v for v in views if v["event_type"] == "text"][:10]
    sample += [v for v in views if v["empty_segment"]][:5]
    by_ep: dict[str, list[dict]] = {}
    for v in sample:
        by_ep.setdefault(v["episode_id"], []).append(v)
    bad = 0
    checked = 0
    for ep, vs in by_ep.items():
        rec0 = data.load_prepared("dev", ep)
        ids = np.asarray(rec0["ids_flat"])
        rids = [int(x) for x in rec0["round_id"].tolist()]
        offs = [int(x) for x in rec0["ids_offsets"].tolist()]
        seq = [int(x) for x in rec0["seq_len"].tolist()]
        pl = [int(x) for x in rec0["prompt_length"].tolist()]
        pos = {int(v): i for i, v in enumerate(rec0["row_index"].tolist())}
        for v in vs:
            i = pos[int(v["row_index"])]
            row = {"row_index": int(v["row_index"]), "event_id": v["event_id"],
                   "event_type": v["event_type"], "token_start": int(v["token_start"]),
                   "token_end": int(v["token_end"]), "row_round_id": int(v["round_id"]),
                   "call_index": (None if v["current"].get("call_index") is None
                                  else int(v["current"]["call_index"]))}
            ri = rids.index(int(v["round_id"]))
            base_ids = ids.copy()
            mut = ids.copy()
            t0, t1 = offs[ri] + int(v["token_end"]), offs[ri] + seq[ri]
            if t1 > t0:
                rng = np.random.default_rng(1234 + i)
                mut[t0:t1] = rng.integers(0, 150000, size=t1 - t0, dtype=np.int64)
            rec_a = view.build_view(episode_id=ep, row=row, round_ids=rids, round_seq_len=seq,
                                    round_prompt_len=pl, offsets=offs, ids=base_ids,
                                    decode=lambda a: runtime_fast.decode_ids(
                                        _tok, a), row_label=True, episode_label=True,
                                    group_id=v["group_id"], scope=v["scope"])
            rec_b = view.build_view(episode_id=ep, row=row, round_ids=rids, round_seq_len=seq,
                                    round_prompt_len=pl, offsets=offs, ids=mut,
                                    decode=lambda a: runtime_fast.decode_ids(
                                        _tok, a), row_label=True, episode_label=True,
                                    group_id=v["group_id"], scope=v["scope"])
            fa, fb = features.extract(rec_a), features.extract(rec_b)
            same = (rec_a["current"] == rec_b["current"]
                    and rec_a["value_sources"] == rec_b["value_sources"]
                    and rec_a["prior"] == rec_b["prior"]
                    and rec_a["relevant_tool_blocks"] == rec_b["relevant_tool_blocks"]
                    and fa["bags"] == fb["bags"] and fa["dense"] == fb["dense"])
            checked += 1
            bad += int(not same)
    ck("suffix_replacement_invariance", bad == 0, {"checked": checked, "bad": bad})

    # exact tau replay
    for arm in ("S0", "S2"):
        fj = json.loads((OUT / f"fusion_frozen_corrected_r3_{arm}.json").read_text())
        tau = float(fj["threshold"])
        ck(f"fusion_tau_exact_hex_{arm}", float(tau).hex() == fj["threshold_hex"],
           {"tau": tau, "hex": fj["threshold_hex"]})
        rz = np.load(OUT / f"replay_{arm}.npz", allow_pickle=True)
        fused = np.asarray(rz["fused"], np.float64)
        labels = np.asarray(rz["labels"], np.int64)
        ck(f"fusion_tau_is_own_5pct_{arm}",
           metrics.threshold_at_fpr(fused, labels, 0.05) == tau,
           metrics.threshold_at_fpr(fused, labels, 0.05))
        conf = metrics.confusion(fused, labels, tau)
        ck(f"fusion_replay_fn_fp_{arm}", (conf["fn"], conf["fp"]) == (fj["fn"], fj["fp"]),
           [conf["fn"], conf["fp"], fj["fn"], fj["fp"]])
        ck(f"fusion_sha_{arm}",
           hashlib.sha256(np.ascontiguousarray(fused).tobytes()).hexdigest() == fj["fused_sha256"],
           fj["fused_sha256"][:16])

    # incremental decode chunking + cache eviction
    chunk_ok = 0
    chunk_n = 0
    for ep in sorted(by_ep)[:6]:
        rec0 = data.load_prepared("dev", ep)
        ids = rec0["ids_flat"]
        rids = [int(x) for x in rec0["round_id"].tolist()]
        offs = [int(x) for x in rec0["ids_offsets"].tolist()]
        pl = [int(x) for x in rec0["prompt_length"].tolist()]
        seq = [int(x) for x in rec0["seq_len"].tolist()]
        for ri in range(len(rids)):
            total = int(seq[ri])
            cuts = sorted(random.sample(range(1, total), min(7, total - 1))) if total > 2 else []
            parts = []
            prev = 0
            for c in cuts + [total]:
                parts.append(runtime_fast.decode_ids(_tok, ids[offs[ri] + prev:offs[ri] + c]))
                prev = c
            full = runtime_fast.decode_ids(_tok, ids[offs[ri]:offs[ri] + total])
            chunk_n += 1
            chunk_ok += int("".join(parts) == full)
    ck("incremental_decode_chunk_invariance", chunk_ok == chunk_n, [chunk_ok, chunk_n])

    # rerun a small episode after clearing caches and compare to reference views
    tc.clear()
    bp.memo.clear()
    ep = sorted(by_ep)[0]
    rec0 = data.load_prepared("dev", ep)
    st = runtime_fast.EpisodeStream(
        episode_id=ep, ids=rec0["ids_flat"],
        offsets=[int(x) for x in rec0["ids_offsets"].tolist()],
        round_ids=[int(x) for x in rec0["round_id"].tolist()],
        prompt_len=[int(x) for x in rec0["prompt_length"].tolist()],
        seq_len=[int(x) for x in rec0["seq_len"].tolist()], tok=_tok, tc=tc, bp=bp)
    bad_evict = 0
    for v in [x for x in views if x["episode_id"] == ep]:
        meta = {"row_index": v["row_index"], "event_id": v["event_id"], "scope": v["scope"],
                "group_id": v["group_id"], "row_label": v["row_label"],
                "episode_label": v["episode_label"]}
        got = st.event(rec_meta=meta, t_start=v["token_start"], t_end=v["token_end"],
                       event_type=v["event_type"], round_id=v["round_id"],
                       token_start=v["token_start"], token_end=v["token_end"],
                       fast=True, call_index=v["current"].get("call_index"))
        feats = features.extract(v)
        bad_evict += int(got["rec"]["current"] != v["current"]
                         or got["feats"]["dense"] != feats["dense"]
                         or [list(x) for x in got["feats"]["bags"]["src"]]
                         != [list(x) for x in feats["bags"]["src"]])
    ck("cache_eviction_recompute_parity", bad_evict == 0, bad_evict)

    # parity files from the A and typed-B runs
    for name in ("parity_views_corrected_r3.json", "parity_views_typed_r3.json"):
        p = OUT / name
        data_p = json.loads(p.read_text()) if p.is_file() else {"bad": {"missing": 1}}
        ck(f"parity_{name}", data_p.get("bad") == {} and data_p.get("events") == 25003,
           data_p.get("bad"))

    out = {"checks": CHECKS, "passed": sum(1 for c in CHECKS if c["ok"]),
           "total": len(CHECKS), "all_ok": all(c["ok"] for c in CHECKS)}
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if out["all_ok"] else 1


_tok = None

if __name__ == "__main__":
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)
    raise SystemExit(main())
