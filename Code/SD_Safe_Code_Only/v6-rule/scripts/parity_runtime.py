"""Full common-DEV parity: r1 reference views/features vs the incremental runtime_fast path.

No labels are read for scoring decisions.  The stored r1 view records are the reference.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdsafe_v6 import data, features, runtime_fast, view  # noqa: E402


def load_reference() -> dict[str, list[dict]]:
    views = []
    for name in ("dev_common.jsonl", "dev_common_fixup.jsonl"):
        views += [json.loads(l) for l in (runtime_fast.ART / "views" / name).read_text().splitlines()]
    by_ep = collections.defaultdict(list)
    for v in views:
        by_ep[v["episode_id"]].append(v)
    for vs in by_ep.values():
        vs.sort(key=lambda r: (int(r["round_id"]), int(r["token_end"]), int(r["row_index"])))
    return by_ep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0)
    ap.add_argument("--episode", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fast", type=int, default=1)
    ap.add_argument("--with-sha", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(data.MODEL_DIR), trust_remote_code=False)
    ref = load_reference()
    eps = sorted(ref)
    if args.episode:
        eps = [args.episode]
    elif not args.all and args.episodes:
        eps = eps[: args.episodes]
    tc = runtime_fast.TextCache()
    bp = runtime_fast.BlockProcessor(tc)
    bad = collections.Counter()
    examples = []
    n_events = 0
    n_episodes = 0
    for ep in eps:
        vs = ref[ep]
        rec0 = data.load_prepared("dev", ep)
        ids = rec0["ids_flat"]
        offsets = [int(x) for x in rec0["ids_offsets"].tolist()]
        rids = [int(x) for x in rec0["round_id"].tolist()]
        plen = [int(x) for x in rec0["prompt_length"].tolist()]
        seq = [int(x) for x in rec0["seq_len"].tolist()]
        st = runtime_fast.EpisodeStream(episode_id=ep, ids=ids, offsets=offsets, round_ids=rids,
                                        prompt_len=plen, seq_len=seq, tok=tok, tc=tc, bp=bp)
        for v in vs:
            meta = {"row_index": v["row_index"], "event_id": v["event_id"],
                    "scope": v["scope"], "group_id": v["group_id"],
                    "row_label": v["row_label"], "episode_label": v["episode_label"]}
            got = st.event(rec_meta=meta, t_start=v["token_start"], t_end=v["token_end"],
                           event_type=v["event_type"], round_id=v["round_id"],
                           token_start=v["token_start"], token_end=v["token_end"],
                           with_prefix_sha=bool(args.with_sha), fast=bool(args.fast))
            r = got["rec"]
            n_events += 1
            checks = {
                "current.kind": r["current"]["kind"] == v["current"]["kind"],
                "current.segment": r["current"]["segment"] == v["current"]["segment"],
                "current.text": r["current"]["text"] == v["current"]["text"],
                "current.visible": r["current"]["visible_so_far"] == v["current"]["visible_so_far"],
                "current.tool": r["current"]["tool"] == v["current"]["tool"],
                "current.args": r["current"]["arguments"] == v["current"]["arguments"],
                "user_request": r["user_request"] == v["user_request"],
                "prior": r["prior"] == v["prior"],
                "relevant": r["relevant_tool_blocks"] == v["relevant_tool_blocks"],
                "value_sources": r["value_sources"] == v["value_sources"],
                "counts": r["counts"] == v["counts"],
                "prefix_chars": r["prefix_chars"] == v["prefix_chars"],
                "has_reasoning": r["has_reasoning_before"] == v["has_reasoning_before"],
                "event_type": r["event_type"] == v["event_type"],
                "round": r["round_id"] == v["round_id"],
            }
            if args.with_sha:
                checks["prefix_sha"] = r["prefix_sha256"] == v["prefix_sha256"]
                checks["current_sha"] = r["current_sha256"] == v["current_sha256"]
            for k, ok in checks.items():
                if not ok:
                    bad[k] += 1
                    if len(examples) < 12:
                        examples.append((ep, v["row_index"], k))
            # feature parity
            ref_feats = features.extract(v)
            fb = got["feats"]["bags"]
            rb = ref_feats["bags"]
            if list(fb["cur"]) != list(rb["cur"]):
                bad["bags.cur"] += 1
            if list(fb["vis"]) != list(rb["vis"]):
                bad["bags.vis"] += 1
            if list(fb["user"]) != list(rb["user"]):
                bad["bags.user"] += 1
            if [list(x) for x in fb["src"]] != [list(x) for x in rb["src"]]:
                bad["bags.src"] += 1
            if list(got["feats"]["dense"]) != list(ref_feats["dense"]):
                bad["dense"] += 1
                if len(examples) < 12:
                    examples.append((ep, v["row_index"], "dense"))
        n_episodes += 1
        if (not args.all) and args.episodes and n_episodes >= args.episodes:
            break
    print(json.dumps({"episodes": n_episodes, "events": n_events, "bad": dict(bad),
                      "examples": examples}, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"episodes": n_episodes, "events": n_events,
                                              "bad": dict(bad), "examples": examples}, indent=1) + "\n")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
