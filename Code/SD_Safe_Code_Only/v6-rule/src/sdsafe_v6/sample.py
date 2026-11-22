"""Group-stratified event sampling for teacher labelling (TRAIN only, deterministic)."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from typing import Any

# Pre-registered strata.  ``row_label`` is the accepted *event-level* atomic checker flag;
# ``episode_label`` is the episode-level ``strict_final`` the benchmark scores.
STRATA = {
    "pos_call": lambda r: bool(r["label_value"]) and r["event_type"] == "tool_call",
    "pos_text": lambda r: bool(r["label_value"]) and r["event_type"] == "content_segment",
    "pre_call": lambda r: (not bool(r["label_value"])) and bool(r["strict_final"]) and r["event_type"] == "tool_call",
    "neg_call": lambda r: (not bool(r["label_value"])) and (not bool(r["strict_final"])) and r["event_type"] == "tool_call",
    "neg_text": lambda r: (not bool(r["label_value"])) and (not bool(r["strict_final"])) and r["event_type"] == "content_segment",
}


def _stable_rank(row: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(f"{salt}|{row['row_index']}|{row['event_id']}".encode()).hexdigest()


def sample_events(
    rows: dict[int, dict[str, Any]],
    *,
    quotas: dict[str, int],
    salt: str = "v6param-pilot",
    max_per_group: int = 2,
    max_per_episode: int = 1,
    exclude_row_index: set[int] | None = None,
) -> dict[str, Any]:
    """Pick events per stratum with group/episode diversity.

    ``max_per_group`` keeps a handful of groups from dominating the teacher budget;
    ``max_per_episode`` avoids labelling many near-identical segments of one episode.
    Selection never looks at a model score, so it is not "hardest examples only".
    """
    exclude_row_index = exclude_row_index or set()
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.values():
        if int(row["row_index"]) in exclude_row_index:
            continue
        for name, pred in STRATA.items():
            if pred(row):
                by_stratum[name].append(row)
                break
    picked: dict[str, list[dict[str, Any]]] = {}
    for name, quota in quotas.items():
        pool = sorted(by_stratum.get(name, []), key=lambda r: _stable_rank(r, salt))
        group_count: dict[str, int] = defaultdict(int)
        ep_count: dict[str, int] = defaultdict(int)
        chosen = []
        for row in pool:
            if len(chosen) >= quota:
                break
            g, e = row["group_id"], row["episode_id"]
            if group_count[g] >= max_per_group or ep_count[e] >= max_per_episode:
                continue
            group_count[g] += 1
            ep_count[e] += 1
            chosen.append(row)
        picked[name] = chosen
    return {
        "salt": salt,
        "quotas": quotas,
        "available": {k: len(v) for k, v in by_stratum.items()},
        "picked": {k: [int(r["row_index"]) for r in v] for k, v in picked.items()},
        "rows": {k: v for k, v in picked.items()},
        "counts": {k: len(v) for k, v in picked.items()},
        "total": sum(len(v) for v in picked.values()),
    }
