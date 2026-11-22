"""Prepared token-id jobs: the CPU node reads the frozen sources and exports exact accepted
token ids; the GPU node only forwards them.  This keeps the *only* serializer on the CPU
(the node that owns the read-only source) and avoids copying the raw traces/artifacts to
the GPU.

Format (one ``.npz`` per episode, plus ``index.json``/``manifest.json``):

    episode_id str, scope str, group_id str, label bool
    round_id int32 [r], seq_len int32 [r], prompt_length int32 [r], round_sha object [r]
    ids_flat int32 [sum(seq_len)], ids_offsets int32 [r+1]
    row_index int64 [n], cache_row int64 [n], message_index int32 [n],
    call_index int32 [n], segment_index int32 [n], token_start int32 [n],
    token_end int32 [n], row_round_id int32 [n], positions int32 [n],
    row_sha object [n], event_id object [n]

The GPU reader re-proves every round against the accepted row digests before capture, so a
corrupted/truncated prepared file cannot silently enter a shard.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

IM_END_ID = 151645


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def job_key(episode_id: str) -> str:
    return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:24]


def write_episode(pairs: list[tuple[Any, list[int]]], outdir: Path) -> Path:
    """Export all rounds of one episode (CPU side).  ``pairs`` are (RoundTask, ids)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = sorted(pairs, key=lambda p: int(p[0].round_id))
    first = pairs[0][0]
    flat: list[int] = []
    offsets = [0]
    round_ids, seq_lens, prompt_lengths, round_shas = [], [], [], []
    col = {k: [] for k in ("row_index", "cache_row", "message_index", "call_index",
                           "segment_index", "token_start", "token_end", "row_round_id",
                           "positions", "row_sha", "event_id")}
    for task, ids in pairs:
        flat.extend(int(x) for x in ids)
        offsets.append(len(flat))
        round_ids.append(int(task.round_id))
        seq_lens.append(len(ids))
        prompt_lengths.append(int(task.prompt_length))
        round_shas.append(str(task.sequence_sha256))
        for r in task.events:
            col["row_index"].append(int(r["row_index"]))
            col["cache_row"].append(int(r["cache_row"]))
            col["message_index"].append(int(r["message_index"]))
            col["call_index"].append(-1 if r.get("call_index") is None else int(r["call_index"]))
            col["segment_index"].append(-1 if r.get("segment_index") is None else int(r["segment_index"]))
            col["token_start"].append(int(r["token_start"]))
            col["token_end"].append(int(r["token_end"]))
            col["row_round_id"].append(int(task.round_id))
            col["positions"].append(int(r["feature_token_index"]))
            col["row_sha"].append(str(r["sequence_sha256"]))
            col["event_id"].append(str(r["event_id"]))
    payload = {
        "episode_id": str(first.episode_id),
        "scope": str(first.scope),
        "group_id": str(first.group_id),
        "label": np.asarray(bool(first.label_value)),
        "round_id": np.asarray(round_ids, dtype=np.int32),
        "seq_len": np.asarray(seq_lens, dtype=np.int32),
        "prompt_length": np.asarray(prompt_lengths, dtype=np.int32),
        "round_sha": np.asarray(round_shas, dtype=object),
        "ids_flat": np.asarray(flat, dtype=np.int32),
        "ids_offsets": np.asarray(offsets, dtype=np.int32),
        "row_index": np.asarray(col["row_index"], dtype=np.int64),
        "cache_row": np.asarray(col["cache_row"], dtype=np.int64),
        "message_index": np.asarray(col["message_index"], dtype=np.int32),
        "call_index": np.asarray(col["call_index"], dtype=np.int32),
        "segment_index": np.asarray(col["segment_index"], dtype=np.int32),
        "token_start": np.asarray(col["token_start"], dtype=np.int32),
        "token_end": np.asarray(col["token_end"], dtype=np.int32),
        "row_round_id": np.asarray(col["row_round_id"], dtype=np.int32),
        "positions": np.asarray(col["positions"], dtype=np.int32),
        "row_sha": np.asarray(col["row_sha"], dtype=object),
        "event_id": np.asarray(col["event_id"], dtype=object),
    }
    path = outdir / f"{job_key(str(first.episode_id))}.npz"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    tmp.replace(path)
    return path


class PreparedRound:
    def __init__(self, episode_id: str, scope: str, label: bool, group_id: str,
                 round_id: int, sequence_sha256: str, positions: list[int],
                 events: list[dict[str, Any]]):
        self.episode_id = episode_id
        self.scope = scope
        self.label_value = label
        self.group_id = group_id
        self.round_id = round_id
        self.sequence_sha256 = sequence_sha256
        self.positions = positions
        self.events = events


class PreparedEpisode:
    def __init__(self, episode_id: str, scope: str, group_id: str, label: bool,
                 rounds: list[tuple[PreparedRound, list[int]]]):
        self.episode_id = episode_id
        self.scope = scope
        self.group_id = group_id
        self.label = label
        self.rounds = rounds


def load_episode(path: Path) -> PreparedEpisode:
    z = np.load(path, allow_pickle=True)
    episode_id = str(z["episode_id"])
    scope = str(z["scope"])
    group_id = str(z["group_id"])
    label = bool(z["label"])
    round_ids = [int(x) for x in np.asarray(z["round_id"]).tolist()]
    seq_lens = [int(x) for x in np.asarray(z["seq_len"]).tolist()]
    offsets = [int(x) for x in np.asarray(z["ids_offsets"]).tolist()]
    flat = np.asarray(z["ids_flat"], dtype=np.int64)
    round_sha = [str(x) for x in np.asarray(z["round_sha"]).tolist()]
    row_round = [int(x) for x in np.asarray(z["row_round_id"]).tolist()]
    n = len(row_round)
    row_arrays = {
        "row_index": np.asarray(z["row_index"], dtype=np.int64),
        "cache_row": np.asarray(z["cache_row"], dtype=np.int64),
        "message_index": np.asarray(z["message_index"], dtype=np.int32),
        "call_index": np.asarray(z["call_index"], dtype=np.int32),
        "segment_index": np.asarray(z["segment_index"], dtype=np.int32),
        "token_start": np.asarray(z["token_start"], dtype=np.int32),
        "token_end": np.asarray(z["token_end"], dtype=np.int32),
        "positions": np.asarray(z["positions"], dtype=np.int32),
        "row_sha": [str(x) for x in np.asarray(z["row_sha"]).tolist()],
        "event_id": [str(x) for x in np.asarray(z["event_id"]).tolist()],
    }
    by_round: dict[int, list[int]] = {}
    for i in range(n):
        by_round.setdefault(row_round[i], []).append(i)
    rounds: list[tuple[PreparedRound, list[int]]] = []
    for j, rid in enumerate(round_ids):
        ids = [int(x) for x in flat[offsets[j]:offsets[j + 1]].tolist()]
        if len(ids) != seq_lens[j]:
            raise ValueError(f"{episode_id} r{rid}: prepared seq_len mismatch")
        if ids[-1] != IM_END_ID:
            raise ValueError(f"{episode_id} r{rid}: prepared ids do not end at <|im_end|>")
        idxs = sorted(by_round.get(rid, []))
        events = []
        positions = []
        for i in idxs:
            pos = int(row_arrays["positions"][i])
            if pos != int(row_arrays["token_end"][i]):
                raise ValueError(f"{episode_id} r{rid}: position != token_end")
            if not (0 <= pos < len(ids)):
                raise ValueError(f"{episode_id} r{rid}: endpoint outside sequence")
            # identity re-proof against the accepted per-row digest (the round's full
            # returned ids; all rows of a round share this digest by construction)
            want = row_arrays["row_sha"][i]
            got = canonical_sha256(ids)
            if got != want:
                raise ValueError(f"{episode_id} r{rid}: prepared ids digest mismatch at row {i}")
            positions.append(pos)
            events.append({
                "row_index": int(row_arrays["row_index"][i]),
                "cache_row": int(row_arrays["cache_row"][i]),
                "message_index": int(row_arrays["message_index"][i]),
                "call_index": int(row_arrays["call_index"][i]) if int(row_arrays["call_index"][i]) >= 0 else None,
                "segment_index": int(row_arrays["segment_index"][i]) if int(row_arrays["segment_index"][i]) >= 0 else None,
                "token_start": int(row_arrays["token_start"][i]),
                "token_end": int(row_arrays["token_end"][i]),
                "feature_token_index": pos,
                "sequence_sha256": want,
                "event_id": row_arrays["event_id"][i],
            })
        if positions != sorted(set(positions)):
            raise ValueError(f"{episode_id} r{rid}: CT endpoints not unique/increasing")
        rounds.append((PreparedRound(episode_id, scope, label, group_id, rid,
                                     round_sha[j], positions, events), ids))
    rounds.sort(key=lambda pair: pair[0].round_id)
    return PreparedEpisode(episode_id, scope, group_id, label, rounds)


def iter_prepared_episodes(prepared_dir: Path, episode_ids: Sequence[str]) -> Iterator[PreparedEpisode]:
    d = Path(prepared_dir)
    index = json.loads((d / "index.json").read_text())
    for e in episode_ids:
        name = index.get(e)
        if name is None:
            raise ValueError(f"prepared job missing for {e}")
        yield load_episode(d / name)
