"""Exact binding between accepted sequence rows and the archived/render token ids.

The accepted `sequence_v1` row files already froze, per event:

* ``sequence_sha256`` / ``round_sequence_sha256`` -- the round's token-id digest,
* ``sequence_length`` / ``prompt_length``,
* ``feature_token_index`` (the endpoint whose final-layer hidden is in
  ``features_bf16.bin``) and ``token_start``/``token_end``,
* ``terminal_token_id`` (must be ``<|im_end|>`` = 151645).

This module rebuilds the same per-round token ids through the *accepted* sequence API
(``sdsafe_fine_confirm.sequences.iter_sequences``, which for ``cot`` copies the archive's
real ``generation.prompt_token_ids + completion_token_ids`` and for the other three
sources re-builds the frozen serializer and verifies it against the frozen event store),
and then refuses to continue unless the rebuilt digest equals the recorded one.  Nothing
is re-encoded approximately and no sequence is truncated.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from . import INDEX_DEV, INDEX_TRAIN, SDSAFE, SEQ_DEV_ROWS, SEQ_TRAIN_ROWS, V2_DEV_ARRAYS
from .common import StudyError

IM_END_ID = 151645


def _load_rows(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


@dataclass
class RoundTask:
    """One independently forwarded round plus every accepted CT endpoint inside it."""

    scope: str
    episode_id: str
    namespaced_episode_id: str
    round_id: int
    sequence_sha256: str
    sequence_length: int
    prompt_length: int
    terminal_token_id: int
    label_value: bool
    group_id: str
    group_hash: str
    scope_tag: str
    positions: list[int] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class RowIndex:
    """Accepted per-event rows, grouped by (scope, episode, round), CT rows only."""

    def __init__(self, split: str) -> None:
        if split not in ("train", "dev"):
            raise StudyError("split must be train or dev")
        self.split = split
        path = SEQ_TRAIN_ROWS if split == "train" else SEQ_DEV_ROWS
        self.path = path
        self.by_round: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        self.episode_ids: list[str] = []          # namespaced ids (unique per scope view)
        self.bare_to_ns: dict[tuple[str, str], str] = {}
        self.episode_scope: dict[str, str] = {}
        self.episode_group: dict[str, str] = {}
        self.episode_label: dict[str, bool] = {}
        self.episode_tokens = 0.0
        seen: set[str] = set()
        for row in _load_rows(path):
            if not row["ct_mask"]:
                continue
            key = (row["scope"], row["episode_id"], int(row["round_id"]))
            self.by_round[key].append(row)
            if row["episode_id"] not in seen:
                seen.add(row["episode_id"])
                self.episode_ids.append(row["episode_id"])
                bare = str(row["episode_id"]).split("::", 1)[-1]
                self.bare_to_ns[(row["scope"], bare)] = str(row["episode_id"])
                self.episode_scope[row["episode_id"]] = row["scope"]
                self.episode_group[row["episode_id"]] = row["group_id"]
                self.episode_label[row["episode_id"]] = bool(row["strict_final"])
        for rows in self.by_round.values():
            rows.sort(key=lambda r: (int(r["message_index"]), int(r["feature_token_index"])))
        self.n_ct_rows = sum(len(v) for v in self.by_round.values())
        self.n_rounds = len(self.by_round)

    def episodes(self, universe: Sequence[str] | None = None) -> list[str]:
        if universe is None:
            return list(self.episode_ids)
        want = set(universe)
        return [e for e in self.episode_ids if e in want]


def dev_common_universe() -> list[str]:
    """The accepted common DEV universe (2,028 episodes / 25,003 CT rows)."""
    import numpy as np

    z = np.load(V2_DEV_ARRAYS, allow_pickle=True)
    ids = [str(x) for x in z["episode_id"].tolist()]
    if len(ids) != 2028:
        raise StudyError(f"common DEV universe changed size: {len(ids)}")
    if int(z["row_index"].shape[0]) != 25_003:
        raise StudyError("common DEV CT row count changed")
    return ids


def dev_universe_from_index() -> list[str]:
    """All 2,386 DEV episodes (superset of the common universe; used for coverage QC)."""
    import numpy as np

    z = np.load(INDEX_DEV, allow_pickle=True)
    return [str(x) for x in z["ep_episode_id"].tolist()]


def train_group_of() -> dict[str, str]:
    import numpy as np

    z = np.load(INDEX_TRAIN, allow_pickle=True)
    return {str(e): str(g) for e, g in zip(z["ep_episode_id"].tolist(), z["ep_group_id"].tolist())}


def _install_nothink_tools() -> None:
    """Let the frozen no-think serializer rebuild its template without the AgentDojo deps.

    ``sdsafe_cot.rounds.tools_for_suite`` needs the AgentDojo suite implementation (pydantic
    only) and the ``openai`` TypedDicts.  The tool schema is rebuilt by this study's shim and
    the returned list is verified against the archive's ``request_tools_sha256`` the first
    time it is used, so a wrong schema cannot silently enter a sequence.
    """
    import sys

    if str(SDSAFE / "src") not in sys.path:
        sys.path.insert(0, str(SDSAFE / "src"))
    from functools import lru_cache

    from . import agentdojo_shim

    agentdojo_shim.install()
    import sdsafe_cot.rounds as rounds  # noqa: E402

    @lru_cache(maxsize=None)
    def cached(benchmark_version: str, suite_name: str):
        return agentdojo_shim.tools_for_suite(benchmark_version, suite_name)

    if getattr(rounds, "_v4_tools_patched", False) is not True:
        rounds.tools_for_suite = cached  # type: ignore[assignment]
        rounds._v4_tools_patched = True  # type: ignore[attr-defined]


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SDSAFE / "models/Qwen3-8B", local_files_only=True)


def iter_round_tasks(
    row_index: RowIndex,
    episode_ids: Sequence[str],
    *,
    tokenizer: Any | None = None,
    scope: str | None = None,
) -> Iterator[tuple[RoundTask, list[int]]]:
    """Yield ``(task, input_ids)`` with the token ids already proved against the row digest."""
    import sys

    if str(SDSAFE / "src") not in sys.path:
        sys.path.insert(0, str(SDSAFE / "src"))
    from sdsafe_fine_confirm.sequences import iter_sequences  # noqa: E402

    tok = tokenizer if tokenizer is not None else _tokenizer()
    if "nothink" in {row_index.episode_scope.get(e) for e in episode_ids}:
        _install_nothink_tools()
    wanted = set(episode_ids)
    scopes = sorted({row_index.episode_scope[e] for e in episode_ids if e in row_index.episode_scope})
    if scope is not None:
        scopes = [scope]
    for scope_name in scopes:
        pairs = [
            (row_index.bare_to_ns[(scope_name, str(e).split("::", 1)[-1])], str(e))
            for e in episode_ids
            if row_index.episode_scope.get(e) == scope_name
        ]
        if not pairs:
            continue
        bare_to_ns = {str(ns).split("::", 1)[-1]: ns for ns, _ in pairs}
        for episode in iter_sequences(
            SDSAFE, scope_name, episode_ids=sorted(bare_to_ns), tokenizer=tok
        ):
            eid = bare_to_ns.get(episode.episode.episode_id)
            if eid is None:
                eid = str(episode.episode.namespaced_episode_id)
            if eid not in wanted:
                continue
            for round_id in sorted(episode.rounds):
                sequence = episode.rounds[round_id]
                key = (scope_name, eid, int(round_id))
                rows = row_index.by_round.get(key)
                if not rows:
                    continue
                ids_list = list(sequence.input_ids)
                recorded = {str(r["sequence_sha256"]) for r in rows}
                if recorded != {sequence.sequence_sha256}:
                    raise StudyError(
                        f"{eid} r{round_id}: rebuilt sequence digest {sequence.sequence_sha256} "
                        f"!= recorded {sorted(recorded)}"
                    )
                if len(ids_list) != int(rows[0]["sequence_length"]):
                    raise StudyError(f"{eid} r{round_id}: sequence length {len(ids_list)} != recorded")
                if ids_list[-1] != IM_END_ID:
                    raise StudyError(f"{eid} r{round_id}: sequence does not end at <|im_end|>")
                positions = []
                for row in rows:
                    pos = int(row["feature_token_index"])
                    if pos != int(row["token_end"]):
                        raise StudyError(f"{row['event_id']}: feature_token_index != token_end")
                    if not (0 <= pos < len(ids_list)):
                        raise StudyError(f"{row['event_id']}: endpoint outside the sequence")
                    if int(row["terminal_token_id"]) != IM_END_ID:
                        raise StudyError(f"{row['event_id']}: unexpected terminal token id")
                    positions.append(pos)
                if positions != sorted(set(positions)):
                    raise StudyError(f"{eid} r{round_id}: CT endpoints are not unique/increasing")
                task = RoundTask(
                    scope=scope_name,
                    episode_id=eid,
                    namespaced_episode_id=str(episode.episode.namespaced_episode_id),
                    round_id=int(round_id),
                    sequence_sha256=sequence.sequence_sha256,
                    sequence_length=len(ids_list),
                    prompt_length=int(sequence.prompt_length),
                    terminal_token_id=int(ids_list[-1]),
                    label_value=bool(rows[0]["strict_final"]),
                    group_id=str(rows[0]["group_id"]),
                    group_hash=str(rows[0]["group_hash"]),
                    scope_tag=str(rows[0]["scope"]),
                    positions=positions,
                    events=rows,
                )
                yield task, ids_list


def canonical_ids_sha(ids: Sequence[int]) -> str:
    """Digest used for our own capture records (comma-joined decimal ids)."""
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ direct cot source
COT_TRACES = SDSAFE / "data/raw/task_injection_cot32k_r1/traces"


def canonical_sha256(value: Any) -> str:
    """Identical to ``sdsafe_features.common.canonical_sha256`` (the accepted digest)."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def token_ids_sha256(ids: Sequence[int]) -> str:
    """The archive's own digest format: SHA256 of the comma-joined decimal ids."""
    return hashlib.sha256(",".join(str(int(v)) for v in ids).encode("utf-8")).hexdigest()


def iter_cot_round_tasks(
    row_index: RowIndex, episode_ids: Sequence[str], limit: int | None = None
) -> Iterator[tuple[RoundTask, list[int]]]:
    """Copy ``prompt_token_ids + completion_token_ids`` straight out of the frozen archive.

    The archived per-round digests are re-derived, the ids are then bound to the accepted
    row through ``canonical_sha256(ids) == row['sequence_sha256']``.  The AgentDojo tool
    digest (``request_tools_sha256``) is *not* recomputable in this environment, so it is
    only recorded, never used as a substitute for the sequence digest.
    """
    sel = [e for e in episode_ids if row_index.episode_scope.get(e) == "cot"]
    for count, ns_id in enumerate(sel):
        if limit is not None and count >= limit:
            return
        bare = ns_id.split("::", 1)[-1]
        path = COT_TRACES / bare / "raw_trace.json"
        if not path.is_file():
            raise StudyError(f"cot raw trace missing for {ns_id}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        messages = raw["trajectory"]["messages"]
        for round_id in sorted({k[2] for k in row_index.by_round if k[0] == "cot" and k[1] == ns_id}):
            rows = row_index.by_round[("cot", ns_id, round_id)]
            index = int(rows[0]["message_index"])
            generation = messages[index].get("generation")
            if not isinstance(generation, dict):
                raise StudyError(f"{ns_id} r{round_id}: assistant message has no generation metadata")
            prompt_ids = [int(v) for v in generation["prompt_token_ids"]]
            completion_ids = [int(v) for v in generation["completion_token_ids"]]
            if generation.get("prompt_token_count") != len(prompt_ids):
                raise StudyError(f"{ns_id} r{round_id}: prompt_token_count mismatch")
            if generation.get("completion_token_count") != len(completion_ids):
                raise StudyError(f"{ns_id} r{round_id}: completion_token_count mismatch")
            if generation.get("prompt_token_ids_sha256") != token_ids_sha256(prompt_ids):
                raise StudyError(f"{ns_id} r{round_id}: archived prompt id digest does not close")
            if generation.get("completion_token_ids_sha256") != token_ids_sha256(completion_ids):
                raise StudyError(f"{ns_id} r{round_id}: archived completion id digest does not close")
            for name, ids in (("prompt", prompt_ids), ("completion", completion_ids)):
                head = generation.get(f"{name}_token_ids_head")
                tail = generation.get(f"{name}_token_ids_tail")
                if head is not None and list(head) != ids[:8]:
                    raise StudyError(f"{ns_id} r{round_id}: {name}_token_ids_head mismatch")
                if tail is not None and list(tail) != ids[-8:]:
                    raise StudyError(f"{ns_id} r{round_id}: {name}_token_ids_tail mismatch")
            if generation.get("finish_reason") == "length":
                raise StudyError(f"{ns_id} r{round_id}: archived round was cut off at max tokens")
            ids_list = prompt_ids + completion_ids
            if ids_list[-1] != IM_END_ID:
                raise StudyError(f"{ns_id} r{round_id}: completion does not end at <|im_end|>")
            recorded = {str(r["sequence_sha256"]) for r in rows}
            if recorded != {canonical_sha256(ids_list)}:
                raise StudyError(
                    f"{ns_id} r{round_id}: archived ids digest {canonical_sha256(ids_list)} != row digest {sorted(recorded)}"
                )
            if len(ids_list) != int(rows[0]["sequence_length"]):
                raise StudyError(f"{ns_id} r{round_id}: sequence length mismatch")
            if len(prompt_ids) != int(rows[0]["prompt_length"]):
                raise StudyError(f"{ns_id} r{round_id}: prompt length mismatch")
            positions = []
            for row in rows:
                pos = int(row["feature_token_index"])
                if pos != int(row["token_end"]):
                    raise StudyError(f"{row['event_id']}: feature_token_index != token_end")
                positions.append(pos)
            if positions != sorted(set(positions)):
                raise StudyError(f"{ns_id} r{round_id}: CT endpoints not unique/increasing")
            yield (
                RoundTask(
                    scope="cot",
                    episode_id=ns_id,
                    namespaced_episode_id=ns_id,
                    round_id=int(round_id),
                    sequence_sha256=str(rows[0]["sequence_sha256"]),
                    sequence_length=len(ids_list),
                    prompt_length=len(prompt_ids),
                    terminal_token_id=int(ids_list[-1]),
                    label_value=bool(rows[0]["strict_final"]),
                    group_id=str(rows[0]["group_id"]),
                    group_hash=str(rows[0]["group_hash"]),
                    scope_tag=str(rows[0]["scope"]),
                    positions=positions,
                    events=rows,
                ),
                ids_list,
            )


# ----------------------------------------------------------------- legacy sources
def legacy_episode_ids(row_index: RowIndex, scope: str) -> list[str]:
    return [e for e in row_index.episode_ids if row_index.episode_scope.get(e) == scope]


def iter_legacy_round_tasks(
    row_index: RowIndex,
    episode_ids: Sequence[str],
    *,
    tokenizer: Any | None = None,
    limit: int | None = None,
) -> Iterator[tuple[RoundTask, list[int]]]:
    """One frozen legacy render per episode; every round proven as a token prefix of it.

    The accepted legacy producer forwarded exactly ``sdsafe_features.trajectory``'s render
    of the whole archived trajectory (``normalize_messages`` + chat template, *no* tool
    schema) and read the endpoint vector of each round from that single forward.  We
    reproduce that render once per episode and accept a round only when
    ``canonical_sha256(ids[:pos + 1])`` equals the row's frozen ``sequence_sha256``.
    """
    import sys

    if str(SDSAFE / "src") not in sys.path:
        sys.path.insert(0, str(SDSAFE / "src"))
    from sdsafe_features.trajectory import normalize_messages  # noqa: E402
    from sdsafe_fine_confirm.sources import load_cohort, source_paths  # noqa: E402

    tok = tokenizer if tokenizer is not None else _tokenizer()
    selected = [e for e in episode_ids if row_index.episode_scope.get(e) in ("ctx40k", "agentlab")]
    by_scope: dict[str, list[str]] = defaultdict(list)
    for e in selected:
        by_scope[row_index.episode_scope[e]].append(e)
    produced = 0
    for scope_name in sorted(by_scope):
        paths = source_paths(SDSAFE, scope_name)
        cohort = load_cohort(SDSAFE, scope_name)
        wanted = {str(e).split("::", 1)[-1] for e in by_scope[scope_name]}
        for episode in cohort.episodes:
            if episode.episode_id not in wanted:
                continue
            ns = str(episode.namespaced_episode_id)
            if ns not in row_index.episode_scope:
                ns = str(by_scope[scope_name][0]).split("::")[0] + "::" + episode.episode_id
            record = episode.record
            messages_path = Path(paths.source_root) / str(record["source_path"]) / "newest_messages.json"
            if not messages_path.is_file():
                raise StudyError(f"legacy messages missing: {messages_path}")
            raw = json.loads(messages_path.read_text(encoding="utf-8"))
            text = tok.apply_chat_template(normalize_messages(raw), tokenize=False, add_generation_prompt=False)
            ids_list = list(tok(text, add_special_tokens=False, truncation=False).input_ids)
            for round_id in sorted({k[2] for k in row_index.by_round if k[0] == scope_name and k[1] == ns}):
                rows = row_index.by_round[(scope_name, ns, round_id)]
                positions = []
                for row in rows:
                    pos = int(row["feature_token_index"])
                    if pos != int(row["token_end"]):
                        raise StudyError(f"{row['event_id']}: feature_token_index != token_end")
                    digest = canonical_sha256(ids_list[: pos + 1])
                    if digest != str(row["sequence_sha256"]):
                        raise StudyError(
                            f"{row['event_id']}: legacy render prefix digest {digest} != row digest"
                        )
                    positions.append(pos)
                if positions != sorted(set(positions)):
                    raise StudyError(f"{ns} r{round_id}: CT endpoints not unique/increasing")
                produced += 1
                yield (
                    RoundTask(
                        scope=scope_name,
                        episode_id=ns,
                        namespaced_episode_id=ns,
                        round_id=int(round_id),
                        sequence_sha256=str(rows[0]["sequence_sha256"]),
                        sequence_length=len(ids_list),
                        prompt_length=int(rows[0]["prompt_length"]),
                        terminal_token_id=IM_END_ID,
                        label_value=bool(rows[0]["strict_final"]),
                        group_id=str(rows[0]["group_id"]),
                        group_hash=str(rows[0]["group_hash"]),
                        scope_tag=str(rows[0]["scope"]),
                        positions=positions,
                        events=rows,
                    ),
                    ids_list,
                )
            if limit is not None and produced >= limit:
                return
