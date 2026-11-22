"""Frozen source registry and the strict four-source cohort join.

Every source is addressed by its *scope* (``cot`` / ``nothink`` / ``ctx40k`` /
``agentlab``). The unified ``artifacts/all_data_mixed/shared/split`` is the only split
authority; the old per-source split label is kept as an annotation (``source_split``) and
is allowed to differ in any direction, because the unified whole-group split supersedes
it. In particular:

* ``ctx40k`` has one ``test -> dev`` episode and 126 ``train -> dev`` episodes;
* ``ramp_ablation`` (AgentLAB) has arbitrary migrations (e.g. ``test -> train``).

Identity, however, is still strict. For every episode the replay ``eligible_episodes``
record, the unified episode row and the old per-source split row must agree on
``suite`` / ``user_task_id`` / ``injection_task_id`` / ``group_id`` / ``group_hash`` /
``round_count`` / ``strict_label`` and (where the old row records them)
``benchmark_version`` / ``attack_type``; every round must agree on its labels. The two
legacy split roots also carry no ``config.json``, so the old ``load_split`` path is not
reused: metadata is derived from the replay record plus the unified row, and the old row
is an *additional* witness, never the authority.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sdsafe_features.common import read_jsonl
from sdsafe_ramp_ablation.common import group_hash

from . import (
    EXPECTED_EPISODES,
    EXPECTED_SPLIT_EPISODES,
    EXPECTED_TOTAL_EPISODES,
    FineConfirmDataError,
    FROZEN_EVENT_STORE,
    LEGACY_SCOPES,
    SCOPE_DIR,
    SOURCE_TAG,
    SPLIT_AUTHORITY,
    SPLIT_NAMES,
    victim_model_of,
)

REPLAY_ROOT = {
    "cot": "artifacts/cot32k/replay_semantic",
    "nothink": "artifacts/nothink_ctx40k/replay_semantic",
    "ctx40k": "artifacts/ctx40k/replay_semantic",
    "agentlab": "artifacts/replay_full",
}
SPLIT_DIR = {
    "cot": "artifacts/cot32k/shared/split",
    "nothink": "artifacts/nothink_ctx40k/shared/split",
    "ctx40k": "artifacts/ctx40k/new_only/shared/split",
    "agentlab": "artifacts/ramp_ablation/shared/split",
}
SOURCE_ROOT = {
    "cot": "data/raw/task_injection_cot32k_r1",
    "nothink": "data/compat/task_injection_nothink_ctx40k_r1",
    "ctx40k": "data/compat/task_injection_ctx40k_r1",
    "agentlab": "external/AgentLAB/Task-Injection/agentdojo/res",
}
FEATURES_DIR = {
    "ctx40k": "artifacts/ctx40k/new_only/shared/features",
    "agentlab": "artifacts/ramp_ablation/shared/features",
}
AGENTLAB_ROOT = "external/AgentLAB"

ADAPTER = {
    "cot": "archived_ids",
    "nothink": "frozen_nothink_template",
    "ctx40k": "DirectInjectionAdapter",
    "agentlab": "VendoredAgentDojo",
}
"""The legacy AgentLAB injection archive is the Human_Reasoning snippet format, so it needs
``VendoredAgentDojo`` to render payloads; CTX40K stores direct ``{vector: payload}`` strings
and needs ``DirectInjectionAdapter``. Using one adapter for both is a semantic defect."""

IDENTITY_FIELDS = (
    "suite",
    "user_task_id",
    "injection_task_id",
    "group_id",
    "group_hash",
    "round_count",
    "strict_label",
)
ROUND_FIELDS = ("train_label", "first_hit", "final_hijacked", "group_id", "group_hash", "strict_label")


@dataclass(frozen=True)
class SourcePaths:
    root: Path
    scope: str
    source_tag: str
    split_dir: Path
    replay_root: Path
    source_root: Path
    agentlab_root: Path
    output_dir: Path
    features_dir: Path | None
    expected_episodes: int
    adapter: str

    @property
    def eligible_path(self) -> Path:
        return self.replay_root / "eligible_episodes.jsonl"

    @property
    def is_legacy(self) -> bool:
        return self.scope in LEGACY_SCOPES

    @property
    def endpoint_index_path(self) -> Path | None:
        return None if self.features_dir is None else self.features_dir / "endpoint_index.jsonl"

    def describe(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "source_tag": self.source_tag,
            "split_dir": str(self.split_dir),
            "replay_root": str(self.replay_root),
            "source_root": str(self.source_root),
            "agentlab_root": str(self.agentlab_root),
            "output_dir": str(self.output_dir),
            "features_dir": None if self.features_dir is None else str(self.features_dir),
            "expected_episodes": self.expected_episodes,
            "adapter": self.adapter,
        }


def source_paths(root: Path | str, scope: str, build_dir: Path | str | None = None) -> SourcePaths:
    """Frozen input paths of one scope; ``build_dir`` overrides where its own index lives."""
    if scope not in SOURCE_TAG:
        raise ValueError(f"Unknown scope: {scope}")
    base = Path(root)
    default_output = SCOPE_DIR[scope] if scope in SCOPE_DIR else FROZEN_EVENT_STORE[scope]
    output = Path(build_dir) if build_dir is not None else base / default_output
    features = base / FEATURES_DIR[scope] if scope in FEATURES_DIR else None
    return SourcePaths(
        root=base,
        scope=scope,
        source_tag=SOURCE_TAG[scope],
        split_dir=base / SPLIT_DIR[scope],
        replay_root=base / REPLAY_ROOT[scope],
        source_root=base / SOURCE_ROOT[scope],
        agentlab_root=base / AGENTLAB_ROOT,
        output_dir=output,
        features_dir=features,
        expected_episodes=EXPECTED_EPISODES[scope],
        adapter=ADAPTER[scope],
    )


def all_source_paths(
    root: Path | str, build_dirs: dict[str, Path | str] | None = None
) -> dict[str, SourcePaths]:
    build_dirs = build_dirs or {}
    return {scope: source_paths(root, scope, build_dirs.get(scope)) for scope in SOURCE_TAG}


def _rows_by_episode(rows: list[dict[str, Any]], path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id:
            raise FineConfirmDataError(f"{path}: a row has no episode_id")
        grouped.setdefault(episode_id, []).append(row)
    return grouped


def _consecutive_rounds(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    keys = sorted(int(row["round_id"]) for row in rows)
    if keys != list(range(1, len(rows) + 1)):
        raise FineConfirmDataError(f"{label}: round ids are not 1..{len(rows)}: {keys[:5]}")
    return {int(row["round_id"]): row for row in rows}


@dataclass(frozen=True)
class CohortEpisode:
    """One episode of the four-source cohort with every frozen join row it must agree with."""

    scope: str
    source_tag: str
    episode_id: str
    record: dict[str, Any]
    unified_row: dict[str, Any]
    source_row: dict[str, Any] | None
    round_rows: dict[int, dict[str, Any]]
    unified_round_rows: dict[int, dict[str, Any]]
    split: str
    source_split: str | None
    victim_model: str
    round_count: int

    @property
    def namespaced_episode_id(self) -> str:
        return f"{self.source_tag}::{self.episode_id}"

    @property
    def group_id(self) -> str:
        return str(self.unified_row["group_id"])

    @property
    def group_hash(self) -> str:
        return str(self.unified_row["group_hash"])

    @property
    def strict_final(self) -> bool:
        return bool(self.record["replay_strict_hijacked"])

    @property
    def suite(self) -> str:
        return str(self.unified_row["suite"])

    @property
    def user_task_id(self) -> str:
        return str(self.unified_row["user_task_id"])

    @property
    def injection_task_id(self) -> str:
        return str(self.unified_row["injection_task_id"])

    @property
    def benchmark_version(self) -> str:
        return str(self.unified_row["benchmark_version"])

    @property
    def attack_type(self) -> str | None:
        return self.unified_row.get("attack_type")

    @property
    def evaluation_scope(self) -> str | None:
        return self.unified_row.get("evaluation_scope")

    def canonical_episode_row(self) -> dict[str, Any]:
        """The identity row every event of this episode is written against."""
        return {
            "episode_id": self.episode_id,
            "namespaced_episode_id": self.namespaced_episode_id,
            "source_tag": self.source_tag,
            "scope": self.scope,
            "victim_model": self.victim_model,
            "suite": self.unified_row["suite"],
            "user_task_id": self.unified_row["user_task_id"],
            "injection_task_id": self.unified_row["injection_task_id"],
            "benchmark_version": self.unified_row["benchmark_version"],
            "attack_type": self.unified_row["attack_type"],
            "group_id": self.unified_row["group_id"],
            "group_hash": self.unified_row["group_hash"],
            "round_count": self.round_count,
            "strict_final": self.strict_final,
            "split": self.split,
            "source_split": self.source_split,
            "evaluation_scope": self.unified_row.get("evaluation_scope"),
            "split_authority": SPLIT_AUTHORITY,
        }

    def canonical_round_rows(self, sequence_sha256: dict[int, str]) -> dict[int, dict[str, Any]]:
        """Per-round rows carrying the unified split and the verified round sequence hash.

        The old legacy round splits record no ``sequence_sha256``; the reconstructed hash of
        the frozen-verified legacy render prefix is used instead and the event rows are
        annotated with that provenance, never presented as an archived hash.
        """
        rows: dict[int, dict[str, Any]] = {}
        for round_id in sorted(self.round_rows):
            source = self.round_rows[round_id]
            unified = self.unified_round_rows[round_id]
            canonical = dict(source)
            canonical.update(
                {
                    "episode_id": self.episode_id,
                    "round_id": round_id,
                    "split": self.split,
                    "source_split": self.source_split,
                    "sequence_sha256": sequence_sha256[round_id],
                    "sequence_sha256_source": (
                        "archived_round_split"
                        if "sequence_sha256" in source
                        else "reconstructed_legacy_render_prefix"
                    ),
                    "message_index": source.get("message_index", unified.get("message_index")),
                    "train_label": bool(unified["train_label"]),
                    "first_hit": bool(unified["first_hit"]),
                    "final_hijacked": bool(unified["final_hijacked"]),
                    "strict_label": bool(unified["strict_label"]),
                }
            )
            rows[round_id] = canonical
        return rows


@dataclass
class Cohort:
    paths: SourcePaths
    episodes: list[CohortEpisode]
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    migrations: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    victim_counts: dict[str, int] = field(default_factory=dict)

    def by_id(self) -> dict[str, CohortEpisode]:
        return {episode.episode_id: episode for episode in self.episodes}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FineConfirmDataError(message)


def load_unified_episode_rows(root: Path | str) -> dict[str, dict[str, Any]]:
    path = Path(root) / SPLIT_AUTHORITY / "episode_split.jsonl"
    rows = read_jsonl(path)
    rows_by_id = {str(row["episode_id"]): row for row in rows}
    _require(len(rows_by_id) == len(rows), "unified episode split has duplicate episode ids")
    return rows_by_id


def load_unified_round_rows(root: Path | str, source_tag: str) -> dict[str, dict[int, dict[str, Any]]]:
    path = Path(root) / SPLIT_AUTHORITY / "round_split.jsonl"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        if row.get("source_tag") != source_tag:
            continue
        grouped.setdefault(str(row["source_original_id"]), []).append(row)
    return {episode_id: {int(row["round_id"]): row for row in rows} for episode_id, rows in grouped.items()}


def load_cohort(root: Path | str, scope: str, *, expected_episodes: int | None = None) -> Cohort:
    """Join one frozen source to the unified split; raise on any unexplained identity drift.

    ``expected_episodes`` exists only for synthetic fixtures; production callers use the
    frozen per-source count.
    """
    paths = source_paths(root, scope)
    expected = paths.expected_episodes if expected_episodes is None else int(expected_episodes)
    _require(paths.split_dir.is_dir(), f"missing per-source split dir: {paths.split_dir}")
    _require(paths.eligible_path.is_file(), f"missing replay cohort: {paths.eligible_path}")

    unified_episodes = load_unified_episode_rows(root)
    unified_rounds = load_unified_round_rows(root, paths.source_tag)
    records = {str(row["episode_id"]): row for row in read_jsonl(paths.eligible_path)}
    split_rows = _rows_by_episode(read_jsonl(paths.split_dir / "episode_split.jsonl"), paths.split_dir)
    round_rows = _rows_by_episode(read_jsonl(paths.split_dir / "round_split.jsonl"), paths.split_dir)
    _require(len(split_rows) == expected, f"{scope}: {len(split_rows)} split episodes")

    exclusions: list[dict[str, Any]] = []
    exclusion_path = paths.split_dir / "exclusions.jsonl"
    if exclusion_path.is_file():
        exclusions = [{"stage": "source_split", **row} for row in read_jsonl(exclusion_path)]

    # Cohort membership is the per-source split; an eligible replay episode that never
    # entered the split must be documented as a source-stage exclusion.
    excluded_ids = {str(row["episode_id"]) for row in exclusions}
    if paths.is_legacy:
        _require(set(records) == set(split_rows), f"{scope}: replay cohort and split membership differ")
    else:
        undocumented = set(records) - set(split_rows) - excluded_ids
        _require(
            not undocumented,
            f"{scope}: eligible episodes absent from the split without an exclusion: {sorted(undocumented)[:5]}",
        )

    episodes: list[CohortEpisode] = []
    migrations: Counter = Counter()
    split_counts: Counter = Counter()
    victim_counts: Counter = Counter()
    for episode_id in sorted(split_rows):
        record = records.get(episode_id)
        _require(record is not None, f"{scope}/{episode_id}: split episode has no replay record")
        _require(
            record.get("eligible_for_hidden_states") is True,
            f"{scope}/{episode_id}: replay record is not C-qualified",
        )
        namespaced = f"{paths.source_tag}::{episode_id}"
        unified = unified_episodes.get(namespaced)
        _require(unified is not None, f"{scope}/{episode_id}: absent from the unified episode split")
        _require(
            str(unified.get("source_original_id")) == episode_id,
            f"{scope}/{episode_id}: unified row has the wrong source_original_id",
        )
        _require(
            unified.get("source_tag") == paths.source_tag,
            f"{scope}/{episode_id}: unified row has the wrong source_tag",
        )
        _require(episode_id in split_rows, f"{scope}/{episode_id}: absent from its own split")
        source_row = split_rows[episode_id][0] if len(split_rows[episode_id]) == 1 else None
        _require(source_row is not None, f"{scope}/{episode_id}: duplicate per-source split rows")

        # ---- identity: replay record vs unified row -------------------------------
        for name in ("suite", "user_task_id", "injection_task_id"):
            _require(
                record.get(name) == unified.get(name),
                f"{scope}/{episode_id}: {name} {record.get(name)!r} != unified {unified.get(name)!r}",
            )
        _require(
            int(record["round_count"]) == int(unified["round_count"]),
            f"{scope}/{episode_id}: round_count disagrees with the unified split",
        )
        _require(
            bool(record["replay_strict_hijacked"]) is bool(unified["strict_label"]),
            f"{scope}/{episode_id}: strict label disagrees with the unified split",
        )
        _require(
            bool(record["replay_final"]) is bool(record["replay_strict_hijacked"]),
            f"{scope}/{episode_id}: replay_final != replay_strict_hijacked",
        )
        _require(
            record.get("benchmark_version") == unified.get("benchmark_version"),
            f"{scope}/{episode_id}: benchmark_version disagrees with the unified split",
        )
        if "attack_type" in record and record.get("attack_type") is not None:
            _require(
                record.get("attack_type") == unified.get("attack_type"),
                f"{scope}/{episode_id}: attack_type disagrees with the unified split",
            )
        expected_group = "|".join(
            (str(record["suite"]), str(record["user_task_id"]), str(record["injection_task_id"]))
        )
        _require(
            unified.get("group_id") == expected_group,
            f"{scope}/{episode_id}: unified group_id is not suite|user_task|injection",
        )
        _require(
            unified.get("group_hash") == group_hash(expected_group),
            f"{scope}/{episode_id}: unified group_hash does not close over the group id",
        )

        # ---- identity: replay record / unified row vs the old per-source row ------
        for name in IDENTITY_FIELDS:
            if name not in source_row:
                continue
            canonical_value = unified.get(name)
            actual = source_row.get(name)
            if name in ("round_count",):
                _require(int(actual) == int(canonical_value), f"{scope}/{episode_id}: source {name} differs")
            elif name == "strict_label":
                _require(bool(actual) is bool(canonical_value), f"{scope}/{episode_id}: source {name} differs")
            else:
                _require(
                    actual == canonical_value,
                    f"{scope}/{episode_id}: source {name} {actual!r} != {canonical_value!r}",
                )
        for name in ("benchmark_version", "attack_type"):
            if name in source_row and source_row.get(name) is not None:
                _require(
                    source_row.get(name) == unified.get(name),
                    f"{scope}/{episode_id}: source {name} disagrees with the unified split",
                )

        # ---- split authority ------------------------------------------------------
        unified_split = str(unified["split"])
        _require(unified_split in SPLIT_NAMES, f"{scope}/{episode_id}: bad unified split {unified_split!r}")
        source_split = str(source_row["split"]) if "split" in source_row else None
        _require(source_split in SPLIT_NAMES, f"{scope}/{episode_id}: bad source split {source_split!r}")
        if source_split != unified_split:
            migrations[f"{source_split}->{unified_split}"] += 1

        # ---- rounds ---------------------------------------------------------------
        round_count = int(record["round_count"])
        own_rounds = _consecutive_rounds(round_rows.get(episode_id, []), f"{scope}/{episode_id}")
        _require(len(own_rounds) == round_count, f"{scope}/{episode_id}: round row count mismatch")
        unified_episode_rounds = unified_rounds.get(episode_id, {})
        _require(len(unified_episode_rounds) == round_count, f"{scope}/{episode_id}: unified rounds missing")
        for round_id, own in own_rounds.items():
            unified_round = unified_episode_rounds.get(round_id)
            _require(unified_round is not None, f"{scope}/{episode_id} r{round_id}: missing unified round row")
            for name in ROUND_FIELDS:
                if name not in own:
                    continue
                if name in ("train_label", "first_hit", "final_hijacked", "strict_label"):
                    _require(
                        bool(own[name]) is bool(unified_round[name]),
                        f"{scope}/{episode_id} r{round_id}: round {name} disagrees with the unified split",
                    )
                else:
                    _require(
                        own[name] == unified_round[name],
                        f"{scope}/{episode_id} r{round_id}: round {name} disagrees with the unified split",
                    )
            _require(
                bool(unified_round["final_hijacked"]) is bool(record["replay_final"]),
                f"{scope}/{episode_id} r{round_id}: unified round final label disagrees with the replay",
            )

        victim = victim_model_of(
            scope,
            str(record["source_path"]),
            suite=str(record["suite"]),
            benchmark_version=str(record["benchmark_version"]),
        )
        episode = CohortEpisode(
            scope=scope,
            source_tag=paths.source_tag,
            episode_id=episode_id,
            record=record,
            unified_row=unified,
            source_row=source_row,
            round_rows=own_rounds,
            unified_round_rows=unified_episode_rounds,
            split=unified_split,
            source_split=source_split,
            victim_model=victim,
            round_count=round_count,
        )
        episodes.append(episode)
        split_counts[unified_split] += 1
        victim_counts[victim] += 1

    _require(
        len(episodes) == expected,
        f"{scope}: cohort closure failed: {len(episodes)} != {expected}",
    )
    _require(sum(split_counts.values()) == expected, f"{scope}: split counts do not sum to the cohort")
    return Cohort(
        paths=paths,
        episodes=episodes,
        exclusions=exclusions,
        migrations=dict(sorted(migrations.items())),
        split_counts=dict(sorted(split_counts.items())),
        victim_counts=dict(sorted(victim_counts.items())),
    )


def load_all_cohorts(root: Path | str) -> dict[str, Cohort]:
    cohorts = {scope: load_cohort(root, scope) for scope in SOURCE_TAG}
    validate_unified_totals(cohorts)
    return cohorts


def validate_unified_totals(cohorts: dict[str, Cohort]) -> dict[str, Any]:
    """The four cohorts must be exactly the unified 12,618 / 8,229 / 2,386 / 2,003 split."""
    episodes = sum(len(cohort.episodes) for cohort in cohorts.values())
    if episodes != EXPECTED_TOTAL_EPISODES:
        raise FineConfirmDataError(f"four-source cohort has {episodes} episodes, expected {EXPECTED_TOTAL_EPISODES}")
    split_totals: Counter = Counter()
    victim_totals: Counter = Counter()
    namespaced: set[str] = set()
    for scope, cohort in cohorts.items():
        if len(cohort.episodes) != EXPECTED_EPISODES[scope]:
            raise FineConfirmDataError(f"{scope}: {len(cohort.episodes)} episodes")
        for episode in cohort.episodes:
            split_totals[episode.split] += 1
            victim_totals[episode.victim_model] += 1
            if episode.namespaced_episode_id in namespaced:
                raise FineConfirmDataError(f"duplicate namespaced episode id: {episode.namespaced_episode_id}")
            namespaced.add(episode.namespaced_episode_id)
    for split, expected in EXPECTED_SPLIT_EPISODES.items():
        if split_totals[split] != expected:
            raise FineConfirmDataError(f"unified {split} total is {split_totals[split]}, expected {expected}")
    return {
        "episodes": episodes,
        "split": dict(sorted(split_totals.items())),
        "victim_model": dict(sorted(victim_totals.items())),
        "namespaced_ids_unique": len(namespaced),
        "per_scope": {scope: len(cohort.episodes) for scope, cohort in cohorts.items()},
        "migrations": {scope: cohort.migrations for scope, cohort in cohorts.items()},
    }
