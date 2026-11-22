"""Source-neutral fine-grained event data for the four-source full cohort.

This package is the data/source/event-index seam of the fine-grained confirmation
experiment. It owns exactly three things and nothing else:

1. a *source-neutral* cohort API over the four frozen sources of the unified 12,618
   episode split (``cot32k`` 3,837 / ``nothink_ctx40k`` 3,572 / ``ctx40k`` 3,857 /
   ``ramp_ablation`` 1,352 AgentLAB episodes), using the unified
   ``all_data_mixed/shared/split`` labels as the *only* split authority;
2. deterministic per-round token sequences for every episode, reusing each source's
   frozen serialization verbatim (real archived CoT ids, the frozen no-think render, and
   the two legacy *rendered prefix* reconstructions) and proving every legacy round
   against that source's frozen endpoint prefix hashes/offsets;
3. an event index for all four sources that exposes **every** ``tool_call``,
   ``content_segment`` and ``round_end`` of **every** goal type (plus ``strict_final``),
   with the official checker labels from the two existing event stores reused read-only
   and the two legacy sources built under ``artifacts/fine_confirm/data/{ctx40k,agentlab}``.

The frozen inference rule this data must support (see ``artifacts/fine_confirm/PROTOCOL.md``):
every ``tool_call`` endpoint and every visible ``content_segment`` endpoint of every
episode is a scored position, regardless of goal type, source or final label. Labels are
supervision/evaluation only and never select an inference position. An ``unknown`` label
is ``null`` and is never coerced to ``0``.

Nothing in this package runs the victim model, extracts hidden states, trains a probe,
scores a test set, or writes a receipt. Historical artifacts, splits, checkpoints and the
two existing event stores are read-only.

Scopes and source tags
----------------------
================  =================  ========  ===============  ===============================
scope             ``source_tag``     victim    episodes         sequence provenance
================  =================  ========  ===============  ===============================
``cot``           ``cot32k``         Qwen3-8B  3,837            real archived prompt+completion ids
``nothink``       ``nothink_ctx40k`` Qwen3-8B  3,572            frozen no-think template render
``ctx40k``        ``ctx40k``         Qwen3-8B  3,857            legacy rendered trajectory prefix
``agentlab``      ``ramp_ablation``  per archive 1,352          legacy rendered trajectory prefix
================  =================  ========  ===============  ===============================

``agentlab`` is the AgentLAB ``ramp_ablation`` cohort; its episode ids are namespaced with
the unified tag ``ramp_ablation`` so every id joins the unified split directly.
"""

from __future__ import annotations

from pathlib import PurePosixPath

# Re-exported read-only vocabulary: the event/label language is shared with the two frozen
# event stores, so downstream code must not have to know which store a row came from.
from sdsafe_event_data import (  # noqa: F401
    CHECKER,
    CONTROL_EXACT,
    CONTROL_LEVELS,
    CONTROL_MATERIAL,
    CONTROL_SEMANTIC_WHITELIST,
    EVENT_CONTENT_SEGMENT,
    EVENT_ROUND_END,
    EVENT_TOOL_CALL,
    EVENT_TYPE_ORDER,
    EVENT_TYPES,
    FROZEN_HASH_EXACT,
    FROZEN_HASH_LEVELS,
    FROZEN_HASH_UNAVAILABLE,
    LABEL_EXACT,
    LABEL_UNKNOWN,
    MAX_SEGMENT_TOKENS,
    MIN_SEGMENT_CHARS,
    SENTENCE_TERMINATORS,
    UNKNOWN_REASONS,
)

SCHEMA_VERSION = 1
CONFIG_STAGE = "fine_confirm.data.config"
MANIFEST_STAGE = "fine_confirm.data.manifest"
VALIDATION_STAGE = "fine_confirm.data.validation"
PROGRESS_STAGE = "fine_confirm.data.progress"

SCOPES = ("cot", "nothink", "ctx40k", "agentlab")
FROZEN_SCOPES = ("cot", "nothink")
LEGACY_SCOPES = ("ctx40k", "agentlab")

SOURCE_TAG = {
    "cot": "cot32k",
    "nothink": "nothink_ctx40k",
    "ctx40k": "ctx40k",
    "agentlab": "ramp_ablation",
}
SCOPE_BY_SOURCE_TAG = {tag: scope for scope, tag in SOURCE_TAG.items()}

QWEN3_8B = "Qwen3-8B"
UNLABELED_DEFAULT = "unlabeled_default"
"""AgentLAB ``long_horizon/<suite>/<version>/<pair>`` layout: the archive names no model.
The value is written verbatim and never guessed."""

VICTIM_MODEL = {
    "cot": QWEN3_8B,
    "nothink": QWEN3_8B,
    "ctx40k": QWEN3_8B,
    "agentlab": None,  # per episode, parsed from the frozen source path
}

EXPECTED_EPISODES = {"cot": 3837, "nothink": 3572, "ctx40k": 3857, "agentlab": 1352}
EXPECTED_TOTAL_EPISODES = 12618
EXPECTED_SPLIT_EPISODES = {"train": 8229, "dev": 2386, "test": 2003}
SPLIT_NAMES = ("train", "dev", "test")

SCOPE_DIR = {
    "ctx40k": "artifacts/fine_confirm/data/ctx40k",
    "agentlab": "artifacts/fine_confirm/data/agentlab",
}
"""Output roots of the two new legacy event indexes. The frozen stores are elsewhere."""

FROZEN_EVENT_STORE = {
    "cot": "artifacts/event_data/cot",
    "nothink": "artifacts/event_data/nothink",
}

# --- sequence provenance: three serializers, never interchangeable -------------------------
LEGACY_SERIALIZER = "legacy_render_normalize_messages_empty_scaffold"
"""The two legacy sources never stored per-round prompt/completion ids. Their frozen features
were built from ``sdsafe_features.trajectory.normalize_messages`` + ``render_trajectory``:
one chat-template render of the whole archived trajectory, history assistants carrying their
visible content and tool calls only, an empty (whitespace-only reasoning) think scaffold, and
*no appended tool schema*. A per-round sequence is the token prefix of that frozen render that
ends at the round's ``<|im_end|>``. Calling the no-think serializer here would change every
token id (177/159 tokens -> 1,379/1,622 on a train episode), so it is forbidden."""

LEGACY_SEQUENCE_PROVENANCE = {
    "ctx40k": (
        "legacy CTX40K rendered trajectory prefix: the served run used CoT but the archive dropped "
        "reasoning, so the sequence is the archived visible trajectory reconstruction, not archived ids"
    ),
    "agentlab": (
        "legacy AgentLAB rendered trajectory prefix: full archived trajectory / empty-scaffold "
        "reconstruction of the old archive, not re-extracted and not re-rendered with the no-think serializer"
    ),
}

FROZEN_SEQUENCE_PROVENANCE = {
    "cot": (
        "archived real prompt/completion token ids of every assistant round (reasoning plus visible "
        "content plus tool calls); the local Qwen3 template is not used to rebuild them"
    ),
    "nothink": (
        "declared real no-think rounds re-rendered with the frozen Qwen3 template (enable_thinking=false); "
        "the archive does not prove the served run disabled thinking"
    ),
}

RECONSTRUCTED = "reconstructed"
"""Provenance tag of every legacy sequence and event. It is never the victim's actual hidden
state input: the tokens are a frozen-verified reconstruction of the visible trajectory."""

LEGACY_ALIGNMENT_ROUNDTRIP = "legacy_frozen_render_roundtrip"
LEGACY_ALIGNMENT_COMPLETION = "legacy_frozen_render_completion"
LEGACY_ALIGNMENT_SOURCE = "frozen_render_offsets_legacy"

LEGACY_CALL_SPAN_TEXT = "legacy_render_text"
LEGACY_CALL_SPAN_TOKEN_FALLBACK = "token_id_last_n_fallback"

SPLIT_AUTHORITY = "artifacts/all_data_mixed/shared/split"

# --- scoring positions: the universal inference rule --------------------------------------
SCORE_POSITIONS = EVENT_TYPES
"""All three event families are scored at inference. There is no goal-type oracle: a
``content_segment`` of a tool-target episode is scored even when its own label is unknown, and
a ``round_end`` is kept for the matched whole-round control."""

# --- validation / build sizes -------------------------------------------------------------
N_SHARDS = 32
ESTIMATED_BYTES_PER_EVENT = 1200
ESTIMATED_EVENTS_PER_EPISODE = 8
DISK_HEADROOM_BYTES = 2 * 1024**3

EVENT_FIELD_SPEC = {
    "identity": (
        "event_id, source_tag, scope, mode, episode_id, namespaced_episode_id, victim_model, "
        "group_id, group_hash, suite, user_task_id, injection_task_id, benchmark_version, attack_type"
    ),
    "split": (
        "split (unified all_data_mixed authority), source_split (the old per-source label, annotation "
        "only), round_split, mixed_split, evaluation_scope"
    ),
    "position": (
        "round_id, message_index, event_type, index, feature_token_index, token_start, token_end, "
        "sequence_sha256, round_sequence_sha256, sequence_length, prompt_length, "
        "sequence_provenance, sequence_serializer, inference_position"
    ),
    "label": (
        "previous_security, prefix_security, security_flip, episode_final, strict_final, label_status, "
        "label_reason, label_usable"
    ),
    "environment_evidence": (
        "atomic_vs_control, control_results, control_round_end_security, control_round_end_matches_atomic, "
        "control_dropped_fields, control_canonicalized_lists, control_first_difference, "
        "control_post_environment_sha256, round_post_environment_sha256, frozen_hash, "
        "frozen_post_environment_sha256"
    ),
}
"""Stable event-row schema across the four sources. ``split`` is always the unified label;
``label_status == "unknown"`` always means ``prefix_security is None`` (never ``False``)."""


class FineConfirmDataError(RuntimeError):
    """The frozen inputs do not form the cohort/sequence/index this contract names."""


def namespaced_episode_id(source_tag: str, episode_id: str) -> str:
    return f"{source_tag}::{episode_id}"


def victim_model_of(scope: str, source_path: str, *, suite: str, benchmark_version: str) -> str:
    """Resolve the victim model of one episode without guessing.

    The three Qwen sources are always ``Qwen3-8B``. AgentLAB stores the victim as a path
    component when it knows it, and *no* model component when the archive ran the
    unlabeled default layout; the latter is reported as ``unlabeled_default``.
    """
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope: {scope}")
    if scope != "agentlab":
        return VICTIM_MODEL[scope]
    parts = PurePosixPath(str(source_path)).parts
    if not parts or parts[0] != "long_horizon":
        raise FineConfirmDataError(
            f"AgentLAB source path {source_path!r} is not the frozen long_horizon layout"
        )
    if len(parts) == 5:
        model, path_suite, version, pair = parts[1], parts[2], parts[3], parts[4]
        victim = model
    elif len(parts) == 4:
        path_suite, version, pair = parts[1], parts[2], parts[3]
        victim = UNLABELED_DEFAULT
    else:
        raise FineConfirmDataError(
            f"AgentLAB source path {source_path!r} has {len(parts)} components; expected "
            "long_horizon/[model/]<suite>/<version>/<pair>"
        )
    if path_suite != suite:
        raise FineConfirmDataError(
            f"AgentLAB source path suite {path_suite!r} != record suite {suite!r} ({source_path})"
        )
    if version != benchmark_version:
        raise FineConfirmDataError(
            f"AgentLAB source path version {version!r} != record benchmark_version {benchmark_version!r} "
            f"({source_path})"
        )
    if not pair.startswith("user_task_"):
        raise FineConfirmDataError(f"AgentLAB source path pair {pair!r} is not a user/injection pair")
    return victim


def is_scored_event(event: dict) -> bool:
    """The universal inference rule: every one of the three event families is scored."""
    return event.get("event_type") in SCORE_POSITIONS
