"""Lane D seam for the fine_label_repair_v1 DEV-only study.

This package owns only durable CPU metadata:

* authoritative TRAIN/DEV row identity and feature-ref contracts;
* the proved 111 CoT fallback-alignment positions;
* the full TRAIN/DEV CoT trailing-UTF8 endpoint audit and replacement map;
* exact official-checker labels for new/changed positions;
* the five fixed supervision policies L0..L4 and the 36-run menu encoded in
  ``artifacts/fine_label_repair_v1/protocol.json``.

It never reads a feature vector, never forwards Qwen, never trains, and never touches a
TEST raw trace.  Existing cached BF16 vectors are referenced by immutable
``(scope, local_feature_row)`` coordinates; repaired/new endpoints get a supplemental
``actual IDs / round / token index`` reference for lane F to materialise later.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = 1
STUDY_ID = "fine_label_repair_v1"

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/fine_label_repair_v1"
DATA_ROOT = ARTIFACT_ROOT / "data"
HANDOFF_ROOT = ARTIFACT_ROOT / "handoffs"

# Immutable study artifacts.
FINE_AUDIT_ROOT = ROOT / "artifacts/fine_error_audit_v1"
COT_EVENT_STORE = ROOT / "artifacts/event_data/cot"
COT_REPLAY_ROOT = ROOT / "artifacts/cot32k/replay_semantic"
COT_SOURCE_ROOT = ROOT / "data/raw/task_injection_cot32k_r1"
COT_COMPAT_ROOT = ROOT / "data/compat/task_injection_cot32k_r1"
AGENTLAB_ROOT = ROOT / "external/AgentLAB"
MODEL_ROOT = ROOT / "models/Qwen3-8B"

CANDIDATE_POSITIONS_PATH = FINE_AUDIT_ROOT / "candidate_positions.jsonl"
ALIGNMENT_PROOFS_PATH = FINE_AUDIT_ROOT / "alignment_proofs.jsonl"
REPAIR_PLAN_PATH = FINE_AUDIT_ROOT / "repair_plan.json"

EVENT_TOOL_CALL = "tool_call"
EVENT_CONTENT_SEGMENT = "content_segment"
EVENT_ROUND_END = "round_end"
CT_EVENT_TYPES = (EVENT_TOOL_CALL, EVENT_CONTENT_SEGMENT)
TARGET_EVENT_TYPES = CT_EVENT_TYPES

SPLIT_TRAIN = "train"
SPLIT_DEV = "dev"
PROTECTED_SPLITS = (SPLIT_TRAIN, SPLIT_DEV)
FORBIDDEN_SPLIT = "test"

LABEL_EXACT = "exact"
LABEL_UNKNOWN = "unknown"

POLICIES = ("L0", "L1", "L2", "L3", "L4")
POLICY_NAMES = {
    "L0": "legacy_first_hit_recomputed",
    "L1": "first_hit_censored",
    "L2": "first_round_state_censored",
    "L3": "prefix_state",
    "L4": "random_drop_control",
}
LEARNING_RATES = ("1e-3", "3e-4")
SEEDS = (0, 1, 2)
BALANCED_POLICIES = POLICIES
NATURAL_CONTROL_COHORTS = ("repaired_train_L0_natural", "legacy_train_L0_natural")

# Expected old anchor counts.  The new repaired counts are computed, never hardcoded.
EXPECTED_LEGACY_CT_FIRST_HIT_ROWS = 107_215
EXPECTED_LEGACY_CT_FIRST_HIT_POSITIVES = 1_721
EXPECTED_LEGACY_CT_FIRST_HIT_NEGATIVES = 105_494
EXPECTED_LEGACY_CT_LABEL0_PREFIX_POSITIVE = 14_045
EXPECTED_TRAIN_EPISODES = 8_229
EXPECTED_COMMON_DEV_EPISODES = 2_028
EXPECTED_DIAGNOSTIC_DEV_EPISODES = 358

# Fixed L4 control RNG.
L4_MASK_RNG_SEED = 20_261_109

# Feature contract.
HIDDEN_SIZE = 4_096
FEATURE_DTYPE = "bfloat16"
LEGACY_FEATURE_KIND = "cached_canonical_bf16"
SUPPLEMENTAL_FEATURE_KIND = "supplemental_bf16_v1"
SUPPLEMENTAL_FEATURE_ROOT = "artifacts/fine_label_repair_v1/features/"


class FineLabelRepairError(RuntimeError):
    """Base error for the lane-D repair seam."""


class DataContractError(FineLabelRepairError):
    """A row/identity/split/feature-ref contract was violated."""


class AlignmentRepairError(FineLabelRepairError):
    """A candidate or UTF-8 alignment proof cannot be honoured."""


class LabelRepairError(FineLabelRepairError):
    """A new/changed position cannot be labelled exactly and safely."""


def source_tag_for_scope(scope: str) -> str:
    mapping = {
        "cot": "cot32k",
        "nothink": "nothink_ctx40k",
        "ctx40k": "ctx40k",
        "agentlab": "ramp_ablation",
    }
    try:
        return mapping[str(scope)]
    except KeyError as error:
        raise DataContractError(f"unknown scope: {scope!r}") from error


def assert_protected_split(split: object, *, where: str) -> str:
    """The only raw/vector-visible splits are TRAIN and DEV; TEST is never opened."""
    value = str(split or "")
    if value not in PROTECTED_SPLITS:
        raise DataContractError(f"{where}: refusing split {value!r}; only TRAIN/DEV are allowed")
    return value
