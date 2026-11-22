"""Lane D sequence data/label contract for fine_sequence_explore_v1.

This package owns only durable, read-only sequence data materialisation:

* per-episode chronological E/C/T sequences built from immutable cached BF16 rows;
* authoritative TRAIN/DEV episode identity and whole-group FIT/CAL/SELECT partitions;
* fixed L0..L4 provenance labels plus explicit prefix-state/onset/local semantics;
* FIT-only and FULL-TRAIN scalers with deterministic label-blind populations;
* a read-only tensor/metadata API (shared BF16 matrix + memmap) for downstream models.

It never forwards Qwen, never trains a model, never calls a historical writer/validator
with side effects, and never opens a TEST raw trace or vector.  All feature reads go
through ``sdsafe_fine_label_repair.features.FeatureStore`` which resolves a reference to
its physical row and re-checks the authoritative split before the vector is opened.
"""

from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = 1
STUDY_ID = "fine_sequence_explore_v1"
DATASET_VERSION = "sequence_v1"

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "artifacts/fine_sequence_explore_v1"
DATA_ROOT = ARTIFACT_ROOT / "data"
HANDOFF_ROOT = ARTIFACT_ROOT / "handoffs"
DEFAULT_DATASET_DIR = DATA_ROOT / DATASET_VERSION

#: Immutable producer roots (read only).
REPAIR_ROOT = ROOT / "artifacts/fine_label_repair_v1"
REPAIR_DATA_ROOT = REPAIR_ROOT / "data"
FINE_CONFIRM_FEATURES_ROOT = ROOT / "artifacts/fine_confirm/features"
REPAIR_FEATURES_ROOT = REPAIR_ROOT / "features"

#: Authoritative split source.
SPLIT_AUTHORITY = "artifacts/all_data_mixed/shared/split"
SPLIT_SEED = 20_261_110

SPLIT_TRAIN = "train"
SPLIT_DEV = "dev"
SPLIT_TEST = "test"
MATERIALIZED_SPLITS = (SPLIT_TRAIN, SPLIT_DEV)
PARTITIONS = ("fit", "cal", "select")
PARTITION_FRACTIONS = {"fit": 0.8, "cal": 0.1, "select": 0.1}

HIDDEN_SIZE = 4_096
FEATURE_DTYPE = "bfloat16"
MATRIX_FILE = "features_bf16.bin"
ROW_BYTES = HIDDEN_SIZE * 2

EVENT_TOOL_CALL = "tool_call"
EVENT_CONTENT_SEGMENT = "content_segment"
EVENT_ROUND_END = "round_end"
CT_EVENT_TYPES = (EVENT_TOOL_CALL, EVENT_CONTENT_SEGMENT)
E_EVENT_TYPES = (EVENT_ROUND_END,)
EVENT_FAMILIES = ("ct", "e")
FAMILY_CT = "ct"
FAMILY_E = "e"

POLICIES = ("L0", "L1", "L2", "L3", "L4")
POLICY_NAMES = {
    "L0": "onset_first_exact_hit",
    "L1": "conservative_prefix_positive_censored",
    "L2": "local_same_turn_prefix_positive",
    "L3": "exact_prefix_state_including_transient",
    "L4": "random_drop_negative_control",
}
LABEL_EXACT = "exact"
LABEL_UNKNOWN = "unknown"

#: Frozen expected populations.  Counts are asserted after actual materialisation.
EXPECTED_TRAIN_EPISODES = 8_229
EXPECTED_DEV_EPISODES = 2_386
EXPECTED_COMMON_DEV_EPISODES = 2_028
EXPECTED_DIAGNOSTIC_DEV_EPISODES = 358
EXPECTED_COMMON_DEV_POSITIVES = 414
EXPECTED_DEV_POSITIVES = 664
EXPECTED_TRAIN_POSITIVES = 1_722
EXPECTED_TRAIN_GROUPS = 616
EXPECTED_DEV_GROUPS = 166
EXPECTED_TRAIN_CT_ROWS = 114_799
EXPECTED_TRAIN_CT_EXACT = 107_294
EXPECTED_TRAIN_CT_UNKNOWN = 7_505
EXPECTED_TRAIN_E_ROWS = 36_662
EXPECTED_DEV_CT_ROWS = 31_913
EXPECTED_DEV_E_ROWS = 10_838
EXPECTED_DEV_ROWS = 42_751
EXPECTED_DEV_CT_EXACT = 29_178
EXPECTED_DEV_CT_UNKNOWN = 2_735

#: Exact accepted policy support (TRAIN CT only).  Unknown rows are counted separately.
EXPECTED_POLICY_COUNTS = {
    "L0": {"positive": 1_722, "negative": 105_572, "masked_exact": 0, "unknown": 7_505},
    "L1": {"positive": 1_722, "negative": 91_511, "masked_exact": 14_061, "unknown": 7_505},
    "L2": {"positive": 2_480, "negative": 91_511, "masked_exact": 13_303, "unknown": 7_505},
    "L3": {"positive": 15_783, "negative": 91_511, "masked_exact": 0, "unknown": 7_505},
    "L4": {"positive": 1_722, "negative": 91_511, "masked_exact": 14_061, "unknown": 7_505},
}
EXPECTED_EXACT_PREFIX_POSITIVE_ROWS = 15_783
EXPECTED_TRANSIENT_PREFIX_POSITIVE_EPISODES = 18

STD_FLOOR = 1e-6
SCALER_POPULATION = "all TRAIN/FIT CT rows including unknown; label-blind, deterministic order"

#: Explicit producer/dependency ledger.  Future consumer modules (T/I) are deliberately
#: absent so adding them cannot invalidate this completed data version.
PRODUCER_FILES = (
    "src/sdsafe_sequence_explore/__init__.py",
    "src/sdsafe_sequence_explore/data.py",
    "src/sdsafe_sequence_explore/labels.py",
    "tests/test_sequence_explore_data.py",
    "tests/test_sequence_explore_labels.py",
)
DEPENDENCY_SOURCE_FILES = (
    "src/sdsafe_fine_label_repair/__init__.py",
    "src/sdsafe_fine_label_repair/data.py",
    "src/sdsafe_fine_label_repair/labels.py",
    "src/sdsafe_fine_label_repair/features.py",
    "src/sdsafe_fine_confirm/__init__.py",
    "src/sdsafe_fine_confirm/sources.py",
    "src/sdsafe_fine_confirm/train_data.py",
    "src/sdsafe_features/common.py",
)
FUTURE_CONSUMER_MODULES = (
    "src/sdsafe_sequence_explore/model.py",
    "src/sdsafe_sequence_explore/objectives.py",
    "src/sdsafe_sequence_explore/train.py",
    "src/sdsafe_sequence_explore/evaluation.py",
    "src/sdsafe_sequence_explore/cli.py",
    "src/sdsafe_sequence_explore/__main__.py",
    "src/sdsafe_sequence_explore/search.py",
    "src/sdsafe_sequence_explore/report.py",
)


class SequenceExploreError(RuntimeError):
    """Base error for the Lane-D sequence seam."""


class DataContractError(SequenceExploreError):
    """A split/identity/feature-ref/data contract was violated."""


class LabelContractError(SequenceExploreError):
    """A label mask/state/semantics contract was violated."""


class DatasetError(SequenceExploreError):
    """A materialised dataset is missing, corrupt, or inconsistent."""


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


def assert_materialized_split(split: object, *, where: str) -> str:
    value = str(split or "")
    if value not in MATERIALIZED_SPLITS:
        raise DataContractError(f"{where}: refusing split {value!r}; only TRAIN/DEV are materialized")
    return value
