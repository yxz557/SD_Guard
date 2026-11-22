"""sdsafe-v5-depth constants: read-only upstream inputs, new writes only under this root.

Stage scope (authorised): a **new dataset version** of frozen-prefix residual-stream
captures from the same unmodified Qwen3-8B forward -- after block 16 and after block 24
(HF ``output_hidden_states[16]`` / ``[24]``), at the already accepted endpoint token
indices.  The pre-existing block 4/8 shards are read-only dependencies and are never
overwritten or mixed into the block-16/24 matrices.  Head training uses the same small
MLP recipe; deployment still only reads what the main model's own forward already
computes, so added cost is hook/copy/scaler/head only.
"""

from __future__ import annotations

import os
from pathlib import Path

STUDY_ID = "sdsafe-lowcost-authorization-v5-depth"
ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"

# ---------------------------------------------------------------- read-only roots
# Node migration 2026-11-13: the new GPU node has no canonical upstream copy (its data disk
# is full).  A byte-identical, SHA-verified read-only mirror is staged under /dev/shm and
# selected through SDSAFE_UPSTREAM_MIRROR.  With the variable unset these resolve to the
# exact original absolute paths, so the frozen code is unchanged by default.
_UPSTREAM = Path(os.environ.get("SDSAFE_UPSTREAM_MIRROR", "/root/autodl-tmp"))
SDSAFE = _UPSTREAM / "SD-Safe"
V1 = _UPSTREAM / "sdsafe-lowcost-authorization-v1"
V2 = _UPSTREAM / "sdsafe-lowcost-authorization-v2"
V3 = _UPSTREAM / "sdsafe-lowcost-authorization-v3"

MODEL_PATH = SDSAFE / "models/Qwen3-8B"
SEQ = SDSAFE / "artifacts/fine_sequence_explore_v1/data/sequence_v1"
SEQ_TRAIN_ROWS = SEQ / "rows_train.jsonl"
SEQ_DEV_ROWS = SEQ / "rows_dev.jsonl"
ACCEPTED_FEATURE_BIN = SEQ / "features_bf16.bin"  # accepted final-layer hidden, bf16

INDEX_TRAIN = V1 / "artifacts/index/index_train.npz"
INDEX_DEV = V1 / "artifacts/index/index_dev.npz"
V2_TRAIN_ARRAYS = V2 / "artifacts/tensors/train_pool_arrays.npz"
V2_DEV_ARRAYS = V2 / "artifacts/tensors/dev_common_arrays.npz"
V2_DEV_HIDDEN = V2 / "artifacts/tensors/dev_common_hidden_probe_f32.pt"
V2_TRAIN_HIDDEN = V2 / "artifacts/tensors/train_hidden_probe_f32.pt"
V2_BASELINE = V2 / "artifacts/baseline/baseline_scores_v2.npz"
OLD_BASE_CKPT = SDSAFE / "artifacts/fine_label_repair_v1/runs/balanced-L4-lr3e-4-seed{seed}/checkpoint_best_fn5.pt"

HIDDEN = 4096
DEV_EPISODES = 2_028
DEV_CT_ROWS = 25_003
DEV_POSITIVES = 414
DEV_NEGATIVES = 1_614
DEV_GROUPS = 166
TRAIN_POOL_ROWS = 93_322
N_GROUPS = 616

# The ACCEPTED canonical frozen point: three-frozen-base ensemble, per-seed ALL-CT raw-logit
# episode max, then 3-seed episode mean; common DEV 2028 / 414 P / 1614 N / 166 groups,
# strict_final, threshold is the FROZEN canonical value below.  It gives 21 FN / 78 FP.
BASELINE = {
    "tp": 393, "fn": 21, "fp": 78, "tn": 1_536,
    "auc": 0.9811267951319673, "threshold": -0.6190447211265564,
    "rule": "canonical frozen threshold (accepted artifact); do not re-budget",
}
# The v4 source's quantile-budget replay selected a *different* threshold on the same
# baseline scores (FN 21 / FP 80).  It is a budget-optimal *point*, not the canonical
# frozen point; both are replayed independently in this stage and never conflated.
BASELINE_BUDGET_POINT_NOTE = (
    "baseline budget-optimal threshold replay is reported separately from the canonical "
    "frozen point; the canonical point is never silently re-budgeted"
)
PROGRESS_FN = 14
STRONG_FN = 8
FP_BUDGET = 0.05
MAX_FP = 80
COST_GATE_MS = 1.0

# ---------------------------------------------------------------- capture design
# ---------------------------------------------------------------- depth capture design
# One prefix-24 forward (unless the smoke proves 24 illegal on the longest real shape)
# captures BOTH depth endpoints; the block-8 tap is included only as a cross-check that
# this forward reproduces the already-accepted block-8 values within tolerance.
PREFIX_LAYERS = 24         # number of decoder blocks actually executed
CAPTURE_BLOCKS = (16, 24)  # "after 16 blocks" = HF hidden_states[16]; "after 24" = [24]
CROSSCHECK_BLOCK = 8
DEPTH_BLOCKS = (16, 24)
MODEL_LAYERS = 36
# fixed numerical tolerances for the chunked-prefill equivalence proof (bf16 GPU)
KV_CHUNK_COS_MIN = 0.9995
KV_CHUNK_MAX_ABS = 0.5
FEATURE_DTYPE = "bfloat16"
FOLDS = 4
FOLD_SEED = 20261113
TRAIN_SEEDS = (0, 1, 2)
