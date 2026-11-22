"""sdsafe-v4 constants: read-only upstream inputs, new writes only under this study root.

Stage scope (authorised): offline *prefix* forwards of the original frozen Qwen3-8B
backbone (embedding + RoPE + first K decoder layers only) to obtain earlier-layer
activations at the **already accepted** per-round endpoint token indices, then a small
head trained on those activations.  No second backbone pass may be added at deployment
time: the captured tensors are advertised as what the main model's own forward already
computes at layer K, and the added deployment cost is only hook/copy/projection/head.
"""

from __future__ import annotations

import os
from pathlib import Path

STUDY_ID = "sdsafe-lowcost-authorization-v4"
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

BASELINE = {
    "tp": 393, "fn": 21, "fp": 78, "tn": 1_536,
    "auc": 0.9811267951319673, "threshold": -0.6190447211265564,
}
PROGRESS_FN = 14
STRONG_FN = 8
FP_BUDGET = 0.05
MAX_FP = 80
COST_GATE_MS = 1.0

# ---------------------------------------------------------------- capture design
PREFIX_LAYERS = 8          # number of decoder blocks actually executed
CAPTURE_BLOCKS = (4, 8)    # "after 4 blocks" = HF hidden_states[4]; "after 8" = [8]
FEATURE_DTYPE = "bfloat16"
FOLDS = 4
FOLD_SEED = 20261113
TRAIN_SEEDS = (0, 1, 2)
