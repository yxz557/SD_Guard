#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

exec python run_matrix.py \
  --dataset-path "${DATASET_PATH:-/home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest}" \
  --draft-model "${DRAFT_MODEL:-/home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2}" \
  --target-model "${TARGET_MODEL:-/home/model_and_data/yxz/model/Huihui-Qwen3-14B-abliterated-v2}" \
  --guard-model "${GUARD_MODEL:-/home/model_and_data/yxz/model/Qwen3Guard-Stream-4B}" \
  --sample-count "${SAMPLE_COUNT:-100}" \
  --seed "${SEED:-42}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-4096}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --num-speculative-tokens "${NUM_SPECULATIVE_TOKENS:-5}" \
  --gpu-id "${GPU_ID:-0}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.72}" \
  --output-dir "${OUTPUT_DIR:-$PROJECT_DIR/outputs/sd5_2x4_100}" \
  "$@"
