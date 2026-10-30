#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-/home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/model_and_data/yxz/model/Qwen3-0.6B}"
TARGET_MODEL="${TARGET_MODEL:-/home/model_and_data/yxz/model/Qwen3-4B}"
GUARD_MODEL="${GUARD_MODEL:-/home/model_and_data/yxz/model/Qwen3Guard-Stream-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/style_control_first_100}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" run_experiment.py \
  --dataset-path "${DATASET_PATH}" \
  --draft-model "${DRAFT_MODEL}" \
  --target-model "${TARGET_MODEL}" \
  --guard-model "${GUARD_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-count 100 \
  --seed 42 \
  --enable-thinking \
  --style-control \
  --do-sample \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --num-assistant-tokens 5 \
  --max-new-tokens 2048 \
  --target-device cuda:0 \
  --draft-device cuda:0 \
  --guard-device cuda:1 \
  --dtype bfloat16 \
  "$@"

