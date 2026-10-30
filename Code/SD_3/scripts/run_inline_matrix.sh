#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-/home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2}"
TARGET_MODEL="${TARGET_MODEL:-/home/model_and_data/yxz/model/Huihui-Qwen3-8B-abliterated-v2}"
GUARD_MODEL="${GUARD_MODEL:-/home/model_and_data/yxz/model/Qwen3Guard-Stream-4B}"
MATRIX_DIR="${MATRIX_DIR:-/home/model_and_data/yxz/Code/SD_Guard/SD_3/outputs/abliterated_2x3_80_4096}"

SAMPLE_COUNT="${SAMPLE_COUNT:-80}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
NUM_ASSISTANT_TOKENS="${NUM_ASSISTANT_TOKENS:-5}"
ASSISTANT_CONFIDENCE_THRESHOLD="${ASSISTANT_CONFIDENCE_THRESHOLD:-0.4}"
DTYPE="${DTYPE:-bfloat16}"

TARGET_DEVICE="${TARGET_DEVICE:-cuda:0}"
DRAFT_DEVICE="${DRAFT_DEVICE:-cuda:0}"
GUARD_DEVICE="${GUARD_DEVICE:-cuda:1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

EXTRA_ARGS=("$@")

run_inline_cell() {
  local thinking_name="$1"
  local thinking_flag="$2"
  local reference_dir="${MATRIX_DIR}/${thinking_name}/speculative"
  local output_dir="${MATRIX_DIR}/${thinking_name}/inline_guard"

  echo
  echo "=== online inline Guard: ${thinking_name} ==="
  "${PYTHON_BIN}" run_experiment.py \
    --generation-mode inline_guard \
    --dataset-path "${DATASET_PATH}" \
    --draft-model "${DRAFT_MODEL}" \
    --target-model "${TARGET_MODEL}" \
    --guard-model "${GUARD_MODEL}" \
    --output-dir "${output_dir}" \
    --sample-manifest "${reference_dir}/sample_manifest.json" \
    --reference-experiment-dir "${reference_dir}" \
    --sample-count "${SAMPLE_COUNT}" \
    --seed "${SEED}" \
    "${thinking_flag}" \
    --do-sample \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --top-k "${TOP_K}" \
    --num-assistant-tokens "${NUM_ASSISTANT_TOKENS}" \
    --assistant-confidence-threshold "${ASSISTANT_CONFIDENCE_THRESHOLD}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --target-device "${TARGET_DEVICE}" \
    --draft-device "${DRAFT_DEVICE}" \
    --guard-device "${GUARD_DEVICE}" \
    --dtype "${DTYPE}" \
    "${EXTRA_ARGS[@]}"

  local copied_labels="${output_dir}/blind/judge_labels.jsonl"
  if [[ -f "${copied_labels}" ]]; then
    "${PYTHON_BIN}" analyze_judgments.py \
      --experiment-dir "${output_dir}" \
      --judge-labels "${copied_labels}" \
      --guard-view proposal_immediate \
      --overwrite
  fi
}

cd "${PROJECT_DIR}"
run_inline_cell think_on --enable-thinking
run_inline_cell think_off --no-enable-thinking

"${PYTHON_BIN}" summarize_matrix.py \
  --matrix-dir "${MATRIX_DIR}" \
  --overwrite

echo
echo "2x4 matrix completed: ${MATRIX_DIR}"
