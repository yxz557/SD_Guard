#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-/home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2}"
TARGET_MODEL="${TARGET_MODEL:-/home/model_and_data/yxz/model/Huihui-Qwen3-8B-abliterated-v2}"
GUARD_MODEL="${GUARD_MODEL:-/home/model_and_data/yxz/model/Qwen3Guard-Stream-4B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/matrix_2x3}"

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
SHARED_MANIFEST=""

run_cell() {
  local thinking_name="$1"
  local generation_mode="$2"
  local thinking_flag="$3"
  local output_dir="${OUTPUT_ROOT}/${thinking_name}/${generation_mode}"
  local manifest_args=()

  if [[ -n "${SHARED_MANIFEST}" ]]; then
    manifest_args=(--sample-manifest "${SHARED_MANIFEST}")
  fi

  echo
  echo "=== ${thinking_name} / ${generation_mode} ==="
  "${PYTHON_BIN}" run_experiment.py \
    --generation-mode "${generation_mode}" \
    --dataset-path "${DATASET_PATH}" \
    --draft-model "${DRAFT_MODEL}" \
    --target-model "${TARGET_MODEL}" \
    --guard-model "${GUARD_MODEL}" \
    --output-dir "${output_dir}" \
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
    "${manifest_args[@]}" \
    "${EXTRA_ARGS[@]}"
}

cd "${PROJECT_DIR}"

# The first cell creates the sole random sample manifest. Every other cell must reuse it.
run_cell think_on target_only --enable-thinking
SHARED_MANIFEST="${OUTPUT_ROOT}/think_on/target_only/sample_manifest.json"

run_cell think_on speculative --enable-thinking
run_cell think_on draft_only --enable-thinking
run_cell think_off target_only --no-enable-thinking
run_cell think_off speculative --no-enable-thinking
run_cell think_off draft_only --no-enable-thinking

"${PYTHON_BIN}" summarize_matrix.py \
  --matrix-dir "${OUTPUT_ROOT}" \
  --overwrite

echo
echo "Matrix completed: ${OUTPUT_ROOT}"

