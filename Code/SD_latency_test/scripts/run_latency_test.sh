#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/model_and_data/yxz/model/Qwen3-0.6B}"
TARGET_MODEL="${TARGET_MODEL:-/home/model_and_data/yxz/model/Qwen3-4B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

OUTPUT_ARGS=()
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_ARGS=(--output-dir "${OUTPUT_DIR}")
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" run_latency_test.py \
  --draft-model "${DRAFT_MODEL}" \
  --target-model "${TARGET_MODEL}" \
  --device cuda:0 \
  --dtype bfloat16 \
  --benchmark-mode both \
  --benchmark-warmup "${BENCHMARK_WARMUP:-1}" \
  --benchmark-repeats "${BENCHMARK_REPEATS:-3}" \
  --num-assistant-tokens 5 \
  --assistant-schedule constant \
  --assistant-confidence-threshold 0.4 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  "${OUTPUT_ARGS[@]}" \
  "$@"
