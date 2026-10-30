#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2}"
TARGET_MODEL="${TARGET_MODEL:-/home/model_and_data/yxz/model/Huihui-Qwen3-8B-abliterated-v2}"
ASSISTANT_CONFIDENCE_THRESHOLD="${ASSISTANT_CONFIDENCE_THRESHOLD:-0.4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" sanity_check.py \
  --draft-model "${DRAFT_MODEL}" \
  --target-model "${TARGET_MODEL}" \
  --draft-device cuda:0 \
  --target-device cuda:0 \
  --dtype bfloat16 \
  --assistant-confidence-threshold "${ASSISTANT_CONFIDENCE_THRESHOLD}" \
  --benchmark-warmup "${BENCHMARK_WARMUP:-1}" \
  --benchmark-repeats "${BENCHMARK_REPEATS:-3}" \
  "$@"

