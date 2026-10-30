#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRAFT_MODEL="${DRAFT_MODEL:-/home/model_and_data/yxz/model/Qwen3-0.6B}"
TARGET_MODEL="${TARGET_MODEL:-/home/model_and_data/yxz/model/Qwen3-4B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

"${PYTHON_BIN}" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 15), f"vllm==0.31.0 requires Python >=3.10,<3.15; found {sys.version} at {sys.executable}"'

OUTPUT_ARGS=()
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_ARGS=(--output-dir "${OUTPUT_DIR}")
fi

PROFILE_ARGS=()
if [[ "${PROFILE_VLLM:-0}" == "1" ]]; then
  PROFILE_ARGS=(
    --profile-vllm
    --profile-max-new-tokens "${PROFILE_MAX_NEW_TOKENS:-64}"
  )
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" run_vllm_latency_test.py \
  --draft-model "${DRAFT_MODEL}" \
  --target-model "${TARGET_MODEL}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 2048 \
  --benchmark-mode greedy \
  --benchmark-warmup "${BENCHMARK_WARMUP:-2}" \
  --benchmark-repeats "${BENCHMARK_REPEATS:-10}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-new-tokens 256 \
  --num-speculative-tokens 5 \
  --draft-sample-method greedy \
  --no-enable-prefix-caching \
  --no-enforce-eager \
  "${PROFILE_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}" \
  "$@"

