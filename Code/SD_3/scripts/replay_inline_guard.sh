#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MATRIX_DIR="${MATRIX_DIR:-/home/model_and_data/yxz/Code/SD_Guard/SD_3/outputs/abliterated_2x3_80_4096}"
GUARD_DEVICE="${GUARD_DEVICE:-cuda:1}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"

EXTRA_ARGS=("$@")
LIMIT_ARGS=()
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  LIMIT_ARGS=(--sample-limit "${SAMPLE_LIMIT}")
fi

cd "${PROJECT_DIR}"
for thinking_name in think_on think_off; do
  source_dir="${MATRIX_DIR}/${thinking_name}/speculative"
  output_dir="${MATRIX_DIR}/${thinking_name}/inline_replay"
  echo
  echo "=== offline replay: ${thinking_name} ==="
  "${PYTHON_BIN}" replay_inline_guard.py \
    --speculative-experiment-dir "${source_dir}" \
    --output-dir "${output_dir}" \
    --guard-device "${GUARD_DEVICE}" \
    "${LIMIT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
done

echo
echo "Offline replay completed under: ${MATRIX_DIR}"
