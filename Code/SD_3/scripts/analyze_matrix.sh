#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MATRIX_DIR="${MATRIX_DIR:-${PROJECT_DIR}/outputs/matrix_2x3}"

cd "${PROJECT_DIR}"
for thinking_name in think_on think_off; do
  for generation_mode in target_only speculative draft_only inline_guard; do
    experiment_dir="${MATRIX_DIR}/${thinking_name}/${generation_mode}"
    judge_labels="${experiment_dir}/blind/judge_labels.jsonl"
    if [[ ! -f "${judge_labels}" ]]; then
      echo "Missing blind judgments: ${judge_labels}" >&2
      exit 1
    fi
    "${PYTHON_BIN}" analyze_judgments.py \
      --experiment-dir "${experiment_dir}" \
      --judge-labels "${judge_labels}" \
      "$@"
  done
done

"${PYTHON_BIN}" summarize_matrix.py \
  --matrix-dir "${MATRIX_DIR}" \
  --overwrite

