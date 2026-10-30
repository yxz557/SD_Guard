#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MATRIX_DIR="${MATRIX_DIR:-$PROJECT_DIR/outputs/sd5_2x4_100}"
cd "$PROJECT_DIR"

"$PYTHON_BIN" sync_inline_labels.py --matrix-dir "$MATRIX_DIR" "$@"

for thinking in think_on think_off; do
  for mode in target_only speculative draft_only inline_guard; do
    cell="$MATRIX_DIR/$thinking/$mode"
    labels="$cell/blind/judge_labels.jsonl"
    if [[ ! -f "$labels" ]]; then
      echo "Missing blind judgments: $labels" >&2
      exit 1
    fi
    "$PYTHON_BIN" analyze_judgments.py \
      --experiment-dir "$cell" \
      --judge-labels "$labels" \
      --overwrite
  done
done

"$PYTHON_BIN" summarize_matrix.py --matrix-dir "$MATRIX_DIR" --overwrite
