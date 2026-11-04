#!/usr/bin/env bash

# Compatibility wrapper. New experiments should call
# run_qwen3_8b_task_injection.sh directly.

set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_qwen3_8b_task_injection.sh" \
  --suite workspace \
  --pairs 10 \
  --attack direct \
  --run-tag "${RUN_TAG:-workspace_direct_sample10_ctx40k_r1}" \
  "$@"

