#!/usr/bin/env bash

# Minimal end-to-end AgentDojo smoke run:
# sample workspace pairs once, run 10 direct-attack pairs, freeze, and verify.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTDOJO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(git -C "${AGENTDOJO_ROOT}" rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_LLM_PORT="${LOCAL_LLM_PORT:-8000}"
BENCHMARK_VERSION="${BENCHMARK_VERSION:-v1.2.1}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-/home/model_and_data/yxz/model/Qwen3-8B}"
RUN_TAG="${RUN_TAG:-workspace_direct_sample10_ctx40k_r1}"
SAMPLE_SEED="${SAMPLE_SEED:-20261030}"
PAIRS_PER_SUITE=10

RUN_ROOT="${RUN_ROOT:-${AGENTDOJO_ROOT}/runs}"
DIRECT_LOGDIR="${DIRECT_LOGDIR:-${RUN_ROOT}/qwen3_8b_${RUN_TAG}}"
RESULT_ROOT="${SDG_RESULT_ROOT:-${PROJECT_ROOT}/results/agentdojo/${RUN_TAG}}"
MANIFEST_PATH="${RESULT_ROOT}/sample_manifest.json"
TRACE_ROOT="${RESULT_ROOT}/traces"
RUN_ID="qwen3-8b-${RUN_TAG}"

export LOCAL_LLM_PORT

on_error() {
  local exit_code=$?
  echo >&2
  echo "ERROR: workspace sample-10 run stopped at line $1 (exit ${exit_code})." >&2
  echo "Existing logs and the one-time sample manifest were kept; rerun this script to resume." >&2
  exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR

announce() {
  echo
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

cd "${AGENTDOJO_ROOT}"

announce "Preflight checks"
"${PYTHON_BIN}" -c "import agentdojo; print('agentdojo:', agentdojo.__file__)"
curl --fail --silent --show-error "http://127.0.0.1:${LOCAL_LLM_PORT}/health" >/dev/null
echo "vLLM health check: OK"

announce "Creating or reusing the one-time workspace 10-pair manifest"
"${PYTHON_BIN}" "${SCRIPT_DIR}/sample_task_injection_pairs.py" \
  --manifest "${MANIFEST_PATH}" \
  --benchmark-version "${BENCHMARK_VERSION}" \
  --pairs-per-suite "${PAIRS_PER_SUITE}" \
  --seed "${SAMPLE_SEED}" \
  --suite workspace
sha256sum "${MANIFEST_PATH}"

groups_output="$("${PYTHON_BIN}" "${SCRIPT_DIR}/sample_task_injection_pairs.py" \
  --manifest "${MANIFEST_PATH}" \
  --benchmark-version "${BENCHMARK_VERSION}" \
  --pairs-per-suite "${PAIRS_PER_SUITE}" \
  --seed "${SAMPLE_SEED}" \
  --suite workspace \
  --emit-groups)"

announce "Running 10 sampled workspace/direct pairs"
while read -r suite_name user_task_id injection_task_list; do
  [[ -n "${suite_name}" ]] || continue
  injection_args=()
  for injection_task_id in ${injection_task_list}; do
    injection_args+=(--injection-task "${injection_task_id}")
  done
  echo "[direct] ${suite_name}/${user_task_id}: ${injection_task_list}"
  "${PYTHON_BIN}" -m agentdojo.scripts.benchmark \
    --suite "${suite_name}" \
    --user-task "${user_task_id}" \
    "${injection_args[@]}" \
    --model VLLM_PARSED \
    --benchmark-version "${BENCHMARK_VERSION}" \
    --attack direct \
    --logdir "${DIRECT_LOGDIR}" \
    --max-workers 1
done <<< "${groups_output}"

announce "Freezing immutable trajectories"
"${PYTHON_BIN}" -m agentdojo.guard_eval.export_trace \
  --input "${DIRECT_LOGDIR}" \
  --output "${TRACE_ROOT}" \
  --target-model-id "${TARGET_MODEL_ID}" \
  --benchmark-version "${BENCHMARK_VERSION}" \
  --run-id "${RUN_ID}"

announce "Verifying trajectory hashes"
"${PYTHON_BIN}" -m agentdojo.guard_eval.verify --input "${TRACE_ROOT}"

announce "Workspace sample-10 direct run completed"
echo "Sample manifest: ${MANIFEST_PATH}"
echo "Frozen traces:   ${TRACE_ROOT}"
echo "The API Guard has not been run."

