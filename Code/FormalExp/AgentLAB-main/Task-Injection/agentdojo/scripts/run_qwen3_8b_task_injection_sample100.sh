#!/usr/bin/env bash

# Full clean baseline plus a fixed sample of 100 attack pairs per suite.
# The same persisted 400-pair manifest is used by direct and injecagent.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTDOJO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(git -C "${AGENTDOJO_ROOT}" rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_LLM_PORT="${LOCAL_LLM_PORT:-8000}"
BENCHMARK_VERSION="${BENCHMARK_VERSION:-v1.2.1}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-/home/model_and_data/yxz/model/Qwen3-8B}"
RUN_TAG="${RUN_TAG:-sample100_ctx40k_r1}"
SAMPLE_SEED="${SAMPLE_SEED:-20261030}"
PAIRS_PER_SUITE=100

RUN_ROOT="${RUN_ROOT:-${AGENTDOJO_ROOT}/runs}"
CLEAN_LOGDIR="${CLEAN_LOGDIR:-${RUN_ROOT}/qwen3_8b_task_injection_full_no_attack_${RUN_TAG}}"
DIRECT_LOGDIR="${DIRECT_LOGDIR:-${RUN_ROOT}/qwen3_8b_task_injection_direct_${RUN_TAG}}"
INJECAGENT_LOGDIR="${INJECAGENT_LOGDIR:-${RUN_ROOT}/qwen3_8b_task_injection_injecagent_${RUN_TAG}}"
RESULT_ROOT="${SDG_RESULT_ROOT:-${PROJECT_ROOT}/results/agentdojo/task_injection_${RUN_TAG}}"
MANIFEST_PATH="${RESULT_ROOT}/sample_manifest.json"
TRACE_ROOT="${RESULT_ROOT}/traces"
RUN_ID="qwen3-8b-task-injection-${RUN_TAG}"

export LOCAL_LLM_PORT

on_error() {
  local exit_code=$?
  echo >&2
  echo "ERROR: sample-100 run stopped at line $1 (exit ${exit_code})." >&2
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

announce "Creating or reusing the one-time 400-pair manifest"
"${PYTHON_BIN}" "${SCRIPT_DIR}/sample_task_injection_pairs.py" \
  --manifest "${MANIFEST_PATH}" \
  --benchmark-version "${BENCHMARK_VERSION}" \
  --pairs-per-suite "${PAIRS_PER_SUITE}" \
  --seed "${SAMPLE_SEED}"
sha256sum "${MANIFEST_PATH}"

announce "Running the complete clean/no-attack baseline"
"${PYTHON_BIN}" -m agentdojo.scripts.benchmark \
  --model VLLM_PARSED \
  --benchmark-version "${BENCHMARK_VERSION}" \
  --logdir "${CLEAN_LOGDIR}" \
  --max-workers 1

run_sampled_attack() {
  local attack_name=$1
  local logdir=$2
  local groups_output

  groups_output="$("${PYTHON_BIN}" "${SCRIPT_DIR}/sample_task_injection_pairs.py" \
    --manifest "${MANIFEST_PATH}" \
    --benchmark-version "${BENCHMARK_VERSION}" \
    --pairs-per-suite "${PAIRS_PER_SUITE}" \
    --seed "${SAMPLE_SEED}" \
    --emit-groups)"

  announce "Running sampled ${attack_name} attack"
  while read -r suite_name user_task_id injection_task_list; do
    [[ -n "${suite_name}" ]] || continue
    injection_args=()
    for injection_task_id in ${injection_task_list}; do
      injection_args+=(--injection-task "${injection_task_id}")
    done
    echo "[${attack_name}] ${suite_name}/${user_task_id}: ${injection_task_list}"
    "${PYTHON_BIN}" -m agentdojo.scripts.benchmark \
      --suite "${suite_name}" \
      --user-task "${user_task_id}" \
      "${injection_args[@]}" \
      --model VLLM_PARSED \
      --benchmark-version "${BENCHMARK_VERSION}" \
      --attack "${attack_name}" \
      --logdir "${logdir}" \
      --max-workers 1
  done <<< "${groups_output}"
}

run_sampled_attack direct "${DIRECT_LOGDIR}"
run_sampled_attack injecagent "${INJECAGENT_LOGDIR}"

announce "Freezing immutable trajectories"
"${PYTHON_BIN}" -m agentdojo.guard_eval.export_trace \
  --input "${CLEAN_LOGDIR}" "${DIRECT_LOGDIR}" "${INJECAGENT_LOGDIR}" \
  --output "${TRACE_ROOT}" \
  --target-model-id "${TARGET_MODEL_ID}" \
  --benchmark-version "${BENCHMARK_VERSION}" \
  --run-id "${RUN_ID}"

announce "Verifying trajectory hashes"
"${PYTHON_BIN}" -m agentdojo.guard_eval.verify --input "${TRACE_ROOT}"

announce "Sample-100 run completed"
echo "Sample manifest: ${MANIFEST_PATH}"
echo "Frozen traces:   ${TRACE_ROOT}"
echo "Direct and injecagent used the same persisted 400 attack pairs."
echo "The API Guard has not been run."

