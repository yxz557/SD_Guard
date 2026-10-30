#!/usr/bin/env bash

# Run the complete Qwen3-8B Task Injection baseline in sequence:
#   1. clean/no-attack baseline
#   2. direct attack
#   3. injecagent attack
#   4. immutable trace export and hash verification
#
# The vLLM server must already be running. This script intentionally does not
# call the API Guard: Guard evaluation belongs to the independent branch after
# raw trajectories have been frozen.
#
# Safe resume behavior:
#   - The benchmark commands do not use --force-rerun.
#   - Re-running this script skips completed AgentDojo task logs.
#   - Immutable trace export accepts identical artifacts and rejects changes.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTDOJO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(git -C "${AGENTDOJO_ROOT}" rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_LLM_PORT="${LOCAL_LLM_PORT:-8000}"
BENCHMARK_VERSION="${BENCHMARK_VERSION:-v1.2.1}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-/home/model_and_data/yxz/model/Qwen3-8B}"
RUN_TAG="${RUN_TAG:-ctx40k_r1}"

RUN_ROOT="${RUN_ROOT:-${AGENTDOJO_ROOT}/runs}"
CLEAN_LOGDIR="${RUN_ROOT}/qwen3_8b_task_injection_full_no_attack_${RUN_TAG}"
DIRECT_LOGDIR="${RUN_ROOT}/qwen3_8b_task_injection_full_direct_${RUN_TAG}"
INJECAGENT_LOGDIR="${RUN_ROOT}/qwen3_8b_task_injection_full_injecagent_${RUN_TAG}"

RESULT_ROOT="${SDG_RESULT_ROOT:-${PROJECT_ROOT}/results/agentdojo/task_injection_full_${RUN_TAG}}"
TRACE_ROOT="${RESULT_ROOT}/traces"
RUN_ID="qwen3-8b-task-injection-full-${RUN_TAG}"

export LOCAL_LLM_PORT

on_error() {
  local exit_code=$?
  local line_number=$1
  echo >&2
  echo "ERROR: full Task Injection run stopped at line ${line_number} (exit ${exit_code})." >&2
  echo "Completed AgentDojo logs were kept. Fix the cause, then run this same script again to resume." >&2
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

announce "Configuration"
echo "AgentDojo root:       ${AGENTDOJO_ROOT}"
echo "Project root:         ${PROJECT_ROOT}"
echo "Python:               ${PYTHON_BIN}"
echo "Benchmark version:    ${BENCHMARK_VERSION}"
echo "Local vLLM endpoint:  http://127.0.0.1:${LOCAL_LLM_PORT}"
echo "Target model:         ${TARGET_MODEL_ID}"
echo "Run tag:              ${RUN_TAG}"
echo "Result root:          ${RESULT_ROOT}"

announce "Preflight checks"
"${PYTHON_BIN}" -c "import agentdojo; print('agentdojo:', agentdojo.__file__)"
curl --fail --silent --show-error "http://127.0.0.1:${LOCAL_LLM_PORT}/health" >/dev/null
echo "vLLM health check: OK"
curl --fail --silent --show-error "http://127.0.0.1:${LOCAL_LLM_PORT}/v1/models"
echo

"${PYTHON_BIN}" -c "from agentdojo.task_suite.load_suites import get_suites; suites=get_suites('${BENCHMARK_VERSION}'); [print(name, 'user_tasks=', len(suite.user_tasks), 'injection_tasks=', len(suite.injection_tasks), 'pairs=', len(suite.user_tasks)*len(suite.injection_tasks)) for name, suite in suites.items()]; print('clean_total=', sum(len(suite.user_tasks) for suite in suites.values())); print('pairs_per_attack=', sum(len(suite.user_tasks)*len(suite.injection_tasks) for suite in suites.values()))"

run_benchmark() {
  local label=$1
  local logdir=$2
  shift 2

  announce "Running ${label}"
  "${PYTHON_BIN}" -m agentdojo.scripts.benchmark \
    --model VLLM_PARSED \
    --benchmark-version "${BENCHMARK_VERSION}" \
    --logdir "${logdir}" \
    --max-workers 1 \
    "$@"
}

run_benchmark "clean/no-attack baseline" "${CLEAN_LOGDIR}"
run_benchmark "complete direct baseline" "${DIRECT_LOGDIR}" --attack direct
run_benchmark "complete injecagent baseline" "${INJECAGENT_LOGDIR}" --attack injecagent

announce "Freezing immutable raw trajectories"
"${PYTHON_BIN}" -m agentdojo.guard_eval.export_trace \
  --input "${CLEAN_LOGDIR}" "${DIRECT_LOGDIR}" "${INJECAGENT_LOGDIR}" \
  --output "${TRACE_ROOT}" \
  --target-model-id "${TARGET_MODEL_ID}" \
  --benchmark-version "${BENCHMARK_VERSION}" \
  --run-id "${RUN_ID}"

announce "Verifying raw trajectory hashes"
"${PYTHON_BIN}" -m agentdojo.guard_eval.verify --input "${TRACE_ROOT}"

announce "Task Injection run completed"
echo "Frozen traces: ${TRACE_ROOT}"
echo "The API Guard has not been run. Its predictions remain an independent branch."

