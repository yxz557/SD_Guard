#!/usr/bin/env bash

# Unified Qwen3-8B AgentDojo Task Injection runner.
#
# A stable --log-root is a resumable campaign. AgentDojo stores each result at
# <log-root>/<pipeline>/<suite>/<user-task>/<attack>/<injection-task>.json and
# skips a valid existing result unless --force-rerun is explicitly requested.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTDOJO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(git -C "${AGENTDOJO_ROOT}" rev-parse --show-toplevel)"
INVOCATION_DIR="$(pwd -P)"

SUITE_CHOICES=(workspace slack travel banking)
ATTACK_CHOICES=(direct ignore_previous injecagent important_instructions tool_knowledge)

PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_LLM_PORT="${LOCAL_LLM_PORT:-8000}"
BENCHMARK_VERSION="${BENCHMARK_VERSION:-v1.2.1}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-/home/model_and_data/yxz/model/Qwen3-8B}"
RUN_TAG="${RUN_TAG:-ctx40k_r1}"
SAMPLE_SEED="${SAMPLE_SEED:-20261030}"
WORKERS="${TASK_INJECTION_WORKERS:-1}"

SUITE_SPEC=""
PAIR_SPEC=""
ATTACK_SPEC=""
LOG_ROOT_ARG=""
RESULT_ROOT_ARG=""
MANIFEST_PATH=""
RUN_ID_ARG=""
ROLLOUT_ID_ARG=""
INCLUDE_CLEAN=false
SKIP_EXPORT=false
FORCE_RERUN=false

usage() {
  cat <<'EOF'
Usage:
  run_qwen3_8b_task_injection.sh \
    --suite <suite|full|csv> \
    --pairs <positive-integer|full> \
    --attack <attack|full|csv> [options]

Required experiment selectors:
  --suite VALUE       workspace, slack, travel, banking, full, or a comma list.
  --pairs VALUE       Sampled user/injection pairs PER SUITE, or full.
  --attack VALUE      direct, ignore_previous, injecagent,
                      important_instructions, tool_knowledge, full, or a comma list.

Reproducibility and storage:
  --seed N            Sampling seed (default: 20261030).
  --log-root PATH     Intermediate logs. Reuse this exact path to resume.
  --result-root PATH  Final result root; frozen traces go under PATH/traces.
  --manifest PATH     Sample manifest path. Numeric --pairs only. The default is
                      result-root/manifests/pairs_<N>_seed_<seed>.json.
  --run-id ID         Stable provenance id used by frozen traces.
  --rollout-id ID     Repeated-rollout identity; makes sample ids unique across trials.
  --run-tag TAG       Changes default paths and the default run id.
  --workers N         Concurrent independent benchmark jobs (default: 1).

Runtime overrides:
  --python-bin PATH
  --port N
  --benchmark-version VERSION
  --target-model-id PATH
  --include-clean     Also run the complete clean baseline for selected suites.
  --skip-export       Keep intermediate logs only; do not freeze/verify yet.
  --force-rerun       Intentionally rerun completed pairs instead of resuming.
  -h, --help

Resume contract:
  Keep --log-root, model, benchmark version, seed, and run identity unchanged.
  Complete JSON results are skipped at pair granularity. Missing, truncated, or
  invalid results are rerun.

Examples:
  # Two smoke pairs per suite for the three attacks not previously exercised.
  ./scripts/run_qwen3_8b_task_injection.sh \
    --suite full --pairs 2 \
    --attack ignore_previous,important_instructions,tool_knowledge \
    --workers 2

  # All pairs in all suites for the two high-yield attacks.
  ./scripts/run_qwen3_8b_task_injection.sh \
    --suite full --pairs full \
    --attack important_instructions,tool_knowledge \
    --workers 4

  # Later fill in the other three attacks in the same campaign.
  ./scripts/run_qwen3_8b_task_injection.sh \
    --suite full --pairs full \
    --attack direct,ignore_previous,injecagent
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_value() {
  local option=$1
  local remaining=$2
  (( remaining >= 2 )) || die "${option} requires a value"
}

while (( $# > 0 )); do
  case "$1" in
    --suite)
      require_value "$1" "$#"
      SUITE_SPEC=$2
      shift 2
      ;;
    --pairs)
      require_value "$1" "$#"
      PAIR_SPEC=$2
      shift 2
      ;;
    --attack)
      require_value "$1" "$#"
      ATTACK_SPEC=$2
      shift 2
      ;;
    --seed)
      require_value "$1" "$#"
      SAMPLE_SEED=$2
      shift 2
      ;;
    --log-root)
      require_value "$1" "$#"
      LOG_ROOT_ARG=$2
      shift 2
      ;;
    --result-root)
      require_value "$1" "$#"
      RESULT_ROOT_ARG=$2
      shift 2
      ;;
    --manifest)
      require_value "$1" "$#"
      MANIFEST_PATH=$2
      shift 2
      ;;
    --run-id)
      require_value "$1" "$#"
      RUN_ID_ARG=$2
      shift 2
      ;;
    --rollout-id)
      require_value "$1" "$#"
      ROLLOUT_ID_ARG=$2
      shift 2
      ;;
    --run-tag)
      require_value "$1" "$#"
      RUN_TAG=$2
      shift 2
      ;;
    --workers)
      require_value "$1" "$#"
      WORKERS=$2
      shift 2
      ;;
    --python-bin)
      require_value "$1" "$#"
      PYTHON_BIN=$2
      shift 2
      ;;
    --port)
      require_value "$1" "$#"
      LOCAL_LLM_PORT=$2
      shift 2
      ;;
    --benchmark-version)
      require_value "$1" "$#"
      BENCHMARK_VERSION=$2
      shift 2
      ;;
    --target-model-id)
      require_value "$1" "$#"
      TARGET_MODEL_ID=$2
      shift 2
      ;;
    --include-clean)
      INCLUDE_CLEAN=true
      shift
      ;;
    --skip-export)
      SKIP_EXPORT=true
      shift
      ;;
    --force-rerun)
      FORCE_RERUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1 (use --help)"
      ;;
  esac
done

[[ -n "${SUITE_SPEC}" ]] || die "--suite is required"
[[ -n "${PAIR_SPEC}" ]] || die "--pairs is required"
[[ -n "${ATTACK_SPEC}" ]] || die "--attack is required"
[[ "${SAMPLE_SEED}" =~ ^-?[0-9]+$ ]] || die "--seed must be an integer"
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || die "--workers must be a positive integer"
if (( WORKERS > 1 )) && (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
  die "--workers greater than 1 requires Bash 5.1 or newer"
fi
if [[ "${PAIR_SPEC}" != "full" && ! "${PAIR_SPEC}" =~ ^[1-9][0-9]*$ ]]; then
  die "--pairs must be a positive integer or full"
fi
if [[ "${PAIR_SPEC}" == "full" && -n "${MANIFEST_PATH}" ]]; then
  die "--manifest is only valid with numeric --pairs"
fi

contains_value() {
  local requested=$1
  shift
  local candidate
  for candidate in "$@"; do
    [[ "${requested}" == "${candidate}" ]] && return 0
  done
  return 1
}

expand_suites() {
  local value
  local raw_values=()
  IFS=',' read -r -a raw_values <<< "${SUITE_SPEC}"
  if contains_value full "${raw_values[@]}"; then
    (( ${#raw_values[@]} == 1 )) || die "suite value full cannot be combined with other suites"
    SELECTED_SUITES=("${SUITE_CHOICES[@]}")
    return
  fi
  SELECTED_SUITES=()
  for value in "${raw_values[@]}"; do
    [[ -n "${value}" ]] || die "empty suite in --suite"
    contains_value "${value}" "${SUITE_CHOICES[@]}" || die "unknown suite: ${value}"
    contains_value "${value}" "${SELECTED_SUITES[@]}" || SELECTED_SUITES+=("${value}")
  done
}

expand_attacks() {
  local value
  local raw_values=()
  IFS=',' read -r -a raw_values <<< "${ATTACK_SPEC}"
  if contains_value full "${raw_values[@]}"; then
    (( ${#raw_values[@]} == 1 )) || die "attack value full cannot be combined with other attacks"
    SELECTED_ATTACKS=("${ATTACK_CHOICES[@]}")
    return
  fi
  SELECTED_ATTACKS=()
  for value in "${raw_values[@]}"; do
    [[ -n "${value}" ]] || die "empty attack in --attack"
    contains_value "${value}" "${ATTACK_CHOICES[@]}" || die "unknown attack: ${value}"
    contains_value "${value}" "${SELECTED_ATTACKS[@]}" || SELECTED_ATTACKS+=("${value}")
  done
}

expand_suites
expand_attacks

absolute_from_invocation() {
  local path=$1
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s/%s\n' "${INVOCATION_DIR}" "${path}"
  fi
}

RUN_BASE="${RUN_ROOT:-${AGENTDOJO_ROOT}/runs}"
LOG_ROOT="${LOG_ROOT_ARG:-${TASK_INJECTION_LOG_ROOT:-${RUN_BASE}/qwen3_8b_task_injection_${RUN_TAG}}}"
RESULT_ROOT="${RESULT_ROOT_ARG:-${SDG_RESULT_ROOT:-${PROJECT_ROOT}/results/agentdojo/task_injection_${RUN_TAG}}}"
LOG_ROOT="$(absolute_from_invocation "${LOG_ROOT}")"
RESULT_ROOT="$(absolute_from_invocation "${RESULT_ROOT}")"
TRACE_ROOT="${RESULT_ROOT}/traces"
RUN_ID="${RUN_ID_ARG:-qwen3-8b-task-injection-${RUN_TAG}}"
if [[ "${PAIR_SPEC}" != "full" && -z "${MANIFEST_PATH}" ]]; then
  MANIFEST_PATH="${RESULT_ROOT}/manifests/pairs_${PAIR_SPEC}_seed_${SAMPLE_SEED}.json"
elif [[ -n "${MANIFEST_PATH}" ]]; then
  MANIFEST_PATH="$(absolute_from_invocation "${MANIFEST_PATH}")"
fi
CAMPAIGN_CONFIG="${LOG_ROOT}/campaign_config.json"

ACTIVE_JOB_COUNT=0
declare -A ACTIVE_JOB_LABELS=()

export LOCAL_LLM_PORT

on_error() {
  local exit_code=$?
  local line_number=$1
  if declare -F stop_running_jobs >/dev/null; then
    stop_running_jobs
  fi
  echo >&2
  echo "ERROR: Task Injection run stopped at line ${line_number} (exit ${exit_code})." >&2
  echo "Completed pair logs were kept under: ${LOG_ROOT}" >&2
  echo "Fix the cause and rerun with the same --log-root to resume." >&2
  exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR

on_signal() {
  echo >&2
  echo "Interrupted; stopping active benchmark jobs..." >&2
  stop_running_jobs
  exit 130
}
trap on_signal INT TERM

announce() {
  echo
  echo "======================================================================"
  echo "$1"
  echo "======================================================================"
}

stop_running_jobs() {
  local pid
  for pid in "${!ACTIVE_JOB_LABELS[@]}"; do
    # The background shell normally has the Python benchmark as its direct
    # child. Stop that child first so Ctrl+C or a failed sibling cannot leave
    # an orphan benchmark process behind.
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${!ACTIVE_JOB_LABELS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  ACTIVE_JOB_LABELS=()
  ACTIVE_JOB_COUNT=0
}

reap_one_job() {
  local completed_pid=""
  local status=0
  local label=""

  if wait -n -p completed_pid; then
    status=0
  else
    status=$?
  fi
  label="${ACTIVE_JOB_LABELS[${completed_pid}]:-unknown}"
  unset 'ACTIVE_JOB_LABELS['"${completed_pid}"']'
  ACTIVE_JOB_COUNT=$((ACTIVE_JOB_COUNT - 1))

  if (( status != 0 )); then
    echo "ERROR: parallel job failed (${label}, pid=${completed_pid}, exit=${status})." >&2
    return "${status}"
  fi
  echo "Completed job: ${label} (pid=${completed_pid})"
}

submit_job() {
  local label=$1
  shift

  if (( WORKERS == 1 )); then
    "$@"
    return
  fi

  while (( ACTIVE_JOB_COUNT >= WORKERS )); do
    if ! reap_one_job; then
      stop_running_jobs
      return 1
    fi
  done

  (
    trap - ERR INT TERM
    "$@"
  ) &
  local pid=$!
  ACTIVE_JOB_LABELS["${pid}"]="${label}"
  (( ACTIVE_JOB_COUNT += 1 ))
  echo "Started job: ${label} (pid=${pid}, active=${ACTIVE_JOB_COUNT}/${WORKERS})"
}

wait_for_all_jobs() {
  while (( ACTIVE_JOB_COUNT > 0 )); do
    if ! reap_one_job; then
      stop_running_jobs
      return 1
    fi
  done
}

suite_is_selected() {
  contains_value "$1" "${SELECTED_SUITES[@]}"
}

ensure_campaign_config() {
  "${PYTHON_BIN}" - \
    "${CAMPAIGN_CONFIG}" \
    "${BENCHMARK_VERSION}" \
    "${TARGET_MODEL_ID}" \
    "${SAMPLE_SEED}" \
    "${RUN_ID}" \
    "${ROLLOUT_ID_ARG}" \
    "${AGENTDOJO_LLM_TEMPERATURE:-}" \
    "${AGENTDOJO_LLM_TOP_P:-}" \
    "${AGENTDOJO_LLM_TOP_K:-}" \
    "${AGENTDOJO_LLM_MAX_TOKENS:-}" \
    "${AGENTDOJO_GENERATION_SEED:-}" \
    "${AGENTDOJO_ENABLE_THINKING:-}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
generation_config = {
    key: value
    for key, value in zip(
        ("temperature", "top_p", "top_k", "max_tokens", "generation_seed", "enable_thinking"),
        sys.argv[7:13],
        strict=True,
    )
    if value != ""
}
expected = {
    "schema_version": "sd_guard.task_injection_campaign.v1",
    "pipeline_model": "VLLM_PARSED",
    "benchmark_version": sys.argv[2],
    "target_model_id": sys.argv[3],
    "sample_seed": int(sys.argv[4]),
    "run_id": sys.argv[5],
}
if sys.argv[6] or generation_config:
    expected.update(
        {
            "schema_version": "sd_guard.task_injection_campaign.v2",
            "rollout_id": sys.argv[6] or None,
            "generation_config": generation_config,
        }
    )
if path.exists():
    with path.open(encoding="utf-8") as file:
        actual = json.load(file)
    if actual != expected:
        raise SystemExit(
            "Existing log root belongs to an incompatible campaign.\n"
            f"path: {path}\nexpected: {expected}\nactual:   {actual}"
        )
    print(f"Reusing compatible campaign config: {path}")
else:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(expected, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Created campaign config: {path}")
PY
}

benchmark_base_args() {
  BENCHMARK_ARGS=(
    --model VLLM_PARSED
    --benchmark-version "${BENCHMARK_VERSION}"
    --logdir "${LOG_ROOT}"
    --max-workers 1
  )
  if [[ "${FORCE_RERUN}" == true ]]; then
    BENCHMARK_ARGS+=(--force-rerun)
  fi
}

run_clean_suite() {
  local suite_name=$1
  announce "Clean baseline: ${suite_name}"
  benchmark_base_args
  "${PYTHON_BIN}" -m agentdojo.scripts.benchmark \
    --suite "${suite_name}" \
    "${BENCHMARK_ARGS[@]}"
}

run_full_attack_suite() {
  local suite_name=$1
  local attack_name=$2
  announce "Full attack: ${attack_name} / ${suite_name}"
  benchmark_base_args
  "${PYTHON_BIN}" -m agentdojo.scripts.benchmark \
    --suite "${suite_name}" \
    --attack "${attack_name}" \
    "${BENCHMARK_ARGS[@]}"
}

run_sampled_attack_group() {
  local attack_name=$1
  local suite_name=$2
  local user_task_id=$3
  local injection_task_list=$4
  local injection_task_id
  local injection_args=()

  for injection_task_id in ${injection_task_list}; do
    injection_args+=(--injection-task "${injection_task_id}")
  done
  echo "[${attack_name}] ${suite_name}/${user_task_id}: ${injection_task_list}"
  benchmark_base_args
  "${PYTHON_BIN}" -m agentdojo.scripts.benchmark \
    --suite "${suite_name}" \
    --user-task "${user_task_id}" \
    "${injection_args[@]}" \
    --attack "${attack_name}" \
    "${BENCHMARK_ARGS[@]}"
}

schedule_sampled_attack() {
  local attack_name=$1
  local suite_name user_task_id injection_task_list

  announce "Sampled attack: ${attack_name} / ${PAIR_SPEC} pair(s) per selected suite"
  while read -r suite_name user_task_id injection_task_list; do
    [[ -n "${suite_name}" ]] || continue
    suite_is_selected "${suite_name}" || continue
    submit_job \
      "${attack_name}/${suite_name}/${user_task_id}/${injection_task_list}" \
      run_sampled_attack_group \
      "${attack_name}" "${suite_name}" "${user_task_id}" "${injection_task_list}"
  done <<< "${GROUPS_OUTPUT}"
}

cd "${AGENTDOJO_ROOT}"

announce "Configuration"
echo "AgentDojo root:       ${AGENTDOJO_ROOT}"
echo "Python:               ${PYTHON_BIN}"
echo "Benchmark version:    ${BENCHMARK_VERSION}"
echo "Local vLLM endpoint:  http://127.0.0.1:${LOCAL_LLM_PORT}"
echo "Target model:         ${TARGET_MODEL_ID}"
echo "Suites:               ${SELECTED_SUITES[*]}"
echo "Pairs per suite:      ${PAIR_SPEC}"
echo "Attacks:              ${SELECTED_ATTACKS[*]}"
echo "Sample seed:          ${SAMPLE_SEED}"
echo "Concurrent jobs:      ${WORKERS}"
echo "Intermediate logs:    ${LOG_ROOT}"
echo "Result root:          ${RESULT_ROOT}"
echo "Run id:               ${RUN_ID}"
echo "Rollout id:           ${ROLLOUT_ID_ARG:-<none>}"
echo "Generation config:    temperature=${AGENTDOJO_LLM_TEMPERATURE:-<server-default>} top_p=${AGENTDOJO_LLM_TOP_P:-<server-default>} top_k=${AGENTDOJO_LLM_TOP_K:-<server-default>} max_tokens=${AGENTDOJO_LLM_MAX_TOKENS:-<server-default>} seed=${AGENTDOJO_GENERATION_SEED:-<server-default>} thinking=${AGENTDOJO_ENABLE_THINKING:-<template-default>}"
echo "Include clean:        ${INCLUDE_CLEAN}"
echo "Force rerun:          ${FORCE_RERUN}"

announce "Preflight checks"
"${PYTHON_BIN}" -c "import agentdojo; print('agentdojo:', agentdojo.__file__)"
ensure_campaign_config
curl --fail --silent --show-error "http://127.0.0.1:${LOCAL_LLM_PORT}/health" >/dev/null
echo "vLLM health check: OK"
"${PYTHON_BIN}" -c "from agentdojo.task_suite.load_suites import get_suites; suites=get_suites('${BENCHMARK_VERSION}'); [print(name, 'user_tasks=', len(suite.user_tasks), 'injection_tasks=', len(suite.injection_tasks), 'pairs=', len(suite.user_tasks)*len(suite.injection_tasks)) for name, suite in suites.items()]"

GROUPS_OUTPUT=""
if [[ "${PAIR_SPEC}" != "full" ]]; then
  announce "Creating or reusing deterministic sample manifest"
  # Always persist all four suites. A workspace-only run can later expand to
  # --suite full without changing the pairs already chosen for workspace.
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sample_task_injection_pairs.py" \
    --manifest "${MANIFEST_PATH}" \
    --benchmark-version "${BENCHMARK_VERSION}" \
    --pairs-per-suite "${PAIR_SPEC}" \
    --seed "${SAMPLE_SEED}"
  sha256sum "${MANIFEST_PATH}"
  GROUPS_OUTPUT="$("${PYTHON_BIN}" "${SCRIPT_DIR}/sample_task_injection_pairs.py" \
    --manifest "${MANIFEST_PATH}" \
    --benchmark-version "${BENCHMARK_VERSION}" \
    --pairs-per-suite "${PAIR_SPEC}" \
    --seed "${SAMPLE_SEED}" \
    --emit-groups)"
fi

if [[ "${INCLUDE_CLEAN}" == true ]]; then
  for suite_name in "${SELECTED_SUITES[@]}"; do
    submit_job "clean/${suite_name}" run_clean_suite "${suite_name}"
  done
  wait_for_all_jobs
fi

for attack_name in "${SELECTED_ATTACKS[@]}"; do
  if [[ "${PAIR_SPEC}" == "full" ]]; then
    for suite_name in "${SELECTED_SUITES[@]}"; do
      submit_job \
        "${attack_name}/${suite_name}" \
        run_full_attack_suite "${suite_name}" "${attack_name}"
    done
  else
    schedule_sampled_attack "${attack_name}"
  fi
done
wait_for_all_jobs

if [[ "${SKIP_EXPORT}" == false ]]; then
  announce "Freezing all completed campaign trajectories"
  EXPORT_ARGS=(
    --input "${LOG_ROOT}"
    --output "${TRACE_ROOT}"
    --target-model-id "${TARGET_MODEL_ID}"
    --benchmark-version "${BENCHMARK_VERSION}"
    --run-id "${RUN_ID}"
    --seed "${SAMPLE_SEED}"
  )
  if [[ -n "${ROLLOUT_ID_ARG}" ]]; then
    EXPORT_ARGS+=(--rollout-id "${ROLLOUT_ID_ARG}")
  fi
  "${PYTHON_BIN}" -m agentdojo.guard_eval.export_trace "${EXPORT_ARGS[@]}"

  announce "Verifying frozen trajectory hashes"
  "${PYTHON_BIN}" -m agentdojo.guard_eval.verify --input "${TRACE_ROOT}"
fi

announce "Task Injection run completed"
echo "Intermediate logs: ${LOG_ROOT}"
if [[ "${PAIR_SPEC}" != "full" ]]; then
  echo "Sample manifest:   ${MANIFEST_PATH}"
fi
if [[ "${SKIP_EXPORT}" == false ]]; then
  echo "Frozen traces:     ${TRACE_ROOT}"
else
  echo "Trace export:      skipped by request"
fi
echo "Reusing the same --log-root skips complete pairs and continues missing pairs."

