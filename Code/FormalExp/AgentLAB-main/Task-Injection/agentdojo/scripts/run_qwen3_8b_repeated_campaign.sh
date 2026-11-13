#!/usr/bin/env bash

# Run the same selected AgentDojo task pairs multiple times without overwriting
# earlier rollouts, then build a clean suite/pair/attack/positive|negative view.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTDOJO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_LLM_PORT="${LOCAL_LLM_PORT:-8000}"
TARGET_MODEL_ID="${TARGET_MODEL_ID:-/root/autodl-tmp/models/Qwen3-8B}"
WORKERS="${TASK_INJECTION_WORKERS:-20}"
REPLICATES=3
START_REPLICATE=1
BASE_GENERATION_SEED=20261113
ADAPTIVE_ROUNDS=2
ADAPTIVE_START_ROUND=1
RUN_TAG="qwen3-8b-task-injection-think-repeat"
MANIFEST_PATH=""
CAMPAIGN_ROOT=""
DRY_RUN=false
SKIP_CURATION=false
REQUIRE_THINKING=true

usage() {
  cat <<'EOF'
Usage:
  run_qwen3_8b_repeated_campaign.sh \
    --manifest PATH --campaign-root PATH [options]

Required:
  --manifest PATH          Pair manifest made by build_repeat_pair_manifest.py.
  --campaign-root PATH     Root for raw replicates and the curated dataset view.

Options:
  --replicates N           Number of rollout rounds (default: 3).
  --start-replicate N      First replicate number, for extending a campaign.
  --generation-seed N      Base request seed; replicate N uses base + N - 1.
  --workers N              Parallel AgentDojo jobs (default: 20).
  --adaptive-rounds N      Missing-label-only rounds after the full rounds (default: 2).
  --adaptive-start-round N First adaptive round id; use a larger value to extend later.
  --python-bin PATH
  --port N
  --target-model-id PATH
  --run-tag TAG
  --allow-missing-thinking Do not reject traces lacking saved reasoning blocks.
  --skip-curation          Run and export raw rollouts only.
  --dry-run                Validate inputs and print planned paths without running.
  -h, --help

Locked paper-aligned generation settings:
  temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096 per step,
  enable_thinking=true. Every replicate uses its own request seed and directories.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_value() {
  (( $2 >= 2 )) || die "$1 requires a value"
}

while (( $# > 0 )); do
  case "$1" in
    --manifest) require_value "$1" "$#"; MANIFEST_PATH=$2; shift 2 ;;
    --campaign-root) require_value "$1" "$#"; CAMPAIGN_ROOT=$2; shift 2 ;;
    --replicates) require_value "$1" "$#"; REPLICATES=$2; shift 2 ;;
    --start-replicate) require_value "$1" "$#"; START_REPLICATE=$2; shift 2 ;;
    --generation-seed) require_value "$1" "$#"; BASE_GENERATION_SEED=$2; shift 2 ;;
    --workers) require_value "$1" "$#"; WORKERS=$2; shift 2 ;;
    --adaptive-rounds) require_value "$1" "$#"; ADAPTIVE_ROUNDS=$2; shift 2 ;;
    --adaptive-start-round) require_value "$1" "$#"; ADAPTIVE_START_ROUND=$2; shift 2 ;;
    --python-bin) require_value "$1" "$#"; PYTHON_BIN=$2; shift 2 ;;
    --port) require_value "$1" "$#"; LOCAL_LLM_PORT=$2; shift 2 ;;
    --target-model-id) require_value "$1" "$#"; TARGET_MODEL_ID=$2; shift 2 ;;
    --run-tag) require_value "$1" "$#"; RUN_TAG=$2; shift 2 ;;
    --allow-missing-thinking) REQUIRE_THINKING=false; shift ;;
    --skip-curation) SKIP_CURATION=true; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ -n "${MANIFEST_PATH}" ]] || die "--manifest is required"
[[ -n "${CAMPAIGN_ROOT}" ]] || die "--campaign-root is required"
[[ -f "${MANIFEST_PATH}" ]] || die "manifest not found: ${MANIFEST_PATH}"
[[ "${REPLICATES}" =~ ^[1-9][0-9]*$ ]] || die "--replicates must be positive"
[[ "${START_REPLICATE}" =~ ^[1-9][0-9]*$ ]] || die "--start-replicate must be positive"
[[ "${BASE_GENERATION_SEED}" =~ ^-?[0-9]+$ ]] || die "--generation-seed must be an integer"
[[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]] || die "--workers must be positive"
[[ "${ADAPTIVE_ROUNDS}" =~ ^[0-9]+$ ]] || die "--adaptive-rounds must be a non-negative integer"
[[ "${ADAPTIVE_START_ROUND}" =~ ^[1-9][0-9]*$ ]] || die "--adaptive-start-round must be positive"
if [[ "${SKIP_CURATION}" == true && "${ADAPTIVE_ROUNDS}" != "0" ]]; then
  die "--skip-curation requires --adaptive-rounds 0"
fi

MANIFEST_PATH="$(cd -- "$(dirname -- "${MANIFEST_PATH}")" && pwd)/$(basename -- "${MANIFEST_PATH}")"
mkdir -p "${CAMPAIGN_ROOT}"
CAMPAIGN_ROOT="$(cd -- "${CAMPAIGN_ROOT}" && pwd)"

PAIR_COUNT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["pairs_per_suite"])' "${MANIFEST_PATH}")"
PAIR_SEED="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["seed"])' "${MANIFEST_PATH}")"
BENCHMARK_VERSION="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["benchmark_version"])' "${MANIFEST_PATH}")"

echo "Repeated AgentDojo campaign"
echo "  manifest:          ${MANIFEST_PATH}"
echo "  campaign root:     ${CAMPAIGN_ROOT}"
echo "  pairs per suite:   ${PAIR_COUNT}"
echo "  replicate range:   ${START_REPLICATE}..$((START_REPLICATE + REPLICATES - 1))"
echo "  workers:           ${WORKERS}"
echo "  adaptive rounds:   ${ADAPTIVE_ROUNDS} (starting at ${ADAPTIVE_START_ROUND})"
echo "  require thinking:  ${REQUIRE_THINKING}"

for (( offset=0; offset<REPLICATES; offset++ )); do
  replicate_number=$((START_REPLICATE + offset))
  replicate_id="$(printf 'replicate_%03d' "${replicate_number}")"
  generation_seed=$((BASE_GENERATION_SEED + replicate_number - 1))
  replicate_root="${CAMPAIGN_ROOT}/raw/${replicate_id}"
  log_root="${replicate_root}/logs"
  result_root="${replicate_root}/results"
  run_id="${RUN_TAG}-${replicate_id}"

  echo
  echo "[${replicate_id}] generation_seed=${generation_seed}"
  echo "[${replicate_id}] logs=${log_root}"
  echo "[${replicate_id}] results=${result_root}"
  if [[ "${DRY_RUN}" == true ]]; then
    continue
  fi

  (
    export LOCAL_LLM_PORT
    export AGENTDOJO_LLM_TEMPERATURE=0.6
    export AGENTDOJO_LLM_TOP_P=0.95
    export AGENTDOJO_LLM_TOP_K=20
    export AGENTDOJO_LLM_MAX_TOKENS=4096
    export AGENTDOJO_GENERATION_SEED="${generation_seed}"
    export AGENTDOJO_ENABLE_THINKING=true
    bash "${SCRIPT_DIR}/run_qwen3_8b_task_injection.sh" \
      --suite full \
      --pairs "${PAIR_COUNT}" \
      --attack full \
      --workers "${WORKERS}" \
      --python-bin "${PYTHON_BIN}" \
      --port "${LOCAL_LLM_PORT}" \
      --benchmark-version "${BENCHMARK_VERSION}" \
      --target-model-id "${TARGET_MODEL_ID}" \
      --seed "${PAIR_SEED}" \
      --manifest "${MANIFEST_PATH}" \
      --log-root "${log_root}" \
      --result-root "${result_root}" \
      --run-id "${run_id}" \
      --rollout-id "${replicate_id}"
  )

  if [[ "${SKIP_CURATION}" == false ]]; then
    CURATION_ARGS=(
      --campaign-root "${CAMPAIGN_ROOT}"
      --manifest "${MANIFEST_PATH}"
    )
    if [[ "${REQUIRE_THINKING}" == true ]]; then
      CURATION_ARGS+=(--require-thinking)
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/curate_repeated_task_injection.py" "${CURATION_ARGS[@]}"
  fi
done

if (( ADAPTIVE_ROUNDS > 0 )); then
  adaptive_generation_seed=$((BASE_GENERATION_SEED + START_REPLICATE + REPLICATES - 1))
  echo
  echo "Adaptive phase: ${ADAPTIVE_ROUNDS} missing-label-only round(s)"
  echo "Adaptive generation seed starts at: ${adaptive_generation_seed}"
  if [[ "${DRY_RUN}" == false ]]; then
    ADAPTIVE_ARGS=(
      --campaign-root "${CAMPAIGN_ROOT}"
      --manifest "${MANIFEST_PATH}"
      --rounds "${ADAPTIVE_ROUNDS}"
      --start-round "${ADAPTIVE_START_ROUND}"
      --workers "${WORKERS}"
      --python-bin "${PYTHON_BIN}"
      --port "${LOCAL_LLM_PORT}"
      --target-model-id "${TARGET_MODEL_ID}"
      --generation-seed "${adaptive_generation_seed}"
      --run-tag "${RUN_TAG}-adaptive"
    )
    if [[ "${REQUIRE_THINKING}" == false ]]; then
      ADAPTIVE_ARGS+=(--allow-missing-thinking)
    fi
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_missing_rollout_rounds.py" "${ADAPTIVE_ARGS[@]}"
  fi
fi

echo
if [[ "${DRY_RUN}" == true ]]; then
  echo "Dry run complete; no benchmark requests were sent."
else
  echo "Campaign complete: ${CAMPAIGN_ROOT}"
  [[ "${SKIP_CURATION}" == true ]] || echo "Curated view: ${CAMPAIGN_ROOT}/curated"
fi
