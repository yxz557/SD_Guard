#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 DATASET_PATH MODEL_PATH OUTPUT_DIR [SAMPLE_COUNT] [SEED]" >&2
  exit 2
fi

dataset_path=$1
model_path=$2
output_dir=$3
sample_count=${4:-100}
seed=${5:-42}

python_bin=${PYTHON_BIN:-python}
guard_gpu=${GUARD_GPU:-0}
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
custom_dir="${output_dir}/custom"
official_output="${output_dir}/official_predictions.jsonl"
analysis_dir="${output_dir}/analysis"

CUDA_VISIBLE_DEVICES="${guard_gpu}" "${python_bin}" -u "${root_dir}/eval_stream_guard.py" \
  --model-path "${model_path}" \
  --dataset-path "${dataset_path}" \
  --output-dir "${custom_dir}" \
  --sample-count "${sample_count}" \
  --seed "${seed}" \
  --overwrite

CUDA_VISIBLE_DEVICES="${guard_gpu}" "${python_bin}" -u "${root_dir}/official/eval_stream_sampled.py" \
  --model-path "${model_path}" \
  --dataset-path "${dataset_path}" \
  --sample-manifest "${custom_dir}/sample_manifest.json" \
  --output-path "${official_output}" \
  --overwrite

"${python_bin}" -u "${root_dir}/analyze_stream_results.py" \
  --custom-dir "${custom_dir}" \
  --official-output "${official_output}" \
  --output-dir "${analysis_dir}" \
  --overwrite

echo "Comparison complete: ${output_dir}"
