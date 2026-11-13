#!/usr/bin/env bash

# Qwen3-8B vLLM server matching the recorded ctx40k AgentDojo experiment.
# Sampling and thinking parameters are sent explicitly by the rollout runner.

set -Eeuo pipefail

AD_MODEL="${AD_MODEL:-/root/autodl-tmp/models/Qwen3-8B}"
LOCAL_LLM_PORT="${LOCAL_LLM_PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"

export CUDA_VISIBLE_DEVICES

echo "Starting Qwen3-8B vLLM server"
echo "  model:                  ${AD_MODEL}"
echo "  CUDA_VISIBLE_DEVICES:   ${CUDA_VISIBLE_DEVICES}"
echo "  port:                   ${LOCAL_LLM_PORT}"
echo "  tensor parallel:        ${TENSOR_PARALLEL_SIZE}"
echo "  GPU memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  max model length:       ${MAX_MODEL_LEN}"
echo "  max concurrent seqs:    ${MAX_NUM_SEQS}"
echo "  tool parser:            hermes"
echo "  reasoning parser:       qwen3"

exec vllm serve "${AD_MODEL}" \
  --host 127.0.0.1 \
  --port "${LOCAL_LLM_PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
