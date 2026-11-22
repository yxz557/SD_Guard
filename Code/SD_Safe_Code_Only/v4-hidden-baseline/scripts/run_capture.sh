#!/usr/bin/env bash
# resumable capture driver; PYTHONDONTWRITEBYTECODE and no system-package changes
set -u
cd /root/autodl-tmp/sdsafe-lowcost-authorization-v4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /root/vllm_env_backup/bin/python -m sdsafe_v4.capture "$@"
