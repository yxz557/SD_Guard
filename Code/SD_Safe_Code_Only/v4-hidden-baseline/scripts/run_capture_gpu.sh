#!/bin/bash
# Single GPU producer for resumed v4 TRAIN capture, with periodic durable sync to CPU.
# Stop: kill "$(cat logs/capture_train_gpu.pid)"  (TERM; producer children are daemonic)
set -u
cd /root/sdsafe-compute/v4-gpu
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/root/sdsafe-compute/v4-gpu/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SDSAFE_UPSTREAM_MIRROR=/dev/shm/sdsafe-gpu-v4/upstream
export SDSAFE_PREPARED_DIR=/root/sdsafe-compute/v4-gpu/manifests/prepared/train
export SDSAFE_PRESKIP_DONE=/root/sdsafe-compute/v4-gpu/artifacts/oldv4/train/progress.jsonl
mkdir -p logs artifacts/capture/train/episodes artifacts/capture/dev/episodes
LOG=logs/capture_train_gpu.log
/root/vllm_env_backup/bin/python -u -m sdsafe_v4.capture --split train >>"$LOG" 2>&1 &
CPID=$!
echo "$CPID" > logs/capture_train_gpu.pid
printf '{"pid":%s,"cwd":"%s","split":"train","started":"%s","stop":"kill $(cat logs/capture_train_gpu.pid)"}\n' \
  "$CPID" "$PWD" "$(date -Is)" > logs/capture_owner.json
echo "[wrapper] capture pid=$CPID" >>"$LOG"
while kill -0 "$CPID" 2>/dev/null; do
  sleep 120
  bash scripts/sync_to_cpu.sh train >>logs/sync_train.log 2>&1
done
wait "$CPID"; rc=$?
bash scripts/sync_to_cpu.sh train >>logs/sync_train.log 2>&1
echo "$rc" > logs/capture_train_gpu.rc
echo "[wrapper] capture exited rc=$rc" >>"$LOG"
