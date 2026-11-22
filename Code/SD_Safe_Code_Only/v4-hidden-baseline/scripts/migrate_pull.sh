#!/bin/bash
# Resumable, read-only rsync pull of the frozen upstream inputs from the CPU/storage node
# into the volatile /dev/shm mirror. No --delete; source is never modified.
set -u
KEY=/root/sdsafe-runner-runtime/keys/id_ed25519_cpu_store
KH=/root/sdsafe-runner-runtime/known_hosts_cpu
SSH_OPTS="-p 13661 -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KH"
SRC=root@connect.bjb2.seetacloud.com
M=/dev/shm/sdsafe-gpu-v4/upstream
LOG=/dev/shm/sdsafe-gpu-v4/logs/migrate_pull.log
mkdir -p /dev/shm/sdsafe-gpu-v4/logs
echo "=== pull start $(date -Is) ===" >>"$LOG"

pull() {
  local label="$1"; local src="$2"; local dst="$3"
  echo "--- [$label] $src -> $dst @ $(date -Is)" >>"$LOG"
  local t0=$(date +%s)
  rsync -a --partial --append-verify \
    -e "ssh $SSH_OPTS" \
    "$SRC:$src" "$dst" >>"$LOG" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "--- [$label] rc=$rc bytes_dst=$(du -sb "$dst" 2>/dev/null | cut -f1) secs=$((t1-t0)) @ $(date -Is)" >>"$LOG"
  return $rc
}

r=0
pull model      /root/autodl-tmp/SD-Safe/models/Qwen3-8B/            "$M/SD-Safe/models/"            || r=1
pull sdsafe_src /root/autodl-tmp/SD-Safe/src/                       "$M/SD-Safe/"                   || r=1
pull sdsafe_data /root/autodl-tmp/SD-Safe/data/                     "$M/SD-Safe/"                   || r=1
pull ext_agentdojo /root/autodl-tmp/SD-Safe/external/AgentLAB/Task-Injection/agentdojo/src/ "$M/SD-Safe/external/AgentLAB/Task-Injection/agentdojo/" || r=1
pull ext_agentlab  /root/autodl-tmp/SD-Safe/external/AgentLAB/src/   "$M/SD-Safe/external/AgentLAB/" || r=1
pull v1_index   /root/autodl-tmp/sdsafe-lowcost-authorization-v1/artifacts/index/   "$M/sdsafe-lowcost-authorization-v1/artifacts/" || r=1
pull v2_tensors /root/autodl-tmp/sdsafe-lowcost-authorization-v2/artifacts/tensors/ "$M/sdsafe-lowcost-authorization-v2/artifacts/" || r=1
pull v2_baseline /root/autodl-tmp/sdsafe-lowcost-authorization-v2/artifacts/baseline/ "$M/sdsafe-lowcost-authorization-v2/artifacts/" || r=1

echo "=== pull done rc=$r $(date -Is) ===" >>"$LOG"
echo "$r" > /dev/shm/sdsafe-gpu-v4/logs/migrate_pull.rc
