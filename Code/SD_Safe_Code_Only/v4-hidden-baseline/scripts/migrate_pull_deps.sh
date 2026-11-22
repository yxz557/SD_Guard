#!/bin/bash
# Pull the non-model read-only dependencies into the /dev/shm mirror (parallel streams).
set -u
KEY=/root/sdsafe-runner-runtime/keys/id_ed25519_cpu_store
KH=/root/sdsafe-runner-runtime/known_hosts_cpu
SSH_OPTS="-p 13661 -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KH"
SRC=root@connect.bjb2.seetacloud.com
M=/dev/shm/sdsafe-gpu-v4/upstream
L=/dev/shm/sdsafe-gpu-v4/logs
mkdir -p "$M/sdsafe-lowcost-authorization-v1/artifacts" "$M/sdsafe-lowcost-authorization-v2/artifacts" "$M/SD-Safe"

start() { # label src dst
  echo "--- [$1] start $(date -Is)" >>"$L/deps_$1.log"
  rsync -a --partial --append-verify -e "ssh $SSH_OPTS" "$SRC:$2" "$3" >>"$L/deps_$1.log" 2>&1
  echo "--- [$1] rc=$? bytes=$(du -sb "$3" 2>/dev/null | cut -f1) $(date -Is)" >>"$L/deps_$1.log"
}

start sdsafe_data /root/autodl-tmp/SD-Safe/data/            "$M/SD-Safe/" &
start v2_tensors  /root/autodl-tmp/sdsafe-lowcost-authorization-v2/artifacts/tensors/ "$M/sdsafe-lowcost-authorization-v2/artifacts/" &
start v2_baseline /root/autodl-tmp/sdsafe-lowcost-authorization-v2/artifacts/baseline/ "$M/sdsafe-lowcost-authorization-v2/artifacts/" &
start v1_index    /root/autodl-tmp/sdsafe-lowcost-authorization-v1/artifacts/index/    "$M/sdsafe-lowcost-authorization-v1/artifacts/" &
start sdsafe_src  /root/autodl-tmp/SD-Safe/src/             "$M/SD-Safe/" &
start ext_agentdojo /root/autodl-tmp/SD-Safe/external/AgentLAB/Task-Injection/agentdojo/src/ "$M/SD-Safe/external/AgentLAB/Task-Injection/agentdojo/" &
start ext_agentlab  /root/autodl-tmp/SD-Safe/external/AgentLAB/src/ "$M/SD-Safe/external/AgentLAB/" &
wait
echo "=== deps pull done $(date -Is) ===" >>"$L/deps_done.log"
