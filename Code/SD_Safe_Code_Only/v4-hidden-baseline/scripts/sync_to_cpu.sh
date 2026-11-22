#!/bin/bash
# Push local new-capture state to the CPU durable dir (atomic-ish, no --delete), then have
# the CPU verify each npz against the digest in progress.jsonl.
set -u
KEY=/root/sdsafe-runner-runtime/keys/id_ed25519_cpu_store
KH=/root/sdsafe-runner-runtime/known_hosts_cpu
SSH_OPTS="-p 13661 -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KH"
SRC=root@connect.bjb2.seetacloud.com
SPLIT="${1:-train}"
LOCAL=/root/sdsafe-compute/v4-gpu/artifacts/capture/$SPLIT
DUR=/root/autodl-tmp/sdsafe-lowcost-authorization-v4-gpu/capture/$SPLIT
mkdir -p "$DUR/episodes"
ssh $SSH_OPTS "$SRC" "mkdir -p '$DUR/episodes'" || exit 2
# shards first, then the commit log (progress.jsonl last so a verifier never sees a digest
# before its file)
rsync -a --partial -e "ssh $SSH_OPTS" "$LOCAL/episodes/" "$SRC:$DUR/episodes/" || exit 3
rsync -a --partial -e "ssh $SSH_OPTS" "$LOCAL/progress.jsonl" "$SRC:$DUR/progress.jsonl" || exit 4
ssh $SSH_OPTS "$SRC" "PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python /root/autodl-tmp/sdsafe-gpu-migration-v1/bridge/verify_synced.py $SPLIT" || exit 5
