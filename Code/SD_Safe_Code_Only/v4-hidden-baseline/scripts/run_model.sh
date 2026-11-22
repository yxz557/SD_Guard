#!/usr/bin/env bash
# post-capture pipeline: datasets -> 18 fits -> selection/eval -> cost -> validator -> report
set -eu
cd /root/autodl-tmp/sdsafe-lowcost-authorization-v4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
PY=/root/vllm_env_backup/bin/python
echo "== build + fit + select + dev eval =="
$PY -m sdsafe_v4.run_all 2>&1 | tee logs/run_all.log
echo "== cost =="
for r in 0 1 2; do $PY -m sdsafe_v4.cost --rep $r > logs/cost_rep$r.log 2>&1 || echo "cost rep$r failed"; done
echo "== hook cost inside the real forward =="
$PY -m sdsafe_v4.cost --rep 0 --hook > logs/hook_rep0.log 2>&1 || echo "hook cost failed"
echo "== validate =="
$PY -m sdsafe_v4.validate 2>&1 | tee logs/validate.log
echo "== finalize =="
$PY -m sdsafe_v4.finalize 2>&1 | tee logs/finalize.log
$PY -m sdsafe_v4.report_md 2>&1 | tail -20
echo DONE
