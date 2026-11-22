# GPU v4 run handoff — migration + resumed research (new RTX 4090 node)

Writer: sole GPU worker (Pi 1.5 + deepseek-flash). Old GPU retired; CPU node is read-only
source + durable store. No parallel writer.

## TL;DR

* **Migration complete and durable.** Frozen v4 source HEAD `ebed683…` imported, all upstream
  inputs SHA-verified, 6-episode cross-GPU canary PASS, remaining 4617 TRAIN captured →
  **full TRAIN 8229 / 114 799 rows / 122.57 M tokens (all 4 scopes)**, DEV 2028 unchanged.
* **Pre-registered v4 result (raw bf16): negative.** Best `F:full` DEV **FN 31 / FP 80 /
  AUC 0.979620** vs accepted baseline **FN 21 / FP 80 / AUC 0.981127**; F vs baseline ΔFN +10.0,
  CI [1,21] (worse). Progress FN≤14 / strong FN≤8 **not met**.
* **Post-hoc, pre-declared arm G (TRAIN-only standardization):** DEV **FN 21 / FP 80 /
  AUC 0.984879** (per-seed .9827–.9847), G vs raw F ΔFN −10.1 CI [−20,−1] (better), G vs
  baseline ΔFN −0.08 CI [−8,8] (tied on FN, AUC +0.0038). Still no FN≤14.
* **3 latent frozen-code bugs fixed** (the old pause preceded P5): bf16→f32 resident path
  missing `<<16` (this collapsed all arms to AUC 0.5 on the first attempt), fused 3-head cost
  matmul on a 3-D weight, validator report-level/tensor/JSON defects. Validator now 12/12.
* Cost: host increment p95 ~0.011–0.013 ms, device ≤0.002 ms, real hook ~0.12 ms/event.
* Next method (block 16/24, new dataset version) is **pre-registered but not started**: measured
  16 blk 11.54 GiB/12 922 tok/s, 24 blk 22.93 GiB/8 586 tok/s; full sweep 3.2–4.8 h > remaining
  phase. Exact next commands below.

## Key numbers

| item | value |
|---|---|
| GPU | `autodl-container-ujxcycmw77-6fc011ce`, RTX 4090 24564 MiB |
| CPU durable | `/root/autodl-tmp/sdsafe-lowcost-authorization-v4-gpu` (2.5 GB) |
| model verify | 5 shard sizes + index/config/tokenizer/vocab/merges SHA == manifest; 36 layers, 399 tensors |
| old shards | 3612 train + 2028 dev SHA-ok, 0 bad |
| transfer | model 16.40 GB ~42 min (~6.5 MB/s, 5 streams); prepared ids 273 MB / 4623 eps / 57.88 M tokens; old shards 1.28 GB tar |
| canary | 6 eps; block4 cos_min 0.9999468, block8 0.9999292; bindings identical |
| new capture | 4617 eps in 27.0 GPU-min, ~35 130 tok/s, peak 6.34 GiB, 4617/4617 durable SHA-ok |
| v4 heldout fold0 | C FN11/FP82/AUC .99069; E FN26; F FN16/AUC .99108 (TRAIN only) |
| v4 DEV | C:full FN33 .97593; E:full FN50; F:full FN31 .97962; baseline FN21 .98113 |
| G DEV | FN21/FP80/AUC .984879; G:ho FN26 .978497 |
| validation | 12/12 checks; 6 DEV variants float64-recomputed, 0 mismatch; 5/6 mutations detected |
| cost | host p95 0.0111–0.0127 ms; device p95 ≤0.0021 ms; hook ~0.12 ms/event |

## Durable artifacts (CPU, SHA read-back verified)

`REPORT.md`, `MIGRATION_REPORT.md`, `HANDOFF.json`, `STATUS.json`,
`artifacts/eval/{report.json,report_slim.json,v4b_report.json}`,
`artifacts/{models,models_v4b}/`, `artifacts/cost/`, `artifacts/validation/validation.json`,
`artifacts/canary/canary_report.json`, `artifacts/capture/train/` (4617 shards +
`verify_summary.json`), `artifacts/manifests/results_sha256sums.txt` (72 files, verified on CPU).
Source is an independent git; lineages: `aa8168e` import, `ddc144b` path map, `55da3d5` pre-skip
+ union view, `8923034` prepared jobs, `80d9cef` prepared validator/cost, `9aca299` bf16 fix,
`0124f84` cost/validate fixes, `7f0b1f9` docs.

## Resume / next commands

The `/dev/shm` mirror is volatile (rebuild with `scripts/migrate_pull*.sh`); the durable results
are on the CPU. To continue the depth test:

1. CPU: export prepared ids for **all** 8229 TRAIN + 2028 DEV (extend
   `bridge/export_prepared.py`; current `prepared/train` has 4623).
2. GPU: capture into a **new** ART version (never overwrite block4/8):
   `SDSAFE_PREPARED_DIR=.../prepared_all PYTHONPATH=src python -m sdsafe_v4.capture --split train`
   with `PrefixModel(blocks=16, captures=(8,16))` (or 24) writing `artifacts/capture16/...`.
3. Train pre-registered arm `H = standardized [final‖block16]`, 3 seeds × {ho,full}, DEV
   exploratory; compare to G and the accepted baseline with paired whole-group bootstrap.

## Residual risks

1. Block16/24 route unexecuted; the 24-block path is near the 24 GiB limit (22.93 GiB at 35.7 k
   tokens) and must be re-smoked per shape before capture.
2. The validator's `forged_metric` mutation is vacuous (never tampers a stored number); the real
   guard is the independent float64 recompute (passed, 0 mismatches). Should be strengthened.
3. Old shards were produced on a 3080 Ti; new shards on a 4090. Canary is within bf16 tolerance
   (cos ≥0.99993) but the union is not bitwise homogeneous. The validator's bit-exact recapture
   only samples the new 4090 shards.
4. DEV remains an exploratory reuse (baseline itself was DEV-selected); no independent confirm.
5. `/dev/shm` mirror is not durable; `/root` is at 55% (14 GB free). Cleanup: remove the
   restricted migration key when the pair is retired; keep the `-p13661` runner key for sync.
