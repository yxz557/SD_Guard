# REPORT — `sdsafe-lowcost-authorization-v5-depth` (block 16 / 24 new data version)

## 0. Headline

* New dataset version captured from the **same frozen Qwen3-8B forward**: residual stream after block **16** and block **24**, at the already accepted endpoint token of every CT event. Backend: `monolithic` (prefix 24 blocks).
* TRAIN **8229 / 114799 rows / 122,566,048 tokens**; common DEV **2028 / 25003 rows / 26,841,965 tokens**.
* **TRAIN-fold-0-selected** candidate `K:ho`: DEV **FN 26 / FP 80 / AUC 0.978894** (threshold -2.025446).
* **DEV-best** (exploratory only, never used for selection) `K:full`: FN 22 / FP 80 / AUC 0.982939. Neither candidate beats the canonical baseline FN 21 / FP 78.
* Accepted baseline canonical frozen point replayed: **FN 21 / FP 78 / AUC 0.981127** at threshold -0.6190447409947714 (independent replay, exact).
* Progress bar FN ≤ 14: **NOT met**; strong FN ≤ 8: **NOT met**.
* Several depth arms raise AUC above the accepted baseline (e.g. J:full .98507, imported G48 .98488, K:full .98294) but their FN is 22–23, not below 21; an AUC gain with FN unchanged/worse is **not** counted as progress here.
* Recovered / new FN vs the canonical baseline: **6 recovered / 11 new** (episode IDs in HANDOFF.json).
* DEV remains an exploratory reuse (the accepted baseline itself was DEV-selected). No TEST material, no API/teacher call, no quantisation, no weight/template/RoPE change.

## 1. What is new

A **new dataset version** of activations from the same unmodified frozen backbone at two greater depths (HF `output_hidden_states[16]` and `[24]`), read at the already accepted endpoint token. The pre-existing block-4/8 shards are read-only dependencies and are **not** mixed into the block-16/24 matrices. Nothing was re-tokenised, truncated, quantised or re-templated.

### 1.1 Exactness / smoke (measured, fresh process per measurement)

| blocks | shape | seq len | peak alloc GiB | tok/s | determinism | suffix | wrong endpoint | no future | outside prefix rejected | lm_head | norm meta |
|---|---|---:|---:|---:|---|---|---|---|---|---|---|
| 16 | boundary | 38006 | 10.71 | 12585.1 | None | None | None | None | None | False | True |
| 16 | equiv_c2048 | 7000 | 7.62 | 18107.9 | True | True | True | True | True | False | True |
| 16 | equiv_mono | 7000 | 7.62 | 18001.0 | True | True | True | True | True | False | True |
| 16 | medium | 7000 | 7.62 | 18039.4 | None | None | None | None | None | False | True |
| 16 | short | 298 | 6.95 | 7359.5 | None | None | None | None | None | False | True |
| 24 | boundary | 38006 | 13.58 | 8379.5 | None | None | None | None | None | False | True |
| 24 | equiv_c2048 | 7000 | 10.49 | 12011.0 | True | True | True | True | True | False | True |
| 24 | equiv_mono | 7000 | 10.49 | 12024.7 | True | True | True | True | True | False | True |
| 24 | medium | 7000 | 10.49 | 12017.4 | None | None | None | None | None | False | True |
| 24 | short | 298 | 9.82 | 5836.7 | None | None | None | None | None | False | True |

KV-chunked prefill (2048-token chunks) was also measured: at block 24 it exceeded the pre-set `max_abs ≤ 0.5` tolerance (max_abs 1.0, cos 0.999708), so **monolithic prefix-24 is used** for the whole capture. The monolithic 24-block forward is legal on the longest prepared TRAIN round (38 006 tokens, 15.25 GiB reserved), so the pre-registered 16-only fallback was not needed.

## 2. Capture coverage

| split | episodes | CT rows | tokens | GPU min | tok/s | peak alloc GiB | backend |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 8229 | 114799 | 122,566,048 | 170.4 | 11989 | 13.58 | monolithic |
| dev | 2028 | 25003 | 26,841,965 | 36.6 | 12219 | 13.58 | monolithic |

Per-scope (episodes / rows / tokens):

* train: cot 2558/32973/52,014,842, ctx40k 2539/33086/13,710,801, nothink 2328/32743/51,576,635, agentlab 804/15997/5,263,770
* dev: cot 675/8091/11,772,806, ctx40k 697/8326/2,751,509, nothink 656/8586/12,317,650

Every shard is per-episode atomic with a SHA-256 commit record in `progress.jsonl`; finished batches were pushed to the CPU durable tree and verified by `sha256sum -c` read-back (34 verified batches, 0 failed).

## 3. Dataset and label pool

* train: 114799 rows / 8229 episodes; in-loss 93233 (pos 1722 / neg 91511); label pool {'pool_rows': 93322, 'pool_labeled': 93233, 'pool_in_loss': 93233, 'matched_capture_rows': 93322}
* dev: 25003 rows / 2028 episodes; in-loss 0 (pos 0 / neg 0); label pool {}

Standardization μ/σ is fit on **TRAIN in-loss rows only** (93233 rows); DEV never enters the scaler.

## 4. TRAIN held-out fold 0

Only the `:ho` rows are honest held-out numbers (fold 0 was **not** fitted); the `:full` rows include folds 0–3 in training, so their fold-0 numbers are optimistic diagnostics and are not used to compare methods.

| variant | FN | FP | AUC | episodes |
|---|---:|---:|---:|---:|
| C:ho | 10 | 82 | 0.990640 | 2015 |
| C:full | 0 | 82 | 0.999600 | 2015 |
| Craw:ho | 12 | 82 | 0.989954 | 2015 |
| Craw:full | 0 | 82 | 0.998421 | 2015 |
| H:ho | 9 | 82 | 0.993089 | 2015 |
| H:full | 0 | 82 | 0.999788 | 2015 |
| I:ho | 5 | 82 | 0.994335 | 2015 |
| I:full | 0 | 82 | 0.999726 | 2015 |
| J:ho | 5 | 82 | 0.993766 | 2015 |
| J:full | 0 | 82 | 0.999818 | 2015 |
| K:ho | 5 | 82 | 0.994445 | 2015 |
| K:full | 0 | 82 | 0.999880 | 2015 |

## 5. Common DEV 2028 (exploratory reuse; each method its own 5 %-FPR threshold)

| variant | FN | FP | TP | AUC | threshold | per-seed AUC | mean individual FN |
|---|---:|---:|---:|---:|---:|---|---:|
| **accepted baseline (canonical frozen)** | 21 | 78 | 393 | 0.981127 | -0.6190447409947714 | — | — |
| accepted baseline (budget-optimal point) | 21 | 80 | 393 | 0.981127 | -0.744608 | — | — |
| v4 literal threshold replay | 22 | 78 | 392 | 0.981127 | -0.6190447211265564 | — | — |
| C:ho | 35 | 80 | 379 | 0.974190 | -0.871932 | 0.97001, 0.97345, 0.97207 | 45.00 |
| C:full | 32 | 80 | 382 | 0.978048 | -0.502804 | 0.97848, 0.97246, 0.97826 | 31.67 |
| Craw:ho | 42 | 80 | 372 | 0.974739 | -0.215481 | 0.97300, 0.97352, 0.97314 | 42.67 |
| Craw:full | 36 | 80 | 378 | 0.975848 | 0.232570 | 0.97632, 0.97421, 0.97465 | 34.00 |
| H:ho | 33 | 80 | 381 | 0.977318 | -2.130525 | 0.97310, 0.97906, 0.97404 | 39.33 |
| H:full | 24 | 80 | 390 | 0.982602 | -2.550464 | 0.98097, 0.98133, 0.98220 | 26.00 |
| I:ho | 26 | 80 | 388 | 0.978892 | -2.592593 | 0.97610, 0.97631, 0.97976 | 30.00 |
| I:full | 25 | 80 | 389 | 0.983157 | -2.283902 | 0.97977, 0.98363, 0.98178 | 25.33 |
| J:ho | 26 | 80 | 388 | 0.977843 | -1.757935 | 0.97367, 0.97918, 0.97635 | 27.33 |
| J:full | 23 | 80 | 391 | 0.985067 | -3.351301 | 0.98385, 0.98342, 0.98302 | 23.00 |
| K:ho | 26 | 80 | 388 | 0.978894 | -2.025446 | 0.97699, 0.97661, 0.97673 | 29.00 |
| K:full | 22 | 80 | 392 | 0.982939 | -3.427180 | 0.97871, 0.98027, 0.98534 | 27.33 |
| G48:full (imported v4b) | 21 | 80 | 393 | 0.984879 | -4.177976 | — | — |

The canonical baseline point uses the threshold **stored in the accepted artifact** (`baseline_scores_v2.npz['threshold']`), which replays exactly 21 FN / 78 FP. The v4 source's literal `BASELINE['threshold']` (-0.6190447211265564) differs from that stored value by ~1.99e-08 and flips one borderline positive, so it replays 22 FN / 78 FP; that literal point is reported for transparency and is **not** used as the canonical comparison. Neither is re-budgeted. The budget-optimal point is the same baseline scores under the study's 5 %-FPR rule (21 FN / 80 FP) and is never substituted for the canonical point.

Borderline episode between the two threshold literals: `nothink_ctx40k::agentlab-613bb1c0d66f4d84686724e8f3a6de1d` (score -0.6190447409947714, label True; artifact threshold -0.6190447409947714 vs v4 constant -0.6190447211265564).

## 6. Paired whole-group bootstrap (fixed thresholds; no in-draw ROC re-selection)

| comparison | ΔFN mean [CI95] | ΔFP mean [CI95] | replicates |
|---|---|---|---:|
| K:ho_vs_baseline_canonical | 5.03 [-6.0, 18.0] | 2.15 [-11.0, 16.0] | 2000 |
| K:ho_vs_baseline_budget | 5.03 [-6.0, 18.0] | 0.13 [-13.0, 14.0] | 2000 |
| C:full_vs_baseline_canonical | 11.07 [3.0, 22.0] | 2.02 [-7.0, 11.0] | 2000 |
| C:full_vs_baseline_budget | 11.07 [3.0, 22.0] | -0.01 [-9.0, 9.0] | 2000 |
| Craw:full_vs_baseline_canonical | 15.02 [6.0, 26.0] | 1.95 [-7.0, 11.0] | 2000 |
| Craw:full_vs_baseline_budget | 15.02 [6.0, 26.0] | -0.07 [-9.0, 9.0] | 2000 |
| H:full_vs_baseline_canonical | 3.04 [-8.0, 15.0] | 2.13 [-13.0, 17.0] | 2000 |
| H:full_vs_baseline_budget | 3.04 [-8.0, 15.0] | 0.11 [-15.0, 15.0] | 2000 |
| I:full_vs_baseline_canonical | 3.94 [-4.0, 13.0] | 2.07 [-11.0, 16.0] | 2000 |
| I:full_vs_baseline_budget | 3.94 [-4.0, 13.0] | 0.05 [-14.0, 14.0] | 2000 |
| J:full_vs_baseline_canonical | 2.02 [-7.0, 12.0] | 2.24 [-14.0, 19.0] | 2000 |
| J:full_vs_baseline_budget | 2.02 [-7.0, 12.0] | 0.22 [-16.0, 17.0] | 2000 |
| K:full_vs_baseline_canonical | 1.09 [-8.0, 12.0] | 2.15 [-14.0, 18.0] | 2000 |
| K:full_vs_baseline_budget | 1.09 [-8.0, 12.0] | 0.12 [-16.0, 17.0] | 2000 |

`ensemble` = per-seed episode maxima then mean over heads; `mean individual FN` is the mean of the three per-seed confusion values, reported separately and not used as the ensemble number.

## 7. Error movement and breakdowns

* K:ho: baseline FN 21 → candidate FN 26; recovered 6, new 11; baseline FP 78 → candidate FP 80 (recovered 17, new 19)
* C:full: baseline FN 21 → candidate FN 32; recovered 0, new 11; baseline FP 78 → candidate FP 80 (recovered 9, new 11)
* Craw:full: baseline FN 21 → candidate FN 36; recovered 1, new 16; baseline FP 78 → candidate FP 80 (recovered 7, new 9)
* H:full: baseline FN 21 → candidate FN 24; recovered 6, new 9; baseline FP 78 → candidate FP 80 (recovered 19, new 21)
* I:full: baseline FN 21 → candidate FN 25; recovered 4, new 8; baseline FP 78 → candidate FP 80 (recovered 17, new 19)
* J:full: baseline FN 21 → candidate FN 23; recovered 6, new 8; baseline FP 78 → candidate FP 80 (recovered 20, new 22)
* K:full: baseline FN 21 → candidate FN 22; recovered 6, new 7; baseline FP 78 → candidate FP 80 (recovered 21, new 23)

Candidate breakdown by scope / suite / IT6:

```json
{
 "scope": {
  "cot": {
   "episodes": 675,
   "positives": 125,
   "tp": 117,
   "fp": 22,
   "fn": 8
  },
  "ctx40k": {
   "episodes": 697,
   "positives": 134,
   "tp": 124,
   "fp": 21,
   "fn": 10
  },
  "nothink": {
   "episodes": 656,
   "positives": 155,
   "tp": 147,
   "fp": 37,
   "fn": 8
  }
 },
 "suite": {
  "banking": {
   "episodes": 414,
   "positives": 84,
   "tp": 76,
   "fp": 33,
   "fn": 8
  },
  "slack": {
   "episodes": 406,
   "positives": 237,
   "tp": 236,
   "fp": 18,
   "fn": 1
  },
  "travel": {
   "episodes": 274,
   "positives": 85,
   "tp": 71,
   "fp": 28,
   "fn": 14
  },
  "workspace": {
   "episodes": 934,
   "positives": 8,
   "tp": 5,
   "fp": 1,
   "fn": 3
  }
 },
 "it6": {
  "IT6": {
   "episodes": 155,
   "positives": 21,
   "tp": 8,
   "fp": 13,
   "fn": 13
  },
  "not_IT6": {
   "episodes": 1873,
   "positives": 393,
   "tp": 380,
   "fp": 67,
   "fn": 13
  }
 }
}
```

## 8. Cost of the added deployment operations (same 4090)

| rep | base host p95 ms | cand host p95 ms | Δ host p95 | base dev p95 | cand dev p95 | Δ dev p95 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.1912 | 0.2973 | +0.1061 | 0.1233 | 0.1243 | +0.0010 |
| 1 | 0.1995 | 0.2938 | +0.0943 | 0.1358 | 0.1303 | -0.0055 |
| 2 | 0.1946 | 0.2846 | +0.0900 | 0.1248 | 0.1248 | +0.0000 |

Real hook/select/scale/head inside a real prefix forward (episode `cot32k::agentlab-04506950aa51b4719692fb10e306da80`, 2 endpoints, seq 7824): select p95 0.015650689601898186 ms, scaler p95 0.12388136237859719 ms, head p95 0.27172639966011025 ms, total p95 **0.4112584516406056 ms/CT**.

Gate: added p95 ≤ 1 ms/CT — host pass True, device pass True.

### 8.1 Paired complete 36-block backbone replay (no lm_head, no token logits)

| shape | hook | seq len | status | median ms | p95 ms | peak alloc GiB |
|---|---:|---:|---|---:|---:|---:|
| boundary | False | 26231 | ok | 4172.415420413017 | 4195.330399274826 | 16.72 |
| boundary | True | 26231 | ok | 4192.081330344081 | 4205.427116900682 | 17.12 |
| short | False | 219 | ok | 32.06024877727032 | 32.94763267040253 | 14.13 |
| short | True | 219 | ok | 31.35950118303299 | 33.41641668230295 | 14.13 |

This is a real shared-backbone pairing (same weights, same token ids, hooks on the actual activation); it is not a claim about the full online service with lm_head sampling.

## 9. Validation

* 24 formal checkpoint re-scores in independent float64 numpy; passed: True.
* capture source/digest checks passed: True; bitwise recapture: True.
* mutations all detected: True; missing: [].
* forged-metric mutation (tamper a stored FN in a shadow copy of the real report) detected: True — this is the non-vacuous guard the v4 stage lacked.

```json
{
 "mutations": {
  "bf16_conversion_detected": true,
  "wrong_endpoint_detected": true,
  "endpoint_guard_accepts_clean": true,
  "wrong_layer_detected": true,
  "layer_guard_accepts_clean": true,
  "row_order_detected": true,
  "row_order_guard_accepts_clean": true,
  "illegal_cohort_detected": true,
  "cohort_guard_accepts_clean": true,
  "dev_cohort_guard_ok": true,
  "oof_leak_detected": true,
  "oof_guard_accepts_clean": true,
  "model_sha_detected": true,
  "model_sha_guard_accepts_clean": true,
  "manifest_tamper_detected": true,
  "manifest_guard_accepts_clean": true,
  "future_token_detected": true,
  "future_token_clean": true,
  "forged_metric_detected": true,
  "forged_metric_mismatches": [
   "H:full:dev fn stored=21 independent=24"
  ],
  "clean_report_mismatches": []
 }
}
```

## 10. What this stage does not establish

* It is not an independent confirmation: DEV was reused and the accepted baseline was DEV-selected.
* The full-36-block cost pairing covers the backbone forward only (no `lm_head` token logits), and only for the shapes actually measured; long full-36 replay is reported if it ran, otherwise its OOM is explicit.
* Sequences are exact accepted ids; the proof is the per-row digest, not the producer's file hash.
* No teacher/API annotation was added. No TEST raw/features/labels/predictions were read.

## 10.1 Code fixes made in this stage (all committed)

* **Capture producer completion bug (real, cost episodes).** The first full TRAIN run stopped on the *first* producer's `done` message, dropping 13 episodes that were still queued from the other five producers (8 216/8 229). Fixed by counting all producers; the resumable rerun captured the remaining 13 and the final coverage is exact 8 229 / 114 799 / 122 566 048. The durable shards were never corrupted.
* **Durable sync bugs (real).** `rsync -e` was given `ssh ssh …` (double command) and the append-only `progress.jsonl` was hashed while being written. Fixed: correct `-e` string, and mutable files are now snapshot-hashed and streamed as exactly the bytes hashed, then read back on the CPU.
* **Validator gaps (real).** The v4 `forged_metric` mutation never tampered a stored number and the v5 manifest guard initially had no manifest to test. Fixed: the validator now builds its own artifact manifest first and the forged-metric guard edits a stored FN in a shadow copy of the real report (detected, 0 clean-report mismatches). K checkpoints are re-scored by an independent float64 numpy gated forward; a ±1 confusion tolerance documents the float32-vs-float64 threshold boundary (0 clean mismatches observed).
* **Complete 36-block replay support.** `PrefixModel(blocks=36)` previously left `model.norm` on the meta device; it is now materialised for the paired cost replay (no `lm_head`).
* Carried from v4 (unchanged invariants): bf16 `uint16 → float32` must shift by 16 bits (pinned by tests), the fused 3-head matmul must not `.t()` a 3-D weight, and the validator must not share the pipeline's loader/metric code.

## 11. Provenance

```
5ec2271 stage complete: exact TRAIN 8229/114799 rows/122.57M tokens + DEV 2028/25003/26.84M captured monolithic prefix-24 (blocks 16/24); H/I/J + C/Craw first wave, conditional gated-low-rank K; DEV baseline replayed at artifact threshold 21/78 (v4 literal 22/78 documented); cost increments <=0.11ms host / ~0ms device, real hook 0.41ms/CT, paired full-36 backbone replay; validator 24/24 formal + all 10 mutations incl non-vacuous forged metric; report/handoff/manifest
fac29e0 eval: select primary candidate on TRAIN fold0 only; report DEV-best separately as exploratory
c552155 conditional gated-low-rank arm K (validated on partial data), validator tolerance for float32/float64 boundary, v4 block-8 and full-36 bitwise cross-checks
178d52a eval: recovered/new FP sets alongside FN sets
e0134ac pre-register conditional second-wave gated low-rank arm K and coverage/label-pool accounting
58d9da0 eval: replay canonical baseline at the artifact-stored threshold (21/78) and report the v4 literal (22/78) transparently; fix durable-sync rsync ssh + mutable progress snapshot; report generator baseline rows; document threshold discrepancy
1b07c6a depth pipeline: endpoint-only prefix capture, KV-chunk equivalence smoke (monolithic chosen: 15.25GiB peak at 38k, chunked exceeds maxabs tol), dataset builder, standardized H/I/J + C/Craw trainer, eval with canonical-vs-budget baseline replay, independent float64 validator with non-vacuous forged-metric mutation, cost + full36 replay, status watcher
ba4c6e5 import v4 source (read-only copy, HEAD ebed683 lineage) into independent v5-depth git; v5 constants, endpoint-only prefix capture, independent bf16 loader + bitpattern tests, CPU all-prepared exporter, smoke driver; PLAN/PROTOCOL/STATUS
```

All **16** upstream input files (model shards, rows, accepted features, indices, label pool, baseline scores) are bit-identical between the CPU authority and the GPU mirror, **before and after** this stage: **YES** (input_hashes_cpu.json, input_hashes_gpu.json, input_hashes_cpu_after.json, input_hashes_gpu_after.json; mismatches []).

Manifest `artifacts/manifests/artifact_hashes.json`: 10454 files, 2,694,278,083 bytes. Large tensors are listed by size (git-ignored).

Source is an independent git (`sdsafe-compute/v5-depth`); the v4 tree is read-only. All writes stayed inside the assigned GPU/CPU paths.

