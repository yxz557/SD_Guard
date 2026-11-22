# REPORT — `sdsafe-lowcost-authorization-v4` (earlier-layer activation route)

## 0. Headline

* All 18 pre-registered fits ran. Best DEV variant on the captured common-DEV universe: `F:full` → FN 31 / FP 80 / AUC 0.979620 (accepted baseline on the same episodes: FN 21 / FP 80 / AUC 0.981127).
* Progress bar FN ≤ 14: **not met**; strong FN ≤ 8: **not met**.
* Captured 2028 / 2028 common-DEV episodes and 8229 TRAIN episodes; DEV coverage FULL common DEV.
* DEV is an exploratory reuse of the accepted common-DEV universe. No TEST material, no API call, no deployment-time second backbone pass.

## 1. What is new (and what is not)

New: the residual stream of the **same frozen Qwen3-8B** after block 4 and after block 8, read at the **already accepted** endpoint token of every CT event. It was never in any accepted artifact, so the v1–v3 finding ("re-weighting the final layer does not move the operating point") says nothing about it.

Not new / not claimed: no new backbone, no re-training of Qwen, no quantisation, no truncation, no re-tokenisation, no second deployment pass, no TEST read, no teacher/API annotation.

### Exactness of the offline prefix execution

`embed_tokens + rotary_emb + layers[0:8]` of the unmodified frozen weights, executed through `transformers`'s own `Qwen3Model.forward` with an early-exit `forward_pre_hook` on `layers[8]` (so `layers[8..35]` never run and `model.norm` is never applied). Weight keys are asserted equal to the safetensors key map; the token ids are the accepted archived/render ids, verified per round against the row's frozen `sequence_sha256`.

| smoke check | result |
|---|---|
| early-exit hook vs independent replay of the same loop | bitwise equal (blocks True, True) |
| suffix invariance at block 4 | bitwise equal |
| block 4: GPU bf16 vs CPU fp32 | max abs 0.01873, cos 0.999988 |
| block 4: GPU prefix vs complete 36-block CPU bf16 | max abs 0.01562, cos 0.999992 |
| block 8: GPU bf16 vs CPU fp32 | max abs 0.1239, cos 0.999984 |
| block 8: GPU prefix vs complete 36-block CPU bf16 | max abs 0.25, cos 0.999972 |
| complete 36-block CPU bf16 → `model.norm` vs the **accepted feature matrix** | cos 0.999787, mean abs 0.02534 |

GPU peak during capture 4.90 GB (weights 4.33 GB); the full 16 GiB BF16 model still cannot be resident, which is why the *complete-model* reference runs on CPU (220 s for 321 tokens) and why no end-to-end main-model wall clock is claimed.

## 2. Capture

| split | episodes | CT rows | tokens prefilled | GPU time | peak |
|---|---:|---:|---:|---:|---:|
| train | 8229 | 114799 | 122,566,048 | 26.9 min | 8.72 GB |
| dev | 2028 | 25003 | 26,841,965 | 0.0 min | 0.00 GB |

Per-scope coverage:

| split | scope | episodes | rows |
|---|---|---:|---:|
| train | agentlab | 804 | 15997 |
| train | cot | 2558 | 32973 |
| train | ctx40k | 2539 | 33086 |
| train | nothink | 2328 | 32743 |
| dev | cot | 675 | 8091 |
| dev | ctx40k | 697 | 8326 |
| dev | nothink | 656 | 8586 |

The only missing scope is now captured: the no-think template needed the frozen AgentDojo tool schema, which this study rebuilt through a local import shim and **verified against the archive's own `request_tools_sha256`** (travel suite digest matches bit for bit) before any sequence was accepted. Every no-think round was additionally bound to its accepted row digest.

## 3. TRAIN-only selection (fold 0 groups never fitted)

| arm | FN | FP | AUC | episodes |
|---|---:|---:|---:|---:|
| C | 11 | 82 | 0.990687 | 2015 |
| E | 26 | 82 | 0.980844 | 2015 |
| F | 16 | 82 | 0.991078 | 2015 |

## 4. DEV (exploratory reuse; own 5 %-FPR threshold)

| variant | FN | FP | AUC | FNR | FPR | threshold |
|---|---:|---:|---:|---:|---:|---:|
| C:ho | 35 | 80 | 0.971261 | 0.0845 | 0.0496 | 1.2219 |
| C:full | 33 | 80 | 0.975928 | 0.0797 | 0.0496 | -1.0496 |
| E:ho | 47 | 80 | 0.970347 | 0.1135 | 0.0496 | -0.6185 |
| E:full | 50 | 80 | 0.972575 | 0.1208 | 0.0496 | 0.7153 |
| F:ho | 32 | 80 | 0.975670 | 0.0773 | 0.0496 | -0.5935 |
| F:full | 31 | 80 | 0.979620 | 0.0749 | 0.0496 | 0.1905 |

Accepted baseline replayed on its own 2 028-episode universe: FN 21 / FP 80 / AUC 0.981127 (thr -0.744608).

Paired whole-group bootstrap (2 000 replicates, fixed thresholds):

* `F_vs_C`: ΔFN -1.98 [-9.0, +6.0], ΔFP -0.00 [-11.0, +11.0], ΔFNR -0.0049 [-0.0233, +0.0133]
* `E_vs_C`: ΔFN +16.98 [+1.0, +39.0], ΔFP +0.10 [-22.0, +25.0], ΔFNR +0.0407 [+0.0023, +0.0892]
* `F_vs_accepted_base`: ΔFN +10.03 [+1.0, +21.0], ΔFP +0.04 [-12.0, +12.0], ΔFNR +0.0242 [+0.0024, +0.0499]

## 5. Cost of the added deployment operations

| path | rep | baseline p95 ms | candidate p95 ms | increment p95 ms |
|---|---|---:|---:|---:|
| host | rep0 | 0.200 | 0.212 | +0.013 |
| device | rep0 | 0.220 | 0.189 | -0.031 |
| host | rep1 | 0.202 | 0.213 | +0.011 |
| device | rep1 | 0.140 | 0.142 | +0.002 |
| host | rep2 | 0.205 | 0.217 | +0.013 |
| device | rep2 | 0.244 | 0.143 | -0.101 |

Boundary of the claim: the added cost is measured on top of the main model's own forward (hook/copy of the already-computed activation, projection, three fused per-seed heads, score and running episode max). The 16 GiB backbone cannot be resident on a 12 GiB card, so no end-to-end main-model number is claimed; what is measured is exactly the increment plus, separately, the real index-select + host copy inside the real capture forward.

## 6. Validation

12 checks, 12 passed, failures: none. Mutations: {"token_offset_plus_one_detected": true, "block_swap_detected": true, "future_round_digest_mismatch_detected": true, "dev_rows_in_fit_detected": true, "forged_metric_detected": false, "manifest_digest_check_available": true}

## 7. What this stage does not establish

* It is not an independent confirmation: DEV was reused and the accepted baseline was DEV-selected.
* It is not a full-deployment measurement of the main model on GPU (hardware-infeasible here).
* `ctx40k`/`agentlab`/`nothink` sequences are rebuilt by re-executing the frozen serializers; the proof is the per-row accepted digest, which is enforced, not the producer's own file hash.
* No teacher/API annotation was added; the inherited v1 consensus is untouched.

## 8. Artifacts and provenance

```
0124f84 fix(cost): batched fused-head matmul + deferred __main__ + CAPTURE_BLOCKS import; fix(validate): read DEV variants from report['dev']
9aca299 fix(train_eval,validate): bf16->f32 resident/matrix path must shift uint16 bits by 16; torch paths already correct The device-resident arm matrices (HEAD 'single gather per batch') reinterpreted bf16 bit patterns as low bits of float32, feeding ~denormal values and collapsing all arms to constant episode scores (AUC 0.5). Frozen P5 had never been executed. Adds no method change; conversion now equals the torch bfloat16 view (checked maxabs 0.0).
80d9cef migration: prepared-aware validator recapture and hook-cost (source-free GPU path)
8923034 migration: CPU-prepared token-id jobs (writer/reader) + prepared capture producer + prepared canary
502d971 migration: allow {split} placeholder in SDSAFE_EPISODE_DIRS union view
55da3d5 migration: env-gated old-shard pre-skip for capture + union episode view for train/validate (default behavior unchanged)
ddc144b migration: explicit SDSAFE_UPSTREAM_MIRROR path mapping (default unchanged) + PLAN/PROTOCOL/STATUS/path-map
aa8168e import frozen v4 source from CPU HEAD ebed68372028654ca2d9bc33e29beafb915e858f (read-only copy; before path mapping)
```

Manifest: `artifacts/manifests/artifact_hashes.json`. Capture: `artifacts/capture/<split>/episodes/*.npz` (per-episode atomic, digest in `progress.jsonl`). Datasets: `artifacts/dataset/<split>`. Runs: `artifacts/runs.json`. Validation: `artifacts/validation/validation.json`.

## 7. Migration to the new GPU node (RTX 4090, 24 GiB)

All frozen v4 work was re-run on `autodl-container-ujxcycmw77-6fc011ce` after the old
3080 Ti node was retired. Source HEAD `ebed683…` was imported as a clean git and all
upstream inputs were SHA-verified before use (model: 5 shard sizes + index/config/tokenizer
SHA; accepted features `af86fa7a…`; old shards 3612+2028 SHA-ok). Cross-GPU canary on 6 old
episodes: block-4 cos_min 0.9999468, block-8 cos_min 0.9999292, round/row/endpoint bindings
identical. The remaining 4617 TRAIN episodes were captured in 27 GPU-minutes at ~35 k tok/s
(peak 6.34 GiB); every shard was copied to the CPU durable store and SHA-verified (4617/4617,
0 bad). Full TRAIN is now 8229 episodes / 114799 CT rows / 122.57 M tokens, all four scopes.
See `MIGRATION_REPORT.md` for transfers, and the repaired latent bugs below.

Three latent defects in the frozen code were found and fixed (the old run never reached P5):
the bf16→f32 device-resident matrix path was missing `<<16` (collapsed every arm to AUC 0.5),
the fused 3-head cost matmul called `.t()` on a 3-D weight, and the validator read DEV
variants from the wrong report level. The first is why the pre-fix v4 run reported constant
scores; after the fix the same data yields the AUCs above.

## 8. Post-hoc v4b arm G (standardization) — exploratory

Motivated by the accepted probe's own `probe_standardize`, arm **G** applies TRAIN-only
per-feature standardization to F's exact input `[final‖block4‖block8]`, same MLP/recipe/
seeds/folds, and was pre-declared before running. TRAIN-only μ/σ from 93 233 in-loss rows.

| variant | TP | FP | FN | AUC | per-seed AUC |
|---|---:|---:|---:|---:|---|
| G:full | 393 | 80 | **21** | **0.984879** | 0.98474 / 0.98306 / 0.98268 |
| G:ho (fold-0 held out) | 388 | 80 | 26 | 0.978497 | — |
| accepted baseline | 393 | 80 | 21 | 0.981127 | — |

Paired whole-group bootstrap: G vs F → ΔFN −10.1, CI [−20,−1] (G significantly better);
G vs accepted baseline → ΔFN −0.08, CI [−8,8] (statistically tied on the operating point)
with AUC +0.0038. So standardization explains the raw-F gap, but the representation still
does not move FN below the accepted 21 (progress bar ≤14 not met).

## 9. Next method (pre-registered, not started)

The 24 GiB card can execute longer prefixes (measured on a 35 722-token round: 16 blocks,
11.54 GiB, 12 922 tok/s; 24 blocks, 22.93 GiB, 8 586 tok/s). A new dataset version capturing
**block 16 / block 24** from the same frozen forward (never mixed with block 4/8) would test
whether the gap is depth-limited. Full TRAIN+DEV capture is ~3.2 h (block 16) / ~4.8 h
(block 24) over 149.4 M tokens, which exceeds this bounded phase; it is left as the next
resumable step (prepared ids for all 8229 TRAIN + 2028 DEV must be exported first).
