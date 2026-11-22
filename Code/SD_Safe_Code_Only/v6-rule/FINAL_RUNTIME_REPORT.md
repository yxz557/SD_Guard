# FINAL_RUNTIME_REPORT — bounded final mechanical round on the frozen runtime

**Stage:** `runtime-final` (one bounded 45-min mechanical round on top of `runtime-r4`).
**Scope:** cache-only runtime substrate only. **No** new annotation task, **no** API request,
**no** new label, **no** training, **no** TEST read, **no** model/scaler/fusion/threshold/feature
semantics change. Python `/root/vllm_env_backup/bin/python` (3.12.3, torch 2.14.0+cu130),
RTX 4090 24 GiB, single GPU writer `/root/sdsafe-compute/v6-param`, `PYTHONDONTWRITEBYTECODE=1`,
2 threads, no other GPU job. CPU authoritative copy:
`/root/autodl-tmp/sdsafe-lowcost-authorization-v6-param/runtime-final/`.

## Bottom line

The r4 CUDA-graph backend was replaced by an **equivalent single-transfer static-graph backend**
(one pinned host block, one blocking H2D, unchanged graph math). Evidence-selected from a CPU
profile of the detector path, not from guesswork. Result over the full 25 003-event /
2 028-episode DEV scope, 3 independent OS processes × 3 repeats (9 passes):

| candidate (integrable scope) | p50 | **p95 median** | **p95 worst** | p99 | ≤1 ms strict? |
|---|---:|---:|---:|---:|---|
| **R rawtext** (parse/JSON/role/source all charged) | 0.372 | **0.9932** | **1.0260** | 1.625 | **no (median yes, worst +2.6 %)** |
| **T integrated** (detector + measured adapter) | 0.371 | **0.9930** | **1.0136** | 1.628 | **no (median yes, worst +1.4 %)** |
| T pure (adapter excluded; no caller in this source) | 0.347 | 0.8721 | 0.8845 | 1.488 | upper bound only, **not claimed** |

`cost_passed=false` under the pre-registered **worst-of-9 ≤ 1.0 ms** definition. The definition was
not relaxed, τ was not moved, no ε and no slice/coversion trick. This is the honest final state:
**~0.99 ms p95 median, 1.01–1.03 ms worst**, i.e. a ~1.4–2.6 % miss (r4 was 3.7–6.8 %).
Accuracy is unchanged and identical to r4: **FUSION_S2 12 FN / 79 FP, AUC 0.9903471437**,
S2 13/79, frozen τ `0.6289290373189838` (`0x1.4202fc9d94608p-1`), `>=`, all 9 passes.

## 1. Evidence → change (only 1–2 hotspots touched)

CPU `cProfile` of the R path (chosen only to *select* hotspots; profile times are not reported as
cost), then a wrapped per-stage decomposition on 150 whole episodes / 1 851 scored events:

| stage | p50 | p95 | mean |
|---|---:|---:|---:|
| graph backend `forward_gpu` (stage+copy+replay) | **0.155** | **0.188** | 0.157 |
| `fast_extract` (features/bags) | 0.123 | 0.283 | 0.142 |
| `view.parse_blocks` (first event of round) | 0.071 | 0.238 | 0.097 |
| segment decode | 0.010 | 0.166 | 0.044 |
| value sources + retrieval | 0.021 | 0.095 | 0.033 |

The graph backend was the only *constant, mechanical* cost (≈0.19 ms p95 on **every** event) whose
reduction cannot change semantics, so only it was touched. Micro-benchmark inside it:

| component | before | after |
|---|---:|---:|
| `_stage` (host packing) | 0.068 ms | 0.049 ms |
| `_copy` (5 pageable H2D) | 0.048 ms | **0.015–0.023 ms** (1 pinned H2D) |
| graph replay | 0.007 ms | 0.006 ms |
| **total** | **0.123 ms** | **0.078 ms** |

Changes (all in `src/sdsafe_v6/runtime_r4.py`, `TriStudentGraph` only):

1. **One pinned staging block per capacity.** Indices (int32), dense features + source count
   (float32) and bag offsets/counts (int64) now live in one pinned host buffer with numpy views;
   device side is one buffer with typed views. Per event: **1 blocking H2D copy** instead of 5
   `from_numpy` pageable copies. Blocking (not `non_blocking`) is deliberate — the same host block
   is rewritten by the next event, so the transfer must have completed before reuse; there is no
   async reuse race and no untimed overlap.
2. **Vectorised packing.** Concatenate the real bags once, shift by the seed row base
   (`np.add(cat, g*B1, out=...)`), offsets/counts from the same length vector (18 scalar writes
   instead of `arange/ravel` temporaries), dense standardisation written in place into the pinned
   view. The per-seed/per-bag Python loop is gone; **values, dtype, padding and arithmetic are
   unchanged**.
3. `dense` and the "source present" count now share one 33-float buffer; the count is stored as an
   exact small float32 (`clamp(min=1)` still applied), which is the same arithmetic as the previous
   int64→`.float()` path.
4. `scripts/bench_r4.py`: `--art-name` (default `runtime-r4`) so the final round writes
   `artifacts/corrected-r3/runtime-final/` without touching r4 evidence. No timing-scope change.

Not touched: `view.parse_blocks`, `_true_starts`, `select_current_call`, `scan_tool_calls`
(historical parser), `fast_extract`/dict caches, `features.extract`, teacher/scaler/fusion, and no
source index or parse step was moved outside the timer. Base (`float32` mean, τ −0.6190447211265564)
and fusion math are byte-for-byte the same code path.

### Why not more (residual headroom, not attempted)

The remaining p95 tail is first-event-of-round / long COT: `view.parse_blocks` (0.24 ms p95),
`fast_extract` (0.28 ms p95: whole-current-text regex + hash per event) and segment decode
(0.17 ms p95). A further win needs an incremental boundary/word-hash path (a parser/algorithm
change) or a controller that genuinely supplies a typed message list — explicitly out of scope for
this bounded round. Steady-state events are already **0.567 ms p95** (R) / **0.566 ms** (T).

## 2. Cost — full scope, 3 processes × 3 repeats, batch 1, cold process then warm repeats

Every round/event re-decodes the causal stream; new-message/role parsing, current JSON, value
sources, word hashes, all 3 seeds, real base cache, per-seed max, float32 seed mean and float64
fusion are inside the timer. All 9 004 first-events-of-round, all long contexts, empty rows and
new-tool-message index costs are included. No view is pre-warmed (fresh `TextCache`/`BlockProcessor`
per pass; only 2 warm-up episodes before the measured passes).

| mode / variant | p50 | p95 med | p95 worst | p99 | full p95 (incl. shared controller decode) | wall ev/s |
|---|---:|---:|---:|---:|---:|---:|
| R rawtext gpu | 0.372 | 0.9932 | 1.0260 | 1.625 | 2.261 | 1 028 |
| T integrated gpu | 0.371 | 0.9930 | 1.0136 | 1.628 | 2.285 | 1 035 |
| T detector-only gpu | 0.347 | 0.8721 | 0.8845 | 1.488 | — | 1 035 |

Component p95 (9-pass): `view_feat` 0.834 (R), `model` **0.135** (was 0.186 in r4),
`shared` decode 1.334 (charged only in `full`), `episode_final` D2H 0.122 (charged to the last
event of the episode), adapter 0.167 (T only).

### Slices (R gpu, 9-pass median of per-pass p95)

| slice | n | p50 | p95 |
|---|---:|---:|---:|
| first event of round | 9 004 | 0.524 | 1.369 |
| later events (steady) | 15 999 | 0.328 | **0.567** |
| content_segment | 17 190 | 0.339 | 0.658 |
| tool_call | 7 813 | 0.497 | 1.374 |
| scope cot / ctx40k / nothink | 8 091 / 8 326 / 8 586 | 0.433 / 0.343 / 0.372 | 1.335 / 0.644 / 0.773 |
| prefix ≥ 20 k chars | 6 002 | 0.460 | 1.445 |
| 8 k–20 k chars | 9 873 | 0.373 | 0.971 |

Resources: graph capacity 768/1536/3072/4096 used 7 735/12 114/4 968/186 calls (R parity pass),
**0 fallbacks** in every pass; GPU peak allocated 94 067 712 B (89.7 MiB), reserved 160 MiB,
0 MiB after run; CPU max RSS 2.13 GiB; no background job after the run.

## 3. Full parity against the frozen corrected-r3 reference (all 25 003 events, R and T)

`scripts/parity_r4.py` (unmodified logic), 65.9 s, both modes identical:

| layer | result |
|---|---|
| views (role, call binding/ordinal/segment, current JSON, prior blocks, retrieval, value sources, flags, prefix/current hashes) | **0 bad rows** |
| bag indices (cur/vis/user/src) | **0 mismatches** |
| dense features (32 float32) | **max |Δ| = 0.0** (bitwise) |
| 3-seed CT logits | max |Δ| = **3.0517578125e-05** (float32 noise) |
| per-seed episode max / float32 seed mean | max |Δ| = 5.7220458984375e-06 |
| float64 fusion | max |Δ| = **2.841287660704239e-07** |
| frozen τ applied with `>=` | FUSION_S2 **12 FN / 79 FP** (reference 12/80) ; S2 **13/79** (reference 13/80) |
| graph fallbacks | **0** |

The single 80→79 FP difference is the same exact-τ boundary episode as r4
(`nothink_ctx40k::agentlab-47443fea3548818726d3e2187fc2607e`, label 0, fused
`0.6289290136415866` = `0x1.4202fbd2313aep-1`, margin −2.37e-8 below τ); the reference sits exactly
on τ. No ε, no rounding, no τ change. AUC `0.9903471436524612` (fusion), `0.9876218953720166` (S2),
baseline 21 FN / 78 FP at τ −0.6190447211265564.

Caller invariants inherited from r3 and still passing (unit tests): differential token chunk steps /
incremental word state, cold text cache, forged ChatML role injection inside tool data,
JSON-embedded tags, future-suffix mutation, own-segment semantics, multi-call rebinding,
chunk/cache regression. Frozen checkpoints/teacher caches/scaler/fusion JSON:
**60/60 SHA-256 identical to HANDOFF_R3** (`frozen_sha_final.json`, 0 mismatches) before and after
the round.

## 4. Validation

- Unit tests: **40/40 OK** (`logs/tests_final.log`); the new backend is exercised by the existing
  graph/cap-boundary/typed=rawtext/frozen-τ tests.
- Cost equivalence vs the r4 scores (400-episode cross-check, same scope):
  `per_seed`, `ensemble`, `fused` **max |Δ| = 0.0** — the faster staging is numerically identical,
  not merely close.
- Full-cost pass scores: `stud_maxdiff_vs_saved` 0.0 and `fused_maxdiff_vs_saved` 0.0 in all 9
  passes, so the reported 12/79 is the frozen-τ application on the real scores.
- Output: `FINAL_RUNTIME_JSON` printed by `scripts/final_runtime_json.py`.

## 5. Artifacts (`artifacts/corrected-r3/runtime-final/`, 26 files, 2.7 MiB)

`cost_r4_p{0,1,2}.json`, `scores_{R,T}_gpu_p{0,1,2}_r{0,1,2}.npz`, `cost_final_summary.json`,
`parity_r4.json`, `parity_r4_row_logits.npz` (row logits/per-seed/ensemble/fused + τ hex),
`accuracy_final.json`, `frozen_sha_final.json`, `MANIFEST_FINAL.json`, `SHA256SUMS.txt`.
CPU authoritative copy (atomic tar → `sha256sum -c` 26/26 → rename → readback 10/10 identical):
`/root/autodl-tmp/sdsafe-lowcost-authorization-v6-param/runtime-final/` (`CPU_SYNC_FINAL.json`).

## 6. Residual risks

1. **Strict cost gate still not met**: worst-of-9 p95 1.026 ms (R) / 1.014 ms (T integrated);
   median 0.993 ms. Quote both; do not present 0.99 ms as a strict pass.
2. T pure 0.872/0.885 ms remains an upper bound: the prepared source has no separate
   controller-provided message list, so it is not a demonstrable integration.
3. The remaining p95 tail is parse/feature/decode on first events and ≥20 k-char COT contexts
   (within-slice p95 up to 1.45 ms) — a bounded further win needs an incremental parser/feature
   path (algorithm change) or a typed caller; neither was in scope here.
4. The 80→79 FP drift is a float32 backend boundary effect on an exact-τ tie; accuracy is otherwise
   bit-identical and τ was not adjusted.
5. Base probes remain in-sample on TRAIN and DEV accuracy is exploratory reuse of the frozen r3
   menu; this round changes nothing about selection or accuracy claims.
6. `artifacts/corrected-r3/runtime-r4/parity_r4.json` was regenerated by the re-run of the frozen
   parity script: the numeric payload is identical to r4 (only the elapsed-seconds field differs);
   the r4 artifacts were otherwise untouched.

## 7. Reproduction

```bash
cd /root/sdsafe-compute/v6-param
PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python scripts/parity_r4.py
for p in 0 1 2; do
  PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python scripts/bench_r4.py \
    --process-id $p --repeats 3 --modes R,T --variants gpu --warmup-episodes 2 \
    --art-name runtime-final
done
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python scripts/summarize_final.py
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python scripts/accuracy_r4.py
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python -m unittest discover -s tests
```
