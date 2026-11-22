# RUNTIME_R4_REPORT — mechanically equivalent runtime for frozen corrected-r3

**Scope:** cache-only runtime optimization of the *frozen* corrected-r3 detector. No training, no
new labels, no API requests, no threshold/feature/model/fusion-semantics change, no TEST read.
Old r1/r2/r3 artifacts are read-only and retained under their old names; all new code/results use
the `runtime-r4` name. GPU: RTX 4090 24 GiB, `/root/vllm_env_backup/bin/python`
(torch 2.14.0+cu130), `PYTHONDONTWRITEBYTECODE=1`, no other GPU job. New results are
CPU-authoritative under
`/root/autodl-tmp/sdsafe-lowcost-authorization-v6-param/runtime-r4/`.

**Bottom line.** The new static-shape CUDA-graph backend and the corrected timing scope are in
place. Full common-DEV parity is exact on views/bags/dense and within float32 noise on logits.
The honest integrable detector p95 is **1.037 ms (R rawtext, parse charged) / 1.039 ms
(T typed + measured adapter)** with 9-pass worst **1.062 / 1.068 ms**; that is **3.7–6.8 % above
the original 1 ms gate**, so `cost_passed=false` (borderline, not hidden). The pure typed path is
0.916 ms median / 0.940 ms worst, but this source has no separate caller-provided message list,
so it is reported as an upper bound, not as an achieved integration. Accuracy is unchanged:
FUSION_S2 **12 FN / 79 FP**, AUC **0.9903471437** (reference 12/80 due to one exact-τ tie that
moves 2.4e-8 below τ under the graph backend; no threshold was touched).

## 1. What changed (mechanical only)

| file | change | equivalence |
|---|---|---|
| `src/sdsafe_v6/runtime_r4.py` | new `TriStudentGraph`, `TriStudentCPU` wrapper, `EpisodeStreamR4`, `FrozenScorerR4`, `typed_source_blocks` | same three frozen S2 checkpoints, same embedding-bag mean / LayerNorm / GELU / linear / head / float32 arithmetic |
| `src/sdsafe_v6/view.py` | `_true_starts` no longer copies `prefix_text[:m.start()]` per candidate; it checks the trailing `<|im_end|>` in place | same returned starts; full parity 25 003/25 003 |
| `scripts/bench_r4.py` | corrected timing scope (A/R/T), 3 OS processes × 3 repeats, cold/warm slices | all detector work inside the timer; shared decode/adapter measured separately and charged in `full` |
| `scripts/parity_r4.py` | full 25 003-event layer-by-layer parity | views/roles/call-bind/retrieval/value-sources/bags/dense/logits/episode-max/mean/fusion |
| `scripts/pilot_r4.py`, `summarize_r4.py`, `accuracy_r4.py`, `manifest_r4.py` | pilot, 9-pass summary, accuracy, SHA manifest | no scoring change |
| `tests/test_runtime_r4.py` | 8 new tests (backends, cap boundaries, typed=rawtext, frozen τ/teacher) | 40/40 tests pass |

Model files, teacher caches, scaler and both frozen fusion JSONs are byte-identical to r3
(`frozen_sha_r4.json`: 0 mismatches against `HANDOFF_R3.json`). The graph backend adds one
all-zero embedding row per seed purely for padding; real indices, bag means, channel order,
source-bag mean, float32 per-seed logits, float32 seed mean and float64 fusion are unchanged.

### Graph backend

- Three frozen students are evaluated as one batched graph, exactly as `TriStudent` did.
- Real bag indices are packed compactly into the smallest fixed capacity of
  `{768, 1536, 3072, 4096}` (per-event total across 3 seeds); dynamic bag offsets are supplied as
  an input. Padding uses the zero row; `sum / true_count` is algebraically the reference
  `EmbeddingBag(mode="mean")` (adding exact zeros does not change a float sum; empty bags stay
  zero). Missing source slots are excluded by `src_present`, preserving the reference mean over
  the source bags that actually exist.
- 0 fallbacks over all 25 003 events in every pass. Capacity use (one process, all passes):
  768: 185 640, 1536: 290 736, 3072: 119 232, 4096: 4 464.

### Pilot before the full run (8 stratified whole episodes, 199 scored events)

| backend | forward p50 | forward p95 |
|---|---:|---:|
| CPU, 1 thread | 0.421 ms | 1.206 ms |
| CPU, 2 threads | 0.384 ms | 0.414 ms |
| original merged CUDA | 0.456 ms | 0.484 ms |
| static-shape CUDA graph (with per-event D2H) | 0.193 ms | 0.238 ms |

CPU-vs-GPU logit max |Δ| = 2.29e-5 (< 1e-4 numeric check). CPU 2-thread is competitive per
forward but has a heavier p95 and no GPU-resident episode accumulation, so the graph backend was
chosen for the full run. Pilot also showed the dominant non-model cost is view/feature/parse, not
the student, which is why the timing-scope correction matters more than a further model tweak.

## 2. Timing scope (the r3 defect, corrected)

r3 mode B called `view.parse_blocks(prompt_override)` **before** its timer and counted the
prompt role/JSON/source boundary recovery as “controller shared”. r4 measures three explicit
scopes:

1. **A — tokens-only.** Every round/event re-decodes the causal token stream; decode + parse +
   index + retrieval + value sources + features + 3 students + base lookup + max/mean/fusion are
   inside the timer. Requirement: report honestly, not force under 1 ms.
2. **R — rawtext interface (the integrable path for this source).** The controller already has
   the decoded prompt and assistant deltas; that shared decode is measured separately and charged
   in `full`. **All detector work — `view.parse_blocks` of the full prompt, JSON/role/source
   boundary recovery, current-call selection, retrieval, value sources, hashing, feature
   transfer, all three students, max/mean/fusion — is inside the detector timer.**
3. **T — typed interface.** Concrete contract: the caller passes an ordered
   `[(role, text), …]` list at its prompt-formatter boundary plus the raw current assistant
   text/delta; the detector never re-parses the full prompt. This source store only has token
   ids, so `typed_source_blocks()` recovers the boundaries with `view.parse_blocks`; that adapter
   is **measured separately and charged** in `T integrated`. No eval view, retrieval result or
   value-source annotation is reused as a free field. Pure T (adapter excluded) is shown only as
   an upper bound; it is **not** claimed as an actual integration because no direct typed caller
   is present in this source.

`full` = detector + shared controller decode (+ adapter for T). Per-event logits are either
copied to CPU each event (`sync`, r3 behaviour) or kept on the GPU and reduced once per episode
(`gpu`, deployment streaming). Both are reported; all transfers are counted, and the episode-end
D2H is charged to the last event and reported separately as `episode_final`.

## 3. Full parity (all 25 003 current events / 2 028 episodes / 166 groups)

Run for both R and T; results identical.

| layer | result |
|---|---|
| views: role, call binding/ordinal/segment, current JSON, prior blocks, retrieval, value sources, flags, prefix/current hashes | **0 bad rows** |
| bag indices (cur/vis/user/src) | **0 mismatches** |
| dense features (32 float32) | **max |Δ| = 0.0** (bitwise) |
| 3-seed CT logits | **max |Δ| = 3.0517578125e-05** (float32; 1e-4 numeric bound) |
| per-seed episode max | **max |Δ| = 5.7220458984375e-06** |
| float32 seed mean | **max |Δ| = 5.7220458984375e-06** |
| float64 fusion | **max |Δ| = 2.841287660704239e-07** |
| graph fallbacks | **0** |

Parity was run three times with identical results. Caller invariants inherited and re-checked:
different token chunk steps / incremental word state, cold text cache, forged ChatML role
injection inside tool data, JSON-embedded tags, future-suffix mutation — all still pass
(r3 validation + r4 tests).

## 4. Accuracy at the frozen thresholds (no re-calibration)

| method | FN | FP | AUC | threshold |
|---|---:|---:|---:|---|
| frozen baseline | 21 | 78 | 0.9811267951 | −0.6190447211265564 |
| S2, r3 reference | 13 | 80 | 0.9876218954 | −0.7427853941917419 |
| **S2, r4 graph** | **13** | **79** | **0.9876218954** | same, `>=` |
| FUSION_S2, r3 reference | 12 | 80 | 0.9903471437 | 0.6289290373189838 (`0x1.4202fc9d94608p-1`) |
| **FUSION_S2, r4 graph** | **12** | **79** | **0.9903471437** | same exact τ, `>=` |

One boundary tie explains the 80→79 FP: episode
`nothink_ctx40k::agentlab-47443fea3548818726d3e2187fc2607e` (label 0) has reference fused
exactly `0.6289290373189838` (on τ, counted FP) and r4 fused `0.6289290136415866`
(`0x1.4202fbd2313aep-1`, margin −2.37e-8), so it falls just below τ. **No ε, no rounding, no
τ adjustment.** All 9 cost passes report fusion 12/79 and S2 13/79; AUCS are bit-identical to r3.

## 5. Cost — 3 sequential OS processes × 3 repeats, full 25 003 events, batch 1

No other GPU job. Each process is independent; every pass starts with a fresh `TextCache` /
`BlockProcessor` (no pre-warmed 25 k semantic cache). p95 values are medians over the 9 passes;
`worst` is the worst of the 9. `full` includes shared controller decode (+ adapter for T).

| mode / variant | p50 | p95 median | p95 worst | p99 | full p95 median | wall throughput |
|---|---:|---:|---:|---:|---:|---:|
| A tokens-only, gpu | 0.456 | **1.669** | 1.736 | 2.810 | 1.669 | 1 185 ev/s |
| A tokens-only, sync | 0.539 | 1.724 | 1.798 | 2.799 | 1.724 | 1 059 ev/s |
| R rawtext, gpu | 0.416 | **1.037** | **1.062** | 1.674 | 2.283 | 991 ev/s |
| R rawtext, sync | 0.490 | 1.107 | 1.129 | 1.746 | 2.368 | 906 ev/s |
| T typed detector only, gpu | 0.391 | **0.916** | **0.940** | 1.532 | — | 1 009 ev/s |
| T **integrated** (detector+adapter), gpu | 0.411 | **1.039** | **1.068** | 1.645 | 2.289 | 1 009 ev/s |
| T typed detector only, sync | 0.464 | 0.996 | 1.004 | 1.628 | — | 900 ev/s |
| T integrated, sync | 0.482 | 1.123 | 1.132 | 1.727 | 2.377 | 900 ev/s |

Comparison with r3 (same machine, same full DEV): A detector p95 2.172 → **1.669 ms (−23 %)**;
B detector p95 1.371 → **R 1.037 ms (−24 %, now including parse)**; B full p95 2.842 →
**2.283 ms (−20 %)**. The median is halved (B p50 0.801 → R 0.416 ms); the remaining tail is
view/feature/parse work on first events and long COT/tool-call contexts, not the student.

### Cold vs steady / slices (R gpu, 9-pass median of per-pass p95)

| slice | n | p50 | p95 |
|---|---:|---:|---:|
| first event of round (cold message finalisation/index) | 9 004 | 0.570 | **1.431** |
| later events (steady) | 15 999 | 0.375 | **0.625** |
| content_segment | 17 190 | 0.387 | 0.715 |
| tool_call | 7 813 | 0.540 | 1.437 |
| scope cot / ctx40k / nothink | 8 091 / 8 326 / 8 586 | 0.482 / 0.384 / 0.414 | 1.387 / 0.698 / 0.825 |
| prefix ≥ 20 k chars | 6 002 | 0.514 | 1.499 |
| zero-span content row (P1) | 59 | 0.442 | 0.633 |

No first event, long context, empty row or new-tool-message index cost is excluded.
T integrated slices are the same within noise plus the adapter p95 ≈ 0.168 ms.

### Resources / transfer

- Graph capacity use: 768 / 1536 / 3072 / 4096 = 185 640 / 290 736 / 119 232 / 4 464 calls per
  process; **0 fallbacks**.
- GPU peak allocated 94 070 784 B (89.7 MiB), reserved 167 772 160 B (160 MiB); 0 MiB after run.
- CPU max RSS ≈ 2.13 GiB per process; Python threads 2.
- Per-event input transfer is only the packed real indices + offsets/counts/dense; episode-end
  output copy p95 ≈ 0.116 ms (charged to the episode's last event and reported as `episode_final`).

## 6. Gate

| candidate | integrable p95 median | worst | ≤ 1 ms? |
|---|---:|---:|---|
| R rawtext (parse/JSON/role/source charged) | 1.037 | 1.062 | **no** |
| T integrated (detector + measured adapter) | 1.039 | 1.068 | **no** |
| T pure (adapter excluded; no direct caller in this source) | 0.916 | 0.940 | technically yes, **not claimed** |
| A tokens-only (reported only) | 1.669 | 1.736 | no |

`cost_passed=false`. This is a **borderline finite tradeoff**: the corrected-scope detector is
within 3.7–6.8 % of the original 1 ms bar on the only demonstrable integrations. The 1 ms bar was
not relaxed, τ was not re-calibrated, no epsilon was introduced, and no event slice was dropped.
To cross 1 ms one needs a cheaper view/feature/parse path (incremental boundary recovery on
successive prompt prefixes) or a caller that genuinely supplies typed messages; both are future
engineering, not a threshold change.

## 7. Validation / artifacts

- `artifacts/corrected-r3/runtime-r4/parity_r4.json` + `parity_r4_row_logits.npz`
  (row logits float32, per-seed float32, ensemble float32, fused float64, τ/hex).
- `cost_r4_p{0,1,2}.json`, `scores_{mode}_{variant}_p{id}_r{rep}.npz`,
  `cost_r4_summary.json`, `accuracy_r4.json`, `pilot_r4.json`, `frozen_sha_r4.json`,
  `MANIFEST_R4.json`.
- Unit tests `logs/tests_r4.log`: **40/40 OK** (32 inherited + 8 new).
- Frozen inputs: teacher caches, checkpoints, scaler, both fusion JSONs and all old r3 artifacts
  match `HANDOFF_R3.json` (0 mismatches). No cache line edited; no retry; no annotation request.
- TEST material: not read. Background jobs after the run: none (GPU 0 MiB).
- CPU authoritative copy: `/root/autodl-tmp/sdsafe-lowcost-authorization-v6-param/runtime-r4/`
  (atomic tar → SHA256SUMS verify → rename → readback; see `CPU_SYNC_R4.json`).

## 8. Residual risks

1. **Cost gate still fails at the honest scope** (1.04–1.07 ms p95). T pure passes but is not a
   demonstrated integration in this source; do not quote 0.94 ms as achieved.
2. The 80→79 FP change is a genuine float32 backend drift on an exact-τ tie (margin −2.4e-8).
   Accuracy is otherwise bit-identical; the frozen τ was not adjusted.
3. The typed contract requires a controller that actually exposes its message list. If it can
   only expose decoded text, the R number (parse charged) is the correct one.
4. Long COT / ≥20 k-char contexts and first-event-of-round remain the p95 tail (~1.43–1.50 ms
   within those slices); later events are already 0.625 ms p95.
5. The CPU backend was piloted but not used for the formal run; 2-thread CPU is close in median
   but has a heavier p95 and no GPU-resident episode accumulation.
6. Base probes remain in-sample on TRAIN and the r3 DEV accuracy is exploratory reuse; r4 does
   not change or re-confirm the selection story.

## 9. Reproduction

```bash
cd /root/sdsafe-compute/v6-param
PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python scripts/pilot_r4.py
PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python scripts/parity_r4.py
for p in 0 1 2; do
  PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python scripts/bench_r4.py \
    --process-id $p --repeats 3 --modes A,R,T --variants gpu,sync --warmup-episodes 2
done
PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python scripts/summarize_r4.py
PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python scripts/accuracy_r4.py
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 /root/vllm_env_backup/bin/python -m unittest discover -s tests
```
