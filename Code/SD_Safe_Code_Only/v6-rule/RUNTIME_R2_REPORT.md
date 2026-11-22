# RUNTIME_R2_REPORT — frozen S2 / FUSION_S2 true streaming cost (cache-only-r2)

**Scope:** cache-only runtime engineering. **New annotation tasks/requests: 0.** No new labels,
no retraining, no threshold search, no label change, no teacher call, no TEST read, no coding-agent
data labelling. All guards/frozen teacher caches/S2 checkpoints/fusion mapping/thresholds are
unchanged. GPU `/root/sdsafe-compute/v6-param` (base commit `6edf1ff`, tag `runtime-r2-base`),
PyTorch 2.14/cu130, `/root/vllm_env_backup/bin/python`.

## 1. Result in one paragraph

The frozen candidate replays exactly: **S2 = 13 FN / 80 FP, AUC 0.989150** and
**FUSION_S2 = 11 FN / 80 FP, AUC 0.990280** from the saved predictions/fusion formula.
A new incremental runtime (`src/sdsafe_v6/runtime_fast.py`) reproduces the r1 reference view and
all sparse/dense features on **24 983/24 983** common-DEV events with **zero** differences
(incl. prefix SHA, prior roles/text, retrieval, value→source, bags, dense). The 3-seed merged
forward changes row logits by at most **1.53e-5** (p99 7.6e-6) versus the saved batched scores;
S2 predictions are **identical**. FUSION_S2 under the frozen threshold is **11 FN / 79 FP** in
the batch-1 stream because the threshold-defining negative episode rounds **2.23e-8 below its own
threshold**; the authoritative saved/eval result remains **11/80**, and no threshold was changed.
Cost target is **not met**: full-stream p95 is **1.94 ms isolated**, **2.32 ms with 3 concurrent
processes** (p50 0.82/1.19 ms). The gap is the honest token-decode burst at the first event of
each round; excluding that burst, later-event p95 is **0.977 ms isolated** / 1.37 ms concurrent.

## 2. Why the r1 cost was incomplete (fixed here)

The r1 `cost.py` charged only `ckpt_S0_v3_seed0` (one `Student.forward`), no 3-seed or fusion,
counted base-cache hits without a real lookup/score update, used only the first 600 sorted events,
re-encoded then decoded a prepared segment, did not maintain a real prior/source index, and its
stream features were not the evaluation features (`prefix_chars`, clipped `full_text`, retrieval
query). `runtime_fast.py` uses the real prepared token stream and all 2 028 episodes / 25 003 CT
rows, all 3 S2 seeds, the frozen base logits, per-seed running max, float32 mean, frozen
FUSION_S2 mapping, and a true incremental message/source index. All decode/parse/retrieval work
is inside the timer; no precomputed view is used as a runtime input.

## 3. Independent replay before optimisation

From `artifacts/dev_episode_S2_v3.npz`, `artifacts/dev_scores_S2_v3.npz`,
`artifacts/baseline_dev.npz`, `artifacts/fusion_frozen_v3.json`:

| method | FN | FP | AUC | threshold |
|---|---:|---:|---:|---:|
| baseline float32 (frozen literal) | 21 | 78 | 0.981127 | −0.6190447211265564 |
| S2 3-seed | 13 | 80 | 0.989150 | −0.7444993853569031 (refit from stream) |
| FUSION_S2 frozen mapping | 11 | 80 | 0.990280 | 0.6624409698441314 |

Row→episode max and per-seed→ensemble replay matches the saved arrays (`0.0` / `0.0`).

## 4. Runtime parity (all common DEV, no sampling)

`scripts/parity_runtime.py --all --fast 1 --with-sha 1` → `artifacts/runtime-r2/parity_views_final.json`

| compared item | events | mismatches |
|---|---:|---:|
| current kind/segment/text/visible/tool/args | 24 983 | 0 |
| user_request, prior (role/text/chars/index) | 24 983 | 0 |
| relevant_tool_blocks (generic retrieval order/ties) | 24 983 | 0 |
| value_sources (exact/normalised, offsets) | 24 983 | 0 |
| counts, prefix_chars, has_reasoning | 24 983 | 0 |
| prefix SHA-256, current SHA-256 | 24 983 | 0 |
| bag indices cur/vis/user/src | 24 983 | 0 |
| dense vector (all 32 values, bitwise) | 24 983 | 0 |

Implementation notes: per-message index; prompt/prefix reuse across rounds when the actual prompt
token prefix is identical (6 972/6 976 transitions; the 4 rewrites fall back to full decode);
incremental current-block word set; exact cached `words`/bags; lazy per-block source features; a
single merged 3-seed embedding/head forward. The 20 rows without a frozen student view are
`no_new_emission` events: decode/index advance only, no score (all 2 028 episodes and 25 003 rows
retained; base running max over scored rows matches the saved base per-seed episode score to 0.0).

## 5. Accuracy replay of the optimised path

`scripts/replay_runtime.py` → `artifacts/runtime-r2/replay_s2_fusion.json`

* row logit max |diff| vs saved scores: **1.52587890625e-5**, p99 7.63e-6
* per-seed episode max |diff|: **7.63e-6**; ensemble |diff|: **5.72e-6**
* S2: 13 FN / 80 FP, AUC 0.989150, **0 prediction changes**
* FUSION_S2 with frozen `tau=0.6624409698441314`: **11 FN / 79 FP** (saved/eval **11/80**).
  Only changed episode: `ctx40k::agentlab-40dd0f4b501622d0c2c1a2b28da929c4` (negative), stream
  fused `0.6624409475899076` vs saved/frozen `0.6624409698441314` (delta −2.23e-8). Refitting tau
  on DEV would restore 11/80 but is **not** done (implicit tau change forbidden). Parent decision
  required on whether the ship threshold is the frozen float64 literal or a re-derived float32
  boundary; no learning artefact was changed.

## 6. Cost — full stream, 3 OS processes × 3 repeats

`scripts/bench_runtime.py`: 25 003 events / 2 028 episodes per pass, causal episode order,
24 983 scored + 20 no-emission; each pass rebuilds all data caches (warm-up only warms code/kernels
and is followed by a cache clear). Included: token decode of every new emission, index append,
current-block finalisation, retrieval, value→source, hashed features, H2D, 3-student forward,
base-cache lookup/score update, per-seed max, float32 ensemble, fusion update.

| condition | p50 | p95 | p99 | throughput |
|---|---:|---:|---:|---:|
| 1 process isolated, median of 3 | 0.822 | **1.938** | 2.877 | 1 020 ev/s |
| 3 processes concurrent, median | 1.193 | **2.316** | 3.345 | 767 ev/s |
| 3 processes, worst repeat p95 | — | 2.345 | — | — |

Component p95 (isolated / concurrent): view+features 1.340 / 1.435 ms, model 0.554 / 1.047 ms,
fusion+max 0.056 / 0.058 ms, no-emission 0.017 / 0.020 ms. p50 components: view+features
0.264/0.269, model 0.507/0.847.

Slices (isolated): first event of a round (9 004) p95 **2.461 ms**; later events (15 979) p95
**0.977 ms**; content_segment p95 1.140; tool_call p95 2.467; scopes cot 2.387, ctx40k 1.135,
nothink 1.777. Length buckets (prefix chars) 0–2k 1.001, 2–8k 1.253, 8–20k 2.067, ≥20k 2.595.
Concurrent slices are ~1.4–1.5× higher.

**Target `p95 ≤ 1 ms` is not met.** The dominant extra cost versus r1 is honest: r1 decoded only a
prepared segment, while the real stream must decode all newly emitted tokens since the previous
event; the first event of a round carries the accumulated generation (up to tens of thousands of
tokens). Later-event p95 is already 0.977 ms isolated. Remaining optimisation headroom is the
3-seed GPU launch/sync chain (0.51 ms p50 isolated) and the Python feature path; no threshold,
model, or view semantics were traded for speed.

Peak resources: GPU 34.8 MB allocated / 48.2 MB reserved per process; CPU max RSS ~2.07 GB per
process (tokenizer + caches); 3 CUDA processes shared one RTX 4090 24 GiB. No background jobs
after measurement; GPU 0 MiB.

## 7. Tests / mutations

`PYTHONPATH=src python -m unittest discover -s tests` → **21/21 OK**
(`logs/tests_runtime_r2.log`), including new `tests/test_runtime_fast.py`: incremental word-set
and bag-memo equality; duplicate identical blocks keep distinct indices; future-suffix mutation
does not change the event view/features; forged ChatML in a tool response stays data; no-emission
advance does not score; merged 3-seed net matches three separate students (<5e-6); frozen teacher
cache SHA unchanged. Existing view-provenance/aux/guard/bootstrap tests still pass.

## 8. Artifacts / identity

Local: `artifacts/runtime-r2/` (`parity_views_final.json`, `replay_s2_fusion.json`,
`replay_stream_scores.npz`, `cost_r2_p0..2.json`, `cost_r2_isolated.json`,
`cost_r2_summary.json`). Source: git commit recorded in `HANDOFF_R2.json`; base tag
`runtime-r2-base` = `6edf1ff`. Teacher caches SHA-256 unchanged
(`aux2 9f6f4d…a20c`, `aux_calls_v2 089e48…0f2b`, `pilot400 ef11c7…b3ef`). S2 checkpoints
`ckpt_S2_v3_seed{0,1,2}.pt` were only read. The bundle is atomically copied to the CPU store
and SHA-read back (see `HANDOFF_R2.json` / CPU sync file).

## 9. Residual risks / decisions for parent

1. **Fusion boundary (decision):** frozen-tau batch-1 stream gives 11/79; saved/eval is 11/80.
   Cause is pure float32 batching rounding at a threshold-defining negative (2.23e-8), not a
   logic/model change. No tau adjustment was made.
2. `p95 ≤ 1 ms` not met when the first-event token-decode burst is charged; steady-state later
   events are 0.977 ms p95 isolated. Further gains require a CUDA-graph/fixed-shape student or
   streaming token decode attribution, both beyond this bounded run.
3. FUSION_S2 remains exploratory DEV reuse (unchanged); S2/FUSION are still short of FN≤8.
4. The 20 no-view rows are handled by an explicit no-new-emission policy, not GT fallback; the
   frozen evaluation already excluded them from student scoring.
5. Teacher labels remain model-generated; no label repair was attempted.
