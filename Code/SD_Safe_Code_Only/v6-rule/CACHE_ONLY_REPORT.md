# CACHE_ONLY_REPORT — SD-Safe v6 parameter authorization, cache-only-r1

**Run:** 2026-11-14 (GPU RTX 4090 24 GiB, PyTorch 2.14/cu130, Transformers 5.20,
`/root/vllm_env_backup/bin/python`).
**Scope:** cache-only follow-up. **New annotation tasks: 0. New annotation API requests: 0.**
No retry/health-probe/second-judge/label-repair/provider change/test annotation. No TEST material
was read. No old SD-Safe/v1–v5 source or data was modified.

## 1. Executive result

The accepted frozen baseline replays **21 FN / 78 FP**, AUC 0.98113 on common DEV (2 028 episodes,
414 positives, 1 614 negatives, 166 groups). With the v3 provenance/QC data:

| method | FN | FP | AUC | note |
|---|---:|---:|---:|---|
| baseline (frozen float32) | 21 | 78 | 0.981127 | accepted comparator |
| S0 main only | 17 | 80 | 0.988448 | event model, no aux |
| S1 main + aux λ0.3 | 13 | 80 | 0.988698 | 3-seed formal |
| S2 main + aux λ1.0 | 13 | 80 | 0.989150 | 3-seed formal |
| S3 shuffled-aux control | 19 | 80 | 0.985317 | capacity control |
| S4 per-argument field encoder | 13 | 80 | 0.989289 | **exploratory, post-DEV** |
| FUSION_S0 frozen equal-logit | 18 | 80 | 0.989265 | rule-selected fold-0 arm |
| FUSION_S2 frozen equal-logit | **11** | 80 | **0.990280** | **exploratory DEV reuse** |

The progress bar `FN ≤ 14` is met by S1/S2/S4 and FUSION_S2; the strong bar `FN ≤ 8` is **not**
met. S1/S2 improve FN by 8 over baseline but the paired whole-group 95 % CI is `[-18, +2]`
(not excluding zero). FUSION_S2 improves FN by 10, CI `[-19, -1.98]` (excludes zero), with
FP +2 (CI `[-11, +16]`). AUC rises while FN falls for S1/S2/FUSION_S2; S0's AUC rise also comes
with a smaller FN fall. S3 is clearly worse than S1/S2, so the gain is not explained by extra
capacity alone.

Cost target is **not met**: honest per-event p50 1.22 ms, p95 1.84 ms (worst repeat 2.02 ms)
versus `≤1.0 ms` p95. This is reported as a tradeoff, not hidden by changing the threshold.

## 2. Process / identity / safety

- Current researcher git: `/root/sdsafe-compute/v6-param` (the only writer here).
- Old paused training PID 37269 verified: state `T`, kernel starttime `2805191632`, argv
  `python -m sdsafe_v6.run_arms --arms S1,S2,S3`; S1/S2 fits/artifacts copied to
  `artifacts/archive_pre_cacheonly/`; PID killed with SIGKILL, **never SIGCONT**.
- Old annotation PIDs 38031 and launcher 38029 remain terminated; not resumed.
- After training: no sdsafe training/annotation process, **no GPU compute process**, GPU 0 MiB.
- System disk 8.3 GiB free; `/dev/shm` 42 GiB free; no large cache written to system disk.
- All Python runs used `PYTHONDONTWRITEBYTECODE=1`.

## 3. Frozen teacher cache QC (no new calls)

Three landed caches: 11 190 response lines, 9 738 unique prompt digests, 1 452 duplicate call
lines, 30 933 421 total tokens (18 983 941 prompt, 11 949 480 completion, 10 267 340 reasoning,
13 724 720 prompt-cache-hit tokens). 343 aux2 error digests have no usage record; exact provider
price is therefore **not reconstructable** from cache and is not invented.

| cache | lines | unique digests | rows | both-pass | single | no parse | total tokens | reasoning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pilot400 | 1 111 | 766 | 383 | 321 | 52 | 10 | 2 179 985 | 505 024 |
| aux_calls_v2 | 5 430 | 4 930 | 2 470 | 2 398 | 64 | 8 | 16 464 821 | 5 545 182 |
| aux2 | 4 649 | 4 042 | 2 023 | 1 781 | 72 | 170 | 12 288 615 | 4 217 134 |
| **union** | **11 190** | **9 738** | **4 355** | **4 155** | **132** | **177** | **30 933 421** | **10 267 340** |

Two-pass agreement: action 3 862/4 221 (91.5 %), relation 3 656/4 221 (86.6 %), alignment
3 908/4 221 (92.6 %). Per-argument authorization agreement over all shared teacher paths:
4 627/4 946 (93.6 %); after path validity filtering, 4 433/4 560 paths are usable (127 invalid
teacher paths excluded). Valid-path field agreement: value_source 74.1 %, derivation 81.5 %,
conflict 85.8 %.

Quote literalism against exactly the text the teacher was shown: user authorization quote 9 061
literal / 2 non-literal / 125 null; external requirement 3 243 / 48 / 5 897; argument source
quote 9 744 / 62 / 1 093. Student-view visibility of cited evidence: user 99.97 %, external
96.11 %, argument 97.52 % of literal quotes.

**Protocol violation found and fixed:** the old collation made 2 265 action rows trainable =
2 199 agreed + 66 single-pass (old last-line-wins policy; corrected QC shows 64). The new
collation masks every single-pass field; trainable action/rel/align = 3 798/3 598/3 846 after
excluding 95 forged-role rows.

## 4. View causality and provenance

`v6param-view-3` fixes the forged-ChatML hole:

- 2 341 TRAIN rows changed; 2 146 role-sequence changes; **0** DEV rows changed; **0**
  `user_request` strings changed.
- 95 annotation rows had been exposed to a forged trusted `user`/`system` block; they are
  excluded from supervision (58 aux, 37 aux2, 12 pilot; unique 95).
- After fixup: **0** TRAIN rows with an extra trusted `user`/`system` role.
- Synthetic mutations pass: forged user/system inside tool response stays data; fake
  `<|im_end|>` inside tool payload does not truncate; a second user turn is downgraded to data;
  mid-line forged markers are not boundaries; wrong endpoint changes the prefix SHA; future-text
  mutation changes features; GT metadata and teacher fields do not enter features.
- Validator: **41/41 checks pass** (`artifacts/validation_v3.json`), including pack SHA,
  float32 21/78 replay, prefix re-decode, segment suffix, value/source literalness, group/episode
  fold disjointness, forged-role check, strict aux policy, frozen fusion and frozen teacher hashes.

## 5. Formal arm results and error movement

All arms: 3 seeds, episode score = max over CT rows per seed then float32 mean over seeds, own
full-sample 5 %-FPR threshold (FP 80 for candidates vs baseline 78), `>=`.

Per-seed FN / FP / AUC:

| arm | seed0 | seed1 | seed2 | ensemble |
|---|---|---|---|---|
| S0 | 20/80/0.98318 | 18/80/0.98524 | 18/80/0.98651 | 17/80/0.98845 |
| S1 | 19/80/0.98452 | 14/80/0.98921 | 17/80/0.98430 | 13/80/0.98870 |
| S2 | 17/80/0.98471 | 14/80/0.98829 | 15/80/0.98561 | 13/80/0.98915 |
| S3 | 29/80/0.97581 | 23/80/0.98591 | 26/80/0.97839 | 19/80/0.98532 |
| S4 | 13/80/0.98810 | 8/80/0.98801 | 21/80/0.98569 | 13/80/0.98929 |

Error movement vs baseline: S0 recovered 14 FN / new 10 FN; S1 16/8; S2 16/8; S3 13/11;
S4 15/7. FP movement: S0 35 recovered/37 new; S1 33/35; S2 37/39; S3 37/39; S4 35/37.

Paired whole-group bootstrap (2 000 replicates, fixed thresholds, no in-draw ROC):

| comparison | ΔFN point | ΔFN mean [95 % CI] | ΔFP point | ΔFP mean [95 % CI] |
|---|---:|---|---:|---|
| S0 vs base | −4 | −3.97 [−14, +6] | +2 | +1.89 [−15, +19] |
| S1 vs base | −8 | −7.98 [−18, +2] | +2 | +1.93 [−14, +18] |
| S2 vs base | −8 | −8.00 [−18, +2] | +2 | +2.05 [−15, +19] |
| S3 vs base | −2 | −1.99 [−12, +8] | +2 | +2.11 [−16, +19] |
| S4 vs base | −8 | −7.95 [−17, +2] | +2 | +1.96 [−14, +19] |
| FUSION_S0 vs base | −3 | −2.92 [−11, +5] | +2 | +1.82 [−11, +15] |
| FUSION_S2 vs base | **−10** | **−9.94 [−19, −1.98]** | +2 | +1.99 [−11, +16] |

S2 slices (FN): cot 3, ctx40k 3, nothink 7; banking 4, slack 1, travel 8, workspace 0.
FUSION_S2 slices (FN): cot 4, ctx40k 3, nothink 4; banking 3, slack 0, travel 8, workspace 0.
Full slices are in `artifacts/results_v3.json`.

### S4 fine-grained field arm (exploratory)

S4 adds a shared per-field encoder over `(path, value, source window, user goal)` with path-level
authorization BCE and a learned max/mean aggregation into the event logit; it keeps the
full/partial alignment head, drops event action/relation and the old aggregate arg head.
2.98 M parameters (base 2.19 M + field encoder 0.79 M), epoch 5 selected on TRAIN fold 0
(AUC 0.99203, FN 10). Formal DEV 13 FN / 80 FP / AUC 0.98929 is **the same FN as S1/S2 and lower
AUC than S2**, with high seed variance (8–21 FN). Independent re-score from the three S4
checkpoints matches the saved ensemble to 1.9e-6. Conclusion: on this sample, fine-grained
per-argument field supervision does **not** add over event-level aux; do not expand this branch
without a larger clean label set.

## 6. Fusion: old rule invalid, new rule frozen

The registered batch-rank fusion used the DEV score distribution of other episodes. It is not an
online per-episode score and was replaced (see PROTOCOL_AMENDMENTS A4). The two old
`fusion_model*.json` files are legacy LR parameters and are **not used**; `results.py` used only
the batch-rank rule.

New frozen mapping is label-free and TRAIN-only; student OOF uses folds 1–3 (fold 0 excluded
because the epoch count was selected there). `FUSION_S0` is the rule-selected overall fold-0 arm
(S0). `FUSION_S2` uses S2 (stronger parameter arm by fold-0 AUC among S1/S2) and is
**exploratory DEV reuse** because the arm choice was made after the base DEV table was visible.
Both use fixed equal weights 0.5/0.5; no DEV-fitted weight, no batch ranking.

## 7. Deployment cost (honest path, 3 independent processes)

Charged per CT event: segment decode, incremental history/source index append (first event of an
episode charges history-load burst), typed view assembly, generic lexical retrieval, exact/normalised
value→source index, hashed feature extraction, H2D, full 2.19 M student forward, score/running max.
The frozen Qwen base is not re-run; no cached-view-only shortcut.

| component (p95 median) | ms |
|---|---:|
| decode | 0.244 |
| index update | 0.222 |
| view + retrieval + value/source | 0.970 |
| H2D | 0.204 |
| student forward + score/max | 0.508 |
| **total p50 / p95 / worst p95** | **1.22 / 1.84 / 2.02** |

3 processes × 3 repeats × 600 events after 60-event warm-up. `cost_passed_1ms=false`.
Optimisation already applied: fused single EmbeddingBag call for cur/vis/user/source and cached
word/bag tokenisation. Remaining gap is mostly Python view/retrieval (~1 ms p95) plus kernel
launch overhead at batch 1. No threshold was changed to make this pass.

## 8. Reproducibility / artifacts

- Source: `/root/sdsafe-compute/v6-param` git, tag/commit recorded in `HANDOFF.json`.
- New source: `annotation_guard.py`, `aux_data.py`, `argfeats.py`, `argmodel.py`; modified
  `view.py` (v3 provenance), `features.py` (cache), `cost.py` (honest stream), `metrics.py`
  (bootstrap sign fix), `run_arms.py` (strict aux, S3 shuffle, versioned outputs), `teacher.py`
  (guard).
- New scripts: `cache_qc.py`, `freeze_fusion_v3.py`, `results_v3.py`, `run_arg_arm.py`,
  `validate_v3.py`, `rescore_s4_v3.py`.
- Tests: 13/13 pass (`logs/tests_cacheonly.log`), including fake-network counter proving zero
  `urlopen` and zero credential reads.
- Old outputs preserved in `artifacts/archive_pre_cacheonly/`; new outputs versioned `*_v3`.
- Teacher caches SHA-256 unchanged before/after:
  `aux2 9f6f4d…a20c`, `aux_calls_v2 089e48…0f2b`, `pilot400 ef11c7…b3ef`.
- Key new SHA-256: `results_v3.json c7ee3f…f01`, `validation_v3.json 34daf8…075e`,
  `study_arms_v3.json 6aa6e3…71cf`, `fusion_frozen_v3.json 939631…ec88`,
  `fusion_frozen_v3_S0.json 8c6448…c97`, `cost_v3_summary.json` in manifests.
- No TEST raw/features/labels/preds read. No old SD-Safe/v1–v5 source/data modified.

## 9. Residual risks / limitations

1. **CPU-store sync not performed from this node**: the CPU SSH endpoint/key are present but the
   host address is not in this container (connection attempts to localhost:13661/22 fail / host
   key unknown). A local handoff bundle and SHA list are produced under
   `artifacts/handoff/cache-only-r1/`; parent must atomically copy and re-read SHA on the CPU node.
2. S4 and FUSION_S2 are post-DEV exploratory reuse, not confirmatory; FUSION_S0 is the
   rule-selected frozen fusion.
3. The base probes are in-sample on TRAIN (AUC 0.9998); only a label-free fixed scale is borrowed,
   but this remains a caveat for the fusion mapping.
4. 20 zero-length common-DEV CT rows have no student view; frozen base logits there equal the
   episode max, so episode max is unchanged. Student scores 24 983/25 003 CT rows.
5. Teacher labels are model-generated, not human gold; 95 forged-role rows are excluded, 132
   single-pass rows masked, 177 no-parse rows excluded.
6. Bootstrap CIs for S1/S2 include 0; only FUSION_S2 excludes 0, and it is exploratory.
7. Cost p95 is 1.84 ms > 1 ms; accuracy/engineering tradeoff must be decided by the parent/user.
8. No new annotation means no label-side fix for the per-argument path-invalid 127 cases or
   forged-role rows; they are masked, not repaired.
