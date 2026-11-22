# PLAN — `sdsafe-lowcost-authorization-v5-depth` (deeper frozen-prefix activation version)

Sole writer: this Pi worker on `autodl-container-ujxcycmw77-6fc011ce` (RTX 4090 24 GiB).
CPU/storage node `autodl-container-rml149b1gp-b62cb45e` holds the read-only sources and the
durable copy.  The previous GPU stage (v4, blocks 4/8) is complete and its tree is read-only.

## 0. Question

The accepted best point is FN 21 / FP 78 (AUC .981127) on common DEV 2028 / 414 P / 1614 N /
166 groups.  Its canonical threshold is the value stored in the accepted baseline artifact
(`baseline_scores_v2.npz['threshold']`); the v4 source's literal constant
−0.6190447211265564 differs by ~2e-8 and flips one borderline positive (22/78), so both are
replayed and reported and neither is silently substituted; the progress bar is FN ≤ 14
(strong ≤ 8).  Block-4/8 data carries information but did not move FN (raw F FN 31,
TRAIN-standardized G FN 21) — so the open question for this stage is **whether the *depth*
of the already-accepted frozen forward supplies the missing error-changing signal**.

## 1. Pre-registered design

* New **dataset version**: residual stream after block **16** and block **24**
  (HF `output_hidden_states[16]` / `[24]`) of the *same unmodified* Qwen3-8B forward, at the
  already accepted endpoint token of every CT event.  A single prefix-24 forward captures
  both endpoints (plus block 8 as a cross-check only).  Never mixed with the old 4/8 shards.
* Pre-registered arms (all standardized with TRAIN-only per-feature μ/σ, same MLP
  4096→256→1 / GELU / dropout .1 / 16 epochs / balanced sampler / lr 3e-4 / 3 seeds):
  * **H** = `[accepted_final ‖ block16]` (8192)
  * **I** = `[accepted_final ‖ block24]` (8192)
  * **J** = `[accepted_final ‖ block16 ‖ block24]` (12288)
* Matched controls, identical recipe/capacity/budget/seeds:
  * **C-final** = `[accepted_final]` (4096) — TRAIN-only control;
  * **G48** = standardized `[accepted_final ‖ block4 ‖ block8]` (12288) — the v4b arm, refit
    here on the identical row set so that "new layer" is not confounded with pipeline drift.
* Selection: TRAIN held-out fold 0 (`ho`) for every arm; `full` (all TRAIN groups) for the
  DEV evaluation.  DEV is an exploratory reuse (the accepted baseline itself was
  DEV-selected) and is labelled as such everywhere.
* Every method gets **its own** threshold at FPR ≤ 5 % (≤ 80 FP) with one fixed rule
  (accepted `fpr_threshold`: highest threshold with FPR ≤ 5 %).  Thresholds are fixed on the
  full sample before any bootstrap resample; no ROC re-selection inside a draw.
* The accepted baseline is replayed at its **canonical frozen threshold** (artifact-stored
  value, 21/78), at the v4 literal constant (22/78, reported transparently), and separately
  at its own budget-optimal threshold (the v4 report's 21/80 point).  They are never
  conflated and the canonical point is never re-budgeted.

## 2. Execution order (each step has a durable checkpoint)

1. Independent bf16 loader + pinned bit-pattern tests (0x3f80=1, 0xbf80=−1, 0x4000=2, …,
   cross-checked against torch `bfloat16 → float32`).  Validator uses its own reader.
2. CPU export of **all** prepared exact token-id jobs (reuse the 4 623 validated TRAIN
   jobs; add the rest of TRAIN and all 2 028 common DEV).  Verified token totals must equal
   the v4 contract: TRAIN 122 566 048, DEV 26 841 965.
3. Fresh-process smoke per depth (16, 24): short / ~7k / longest real round (>35.7k),
   endpoint-only capture, no `lm_head`, `reset_peak_memory_stats` after load.  KV-chunked
   prefill is only used if the monolithic 24-block forward is illegal; its equivalence to
   the monolithic forward is proved at a fixed tolerance plus suffix-invariance,
   wrong-endpoint and future-token mutations.
4. Full capture TRAIN then common DEV into a **new** artifact version; per-episode atomic
   `.npz` shards, SHA in `progress.jsonl`, immediate push to the CPU durable tree with
   SHA read-back.  Resume from committed shards only.
5. Dataset build (row join by accepted global row index; final vector from the accepted
   feature matrix; block16/24 from the new shards; TRAIN labels from the accepted v2 pool).
6. Fits H/I/J + C-final + G48 × 3 seeds × {ho, full}; TRAIN fold-0 selection table; DEV
   exploratory table; paired whole-group bootstrap; recovered/new FN analysis; scope /
   workspace / IT6 breakdown.
7. Cost: paired baseline vs candidate, 3 independent processes, host-resident and
   device-resident paths, p95 increment ≤ 1 ms/CT, real hook/copy included.
8. Validation: independent float64 metric recompute of every formal checkpoint on a fresh
   score pass, capture-source re-derivation, and a **non-vacuous** forged-metric mutation
   (tamper a stored metric in a shadow copy → must FAIL) plus wrong layer/endpoint,
   bf16 conversion, row order, illegal cohort, future token, model SHA, OOF-held-group,
   manifest tampering.
9. REPORT / HANDOFF.json / STATUS / PROTOCOL / manifest + CPU durable sync with SHA
   read-back.  Final acceptance of a candidate is explicitly left to the parent.

## 3. Budgets

* Phase wall clock ≤ 8 h; GPU job count: one capture job at a time, single owner.
* Storage: new study ≤ 20 GiB on CPU with ≥ 20 GiB free remaining; GPU root keeps ≥ 4 GiB
  free, `/dev/shm` ≤ 24 GiB for the study (model mirror 16 GiB already resident).
* No API/teacher calls.  No TEST read.  No quantisation, no weight/template/RoPE change.
* Everything Python runs with `PYTHONDONTWRITEBYTECODE=1` and the declared environment
  `/root/vllm_env_backup/bin/python`.
