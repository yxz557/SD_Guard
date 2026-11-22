# PROTOCOL — `sdsafe-lowcost-authorization-v5-depth`

Contract recovered from the frozen v4 stage (`/root/sdsafe-compute/v4-gpu`, read-only,
HEAD lineage …`b1b1548`) and from the CPU durable store.  Nothing here changes an accepted
input; the new writes live only in this study root and in the CPU durable v5 tree.

## 1. Authoritative inputs (read-only)

| input | path (GPU) | verification |
|---|---|---|
| frozen Qwen3-8B | `/dev/shm/sdsafe-gpu-v4/upstream/SD-Safe/models/Qwen3-8B` | 5 shards + config/tokenizer SHA match the migration `model_manifest.json`; 36 layers / 399 tensors |
| accepted rows | `.../SD-Safe/artifacts/fine_sequence_explore_v1/data/sequence_v1/rows_{train,dev}.jsonl` | per-row `sequence_sha256`, `feature_token_index`, `strict_final` |
| accepted final hidden | `.../sequence_v1/features_bf16.bin` (bf16 `uint16`) | row-indexed by `cache_row` |
| common DEV universe | `.../sdsafe-lowcost-authorization-v2/artifacts/tensors/dev_common_arrays.npz` | exactly 2 028 episodes / 25 003 rows / 166 groups |
| TRAIN label pool | `.../sdsafe-lowcost-authorization-v2/artifacts/tensors/train_pool_arrays.npz` | `security_label`, `security_mask`; 93 233 in-loss rows |
| accepted baseline scores | `.../sdsafe-lowcost-authorization-v2/artifacts/baseline/baseline_scores_v2.npz` | `ensemble_score`, frozen canonical threshold −0.6190447211265564 |
| prepared token ids | `/root/sdsafe-compute/v5-depth/prepared/{train,dev}` (from CPU bridge) | every round re-proved by `canonical_sha256(ids) == sequence_sha256` inside `load_episode` |
| old block 4/8 shards | v4 tree (read-only) | only used for the G48 control; never modified |

TRAIN total contract: **8 229 episodes / 114 799 CT rows / 122 566 048 tokens**, scopes
cot / ctx40k / nothink / agentlab.  common DEV: **2 028 / 25 003 / 26 841 965 tokens**,
cot / ctx40k / nothink.  All CT / unknown / zero-CT episodes are kept; no episode is
dropped for length.

## 2. Depth capture contract

* `PrefixModel(blocks=24, captures=(16,24))` — only decoder blocks 0..23 materialised from
  the original safetensors; weights are never modified; the `lm_head` is never loaded or
  executed; `model.norm` is never applied.
* Endpoint-only extraction in the layer pre-hook (index-select of the accepted endpoint
  rows); the full `[1, L, 4096]` activation tensor is not retained.
* One prefix-24 forward per round unless the smoke proves 24 blocks illegal for the longest
  real shape.  In that case a KV-cache chunked prefill is used **only after** the smoke
  proves, at a fixed tolerance (`cos_min ≥ 0.9995`, `max_abs ≤ 0.5`), that it matches the
  monolithic forward at blocks 16/24, plus suffix-invariance (endpoint row unchanged when
  the sequence is truncated at that endpoint), wrong-endpoint (row p−1 differs from p), and
  future-token (appending tokens changes nothing at p).  Positions/RoPE are absolute.
* Per episode: atomic `.npz` rename + `npz_sha256` commit record in `progress.jsonl`.
  Resume is "committed shard exists and its digest is recorded"; a partial `.tmp` is ignored.
* Shard fields: `episode_id, scope, group_id, label, seq_len[r], row_index, cache_row,
  message_index, call_index, segment_index, token_start, token_end, round_id, positions,
  sequence_sha256, event_id, block16 uint16[n,4096], block24 uint16[n,4096]`.
* SHA-sync each committed batch to
  `/root/autodl-tmp/sdsafe-lowcost-authorization-v5-depth/artifacts/capture/…` and read the
  digest back on the CPU.  The GPU tree is never the only copy of a finished batch.

## 3. Training / evaluation contract

* Standardization μ/σ from **TRAIN in-loss rows only** (`in_loss & row_label ≥ 0`,
  93 233 rows); never from DEV, never from the evaluation fold being reported as held out.
* Row classifier `Linear(d,256) → GELU → Dropout(.1) → Linear(256,1)`; AdamW lr 3e-4,
  wd .01, 16 epochs, balanced 16 pos + 48 neg per step, grad clip 1.0, seeds 0/1/2.
* Episode score = max over CT rows of the row logit, then mean over the 3 seeds
  (`ensemble of per-seed episode maxima`), matching the accepted baseline composition.
* Threshold rule: highest threshold with FPR ≤ 5 % on the full evaluation sample (same
  function as the accepted stage; ≤ 80 FP on common DEV).  Each method uses its own
  threshold.  Bootstrap reuses those fixed thresholds (no in-draw re-selection).
* Bootstrap: paired whole-group, 2 000 replicates, seed 20261113; report point Δ and
  bootstrap mean/CI separately.  `AUC up but FN unchanged` is not called progress.
* Baseline replay: canonical frozen point (thr −0.6190447211265564 → 21 FN / 78 FP) and the
  budget-optimal point (own 5 %-FPR threshold → 21 FN / 80 FP) are both reported.
* DEV is exploratory reuse.  No TEST material is read anywhere.

## 4. Cost contract

Same 4090, warmup + `cuda.synchronize` in every timed step, 3 independent processes,
600 CT events.  Baseline and candidate heads are timed on identical rows, host-start
(memmap read + H2D) and device-resident starts.  The claim is the **increment** on top of
the main model's own forward (hook/copy of the already-computed activation, scaler, three
fused per-seed heads, score, running episode max), plus the real hook/copy cost inside the
actual capture forward.  No end-to-end main-model wall clock is claimed unless the full
36-block replay actually runs on this card.

## 5. Determinism and provenance

* Source checkpointed in an independent git (every code change committed with a message).
* Model / feature / row / prepared / capture SHAs recorded; the manifest hashes every small
  artifact and lists sizes for the large tensors.
* `PYTHONDONTWRITEBYTECODE=1` everywhere; the declared interpreter is
  `/root/vllm_env_backup/bin/python` (torch 2.14.0+cu130, transformers 5.20.2).
* Known code-fix invariants carried into v5:
  * bf16 `uint16 → float32` must be `(bits.astype(uint32) << 16).view(float32)` —
    pinned by `tests/test_loader_bits.py`;
  * the fused 3-head cost matmul must not call `.t()` on a 3-D weight (v5 uses
    `bmm(w1, x.unsqueeze(-1))` with `[heads, 256, dim]`);
  * the validator's `forged_metric` mutation must mutate a **stored metric value** in a
    shadow copy of the real report and must FAIL; a zero-input dummy is vacuous and is not
    used;
  * the validator does not share the pipeline's metric/loader functions: float64
    recomputation and the independent `loader.bf16_bits_to_f32`.

## 6. Conditional second wave (pre-declared before any result is seen)

Trigger: the first-wave TRAIN held-out fold-0 table shows at least one depth arm (H, I or J)
with fewer FN than the matched final-only control C **and** no arm reaches FN ≤ 14 on DEV.
If the trigger does not fire, the depth route is reported negative and no second wave is run
(no fishing).  If it fires, exactly one lightweight, pre-declared arm is fitted:

* **K = gated low-rank fusion** of the same three blocks:
  `h_b = GELU(W_b x_b + c_b) ∈ R^256` for `b ∈ {final, 16, 24}` (each `W_b: 4096→256`),
  `z = Σ_b softplus(g_b) · h_b` with one learned scalar gate per block, `logit = w2·z + b2`.
  Parameter count is within ~1 % of the J head (3.15 M vs 3.15 M) so "capacity" is matched;
  the gates make the depth contribution learnable rather than concatenated-and-ignored.
* Same TRAIN-only scaler, same recipe/epochs/seeds/folds, same fold-0 selection, same
  5 %-FPR threshold rule and the same bootstrap.  It is reported as an exploratory
  second-wave arm, not as an independent confirmation.

## 7. Coverage/label-pool accounting

* TRAIN capture universe: 8 229 episodes over cot / ctx40k / nothink / agentlab, every CT /
  unknown / zero-CT episode retained, no length-based skipping.
* common DEV: exactly the accepted 2 028-episode universe, 25 003 rows, 166 groups.
* Label pool used for fits: the accepted v2 pool, `security_label` with `security_mask`
  false → **93 233** in-loss rows.  Other historical pool sizes (old 107 215, repair 107 294,
  L4 ≈ 93 322) are *not* used here and are named only to avoid conflation.
* Any episode that fails a prepared digest check is a hard error, never a silent drop.
