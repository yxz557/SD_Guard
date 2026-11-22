# PROTOCOL — `sdsafe-lowcost-authorization-v4` (earlier-layer activation route)

Frozen before any TRAIN/DEV capture is used for modelling. All writes stay inside
`/root/autodl-tmp/sdsafe-lowcost-authorization-v4`. Upstream roots (SD-Safe, v1, v2, v3)
are read-only. No API/teacher call, no TEST material, no system-package change, no new
dependency install.

## 1. Question

The accepted baseline (`three-frozen-base_ensemble_B`, FN 21 / FP 78 / AUC .9811267951,
thr −0.6190447211265564, common DEV 2 028 eps / 414 P / 1 614 N / 166 groups) is a probe on
the **final** Qwen3-8B hidden state (36 blocks + final RMSNorm). v1–v3 showed that
re-weighting, correcting, re-bounding or re-aggregating that single representation does not
move the operating point. v4 therefore asks a *representation* question:

> Does the residual stream **after block 4 and after block 8** of the same frozen Qwen3-8B,
> read at the same accepted endpoint token, carry information the final layer lost?

This is a genuinely new information source (the earlier layers are not in any accepted
artifact) and it can be captured offline without violating the deployment contract, because
in deployment the main model's own forward already computes those tensors.

## 2. Exactness: what "the original model" means here

A block-`B` activation depends only on blocks `0..B−1`, the token ids and the positions;
later blocks cannot change it. v4 executes `embed_tokens + rotary_emb + layers[0..7]` of the
**unmodified frozen** `Qwen3-8B` weights and reads `output_hidden_states[4]` and
`output_hidden_states[8]` (i.e. after 4 and after 8 blocks). Nothing is quantised,
truncated, re-templated, re-encoded or regenerated:

* weights are read byte-for-byte from the original 5 safetensors shards through the
  original `model.safetensors.index.json` key map; the per-layer `state_dict` key set is
  asserted equal to the shard key set (11 tensors per block + `embed_tokens.weight`);
* the module graph is `transformers.models.qwen3`'s own `Qwen3Model` / `Qwen3DecoderLayer`
  (built on the meta device, only the executed prefix materialised), the loop is the
  library's own `Qwen3Model.forward`, and the exit is a `forward_pre_hook` on
  `layers[8]` that raises — so `layers[8..35]` never run and `model.norm` is never applied;
* RoPE is the library's `Qwen3RotaryEmbedding` built from the accepted config
  (`rope_theta = 1e6`, `head_dim = 128`, `max_position_embeddings = 40960`);
* positions are the plain `arange(L)` of a single unpadded sequence, causal SDPA, batch 1,
  bf16, `use_cache=False`, `logits_used = False`.

**Token ids are the accepted ones.** One sequence = one assistant round, exactly as the
accepted producer forwarded it:

| scope | source of ids | binding check |
|---|---|---|
| `cot` (`cot32k`) | archive `generation.prompt_token_ids + completion_token_ids` | archive's own `*_token_ids_sha256` + head/tail, then `canonical_sha256(ids) == row.sequence_sha256` |
| `ctx40k`, `agentlab` (`ramp_ablation`) | frozen legacy full-trajectory render (`normalize_messages` + chat template, no tool schema) | `canonical_sha256(ids[:feature_token_index+1]) == row.sequence_sha256` |

`nothink` (`nothink_ctx40k`) is **not captured** (see §7); it is not replaced by any other
serializer.

Endpoint index = the row's own `feature_token_index` (asserted equal to `token_end`), the
same index whose accepted final-layer vector is stored in `features_bf16.bin`.

## 3. Smoke (2 authoritative TRAIN episodes, executed)

`artifacts/smoke/smoke_report.json`. Episodes: one short `ctx40k` episode and one
multi-call `cot` episode with text segments (different lengths, 3 rounds each).

| check | result |
|---|---|
| hook path vs independent replay of `Qwen3Model.forward` | **bitwise equal** at blocks 4 and 8 |
| suffix invariance (truncate to `pos+1` vs read at `pos` of the longer round) | **bitwise equal** (block 4) |
| GPU bf16 prefix vs CPU **fp32** prefix | block 4 max abs 1.9e-2, cos 0.99999; block 8 max abs 0.124, cos 0.99998 |
| GPU bf16 prefix vs **complete 36-block CPU bf16** model at the same block | block 4 max abs 1.6e-2 cos 0.99999; block 8 max abs 0.25 cos 0.99997 |
| complete 36-block CPU bf16 → `model.norm` vs the **accepted bf16 feature matrix** | cos 0.99979, mean abs 2.5e-2, max abs 2.0 |

The last row is the end-to-end identity proof (ids + endpoint index + weights + norm
placement). The 36-block CPU run costs 220 s for 321 tokens and is therefore a one-off
reference, not a capture path. GPU peak 4.90 GB (weights 4.33 GB); measured prefix
throughput 17–133 k token-layers/s depending on length (≈18 k tokens/s at 8 blocks).

## 4. Capture

One prefix forward per round; every CT endpoint of that round is read from that single
forward; per-episode atomic `.npz` (`tmp` + `fsync` + `rename`) with the file digest in
`progress.jsonl`. Resume = skip episodes whose progress line carries a digest.

DEV = the accepted common universe (2 028 episodes) intersected with the capturable scopes;
TRAIN = all capturable episodes, shuffled with a fixed seed so partial coverage is unbiased.
Stored: `block4`, `block8` (bf16, 8 KiB per row per block) at the accepted index, plus row
binding (`row_index`, `cache_row`, `round_id`, `message_index`, `call_index`,
`segment_index`, `token_start`, `token_end`, `positions`, `sequence_sha256`, `event_id`).

Coverage is reported per scope and per split, including the explicitly missing scope.

## 5. Model menu (pre-registered; ≤ 24 formal runs)

Row classifier `Linear(d,256) → GELU → Dropout(0.1) → Linear(256,1)`, accepted balanced
recipe (16 pos / 48 neg per 64-row batch, AdamW 3e-4, 16 epochs, `2·ΣBCE/Σn`), 3 seeds
(0,1,2), episode score = `max` over the episode's CT rows, ensemble = mean over seeds.

| arm | input | role |
|---|---|---|
| `C` | accepted final hidden, raw bf16 (4096) | matched terminal-only control (isolates "new information", not capacity) |
| `E` | block4 ‖ block8 (8192) | early layers only |
| `F` | final ‖ block4 ‖ block8 (12288) | fused |

Two fits per (arm, seed): `ho` = fit on group folds 1–3 (fold 0's groups never seen) used
for TRAIN-only selection; `full` = fit on all TRAIN groups, used for the DEV evaluation.
3 arms × 3 seeds × 2 fits = **18 runs**. No run is selected on DEV.

Arm `C` consumes the raw accepted bf16 final-layer vector.  The accepted probe of v2/v3
consumed a *probe-standardised* copy of the same vector (`probe_standardize`), so `C` is a
matched capacity/recipe control on the same information, not a re-run of the accepted
probe; the accepted baseline numbers are quoted from the accepted artifacts instead.

## 6. Evaluation rules

* Primary metric: the accepted common-DEV episode universe, strict `final` labels,
  own 5 %-FPR ROC operating point (`fpr_threshold`), `FP ≤ 80`, `FN` and `AUC` reported.
* DEV is an **exploratory reuse** (the accepted baseline itself was DEV-selected); no run
  here is independent confirmation.
* Missing-scope episodes stay in the denominator: variant "covered subset" numbers and the
  full-2 028 number with explicit old-base fallback are both reported, together with the
  per-episode coverage flag. No episode, label, denominator or hard example is dropped.
* TRAIN-only selection uses held-out groups (fold 0: 154 groups) and is reported even when
  it disagrees with DEV.
* Paired whole-group bootstrap over DEV groups (2 000 replicates) for FN/FP/FNR/FPR deltas;
  fixed threshold, no ROC re-selection inside a replicate; point delta and bootstrap mean
  reported separately.
* Semantic re-labelling is reported separately if at all; original labels are never changed.

## 7. Known limitation (declared, not hidden)

`nothink_ctx40k` (656 of the 2 028 common-DEV episodes, ≈32 %) cannot be re-built on this
host: its round ids are a frozen-template render that needs the AgentDojo tool schema, the
only torch/transformers environment has no `agentdojo` runtime dependencies
(`cohere`, `tenacity`, `google-genai`, `openai` all absent) and there is no network. Those
rows are explicit unknown + old-base fallback; they are never re-encoded with another
serializer. Any candidate from this stage is therefore **not** a full-coverage candidate.

`ctx40k`/`agentlab` ids are reproduced, but through a re-execution of the frozen render
(the accepted `full_sequence_sha256` is not the row digest); the row's own
`sequence_sha256` is the binding check and it is enforced per row.

## 8. Cost accounting (deployment)

Deployment must not add a second backbone pass. The advertised increment is only:
capture of the block-4/8 tensor the main forward already produces (hook + copy), the
optional per-token running pool update, the H2D/D2H of the needed rows, the projection and
all heads/scores/episode max. Offline prefix-layer computation is an **offline** cost and is
reported separately, never as deployment cost. Gate: ≤ 1 ms extra p95 per CT event,
batch-1 streaming, ≥ 3 independent processes, host and device paths reported separately,
warm-up excluded, CUDA synchronisation included.

## 9. Validator (independent)

Re-derives every formal checkpoint's row scores and episode metrics from the saved weights
and the captured tensors, re-checks the row binding (row id ↔ episode ↔ cache row ↔ token
index), re-checks group-disjointness of the fits, recomputes AUC/FN/FP/FPR/FNR with an
independent implementation, and runs negative mutations that must fail: wrong block
(layer swap), wrong token offset, future-call/text injection, DEV rows in the scaler/fit,
non-authoritative cohort, wrong model weights, held-group leakage, forged metric and forged
manifest digest.
