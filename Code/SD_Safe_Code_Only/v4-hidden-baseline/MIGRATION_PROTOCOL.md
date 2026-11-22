# MIGRATION_PROTOCOL — v4 node migration addendum (frozen before new capture)

This addendum does **not** change the scientific protocol in `PROTOCOL.md`. It fixes the
execution/identity rules for running the frozen v4 pipeline on the new GPU node.

## 1. Identity / exactness invariants (unchanged)

* Same accepted token ids, same endpoint `feature_token_index == token_end`, same
  `output_hidden_states[4]`/`[8]` after 4/8 blocks, same `model.norm` never applied,
  same bf16 causal SDPA batch-1 `use_cache=False`.
* Weights read byte-for-byte from the original 5 shards; per-layer key set asserted.
* No re-serialisation, no truncation of `ctx40k`, no re-labelling, no future call/text.

## 2. New-node identity checks (must pass before capture)

1. Model: 5 shard sizes == `model_manifest.json`; `model.safetensors.index.json` SHA ==
   `f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc`; config/tokenizer
   small-file SHAs == manifest; tensor count via index = 399, 36 blocks.
2. Inputs: `features_bf16.bin`, `rows_train.jsonl`, `rows_dev.jsonl` SHA == CPU value
   (`af86fa7a…`, `1b68abc6…`, `8bdc5b63…`); v1 index + v2 tensors/baseline SHA == CPU.
3. Source: independent git, HEAD-lineage `ebed683…` + one recorded path-mapping commit;
   `git status --porcelain` clean apart from ignored `artifacts/logs/manifests`.

## 3. Cross-GPU canary (before trusting the new backend)

Recompute a small, pre-declared set of already-captured episodes on the 4090 and compare
against the old shard (`block4`, `block8`):

* report per-block `cos`, `maxabs`, `RMS(diff)` and the bf16-bitwise-equal fraction;
* assert `row_index`, `cache_row`, `positions`, `sequence_sha256`, `event_id`, `token_end`
  and per-round `seq_len` are identical to the old shard;
* tolerance is chosen from the *existing* smoke evidence (GPU-bf16 vs CPU-fp32 and vs
  full-36 CPU: cos ≥ 0.99997) and the dtype (bf16). Because both GPUs run the same
  **mathematical** path but different kernels/SM counts, bitwise equality is *not*
  assumed; a `cos ≥ 0.9999` and small `maxabs` bound is the acceptance gate.
* provenance per shard: GPU name, torch/cuda/transformers versions, backend
  (`sdpa`), dtype, driver. If the gate fails, store the new backend version separately
  and quantify the impact; never average old+new silently.

## 4. Resume / durability

* Capture resume key = `progress.jsonl` line with `npz_sha256` **and** file present. The
  single `.npz.tmp` fragment is ignored.
* Every finished episode is copied to
  `/root/autodl-tmp/sdsafe-lowcost-authorization-v4-gpu/capture/<split>/episodes/` via
  temp + fsync + atomic rename, then SHA-256 read back and compared. Only then is it
  "durable"; local `progress.jsonl` lines are mirrored too.
* `/dev/shm` is rebuildable only; never the sole copy of a result.

## 5. DEV / TEST boundary

DEV = the accepted common universe (2 028 / 414P / 1 614N / 166 groups), evaluation and
calibration only, exploratory reuse. TEST (including authorityTEST raw/features/labels/
predictions) is never read. `load_cohort` unified split is authority, not `source_split`.

## 6. Cost claim boundary

The deployment increment is hook + copy + per-token pool + scaler + all heads + score +
running-max/E-update, all counted. It is measured **on top of the same single main-model
forward**, never as a separate backbone/API pass. If full-GPU timing is not yet realised,
the boundary is stated as offline-only.
