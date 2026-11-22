# PLAN — v4 earlier-layer activation route

## Phases (each one produces a durable artifact)

1. **P0 provenance** (done) — new independent git, checkpoint commit before any change,
   read-only upstream roots recorded, `PROTOCOL.md` frozen before modelling.
2. **P1 exact prefix execution** (done) — `src/sdsafe_v4/prefix.py`:
   `embed_tokens + rotary_emb + layers[0:8]` of the unmodified frozen Qwen3-8B, original
   module graph, original forward loop, early exit via a `forward_pre_hook` on `layers[8]`
   (so `model.norm` is never applied). Weight keys asserted equal to the shard key set.
3. **P2 smoke** (done) — two authoritative TRAIN episodes (one short `ctx40k`, one
   multi-call `cot` with text segments): GPU peak 4.90 GB, 18 k tokens/s, hook path
   bitwise equal to an independent replay, suffix invariance bitwise equal, GPU-bf16 vs
   CPU-fp32 cos ≥ 0.99998, GPU prefix vs complete 36-block CPU model cos ≥ 0.99997, and the
   complete 36-block CPU forward reproduces the **accepted** feature matrix (cos 0.99979).
4. **P3 capture** — resumable per-episode atomic shards for all four scopes; DEV first
   (must cover the common 2 028 universe), then TRAIN with a fixed shuffle.
5. **P4 pilot accounting** — measured throughput → exact remaining budget for the full
   TRAIN sweep (recorded in `artifacts/capture/*/coverage_run.json`).
6. **P5 modelling** — datasets (`block4`, `block8`, accepted final hidden), arms C/E/F,
   matched recipe, TRAIN held-out-group selection, DEV evaluation, paired group bootstrap.
7. **P6 cost** — real batch-1 streaming increment (3 processes, host + device), plus the
   measured hook cost inside the real capture forward. No full-model-on-GPU claim.
8. **P7 validation** — independent recompute of capture binding, fits, scores and metrics,
   with a negative-mutation suite.
9. **P8 handoff** — `REPORT.md`, `HANDOFF.json`, `STATUS.json`, manifest of artifact SHAs,
   git commits, and an explicit list of what is *not* established.

## Non-goals

* No quantisation, no truncation, no re-templating, no second serializer for any scope.
* No deployment-time second backbone pass; the added cost is hook + copy + head only.
* No TEST material, no DEV labels in training, no DEV-selected checkpoint as "confirmation".
* No claim of full-model-on-GPU measurement (12 GiB < 16 GiB is a hardware fact).

## Time budget

Capture is the long pole; the model fits are seconds-to-minutes because the activation is
cached. Remaining budget is recomputed after P4 and written into `STATUS.json`.
