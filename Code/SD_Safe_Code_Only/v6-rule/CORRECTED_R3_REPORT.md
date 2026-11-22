# CORRECTED_R3_REPORT — SD-Safe v6 parameter authorization, corrected-r3

**Scope:** cache-only correction of the independently audited F1/F2 field-semantics defects,
teacher-cache rebinding, explicit zero-new-token policy, fixed-menu retraining, frozen fusion and
two honest runtime interfaces.
**New annotation tasks / API requests: 0.** No retry, health probe, second judge, label repair,
provider change or coding-agent labelling. **No TEST material read.** Old r1/r2 artifacts,
checkpoints, thresholds, views and reports are read-only and retained. All new artifacts use the
`corrected-r3` name. GPU: RTX 4090 24 GiB, `/root/vllm_env_backup/bin/python` (torch 2.14/cu130),
`PYTHONDONTWRITEBYTECODE=1`. New results are CPU-authoritative under
`/root/autodl-tmp/sdsafe-lowcost-authorization-v6-param/corrected-r3/`.

## 1. Executive result

The corrected field semantics are in place (F1), the cached teacher labels are rebound/masked,
and the zero-new-token policy is explicit and uniformly applied. On the fresh corrected-r3
views/features the fixed-menu DEV result is:

| method | FN | FP | AUC | threshold |
|---|---:|---:|---:|---|
| baseline, frozen float32, `>=` original literal | **21** | **78** | 0.981127 | −0.6190447211265564 |
| S0 main only, 3-seed | 16 | 80 | 0.987914 | −4.333139896392822 |
| **S2 λ1**, 3-seed | **13** | **80** | 0.987622 | −0.7427853941917419 |
| S3 shuffled-aux control, 3-seed | 16 | 80 | 0.985519 | −0.41384175419807434 |
| FUSION_S0 frozen equal-logit | 18 | 80 | 0.989365 | 0.6942636598006706 |
| **FUSION_S2 frozen equal-logit** | **12** | **80** | **0.990347** | 0.6289290373189838 |

- S2 keeps the old 13 FN but AUC falls slightly (0.98915 → 0.98762) under the corrected
  current-call binding and the smaller clean teacher pool. The control S3 improves to 16 FN, so
  the S2-vs-S3 “parameter supervision” separation weakens to ΔFN −3, CI [−8, 0] (includes 0).
- FUSION_S2 is 12 FN (old exploratory 11), AUC 0.990347 (slightly above old 0.990280). Its
  ΔFN vs the frozen baseline is −9, whole-group 95 % CI [−17, −2]; exploratory DEV reuse only,
  not independent confirmation.
- **Cost gate still fails.** Mode A (tokens-only full path) p95 = **2.172 ms**; the real typed
  controller path’s detector-extra p95 is **1.371 ms** (p50 0.801 ms), and its full cost
  including controller-side decode is **2.842 ms** p95. No threshold or model was traded for
  speed.

## 2. F1 — current-call localisation (fixed)

`view.py` no longer takes the first `<tool_call>` of the assistant block. A string/escape-aware
balanced-JSON scanner enumerates real calls; `<think>` spans are reasoning; the row’s
`call_index` ordinal selects a candidate; the row’s own decoded CT `segment` must agree on tool
name **and** decoded argument object. On disagreement the independently parsed segment wins when
unique and the conflict is recorded. Nothing after the row endpoint is decoded or parsed.

Balanced-JSON block parsing also suppresses forged `<|im_end|>`/`<|im_start|>role` markers inside
tool-call argument strings, and the assistant block-end test now ignores `<|im_end|>` inside an
open tool-call JSON string. This removes both a wrong-current-call binding and a prior-text
truncation class.

DEV (25 003 rows):

- 7 813/7 813 `tool_call` rows resolve `call_match="ordinal+segment"`, **0** binding conflicts.
- **830** rows change current tool/arguments versus the old view (549 with call_index 1,
  132 with 2, 65 with 3, 26 with 4, …); **787** rows change `value_sources`.
- 0 DEV rows need the defensive segment-only/block fallback.

TRAIN (114 799 rows):

- 31 853 `ordinal+segment`, 6 segment-only adversarial nested-template rows (resolved from the
  accepted segment, `block_fallback=true`), 8 total block-fallback rows, **3 401** rows change
  current tool/arguments, 3 833 change `value_sources`.
- Prior-text correction changes 11 512 TRAIN rows (forged markers inside argument strings);
  300 prior-role changes. These are TRAIN-only adversarial `agentlab` cases; DEV has none.

## 3. F1 consequence — cached teacher rebinding/masking

Each cached teacher row was re-rendered against the corrected view and compared with the exact
material the teacher saw (user request, prior roles/text, current tool/arguments, value-source
paths). The old global digest-only dedupe undercounted rows when two different rows shared an
identical prompt; keying by `(digest,row_index)` recovers the true **5 045** cached rows (the r1
report’s 4 355 was an undercount caused by that collision; no cache line changed).

| teacher item | count |
|---|---:|
| cached rows (deduped by digest+row) | 5 045 |
| masked: prompt bound the previous call | 312 |
| masked: prior material changed (overlapping reason) | 897 |
| **masked for rebind/material** | **1 121** |
| masked single-pass or unparseable | 131 |
| both-pass present and clean | 3 793 |
| trainable action / relation / alignment | 3 476 / 3 317 / 3 501 |
| trainable argument events (positive) | 1 721 (524) |
| argument paths valid / invalid | 4 117 total, 122 invalid |
| quote-invalid masks (align/action/rel) | 69 / 2 / 2 |
| argument quote-invalid | 45 |

The frozen teacher caches were **not** modified; their three SHA-256 values still equal
`FROZEN_MANIFEST.json` (`aux_calls_v2 089e48…`, `aux2 9f6f4d…`, `pilot400 ef11c7…`). The old
95 forged-role exclusions and 132 single-pass masks are retained. GT labels / teacher answers are
never features. Net effect: S2 is trained on a smaller, correctly bound clean teacher pool
(3 476 action targets vs the old 3 798), which is the honest cost of not reusing mis-bound labels.

## 4. F2 — zero-new-token rows: corrected accounting and declared policy

The audit’s “20 zero-length rows” is incomplete: common DEV has **59** rows with
`token_start == token_end`, all `content_segment`. 39 of them had already been silently scored by
the old fixup path; 20 had been dropped by an old strict span check. The policy was declared
before any corrected score (`PROTOCOL_CORRECTED_R3.md` C3):

- P1 `empty_segment_with_visible_text`: zero new segment tokens but non-empty causal assistant
  text → kept and **scored** from the same causal visible prefix (segment bag empty, vis channel
  non-empty). All 59 DEV rows take this branch; 0 DEV rows are `no_new_emission`.
- P2 `no_new_emission`: no visible text at all → retained in the stream, **no student logit**, no
  running-max update.
- P3 `UNKNOWN_ADAPTER`: zero-span `tool_call`/dropped visible call → no logit, reported.
- Episode score is **defined** as max over rows with a valid student view; an episode with no
  valid row would be `-inf`. This is a policy definition, not a claim of equality to a
  non-existent row’s max.
- All 2 028 episodes / 25 003 CT rows / 166 groups are retained and scored.

Retraction of `PROTOCOL_AMENDMENTS.md` A6: the claim that the zero-length rows’ “base logits equal
the episode maximum” was wrong. The 20 previously-missing rows are **never** the base episode
argmax (0/60 seed-row checks) and are never the corrected student argmax (0/20 rows). Therefore
adding them changes no S2/base episode max. Among the 39 previously-scored rows, two (167541,
170534) are the student episode argmax in two seeds each (4/177 seed-row checks). The correct
statement is: base max is unchanged because those rows are never selected; the student score is
defined by the explicit P1/P2 policy above.

## 5. Registered menu and fitting

Exactly three arms, fixed before reading corrected DEV: S2 (main + parameter aux, λ=1), S0
(main only), S3 (S2 with shuffled aux labels), fixed **5 epochs**, batch 64 (16 pos/48 neg),
600 steps/epoch, AdamW 1e-3/wd 1e-2/dropout 0.2, 3 seeds {0,1,2} → 9 formal fits. Group-disjoint
folds by `blake2b(group_id)%4`; normalizer/dictionary/fit use TRAIN in-pool rows only; main label
is the frozen pool label. 8 extra TRAIN-only outer-held-group OOF fits (S0/S2, one per fold, 5
epochs) are used only for the label-free fusion scale. No epoch scan, LR sweep, S4 or extra arm.

TRAIN rows 114 799 (in-pool 93 322, positives 1 722); folds 28 983 / 27 365 / 29 424 / 29 027;
8 229 episodes; group-disjoint check passed. All 9 formal fits + 8 OOF fits completed in 928 s.

Per-seed FN at each seed’s own 5 %-FPR threshold: S0 18/17/19, S2 18/16/15, S3 28/19/24
(dev ensemble above; FP 80 each). S3’s per-seed variance is large but its 3-seed mean is better
than in r1; the corrected binding did not create a stronger seed-stable S2 arm.

## 6. Fixed frozen fusion and exact τ

`fused = 0.5·z_base + 0.5·z_student` with no fitted weight:

- `z_base`: TRAIN episode base mean/std −14.176067679492869 / 22.363694741629555 (8 227 finite
  episodes; base probes remain in-sample on TRAIN, a recorded scale caveat).
- `z_student`: TRAIN-only outer-held-group OOF episode max; all four folds are usable because r3
  has no epoch/arm selection. S2 scale −8.885196… / 10.069459… over 8 229 OOF episodes; S0 scale
  −22.969588… / 22.402623….
- Fusion arithmetic is float64; per-seed CT logits/episode max and seed mean are float32 as in
  the reference.
- τ was calibrated **once** on DEV at own full-sample ≤5 % FPR using the **deployment batch-1
  stream** fused score, then frozen with exact `float.hex()`:
  FUSION_S2 τ = `0.6289290373189838` (`0x1.4202fc9d94608p-1`), FUSION_S0 τ =
  `0.6942636598006706` (`0x1.637686c34a512p-1`). `>=`, no epsilon, no decimal rounding.
  Replay reproduces 12/80 and 18/80 exactly; batch-vs-stream max |Δfused| ≈ 3.4e-7.
- The old r2 literal τ and the 11/79 batch-1 boundary remain frozen legacy; nothing was
  re-derived to force 80.

## 7. Error movement and slices (S2 / FUSION_S2)

S2 vs baseline: recovered 16 FN / new 8 FN; recovered 40 FP / new 42 FP. FUSION_S2: recovered
13 FN / new 4 FN; recovered 24 FP / new 26 FP.

| comparison | ΔFN point | ΔFN 95 % CI | ΔFP point | ΔFP 95 % CI |
|---|---:|---|---:|---|
| S0 vs base | −5 | [−16, +6] | +2 | [−25, +30] |
| S2 vs base | −8 | [−19, +2] | +2 | [−23, +28] |
| S3 vs base | −5 | [−15, +5] | +2 | [−23, +28] |
| FUSION_S0 vs base | −3 | [−12, +7] | +2 | [−17, +23] |
| FUSION_S2 vs base | **−9** | **[−17, −2]** | +2 | [−16, +21] |
| S2 − S0 | −3 | [−9, +2] | — | — |
| S2 − S3 shuffled | −3 | [−8, 0] | — | — |
| FUSION_S2 − S2 | −1 | [−7, +5] | — | — |

S2 FN by scope: cot 2, ctx40k 5, nothink 6; suite: banking 3, slack 1, travel 9, workspace 0.
FUSION_S2: cot 3, ctx40k 5, nothink 4; banking 2, slack 1, travel 9, workspace 0.
Travel/injection_task_6 is the hard slice for both (S2 9 FN of 16 positives in 45 episodes;
FUSION_S2 also 9 FN, 1 FP). New-FN groups for both: `banking|user_task_2|injection_task_4`,
`slack|user_task_1|injection_task_1`, `travel|user_task_5|injection_task_6`.
All CIs are whole-group paired bootstrap (2 000 reps, fixed thresholds); these are exploratory
DEV-reuse estimates, not population-level confirmation under repeated selection.

## 8. Cost — two interfaces, sequential 3 OS processes × 3 repeats

No other GPU job; per event batch 1; causal in-episode order; all 2 028 episodes / 25 003 events;
cold/warm and slices reported; pre-warmed views are never used as runtime input. Three processes
ran **sequentially**; medians below are over all 9 passes.

| mode | p50 | p95 (median) | p95 worst | p99 |
|---|---:|---:|---:|---:|
| A tokens-only (history re-decode + all work) | 0.867 | **2.172** | 2.218 | 3.369 |
| B typed controller, **detector extra only** | 0.801 | **1.371** | 1.400 | 2.072 |
| B typed controller, full (detector + controller decode) | 0.876 | 2.842 | 2.915 | 4.333 |

Mode B is the real typed-message adapter: the controller supplies the role-tagged prompt text and
raw assistant text deltas it already has; the adapter charges message ingest/byte handling,
current JSON parse, word hashing, source/value index, feature transfer, all three students, base
lookup, per-seed max, float32 mean and fusion. Controller-side prompt/delta decode is measured
separately (p95 1.524 ms) and charged in the “full” variant. Inputs were proven against the
actual source, not an oracle view: **25 003/25 003** events have bit-identical text/view/dense
features and 3-seed layer outputs for both mode A and typed-B parity (0 mismatches), and the
stream reproduces S2 13/80 and FUSION_S2 12/80 with per-seed episode max diff 0.0 vs the saved
replay.

Slices (full, p95 median):

| slice | A | B full | B detector |
|---|---:|---:|---:|
| first event of a round (9 004) | 2.891 | 3.713 | — |
| later events (15 999) | 1.051 | 1.051 | — |
| content_segment (17 190) | 1.272 | 1.984 | — |
| tool_call (7 813) | 2.900 | 3.556 | — |
| scope cot / ctx40k / nothink | 2.787 / 1.274 / 1.963 | 3.328 / 1.535 / 2.689 | — |
| prefix ≥20 k chars (6 002) | 3.053 | 4.143 | — |

Remaining B detector p95 is view+feature indexing ≈0.73 ms plus the merged 3-student forward
≈0.60 ms plus fusion ≈0.06 ms. p95 is dominated by first-event-of-round message/finalisation
work; later events are already ≈1.05 ms full. **The 1 ms p95 gate is not met** for either the
tokens-only or the typed full path, and not met by the typed detector-extra path on its own. This
is reported as a tradeoff, not hidden by a threshold change.

## 9. Validation evidence

`artifacts/corrected-r3/validation_corrected_r3.json` — **32/32 checks pass**:

- export contracts: 25 003 DEV / 114 799 TRAIN rows, 0 failures, all 2 028 episodes and 166
  groups retained;
- call localisation: 7 813/7 813 DEV ordinal+segment, 0 conflicts, 830 changed current calls;
  31 853 TRAIN ordinal+segment, 6 segment-only/8 fallback adversarial rows;
- teacher: frozen cache SHA match, 1 121 rebind masks, a concrete old-bound row masked and a
  clean row retained;
- empty policy: 59 DEV empty-segment rows scored, 0 no-new-emission/unknown-adapter; the 20
  formerly-missing rows are never student/base argmax;
- causality: 25 sampled events (multi-call, text, empty) with all tokens after the fixed endpoint
  replaced → view/features identical; forged ChatML inside argument strings stays data;
- exact τ: `float.hex()` round-trip, τ equals the observed own-5 % FPR threshold exactly,
  replay FN/FP and fused SHA match;
- chunking/cache: 30/30 rounds decode identically under random token-chunk splits; clearing the
  text/block caches and recomputing an episode reproduces the reference view/features;
- parity: mode A and typed-B full parity JSON both 25 003 events, 0 mismatches.

Unit tests: **32/32 OK** (`logs/tests_r3.log`), including the new 11-test call-localisation oracle
suite (first/second/third, same name different args, literal tag in JSON, nested/escaped JSON,
CoT pseudo-tags, unclosed/incomplete calls, forged markers inside arguments, future-append
invariance).

Old-data integrity: `sha256sum -c artifacts/independent-final-audit/sha/before.sha256` → 83/85 OK;
the only two changed files are `src/sdsafe_v6/view.py` and `src/sdsafe_v6/runtime_fast.py` (the
intended source fix). Every pack, model checkpoint, old cache, view, fusion JSON and teacher cache
is byte-identical.

## 10. Legacy values retained (not rewritten)

| legacy item | value | status |
|---|---|---|
| r1 canonical baseline | 21 FN / 78 FP, AUC 0.981127 | unchanged |
| r1 S2 λ1 (old field semantics) | 13 FN / 80 FP, AUC 0.989150 | retained as legacy |
| r1 FUSION_S2 (old) | 11 FN / 80 FP, AUC 0.990280 | retained as legacy/exploratory |
| r2 batch-1 frozen literal τ | 11 FN / 79 FP | retained; boundary artifact documented |
| old S2/FUSION checkpoints, packs, scores, thresholds | — | untouched |

The r3 numbers above are a **new method on new inputs**, not a re-scoring of the old model.

## 11. Residual risks / limitations

1. **Cost gate fails**: typed detector-extra p95 1.371 ms; full typed p95 2.842 ms. Further gains
   need a CUDA-graph/fixed-shape student and/or a fundamentally cheaper view/index path, not a
   threshold change. The later-event full p95 is 1.051 ms.
2. Teacher labels remain model-generated; the rebind correction removes 1 121 rows rather than
   repairing them (no new annotation). The clean pool is smaller and may be less diverse.
3. S2-vs-S3 and S2-vs-S0 CIs include 0; the parameter-supervision attribution is not established.
   FUSION_S2 is exploratory DEV reuse with a one-shot r3 τ calibration, not independent
   confirmation; its ΔFN CI excludes 0 but is selection-exposed.
4. The 8 adversarial TRAIN block-fallback rows are included via the accepted segment; they are
   only 0.007 % of TRAIN and absent from DEV, but a true controller integration should expose
   typed message boundaries rather than re-parse adversarial text.
5. Base probes are in-sample on TRAIN; the fusion scale is label-free but not an OOF base model.
6. 39 previously-scored zero-span rows and 20 newly scored ones are handled by the declared P1
   policy; 2 old rows are seed argmax. The student episode max is a defined policy, not a
   theoretical equivalence.

## 12. Key SHA-256

| artifact | SHA-256 |
|---|---|
| views_train.jsonl | `ae5c7714a3f67fabc50455bb1c6996aaec5104fde6b1f60b9ac16797a64f5b7c` |
| views_dev.jsonl | `f56ac9c3439fa958001106082a102574c8b784b0faf0e30e00ea5fdf5920bb46` |
| cache_train.npz | `c7b3bc26bd53e4f0c54ad629c34486dea94458d5f022e3969a83e3b59edb3e50` |
| cache_dev.npz | `0cc941ca8b5171e4e7f0be95d08e498acbf351fe8a3d18faca3fb6a6f07113bc` |
| aux_labels_corrected_r3.json | `14cc584979742edfa07bbf84456c85c04d813abb6f11eae8dba80273d1127ceb` |
| results_corrected_r3.json | `1e4cc48e4a03f25fd4edfee301872a85a3dc3500fe826a95a9954def7c680d3a` |
| validation_corrected_r3.json | `e92725c7678fd5d24361f42b18787a24ee8bdcca498bd1035ca8f161a1db4236` |
| cost_r3_summary.json | `cb239525c42870787368f491a0c60f1608c41cfafd64f959880e9ce1f5e377fc` |
| fusion_frozen_corrected_r3_S2.json | `202c5eb4dfbca85184daf0b98fbfac5970223c4e0bdfef36468e36b8a03ece85` |
| fusion_frozen_corrected_r3_S0.json | `4ce6d133d2fb2e480eeacb56fb655ddc6c3456c7a388a3d1bd84ee103e835bd2` |
| replay_S2.npz | `7d2d860fa9a66091b49a3498ade9410431ee62e3d2252944a4964ab53574493b` |
| teacher caches | unchanged (`089e48…`, `9f6f4d…`, `ef11c7…`) |
