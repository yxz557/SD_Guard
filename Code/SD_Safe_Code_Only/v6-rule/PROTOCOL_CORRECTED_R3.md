# PROTOCOL_CORRECTED_R3 — corrected-r3 (declared before any corrected DEV score)

This file is written **before** the corrected-r3 views/features are exported and before any
r3 model is fitted or scored. New annotation tasks / API requests remain **0**. The three
teacher caches are frozen (SHA-256 unchanged). No TEST material is read. Old r1/r2 artifacts,
checkpoints, thresholds and reports are read-only and retained under their old names; all new
data/results use the `corrected-r3` name.

## C1. Call-localisation fix (F1)

The current action of a `tool_call` CT row is the call that the row's own readout span covers,
not the first `<tool_call>` in the assistant block.

Localisation uses, in this order:

1. A balanced-brace scanner over the decoded assistant block that is aware of JSON string and
   escape state, so a literal `<tool_call>` inside a JSON string, nested JSON, escaped quotes,
   unclosed later calls and quoted/pseudo tags do not create false calls.
2. `<think>...</think>` spans are treated as reasoning: tags inside them are not tool calls.
3. The row's `call_index` ordinal selects the candidate call among the calls visible in the
   causal prefix (ordinal is an event identity, not a model feature).
4. The row's actual CT `segment` (decoded `ids[token_start:token_end]`) is parsed independently
   and must agree with the ordinal candidate on tool name **and** decoded argument object.
   If it disagrees and the segment candidate is unique, the segment wins and the conflict is
   recorded (`call_match="segment"` / `call_binding_conflict=true`). Only unique
   name+argument equality is used; the mapping is never keyed by a label or metadata id.

No token after the row endpoint is ever decoded or parsed to build the current action. All
TRAIN/DEV views and feature caches are regenerated under `artifacts/corrected-r3/`;
old view/feature files are untouched. The number of rows whose current tool/arguments change
is reported, not estimated.

## C2. Cached-teacher binding mask (F1 consequence)

Teacher prompts in the frozen caches were rendered from the stored aux views. For every cached
teacher row we rebuild the corrected view and compare the exact material the teacher saw:
`user_request`, the `prior` block list (role/text), `current.tool`, `current.arguments`, and the
argument `value_sources` paths.

- Any row whose current call (name or arguments) differs from the corrected row is **masked in
  all teacher fields** (action, relation, alignment, argument-level), because the teacher judged
  the previous call, not this one.
- Any row whose user request or prior material differs, or whose cited quote / argument path is
  not literally present in the corrected causal material, is masked in the affected fields.
- Both passes must remain present, parseable, identical on the field, and valid on the quotes.
- The existing 95 forged-trusted-role exclusions and the 132 single-pass masks are retained.
- No cache line is edited, no retry, no relabel, no new request. GT labels and teacher answers
  remain outside features.

## C3. Zero-new-token row policy (declared before scoring; no GT used)

Diagnostics before this declaration found 59 common-DEV CT rows with `token_start == token_end`
(all `content_segment`; 39 had been silently scored by the old path and 20 had been dropped by
an old strict span check). The policy is structural, decided without labels:

- **P1 `empty_segment_with_visible_text`**: a zero-new-token row whose causal assistant text
  emitted before the endpoint (`visible_so_far`, after removing `<think>` reasoning) is
  non-empty is kept as a text event with `segment=""`. It is scored from the same causal visible
  prefix (the vis channel is non-empty; nothing is invented, no future token is used). A
  `empty_segment` flag is stored.
- **P2 `no_new_emission`**: a zero-new-token row with no visible assistant text is kept in the
  stream (history/index may advance) but produces **no student logit**; the per-seed running
  max is not updated. This is an explicit model-applicability policy, not a claim that a
  non-existent row equals a max.
- **P3 `UNKNOWN_ADAPTER`**: a zero-span row typed `tool_call` (or any row where the adapter
  demonstrably dropped a visible tool call) produces no logit and is reported separately; it
  is never repaired from future tokens.
- Episode score is defined as the max over rows with a valid student view. If an episode has no
  valid row, its score is `-inf` (policy). All 2 028 episodes / 25 003 CT rows are retained and
  coverage is reported. The old statement that these rows "equal the episode max" is retracted.

## C4. Registered r3 menu (fixed before corrected DEV is read)

Exactly three arms, matched capacity/label pool/steps:

| arm | supervision |
|---|---|
| S2 | main security BCE + parameter-level aux heads, lambda = 1.0 |
| S0 | main security BCE only |
| S3 | S2 with the aux labels shuffled (capacity/supervision control) |

- Fixed **5 epochs**, batch 64 (16 pos / 48 neg), 600 steps/epoch, AdamW lr 1e-3, wd 1e-2,
  dropout 0.2, hashed bags 2^17x16, hidden 256 (unchanged architecture).
- **3 seeds {0,1,2} -> 9 formal fits**. No epoch scan, no LR sweep, no S4, no extra arm.
- Group-disjoint folds from `blake2b(group_id) % 4`; a group never spans sides. The same source
  crosses scopes but never crosses the TRAIN/DEV side.
- Normaliser, dictionary, retrieval, few-shot and fitting read TRAIN only. Main label is the
  real frozen pool label; no teacher unsafe label is treated as a strict goal.
- Auxiliary TEACHER checks needed for the fixed menu are listed separately from the 9 formal
  fits (8 TRAIN-OOF fits used only for the frozen fusion scale).

## C5. Fixed frozen fusion (no fitted DEV meta-model)

`fused = 0.5*z_base + 0.5*z_student`, labels never used for weights.

- `z_base` uses the frozen base TRAIN episode mean/std (recomputed from the unchanged frozen
  base packs, float64).
- `z_student` uses TRAIN-only **outer-held-group OOF** episode scores (one model per held-out
  fold, 5 epochs, group-disjoint; 8 auxiliary fits) and `finite` rows only; no DEV label, no
  fold-0 selection leak (there is no epoch selection in r3).
- Fusion arithmetic in float64; per-seed CT logits float32, per-seed episode max float32,
  seed mean float32 arithmetic, as in the reference.
- The r3 fusion threshold is calibrated **once**, on DEV, at its own full-sample `<=5 %` FPR
  using the **actual deployment batch-1/fusion implementation** (`>=`, exact float64), then
  serialised as exact `float.hex()` and frozen. No epsilon, no rounding to decimal places, no
  repeated DEV tuning. This single calibration is declared now; r3 DEV numbers are therefore
  exploratory reuse, not an independent confirmation.
- The old base packs and old thresholds (21/78, 13/80, 11/80, literal 11/79) are never moved or
  re-derived. They remain as the legacy comparator, including the old field-semantics version.

## C6. Cost interfaces (both reported)

- **A tokens-only compatible path**: every event re-establishes history by decoding the actual
  causal token stream; all decode/index/retrieval/value-source/3-student/base-cache/max/mean/
  fusion work is inside the timer. It may remain > 1 ms; that is reported.
- **B controller-native typed-message path**: the controller supplies the role-tagged message
  text it already has (user/tool raw text; assistant deltas up to the CT endpoint) to an
  incremental adapter. The detector charges message ingest, byte handling, current JSON parse,
  word hashing, source/value index, feature transfer, all three student forwards, per-seed max,
  mean and fusion. The controller-side shared decode/formatter is measured and reported
  separately, never hidden in fixture construction. Inputs are proofed against the reference
  source by per-event text/view/feature/logit parity over all common-DEV events; if parity is
  impossible the path is reported as a prototype limitation, not as <=1 ms.
- Both paths: batch 1, causal in-episode order, no shuffling, cold/warm/peak and per-scope /
  per-event-kind reported, full common DEV including long contexts. Three independent OS
  processes run **sequentially** (3 repeats each), no other GPU job. The 1 ms p95 gate is the
  original bar; if the real integrable path is 1–2 ms the complete tradeoff is reported.

## C7. Reporting / gates

- Correctness first: call-localisation oracle tests (first/second/third, same name different
  args, literal tag in JSON, nested/escaped JSON, CoT pseudo-tags, unclosed later call), teacher
  rebind/mask tests, empty policy, forged-role and future-suffix invariance, exact-tau replay,
  cache chunk/role/eviction vs full parity.
- Then accuracy: baseline 21/78 independent; S2/S0/S3 own-FPR; fused; point deltas and
  fixed-threshold whole-group bootstrap CIs; scope/workspace/travel-IT6 slices and error
  movement. All r3 DEV results are exploratory reuse.
- Then cost. The final line is `CORRECTED_STATUS_JSON` with the required fields.
