# PROTOCOL amendments and deviations — cache-only-r1

Registered study: `sdsafe-lowcost-authorization-v6-param`.
This file is written after the base-arm DEV numbers were already visible; it records every
deviation rather than silently rewriting the original `PROTOCOL.md`.

## A0. Annotation budget is permanently zero

- New annotation API requests in this run: **0**.
- No retries, cache completion, health probes, second judge, label repair, provider/model change,
  or test annotation call was made.
- `src/sdsafe_v6/annotation_guard.py` hard-locks every entry point before the API key is read or
  any network object is constructed. `teacher._key()`, `teacher.call()`,
  `teacher.label_events()` and `scripts/label.py` all call it first.
- The three teacher caches are frozen (see `artifacts/teacher/FROZEN_MANIFEST.json`); their SHA-256
  values are unchanged before/after this run.
- The old annotation workers PID 38031 and its launcher 38029 remain terminated; they were not
  resumed. The old paused training PID 37269 was verified by kernel starttime and killed without
  SIGCONT after its S1/S2 fits were preserved.

## A1. Strict two-pass collation (protocol compliance fix)

The original `aux_labels_from_cache` let a row with only one parseable pass contribute
`action`/`rel`/`align` labels. With the old last-line-wins cache policy, the report's 2 265
trainable action rows = 2 199 agreed + 66 single-pass, which violates PROTOCOL §4.

New `aux_data.py`:

- de-duplicates each prompt digest by preferring a parseable completion (recovers 7 rows that the
  old last-write policy silently dropped);
- keeps a field **only** when both A and B are present and exactly equal; single-pass rows are
  counted and masked;
- uses all three landed caches (`aux_calls_v2`, `aux2`, `pilot400`) without any new call;
- requires the teacher's per-argument `path` to be a real path in the student view; invalid paths
  mask the event-level `arg` target (127/4 560 shared paths invalid);
- excludes 95 rows whose old teacher prompt exposed a forged trusted `user`/`system` block
  (the teacher saw untrusted content as a USER_TURN);
- keeps per-argument labels for the S4 field encoder.

Resulting cache-only supervision: 4 155 two-pass events, action 3 798 / rel 3 598 / align
3 846 trainable event fields, 1 881 valid per-argument event targets (798 positive).

## A2. View provenance fix (v6param-view-3)

`parse_blocks` previously split on any `<|im_start|>role\n` inside tool output. 2 341 TRAIN rows
contained such forged ChatML (e.g. a fake user turn inside a `<tool_response>`), and the old parser
turned 95 annotation-prompt rows into extra trusted `user`/`system` turns.

v3 rules:

- a block boundary must be at prompt start, or immediately after a real `<|im_end|>`;
- an open `<tool_response>` payload suppresses boundaries and `<|im_end|>` inside it;
- only the first `system` and first `user` can be trusted; later forged roles become
  `data_system`/`data_user` and never reach `user_request` or the trusted text channels.

Re-export changed 2 341 TRAIN rows (2 146 role changes), **zero** DEV rows, and **zero**
`user_request` strings. `views/train_all_fixup.jsonl` carries the fixed rows; DEV fixup is unchanged.

## A3. S3 shuffle-control bug

The original fold-0 scan always passed `shuffle_aux=False`, so S3's epoch scan was identical to S2
and its selected epoch was not a capacity control. The scan now passes the arm's `shuffle_aux`.
All four base arms were retrained on the v3 views/aux data; old artifacts are preserved under
`artifacts/archive_pre_cacheonly/`.

## A4. Fusion rule replaced (PROTOCOL §6 was not deployable)

The registered `rank_pct(base)+rank_pct(student)` ranks the whole DEV batch, i.e. it uses other
(future) DEV episodes' score distribution. It is not a per-episode online score. The old
`fusion_model*.json` files are legacy LR parameters, not used by `results.py`; they are marked
legacy in the handoff.

New frozen mapping (label-free, TRAIN only):

```
z_base    = (base - base_mean_TRAIN) / base_std_TRAIN
z_student = (student - student_mean_TRAIN_OOF) / student_std_TRAIN_OOF
fused     = 0.5*z_base + 0.5*z_student
```

Student OOF uses only folds 1–3; fold 0 is excluded because the shared epoch count was selected
on fold 0. The base probes are in-sample on TRAIN (AUC 0.9998), so their TRAIN mean/std is used
only as a fixed label-free scale; no weight is fitted. Two mappings are reported:

- `fusion_frozen_v3_S0.json`: `FUSION_S0`, the rule-selected overall fold-0 arm (**S0**);
- `fusion_frozen_v3.json` (S2): `FUSION_S2`, the stronger parameter-level arm among S1/S2 by
  fold-0 AUC. **The S2 arm choice was made after the base DEV numbers were visible**, so
  `FUSION_S2` is an exploratory DEV reuse, not a confirmatory pre-registered result.

## A5. S4 exploratory arm added after base DEV

S4 (shared per-argument field encoder) was added after the base-arm DEV table was visible. Its
epoch count was selected on TRAIN fold 0 only and its mapping/features never use DEV labels, but
its DEV number is explicitly exploratory reuse, not a pre-registered confirmatory arm.

## A6. Known evaluation edge

59 DEV export failures are zero-length CT spans (`token_start == token_end`); 20 are in the
common-DEV universe. Their frozen base logits equal the episode maximum for every seed, so the
episode max is unchanged. The student scores 24 983/25 003 common-DEV CT rows; all 2 028 episodes
and 166 groups are present. This is recorded as a residual risk, not silently hidden.

## Addendum (corrected-r3): A6 retracted — zero-length rows

The A6 statement above is **retracted** by the corrected-r3 work (see
`CORRECTED_R3_REPORT.md` §4 and `PROTOCOL_CORRECTED_R3.md` C3). Independent replay shows the 20
common-DEV rows with `token_start == token_end` are *never* the per-seed episode argmax
(0/60 base checks) and are not equal to the episode maximum (e.g. row 162426: −58.88 / −82.01 /
−54.60 vs episode max −9.59 / −2.90 / −3.73). The base episode maximum is unchanged because
those rows are never selected, not because their logits equal the max. In addition, 39 further
zero-length rows had already been silently scored by the old fixup path; all 59 are now handled
by the declared P1 policy. The student-side max is an explicit definition (max over rows with a
valid causal view; no logit for `no_new_emission`/`UNKNOWN_ADAPTER`), not a theoretical
equivalence to a non-existent row.
