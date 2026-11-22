# STATUS — runtime-r4 (2026-11-14, after corrected-r3)

**Stage:** cache-only runtime optimization of the frozen corrected-r3 detector. **New annotation
tasks/API requests: 0.** No TEST read. No training, no label/feature/threshold/fusion change.
Old r1/r2/r3 artifacts and reports are retained untouched; all new code/results use `runtime-r4`.

## Runtime result

- Static-shape CUDA-graph backend for the same 3 frozen S2 students (0 fallbacks / 25 003
  events), plus a CPU 2-thread pilot backend. `view._true_starts` no longer copies the prefix per
  candidate (mechanically identical; full parity 0 bad rows).
- Full common-DEV parity 25 003/25 003 events for both rawtext and typed ingest: views/roles/
  call-bind/retrieval/value-sources 0 bad, bags exact, dense bitwise 0, CT logits max |Δ| 3.05e-5,
  per-seed episode max 5.72e-6, fused 2.84e-7.
- Frozen τ unchanged: `0.6289290373189838` (`0x1.4202fc9d94608p-1`), `>=`, no ε. Accuracy
  FUSION_S2 **12 FN / 79 FP**, AUC 0.9903471437; S2 13/79, AUC 0.9876218954. One reference
  exact-τ tie moves 2.4e-8 below τ under the graph backend (80→79 FP); reported, not tuned.
- Timing scope corrected: R charges `parse_blocks`/JSON/role/source in the detector; T’s adapter
  is measured and charged in `T integrated`. r3’s “parse outside the timer” is no longer used.

## Cost (3 OS processes × 3 repeats, full 25 003 events, batch 1)

| candidate | p95 median | p95 worst | gate |
|---|---:|---:|---|
| A tokens-only | 1.669 ms | 1.736 ms | reported only |
| **R rawtext (all parse charged)** | **1.037 ms** | **1.062 ms** | **fail, borderline** |
| **T integrated (detector + adapter)** | **1.039 ms** | **1.068 ms** | **fail, borderline** |
| T pure (no caller in this source) | 0.916 ms | 0.940 ms | not claimed |

`cost_passed=false`; 3.7–6.8 % above the original 1 ms bar. Vs r3: A p95 −23 %, B detector
−24 % (now including parse), B full p95 −20 %. Late events 0.625 ms p95; first events 1.43 ms.

## Validation

- `logs/tests_r4.log`: **40/40 OK** (32 inherited + 8 new).
- Teacher caches, checkpoints, scaler, fusion JSONs and old r3 artifacts match `HANDOFF_R3.json`
  (0 mismatches); no cache line edited.
- CPU authoritative copy `/root/autodl-tmp/sdsafe-lowcost-authorization-v6-param/runtime-r4/`
  (atomic, SHA-verified; `CPU_SYNC_R4.json`).
- No background jobs after the run; GPU 0 MiB.

## Files

`RUNTIME_R4_REPORT.md`, `HANDOFF_R4.json`, `src/sdsafe_v6/runtime_r4.py`,
`scripts/{bench,parity,pilot,summarize,accuracy,manifest}_r4.py`, `tests/test_runtime_r4.py`,
`artifacts/corrected-r3/runtime-r4/`.

---

# STATUS — corrected-r3 (2026-11-14)

**Stage:** cache-only corrected-r3 complete. **New annotation tasks/API requests: 0.** No TEST
read. Old r1/r2 data/results/checkpoints/thresholds unchanged and retained.

## Correctness

- F1 fixed: 7 813/7 813 DEV `tool_call` rows bind `call_index` ordinal **and** the row’s own CT
  segment (0 conflicts); 830 DEV / 3 401 TRAIN current-call changes reported. JSON-string-aware
  block parsing also fixes forged `<|im_end|>`/`<|im_start|>` inside tool-call arguments.
- Teacher rebind: 5 045 truly deduped cached rows (old 4 355 was a digest-collision undercount);
  1 121 masked (312 bound a previous call, 897 prior material changed; overlap), 3 793 clean
  both-pass, trainable action 3 476 / rel 3 317 / align 3 501 / arg 1 721. Frozen teacher SHAs
  unchanged; no cache line edited, no retry.
- Empty policy: 59 DEV zero-new-token rows (not 20) all scored under declared P1
  (`empty_segment_with_visible_text`); 0 `no_new_emission`, 0 `UNKNOWN_ADAPTER`. A6 retracted:
  the 20 previously-missing rows are never base/student argmax; 2 old-39 rows are seed argmax.
- Validation 32/32; unit tests 32/32; full mode-A and typed-B parity 25 003/25 003 events, 0
  mismatches; 83/85 old SHA checks OK (only the two intended source files changed).

## Accuracy (fixed menu, corrected DEV, exploratory reuse)

| method | FN | FP | AUC |
|---|---:|---:|---:|
| frozen baseline (21/78) | 21 | 78 | 0.981127 |
| S0 main | 16 | 80 | 0.987914 |
| **S2 λ1** | **13** | **80** | 0.987622 |
| S3 shuffled | 16 | 80 | 0.985519 |
| FUSION_S0 | 18 | 80 | 0.989365 |
| **FUSION_S2** | **12** | **80** | **0.990347** |

FUSION_S2 vs base ΔFN −9, 95 % group-bootstrap CI [−17, −2] (exploratory; one-shot r3 τ).
S2 vs S3 ΔFN −3, CI [−8, 0] — parameter-supervision attribution still not established.
Old 13/80, 11/80 and r2 11/79 remain legacy and untouched.

## Cost (3 sequential OS processes × 3 repeats, full DEV, batch 1)

| mode | p50 | p95 | p95 worst |
|---|---:|---:|---:|
| A tokens-only | 0.867 | 2.172 | 2.218 |
| B typed, detector-extra | 0.801 | **1.371** | 1.400 |
| B typed, full incl. controller decode | 0.876 | 2.842 | 2.915 |

`p95 ≤ 1 ms` **not met**; cost_passed=false. Typed path is proven against the real source
(25 003-event parity), not an oracle view. Later events full p95 1.051 ms; first-event message
finalisation dominates.

## Gate summary

- F1 / teacher mis-binding fixed: **yes** (validation + oracle tests).
- Accuracy candidate: S2 13/80, FUSION_S2 12/80; FUSION_S2 CI excludes 0 but exploratory.
- Cost gate: **no** (typed detector-extra 1.371 ms p95; full 2.842 ms).
- New annotations: **0**. TEST read: **no**. Background jobs after run: **none**.

## Files

`CORRECTED_R3_REPORT.md`, `HANDOFF_R3.json`, `PROTOCOL_CORRECTED_R3.md`,
`artifacts/corrected-r3/`, CPU authoritative
`/root/autodl-tmp/sdsafe-lowcost-authorization-v6-param/corrected-r3/`.
