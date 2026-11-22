# SD-Safe code-only archive

This folder is a **code and documentation archive only**. It deliberately excludes every dataset, trace, feature cache, label cache, model checkpoint, model weight, DeepSeek response cache, log, and credential.

## Contents

| Folder | Purpose |
|---|---|
| `upstream-sequence/` | Historical source snapshots for converting raw AgentDojo/Qwen traces into the accepted `sequence_v1` event dataset and labels. |
| `v4-hidden-baseline/` | The earlier hidden-state detector and its evaluation/migration scripts. |
| `v5-depth-preprocessing/` | Exact token/round/event-span preparation used before V6 builds event views. |
| `v6-rule/` | Current V6 hashed-text detector, DeepSeek auxiliary-label pipeline, FUSION_S2 scripts, and causal-flow rule. |

## Important limitation

This archive is **not runnable by itself**. The experiment code refers to external inputs that are intentionally not included here: the original traces, accepted `sequence_v1` data, prepared token arrays, old baseline scores/training pool, Qwen tokenizer/model, and trained checkpoints.

It is intended to preserve the method and every currently available processing script, so that a future environment with those inputs can recreate or modify the pipeline.

## Code-path overview

```text
raw AgentDojo/Qwen trace
  -> upstream-sequence (source parsing, label repair, sequence construction)
  -> v5-depth-preprocessing (token/round/span preparation)
  -> v6-rule (event view, hashed features, DS auxiliary labels, student training, fusion)
  -> v6-rule/rule_taint.py (causal directive-to-sink rule)
```

`v6-rule/src/sdsafe_v6/rule_taint.py`, `v6-rule/scripts/evaluate_rule.py`, and `v6-rule/tests/test_rule_taint.py` are included even though they were untracked in the server Git checkout; they are required for the current causal-flow rule.

See [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) for source origins, exclusions, and hard-coded path notes.
