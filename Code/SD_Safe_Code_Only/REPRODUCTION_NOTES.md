# Reproduction notes for the code-only archive

## Collection scope

Collected from server `autodl-2` on 2026-11-22. This archive contains only source code, tests, scripts, protocols, reports, and integrity manifests. It contains no raw trajectory, no processed data, no `.npz`, `.pt`, `.bin`, `.jsonl`, DeepSeek cache, model/tokenizer file, or credential.

## Upstream trace and label processing

The former top-level repository `/root/autodl-tmp/SD-Safe/src` and `/scripts` had already been cleared. The retained historical source snapshot is therefore under:

```text
upstream-sequence/source_snapshot/
```

The most relevant modules are:

- `src/sdsafe_fine_confirm/sources.py`: trace/source ingestion;
- `src/sdsafe_fine_confirm/train_data.py`: upstream training-data assembly;
- `src/sdsafe_fine_label_repair/data.py`: event/context reconstruction;
- `src/sdsafe_fine_label_repair/features.py`: upstream hidden-state feature processing;
- `src/sdsafe_fine_label_repair/labels.py`: checker-label repair and masks;
- `src/sdsafe_sequence_explore/data.py`: `sequence_v1` construction;
- `src/sdsafe_sequence_explore/labels.py`: sequence and prefix labels.

`upstream-sequence/accepted-data-builder/` is an additional accepted-data construction snapshot, including source/label tests and integrity manifests. Dataset handoff metadata was intentionally excluded with the other data-facing artifacts.

## V6/Rule method

The primary implementation lives in `v6-rule/src/sdsafe_v6/`:

- `view.py`: causal event view (`user`, `src`, `cur`, `vis`);
- `features.py`: hashed unigram/bigram buckets and 32 dense relation features;
- `teacher.py`, `aux_data.py`: offline DeepSeek auxiliary-label prompting, parsing, and two-pass quality controls;
- `model.py`, `train.py`: V6 student detector and auxiliary heads;
- `baseline.py`, `freeze_fusion_v3.py`: old hidden-state score handling and FUSION_S2;
- `rule_taint.py`: causal directive-to-sink flow rule;
- `evaluate_rule.py`: combines rule evidence with FUSION_S2.

## Original runtime reference

The recorded final runtime was:

```text
Python       3.12.3
PyTorch      2.14.0+cu130
NumPy        2.3.5
Transformers 5.20.2
```

This is a reference, not a bundled environment.

## Required path refactor before reuse

Some historical code has hard-coded paths such as:

```text
/root/autodl-tmp/SD-Safe
/root/sdsafe-compute/v4-gpu
/root/sdsafe-compute/v5-depth
/root/sdsafe-compute/v6-param
/root/sdsafe-compute/v6-rule
/dev/shm/sdsafe-gpu-v4/upstream/...
```

Before running it on a new server, replace these with a config file, command-line arguments, or environment variables. The code archive itself deliberately does not recreate those data locations.

## Deliberately excluded

- raw Task Injection traces and ground truth;
- `sequence_v1` rows, labels, BF16 hidden states, and split data;
- V5 prepared token arrays;
- V6 views, feature caches, DS teacher caches, score files, and checkpoints;
- Qwen3-8B tokenizer/model and all other weights;
- credentials, including the DeepSeek API key;
- runtime logs, generated reports, `.git` object databases, and `__pycache__`.
