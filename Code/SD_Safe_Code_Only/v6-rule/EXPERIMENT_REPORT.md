# v6-rule general causal-flow experiment

## Result

This is a whole-DEV development evaluation, not an evaluation of only the nine inspected Travel
false negatives.

| Common DEV metric | corrected-r3 FUSION_S2 | General causal fusion |
|---|---:|---:|
| TP | 402 | 412 |
| FN | 12 | 2 |
| FN rate | 2.8986% | 0.4831% |
| FP | 80 | 80 |
| FP rate | 4.9566% | 4.9566% |
| AUC | 0.990347 | 0.992980 |

Ten of the twelve frozen-base false negatives are recovered, no previously detected positive is
lost, and no false positive is added at the calibrated 5% DEV budget. The recovered cases comprise
nine Travel text-propagation episodes and one Slack URL-to-message episode. The two remaining
Banking cases are recognized by the tool-flow feature, but TRAIN selects a conservative tool
weight and their combined scores remain below the calibrated threshold.

## Evidence that the rule is broader than one Travel task

The text branch is necessarily concentrated in text-output attacks, but the same source-to-sink
tracker is also exercised by tool attacks. On common DEV:

| Raw feature | Episodes | Positive | Negative | Suites represented |
|---|---:|---:|---:|---|
| `text_object_flow` | 16 | 16 | 0 | Travel |
| `tool_identifier_flow` | 376 | 312 | 64 | Banking, Slack, Travel, Workspace |
| Either feature | 392 | 328 | 64 | Banking, Slack, Travel, Workspace |

On TRAIN, the tool feature hits 896 episodes (754 positive and 142 negative), again spanning all
four suites. This is why the feature cannot honestly be used as a hard attack label: a copied
payload can be an intermediate or benchmark-negative step. The learned score and calibrated
threshold retain responsibility for the final decision.

The implementation contains no suite/task branch, no known hotel/account/URL, no title-case
requirement, and no fixed two- or three-word threshold. Unit tests explicitly cover one-, two-,
and three-word text objects plus Slack-like URL flow and Banking-like account mutation.

## Selection protocol

- Universe: 8,229 TRAIN episodes and 2,028 common-DEV episodes (414 positive, 1,614 negative).
- Rule weights: selected only from grouped TRAIN out-of-fold scores.
- Selected weights: text `0.9`, tool `0.1`.
- Mean grouped TRAIN AUC used for selection: `0.998949`.
- DEV threshold: calibrated to the 5% false-positive budget.
- TEST: not present and not opened.

A DEV-only diagnostic grid can improve the number further, but it is intentionally not the
reported configuration because choosing it after reading DEV errors would be overfitting.

## Reproduce

From `/root/sdsafe-compute/v6-rule`:

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m unittest tests.test_rule_taint
PYTHONPATH=src /root/miniconda3/bin/python scripts/evaluate_rule.py
```

Generated files under `artifacts/rule-v2/`:

- `report.json`: selection protocol, whole-TRAIN/DEV feature audit, final metrics, and exact FN IDs;
- `evidence_train.jsonl` and `evidence_dev.jsonl`: auditable source-to-sink evidence;
- `scores_dev.npz`: aligned base score, two rule features, hybrid score, and thresholds.

## Validity boundary

The hypothesis was motivated by inspected DEV errors, and the common DEV threshold is calibrated
on that same development split. The result demonstrates internal consistency and a useful
candidate method, not unseen-data generalization. Freeze this version before evaluating any
future untouched TEST split.
