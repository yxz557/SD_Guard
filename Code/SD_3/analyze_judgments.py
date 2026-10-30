#!/usr/bin/env python3
"""Merge blind human/LLM judgments with Guard outputs and compute metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from sdguard_sd.io_utils import iter_jsonl, write_json, write_jsonl
from sdguard_sd.metrics import confusion_metrics, is_unsafe
from sdguard_sd.policies import SAFE, UNSAFE, normalize_label


SCOPES = ("full_trace", "answer_only")
POLICIES = ("strict", "loose")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join blind Safe/Unsafe labels by sample_id and score Stream Guard."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--judge-labels", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <experiment-dir>/analysis",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Score only supplied sample_ids instead of requiring the complete blind set.",
    )
    parser.add_argument(
        "--guard-view",
        choices=("auto", "posthoc", "proposal_immediate", "verified_commit"),
        default="auto",
        help=(
            "auto uses proposal_immediate for inline_guard and posthoc otherwise. "
            "The other choices explicitly select one saved decision view."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _scope_judgment(row: Mapping[str, Any], scope: str) -> dict[str, Any]:
    nested = row.get(scope)
    if isinstance(nested, Mapping):
        raw_label = nested.get("label")
        categories = nested.get("categories", [])
        reason = nested.get("reason", "")
    else:
        prefix = "trace" if scope == "full_trace" else "answer"
        raw_label = row.get(f"{prefix}_label")
        categories = row.get(f"{prefix}_categories", [])
        reason = row.get(f"{prefix}_reason", row.get("reason", ""))
    if raw_label is None:
        raise ValueError(f"Missing {scope} binary label")
    label = normalize_label(raw_label)
    if label not in {SAFE, UNSAFE}:
        raise ValueError(
            f"{scope}.label must be Safe or Unsafe; got {raw_label!r}"
        )
    if categories is None:
        categories = []
    if isinstance(categories, str):
        categories = [categories]
    if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
        raise ValueError(f"{scope}.categories must be a list of strings")
    if not isinstance(reason, str):
        raise ValueError(f"{scope}.reason must be a string")
    return {"label": label, "categories": categories, "reason": reason}


def load_judgments(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(iter_jsonl(path), start=1):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"{path}:{line_number} has no sample_id")
        if sample_id in result:
            raise ValueError(f"Duplicate judgment for sample_id {sample_id!r}")
        try:
            result[sample_id] = {
                scope: _scope_judgment(row, scope) for scope in SCOPES
            } | {
                "confidence": row.get("confidence"),
                "overall_reason": row.get("reason", ""),
            }
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} ({sample_id}): {exc}") from exc
    if not result:
        raise ValueError(f"No judgments found in {path}")
    return result


def _load_generation_records(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ValueError(f"Duplicate generation record for sample_id {sample_id!r}")
        result[sample_id] = row
    if not result:
        raise ValueError(f"No generation records found in {path}")
    return result


def _guard_payload(row: Mapping[str, Any], view: str) -> tuple[str, Mapping[str, Any]]:
    selected = view
    if selected == "auto":
        selected = (
            "proposal_immediate"
            if row.get("generation_mode") == "inline_guard"
            else "posthoc"
        )
    if selected == "posthoc":
        return selected, row["guard"]
    inline = row.get("inline_guard")
    if not isinstance(inline, Mapping) or selected not in inline:
        raise ValueError(
            f"Generation record {row.get('sample_id')!r} has no inline Guard view {selected!r}"
        )
    return selected, inline[selected]


def _prepare_output(path: Path, overwrite: bool) -> None:
    expected = ("metrics.json", "metrics.csv", "merged_results.jsonl", "errors.jsonl")
    existing = [path / name for name in expected if (path / name).exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(item) for item in existing)
        raise FileExistsError(f"Analysis outputs already exist: {rendered}; pass --overwrite")
    path.mkdir(parents=True, exist_ok=True)
    for item in existing:
        item.unlink()


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scope",
        "policy",
        "sample_count",
        "excluded_unavailable_scope",
        "tp",
        "tn",
        "fp",
        "fn",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "specificity",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else experiment_dir / "analysis"
    )
    _prepare_output(output_dir, args.overwrite)

    generations = _load_generation_records(experiment_dir / "raw" / "generation_records.jsonl")
    judgments = load_judgments(args.judge_labels.expanduser().resolve())
    generation_ids = set(generations)
    judgment_ids = set(judgments)
    unknown = sorted(judgment_ids - generation_ids)
    missing = sorted(generation_ids - judgment_ids)
    if unknown:
        raise ValueError(f"Judgment file contains unknown sample_ids: {unknown[:10]}")
    if missing and not args.allow_partial:
        raise ValueError(
            f"Judgment file is missing {len(missing)} sample_ids; examples: {missing[:10]}. "
            "Pass --allow-partial only for an intentional partial analysis."
        )

    selected_ids = [sample_id for sample_id in generations if sample_id in judgments]
    merged_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selected_views = {_guard_payload(row, args.guard_view)[0] for row in generations.values()}
    if len(selected_views) != 1:
        raise ValueError(f"Analysis mixes Guard views: {sorted(selected_views)}")
    selected_view = next(iter(selected_views))

    for scope in SCOPES:
        for policy in POLICIES:
            pairs: list[tuple[bool, bool]] = []
            excluded = 0
            for sample_id in selected_ids:
                generation = generations[sample_id]
                _, guard_payload = _guard_payload(generation, args.guard_view)
                guard_scope = guard_payload["scopes"][scope]
                if not guard_scope["available"]:
                    excluded += 1
                    continue
                truth_label = judgments[sample_id][scope]["label"]
                guard_label = normalize_label(guard_scope["decisions"][policy]["prediction"])
                truth_unsafe = is_unsafe(truth_label, policy)
                prediction_unsafe = guard_label == UNSAFE
                pairs.append((truth_unsafe, prediction_unsafe))
                if truth_unsafe != prediction_unsafe:
                    kind = "false_negative" if truth_unsafe else "false_positive"
                    detail = {
                        "sample_id": sample_id,
                        "scope": scope,
                        "policy": policy,
                        "error_type": kind,
                        "judge_label": truth_label,
                        "judge_categories": judgments[sample_id][scope]["categories"],
                        "judge_reason": judgments[sample_id][scope]["reason"],
                        "guard_prediction": guard_label,
                        "guard_decision": guard_scope["decisions"][policy],
                        "prompt": generation["prompt"],
                        "thinking": generation["completion"]["thinking"],
                        "final_answer": generation["completion"]["final_answer"],
                    }
                    error_rows.append(detail)
            scores = confusion_metrics(pairs)
            metric_rows.append(
                {
                    "scope": scope,
                    "policy": policy,
                    "excluded_unavailable_scope": excluded,
                    **scores,
                }
            )

    for sample_id in selected_ids:
        generation = generations[sample_id]
        _, guard_payload = _guard_payload(generation, args.guard_view)
        guard_scopes = guard_payload["scopes"]
        merged_rows.append(
            {
                "sample_id": sample_id,
                "judgment": judgments[sample_id],
                "guard_scopes": guard_scopes,
                "prompt": generation["prompt"],
                "completion": generation["completion"],
            }
        )

    category_counts: Counter[str] = Counter()
    for sample_id in selected_ids:
        for scope in SCOPES:
            category_counts.update(judgments[sample_id][scope]["categories"])

    report = {
        "schema_version": 1,
        "experiment_dir": str(experiment_dir),
        "judge_labels": str(args.judge_labels.expanduser().resolve()),
        "guard_view": selected_view,
        "generation_sample_count": len(generations),
        "judged_sample_count": len(selected_ids),
        "partial_analysis": bool(missing),
        "missing_sample_count": len(missing),
        "label_semantics": {
            "ground_truth": "Unsafe is positive; Safe is negative",
            "strict_guard": "earliest adjacent UU or CC pair is positive",
            "loose_guard": "earliest adjacent UU pair is positive",
        },
        "metrics": metric_rows,
        "judge_category_counts": dict(category_counts.most_common()),
    }
    write_json(output_dir / "metrics.json", report)
    _write_metrics_csv(output_dir / "metrics.csv", metric_rows)
    write_jsonl(output_dir / "merged_results.jsonl", merged_rows)
    write_jsonl(output_dir / "errors.jsonl", error_rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nAnalysis written to: {output_dir}")


if __name__ == "__main__":
    main()
