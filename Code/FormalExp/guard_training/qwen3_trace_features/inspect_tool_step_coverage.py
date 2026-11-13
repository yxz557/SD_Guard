from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

COVERAGE_KINDS = ("both", "call_only", "result_only", "neither")


def iter_feature_metadata(feature_dir: Path):
    """Yield feature metadata without importing torch/safetensors dependencies."""
    for path in sorted(feature_dir.glob("shard_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect whether extracted AgentDojo traces contain tool-call and "
            "tool-result step boundaries. This command reads metadata only; it "
            "does not load hidden-state tensors or modify a dataset."
        )
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        action="append",
        required=True,
        help="Feature-store root containing features/shard_*.jsonl. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--source-name",
        action="append",
        default=None,
        help=(
            "Display name corresponding to each --feature-root. If supplied, "
            "the number and order must match --feature-root."
        ),
    )
    parser.add_argument(
        "--offline-dataset",
        type=Path,
        default=None,
        help=(
            "Optional compact offline-dataset directory. When supplied, the "
            "report also summarizes only sample IDs present in *.metadata.jsonl."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--examples-per-category", type=int, default=5)
    return parser.parse_args()


def coverage_kind(call_count: int, result_count: int) -> str:
    if call_count > 0 and result_count > 0:
        return "both"
    if call_count > 0:
        return "call_only"
    if result_count > 0:
        return "result_only"
    return "neither"


def enrich_row(row: dict[str, Any], source_name: str) -> dict[str, Any]:
    steps = row.get("steps") or []
    if not isinstance(steps, list):
        raise ValueError(f"Sample {row.get('sample_id')} has a non-list steps field")
    kinds = Counter(
        str(step.get("kind"))
        for step in steps
        if isinstance(step, dict) and step.get("kind") is not None
    )
    call_count = int(kinds["tool_call_end"])
    result_count = int(kinds["tool_result_end"])
    output = dict(row)
    output["data_source"] = source_name
    output["tool_call_count"] = call_count
    output["tool_result_count"] = result_count
    output["tool_step_coverage"] = coverage_kind(call_count, result_count)
    return output


def load_feature_rows(feature_roots: list[Path], source_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for root, source_name in zip(feature_roots, source_names, strict=True):
        feature_dir = root / "features"
        source_rows = list(iter_feature_metadata(feature_dir))
        if not source_rows:
            raise ValueError(f"No shard_*.jsonl metadata found under {feature_dir}")
        for row in source_rows:
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"Feature metadata row under {feature_dir} lacks sample_id")
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id across feature roots: {sample_id}")
            seen_sample_ids.add(sample_id)
            rows.append(enrich_row(row, source_name))
    return rows


def load_offline_annotations(offline_dataset: Path) -> dict[str, dict[str, Any]]:
    paths = sorted(offline_dataset.glob("*.metadata.jsonl"))
    if not paths:
        raise ValueError(f"No *.metadata.jsonl files found under {offline_dataset}")
    annotations: dict[str, dict[str, Any]] = {}
    suffix = ".metadata.jsonl"
    for path in paths:
        output_split = path.name[: -len(suffix)]
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{line_number}")
                sample_id = str(row.get("sample_id") or "")
                if not sample_id:
                    raise ValueError(f"Missing sample_id at {path}:{line_number}")
                if sample_id in annotations:
                    raise ValueError(
                        f"Sample {sample_id} appears in multiple offline metadata rows"
                    )
                annotations[sample_id] = {
                    "offline_split": output_split,
                    "offline_data_source": row.get("data_source"),
                }
    return annotations


def numeric_summary(values: Iterable[int]) -> dict[str, int | float | None]:
    materialized = list(values)
    if not materialized:
        return {"total": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "total": int(sum(materialized)),
        "min": int(min(materialized)),
        "max": int(max(materialized)),
        "mean": float(statistics.fmean(materialized)),
        "median": float(statistics.median(materialized)),
    }


def basic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    labels = Counter(str(int(row.get("label", 0))) for row in rows)
    coverage: dict[str, Any] = {}
    for kind in COVERAGE_KINDS:
        selected = [row for row in rows if row["tool_step_coverage"] == kind]
        positives = sum(int(row.get("label", 0)) for row in selected)
        coverage[kind] = {
            "samples": len(selected),
            "rate": (len(selected) / samples) if samples else None,
            "positives": positives,
            "negatives": len(selected) - positives,
        }
    equal = sum(
        int(row["tool_call_count"] == row["tool_result_count"]) for row in rows
    )
    call_gt = sum(
        int(row["tool_call_count"] > row["tool_result_count"]) for row in rows
    )
    result_gt = samples - equal - call_gt
    positives = int(labels.get("1", 0))
    return {
        "samples": samples,
        "labels": {"0": int(labels.get("0", 0)), "1": positives},
        "positive_rate": (positives / samples) if samples else None,
        "eligible_call_and_result": coverage["both"],
        "coverage": coverage,
        "tool_call_steps": numeric_summary(
            int(row["tool_call_count"]) for row in rows
        ),
        "tool_result_steps": numeric_summary(
            int(row["tool_result_count"]) for row in rows
        ),
        "call_result_count_relation": {
            "equal": equal,
            "calls_greater": call_gt,
            "results_greater": result_gt,
        },
    }


def grouped_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    def group_value(row: dict[str, Any]) -> str:
        value = row.get(key)
        return "<missing>" if value is None or value == "" else str(value)

    values = sorted({group_value(row) for row in rows})
    return {
        value: basic_summary(
            [row for row in rows if group_value(row) == value]
        )
        for value in values
    }


def build_scope_report(rows: list[dict[str, Any]], examples_per_category: int) -> dict[str, Any]:
    examples: dict[str, list[dict[str, Any]]] = {}
    for kind in COVERAGE_KINDS:
        selected = [row for row in rows if row["tool_step_coverage"] == kind]
        examples[kind] = [
            {
                "sample_id": row["sample_id"],
                "data_source": row.get("data_source"),
                "offline_split": row.get("offline_split"),
                "label": int(row.get("label", 0)),
                "suite_name": row.get("suite_name"),
                "attack_type": row.get("attack_type"),
                "user_task_id": row.get("user_task_id"),
                "injection_task_id": row.get("injection_task_id"),
                "tool_call_count": row["tool_call_count"],
                "tool_result_count": row["tool_result_count"],
            }
            for row in selected[:examples_per_category]
        ]
    report = {
        "overall": basic_summary(rows),
        "by_label": grouped_summary(rows, "label"),
        "by_suite": grouped_summary(rows, "suite_name"),
        "by_attack": grouped_summary(rows, "attack_type"),
        "by_source": grouped_summary(rows, "data_source"),
        "examples_by_coverage": examples,
    }
    if any(row.get("offline_split") is not None for row in rows):
        report["by_offline_split"] = grouped_summary(rows, "offline_split")
    return report


def build_report(
    rows: list[dict[str, Any]],
    feature_roots: list[Path],
    source_names: list[str],
    offline_annotations: dict[str, dict[str, Any]] | None,
    offline_dataset: Path | None,
    examples_per_category: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "sd_guard.tool_step_coverage.v1",
        "eligibility_rule": "tool_call_count > 0 and tool_result_count > 0",
        "inputs": [
            {"source_name": name, "feature_root": str(root.resolve())}
            for root, name in zip(feature_roots, source_names, strict=True)
        ],
        "all_feature_rows": build_scope_report(rows, examples_per_category),
    }
    if offline_annotations is not None:
        rows_by_id = {str(row["sample_id"]): row for row in rows}
        missing = sorted(set(offline_annotations) - set(rows_by_id))
        if missing:
            raise ValueError(
                "Offline dataset references samples absent from supplied feature roots: "
                f"{missing[:5]} (total={len(missing)})"
            )
        selected_rows: list[dict[str, Any]] = []
        for sample_id, annotation in offline_annotations.items():
            row = dict(rows_by_id[sample_id])
            row.update(annotation)
            selected_rows.append(row)
        report["offline_dataset"] = str(offline_dataset.resolve()) if offline_dataset else None
        report["offline_selection"] = {
            "selected_samples": len(selected_rows),
            "excluded_feature_rows": len(rows) - len(selected_rows),
            **build_scope_report(selected_rows, examples_per_category),
        }
    return report


def resolve_source_names(feature_roots: list[Path], requested: list[str] | None) -> list[str]:
    if requested is None:
        names = [root.name for root in feature_roots]
    else:
        if len(requested) != len(feature_roots):
            raise ValueError(
                "--source-name count must match --feature-root count: "
                f"{len(requested)} != {len(feature_roots)}"
            )
        names = requested
    if len(names) != len(set(names)):
        raise ValueError(f"Source names must be unique: {names}")
    return names


def main() -> None:
    args = parse_args()
    if args.examples_per_category < 0:
        raise ValueError("--examples-per-category must be non-negative")
    feature_roots = [root.resolve() for root in args.feature_root]
    source_names = resolve_source_names(feature_roots, args.source_name)
    rows = load_feature_rows(feature_roots, source_names)
    annotations = (
        load_offline_annotations(args.offline_dataset.resolve())
        if args.offline_dataset is not None
        else None
    )
    report = build_report(
        rows,
        feature_roots,
        source_names,
        annotations,
        args.offline_dataset,
        args.examples_per_category,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")


if __name__ == "__main__":
    main()
