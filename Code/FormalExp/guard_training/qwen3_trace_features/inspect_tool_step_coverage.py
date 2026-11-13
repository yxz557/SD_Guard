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
    parser.add_argument(
        "--trace-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional raw-trace root corresponding to each --feature-root. "
            "Repeat in the same order. The metadata source_path is tried first; "
            "trace roots provide a fallback when feature stores were moved."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--anomaly-output",
        type=Path,
        default=None,
        help=(
            "Write one JSONL row per diagnostically interesting sample. If omitted "
            "while --output is set, <output-stem>_anomalies.jsonl is used."
        ),
    )
    parser.add_argument("--examples-per-category", type=int, default=5)
    parser.add_argument("--max-error-chars", type=int, default=2000)
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


def build_trace_indices(
    trace_roots: list[Path] | None,
    source_names: list[str],
) -> dict[str, dict[str, Path]]:
    if trace_roots is None:
        return {}
    if len(trace_roots) != len(source_names):
        raise ValueError(
            "--trace-root count must match --feature-root count: "
            f"{len(trace_roots)} != {len(source_names)}"
        )
    indices: dict[str, dict[str, Path]] = {}
    for source_name, root in zip(source_names, trace_roots, strict=True):
        index: dict[str, Path] = {}
        for path in sorted(root.rglob("raw_trace.json")):
            sample_id = path.parent.name
            if sample_id in index:
                raise ValueError(
                    f"Duplicate raw trace directory name under {root}: {sample_id}"
                )
            index[sample_id] = path.resolve()
        indices[source_name] = index
    return indices


def resolve_raw_trace_path(
    row: dict[str, Any],
    trace_indices: dict[str, dict[str, Path]],
) -> Path | None:
    source_path = row.get("source_path")
    if source_path:
        candidate = Path(str(source_path))
        if candidate.is_file():
            return candidate.resolve()
    source_name = str(row["data_source"])
    sample_id = str(row["sample_id"])
    return trace_indices.get(source_name, {}).get(sample_id)


def content_preview(content: Any, limit: int = 500) -> str:
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(
            content_preview(item, limit=limit)
            for item in content
            if item is not None
        )
    elif isinstance(content, dict):
        text = str(content.get("content") or content.get("text") or content)
    else:
        text = str(content)
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"


def inspect_raw_trace(path: Path, max_error_chars: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    trajectory = raw.get("trajectory") or {}
    messages = trajectory.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError(f"Expected trajectory.messages list in {path}")

    tool_messages: list[tuple[int, dict[str, Any]]] = [
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") == "tool"
    ]
    errors: list[dict[str, Any]] = []
    for message_index, message in tool_messages:
        error = message.get("error")
        if error is None or str(error).strip() == "":
            continue
        tool_call = message.get("tool_call") or {}
        error_text = str(error)
        if len(error_text) > max_error_chars:
            error_text = error_text[:max_error_chars] + "…"
        errors.append(
            {
                "message_index": message_index,
                "tool_call_id": message.get("tool_call_id"),
                "function": (
                    tool_call.get("function") or tool_call.get("name")
                    if isinstance(tool_call, dict)
                    else None
                ),
                "error": error_text,
                "content_preview": content_preview(message.get("content")),
            }
        )

    last_tool_error = False
    if tool_messages:
        last_error = tool_messages[-1][1].get("error")
        last_tool_error = last_error is not None and str(last_error).strip() != ""
    provenance = raw.get("provenance") or {}
    run_error = provenance.get("run_error") if isinstance(provenance, dict) else None
    return {
        "raw_trace_path": str(path.resolve()),
        "raw_trace_status": "inspected",
        "tool_message_count": len(tool_messages),
        "structured_tool_error_count": len(errors),
        "successful_tool_result_count": len(tool_messages) - len(errors),
        "has_structured_tool_error": bool(errors),
        "all_tool_results_structured_errors": bool(tool_messages)
        and len(errors) == len(tool_messages),
        "last_tool_result_structured_error": last_tool_error,
        "structured_tool_errors": errors,
        "provenance_run_error": run_error,
    }


def attach_raw_trace_diagnostics(
    rows: list[dict[str, Any]],
    trace_indices: dict[str, dict[str, Path]],
    max_error_chars: int,
) -> None:
    for row in rows:
        path = resolve_raw_trace_path(row, trace_indices)
        if path is None:
            row.update(
                {
                    "raw_trace_path": row.get("source_path"),
                    "raw_trace_status": "missing",
                    "tool_message_count": None,
                    "structured_tool_error_count": None,
                    "successful_tool_result_count": None,
                    "has_structured_tool_error": None,
                    "all_tool_results_structured_errors": None,
                    "last_tool_result_structured_error": None,
                    "structured_tool_errors": [],
                    "provenance_run_error": None,
                }
            )
            continue
        try:
            row.update(inspect_raw_trace(path, max_error_chars))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            row.update(
                {
                    "raw_trace_path": str(path),
                    "raw_trace_status": "inspection_error",
                    "raw_trace_inspection_error": f"{type(exc).__name__}: {exc}",
                    "tool_message_count": None,
                    "structured_tool_error_count": None,
                    "successful_tool_result_count": None,
                    "has_structured_tool_error": None,
                    "all_tool_results_structured_errors": None,
                    "last_tool_result_structured_error": None,
                    "structured_tool_errors": [],
                    "provenance_run_error": None,
                }
            )


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
    summary = {
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
    if any("raw_trace_status" in row for row in rows):
        inspected = [row for row in rows if row.get("raw_trace_status") == "inspected"]
        missing = [row for row in rows if row.get("raw_trace_status") == "missing"]
        inspection_errors = [
            row for row in rows if row.get("raw_trace_status") == "inspection_error"
        ]
        with_error = [row for row in inspected if row.get("has_structured_tool_error")]
        all_error = [
            row for row in inspected if row.get("all_tool_results_structured_errors")
        ]
        last_error = [
            row for row in inspected if row.get("last_tool_result_structured_error")
        ]
        summary["raw_trace_diagnostics"] = {
            "inspected_samples": len(inspected),
            "missing_samples": len(missing),
            "inspection_error_samples": len(inspection_errors),
            "samples_with_any_structured_tool_error": len(with_error),
            "samples_with_all_tool_results_structured_errors": len(all_error),
            "samples_with_last_tool_result_structured_error": len(last_error),
            "structured_tool_error_messages": sum(
                int(row.get("structured_tool_error_count") or 0) for row in inspected
            ),
            "samples_with_provenance_run_error": sum(
                int(bool(row.get("provenance_run_error"))) for row in inspected
            ),
        }
    return summary


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


def anomaly_types(row: dict[str, Any]) -> list[str]:
    kinds: list[str] = []
    coverage = str(row["tool_step_coverage"])
    if coverage != "both":
        kinds.append(coverage)
    if int(row["tool_call_count"]) != int(row["tool_result_count"]):
        kinds.append("call_result_count_mismatch")
    if row.get("has_structured_tool_error"):
        kinds.append("structured_tool_error")
    if row.get("all_tool_results_structured_errors"):
        kinds.append("all_tool_results_structured_errors")
    if row.get("last_tool_result_structured_error"):
        kinds.append("last_tool_result_structured_error")
    if row.get("raw_trace_status") == "missing":
        kinds.append("raw_trace_missing")
    if row.get("raw_trace_status") == "inspection_error":
        kinds.append("raw_trace_inspection_error")
    if row.get("provenance_run_error"):
        kinds.append("provenance_run_error")
    return kinds


def anomaly_record(row: dict[str, Any]) -> dict[str, Any] | None:
    kinds = anomaly_types(row)
    if not kinds:
        return None
    keep = (
        "sample_id",
        "data_source",
        "offline_split",
        "label",
        "suite_name",
        "attack_type",
        "user_task_id",
        "injection_task_id",
        "group_id",
        "tool_call_count",
        "tool_result_count",
        "raw_trace_path",
        "raw_trace_status",
        "raw_trace_inspection_error",
        "tool_message_count",
        "structured_tool_error_count",
        "successful_tool_result_count",
        "has_structured_tool_error",
        "all_tool_results_structured_errors",
        "last_tool_result_structured_error",
        "structured_tool_errors",
        "provenance_run_error",
    )
    output = {key: row.get(key) for key in keep if key in row}
    output["diagnostic_flags"] = kinds
    return output


def write_anomaly_manifest(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for row in rows if (record := anomaly_record(row)) is not None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    flag_counts = Counter(
        flag for record in records for flag in record["diagnostic_flags"]
    )
    return {
        "path": str(path.resolve()),
        "samples": len(records),
        "flag_counts": dict(sorted(flag_counts.items())),
    }


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
    if args.max_error_chars < 0:
        raise ValueError("--max-error-chars must be non-negative")
    feature_roots = [root.resolve() for root in args.feature_root]
    source_names = resolve_source_names(feature_roots, args.source_name)
    rows = load_feature_rows(feature_roots, source_names)
    trace_roots = (
        [root.resolve() for root in args.trace_root]
        if args.trace_root is not None
        else None
    )
    trace_indices = build_trace_indices(trace_roots, source_names)
    attach_raw_trace_diagnostics(rows, trace_indices, args.max_error_chars)
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
    selected_rows = rows
    if annotations is not None:
        selected_ids = set(annotations)
        selected_rows = []
        for original in rows:
            sample_id = str(original["sample_id"])
            if sample_id not in selected_ids:
                continue
            row = dict(original)
            row.update(annotations[sample_id])
            selected_rows.append(row)

    anomaly_output = args.anomaly_output
    if anomaly_output is None and args.output is not None:
        anomaly_output = args.output.with_name(args.output.stem + "_anomalies.jsonl")
    if anomaly_output is not None:
        report["anomaly_manifest"] = write_anomaly_manifest(
            anomaly_output.resolve(), selected_rows
        )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="")


if __name__ == "__main__":
    main()
