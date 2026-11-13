from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ATTACKS = (
    "direct",
    "ignore_previous",
    "injecagent",
    "important_instructions",
    "tool_knowledge",
)
LABEL_NAMES = {True: "positive", False: "negative"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    path.write_text(content, encoding="utf-8", newline="\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RolloutRecord:
    suite_name: str
    user_task_id: str
    injection_task_id: str
    attack_type: str
    hijacked: bool | None
    sample_id: str
    rollout_id: str | None
    raw_trace_path: Path
    ground_truth_path: Path
    tool_call_count: int
    tool_result_count: int
    tool_error_count: int
    thinking_block_count: int
    utility_passed: bool | None
    accepted: bool
    rejection_reasons: tuple[str, ...]

    @property
    def cell(self) -> tuple[str, str, str, str]:
        return self.suite_name, self.user_task_id, self.injection_task_id, self.attack_type

    def to_record(self, *, selected: bool = False) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "user_task_id": self.user_task_id,
            "injection_task_id": self.injection_task_id,
            "attack_type": self.attack_type,
            "label": None if self.hijacked is None else int(self.hijacked),
            "label_name": None if self.hijacked is None else LABEL_NAMES[self.hijacked],
            "sample_id": self.sample_id,
            "rollout_id": self.rollout_id,
            "raw_trace_path": str(self.raw_trace_path),
            "ground_truth_path": str(self.ground_truth_path),
            "tool_call_count": self.tool_call_count,
            "tool_result_count": self.tool_result_count,
            "tool_error_count": self.tool_error_count,
            "thinking_block_count": self.thinking_block_count,
            "utility_passed": self.utility_passed,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "selected": selected,
        }


def _expected_cells(manifest: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    cells: list[tuple[str, str, str, str]] = []
    for suite_name in manifest["suite_names"]:
        for pair in manifest["suites"][suite_name]["sampled_pairs"]:
            for attack_type in ATTACKS:
                cells.append(
                    (
                        suite_name,
                        pair["user_task_id"],
                        pair["injection_task_id"],
                        attack_type,
                    )
                )
    return cells


def _discover_trace_roots(campaign_root: Path, explicit_roots: list[Path]) -> list[Path]:
    if explicit_roots:
        roots = explicit_roots
    else:
        roots = sorted(campaign_root.glob("raw/*/results/traces"))
    missing = [path for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Trace roots do not exist: {missing}")
    return roots


def _parse_rollout(
    raw_trace_path: Path,
    *,
    require_thinking: bool,
    allow_unbalanced_tool_trace: bool,
) -> RolloutRecord:
    ground_truth_path = raw_trace_path.with_name("ground_truth.json")
    rejection_reasons: list[str] = []
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Missing ground truth beside {raw_trace_path}")
    raw_trace = _load_json(raw_trace_path)
    ground_truth = _load_json(ground_truth_path)
    task = raw_trace.get("task") or {}
    provenance = raw_trace.get("provenance") or {}
    trajectory = raw_trace.get("trajectory") or {}
    messages = trajectory.get("messages") or []
    tool_trace = trajectory.get("tool_trace") or []
    tool_call_count = sum(event.get("event") == "tool_call_proposed" for event in tool_trace)
    tool_result_count = sum(event.get("event") == "tool_result" for event in tool_trace)
    tool_error_count = sum(
        event.get("event") == "tool_result" and event.get("error") not in {None, ""}
        for event in tool_trace
    )
    thinking_block_count = sum(
        block.get("type") in {"thinking", "redacted_thinking"}
        for message in messages
        for block in (message.get("content") or [])
        if isinstance(block, dict)
    )

    if not ground_truth.get("valid"):
        rejection_reasons.append("invalid_ground_truth")
    if not ground_truth.get("payload_present"):
        rejection_reasons.append("payload_missing")
    if ground_truth.get("hijacked") not in {True, False}:
        rejection_reasons.append("label_missing")
    if provenance.get("run_error") not in {None, ""}:
        rejection_reasons.append("run_error")
    if tool_call_count == 0 or tool_result_count == 0:
        rejection_reasons.append("no_tool_steps")
    if not allow_unbalanced_tool_trace and tool_call_count != tool_result_count:
        rejection_reasons.append("unexecuted_or_unmatched_tool_call")
    if require_thinking and thinking_block_count == 0:
        rejection_reasons.append("thinking_not_saved")
    expected_hash = ground_truth.get("trace_sha256")
    if not isinstance(expected_hash, str) or _sha256_file(raw_trace_path) != expected_hash:
        rejection_reasons.append("trace_hash_mismatch")

    return RolloutRecord(
        suite_name=str(raw_trace.get("suite_name")),
        user_task_id=str(task.get("user_task_id")),
        injection_task_id=str(task.get("injection_task_id")),
        attack_type=str(task.get("attack_type")),
        hijacked=ground_truth.get("hijacked"),
        sample_id=str(raw_trace.get("sample_id")),
        rollout_id=provenance.get("rollout_id"),
        raw_trace_path=raw_trace_path.resolve(),
        ground_truth_path=ground_truth_path.resolve(),
        tool_call_count=tool_call_count,
        tool_result_count=tool_result_count,
        tool_error_count=tool_error_count,
        thinking_block_count=thinking_block_count,
        utility_passed=ground_truth.get("utility_passed"),
        accepted=not rejection_reasons,
        rejection_reasons=tuple(rejection_reasons),
    )


def _copy_selected(record: RolloutRecord, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record.raw_trace_path, destination / "raw_trace.json")
    shutil.copyfile(record.ground_truth_path, destination / "ground_truth.json")
    hash_source = record.raw_trace_path.with_suffix(".sha256")
    if hash_source.is_file():
        shutil.copyfile(hash_source, destination / "raw_trace.sha256")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean suite/pair/attack/label view over repeated AgentDojo rollouts."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--trace-root", type=Path, action="append", default=[])
    parser.add_argument("--require-thinking", action="store_true")
    parser.add_argument("--allow-unbalanced-tool-trace", action="store_true")
    args = parser.parse_args()

    campaign_root = args.campaign_root.resolve()
    output_root = (args.output or campaign_root / "curated").resolve()
    manifest = _load_json(args.manifest)
    expected_cells = _expected_cells(manifest)
    expected_set = set(expected_cells)
    trace_roots = _discover_trace_roots(campaign_root, args.trace_root)

    records: list[RolloutRecord] = []
    for trace_root in trace_roots:
        for raw_trace_path in sorted(trace_root.glob("*/raw_trace.json")):
            records.append(
                _parse_rollout(
                    raw_trace_path,
                    require_thinking=args.require_thinking,
                    allow_unbalanced_tool_trace=args.allow_unbalanced_tool_trace,
                )
            )

    grouped: dict[tuple[tuple[str, str, str, str], bool], list[RolloutRecord]] = defaultdict(list)
    for record in records:
        if record.accepted and record.cell in expected_set and record.hijacked in {True, False}:
            grouped[(record.cell, bool(record.hijacked))].append(record)
    for candidates in grouped.values():
        candidates.sort(
            key=lambda record: (
                record.tool_error_count,
                record.rollout_id or "",
                record.sample_id,
            )
        )

    selected_keys: set[tuple[str, bool]] = set()
    missing_records: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    complete_cells = 0
    for cell in expected_cells:
        suite_name, user_task_id, injection_task_id, attack_type = cell
        pair_name = f"{user_task_id}__{injection_task_id}"
        label_counts: dict[str, int] = {}
        missing_labels: list[str] = []
        for hijacked, label_name in LABEL_NAMES.items():
            candidates = grouped.get((cell, hijacked), [])
            destination = output_root / suite_name / pair_name / attack_type / label_name
            label_counts[label_name] = len(candidates)
            if candidates:
                selected = candidates[0]
                selected_keys.add((selected.sample_id, hijacked))
                _copy_selected(selected, destination)
                status = {
                    "status": "selected",
                    "selected_sample_id": selected.sample_id,
                    "selected_rollout_id": selected.rollout_id,
                    "available_clean_rollouts": len(candidates),
                    "source_raw_trace": str(selected.raw_trace_path),
                }
            else:
                missing_labels.append(label_name)
                status = {
                    "status": "missing",
                    "selected_sample_id": None,
                    "selected_rollout_id": None,
                    "available_clean_rollouts": 0,
                }
                missing_records.append(
                    {
                        "suite_name": suite_name,
                        "user_task_id": user_task_id,
                        "injection_task_id": injection_task_id,
                        "attack_type": attack_type,
                        "missing_label": label_name,
                    }
                )
            _write_json(destination / "status.json", status)
        if not missing_labels:
            complete_cells += 1
        cell_summaries.append(
            {
                "suite_name": suite_name,
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "attack_type": attack_type,
                "clean_rollouts": label_counts,
                "missing_labels": missing_labels,
                "complete_positive_negative_pair": not missing_labels,
            }
        )

    all_rollout_records = [
        record.to_record(selected=(record.sample_id, bool(record.hijacked)) in selected_keys)
        for record in sorted(
            records,
            key=lambda item: (item.cell, item.rollout_id or "", item.sample_id),
        )
    ]
    rejection_counts = Counter(
        reason for record in records for reason in record.rejection_reasons
    )
    coverage = {
        "schema_version": "sd_guard.repeated_task_injection_coverage.v1",
        "campaign_root": str(campaign_root),
        "manifest": str(args.manifest.resolve()),
        "trace_roots": [str(path.resolve()) for path in trace_roots],
        "quality_policy": {
            "require_valid_ground_truth": True,
            "require_payload": True,
            "require_tool_calls_and_results": True,
            "require_balanced_tool_trace": not args.allow_unbalanced_tool_trace,
            "require_thinking": args.require_thinking,
            "structured_tool_errors_allowed": True,
        },
        "counts": {
            "expected_task_pairs": sum(
                len(manifest["suites"][suite]["sampled_pairs"])
                for suite in manifest["suite_names"]
            ),
            "expected_attack_cells": len(expected_cells),
            "complete_positive_negative_cells": complete_cells,
            "missing_label_slots": len(missing_records),
            "discovered_rollouts": len(records),
            "accepted_rollouts": sum(record.accepted for record in records),
            "rejected_rollouts": sum(not record.accepted for record in records),
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "cells": cell_summaries,
    }
    _write_json(output_root / "coverage.json", coverage)
    _write_jsonl(output_root / "all_rollouts.jsonl", all_rollout_records)
    _write_jsonl(output_root / "missing_labels.jsonl", missing_records)
    print(
        f"Curated {len(records)} rollout(s): complete_cells={complete_cells}/{len(expected_cells)}, "
        f"missing_label_slots={len(missing_records)}, output={output_root}"
    )


if __name__ == "__main__":
    main()
