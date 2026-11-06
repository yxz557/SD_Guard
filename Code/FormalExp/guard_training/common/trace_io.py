from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class TraceRecord:
    sample_id: str
    source_path: Path
    label: int
    suite_name: str
    attack_type: str
    user_task_id: str
    injection_task_id: str
    group_id: str
    messages: list[dict[str, Any]]
    trace_sha256: str

    def metadata(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_path": str(self.source_path),
            "label": self.label,
            "suite_name": self.suite_name,
            "attack_type": self.attack_type,
            "user_task_id": self.user_task_id,
            "injection_task_id": self.injection_task_id,
            "group_id": self.group_id,
            "trace_sha256": self.trace_sha256,
        }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_group_id(
    suite_name: str,
    user_task_id: str,
    injection_task_id: str,
) -> str:
    # JSON avoids ambiguity if an identifier ever contains a separator.
    return json.dumps(
        [suite_name, user_task_id, injection_task_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def load_trace_record(raw_trace_path: Path) -> TraceRecord | None:
    raw = read_json(raw_trace_path)
    ground_truth_path = raw_trace_path.with_name("ground_truth.json")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Missing ground truth next to {raw_trace_path}")
    truth = read_json(ground_truth_path)

    if not bool(truth.get("valid", False)):
        return None
    if truth.get("invalid_reason") is not None:
        return None

    task = raw.get("task") or {}
    trajectory = raw.get("trajectory") or {}
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Trace has no trajectory.messages: {raw_trace_path}")

    sample_id = str(raw.get("sample_id") or truth.get("sample_id") or raw_trace_path.parent.name)
    suite_name = str(raw.get("suite_name") or "")
    attack_type = str(task.get("attack_type") or "")
    user_task_id = str(task.get("user_task_id") or "")
    injection_task_id = str(task.get("injection_task_id") or "")
    missing = [
        name
        for name, value in (
            ("suite_name", suite_name),
            ("attack_type", attack_type),
            ("user_task_id", user_task_id),
            ("injection_task_id", injection_task_id),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"Missing {missing} in {raw_trace_path}")

    trace_sha256 = str(truth.get("trace_sha256") or sha256_file(raw_trace_path))
    return TraceRecord(
        sample_id=sample_id,
        source_path=raw_trace_path.resolve(),
        label=int(bool(truth.get("hijacked", False))),
        suite_name=suite_name,
        attack_type=attack_type,
        user_task_id=user_task_id,
        injection_task_id=injection_task_id,
        group_id=make_group_id(suite_name, user_task_id, injection_task_id),
        messages=messages,
        trace_sha256=trace_sha256,
    )


def iter_trace_records(trace_root: Path) -> Iterator[TraceRecord]:
    for raw_trace_path in sorted(trace_root.rglob("raw_trace.json")):
        record = load_trace_record(raw_trace_path)
        if record is not None:
            yield record


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(filter(None, (content_to_text(item) for item in content)))
    if isinstance(content, dict):
        for key in ("content", "text", "value"):
            if key in content:
                return content_to_text(content[key])
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)

