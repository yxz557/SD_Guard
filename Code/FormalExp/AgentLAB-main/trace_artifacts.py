"""Shared immutable raw-trace writer for the non-AgentDojo AgentLAB tracks.

The Task Injection runner already freezes AgentDojo logs as
``sd_guard.raw_trace.v1`` artifacts.  This module gives the other four tracks
the same on-disk contract without requiring labels to be produced at run time.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


RAW_TRACE_SCHEMA_VERSION = "sd_guard.raw_trace.v1"


def _jsonable(value: Any) -> Any:
    """Convert SDK objects and dataclasses to stable JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump(mode="json"))
        except TypeError:
            return _jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dict__"):
        public = {
            key: item for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if public:
            return _jsonable(public)
    return str(value)


def _compact_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_json(value: Any) -> bytes:
    text = json.dumps(
        _jsonable(value), ensure_ascii=False, indent=2, sort_keys=True,
        separators=(",", ": "),
    )
    return (text + "\n").encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_compact_json(value)).hexdigest()


def _make_read_only(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def _write_immutable(path: Path, content: bytes) -> None:
    """Create once, allow identical retries, and reject changed content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"Refusing to overwrite immutable trace artifact: {path}"
            )
        return
    with path.open("xb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    _make_read_only(path)


def build_sample_id(component: str, identity: Mapping[str, Any]) -> str:
    digest = sha256_json({
        "dataset": "AgentLAB",
        "benchmark_component": component,
        "identity": identity,
    })
    return f"agentlab-{digest[:32]}"


def extract_tool_trace(messages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive a convenient tool-event view while retaining the raw messages."""
    events: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            for call_index, tool_call in enumerate(message.get("tool_calls") or []):
                events.append({
                    "event": "tool_call_proposed",
                    "message_index": message_index,
                    "call_index": call_index,
                    "tool_call": _jsonable(tool_call),
                })
        elif role == "tool":
            events.append({
                "event": "tool_result",
                "message_index": message_index,
                "tool_call_id": message.get("tool_call_id"),
                "name": message.get("name"),
                "content": _jsonable(message.get("content")),
                "error": _jsonable(message.get("error")),
            })
    return events


def assistant_message_dict(message: Any) -> dict[str, Any]:
    """Serialize an OpenAI-compatible assistant message without losing reasoning."""
    content = getattr(message, "content", None)
    result: dict[str, Any] = {"role": "assistant", "content": content}
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning not in (None, ""):
        result["reasoning_content"] = reasoning

    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        calls.append({
            "id": getattr(call, "id", None),
            "type": getattr(call, "type", "function"),
            "function": {
                "name": getattr(function, "name", None),
                "arguments": getattr(function, "arguments", None),
            },
        })
    if calls:
        result["tool_calls"] = calls
    return _jsonable(result)


def write_raw_trace(
    output_root: str | Path,
    *,
    component: str,
    identity: Mapping[str, Any],
    task: Mapping[str, Any],
    target: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    provenance: Mapping[str, Any],
    attack: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
    evaluation: Mapping[str, Any] | None = None,
) -> Path:
    """Freeze one label-free raw trace and return its ``raw_trace.json`` path."""
    normalized_identity = _jsonable(identity)
    sample_id = build_sample_id(component, normalized_identity)
    trace: dict[str, Any] = {
        "schema_version": RAW_TRACE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "dataset": "AgentLAB",
        "benchmark_component": component,
        "task": _jsonable(task),
        "target": _jsonable(target),
        "trajectory": _jsonable(trajectory),
        "provenance": _jsonable(provenance),
    }
    if attack is not None:
        trace["attack"] = _jsonable(attack)
    if environment is not None:
        trace["environment"] = _jsonable(environment)
    if evaluation is not None:
        trace["evaluation"] = _jsonable(evaluation)

    trace_path = Path(output_root) / sample_id / "raw_trace.json"
    content = _canonical_json(trace)
    _write_immutable(trace_path, content)
    trace_hash = hashlib.sha256(content).hexdigest()
    _write_immutable(
        trace_path.with_suffix(".sha256"),
        f"{trace_hash}  {trace_path.name}\n".encode("utf-8"),
    )
    return trace_path


__all__ = [
    "RAW_TRACE_SCHEMA_VERSION",
    "assistant_message_dict",
    "build_sample_id",
    "extract_tool_trace",
    "sha256_json",
    "write_raw_trace",
]
