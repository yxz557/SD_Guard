from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from agentdojo.guard_eval import GROUND_TRUTH_SCHEMA_VERSION, RAW_TRACE_SCHEMA_VERSION
from agentdojo.guard_eval.artifacts import (
    discover_files,
    load_json,
    sha256_file,
    sha256_json,
    write_immutable_json,
    write_trace_hash,
)


REQUIRED_LOG_FIELDS = {
    "suite_name",
    "pipeline_name",
    "user_task_id",
    "injections",
    "messages",
    "utility",
    "security",
}


def _normalize_optional_id(value: Any) -> str | None:
    if value in (None, "", "none", "None"):
        return None
    return str(value)


def _looks_like_agentdojo_log(value: Any) -> bool:
    return isinstance(value, dict) and REQUIRED_LOG_FIELDS.issubset(value)


def _extract_tool_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_trace: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                tool_trace.append(
                    {
                        "event": "tool_call_proposed",
                        "message_index": message_index,
                        "tool_call": tool_call,
                    }
                )
        elif role == "tool":
            tool_trace.append(
                {
                    "event": "tool_result",
                    "message_index": message_index,
                    "tool_call": message.get("tool_call"),
                    "content": message.get("content"),
                    "error": message.get("error"),
                }
            )
    return tool_trace


def _build_sample_id(
    *,
    suite_name: str,
    benchmark_version: str,
    user_task_id: str,
    injection_task_id: str | None,
    attack_type: str | None,
    payload_sha256: str,
    rollout_id: str | None = None,
) -> str:
    identity = {
        "dataset": "AgentLAB",
        "benchmark_component": "Task Injection",
        "suite_name": suite_name,
        "benchmark_version": benchmark_version,
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
        "attack_type": attack_type,
        "payload_sha256": payload_sha256,
    }
    if rollout_id is not None:
        identity["rollout_id"] = rollout_id
    return f"agentlab-{sha256_json(identity)[:32]}"


def _generation_config_from_environment() -> dict[str, Any]:
    environment_fields = {
        "temperature": "AGENTDOJO_LLM_TEMPERATURE",
        "top_p": "AGENTDOJO_LLM_TOP_P",
        "top_k": "AGENTDOJO_LLM_TOP_K",
        "max_tokens": "AGENTDOJO_LLM_MAX_TOKENS",
        "generation_seed": "AGENTDOJO_GENERATION_SEED",
        "enable_thinking": "AGENTDOJO_ENABLE_THINKING",
    }
    config: dict[str, Any] = {}
    for key, environment_name in environment_fields.items():
        value = os.getenv(environment_name)
        if value in (None, ""):
            continue
        if key in {"top_k", "max_tokens", "generation_seed"}:
            config[key] = int(value)
        elif key in {"temperature", "top_p"}:
            config[key] = float(value)
        else:
            config[key] = value.strip().lower() in {"1", "true", "yes", "on"}
    return config


def export_agentdojo_log(
    log_path: Path,
    output_root: Path,
    *,
    target_model_id: str | None = None,
    benchmark_version_fallback: str = "v1.2.1",
    run_id: str = "baseline-r0",
    rollout_id: str | None = None,
    seed: int | None = None,
) -> tuple[Path, Path]:
    source = load_json(log_path)
    if not _looks_like_agentdojo_log(source):
        raise ValueError(f"Not an AgentDojo task log: {log_path}")

    suite_name = str(source["suite_name"])
    user_task_id = str(source["user_task_id"])
    injection_task_id = _normalize_optional_id(source.get("injection_task_id"))
    attack_type = _normalize_optional_id(source.get("attack_type"))
    benchmark_version = str(source.get("benchmark_version") or benchmark_version_fallback)
    injections = source.get("injections") or {}
    messages = source.get("messages") or []
    if not isinstance(injections, dict) or not isinstance(messages, list):
        raise TypeError(f"Invalid injections/messages in {log_path}")

    payload_present = bool(injection_task_id and attack_type and injections)
    payload_sha256 = sha256_json(injections)
    sample_id = _build_sample_id(
        suite_name=suite_name,
        benchmark_version=benchmark_version,
        user_task_id=user_task_id,
        injection_task_id=injection_task_id,
        attack_type=attack_type,
        payload_sha256=payload_sha256,
        rollout_id=rollout_id,
    )
    source_log_sha256 = sha256_file(log_path)
    provenance = {
        "run_id": run_id,
        "seed": seed,
        "source_log_name": log_path.name,
        "source_log_sha256": source_log_sha256,
        "evaluation_timestamp": source.get("evaluation_timestamp"),
        "agentdojo_package_version": source.get("agentdojo_package_version"),
        "duration_seconds": source.get("duration"),
        "run_error": source.get("error"),
    }
    if rollout_id is not None:
        provenance["rollout_id"] = rollout_id
    generation_config = _generation_config_from_environment()
    if generation_config:
        provenance["generation_config"] = generation_config

    raw_trace = {
        "schema_version": RAW_TRACE_SCHEMA_VERSION,
        "sample_id": sample_id,
        "dataset": "AgentLAB",
        "benchmark_component": "Task Injection",
        "benchmark_version": benchmark_version,
        "suite_name": suite_name,
        "task": {
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id,
            "attack_type": attack_type,
            "payload_present": payload_present,
            "payload_sha256": payload_sha256,
            "injections": injections,
        },
        "target": {
            "pipeline_name": str(source["pipeline_name"]),
            "model_id": target_model_id,
        },
        "trajectory": {
            "messages": messages,
            "tool_trace": _extract_tool_trace(messages),
        },
        "provenance": provenance,
    }

    sample_dir = output_root / sample_id
    trace_path = sample_dir / "raw_trace.json"
    trace_sha256 = write_immutable_json(trace_path, raw_trace)
    write_trace_hash(trace_path, trace_sha256)

    run_error = source.get("error")
    valid = run_error in (None, "") and isinstance(source.get("utility"), bool) and isinstance(
        source.get("security"), bool
    )
    if valid:
        hijacked: bool | None = bool(source["security"]) if payload_present else False
        utility_passed: bool | None = bool(source["utility"])
    else:
        hijacked = None
        utility_passed = None

    ground_truth = {
        "schema_version": GROUND_TRUTH_SCHEMA_VERSION,
        "sample_id": sample_id,
        "trace_sha256": trace_sha256,
        "truth_source": "agentdojo_programmatic_rule",
        "valid": valid,
        "invalid_reason": None if valid else (run_error or "missing/non-boolean utility or security result"),
        "payload_present": payload_present,
        "hijacked": hijacked,
        "utility_passed": utility_passed,
        "raw_agentdojo_security": source.get("security"),
        "label_semantics": "For Task Injection, raw_agentdojo_security=true means the injection goal executed.",
    }
    truth_path = sample_dir / "ground_truth.json"
    write_immutable_json(truth_path, ground_truth)
    return trace_path, truth_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze AgentDojo logs into immutable SD_Guard trace artifacts.")
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="AgentDojo log file(s) or directories.")
    parser.add_argument("--output", type=Path, required=True, help="Artifact root directory.")
    parser.add_argument("--target-model-id", default=None)
    parser.add_argument("--benchmark-version", default="v1.2.1")
    parser.add_argument("--run-id", default="baseline-r0")
    parser.add_argument(
        "--rollout-id",
        default=None,
        help="Stable repeated-rollout id. When set, it is included in the sample identity.",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    candidates = discover_files(args.input, "*.json")
    exported = 0
    skipped = 0
    for candidate in candidates:
        try:
            value = load_json(candidate)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        if not _looks_like_agentdojo_log(value):
            skipped += 1
            continue
        trace_path, truth_path = export_agentdojo_log(
            candidate,
            args.output,
            target_model_id=args.target_model_id,
            benchmark_version_fallback=args.benchmark_version,
            run_id=args.run_id,
            rollout_id=args.rollout_id,
            seed=args.seed,
        )
        exported += 1
        print(f"exported: {trace_path}")
        print(f"truth:    {truth_path}")
    if exported == 0:
        raise SystemExit(f"No AgentDojo task logs found under: {', '.join(map(str, args.input))}")
    print(f"Done. Exported {exported} trace(s); skipped {skipped} unrelated JSON file(s).")


if __name__ == "__main__":
    main()
