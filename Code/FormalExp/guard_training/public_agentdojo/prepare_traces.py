from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ATTACKS = (
    "direct",
    "ignore_previous",
    "important_instructions",
)

DEFENSE_SUFFIXES = (
    "repeat_user_prompt",
    "spotlighting_with_delimiting",
    "tool_filter",
    "transformers_pi_detector",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert selected public AgentDojo run JSON files into the canonical "
            "raw_trace.json/ground_truth.json layout consumed by feature extraction."
        )
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-version", default="v1.2.1")
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=list(DEFAULT_ATTACKS),
        help="Attacks to retain. The default matches the explicit v1.2.1 public runs.",
    )
    parser.add_argument(
        "--defense",
        choices=("none", "all"),
        default="none",
        help="Keep only no-defense runs by default.",
    )
    parser.add_argument(
        "--source-runs",
        nargs="*",
        default=None,
        help="Optional allow-list of top-level run directory names.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_defense(source_run: str) -> str:
    for defense in DEFENSE_SUFFIXES:
        if source_run.endswith(f"-{defense}") or source_run.endswith(f"_{defense}"):
            return defense
    return "none"


def iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)


def payload_was_observed(injections: Any, messages: list[dict[str, Any]]) -> bool:
    if not isinstance(injections, dict) or not injections:
        return False
    message_text = "\n".join(iter_strings(messages))
    return any(
        isinstance(payload, str) and payload and payload in message_text
        for payload in injections.values()
    )


def safe_slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalized[:80] or "unknown"


def write_new_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build_sample(
    source_path: Path,
    runs_root: Path,
    source_run: str,
    public: dict[str, Any],
) -> tuple[str, bytes, bytes, dict[str, Any]]:
    relative_path = source_path.relative_to(runs_root).as_posix()
    source_sha256 = sha256_file(source_path)
    sample_suffix = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    sample_id = f"agentdojo-public-{safe_slug(source_run)}-{sample_suffix}"

    messages = public["messages"]
    injections = public.get("injections") or {}
    payload_present = payload_was_observed(injections, messages)
    raw_trace = {
        "benchmark_component": "task_injection",
        "benchmark_version": public["benchmark_version"],
        "dataset": "agentdojo_public_runs",
        "provenance": {
            "agentdojo_package_version": public.get("agentdojo_package_version"),
            "evaluation_timestamp": public.get("evaluation_timestamp"),
            "pipeline_name": public.get("pipeline_name"),
            "source_relative_path": relative_path,
            "source_run": source_run,
            "source_sha256": source_sha256,
        },
        "sample_id": sample_id,
        "schema_version": "sd_guard.raw_trace.v1",
        "suite_name": public["suite_name"],
        "target": {
            "model_id": source_run,
            "pipeline_name": public.get("pipeline_name"),
        },
        "task": {
            "attack_type": public["attack_type"],
            "injection_task_id": public["injection_task_id"],
            "injections": injections,
            "payload_present": payload_present,
            "user_task_id": public["user_task_id"],
        },
        "trajectory": {"messages": messages},
    }
    raw_bytes = canonical_json_bytes(raw_trace)
    trace_sha256 = sha256_bytes(raw_bytes)
    ground_truth = {
        "hijacked": bool(public["security"]),
        "invalid_reason": None,
        "label_semantics": "1=AgentDojo injection goal succeeded; 0=did not succeed",
        "payload_present": payload_present,
        "raw_agentdojo_security": bool(public["security"]),
        "sample_id": sample_id,
        "schema_version": "sd_guard.ground_truth.v1",
        "trace_sha256": trace_sha256,
        "truth_source": "agentdojo_public_runs.security",
        "utility_passed": bool(public.get("utility", False)),
        "valid": True,
    }
    truth_bytes = canonical_json_bytes(ground_truth)
    index_row = {
        "attack_type": public["attack_type"],
        "benchmark_version": public["benchmark_version"],
        "defense": infer_defense(source_run),
        "hijacked": bool(public["security"]),
        "injection_task_id": public["injection_task_id"],
        "payload_present": payload_present,
        "pipeline_name": public.get("pipeline_name"),
        "sample_id": sample_id,
        "source_relative_path": relative_path,
        "source_run": source_run,
        "source_sha256": source_sha256,
        "suite_name": public["suite_name"],
        "trace_sha256": trace_sha256,
        "user_task_id": public["user_task_id"],
        "utility_passed": bool(public.get("utility", False)),
    }
    return sample_id, raw_bytes, truth_bytes, index_row


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    output_root = args.output_root.resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Public runs root does not exist: {runs_root}")

    allowed_attacks = set(args.attacks)
    allowed_runs = set(args.source_runs) if args.source_runs else None
    counters: Counter[str] = Counter()
    index_rows: list[dict[str, Any]] = []

    for source_path in sorted(runs_root.rglob("*.json")):
        if args.limit is not None and counters["accepted"] >= args.limit:
            break
        relative = source_path.relative_to(runs_root)
        if len(relative.parts) < 2:
            counters["malformed_path"] += 1
            continue
        source_run = relative.parts[0]
        if allowed_runs is not None and source_run not in allowed_runs:
            counters["source_run_filtered"] += 1
            continue
        if args.defense == "none" and infer_defense(source_run) != "none":
            counters["defense_filtered"] += 1
            continue

        try:
            public = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counters["invalid_json"] += 1
            continue
        if not isinstance(public, dict):
            counters["invalid_json_object"] += 1
            continue
        if public.get("benchmark_version") != args.benchmark_version:
            counters["version_filtered"] += 1
            continue
        if public.get("attack_type") not in allowed_attacks:
            counters["attack_filtered"] += 1
            continue
        if public.get("error") not in (None, ""):
            counters["run_error"] += 1
            continue
        if not isinstance(public.get("security"), bool):
            counters["invalid_security_label"] += 1
            continue
        messages = public.get("messages")
        if not isinstance(messages, list) or not messages:
            counters["missing_messages"] += 1
            continue
        required = ("suite_name", "user_task_id", "injection_task_id", "attack_type")
        if any(not public.get(key) for key in required):
            counters["missing_task_metadata"] += 1
            continue

        sample_id, raw_bytes, truth_bytes, index_row = build_sample(
            source_path, runs_root, source_run, public
        )
        trace_dir = output_root / "traces" / sample_id
        write_new_or_verify(trace_dir / "raw_trace.json", raw_bytes)
        write_new_or_verify(trace_dir / "ground_truth.json", truth_bytes)
        index_rows.append(index_row)
        counters["accepted"] += 1
        counters["positive" if index_row["hijacked"] else "negative"] += 1

    if not index_rows:
        raise ValueError("No public traces passed the requested filters")

    index_payload = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in sorted(index_rows, key=lambda row: row["sample_id"])
    )
    summary = {
        "attacks": sorted(allowed_attacks),
        "benchmark_version": args.benchmark_version,
        "counts": dict(sorted(counters.items())),
        "defense": args.defense,
        "output_root": str(output_root),
        "runs_root": str(runs_root),
        "schema_version": "sd_guard.public_agentdojo_preparation.v1",
        "source_runs": sorted({row["source_run"] for row in index_rows}),
        "task_pairs": len(
            {
                (row["suite_name"], row["user_task_id"], row["injection_task_id"])
                for row in index_rows
            }
        ),
    }
    write_new_or_verify(output_root / "records.jsonl", index_payload)
    write_new_or_verify(output_root / "preparation_summary.json", canonical_json_bytes(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

