"""Summarize vLLM/PyTorch profiler traces for speculative decoding.

The profiler trace remains the source of truth.  This module only produces a
compact GPU-kernel-time view by attributing CUDA launches to the innermost
recognizable CPU range or Python stack.  It deliberately does not present the
result as end-to-end wall-clock phase timing: CUDA work can overlap and some
third-party kernels do not retain enough correlation metadata to classify.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STAGE_ORDER = (
    "guard_prefill",
    "guard_proposal",
    "guard_commit",
    "guard_posthoc",
    "draft_proposal",
    "target_verification",
    "rejection_sampling",
    "scheduler_and_bookkeeping",
    "other_or_unattributed",
)


def _event_text(event: dict[str, Any]) -> str:
    args = event.get("args")
    useful_args: list[str] = []
    if isinstance(args, dict):
        for key, value in args.items():
            normalized_key = str(key).lower()
            if "stack" in normalized_key or "trace" in normalized_key:
                useful_args.append(str(value))
    return " ".join([str(event.get("name", "")), *useful_args]).lower()


def classify_profile_event(event: dict[str, Any]) -> str:
    """Classify a CPU profiler event using its name and captured call stack."""

    text = _event_text(event)
    if "sd4_guard:prefill" in text:
        return "guard_prefill"
    if "sd4_guard:proposal" in text:
        return "guard_proposal"
    if "sd4_guard:commit" in text:
        return "guard_commit"
    if "sd4_guard:posthoc" in text:
        return "guard_posthoc"
    if any(
        marker in text
        for marker in (
            "plain_draft_model_speculator",
            "plaindraftmodelspeculator",
            "spec_decode/draft_model/speculator.py",
            "spec_decode\\draft_model\\speculator.py",
            "_multi_step_decode",
            "sample_draft",
            "draft_proposal",
            "gpu_model_runner: draft",
        )
    ):
        return "draft_proposal"
    if any(
        marker in text
        for marker in (
            "rejection_sampler",
            "rejectionsampler",
            "rejection_sampling",
            "spec_decode/rejection_sampler.py",
            "spec_decode\\rejection_sampler.py",
        )
    ):
        return "rejection_sampling"
    if any(
        marker in text
        for marker in (
            "gpu_model_runner: forward",
            "gpu_model_runner: postprocess",
            "_model_forward",
            "compute_logits",
            "target_verification",
        )
    ):
        return "target_verification"
    if any(
        marker in text
        for marker in (
            "scheduler",
            "bookkeeping",
            "enginecore",
            "engine_core",
            "llm_engine step",
        )
    ):
        return "scheduler_and_bookkeeping"
    return "other_or_unattributed"


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _argument(event: dict[str, Any], *names: str) -> Any:
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    lowered = {str(key).lower(): value for key, value in args.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _is_cuda_runtime_event(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    name = str(event.get("name", "")).lower()
    return (
        "cuda_runtime" in category
        or "cuda_driver" in category
        or name.startswith("cudalaunch")
        or name.startswith("culaunch")
        or name.startswith("cudagraphlaunch")
    )


def _is_gpu_work_event(event: dict[str, Any]) -> bool:
    category = str(event.get("cat", "")).lower()
    return any(marker in category for marker in ("kernel", "gpu_memcpy", "gpu_memset"))


def _correlation_key(event: dict[str, Any]) -> str | None:
    value = _argument(event, "correlation", "correlation id")
    return None if value is None else str(value)


def _external_id(event: dict[str, Any]) -> str | None:
    value = _argument(event, "external id", "external_id")
    return None if value is None else str(value)


def _iter_complete_events(events: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        if _number(event.get("ts")) is None or _number(event.get("dur")) is None:
            continue
        yield event


def summarize_trace_events(events: list[Any]) -> dict[str, Any]:
    """Return attributed GPU kernel times from Chrome trace events."""

    complete = list(_iter_complete_events(events))
    external_stage: dict[str, str] = {}
    cpu_range_us = {stage: 0.0 for stage in STAGE_ORDER}
    for event in complete:
        stage = classify_profile_event(event)
        if not _is_gpu_work_event(event) and not _is_cuda_runtime_event(event):
            cpu_range_us[stage] += float(event.get("dur", 0.0))
        external_id = _external_id(event)
        if external_id is not None and stage != "other_or_unattributed":
            external_stage[external_id] = stage

    thread_events: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for event in complete:
        if not _is_gpu_work_event(event):
            thread_events[(event.get("pid"), event.get("tid"))].append(event)

    correlation_stage: dict[str, str] = {}
    cuda_graph_launch_count = 0
    for same_thread in thread_events.values():
        same_thread.sort(
            key=lambda event: (
                float(event["ts"]),
                -float(event["dur"]),
            )
        )
        stack: list[tuple[float, str]] = []
        for event in same_thread:
            started = float(event["ts"])
            while stack and started >= stack[-1][0]:
                stack.pop()

            direct_stage = classify_profile_event(event)
            inherited_stage = next(
                (
                    stage
                    for _, stage in reversed(stack)
                    if stage != "other_or_unattributed"
                ),
                "other_or_unattributed",
            )
            stage = direct_stage if direct_stage != "other_or_unattributed" else inherited_stage
            if stage == "other_or_unattributed":
                external_id = _external_id(event)
                if external_id is not None:
                    stage = external_stage.get(external_id, stage)

            if _is_cuda_runtime_event(event):
                correlation = _correlation_key(event)
                if correlation is not None:
                    correlation_stage[correlation] = stage
                if "graphlaunch" in str(event.get("name", "")).lower():
                    cuda_graph_launch_count += 1

            duration = float(event["dur"])
            if duration > 0 and not _is_cuda_runtime_event(event):
                stack.append((started + duration, stage))

    stage_us = {stage: 0.0 for stage in STAGE_ORDER}
    top_kernels: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"gpu_time_us": 0.0, "calls": 0}
    )
    kernel_count = 0
    for event in complete:
        if not _is_gpu_work_event(event):
            continue
        duration = float(event["dur"])
        if duration < 0:
            continue
        stage = "other_or_unattributed"
        correlation = _correlation_key(event)
        if correlation is not None:
            stage = correlation_stage.get(correlation, stage)
        if stage == "other_or_unattributed":
            external_id = _external_id(event)
            if external_id is not None:
                stage = external_stage.get(external_id, stage)
        stage_us[stage] += duration
        kernel_name = str(event.get("name", "<unnamed>"))
        top_kernels[kernel_name]["gpu_time_us"] = (
            float(top_kernels[kernel_name]["gpu_time_us"]) + duration
        )
        top_kernels[kernel_name]["calls"] = int(top_kernels[kernel_name]["calls"]) + 1
        kernel_count += 1

    total_us = sum(stage_us.values())
    stages = {
        stage: {
            "gpu_kernel_ms": stage_us[stage] / 1000.0,
            "cpu_range_inclusive_ms": cpu_range_us[stage] / 1000.0,
            "share_of_total_gpu_kernel_time": (
                None if total_us <= 0 else stage_us[stage] / total_us
            ),
        }
        for stage in STAGE_ORDER
    }
    attributed_us = total_us - stage_us["other_or_unattributed"]
    ranked_kernels = sorted(
        (
            {
                "name": name,
                "gpu_kernel_ms": float(values["gpu_time_us"]) / 1000.0,
                "calls": int(values["calls"]),
            }
            for name, values in top_kernels.items()
        ),
        key=lambda row: float(row["gpu_kernel_ms"]),
        reverse=True,
    )
    return {
        "available": total_us > 0,
        "total_gpu_kernel_ms": total_us / 1000.0,
        "kernel_event_count": kernel_count,
        "attribution_coverage": None if total_us <= 0 else attributed_us / total_us,
        "cuda_graph_launch_count": cuda_graph_launch_count,
        "stages": stages,
        "top_gpu_kernels": ranked_kernels[:20],
        "method": (
            "GPU kernel durations from the PyTorch Chrome trace, attributed through "
            "CUDA launch correlation IDs to recognizable vLLM CPU ranges/call stacks. "
            "cpu_range_inclusive_ms separately sums matching CPU ranges."
        ),
        "caveat": (
            "These percentages divide GPU kernel time, not end-to-end wall time. "
            "CPU ranges are inclusive and may overlap/nest, so their values are also "
            "diagnostic rather than additive wall-clock phases. Overlapping GPU work "
            "and kernels without correlation metadata can make them differ from a "
            "synchronized phase timer; inspect the raw trace when attribution coverage "
            "is low."
        ),
    }


def _load_trace(path: Path) -> list[Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
    else:
        events = payload
    if not isinstance(events, list):
        raise ValueError(f"Profiler trace does not contain a traceEvents list: {path}")
    return events


def summarize_profile_directory(directory: Path) -> dict[str, Any]:
    trace_files = sorted(directory.rglob("*.pt.trace.json"))
    trace_files.extend(sorted(directory.rglob("*.pt.trace.json.gz")))
    summary_files = sorted(directory.rglob("*profiler_out*.txt"))
    all_events: list[Any] = []
    errors: list[str] = []
    for trace_file in trace_files:
        try:
            all_events.extend(_load_trace(trace_file))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{trace_file}: {exc}")
    summary = summarize_trace_events(all_events)
    summary.update(
        {
            "trace_files": [str(path) for path in trace_files],
            "profiler_table_files": [str(path) for path in summary_files],
            "parse_errors": errors,
        }
    )
    return summary


def summarize_profile_files(paths: Iterable[Path]) -> dict[str, Any]:
    """Summarize an explicit trace set so adjacent diagnostics stay separate."""

    trace_files = sorted(Path(path) for path in paths)
    all_events: list[Any] = []
    errors: list[str] = []
    for trace_file in trace_files:
        try:
            all_events.extend(_load_trace(trace_file))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{trace_file}: {exc}")
    summary = summarize_trace_events(all_events)
    summary.update(
        {
            "trace_files": [str(path) for path in trace_files],
            "profiler_table_files": [],
            "parse_errors": errors,
        }
    )
    return summary

