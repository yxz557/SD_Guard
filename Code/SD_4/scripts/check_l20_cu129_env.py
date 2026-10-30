#!/usr/bin/env python3
"""Fail-fast validation for the two-L20 SD_4 CUDA 12.9 environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED = {
    "vllm": "0.31.0",
    "torch": "2.13.0",
    "transformers": "5.17.0",
    "tokenizers": "0.23.3",
    "datasets": "4.8.5",
    "accelerate": "1.15.0",
}


def public_version(value: str) -> str:
    return value.split("+", 1)[0]


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))
    return ordered[index]


def transfer_benchmark(torch: Any, token_count: int) -> dict[str, Any]:
    source = torch.arange(token_count, dtype=torch.int64, device="cuda:0")
    for _ in range(10):
        copied = source.to("cuda:1", non_blocking=False)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        del copied
    measurements: list[float] = []
    for _ in range(100):
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        started_ns = time.perf_counter_ns()
        copied = source.to("cuda:1", non_blocking=False)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        measurements.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
        del copied
    return {
        "token_count": token_count,
        "bytes": token_count * 8,
        "count": len(measurements),
        "mean_ms": statistics.fmean(measurements),
        "median_ms": statistics.median(measurements),
        "p90_ms": percentile(measurements, 0.9),
        "min_ms": min(measurements),
        "max_ms": max(measurements),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-transfer-benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    versions: dict[str, str | None] = {}
    for name, expected in EXPECTED.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
            errors.append(f"missing required package: {name}=={expected}")
        versions[name] = installed
        if installed is not None and public_version(installed) != expected:
            errors.append(f"expected {name} {expected}, found {installed}")

    if sys.version_info[:2] != (3, 10):
        errors.append(f"expected Python 3.10.x, found {sys.version.split()[0]}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        errors.append(
            "expected Linux x86_64, found "
            f"{platform.system()} {platform.machine()}"
        )

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch import failed: {exc}") from exc

    if torch.version.cuda != "12.9":
        errors.append(f"expected Torch CUDA runtime 12.9, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is False")
    if torch.cuda.device_count() != 2:
        errors.append(
            "expected exactly two visible GPUs ordered as engine,guard; "
            f"found {torch.cuda.device_count()}"
        )

    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            name = str(props.name)
            if "L20" not in name:
                warnings.append(f"logical cuda:{index} is {name}, not NVIDIA L20")
            gpus.append(
                {
                    "logical_device": f"cuda:{index}",
                    "name": name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )

    peer_access: dict[str, bool] | None = None
    transfer: dict[str, Any] | None = None
    if torch.cuda.is_available() and torch.cuda.device_count() == 2:
        peer_access = {
            "0_to_1": bool(torch.cuda.can_device_access_peer(0, 1)),
            "1_to_0": bool(torch.cuda.can_device_access_peer(1, 0)),
        }
        if not all(peer_access.values()):
            warnings.append("CUDA peer access is unavailable in at least one direction")
        if not args.skip_transfer_benchmark:
            transfer = {
                "path": "direct torch cuda:0 -> cuda:1 Tensor.to diagnostic",
                "formal_latency_included": False,
                "measurements": {
                    str(count): transfer_benchmark(torch, count)
                    for count in (5, 4096)
                },
            }

    pip_check = command_output([sys.executable, "-m", "pip", "check"])
    if pip_check.get("returncode") not in (None, 0):
        errors.append("pip check reported broken requirements")
    topology = command_output(["nvidia-smi", "topo", "-m"])
    topology_text = str(topology.get("stdout", ""))
    if "SYS" in topology_text:
        warnings.append(
            "GPU0/GPU1 topology traverses SYS/NUMA; measure handoff latency and "
            "do not describe it as NVLink"
        )

    report = {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpus,
        "peer_access": peer_access,
        "direct_transfer_diagnostic": transfer,
        "nvidia_smi_topology": topology,
        "pip_check": pip_check,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
