#!/usr/bin/env python3
"""Fail-fast validation for the SD_4 CUDA 13.0 runtime."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
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


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in EXPECTED}


def driver_version() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return values[0] if values else None


def main() -> None:
    errors: list[str] = []
    if sys.version_info[:2] != (3, 10):
        errors.append(f"expected Python 3.10.x, found {sys.version.split()[0]}")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        errors.append(
            "expected Linux x86_64, found "
            f"{platform.system()} {platform.machine()}"
        )

    versions = package_versions()
    for name, expected in EXPECTED.items():
        actual = public_version(versions[name])
        if actual != expected:
            errors.append(f"expected {name} {expected}, found {versions[name]}")

    import torch

    if public_version(str(torch.__version__)) != EXPECTED["torch"]:
        errors.append(f"unexpected torch runtime version: {torch.__version__}")
    if torch.version.cuda != "13.0":
        errors.append(f"expected Torch CUDA 13.0, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is False")
    if torch.cuda.device_count() != 1:
        errors.append(
            "SD_4 requires exactly one visible GPU; "
            f"found {torch.cuda.device_count()}"
        )

    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        capability = tuple(torch.cuda.get_device_capability(0))
        if capability < (7, 5):
            errors.append(
                "vLLM requires compute capability >= 7.5; "
                f"found {capability[0]}.{capability[1]}"
            )
        gpu = {
            "name": props.name,
            "compute_capability": list(capability),
            "memory_gib": round(props.total_memory / (1024**3), 2),
        }

    driver = driver_version()
    if driver is None:
        errors.append("could not read the NVIDIA driver version with nvidia-smi")
    else:
        match = re.match(r"(\d+)", driver)
        if match is None or int(match.group(1)) < 580:
            errors.append(
                "CUDA 13.0 requires an R580-or-newer driver; "
                f"found {driver}"
            )

    report = {
        "status": "ok" if not errors else "failed",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "torch_runtime": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": driver,
        "visible_gpu_count": torch.cuda.device_count(),
        "gpu": gpu,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
