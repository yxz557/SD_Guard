#!/usr/bin/env python3
"""Load Qwen3Guard alone and run SD_4's Transformers compatibility probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from sdguard_sd.runtime import load_guard_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/model_and_data/yxz/model/Qwen3Guard-Stream-4B",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--revision")
    parser.add_argument("--attention-implementation")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _ = load_guard_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        revision=args.revision,
        attention_implementation=args.attention_implementation,
    )
    report = getattr(model, "_sd4_compatibility_report", None)
    if not isinstance(report, dict) or report.get("status") != "passed":
        raise RuntimeError(f"Guard compatibility probe did not pass: {report!r}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
