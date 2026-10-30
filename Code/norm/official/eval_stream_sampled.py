#!/usr/bin/env python3
"""Run the released full-forward evaluator on an existing sample manifest.

The model forward, token boundary, and risk-head decoding are imported from
the untouched released ``eval_stream.py`` beside this file.  This wrapper only
adds deterministic sample selection, single-GPU validation, and JSON metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

# Allow direct execution as ``python official/eval_stream_sampled.py`` while
# keeping the released evaluator importable from this script's own directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_stream import ContentEvaluator

from guard_eval.dataset import load_split
from guard_eval.result_io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official Qwen3Guard full-forward logic on a shared manifest."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_one_visible_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The official comparison requires a CUDA GPU")
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA GPU, found {count}. "
            "Launch with CUDA_VISIBLE_DEVICES=<one physical GPU>."
        )


def require_unsharded_model(model: Any) -> None:
    expected = torch.device("cuda", 0)
    actual = {
        tensor.device
        for tensor in list(model.parameters()) + list(model.buffers())
    }
    if actual != {expected}:
        raise RuntimeError(
            "Official evaluator model was sharded or offloaded: "
            f"{sorted(str(device) for device in actual)!r}"
        )


def main() -> None:
    args = parse_args()
    output_path = args.output_path.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Official output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = read_json(args.sample_manifest.expanduser().resolve())
    if manifest.get("split") != "thinking":
        raise ValueError("Official comparison manifest must target thinking")
    entries = manifest.get("samples")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Sample manifest has no entries")

    dataset = load_split(
        args.dataset_path,
        "thinking",
        cache_dir=args.cache_dir,
    )
    selected: list[tuple[dict[str, Any], Any]] = []
    for expected_order, entry in enumerate(entries):
        order = int(entry.get("sample_order", expected_order))
        index = int(entry["dataset_index"])
        sample_id = str(entry["sample_id"])
        if order != expected_order:
            raise ValueError("Manifest sample_order is not contiguous")
        row = dataset[index]
        if str(row["unique_id"]) != sample_id:
            raise ValueError(
                f"Manifest mismatch at dataset index {index}: "
                f"expected {sample_id!r}, found {row['unique_id']!r}"
            )
        selected.append((dict(entry), row))

    require_one_visible_gpu()
    evaluator = ContentEvaluator(args.model_path)
    require_unsharded_model(evaluator.model)
    print("Verified official evaluator placement: logical cuda:0 only")

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    with output_path.open("w", encoding="utf-8", newline="\n") as writer:
        for entry, row in tqdm(selected, desc="official full forward · thinking"):
            result = evaluator.process_conversation_response(
                row["message"], thinking=True
            )
            output = {
                **dict(row),
                **result,
                "sample_id": str(row["unique_id"]),
                "sample_order": int(entry["sample_order"]),
                "dataset_index": int(entry["dataset_index"]),
            }
            writer.write(json.dumps(output, ensure_ascii=False) + "\n")
            writer.flush()

    metadata = {
        "status": "completed",
        "method": "released eval_stream.py ContentEvaluator full forward",
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "split": "thinking",
        "sample_count": len(selected),
        "sample_manifest": str(args.sample_manifest.expanduser().resolve()),
        "output_path": str(output_path),
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": time.perf_counter() - started,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpu_name": torch.cuda.get_device_name(0),
        "hf_device_map": {
            str(key): str(value)
            for key, value in getattr(evaluator.model, "hf_device_map", {}).items()
        },
        "metric_note": (
            "Strict/loose metrics are recomputed by analyze_stream_results.py from "
            "pred_risk_levels because the released script prints the two lists swapped."
        ),
    }
    write_json(output_path.with_suffix(output_path.suffix + ".metadata.json"), metadata)
    print(f"Official sampled predictions written to: {output_path}")


if __name__ == "__main__":
    main()
