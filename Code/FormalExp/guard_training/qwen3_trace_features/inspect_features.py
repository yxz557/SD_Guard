from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from guard_training.common.feature_store import iter_feature_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate extracted feature shards.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--check-finite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_dir = args.feature_root / "features"
    rows = list(iter_feature_metadata(feature_dir))
    by_shard = Counter(int(row["shard"]) for row in rows)
    labels = Counter(int(row["label"]) for row in rows)
    suites = Counter(str(row["suite_name"]) for row in rows)
    attacks = Counter(str(row["attack_type"]) for row in rows)
    token_counts = [int(row["token_count"]) for row in rows]
    step_counts = [len(row.get("steps") or []) for row in rows]
    shard_reports: list[dict[str, Any]] = []
    expected_keys: set[str] | None = None

    for shard_index in sorted(by_shard):
        path = feature_dir / f"shard_{shard_index:05d}.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(f"Tensor keys differ in {path}: {keys} != {expected_keys}")
            shapes = {key: list(handle.get_slice(key).get_shape()) for key in sorted(keys)}
            first_global = next(
                key for key in sorted(keys) if key not in {"step_hidden", "step_offsets", "step_kind", "step_token_index"}
            )
            if shapes[first_global][0] != by_shard[shard_index]:
                raise ValueError(f"Metadata/tensor row mismatch in {path}")
            if shapes["step_offsets"][0] != by_shard[shard_index] + 1:
                raise ValueError(f"Invalid step_offsets shape in {path}")
            if args.check_finite:
                for key in sorted(keys):
                    if key.startswith("step_") and key != "step_hidden":
                        continue
                    tensor = handle.get_tensor(key)
                    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                        raise ValueError(f"Non-finite values found in {path}:{key}")
            shard_reports.append(
                {
                    "shard": shard_index,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "metadata_rows": by_shard[shard_index],
                    "shapes": shapes,
                }
            )

    report = {
        "samples": len(rows),
        "unique_sample_ids": len({row["sample_id"] for row in rows}),
        "labels": dict(sorted(labels.items())),
        "suites": dict(sorted(suites.items())),
        "attacks": dict(sorted(attacks.items())),
        "token_count": {
            "min": min(token_counts) if token_counts else None,
            "max": max(token_counts) if token_counts else None,
            "mean": sum(token_counts) / len(token_counts) if token_counts else None,
        },
        "step_count": {
            "min": min(step_counts) if step_counts else None,
            "max": max(step_counts) if step_counts else None,
            "mean": sum(step_counts) / len(step_counts) if step_counts else None,
        },
        "total_feature_bytes": sum(item["bytes"] for item in shard_reports),
        "shards": shard_reports,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized)


if __name__ == "__main__":
    main()

