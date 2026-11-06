from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from guard_training.common.feature_loading import read_jsonl
from guard_training.common.feature_store import iter_feature_metadata
from guard_training.frozen_hidden_mlp.model import HiddenStateClassifier
from guard_training.frozen_hidden_mlp.train import predict_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a whole-trace classifier head to causal tool-step hidden states."
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layer = int(checkpoint["layer"])
    model = HiddenStateClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    manifest_rows = [
        row for row in read_jsonl(args.manifest) if row["split"] == args.split
    ]
    feature_metadata = {
        row["sample_id"]: row
        for row in iter_feature_metadata(args.feature_root / "features")
    }
    by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        by_shard[int(row["shard"])].append(row)

    timelines = []
    for shard_index in sorted(by_shard):
        shard_path = (
            args.feature_root
            / "features"
            / f"shard_{shard_index:05d}.safetensors"
        )
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            step_hidden = handle.get_tensor("step_hidden")
        resolved_layer = layer if layer >= 0 else step_hidden.shape[1] + layer
        for row in sorted(by_shard[shard_index], key=lambda item: int(item["row"])):
            metadata = feature_metadata[row["sample_id"]]
            start = int(metadata["step_start"])
            end = int(metadata["step_end"])
            steps = metadata["steps"]
            if end <= start:
                scores = []
            else:
                features = step_hidden[start:end, resolved_layer].to(torch.float32)
                scores = predict_scores(model, features, args.batch_size, device).tolist()
            if len(scores) != len(steps):
                raise ValueError(f"Step score mismatch for {row['sample_id']}")
            timeline_steps = []
            for step, score in zip(steps, scores, strict=True):
                timeline_steps.append({**step, "risk_score": float(score)})
            timelines.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "label": int(row["label"]),
                    "suite_name": row["suite_name"],
                    "attack_type": row["attack_type"],
                    "split": row["split"],
                    "steps": timeline_steps,
                    "interpretation_warning": (
                        "The head was trained on complete traces. Step scores are exploratory "
                        "prefix-risk scores, not supervised step labels."
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for timeline in timelines:
            handle.write(json.dumps(timeline, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"samples": len(timelines), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

