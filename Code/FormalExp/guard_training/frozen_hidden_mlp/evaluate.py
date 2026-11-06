from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from guard_training.common.feature_loading import load_feature_matrix
from guard_training.common.metrics import binary_metrics
from guard_training.frozen_hidden_mlp.model import HiddenStateClassifier
from guard_training.frozen_hidden_mlp.train import predict_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen-hidden-state classifier.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def grouped_metrics(
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    key: str,
) -> dict[str, Any]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices[str(row[key])].append(index)
    return {
        value: binary_metrics(labels[selected], scores[selected], threshold)
        for value, selected in sorted(indices.items())
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    features, labels, rows = load_feature_matrix(
        args.feature_root,
        args.manifest,
        args.split,
        checkpoint["feature"],
        int(checkpoint["layer"]),
    )
    model = HiddenStateClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    scores = predict_scores(model, features, args.batch_size, device)
    labels_np = labels.numpy()
    threshold = float(checkpoint["threshold"])

    report = {
        "split": args.split,
        "feature": checkpoint["feature"],
        "layer": checkpoint["layer"],
        "threshold_selected_on_validation": threshold,
        "overall": binary_metrics(labels_np, scores, threshold),
        "by_suite": grouped_metrics(rows, labels_np, scores, threshold, "suite_name"),
        "by_attack": grouped_metrics(rows, labels_np, scores, threshold, "attack_type"),
    }
    (args.output_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    prediction_path = args.output_dir / f"{args.split}_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, label, score in zip(rows, labels_np, scores, strict=True):
            payload = {
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "suite_name": row["suite_name"],
                "attack_type": row["attack_type"],
                "label": int(label),
                "score": float(score),
                "prediction": int(score >= threshold),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

