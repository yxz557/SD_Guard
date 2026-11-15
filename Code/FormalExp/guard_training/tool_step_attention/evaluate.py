from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from guard_training.common.metrics import binary_metrics
from guard_training.tool_step_attention.dataset import (
    CALL_VARIANTS,
    RESULT_VARIANTS,
    ToolStepAttentionDataset,
    collate_attention,
)
from guard_training.tool_step_attention.model import (
    ToolStepAttentionClassifier,
    move_batch,
)
from guard_training.tool_step_attention.train import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a readout-conditioned tool-step attention classifier."
    )
    parser.add_argument("--offline-dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attention-top-k", type=int, default=3)
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


def top_entries(
    weights: torch.Tensor,
    valid_count: int,
    metadata: list[dict[str, Any]],
    top_k: int,
    valid_positions: list[int] | None = None,
) -> list[dict[str, Any]]:
    if valid_count <= 0 or top_k <= 0:
        return []
    positions = (
        list(range(valid_count)) if valid_positions is None else valid_positions
    )
    if not positions:
        return []
    candidate_weights = weights[positions]
    local = torch.topk(
        candidate_weights, k=min(top_k, len(positions))
    ).indices.tolist()
    selected = [positions[index] for index in local]
    return [
        {
            "position": int(position),
            "weight": float(weights[position].item()),
            "step": metadata[position] if position < len(metadata) else None,
        }
        for position in selected
    ]


def attention_summary(
    row: dict[str, Any],
    variant: str,
    attention: dict[str, torch.Tensor],
    batch_position: int,
    top_k: int,
) -> dict[str, Any]:
    steps = list(row.get("steps", []))
    if variant in CALL_VARIANTS:
        metadata = [step for step in steps if step.get("kind") == "tool_call_end"]
        weights = attention["step_weights"][batch_position].cpu()
        return {"top_calls": top_entries(weights, len(metadata), metadata, top_k)}
    if variant in RESULT_VARIANTS:
        metadata = [step for step in steps if step.get("kind") == "tool_result_end"]
        weights = attention["step_weights"][batch_position].cpu()
        return {"top_results": top_entries(weights, len(metadata), metadata, top_k)}
    pairs = list(row.get("round_pairs", []))
    decision_weights = attention["decision_weights"][batch_position].cpu()
    result_weights = attention["result_weights"][batch_position].cpu()
    return {
        "top_decisions": top_entries(decision_weights, len(pairs), pairs, top_k),
        "top_round_results": top_entries(
            result_weights,
            len(pairs),
            pairs,
            top_k,
            [
                index
                for index, pair in enumerate(pairs)
                if pair.get("result_step_index") is not None
            ],
        ),
    }


def predict(
    model: ToolStepAttentionClassifier,
    dataset: ToolStepAttentionDataset,
    batch_size: int,
    device: torch.device,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_attention, variant=dataset.variant),
    )
    scores = np.empty(len(dataset), dtype=np.float64)
    labels = np.empty(len(dataset), dtype=np.int64)
    summaries: list[dict[str, Any] | None] = [None] * len(dataset)
    model.eval()
    with torch.inference_mode():
        for original_batch in loader:
            indices = original_batch["indices"].tolist()
            batch = move_batch(original_batch, device)
            output = model(batch, return_attention=True)
            assert isinstance(output, tuple)
            logits, attention = output
            probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            batch_labels = batch["labels"].cpu().numpy()
            for batch_position, dataset_index in enumerate(indices):
                scores[dataset_index] = probabilities[batch_position]
                labels[dataset_index] = batch_labels[batch_position]
                summaries[dataset_index] = attention_summary(
                    dataset.rows[dataset_index],
                    dataset.variant,
                    attention,
                    batch_position,
                    top_k,
                )
    if any(value is None for value in summaries):
        raise AssertionError("Failed to generate every attention summary")
    return scores, labels, [value for value in summaries if value is not None]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ToolStepAttentionClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    variant = str(checkpoint["model_config"]["variant"])
    dataset = ToolStepAttentionDataset(
        args.offline_dataset.resolve(), args.split, variant
    )
    if dataset.config != checkpoint["offline_dataset_config"]:
        raise ValueError("Checkpoint was trained on a different attention dataset")
    scores, labels, summaries = predict(
        model, dataset, args.batch_size, device, args.attention_top_k
    )
    threshold = float(checkpoint["threshold"])
    operating = {
        name: {
            "threshold_selected_on_validation": float(value),
            "metrics": binary_metrics(labels, scores, float(value)),
        }
        for name, value in checkpoint.get("operating_thresholds", {}).items()
    }
    report = {
        "split": args.split,
        "feature": variant,
        "layer": int(dataset.config["layer"]),
        "threshold_selected_on_validation": threshold,
        "overall": binary_metrics(labels, scores, threshold),
        "by_suite": grouped_metrics(
            dataset.rows, labels, scores, threshold, "suite_name"
        ),
        "by_attack": grouped_metrics(
            dataset.rows, labels, scores, threshold, "attack_type"
        ),
        "operating_points": operating,
    }
    metrics_path = args.output_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    prediction_path = args.output_dir / f"{args.split}_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, label, score, attention in zip(
            dataset.rows, labels, scores, summaries, strict=True
        ):
            payload = {
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "suite_name": row["suite_name"],
                "attack_type": row["attack_type"],
                "label": int(label),
                "score": float(score),
                "prediction": int(score >= threshold),
                "attention": attention,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

