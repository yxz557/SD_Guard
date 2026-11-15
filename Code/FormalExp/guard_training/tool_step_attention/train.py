from __future__ import annotations

import argparse
import json
import math
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from guard_training.common.metrics import binary_metrics, choose_fbeta_threshold
from guard_training.frozen_hidden_mlp.train import resolve_positive_weight, set_seed
from guard_training.tool_step_attention.dataset import (
    VARIANTS,
    ToolStepAttentionDataset,
    collate_attention,
)
from guard_training.tool_step_attention.model import (
    ToolStepAttentionClassifier,
    model_config,
    model_parameter_count,
    move_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a readout-conditioned tool-step attention classifier."
    )
    parser.add_argument("--offline-dataset", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--positive-weight", default="none")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def loader_for(
    dataset: ToolStepAttentionDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        collate_fn=partial(collate_attention, variant=dataset.variant),
    )


def predict_scores(
    model: ToolStepAttentionClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            assert isinstance(logits, torch.Tensor)
            scores.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
            labels.append(batch["labels"].cpu())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def evaluate_loss(
    model: ToolStepAttentionClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            assert isinstance(logits, torch.Tensor)
            loss = criterion(logits, batch["labels"])
            count = int(batch["labels"].shape[0])
            total_loss += float(loss.item()) * count
            total_examples += count
    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return total_loss / total_examples


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(np.asarray(scores, dtype=np.float64))[::-1]
    return np.concatenate(([np.inf], unique, [-np.inf]))


def confusion_rates(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float]:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    recall = float(np.mean(predictions[positives])) if positives.any() else 0.0
    fpr = float(np.mean(predictions[negatives])) if negatives.any() else 0.0
    return recall, fpr


def choose_threshold_at_fpr(
    labels: np.ndarray, scores: np.ndarray, target_fpr: float
) -> float:
    candidates: list[tuple[float, float, float]] = []
    for threshold in candidate_thresholds(scores):
        recall, fpr = confusion_rates(labels, scores, float(threshold))
        if fpr <= target_fpr + 1e-12:
            candidates.append((recall, -fpr, -float(threshold)))
    if not candidates:
        return float("inf")
    best = max(candidates)
    return -best[2]


def choose_threshold_at_recall(
    labels: np.ndarray, scores: np.ndarray, target_recall: float
) -> float:
    candidates: list[tuple[float, float, float]] = []
    for threshold in candidate_thresholds(scores):
        recall, fpr = confusion_rates(labels, scores, float(threshold))
        if recall + 1e-12 >= target_recall:
            candidates.append((-fpr, recall, float(threshold)))
    if not candidates:
        return float("-inf")
    return max(candidates)[2]


def operating_thresholds(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "validation_f2": choose_fbeta_threshold(labels, scores, beta=2.0),
        "validation_fpr_05": choose_threshold_at_fpr(labels, scores, 0.05),
        "validation_fpr_10": choose_threshold_at_fpr(labels, scores, 0.10),
        "validation_recall_95": choose_threshold_at_recall(labels, scores, 0.95),
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    train_dataset = ToolStepAttentionDataset(
        args.offline_dataset.resolve(), "train", args.variant
    )
    validation_dataset = ToolStepAttentionDataset(
        args.offline_dataset.resolve(), "validation", args.variant
    )
    if train_dataset.config != validation_dataset.config:
        raise ValueError("Attention dataset config changed between split loads")
    train_loader = loader_for(
        train_dataset,
        args.batch_size,
        True,
        args.seed,
        args.num_workers,
    )
    validation_loader = loader_for(
        validation_dataset,
        args.batch_size,
        False,
        args.seed,
        args.num_workers,
    )
    train_labels = train_dataset.tensors["labels"].to(torch.long)
    positive_weight = resolve_positive_weight(args.positive_weight, train_labels)
    class_weights = torch.tensor([1.0, positive_weight], dtype=torch.float32, device=device)
    model = ToolStepAttentionClassifier(
        hidden_size=int(train_dataset.config["hidden_size"]),
        variant=args.variant,
        attention_dim=args.attention_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    config = vars(args).copy()
    config.update(
        {
            "offline_dataset": str(args.offline_dataset.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "device_resolved": str(device),
            "head": "mlp",
            "hidden_dim": args.head_hidden_dim,
            "input_dim": model.output_dim,
            "model_parameters": model_parameter_count(model),
            "train_samples": len(train_dataset),
            "train_positives": int(train_labels.sum().item()),
            "validation_samples": len(validation_dataset),
            "validation_positives": int(
                validation_dataset.tensors["labels"].sum().item()
            ),
            "positive_weight_resolved": positive_weight,
        }
    )
    (args.output_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    best_ap = -math.inf
    stale_epochs = 0
    checkpoint_path = args.output_dir / "best.pt"
    history_path = args.output_dir / "history.jsonl"
    with history_path.open("w", encoding="utf-8", newline="\n") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch)
                assert isinstance(logits, torch.Tensor)
                loss = criterion(logits, batch["labels"])
                loss.backward()
                optimizer.step()
                count = int(batch["labels"].shape[0])
                total_loss += float(loss.item()) * count
                total_examples += count

            validation_scores, validation_labels = predict_scores(
                model, validation_loader, device
            )
            validation_loss = evaluate_loss(
                model, validation_loader, criterion, device
            )
            thresholds = operating_thresholds(validation_labels, validation_scores)
            metrics = binary_metrics(
                validation_labels,
                validation_scores,
                thresholds["validation_f2"],
            )
            metrics.update(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(total_examples, 1),
                    "validation_loss": validation_loss,
                }
            )
            history.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            history.flush()
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            score = float(metrics.get("average_precision") or -math.inf)
            if score > best_ap:
                best_ap = score
                stale_epochs = 0
                atomic_torch_save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": model_config(model),
                        "offline_dataset_config": train_dataset.config,
                        "threshold": thresholds["validation_f2"],
                        "operating_thresholds": thresholds,
                        "validation_metrics": metrics,
                        "train_config": config,
                    },
                    checkpoint_path,
                )
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    (args.output_dir / "best_validation_metrics.json").write_text(
        json.dumps(
            {
                **checkpoint["validation_metrics"],
                "operating_thresholds": checkpoint["operating_thresholds"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
