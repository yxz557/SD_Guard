from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from guard_training.common.feature_loading import load_feature_matrix
from guard_training.common.metrics import binary_metrics, choose_fbeta_threshold
from guard_training.frozen_hidden_mlp.model import HiddenStateClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frozen-hidden-state classifier head.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature", default="trace_last")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--head", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--positive-weight", default="inverse")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_positive_weight(mode: str, labels: torch.Tensor) -> float:
    positives = int(labels.sum().item())
    negatives = int(labels.numel() - positives)
    if positives == 0:
        raise ValueError("Training split contains no positive examples")
    ratio = negatives / positives
    if mode == "none":
        return 1.0
    if mode == "inverse":
        return ratio
    if mode == "sqrt_inverse":
        return math.sqrt(ratio)
    value = float(mode)
    if value <= 0:
        raise ValueError("Positive class weight must be positive")
    return value


def predict_scores(
    model: nn.Module,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size):
            logits = model(batch.to(device))
            outputs.append(torch.softmax(logits, dim=-1)[:, 1].cpu())
    return torch.cat(outputs).numpy()


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )

    train_x, train_y, _ = load_feature_matrix(
        args.feature_root, args.manifest, "train", args.feature, args.layer
    )
    validation_x, validation_y, validation_rows = load_feature_matrix(
        args.feature_root, args.manifest, "validation", args.feature, args.layer
    )
    positive_weight = resolve_positive_weight(args.positive_weight, train_y)
    class_weights = torch.tensor([1.0, positive_weight], dtype=torch.float32, device=device)

    model = HiddenStateClassifier(
        input_dim=int(train_x.shape[1]),
        head=args.head,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
    )

    config = vars(args).copy()
    config.update(
        {
            "feature_root": str(args.feature_root.resolve()),
            "manifest": str(args.manifest.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "device_resolved": str(device),
            "input_dim": int(train_x.shape[1]),
            "train_samples": int(train_y.numel()),
            "train_positives": int(train_y.sum().item()),
            "validation_samples": int(validation_y.numel()),
            "validation_positives": int(validation_y.sum().item()),
            "positive_weight_resolved": positive_weight,
        }
    )
    (args.output_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    best_average_precision = -math.inf
    stale_epochs = 0
    history_path = args.output_dir / "history.jsonl"
    checkpoint_path = args.output_dir / "best.pt"
    validation_labels_np = validation_y.numpy()

    with history_path.open("w", encoding="utf-8", newline="\n") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * batch_y.shape[0]
                total_examples += int(batch_y.shape[0])

            validation_scores = predict_scores(
                model, validation_x, args.batch_size, device
            )
            threshold = choose_fbeta_threshold(validation_labels_np, validation_scores, beta=2.0)
            metrics = binary_metrics(validation_labels_np, validation_scores, threshold)
            metrics.update(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(total_examples, 1),
                }
            )
            history.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")
            history.flush()
            print(json.dumps(metrics, ensure_ascii=False), flush=True)

            average_precision = metrics.get("average_precision")
            score = float(average_precision) if average_precision is not None else -math.inf
            if score > best_average_precision:
                best_average_precision = score
                stale_epochs = 0
                atomic_torch_save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": {
                            "input_dim": int(train_x.shape[1]),
                            "head": args.head,
                            "hidden_dim": args.hidden_dim,
                            "dropout": args.dropout,
                        },
                        "feature": args.feature,
                        "layer": args.layer,
                        "threshold": threshold,
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
            checkpoint["validation_metrics"],
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

