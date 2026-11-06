from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .feature_loading import read_jsonl


def load_offline_config(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "dataset_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing compact dataset configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_offline_split(
    dataset_dir: Path,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    tensor_path = dataset_dir / f"{split}.safetensors"
    metadata_path = dataset_dir / f"{split}.metadata.jsonl"
    if not tensor_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing compact dataset split: {split}")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        features = handle.get_tensor("features").to(torch.float32)
        labels = handle.get_tensor("labels").to(torch.long)
    rows = read_jsonl(metadata_path)
    if features.ndim != 2:
        raise ValueError(f"Expected compact features [samples,hidden], got {features.shape}")
    if features.shape[0] != labels.numel() or features.shape[0] != len(rows):
        raise ValueError(f"Feature/label/metadata mismatch for split={split}")
    return features, labels, rows, load_offline_config(dataset_dir)

