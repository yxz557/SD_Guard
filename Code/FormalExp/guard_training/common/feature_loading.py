from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_feature_matrix(
    feature_root: Path,
    manifest_path: Path,
    split: str,
    feature_name: str,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    selected = [row for row in read_jsonl(manifest_path) if row["split"] == split]
    if not selected:
        raise ValueError(f"No rows found for split={split}")

    by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_shard[int(row["shard"])].append(row)

    matrices: list[torch.Tensor] = []
    labels: list[int] = []
    ordered_rows: list[dict[str, Any]] = []
    feature_dir = feature_root / "features"
    for shard_index in sorted(by_shard):
        path = feature_dir / f"shard_{shard_index:05d}.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if feature_name not in handle.keys():
                raise KeyError(f"Feature {feature_name!r} is absent from {path}")
            tensor = handle.get_tensor(feature_name)
        if tensor.ndim != 3:
            raise ValueError(f"Expected [samples,layers,hidden] for {feature_name}, got {tensor.shape}")
        resolved_layer = layer if layer >= 0 else tensor.shape[1] + layer
        if not 0 <= resolved_layer < tensor.shape[1]:
            raise IndexError(f"Layer {layer} is invalid for {tensor.shape[1]} saved layers")
        for row in sorted(by_shard[shard_index], key=lambda item: int(item["row"])):
            matrices.append(tensor[int(row["row"]), resolved_layer].to(torch.float32))
            labels.append(int(row["label"]))
            ordered_rows.append(row)

    return torch.stack(matrices), torch.tensor(labels, dtype=torch.long), ordered_rows

