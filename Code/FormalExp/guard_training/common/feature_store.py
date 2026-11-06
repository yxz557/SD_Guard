from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


STEP_KIND_TO_ID = {
    "tool_call_end": 0,
    "assistant_decision_end": 1,
    "tool_result_end": 2,
    "tool_round_end": 3,
}


def iter_feature_metadata(feature_dir: Path):
    for path in sorted(feature_dir.glob("shard_*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_processed_sample_ids(feature_dir: Path) -> set[str]:
    return {str(row["sample_id"]) for row in iter_feature_metadata(feature_dir)}


class FeatureShardWriter:
    def __init__(self, feature_dir: Path, shard_size: int) -> None:
        self.feature_dir = feature_dir
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.rows: list[dict[str, Any]] = []
        self.global_features: list[dict[str, torch.Tensor]] = []
        self.step_features: list[torch.Tensor] = []
        existing = sorted(feature_dir.glob("shard_*.jsonl"))
        self.next_shard = (
            max(int(path.stem.split("_")[1]) for path in existing) + 1 if existing else 0
        )

    def add(
        self,
        metadata: dict[str, Any],
        global_features: dict[str, torch.Tensor],
        step_features: torch.Tensor,
        step_metadata: list[dict[str, Any]],
    ) -> None:
        if step_features.ndim != 3:
            raise ValueError(f"Expected step features [steps,layers,hidden], got {step_features.shape}")
        if step_features.shape[0] != len(step_metadata):
            raise ValueError("Step feature count does not match step metadata")
        if self.global_features and global_features.keys() != self.global_features[0].keys():
            raise ValueError("Global feature keys changed within a shard")

        row = dict(metadata)
        row["steps"] = step_metadata
        self.rows.append(row)
        self.global_features.append({key: value.contiguous() for key, value in global_features.items()})
        self.step_features.append(step_features.contiguous())
        if len(self.rows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return

        shard_name = f"shard_{self.next_shard:05d}"
        tensor_path = self.feature_dir / f"{shard_name}.safetensors"
        metadata_path = self.feature_dir / f"{shard_name}.jsonl"
        if tensor_path.exists() or metadata_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing shard {shard_name}")

        tensors: dict[str, torch.Tensor] = {}
        for feature_name in self.global_features[0]:
            tensors[feature_name] = torch.stack(
                [row[feature_name] for row in self.global_features], dim=0
            ).contiguous()

        offsets = [0]
        for step_tensor in self.step_features:
            offsets.append(offsets[-1] + int(step_tensor.shape[0]))
        tensors["step_offsets"] = torch.tensor(offsets, dtype=torch.int64)
        if offsets[-1] > 0:
            tensors["step_hidden"] = torch.cat(self.step_features, dim=0).contiguous()
        else:
            reference = next(iter(self.global_features[0].values()))
            tensors["step_hidden"] = torch.empty(
                (0, reference.shape[0], reference.shape[1]), dtype=reference.dtype
            )

        step_kind_ids: list[int] = []
        step_token_indices: list[int] = []
        for row in self.rows:
            for step in row["steps"]:
                step_kind_ids.append(STEP_KIND_TO_ID[step["kind"]])
                step_token_indices.append(int(step["token_index"]))
        tensors["step_kind"] = torch.tensor(step_kind_ids, dtype=torch.int16)
        tensors["step_token_index"] = torch.tensor(step_token_indices, dtype=torch.int32)

        tensor_tmp = tensor_path.with_name(tensor_path.name + ".tmp")
        metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
        save_file(tensors, str(tensor_tmp))

        with metadata_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for row_index, row in enumerate(self.rows):
                output = dict(row)
                output["shard"] = self.next_shard
                output["row"] = row_index
                output["step_start"] = offsets[row_index]
                output["step_end"] = offsets[row_index + 1]
                handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")

        os.replace(tensor_tmp, tensor_path)
        os.replace(metadata_tmp, metadata_path)
        self.next_shard += 1
        self.rows.clear()
        self.global_features.clear()
        self.step_features.clear()

