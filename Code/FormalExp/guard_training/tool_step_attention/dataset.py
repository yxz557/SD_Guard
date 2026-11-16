from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from guard_training.common.feature_loading import read_jsonl


VARIANTS = (
    "c_qattn",
    "t_qattn",
    "rc_qattn",
    "rt_qattn",
    "rct_independent",
    "rct_shared",
    "rct_tool_independent",
    "rct_tool_shared",
    "rct_round_tool_independent",
    "rct_round_tool_shared",
)
CALL_VARIANTS = {"c_qattn", "rc_qattn"}
RESULT_VARIANTS = {"t_qattn", "rt_qattn"}
# Legacy variants pool assistant_decision_end/tool_round_end themselves.
ROUND_VARIANTS = {"rct_independent", "rct_shared"}
TOOL_RCT_VARIANTS = {"rct_tool_independent", "rct_tool_shared"}
ROUND_TOOL_RCT_VARIANTS = {
    "rct_round_tool_independent",
    "rct_round_tool_shared",
}
INDIVIDUAL_RCT_VARIANTS = TOOL_RCT_VARIANTS | ROUND_TOOL_RCT_VARIANTS
SHARED_RESULT_VARIANTS = {"rct_tool_shared", "rct_round_tool_shared"}


def load_config(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "dataset_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing attention dataset config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tensors(dataset_dir: Path, split: str, variant: str) -> dict[str, torch.Tensor]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown attention variant: {variant}")
    path = dataset_dir / f"{split}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Missing attention split tensors: {path}")
    common = {"readout", "labels"}
    if variant in CALL_VARIANTS:
        required = common | {"call_hidden", "call_offsets"}
    elif variant in RESULT_VARIANTS:
        required = common | {"result_hidden", "result_offsets"}
    elif variant in ROUND_VARIANTS:
        required = common | {
            "decision_hidden",
            "round_result_hidden",
            "round_result_mask",
            "round_offsets",
        }
    elif variant in INDIVIDUAL_RCT_VARIANTS:
        required = common | {
            "call_hidden",
            "call_offsets",
            "result_hidden",
            "result_offsets",
            "result_to_call",
        }
        if variant in ROUND_TOOL_RCT_VARIANTS:
            required |= {
                "call_round_context",
                "call_round_context_mask",
                "call_round_group",
                "result_round_context",
                "result_round_context_mask",
                "result_round_group",
            }
    else:
        raise ValueError(f"Unknown attention variant: {variant}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        missing = required - set(handle.keys())
        if missing:
            raise KeyError(f"Missing tensors in {path}: {sorted(missing)}")
        return {key: handle.get_tensor(key) for key in required}


class ToolStepAttentionDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str, variant: str) -> None:
        self.dataset_dir = dataset_dir
        self.split = split
        self.variant = variant
        self.config = load_config(dataset_dir)
        self.tensors = load_tensors(dataset_dir, split, variant)
        self.rows = read_jsonl(dataset_dir / f"{split}.metadata.jsonl")
        sample_count = int(self.tensors["labels"].numel())
        if len(self.rows) != sample_count or self.tensors["readout"].shape[0] != sample_count:
            raise ValueError(f"Metadata/tensor mismatch for split={split}")

    def __len__(self) -> int:
        return int(self.tensors["labels"].numel())

    @staticmethod
    def _slice(
        hidden: torch.Tensor, offsets: torch.Tensor, index: int
    ) -> torch.Tensor:
        start = int(offsets[index].item())
        end = int(offsets[index + 1].item())
        return hidden[start:end]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "index": index,
            "readout": self.tensors["readout"][index],
            "label": self.tensors["labels"][index],
        }
        if self.variant in CALL_VARIANTS:
            item["steps"] = self._slice(
                self.tensors["call_hidden"], self.tensors["call_offsets"], index
            )
        elif self.variant in RESULT_VARIANTS:
            item["steps"] = self._slice(
                self.tensors["result_hidden"], self.tensors["result_offsets"], index
            )
        elif self.variant in ROUND_VARIANTS:
            offsets = self.tensors["round_offsets"]
            item["decisions"] = self._slice(
                self.tensors["decision_hidden"], offsets, index
            )
            item["round_results"] = self._slice(
                self.tensors["round_result_hidden"], offsets, index
            )
            item["round_result_mask"] = self._slice(
                self.tensors["round_result_mask"], offsets, index
            )
        elif self.variant in INDIVIDUAL_RCT_VARIANTS:
            call_offsets = self.tensors["call_offsets"]
            result_offsets = self.tensors["result_offsets"]
            item["calls"] = self._slice(
                self.tensors["call_hidden"], call_offsets, index
            )
            item["results"] = self._slice(
                self.tensors["result_hidden"], result_offsets, index
            )
            item["result_to_call"] = self._slice(
                self.tensors["result_to_call"], result_offsets, index
            )
            if self.variant in ROUND_TOOL_RCT_VARIANTS:
                item["call_round_context"] = self._slice(
                    self.tensors["call_round_context"], call_offsets, index
                )
                item["call_round_context_mask"] = self._slice(
                    self.tensors["call_round_context_mask"], call_offsets, index
                )
                item["call_round_group"] = self._slice(
                    self.tensors["call_round_group"], call_offsets, index
                )
                item["result_round_context"] = self._slice(
                    self.tensors["result_round_context"], result_offsets, index
                )
                item["result_round_context_mask"] = self._slice(
                    self.tensors["result_round_context_mask"], result_offsets, index
                )
                item["result_round_group"] = self._slice(
                    self.tensors["result_round_group"], result_offsets, index
                )
        else:
            raise ValueError(f"Unknown attention variant: {self.variant}")
        return item


def pad_sequences(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences or any(sequence.ndim != 2 for sequence in sequences):
        raise ValueError("Expected a non-empty list of [steps,hidden] tensors")
    if any(sequence.shape[0] == 0 for sequence in sequences):
        raise ValueError("Attention sequences must contain at least one step")
    batch = len(sequences)
    max_steps = max(int(sequence.shape[0]) for sequence in sequences)
    hidden_size = int(sequences[0].shape[1])
    output = sequences[0].new_zeros((batch, max_steps, hidden_size))
    mask = torch.zeros((batch, max_steps), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = int(sequence.shape[0])
        output[index, :length] = sequence
        mask[index, :length] = True
    return output, mask


def pad_boolean_sequences(
    sequences: list[torch.Tensor], max_steps: int
) -> torch.Tensor:
    output = torch.zeros((len(sequences), max_steps), dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        length = int(sequence.numel())
        output[index, :length] = sequence.to(torch.bool)
    return output


def pad_integer_sequences(
    sequences: list[torch.Tensor], max_steps: int, fill_value: int = -1
) -> torch.Tensor:
    output = torch.full(
        (len(sequences), max_steps), fill_value, dtype=torch.int64
    )
    for index, sequence in enumerate(sequences):
        length = int(sequence.numel())
        output[index, :length] = sequence.to(torch.int64)
    return output


def collate_attention(items: list[dict[str, Any]], variant: str) -> dict[str, torch.Tensor]:
    batch: dict[str, torch.Tensor] = {
        "indices": torch.tensor([int(item["index"]) for item in items], dtype=torch.int64),
        "readout": torch.stack([item["readout"] for item in items]),
        "labels": torch.stack([item["label"] for item in items]).to(torch.int64),
    }
    if variant in CALL_VARIANTS or variant in RESULT_VARIANTS:
        steps, mask = pad_sequences([item["steps"] for item in items])
        batch["steps"] = steps
        batch["step_mask"] = mask
    elif variant in ROUND_VARIANTS:
        decisions, decision_mask = pad_sequences(
            [item["decisions"] for item in items]
        )
        round_results, _ = pad_sequences(
            [item["round_results"] for item in items]
        )
        batch["decisions"] = decisions
        batch["decision_mask"] = decision_mask
        batch["round_results"] = round_results
        batch["round_result_mask"] = pad_boolean_sequences(
            [item["round_result_mask"] for item in items], decisions.shape[1]
        )
    elif variant in INDIVIDUAL_RCT_VARIANTS:
        calls, call_mask = pad_sequences([item["calls"] for item in items])
        results, result_mask = pad_sequences([item["results"] for item in items])
        batch["calls"] = calls
        batch["call_mask"] = call_mask
        batch["results"] = results
        batch["result_mask"] = result_mask
        batch["result_to_call"] = pad_integer_sequences(
            [item["result_to_call"] for item in items], results.shape[1]
        )
        if variant in ROUND_TOOL_RCT_VARIANTS:
            call_round_context, _ = pad_sequences(
                [item["call_round_context"] for item in items]
            )
            result_round_context, _ = pad_sequences(
                [item["result_round_context"] for item in items]
            )
            batch["call_round_context"] = call_round_context
            batch["call_round_context_mask"] = pad_boolean_sequences(
                [item["call_round_context_mask"] for item in items], calls.shape[1]
            )
            batch["call_round_group"] = pad_integer_sequences(
                [item["call_round_group"] for item in items], calls.shape[1]
            )
            batch["result_round_context"] = result_round_context
            batch["result_round_context_mask"] = pad_boolean_sequences(
                [item["result_round_context_mask"] for item in items],
                results.shape[1],
            )
            batch["result_round_group"] = pad_integer_sequences(
                [item["result_round_group"] for item in items], results.shape[1]
            )
    else:
        raise ValueError(f"Unknown attention variant: {variant}")
    return batch

