from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from guard_training.common.feature_loading import read_jsonl
from guard_training.common.offline_dataset import load_offline_config
from guard_training.tool_step_mean_pool.build_offline_datasets import (
    FeatureLocation,
    atomic_save_tensors,
    attach_feature_locations,
    build_feature_index,
    discover_offline_splits,
    file_sha256,
    resolve_feature_roots,
    summary,
    validate_feature_roots,
    write_json,
    write_jsonl,
)


CALL_KIND = "tool_call_end"
RESULT_KIND = "tool_result_end"
DECISION_KIND = "assistant_decision_end"
ROUND_KIND = "tool_round_end"


@dataclass
class SampleFeatures:
    metadata: dict[str, Any]
    label: int
    readout: torch.Tensor
    calls: torch.Tensor
    results: torch.Tensor
    decisions: torch.Tensor
    round_results: torch.Tensor
    round_result_mask: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize final-layer variable-length call/result/round features "
            "for readout-conditioned attention training."
        )
    )
    parser.add_argument("--input-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, action="append", default=None)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser.parse_args()


def resolve_layer(layer: int, count: int) -> int:
    resolved = layer if layer >= 0 else count + layer
    if not 0 <= resolved < count:
        raise IndexError(f"Layer {layer} is invalid for {count} saved layers")
    return resolved


def make_offsets(sequences: list[torch.Tensor]) -> torch.Tensor:
    values = [0]
    for sequence in sequences:
        values.append(values[-1] + int(sequence.shape[0]))
    return torch.tensor(values, dtype=torch.int64)


def concatenate_sequences(
    sequences: list[torch.Tensor], hidden_size: int, dtype: torch.dtype
) -> torch.Tensor:
    if not sequences or sum(int(value.shape[0]) for value in sequences) == 0:
        return torch.empty((0, hidden_size), dtype=dtype)
    return torch.cat(sequences, dim=0).to(dtype)


def pair_rounds(
    steps: list[dict[str, Any]], step_vectors: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]], int]:
    decision_indices = [
        index for index, step in enumerate(steps) if step.get("kind") == DECISION_KIND
    ]
    if not decision_indices:
        hidden = int(step_vectors.shape[1])
        return (
            step_vectors.new_empty((0, hidden)),
            step_vectors.new_empty((0, hidden)),
            torch.empty(0, dtype=torch.bool),
            [],
            sum(step.get("kind") == ROUND_KIND for step in steps),
        )

    pairs: list[dict[str, Any]] = []
    by_round: dict[int, int] = {}
    for decision_position, step_index in enumerate(decision_indices):
        round_index = steps[step_index].get("round_index")
        pair = {
            "decision_position": decision_position,
            "decision_step_index": step_index,
            "decision_message_index": steps[step_index].get("message_index"),
            "round_index": round_index,
            "result_step_index": None,
            "result_message_index": None,
        }
        pairs.append(pair)
        if round_index is not None:
            by_round[int(round_index)] = decision_position

    unpaired_results = 0
    for result_step_index, step in enumerate(steps):
        if step.get("kind") != ROUND_KIND:
            continue
        round_index = step.get("round_index")
        target: int | None = None
        if round_index is not None:
            candidate = by_round.get(int(round_index))
            if candidate is not None and pairs[candidate]["result_step_index"] is None:
                target = candidate
        if target is None:
            # Fallback for traces whose tool messages omit tool_call_id/round_index:
            # attach the result to the latest preceding unmatched decision.
            preceding = [
                index
                for index, pair in enumerate(pairs)
                if pair["decision_step_index"] < result_step_index
                and pair["result_step_index"] is None
            ]
            if preceding:
                target = preceding[-1]
        if target is None:
            unpaired_results += 1
            continue
        pairs[target]["result_step_index"] = result_step_index
        pairs[target]["result_message_index"] = step.get("message_index")

    decisions = step_vectors[decision_indices]
    round_results = torch.zeros_like(decisions)
    result_mask = torch.zeros(len(pairs), dtype=torch.bool)
    for position, pair in enumerate(pairs):
        result_step_index = pair["result_step_index"]
        if result_step_index is not None:
            round_results[position] = step_vectors[int(result_step_index)]
            result_mask[position] = True
    return decisions, round_results, result_mask, pairs, unpaired_results


def materialize_samples(
    selected: list[tuple[dict[str, Any], FeatureLocation]], layer: int
) -> tuple[list[SampleFeatures], dict[str, int]]:
    by_shard: dict[
        tuple[Path, int], list[tuple[int, dict[str, Any], FeatureLocation]]
    ] = defaultdict(list)
    for output_index, (offline_row, location) in enumerate(selected):
        by_shard[(location.root, int(location.metadata["shard"]))].append(
            (output_index, offline_row, location)
        )
    output: list[SampleFeatures | None] = [None] * len(selected)
    diagnostics = Counter()

    for (root, shard), bucket in sorted(by_shard.items(), key=lambda item: str(item[0])):
        tensor_path = root / "features" / f"shard_{shard:05d}.safetensors"
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            readout_tensor = handle.get_tensor("readout_last")
            step_tensor = handle.get_tensor("step_hidden")
        readout_layer = resolve_layer(layer, int(readout_tensor.shape[1]))
        step_layer = resolve_layer(layer, int(step_tensor.shape[1]))
        if readout_tensor.shape[2] != step_tensor.shape[2]:
            raise ValueError(f"Hidden-size mismatch in {tensor_path}")

        for output_index, offline_row, location in bucket:
            metadata = location.metadata
            steps = list(metadata.get("steps", []))
            start = int(metadata["step_start"])
            end = int(metadata["step_end"])
            step_vectors = step_tensor[start:end, step_layer].to(torch.float32)
            if step_vectors.shape[0] != len(steps):
                raise ValueError(
                    f"Step metadata/tensor mismatch for {offline_row['sample_id']}"
                )
            call_indices = [
                index for index, step in enumerate(steps) if step.get("kind") == CALL_KIND
            ]
            result_indices = [
                index for index, step in enumerate(steps) if step.get("kind") == RESULT_KIND
            ]
            if not call_indices or not result_indices:
                raise AssertionError("Ineligible sample reached attention materialization")
            decisions, round_results, round_mask, pairs, unpaired = pair_rounds(
                steps, step_vectors
            )
            if decisions.shape[0] == 0:
                raise ValueError(
                    f"Eligible sample has no assistant_decision_end: {offline_row['sample_id']}"
                )
            if not bool(round_mask.any()):
                diagnostics["samples_without_paired_round_result"] += 1
            diagnostics["unpaired_tool_round_end"] += unpaired
            diagnostics["unmatched_decision_rounds"] += int((~round_mask).sum().item())

            enriched = dict(offline_row)
            enriched.update(
                {
                    "attention_counts": {
                        "tool_calls": len(call_indices),
                        "tool_results": len(result_indices),
                        "assistant_decisions": len(pairs),
                        "paired_tool_rounds": int(round_mask.sum().item()),
                        "unmatched_decisions": int((~round_mask).sum().item()),
                        "unpaired_tool_round_ends": unpaired,
                    },
                    "round_pairs": pairs,
                }
            )
            output[output_index] = SampleFeatures(
                metadata=enriched,
                label=int(offline_row["label"]),
                readout=readout_tensor[
                    int(metadata["row"]), readout_layer
                ].to(torch.float32),
                calls=step_vectors[call_indices],
                results=step_vectors[result_indices],
                decisions=decisions,
                round_results=round_results,
                round_result_mask=round_mask,
            )

    if any(sample is None for sample in output):
        raise AssertionError("Failed to materialize every selected attention sample")
    return [sample for sample in output if sample is not None], dict(diagnostics)


def save_split(
    output_dir: Path,
    split: str,
    samples: list[SampleFeatures],
    target_dtype: torch.dtype,
) -> dict[str, Any]:
    if not samples:
        raise ValueError(f"No attention samples for split={split}")
    hidden_size = int(samples[0].readout.numel())
    tensors = {
        "readout": torch.stack([sample.readout for sample in samples]).to(target_dtype),
        "labels": torch.tensor([sample.label for sample in samples], dtype=torch.int64),
        "call_hidden": concatenate_sequences(
            [sample.calls for sample in samples], hidden_size, target_dtype
        ),
        "call_offsets": make_offsets([sample.calls for sample in samples]),
        "result_hidden": concatenate_sequences(
            [sample.results for sample in samples], hidden_size, target_dtype
        ),
        "result_offsets": make_offsets([sample.results for sample in samples]),
        "decision_hidden": concatenate_sequences(
            [sample.decisions for sample in samples], hidden_size, target_dtype
        ),
        "round_result_hidden": concatenate_sequences(
            [sample.round_results for sample in samples], hidden_size, target_dtype
        ),
        "round_result_mask": torch.cat(
            [sample.round_result_mask for sample in samples], dim=0
        ),
        "round_offsets": make_offsets([sample.decisions for sample in samples]),
    }
    tensor_path = output_dir / f"{split}.safetensors"
    metadata_path = output_dir / f"{split}.metadata.jsonl"
    atomic_save_tensors(tensors, tensor_path)
    write_jsonl(
        metadata_path,
        ({**sample.metadata, "compact_row": index} for index, sample in enumerate(samples)),
    )
    labels = tensors["labels"]
    return {
        "samples": len(samples),
        "positives": int(labels.sum().item()),
        "negatives": int(labels.numel() - labels.sum().item()),
        "tool_calls": int(tensors["call_hidden"].shape[0]),
        "tool_results": int(tensors["result_hidden"].shape[0]),
        "assistant_decisions": int(tensors["decision_hidden"].shape[0]),
        "paired_round_results": int(tensors["round_result_mask"].sum().item()),
        "tensor_file": tensor_path.name,
        "tensor_bytes": tensor_path.stat().st_size,
        "tensor_sha256": file_sha256(tensor_path),
        "metadata_file": metadata_path.name,
        "metadata_sha256": file_sha256(metadata_path),
    }


def build_dataset(
    input_dataset: Path,
    output_dir: Path,
    feature_roots: list[Path] | None,
    layer: int,
    storage_dtype_name: str,
) -> dict[str, Any]:
    input_dataset = input_dataset.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    input_config = load_offline_config(input_dataset)
    requested_roots = feature_roots
    if requested_roots is None and input_config.get("source_feature_roots"):
        requested_roots = [Path(value) for value in input_config["source_feature_roots"]]
    roots = resolve_feature_roots(input_config, requested_roots)
    compatibility = validate_feature_roots(roots)
    feature_index = build_feature_index(roots)
    splits = discover_offline_splits(input_dataset)
    selected_all: list[tuple[dict[str, Any], FeatureLocation]] = []
    split_ranges: dict[str, tuple[int, int]] = {}
    source_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    all_excluded: list[dict[str, Any]] = []

    for split in splits:
        source_rows = read_jsonl(input_dataset / f"{split}.metadata.jsonl")
        source_rows_by_split[split] = source_rows
        selected, excluded = attach_feature_locations(source_rows, feature_index)
        start = len(selected_all)
        selected_all.extend(
            ({**row, "offline_split": split}, location)
            for row, location in selected
        )
        split_ranges[split] = (start, len(selected_all))
        all_excluded.extend({**row, "offline_split": split} for row in excluded)

    samples_all, pairing_diagnostics = materialize_samples(selected_all, layer)
    target_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[storage_dtype_name]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent))
    try:
        split_configs: dict[str, Any] = {}
        split_report: dict[str, Any] = {}
        for split in splits:
            start, end = split_ranges[split]
            samples = samples_all[start:end]
            split_configs[split] = save_split(staging, split, samples, target_dtype)
            output_rows = [sample.metadata for sample in samples]
            excluded = [row for row in all_excluded if row["offline_split"] == split]
            split_report[split] = {
                "before_filter": summary(source_rows_by_split[split]),
                "after_filter": summary(output_rows),
                "excluded": summary(excluded),
                "exclusion_reasons": dict(
                    sorted(Counter(row["exclusion_reason"] for row in excluded).items())
                ),
            }
        write_jsonl(staging / "excluded_by_eligibility.jsonl", all_excluded)
        config = {
            "schema_version": "sd_guard.tool_step_attention_dataset.v1",
            "source_offline_dataset": str(input_dataset),
            "source_offline_dataset_config_sha256": file_sha256(
                input_dataset / "dataset_config.json"
            ),
            "source_feature_roots": [str(root) for root in roots],
            "feature_compatibility": compatibility,
            "layer": layer,
            "hidden_size": int(compatibility["hidden_size"]),
            "storage_dtype": storage_dtype_name,
            "eligibility_rule": "tool_call_count > 0 and tool_result_count > 0",
            "features": {
                "readout": "readout_last",
                "calls": CALL_KIND,
                "results": RESULT_KIND,
                "decisions": DECISION_KIND,
                "round_results": ROUND_KIND,
            },
            "round_pairing": {
                "primary_key": "round_index",
                "fallback": "latest preceding unmatched assistant_decision_end",
                "unmatched_decision_result": "zero vector plus false result mask",
            },
            "pairing_diagnostics": pairing_diagnostics,
            "splits": split_configs,
        }
        write_json(staging / "dataset_config.json", config)
        write_json(
            staging / "build_report.json",
            {
                "schema_version": "sd_guard.tool_step_attention_build.v1",
                "splits": split_report,
                "pairing_diagnostics": pairing_diagnostics,
                "excluded_samples": len(all_excluded),
            },
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return config


def main() -> None:
    args = parse_args()
    config = build_dataset(
        input_dataset=args.input_dataset,
        output_dir=args.output_dir,
        feature_roots=args.feature_root,
        layer=args.layer,
        storage_dtype_name=args.storage_dtype,
    )
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
