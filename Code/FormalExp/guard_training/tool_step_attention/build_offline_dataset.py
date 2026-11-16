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
    call_round_context: torch.Tensor
    call_round_context_mask: torch.Tensor
    call_round_group: torch.Tensor
    result_round_context: torch.Tensor
    result_round_context_mask: torch.Tensor
    result_round_group: torch.Tensor
    result_to_call: torch.Tensor


@dataclass
class IndividualToolAlignment:
    call_round_context: torch.Tensor
    call_round_context_mask: torch.Tensor
    call_round_group: torch.Tensor
    result_round_context: torch.Tensor
    result_round_context_mask: torch.Tensor
    result_round_group: torch.Tensor
    result_to_call: torch.Tensor
    tool_pairs: list[dict[str, Any]]
    diagnostics: Counter[str]


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


def _round_index(step: dict[str, Any]) -> int | None:
    value = step.get("round_index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def align_individual_tools(
    steps: list[dict[str, Any]],
    step_vectors: torch.Tensor,
    call_indices: list[int],
    result_indices: list[int],
) -> IndividualToolAlignment:
    """Align individual call/result endpoints without replacing their values.

    Exact ``tool_call_id`` matching is preferred. Missing IDs fall back to an
    unused call from the same round (and function when available), then to an
    unused call in trace order. The round endpoint vectors are repeated only as
    scoring contexts; the actual values remain tool_call_end/tool_result_end.
    """

    hidden_size = int(step_vectors.shape[1])
    diagnostics: Counter[str] = Counter()
    decision_by_round: dict[int, int] = {}
    result_round_by_round: dict[int, int] = {}
    result_round_by_order: dict[int, int] = {}
    for step_index, step in enumerate(steps):
        round_index = _round_index(step)
        if round_index is None:
            continue
        if step.get("kind") == DECISION_KIND:
            decision_by_round[round_index] = step_index
        elif step.get("kind") == ROUND_KIND:
            result_round_by_round[round_index] = step_index

    # Public AgentDojo traces may omit tool_call_id on tool messages. Their
    # tool_result_end and tool_round_end boundaries are still serialized in
    # order, but both carry round_index=None. Associate every contiguous result
    # block with its immediately following round endpoint before considering a
    # result unmatched.
    pending_results: list[int] = []
    for step_index, step in enumerate(steps):
        kind = step.get("kind")
        if kind == RESULT_KIND:
            pending_results.append(step_index)
        elif kind == ROUND_KIND:
            for result_step_index in pending_results:
                result_round_by_order[result_step_index] = step_index
            pending_results.clear()
        elif pending_results:
            # A different boundary before tool_round_end means this is not a
            # structurally complete result block; do not attach it to a later
            # unrelated round.
            pending_results.clear()

    call_id_to_position: dict[str, int] = {}
    calls_by_round: dict[int, list[int]] = defaultdict(list)
    for call_position, step_index in enumerate(call_indices):
        step = steps[step_index]
        call_id = str(step.get("tool_call_id") or "")
        if call_id:
            if call_id in call_id_to_position:
                diagnostics["duplicate_tool_call_ids"] += 1
            else:
                call_id_to_position[call_id] = call_position
        round_index = _round_index(step)
        if round_index is not None:
            calls_by_round[round_index].append(call_position)

    call_round_context = step_vectors.new_zeros((len(call_indices), hidden_size))
    call_round_context_mask = torch.zeros(len(call_indices), dtype=torch.bool)
    call_round_group = torch.full((len(call_indices),), -1, dtype=torch.int64)
    for call_position, step_index in enumerate(call_indices):
        round_index = _round_index(steps[step_index])
        context_index = (
            decision_by_round.get(round_index) if round_index is not None else None
        )
        if context_index is None:
            diagnostics["calls_without_round_context"] += 1
            continue
        call_round_context[call_position] = step_vectors[context_index]
        call_round_context_mask[call_position] = True
        call_round_group[call_position] = int(context_index)

    result_round_context = step_vectors.new_zeros((len(result_indices), hidden_size))
    result_round_context_mask = torch.zeros(len(result_indices), dtype=torch.bool)
    result_round_group = torch.full((len(result_indices),), -1, dtype=torch.int64)
    result_to_call = torch.full((len(result_indices),), -1, dtype=torch.int64)
    used_calls: set[int] = set()
    tool_pairs: list[dict[str, Any]] = []

    for result_position, step_index in enumerate(result_indices):
        step = steps[step_index]
        round_index = _round_index(step)
        context_index = (
            result_round_by_round.get(round_index) if round_index is not None else None
        )
        if context_index is not None:
            diagnostics["result_round_context_exact"] += 1
        else:
            context_index = result_round_by_order.get(step_index)
            if context_index is not None:
                diagnostics["result_round_context_order_fallback"] += 1
        if context_index is None:
            diagnostics["results_without_round_context"] += 1
        else:
            result_round_context[result_position] = step_vectors[context_index]
            result_round_context_mask[result_position] = True
            result_round_group[result_position] = int(context_index)

        call_position: int | None = None
        match_method: str | None = None
        call_id = str(step.get("tool_call_id") or "")
        if call_id and call_id in call_id_to_position:
            call_position = call_id_to_position[call_id]
            match_method = "tool_call_id"
            diagnostics["tool_pairs_exact_id"] += 1

        if call_position is None and round_index is not None:
            candidates = [
                value
                for value in calls_by_round.get(round_index, [])
                if value not in used_calls
            ]
            function = str(step.get("function") or "")
            same_function = [
                value
                for value in candidates
                if function
                and str(steps[call_indices[value]].get("function") or "") == function
            ]
            if same_function:
                call_position = same_function[0]
                match_method = "round_function_fallback"
                diagnostics["tool_pairs_round_function_fallback"] += 1
            elif candidates:
                call_position = candidates[0]
                match_method = "round_order_fallback"
                diagnostics["tool_pairs_round_order_fallback"] += 1

        if call_position is None:
            candidates = [
                value for value in range(len(call_indices)) if value not in used_calls
            ]
            if candidates:
                call_position = candidates[0]
                match_method = "trace_order_fallback"
                diagnostics["tool_pairs_trace_order_fallback"] += 1

        if call_position is None:
            diagnostics["unmatched_tool_results"] += 1
        else:
            result_to_call[result_position] = call_position
            used_calls.add(call_position)

        tool_pairs.append(
            {
                "result_position": result_position,
                "result_step_index": step_index,
                "result_message_index": step.get("message_index"),
                "result_tool_call_id": step.get("tool_call_id"),
                "round_index": round_index,
                "call_position": call_position,
                "call_step_index": (
                    call_indices[call_position] if call_position is not None else None
                ),
                "match_method": match_method,
            }
        )

    return IndividualToolAlignment(
        call_round_context=call_round_context,
        call_round_context_mask=call_round_context_mask,
        call_round_group=call_round_group,
        result_round_context=result_round_context,
        result_round_context_mask=result_round_context_mask,
        result_round_group=result_round_group,
        result_to_call=result_to_call,
        tool_pairs=tool_pairs,
        diagnostics=diagnostics,
    )


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
            individual = align_individual_tools(
                steps, step_vectors, call_indices, result_indices
            )
            if decisions.shape[0] == 0:
                raise ValueError(
                    f"Eligible sample has no assistant_decision_end: {offline_row['sample_id']}"
                )
            if not bool(round_mask.any()):
                diagnostics["samples_without_paired_round_result"] += 1
            diagnostics["unpaired_tool_round_end"] += unpaired
            diagnostics["unmatched_decision_rounds"] += int((~round_mask).sum().item())
            diagnostics.update(individual.diagnostics)
            if not bool((individual.result_to_call >= 0).any()):
                diagnostics["samples_without_any_tool_pair"] += 1
            if not bool(individual.call_round_context_mask.any()):
                diagnostics["samples_without_any_call_round_context"] += 1
            if not bool(individual.result_round_context_mask.any()):
                diagnostics["samples_without_any_result_round_context"] += 1

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
                        "matched_individual_results": int(
                            (individual.result_to_call >= 0).sum().item()
                        ),
                        "calls_with_round_context": int(
                            individual.call_round_context_mask.sum().item()
                        ),
                        "results_with_round_context": int(
                            individual.result_round_context_mask.sum().item()
                        ),
                    },
                    "round_pairs": pairs,
                    "individual_tool_pairs": individual.tool_pairs,
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
                call_round_context=individual.call_round_context,
                call_round_context_mask=individual.call_round_context_mask,
                call_round_group=individual.call_round_group,
                result_round_context=individual.result_round_context,
                result_round_context_mask=individual.result_round_context_mask,
                result_round_group=individual.result_round_group,
                result_to_call=individual.result_to_call,
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
        "call_round_context": concatenate_sequences(
            [sample.call_round_context for sample in samples], hidden_size, target_dtype
        ),
        "call_round_context_mask": torch.cat(
            [sample.call_round_context_mask for sample in samples], dim=0
        ),
        "call_round_group": torch.cat(
            [sample.call_round_group for sample in samples], dim=0
        ),
        "result_round_context": concatenate_sequences(
            [sample.result_round_context for sample in samples],
            hidden_size,
            target_dtype,
        ),
        "result_round_context_mask": torch.cat(
            [sample.result_round_context_mask for sample in samples], dim=0
        ),
        "result_round_group": torch.cat(
            [sample.result_round_group for sample in samples], dim=0
        ),
        "result_to_call": torch.cat(
            [sample.result_to_call for sample in samples], dim=0
        ),
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
        "matched_individual_results": int(
            (tensors["result_to_call"] >= 0).sum().item()
        ),
        "calls_with_round_context": int(
            tensors["call_round_context_mask"].sum().item()
        ),
        "results_with_round_context": int(
            tensors["result_round_context_mask"].sum().item()
        ),
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
            "schema_version": "sd_guard.tool_step_attention_dataset.v2.1",
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
                "call_round_context": (
                    "assistant_decision_end repeated for each call in the round"
                ),
                "result_round_context": (
                    "tool_round_end repeated for each result in the round"
                ),
                "result_to_call": "per-result local call position",
                "call_round_group": "per-call local round_index",
                "result_round_group": "per-result local round_index",
            },
            "round_pairing": {
                "primary_key": "round_index",
                "fallback": "latest preceding unmatched assistant_decision_end",
                "unmatched_decision_result": "zero vector plus false result mask",
            },
            "individual_tool_pairing": {
                "primary_key": "tool_call_id",
                "fallbacks": [
                    "same round and function",
                    "same round in trace order",
                    "global trace order",
                ],
                "round_context_role": (
                    "score-only context; pooled values remain individual "
                    "tool_call_end/tool_result_end vectors"
                ),
                "result_round_context_fallback": (
                    "contiguous tool_result_end block followed immediately by "
                    "tool_round_end"
                ),
            },
            "pairing_diagnostics": pairing_diagnostics,
            "splits": split_configs,
        }
        write_json(staging / "dataset_config.json", config)
        write_json(
            staging / "build_report.json",
            {
                "schema_version": "sd_guard.tool_step_attention_build.v2.1",
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
