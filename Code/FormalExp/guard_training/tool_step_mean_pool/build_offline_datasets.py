from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from guard_training.common.feature_loading import read_jsonl
from guard_training.common.feature_store import iter_feature_metadata
from guard_training.common.offline_dataset import load_offline_config


MAIN_SPLITS = ("train", "validation", "test")
CALL_KIND = "tool_call_end"
RESULT_KIND = "tool_result_end"

VARIANT_COMPONENTS: dict[str, tuple[str, ...]] = {
    "c0": ("call_pool",),
    "t0": ("result_pool",),
    "rc": ("readout_last", "call_pool"),
    "rt": ("readout_last", "result_pool"),
    "rct": ("readout_last", "call_pool", "result_pool"),
}


@dataclass(frozen=True)
class FeatureLocation:
    root: Path
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build C0/T0/RC/RT/RCT compact datasets by mean-pooling all "
            "tool-call and tool-result step-boundary hidden states."
        )
    )
    parser.add_argument("--input-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--feature-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Original Qwen3 feature root. Repeat for local and public roots. "
            "When omitted, roots are read from the input dataset config."
        ),
    )
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--include-r0-common",
        action="store_true",
        help=(
            "Also write a readout-only control after the same eligibility filter. "
            "This is recommended for a strictly matched R0 comparison."
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_tensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(temporary))
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_feature_roots(
    input_config: dict[str, Any], requested: list[Path] | None
) -> list[Path]:
    if requested:
        roots = [path.resolve() for path in requested]
    else:
        configured = [
            input_config.get("local_feature_root"),
            input_config.get("public_feature_root"),
            input_config.get("source_feature_root"),
        ]
        roots = [Path(value).resolve() for value in configured if value]
    roots = list(dict.fromkeys(roots))
    if not roots:
        raise ValueError(
            "No source feature roots were supplied and none were found in dataset_config.json"
        )
    for root in roots:
        if not (root / "features").is_dir():
            raise FileNotFoundError(f"Feature root has no features directory: {root}")
        if not (root / "feature_config.json").is_file():
            raise FileNotFoundError(f"Feature root has no feature_config.json: {root}")
    return roots


def validate_feature_roots(roots: list[Path]) -> dict[str, Any]:
    configs = [
        json.loads((root / "feature_config.json").read_text(encoding="utf-8"))
        for root in roots
    ]
    checked_keys = (
        "schema_version",
        "serialization_version",
        "model_config_sha256",
        "tokenizer_class",
        "tokenizer_chat_template_sha256",
        "saved_layer_count",
        "hidden_size",
        "storage_dtype",
        "readout_prompt",
        "global_features",
        "step_kinds",
    )
    reference = configs[0]
    mismatches: dict[str, Any] = {}
    for root, config in zip(roots[1:], configs[1:], strict=True):
        changed = {
            key: {"reference": reference.get(key), "other": config.get(key)}
            for key in checked_keys
            if reference.get(key) != config.get(key)
        }
        if changed:
            mismatches[str(root)] = changed
    if mismatches:
        raise ValueError(
            "Feature roots are incompatible:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2, sort_keys=True)
        )
    global_features = set(reference.get("global_features") or [])
    if "readout_last" not in global_features:
        raise ValueError("Source feature store does not contain readout_last")
    step_kinds = reference.get("step_kinds") or {}
    for kind in (CALL_KIND, RESULT_KIND):
        if kind not in step_kinds:
            raise ValueError(f"Source feature store does not contain step kind {kind}")
    return {
        "checked_keys": list(checked_keys),
        "hidden_size": int(reference["hidden_size"]),
        "saved_layer_count": int(reference["saved_layer_count"]),
        "roots": [
            {
                "path": str(root),
                "feature_config_sha256": file_sha256(root / "feature_config.json"),
            }
            for root in roots
        ],
    }


def build_feature_index(roots: list[Path]) -> dict[str, FeatureLocation]:
    index: dict[str, FeatureLocation] = {}
    for root in roots:
        for metadata in iter_feature_metadata(root / "features"):
            sample_id = str(metadata["sample_id"])
            if sample_id in index:
                raise ValueError(f"Sample ID appears in multiple feature roots: {sample_id}")
            index[sample_id] = FeatureLocation(root=root, metadata=metadata)
    if not index:
        raise ValueError("Source feature roots contain no samples")
    return index


def discover_offline_splits(input_dataset: Path) -> list[str]:
    suffix = ".metadata.jsonl"
    splits = sorted(
        path.name[: -len(suffix)]
        for path in input_dataset.glob(f"*{suffix}")
    )
    if not splits:
        raise ValueError(f"No *.metadata.jsonl files found under {input_dataset}")
    missing = set(MAIN_SPLITS) - set(splits)
    if missing:
        raise ValueError(f"Input dataset is missing main splits: {sorted(missing)}")
    return [*MAIN_SPLITS, *[split for split in splits if split not in MAIN_SPLITS]]


def count_step_kinds(metadata: dict[str, Any]) -> tuple[int, int]:
    kinds = Counter(str(step.get("kind")) for step in metadata.get("steps", []))
    return int(kinds[CALL_KIND]), int(kinds[RESULT_KIND])


def attach_feature_locations(
    rows: list[dict[str, Any]], index: dict[str, FeatureLocation]
) -> tuple[list[tuple[dict[str, Any], FeatureLocation]], list[dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], FeatureLocation]] = []
    excluded: list[dict[str, Any]] = []
    for offline_row in rows:
        sample_id = str(offline_row["sample_id"])
        location = index.get(sample_id)
        if location is None:
            raise ValueError(f"Offline sample is absent from source feature roots: {sample_id}")
        call_count, result_count = count_step_kinds(location.metadata)
        annotated = dict(offline_row)
        annotated["tool_call_count"] = call_count
        annotated["tool_result_count"] = result_count
        if call_count == 0 or result_count == 0:
            annotated["exclusion_reason"] = (
                "missing_tool_call" if call_count == 0 else "missing_tool_result"
            )
            if call_count == 0 and result_count == 0:
                annotated["exclusion_reason"] = "missing_tool_call_and_result"
            excluded.append(annotated)
            continue
        selected.append((annotated, location))
    return selected, excluded


def _resolve_layer(layer: int, count: int) -> int:
    resolved = layer if layer >= 0 else count + layer
    if not 0 <= resolved < count:
        raise IndexError(f"Layer {layer} is invalid for {count} saved layers")
    return resolved


def load_base_matrices(
    selected: list[tuple[dict[str, Any], FeatureLocation]], layer: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[dict[str, Any]]]:
    if not selected:
        raise ValueError("Eligibility filter removed every sample in a split")
    by_shard: dict[tuple[Path, int], list[tuple[int, dict[str, Any], FeatureLocation]]] = (
        defaultdict(list)
    )
    for output_index, (offline_row, location) in enumerate(selected):
        shard = int(location.metadata["shard"])
        by_shard[(location.root, shard)].append((output_index, offline_row, location))

    output_rows: list[dict[str, Any] | None] = [None] * len(selected)
    labels = torch.empty(len(selected), dtype=torch.int64)
    readout_vectors: list[torch.Tensor | None] = [None] * len(selected)
    call_vectors: list[torch.Tensor | None] = [None] * len(selected)
    result_vectors: list[torch.Tensor | None] = [None] * len(selected)

    for (root, shard), bucket in sorted(by_shard.items(), key=lambda item: str(item[0])):
        tensor_path = root / "features" / f"shard_{shard:05d}.safetensors"
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if "readout_last" not in keys or "step_hidden" not in keys:
                raise KeyError(f"Missing readout_last or step_hidden in {tensor_path}")
            readout = handle.get_tensor("readout_last")
            step_hidden = handle.get_tensor("step_hidden")
        if readout.ndim != 3 or step_hidden.ndim != 3:
            raise ValueError(
                f"Unexpected feature shapes in {tensor_path}: "
                f"readout={tuple(readout.shape)}, steps={tuple(step_hidden.shape)}"
            )
        readout_layer = _resolve_layer(layer, int(readout.shape[1]))
        step_layer = _resolve_layer(layer, int(step_hidden.shape[1]))
        if readout.shape[2] != step_hidden.shape[2]:
            raise ValueError(f"Global and step hidden dimensions differ in {tensor_path}")

        for output_index, offline_row, location in bucket:
            metadata = location.metadata
            row_index = int(metadata["row"])
            start = int(metadata["step_start"])
            end = int(metadata["step_end"])
            steps = metadata.get("steps", [])
            sample_steps = step_hidden[start:end, step_layer].to(torch.float32)
            if sample_steps.shape[0] != len(steps):
                raise ValueError(
                    f"Step metadata/tensor mismatch for sample {offline_row['sample_id']}"
                )
            call_indices = [
                index for index, step in enumerate(steps) if step.get("kind") == CALL_KIND
            ]
            result_indices = [
                index for index, step in enumerate(steps) if step.get("kind") == RESULT_KIND
            ]
            if not call_indices or not result_indices:
                raise AssertionError("Ineligible sample reached mean-pooling stage")
            call_pool = sample_steps[call_indices].mean(dim=0)
            result_pool = sample_steps[result_indices].mean(dim=0)
            readout_last = readout[row_index, readout_layer].to(torch.float32)

            output_rows[output_index] = dict(offline_row)
            labels[output_index] = int(offline_row["label"])
            readout_vectors[output_index] = readout_last
            call_vectors[output_index] = call_pool
            result_vectors[output_index] = result_pool

    if any(row is None for row in output_rows):
        raise AssertionError("Failed to materialize every selected sample")
    base = {
        "readout_last": torch.stack([value for value in readout_vectors if value is not None]),
        "call_pool": torch.stack([value for value in call_vectors if value is not None]),
        "result_pool": torch.stack([value for value in result_vectors if value is not None]),
    }
    return base, labels, [row for row in output_rows if row is not None]


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row.get(key, "<missing>")) for row in rows).items()))

    step_counts = [
        (
            int(row.get("tool_call_count", count_step_kinds(row)[0])),
            int(row.get("tool_result_count", count_step_kinds(row)[1])),
        )
        for row in rows
    ]

    def numeric(values: list[int]) -> dict[str, int | float | None]:
        return {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": (sum(values) / len(values)) if values else None,
            "total": sum(values),
        }

    positives = sum(int(row["label"]) for row in rows)
    return {
        "samples": len(rows),
        "labels": {"0": len(rows) - positives, "1": positives},
        "sources": counts("data_source"),
        "suites": counts("suite_name"),
        "attacks": counts("attack_type"),
        "groups": len({str(row["group_id"]) for row in rows}),
        "tool_calls": numeric([calls for calls, _ in step_counts]),
        "tool_results": numeric([results for _, results in step_counts]),
        "call_result_count_relation": {
            "equal": sum(calls == results for calls, results in step_counts),
            "calls_greater": sum(calls > results for calls, results in step_counts),
            "results_greater": sum(calls < results for calls, results in step_counts),
        },
    }


def variant_matrix(base: dict[str, torch.Tensor], components: tuple[str, ...]) -> torch.Tensor:
    tensors = [base[name] for name in components]
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=1)


def build_offline_datasets(
    input_dataset: Path,
    output_root: Path,
    feature_roots: list[Path] | None,
    layer: int,
    storage_dtype_name: str,
    include_r0_common: bool,
) -> dict[str, Any]:
    input_dataset = input_dataset.resolve()
    output_root = output_root.resolve()
    if not input_dataset.is_dir():
        raise FileNotFoundError(f"Input offline dataset does not exist: {input_dataset}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")

    input_config = load_offline_config(input_dataset)
    roots = resolve_feature_roots(input_config, feature_roots)
    compatibility = validate_feature_roots(roots)
    feature_index = build_feature_index(roots)
    splits = discover_offline_splits(input_dataset)
    variants = dict(VARIANT_COMPONENTS)
    if include_r0_common:
        variants = {"r0_common": ("readout_last",), **variants}
    target_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[storage_dtype_name]

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_root.name + ".tmp-", dir=output_root.parent))
    report: dict[str, Any] = {
        "schema_version": "sd_guard.tool_step_mean_pool_build.v1",
        "input_dataset": str(input_dataset),
        "input_dataset_config_sha256": file_sha256(input_dataset / "dataset_config.json"),
        "feature_compatibility": compatibility,
        "eligibility_rule": "tool_call_count > 0 and tool_result_count > 0",
        "pooling": {
            "call_pool": "arithmetic mean of all tool_call_end vectors",
            "result_pool": "arithmetic mean of all tool_result_end vectors",
            "call_and_result_are_never_averaged_together": True,
            "layer": layer,
        },
        "variants": {name: list(parts) for name, parts in variants.items()},
        "splits": {},
    }
    variant_split_configs: dict[str, dict[str, Any]] = {name: {} for name in variants}
    all_excluded: list[dict[str, Any]] = []
    selected_all: list[tuple[dict[str, Any], FeatureLocation]] = []
    split_ranges: dict[str, tuple[int, int]] = {}
    source_rows_by_split: dict[str, list[dict[str, Any]]] = {}

    try:
        for name in variants:
            (staging / name).mkdir(parents=True)

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

        # Materialize every selected sample in one pass over the feature shards.
        # The source feature store is large, while the three final-layer base
        # matrices fit comfortably in CPU memory.
        base_all, labels_all, output_rows_all = load_base_matrices(selected_all, layer)

        for split in splits:
            start, end = split_ranges[split]
            if start == end:
                raise ValueError(f"Eligibility filter removed every sample in split={split}")
            base = {name: tensor[start:end] for name, tensor in base_all.items()}
            labels = labels_all[start:end]
            output_rows = output_rows_all[start:end]
            source_rows = source_rows_by_split[split]
            excluded = [
                row for row in all_excluded if row["offline_split"] == split
            ]
            report["splits"][split] = {
                "before_filter": summary(source_rows),
                "after_filter": summary(output_rows),
                "excluded": summary(excluded),
                "exclusion_reasons": dict(
                    sorted(Counter(str(row["exclusion_reason"]) for row in excluded).items())
                ),
            }

            for variant_name, components in variants.items():
                variant_dir = staging / variant_name
                features = variant_matrix(base, components).to(target_dtype)
                tensor_path = variant_dir / f"{split}.safetensors"
                metadata_path = variant_dir / f"{split}.metadata.jsonl"
                atomic_save_tensors({"features": features, "labels": labels}, tensor_path)
                write_jsonl(
                    metadata_path,
                    ({**row, "compact_row": index} for index, row in enumerate(output_rows)),
                )
                variant_split_configs[variant_name][split] = {
                    "samples": int(labels.numel()),
                    "positives": int(labels.sum().item()),
                    "negatives": int(labels.numel() - labels.sum().item()),
                    "tensor_file": tensor_path.name,
                    "tensor_sha256": file_sha256(tensor_path),
                    "tensor_bytes": tensor_path.stat().st_size,
                    "metadata_file": metadata_path.name,
                    "metadata_sha256": file_sha256(metadata_path),
                }

        write_jsonl(staging / "excluded_by_eligibility.jsonl", all_excluded)
        report["excluded_samples"] = len(all_excluded)
        report["excluded_manifest"] = "excluded_by_eligibility.jsonl"
        report["excluded_manifest_sha256"] = file_sha256(
            staging / "excluded_by_eligibility.jsonl"
        )

        hidden_size = int(compatibility["hidden_size"])
        for variant_name, components in variants.items():
            config = {
                "schema_version": "sd_guard.compact_hidden_dataset.tool_step_mean.v1",
                "source_offline_dataset": str(input_dataset),
                "source_offline_dataset_config_sha256": report[
                    "input_dataset_config_sha256"
                ],
                "source_feature_roots": [str(root) for root in roots],
                "feature": variant_name,
                "components": list(components),
                "pooling": report["pooling"],
                "eligibility_rule": report["eligibility_rule"],
                "layer": layer,
                "input_dim": hidden_size * len(components),
                "storage_dtype": storage_dtype_name,
                "splits": variant_split_configs[variant_name],
            }
            write_json(staging / variant_name / "dataset_config.json", config)

        write_json(staging / "build_report.json", report)
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def main() -> None:
    args = parse_args()
    report = build_offline_datasets(
        input_dataset=args.input_dataset,
        output_root=args.output_root,
        feature_roots=args.feature_root,
        layer=args.layer,
        storage_dtype_name=args.storage_dtype,
        include_r0_common=args.include_r0_common,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
