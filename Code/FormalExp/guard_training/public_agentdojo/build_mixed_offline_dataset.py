from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from guard_training.common.feature_store import iter_feature_metadata
from guard_training.common.splitting import SPLITS, find_group_stratified_split


AUXILIARY_SPLITS = ("public_validation", "public_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact nothink+public offline dataset with one frozen "
            "task-pair assignment. Public traces augment train only; public "
            "validation/test pairs are saved as auxiliary evaluation splits."
        )
    )
    parser.add_argument("--local-feature-root", type=Path, required=True)
    parser.add_argument("--public-feature-root", type=Path, required=True)
    parser.add_argument(
        "--local-manifest",
        type=Path,
        default=None,
        help=(
            "Existing nothink split manifest to freeze. If omitted, "
            "<local-feature-root>/split_manifest.jsonl is used when present; "
            "otherwise a new local-only group split is generated."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature", default="trace_last")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument(
        "--unmapped-public",
        choices=("exclude", "train"),
        default="exclude",
        help="What to do with a public task pair absent from the local master split.",
    )
    parser.add_argument("--local-source-name", default="qwen3_8b_nothink")
    parser.add_argument("--public-source-name", default="agentdojo_public_v1.2.1")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_save_tensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file({key: value.contiguous() for key, value in tensors.items()}, str(temporary))
    os.replace(temporary, path)


def validate_unique_sample_ids(rows: Iterable[dict[str, Any]], source_name: str) -> None:
    sample_ids = [str(row["sample_id"]) for row in rows]
    duplicates = [sample_id for sample_id, count in Counter(sample_ids).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"Duplicate sample IDs in {source_name}: {duplicates[:5]}"
        )


def validate_feature_configs(local_root: Path, public_root: Path) -> dict[str, Any]:
    local_path = local_root / "feature_config.json"
    public_path = public_root / "feature_config.json"
    local = read_json(local_path)
    public = read_json(public_path)
    keys = (
        "schema_version",
        "serialization_version",
        "model_config_sha256",
        "tokenizer_class",
        "tokenizer_chat_template_sha256",
        "num_decoder_layers",
        "saved_layer_count",
        "hidden_size",
        "max_length",
        "storage_dtype",
        "readout_prompt",
        "global_features",
    )
    mismatches = {
        key: {"local": local.get(key), "public": public.get(key)}
        for key in keys
        if local.get(key) != public.get(key)
    }
    if mismatches:
        raise ValueError(
            "Local and public feature stores are incompatible:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2, sort_keys=True)
        )
    return {
        "checked_keys": list(keys),
        "local_config": str(local_path.resolve()),
        "local_config_sha256": file_sha256(local_path),
        "public_config": str(public_path.resolve()),
        "public_config_sha256": file_sha256(public_path),
    }


def group_assignment_from_manifest(path: Path) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for row in read_jsonl(path):
        group_id = str(row["group_id"])
        split = str(row["split"])
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split!r} in {path}")
        previous = assignment.setdefault(group_id, split)
        if previous != split:
            raise ValueError(f"Group crosses splits in {path}: {group_id}")
    if not assignment:
        raise ValueError(f"Manifest contains no task groups: {path}")
    return assignment


def choose_assignment(
    local_rows: list[dict[str, Any]],
    local_feature_root: Path,
    requested_manifest: Path | None,
    ratios: tuple[float, float, float],
    seed: int,
    trials: int,
) -> tuple[dict[str, str], Path | None, dict[str, Any]]:
    manifest = requested_manifest
    if manifest is None:
        candidate = local_feature_root / "split_manifest.jsonl"
        if candidate.exists():
            manifest = candidate
    if manifest is not None:
        manifest = manifest.resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"Local split manifest does not exist: {manifest}")
        assignment = group_assignment_from_manifest(manifest)
        missing = sorted({str(row["group_id"]) for row in local_rows} - set(assignment))
        if missing:
            raise ValueError(
                f"Local manifest misses {len(missing)} feature groups; first={missing[:3]}"
            )
        return assignment, manifest, {
            "mode": "frozen_local_manifest",
            "manifest": str(manifest),
            "manifest_sha256": file_sha256(manifest),
        }

    result = find_group_stratified_split(
        local_rows,
        ratios=ratios,
        seed=seed,
        trials=trials,
    )
    return result.group_to_split, None, {
        "mode": "generated_from_local_nothink_only",
        "score": result.score,
        "winning_trial": result.trial,
    }


def with_assignment(
    rows: Iterable[dict[str, Any]],
    assignment: dict[str, str],
    source_name: str,
    unmapped: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    excluded = []
    for original in rows:
        row = dict(original)
        group_id = str(row["group_id"])
        split = assignment.get(group_id)
        if split is None:
            if unmapped == "exclude":
                row["exclusion_reason"] = "group_absent_from_local_master_split"
                excluded.append(row)
                continue
            split = "train"
            row["assignment_note"] = "public_only_group_forced_to_train"
        row["split"] = split
        row["data_source"] = source_name
        selected.append(row)
    return selected, excluded


def deduplicate_serialized(
    local_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    by_hash: dict[str, dict[str, Any]] = {}
    # Local rows win an exact duplicate tie because local nothink is the target domain.
    for row in [*local_rows, *public_rows]:
        serialized_hash = str(row.get("serialized_sha256") or "")
        if not serialized_hash:
            kept.append(row)
            continue
        previous = by_hash.get(serialized_hash)
        if previous is None:
            by_hash[serialized_hash] = row
            kept.append(row)
            continue
        if int(previous["label"]) != int(row["label"]):
            raise ValueError(
                "Identical serialized trace has conflicting labels: "
                f"{previous['sample_id']} vs {row['sample_id']}"
            )
        if str(previous["group_id"]) != str(row["group_id"]):
            raise ValueError(
                "Identical serialized trace belongs to different task groups: "
                f"{previous['sample_id']} vs {row['sample_id']}"
            )
        if str(previous["split"]) != str(row["split"]):
            raise ValueError(
                "Identical serialized trace crosses splits: "
                f"{previous['sample_id']} vs {row['sample_id']}"
            )
        duplicate = dict(row)
        duplicate["exclusion_reason"] = (
            f"exact_serialized_duplicate_of:{previous['sample_id']}"
        )
        removed.append(duplicate)
    local_source = str(local_rows[0]["data_source"]) if local_rows else ""
    return (
        [row for row in kept if row.get("data_source") == local_source],
        [row for row in kept if row.get("data_source") != local_source],
        removed,
    )


def validate_group_isolation(
    local_rows: list[dict[str, Any]], public_rows: list[dict[str, Any]]
) -> None:
    groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in [*local_rows, *public_rows]:
        groups[str(row["split"])].add(str(row["group_id"]))
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(
                    f"Task groups cross {left}/{right}: {sorted(overlap)[:3]}"
                )


def load_selected_features(
    feature_root: Path,
    rows: list[dict[str, Any]],
    feature_name: str,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if not rows:
        raise ValueError(f"No rows selected from {feature_root}")
    by_shard: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[int(row["shard"])].append(row)

    matrices: list[torch.Tensor] = []
    labels: list[int] = []
    ordered_rows: list[dict[str, Any]] = []
    feature_dir = feature_root / "features"
    for shard_index in sorted(by_shard):
        tensor_path = feature_dir / f"shard_{shard_index:05d}.safetensors"
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            if feature_name not in handle.keys():
                raise KeyError(f"Feature {feature_name!r} is absent from {tensor_path}")
            tensor = handle.get_tensor(feature_name)
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected [samples,layers,hidden] for {feature_name}, got {tensor.shape}"
            )
        resolved_layer = layer if layer >= 0 else tensor.shape[1] + layer
        if not 0 <= resolved_layer < tensor.shape[1]:
            raise IndexError(f"Layer {layer} is invalid for {tensor.shape[1]} saved layers")
        for row in sorted(by_shard[shard_index], key=lambda item: int(item["row"])):
            matrices.append(tensor[int(row["row"]), resolved_layer].to(torch.float32))
            labels.append(int(row["label"]))
            ordered_rows.append(row)
    return torch.stack(matrices), torch.tensor(labels, dtype=torch.long), ordered_rows


def materialize_split(
    output_dir: Path,
    split_name: str,
    parts: list[tuple[Path, list[dict[str, Any]]]],
    feature_name: str,
    layer: int,
    target_dtype: torch.dtype,
) -> dict[str, Any]:
    matrices = []
    labels = []
    metadata = []
    for feature_root, rows in parts:
        if not rows:
            continue
        matrix, part_labels, part_rows = load_selected_features(
            feature_root, rows, feature_name, layer
        )
        matrices.append(matrix)
        labels.append(part_labels)
        metadata.extend(part_rows)
    if not matrices:
        raise ValueError(f"No samples available for output split {split_name}")
    features = torch.cat(matrices, dim=0)
    targets = torch.cat(labels, dim=0)

    tensor_path = output_dir / f"{split_name}.safetensors"
    metadata_path = output_dir / f"{split_name}.metadata.jsonl"
    atomic_save_tensors(
        {"features": features.to(target_dtype), "labels": targets.to(torch.int64)},
        tensor_path,
    )
    metadata_bytes = b"".join(
        (
            json.dumps({**row, "compact_row": index}, ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        for index, row in enumerate(metadata)
    )
    atomic_write(metadata_path, metadata_bytes)
    source_counts = Counter(str(row["data_source"]) for row in metadata)
    return {
        "metadata_file": metadata_path.name,
        "metadata_sha256": file_sha256(metadata_path),
        "negatives": int(targets.numel() - targets.sum().item()),
        "positives": int(targets.sum().item()),
        "samples": int(targets.numel()),
        "sources": dict(sorted(source_counts.items())),
        "tensor_bytes": tensor_path.stat().st_size,
        "tensor_file": tensor_path.name,
        "tensor_sha256": file_sha256(tensor_path),
    }


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "attacks": dict(sorted(Counter(str(row["attack_type"]) for row in rows).items())),
        "groups": len({str(row["group_id"]) for row in rows}),
        "labels": dict(sorted(Counter(int(row["label"]) for row in rows).items())),
        "samples": len(rows),
        "sources": dict(sorted(Counter(str(row["data_source"]) for row in rows).items())),
        "suites": dict(sorted(Counter(str(row["suite_name"]) for row in rows).items())),
    }


def main() -> None:
    args = parse_args()
    if args.local_source_name == args.public_source_name:
        raise ValueError("Local and public source names must differ")
    local_root = args.local_feature_root.resolve()
    public_root = args.public_feature_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == local_root or output_dir == public_root:
        raise ValueError("Output directory must differ from both input feature roots")
    output_dir.mkdir(parents=True, exist_ok=True)

    compatibility = validate_feature_configs(local_root, public_root)
    local_metadata = list(iter_feature_metadata(local_root / "features"))
    public_metadata = list(iter_feature_metadata(public_root / "features"))
    if not local_metadata or not public_metadata:
        raise ValueError("Both local and public feature stores must contain metadata rows")
    validate_unique_sample_ids(local_metadata, args.local_source_name)
    validate_unique_sample_ids(public_metadata, args.public_source_name)
    sample_id_overlap = {
        str(row["sample_id"]) for row in local_metadata
    } & {str(row["sample_id"]) for row in public_metadata}
    if sample_id_overlap:
        raise ValueError(
            "Sample IDs overlap across local/public feature stores: "
            f"{sorted(sample_id_overlap)[:5]}"
        )

    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    assignment, local_manifest, assignment_provenance = choose_assignment(
        local_metadata,
        local_root,
        args.local_manifest,
        ratios,
        args.seed,
        args.trials,
    )
    local_rows, local_excluded = with_assignment(
        local_metadata, assignment, args.local_source_name, "exclude"
    )
    if local_excluded:
        raise AssertionError("Local rows unexpectedly fell outside the local assignment")
    public_rows, public_unmapped = with_assignment(
        public_metadata, assignment, args.public_source_name, args.unmapped_public
    )
    local_rows, public_rows, exact_duplicates = deduplicate_serialized(
        local_rows, public_rows
    )
    validate_group_isolation(local_rows, public_rows)

    local_by_split = {
        split: [row for row in local_rows if row["split"] == split] for split in SPLITS
    }
    public_by_split = {
        split: [row for row in public_rows if row["split"] == split] for split in SPLITS
    }
    target_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.storage_dtype]

    output_splits: dict[str, Any] = {}
    output_splits["train"] = materialize_split(
        output_dir,
        "train",
        [
            (local_root, local_by_split["train"]),
            (public_root, public_by_split["train"]),
        ],
        args.feature,
        args.layer,
        target_dtype,
    )
    output_splits["validation"] = materialize_split(
        output_dir,
        "validation",
        [(local_root, local_by_split["validation"])],
        args.feature,
        args.layer,
        target_dtype,
    )
    output_splits["test"] = materialize_split(
        output_dir,
        "test",
        [(local_root, local_by_split["test"])],
        args.feature,
        args.layer,
        target_dtype,
    )
    if public_by_split["validation"]:
        output_splits["public_validation"] = materialize_split(
            output_dir,
            "public_validation",
            [(public_root, public_by_split["validation"])],
            args.feature,
            args.layer,
            target_dtype,
        )
    if public_by_split["test"]:
        output_splits["public_test"] = materialize_split(
            output_dir,
            "public_test",
            [(public_root, public_by_split["test"])],
            args.feature,
            args.layer,
            target_dtype,
        )

    combined_manifest_rows = []
    for row in local_rows:
        output = dict(row)
        output["dataset_role"] = f"main_{row['split']}"
        combined_manifest_rows.append(output)
    for row in public_rows:
        output = dict(row)
        output["dataset_role"] = (
            "main_train" if row["split"] == "train" else f"auxiliary_public_{row['split']}"
        )
        combined_manifest_rows.append(output)
    manifest_path = output_dir / "combined_manifest.jsonl"
    manifest_bytes = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        for row in sorted(
            combined_manifest_rows,
            key=lambda row: (str(row["split"]), str(row["group_id"]), str(row["sample_id"])),
        )
    )
    atomic_write(manifest_path, manifest_bytes)

    excluded_path = output_dir / "excluded.jsonl"
    excluded_rows = [*public_unmapped, *exact_duplicates]
    excluded_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in excluded_rows
    )
    atomic_write(excluded_path, excluded_bytes)

    summary = {
        "assignment": assignment_provenance,
        "excluded": {
            "exact_serialized_duplicates": len(exact_duplicates),
            "public_groups_absent_from_local_split": len(
                {str(row["group_id"]) for row in public_unmapped}
            ),
            "public_samples_absent_from_local_split": len(public_unmapped),
        },
        "feature_compatibility": compatibility,
        "local": {
            split: summarize_rows(local_by_split[split]) for split in SPLITS
        },
        "public": {
            split: summarize_rows(public_by_split[split]) for split in SPLITS
        },
        "ratios": dict(zip(SPLITS, ratios, strict=True)),
        "schema_version": "sd_guard.mixed_task_pair_split.v1",
    }
    summary_path = output_dir / "split_summary.json"
    atomic_write(
        summary_path,
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )

    input_dim = int(read_json(local_root / "feature_config.json")["hidden_size"])
    config = {
        "feature": args.feature,
        "feature_compatibility": compatibility,
        "input_dim": input_dim,
        "layer": args.layer,
        "local_feature_root": str(local_root),
        "local_manifest": str(local_manifest) if local_manifest is not None else None,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "policy": {
            "group_key": ["suite_name", "user_task_id", "injection_task_id"],
            "main_validation_test_source": args.local_source_name,
            "public_train_pairs_only": True,
            "public_validation_test_saved_as_auxiliary": True,
        },
        "public_feature_root": str(public_root),
        "schema_version": "sd_guard.compact_hidden_dataset.mixed.v1",
        "source_split_summary": str(summary_path),
        "splits": output_splits,
        "storage_dtype": args.storage_dtype,
    }
    config_path = output_dir / "dataset_config.json"
    atomic_write(
        config_path,
        (json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

