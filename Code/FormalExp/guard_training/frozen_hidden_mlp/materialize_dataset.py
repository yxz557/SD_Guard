from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from guard_training.common.feature_loading import load_feature_matrix


SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize one feature/layer into compact train/validation/test files."
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature", default="trace_last")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--storage-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.storage_dtype]
    split_summary: dict[str, Any] = {}
    input_dim: int | None = None

    for split in SPLITS:
        features, labels, rows = load_feature_matrix(
            args.feature_root,
            args.manifest,
            split,
            args.feature,
            args.layer,
        )
        if input_dim is None:
            input_dim = int(features.shape[1])
        elif int(features.shape[1]) != input_dim:
            raise ValueError("Feature dimensions differ across splits")

        tensor_path = args.output_dir / f"{split}.safetensors"
        metadata_path = args.output_dir / f"{split}.metadata.jsonl"
        atomic_save_tensors(
            {
                "features": features.to(target_dtype),
                "labels": labels.to(torch.int64),
            },
            tensor_path,
        )
        metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
        with metadata_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(rows):
                output = dict(row)
                output["compact_row"] = index
                handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(metadata_tmp, metadata_path)
        split_summary[split] = {
            "samples": int(labels.numel()),
            "positives": int(labels.sum().item()),
            "negatives": int(labels.numel() - labels.sum().item()),
            "tensor_file": tensor_path.name,
            "tensor_sha256": file_sha256(tensor_path),
            "tensor_bytes": tensor_path.stat().st_size,
            "metadata_file": metadata_path.name,
            "metadata_sha256": file_sha256(metadata_path),
        }

    manifest_sha256 = file_sha256(args.manifest)
    config = {
        "schema_version": "sd_guard.compact_hidden_dataset.v1",
        "source_feature_root": str(args.feature_root.resolve()),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "feature": args.feature,
        "layer": args.layer,
        "input_dim": input_dim,
        "storage_dtype": args.storage_dtype,
        "splits": split_summary,
    }
    (args.output_dir / "dataset_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

