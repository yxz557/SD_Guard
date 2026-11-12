from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REQUIRED_MAIN_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an existing compact offline hidden-state dataset without "
            "changing its task-group split assignment."
        )
    )
    parser.add_argument("--input-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suite",
        dest="suites",
        nargs="+",
        required=True,
        help="One or more suite_name values to retain, for example: --suite travel",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def normalize_suites(values: Iterable[str]) -> tuple[str, ...]:
    suites = tuple(sorted({value.strip() for value in values if value.strip()}))
    if not suites:
        raise ValueError("At least one non-empty suite name is required")
    return suites


def select_indices(
    rows: list[dict[str, Any]], suites: tuple[str, ...]
) -> list[int]:
    selected = set(suites)
    return [
        index
        for index, row in enumerate(rows)
        if str(row.get("suite_name", "")) in selected
    ]


def summarize_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    labels = Counter(int(row["label"]) for row in rows)
    unexpected_labels = set(labels) - {0, 1}
    if unexpected_labels:
        raise ValueError(f"Expected binary labels, got {sorted(unexpected_labels)}")
    return {
        "attacks": dict(
            sorted(
                Counter(
                    str(row.get("attack_type", "unknown")) for row in rows
                ).items()
            )
        ),
        "groups": len({str(row.get("group_id", "unknown")) for row in rows}),
        "labels": {str(label): count for label, count in sorted(labels.items())},
        "negatives": labels[0],
        "positives": labels[1],
        "samples": len(rows),
        "sources": dict(
            sorted(
                Counter(
                    str(row.get("data_source", "unknown")) for row in rows
                ).items()
            )
        ),
        "suites": dict(
            sorted(Counter(str(row["suite_name"]) for row in rows).items())
        ),
    }


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def filter_split(
    input_dir: Path,
    output_dir: Path,
    split_name: str,
    suites: tuple[str, ...],
) -> dict[str, Any] | None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    input_tensor = input_dir / f"{split_name}.safetensors"
    input_metadata = input_dir / f"{split_name}.metadata.jsonl"
    if not input_tensor.is_file() or not input_metadata.is_file():
        raise FileNotFoundError(f"Missing compact input split: {split_name}")

    rows = read_jsonl(input_metadata)
    with safe_open(str(input_tensor), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != {"features", "labels"}:
            raise ValueError(
                f"Unexpected tensor keys for split={split_name}: {sorted(keys)}"
            )
        features = handle.get_tensor("features")
        labels = handle.get_tensor("labels").to(torch.long)

    if features.ndim != 2:
        raise ValueError(
            f"Expected features [samples, hidden] for split={split_name}, "
            f"got {tuple(features.shape)}"
        )
    if features.shape[0] != labels.numel() or features.shape[0] != len(rows):
        raise ValueError(f"Feature/label/metadata mismatch for split={split_name}")
    for index, (row, tensor_label) in enumerate(zip(rows, labels.tolist(), strict=True)):
        if int(row["label"]) != int(tensor_label):
            raise ValueError(
                f"Tensor/metadata label mismatch for split={split_name}, row={index}"
            )

    indices = select_indices(rows, suites)
    if not indices:
        if split_name in REQUIRED_MAIN_SPLITS:
            raise ValueError(
                f"Suite filter {list(suites)} selected no rows from required split "
                f"{split_name}"
            )
        return None

    index_tensor = torch.tensor(indices, dtype=torch.long)
    selected_features = features.index_select(0, index_tensor).contiguous()
    selected_labels = labels.index_select(0, index_tensor).contiguous()
    selected_rows = []
    for compact_row, source_index in enumerate(indices):
        row = dict(rows[source_index])
        row["compact_row"] = compact_row
        selected_rows.append(row)

    output_tensor = output_dir / f"{split_name}.safetensors"
    tensor_temporary = output_tensor.with_name(output_tensor.name + ".tmp")
    save_file(
        {"features": selected_features, "labels": selected_labels},
        str(tensor_temporary),
    )
    os.replace(tensor_temporary, output_tensor)

    output_metadata = output_dir / f"{split_name}.metadata.jsonl"
    metadata_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in selected_rows
    )
    atomic_write(output_metadata, metadata_bytes)

    summary = summarize_rows(selected_rows)
    summary.update(
        {
            "metadata_file": output_metadata.name,
            "metadata_sha256": file_sha256(output_metadata),
            "tensor_bytes": output_tensor.stat().st_size,
            "tensor_file": output_tensor.name,
            "tensor_sha256": file_sha256(output_tensor),
        }
    )
    return summary


def build_filtered_dataset(
    input_dir: Path, output_dir: Path, suites: tuple[str, ...]
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output dataset directories must differ")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {output_dir}"
        )

    input_config_path = input_dir / "dataset_config.json"
    if not input_config_path.is_file():
        raise FileNotFoundError(f"Missing input dataset config: {input_config_path}")
    input_config = json.loads(input_config_path.read_text(encoding="utf-8"))
    configured_splits = input_config.get("splits")
    if not isinstance(configured_splits, dict):
        raise ValueError("Input dataset config does not contain a splits mapping")
    missing_main = set(REQUIRED_MAIN_SPLITS) - set(configured_splits)
    if missing_main:
        raise ValueError(f"Input dataset is missing main splits: {sorted(missing_main)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        output_splits: dict[str, Any] = {}
        for split_name in configured_splits:
            summary = filter_split(input_dir, staging_dir, split_name, suites)
            if summary is not None:
                output_splits[split_name] = summary

        config = {
            "feature": input_config["feature"],
            "filter": {"suite_names": list(suites)},
            "input_dim": int(input_config["input_dim"]),
            "layer": int(input_config["layer"]),
            "parent_config_sha256": file_sha256(input_config_path),
            "parent_dataset": str(input_dir),
            "parent_schema_version": input_config.get("schema_version"),
            "policy": {
                "preserve_parent_split_assignment": True,
                "selection_field": "suite_name",
            },
            "schema_version": "sd_guard.compact_hidden_dataset.filtered.v1",
            "splits": output_splits,
            "storage_dtype": input_config["storage_dtype"],
        }
        config_path = staging_dir / "dataset_config.json"
        atomic_write(
            config_path,
            (json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
        os.replace(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    return config


def main() -> None:
    args = parse_args()
    suites = normalize_suites(args.suites)
    config = build_filtered_dataset(args.input_dataset, args.output_dir, suites)
    print(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
