"""Deterministic sample manifests shared by every evaluation path."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .dataset import GuardSample


MANIFEST_VERSION = 1


def create_sample_manifest(
    samples: Sequence[GuardSample],
    *,
    sample_count: int,
    seed: int,
    dataset_path: str,
) -> dict[str, Any]:
    """Randomly choose samples once and return a reproducible manifest."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(samples):
        raise ValueError(
            f"sample_count={sample_count} exceeds thinking split size {len(samples)}"
        )

    ids = [sample.unique_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("thinking split contains duplicate unique_id values")

    indices = random.Random(seed).sample(range(len(samples)), sample_count)
    return {
        "version": MANIFEST_VERSION,
        "dataset_path": str(dataset_path),
        "split": "thinking",
        "seed": int(seed),
        "dataset_sample_count": len(samples),
        "sample_count": sample_count,
        "samples": [
            {
                "sample_order": order,
                "dataset_index": dataset_index,
                "sample_id": samples[dataset_index].unique_id,
            }
            for order, dataset_index in enumerate(indices)
        ],
    }


def select_samples_from_manifest(
    samples: Sequence[GuardSample], manifest: Mapping[str, Any]
) -> list[tuple[int, int, GuardSample]]:
    """Resolve and validate ``(order, dataset index, sample)`` entries."""

    if int(manifest.get("version", -1)) != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported sample manifest version: {manifest.get('version')!r}"
        )
    if manifest.get("split") != "thinking":
        raise ValueError("Sample manifest must target the thinking split")

    entries = manifest.get("samples")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Sample manifest has no samples")

    selected: list[tuple[int, int, GuardSample]] = []
    seen_indices: set[int] = set()
    seen_ids: set[str] = set()
    for expected_order, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Manifest sample {expected_order} is not an object")
        order = int(entry.get("sample_order", expected_order))
        index = int(entry["dataset_index"])
        expected_id = str(entry["sample_id"])
        if order != expected_order:
            raise ValueError(
                f"Manifest sample_order must be contiguous: {order} != {expected_order}"
            )
        if index < 0 or index >= len(samples):
            raise ValueError(f"Manifest dataset_index {index} is out of range")
        sample = samples[index]
        if sample.unique_id != expected_id:
            raise ValueError(
                "Manifest no longer matches the dataset: "
                f"index {index} expected {expected_id!r}, found {sample.unique_id!r}"
            )
        if index in seen_indices or expected_id in seen_ids:
            raise ValueError("Sample manifest contains duplicate entries")
        seen_indices.add(index)
        seen_ids.add(expected_id)
        selected.append((order, index, sample))

    declared_count = int(manifest.get("sample_count", len(entries)))
    if declared_count != len(selected):
        raise ValueError(
            f"Manifest sample_count={declared_count} but contains {len(selected)} entries"
        )
    return selected


__all__ = [
    "MANIFEST_VERSION",
    "create_sample_manifest",
    "select_samples_from_manifest",
]
