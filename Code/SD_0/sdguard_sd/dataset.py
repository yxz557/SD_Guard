"""Load the Qwen3GuardTest thinking split and keep only the user question."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PromptSample:
    sample_id: str
    prompt: str
    dataset_index: int
    original_label: str | None
    original_source: str | None
    original_unsafe_type: str | None


def _load_dataset_rows(dataset_path: str | Path, cache_dir: str | Path | None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised on the GPU host
        raise RuntimeError("Install the requirements before loading Qwen3GuardTest") from exc

    path = Path(dataset_path).expanduser()
    kwargs: dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())

    if path.is_file():
        return load_dataset("json", data_files={"thinking": str(path)}, split="thinking", **kwargs)
    for name in ("thinking.jsonl", "thinking.json"):
        candidate = path / name
        if candidate.is_file():
            return load_dataset(
                "json", data_files={"thinking": str(candidate)}, split="thinking", **kwargs
            )
    source = str(path) if path.exists() else str(dataset_path)
    return load_dataset(source, split="thinking", **kwargs)


def _last_user_content(messages: Any) -> str:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("Dataset field 'message' must be a list of chat messages")
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("The last user message has empty or non-text content")
            return content
    raise ValueError("Dataset row has no user message")


def load_prompt_samples(
    dataset_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> list[PromptSample]:
    rows = _load_dataset_rows(dataset_path, cache_dir)
    result: list[PromptSample] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = str(row.get("unique_id", index))
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in dataset: {sample_id!r}")
        seen.add(sample_id)
        result.append(
            PromptSample(
                sample_id=sample_id,
                prompt=_last_user_content(row.get("message")),
                dataset_index=index,
                original_label=None if row.get("label") is None else str(row.get("label")),
                original_source=None if row.get("source") is None else str(row.get("source")),
                original_unsafe_type=(
                    None if row.get("unsafe_type") is None else str(row.get("unsafe_type"))
                ),
            )
        )
    return result


def create_manifest(
    samples: Sequence[PromptSample],
    *,
    dataset_path: str | Path,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(samples):
        raise ValueError(f"Requested {sample_count} samples, but the split has {len(samples)}")
    chosen = random.Random(seed).sample(range(len(samples)), sample_count)
    return {
        "schema_version": 1,
        "dataset_path": str(dataset_path),
        "split": "thinking",
        "seed": seed,
        "sample_count": sample_count,
        "samples": [
            {
                "sample_order": order,
                "dataset_index": samples[index].dataset_index,
                "sample_id": samples[index].sample_id,
            }
            for order, index in enumerate(chosen)
        ],
    }


def select_from_manifest(
    samples: Sequence[PromptSample], manifest: Mapping[str, Any]
) -> list[tuple[int, PromptSample]]:
    by_id = {sample.sample_id: sample for sample in samples}
    selected: list[tuple[int, PromptSample]] = []
    seen: set[str] = set()
    entries = manifest.get("samples")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest has no non-empty 'samples' list")
    for position, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Manifest sample {position} is not an object")
        sample_id = str(entry.get("sample_id"))
        if sample_id in seen:
            raise ValueError(f"Manifest contains duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        try:
            sample = by_id[sample_id]
        except KeyError as exc:
            raise ValueError(f"Manifest sample_id not found in dataset: {sample_id!r}") from exc
        expected_index = int(entry.get("dataset_index", sample.dataset_index))
        if sample.dataset_index != expected_index:
            raise ValueError(
                f"Dataset index changed for {sample_id!r}: "
                f"manifest={expected_index}, current={sample.dataset_index}"
            )
        selected.append((int(entry.get("sample_order", position)), sample))
    selected.sort(key=lambda item: item[0])
    return selected


__all__ = [
    "PromptSample",
    "create_manifest",
    "load_prompt_samples",
    "select_from_manifest",
]

