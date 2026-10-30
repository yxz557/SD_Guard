"""Dataset loading and validation for the public Qwen3GuardTest benchmark.

The module deliberately keeps :mod:`datasets` as an optional, lazy dependency.
This lets the analysis-only modules import the shared data types without loading
the Hugging Face stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


VALID_SPLITS: tuple[str, ...] = ("thinking", "thinking_loc", "response_loc")
LOCATION_SPLITS: frozenset[str] = frozenset({"thinking_loc", "response_loc"})


class DatasetFormatError(ValueError):
    """Raised when a benchmark row does not have the expected public schema."""


def normalize_label(value: Any) -> str:
    """Return the benchmark label as ``"Safe"`` or ``"Unsafe"``.

    The published dataset has used both title-cased and lower-cased spellings
    in surrounding documentation. Normalising at the boundary keeps downstream
    metric code deterministic while still rejecting accidental third classes.
    """

    label = str(value).strip().casefold()
    if label == "safe":
        return "Safe"
    if label == "unsafe":
        return "Unsafe"
    raise DatasetFormatError(
        f"Expected a binary Safe/Unsafe label, received {value!r}."
    )


def _normalise_messages(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DatasetFormatError("'message' must be a sequence of chat messages.")

    messages: list[dict[str, Any]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise DatasetFormatError(f"message[{index}] must be a mapping.")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise DatasetFormatError(
                f"message[{index}] has unsupported role {role!r}."
            )
        if "content" not in message:
            raise DatasetFormatError(f"message[{index}] is missing 'content'.")
        messages.append(dict(message))

    if not messages:
        raise DatasetFormatError("'message' must contain at least one message.")
    if not any(message["role"] == "user" for message in messages):
        raise DatasetFormatError("Conversation does not contain a user turn.")
    if not any(message["role"] == "assistant" for message in messages):
        raise DatasetFormatError("Conversation does not contain an assistant turn.")
    return tuple(messages)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetFormatError(f"{field_name!r} must be an integer or null.") from exc


@dataclass(frozen=True, slots=True)
class GuardSample:
    """A validated row from one Qwen3GuardTest split."""

    split: str
    unique_id: str
    messages: tuple[dict[str, Any], ...]
    label: str
    source: str | None = None
    input_ids: tuple[int, ...] | None = None
    unsafe_type: str | None = None
    unsafe_start_index: int | None = None
    unsafe_end_index: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def has_location_annotation(self) -> bool:
        return self.unsafe_start_index is not None and self.unsafe_end_index is not None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], split: str) -> "GuardSample":
        """Validate and convert one dataset row."""

        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}.")

        missing = [key for key in ("unique_id", "message", "label") if key not in row]
        if missing:
            raise DatasetFormatError(f"Dataset row is missing required fields: {missing}.")

        raw_input_ids = row.get("input_ids")
        input_ids: tuple[int, ...] | None
        if raw_input_ids is None:
            input_ids = None
        elif isinstance(raw_input_ids, Sequence) and not isinstance(
            raw_input_ids, (str, bytes)
        ):
            try:
                input_ids = tuple(int(token_id) for token_id in raw_input_ids)
            except (TypeError, ValueError) as exc:
                raise DatasetFormatError("'input_ids' must contain integers.") from exc
        else:
            raise DatasetFormatError("'input_ids' must be a sequence or null.")

        unsafe_start = _optional_int(row.get("unsafe_start_index"), "unsafe_start_index")
        unsafe_end = _optional_int(row.get("unsafe_end_index"), "unsafe_end_index")
        if (unsafe_start is None) != (unsafe_end is None):
            raise DatasetFormatError(
                "unsafe_start_index and unsafe_end_index must either both exist or both be null."
            )
        if unsafe_start is not None and unsafe_end is not None:
            if unsafe_start < 0 or unsafe_end < unsafe_start:
                raise DatasetFormatError(
                    "Unsafe location indices must satisfy 0 <= start <= end."
                )

        if split in LOCATION_SPLITS:
            if input_ids is None:
                raise DatasetFormatError(f"Split {split!r} requires annotated input_ids.")
            if unsafe_start is None:
                raise DatasetFormatError(f"Split {split!r} requires unsafe location indices.")
            # Qwen3GuardTest contains terminal unsafe spans whose end index is
            # exactly len(input_ids).  The released evaluator accepts these
            # records and effectively treats the annotation as extending to
            # the end of the sequence, so only values strictly beyond the
            # one-past-the-end boundary are invalid here.
            if unsafe_start is not None and unsafe_start >= len(input_ids):
                raise DatasetFormatError(
                    f"unsafe_start_index={unsafe_start} is outside input_ids "
                    f"length {len(input_ids)}."
                )
            if unsafe_end is not None and unsafe_end > len(input_ids):
                raise DatasetFormatError(
                    f"unsafe_end_index={unsafe_end} exceeds input_ids length {len(input_ids)}."
                )

        return cls(
            split=split,
            unique_id=str(row["unique_id"]),
            messages=_normalise_messages(row["message"]),
            label=normalize_label(row["label"]),
            source=None if row.get("source") is None else str(row["source"]),
            input_ids=input_ids,
            unsafe_type=(
                None if row.get("unsafe_type") is None else str(row["unsafe_type"])
            ),
            unsafe_start_index=unsafe_start,
            unsafe_end_index=unsafe_end,
            raw=dict(row),
        )


def _import_load_dataset() -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised on the remote runtime
        raise RuntimeError(
            "Loading Qwen3GuardTest requires the 'datasets' package. "
            "Install the project requirements before running an entry script."
        ) from exc
    return load_dataset


def load_split(
    dataset_path: str | Path,
    split: str,
    *,
    cache_dir: str | Path | None = None,
    **load_kwargs: Any,
) -> Any:
    """Load one public benchmark split as a Hugging Face ``Dataset``.

    A downloaded dataset snapshot often contains ``thinking.jsonl`` and the two
    location JSONL files directly. Those files are loaded explicitly when found;
    otherwise ``datasets.load_dataset`` receives ``dataset_path`` unchanged, so
    both a Hub dataset id and a dataset-script directory remain supported.
    """

    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}.")

    load_dataset = _import_load_dataset()
    path = Path(dataset_path).expanduser()
    common: dict[str, Any] = dict(load_kwargs)
    if cache_dir is not None:
        common["cache_dir"] = str(Path(cache_dir).expanduser())

    if path.is_file():
        return load_dataset(
            "json", data_files={split: str(path)}, split=split, **common
        )

    jsonl_path = path / f"{split}.jsonl"
    json_path = path / f"{split}.json"
    if jsonl_path.is_file() or json_path.is_file():
        data_file = jsonl_path if jsonl_path.is_file() else json_path
        return load_dataset(
            "json", data_files={split: str(data_file)}, split=split, **common
        )

    source = str(path) if path.exists() else str(dataset_path)
    return load_dataset(source, split=split, **common)


def iter_guard_samples(rows: Iterable[Mapping[str, Any]], split: str) -> Iterator[GuardSample]:
    """Yield validated :class:`GuardSample` objects without materialising a split."""

    for row_index, row in enumerate(rows):
        try:
            yield GuardSample.from_mapping(row, split)
        except DatasetFormatError as exc:
            raise DatasetFormatError(f"Invalid row {row_index} in split {split!r}: {exc}") from exc


def load_guard_samples(
    dataset_path: str | Path,
    split: str,
    *,
    cache_dir: str | Path | None = None,
    **load_kwargs: Any,
) -> list[GuardSample]:
    """Load and validate a complete split."""

    rows = load_split(dataset_path, split, cache_dir=cache_dir, **load_kwargs)
    return list(iter_guard_samples(rows, split))


__all__ = [
    "DatasetFormatError",
    "GuardSample",
    "LOCATION_SPLITS",
    "VALID_SPLITS",
    "iter_guard_samples",
    "load_guard_samples",
    "load_split",
    "normalize_label",
]
