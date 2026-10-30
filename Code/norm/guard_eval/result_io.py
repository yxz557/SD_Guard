"""Small, dependency-free helpers for evaluation result files."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO


PathLike = str | os.PathLike[str]


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set):
        return sorted(value, key=str)
    # NumPy and similar scalar/array objects expose item()/tolist(). Avoid a
    # hard dependency just for serialisation.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _json_dumps(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        allow_nan=False,
        default=_json_default,
    )


def ensure_parent(path: PathLike) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def iter_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    """Yield JSON objects while reporting exact file/line parse failures."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {source} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {source} at line {line_number}"
                )
            yield value


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _atomic_text_writer(path: Path) -> tuple[TextIO, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    return os.fdopen(descriptor, "w", encoding="utf-8", newline=""), Path(
        temporary_name
    )


def write_jsonl(
    path: PathLike,
    records: Iterable[Mapping[str, Any]],
    *,
    append: bool = False,
) -> int:
    """Write JSON Lines and return the number of records written."""

    output = ensure_parent(path)
    count = 0
    if append:
        with output.open("a", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(_json_dumps(dict(record)) + "\n")
                count += 1
        return count

    handle, temporary = _atomic_text_writer(output)
    try:
        with handle:
            for record in records:
                handle.write(_json_dumps(dict(record)) + "\n")
                count += 1
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


class JsonlWriter:
    """Streaming JSONL writer for long model runs."""

    def __init__(self, path: PathLike, *, append: bool = False, flush: bool = False):
        self.path = ensure_parent(path)
        self.append = append
        self.flush_each_record = flush
        self._handle: TextIO | None = None
        self.count = 0

    def __enter__(self) -> "JsonlWriter":
        self._handle = self.path.open(
            "a" if self.append else "w", encoding="utf-8", newline="\n"
        )
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("JsonlWriter must be used as a context manager")
        self._handle.write(_json_dumps(dict(record)) + "\n")
        self.count += 1
        if self.flush_each_record:
            self._handle.flush()

    def flush(self) -> None:
        """Flush buffered records without closing the writer.

        Long streaming runs call this once after a complete sample.  That
        keeps completed samples durable without forcing one flush for every
        token record.
        """

        if self._handle is None:
            raise RuntimeError("JsonlWriter must be used as a context manager")
        self._handle.flush()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def append_jsonl(path: PathLike, record: Mapping[str, Any]) -> None:
    write_jsonl(path, [record], append=True)


def read_json(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: PathLike, value: Any, *, indent: int = 2) -> Path:
    output = ensure_parent(path)
    handle, temporary = _atomic_text_writer(output)
    try:
        with handle:
            handle.write(_json_dumps(value, indent=indent))
            handle.write("\n")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return _json_dumps(value)
    return value


def write_csv(
    path: PathLike,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write dictionaries to UTF-8 CSV, preserving later-discovered columns."""

    output = ensure_parent(path)
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        ordered_fields: list[str] = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered_fields.append(key)
        fieldnames = ordered_fields

    handle, temporary = _atomic_text_writer(output)
    try:
        with handle:
            if fieldnames:
                writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
                writer.writeheader()
                for row in materialized:
                    writer.writerow(
                        {field: _csv_cell(row.get(field)) for field in fieldnames}
                    )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output

