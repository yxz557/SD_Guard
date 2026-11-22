"""Hashing, atomic IO, read-only guards and small numpy helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

STUDY_ROOT = Path(__file__).resolve().parents[2]


class StudyError(RuntimeError):
    pass


def sha256_file(path: Path | str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_bytes(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def atomic_write_json(path: Path | str, payload: Any) -> None:
    atomic_write_bytes(
        path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False).encode("utf-8") + b"\n"
    )


def atomic_write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    buf = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n" for row in rows
    )
    atomic_write_bytes(path, buf)


def json_load(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assert_inside_study(path: Path | str) -> Path:
    """Every new write must stay inside the assigned study root."""
    p = Path(path).resolve()
    if not p.is_relative_to(STUDY_ROOT):
        raise StudyError(f"refusing to write outside the study root: {p}")
    return p


def assert_read_only(path: Path | str, read_roots: Sequence[Path]) -> Path:
    p = Path(path).resolve()
    if p.is_relative_to(STUDY_ROOT):
        raise StudyError(f"expected a read-only upstream path, got a study path: {p}")
    if not any(p.is_relative_to(r.resolve()) for r in read_roots):
        raise StudyError(f"path is not inside a declared read-only root: {p}")
    return p


def guard_upstream_hashes(pairs: dict[str, str]) -> dict[str, Any]:
    """Re-verify upstream file hashes; any drift aborts the stage."""
    report: dict[str, Any] = {"checked": 0, "mismatch": [], "missing": []}
    for rel, expected in pairs.items():
        p = Path(rel)
        if not p.is_file():
            report["missing"].append(rel)
            continue
        actual = sha256_file(p)
        report["checked"] += 1
        if actual != expected:
            report["mismatch"].append({"path": rel, "expected": expected, "actual": actual})
    report["status"] = "PASS" if not report["mismatch"] and not report["missing"] else "FAIL"
    return report
