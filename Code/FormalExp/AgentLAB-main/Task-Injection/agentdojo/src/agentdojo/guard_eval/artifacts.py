from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically so hashes are reproducible."""
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, separators=(",", ": "))
    return (text + "\n").encode("utf-8")


def compact_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically without presentation whitespace."""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text.encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(compact_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _make_read_only(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        # Hash verification is authoritative on filesystems that do not honor chmod.
        pass


def write_immutable_bytes(path: Path, content: bytes, *, read_only: bool = True) -> None:
    """Create a file once; permit idempotent reruns but reject changed content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Refusing to overwrite immutable artifact with different content: {path}")
        return

    with path.open("xb") as file:
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    if read_only:
        _make_read_only(path)


def write_immutable_json(path: Path, value: Any, *, read_only: bool = True) -> str:
    content = canonical_json_bytes(value)
    write_immutable_bytes(path, content, read_only=read_only)
    return sha256_bytes(content)


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically replace a mutable derived artifact such as a prediction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_trace_hash(trace_path: Path, trace_sha256: str) -> Path:
    hash_path = trace_path.with_suffix(".sha256")
    content = f"{trace_sha256}  {trace_path.name}\n".encode()
    write_immutable_bytes(hash_path, content)
    return hash_path


def verify_trace(trace_path: Path) -> str:
    hash_path = trace_path.with_suffix(".sha256")
    if not hash_path.exists():
        raise FileNotFoundError(f"Missing trace hash sidecar: {hash_path}")
    fields = hash_path.read_text(encoding="utf-8").strip().split()
    if not fields:
        raise ValueError(f"Empty trace hash sidecar: {hash_path}")
    expected = fields[0]
    actual = sha256_file(trace_path)
    if actual != expected:
        raise ValueError(f"Trace hash mismatch for {trace_path}: expected {expected}, got {actual}")
    return actual


def discover_files(inputs: list[Path], filename: str) -> list[Path]:
    discovered: set[Path] = set()
    for input_path in inputs:
        path = input_path.resolve()
        if path.is_file():
            if path.name == filename or filename == "*.json":
                discovered.add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        pattern = "*.json" if filename == "*.json" else filename
        discovered.update(candidate.resolve() for candidate in path.rglob(pattern))
    return sorted(discovered)
