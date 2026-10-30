"""Shared output-directory policy for SD latency experiments."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    """Locate the SD_Guard root regardless of the harness's copied layout."""

    source = Path(__file__).resolve()
    for parent in source.parents:
        if parent.name == "SD_Guard":
            return parent
    raise RuntimeError(f"Cannot locate the SD_Guard repository above {source}")


def default_output_root() -> Path:
    """Return <repository>/SD_latency_test/outputs on every host."""

    return repository_root() / "SD_latency_test" / "outputs"


def resolve_experiment_output_dir(
    requested: Path | None,
    *,
    timestamp: str,
    descriptor: str,
) -> Path:
    """Resolve an explicit path or allocate a unique default experiment path."""

    if requested is not None:
        return requested.expanduser().resolve()

    base = default_output_root()
    candidate = base / f"{timestamp}_{descriptor}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{timestamp}_{descriptor}_{suffix:02d}"
        suffix += 1
    return candidate
