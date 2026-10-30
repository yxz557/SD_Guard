"""Deterministic sample-level seeds shared by every cell in the 2x4 matrix."""

from __future__ import annotations

import hashlib


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    """Match SD_3 exactly: SHA256(base_seed:sample_id) modulo 2**31-1."""

    digest = hashlib.sha256(f"{int(base_seed)}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


__all__ = ["stable_sample_seed"]
