"""Independent bf16 bit-pattern loader (the only mapping used by v5 datasets).

The frozen bf16 storage keeps the *bit pattern* as ``uint16``.  The correct float32 value
is obtained by shifting the pattern into the high half of a ``uint32`` and reinterpreting:
``f32 = uint16_bits.astype(uint32) << 16``.  A plain ``astype(float32)`` reinterprets the
pattern as a small integer and silently feeds denormal-scale numbers (the exact bug that
collapsed the frozen v4 P5 stage to AUC 0.5).

This module is deliberately small and separate from the training/pipeline reader, so the
validator can use it as an independent implementation.  ``tests/test_loader_bits.py``
pins known IEEE-754 bf16 patterns and cross-checks against torch's own
``bfloat16.view(float32)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

KNOWN_BITPATTERNS = {
    0x3F80: 1.0,      # +1.0
    0xBF80: -1.0,     # -1.0
    0x4000: 2.0,      # +2.0
    0xC000: -2.0,     # -2.0
    0x0000: 0.0,
    0x8000: -0.0,
    0x3F00: 0.5,
    0x7F80: float("inf"),
    0xFF80: float("-inf"),
}


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """uint16 bf16 bit patterns -> float32 values (left-shift, never a numeric cast)."""
    raw = np.asarray(bits)
    if raw.dtype != np.uint16:
        if raw.dtype.kind not in "iu":
            raise TypeError(f"expected integer bf16 bit patterns, got {raw.dtype}")
        raw = raw.astype(np.uint16)
    shifted = raw.astype(np.uint32) << np.uint32(16)
    return shifted.view(np.float32)


def bf16_bits_to_f32_torch(bits):
    """Same conversion through torch's own dtype view (cross-check only)."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(np.asarray(bits, dtype=np.uint16)))
    # numeric upcast: bfloat16 -> float32 (a same-itemsize dtype *view* would reinterpret
    # pairs of bf16 patterns as one float32 and is therefore the wrong cross-check)
    return t.view(torch.bfloat16).to(torch.float32)


def assert_known_bitpatterns() -> dict[str, bool]:
    """Bit-exact sanity table; any mismatch is a hard failure."""
    out = {}
    for bits, want in KNOWN_BITPATTERNS.items():
        got = float(bf16_bits_to_f32(np.asarray([bits], dtype=np.uint16))[0])
        if np.isnan(want):
            ok = bool(np.isnan(got))
        else:
            ok = bool(got == want) and bool(np.signbit(got) == np.signbit(want))
        out[f"0x{bits:04x}"] = ok
        if not ok:
            raise AssertionError(f"bit pattern 0x{bits:04x}: got {got!r}, want {want!r}")
    return out


def load_bf16_matrix(path: Path | str, rows: int, dim: int = 4096) -> np.ndarray:
    """Read-only memmap of a bf16 matrix stored as raw little-endian uint16 rows."""
    flat = np.memmap(path, dtype="<u2", mode="r")
    if flat.size != rows * dim:
        raise ValueError(f"{path}: expected {rows * dim} uint16 elements, got {flat.size}")
    return flat.reshape(rows, dim)


def decode_rows(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """float32 rows for ``idx`` (gather at uint16 level, then shift)."""
    return bf16_bits_to_f32(np.asarray(mat[np.asarray(idx, dtype=np.int64)]))


def bf16_quantize(x: np.ndarray) -> np.ndarray:
    """Round-trip helper: float -> bf16 bit pattern (truncation, as torch bf16 cast)."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))
    return t.to(torch.bfloat16).view(torch.uint16).numpy()
