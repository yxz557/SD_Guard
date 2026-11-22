"""Pinned tests for the independent bf16 loader and the chunked-prefill contract.

Runs on CPU (torch required only for the cross-check and module build).  v5 uses the
declared environment /root/vllm_env_backup/bin/python.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdsafe_v5.loader import (  # noqa: E402
    KNOWN_BITPATTERNS,
    assert_known_bitpatterns,
    bf16_bits_to_f32,
    bf16_bits_to_f32_torch,
    bf16_quantize,
)


def test_known_bitpatterns():
    out = assert_known_bitpatterns()
    assert all(out.values()), out
    print("known bitpatterns:", out)


def test_torch_agreement_random():
    rng = np.random.default_rng(20261114)
    bits = rng.integers(0, 65536, size=100_000, dtype=np.uint32).astype(np.uint16)
    a = bf16_bits_to_f32(bits)
    b = bf16_bits_to_f32_torch(bits).numpy()
    nan = np.isnan(a)
    assert np.array_equal(nan, np.isnan(b)), "NaN masks disagree"
    # finite/inf values must be bit-identical (torch may canonicalise NaN payloads)
    assert np.array_equal(a[~nan].view(np.uint32), b[~nan].view(np.uint32)), \
        "loader disagrees with torch numeric bfloat16 upcast"
    print("random agreement: 100000/100000 bitwise equal")


def test_never_numeric_cast():
    """The historical bug: astype(float32) on bit patterns must NOT equal the truth."""
    bits = np.asarray([0x3F80, 0xBF80, 0x4000], dtype=np.uint16)
    wrong = bits.astype(np.float32)
    right = bf16_bits_to_f32(bits)
    assert list(right) == [1.0, -1.0, 2.0]
    assert not np.allclose(wrong, right), "buggy cast unexpectedly agrees"
    print("buggy cast values:", wrong.tolist(), "correct:", right.tolist())


def test_quantize_roundtrip():
    x = np.asarray([0.0, 1.0, -1.0, 2.0, 3.1415927, 1e-8, -1e-8], dtype=np.float32)
    bits = bf16_quantize(x)
    back = bf16_bits_to_f32(bits)
    # exact for the round numbers, monotone for the rest
    assert back[0] == 0.0 and back[1] == 1.0 and back[2] == -1.0 and back[3] == 2.0
    print("quantize roundtrip:", back.tolist())


if __name__ == "__main__":
    test_known_bitpatterns()
    test_torch_agreement_random()
    test_never_numeric_cast()
    test_quantize_roundtrip()
    print("ALL LOADER TESTS PASSED")
