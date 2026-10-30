"""Standalone latency and output-difference harness for the SD_0 strategy."""

from .generation import (
    GenerationResult,
    GenerationSettings,
    SpeculativeRound,
    generate_assisted,
    generate_target_only,
    seed_everything,
)

__all__ = [
    "GenerationResult",
    "GenerationSettings",
    "SpeculativeRound",
    "generate_assisted",
    "generate_target_only",
    "seed_everything",
]
