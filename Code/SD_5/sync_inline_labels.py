#!/usr/bin/env python3
"""Reuse speculative judgments for inline cells only after exact output checks."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sdguard_sd.io_utils import iter_jsonl


def _rows(cell: Path) -> dict[str, dict]:
    return {
        str(row["sample_id"]): row
        for row in iter_jsonl(cell / "raw" / "generation_records.jsonl")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.matrix_dir.expanduser().resolve()
    for thinking in ("think_on", "think_off"):
        standard_cell = root / thinking / "speculative"
        inline_cell = root / thinking / "inline_guard"
        standard = _rows(standard_cell)
        inline = _rows(inline_cell)
        if standard.keys() != inline.keys():
            raise RuntimeError(f"Sample IDs differ for {thinking}")
        for sample_id in standard:
            left = standard[sample_id]
            right = inline[sample_id]
            if left["generation"]["generated_ids"] != right["generation"]["generated_ids"]:
                raise RuntimeError(
                    f"Cannot reuse labels: generated IDs differ for {thinking}/{sample_id}"
                )
            equivalence = right.get("inline_guard", {}).get("reference_equivalence", {})
            if not all(
                equivalence.get(key) is True
                for key in (
                    "generated_ids_match",
                    "completion_match",
                    "acceptance_trace_match",
                    "human_labels_reusable",
                )
            ):
                raise RuntimeError(
                    f"Cannot reuse labels: equivalence proof missing for {thinking}/{sample_id}"
                )
        source = standard_cell / "blind" / "judge_labels.jsonl"
        destination = inline_cell / "blind" / "judge_labels.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"Missing speculative judgments: {source}")
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Inline judgment file exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        print(f"Verified {len(standard)} exact pairs; copied {source} -> {destination}")


if __name__ == "__main__":
    main()
