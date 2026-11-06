from __future__ import annotations

import argparse
import json
from pathlib import Path

from guard_training.common.feature_store import iter_feature_metadata
from guard_training.common.splitting import find_group_stratified_split, summarize_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a leakage-safe group-stratified split.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--trials", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_rows = list(iter_feature_metadata(args.feature_root / "features"))
    sample_ids = [row["sample_id"] for row in metadata_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs found in extracted feature metadata")

    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    result = find_group_stratified_split(
        metadata_rows,
        ratios=ratios,
        seed=args.seed,
        trials=args.trials,
    )

    output_rows = []
    keep_keys = (
        "sample_id",
        "group_id",
        "label",
        "suite_name",
        "attack_type",
        "user_task_id",
        "injection_task_id",
        "trace_sha256",
        "serialized_sha256",
        "token_count",
        "shard",
        "row",
        "step_start",
        "step_end",
    )
    for row in metadata_rows:
        output = {key: row.get(key) for key in keep_keys}
        output["split"] = result.group_to_split[str(row["group_id"])]
        output_rows.append(output)
    output_rows.sort(key=lambda row: (row["split"], row["group_id"], row["sample_id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "schema_version": "sd_guard.group_split.v1",
        "group_key": ["suite_name", "user_task_id", "injection_task_id"],
        "stratification_target": ["suite_name", "attack_type", "label"],
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "trials": args.trials,
        "winning_trial": result.trial,
        "score": result.score,
        "counts": summarize_split(output_rows),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

