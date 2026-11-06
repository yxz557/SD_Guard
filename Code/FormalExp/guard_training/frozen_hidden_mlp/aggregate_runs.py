from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate repeated classifier test runs.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def aggregate_dicts(values: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    common_keys = set.intersection(*(set(value) for value in values)) if values else set()
    for key in sorted(common_keys):
        items = [value[key] for value in values]
        if all(_is_number(item) for item in items):
            numbers = [float(item) for item in items]
            result[key] = {
                "mean": statistics.mean(numbers),
                "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
                "min": min(numbers),
                "max": max(numbers),
                "values": numbers,
            }
        elif all(isinstance(item, dict) for item in items):
            nested = aggregate_dicts(items)
            if nested:
                result[key] = nested
    return result


def collect_reports(runs_root: Path) -> list[dict[str, Any]]:
    reports = []
    for metrics_path in sorted(runs_root.glob("seed_*/evaluation/test_metrics.json")):
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        train_config_path = metrics_path.parents[1] / "train_config.json"
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        reports.append(
            {
                "seed": int(train_config["seed"]),
                "metrics_path": str(metrics_path.resolve()),
                "checkpoint_path": str((metrics_path.parents[1] / "best.pt").resolve()),
                "report": report,
            }
        )
    return sorted(reports, key=lambda item: item["seed"])


def build_aggregate(runs_root: Path) -> dict[str, Any]:
    reports = collect_reports(runs_root)
    if not reports:
        raise ValueError(f"No seed_*/evaluation/test_metrics.json files under {runs_root}")
    test_reports = [item["report"] for item in reports]
    return {
        "schema_version": "sd_guard.repeated_test_aggregate.v1",
        "runs": len(reports),
        "seeds": [item["seed"] for item in reports],
        "std_definition": "sample standard deviation (N-1 denominator)",
        "aggregate": aggregate_dicts(test_reports),
        "individual_runs": [
            {
                "seed": item["seed"],
                "metrics_path": item["metrics_path"],
                "checkpoint_path": item["checkpoint_path"],
                "overall": item["report"]["overall"],
            }
            for item in reports
        ],
    }


def main() -> None:
    args = parse_args()
    report = build_aggregate(args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

