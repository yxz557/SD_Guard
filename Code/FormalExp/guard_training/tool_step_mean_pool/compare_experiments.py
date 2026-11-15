from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DISPLAY_METRICS = (
    "average_precision",
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "positive_precision",
    "positive_recall",
    "false_positive_rate",
    "false_negative_rate",
    "positive_f1",
    "threshold",
)

DISPLAY_NAMES = {
    "average_precision": "AP",
    "roc_auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced acc.",
    "positive_precision": "Precision",
    "positive_recall": "Recall",
    "false_positive_rate": "FPR",
    "false_negative_rate": "FNR",
    "positive_f1": "Positive F1",
    "threshold": "Threshold",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare repeated R0/C0/T0/RC/RT/RCT test results."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="NAME=RUNS_ROOT",
        help="Repeat once per experiment, for example C0=/path/to/c0.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_experiments(values: Iterable[str]) -> list[tuple[str, Path]]:
    experiments: list[tuple[str, Path]] = []
    names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Experiment must have NAME=RUNS_ROOT form: {value}")
        name, path_text = value.split("=", 1)
        name = name.strip()
        if not name or not path_text.strip():
            raise ValueError(f"Experiment must have NAME=RUNS_ROOT form: {value}")
        if name in names:
            raise ValueError(f"Duplicate experiment name: {name}")
        path = Path(path_text).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Experiment runs root does not exist: {path}")
        names.add(name)
        experiments.append((name, path))
    return experiments


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def normalized_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    matrix = metrics.get("confusion_matrix")
    if not (
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
    ):
        raise ValueError("Metrics report has no 2x2 confusion_matrix")
    tn, fp = (int(value) for value in matrix[0])
    fn, tp = (int(value) for value in matrix[1])
    samples = tn + fp + fn + tp
    output: dict[str, float | None] = {
        "average_precision": metrics.get("average_precision"),
        "roc_auc": metrics.get("roc_auc"),
        "accuracy": safe_rate(tn + tp, samples),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "positive_precision": metrics.get("positive_precision"),
        "positive_recall": metrics.get("positive_recall"),
        "false_positive_rate": safe_rate(fp, fp + tn),
        "false_negative_rate": safe_rate(fn, fn + tp),
        "positive_f1": metrics.get("positive_f1"),
        "threshold": metrics.get("threshold"),
    }
    return {
        key: (float(value) if isinstance(value, (int, float)) else None)
        for key, value in output.items()
    }


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def aggregate_metric_views(views: list[dict[str, float | None]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in DISPLAY_METRICS:
        values = [
            float(view[metric])
            for view in views
            if view.get(metric) is not None and math.isfinite(float(view[metric]))
        ]
        if values:
            output[metric] = summarize(values)
    return output


def aggregate_groups(reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    group_names = sorted(
        set.intersection(*(set(report.get(key, {})) for report in reports))
    )
    output: dict[str, Any] = {}
    for group in group_names:
        raw = [report[key][group] for report in reports]
        supports = {
            (int(value["samples"]), int(value["positives"])) for value in raw
        }
        if len(supports) != 1:
            raise ValueError(f"Group support changed across runs for {key}/{group}")
        samples, positives = next(iter(supports))
        output[group] = {
            "support": {
                "samples": samples,
                "positives": positives,
                "prevalence": positives / samples if samples else 0.0,
            },
            **aggregate_metric_views([normalized_metrics(value) for value in raw]),
        }
    return output


def count_mlp_parameters(config: dict[str, Any]) -> int | None:
    if config.get("head") != "mlp":
        return None
    input_dim = int(config["input_dim"])
    hidden_dim = int(config["hidden_dim"])
    # LayerNorm weight+bias, first Linear, and two-logit output Linear.
    return 2 * input_dim + input_dim * hidden_dim + hidden_dim + hidden_dim * 2 + 2


def load_predictions(path: Path) -> tuple[set[str], Counter[str]]:
    ids: set[str] = set()
    labels: Counter[str] = Counter()
    if not path.is_file():
        return ids, labels
    for row in read_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in ids:
            raise ValueError(f"Duplicate prediction sample ID in {path}: {sample_id}")
        ids.add(sample_id)
        labels[str(int(row["label"]))] += 1
    return ids, labels


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON objects in {path}")
                rows.append(value)
    return rows


def load_experiment(name: str, runs_root: Path) -> tuple[dict[str, Any], set[str]]:
    metrics_paths = sorted(runs_root.glob("seed_*/evaluation/test_metrics.json"))
    if not metrics_paths:
        raise ValueError(f"No seed_*/evaluation/test_metrics.json under {runs_root}")
    reports: list[dict[str, Any]] = []
    seeds: list[int] = []
    dataset_paths: set[str] = set()
    dataset_hashes: set[str] = set()
    parameter_counts: set[int] = set()
    reference_ids: set[str] | None = None
    label_counts: Counter[str] | None = None

    for metrics_path in metrics_paths:
        reports.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        run_dir = metrics_path.parents[1]
        config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
        seeds.append(int(config["seed"]))
        offline_dataset = str(config.get("offline_dataset") or "")
        dataset_paths.add(offline_dataset)
        if offline_dataset:
            dataset_config = Path(offline_dataset) / "dataset_config.json"
            if dataset_config.is_file():
                dataset_hashes.add(file_sha256(dataset_config))
        parameter_count = count_mlp_parameters(config)
        if parameter_count is not None:
            parameter_counts.add(parameter_count)
        prediction_path = metrics_path.with_name("test_predictions.jsonl")
        ids, counts = load_predictions(prediction_path)
        if ids:
            if reference_ids is None:
                reference_ids = ids
                label_counts = counts
            elif ids != reference_ids:
                raise ValueError(f"Test sample IDs differ across seeds in {runs_root}")

    if len(dataset_paths) != 1:
        raise ValueError(f"Seeds use different offline datasets in {runs_root}")
    if len(parameter_counts) > 1:
        raise ValueError(f"Seeds use different model sizes in {runs_root}")
    report = {
        "name": name,
        "runs_root": str(runs_root),
        "runs": len(reports),
        "seeds": seeds,
        "offline_dataset": next(iter(dataset_paths)),
        "dataset_config_sha256": (
            next(iter(dataset_hashes)) if len(dataset_hashes) == 1 else None
        ),
        "model_parameters": (
            next(iter(parameter_counts)) if len(parameter_counts) == 1 else None
        ),
        "test_samples": len(reference_ids or set()),
        "test_labels": dict(sorted((label_counts or Counter()).items())),
        "overall": aggregate_metric_views(
            [normalized_metrics(value["overall"]) for value in reports]
        ),
        "by_suite": aggregate_groups(reports, "by_suite"),
        "by_attack": aggregate_groups(reports, "by_attack"),
    }
    return report, reference_ids or set()


def matched_test_sets(
    experiments: list[dict[str, Any]], sample_sets: dict[str, set[str]]
) -> dict[str, Any]:
    reference_name = experiments[0]["name"]
    reference = sample_sets[reference_name]
    details: dict[str, Any] = {}
    for experiment in experiments:
        name = experiment["name"]
        current = sample_sets[name]
        details[name] = {
            "matches_reference": current == reference if reference and current else None,
            "missing_vs_reference": len(reference - current),
            "extra_vs_reference": len(current - reference),
        }
    comparable = [value["matches_reference"] for value in details.values()]
    return {
        "reference": reference_name,
        "all_exactly_matched": all(value is True for value in comparable),
        "prediction_files_available_for_all": all(value is not None for value in comparable),
        "details": details,
    }


def fmt_metric(value: dict[str, Any] | None) -> str:
    if value is None:
        return "—"
    return f"{value['mean']:.4f} ± {value['std']:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def build_markdown(report: dict[str, Any]) -> str:
    experiments = report["experiments"]
    lines = [
        "# Tool-step mean-pooling 实验对比",
        "",
        "所有数值均为多随机种子 test 结果的 mean ± sample SD。阈值由各次训练的 validation 集独立选择。",
        "",
        "## 数据与模型一致性",
        "",
    ]
    match = report["test_set_comparability"]
    data_rows = []
    for experiment in experiments:
        detail = match["details"][experiment["name"]]
        data_rows.append(
            [
                experiment["name"],
                str(experiment["runs"]),
                str(experiment["test_samples"]),
                str(experiment["test_labels"]),
                str(experiment["model_parameters"] or "—"),
                str(detail["matches_reference"]),
            ]
        )
    lines.extend(
        markdown_table(
            ["实验", "runs", "test N", "test labels", "MLP 参数量", "与首项同 test IDs"],
            data_rows,
        )
    )
    lines.extend(["", "## 总体指标", ""])
    overall_rows = []
    for experiment in experiments:
        overall_rows.append(
            [experiment["name"]]
            + [fmt_metric(experiment["overall"].get(metric)) for metric in DISPLAY_METRICS]
        )
    lines.extend(
        markdown_table(
            ["实验"] + [DISPLAY_NAMES[metric] for metric in DISPLAY_METRICS],
            overall_rows,
        )
    )

    for group_key, title in (("by_suite", "按 suite"), ("by_attack", "按 attack")):
        lines.extend(["", f"## {title}", ""])
        group_names = sorted(
            set().union(*(set(experiment[group_key]) for experiment in experiments))
        )
        rows: list[list[str]] = []
        selected_metrics = (
            "average_precision",
            "roc_auc",
            "positive_precision",
            "positive_recall",
            "false_positive_rate",
            "false_negative_rate",
        )
        for group in group_names:
            for experiment in experiments:
                metrics = experiment[group_key].get(group, {})
                support = metrics.get("support", {})
                rows.append(
                    [
                        group,
                        experiment["name"],
                        str(support.get("samples", "—")),
                        str(support.get("positives", "—")),
                    ]
                    + [fmt_metric(metrics.get(metric)) for metric in selected_metrics]
                )
        lines.extend(
            markdown_table(
                ["分组", "实验", "N", "正例"]
                + [DISPLAY_NAMES[metric] for metric in selected_metrics],
                rows,
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, experiments: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "metric", "mean", "std", "min", "max"])
        for experiment in experiments:
            for metric in DISPLAY_METRICS:
                value = experiment["overall"].get(metric)
                if value is not None:
                    writer.writerow(
                        [
                            experiment["name"],
                            metric,
                            value["mean"],
                            value["std"],
                            value["min"],
                            value["max"],
                        ]
                    )


def plot_overall(path: Path, experiments: list[dict[str, Any]], dpi: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Plotting requires matplotlib and numpy") from error

    metrics = ("average_precision", "roc_auc", "accuracy", "false_positive_rate", "false_negative_rate")
    names = [experiment["name"] for experiment in experiments]
    x = np.arange(len(names))
    width = 0.15
    figure, axis = plt.subplots(figsize=(max(9.0, 1.25 * len(names)), 5.5))
    for index, metric in enumerate(metrics):
        means = [experiment["overall"][metric]["mean"] for experiment in experiments]
        errors = [experiment["overall"][metric]["std"] for experiment in experiments]
        offset = (index - (len(metrics) - 1) / 2) * width
        axis.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=2,
            label=DISPLAY_NAMES[metric],
        )
    axis.set_xticks(x, names)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Test metric (mean ± sample SD)")
    axis.set_title("Tool-step mean-pooling comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def compare_experiments(
    experiment_specs: list[tuple[str, Path]], output_dir: Path, plot: bool, dpi: int
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments: list[dict[str, Any]] = []
    sample_sets: dict[str, set[str]] = {}
    for name, runs_root in experiment_specs:
        experiment, sample_ids = load_experiment(name, runs_root)
        experiments.append(experiment)
        sample_sets[name] = sample_ids
    report = {
        "schema_version": "sd_guard.tool_step_experiment_comparison.v1",
        "experiments": experiments,
        "test_set_comparability": matched_test_sets(experiments, sample_sets),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output_dir / "comparison.md").write_text(
        build_markdown(report), encoding="utf-8", newline="\n"
    )
    write_csv(output_dir / "overall_metrics.csv", experiments)
    if plot:
        plot_overall(output_dir / "overall_metrics.png", experiments, dpi)
    return report


def main() -> None:
    args = parse_args()
    experiment_specs = parse_experiments(args.experiment)
    report = compare_experiments(
        experiment_specs, args.output_dir.resolve(), not args.no_plot, args.dpi
    )
    print(
        json.dumps(
            {
                "experiments": [value["name"] for value in report["experiments"]],
                "output_dir": str(args.output_dir.resolve()),
                "test_set_comparability": report["test_set_comparability"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

