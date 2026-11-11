from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunHistory:
    name: str
    rows: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot train and validation loss for repeated frozen-hidden-state runs."
        )
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_history(path: Path) -> RunHistory:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {
                key
                for key in ("epoch", "train_loss", "validation_loss")
                if key not in row
            }
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(
                    f"{path}:{line_number} is missing required fields: {missing_text}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"History is empty: {path}")
    return RunHistory(name=path.parent.name, rows=tuple(rows))


def collect_histories(runs_root: Path) -> list[RunHistory]:
    paths = sorted(runs_root.glob("seed_*/history.jsonl"))
    if not paths and (runs_root / "history.jsonl").is_file():
        paths = [runs_root / "history.jsonl"]
    if not paths:
        raise ValueError(f"No seed_*/history.jsonl files under {runs_root}")
    return [load_history(path) for path in paths]


def summarize_metric(
    histories: list[RunHistory], metric: str
) -> tuple[list[int], list[float], list[float], list[int]]:
    values_by_epoch: dict[int, list[float]] = {}
    for history in histories:
        for row in history.rows:
            epoch = int(row["epoch"])
            values_by_epoch.setdefault(epoch, []).append(float(row[metric]))

    epochs = sorted(
        epoch
        for epoch, values in values_by_epoch.items()
        if len(values) == len(histories)
    )
    if not epochs:
        raise ValueError(f"No common epochs found for metric {metric}")
    means = [statistics.mean(values_by_epoch[epoch]) for epoch in epochs]
    standard_deviations = [
        statistics.stdev(values_by_epoch[epoch])
        if len(values_by_epoch[epoch]) > 1
        else 0.0
        for epoch in epochs
    ]
    counts = [len(values_by_epoch[epoch]) for epoch in epochs]
    return epochs, means, standard_deviations, counts


def plot_histories(
    histories: list[RunHistory], output: Path, title: str | None, dpi: int
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting requires matplotlib. Install guard_training/requirements.txt."
        ) from error

    colors = {"train_loss": "#1f77b4", "validation_loss": "#ff7f0e"}
    labels = {"train_loss": "train", "validation_loss": "validation"}
    figure, axis = plt.subplots(figsize=(9.0, 5.5))

    for history in histories:
        epochs = [int(row["epoch"]) for row in history.rows]
        for metric in ("train_loss", "validation_loss"):
            values = [float(row[metric]) for row in history.rows]
            axis.plot(epochs, values, color=colors[metric], alpha=0.16, linewidth=1.0)

    for metric in ("train_loss", "validation_loss"):
        epochs, means, deviations, _ = summarize_metric(histories, metric)
        lower = [mean - deviation for mean, deviation in zip(means, deviations, strict=True)]
        upper = [mean + deviation for mean, deviation in zip(means, deviations, strict=True)]
        axis.fill_between(epochs, lower, upper, color=colors[metric], alpha=0.14)
        axis.plot(
            epochs,
            means,
            color=colors[metric],
            linewidth=2.2,
            label=f"{labels[metric]} mean",
        )

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title(title or f"Training and validation loss ({len(histories)} runs)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.text(
        0.01,
        0.01,
        "Faint lines: individual runs; bands: mean ± 1 sample SD on common epochs",
        transform=axis.transAxes,
        fontsize=8,
        alpha=0.75,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output = args.output or args.runs_root / "training_validation_loss.png"
    histories = collect_histories(args.runs_root)
    plot_histories(histories, output, args.title, args.dpi)
    print(
        json.dumps(
            {
                "runs": len(histories),
                "run_names": [history.name for history in histories],
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
