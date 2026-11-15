from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SEEDS = tuple(range(42, 52))


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset: str
    hidden_dim: int
    purpose: str


EXPERIMENTS = (
    Experiment("c0_h256", "c0", 256, "same-size call baseline"),
    Experiment("t0_h256", "t0", 256, "same-size result baseline"),
    Experiment("rc_h256", "rc", 256, "original two-component model"),
    Experiment("rt_h256", "rt", 256, "original two-component model"),
    Experiment("rct_h256", "rct", 256, "original three-component model"),
    Experiment("rc_pm_h127", "rc", 127, "parameter-matched to C0"),
    Experiment("rt_pm_h127", "rt", 127, "parameter-matched to T0"),
    Experiment("rct_pm_h84", "rct", 84, "parameter-matched to C0"),
    Experiment("c0_large_h514", "c0", 514, "capacity control for RC"),
    Experiment("t0_large_h514", "t0", 514, "capacity control for RT"),
    Experiment("c0_large_h771", "c0", 771, "capacity control for RCT"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bidirectional parameter-count controls for mean-pooled models."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[experiment.name for experiment in EXPERIMENTS],
        help="Subset of experiment names to run.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--positive-weight", default="none")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-loss-plots", action="store_true")
    parser.add_argument("--no-comparison-plot", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parameter_count(input_dim: int, hidden_dim: int) -> int:
    return input_dim * (hidden_dim + 2) + 3 * hidden_dim + 2


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    by_name = {experiment.name: experiment for experiment in EXPERIMENTS}
    unknown = sorted(set(args.experiments) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_specs: list[str] = []

    for name in args.experiments:
        experiment = by_name[name]
        dataset = dataset_root / experiment.dataset
        config_path = dataset / "dataset_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing dataset for {name}: {dataset}")
        dataset_config = json.loads(config_path.read_text(encoding="utf-8"))
        input_dim = int(dataset_config["input_dim"])
        parameters = parameter_count(input_dim, experiment.hidden_dim)
        print(
            f"experiment={name} dataset={experiment.dataset} input_dim={input_dim} "
            f"hidden_dim={experiment.hidden_dim} parameters={parameters} "
            f"purpose={experiment.purpose}",
            flush=True,
        )
        runs_root = output_root / name
        command = [
            sys.executable,
            "-m",
            "guard_training.frozen_hidden_mlp.run_repeated",
            "--offline-dataset",
            str(dataset),
            "--output-root",
            str(runs_root),
            "--seeds",
            *(str(seed) for seed in args.seeds),
            "--head",
            "mlp",
            "--hidden-dim",
            str(experiment.hidden_dim),
            "--dropout",
            str(args.dropout),
            "--positive-weight",
            args.positive_weight,
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--device",
            args.device,
        ]
        if args.force:
            command.append("--force")
        run(command)
        if not args.skip_loss_plots:
            run(
                [
                    sys.executable,
                    "-m",
                    "guard_training.frozen_hidden_mlp.plot_loss_curves",
                    "--runs-root",
                    str(runs_root),
                    "--output",
                    str(runs_root / "training_validation_loss.png"),
                    "--title",
                    f"{name}: train vs validation loss",
                ]
            )
        comparison_specs.append(f"{name}={runs_root}")

    compare = [
        sys.executable,
        "-m",
        "guard_training.tool_step_mean_pool.compare_experiments",
    ]
    for spec in comparison_specs:
        compare.extend(["--experiment", spec])
    compare.extend(["--output-dir", str(output_root / "comparison")])
    if args.no_comparison_plot:
        compare.append("--no-plot")
    run(compare)


if __name__ == "__main__":
    main()
