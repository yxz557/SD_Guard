from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from guard_training.tool_step_attention.dataset import VARIANTS


DEFAULT_SEEDS = tuple(range(42, 52))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all readout-conditioned attention variants and compare them."
    )
    parser.add_argument("--offline-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="NAME=RUNS_ROOT",
        help="Existing repeated mean baseline to include in the final comparison.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--positive-weight", default="none")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
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


def main() -> None:
    args = parse_args()
    dataset = args.offline_dataset.resolve()
    output_root = args.output_root.resolve()
    if not (dataset / "dataset_config.json").is_file():
        raise FileNotFoundError(f"Missing attention dataset: {dataset}")
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_specs: list[str] = list(args.reference)
    for variant in args.variants:
        runs_root = output_root / variant
        command = [
            sys.executable,
            "-m",
            "guard_training.tool_step_attention.run_repeated",
            "--offline-dataset",
            str(dataset),
            "--variant",
            variant,
            "--output-root",
            str(runs_root),
            "--seeds",
            *(str(seed) for seed in args.seeds),
            "--attention-dim",
            str(args.attention_dim),
            "--head-hidden-dim",
            str(args.head_hidden_dim),
            "--dropout",
            str(args.dropout),
            "--attention-dropout",
            str(args.attention_dropout),
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
                    f"{variant}: train vs validation loss",
                ]
            )
        comparison_specs.append(f"{variant}={runs_root}")

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

