from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = tuple(range(42, 52))
DEFAULT_VARIANTS = ("c0", "t0", "rc", "rt", "rct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated MLP training for C0/T0/RC/RT/RCT and produce one comparison report."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--include-r0-common", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--head", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=256)
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


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    variants = list(args.variants)
    if args.include_r0_common:
        variants.insert(0, "r0_common")
    unknown = sorted(set(variants) - {"r0_common", *DEFAULT_VARIANTS})
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    output_root.mkdir(parents=True, exist_ok=True)

    comparison_specs: list[str] = []
    for variant in variants:
        dataset = dataset_root / variant
        if not (dataset / "dataset_config.json").is_file():
            raise FileNotFoundError(f"Missing compact dataset for {variant}: {dataset}")
        runs_root = output_root / variant
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
            args.head,
            "--hidden-dim",
            str(args.hidden_dim),
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
                    f"{variant.upper()}: train vs validation loss",
                ]
            )
        display_name = "R0-common" if variant == "r0_common" else variant.upper()
        comparison_specs.append(f"{display_name}={runs_root}")

    compare_command = [
        sys.executable,
        "-m",
        "guard_training.tool_step_mean_pool.compare_experiments",
    ]
    for spec in comparison_specs:
        compare_command.extend(["--experiment", spec])
    compare_command.extend(["--output-dir", str(output_root / "comparison")])
    if args.no_comparison_plot:
        compare_command.append("--no-plot")
    run(compare_command)


if __name__ == "__main__":
    main()
