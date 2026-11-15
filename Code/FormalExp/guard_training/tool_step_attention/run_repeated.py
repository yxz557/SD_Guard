from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from guard_training.tool_step_attention.dataset import VARIANTS


DEFAULT_SEEDS = tuple(range(42, 52))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate repeated readout-conditioned attention models."
    )
    parser.add_argument("--offline-dataset", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    common = [
        "--offline-dataset",
        str(args.offline_dataset),
        "--variant",
        args.variant,
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
    for seed in args.seeds:
        run_dir = args.output_root / f"seed_{seed}"
        checkpoint = run_dir / "best.pt"
        metrics = run_dir / "evaluation" / "test_metrics.json"
        if args.force or not checkpoint.exists():
            run_logged(
                [
                    sys.executable,
                    "-m",
                    "guard_training.tool_step_attention.train",
                    *common,
                    "--output-dir",
                    str(run_dir),
                    "--seed",
                    str(seed),
                ],
                run_dir / "train.log",
            )
        else:
            print(f"seed={seed}: checkpoint exists, skip training")
        if args.force or not metrics.exists():
            run_logged(
                [
                    sys.executable,
                    "-m",
                    "guard_training.tool_step_attention.evaluate",
                    "--offline-dataset",
                    str(args.offline_dataset),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(run_dir / "evaluation"),
                    "--split",
                    "test",
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                ],
                run_dir / "evaluate.log",
            )
        else:
            print(f"seed={seed}: test metrics exist, skip evaluation")
    run_logged(
        [
            sys.executable,
            "-m",
            "guard_training.frozen_hidden_mlp.aggregate_runs",
            "--runs-root",
            str(args.output_root),
            "--output",
            str(args.output_root / "aggregate_test_metrics.json"),
        ],
        args.output_root / "aggregate.log",
    )


if __name__ == "__main__":
    main()
