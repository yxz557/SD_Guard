#!/usr/bin/env python3
"""Plot boundary-relative detection-latency ECDFs from location evaluation.

The script consumes the existing ``sample_records.jsonl`` files produced by
``eval_guard_location_full_forward.py``. It does not load the model or rerun
inference.

AutoDL example
---------------

.. code-block:: bash

    pip install matplotlib

    python Code/norm/plot_location_cdf.py \
      --results-dir /root/autodl-tmp/SD_Guard/outputs/location-full-forward-4B

This creates ``location_latency_cdf.png`` and
``location_latency_cdf.pdf`` inside the result directory.

Latency definition
------------------

For trigger position ``t`` and annotated unsafe interval ``[s, e]``:

* ``t < s``: latency = ``t - s`` (negative; ahead of the unsafe interval)
* ``s <= t <= e``: latency = ``0`` (exact hit inside the unsafe interval)
* ``t > e``: latency = ``t - e`` (positive; tokens after the interval)
* missed response: positive infinity

Missed responses are kept in the denominator but are not plotted at a finite
x coordinate. Consequently, each ECDF plateaus at response recall rather than
artificially reaching 1.0. In this coordinate system, F(0) is the fraction
detected by the end of the annotated interval, and F(128) is the benchmark's
first-128 rate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'matplotlib'. Install it with: pip install matplotlib"
    ) from exc


SPLITS = ("thinking_loc", "response_loc")
POLICIES = ("strict", "loose")
COLORS = {"strict": "#C44E52", "loose": "#4C72B0"}
LINESTYLES = {"strict": "-", "loose": "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot strict/loose empirical CDFs of Stream Guard detection "
            "latency using existing location-evaluation sample records."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=(
            "Evaluation result directory containing "
            "raw/<split>/sample_records.jsonl."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output path without extension. Defaults to "
            "<results-dir>/location_latency_cdf."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="PNG resolution (default: 240).",
    )
    parser.add_argument(
        "--title",
        default="Qwen3Guard-Stream-4B detection latency",
        help="Figure title.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample record file: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc

    if not rows:
        raise ValueError(f"No sample records found in {path}")
    return rows


def boundary_relative_latency(row: dict[str, Any], policy: str) -> float:
    """Return benchmark-aligned signed latency, or +inf for a miss."""

    decision = row["policies"][policy]
    trigger = decision.get("trigger_full_index")
    if not decision.get("triggered", False) or trigger is None:
        return math.inf

    start = int(row["unsafe_start_index"])
    end = int(row["unsafe_end_index"])
    trigger = int(trigger)

    if trigger < start:
        return float(trigger - start)
    if trigger <= end:
        return 0.0
    return float(trigger - end)


def aggregate_ecdf(
    latencies: Iterable[float],
) -> tuple[list[float], list[float], int, int]:
    """Aggregate duplicate x values while retaining misses in denominator."""

    values = list(latencies)
    total = len(values)
    finite = [value for value in values if math.isfinite(value)]
    counts = Counter(finite)

    xs = sorted(counts)
    cumulative = 0
    ys: list[float] = []
    for x in xs:
        cumulative += counts[x]
        ys.append(cumulative / total)
    return xs, ys, total, len(finite)


def cdf_at(latencies: Iterable[float], threshold: float) -> float:
    values = list(latencies)
    return sum(value <= threshold for value in values) / len(values)


def rounded_axis_limit(value: float, step: int) -> float:
    return math.ceil(value / step) * step


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix is not None
        else results_dir / "location_latency_cdf"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows_by_split = {
        split: read_jsonl(
            results_dir / "raw" / split / "sample_records.jsonl"
        )
        for split in SPLITS
    }
    latencies = {
        (split, policy): [
            boundary_relative_latency(row, policy)
            for row in rows_by_split[split]
        ]
        for split in SPLITS
        for policy in POLICIES
    }

    all_finite = [
        value
        for values in latencies.values()
        for value in values
        if math.isfinite(value)
    ]
    min_x = min(all_finite)
    max_x = max(all_finite)
    plot_min = min(-256.0, -rounded_axis_limit(abs(min_x), 32))
    plot_max = max(256.0, rounded_axis_limit(max_x, 32))

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, split in zip(axes, SPLITS):
        for policy in POLICIES:
            values = latencies[(split, policy)]
            xs, ys, total, detected = aggregate_ecdf(values)
            f0 = cdf_at(values, 0)
            f128 = cdf_at(values, 128)
            recall = detected / total

            step_x = [plot_min, *xs, plot_max]
            step_y = [0.0, *ys, recall]
            label = (
                f"{policy.capitalize()}  "
                f"end={f0:.1%}, +128={f128:.1%}, recall={recall:.1%}"
            )
            ax.step(
                step_x,
                step_y,
                where="post",
                color=COLORS[policy],
                linestyle=LINESTYLES[policy],
                linewidth=2.1,
                label=label,
            )
            ax.scatter(
                [0, 128],
                [f0, f128],
                color=COLORS[policy],
                s=24,
                zorder=4,
                edgecolors="white",
                linewidths=0.6,
            )

            print(
                f"{split:12s} {policy:6s} "
                f"N={total:3d} detected={detected:3d} "
                f"F(0)={f0:.6f} F(128)={f128:.6f} "
                f"recall={recall:.6f} FNR={1.0 - recall:.6f}"
            )

        ax.axvline(0, color="#444444", linewidth=1.0, alpha=0.8)
        ax.axvline(128, color="#888888", linewidth=1.0, alpha=0.8)
        ax.text(
            0,
            0.015,
            "unsafe interval end",
            rotation=90,
            va="bottom",
            ha="right",
            color="#444444",
            fontsize=8,
        )
        ax.text(
            128,
            0.015,
            "+128",
            rotation=90,
            va="bottom",
            ha="right",
            color="#777777",
            fontsize=8,
        )
        ax.set_title(split)
        ax.set_xscale("symlog", base=2, linthresh=32, linscale=1.0)
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(0, 1.01)
        ax.grid(True, which="major", color="#D9D9D9", linewidth=0.7, alpha=0.8)
        ax.grid(True, which="minor", color="#EEEEEE", linewidth=0.5, alpha=0.7)
        ax.legend(loc="lower right", frameon=True, framealpha=0.95)
        ax.set_xlabel(
            "Boundary-relative latency (tokens)\n"
            "negative = ahead, 0 = hit, positive = after interval"
        )

    axes[0].set_ylabel("Cumulative fraction of all unsafe responses")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    fig.suptitle(args.title)

    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
