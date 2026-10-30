"""Generate publication/PPT figures with matplotlib.

The script reads the existing JSON reports under ``results`` and writes both
SVG and 300-DPI PNG files.  It is intended for the Linux GPU server (or any
machine with matplotlib installed); it does not rerun model inference.

Server usage from the repository root::

    python -m pip install matplotlib
    python Code/scripts/generate_ppt_figures_matplotlib.py

If the server cannot render Chinese text, pass a CJK font file explicitly::

    python Code/scripts/generate_ppt_figures_matplotlib.py \
        --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError as exc:  # pragma: no cover - exercised on the server
    raise SystemExit(
        "matplotlib is required for this version. Install it with: "
        "python -m pip install matplotlib"
    ) from exc

from ppt_figure_data import (
    MODE_ORDER,
    THINKING_ORDER,
    collect_figure_data,
    default_repo_root,
)


TARGET_COLOR = "#4C78A8"
SD_COLOR = "#F58518"
THINK_ON_COLOR = "#4C78A8"
THINK_OFF_COLOR = "#59A14F"
GRID_COLOR = "#D9DEE7"
TEXT_COLOR = "#273142"
MUTED_COLOR = "#667085"

STAGE_COLORS = {
    "draft_proposal": "#4C78A8",
    "target_forward": "#F58518",
    "posthoc_guard": "#B279A2",
    "inline_guard_prefill": "#72B7B2",
    "inline_guard_proposal": "#54A24B",
    "inline_guard_commit": "#E45756",
    "other": "#BAB0AC",
}


def parse_args() -> argparse.Namespace:
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(
        description="Generate four SD_Guard PPT figures with matplotlib."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results",
        help="Directory containing SD_4 and SD_latency_test results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "ppt_figures" / "matplotlib",
        help="Output directory for SVG, PNG, and the figure-data manifest.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional path to a Chinese TTF/TTC/OTF font file.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_fonts(font_path: Path | None) -> str:
    preferred = (
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "SimHei",
        "Arial Unicode MS",
    )

    if font_path is not None:
        if not font_path.is_file():
            raise FileNotFoundError(f"Font file not found: {font_path}")
        font_manager.fontManager.addfont(str(font_path))
        selected = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        installed = {font.name for font in font_manager.fontManager.ttflist}
        selected = next((name for name in preferred if name in installed), "DejaVu Sans")
        if selected == "DejaVu Sans":
            print(
                "WARNING: no common CJK font was found. Chinese glyphs may be missing. "
                "Use --font-path to provide a Noto/Source Han CJK font.",
                file=sys.stderr,
            )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#98A2B3",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "svg.fonttype": "none",
        }
    )
    return selected


def style_axis(ax: Any, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: Any, output_dir: Path, stem: str, dpi: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [str(svg_path), str(png_path)]


def plot_figure1(data: dict[str, Any], output_dir: Path, dpi: int) -> list[str]:
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    x = list(range(len(rows)))
    width = 0.30
    target = [row["target_mean_s"] for row in rows]
    sd = [row["sd_mean_s"] for row in rows]

    target_bars = ax.bar(
        [value - width / 2 for value in x],
        target,
        width,
        label="Target-only",
        color=TARGET_COLOR,
    )
    sd_bars = ax.bar(
        [value + width / 2 for value in x],
        sd,
        width,
        label="Standard SD",
        color=SD_COLOR,
    )

    ax.set_title("vLLM 推测解码的端到端加速", fontsize=22, fontweight="bold", pad=18)
    ax.set_ylabel("平均端到端延迟（秒）", fontsize=14)
    ax.set_xticks(x, [row["label"] for row in rows], fontsize=14)
    ax.set_ylim(0, max(target) * 1.28)
    style_axis(ax)
    ax.legend(frameon=False, fontsize=13, loc="upper right")

    for bars in (target_bars, sd_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.2f} s",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="semibold",
            )

    for index, row in enumerate(rows):
        ax.text(
            index + width / 2,
            row["sd_mean_s"] * 0.53,
            f"{row['mean_speedup']:.2f}×",
            ha="center",
            va="center",
            fontsize=15,
            color="white",
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.015,
        "100 samples · 2×NVIDIA L20 · vLLM 0.31 · K=3 · batch=1 · BF16 · 模型加载与 warm-up 排除",
        ha="center",
        fontsize=10.5,
        color=MUTED_COLOR,
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.98, 0.98))
    return save_figure(fig, output_dir, "fig1_vllm_end_to_end_speedup", dpi)


def plot_figure2(data: dict[str, Any], output_dir: Path, dpi: int) -> list[str]:
    row_lookup = {
        (row["thinking"], row["mode"]): row for row in data["rows"]
    }
    panels = (
        ("accuracy_pct", "Accuracy ↑", (60, 100)),
        ("f1_pct", "F1 ↑", (60, 100)),
        ("fnr_pct", "漏报率 FNR ↓", (0, 12)),
        ("fpr_pct", "误报率 FPR ↓", (0, 75)),
    )
    mode_labels = [
        row_lookup[("think_on", mode)]["mode_label"] for mode in MODE_ORDER
    ]
    x = list(range(len(MODE_ORDER)))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2))
    fig.suptitle(
        "完整方法的 Strict 安全检测效果",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )

    for ax, (metric, title, bounds) in zip(axes.flat, panels):
        think_on_values = [
            row_lookup[("think_on", mode)][metric] for mode in MODE_ORDER
        ]
        think_off_values = [
            row_lookup[("think_off", mode)][metric] for mode in MODE_ORDER
        ]
        on_bars = ax.bar(
            [value - width / 2 for value in x],
            think_on_values,
            width,
            label="Think on",
            color=THINK_ON_COLOR,
        )
        off_bars = ax.bar(
            [value + width / 2 for value in x],
            think_off_values,
            width,
            label="Think off",
            color=THINK_OFF_COLOR,
        )
        ax.set_title(title, fontsize=15, fontweight="semibold")
        ax.set_xticks(x, mode_labels, rotation=12, ha="right", fontsize=10.5)
        ax.set_ylabel("百分比（%）")
        ax.set_ylim(*bounds)
        style_axis(ax)
        for bars in (on_bars, off_bars):
            for bar in bars:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (bounds[1] - bounds[0]) * 0.018,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.8,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.text(
        0.5,
        0.018,
        "full_trace · strict · n=100/cell；Standard SD 与 Inline Guard 在两种 thinking 条件下均保持 100/100 最终 token 完全一致",
        ha="center",
        fontsize=10.3,
        color=MUTED_COLOR,
    )
    fig.tight_layout(rect=(0.025, 0.055, 0.99, 0.91), h_pad=2.2, w_pad=1.6)
    return save_figure(fig, output_dir, "fig2_strict_safety_metrics", dpi)


def plot_figure3(data: dict[str, Any], output_dir: Path, dpi: int) -> list[str]:
    rows = data["rows"]
    ks = [row["k"] for row in rows]
    speedups = [row["throughput_speedup"] for row in rows]
    acceptance = [row["acceptance_rate_pct"] for row in rows]
    best_k = data["best_k"]

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    colors = [SD_COLOR if k == best_k else "#9CB7D4" for k in ks]
    bars = ax.bar(ks, speedups, width=0.58, color=colors, label="吞吐加速比")
    ax.axhline(1.0, color="#8B95A5", linewidth=1.2, linestyle="--")
    ax.set_xlabel("候选 token 数 K", fontsize=14)
    ax.set_ylabel("吞吐加速比（×）", fontsize=14, color=TARGET_COLOR)
    ax.set_xticks(ks)
    ax.set_ylim(0, max(speedups) * 1.32)
    style_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(
        ks,
        acceptance,
        color="#E45756",
        marker="o",
        markersize=8,
        linewidth=2.5,
        label="Draft 接受率",
    )
    ax2.set_ylabel("Draft 接受率（%）", fontsize=14, color="#E45756")
    ax2.set_ylim(0, max(acceptance) * 1.35)
    ax2.spines["top"].set_visible(False)

    for bar, value, k in zip(bars, speedups, ks):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.06,
            f"{value:.2f}×",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="semibold",
            color="white" if k == best_k else TEXT_COLOR,
        )
    for k, value in zip(ks, acceptance):
        ax2.text(k, value + 1.7, f"{value:.1f}%", ha="center", fontsize=11, color="#B42318")

    ax.set_title("K 值消融：K=3 在该微基准中取得最佳吞吐", fontsize=21, fontweight="bold", pad=18)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right", fontsize=12)
    fig.text(
        0.5,
        0.015,
        "单 Prompt 微基准 · RTX PRO 6000 Blackwell 96GB · vLLM 0.31 · 0.6B→14B · 256 tokens · warm-up 2 · repeats 10",
        ha="center",
        fontsize=10.3,
        color=MUTED_COLOR,
    )
    fig.tight_layout(rect=(0.035, 0.06, 0.98, 0.98))
    return save_figure(fig, output_dir, "fig3_k_ablation", dpi)


def plot_figure4(data: dict[str, Any], output_dir: Path, dpi: int) -> list[str]:
    stages = data["stages"]
    fig = plt.figure(figsize=(13.6, 8.4))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.05, 1.35), hspace=0.20)
    ax = fig.add_subplot(grid[0])
    table_ax = fig.add_subplot(grid[1])
    table_ax.axis("off")

    y_positions = [1, 0]
    labels = ["Standard SD", "Inline SD"]
    left = [0.0, 0.0]
    for stage in stages:
        values = [stage["standard_ms"], stage["inline_ms"]]
        if values == [0.0, 0.0]:
            continue
        bars = ax.barh(
            y_positions,
            values,
            left=left,
            height=0.52,
            label=stage["label"],
            color=STAGE_COLORS[stage["key"]],
        )
        for index, (bar, value) in enumerate(zip(bars, values)):
            if value >= 145:
                total = (
                    data["standard_total_ms"]
                    if index == 0
                    else data["inline_total_ms"]
                )
                ax.text(
                    left[index] + value / 2,
                    y_positions[index],
                    f"{value:.0f} ms\n{value / total * 100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    color="white" if stage["key"] != "other" else TEXT_COLOR,
                    fontweight="semibold",
                )
        left = [left[index] + values[index] for index in range(2)]

    ax.set_yticks(y_positions, labels, fontsize=12.5)
    ax.set_xlabel("阶段耗时（ms）", fontsize=12.5)
    ax.set_xlim(0, data["inline_total_ms"] * 1.16)
    style_axis(ax, grid_axis="x")
    ax.text(
        data["standard_total_ms"] + 35,
        1,
        f"总计 {data['standard_total_ms']:.2f} ms",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.text(
        data["inline_total_ms"] + 35,
        0,
        f"总计 {data['inline_total_ms']:.2f} ms",
        va="center",
        fontsize=11.5,
        fontweight="bold",
    )

    cell_text: list[list[str]] = []
    for stage in stages:
        standard = (
            "—"
            if stage["standard_ms"] == 0
            else f"{stage['standard_ms']:.2f} ms"
        )
        standard_share = (
            "—"
            if stage["standard_ms"] == 0
            else f"{stage['standard_share_pct']:.2f}%"
        )
        inline = (
            "—" if stage["inline_ms"] == 0 else f"{stage['inline_ms']:.2f} ms"
        )
        inline_share = (
            "—"
            if stage["inline_ms"] == 0
            else f"{stage['inline_share_pct']:.2f}%"
        )
        cell_text.append(
            [stage["label"], standard, standard_share, inline, inline_share]
        )
    cell_text.append(
        [
            "总延迟",
            f"{data['standard_total_ms']:.2f} ms",
            "100%",
            f"{data['inline_total_ms']:.2f} ms",
            "100%",
        ]
    )

    table = table_ax.table(
        cellText=cell_text,
        colLabels=["主要阶段", "Standard SD", "占比", "Inline SD", "占比"],
        cellLoc="right",
        colLoc="center",
        loc="center",
        colWidths=[0.28, 0.18, 0.12, 0.18, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.8)
    table.scale(1, 1.38)
    last_row = len(cell_text)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#E4E7EC")
        if row == 0:
            cell.set_facecolor("#F2F4F7")
            cell.set_text_props(fontweight="bold", ha="center")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")
        if row == last_row:
            cell.set_facecolor("#F9FAFB")
            cell.set_text_props(fontweight="bold")

    fig.suptitle(
        "在线 Guard 的阶段耗时拆分",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "独立同步诊断 · Think on · K=5 · 32 generated tokens（finish_reason=length）· 2×L20 · 非正式 100-sample 延迟",
        ha="center",
        fontsize=10.2,
        color=MUTED_COLOR,
    )
    fig.tight_layout(rect=(0.025, 0.04, 0.99, 0.955))
    return save_figure(fig, output_dir, "fig4_stage_latency_breakdown", dpi)


def main() -> None:
    args = parse_args()
    selected_font = configure_fonts(args.font_path)
    data = collect_figure_data(args.results_dir.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    generated.extend(plot_figure1(data["figure1"], output_dir, args.dpi))
    generated.extend(plot_figure2(data["figure2"], output_dir, args.dpi))
    generated.extend(plot_figure3(data["figure3"], output_dir, args.dpi))
    generated.extend(plot_figure4(data["figure4"], output_dir, args.dpi))

    manifest = {
        "renderer": "matplotlib",
        "font": selected_font,
        "results_dir": str(args.results_dir.resolve()),
        "generated_files": generated,
        "sources": data["sources"],
        "formal_config": data["formal_config"],
        "stage_config": data["stage_config"],
        "k_sweep_config": data["figure3"]["config"],
    }
    manifest_path = output_dir / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"Renderer: matplotlib; font: {selected_font}")
    print(f"Wrote {len(generated)} figure files to: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

