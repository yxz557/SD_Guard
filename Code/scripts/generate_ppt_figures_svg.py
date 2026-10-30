r"""Generate editable PPT-ready SVG figures with no third-party packages.

This version is designed for the local Windows checkout.  It uses only the
Python standard library and reads the same JSON data as the matplotlib version.

Run from the repository root::

    .\.venv\Scripts\python.exe Code\scripts\generate_ppt_figures_svg.py
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

from ppt_figure_data import (
    MODE_ORDER,
    THINKING_ORDER,
    collect_figure_data,
    default_repo_root,
)


WIDTH = 1600
HEIGHT = 900
FONT = "Microsoft YaHei, Noto Sans CJK SC, Source Han Sans SC, Arial, sans-serif"

TARGET_COLOR = "#4C78A8"
SD_COLOR = "#F58518"
THINK_ON_COLOR = "#4C78A8"
THINK_OFF_COLOR = "#59A14F"
TEXT_COLOR = "#273142"
MUTED_COLOR = "#667085"
GRID_COLOR = "#D9DEE7"

STAGE_COLORS = {
    "draft_proposal": "#4C78A8",
    "target_forward": "#F58518",
    "posthoc_guard": "#B279A2",
    "inline_guard_prefill": "#72B7B2",
    "inline_guard_proposal": "#54A24B",
    "inline_guard_commit": "#E45756",
    "other": "#BAB0AC",
}


class SVG:
    def __init__(self, title: str) -> None:
        self.parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
                f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
            ),
            f"<title>{html.escape(title)}</title>",
            "<defs>",
            "<filter id=\"shadow\" x=\"-10%\" y=\"-10%\" width=\"120%\" height=\"130%\">",
            "<feDropShadow dx=\"0\" dy=\"2\" stdDeviation=\"3\" flood-color=\"#101828\" flood-opacity=\"0.12\"/>",
            "</filter>",
            "</defs>",
            '<rect x="0" y="0" width="1600" height="900" fill="#FFFFFF"/>',
        ]

    def raw(self, value: str) -> None:
        self.parts.append(value)

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: str,
        *,
        stroke: str = "none",
        stroke_width: float = 1,
        rx: float = 0,
        opacity: float = 1,
    ) -> None:
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'rx="{rx}" opacity="{opacity}"/>'
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = GRID_COLOR,
        width: float = 1,
        dash: str | None = None,
    ) -> None:
        dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 24,
        fill: str = TEXT_COLOR,
        anchor: str = "start",
        weight: str = "normal",
        rotate: float | None = None,
        baseline: str = "middle",
    ) -> None:
        transform = "" if rotate is None else f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" dominant-baseline="{baseline}" '
            f'font-weight="{weight}"{transform}>{html.escape(value)}</text>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        fill: str,
        *,
        stroke: str = "white",
        stroke_width: float = 2,
    ) -> None:
        self.parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def polyline(
        self,
        points: Iterable[tuple[float, float]],
        *,
        stroke: str,
        width: float = 3,
        fill: str = "none",
    ) -> None:
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.parts.append(
            f'<polyline points="{point_text}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def close(self) -> str:
        return "\n".join(self.parts + ["</svg>", ""])


def parse_args() -> argparse.Namespace:
    repo_root = default_repo_root()
    parser = argparse.ArgumentParser(
        description="Generate four dependency-free SVG PPT figures."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=repo_root / "results"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "ppt_figures" / "local_svg",
    )
    return parser.parse_args()


def write_svg(output_dir: Path, stem: str, svg: SVG) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.svg"
    path.write_text(svg.close(), encoding="utf-8")
    return str(path)


def title_block(svg: SVG, title: str, subtitle: str) -> None:
    svg.text(800, 55, title, size=34, weight="bold", anchor="middle")
    svg.text(800, 96, subtitle, size=17, fill=MUTED_COLOR, anchor="middle")


def footer(svg: SVG, value: str) -> None:
    svg.line(80, 852, 1520, 852, stroke="#EAECF0")
    svg.text(800, 876, value, size=15, fill=MUTED_COLOR, anchor="middle")


def draw_figure1(data: dict[str, Any]) -> SVG:
    svg = SVG("vLLM 推测解码的端到端加速")
    title_block(svg, "vLLM 推测解码的端到端加速", "端到端 pipeline latency（含 posthoc Guard）")

    plot_x, plot_y, plot_w, plot_h = 170, 160, 1320, 570
    max_y = 55.0
    for tick in range(0, 56, 10):
        y = plot_y + plot_h - tick / max_y * plot_h
        svg.line(plot_x, y, plot_x + plot_w, y)
        svg.text(plot_x - 18, y, str(tick), size=16, anchor="end")
    svg.line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#98A2B3", width=1.5)
    svg.line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#98A2B3", width=1.5)
    svg.text(45, plot_y + plot_h / 2, "平均端到端延迟（秒）", size=19, anchor="middle", rotate=-90)

    centers = [510, 1150]
    bar_width = 170
    gap = 30
    for center, row in zip(centers, data["rows"]):
        values = [row["target_mean_s"], row["sd_mean_s"]]
        colors = [TARGET_COLOR, SD_COLOR]
        x_positions = [center - bar_width - gap / 2, center + gap / 2]
        for index, (x, value, color) in enumerate(zip(x_positions, values, colors)):
            height = value / max_y * plot_h
            y = plot_y + plot_h - height
            svg.rect(x, y, bar_width, height, color, rx=5)
            svg.text(x + bar_width / 2, y - 19, f"{value:.2f} s", size=19, anchor="middle", weight="bold")
            if index == 1:
                svg.text(x + bar_width / 2, y + height / 2, f"{row['mean_speedup']:.2f}×", size=24, fill="white", anchor="middle", weight="bold")
        svg.text(center, plot_y + plot_h + 44, row["label"], size=22, anchor="middle", weight="bold")

    svg.rect(1120, 125, 24, 24, TARGET_COLOR, rx=3)
    svg.text(1155, 137, "Target-only", size=18)
    svg.rect(1310, 125, 24, 24, SD_COLOR, rx=3)
    svg.text(1345, 137, "Standard SD", size=18)
    footer(svg, "100 samples · 2×NVIDIA L20 · vLLM 0.31 · K=3 · batch=1 · BF16 · 模型加载与 warm-up 排除")
    return svg


def draw_metric_panel(
    svg: SVG,
    rows: dict[tuple[str, str], dict[str, Any]],
    metric: str,
    title: str,
    bounds: tuple[float, float],
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    svg.rect(x, y, width, height, "#FFFFFF", stroke="#E4E7EC", rx=8)
    svg.text(x + width / 2, y + 28, title, size=22, anchor="middle", weight="bold")
    chart_x, chart_y = x + 60, y + 58
    chart_w, chart_h = width - 85, height - 108
    lower, upper = bounds
    for fraction in (0, 0.5, 1):
        value = lower + (upper - lower) * fraction
        tick_y = chart_y + chart_h - fraction * chart_h
        svg.line(chart_x, tick_y, chart_x + chart_w, tick_y)
        svg.text(chart_x - 10, tick_y, f"{value:.0f}", size=12, anchor="end")

    group_width = chart_w / len(MODE_ORDER)
    bar_width = 35
    for mode_index, mode in enumerate(MODE_ORDER):
        center = chart_x + group_width * (mode_index + 0.5)
        for thinking_index, thinking in enumerate(THINKING_ORDER):
            value = rows[(thinking, mode)][metric]
            bar_x = center + (-bar_width - 3 if thinking_index == 0 else 3)
            normalized = max(0, min(1, (value - lower) / (upper - lower)))
            bar_h = normalized * chart_h
            bar_y = chart_y + chart_h - bar_h
            color = THINK_ON_COLOR if thinking == "think_on" else THINK_OFF_COLOR
            svg.rect(bar_x, bar_y, bar_width, bar_h, color, rx=2)
            svg.text(bar_x + bar_width / 2, bar_y - 11, f"{value:.1f}", size=12, anchor="middle")
        label = rows[("think_on", mode)]["mode_label"]
        svg.text(center, chart_y + chart_h + 30, label, size=12, anchor="middle")


def draw_figure2(data: dict[str, Any]) -> SVG:
    svg = SVG("完整方法的 Strict 安全检测效果")
    title_block(svg, "完整方法的 Strict 安全检测效果", "full_trace · strict · n=100/cell")
    rows = {(row["thinking"], row["mode"]): row for row in data["rows"]}

    panels = (
        ("accuracy_pct", "Accuracy ↑", (60, 100), 70, 140),
        ("f1_pct", "F1 ↑", (60, 100), 815, 140),
        ("fnr_pct", "漏报率 FNR ↓", (0, 12), 70, 490),
        ("fpr_pct", "误报率 FPR ↓", (0, 75), 815, 490),
    )
    for metric, panel_title, bounds, x, y in panels:
        draw_metric_panel(svg, rows, metric, panel_title, bounds, x, y, 715, 315)

    svg.rect(610, 110, 22, 22, THINK_ON_COLOR, rx=3)
    svg.text(642, 121, "Think on", size=16)
    svg.rect(790, 110, 22, 22, THINK_OFF_COLOR, rx=3)
    svg.text(822, 121, "Think off", size=16)
    footer(svg, "Standard SD 与 Inline Guard 在两种 thinking 条件下均保持 100/100 最终 token 完全一致")
    return svg


def draw_figure3(data: dict[str, Any]) -> SVG:
    svg = SVG("K 值消融")
    title_block(svg, "K 值消融：K=3 在该微基准中取得最佳吞吐", "吞吐加速比与 Draft 接受率的权衡")
    rows = data["rows"]
    plot_x, plot_y, plot_w, plot_h = 170, 160, 1260, 560
    left_max, right_max = 2.2, 60.0

    for tick in (0, 0.5, 1.0, 1.5, 2.0):
        y = plot_y + plot_h - tick / left_max * plot_h
        svg.line(plot_x, y, plot_x + plot_w, y)
        svg.text(plot_x - 16, y, f"{tick:.1f}", size=16, anchor="end")
    svg.line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#98A2B3", width=1.5)
    svg.line(plot_x + plot_w, plot_y, plot_x + plot_w, plot_y + plot_h, stroke="#98A2B3", width=1.5)
    svg.line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#98A2B3", width=1.5)
    threshold_y = plot_y + plot_h - 1.0 / left_max * plot_h
    svg.line(plot_x, threshold_y, plot_x + plot_w, threshold_y, stroke="#8B95A5", width=2, dash="8 7")

    for tick in (0, 20, 40, 60):
        y = plot_y + plot_h - tick / right_max * plot_h
        svg.text(plot_x + plot_w + 18, y, str(tick), size=16)

    centers = [plot_x + plot_w * (index + 0.5) / len(rows) for index in range(len(rows))]
    acceptance_points: list[tuple[float, float]] = []
    for center, row in zip(centers, rows):
        bar_h = row["throughput_speedup"] / left_max * plot_h
        bar_y = plot_y + plot_h - bar_h
        color = SD_COLOR if row["k"] == data["best_k"] else "#9CB7D4"
        svg.rect(center - 70, bar_y, 140, bar_h, color, rx=5)
        label_fill = "white" if row["k"] == data["best_k"] else TEXT_COLOR
        svg.text(
            center,
            plot_y + plot_h - 28,
            f"{row['throughput_speedup']:.2f}×",
            size=19,
            fill=label_fill,
            anchor="middle",
            weight="bold",
        )
        svg.text(center, plot_y + plot_h + 42, f"K={row['k']}", size=20, anchor="middle", weight="bold")
        point_y = plot_y + plot_h - row["acceptance_rate_pct"] / right_max * plot_h
        acceptance_points.append((center, point_y))

    svg.polyline(acceptance_points, stroke="#E45756", width=4)
    for (center, point_y), row in zip(acceptance_points, rows):
        svg.circle(center, point_y, 8, "#E45756")
        svg.text(center, point_y - 24, f"{row['acceptance_rate_pct']:.1f}%", size=17, fill="#B42318", anchor="middle", weight="bold")

    svg.text(48, plot_y + plot_h / 2, "吞吐加速比（×）", size=19, anchor="middle", rotate=-90)
    svg.text(1550, plot_y + plot_h / 2, "Draft 接受率（%）", size=19, fill="#B42318", anchor="middle", rotate=90)
    svg.rect(1040, 120, 24, 24, "#9CB7D4", rx=3)
    svg.text(1074, 132, "吞吐加速比", size=17)
    svg.line(1250, 132, 1290, 132, stroke="#E45756", width=4)
    svg.circle(1270, 132, 6, "#E45756")
    svg.text(1300, 132, "Draft 接受率", size=17)
    footer(svg, "单 Prompt 微基准 · RTX PRO 6000 Blackwell 96GB · vLLM 0.31 · 0.6B→14B · 256 tokens · warm-up 2 · repeats 10")
    return svg


def draw_figure4(data: dict[str, Any]) -> SVG:
    svg = SVG("在线 Guard 的阶段耗时拆分")
    title_block(svg, "在线 Guard 的阶段耗时拆分", "32-token 同步阶段诊断：Standard SD vs Inline SD")

    plot_x, plot_y, plot_w = 230, 145, 1200
    max_total = data["inline_total_ms"] * 1.04
    y_values = {"standard": 185, "inline": 285}
    totals = {
        "standard": data["standard_total_ms"],
        "inline": data["inline_total_ms"],
    }
    svg.text(205, y_values["standard"] + 28, "Standard SD", size=19, anchor="end", weight="bold")
    svg.text(205, y_values["inline"] + 28, "Inline SD", size=19, anchor="end", weight="bold")

    for key, value_key in (("standard", "standard_ms"), ("inline", "inline_ms")):
        left = plot_x
        total = totals[key]
        for stage in data["stages"]:
            value = stage[value_key]
            if value <= 0:
                continue
            width = value / max_total * plot_w
            svg.rect(left, y_values[key], width, 56, STAGE_COLORS[stage["key"]], stroke="white", stroke_width=1)
            if width >= 118:
                svg.text(left + width / 2, y_values[key] + 21, f"{value:.0f} ms", size=15, fill="white" if stage["key"] != "other" else TEXT_COLOR, anchor="middle", weight="bold")
                svg.text(left + width / 2, y_values[key] + 42, f"{value / total * 100:.1f}%", size=13, fill="white" if stage["key"] != "other" else TEXT_COLOR, anchor="middle")
            left += width
        svg.text(left + 16, y_values[key] + 28, f"总计 {total:.2f} ms", size=18, weight="bold")

    table_x, table_y = 85, 400
    column_widths = [430, 250, 150, 250, 150]
    headers = ["主要阶段", "Standard SD", "占比", "Inline SD", "占比"]
    row_height = 47
    total_width = sum(column_widths)
    svg.rect(table_x, table_y, total_width, row_height, "#F2F4F7", stroke="#D0D5DD")
    cursor = table_x
    for header, width in zip(headers, column_widths):
        svg.text(cursor + width / 2, table_y + row_height / 2, header, size=17, anchor="middle", weight="bold")
        cursor += width

    rows = []
    for stage in data["stages"]:
        rows.append(
            [
                stage["label"],
                "—" if stage["standard_ms"] == 0 else f"{stage['standard_ms']:.2f} ms",
                "—" if stage["standard_ms"] == 0 else f"{stage['standard_share_pct']:.2f}%",
                "—" if stage["inline_ms"] == 0 else f"{stage['inline_ms']:.2f} ms",
                "—" if stage["inline_ms"] == 0 else f"{stage['inline_share_pct']:.2f}%",
            ]
        )
    rows.append(
        [
            "总延迟",
            f"{data['standard_total_ms']:.2f} ms",
            "100%",
            f"{data['inline_total_ms']:.2f} ms",
            "100%",
        ]
    )

    for row_index, row in enumerate(rows, start=1):
        y = table_y + row_height * row_index
        fill = "#F9FAFB" if row_index == len(rows) else "#FFFFFF"
        svg.rect(table_x, y, total_width, row_height, fill, stroke="#E4E7EC")
        cursor = table_x
        for column_index, (value, width) in enumerate(zip(row, column_widths)):
            anchor = "start" if column_index == 0 else "end"
            x = cursor + 16 if column_index == 0 else cursor + width - 16
            svg.text(
                x,
                y + row_height / 2,
                value,
                size=16,
                anchor=anchor,
                weight="bold" if row_index == len(rows) else "normal",
            )
            cursor += width

    footer(svg, "独立同步诊断 · Think on · K=5 · 32 generated tokens（finish_reason=length）· 2×L20 · 非正式 100-sample 延迟")
    return svg


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = collect_figure_data(results_dir)

    figures = {
        "fig1_vllm_end_to_end_speedup": draw_figure1(data["figure1"]),
        "fig2_strict_safety_metrics": draw_figure2(data["figure2"]),
        "fig3_k_ablation": draw_figure3(data["figure3"]),
        "fig4_stage_latency_breakdown": draw_figure4(data["figure4"]),
    }
    generated = [write_svg(output_dir, stem, svg) for stem, svg in figures.items()]

    manifest = {
        "renderer": "python-standard-library-svg",
        "results_dir": str(results_dir),
        "generated_files": generated,
        "sources": data["sources"],
        "formal_config": data["formal_config"],
        "stage_config": data["stage_config"],
        "k_sweep_config": data["figure3"]["config"],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote {len(generated)} SVG figures to: {output_dir}")
    for path in generated:
        print(path)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

