"""Load and normalize the SD_Guard results used by the PPT figures.

This module only uses the Python standard library.  Both plotting backends
(`generate_ppt_figures_matplotlib.py` and `generate_ppt_figures_svg.py`) import
it so the two versions always visualize the same values.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


THINKING_ORDER = ("think_on", "think_off")
MODE_ORDER = ("target_only", "speculative", "draft_only", "inline_guard")

MODE_LABELS = {
    "target_only": "Target-only",
    "speculative": "Standard SD",
    "draft_only": "Draft-only",
    "inline_guard": "Inline Guard",
}

THINKING_LABELS = {
    "think_on": "Think on",
    "think_off": "Think off",
}


def default_repo_root() -> Path:
    """Return the repository root for scripts stored under Code/scripts/."""

    return Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_cell(
    cells: list[dict[str, Any]], thinking: str, generation_mode: str
) -> dict[str, Any]:
    for cell in cells:
        if (
            cell.get("thinking") == thinking
            and cell.get("generation_mode") == generation_mode
        ):
            return cell
    raise KeyError(f"Missing cell: {thinking}/{generation_mode}")


def _formal_pipeline_stats(cell: dict[str, Any]) -> dict[str, Any]:
    stats = cell["formal"]["statistics"]
    if "pipeline_total_ms" in stats:
        return stats["pipeline_total_ms"]
    if "operational_pipeline_total_ms" in stats:
        return stats["operational_pipeline_total_ms"]
    raise KeyError(
        f"No end-to-end pipeline statistic for "
        f"{cell.get('thinking')}/{cell.get('generation_mode')}"
    )


def _load_figure1(latency_report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cells = latency_report["cells"]
    for thinking in THINKING_ORDER:
        target_stats = _formal_pipeline_stats(
            _find_cell(cells, thinking, "target_only")
        )
        sd_stats = _formal_pipeline_stats(
            _find_cell(cells, thinking, "speculative")
        )
        rows.append(
            {
                "thinking": thinking,
                "label": THINKING_LABELS[thinking],
                "target_mean_s": target_stats["mean"] / 1000.0,
                "target_median_s": target_stats["median"] / 1000.0,
                "target_p90_s": target_stats["p90"] / 1000.0,
                "sd_mean_s": sd_stats["mean"] / 1000.0,
                "sd_median_s": sd_stats["median"] / 1000.0,
                "sd_p90_s": sd_stats["p90"] / 1000.0,
                "mean_speedup": target_stats["mean"] / sd_stats["mean"],
            }
        )
    return {"rows": rows}


def _load_figure2(matrix_summary: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    judged = matrix_summary["judged_metrics"]
    for thinking in THINKING_ORDER:
        for mode in MODE_ORDER:
            cell = _find_cell(judged, thinking, mode)
            strict_full_trace = next(
                metric
                for metric in cell["metrics"]
                if metric["scope"] == "full_trace"
                and metric["policy"] == "strict"
            )
            rows.append(
                {
                    "thinking": thinking,
                    "thinking_label": THINKING_LABELS[thinking],
                    "mode": mode,
                    "mode_label": MODE_LABELS[mode],
                    "accuracy_pct": strict_full_trace["accuracy"] * 100.0,
                    "f1_pct": strict_full_trace["f1"] * 100.0,
                    "fnr_pct": strict_full_trace["false_negative_rate"] * 100.0,
                    "fpr_pct": strict_full_trace["false_positive_rate"] * 100.0,
                    "tp": strict_full_trace["tp"],
                    "tn": strict_full_trace["tn"],
                    "fp": strict_full_trace["fp"],
                    "fn": strict_full_trace["fn"],
                }
            )

    equivalence = {
        item["thinking"]: {
            "all_equal": item["speculative_vs_inline_generated_ids_all_equal"],
            "sample_count": item["matched_sample_count"],
        }
        for item in matrix_summary["speculative_inline_generation_equivalence"]
    }
    return {
        "rows": rows,
        "equivalence": equivalence,
        "judged_metrics_complete": matrix_summary["judged_metrics_complete"],
    }


def _load_figure3(k_sweep_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    directory_pattern = re.compile(r"^k(?P<k>\d+)-tok(?P<tokens>\d+)$")
    for directory in k_sweep_root.iterdir():
        if not directory.is_dir():
            continue
        match = directory_pattern.match(directory.name)
        if match is None:
            continue
        report = read_json(directory / "latency_report.json")
        aggregate = report["aggregate_latency"]["greedy"]
        speculative = report["speculative_decoding_metrics"]["greedy"]
        output_comparison = report["prompts"][0]["modes"]["greedy"][
            "target_vs_sd"
        ]
        rows.append(
            {
                "k": int(match.group("k")),
                "max_new_tokens": int(match.group("tokens")),
                "target_median_ms": aggregate["target_only"]["median_total_ms"],
                "sd_median_ms": aggregate["sd_untraced"]["median_total_ms"],
                "latency_speedup": aggregate["latency_speedup"],
                "throughput_speedup": aggregate["throughput_speedup"],
                "acceptance_rate_pct": speculative["draft_acceptance_rate"]
                * 100.0,
                "mean_acceptance_length": speculative["mean_acceptance_length"],
                "target_token_count": output_comparison["target_token_count"],
                "sd_token_count": output_comparison["sd_token_count"],
                "exact_output_match": output_comparison["exact_token_match"],
            }
        )
    rows.sort(key=lambda item: item["k"])
    if not rows:
        raise FileNotFoundError(f"No K-sweep reports found under {k_sweep_root}")

    best = max(rows, key=lambda item: item["throughput_speedup"])
    sample_report = read_json(
        k_sweep_root / f"k{best['k']}-tok{best['max_new_tokens']}" / "latency_report.json"
    )
    return {
        "rows": rows,
        "best_k": best["k"],
        "config": {
            "backend": sample_report["backend"],
            "vllm_version": sample_report["runtime"]["target_only_worker"][
                "vllm_version"
            ],
            "gpu": sample_report["runtime"]["target_only_worker"]["gpus"][0][
                "name"
            ],
            "draft_model": Path(sample_report["models"]["draft"]).name,
            "target_model": Path(sample_report["models"]["target"]).name,
            "dtype": sample_report["engine"]["dtype"],
            "batch_size": sample_report["benchmark"]["batch_size"],
            "warmup_runs": sample_report["benchmark"]["warmup_runs_per_batch"],
            "measured_runs": sample_report["benchmark"][
                "measured_runs_per_batch"
            ],
            "enable_thinking": sample_report["settings"]["enable_thinking"],
            "prompt_token_count": sample_report["prompts"][0][
                "prompt_token_count"
            ],
            "prefix_caching": sample_report["engine"]["enable_prefix_caching"],
        },
    }


def _load_figure4(stage_report: dict[str, Any]) -> dict[str, Any]:
    think_on = stage_report["diagnostic"]["think_on"]
    standard = think_on["standard_sd"]
    inline = think_on["inline_sd"]

    standard_values = {
        "draft_proposal": standard["stages"]["draft_proposal"]["total_ms"],
        "target_forward": standard["stages"]["target_forward"]["total_ms"],
        "posthoc_guard": standard["result"]["guard_rpc_ms"],
        "inline_guard_prefill": 0.0,
        "inline_guard_proposal": 0.0,
        "inline_guard_commit": 0.0,
    }
    standard_values["other"] = max(
        0.0, standard["operation_wall_ms"] - sum(standard_values.values())
    )

    inline_guard = inline["result"]["guard"]
    inline_values = {
        "draft_proposal": inline["stages"]["draft_proposal"]["total_ms"],
        "target_forward": inline["stages"]["target_forward"]["total_ms"],
        "posthoc_guard": 0.0,
        "inline_guard_prefill": inline_guard["prefill_model_ms"],
        "inline_guard_proposal": inline_guard["proposal_model_ms"],
        "inline_guard_commit": inline_guard["commit_model_ms"],
    }
    inline_values["other"] = max(
        0.0, inline["operation_wall_ms"] - sum(inline_values.values())
    )

    stages = [
        {
            "key": "draft_proposal",
            "label": "Draft proposal",
            "standard_ms": standard_values["draft_proposal"],
            "inline_ms": inline_values["draft_proposal"],
        },
        {
            "key": "target_forward",
            "label": "Target forward",
            "standard_ms": standard_values["target_forward"],
            "inline_ms": inline_values["target_forward"],
        },
        {
            "key": "posthoc_guard",
            "label": "Posthoc Guard",
            "standard_ms": standard_values["posthoc_guard"],
            "inline_ms": inline_values["posthoc_guard"],
        },
        {
            "key": "inline_guard_prefill",
            "label": "Inline Guard prefill",
            "standard_ms": standard_values["inline_guard_prefill"],
            "inline_ms": inline_values["inline_guard_prefill"],
        },
        {
            "key": "inline_guard_proposal",
            "label": "Inline Guard proposal",
            "standard_ms": standard_values["inline_guard_proposal"],
            "inline_ms": inline_values["inline_guard_proposal"],
        },
        {
            "key": "inline_guard_commit",
            "label": "Inline Guard commit",
            "standard_ms": standard_values["inline_guard_commit"],
            "inline_ms": inline_values["inline_guard_commit"],
        },
        {
            "key": "other",
            "label": "Other",
            "standard_ms": standard_values["other"],
            "inline_ms": inline_values["other"],
        },
    ]

    standard_total = standard["operation_wall_ms"]
    inline_total = inline["operation_wall_ms"]
    for stage in stages:
        stage["standard_share_pct"] = stage["standard_ms"] / standard_total * 100.0
        stage["inline_share_pct"] = stage["inline_ms"] / inline_total * 100.0

    return {
        "thinking": "think_on",
        "generated_token_count": standard["result"]["generated_token_count"],
        "finish_reason": standard["result"]["finish_reason"],
        "standard_total_ms": standard_total,
        "inline_total_ms": inline_total,
        "standard_round_count": standard["result"]["speculative_decoding"][
            "round_count"
        ],
        "standard_proposed_token_count": standard["result"][
            "speculative_decoding"
        ]["proposed_token_count"],
        "standard_accepted_token_count": standard["result"][
            "speculative_decoding"
        ]["accepted_draft_token_count"],
        "stages": stages,
    }


def collect_figure_data(results_dir: Path) -> dict[str, Any]:
    """Collect every value needed by the four PPT figures."""

    formal_root = results_dir / "SD_4" / "sd4_formal_k3_100"
    stage_root = results_dir / "SD_4" / "sd4_stage_timing_warmup"
    k_sweep_root = (
        results_dir
        / "SD_latency_test"
        / "pro6000"
        / "vllm-b1-tok256-k-sweep"
    )

    formal_config = read_json(formal_root / "experiment_config.json")
    matrix_summary = read_json(formal_root / "analysis" / "matrix_summary.json")
    latency_report = read_json(formal_root / "analysis" / "latency_report.json")
    stage_config = read_json(stage_root / "experiment_config.json")
    stage_report = read_json(stage_root / "analysis" / "stage_timing_report.json")

    return {
        "figure1": _load_figure1(latency_report),
        "figure2": _load_figure2(matrix_summary),
        "figure3": _load_figure3(k_sweep_root),
        "figure4": _load_figure4(stage_report),
        "formal_config": formal_config,
        "stage_config": stage_config,
        "sources": {
            "formal_config": str(formal_root / "experiment_config.json"),
            "matrix_summary": str(formal_root / "analysis" / "matrix_summary.json"),
            "latency_report": str(formal_root / "analysis" / "latency_report.json"),
            "k_sweep_root": str(k_sweep_root),
            "stage_report": str(
                stage_root / "analysis" / "stage_timing_report.json"
            ),
        },
    }

