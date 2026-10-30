#!/usr/bin/env python3
"""Compare SD_0 against target-only and profile the SD round phases."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from sd_latency.generation import (
    GenerationResult,
    GenerationSettings,
    generate_assisted,
    generate_target_only,
)
from sd_latency.output_paths import resolve_experiment_output_dir
from sd_latency.reporting import compare_outputs, summarize_pair
from sd_latency.runtime import (
    assert_same_tokenizer,
    collect_runtime_metadata,
    load_generation_model,
)


PINNED_TRANSFORMERS = "4.57.6"
DEFAULT_PROMPT = "请用两句话解释为什么天空通常看起来是蓝色的。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone SD_0 correctness, latency, trace-overhead, and phase benchmark."
    )
    parser.add_argument(
        "--draft-model",
        default="/home/model_and_data/yxz/model/Qwen3-0.6B",
    )
    parser.add_argument(
        "--target-model",
        default="/home/model_and_data/yxz/model/Qwen3-4B",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Both Draft and Target are loaded on this device (default: cuda:0).",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--draft-revision")
    parser.add_argument("--target-revision")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-assistant-tokens", type=int, default=5)
    parser.add_argument(
        "--assistant-schedule",
        default="constant",
        choices=("constant", "heuristic", "heuristic_transient"),
    )
    parser.add_argument("--assistant-confidence-threshold", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("greedy", "sample", "both"),
        default="both",
        help=(
            "Greedy gives the cleanest output/latency comparison; sample reproduces SD_0's "
            "temperature/top-p/top-k path. The default runs both."
        ),
    )
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to test; repeat the option for multiple prompts.",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="UTF-8 text (one prompt per line) or JSONL with id/prompt fields.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Exact output directory. If omitted, a unique experiment directory is "
            "created under <repository>/SD_latency_test/outputs/."
        ),
    )
    parser.add_argument(
        "--run-name",
        help="Optional label included in the automatically generated output directory name.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0 or args.num_assistant_tokens <= 0:
        raise ValueError("Token limits must be positive")
    if args.benchmark_warmup < 0 or args.benchmark_repeats <= 0:
        raise ValueError("Warm-up must be non-negative and repeats must be positive")
    if not 0 <= args.assistant_confidence_threshold <= 1:
        raise ValueError("--assistant-confidence-threshold must be in [0, 1]")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Invalid sampling parameters")
    installed = importlib.metadata.version("transformers")
    if installed != PINNED_TRANSFORMERS:
        raise RuntimeError(
            f"This tracer requires transformers=={PINNED_TRANSFORMERS}; found {installed}"
        )


def _load_prompts(args: argparse.Namespace) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for index, prompt in enumerate(args.prompt or []):
        prompts.append({"id": f"cli-{index}", "prompt": prompt})
    if args.prompts_file is not None:
        source = args.prompts_file.expanduser().resolve()
        for line_number, raw_line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                value = json.loads(line)
                prompt = value.get("prompt")
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError(f"Missing prompt at {source}:{line_number}")
                prompt_id = str(value.get("id", f"file-{line_number}"))
            else:
                prompt = line
                prompt_id = f"file-{line_number}"
            prompts.append({"id": prompt_id, "prompt": prompt})
    if not prompts:
        prompts.append({"id": "default", "prompt": DEFAULT_PROMPT})
    return prompts


def _result_row(repeat_index: int, result: GenerationResult) -> dict[str, Any]:
    return {
        "repeat_index": repeat_index,
        "preparation_ms": result.preparation_ms,
        "generate_ms": result.generate_ms,
        "output_transfer_ms": result.output_transfer_ms,
        "total_ms": result.total_ms,
        "generated_token_count": result.generated_token_count,
        "tokens_per_second": result.tokens_per_second,
    }


def _benchmark_pair(
    *,
    target_call: Callable[[], GenerationResult],
    assisted_call: Callable[[], GenerationResult],
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], GenerationResult, GenerationResult]:
    for _ in range(warmup):
        target_call()
        assisted_call()

    target_rows: list[dict[str, Any]] = []
    assisted_rows: list[dict[str, Any]] = []
    first_target: GenerationResult | None = None
    first_assisted: GenerationResult | None = None
    paired_exact_matches: list[bool] = []
    for repeat_index in range(repeats):
        order = ("target", "sd") if repeat_index % 2 == 0 else ("sd", "target")
        outputs: dict[str, GenerationResult] = {}
        for method in order:
            result = target_call() if method == "target" else assisted_call()
            outputs[method] = result
            if method == "target":
                target_rows.append(_result_row(repeat_index, result))
                if first_target is None:
                    first_target = result
            else:
                assisted_rows.append(_result_row(repeat_index, result))
                if first_assisted is None:
                    first_assisted = result
        paired_exact_matches.append(
            outputs["target"].generated_ids == outputs["sd"].generated_ids
        )

    if first_target is None or first_assisted is None:  # pragma: no cover
        raise RuntimeError("Benchmark produced no measured runs")
    summary = summarize_pair(target_rows, assisted_rows)
    summary["paired_exact_output_matches"] = paired_exact_matches
    summary["all_paired_outputs_exactly_match"] = all(paired_exact_matches)
    return summary, first_target, first_assisted


def _phase_profile(
    result: GenerationResult, untraced_median_ms: float
) -> dict[str, Any]:
    phases = result.phase_summary()
    trace_overhead_ms = result.total_ms - untraced_median_ms
    phases.update(
        {
            "input_preparation_ms": result.preparation_ms,
            "generate_total_ms": result.generate_ms,
            "output_transfer_ms": result.output_transfer_ms,
            "end_to_end_ms": result.total_ms,
            "untraced_sd_median_ms": untraced_median_ms,
            "trace_overhead_vs_untraced_median_ms": trace_overhead_ms,
            "trace_slowdown_ratio": (
                None if untraced_median_ms <= 0 else result.total_ms / untraced_median_ms
            ),
            "timing_note": (
                "Draft and verify phase boundaries synchronize CUDA intentionally. "
                "target_forward_cuda_ms is a subset of verify_and_decide_ms, not additive."
            ),
        }
    )
    return phases


def _mode_settings(base: GenerationSettings, mode: str) -> GenerationSettings:
    return replace(base, do_sample=(mode == "sample"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", args.run_name).strip("-._")
        if not safe_label:
            raise ValueError("--run-name must contain at least one letter or number")
        descriptor = safe_label
    else:
        descriptor = (
            f"{args.benchmark_mode}_seed{args.seed}_tok{args.max_new_tokens}"
        )
    return resolve_experiment_output_dir(
        args.output_dir,
        timestamp=timestamp,
        descriptor=descriptor,
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    prompts = _load_prompts(args)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_started_ns = time.perf_counter_ns()
    print(f"Loading Target on {args.device}: {args.target_model}")
    target, tokenizer = load_generation_model(
        args.target_model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        revision=args.target_revision,
        attention_implementation=args.attn_implementation,
    )
    print(f"Loading Draft on {args.device}: {args.draft_model}")
    draft, draft_tokenizer = load_generation_model(
        args.draft_model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        revision=args.draft_revision,
        attention_implementation=args.attn_implementation,
    )
    assert_same_tokenizer(tokenizer, draft_tokenizer)
    model_load_ms = (time.perf_counter_ns() - load_started_ns) / 1_000_000.0

    base_settings = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        num_assistant_tokens=args.num_assistant_tokens,
        assistant_schedule=args.assistant_schedule,
        assistant_confidence_threshold=args.assistant_confidence_threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
    )
    modes = ["greedy", "sample"] if args.benchmark_mode == "both" else [args.benchmark_mode]
    report: dict[str, Any] = {
        "schema_version": 1,
        "runtime": collect_runtime_metadata(),
        "model_load_ms": model_load_ms,
        "models": {"draft": args.draft_model, "target": args.target_model},
        "model_placement": {"draft": args.device, "target": args.device},
        "model_loading": {
            "dtype": args.dtype,
            "requested_attention_implementation": args.attn_implementation,
            "resolved_attention_implementation": {
                "draft": getattr(draft.config, "_attn_implementation", None),
                "target": getattr(target.config, "_attn_implementation", None),
            },
            "local_files_only": args.local_files_only,
            "revisions": {
                "draft": args.draft_revision,
                "target": args.target_revision,
            },
        },
        "settings": asdict(base_settings),
        "benchmark": {
            "modes": modes,
            "warmup_runs_per_method": args.benchmark_warmup,
            "measured_runs_per_method": args.benchmark_repeats,
            "ordering": "alternates target-first and SD-first to reduce order bias",
        },
        "prompts": [],
    }
    output_pair_lines: list[str] = []
    trace_rows: list[dict[str, Any]] = []
    aggregate_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {"target": [], "sd": []} for mode in modes
    }

    for prompt_index, prompt_record in enumerate(prompts):
        prompt_id = prompt_record["id"]
        prompt = prompt_record["prompt"]
        prompt_seed = args.seed + prompt_index
        prompt_report: dict[str, Any] = {
            "id": prompt_id,
            "prompt": prompt,
            "seed": prompt_seed,
            "modes": {},
        }
        greedy_outputs: tuple[GenerationResult, GenerationResult] | None = None

        for mode in modes:
            settings = _mode_settings(base_settings, mode)

            def target_call(settings: GenerationSettings = settings) -> GenerationResult:
                return generate_target_only(
                    target,
                    tokenizer,
                    prompt,
                    device=args.device,
                    settings=settings,
                    seed=prompt_seed,
                )

            def assisted_call(settings: GenerationSettings = settings) -> GenerationResult:
                return generate_assisted(
                    target,
                    draft,
                    tokenizer,
                    prompt,
                    target_device=args.device,
                    draft_device=args.device,
                    settings=settings,
                    seed=prompt_seed,
                    trace=False,
                )

            latency, first_target, first_assisted = _benchmark_pair(
                target_call=target_call,
                assisted_call=assisted_call,
                warmup=args.benchmark_warmup,
                repeats=args.benchmark_repeats,
            )
            traced = generate_assisted(
                target,
                draft,
                tokenizer,
                prompt,
                target_device=args.device,
                draft_device=args.device,
                settings=settings,
                seed=prompt_seed,
                trace=True,
            )
            reconstructed = tuple(
                token_id for round_ in traced.rounds for token_id in round_.committed_ids
            )
            if reconstructed != traced.generated_ids:
                raise AssertionError(
                    f"{prompt_id}/{mode}: SD round trace does not reconstruct its output"
                )
            traced_matches_untraced = traced.generated_ids == first_assisted.generated_ids
            if not traced_matches_untraced:
                print(
                    f"WARNING: {prompt_id}/{mode}: traced SD differs from the measured "
                    "untraced SD run with the same seed. Check CUDA determinism before "
                    "attributing the difference to the tracer."
                )
            target_text = tokenizer.decode(
                list(first_target.generated_ids), skip_special_tokens=False
            )
            assisted_text = tokenizer.decode(
                list(first_assisted.generated_ids), skip_special_tokens=False
            )
            traced_text = tokenizer.decode(
                list(traced.generated_ids), skip_special_tokens=False
            )
            mode_report = {
                "latency": latency,
                "target_vs_sd": compare_outputs(
                    first_target.generated_ids,
                    first_assisted.generated_ids,
                    target_text,
                    assisted_text,
                ),
                "trace_validation": {
                    "traced_matches_untraced_sd": traced_matches_untraced,
                    "trace_reconstructs_traced_output": True,
                    "traced_token_count": traced.generated_token_count,
                },
                "traced_phase_profile": _phase_profile(
                    traced, latency["sd_untraced"]["median_total_ms"]
                ),
                "outputs": {
                    "target_only": target_text,
                    "sd_untraced": assisted_text,
                    "sd_traced": traced_text,
                },
            }
            prompt_report["modes"][mode] = mode_report
            aggregate_rows[mode]["target"].extend(latency["target_only"]["runs"])
            aggregate_rows[mode]["sd"].extend(latency["sd_untraced"]["runs"])
            for round_ in traced.rounds:
                trace_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt_index": prompt_index,
                        "mode": mode,
                        **round_.to_dict(tokenizer),
                    }
                )
            if mode == "greedy":
                greedy_outputs = (first_target, first_assisted)

        if greedy_outputs is None:
            greedy_settings = _mode_settings(base_settings, "greedy")
            greedy_target = generate_target_only(
                target,
                tokenizer,
                prompt,
                device=args.device,
                settings=greedy_settings,
                seed=prompt_seed,
            )
            greedy_sd = generate_assisted(
                target,
                draft,
                tokenizer,
                prompt,
                target_device=args.device,
                draft_device=args.device,
                settings=greedy_settings,
                seed=prompt_seed,
                trace=False,
            )
            greedy_outputs = (greedy_target, greedy_sd)
            prompt_report["greedy_output_only"] = compare_outputs(
                greedy_target.generated_ids,
                greedy_sd.generated_ids,
                tokenizer.decode(list(greedy_target.generated_ids), skip_special_tokens=False),
                tokenizer.decode(list(greedy_sd.generated_ids), skip_special_tokens=False),
            )

        target_output = tokenizer.decode(
            list(greedy_outputs[0].generated_ids), skip_special_tokens=False
        )
        sd_output = tokenizer.decode(
            list(greedy_outputs[1].generated_ids), skip_special_tokens=False
        )
        output_pair_lines.append(
            f"{prompt_index:04d} PROMPT\t{json.dumps(prompt, ensure_ascii=False)}"
        )
        output_pair_lines.append(
            f"{prompt_index:04d} TARGET_ONLY\t{json.dumps(target_output, ensure_ascii=False)}"
        )
        output_pair_lines.append(
            f"{prompt_index:04d} SD\t{json.dumps(sd_output, ensure_ascii=False)}"
        )
        output_pair_lines.append("")
        report["prompts"].append(prompt_report)

    report["aggregate_latency"] = {
        mode: summarize_pair(rows["target"], rows["sd"])
        for mode, rows in aggregate_rows.items()
    }

    report_path = output_dir / "latency_report.json"
    pairs_path = output_dir / "output_pairs.txt"
    rounds_path = output_dir / "sd_round_trace.jsonl"
    _write_json(report_path, report)
    pairs_path.write_text("\n".join(output_pair_lines), encoding="utf-8")
    rounds_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace_rows),
        encoding="utf-8",
    )

    console_summary = {
        "report": str(report_path),
        "adjacent_output_pairs": str(pairs_path),
        "round_trace": str(rounds_path),
        "aggregate_latency": {
            mode: {
                "latency_speedup": summary["latency_speedup"],
                "throughput_speedup": summary["throughput_speedup"],
            }
            for mode, summary in report["aggregate_latency"].items()
        },
        "prompts": [
            {
                "id": row["id"],
                "modes": {
                    mode: {
                        "exact_greedy_match": (
                            details["target_vs_sd"]["exact_token_match"]
                            if mode == "greedy"
                            else None
                        ),
                        "latency_speedup": details["latency"]["latency_speedup"],
                        "throughput_speedup": details["latency"]["throughput_speedup"],
                        "trace_slowdown_ratio": details["traced_phase_profile"][
                            "trace_slowdown_ratio"
                        ],
                    }
                    for mode, details in row["modes"].items()
                },
            }
            for row in report["prompts"]
        ],
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
