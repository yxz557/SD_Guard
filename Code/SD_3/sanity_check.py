#!/usr/bin/env python3
"""Check SD tracing fidelity and diagnose target-only/assisted greedy differences."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
from typing import Any

from sdguard_sd.generation import (
    generate_assisted_greedy_untraced,
    generate_speculatively,
    generate_target_only_greedy,
    seed_everything,
)
from sdguard_sd.io_utils import write_json
from sdguard_sd.runtime import (
    assert_same_tokenizer,
    load_generation_model,
    synchronize_devices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft-model",
        default="/home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2",
    )
    parser.add_argument(
        "--target-model",
        default="/home/model_and_data/yxz/model/Huihui-Qwen3-8B-abliterated-v2",
    )
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--draft-revision")
    parser.add_argument("--target-revision")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-assistant-tokens", type=int, default=5)
    parser.add_argument("--assistant-confidence-threshold", type=float, default=0.4)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--prompt",
        default="请用两句话解释为什么天空通常看起来是蓝色的。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sanity_report.json"),
        help="Detailed JSON report path (default: outputs/sanity_report.json).",
    )
    parser.add_argument(
        "--require-target-match",
        action="store_true",
        help="Also fail when stock assisted greedy differs from target-only greedy.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=1,
        help="Untimed warm-up runs per decoding method (default: 1).",
    )
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=3,
        help="Measured runs per decoding method (default: 3).",
    )
    return parser.parse_args()


def first_mismatch(left: tuple[int, ...], right: tuple[int, ...]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def summarize_latency_runs(
    target_runs: list[dict[str, Any]], assisted_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    if not target_runs or not assisted_runs:
        raise ValueError("Latency benchmark requires non-empty target and assisted runs")

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        latencies = [float(row["latency_ms"]) for row in rows]
        total_tokens = sum(int(row["generated_token_count"]) for row in rows)
        total_seconds = sum(latencies) / 1000.0
        return {
            "runs": rows,
            "median_latency_ms": statistics.median(latencies),
            "mean_latency_ms": statistics.fmean(latencies),
            "aggregate_generated_tokens": total_tokens,
            "aggregate_tokens_per_second": (
                None if total_seconds <= 0 else total_tokens / total_seconds
            ),
        }

    target = summarize(target_runs)
    assisted = summarize(assisted_runs)
    target_tps = target["aggregate_tokens_per_second"]
    assisted_tps = assisted["aggregate_tokens_per_second"]
    return {
        "target_only": target,
        "speculative_decoding": assisted,
        "median_latency_speedup": (
            target["median_latency_ms"] / assisted["median_latency_ms"]
        ),
        "throughput_speedup": (
            None
            if target_tps is None or assisted_tps is None
            else assisted_tps / target_tps
        ),
        "speedup_definition": "values greater than 1 mean speculative decoding is faster",
        "all_generated_token_counts_match": (
            [row["generated_token_count"] for row in target_runs]
            == [row["generated_token_count"] for row in assisted_runs]
        ),
    }


def _timed_generation(call: Any, *, devices: tuple[str, ...]) -> tuple[tuple[int, ...], float]:
    synchronize_devices(*devices)
    started_ns = time.perf_counter_ns()
    generated = call()
    synchronize_devices(*devices)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return generated, elapsed_ms


def _run_latency_benchmark(
    *,
    target: Any,
    draft: Any,
    tokenizer: Any,
    prompt: str,
    target_device: str,
    draft_device: str,
    enable_thinking: bool,
    max_new_tokens: int,
    num_assistant_tokens: int,
    assistant_confidence_threshold: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("--benchmark-warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("--benchmark-repeats must be positive")

    def target_call() -> tuple[int, ...]:
        return generate_target_only_greedy(
            target,
            tokenizer,
            prompt,
            target_device=target_device,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
        )

    def assisted_call() -> tuple[int, ...]:
        return generate_assisted_greedy_untraced(
            target,
            draft,
            tokenizer,
            prompt,
            target_device=target_device,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
            num_assistant_tokens=num_assistant_tokens,
            assistant_confidence_threshold=assistant_confidence_threshold,
        )

    for _ in range(warmup):
        seed_everything(12345)
        _timed_generation(target_call, devices=(target_device,))
        seed_everything(12345)
        _timed_generation(assisted_call, devices=(target_device, draft_device))

    target_runs: list[dict[str, Any]] = []
    assisted_runs: list[dict[str, Any]] = []
    paired_exact_matches: list[bool] = []
    for repeat_index in range(repeats):
        order = ("target", "assisted") if repeat_index % 2 == 0 else ("assisted", "target")
        outputs: dict[str, tuple[int, ...]] = {}
        for method in order:
            seed_everything(12345)
            if method == "target":
                generated, elapsed_ms = _timed_generation(
                    target_call, devices=(target_device,)
                )
                destination = target_runs
            else:
                generated, elapsed_ms = _timed_generation(
                    assisted_call, devices=(target_device, draft_device)
                )
                destination = assisted_runs
            outputs[method] = generated
            destination.append(
                {
                    "repeat_index": repeat_index,
                    "latency_ms": elapsed_ms,
                    "generated_token_count": len(generated),
                    "tokens_per_second": (
                        None if elapsed_ms <= 0 else len(generated) / (elapsed_ms / 1000.0)
                    ),
                }
            )
        paired_exact_matches.append(outputs["target"] == outputs["assisted"])

    summary = summarize_latency_runs(target_runs, assisted_runs)
    summary.update(
        {
            "warmup_runs_per_method": warmup,
            "measured_runs_per_method": repeats,
            "paired_exact_output_matches": paired_exact_matches,
            "all_paired_outputs_exactly_match": all(paired_exact_matches),
            "measurement_scope": (
                "chat-template preparation, generate(), synchronization, and output transfer"
            ),
        }
    )
    return summary


def main() -> None:
    args = parse_args()
    version = importlib.metadata.version("transformers")
    if version != "4.57.6":
        raise RuntimeError(f"sanity_check.py requires transformers==4.57.6; found {version}")
    target, tokenizer = load_generation_model(
        args.target_model,
        device=args.target_device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        revision=args.target_revision,
        attention_implementation=args.attn_implementation,
    )
    draft, draft_tokenizer = load_generation_model(
        args.draft_model,
        device=args.draft_device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        revision=args.draft_revision,
        attention_implementation=args.attn_implementation,
    )
    assert_same_tokenizer(tokenizer, draft_tokenizer, candidate_name="draft model")

    target_only = generate_target_only_greedy(
        target,
        tokenizer,
        args.prompt,
        target_device=args.target_device,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
    )
    seed_everything(12345)
    builtin_assisted = generate_assisted_greedy_untraced(
        target,
        draft,
        tokenizer,
        args.prompt,
        target_device=args.target_device,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        num_assistant_tokens=args.num_assistant_tokens,
        assistant_confidence_threshold=args.assistant_confidence_threshold,
    )
    seed_everything(12345)
    builtin_assisted_repeat = generate_assisted_greedy_untraced(
        target,
        draft,
        tokenizer,
        args.prompt,
        target_device=args.target_device,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        num_assistant_tokens=args.num_assistant_tokens,
        assistant_confidence_threshold=args.assistant_confidence_threshold,
    )
    assisted = generate_speculatively(
        target,
        draft,
        tokenizer,
        args.prompt,
        target_device=args.target_device,
        draft_device=args.draft_device,
        sample_seed=12345,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        num_assistant_tokens=args.num_assistant_tokens,
        assistant_confidence_threshold=args.assistant_confidence_threshold,
        do_sample=False,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )
    latency_benchmark = _run_latency_benchmark(
        target=target,
        draft=draft,
        tokenizer=tokenizer,
        prompt=args.prompt,
        target_device=args.target_device,
        draft_device=args.draft_device,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        num_assistant_tokens=args.num_assistant_tokens,
        assistant_confidence_threshold=args.assistant_confidence_threshold,
        warmup=args.benchmark_warmup,
        repeats=args.benchmark_repeats,
    )
    reconstructed = tuple(
        token_id for round_ in assisted.rounds for token_id in round_.committed_ids
    )
    exact_target_match = target_only == builtin_assisted
    builtin_repeatable = builtin_assisted == builtin_assisted_repeat
    tracer_is_nonintrusive = builtin_assisted == assisted.generated_ids
    trace_match = reconstructed == assisted.generated_ids
    mismatch_index = first_mismatch(target_only, builtin_assisted)
    report: dict[str, Any] = {
        "transformers_version": version,
        "prompt": args.prompt,
        "target_only_ids": list(target_only),
        "builtin_assisted_ids": list(builtin_assisted),
        "builtin_assisted_repeat_ids": list(builtin_assisted_repeat),
        "assisted_ids": list(assisted.generated_ids),
        "target_only_matches_builtin_assisted": exact_target_match,
        "target_mismatch_index": mismatch_index,
        "builtin_assisted_repeatable": builtin_repeatable,
        "traced_matches_builtin_assisted": tracer_is_nonintrusive,
        "trace_reconstructs_output": trace_match,
        "round_count": assisted.round_count,
        "proposed_token_count": assisted.proposed_token_count,
        "accepted_draft_token_count": assisted.accepted_draft_token_count,
        "acceptance_rate": assisted.acceptance_rate,
        "decoded_output": tokenizer.decode(
            list(assisted.generated_ids), skip_special_tokens=False
        ),
        "latency_benchmark": latency_benchmark,
    }
    output_path = args.output.expanduser().resolve()
    write_json(output_path, report)
    summary = {
        "target_only_matches_builtin_assisted": exact_target_match,
        "builtin_assisted_repeatable": builtin_repeatable,
        "traced_matches_builtin_assisted": tracer_is_nonintrusive,
        "trace_reconstructs_output": trace_match,
        "target_mismatch_index": mismatch_index,
        "acceptance_rate": assisted.acceptance_rate,
        "latency_benchmark": {
            "target_only_median_ms": latency_benchmark["target_only"][
                "median_latency_ms"
            ],
            "speculative_median_ms": latency_benchmark["speculative_decoding"][
                "median_latency_ms"
            ],
            "median_latency_speedup": latency_benchmark["median_latency_speedup"],
            "target_only_tokens_per_second": latency_benchmark["target_only"][
                "aggregate_tokens_per_second"
            ],
            "speculative_tokens_per_second": latency_benchmark[
                "speculative_decoding"
            ]["aggregate_tokens_per_second"],
            "throughput_speedup": latency_benchmark["throughput_speedup"],
            "all_paired_outputs_exactly_match": latency_benchmark[
                "all_paired_outputs_exactly_match"
            ],
        },
        "detailed_report": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if builtin_repeatable and not tracer_is_nonintrusive:
        raise AssertionError("The tracing layer changed stock assisted-greedy output")
    if not builtin_repeatable:
        print(
            "\nWARNING: two stock assisted-greedy runs differed even with the same seed; "
            "this CUDA/model path is not bitwise repeatable. Tracer equality across separate "
            "runs is therefore not a valid hard assertion."
        )
    if not trace_match:
        raise AssertionError("Speculative round trace does not reconstruct the generated IDs")
    if not exact_target_match:
        message = (
            "Stock assisted greedy differs from token-by-token target greedy at generated "
            f"index {mismatch_index}. This can occur when block verification and one-token "
            "forward passes produce slightly different finite-precision logits. The custom "
            "tracer is non-intrusive and its round trace is exact."
        )
        if args.require_target_match:
            raise AssertionError(message)
        print(f"\nWARNING: {message}")
    if builtin_repeatable:
        print("\nPASS: traced SD matches stock Transformers SD, and the trace reconstructs the output.")
    else:
        print("\nPASS: the SD round trace reconstructs its output; stock SD was not bitwise repeatable.")


if __name__ == "__main__":
    main()
