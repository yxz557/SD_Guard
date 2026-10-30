#!/usr/bin/env python3
"""Quickly verify that greedy and sampling use different decoding paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from sd_latency.generation import (
    GenerationResult,
    GenerationSettings,
    generate_assisted,
    generate_target_only,
)
from sd_latency.output_paths import resolve_experiment_output_dir
from sd_latency.runtime import (
    assert_same_tokenizer,
    collect_runtime_metadata,
    load_generation_model,
)


PINNED_TRANSFORMERS = "4.57.6"
DEFAULT_PROMPT = "请用两句话解释为什么天空通常看起来是蓝色的。"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run four short generations to verify that greedy and sampling are "
            "actually different for Target-only and SD."
        )
    )
    parser.add_argument(
        "--draft-model", default="/home/model_and_data/yxz/model/Qwen3-0.6B"
    )
    parser.add_argument(
        "--target-model", default="/home/model_and_data/yxz/model/Qwen3-4B"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def resolve_output_dir(
    requested: Path | None, *, seed: int, max_new_tokens: int
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolve_experiment_output_dir(
        requested,
        timestamp=timestamp,
        descriptor=f"mode-check_seed{seed}_tok{max_new_tokens}",
    )


def result_record(
    result: GenerationResult, tokenizer: Any
) -> dict[str, Any]:
    return {
        "token_ids": list(result.generated_ids),
        "text": tokenizer.decode(list(result.generated_ids), skip_special_tokens=False),
        "generated_token_count": result.generated_token_count,
        "total_ms": result.total_ms,
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Invalid sampling parameters")
    installed_transformers = importlib.metadata.version("transformers")
    if installed_transformers != PINNED_TRANSFORMERS:
        raise RuntimeError(
            f"Expected transformers=={PINNED_TRANSFORMERS}; found {installed_transformers}"
        )

    output_dir = resolve_output_dir(
        args.output_dir,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Target on {args.device}: {args.target_model}")
    target, tokenizer = load_generation_model(
        args.target_model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        revision=None,
        attention_implementation=args.attn_implementation,
    )
    print(f"Loading Draft on {args.device}: {args.draft_model}")
    draft, draft_tokenizer = load_generation_model(
        args.draft_model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        revision=None,
        attention_implementation=args.attn_implementation,
    )
    assert_same_tokenizer(tokenizer, draft_tokenizer)

    base = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        num_assistant_tokens=5,
        assistant_schedule="constant",
        assistant_confidence_threshold=0.4,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking=args.enable_thinking,
    )
    greedy = replace(base, do_sample=False)
    sampling = replace(base, do_sample=True)

    print("Running Target-only greedy/sample and SD greedy/sample...")
    target_greedy = generate_target_only(
        target, tokenizer, args.prompt, device=args.device, settings=greedy, seed=args.seed
    )
    target_sample = generate_target_only(
        target, tokenizer, args.prompt, device=args.device, settings=sampling, seed=args.seed
    )
    sd_greedy = generate_assisted(
        target,
        draft,
        tokenizer,
        args.prompt,
        target_device=args.device,
        draft_device=args.device,
        settings=greedy,
        seed=args.seed,
        trace=False,
    )
    sd_sample = generate_assisted(
        target,
        draft,
        tokenizer,
        args.prompt,
        target_device=args.device,
        draft_device=args.device,
        settings=sampling,
        seed=args.seed,
        trace=False,
    )

    results = {
        "target_only": {
            "greedy": result_record(target_greedy, tokenizer),
            "sampling": result_record(target_sample, tokenizer),
        },
        "sd": {
            "greedy": result_record(sd_greedy, tokenizer),
            "sampling": result_record(sd_sample, tokenizer),
        },
    }
    comparisons = {
        "target_greedy_differs_from_sampling": (
            target_greedy.generated_ids != target_sample.generated_ids
        ),
        "sd_greedy_differs_from_sampling": (
            sd_greedy.generated_ids != sd_sample.generated_ids
        ),
        "greedy_target_matches_sd": (
            target_greedy.generated_ids == sd_greedy.generated_ids
        ),
        "sampling_target_matches_sd": (
            target_sample.generated_ids == sd_sample.generated_ids
        ),
    }
    passed = (
        comparisons["target_greedy_differs_from_sampling"]
        and comparisons["sd_greedy_differs_from_sampling"]
    )
    report = {
        "passed": passed,
        "purpose": "mode-switch correctness check; not a latency benchmark",
        "runtime": collect_runtime_metadata(),
        "models": {"draft": args.draft_model, "target": args.target_model},
        "model_loading": {
            "dtype": args.dtype,
            "requested_attention_implementation": args.attn_implementation,
            "resolved_attention_implementation": {
                "draft": getattr(draft.config, "_attn_implementation", None),
                "target": getattr(target.config, "_attn_implementation", None),
            },
            "local_files_only": args.local_files_only,
        },
        "prompt": args.prompt,
        "seed": args.seed,
        "settings": asdict(base),
        "comparisons": comparisons,
        "outputs": results,
    }

    report_path = output_dir / "mode_check.json"
    outputs_path = output_dir / "output_comparison.txt"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_lines = [
        f"TARGET_ONLY GREEDY\t{json.dumps(results['target_only']['greedy']['text'], ensure_ascii=False)}",
        f"TARGET_ONLY SAMPLING\t{json.dumps(results['target_only']['sampling']['text'], ensure_ascii=False)}",
        f"SD GREEDY\t{json.dumps(results['sd']['greedy']['text'], ensure_ascii=False)}",
        f"SD SAMPLING\t{json.dumps(results['sd']['sampling']['text'], ensure_ascii=False)}",
    ]
    outputs_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    verdict = "PASS" if passed else "CHECK FAILED OR INCONCLUSIVE"
    print(f"\n{verdict}")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")
    print(f"Adjacent outputs: {outputs_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
