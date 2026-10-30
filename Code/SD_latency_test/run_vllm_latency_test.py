#!/usr/bin/env python3
"""Compare vLLM Target-only against vLLM draft-model speculative decoding."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sd_latency.output_paths import resolve_experiment_output_dir
from sd_latency.reporting import compare_outputs, summarize_pair
from sd_latency.runtime import assert_same_tokenizer, normalize_model_source
from sd_latency.vllm_worker import PINNED_VLLM


DEFAULT_PROMPT = "请用两句话解释为什么天空通常看起来是蓝色的。"
PROJECT_DIR = Path(__file__).resolve().parent
MINIMUM_PYTHON = (3, 10)
MAXIMUM_PYTHON_EXCLUSIVE = (3, 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark vLLM Target-only against vLLM's official draft_model "
            "speculative-decoding path using isolated engine processes."
        )
    )
    parser.add_argument(
        "--draft-model",
        default="/home/model_and_data/yxz/model/Qwen3-0.6B",
    )
    parser.add_argument(
        "--target-model",
        default="/home/model_and_data/yxz/model/Qwen3-4B",
    )
    parser.add_argument("--draft-revision")
    parser.add_argument("--target-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--num-speculative-tokens",
        "--num-assistant-tokens",
        dest="num_speculative_tokens",
        type=int,
        default=5,
        help="Maximum draft length K. The second option name is a compatibility alias.",
    )
    parser.add_argument(
        "--draft-sample-method",
        choices=("greedy", "probabilistic"),
        default="greedy",
        help="vLLM's public draft_model example uses the default greedy drafter.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=("greedy", "sample", "both"),
        default="greedy",
    )
    parser.add_argument("--benchmark-warmup", type=int, default=2)
    parser.add_argument("--benchmark-repeats", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of distinct prompts submitted in one llm.generate() call.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disabled by default so repeated benchmark prompts cannot reuse prompt KV.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep false for the normal target-runner CUDA-graph path. vLLM 0.31's "
            "plain draft-model speculator does not yet support CUDA graphs."
        ),
    )
    parser.add_argument(
        "--engine-order",
        choices=("target-first", "sd-first"),
        default="target-first",
    )
    parser.add_argument(
        "--profile-vllm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After formal timing, run a same-shape diagnostic baseline and one SD "
            "request under vLLM's official PyTorch profiler; write a GPU-kernel "
            "stage breakdown plus raw trace."
        ),
    )
    parser.add_argument(
        "--profile-max-new-tokens",
        type=int,
        default=64,
        help="Output-token cap for the two diagnostic requests (default: 64).",
    )
    parser.add_argument("--required-vllm-version", default=PINNED_VLLM)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    current_python = sys.version_info[:2]
    if not MINIMUM_PYTHON <= current_python < MAXIMUM_PYTHON_EXCLUSIVE:
        raise RuntimeError(
            "vllm==0.31.0 requires Python >=3.10,<3.15; "
            f"this interpreter is Python {sys.version.split()[0]} at {sys.executable}. "
            "Create the vLLM environment explicitly with Python 3.10 or 3.11."
        )
    if args.max_new_tokens <= 0 or args.num_speculative_tokens <= 0:
        raise ValueError("Token limits must be positive")
    if args.benchmark_warmup < 0 or args.benchmark_repeats <= 0:
        raise ValueError("Warm-up must be non-negative and repeats must be positive")
    if args.batch_size <= 0 or args.tensor_parallel_size <= 0:
        raise ValueError("Batch size and tensor parallel size must be positive")
    if args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if args.max_num_seqs is not None and args.max_num_seqs < args.batch_size:
        raise ValueError("--max-num-seqs cannot be smaller than --batch-size")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Invalid sampling parameters")
    if not args.required_vllm_version:
        raise ValueError("--required-vllm-version cannot be empty")
    if args.profile_max_new_tokens <= 0:
        raise ValueError("--profile-max-new-tokens must be positive")


def _load_prompt_texts(args: argparse.Namespace) -> list[dict[str, str]]:
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
    ids = [record["id"] for record in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt IDs must be unique")
    if len(prompts) % args.batch_size != 0:
        raise ValueError(
            f"Prompt count ({len(prompts)}) must be divisible by batch size "
            f"({args.batch_size}) so every measured batch has the same shape"
        )
    return prompts


def _coerce_token_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, list)
        and len(encoded) == 1
        and isinstance(encoded[0], list)
    ):
        encoded = encoded[0]
    if not isinstance(encoded, list) or any(not isinstance(value, int) for value in encoded):
        raise TypeError("Chat template did not return a flat list of token IDs")
    return [int(value) for value in encoded]


def _prepare_prompts(
    args: argparse.Namespace,
    prompt_texts: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The vLLM environment must also provide Transformers for Qwen chat templates"
        ) from exc

    target_source = normalize_model_source(args.target_model)
    draft_source = normalize_model_source(args.draft_model)
    target_kwargs: dict[str, Any] = {"local_files_only": args.local_files_only}
    draft_kwargs: dict[str, Any] = {"local_files_only": args.local_files_only}
    if args.target_revision is not None:
        target_kwargs["revision"] = args.target_revision
    if args.draft_revision is not None:
        draft_kwargs["revision"] = args.draft_revision
    target_tokenizer = AutoTokenizer.from_pretrained(target_source, **target_kwargs)
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_source, **draft_kwargs)
    assert_same_tokenizer(target_tokenizer, draft_tokenizer)

    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(prompt_texts):
        encoded = target_tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
            return_dict=False,
        )
        token_ids = _coerce_token_ids(encoded)
        if len(token_ids) + args.max_new_tokens > args.max_model_len:
            raise ValueError(
                f"Prompt {record['id']!r} has {len(token_ids)} tokens; prompt plus "
                f"max_new_tokens exceeds max_model_len={args.max_model_len}"
            )
        prepared.append(
            {
                "id": record["id"],
                "prompt": record["prompt"],
                "prompt_token_ids": token_ids,
                "seed": args.seed + index,
            }
        )
    return prepared, target_tokenizer


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", args.run_name).strip("-._")
        if not safe_label:
            raise ValueError("--run-name must contain at least one letter or number")
        descriptor = safe_label
    else:
        descriptor = (
            f"vllm-{args.benchmark_mode}-batch{args.batch_size}-"
            f"k{args.num_speculative_tokens}-tok{args.max_new_tokens}"
        )
    return resolve_experiment_output_dir(
        args.output_dir,
        timestamp=timestamp,
        descriptor=descriptor,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _worker_config(
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    *,
    engine_mode: str,
    profile_output_dir: Path,
) -> dict[str, Any]:
    modes = ["greedy", "sample"] if args.benchmark_mode == "both" else [args.benchmark_mode]
    profile_mode = "greedy" if "greedy" in modes else modes[0]
    return {
        "schema_version": 1,
        "required_vllm_version": args.required_vllm_version,
        "engine_mode": engine_mode,
        "models": {
            "draft": normalize_model_source(args.draft_model),
            "target": normalize_model_source(args.target_model),
        },
        "revisions": {
            "draft": args.draft_revision,
            "target": args.target_revision,
        },
        "engine": {
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "enable_prefix_caching": args.enable_prefix_caching,
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": args.trust_remote_code,
            "seed": args.seed,
            "attention_backend": "auto",
        },
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_sample_method": args.draft_sample_method,
            "rejection_sample_method": "standard",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "enable_thinking": args.enable_thinking,
        },
        "benchmark": {
            "modes": modes,
            "warmup_runs_per_batch": args.benchmark_warmup,
            "measured_runs_per_batch": args.benchmark_repeats,
            "batch_size": args.batch_size,
        },
        "profiling": {
            "enabled": bool(args.profile_vllm and engine_mode == "speculative"),
            "profiler": "torch",
            "mode": profile_mode,
            "max_new_tokens": args.profile_max_new_tokens,
            "output_dir": str(profile_output_dir.resolve()),
            "timing_isolation": (
                "Runs after all untraced measurements and is excluded from speedup."
            ),
        },
        "prompts": prompts,
    }


def _run_worker(
    config_path: Path,
    output_path: Path,
    *,
    local_files_only: bool,
) -> None:
    environment = os.environ.copy()
    if local_files_only:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sd_latency.vllm_worker",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_DIR,
        env=environment,
        check=True,
    )


def _read_worker_result(path: Path, expected_mode: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("engine_mode") != expected_mode:
        raise RuntimeError(
            f"Worker output {path} reports {result.get('engine_mode')!r}, "
            f"expected {expected_mode!r}"
        )
    return result


def _timing_rows(mode_result: dict[str, Any]) -> list[dict[str, Any]]:
    excluded = {"output_token_ids", "prompt_ids", "run_key", "batch_index"}
    return [
        {key: value for key, value in row.items() if key not in excluded}
        for row in mode_result["runs"]
    ]


def _paired_output_matches(
    target_mode: dict[str, Any], sd_mode: dict[str, Any]
) -> list[bool]:
    target_runs = {row["run_key"]: row for row in target_mode["runs"]}
    sd_runs = {row["run_key"]: row for row in sd_mode["runs"]}
    if target_runs.keys() != sd_runs.keys():
        raise RuntimeError("Target-only and SD workers produced different measured run keys")
    matches: list[bool] = []
    for key in sorted(target_runs):
        left = target_runs[key]["output_token_ids"]
        right = sd_runs[key]["output_token_ids"]
        matches.append(left == right)
    return matches


def _build_report(
    args: argparse.Namespace,
    prompts: list[dict[str, Any]],
    tokenizer: Any,
    target: dict[str, Any],
    sd: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    modes = target["benchmark"]["modes"]
    if modes != sd["benchmark"]["modes"]:
        raise RuntimeError("Target-only and SD workers used different benchmark modes")

    report: dict[str, Any] = {
        "schema_version": 2,
        "backend": "vllm",
        "runtime": {
            "target_only_worker": target["runtime"],
            "speculative_worker": sd["runtime"],
        },
        "engine_load_ms": {
            "target_only": target["engine_load_ms"],
            "speculative": sd["engine_load_ms"],
        },
        "models": target["models"],
        "revisions": target["revisions"],
        "engine": target["engine"],
        "settings": target["settings"],
        "benchmark": {
            **target["benchmark"],
            "engine_execution_order": args.engine_order,
            "engine_isolation": "fresh subprocess per Target-only/SD engine",
            "timing_scope": "blocking llm.generate() call; model load excluded",
            "latency_unit": "one full batch",
        },
        "speculative_decoding_metrics": {
            mode: sd["modes"][mode]["speculative_decoding_metrics"] for mode in modes
        },
        "vllm_profile": sd.get("profiling", {"enabled": False}),
        "prompts": [],
        "aggregate_latency": {},
        "notes": [
            "Target-only and SD are compared within vLLM; do not attribute the "
            "Transformers-to-vLLM engine difference to speculative decoding.",
            "assistant_confidence_threshold and Transformers assistant schedules have "
            "no direct mapping in vLLM's fixed-K draft_model strategy.",
            "vLLM chooses its attention backend automatically; Transformers "
            "attn_implementation is intentionally not passed.",
            "When enabled, vLLM profiling runs after the untraced measurements, so "
            "profiler overhead is not included in latency speedup.",
        ],
    }

    for mode in modes:
        target_mode = target["modes"][mode]
        sd_mode = sd["modes"][mode]
        summary = summarize_pair(_timing_rows(target_mode), _timing_rows(sd_mode))
        paired_matches = _paired_output_matches(target_mode, sd_mode)
        summary["paired_exact_output_matches"] = paired_matches
        summary["all_paired_outputs_exactly_match"] = all(paired_matches)
        report["aggregate_latency"][mode] = summary

    output_pair_lines: list[str] = []
    display_mode = "greedy" if "greedy" in modes else modes[0]
    for prompt_index, prompt in enumerate(prompts):
        prompt_id = str(prompt["id"])
        prompt_report: dict[str, Any] = {
            "id": prompt_id,
            "prompt": prompt["prompt"],
            "prompt_token_count": len(prompt["prompt_token_ids"]),
            "seed": prompt["seed"],
            "modes": {},
        }
        for mode in modes:
            target_ids = target["modes"][mode]["first_outputs"][prompt_id]
            sd_ids = sd["modes"][mode]["first_outputs"][prompt_id]
            target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
            sd_text = tokenizer.decode(sd_ids, skip_special_tokens=False)
            prompt_report["modes"][mode] = {
                "target_vs_sd": compare_outputs(
                    target_ids,
                    sd_ids,
                    target_text,
                    sd_text,
                ),
                "outputs": {
                    "target_only": target_text,
                    "speculative": sd_text,
                },
            }
            if mode == display_mode:
                output_pair_lines.extend(
                    [
                        f"{prompt_index:04d} MODE\t{mode}",
                        f"{prompt_index:04d} PROMPT\t"
                        f"{json.dumps(prompt['prompt'], ensure_ascii=False)}",
                        f"{prompt_index:04d} TARGET_ONLY\t"
                        f"{json.dumps(target_text, ensure_ascii=False)}",
                        f"{prompt_index:04d} SD\t"
                        f"{json.dumps(sd_text, ensure_ascii=False)}",
                        "",
                    ]
                )
        report["prompts"].append(prompt_report)
    return report, output_pair_lines


def main() -> None:
    args = parse_args()
    _validate_args(args)
    prompt_texts = _load_prompt_texts(args)
    prompts, tokenizer = _prepare_prompts(args, prompt_texts)
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "target_only": {
            "config": output_dir / "vllm_target_worker_config.json",
            "output": output_dir / "vllm_target_worker.json",
        },
        "speculative": {
            "config": output_dir / "vllm_sd_worker_config.json",
            "output": output_dir / "vllm_sd_worker.json",
        },
    }
    for engine_mode in ("target_only", "speculative"):
        _write_json(
            paths[engine_mode]["config"],
            _worker_config(
                args,
                prompts,
                engine_mode=engine_mode,
                profile_output_dir=output_dir / "vllm_torch_profile",
            ),
        )

    execution_order = (
        ("target_only", "speculative")
        if args.engine_order == "target-first"
        else ("speculative", "target_only")
    )
    for engine_mode in execution_order:
        print(f"Starting isolated vLLM worker: {engine_mode}", flush=True)
        _run_worker(
            paths[engine_mode]["config"],
            paths[engine_mode]["output"],
            local_files_only=args.local_files_only,
        )

    target = _read_worker_result(paths["target_only"]["output"], "target_only")
    sd = _read_worker_result(paths["speculative"]["output"], "speculative")
    report, output_pair_lines = _build_report(args, prompts, tokenizer, target, sd)

    report_path = output_dir / "latency_report.json"
    pairs_path = output_dir / "output_pairs.txt"
    metrics_path = output_dir / "vllm_spec_decode_metrics.json"
    profile_path = output_dir / "vllm_profile_summary.json"
    _write_json(report_path, report)
    pairs_path.write_text("\n".join(output_pair_lines), encoding="utf-8")
    _write_json(metrics_path, report["speculative_decoding_metrics"])
    if report["vllm_profile"].get("enabled"):
        _write_json(profile_path, report["vllm_profile"])

    console_summary = {
        "report": str(report_path),
        "adjacent_output_pairs": str(pairs_path),
        "speculative_metrics": str(metrics_path),
        "vllm_profile_summary": (
            str(profile_path) if report["vllm_profile"].get("enabled") else None
        ),
        "aggregate_latency": {
            mode: {
                "latency_speedup": summary["latency_speedup"],
                "throughput_speedup": summary["throughput_speedup"],
                "all_paired_outputs_exactly_match": summary[
                    "all_paired_outputs_exactly_match"
                ],
            }
            for mode, summary in report["aggregate_latency"].items()
        },
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
