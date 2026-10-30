#!/usr/bin/env python3
"""Run the complete SD_5 2x4 experiment in one command."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sdguard_sd.dataset import create_manifest, load_prompt_samples, select_from_manifest
from sdguard_sd.io_utils import read_json, write_json
from sdguard_sd.runtime import assert_same_tokenizer, normalize_model_source
from sdguard_sd.seeding import stable_sample_seed


PROJECT_DIR = Path(__file__).resolve().parent
THINKING_LEVELS = ("think_on", "think_off")
MODES = ("target_only", "speculative", "draft_only", "inline_guard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SD_3's full 2x4 evaluation with vLLM 0.31 draft-model SD and "
            "an alarm-only Guard inserted between Draft proposal and Target verification."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default="/home/model_and_data/yxz/Code/SD_Guard/datasets/Qwen3GuardTest",
    )
    parser.add_argument(
        "--draft-model",
        default="/home/model_and_data/yxz/model/Huihui-Qwen3-0.6B-abliterated-v2",
    )
    parser.add_argument(
        "--target-model",
        default="/home/model_and_data/yxz/model/Huihui-Qwen3-14B-abliterated-v2",
    )
    parser.add_argument(
        "--guard-model",
        default="/home/model_and_data/yxz/model/Qwen3Guard-Stream-4B",
    )
    parser.add_argument("--draft-revision")
    parser.add_argument("--target-revision")
    parser.add_argument("--guard-revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument(
        "--profile-vllm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--profile-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--guard-attn-implementation")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate(args: argparse.Namespace) -> None:
    if args.sample_count <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Sample and token counts must be positive")
    if args.max_model_len <= args.max_new_tokens:
        raise ValueError("--max-model-len must leave room for prompt tokens")
    if args.num_speculative_tokens <= 0:
        raise ValueError("--num-speculative-tokens must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Invalid sampling settings")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.warmup_samples < 0:
        raise ValueError("--warmup-samples cannot be negative")
    if "," in args.gpu_id:
        raise ValueError("--gpu-id must identify exactly one physical GPU")


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_name or f"sd4-qwen3-06b-14b-n{args.sample_count}-{stamp}"
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-._")
    return (PROJECT_DIR / "outputs" / label).resolve()


def _coerce_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise TypeError("Chat template did not return a flat token-ID list")
    return [int(item) for item in value]


def _load_tokenizers(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-vllm.txt before running SD_5") from exc

    def load(source: str, revision: str | None, *, trust: bool = False) -> Any:
        kwargs: dict[str, Any] = {
            "local_files_only": args.local_files_only,
            "trust_remote_code": trust,
        }
        if revision is not None:
            kwargs["revision"] = revision
        return AutoTokenizer.from_pretrained(normalize_model_source(source), **kwargs)

    target = load(args.target_model, args.target_revision)
    draft = load(args.draft_model, args.draft_revision)
    guard = load(args.guard_model, args.guard_revision, trust=True)
    assert_same_tokenizer(target, draft, candidate_name="Draft")
    assert_same_tokenizer(target, guard, candidate_name="Qwen3Guard-Stream-4B")
    return target, draft, guard


def _prepare_samples(
    args: argparse.Namespace,
    output_dir: Path,
    tokenizer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = load_prompt_samples(args.dataset_path, cache_dir=args.cache_dir)
    if args.sample_manifest is None:
        manifest = create_manifest(
            samples,
            dataset_path=args.dataset_path,
            sample_count=args.sample_count,
            seed=args.seed,
        )
    else:
        manifest = read_json(args.sample_manifest)
        if int(manifest["sample_count"]) != args.sample_count:
            raise ValueError("Manifest sample_count differs from --sample-count")
    selected = select_from_manifest(samples, manifest)
    prepared: list[dict[str, Any]] = []
    for sample_order, sample in selected:
        prompts: dict[str, list[int]] = {}
        for thinking_name, enabled in (("think_on", True), ("think_off", False)):
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": sample.prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enabled,
                return_dict=False,
            )
            ids = _coerce_ids(encoded)
            if len(ids) + args.max_new_tokens > args.max_model_len:
                raise ValueError(
                    f"Sample {sample.sample_id!r}: {len(ids)} prompt tokens + "
                    f"{args.max_new_tokens} output tokens exceeds max_model_len="
                    f"{args.max_model_len}"
                )
            prompts[thinking_name] = ids
        prepared.append(
            {
                "sample_id": sample.sample_id,
                "sample_order": sample_order,
                "dataset_index": sample.dataset_index,
                "sample_seed": stable_sample_seed(args.seed, sample.sample_id),
                "prompt": sample.prompt,
                "original_label": sample.original_label,
                "original_source": sample.original_source,
                "original_unsafe_type": sample.original_unsafe_type,
                "prompt_token_ids": prompts,
            }
        )
    write_json(output_dir / "sample_manifest.json", manifest)
    return manifest, prepared


def _public_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "backend": "vllm",
        "required_vllm_base_version": "0.31.0",
        "dataset_path": str(args.dataset_path),
        "draft_model": normalize_model_source(args.draft_model),
        "target_model": normalize_model_source(args.target_model),
        "guard_model": normalize_model_source(args.guard_model),
        "sample_count": args.sample_count,
        "seed": args.seed,
        "per_sample_seed_strategy": "sha256(base_seed:sample_id) % (2**31-1)",
        "cache_dir": None if args.cache_dir is None else str(args.cache_dir),
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "num_speculative_tokens": args.num_speculative_tokens,
        "draft_sample_method": "greedy",
        "rejection_sample_method": "standard",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
        "batch_size": 1,
        "tensor_parallel_size": 1,
        "target_device": "cuda:0",
        "draft_device": "cuda:0",
        "guard_device": "cuda:0",
        "physical_gpu": str(args.gpu_id),
        "all_models_same_gpu": True,
        "dtype": args.dtype,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "guard_attn_implementation": args.guard_attn_implementation,
        "draft_revision": args.draft_revision,
        "target_revision": args.target_revision,
        "guard_revision": args.guard_revision,
        "local_files_only": args.local_files_only,
        "inline_guard_semantics": "passive_observer_alarm_only",
    }


def _worker_config(
    args: argparse.Namespace,
    output_dir: Path,
    samples: list[dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": role,
        "matrix_dir": str(output_dir),
        "public_config": _public_config(args),
        "models": {
            "draft": normalize_model_source(args.draft_model),
            "target": normalize_model_source(args.target_model),
            "guard": normalize_model_source(args.guard_model),
        },
        "revisions": {
            "draft": args.draft_revision,
            "target": args.target_revision,
            "guard": args.guard_revision,
        },
        "engine": {
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_num_seqs": 1,
            "tensor_parallel_size": 1,
            "enable_prefix_caching": False,
            "enforce_eager": args.enforce_eager,
            "trust_remote_code": args.trust_remote_code,
        },
        "guard": {
            "model": normalize_model_source(args.guard_model),
            "revision": args.guard_revision,
            "dtype": args.dtype,
            "local_files_only": args.local_files_only,
            "attention_implementation": args.guard_attn_implementation,
        },
        "settings": {
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "num_speculative_tokens": args.num_speculative_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "benchmark": {
            "batch_size": 1,
            "warmup_samples": args.warmup_samples,
            "formal_sample_count": args.sample_count,
            "formal_timing_profiled": False,
        },
        "profiling": {
            "enabled": bool(args.profile_vllm and role == "speculative"),
            "max_new_tokens": args.profile_max_new_tokens,
            "output_dir": str(output_dir / "diagnostics" / "vllm_torch_profile"),
            "excluded_from_formal_latency": True,
        },
        "samples": samples,
    }


def _run_worker(
    config_path: Path,
    result_path: Path,
    *,
    args: argparse.Namespace,
) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    if args.local_files_only:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    command = [
        sys.executable,
        "-m",
        "sdguard_sd.vllm_worker",
        "--config",
        str(config_path),
        "--output",
        str(result_path),
    ]
    log_path = result_path.with_suffix(".log")
    recent_output: deque[str] = deque(maxlen=160)
    with log_path.open("w", encoding="utf-8", newline="\n") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:  # pragma: no cover - guaranteed by PIPE
            raise RuntimeError("Worker stdout pipe was not created")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
            log_stream.flush()
            recent_output.append(line)
        return_code = process.wait()
    if return_code != 0:
        tail = "".join(recent_output).rstrip()
        raise RuntimeError(
            f"Worker failed with exit status {return_code}. Full log: {log_path}\n"
            f"--- worker log tail ---\n{tail}"
        )


def _latency_index(output_dir: Path) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for thinking in THINKING_LEVELS:
        for mode in MODES:
            cell = output_dir / thinking / mode
            cells.append(
                {
                    "thinking": thinking,
                    "generation_mode": mode,
                    "formal": read_json(cell / "raw" / "latency_summary.json"),
                    "profile_diagnostic": (
                        read_json(cell / "raw" / "vllm_profile_summary.json")
                        if (cell / "raw" / "vllm_profile_summary.json").exists()
                        else None
                    ),
                }
            )
    return {
        "schema_version": 1,
        "cells": cells,
        "timing_contract": {
            "formal": "100 sample, unprofiled wall latency; warm-up and model load excluded",
            "profile": "separate diagnostic requests; GPU kernel time is not wall latency",
            "target_only": "target model generation, posthoc Guard, total pipeline",
            "draft_only": "draft model generation, posthoc Guard, total pipeline",
            "standard_sd": "engine generation, posthoc Guard, total pipeline + profiler stages",
            "inline_sd": "Guard prefill/proposal/commit, operational total + profiler stages",
            "inline_posthoc_audit": "excluded from operational latency",
        },
    }


def main() -> None:
    args = parse_args()
    _validate(args)
    output_dir = _output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing",
        "started_at": _utc_now(),
        "project": "SD_5",
        "one_command_matrix": True,
    }
    write_json(metadata_path, metadata)
    started_ns = time.perf_counter_ns()
    try:
        target_tokenizer, _, _ = _load_tokenizers(args)
        manifest, samples = _prepare_samples(args, output_dir, target_tokenizer)
        write_json(output_dir / "experiment_config.json", _public_config(args))
        for thinking in THINKING_LEVELS:
            for mode in MODES:
                cell = output_dir / thinking / mode
                cell.mkdir(parents=True, exist_ok=True)
                write_json(cell / "sample_manifest.json", manifest)

        worker_dir = output_dir / "workers"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_results: dict[str, Any] = {}
        for role in ("target", "draft", "speculative"):
            config_path = worker_dir / f"{role}_config.json"
            result_path = worker_dir / f"{role}_result.json"
            write_json(config_path, _worker_config(args, output_dir, samples, role))
            print(f"Starting isolated {role} worker on physical GPU {args.gpu_id}", flush=True)
            _run_worker(config_path, result_path, args=args)
            worker_results[role] = read_json(result_path)

        summary_command = [
            sys.executable,
            str(PROJECT_DIR / "summarize_matrix.py"),
            "--matrix-dir",
            str(output_dir),
            "--overwrite",
        ]
        subprocess.run(summary_command, cwd=PROJECT_DIR, check=True)
        latency = _latency_index(output_dir)
        write_json(output_dir / "analysis" / "latency_report.json", latency)
        metadata.update(
            {
                "status": "complete",
                "finished_at": _utc_now(),
                "wall_time_ms": (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                "worker_results": worker_results,
                "sample_count_per_cell": args.sample_count,
                "cell_count": 8,
                "all_models_physical_gpu": str(args.gpu_id),
                "formal_latency_profiled": False,
            }
        )
        write_json(metadata_path, metadata)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "matrix_dir": str(output_dir),
                    "matrix_summary": str(output_dir / "analysis" / "matrix_summary.json"),
                    "latency_report": str(output_dir / "analysis" / "latency_report.json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception:
        metadata.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "wall_time_ms": (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                "error": traceback.format_exc(),
            }
        )
        write_json(metadata_path, metadata)
        raise


if __name__ == "__main__":
    main()
