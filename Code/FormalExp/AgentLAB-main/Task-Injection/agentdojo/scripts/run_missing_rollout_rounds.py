from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_missing_cells(path: Path) -> list[tuple[str, str, str, str]]:
    cells: set[tuple[str, str, str, str]] = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            cells.add(
                (
                    record["suite_name"],
                    record["user_task_id"],
                    record["injection_task_id"],
                    record["attack_type"],
                )
            )
    return sorted(cells)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_cell(
    cell: tuple[str, str, str, str],
    *,
    python_bin: str,
    benchmark_version: str,
    log_root: Path,
    environment: dict[str, str],
    agentdojo_root: Path,
) -> None:
    suite_name, user_task_id, injection_task_id, attack_type = cell
    command = [
        python_bin,
        "-m",
        "agentdojo.scripts.benchmark",
        "--model",
        "VLLM_PARSED",
        "--benchmark-version",
        benchmark_version,
        "--logdir",
        str(log_root),
        "--max-workers",
        "1",
        "--suite",
        suite_name,
        "--user-task",
        user_task_id,
        "--injection-task",
        injection_task_id,
        "--attack",
        attack_type,
    ]
    completed = subprocess.run(command, env=environment, cwd=agentdojo_root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Cell failed with exit {completed.returncode}: {cell}")


def _run_checked(command: list[str], environment: dict[str, str], agentdojo_root: Path) -> None:
    subprocess.run(command, env=environment, cwd=agentdojo_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptively rerun only attack cells still missing a clean positive or negative rollout."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--target-model-id", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument("--generation-seed", type=int, default=20261213)
    parser.add_argument("--run-tag", default="qwen3-8b-task-injection-think-adaptive")
    parser.add_argument("--allow-missing-thinking", action="store_true")
    args = parser.parse_args()
    if args.rounds <= 0 or args.start_round <= 0 or args.workers <= 0:
        raise SystemExit("--rounds, --start-round, and --workers must be positive")

    script_dir = Path(__file__).resolve().parent
    agentdojo_root = script_dir.parent
    campaign_root = args.campaign_root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = _load_json(manifest_path)
    benchmark_version = str(manifest["benchmark_version"])
    sample_seed = int(manifest["seed"])
    missing_path = campaign_root / "curated" / "missing_labels.jsonl"
    if not missing_path.is_file():
        raise FileNotFoundError(
            f"Run the repeated campaign/curator first; missing coverage file: {missing_path}"
        )

    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM health check returned HTTP {response.status}")

    for offset in range(args.rounds):
        round_number = args.start_round + offset
        cells = _load_missing_cells(missing_path)
        if not cells:
            print("All expected attack cells already have a clean positive and negative rollout.")
            break

        rollout_id = f"adaptive_{round_number:03d}"
        generation_seed = args.generation_seed + round_number - 1
        round_root = campaign_root / "raw" / rollout_id
        log_root = round_root / "logs"
        result_root = round_root / "results"
        trace_root = result_root / "traces"
        run_id = f"{args.run_tag}-{rollout_id}"
        log_root.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment.update(
            {
                "LOCAL_LLM_PORT": str(args.port),
                "AGENTDOJO_LLM_TEMPERATURE": "0.6",
                "AGENTDOJO_LLM_TOP_P": "0.95",
                "AGENTDOJO_LLM_TOP_K": "20",
                "AGENTDOJO_LLM_MAX_TOKENS": "4096",
                "AGENTDOJO_GENERATION_SEED": str(generation_seed),
                "AGENTDOJO_ENABLE_THINKING": "true",
            }
        )
        _write_json(
            log_root / "campaign_config.json",
            {
                "schema_version": "sd_guard.task_injection_adaptive_campaign.v1",
                "benchmark_version": benchmark_version,
                "run_id": run_id,
                "rollout_id": rollout_id,
                "target_model_id": args.target_model_id,
                "sample_seed": sample_seed,
                "generation_config": {
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "max_tokens": 4096,
                    "generation_seed": generation_seed,
                    "enable_thinking": True,
                },
                "scheduled_missing_cells": [
                    {
                        "suite_name": suite,
                        "user_task_id": user,
                        "injection_task_id": injection,
                        "attack_type": attack,
                    }
                    for suite, user, injection, attack in cells
                ],
            },
        )
        print(
            f"[{rollout_id}] rerunning {len(cells)} cell(s) still missing at least one label; "
            f"generation_seed={generation_seed}"
        )
        failures: list[BaseException] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _run_cell,
                    cell,
                    python_bin=args.python_bin,
                    benchmark_version=benchmark_version,
                    log_root=log_root,
                    environment=environment,
                    agentdojo_root=agentdojo_root,
                )
                for cell in cells
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException as error:
                    failures.append(error)
        if failures:
            raise RuntimeError(f"{len(failures)} adaptive cell(s) failed; first error: {failures[0]}")

        _run_checked(
            [
                args.python_bin,
                "-m",
                "agentdojo.guard_eval.export_trace",
                "--input",
                str(log_root),
                "--output",
                str(trace_root),
                "--target-model-id",
                args.target_model_id,
                "--benchmark-version",
                benchmark_version,
                "--run-id",
                run_id,
                "--rollout-id",
                rollout_id,
                "--seed",
                str(sample_seed),
            ],
            environment,
            agentdojo_root,
        )
        _run_checked(
            [args.python_bin, "-m", "agentdojo.guard_eval.verify", "--input", str(trace_root)],
            environment,
            agentdojo_root,
        )
        curate_command = [
            args.python_bin,
            str(script_dir / "curate_repeated_task_injection.py"),
            "--campaign-root",
            str(campaign_root),
            "--manifest",
            str(manifest_path),
        ]
        if not args.allow_missing_thinking:
            curate_command.append("--require-thinking")
        _run_checked(curate_command, environment, agentdojo_root)

    coverage_path = campaign_root / "curated" / "coverage.json"
    coverage = _load_json(coverage_path)
    counts = coverage["counts"]
    print(
        "Adaptive collection finished: "
        f"complete={counts['complete_positive_negative_cells']}/"
        f"{counts['expected_attack_cells']}, "
        f"missing_label_slots={counts['missing_label_slots']}"
    )


if __name__ == "__main__":
    main()
