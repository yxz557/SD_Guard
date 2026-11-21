#!/usr/bin/env python3
"""Start a fixed-path vLLM service used by the Qwen3-8B AgentLAB profile."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

from qwen3_8b_official_profile import (
    COMMON_ENV,
    INTENT_ATTACKER_MODEL_NAME,
    INTENT_ATTACKER_MODEL_PATH,
    RESULTS_ROOT,
    VICTIM_MODEL_NAME,
    VICTIM_MODEL_PATH,
    VLLM_EXECUTABLE,
)


def _service_command(service: str) -> list[str]:
    if service == "victim":
        return [
            str(VLLM_EXECUTABLE), "serve", str(VICTIM_MODEL_PATH),
            "--served-model-name", VICTIM_MODEL_NAME,
            "--host", "127.0.0.1",
            "--port", "8001",
            "--tensor-parallel-size", "1",
            "--dtype", "auto",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "40960",
            "--max-num-seqs", "20",
            "--trust-remote-code",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--reasoning-parser", "qwen3",
            "--generation-config", "vllm",
            "--override-generation-config", '{"temperature":0.7,"top_p":0.8,"top_k":20}',
            "--default-chat-template-kwargs", '{"enable_thinking":false}',
        ]
    if service == "victim-for-intent":
        command = _service_command("victim")
        utilization_index = command.index("--gpu-memory-utilization") + 1
        command[utilization_index] = "0.45"
        context_index = command.index("--max-model-len") + 1
        command[context_index] = "8192"
        max_sequences_index = command.index("--max-num-seqs") + 1
        command[max_sequences_index] = "2"
        command.extend(["--cpu-offload-gb", "8"])
        return command
    return [
        str(VLLM_EXECUTABLE), "serve", str(INTENT_ATTACKER_MODEL_PATH),
        "--served-model-name", INTENT_ATTACKER_MODEL_NAME,
        "--host", "127.0.0.1",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--dtype", "auto",
        "--gpu-memory-utilization", "0.45",
        "--max-model-len", "8192",
        "--max-num-seqs", "2",
        "--cpu-offload-gb", "20",
        "--trust-remote-code",
        "--generation-config", "vllm",
        "--override-generation-config", '{"temperature":0.3,"top_p":0.95,"top_k":20}',
        "--default-chat-template-kwargs", '{"enable_thinking":false}',
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "service",
        choices=["victim", "victim-for-intent", "intent-attacker"],
        help=("Use victim for Tool/Objective/Memory. Intent requires both "
              "victim-for-intent and intent-attacker in separate terminals."),
    )
    args = parser.parse_args()

    model_path = (INTENT_ATTACKER_MODEL_PATH if args.service == "intent-attacker"
                  else VICTIM_MODEL_PATH)
    if not model_path.is_dir():
        print(f"Missing model directory: {model_path}", file=sys.stderr)
        return 2
    if not VLLM_EXECUTABLE.is_file():
        print(f"Missing vLLM executable: {VLLM_EXECUTABLE}", file=sys.stderr)
        return 2

    log_dir = RESULTS_ROOT / "service_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{args.service}_{timestamp}.log"
    command = _service_command(args.service)
    environment = os.environ.copy()
    environment.update(COMMON_ENV)
    environment["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"Starting {args.service}; log: {log_file}")
    with log_file.open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
