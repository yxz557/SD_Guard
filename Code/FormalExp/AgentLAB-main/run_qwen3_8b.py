#!/usr/bin/env python3
"""Launch one formal Qwen3-8B AgentLAB track from the fixed server profile.

Examples:
    python run_qwen3_8b.py tool smoke
    python run_qwen3_8b.py objective full
    python run_qwen3_8b.py memory full --run-tag r2
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from local_env import (
    DEFAULT_SECRETS_FILE,
    OFFICIAL_OPENAI_BASE_URL,
    load_local_env,
    validate_secret_value,
)
from qwen3_8b_official_profile import (
    AGENTLAB_ROOT,
    DATASET_MEMORY,
    DATASET_TOOL_INTENT,
    GPT51_MODEL,
    INTENT_ATTACKER_MODEL_PATH,
    MEMORY_ENV,
    MEM0_EMBEDDER_PATH,
    OPENAI_BASE_URL,
    VICTIM_MODEL_PATH,
    VLLM_ENV,
    build_run_specs,
)


TASK_ALIASES = {
    "tool": "tool-chaining",
    "tool-chaining": "tool-chaining",
    "intent": "intent-hijacking",
    "intent-hijacking": "intent-hijacking",
    "objective": "objective-drifting",
    "objective-drifting": "objective-drifting",
    "memory": "memory-poisoning",
    "memory-poisoning": "memory-poisoning",
}


def _directory_has_content(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _check_health(url: str) -> None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"HTTP {response.status}")
    except (OSError, urllib.error.URLError, RuntimeError) as exc:
        raise RuntimeError(f"Required local service is not healthy at {url}: {exc}") from exc


def _check_auxiliary_api(base_url: str, api_key: str) -> None:
    """Fail fast unless the configured GPT-5.1 endpoint accepts a real request."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps({
            "model": GPT51_MODEL,
            "messages": [{"role": "user", "content": "Reply exactly: OK"}],
            "max_completion_tokens": 16,
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read(2000).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GPT-5.1 preflight returned HTTP {exc.code}: {body}"
        ) from exc
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot reach GPT-5.1 endpoint {base_url}: {exc}"
        ) from exc
    if not payload.get("choices"):
        raise RuntimeError(
            f"GPT-5.1 preflight returned no choices from {base_url}"
        )


def _required_paths(task: str) -> tuple[Path, ...]:
    common = (AGENTLAB_ROOT, VLLM_ENV, VICTIM_MODEL_PATH, DATASET_TOOL_INTENT)
    if task == "intent-hijacking":
        return (*common, INTENT_ATTACKER_MODEL_PATH)
    if task == "memory-poisoning":
        return (AGENTLAB_ROOT, MEMORY_ENV, VICTIM_MODEL_PATH,
                DATASET_MEMORY, MEM0_EMBEDDER_PATH)
    return common


def _run_with_log(command: tuple[str, ...], log_file: Path, environment: dict[str, str]) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env.update(environment)
    with log_file.open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=AGENTLAB_ROOT,
            env=child_env,
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


def _write_manifest(
    spec, task: str, scale: str, run_tag: str | None, auxiliary_base_url: str,
) -> None:
    payload = {
        "schema_version": "sd_guard.agentlab_launcher.v1",
        "task": task,
        "scale": scale,
        "run_tag": run_tag,
        "launcher_recorded_at": datetime.now(timezone.utc).isoformat(),
        "official_evaluator_model": GPT51_MODEL,
        "auxiliary_base_url": auxiliary_base_url,
        "uses_official_openai_endpoint": (
            auxiliary_base_url == OFFICIAL_OPENAI_BASE_URL
        ),
        "command": list(spec.command),
        "environment": dict(spec.environment or {}),
    }
    path = spec.run_dir / "launcher_config.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed official-style Qwen3-8B AgentLAB profile",
    )
    parser.add_argument("task", choices=sorted(TASK_ALIASES))
    parser.add_argument("scale", choices=["smoke", "full"])
    parser.add_argument(
        "--run-tag",
        help="Optional suffix for an intentional rerun, for example r2",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved commands and paths without checking services or writing files",
    )
    parser.add_argument(
        "--skip-health-check", action="store_true",
        help="Advanced debugging only; do not use for formal runs",
    )
    args = parser.parse_args()

    try:
        secrets_path, _ = load_local_env(str(DEFAULT_SECRETS_FILE), override=True)
    except ValueError as exc:
        print(f"Invalid local configuration: {exc}", file=sys.stderr)
        return 2

    auxiliary_base_url = (
        os.environ.get("OPENAI_BASE_URL") or OPENAI_BASE_URL
    ).strip().rstrip("/")
    if not auxiliary_base_url.startswith(("https://", "http://")):
        print(
            "OPENAI_BASE_URL must start with https:// or http://",
            file=sys.stderr,
        )
        return 2

    task = TASK_ALIASES[args.task]
    specs = build_run_specs(
        task, args.scale, args.run_tag,
        auxiliary_base_url=auxiliary_base_url,
    )

    print(f"Task       : {task}")
    print(f"Scale      : {args.scale}")
    print(f"Judge      : {GPT51_MODEL} @ {auxiliary_base_url}")
    for spec in specs:
        print(f"Run dir    : {spec.run_dir}")
        print(f"Command    : {shlex.join(spec.command)}")

    if args.dry_run:
        return 0

    missing = [path for path in _required_paths(task) if not path.exists()]
    if missing:
        for path in missing:
            print(f"Missing required server path: {path}", file=sys.stderr)
        return 2

    try:
        api_key = validate_secret_value(
            "OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY"),
        )
    except ValueError as exc:
        print(f"Invalid OpenAI configuration: {exc}", file=sys.stderr)
        print(f"Fill OPENAI_API_KEY once in: {secrets_path}", file=sys.stderr)
        return 2
    if not secrets_path.is_file():
        print(f"Missing local secrets file: {secrets_path}", file=sys.stderr)
        return 2

    for spec in specs:
        if spec.run_dir.exists() and _directory_has_content(spec.run_dir):
            print(
                f"Refusing to mix results in non-empty directory: {spec.run_dir}\n"
                "Use --run-tag r2 (or another new tag) for an intentional rerun.",
                file=sys.stderr,
            )
            return 2

    if not args.skip_health_check:
        checked = set()
        for spec in specs:
            for url in spec.health_urls:
                if url not in checked:
                    _check_health(url)
                    checked.add(url)
                    print(f"Healthy     : {url}")

    try:
        _check_auxiliary_api(auxiliary_base_url, api_key)
    except RuntimeError as exc:
        print(f"Auxiliary API preflight failed: {exc}", file=sys.stderr)
        return 2
    print(f"GPT-5.1 API : healthy at {auxiliary_base_url}")

    for spec in specs:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(
            spec, task, args.scale, args.run_tag, auxiliary_base_url,
        )
        print(f"\nStarting {spec.name}; log: {spec.log_file}")
        return_code = _run_with_log(
            spec.command, spec.log_file, spec.environment or {},
        )
        if return_code != 0:
            print(f"{spec.name} failed with exit code {return_code}", file=sys.stderr)
            return return_code

    print("All requested runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
