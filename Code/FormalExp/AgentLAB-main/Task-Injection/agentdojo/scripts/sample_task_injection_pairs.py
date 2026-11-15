from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from agentdojo.task_suite.load_suites import get_suites


SCHEMA_VERSION = "sd_guard.task_injection_pair_manifest.v1"


def _task_sort_key(task_id: str) -> tuple[str, int | str]:
    prefix, separator, suffix = task_id.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    return task_id, task_id


def _suite_seed(seed: int, suite_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{suite_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _selected_suites(benchmark_version: str, requested: list[str]) -> dict[str, Any]:
    available = get_suites(benchmark_version)
    if not requested:
        return dict(available)
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown suites for {benchmark_version}: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError(f"Duplicate --suite values are not allowed: {requested}")
    return {name: available[name] for name in requested}


def _build_manifest(
    benchmark_version: str,
    suites: dict[str, Any],
    pairs_per_suite: int,
    seed: int,
) -> dict[str, Any]:
    suite_records: dict[str, Any] = {}
    for suite_name, suite in suites.items():
        user_task_ids = sorted(suite.user_tasks, key=_task_sort_key)
        injection_task_ids = sorted(suite.injection_tasks, key=_task_sort_key)
        population = [
            (user_task_id, injection_task_id)
            for user_task_id in user_task_ids
            for injection_task_id in injection_task_ids
        ]
        if len(population) < pairs_per_suite:
            raise ValueError(
                f"Suite {suite_name!r} has only {len(population)} pairs; "
                f"cannot sample {pairs_per_suite} without replacement"
            )
        rng = random.Random(_suite_seed(seed, suite_name))
        sampled = sorted(
            rng.sample(population, pairs_per_suite),
            key=lambda pair: (_task_sort_key(pair[0]), _task_sort_key(pair[1])),
        )
        suite_records[suite_name] = {
            "pool_user_tasks": len(user_task_ids),
            "pool_injection_tasks": len(injection_task_ids),
            "pool_pairs": len(population),
            "sampled_pairs": [
                {"user_task_id": user_task_id, "injection_task_id": injection_task_id}
                for user_task_id, injection_task_id in sampled
            ],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": benchmark_version,
        "pairs_per_suite": pairs_per_suite,
        "seed": seed,
        "suite_names": list(suites),
        "suites": suite_records,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Manifest root must be a JSON object: {path}")
    return value


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    benchmark_version: str,
    suites: dict[str, Any],
    pairs_per_suite: int,
    seed: int,
) -> None:
    expected_metadata = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": benchmark_version,
        "pairs_per_suite": pairs_per_suite,
        "seed": seed,
    }
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Existing manifest has incompatible {key}: "
                f"expected {expected!r}, got {manifest.get(key)!r}"
            )

    manifest_suite_names = manifest.get("suite_names")
    if (
        not isinstance(manifest_suite_names, list)
        or len(manifest_suite_names) != len(set(manifest_suite_names))
        or set(manifest_suite_names) != set(suites)
    ):
        raise ValueError(
            "Existing manifest has incompatible suite_names: "
            f"expected the suites {list(suites)!r}, got {manifest_suite_names!r}"
        )

    suite_records = manifest.get("suites")
    if not isinstance(suite_records, dict) or set(suite_records) != set(suites):
        raise ValueError("Existing manifest suite records do not match the requested suites")

    for suite_name, suite in suites.items():
        records = suite_records[suite_name].get("sampled_pairs")
        if not isinstance(records, list) or len(records) != pairs_per_suite:
            raise ValueError(f"Invalid sampled pair count for suite {suite_name!r}")
        available_users = set(suite.user_tasks)
        available_injections = set(suite.injection_tasks)
        observed: set[tuple[str, str]] = set()
        for record in records:
            user_task_id = record.get("user_task_id")
            injection_task_id = record.get("injection_task_id")
            pair = (user_task_id, injection_task_id)
            if user_task_id not in available_users or injection_task_id not in available_injections:
                raise ValueError(f"Manifest contains an unavailable pair in {suite_name}: {pair}")
            if pair in observed:
                raise ValueError(f"Manifest contains a duplicate pair in {suite_name}: {pair}")
            observed.add(pair)


def _ensure_manifest(
    path: Path,
    *,
    benchmark_version: str,
    suites: dict[str, Any],
    pairs_per_suite: int,
    seed: int,
) -> dict[str, Any]:
    if path.exists():
        manifest = _load_manifest(path)
        _validate_manifest(
            manifest,
            benchmark_version=benchmark_version,
            suites=suites,
            pairs_per_suite=pairs_per_suite,
            seed=seed,
        )
        print(f"Reusing existing sample manifest: {path}", file=sys.stderr)
        return manifest

    manifest = _build_manifest(benchmark_version, suites, pairs_per_suite, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    print(f"Created sample manifest once: {path}", file=sys.stderr)
    return manifest


def _emit_groups(manifest: dict[str, Any]) -> None:
    for suite_name in manifest["suite_names"]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for pair in manifest["suites"][suite_name]["sampled_pairs"]:
            grouped[pair["user_task_id"]].append(pair["injection_task_id"])
        for user_task_id in sorted(grouped, key=_task_sort_key):
            injection_task_ids = sorted(set(grouped[user_task_id]), key=_task_sort_key)
            print(" ".join([suite_name, user_task_id, *injection_task_ids]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create once and reuse a deterministic AgentDojo user/injection pair manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-version", default="v1.2.1")
    parser.add_argument("--pairs-per-suite", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--emit-groups", action="store_true")
    args = parser.parse_args()
    if args.pairs_per_suite <= 0:
        raise SystemExit("--pairs-per-suite must be positive")

    suites = _selected_suites(args.benchmark_version, args.suite)
    manifest = _ensure_manifest(
        args.manifest,
        benchmark_version=args.benchmark_version,
        suites=suites,
        pairs_per_suite=args.pairs_per_suite,
        seed=args.seed,
    )
    if args.emit_groups:
        _emit_groups(manifest)
    else:
        total = args.pairs_per_suite * len(suites)
        print(
            f"Manifest ready: suites={len(suites)}, pairs_per_suite={args.pairs_per_suite}, total_pairs={total}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

