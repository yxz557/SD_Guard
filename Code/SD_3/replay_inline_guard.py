#!/usr/bin/env python3
"""Replay an existing speculative round trace through the inline Guard logic."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sdguard_sd.generation import SpeculativeRound, prepare_prompt
from sdguard_sd.inline_guard import IncrementalGuardMonitor, aggregate_inline_records
from sdguard_sd.io_utils import JsonlWriter, iter_jsonl, read_json, write_json
from sdguard_sd.runtime import collect_runtime_metadata, load_guard_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline replay of SD round traces. Provisional Draft branches are Guarded "
            "before their recorded Target acceptance decision is applied."
        )
    )
    parser.add_argument("--speculative-experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--guard-model")
    parser.add_argument("--guard-device")
    parser.add_argument("--dtype")
    parser.add_argument("--guard-attn-implementation")
    parser.add_argument("--guard-revision")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _group(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        result.setdefault(str(row["sample_id"]), []).append(row)
    return result


def main() -> None:
    args = parse_args()
    installed = importlib.metadata.version("transformers")
    if installed != "4.57.6":
        raise RuntimeError(
            "Inline replay requires transformers==4.57.6 because KV-cache crop "
            f"semantics are version-sensitive; found {installed}."
        )
    source = args.speculative_experiment_dir.expanduser().resolve()
    config = read_json(source / "run_config.json")
    if config.get("generation_mode") != "speculative":
        raise ValueError("--speculative-experiment-dir must be a speculative cell")
    rows = list(iter_jsonl(source / "raw" / "generation_records.jsonl"))
    if args.sample_limit is not None:
        if args.sample_limit <= 0:
            raise ValueError("--sample-limit must be positive")
        rows = rows[: args.sample_limit]
    rounds_by_sample = _group(source / "raw" / "sd_round_records.jsonl")
    tokens_by_sample = _group(source / "raw" / "guard_token_records.jsonl")

    destination = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else source / "inline_replay"
    )
    record_path = destination / "replay_records.jsonl"
    token_path = destination / "proposal_token_records.jsonl"
    round_path = destination / "proposal_round_records.jsonl"
    summary_path = destination / "replay_summary.json"
    metadata_path = destination / "replay_metadata.json"
    existing = [
        path
        for path in (record_path, token_path, round_path, summary_path, metadata_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(f"Replay outputs already exist: {existing}; pass --overwrite")
    destination.mkdir(parents=True, exist_ok=True)

    guard_model_source = args.guard_model or config["guard_model"]
    guard_device = args.guard_device or config["guard_device"]
    dtype = args.dtype or config["dtype"]
    attention = (
        args.guard_attn_implementation
        if args.guard_attn_implementation is not None
        else config.get("guard_attn_implementation")
    )
    revision = (
        args.guard_revision
        if args.guard_revision is not None
        else config.get("guard_revision")
    )
    local_only = (
        bool(config.get("local_files_only", False))
        if args.local_files_only is None
        else args.local_files_only
    )
    print(f"Loading Stream Guard on {guard_device}: {guard_model_source}")
    guard_model, tokenizer = load_guard_model(
        guard_model_source,
        device=guard_device,
        dtype=dtype,
        local_files_only=local_only,
        revision=revision,
        attention_implementation=attention,
    )

    started = time.perf_counter()
    inline_results: list[dict[str, Any]] = []
    with ExitStack() as stack:
        record_writer = stack.enter_context(JsonlWriter(record_path))
        token_writer = stack.enter_context(JsonlWriter(token_path))
        round_writer = stack.enter_context(JsonlWriter(round_path))
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            def tqdm(iterable: Any, **_: Any) -> Any:
                return iterable

        for row in tqdm(rows, desc="offline inline-Guard replay", unit="sample"):
            sample_id = str(row["sample_id"])
            prompt_mapping = prepare_prompt(
                tokenizer,
                row["prompt"],
                enable_thinking=bool(row["thinking_enabled"]),
            )
            prompt_ids = tuple(
                int(value) for value in prompt_mapping["input_ids"][0].tolist()
            )
            expected_prompt_count = int(row["generation"]["prompt_token_count"])
            if len(prompt_ids) != expected_prompt_count:
                raise RuntimeError(
                    f"Reconstructed prompt length differs for {sample_id}: "
                    f"{len(prompt_ids)} != {expected_prompt_count}"
                )

            monitor = IncrementalGuardMonitor(
                guard_model,
                tokenizer,
                guard_device=guard_device,
                enable_thinking=bool(row["thinking_enabled"]),
            )
            monitor.start(prompt_ids)
            source_rounds = rounds_by_sample.get(sample_id, [])
            for raw in source_rounds:
                round_ = SpeculativeRound(
                    round_index=int(raw["round_index"]),
                    prefix_length=int(raw["prefix_length"]),
                    proposed_ids=tuple(int(value) for value in raw["proposed_ids"]),
                    accepted_count=int(raw["accepted_count"]),
                    committed_ids=tuple(int(value) for value in raw["committed_ids"]),
                    commit_kind=str(raw["commit_kind"]),
                )
                monitor.observe_proposal(
                    round_index=round_.round_index,
                    prefix_length=round_.prefix_length,
                    proposed_ids=round_.proposed_ids,
                )
                monitor.observe_commit(round_)

            posthoc_tokens = tokens_by_sample.get(sample_id, [])
            posthoc = SimpleNamespace(
                token_records=posthoc_tokens,
                scopes=row["guard"]["scopes"],
            )
            generated_ids = tuple(
                int(value) for value in row["generation"]["generated_ids"]
            )
            result = monitor.finalize(
                generated_ids=generated_ids,
                answer_start_generated_index=row["completion"][
                    "answer_start_generated_index"
                ],
                posthoc_guard=posthoc,
            )
            consistency = result["consistency"]
            if not consistency["verified_posthoc_exact_match"]:
                print(
                    "Warning: incremental and saved posthoc Guard labels differ for "
                    f"{sample_id}: token_label_mismatches="
                    f"{consistency['token_label_mismatch_count']}, "
                    "scope_decisions_match="
                    f"{consistency['verified_scope_decisions_match_posthoc']}. "
                    "The difference is recorded and replay will continue."
                )
            result["source_sample_id"] = sample_id
            inline_results.append(result)
            record_writer.write(
                {
                    "schema_version": 1,
                    "sample_id": sample_id,
                    "sample_order": int(row["sample_order"]),
                    "thinking_enabled": bool(row["thinking_enabled"]),
                    "inline_guard": result,
                }
            )
            for token in monitor.tracker.token_records:
                if token["phase"] == "proposal":
                    token_writer.write(
                        {
                            "sample_id": sample_id,
                            "sample_order": int(row["sample_order"]),
                            **token,
                        }
                    )
            for replay_round in monitor.tracker.round_records:
                round_writer.write(
                    {
                        "sample_id": sample_id,
                        "sample_order": int(row["sample_order"]),
                        **replay_round,
                    }
                )

    summary = aggregate_inline_records(inline_results)
    write_json(summary_path, summary)
    metadata = {
        "schema_version": 1,
        "status": "completed",
        "source_speculative_experiment_dir": str(source),
        "sample_count": len(rows),
        "thinking_enabled": bool(config["enable_thinking"]),
        "guard_model": guard_model_source,
        "guard_device": guard_device,
        "dtype": dtype,
        "wall_time_seconds": time.perf_counter() - started,
        "guard_behavior": "alarm_only_no_generation_intervention",
        **collect_runtime_metadata(),
    }
    write_json(metadata_path, metadata)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReplay output: {destination}")


if __name__ == "__main__":
    main()
