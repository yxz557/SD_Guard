#!/usr/bin/env python3
"""Evaluate a shared random thinking subset with streaming and full forward."""

from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guard_eval.dataset import load_guard_samples
from guard_eval.full_forward import compare_stream_and_full, run_full_forward
from guard_eval.model_loader import (
    collect_runtime_metadata,
    load_stream_guard,
    require_single_cuda_model,
)
from guard_eval.policies import evaluate_all_policies
from guard_eval.result_io import JsonlWriter, read_json, write_json
from guard_eval.sampling import create_sample_manifest, select_samples_from_manifest
from guard_eval.sequence import prepare_sequence
from guard_eval.stream_runner import run_stream_sample, warmup_stream_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample the thinking split once, then evaluate every selected "
            "sample with true token streaming and an official-style full forward."
        )
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Random thinking samples when no existing manifest is supplied.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        help="Reuse an existing manifest instead of drawing a new sample.",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--warmup-repeats",
        type=int,
        default=2,
        help="Unrecorded streaming warmups before the first measured sample.",
    )
    parser.add_argument("--warmup-response-tokens", type=int, default=8)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32", "auto"),
    )
    parser.add_argument("--attn-implementation")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_display_name(model_path: str) -> str:
    name = Path(model_path).expanduser().name
    return name or model_path.rstrip("/").rsplit("/", 1)[-1]


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {destination}. Pass --overwrite to replace it."
        )
    if overwrite:
        for name in ("raw", "sample_manifest.json"):
            target = destination / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def select_policy_pair(
    decisions: dict[str, dict[str, Any]], *, stream: bool
) -> dict[str, dict[str, Any]]:
    if stream:
        return {
            "strict": decisions["online_strict"],
            "loose": decisions["online_loose"],
        }
    return {
        "strict": decisions["official_strict"],
        "loose": decisions["official_loose"],
    }


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.warmup_repeats < 0 or args.warmup_response_tokens < 0:
        raise ValueError("Warmup counts must be non-negative")

    print(f"Loading thinking split: {args.dataset_path}")
    all_samples = load_guard_samples(
        args.dataset_path,
        "thinking",
        cache_dir=args.cache_dir,
    )
    if args.sample_manifest is None:
        manifest = create_sample_manifest(
            all_samples,
            sample_count=args.sample_count,
            seed=args.seed,
            dataset_path=args.dataset_path,
        )
        manifest_source = None
    else:
        manifest_path = args.sample_manifest.expanduser().resolve()
        manifest = read_json(manifest_path)
        manifest_source = str(manifest_path)
    selected = select_samples_from_manifest(all_samples, manifest)

    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "sample_manifest.json", manifest)

    print(f"Loading Stream Guard on one GPU: {args.model_path}")
    loaded = load_stream_guard(
        args.model_path,
        dtype=args.dtype,
        # Map the entire model to logical GPU 0. CUDA_VISIBLE_DEVICES chooses
        # which physical GPU is exposed as logical GPU 0.
        device_map={"": 0},
        attention_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        revision=args.revision,
    )
    model_device = require_single_cuda_model(loaded)
    model_name = model_display_name(args.model_path)
    runtime_metadata = collect_runtime_metadata(loaded)
    print(f"Verified unsharded model placement: {model_device}")

    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    metadata: dict[str, Any] = {
        **runtime_metadata,
        "model": model_name,
        "split": "thinking",
        "dataset_path": str(args.dataset_path),
        "started_at_utc": started_at,
        "status": "running",
        "sample_count": len(selected),
        "seed": manifest.get("seed"),
        "manifest_source": manifest_source,
        "manifest_path": str(output_dir / "sample_manifest.json"),
        "warmup_repeats": args.warmup_repeats,
        "warmup_response_tokens": args.warmup_response_tokens,
        "dtype_argument": args.dtype,
        "device_map_argument": {"": 0},
        "attention_implementation_argument": args.attn_implementation,
        "latency_scope": (
            "CUDA sync -> stream_moderate_from_ids -> CUDA sync -> risk label extraction"
        ),
        "strict_policy": "first equal adjacent non-Safe pair: UU or CC",
        "loose_policy": "first adjacent Unsafe pair: UU",
    }
    metadata_path = raw_dir / "run_metadata.json"
    write_json(metadata_path, metadata)

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - requirements include tqdm
        def tqdm(iterable: Any, **_: Any) -> Any:
            return iterable

    processed = 0
    warmed_up = False
    token_path = raw_dir / "token_records.jsonl"
    sample_path = raw_dir / "sample_records.jsonl"
    full_path = raw_dir / "full_forward_records.jsonl"
    mismatch_path = raw_dir / "mismatch_records.jsonl"

    try:
        with (
            JsonlWriter(token_path) as token_writer,
            JsonlWriter(sample_path) as sample_writer,
            JsonlWriter(full_path) as full_writer,
            JsonlWriter(mismatch_path) as mismatch_writer,
        ):
            for sample_order, dataset_index, sample in tqdm(
                selected, desc=f"{model_name} · thinking · stream+full"
            ):
                sequence = prepare_sequence(
                    loaded.tokenizer,
                    sample.messages,
                    "thinking",
                    annotated_input_ids=sample.input_ids,
                    require_alignment=False,
                )

                if not warmed_up and args.warmup_repeats:
                    print(
                        f"Warming up with {args.warmup_repeats} replay(s), "
                        f"{args.warmup_response_tokens} response token(s) each..."
                    )
                    warmup_stream_sample(
                        loaded,
                        sequence,
                        repeats=args.warmup_repeats,
                        response_tokens=args.warmup_response_tokens,
                    )
                    warmed_up = True

                stream = run_stream_sample(
                    loaded,
                    sequence,
                    sample.unique_id,
                    model_name=model_name,
                )
                full = run_full_forward(loaded, sequence)
                consistency = compare_stream_and_full(stream, full)

                stream_all = evaluate_all_policies(
                    stream.risk_levels,
                    full_index_offset=sequence.eval_start_index,
                )
                full_all = evaluate_all_policies(
                    full.risk_levels,
                    full_index_offset=sequence.eval_start_index,
                )
                stream_policies = select_policy_pair(stream_all, stream=True)
                full_policies = select_policy_pair(full_all, stream=False)

                for token_record in stream.token_records:
                    token_writer.write(
                        {
                            **token_record.to_dict(),
                            "sample_order": sample_order,
                            "dataset_index": dataset_index,
                        }
                    )

                full_writer.write(
                    {
                        "model": model_name,
                        "split": "thinking",
                        "sample_id": sample.unique_id,
                        "sample_order": sample_order,
                        "dataset_index": dataset_index,
                        "label": sample.label,
                        **full.to_dict(),
                        "policies": full_policies,
                    }
                )

                mismatch_dicts = consistency.to_dict()["mismatches"]
                for mismatch in mismatch_dicts:
                    mismatch_writer.write(
                        {
                            "model": model_name,
                            "sample_id": sample.unique_id,
                            "sample_order": sample_order,
                            "dataset_index": dataset_index,
                            **mismatch,
                        }
                    )

                token_count = max(
                    consistency.stream_token_count,
                    consistency.full_token_count,
                )
                mismatch_count = len(mismatch_dicts)
                sample_writer.write(
                    {
                        "model": model_name,
                        "split": "thinking",
                        "sample_id": sample.unique_id,
                        "sample_order": sample_order,
                        "dataset_index": dataset_index,
                        "label": sample.label,
                        "source": sample.source,
                        "unsafe_type": sample.unsafe_type,
                        "total_sequence_token_count": len(sequence.input_ids),
                        "prompt_token_count": sequence.prompt_token_count,
                        "bridge_token_count": sequence.bridge_token_count,
                        "response_token_count": stream.response_token_count,
                        "mean_context_length": stream.mean_context_length,
                        "mean_token_latency_ms": stream.mean_token_latency_ms,
                        "prompt_risk_level": stream.prompt.risk_level,
                        "prompt_risk_prob": stream.prompt.risk_prob,
                        "stream_policies": stream_policies,
                        "full_forward_policies": full_policies,
                        "agreement": {
                            "token_count": token_count,
                            "token_label_mismatch_count": mismatch_count,
                            "token_label_agreement": (
                                (token_count - mismatch_count) / token_count
                                if token_count
                                else None
                            ),
                            "all_token_labels_match": consistency.labels_match,
                            "prompt_label_matches": consistency.prompt_label_matches,
                            "strict_prediction_matches": (
                                stream_policies["strict"]["prediction"]
                                == full_policies["strict"]["prediction"]
                            ),
                            "loose_prediction_matches": (
                                stream_policies["loose"]["prediction"]
                                == full_policies["loose"]["prediction"]
                            ),
                        },
                    }
                )

                for writer in (
                    token_writer,
                    sample_writer,
                    full_writer,
                    mismatch_writer,
                ):
                    writer.flush()
                processed += 1

        metadata.update(
            {
                "status": "completed",
                "processed_sample_count": processed,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_time_seconds": time.perf_counter() - wall_start,
            }
        )
        write_json(metadata_path, metadata)
    except BaseException as exc:
        metadata.update(
            {
                "status": "failed",
                "processed_sample_count": processed,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_time_seconds": time.perf_counter() - wall_start,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        write_json(metadata_path, metadata)
        raise

    print(f"Completed {processed}/{len(selected)} shared thinking samples")
    print(f"Manifest: {output_dir / 'sample_manifest.json'}")
    print(f"Raw results: {raw_dir}")


if __name__ == "__main__":
    main()
