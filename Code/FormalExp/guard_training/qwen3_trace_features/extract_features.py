from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from guard_training.common.feature_store import (
    STEP_KIND_TO_ID,
    FeatureShardWriter,
    load_processed_sample_ids,
)
from guard_training.common.trace_io import TraceRecord, iter_trace_records
from guard_training.common.trace_serialization import (
    DEFAULT_READOUT_PROMPT,
    serialize_messages,
    tokenize_serialized,
)


STEP_KINDS = set(STEP_KIND_TO_ID)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefill complete AgentDojo traces with Qwen3 and save pooled hidden states."
    )
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-4B")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--storage-dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="auto",
    )
    parser.add_argument("--readout-prompt", default=DEFAULT_READOUT_PROMPT)
    parser.add_argument("--no-readout", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def discover_layers_and_norm(model: torch.nn.Module):
    layer_candidates = (
        ("layers",),
        ("model", "layers"),
        ("transformer", "h"),
    )
    layers = None
    for path in layer_candidates:
        value: Any = model
        for part in path:
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None:
            layers = value
            break
    if layers is None:
        raise ValueError("Could not locate decoder layers on the loaded model")

    norm_candidates = (
        ("norm",),
        ("model", "norm"),
        ("transformer", "ln_f"),
    )
    final_norm = None
    for path in norm_candidates:
        value = model
        for part in path:
            value = getattr(value, part, None)
            if value is None:
                break
        if value is not None:
            final_norm = value
            break
    if final_norm is None:
        raise ValueError("Could not locate final normalization layer on the loaded model")
    return list(layers), final_norm


class LayerFeatureCollector:
    def __init__(self, layers, final_norm, storage_dtype: torch.dtype) -> None:
        self.storage_dtype = storage_dtype
        self.trace_end = 0
        self.readout_end: int | None = None
        self.step_positions: list[int] = []
        self.captured: list[dict[str, torch.Tensor]] = []
        self.handles = []
        for layer in [*layers, final_norm]:
            self.handles.append(layer.register_forward_hook(self._hook))

    def prepare(self, trace_end: int, readout_end: int | None, step_positions: list[int]) -> None:
        self.trace_end = trace_end
        self.readout_end = readout_end
        self.step_positions = step_positions
        self.captured = []

    def _hook(self, _module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError(f"Expected layer output [1,tokens,hidden], got {hidden.shape}")
        sequence = hidden[0]
        trace_slice = sequence[: self.trace_end + 1]
        last32_start = max(0, self.trace_end - 31)
        values: dict[str, torch.Tensor] = {
            "trace_last": sequence[self.trace_end],
            "trace_last32_mean": sequence[last32_start : self.trace_end + 1].mean(
                dim=0, dtype=torch.float32
            ),
            "trace_mean": trace_slice.mean(dim=0, dtype=torch.float32),
        }
        if self.readout_end is not None:
            values["readout_last"] = sequence[self.readout_end]
        if self.step_positions:
            index = torch.tensor(self.step_positions, device=sequence.device, dtype=torch.long)
            values["steps"] = sequence.index_select(0, index)
        else:
            values["steps"] = sequence.new_empty((0, sequence.shape[-1]))
        self.captured.append(
            {
                key: value.detach().to(device="cpu", dtype=self.storage_dtype)
                for key, value in values.items()
            }
        )

    def finish(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        if not self.captured:
            raise RuntimeError("No layer hooks fired during model forward")
        feature_names = [name for name in self.captured[0] if name != "steps"]
        global_features = {
            name: torch.stack([layer[name] for layer in self.captured], dim=0)
            for name in feature_names
        }
        # Each layer has [steps, hidden]; output is [steps, layers, hidden].
        step_features = torch.stack([layer["steps"] for layer in self.captured], dim=1)
        return global_features, step_features

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise ValueError(f"Existing configuration differs from this run: {path}")
        return
    path.write_text(serialized, encoding="utf-8", newline="\n")


def append_rejected(path: Path, record: TraceRecord, reason: str, **details: Any) -> None:
    payload = record.metadata()
    payload.update({"reason": reason, **details})
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = args.output_dir / "rejected.jsonl"

    model_dtype = torch_dtype(args.model_dtype)
    storage_dtype = torch_dtype(args.storage_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": model_dtype,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    model.to(args.device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    layers, final_norm = discover_layers_and_norm(model)
    collector = LayerFeatureCollector(layers, final_norm, storage_dtype)
    model_limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    max_length = args.max_length or model_limit
    if max_length <= 0:
        raise ValueError("Could not infer max length; pass --max-length explicitly")

    readout_prompt = None if args.no_readout else args.readout_prompt
    model_config_json = json.dumps(
        model.config.to_dict(), ensure_ascii=False, sort_keys=True, default=str
    )
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    config = {
        "schema_version": "sd_guard.qwen3_trace_features.v1",
        "serialization_version": "sd_guard.qwen_chatml_tools.v1",
        "trace_root": str(args.trace_root.resolve()),
        "model_path": args.model_path,
        "model_type": str(getattr(model.config, "model_type", "")),
        "model_config_sha256": hashlib.sha256(model_config_json.encode("utf-8")).hexdigest(),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "num_decoder_layers": len(layers),
        "saved_layer_count": len(layers) + 1,
        "hidden_size": int(getattr(model.config, "hidden_size", 0)),
        "max_length": max_length,
        "model_dtype": args.model_dtype,
        "storage_dtype": args.storage_dtype,
        "attn_implementation": args.attn_implementation,
        "readout_prompt": readout_prompt,
        "global_features": [
            "trace_last",
            "trace_last32_mean",
            "trace_mean",
            *(["readout_last"] if readout_prompt else []),
        ],
        "step_kinds": STEP_KIND_TO_ID,
        "step_feature": "all saved layers at each causal step boundary",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    write_json_once(args.output_dir / "feature_config.json", config)

    writer = FeatureShardWriter(feature_dir, args.shard_size)
    processed = load_processed_sample_ids(feature_dir)
    completed = 0
    skipped = 0
    failed = 0

    try:
        for record_index, record in enumerate(iter_trace_records(args.trace_root)):
            if args.limit is not None and completed >= args.limit:
                break
            if record.sample_id in processed:
                skipped += 1
                continue

            try:
                serialized = serialize_messages(record.messages, readout_prompt=readout_prompt)
                tokenized = tokenize_serialized(tokenizer, serialized)
                token_count = len(tokenized.input_ids)
                if token_count > max_length:
                    append_rejected(
                        rejected_path,
                        record,
                        "overlength",
                        token_count=token_count,
                        max_length=max_length,
                    )
                    failed += 1
                    continue

                step_metadata = [
                    boundary for boundary in tokenized.boundaries if boundary["kind"] in STEP_KINDS
                ]
                step_positions = [int(step["token_index"]) for step in step_metadata]
                collector.prepare(
                    trace_end=tokenized.trace_end_token,
                    readout_end=tokenized.readout_end_token,
                    step_positions=step_positions,
                )

                input_ids = torch.tensor(
                    [tokenized.input_ids], dtype=torch.long, device=args.device
                )
                attention_mask = torch.tensor(
                    [tokenized.attention_mask], dtype=torch.long, device=args.device
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                del output, input_ids, attention_mask
                global_features, step_features = collector.finish()

                metadata = record.metadata()
                metadata.update(
                    {
                        "token_count": token_count,
                        "trace_end_token": tokenized.trace_end_token,
                        "readout_end_token": tokenized.readout_end_token,
                        "serialized_sha256": hashlib.sha256(
                            serialized.text.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                writer.add(metadata, global_features, step_features, step_metadata)
                completed += 1
                if completed % args.log_every == 0:
                    print(
                        f"completed={completed} skipped={skipped} rejected_or_failed={failed} "
                        f"last_tokens={token_count} sample={record.sample_id}",
                        flush=True,
                    )
            except Exception as exc:
                failed += 1
                append_rejected(
                    rejected_path,
                    record,
                    "exception",
                    exception_type=type(exc).__name__,
                    exception=str(exc),
                    traceback=traceback.format_exc(),
                )
                if not args.continue_on_error:
                    raise
                print(f"ERROR sample={record.sample_id}: {exc}", file=sys.stderr, flush=True)
    finally:
        writer.flush()
        collector.close()

    print(
        json.dumps(
            {"completed": completed, "skipped": skipped, "rejected_or_failed": failed},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

