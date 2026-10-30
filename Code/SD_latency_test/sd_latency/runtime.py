"""Minimal model-loading and CUDA timing helpers for the latency harness."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any


def import_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on the GPU host
        raise RuntimeError("Install requirements.txt before running this benchmark") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    try:
        return mapping[name.casefold()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name!r}") from exc


def normalize_model_source(source: str | Path) -> str:
    path = Path(source).expanduser()
    return str(path.resolve()) if path.exists() else str(source)


def load_generation_model(
    source: str | Path,
    *,
    device: str,
    dtype: str,
    local_files_only: bool,
    revision: str | None,
    attention_implementation: str | None,
) -> tuple[Any, Any]:
    torch, AutoModelForCausalLM, AutoTokenizer = import_runtime()
    model_source = normalize_model_source(source)
    shared: dict[str, Any] = {"local_files_only": local_files_only}
    if revision is not None:
        shared["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_source, **shared)
    model_kwargs = dict(shared)
    model_kwargs["torch_dtype"] = resolve_dtype(torch, dtype)
    model_kwargs["device_map"] = {"": device}
    if attention_implementation:
        model_kwargs["attn_implementation"] = attention_implementation
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs).eval()
    require_device(model, device, name=model_source)
    return model, tokenizer


def require_device(model: Any, expected: str, *, name: str) -> None:
    torch, _, _ = import_runtime()
    expected_device = torch.device(expected)
    actual = {parameter.device for parameter in model.parameters()}
    actual.update(buffer.device for buffer in model.buffers())
    if actual != {expected_device}:
        rendered = ", ".join(sorted(str(device) for device in actual)) or "<none>"
        raise RuntimeError(f"Model {name!r} must be entirely on {expected_device}; found {rendered}")


def assert_same_tokenizer(reference: Any, candidate: Any) -> None:
    if len(reference) != len(candidate) or reference.get_vocab() != candidate.get_vocab():
        raise RuntimeError("Draft and target must use the same tokenizer vocabulary")
    for token in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
        left = reference.convert_tokens_to_ids(token)
        right = candidate.convert_tokens_to_ids(token)
        if left != right:
            raise RuntimeError(f"Special token mismatch for {token!r}: {left!r} != {right!r}")


def synchronize_devices(*devices: str) -> None:
    torch, _, _ = import_runtime()
    if not torch.cuda.is_available():
        return
    seen: set[str] = set()
    for value in devices:
        device = torch.device(value)
        rendered = str(device)
        if device.type == "cuda" and rendered not in seen:
            torch.cuda.synchronize(device)
            seen.add(rendered)


def collect_runtime_metadata() -> dict[str, Any]:
    torch, _, _ = import_runtime()
    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "device": f"cuda:{index}",
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "transformers_version": importlib.metadata.version("transformers"),
        "cuda_runtime_version": torch.version.cuda,
        "gpus": gpus,
    }
