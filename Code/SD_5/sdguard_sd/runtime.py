"""Model loading, device validation, and reproducibility metadata."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from functools import wraps
from pathlib import Path
from typing import Any


def import_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised on the GPU host
        raise RuntimeError("Install requirements.txt before running the experiment") from exc
    return torch, AutoModel, AutoModelForCausalLM, AutoTokenizer


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


def ensure_pad_token_id(config: Any, tokenizer: Any) -> Any:
    """Backfill the token ID expected by Qwen3Guard's remote model code."""

    try:
        config.pad_token_id
    except AttributeError:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError(
                "Qwen3Guard config has no pad_token_id and its tokenizer does "
                "not provide a pad_token_id or eos_token_id"
            )
        config.pad_token_id = int(pad_token_id)
    return config


def legacy_default_rope_parameters(
    config: Any,
    device: Any = None,
    seq_len: int | None = None,
    **_: Any,
) -> tuple[Any, float]:
    """Reproduce the standard RoPE initializer removed in Transformers 5."""

    del seq_len
    import torch

    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


def ensure_legacy_default_rope_initializer() -> bool:
    """Restore the legacy default RoPE lookup only when it is absent."""

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return False
    ROPE_INIT_FUNCTIONS["default"] = legacy_default_rope_parameters
    return True


def legacy_masking_api_adapter(function: Any) -> Any:
    """Adapt a Transformers 5 mask factory to the Qwen3Guard 4.55 API."""

    @wraps(function)
    def compatible(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs:
            if "inputs_embeds" in kwargs:
                raise TypeError(
                    "Mask factory received both input_embeds and inputs_embeds"
                )
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        # Transformers 5 derives the query offset from past_key_values and no
        # longer accepts the explicit Transformers 4.x cache_position input.
        kwargs.pop("cache_position", None)
        return function(*args, **kwargs)

    compatible._sd5_legacy_masking_adapter = True
    return compatible


def ensure_legacy_masking_api() -> bool:
    """Bridge Qwen3Guard's Transformers 4.55 mask calls on Transformers 5."""

    import inspect
    from transformers import masking_utils

    changed = False
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        function = getattr(masking_utils, name)
        if getattr(function, "_sd5_legacy_masking_adapter", False):
            continue
        parameters = inspect.signature(function).parameters
        if "input_embeds" in parameters or "inputs_embeds" not in parameters:
            continue
        setattr(masking_utils, name, legacy_masking_api_adapter(function))
        changed = True
    return changed


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
    torch, _, AutoModelForCausalLM, AutoTokenizer = import_runtime()
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


def load_guard_model(
    source: str | Path,
    *,
    device: str,
    dtype: str,
    local_files_only: bool,
    revision: str | None,
    attention_implementation: str | None,
) -> tuple[Any, Any]:
    torch, AutoModel, _, AutoTokenizer = import_runtime()
    from transformers import AutoConfig

    model_source = normalize_model_source(source)
    shared: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    if revision is not None:
        shared["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_source, **shared)
    config = AutoConfig.from_pretrained(model_source, **shared)
    ensure_pad_token_id(config, tokenizer)
    ensure_legacy_default_rope_initializer()
    ensure_legacy_masking_api()
    model_kwargs = dict(shared)
    model_kwargs["config"] = config
    model_kwargs["torch_dtype"] = resolve_dtype(torch, dtype)
    model_kwargs["device_map"] = {"": device}
    if attention_implementation:
        model_kwargs["attn_implementation"] = attention_implementation
    model = AutoModel.from_pretrained(model_source, **model_kwargs).eval()
    if not hasattr(model, "risk_level_head"):
        raise TypeError(f"{model_source!r} is not a Qwen3Guard-Stream checkpoint")
    require_device(model, device, name=model_source)
    return model, tokenizer


def require_device(model: Any, expected: str, *, name: str) -> None:
    torch, _, _, _ = import_runtime()
    expected_device = torch.device(expected)
    actual = {parameter.device for parameter in model.parameters()}
    actual.update(buffer.device for buffer in model.buffers())
    if actual != {expected_device}:
        rendered = ", ".join(sorted(str(device) for device in actual)) or "<none>"
        raise RuntimeError(f"Model {name!r} must be entirely on {expected_device}; found {rendered}")


def assert_same_tokenizer(reference: Any, candidate: Any, *, candidate_name: str) -> None:
    if len(reference) != len(candidate):
        raise RuntimeError(
            f"Tokenizer size mismatch for {candidate_name}: {len(reference)} != {len(candidate)}"
        )
    if reference.get_vocab() != candidate.get_vocab():
        raise RuntimeError(f"Tokenizer vocabulary mismatch for {candidate_name}")
    tokens = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    for token in tokens:
        left = reference.convert_tokens_to_ids(token)
        right = candidate.convert_tokens_to_ids(token)
        if left != right:
            raise RuntimeError(
                f"Special token mismatch for {candidate_name}: {token!r} -> {left!r}/{right!r}"
            )


def synchronize_devices(*devices: str) -> None:
    torch, _, _, _ = import_runtime()
    if not torch.cuda.is_available():
        return
    seen: set[str] = set()
    for value in devices:
        device = torch.device(value)
        if device.type == "cuda" and str(device) not in seen:
            torch.cuda.synchronize(device)
            seen.add(str(device))


def collect_runtime_metadata() -> dict[str, Any]:
    torch, _, _, _ = import_runtime()

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

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
        "transformers_version": version("transformers"),
        "datasets_version": version("datasets"),
        "cuda_runtime_version": torch.version.cuda,
        "gpus": gpus,
    }


__all__ = [
    "assert_same_tokenizer",
    "collect_runtime_metadata",
    "ensure_legacy_default_rope_initializer",
    "ensure_legacy_masking_api",
    "ensure_pad_token_id",
    "import_runtime",
    "legacy_default_rope_parameters",
    "legacy_masking_api_adapter",
    "load_generation_model",
    "load_guard_model",
    "normalize_model_source",
    "require_device",
    "synchronize_devices",
]


