"""Model loading, device validation, and reproducibility metadata."""

from __future__ import annotations

import importlib.metadata
import inspect
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
        raise RuntimeError(
            "Install requirements-vllm.txt in the pinned sd-vllm-cu129 environment"
        ) from exc
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


def normalize_model_source(source: str | Path) -> str:
    path = Path(source).expanduser()
    return str(path.resolve()) if path.exists() else str(source)


def ensure_guard_config(config: Any, tokenizer: Any) -> Any:
    """Materialize legacy config fields required by Qwen3Guard custom code."""

    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError(
                "Qwen3Guard config has no pad_token_id and its tokenizer does "
                "not provide a pad_token_id or eos_token_id"
            )
        config.pad_token_id = int(pad_token_id)

    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        count = int(getattr(config, "num_hidden_layers", 0))
        if count <= 0:
            raise RuntimeError("Qwen3Guard config has no valid num_hidden_layers")
        if bool(getattr(config, "use_sliding_window", False)) or getattr(
            config, "sliding_window", None
        ) is not None:
            raise RuntimeError(
                "Qwen3Guard config is missing layer_types for a sliding-window model"
            )
        config.layer_types = ["full_attention"] * count
    elif len(layer_types) != int(config.num_hidden_layers):
        raise RuntimeError(
            "Qwen3Guard layer_types length does not match num_hidden_layers"
        )
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

    base = float(config.rope_theta)
    partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    if dim <= 0 or dim % 2:
        raise RuntimeError(f"Qwen3Guard has an invalid rotary dimension: {dim}")
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
    """Restore the Qwen3Guard 4.x ``default`` RoPE lookup on Transformers 5."""

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return False
    ROPE_INIT_FUNCTIONS["default"] = legacy_default_rope_parameters
    return True


def legacy_masking_api_adapter(function: Any) -> Any:
    """Translate Qwen3Guard's Transformers 4 mask call to the 5.17 API."""

    @wraps(function)
    def compatible(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs:
            if "inputs_embeds" in kwargs:
                raise TypeError(
                    "Mask factory received both input_embeds and inputs_embeds"
                )
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        # Transformers 5.17 derives the offset from the cache and no longer
        # accepts the explicit Transformers 4.x cache_position argument.
        kwargs.pop("cache_position", None)
        return function(*args, **kwargs)

    compatible._sd4_legacy_masking_adapter = True
    return compatible


def ensure_legacy_masking_api() -> tuple[str, ...]:
    """Patch only the mask entry points whose signatures changed in 5.17."""

    from transformers import masking_utils

    changed: list[str] = []
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        function = getattr(masking_utils, name)
        if getattr(function, "_sd4_legacy_masking_adapter", False):
            continue
        parameters = inspect.signature(function).parameters
        if "input_embeds" in parameters or "inputs_embeds" not in parameters:
            continue
        setattr(masking_utils, name, legacy_masking_api_adapter(function))
        changed.append(name)
    return tuple(changed)


def crop_cache_to_length(cache: Any, retained_length: int) -> None:
    """Roll a Transformers 5.17 cache back using its non-deprecated API."""

    current_length = int(cache.get_seq_length())
    retained = int(retained_length)
    if retained < 0 or retained > current_length:
        raise RuntimeError(
            f"Invalid Guard cache rollback: retain {retained} of {current_length} tokens"
        )
    tokens_to_remove = current_length - retained
    if tokens_to_remove:
        cache.crop(-tokens_to_remove)
    actual = int(cache.get_seq_length())
    if actual != retained:
        raise RuntimeError(
            f"Guard cache rollback produced length {actual}, expected {retained}"
        )


def _validate_guard_outputs(
    outputs: Any, *, token_count: int, config: Any
) -> dict[str, list[int]]:
    expected = {
        "risk_level_logits": int(config.num_risk_level),
        "category_logits": int(config.num_category),
        "query_risk_level_logits": int(config.num_query_risk_level),
        "query_category_logits": int(config.num_query_category),
    }
    shapes: dict[str, list[int]] = {}
    for name, width in expected.items():
        value = getattr(outputs, name, None)
        shape = tuple(getattr(value, "shape", ()))
        if len(shape) != 3 or shape[0] != 1 or shape[1] != token_count or shape[2] != width:
            raise RuntimeError(
                f"Qwen3Guard returned invalid {name} shape {shape}; "
                f"expected (1, {token_count}, {width})"
            )
        shapes[name] = [int(part) for part in shape]
    cache = getattr(outputs, "past_key_values", None)
    if cache is None or not callable(getattr(cache, "get_seq_length", None)):
        raise RuntimeError("Qwen3Guard did not return a readable DynamicCache")
    if not callable(getattr(cache, "crop", None)):
        raise RuntimeError("Qwen3Guard DynamicCache does not support rollback")
    return shapes


def smoke_test_guard_compatibility(
    model: Any, tokenizer: Any, *, device: str
) -> dict[str, Any]:
    """Exercise prefill, incremental decode, outputs, and KV rollback once."""

    torch, _, _, _ = import_runtime()
    encoded = tokenizer.encode("SD4 guard compatibility", add_special_tokens=False)
    token_ids = [int(value) for value in encoded[:2]]
    if not token_ids:
        fallback = getattr(tokenizer, "eos_token_id", None)
        if fallback is None:
            fallback = getattr(tokenizer, "bos_token_id", None)
        if fallback is None:
            raise RuntimeError("Guard tokenizer cannot produce a compatibility-test token")
        token_ids = [int(fallback)]
    next_token = int(
        getattr(tokenizer, "eos_token_id", None)
        or getattr(tokenizer, "bos_token_id", None)
        or token_ids[-1]
    )
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    next_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
    with torch.inference_mode():
        prefill = model.forward(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=0,
            return_dict=True,
        )
        prefill_shapes = _validate_guard_outputs(
            prefill, token_count=len(token_ids), config=model.config
        )
        cache = prefill.past_key_values
        prefill_length = int(cache.get_seq_length())
        if prefill_length != len(token_ids):
            raise RuntimeError(
                f"Guard prefill cache length {prefill_length} != {len(token_ids)}"
            )
        incremental = model.forward(
            input_ids=next_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=0,
            return_dict=True,
        )
        incremental_shapes = _validate_guard_outputs(
            incremental, token_count=1, config=model.config
        )
        cache = incremental.past_key_values
        if int(cache.get_seq_length()) != prefill_length + 1:
            raise RuntimeError("Guard incremental forward did not append one KV token")
        crop_cache_to_length(cache, prefill_length)
    synchronize_devices(device)
    return {
        "status": "passed",
        "prefill_token_count": len(token_ids),
        "prefill_output_shapes": prefill_shapes,
        "incremental_output_shapes": incremental_shapes,
        "cache_class": type(cache).__name__,
        "cache_rollback": "negative tokens_to_remove",
    }


def _model_dtype_keyword() -> str:
    """Return the pinned Transformers 5 loading keyword."""

    major = int(importlib.metadata.version("transformers").split(".", 1)[0])
    return "dtype" if major >= 5 else "torch_dtype"


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
    model_kwargs[_model_dtype_keyword()] = resolve_dtype(torch, dtype)
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
    ensure_guard_config(config, tokenizer)
    rope_patched = ensure_legacy_default_rope_initializer()
    mask_patches = ensure_legacy_masking_api()
    model_kwargs = dict(shared)
    model_kwargs["config"] = config
    model_kwargs[_model_dtype_keyword()] = resolve_dtype(torch, dtype)
    model_kwargs["device_map"] = {"": device}
    if attention_implementation:
        model_kwargs["attn_implementation"] = attention_implementation
    model = AutoModel.from_pretrained(model_source, **model_kwargs).eval()
    if not hasattr(model, "risk_level_head"):
        raise TypeError(f"{model_source!r} is not a Qwen3Guard-Stream checkpoint")
    require_device(model, device, name=model_source)
    compatibility = smoke_test_guard_compatibility(model, tokenizer, device=device)
    compatibility.update(
        {
            "transformers_version": importlib.metadata.version("transformers"),
            "pad_token_id": int(config.pad_token_id),
            "default_rope_initializer_added": rope_patched,
            "mask_functions_adapted": list(mask_patches),
            "model_source": model_source,
        }
    )
    model._sd4_compatibility_report = compatibility
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
    "crop_cache_to_length",
    "ensure_guard_config",
    "ensure_legacy_default_rope_initializer",
    "ensure_legacy_masking_api",
    "import_runtime",
    "legacy_default_rope_parameters",
    "legacy_masking_api_adapter",
    "load_generation_model",
    "load_guard_model",
    "normalize_model_source",
    "require_device",
    "smoke_test_guard_compatibility",
    "synchronize_devices",
]


