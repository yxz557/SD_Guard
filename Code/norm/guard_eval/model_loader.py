"""Lazy loading and runtime metadata for Qwen3Guard-Stream checkpoints."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class LoadedGuard:
    """The tokenizer/model pair plus reproducibility metadata."""

    model: Any
    tokenizer: Any
    model_path: str
    dtype: str
    device_map: Any
    attention_implementation: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def input_device(self) -> Any:
        return model_input_device(self.model)

    @property
    def cuda_devices(self) -> tuple[Any, ...]:
        return model_cuda_devices(self.model)


def _import_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("Qwen3Guard evaluation requires PyTorch.") from exc
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise RuntimeError(
            "Qwen3Guard evaluation requires transformers>=4.55.0."
        ) from exc
    return torch, AutoModel, AutoTokenizer


def resolve_torch_dtype(torch: Any, dtype: str | Any) -> Any:
    """Resolve a CLI-friendly dtype name to a PyTorch dtype."""

    if not isinstance(dtype, str):
        return dtype
    normalised = dtype.strip().casefold().replace("torch.", "")
    mapping = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[normalised]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dtype {dtype!r}; use bfloat16, float16, float32, or auto."
        ) from exc


def _serialise_device_map(device_map: Any) -> Any:
    if isinstance(device_map, Mapping):
        return {str(key): str(value) for key, value in device_map.items()}
    return None if device_map is None else str(device_map)


def unwrap_model(model_or_loaded: Any) -> Any:
    """Return a model while avoiding confusion with the model's backbone field."""

    if isinstance(model_or_loaded, LoadedGuard):
        return model_or_loaded.model
    return model_or_loaded


def model_input_device(model_or_loaded: Any) -> Any:
    """Return the device expected by the model's input embedding layer."""

    model = unwrap_model(model_or_loaded)
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = getattr(model, "device", None)
        if device is None:
            raise RuntimeError("Unable to determine the Guard model input device.")
        return device


def model_cuda_devices(model_or_loaded: Any) -> tuple[Any, ...]:
    """Return every CUDA device involved in a possibly sharded model."""

    torch, _, _ = _import_runtime()
    if not torch.cuda.is_available():
        return ()

    model = unwrap_model(model_or_loaded)
    candidates: list[Any] = []
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        candidates.extend(device_map.values())
    else:
        candidates.append(model_input_device(model))

    devices: dict[int, Any] = {}
    for value in candidates:
        try:
            if isinstance(value, int):
                device = torch.device("cuda", value)
            else:
                device = torch.device(value)
        except (TypeError, RuntimeError, ValueError):
            continue
        if device.type != "cuda":
            continue
        index = torch.cuda.current_device() if device.index is None else device.index
        devices[index] = torch.device("cuda", index)
    return tuple(devices[index] for index in sorted(devices))


def synchronize_model(model_or_loaded: Any) -> None:
    """Synchronize all CUDA devices used by the model; do nothing on CPU."""

    torch, _, _ = _import_runtime()
    for device in model_cuda_devices(model_or_loaded):
        torch.cuda.synchronize(device)


def require_single_cuda_model(
    model_or_loaded: Any, *, expected_index: int = 0
) -> Any:
    """Fail unless every model parameter and buffer is on one CUDA device."""

    torch, _, _ = _import_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3Guard evaluation requires a CUDA GPU")

    model = unwrap_model(model_or_loaded)
    expected = torch.device("cuda", expected_index)
    actual: set[Any] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        actual.add(tensor.device)

    if actual != {expected}:
        rendered = ", ".join(sorted(str(device) for device in actual)) or "<none>"
        raise RuntimeError(
            "The Guard model must be entirely resident on one GPU. "
            f"Expected only {expected}, found: {rendered}. "
            "Set CUDA_VISIBLE_DEVICES to one physical GPU before starting Python."
        )

    mapped = model_cuda_devices(model)
    if mapped != (expected,):
        raise RuntimeError(
            "Unexpected Hugging Face device map for the Guard model: "
            f"{tuple(str(device) for device in mapped)!r}"
        )
    return expected


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_metadata(
    model_or_loaded: Any | None = None,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Collect JSON-ready software, hardware, and checkpoint information."""

    torch, _, _ = _import_runtime()
    model = None if model_or_loaded is None else unwrap_model(model_or_loaded)
    if isinstance(model_or_loaded, LoadedGuard) and model_path is None:
        model_path = model_or_loaded.model_path

    gpu_info: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        devices = (
            model_cuda_devices(model)
            if model is not None
            else tuple(torch.device("cuda", index) for index in range(torch.cuda.device_count()))
        )
        for device in devices:
            properties = torch.cuda.get_device_properties(device)
            gpu_info.append(
                {
                    "device": str(device),
                    "name": properties.name,
                    "compute_capability": list(torch.cuda.get_device_capability(device)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )

    config_summary: dict[str, Any] = {}
    if model is not None and getattr(model, "config", None) is not None:
        config = model.config
        for key in (
            "architectures",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "guard_inner_size",
            "num_risk_level",
            "num_query_risk_level",
            "torch_dtype",
        ):
            value = getattr(config, key, None)
            if value is not None:
                config_summary[key] = str(value) if key == "torch_dtype" else value
        config_summary["attention_implementation"] = getattr(
            config, "_attn_implementation", None
        )
        config_summary["commit_hash"] = getattr(config, "_commit_hash", None)

    return {
        "model_path": None if model_path is None else str(model_path),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "transformers_version": _package_version("transformers"),
        "datasets_version": _package_version("datasets"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": (
            None if not torch.cuda.is_available() else torch.backends.cudnn.version()
        ),
        "gpus": gpu_info,
        "hf_device_map": (
            None
            if model is None
            else _serialise_device_map(getattr(model, "hf_device_map", None))
        ),
        "model_config": config_summary,
    }


def load_stream_guard(
    model_path: str | Path,
    *,
    dtype: str | Any = "bfloat16",
    device_map: str | Mapping[str, Any] | None = "auto",
    attention_implementation: str | None = None,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
    revision: str | None = None,
    **model_kwargs: Any,
) -> LoadedGuard:
    """Load a Qwen3Guard-Stream model and verify its streaming API."""

    torch, AutoModel, AutoTokenizer = _import_runtime()
    path = str(Path(model_path).expanduser()) if Path(model_path).exists() else str(model_path)
    resolved_dtype = resolve_torch_dtype(torch, dtype)

    shared_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if revision is not None:
        shared_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(path, **shared_kwargs)
    load_kwargs = dict(shared_kwargs)
    load_kwargs.update(model_kwargs)
    load_kwargs["torch_dtype"] = resolved_dtype
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if attention_implementation is not None:
        load_kwargs["attn_implementation"] = attention_implementation

    model = AutoModel.from_pretrained(path, **load_kwargs).eval()
    for method_name in ("stream_moderate_from_ids", "close_stream"):
        if not callable(getattr(model, method_name, None)):
            raise TypeError(
                f"Checkpoint {path!r} does not expose {method_name}(); "
                "verify that it is a Qwen3Guard-Stream checkpoint and that "
                "trust_remote_code=True."
            )

    loaded = LoadedGuard(
        model=model,
        tokenizer=tokenizer,
        model_path=path,
        dtype=str(resolved_dtype),
        device_map=_serialise_device_map(getattr(model, "hf_device_map", device_map)),
        attention_implementation=getattr(
            model.config, "_attn_implementation", attention_implementation
        ),
    )
    loaded.metadata = collect_runtime_metadata(loaded)
    return loaded


__all__ = [
    "LoadedGuard",
    "collect_runtime_metadata",
    "load_stream_guard",
    "model_cuda_devices",
    "model_input_device",
    "require_single_cuda_model",
    "resolve_torch_dtype",
    "synchronize_model",
    "unwrap_model",
]
