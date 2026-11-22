"""Exact original-model *prefix* execution for earlier-layer activations.

Why this is the original model and not a new one
------------------------------------------------
A transformer's activation after block ``b`` depends only on blocks ``0..b`` and on the
input token ids / positions.  Later blocks cannot influence it.  So executing
``embed_tokens + rotary_emb + layers[0:B]`` of the *unmodified frozen* Qwen3-8B weights on
the *same* token ids and positions yields bit-identical values to the corresponding
intermediate tensor of a full 32/36-block forward on a machine with enough VRAM.

Nothing is quantised, truncated, re-templated or re-tokenised here: the token ids are the
already accepted archived/render sequences, weights are read byte-for-byte out of the
original safetensors shards, and the module graph is the original
``transformers.models.qwen3.modeling_qwen3`` graph (layers are *materialised*, not
re-implemented).  Layer numbering is HF's: the returned tensor for block ``B`` is
``output_hidden_states[B]``, i.e. the residual-stream state **after** blocks ``0..B-1``.
The final ``model.norm`` is never applied (it belongs after the last block only).

Two execution paths exist here:

``PrefixModel``
    Production capture path.  ``Qwen3Model`` built on the meta device, only
    ``embed_tokens`` and ``layers[0:B]`` materialised on the GPU+bf16, and the original
    ``Qwen3Model.forward`` loop stopped through a ``forward_pre_hook`` on ``layers[B]``
    that raises.  Because the hook fires *before* ``layers[B]`` runs, the loop stops
    exactly at ``output_hidden_states[B]`` and ``model.norm`` is never reached.

``ReplicatedPrefix``
    Independent hand-written loop over the same modules with an explicitly constructed
    causal-mask mapping.  Used only as a cross-check that the hook path is not silently
    reading a different tensor/position/norm.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from . import HIDDEN, MODEL_PATH
from .common import StudyError, sha256_file

EXPECTED_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "rope_theta": 1_000_000,
    "rms_norm_eps": 1e-06,
    "tie_word_embeddings": False,
    "sliding_window": None,
    "use_sliding_window": False,
    "max_position_embeddings": 40960,
}


class _Stop(Exception):
    """Raised by the early-exit pre-hook; carries the captured tensors."""


def verify_config(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    problems = []
    for key, expected in EXPECTED_CONFIG.items():
        if raw.get(key) != expected:
            problems.append({"key": key, "expected": expected, "actual": raw.get(key)})
    if raw.get("architectures") != ["Qwen3ForCausalLM"]:
        problems.append({"key": "architectures", "expected": ["Qwen3ForCausalLM"], "actual": raw.get("architectures")})
    if problems:
        raise StudyError(f"model config is not the accepted Qwen3-8B config: {problems}")
    return raw


def shard_files(model_path: Path = MODEL_PATH) -> list[Path]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    names = sorted(set(index["weight_map"].values()))
    return [model_path / name for name in names]


def weight_map(model_path: Path = MODEL_PATH) -> dict[str, str]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return index["weight_map"]


def expected_prefix_keys(blocks: int) -> list[str]:
    keys = ["model.embed_tokens.weight"]
    for i in range(blocks):
        keys.extend(
            [
                f"model.layers.{i}.input_layernorm.weight",
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"model.layers.{i}.self_attn.q_norm.weight",
                f"model.layers.{i}.self_attn.k_norm.weight",
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
                f"model.layers.{i}.mlp.down_proj.weight",
            ]
        )
    return sorted(keys)


def _read_shard_tensors(files: Sequence[Path], wanted: set[str]) -> dict[str, torch.Tensor]:
    """Read only the wanted keys, straight out of the original safetensors shards."""
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as fh:
            present = set(fh.keys()) & wanted
            if not present:
                continue
            for key in sorted(present):
                out[key] = fh.get_tensor(key)
    missing = wanted - set(out)
    if missing:
        raise StudyError(f"missing weights in the original shards: {sorted(missing)[:5]}")
    return out


def _build_meta_model(model_path: Path):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding
    from transformers import Qwen3Config

    config = Qwen3Config.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = "sdpa"
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = Qwen3Model(config)
    finally:
        torch.set_default_dtype(previous)
    # RoPE tables are derived from the config (no learned weights); build them for real.
    model.rotary_emb = Qwen3RotaryEmbedding(config)
    model.eval()
    model.requires_grad_(False)
    return config, model



class PrefixModel:
    """Original Qwen3-8B modules for ``embed_tokens + layers[0:blocks]``, GPU+bf16."""

    def __init__(
        self,
        blocks: int = 8,
        captures: Sequence[int] = (4, 8),
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_path: Path = MODEL_PATH,
        verify_keys: bool = True,
    ) -> None:
        if not (1 <= blocks <= 36):
            raise StudyError("blocks out of range")
        if any(not (1 <= c <= blocks) for c in captures):
            raise StudyError("capture block must be inside the executed prefix")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        self.config, self.model = _build_meta_model(model_path)
        self.blocks = blocks
        self.captures = tuple(sorted(set(captures) | {blocks}))
        self.device = device
        self.dtype = dtype
        self.model_path = model_path

        files = shard_files(model_path)
        wanted = set(expected_prefix_keys(blocks))
        tensors = _read_shard_tensors(files, wanted)
        if verify_keys:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

            with torch.device("meta"):
                probe = Qwen3DecoderLayer(self.config, layer_idx=0)
            probe_keys = {
                f"model.layers.0.{k}" for k in probe.state_dict().keys()
            }
            got = {k for k in tensors if k.startswith("model.layers.0.")}
            if probe_keys != got:
                raise StudyError(
                    f"layer state_dict keys do not match the shard keys: extra={sorted(probe_keys - got)} "
                    f"missing={sorted(got - probe_keys)}"
                )

        # materialise only what we execute
        self.model.embed_tokens.to_empty(device=device).to(dtype=dtype)
        self.model.rotary_emb.to(device)
        for i in range(blocks):
            self.model.layers[i].to_empty(device=device).to(dtype=dtype)

        # copy in place (load_state_dict) so every matrix exists once, on the GPU
        self.model.embed_tokens.load_state_dict(
            {"weight": tensors.pop("model.embed_tokens.weight").to(dtype=dtype)}, strict=True
        )
        for i in range(blocks):
            prefix = f"model.layers.{i}."
            sub = {
                k[len(prefix):]: v.to(dtype=dtype)
                for k, v in tensors.items()
                if k.startswith(prefix)
            }
            for key in [k for k in tensors if k.startswith(prefix)]:
                del tensors[key]
            self.model.layers[i].load_state_dict(sub, strict=True)
            del sub
        if tensors:
            raise StudyError(f"unused tensors: {sorted(tensors)[:3]}")
        del tensors
        self.loaded_keys = expected_prefix_keys(blocks)

        self._captured: dict[int, torch.Tensor] = {}
        self._handles = []
        for index in self.captures:
            self._handles.append(
                self.model.layers[index].register_forward_pre_hook(self._make_hook(index), with_kwargs=False)
            )

    # ------------------------------------------------------------------ hooks
    def _make_hook(self, index: int):
        def hook(module, args):
            hidden = args[0]
            if index in self.captures:
                self._captured[index] = hidden
            if index == self.blocks:
                raise _Stop()
            return None

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------ forward
    def forward_ids(self, input_ids: Sequence[int] | torch.Tensor) -> dict[int, torch.Tensor]:
        """Run the prefix; return ``{block: hidden[1, L, 4096]}`` on the GPU."""
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.to(device=self.device, dtype=torch.long)
        else:
            ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
        if ids.shape[1] > self.config.max_position_embeddings:
            raise StudyError("sequence exceeds model context; truncation is forbidden")
        self._captured = {}
        with torch.inference_mode():
            try:
                self.model(input_ids=ids, use_cache=False, return_dict=True)
            except _Stop:
                pass
        missing = [b for b in self.captures if b not in self._captured]
        if missing:
            raise StudyError(f"capture hook did not fire for blocks {missing}")
        return dict(self._captured)

    def endpoint_vectors(
        self, input_ids: Sequence[int], positions: Sequence[int]
    ) -> dict[int, torch.Tensor]:
        """Capture block vectors at exact token positions -> cpu bf16 ``[n, 4096]``."""
        pos = list(positions)
        if not pos or pos != sorted(set(pos)) or pos[-1] >= len(input_ids):
            raise StudyError("positions must be unique, increasing and inside the sequence")
        captured = self.forward_ids(input_ids)
        out = {}
        for block, hidden in captured.items():
            idx = torch.tensor(pos, dtype=torch.long, device=hidden.device)
            out[block] = hidden[0].index_select(0, idx).to(dtype=self.dtype, device="cpu").contiguous()
        return out


class ReplicatedPrefix:
    """Hand-written replay of ``Qwen3Model.forward`` truncated after ``blocks`` blocks."""

    def __init__(self, prefix: PrefixModel) -> None:
        self.prefix = prefix
        self.model = prefix.model
        self.config = prefix.config

    def forward_ids(self, input_ids: Sequence[int]) -> dict[int, torch.Tensor]:
        from transformers.masking_utils import create_causal_mask

        model = self.model
        device = self.prefix.device
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
        captured: dict[int, torch.Tensor] = {}
        with torch.inference_mode():
            inputs_embeds = model.embed_tokens(ids)
            position_ids = torch.arange(inputs_embeds.shape[1], device=device).unsqueeze(0)
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": None,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            hidden_states = inputs_embeds
            position_embeddings = model.rotary_emb(hidden_states, position_ids)
            for i in range(self.prefix.blocks):
                layer = model.layers[i]
                if i in self.prefix.captures:
                    captured[i] = hidden_states
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    past_key_values=None,
                    use_cache=False,
                )
            if self.prefix.blocks in self.prefix.captures:
                captured[self.prefix.blocks] = hidden_states
        return captured


# ---------------------------------------------------------------------- reference
def full_cpu_reference(
    input_ids: Sequence[int],
    capture_blocks: Sequence[int] = (4, 8),
    model_path: Path = MODEL_PATH,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    """Ground-truth reference: the *complete* original model on CPU, captured by hooks.

    ``hidden_states[b]`` is captured as the pre-hook input of ``layers[b]``; for
    ``b == 36`` the final ``model.norm`` output is returned so an end-to-end comparison
    with the accepted feature matrix is possible.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding
    from transformers import Qwen3Config

    config = Qwen3Config.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = "sdpa"
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = Qwen3Model(config)
    finally:
        torch.set_default_dtype(previous)
    model.rotary_emb = Qwen3RotaryEmbedding(config)
    wanted = set(weight_map(model_path)) - {"lm_head.weight"}  # logits are never used
    tensors = _read_shard_tensors(shard_files(model_path), wanted)
    embed = {"weight": tensors.pop("model.embed_tokens.weight").to(dtype=dtype)}
    model.embed_tokens.to_empty(device="cpu")
    model.embed_tokens.load_state_dict(embed, strict=True)
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."
        sub = {k[len(prefix):]: v.to(dtype=dtype) for k, v in tensors.items() if k.startswith(prefix)}
        for key in [k for k in tensors if k.startswith(prefix)]:
            del tensors[key]
        model.layers[i].to_empty(device="cpu")
        model.layers[i].load_state_dict(sub, strict=True)
    norm_key = "model.norm.weight"
    model.norm.to_empty(device="cpu")
    model.norm.load_state_dict({"weight": tensors.pop(norm_key).to(dtype=dtype)}, strict=True)
    assert not tensors, sorted(tensors)[:5]
    model.eval()
    model.requires_grad_(False)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for b in capture_blocks:
        handles.append(
            model.layers[b].register_forward_pre_hook(
                (lambda m, args, _b=b: captured.__setitem__(_b, args[0].detach())), with_kwargs=False
            )
        )
    ids = torch.tensor([list(input_ids)], dtype=torch.long)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False, return_dict=True)
    for h in handles:
        h.remove()
    captured["final_norm"] = out.last_hidden_state.detach()
    return captured
