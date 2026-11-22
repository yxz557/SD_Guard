"""Small, explicitly causal sequence models for fine_sequence_explore_v1 (Lane T).

Every model consumes cached per-position LLM vectors (already causal context) and emits
one raw logit per position.  The family is a hypothesis: a pointwise MLP control is
provided alongside projected residual GRU, left-causal TCN and a small causal
Transformer.  No model receives episode/group/task/source identity, labels, future
length or an end tag as input.

The public forward contract is::

    logits = model(features, padding_mask, ct_mask=..., token_gap=..., round_gap=...)

where ``features`` is ``[B,T,4096]``, ``padding_mask`` is TRUE for a real token (the
opposite of PyTorch key_padding_mask), and ``logits`` is ``[B,T]`` raw finite values.
Padded positions are set to zero; objectives must always mask them out.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = ("mlp", "gru", "tcn", "transformer")
META_FEATURES = ("ct", "token_gap", "round_gap", "token_index")


@dataclass(frozen=True)
class ModelConfig:
    family: str = "mlp"
    input_dim: int = 4096
    hidden: int = 256
    proj_dim: int = 128
    num_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.1
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    transformer_ff: int | None = None
    use_meta_features: bool = False
    meta_features: tuple[str, ...] = ("ct", "token_gap", "round_gap")
    aux_head: bool = False

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown model family {self.family!r}")
        if self.input_dim <= 0 or self.hidden <= 0 or self.proj_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.family == "transformer":
            if self.proj_dim % self.num_heads:
                raise ValueError("transformer proj_dim must be divisible by num_heads")
            if self.kernel_size < 1:
                raise ValueError("bad kernel size")
        if self.family == "tcn":
            if self.kernel_size < 1 or any(int(d) < 1 for d in self.dilations):
                raise ValueError("TCN kernel/dilations must be positive")
        for name in self.meta_features:
            if name not in META_FEATURES:
                raise ValueError(f"unknown meta feature {name!r}")
        if self.use_meta_features and not self.meta_features:
            raise ValueError("use_meta_features=True needs at least one meta feature")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dilations"] = list(self.dilations)
        data["meta_features"] = list(self.meta_features)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        data = dict(data)
        data["dilations"] = tuple(int(x) for x in data.get("dilations", (1, 2, 4, 8)))
        data["meta_features"] = tuple(str(x) for x in data.get("meta_features", ("ct", "token_gap", "round_gap")))
        return cls(**data)

    @property
    def receptive_field(self) -> int:
        if self.family != "tcn":
            return -1
        return 1 + sum((int(self.kernel_size) - 1) * int(d) for d in self.dilations)


def _sinusoidal_positions(length: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    if dim > 1:
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


class _BaseCausalModel(nn.Module):
    """Shared projection/head plumbing; subclasses provide the causal encoder."""

    def __init__(self, config: ModelConfig, enc_dim: int, *, meta_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.enc_dim = int(enc_dim)
        self.head = nn.Linear(self.enc_dim, 1)
        self.aux_head = nn.Linear(self.enc_dim, 1) if config.aux_head else None
        self.meta_dim = int(meta_dim if meta_dim is not None else self.enc_dim)
        self.meta_encoder = (
            nn.Linear(len(config.meta_features), self.meta_dim) if config.use_meta_features else None
        )

    # -- label-blind causal metadata -------------------------------------
    def meta_tensor(
        self,
        *,
        ct_mask: torch.Tensor | None,
        token_gap: torch.Tensor | None,
        round_gap: torch.Tensor | None,
        token_index: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.meta_encoder is None:
            return None
        parts: list[torch.Tensor] = []
        for name in self.config.meta_features:
            if name == "ct":
                if ct_mask is None:
                    raise ValueError("meta feature 'ct' needs ct_mask")
                parts.append(ct_mask.to(torch.float32))
            elif name == "token_gap":
                if token_gap is None:
                    raise ValueError("meta feature 'token_gap' needs token_gap")
                parts.append(torch.log1p(torch.clamp(token_gap.to(torch.float32), min=0.0)))
            elif name == "round_gap":
                if round_gap is None:
                    raise ValueError("meta feature 'round_gap' needs round_gap")
                parts.append(torch.log1p(torch.clamp(round_gap.to(torch.float32), min=0.0)))
            elif name == "token_index":
                if token_index is None:
                    raise ValueError("meta feature 'token_index' needs token_index")
                parts.append(torch.log1p(torch.clamp(token_index.to(torch.float32), min=0.0)))
            else:  # pragma: no cover - validated in config
                raise ValueError(name)
        return torch.stack(parts, dim=-1)

    def add_meta(self, hidden: torch.Tensor, meta: torch.Tensor | None) -> torch.Tensor:
        if self.meta_encoder is None or meta is None:
            return hidden
        return hidden + self.meta_encoder(meta.to(hidden.dtype))

    def _encode(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
        meta: torch.Tensor | None,
    ) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        ct_mask: torch.Tensor | None = None,
        token_gap: torch.Tensor | None = None,
        round_gap: torch.Tensor | None = None,
        token_index: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError(f"features must be [B,T,{self.config.input_dim}]")
        valid = padding_mask.to(torch.bool)
        if valid.shape != features.shape[:2]:
            raise ValueError("padding_mask must be [B,T]")
        x = features.to(torch.float32)
        # Structural padding never becomes a feature.  Masked positions cannot leak
        # into causal outputs because every encoder is strictly left-to-right.
        x = x * valid.unsqueeze(-1).to(x.dtype)
        meta = self.meta_tensor(ct_mask=ct_mask, token_gap=token_gap, round_gap=round_gap, token_index=token_index)
        hidden = self._encode(x, valid, meta)
        logits = self.head(hidden).squeeze(-1)
        logits = torch.where(valid, logits, torch.zeros((), dtype=logits.dtype, device=logits.device))
        if not bool(torch.isfinite(logits[valid]).all()):
            raise FloatingPointError("model produced non-finite logits on a real position")
        aux = None
        if self.aux_head is not None:
            aux = self.aux_head(hidden).squeeze(-1)
            aux = torch.where(valid, aux, torch.zeros((), dtype=aux.dtype, device=aux.device))
        if return_aux:
            return logits, aux
        return logits


class PointwiseMLP(_BaseCausalModel):
    """Independent 4096->hidden->1 mapping per real position (no memory)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config, config.hidden)
        self.fc1 = nn.Linear(config.input_dim, config.hidden)
        self.dropout = nn.Dropout(config.dropout)

    def _encode(self, features: torch.Tensor, padding_mask: torch.Tensor, meta: torch.Tensor | None) -> torch.Tensor:
        hidden = self.fc1(features)
        hidden = self.add_meta(hidden, meta)
        return self.dropout(F.gelu(hidden))


class ResidualGRU(_BaseCausalModel):
    """Projected tokenwise-LN residual GRU; state starts at zero for every episode."""

    def __init__(self, config: ModelConfig) -> None:
        # Meta features are added while hidden is still proj_dim wide, before the GRU
        # maps it to config.hidden.  The projection dimension must follow that order.
        super().__init__(config, config.hidden, meta_dim=config.proj_dim)
        self.proj = nn.Linear(config.input_dim, config.proj_dim)
        self.ln = nn.LayerNorm(config.proj_dim)
        gru_dropout = float(config.dropout) if config.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            config.proj_dim,
            config.hidden,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.skip = nn.Identity() if config.proj_dim == config.hidden else nn.Linear(config.proj_dim, config.hidden)
        self.dropout = nn.Dropout(config.dropout)

    def _encode(self, features: torch.Tensor, padding_mask: torch.Tensor, meta: torch.Tensor | None) -> torch.Tensor:
        hidden = self.ln(self.proj(features))
        hidden = self.add_meta(hidden, meta)
        output, _ = self.gru(hidden)
        return self.dropout(output + self.skip(hidden))


class _CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_pad = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=0, dilation=int(dilation))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B,C,T]; pad only the left so no future token is ever read.
        padded = F.pad(hidden, (self.left_pad, 0))
        return hidden + self.dropout(F.gelu(self.conv(padded)))


class CausalTCN(_BaseCausalModel):
    """Left-causal residual dilated TCN; declared receptive field is ``config.receptive_field``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config, config.proj_dim)
        self.proj = nn.Linear(config.input_dim, config.proj_dim)
        self.ln = nn.LayerNorm(config.proj_dim)
        self.blocks = nn.ModuleList(
            [
                _CausalConvBlock(config.proj_dim, config.kernel_size, dilation, config.dropout)
                for dilation in config.dilations
            ]
        )
        self.dropout = nn.Dropout(config.dropout)

    def _encode(self, features: torch.Tensor, padding_mask: torch.Tensor, meta: torch.Tensor | None) -> torch.Tensor:
        hidden = self.ln(self.proj(features)).transpose(1, 2)  # [B,C,T]
        if self.meta_encoder is not None and meta is not None:
            hidden = hidden + self.meta_encoder(meta.to(hidden.dtype)).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        return self.dropout(hidden).transpose(1, 2)


class CausalTransformer(_BaseCausalModel):
    """Small 2-layer causal Transformer with triangular + padding masks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config, config.proj_dim)
        self.proj = nn.Linear(config.input_dim, config.proj_dim)
        self.ln = nn.LayerNorm(config.proj_dim)
        ff = int(config.transformer_ff or 2 * config.proj_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.proj_dim,
            nhead=config.num_heads,
            dim_feedforward=ff,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config.num_layers), enable_nested_tensor=False)

    @staticmethod
    def causal_attn_mask(length: int, *, device: torch.device) -> torch.Tensor:
        # True means "not allowed to attend"; only future positions are forbidden.
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _causal_attn_mask(self, length: int, *, device: torch.device) -> torch.Tensor:
        return self.causal_attn_mask(length, device=device)

    def _encode(self, features: torch.Tensor, padding_mask: torch.Tensor, meta: torch.Tensor | None) -> torch.Tensor:
        hidden = self.ln(self.proj(features))
        hidden = self.add_meta(hidden, meta)
        length = hidden.shape[1]
        if length:
            hidden = hidden + _sinusoidal_positions(length, hidden.shape[-1], device=hidden.device, dtype=hidden.dtype)
        attn = self._causal_attn_mask(length, device=hidden.device)
        # padding_mask is TRUE=real; PyTorch key_padding_mask is TRUE=ignore.
        key_padding = ~padding_mask.to(torch.bool)
        inactive = ~padding_mask.to(torch.bool).any(dim=1)
        if bool(inactive.any()) and length:
            # A fully padded episode would otherwise make every key masked.  Native MHA
            # can produce NaN hidden states/gradients in that case.  Let inactive rows
            # attend to one structural zero key; the row is zeroed again below and can
            # never become a real endpoint.
            key_padding = key_padding.clone()
            key_padding[inactive, 0] = False
        encoded = self.encoder(hidden, mask=attn, src_key_padding_mask=key_padding)
        return encoded * padding_mask.to(torch.bool).unsqueeze(-1).to(encoded.dtype)


def build_model(config: ModelConfig) -> _BaseCausalModel:
    if config.family == "mlp":
        return PointwiseMLP(config)
    if config.family == "gru":
        return ResidualGRU(config)
    if config.family == "tcn":
        return CausalTCN(config)
    if config.family == "transformer":
        return CausalTransformer(config)
    raise ValueError(config.family)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = [
    "FAMILIES",
    "META_FEATURES",
    "ModelConfig",
    "PointwiseMLP",
    "ResidualGRU",
    "CausalTCN",
    "CausalTransformer",
    "build_model",
    "count_parameters",
]
