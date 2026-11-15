from __future__ import annotations

from typing import Any

import torch
from torch import nn

from guard_training.frozen_hidden_mlp.model import HiddenStateClassifier
from guard_training.tool_step_attention.dataset import (
    CALL_VARIANTS,
    RESULT_VARIANTS,
    ROUND_VARIANTS,
    VARIANTS,
)


class ReadoutConditionedAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.step_norm = nn.LayerNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, attention_dim, bias=False)
        self.step_projection = nn.Linear(hidden_size, attention_dim, bias=False)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        readout: torch.Tensor,
        steps: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every sample must expose at least one attention step")
        query = self.query_projection(self.query_norm(readout)).unsqueeze(1)
        keys = self.step_projection(self.step_norm(steps))
        logits = self.score(self.dropout(self.activation(query + keys))).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * steps, dim=1)
        return pooled, weights


class ToolStepAttentionClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        variant: str,
        attention_dim: int = 128,
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown attention variant: {variant}")
        self.hidden_size = hidden_size
        self.variant = variant
        self.call_or_primary_attention = ReadoutConditionedAttention(
            hidden_size, attention_dim, attention_dropout
        )
        self.result_attention: ReadoutConditionedAttention | None = None
        if variant == "rct_independent":
            self.result_attention = ReadoutConditionedAttention(
                hidden_size, attention_dim, attention_dropout
            )
        output_components = 1 if variant in {"c_qattn", "t_qattn"} else 2
        if variant in ROUND_VARIANTS:
            output_components = 3
        self.classifier = HiddenStateClassifier(
            input_dim=hidden_size * output_components,
            head="mlp",
            hidden_dim=head_hidden_dim,
            dropout=dropout,
        )

    @property
    def output_dim(self) -> int:
        return self.classifier.input_dim

    def forward(
        self, batch: dict[str, torch.Tensor], return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        readout = batch["readout"]
        attention: dict[str, torch.Tensor] = {}
        if self.variant in CALL_VARIANTS or self.variant in RESULT_VARIANTS:
            pooled, weights = self.call_or_primary_attention(
                readout, batch["steps"], batch["step_mask"]
            )
            if self.variant in {"c_qattn", "t_qattn"}:
                representation = pooled
            else:
                representation = torch.cat([readout, pooled], dim=-1)
            attention["step_weights"] = weights
        else:
            decision_pool, decision_weights = self.call_or_primary_attention(
                readout, batch["decisions"], batch["decision_mask"]
            )
            if self.variant == "rct_independent":
                assert self.result_attention is not None
                result_pool, result_weights = self.result_attention(
                    readout,
                    batch["round_results"],
                    batch["round_result_mask"],
                )
            else:
                valid = batch["round_result_mask"].to(decision_weights.dtype)
                result_weights = decision_weights * valid
                denominator = result_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
                result_weights = result_weights / denominator
                result_pool = torch.sum(
                    result_weights.unsqueeze(-1) * batch["round_results"], dim=1
                )
            representation = torch.cat([readout, decision_pool, result_pool], dim=-1)
            attention["decision_weights"] = decision_weights
            attention["result_weights"] = result_weights
        logits = self.classifier(representation)
        if return_attention:
            return logits, attention
        return logits


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key in {"labels", "indices"}:
            output[key] = value.to(device)
        elif value.dtype.is_floating_point:
            output[key] = value.to(device=device, dtype=torch.float32)
        else:
            output[key] = value.to(device)
    return output


def model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_config(model: ToolStepAttentionClassifier) -> dict[str, Any]:
    primary = model.call_or_primary_attention
    return {
        "hidden_size": model.hidden_size,
        "variant": model.variant,
        "attention_dim": primary.query_projection.out_features,
        "head_hidden_dim": model.classifier.network[1].out_features,
        "dropout": model.classifier.network[3].p,
        "attention_dropout": primary.dropout.p,
    }
