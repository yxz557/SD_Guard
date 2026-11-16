from __future__ import annotations

from typing import Any

import torch
from torch import nn

from guard_training.frozen_hidden_mlp.model import HiddenStateClassifier
from guard_training.tool_step_attention.dataset import (
    CALL_VARIANTS,
    INDIVIDUAL_RCT_VARIANTS,
    RESULT_VARIANTS,
    ROUND_VARIANTS,
    ROUND_TOOL_RCT_VARIANTS,
    SHARED_RESULT_VARIANTS,
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

    def score_logits(
        self,
        readout: torch.Tensor,
        steps: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_projection(self.query_norm(readout)).unsqueeze(1)
        keys = self.step_projection(self.step_norm(steps))
        return self.score(self.dropout(self.activation(query + keys))).squeeze(-1)

    @staticmethod
    def pool_from_logits(
        steps: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every sample must expose at least one attention step")
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights.unsqueeze(-1) * steps, dim=1)
        return pooled, weights

    def forward(
        self,
        readout: torch.Tensor,
        steps: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pool_from_logits(
            steps, self.score_logits(readout, steps), mask
        )


def hierarchical_logits(
    tool_logits: torch.Tensor,
    round_logits: torch.Tensor,
    round_groups: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Combine round and within-round tool scores without tool-count bias.

    For a tool j in round k the returned logit is
    ``round_score[k] + log_softmax(tool_scores_in_round_k)[j]``. A final
    softmax over all tools therefore equals round-softmax times the within-round
    tool-softmax. Scores from repeated copies of the same round context are
    averaged so every tool in that round receives one identical round prior.
    """

    steps = int(tool_logits.shape[1])
    same_group = round_groups.unsqueeze(2) == round_groups.unsqueeze(1)
    pair_mask = same_group & mask.unsqueeze(2) & mask.unsqueeze(1)
    source_tool_logits = tool_logits.unsqueeze(1).expand(-1, steps, -1)
    negative_infinity = torch.finfo(tool_logits.dtype).min
    within_normalizer = torch.logsumexp(
        source_tool_logits.masked_fill(~pair_mask, negative_infinity), dim=2
    )
    source_round_logits = round_logits.unsqueeze(1).expand(-1, steps, -1)
    group_count = pair_mask.sum(dim=2).clamp_min(1)
    group_round_score = (
        source_round_logits.masked_fill(~pair_mask, 0.0).sum(dim=2) / group_count
    )
    combined = group_round_score + tool_logits - within_normalizer
    return torch.where(mask, combined, torch.zeros_like(combined))


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
        if variant in {
            "rct_independent",
            "rct_tool_independent",
            "rct_round_tool_independent",
        }:
            self.result_attention = ReadoutConditionedAttention(
                hidden_size, attention_dim, attention_dropout
            )
        self.call_round_attention: ReadoutConditionedAttention | None = None
        self.result_round_attention: ReadoutConditionedAttention | None = None
        if variant in ROUND_TOOL_RCT_VARIANTS:
            self.call_round_attention = ReadoutConditionedAttention(
                hidden_size, attention_dim, attention_dropout
            )
        if variant == "rct_round_tool_independent":
            self.result_round_attention = ReadoutConditionedAttention(
                hidden_size, attention_dim, attention_dropout
            )
        output_components = 1 if variant in {"c_qattn", "t_qattn"} else 2
        if variant in ROUND_VARIANTS or variant in INDIVIDUAL_RCT_VARIANTS:
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
        elif self.variant in ROUND_VARIANTS:
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
        elif self.variant in INDIVIDUAL_RCT_VARIANTS:
            call_mask = batch["call_mask"]
            call_logits = self.call_or_primary_attention.score_logits(
                readout, batch["calls"]
            )
            if self.variant in ROUND_TOOL_RCT_VARIANTS:
                assert self.call_round_attention is not None
                call_mask = call_mask & batch["call_round_context_mask"]
                call_round_logits = self.call_round_attention.score_logits(
                    readout, batch["call_round_context"]
                )
                call_logits = hierarchical_logits(
                    call_logits,
                    call_round_logits,
                    batch["call_round_group"],
                    call_mask,
                )
            call_pool, call_weights = ReadoutConditionedAttention.pool_from_logits(
                batch["calls"], call_logits, call_mask
            )

            result_mask = batch["result_mask"]
            if self.variant in SHARED_RESULT_VARIANTS:
                result_to_call = batch["result_to_call"]
                mapping_mask = result_to_call >= 0
                safe_indices = result_to_call.clamp(
                    min=0, max=int(batch["calls"].shape[1]) - 1
                )
                result_logits = torch.gather(call_logits, 1, safe_indices)
                mapped_call_mask = torch.gather(call_mask, 1, safe_indices)
                result_mask = result_mask & mapping_mask & mapped_call_mask
            else:
                assert self.result_attention is not None
                result_logits = self.result_attention.score_logits(
                    readout, batch["results"]
                )
                if self.variant in ROUND_TOOL_RCT_VARIANTS:
                    assert self.result_round_attention is not None
                    result_mask = (
                        result_mask & batch["result_round_context_mask"]
                    )
                    result_round_logits = (
                        self.result_round_attention.score_logits(
                            readout, batch["result_round_context"]
                        )
                    )
                    result_logits = hierarchical_logits(
                        result_logits,
                        result_round_logits,
                        batch["result_round_group"],
                        result_mask,
                    )
            result_pool, result_weights = (
                ReadoutConditionedAttention.pool_from_logits(
                    batch["results"], result_logits, result_mask
                )
            )
            representation = torch.cat([readout, call_pool, result_pool], dim=-1)
            attention["call_weights"] = call_weights
            attention["result_weights"] = result_weights
        else:
            raise ValueError(f"Unknown attention variant: {self.variant}")
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
