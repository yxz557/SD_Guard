from __future__ import annotations

import torch
from torch import nn


class HiddenStateClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        head: str = "mlp",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.head_name = head
        if head == "linear":
            self.network = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 2),
            )
        elif head == "mlp":
            self.network = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            raise ValueError(f"Unsupported head: {head}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.network(hidden)

