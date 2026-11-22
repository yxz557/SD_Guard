"""Few-million-parameter hashed-text student with a main security head and parameter-level aux heads.

This is the allowed "B" discriminator: a fixed, small classification network.  It is not a
second LLM, it never generates a rationale, and its cost is measured inside the real
incremental path (see ``cost.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from . import features

AUX_ACTION = 3   # authorized / unauthorized / unclear
AUX_REL = 5      # direct / prerequisite / extra / conflicting / unclear
AUX_ALIGN = 4    # full / partial / none / unclear


@dataclass
class ModelConfig:
    buckets: int = features.BUCKETS
    emb_dim: int = 16
    hidden: int = 256
    dropout: float = 0.2
    dense_dim: int = features.DENSE_DIM


class Student(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.emb = nn.EmbeddingBag(self.cfg.buckets, self.cfg.emb_dim, mode="mean", sparse=False)
        width = self.cfg.emb_dim * 4 + self.cfg.dense_dim
        self.norm = nn.LayerNorm(width)
        self.trunk = nn.Sequential(
            nn.Linear(width, self.cfg.hidden), nn.GELU(), nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden, self.cfg.hidden), nn.GELU(),
        )
        self.head_main = nn.Linear(self.cfg.hidden, 1)
        self.head_action = nn.Linear(self.cfg.hidden, AUX_ACTION)
        self.head_rel = nn.Linear(self.cfg.hidden, AUX_REL)
        self.head_align = nn.Linear(self.cfg.hidden, AUX_ALIGN)
        self.head_arg = nn.Linear(self.cfg.hidden, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cur = self.emb(batch["cur_idx"], batch["cur_off"])
        vis = self.emb(batch["vis_idx"], batch["vis_off"])
        user = self.emb(batch["user_idx"], batch["user_off"])
        src = self.emb(batch["src_idx"], batch["src_off"])
        x = torch.cat([cur, vis, user, src, batch["dense"]], dim=-1)
        h = self.trunk(self.norm(x))
        return {
            "main": self.head_main(h).squeeze(-1),
            "action": self.head_action(h),
            "rel": self.head_rel(h),
            "align": self.head_align(h),
            "arg": self.head_arg(h),
            "hidden": h,
        }

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def collate(items: list[dict[str, Any]], device: str = "cpu") -> dict[str, torch.Tensor]:
    """items: [{"bags": {...}, "dense": [...]}] -> padded index/offset tensors."""
    out: dict[str, torch.Tensor] = {}
    dense = torch.tensor([it["dense"] for it in items], dtype=torch.float32, device=device)
    out["dense"] = dense
    for channel in features.CHANNELS:
        if channel == "src":
            continue
        flat: list[int] = []
        offsets: list[int] = []
        for it in items:
            idx = it["bags"][channel]
            offsets.append(len(flat))
            flat.extend(idx)
        out[f"{channel}_idx"] = torch.tensor(flat, dtype=torch.long, device=device)
        out[f"{channel}_off"] = torch.tensor(offsets, dtype=torch.long, device=device)
    # the source channel has up to 3 bags per event: average their EmbeddingBag outputs by
    # concatenating them as separate bags and using one row per (event, source) then mean
    flat = []
    offsets = []
    counts: list[int] = []
    for it in items:
        bags = it["bags"]["src"]
        counts.append(len(bags))
        for bag in bags:
            offsets.append(len(flat))
            flat.extend(bag)
    out["src_idx"] = torch.tensor(flat, dtype=torch.long, device=device)
    out["src_off"] = torch.tensor(offsets, dtype=torch.long, device=device)
    out["src_counts"] = torch.tensor(counts, dtype=torch.long, device=device)
    return out


def forward_batched(model: Student, items: list[dict[str, Any]], device: str = "cpu") -> dict[str, torch.Tensor]:
    """Forward with the source-channel mean over per-event source bags."""
    batch = collate(items, device=device)
    cur = model.emb(batch["cur_idx"], batch["cur_off"])
    vis = model.emb(batch["vis_idx"], batch["vis_off"])
    user = model.emb(batch["user_idx"], batch["user_off"])
    src_rows = model.emb(batch["src_idx"], batch["src_off"])
    # mean per event
    src = torch.zeros_like(cur)
    src.index_add_(0, torch.repeat_interleave(
        torch.arange(batch["src_counts"].numel(), device=device), batch["src_counts"]),
        src_rows / torch.repeat_interleave(batch["src_counts"], batch["src_counts"]).float().unsqueeze(-1))
    x = torch.cat([cur, vis, user, src, batch["dense"]], dim=-1)
    h = model.trunk(model.norm(x))
    return {
        "main": model.head_main(h).squeeze(-1),
        "action": model.head_action(h),
        "rel": model.head_rel(h),
        "align": model.head_align(h),
        "arg": model.head_arg(h),
        "hidden": h,
    }
