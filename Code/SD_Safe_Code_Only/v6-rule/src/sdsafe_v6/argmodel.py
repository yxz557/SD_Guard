"""S4: shared per-argument field encoder feeding an event-level authorization logit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import argfeats, model


class ArgTensors:
    """Ragged per-argument tensors aligned to the event cache row order."""

    def __init__(self, cache: dict[str, Any], device: str):
        self.device = device
        self.n = int(cache["arg_count"].shape[0])
        self.counts = np.asarray(cache["arg_count"], dtype=np.int64)
        self.off = np.asarray(cache["arg_off"], dtype=np.int64)
        self.paths = [str(x) for x in cache["arg_paths"].tolist()]
        self.flat = {ch: np.asarray(cache[f"{ch}_idx"], dtype=np.int64)
                     for ch in ("path", "value", "source", "user")}
        self.offs = {ch: np.asarray(cache[f"{ch}_off"], dtype=np.int64)
                     for ch in ("path", "value", "source", "user")}
        self.dense = np.asarray(cache["arg_dense"], dtype=np.float32)
        self.row_index = np.asarray(cache["row_index"], dtype=np.int64)

    def batch(self, rows: np.ndarray) -> dict[str, Any]:
        dev = self.device
        out_flat: dict[str, list[int]] = {ch: [] for ch in self.flat}
        out_off: dict[str, list[int]] = {ch: [] for ch in self.flat}
        dense_parts = []
        paths: list[str] = []
        counts: list[int] = []
        row_ids: list[int] = []
        for local, r in enumerate(rows):
            start, end = int(self.off[r]), int(self.off[r + 1])
            counts.append(end - start)
            for ch in self.flat:
                a, b = int(self.offs[ch][start]), int(self.offs[ch][end])
                base = len(out_flat[ch])
                out_flat[ch].extend(self.flat[ch][a:b].tolist())
                out_off[ch].extend((self.offs[ch][start:end] - self.offs[ch][start] + base).tolist())
            dense_parts.append(self.dense[start:end])
            paths.extend(self.paths[start:end])
            row_ids.extend([local] * (end - start))
        out: dict[str, Any] = {
            "counts": torch.tensor(counts, dtype=torch.long, device=dev),
            "row": torch.tensor(row_ids, dtype=torch.long, device=dev),
            "dense": torch.from_numpy(np.concatenate(dense_parts) if dense_parts else
                                      np.zeros((0, argfeats.ARG_DENSE), np.float32)).to(dev),
            "paths": paths,
        }
        for ch in self.flat:
            out[f"{ch}_idx"] = torch.tensor(out_flat[ch], dtype=torch.long, device=dev)
            out[f"{ch}_off"] = torch.tensor(out_off[ch], dtype=torch.long, device=dev)
        return out


class ArgStudent(nn.Module):
    def __init__(self, cfg: model.ModelConfig | None = None) -> None:
        super().__init__()
        self.base = model.Student(cfg)
        self.emb = nn.EmbeddingBag(argfeats.ARGB, 12, mode="mean", sparse=False)
        self.mlp = nn.Sequential(nn.Linear(4 * 12 + argfeats.ARG_DENSE, 64), nn.GELU(),
                                 nn.Linear(64, 1))
        self.w_max = nn.Parameter(torch.tensor(1.0))
        self.w_mean = nn.Parameter(torch.tensor(0.5))
        self.gain = nn.Parameter(torch.tensor(0.5))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _base_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cur = self.base.emb(batch["cur_idx"], batch["cur_off"])
        vis = self.base.emb(batch["vis_idx"], batch["vis_off"])
        user = self.base.emb(batch["user_idx"], batch["user_off"])
        src_rows = self.base.emb(batch["src_idx"], batch["src_off"])
        rep = torch.repeat_interleave(torch.arange(batch["src_counts"].numel(), device=cur.device),
                                      batch["src_counts"])
        cnt = torch.repeat_interleave(batch["src_counts"], batch["src_counts"]).float().unsqueeze(-1)
        src = torch.zeros_like(cur)
        src.index_add_(0, rep, src_rows / cnt)
        x = torch.cat([cur, vis, user, src, batch["dense"]], dim=-1)
        h = self.base.trunk(self.base.norm(x))
        return {"main": self.base.head_main(h).squeeze(-1), "action": self.base.head_action(h),
                "rel": self.base.head_rel(h), "align": self.base.head_align(h),
                "arg": self.base.head_arg(h), "hidden": h}

    def forward(self, batch: dict[str, torch.Tensor], argb: dict[str, Any]) -> dict[str, Any]:
        base = self._base_forward(batch)
        emb = torch.cat([self.emb(argb[f"{ch}_idx"], argb[f"{ch}_off"])
                         for ch in ("path", "value", "source", "user")], dim=-1)
        arg_logits = self.mlp(torch.cat([emb, argb["dense"]], dim=-1)).squeeze(-1)
        n_rows = int(argb["counts"].numel())
        if arg_logits.numel():
            row = argb["row"]
            maxv = torch.full((n_rows,), -1e9, dtype=arg_logits.dtype, device=arg_logits.device)
            maxv = maxv.index_reduce_(0, row, arg_logits, "amax", include_self=True)
            sumv = torch.zeros(n_rows, dtype=arg_logits.dtype, device=arg_logits.device)
            sumv = sumv.index_add_(0, row, arg_logits)
            cnt = argb["counts"].float().clamp(min=1.0)
            meanv = sumv / cnt
            has = argb["counts"] > 0
            agg = torch.where(has, self.w_max * maxv + self.w_mean * meanv,
                              torch.zeros_like(maxv))
        else:
            agg = torch.zeros(n_rows, dtype=base["main"].dtype, device=base["main"].device)
        main = base["main"] + self.gain * agg
        return {"main": main, "base_main": base["main"], "align": base["align"],
                "action": base["action"], "rel": base["rel"], "arg_event": base["arg"],
                "arg_logits": arg_logits, "arg_row": argb["row"], "agg": agg}
