"""Runtime-r4: mechanically equivalent runtime variants of the frozen corrected-r3 detector.

Nothing here trains, fits, relabels or re-thresholds anything.  The only additions are

* ``TriStudentGraph`` -- the *same* three frozen S2 students evaluated as one static-shape
  CUDA graph.  The graph removes per-event Python/launch/sync overhead; the maths (embedding
  bag mean, LayerNorm, GELU, two linear layers, head, float32) is unchanged.  Bags are padded
  with a dedicated all-zero embedding row and summed, then divided by the true count, which is
  algebraically the same mean as ``EmbeddingBag(mode="mean")`` (adding exact zeros does not
  change a float sum; empty bags stay zero).
* ``TriStudentCPU`` -- the same merged student on CPU (for a single-process CPU pilot).
* ``EpisodeStreamR4`` -- thin instrumentation around the frozen r3 stream so the benchmark can
  report where the time went without moving any work outside the timer.
* ``FrozenScorerR4`` -- the frozen threshold/fusion arithmetic, reused verbatim from r3.

Reference compatibility is proven by ``scripts/parity_r4.py`` over all 25 003 DEV events:
views (roles / call binding / retrieval / value sources), dense and bag features, per-seed CT
logits, per-seed episode max, float32 seed mean, float64 fusion and the frozen ``>=`` tau.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import features, runtime_fast as rf, view

ART = rf.ART
_EMPTY_IDX = np.zeros(0, dtype=np.int32)
# maximum bag length in common DEV is 693 (cur); 768 keeps one fixed graph shape for all rows.
GRAPH_N_SRC = view.MAX_RELEVANT_BLOCKS


class TriStudentGraph:
    """Static-shape CUDA graphs for the three merged frozen S2 students.

    Every event packs its real bag indices compactly in a fixed-capacity buffer (smallest of
    ``CAPS`` that fits) and supplies *dynamic* bag offsets.  Padding uses one all-zero embedding
    row per seed, so a sum followed by division by the true count is exactly the reference
    ``EmbeddingBag(mode="mean")`` (empty bags stay zero).  The trunk is the same LayerNorm/GELU/
    linear stack in float32.  This removes per-event Python/launch overhead without changing any
    model arithmetic.
    """

    CAPS = (768, 1536, 3072, 4096)

    def __init__(self, ckpts: list[Path], *, artdir: Path = ART, scaler: Path | None = None,
                 caps: tuple[int, ...] | None = None) -> None:
        import torch

        self.torch = torch
        self.base = rf.TriStudent(ckpts, device="cuda", scaler=scaler)
        t = torch
        n, B, D = self.base.emb.shape
        self.n_seeds = n
        self.buckets = B
        self.width = self.base.width
        self.hidden = self.base.hidden
        self.dense_mean = self.base.dense_mean
        self.dense_std = self.base.dense_std
        self.n_bags = n * (3 + GRAPH_N_SRC)
        self.caps = tuple(sorted(caps or self.CAPS))
        # one extra all-zero row per seed used only for graph padding
        self.emb_ext = t.cat([self.base.emb,
                              t.zeros(n, 1, D, device="cuda", dtype=self.base.emb.dtype)],
                             dim=1).contiguous()
        self.emb_flat = self.emb_ext.view(n * (B + 1), -1)
        self.gamma = self.base.norm_w.contiguous()
        self.beta = self.base.norm_b.contiguous()
        self.w1 = self.base.w1.contiguous()
        self.b1 = self.base.b1.contiguous()
        self.w2 = self.base.w2.contiguous()
        self.b2 = self.base.b2.contiguous()
        self.wh = self.base.wh.contiguous()
        self.bh = self.base.bh.contiguous()
        self._graphs = {}
        for cap in self.caps:
            self._graphs[cap] = self._capture(cap)
        self.n_graph = 0
        self.n_fallback = 0
        self.fallback_reasons: dict[int, int] = {}
        self.cap_counts = {cap: 0 for cap in self.caps}

    # -- GPU computation ----------------------------------------------------------------
    def _run(self, idx, off, counts, dx, cap):
        t = self.torch
        rows = t.nn.functional.embedding_bag(idx, self.emb_flat, off, mode="sum")
        rows = rows / counts.float().clamp(min=1).unsqueeze(1)
        rows = rows.view(self.n_seeds, 3 + GRAPH_N_SRC, -1)
        x_cur = rows[:, 0, :]
        x_vis = rows[:, 1, :]
        x_user = rows[:, 2, :]
        src_rows = rows[:, 3:, :]
        # ``dx`` = [dense (float32, DENSE_DIM) | n_src_present (float32)] in one pinned buffer;
        # the count is small and exact, so clamp(min=1)/divide is the same arithmetic as before.
        present = dx[features.DENSE_DIM:features.DENSE_DIM + 1].clamp(min=1).unsqueeze(1)
        src = src_rows.sum(dim=1) / present
        dense = dx[:features.DENSE_DIM].unsqueeze(0).expand(self.n_seeds, -1)
        x = t.cat([x_cur, x_vis, x_user, src, dense], dim=-1)
        xn = t.nn.functional.layer_norm(x, (self.width,), None, None, 1e-5)
        x = xn * self.gamma + self.beta
        h = t.baddbmm(self.b1.unsqueeze(1), x.unsqueeze(1),
                      self.w1.transpose(1, 2)).squeeze(1)
        h = t.nn.functional.gelu(h)
        h = t.baddbmm(self.b2.unsqueeze(1), h.unsqueeze(1),
                      self.w2.transpose(1, 2)).squeeze(1)
        h = t.nn.functional.gelu(h)
        return (h * self.wh.squeeze(1)).sum(dim=-1) + self.bh.squeeze(1)

    def _capture(self, cap: int):
        """One static-shape graph + one pre-allocated *pinned* host->device staging block.

        All graph inputs (packed indices int32, dense+source-count float32, bag offsets/counts
        int64) live in a single pinned host buffer, so one event needs exactly **one** blocking
        H2D copy of one contiguous block instead of five small pageable copies.  Blocking (not
        non_blocking) is deliberate: the same host block is rewritten by the next event, so the
        transfer must have completed first (no async-reuse race, no untimed overlap).
        """
        t = self.torch
        n_oc = 2 * self.n_bags
        n_dense = features.DENSE_DIM + 1
        idx_bytes = cap * 4
        dense_bytes = n_dense * 4
        oc_off = idx_bytes + dense_bytes + (-(idx_bytes + dense_bytes)) % 8
        total_bytes = oc_off + n_oc * 8
        p_buf = t.empty(total_bytes, dtype=t.uint8, pin_memory=True)
        d_buf = t.empty(total_bytes, dtype=t.uint8, device="cuda")
        d_idx = d_buf[:idx_bytes].view(t.int32)
        d_dx = d_buf[idx_bytes:idx_bytes + dense_bytes].view(t.float32)
        d_oc = d_buf[oc_off:].view(t.int64)
        d_off = d_oc[:self.n_bags]
        d_counts = d_oc[self.n_bags:]
        d_buf.zero_()
        hb = p_buf.numpy()
        h_idx = hb[:idx_bytes].view(np.int32)
        h_dx = hb[idx_bytes:idx_bytes + dense_bytes].view(np.float32)
        h_oc = hb[oc_off:].view(np.int64)
        # warm the allocator / kernels on the capture stream
        s = t.cuda.Stream()
        s.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(s):
            for _ in range(3):
                self._run(d_idx, d_off, d_counts, d_dx, cap)
        t.cuda.current_stream().wait_stream(s)
        g = t.cuda.CUDAGraph()
        with t.cuda.graph(g):
            out = self._run(d_idx, d_off, d_counts, d_dx, cap)
        return {"graph": g, "out": out, "d_idx": d_idx, "d_oc": d_oc, "d_dx": d_dx,
                "d_buf": d_buf, "p_buf": p_buf,
                "h_idx": h_idx, "h_oc": h_oc, "h_dx": h_dx}

    # -- host staging --------------------------------------------------------------------
    def _stage(self, feats: dict[str, Any]) -> tuple[int, int]:
        """Pack this event's real bag indices into the fixed-capacity pinned buffer.

        Vectorised replacement of the per-seed/per-bag Python loop: the indices are concatenated
        once and shifted by the seed row base, the bag offsets/counts are tiled from the same
        length vector.  Returned values, dtype and padding are unchanged.
        """
        bags = feats["bags"]
        src_bags = bags["src"]
        if len(src_bags) > GRAPH_N_SRC:
            return -1, 0
        empty = _EMPTY_IDX
        arrs = [bags["cur"], bags["vis"], bags["user"]] + list(src_bags) + \
               [empty] * (GRAPH_N_SRC - len(src_bags))
        a32 = [np.asarray(a, dtype=np.int32) for a in arrs]
        lens = np.fromiter((a.size for a in a32), dtype=np.int64, count=len(a32))
        per_seed = int(lens.sum())
        total = self.n_seeds * per_seed
        cap = next((c for c in self.caps if total <= c), None)
        if cap is None:
            return -2, 0
        rec = self._graphs[cap]
        idx = rec["h_idx"]
        oc = rec["h_oc"]
        if per_seed:
            cat = np.concatenate(a32) if len(a32) > 1 else a32[0]
            B1 = self.buckets + 1
            for g in range(self.n_seeds):
                np.add(cat, g * B1, out=idx[g * per_seed:(g + 1) * per_seed])
        cursor = total
        if cursor < cap:
            idx[cursor:cap] = self.buckets      # zero row of the first seed
        # per-seed per-bag offsets (18 scalar writes are cheaper than arange/ravel temporaries)
        slot = 0
        for g in range(self.n_seeds):
            o = g * per_seed
            for L in lens.tolist():
                oc[slot] = o
                o += L
                slot += 1
        oc[self.n_bags:] = np.tile(lens, self.n_seeds)
        dx = rec["h_dx"]
        np.subtract(np.asarray(feats["dense"], dtype=np.float32), self.dense_mean, out=dx[:features.DENSE_DIM])
        np.divide(dx[:features.DENSE_DIM], self.dense_std, out=dx[:features.DENSE_DIM])
        dx[features.DENSE_DIM] = len(src_bags)
        return cap, cursor

    def _copy(self, cap: int, cursor: int) -> None:
        rec = self._graphs[cap]
        rec["d_buf"].copy_(rec["p_buf"], non_blocking=False)

    def forward(self, feats: dict[str, Any]) -> np.ndarray:
        t = self.torch
        with t.inference_mode():
            cap, cursor = self._stage(feats)
            if cap < 0:
                self.n_fallback += 1
                return self.base.forward(feats)
            self._copy(cap, cursor)
            rec = self._graphs[cap]
            rec["graph"].replay()
            self.n_graph += 1
            self.cap_counts[cap] += 1
            return rec["out"].detach().float().cpu().numpy().copy()

    def forward_gpu(self, feats: dict[str, Any]):
        """Stage + replay without the per-event D2H sync; caller accumulates on GPU."""
        t = self.torch
        cap, cursor = self._stage(feats)
        if cap < 0:
            self.n_fallback += 1
            self.fallback_reasons[cap] = self.fallback_reasons.get(cap, 0) + 1
            return self.torch.from_numpy(self.base.forward(feats)).to("cuda")
        self._copy(cap, cursor)
        rec = self._graphs[cap]
        rec["graph"].replay()
        self.n_graph += 1
        self.cap_counts[cap] += 1
        return rec["out"]


class EpisodeStreamR4(rf.EpisodeStream):
    """Frozen r3 stream plus per-round instrumentation (no work is moved out of the event)."""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.last_round_wall = 0.0
        self.last_parse_wall = 0.0
        self.last_process_wall = 0.0

    def begin_round(self, rid, prompt_tokens, first_end, prompt_text_override=None,
                    body_text_override=None, blocks_override=None) -> None:
        t0 = time.perf_counter()
        super().begin_round(rid, prompt_tokens, first_end,
                            prompt_text_override=prompt_text_override,
                            body_text_override=body_text_override,
                            blocks_override=blocks_override)
        self.last_round_wall = time.perf_counter() - t0
        self.last_parse_wall = float(getattr(self, "_t_parse", 0.0))
        self.last_process_wall = float(getattr(self, "_t_process", 0.0))


class FrozenScorerR4:
    """Frozen S2 / FUSION_S2 arithmetic with a selectable equivalent backend."""

    def __init__(self, arm: str = "S2", *, backend: str = "graph", seeds=(0, 1, 2),
                 artdir: Path = ART, tag: str = "", fusion_path: Path | None = None,
                 scaler_path: Path | None = None) -> None:
        self.arm = arm
        self.backend = backend
        self.seeds = tuple(seeds)
        suffix = f"_{tag}" if tag else ""
        scaler = Path(scaler_path) if scaler_path else (
            Path(artdir) / (f"scaler_{tag}.npz" if tag else "scaler_v6_v3.npz"))
        ckpts = [Path(artdir) / f"ckpt_{arm}{suffix}_seed{s}.pt" for s in self.seeds]
        if backend == "graph":
            self.net = TriStudentGraph(ckpts, scaler=scaler)
        elif backend == "cpu":
            self.net = rf.TriStudent(ckpts, device="cpu", scaler=scaler)
        elif backend == "cuda":
            self.net = rf.TriStudent(ckpts, device="cuda", scaler=scaler)
        else:
            raise ValueError(backend)
        self.routing = rf.load_fusion(fusion_path or rf.FUSION_PATH)

    def row_logits(self, feats: dict[str, Any]) -> np.ndarray:
        return self.net.forward(feats)

    def episode_scores(self, per_seed_max: np.ndarray) -> np.ndarray:
        return np.asarray(per_seed_max, dtype=np.float32).mean(axis=0, dtype=np.float32)

    def fused(self, base_ep: np.ndarray, student_ep: np.ndarray) -> np.ndarray:
        r = self.routing
        return 0.5 * ((base_ep.astype(np.float64) - r.base_mean) / r.base_std) + \
            0.5 * ((student_ep.astype(np.float64) - r.stud_mean) / r.stud_std)


def typed_source_blocks(prompt_text: str):
    """Source adapter used by the typed benchmark mode.

    A real controller builds its prompt from a role-tagged message list and still has that list
    at the formatter boundary.  The prepared episode store only keeps token ids, so this adapter
    recovers the same boundaries from the decoded prompt.  Its cost is *measured and charged*
    in ``mode T integrated``; it is never hidden in fixture construction.
    """
    return view.parse_blocks(prompt_text)
