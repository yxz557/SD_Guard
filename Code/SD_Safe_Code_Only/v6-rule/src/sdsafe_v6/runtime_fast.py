"""Deployment-shaped incremental runtime for the frozen S2/FUSION_S2 detector (runtime-r2).

This module is *only* an equivalent engineering implementation of the already-frozen
cache-only-r1 detector.  It never reads a label for scoring, never fits anything, never
changes a threshold, and never calls a model provider.

The stream is the real causal prompt/emission stream from the prepared DEV episodes:

  * a round starts from the actual prompt token ids that the frozen model consumed,
  * each CT event consumes the actual newly-emitted token ids ``[prev_end, token_end)``,
  * the history/source index is maintained per message and reused inside the episode,
  * the current assistant block is the only growing block; prior messages are finalised once,
  * hashed lexical features, value/source provenance and generic retrieval are computed from
    the maintained strings,
  * the three frozen S2 students are evaluated with one merged batched forward,
  * per-seed episode running max -> float32 mean -> frozen FUSION_S2 mapping -> frozen threshold.

Reference compatibility: all text processing reuses ``view``/``features`` helpers so the
produced bags/dense features are bitwise identical to the r1 evaluation path.  The only
deliberate difference is performance (indexes/caches/merged forward), checked by
``scripts/parity_runtime.py`` over every common-DEV event.

No TEST material, no teacher call, no annotation task.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import features, view

ART = Path("/root/sdsafe-compute/v6-param/artifacts")
IM_START = view.IM_START
IM_END = view.IM_END
_BLOCK_RE = view._BLOCK_RE
_NUM_RE = features._NUM_RE
_WORD_RE = features._WORD_RE
_STOP = features._STOP

FROZEN_BASE_THRESHOLD = float(np.float32(-0.6190447211265564))


def decode_ids(tok, arr) -> str:
    """Backend decode (identical output to tok.decode) without the HF wrapper overhead."""
    fast = getattr(tok, "_tokenizer", None)
    if fast is not None:
        return fast.decode(arr.tolist() if hasattr(arr, "tolist") else list(arr),
                           skip_special_tokens=False)
    return tok.decode(arr, skip_special_tokens=False)
FUSION_PATH = ART / "fusion_frozen_v3.json"


# --------------------------------------------------------------------------------------
# text-level caches (never precomputed views; caches are filled while streaming)
# --------------------------------------------------------------------------------------


class TextCache:
    """Per-process caches keyed by the exact strings the reference path uses."""

    def __init__(self) -> None:
        self._words: dict[str, tuple[str, ...]] = {}
        self._ub: dict[str, int] = {}
        self._bb: dict[tuple[str, str], int] = {}
        self._bag: dict[str, tuple[int, ...]] = {}
        self._wordset: dict[str, frozenset[str]] = {}
        self._content: dict[str, list[str]] = {}
        self._content_set: dict[str, set[str]] = {}
        self._six: dict[str, frozenset[tuple[str, ...]]] = {}
        self._norm_lower: dict[str, str] = {}
        self._lower: dict[str, str] = {}
        self._numbers: dict[str, list[float]] = {}

    def clear(self) -> None:
        for d in (self._words, self._ub, self._bb, self._bag, self._wordset, self._content,
                  self._content_set, self._six, self._norm_lower, self._lower, self._numbers):
            d.clear()

    def words(self, text: str) -> tuple[str, ...]:
        got = self._words.get(text)
        if got is None:
            # identical to view.words / features.words for this ASCII word class
            got = tuple(_WORD_RE.findall(text.lower())) if text else ()
            self._words[text] = got
        return got

    def bag_indices(self, text: str) -> tuple[int, ...]:
        got = self._bag.get(text)
        if got is not None:
            return got
        ws = self.words(text)
        ub, bb = self._ub, self._bb
        idx: list[int] = []
        stop = _STOP
        for w in ws:
            if w in stop:
                continue
            v = ub.get(w)
            if v is None:
                v = features.bucket("u:" + w)
                ub[w] = v
            idx.append(v)
        prev = None
        for w in ws:
            if prev is not None and not (prev in stop and w in stop):
                key = (prev, w)
                v = bb.get(key)
                if v is None:
                    v = features.bucket("b:%s|%s" % key)
                    bb[key] = v
                idx.append(v)
            prev = w
        got = tuple(idx)
        self._bag[text] = got
        return got


    def word_set(self, text: str) -> frozenset[str]:
        got = self._wordset.get(text)
        if got is None:
            got = frozenset(w for w in self.words(text) if len(w) >= 4)
            self._wordset[text] = got
        return got

    def content_words(self, text: str) -> list[str]:
        got = self._content.get(text)
        if got is None:
            got = [w for w in self.words(text) if w not in _STOP and len(w) > 1]
            self._content[text] = got
        return got

    def content_set(self, text: str) -> set[str]:
        got = self._content_set.get(text)
        if got is None:
            got = set(self.content_words(text))
            self._content_set[text] = got
        return got

    def sixgrams(self, text: str) -> frozenset[tuple[str, ...]]:
        got = self._six.get(text)
        if got is None:
            c = self.content_words(text)
            got = frozenset(tuple(c[i:i + 6]) for i in range(max(0, len(c) - 5)))
            self._six[text] = got
        return got

    def norm_lower(self, text: str) -> str:
        got = self._norm_lower.get(text)
        if got is None:
            got = view.norm_space(text).lower()
            self._norm_lower[text] = got
        return got

    def lower(self, text: str) -> str:
        got = self._lower.get(text)
        if got is None:
            got = text.lower()
            self._lower[text] = got
        return got

    def numbers(self, text: str) -> list[float]:
        got = self._numbers.get(text)
        if got is None:
            out: list[float] = []
            for raw in _NUM_RE.findall(text or ""):
                try:
                    out.append(float(raw.replace(",", ".")))
                except ValueError:
                    continue
            self._numbers[text] = out
            got = out
        return got


_TAIL_WORD_RE = re.compile(r"[A-Za-z0-9_@.\-]+$")


class IncWordSet:
    """Exact incremental word set for a monotonically growing text."""

    def __init__(self) -> None:
        self.text = ""
        self.set: set[str] = set()
        self.carry = ""

    def reset(self, text: str, cache: TextCache) -> None:
        self.text = ""
        self.set = set()
        self.carry = ""
        self.append(text, cache)

    def append(self, delta: str, cache: TextCache) -> None:
        if not delta:
            return
        combined = self.carry + delta
        m = _TAIL_WORD_RE.search(combined)
        tail = m.group(0) if m else ""
        complete = combined[: len(combined) - len(tail)] if tail else combined
        if complete:
            for w in cache.words(complete):
                if len(w) >= 4:
                    self.set.add(w)
        self.carry = tail
        self.text += delta


    def current(self) -> set[str]:
        out = self.set
        if len(self.carry) >= 4:
            out = out | {self.carry.lower()}
        return out


# --------------------------------------------------------------------------------------
# block processing (exactly the r1 build_view rules)
# --------------------------------------------------------------------------------------


@dataclass
class BlockFeat:
    role: str
    text: str                 # stored, clipped text (what the student reads for sources)
    full_text: str            # normalised full text used by retrieval / value->source
    chars: int
    block_index: int = -1

    def as_prior(self) -> dict[str, Any]:
        return {"block_index": self.block_index, "role": self.role, "text": self.text,
                "chars": self.chars, "full_text": self.full_text}


def _copy_block(b: "BlockFeat") -> "BlockFeat":
    return BlockFeat(role=b.role, text=b.text, full_text=b.full_text, chars=b.chars,
                     block_index=b.block_index)


class BlockProcessor:
    def __init__(self, cache: TextCache) -> None:
        self.cache = cache
        self.memo: dict[tuple[str, str], BlockFeat | None] = {}

    def process(self, block: view.Block) -> BlockFeat | None:
        key = (block.role, block.text)
        if key in self.memo:
            got = self.memo[key]
            return None if got is None else _copy_block(got)
        role = block.role
        full = block.text
        if role == "system":
            full = view.strip_system_tools(full)
            stored = full
        elif role == "tool":
            full = view.strip_tool_wrapper(full)
            stored = view.clip_block(full)
        elif role == "assistant":
            reasoning, visible = view.split_assistant(full)
            stored = view.clip_block((visible + " " + reasoning).strip() if reasoning else visible)
            full = stored
        else:
            stored = view.clip_block(full)
        if not view.norm_space(stored):
            self.memo[key] = None
            return None
        full_text = view.norm_space(full)[:20000]
        got = BlockFeat(role=role, text=stored, full_text=full_text, chars=len(stored))
        self.memo[key] = got
        return _copy_block(got)


# --------------------------------------------------------------------------------------
# fast feature extraction (bitwise-compatible with features.extract)
# --------------------------------------------------------------------------------------


def _lcn_binary(a: list[str], six: frozenset[tuple[str, ...]]) -> int:
    """Equivalent to features.longest_common_ngram(a, b, 6) for the frozen b.

    The reference implementation only ever compares length-6 tuples against a set of
    length-6 tuples, so its return value is 6 iff a common 6-gram exists, else 0.
    """
    if len(a) < 6 or not six:
        return 0
    for i in range(len(a) - 5):
        if tuple(a[i:i + 6]) in six:
            return 6
    return 0


def _jacc(a: set[str], b: set[str]) -> float:
    return len(a & b) / (len(a | b) or 1)


def _rel_flags(value: str, ref_numbers: list[float]) -> tuple[int, int, int]:
    try:
        v = float(value.replace(",", ""))
    except ValueError:
        return 0, 0, 0
    if not ref_numbers:
        return 0, 0, 0
    equal = int(any(abs(v - r) < 1e-9 for r in ref_numbers))
    greater = int(any(v > r for r in ref_numbers))
    less = int(any(v < r for r in ref_numbers))
    return equal, greater, less


def _shape(value: str) -> tuple[int, int, int, int]:
    digits = sum(c.isdigit() for c in value)
    return (int(digits >= 6), int("@" in value), int(any(c.isupper() for c in value)),
            int(len(value) >= 24))


def fast_extract(rec: dict[str, Any], prior: list[BlockFeat], sources: list[BlockFeat],
                 cache: TextCache) -> dict[str, Any]:
    """Return exactly features.extract(view) for the same record/materialised inputs."""
    cur_text = rec["current"]["segment"]
    vis_text = rec["current"]["visible_so_far"] if rec["current"]["kind"] == "text" else ""
    user_text = rec["user_request"]
    cur_key = (rec["current"]["tool"] or "") + " " + cur_text

    src_bags = [cache.bag_indices(b.text) for b in sources] or [()]
    bags = {
        "cur": cache.bag_indices(cur_key),
        "vis": cache.bag_indices(vis_text),
        "user": cache.bag_indices(user_text),
        "src": src_bags,
    }

    cur_w = cache.content_words(cur_text)
    vis_w = cache.content_words(vis_text) if vis_text else []
    user_w = cache.content_words(user_text)
    user_set = cache.content_set(user_text)
    cur_set = set(cur_w)
    src_sets = [cache.content_set(b.text) for b in sources]
    cur_or_vis = cur_w + vis_w

    f: list[float] = []
    f.append(_jacc(cur_set, user_set))
    f.append(len(cur_set & user_set) / (len(cur_set) or 1))
    f.append(min(1.0, _lcn_binary(cur_or_vis, cache.sixgrams(user_text)) / 6.0))
    src_jacc = [_jacc(set(cur_or_vis), s) for s in src_sets] or [0.0]
    src_lcn = [_lcn_binary(cur_or_vis, cache.sixgrams(b.text)) for b in sources] or [0]
    f.append(max(src_jacc))
    f.append(min(1.0, max(src_lcn) / 6.0))
    f.append(max(src_jacc) - _jacc(cur_set, user_set))

    vs = rec.get("value_sources") or []
    n_args = len(vs)
    n_exact_user = n_norm_user = n_in_src = n_no_src = 0
    n_num = n_eq = n_gt = n_lt = 0
    n_digit6 = n_at = n_upper = n_long = 0
    conflicts = 0
    user_nums = cache.numbers(user_text)
    for item in vs:
        u = item.get("user") or {}
        if u.get("exact"):
            n_exact_user += 1
        if u.get("normalized"):
            n_norm_user += 1
        srcs = item.get("sources") or []
        if srcs:
            n_in_src += 1
            if not u.get("normalized"):
                conflicts += 1
        else:
            n_no_src += 1
        eq, gt, lt = _rel_flags(item["value"], user_nums)
        n_eq += eq
        n_gt += gt
        n_lt += lt
        if re.fullmatch(r"-?\d+(?:[.,]\d+)?", item["value"].strip()):
            n_num += 1
        d6, at, up, lg = _shape(item["value"])
        n_digit6 += d6
        n_at += at
        n_upper += up
        n_long += lg
    inv = 1.0 / (n_args or 1)
    f.extend([inv * n_exact_user, inv * n_norm_user, inv * n_in_src, inv * n_no_src,
              inv * conflicts, inv * n_num, inv * n_eq, inv * n_gt, inv * n_lt,
              inv * n_digit6, inv * n_at, inv * n_upper, inv * n_long])
    f.append(min(1.0, n_args / 6.0))
    f.append(float(n_args == 0))

    cur_nums = cache.numbers(cur_text + " " + vis_text)
    f.append(float(bool(cur_nums) and bool(user_nums)))
    f.append(float(bool(set(round(x, 6) for x in cur_nums) & set(round(x, 6) for x in user_nums))))
    f.append(float(bool(cur_nums) and bool(user_nums) and min(cur_nums) > max(user_nums)))

    counts = rec["counts"]
    f.extend([
        float(rec["current"]["kind"] == "tool_call"),
        min(1.0, counts["tool_blocks"] / 6.0),
        min(1.0, counts["user_blocks"] / 4.0),
        min(1.0, counts["prior_blocks"] / 12.0),
        min(1.0, rec["round_id"] / 8.0),
        min(1.0, math.log1p(rec["prefix_chars"]) / 12.0),
        min(1.0, len(cur_or_vis) / 250.0),
        min(1.0, len(user_w) / 250.0),
    ])
    if len(f) != features.DENSE_DIM:
        raise RuntimeError(f"dense width {len(f)}")
    return {"bags": bags, "dense": f}


# --------------------------------------------------------------------------------------
# fused value/source + retrieval using the maintained index (same result as view.py)
# --------------------------------------------------------------------------------------


def fast_value_sources(values: list[tuple[str, str]], user_request: str,
                       prior: list[BlockFeat], cache: TextCache) -> list[dict[str, Any]]:
    user_lower = cache.lower(user_request)
    user_norm = cache.norm_lower(user_request)
    out: list[dict[str, Any]] = []
    for path, value in values:
        if not value:
            hit = {"exact": False, "normalized": False}
        elif value in user_request:
            hit = {"exact": True, "normalized": True, "offset": user_request.find(value)}
        else:
            nv = view.norm_space(value).lower()
            hit = {"exact": False, "normalized": bool(nv) and nv in user_norm}
        rec: dict[str, Any] = {"path": path, "value": value, "user": hit, "sources": []}
        nv = None
        for b in prior:
            if b.role != "tool":
                continue
            ft = b.full_text
            if value in ft:
                rec["sources"].append({"block_index": b.block_index, "exact": True})
                continue
            if nv is None:
                nv = view.norm_space(value).lower()
            if nv and nv in cache.norm_lower(b.full_text):
                rec["sources"].append({"block_index": b.block_index, "exact": False})
        out.append(rec)
    return out


def fast_retrieval(current_text: str, prior: list[BlockFeat], cache: TextCache,
                   k: int = view.MAX_RELEVANT_BLOCKS,
                   cur: frozenset[str] | set[str] | None = None) -> list[dict[str, Any]]:
    if cur is None:
        cur = cache.word_set(current_text)
    denom = 1.0 + len(cur) ** 0.5
    scored: list[tuple[float, int, BlockFeat]] = []
    tool_positions: list[int] = []
    for pos, b in enumerate(prior):
        if b.role != "tool":
            continue
        tool_positions.append(pos)
        overlap = len(cur & cache.word_set(b.full_text))
        scored.append((overlap / denom, pos, b))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    picked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for score, _pos, b in scored:
        if len(picked) >= k:
            break
        if b.block_index in seen:
            continue
        seen.add(b.block_index)
        picked.append({"block_index": b.block_index, "lexical_score": round(float(score), 6)})
    if tool_positions:
        recent = prior[tool_positions[-1]]
        if recent.block_index not in seen:
            picked.append({"block_index": recent.block_index, "lexical_score": 0.0})
    picked.sort(key=lambda r: r["block_index"])
    return picked


# --------------------------------------------------------------------------------------
# merged three-seed student
# --------------------------------------------------------------------------------------


class TriStudent:
    """Three frozen S2 students evaluated as one batched group (same maths, one launch set)."""

    def __init__(self, ckpts: list[Path], device: str = "cuda",
                 scaler: Path | None = None) -> None:
        import torch
        from . import model as model_mod

        self.torch = torch
        self.device = device
        sc = np.load(scaler or (ART / "scaler_v6_v3.npz"), allow_pickle=True)
        self.dense_mean = np.asarray(sc["mean"], dtype=np.float32)
        self.dense_std = np.asarray(sc["std"], dtype=np.float32)
        weights = []
        cfg = None
        for path in ckpts:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if cfg is None:
                cfg = model_mod.ModelConfig(**{k: v for k, v in payload["config"].items()
                                               if k in model_mod.ModelConfig.__dataclass_fields__})
            sd = payload["state_dict"]
            weights.append(sd)
        self.cfg = cfg
        t = torch
        self.emb = t.stack([w["emb.weight"] for w in weights]).to(device).contiguous()
        self.norm_w = t.stack([w["norm.weight"] for w in weights]).to(device).contiguous()
        self.norm_b = t.stack([w["norm.bias"] for w in weights]).to(device).contiguous()
        self.w1 = t.stack([w["trunk.0.weight"] for w in weights]).to(device).contiguous()
        self.b1 = t.stack([w["trunk.0.bias"] for w in weights]).to(device).contiguous()
        self.w2 = t.stack([w["trunk.3.weight"] for w in weights]).to(device).contiguous()
        self.b2 = t.stack([w["trunk.3.bias"] for w in weights]).to(device).contiguous()
        self.wh = t.stack([w["head_main.weight"] for w in weights]).to(device).contiguous()
        self.bh = t.stack([w["head_main.bias"] for w in weights]).to(device).contiguous()
        self.n_seeds = len(ckpts)
        self.buckets = int(self.emb.shape[1])
        self.width = int(self.w1.shape[2])
        self.hidden = int(self.w1.shape[1])
        for p in (self.emb, self.norm_w, self.norm_b, self.w1, self.b1, self.w2, self.b2,
                  self.wh, self.bh):
            p.requires_grad_(False)

    def forward(self, feats: dict[str, Any]) -> "np.ndarray":
        t = self.torch
        dev = self.device
        bags = feats["bags"]
        n = self.n_seeds
        B = self.buckets
        # cur/vis/user are one bag per seed; src up to four bags per seed
        cur = np.asarray(bags["cur"], dtype=np.int64)
        vis = np.asarray(bags["vis"], dtype=np.int64)
        user = np.asarray(bags["user"], dtype=np.int64)
        src_bags = [np.asarray(b, dtype=np.int64) for b in bags["src"]]
        if not src_bags:
            src_bags = [np.zeros(0, dtype=np.int64)]
        # group order: for every seed: cur, vis, user, src bags...
        flat_parts: list[np.ndarray] = []
        offsets: list[int] = []
        counts: list[int] = []
        cursor = 0
        for g in range(n):
            for arr in (cur, vis, user):
                offsets.append(cursor)
                flat_parts.append(arr + g * B)
                cursor += arr.size
            for arr in src_bags:
                offsets.append(cursor)
                flat_parts.append(arr + g * B)
                cursor += arr.size
            counts.append(len(src_bags))
        flat = np.concatenate(flat_parts) if flat_parts else np.zeros(0, np.int64)
        offs = np.asarray(offsets, dtype=np.int64)
        idx_t = t.from_numpy(flat).to(dev, non_blocking=False)
        off_t = t.from_numpy(offs).to(dev)
        dense_np = (np.asarray(feats["dense"], dtype=np.float32) - self.dense_mean) / self.dense_std
        dense = t.from_numpy(dense_np.astype(np.float32)).to(dev)
        rows = t.nn.functional.embedding_bag(idx_t, self.emb.view(n * B, -1), off_t,
                                             mode="mean", per_sample_weights=None)
        # slice channels per seed
        per = 3 + len(src_bags)
        rows = rows.view(n, per, -1)
        x_cur = rows[:, 0, :]
        x_vis = rows[:, 1, :]
        x_user = rows[:, 2, :]
        src_rows = rows[:, 3:, :]                       # (n, nsrc, 16)
        src = src_rows.mean(dim=1)
        x = t.cat([x_cur, x_vis, x_user, src, dense.unsqueeze(0).expand(n, -1)], dim=-1)
        x = t.stack([t.nn.functional.layer_norm(x[g], (self.width,), self.norm_w[g],
                                                self.norm_b[g], 1e-5)
                     for g in range(n)])
        h = t.baddbmm(self.b1.unsqueeze(1), x.unsqueeze(1),
                      self.w1.transpose(1, 2)).squeeze(1)
        h = t.nn.functional.gelu(h)
        h = t.baddbmm(self.b2.unsqueeze(1), h.unsqueeze(1),
                      self.w2.transpose(1, 2)).squeeze(1)
        h = t.nn.functional.gelu(h)
        logit = (h * self.wh.squeeze(1)).sum(dim=-1) + self.bh.squeeze(1)
        return logit.detach().float().cpu().numpy()


# --------------------------------------------------------------------------------------
# episode stream
# --------------------------------------------------------------------------------------


@dataclass
class FrozenRouting:
    base_mean: float
    base_std: float
    stud_mean: float
    stud_std: float


def load_fusion(path: Path = FUSION_PATH) -> FrozenRouting:
    z = json.loads(path.read_text())
    return FrozenRouting(float(z["base_mean_TRAIN"]), float(z["base_std_TRAIN"]),
                         float(z["student_mean_TRAIN_OOF"]), float(z["student_std_TRAIN_OOF"]))


class EpisodeStream:
    """Incremental per-episode state (history/source index + current block)."""

    def __init__(self, *, episode_id: str, ids, offsets, round_ids, prompt_len, seq_len,
                 tok, tc: TextCache, bp: BlockProcessor) -> None:
        self.episode_id = episode_id
        self.ids = ids
        self.offsets = offsets
        self.round_ids = round_ids
        self.prompt_len = prompt_len
        self.seq_len = seq_len
        self.tok = tok
        self.tc = tc
        self.bp = bp
        self.round = None
        self.text = ""
        self.cursor = 0
        self.prompt_text = ""
        self.prompt_char_end = 0
        self.prior: list[BlockFeat] = []
        self.user_request = ""
        self.has_reasoning_before = False
        self.prev_prompt = None   # (round_index, prompt_len)
        self.reused_rounds = 0
        self.fallback_rounds = 0
        self.body_words = IncWordSet()
        self.prev_body = ""

    # -- round setup ------------------------------------------------------------------
    def begin_round(self, rid: int, prompt_tokens: int, first_end: int,
                    prompt_text_override: str | None = None,
                    body_text_override: str | None = None,
                    blocks_override: list | None = None) -> None:
        ri = self.round_ids.index(rid)
        off = self.offsets[ri]
        pl = int(self.prompt_len[ri])
        if pl > int(self.seq_len[ri]) or first_end > int(self.seq_len[ri]):
            raise ValueError(f"{self.episode_id}: bad round span")
        if prompt_text_override is not None:
            # typed controller path: prompt text already arrived; only new emission is decoded
            _d0 = time.perf_counter()
            body = (body_text_override if body_text_override is not None
                    else decode_ids(self.tok, self.ids[off + pl:off + first_end]))
            prompt_text = prompt_text_override
            self.text = prompt_text + body
            self._t_decode = time.perf_counter() - _d0
            self.reused_rounds += 1
            self.round = rid
            self.cursor = int(first_end)
            self.body_words = IncWordSet()
            self.prev_body = ""
            self.prompt_text = prompt_text
            self.prompt_char_end = len(prompt_text)
            self.prev_prompt = (ri, pl)
            blocks = blocks_override if blocks_override is not None else view.parse_blocks(prompt_text)
            self._t_parse = 0.0
            self._t_reason = 0.0
            if not blocks or blocks[-1].role != "assistant":
                raise ValueError(f"{self.episode_id}: prompt does not end in assistant")
            prior_blocks = blocks[:-1]
            self.has_reasoning_before = any("<think>" in b.text for b in prior_blocks)
            self.prior = self._process_window(prior_blocks)
            for b in prior_blocks:
                if b.role == "user":
                    self.user_request = view.norm_space(b.text)[:view.MAX_USER_CHARS]
                    break
            else:
                self.user_request = ""
            return
        reuse = False
        if self.prev_prompt is not None:
            pri, ppl = self.prev_prompt
            if pl >= ppl:
                poff = self.offsets[pri]
                a = self.ids[poff:poff + ppl]
                b = self.ids[off:off + ppl]
                if np.array_equal(a, b):
                    reuse = True
        _d0 = time.perf_counter()
        if reuse:
            prior_text = self.prompt_text
            # decode only the newly added history and the new emission separately
            hist = decode_ids(self.tok, self.ids[off + ppl:off + pl])
            body = decode_ids(self.tok, self.ids[off + pl:off + first_end])
            prompt_text = prior_text + hist
            self.text = prompt_text + body
            self.reused_rounds += 1
        else:
            prompt_text = decode_ids(self.tok, self.ids[off:off + pl])
            body = decode_ids(self.tok, self.ids[off + pl:off + first_end])
            self.text = prompt_text + body
            self.fallback_rounds += 1
        self._t_decode = time.perf_counter() - _d0
        self.round = rid
        self.cursor = int(first_end)
        self.body_words = IncWordSet()
        self.prev_body = ""
        self.prompt_text = prompt_text
        self.prompt_char_end = len(prompt_text)
        self.prev_prompt = (ri, pl)
        _p0 = time.perf_counter()
        blocks = view.parse_blocks(prompt_text)
        self._t_parse = time.perf_counter() - _p0
        _p1 = time.perf_counter()
        self._t_reason = 0.0
        if not blocks or blocks[-1].role != "assistant":
            raise ValueError(f"{self.episode_id}: prompt does not end in assistant")
        prior_blocks = blocks[:-1]
        self.has_reasoning_before = any("<think>" in b.text for b in prior_blocks)
        self.prior = self._process_window(prior_blocks)
        self._t_process = time.perf_counter() - _p1
        _u0 = time.perf_counter()
        for b in prior_blocks:
            if b.role == "user":
                self.user_request = view.norm_space(b.text)[:view.MAX_USER_CHARS]
                break
        else:
            self.user_request = ""
        self._t_user = time.perf_counter() - _u0

    def _process_window(self, blocks: list[view.Block]) -> list[BlockFeat]:
        out: list[BlockFeat] = []
        for block in blocks[-view.MAX_PRIOR_BLOCKS:]:
            got = self.bp.process(block)
            if got is None:
                continue
            got.block_index = len(out)
            out.append(got)
        return out

    def extend(self, t_end: int) -> None:
        off = self.offsets[self.round_ids.index(self.round)]
        delta = decode_ids(self.tok, self.ids[off + self.cursor:off + t_end])
        self.text += delta
        self.cursor = int(t_end)

    def advance_only(self, round_id: int, t_end: int) -> None:
        """No-new-emission event: advance decode/index state, do not score."""
        rid = int(round_id)
        if self.round != rid:
            self.begin_round(rid, 0, int(t_end))
        elif int(t_end) > self.cursor:
            self.extend(int(t_end))

    # -- one event --------------------------------------------------------------------
    def event(self, *, rec_meta: dict[str, Any], t_start: int, t_end: int,
              event_type: str, round_id: int, token_start: int, token_end: int,
              with_prefix_sha: bool = False, fast: bool = True,
              timed: bool = False, call_index: int | None = None,
              prompt_text_override: str | None = None,
              body_text_override: str | None = None,
              text_delta_override: str | None = None,
              blocks_override: list | None = None) -> dict[str, Any]:
        tm: dict[str, float] = {}
        _t0 = time.perf_counter()
        rid = int(round_id)
        if self.round != rid:
            self.begin_round(rid, 0, int(t_end), prompt_text_override=prompt_text_override,
                             body_text_override=body_text_override,
                             blocks_override=blocks_override)
        elif int(t_end) > self.cursor:
            if text_delta_override is not None:
                self.text += text_delta_override
                self.cursor = int(t_end)
            else:
                self.extend(int(t_end))
        if timed:
            tm["round"] = time.perf_counter() - _t0
        body = self.text[self.prompt_char_end:]
        idx = view._find_block_end(body, "assistant")
        if idx >= 0:
            body = body[:idx]
        if body.endswith("\n"):
            body = body[:-1]
        if IM_START in body:
            # a new message boundary inside the current assistant window: fall back to the
            # full reference parser for this event (rare, never expected for CT rows)
            blocks = view.parse_blocks(self.text)
            if not blocks or blocks[-1].role != "assistant":
                raise ValueError(f"{self.episode_id}: current block not assistant")
            body = blocks[-1].text
        current_full = body
        if body == self.prev_body:
            pass
        elif body.startswith(self.prev_body):
            self.body_words.append(body[len(self.prev_body):], self.tc)
        else:
            self.body_words.reset(body, self.tc)
        self.prev_body = body
        reasoning, current_visible = view.split_assistant(current_full)
        current_visible = current_visible[-1200:]
        kind = "tool_call" if str(event_type) == "tool_call" else "text"
        _tseg = time.perf_counter() if timed else 0.0
        _off = self.offsets[self.round_ids.index(rid)]
        seg = decode_ids(self.tok, self.ids[_off + int(t_start):_off + int(t_end)])
        empty_segment = int(t_start) == int(t_end)
        unknown_adapter = bool(empty_segment and kind == "tool_call")
        no_new_emission = bool(empty_segment and not unknown_adapter
                               and not view.norm_space(current_visible))
        block_fallback = False
        call = None
        if kind == "tool_call" and not unknown_adapter:
            call = view.select_current_call(current_full, seg, call_index)
            if call is not None and not call.get("parse_ok"):
                raise ValueError(f"{self.episode_id}: current tool call JSON does not parse")
            if call is None:
                raise ValueError(f"{self.episode_id}: tool_call row has no visible current call")
            if call.get("raw") and call["raw"] not in current_full:
                block_fallback = True
                current_full = seg
                reasoning, current_visible = view.split_assistant(current_full)
                current_visible = current_visible[-1200:]
                self.body_words.reset(current_full, self.tc)
                self.prev_body = current_full
        args = call.get("arguments") if call else None
        values = [(p, v) for p, v in view.arg_values(args) if len(v) >= 2][:12]
        _t1 = time.perf_counter() if timed else 0.0
        if fast:
            sources = fast_value_sources(values, self.user_request, self.prior, self.tc) if values else []
            relevant = fast_retrieval(current_full, self.prior, self.tc,
                                      cur=self.body_words.current())
        else:
            prior_dicts = [b.as_prior() for b in self.prior]
            sources = view.value_sources(values, self.user_request, prior_dicts) if values else []
            relevant = view.retrieval(current_full, prior_dicts)
        if timed:
            tm["segment_decode"] = _t1 - _tseg
            tm["sources_retrieval"] = time.perf_counter() - _t1
            _t3 = time.perf_counter()
        rec: dict[str, Any] = {
            "view_version": view.VIEW_VERSION,
            "episode_id": self.episode_id,
            "row_index": int(rec_meta["row_index"]),
            "event_id": str(rec_meta["event_id"]),
            "event_type": str(event_type),
            "scope": rec_meta["scope"],
            "group_id": rec_meta["group_id"],
            "round_id": rid,
            "token_start": int(token_start),
            "token_end": int(token_end),
            "row_label": bool(rec_meta["row_label"]),
            "episode_label": bool(rec_meta["episode_label"]),
            "user_request": self.user_request,
            "prior": [{"block_index": b.block_index, "role": b.role, "text": b.text,
                       "chars": b.chars} for b in self.prior],
            "relevant_tool_blocks": relevant,
            "current": {"kind": kind, "text": current_full[:view.MAX_CURRENT_CHARS],
                        "segment": seg[:view.MAX_CURRENT_CHARS],
                        "visible_so_far": current_visible,
                        "tool": None if call is None else call.get("name"),
                        "arguments": args,
                        "call_index": None if call_index is None else int(call_index),
                        "call_match": None if not call else call.get("call_match"),
                        "call_ordinal": None if not call else call.get("call_ordinal"),
                        "call_binding_conflict": False if not call else bool(call.get("call_binding_conflict")),
                        "calls_in_block": None if not call else call.get("calls_in_block")},
            "empty_segment": empty_segment,
            "no_new_emission": no_new_emission,
            "unknown_adapter": unknown_adapter,
            "block_fallback": block_fallback,
            "visible_chars": len(view.norm_space(current_visible)),
            "value_sources": sources,
            "prefix_chars": len(self.text),
            "counts": {"prior_blocks": len(self.prior),
                       "tool_blocks": sum(1 for b in self.prior if b.role == "tool"),
                       "user_blocks": sum(1 for b in self.prior if b.role == "user"),
                       "args": len(values)},
            "has_reasoning_before": self.has_reasoning_before,
        }
        if with_prefix_sha:
            rec["prefix_sha256"] = view.sha256_text(self.text)
            rec["current_sha256"] = view.sha256_text(seg)
        if fast:
            picked = [b for b in self.prior if b.block_index in
                      {r["block_index"] for r in relevant}]
            picked = [b for b in sorted(picked, key=lambda b: b.block_index)
                      if b.block_index in {r["block_index"] for r in relevant}][:view.MAX_RELEVANT_BLOCKS]
            # preserve the exact source order of view_text_fields: relevant order, first 3
            by_index = {b.block_index: b for b in self.prior}
            picked = [by_index[r["block_index"]] for r in relevant
                      if r["block_index"] in by_index][:view.MAX_RELEVANT_BLOCKS]
            feats = fast_extract(rec, self.prior, picked, self.tc)
        else:
            feats = features.extract(rec)
        if timed:
            tm["extract"] = time.perf_counter() - _t3
        return {"rec": rec, "feats": feats, "timings": tm}


# --------------------------------------------------------------------------------------
# model / episode score helper used by the benchmark wrapper
# --------------------------------------------------------------------------------------


class FrozenScorer:
    def __init__(self, arm: str = "S2", device: str = "cuda", seeds=(0, 1, 2),
                 artdir: Path = ART, tag: str = "", fusion_path: Path | None = None,
                 scaler_path: Path | None = None) -> None:
        self.arm = arm
        self.device = device
        self.seeds = tuple(seeds)
        suffix = f"_{tag}" if tag else ""
        scaler = Path(scaler_path) if scaler_path else (
            Path(artdir) / (f"scaler_{tag}.npz" if tag else "scaler_v6_v3.npz"))
        self.net = TriStudent([Path(artdir) / f"ckpt_{arm}{suffix}_seed{s}.pt" for s in self.seeds],
                              device=device, scaler=scaler)
        self.routing = load_fusion(fusion_path or FUSION_PATH)

    def row_logits(self, feats: dict[str, Any]) -> np.ndarray:
        return self.net.forward(feats)

    def episode_scores(self, per_seed_max: np.ndarray) -> np.ndarray:
        return np.asarray(per_seed_max, dtype=np.float32).mean(axis=0, dtype=np.float32)

    def fused(self, base_ep: np.ndarray, student_ep: np.ndarray) -> np.ndarray:
        r = self.routing
        return 0.5 * ((base_ep.astype(np.float64) - r.base_mean) / r.base_std) + \
            0.5 * ((student_ep.astype(np.float64) - r.stud_mean) / r.stud_std)
