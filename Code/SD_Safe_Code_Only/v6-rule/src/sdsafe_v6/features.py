"""Hashed text + explicit value/source features for one event view.

Two parts, both computed from strings only (no tokenizer, no LLM):

``bags``   -- hashed word unigram/bigram bags over four text channels, embedded by a learned
              ``EmbeddingBag``.  This is the "small encoder" part (a fixed, few-million-parameter
              classification network; not a second LLM and it generates no review text).
``dense``  -- explicit, interpretable parameter-level relations: where each argument value
              literally comes from, whether the current text echoes a low-trust directive,
              numeric/relational agreement with the user request, and structural counts.

The extractor is the single implementation used by training, evaluation *and* the latency
benchmark, so the measured cost is the real one.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Any, Iterable

from . import view

BUCKETS = 1 << 17
DENSE_DIM = 32
CHANNELS = ("cur", "vis", "user", "src")

_WORD_RE = re.compile(r"[A-Za-z0-9_@.\-]+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_STOP = frozenset("""a an the and or of to in on for with at by from is are was were be been
as that this these those it its i you he she they we my your our their me us them not no
please can could would will shall may might should do does did done have has had if then than
so such into about over under after before while when where which who whom what how why""".split())


def bucket(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big") % BUCKETS


@lru_cache(maxsize=1024)
def words(text: str) -> tuple[str, ...]:
    return tuple(w.lower() for w in _WORD_RE.findall(text or ""))


@lru_cache(maxsize=1024)
def bag_indices(text: str, min_len: int = 1) -> tuple[int, ...]:
    """Unigram + bigram hashed indices over content words (deterministic, order-insensitive)."""
    ws = [w for w in words(text) if len(w) >= min_len]
    idx: list[int] = [bucket("u:" + w) for w in ws if w not in _STOP]
    for a, b in zip(ws, ws[1:]):
        if a in _STOP and b in _STOP:
            continue
        idx.append(bucket("b:%s|%s" % (a, b)))
    return tuple(idx)


def _content(text: str) -> list[str]:
    return [w for w in words(text) if w not in _STOP and len(w) > 1]


def longest_common_ngram(a: list[str], b: list[str], n_max: int = 6) -> int:
    """Length of the longest common contiguous word run (bounded) -- cheap and exact."""
    if not a or not b:
        return 0
    best = 0
    bset = {tuple(b[i:i + n_max]) for i in range(max(0, len(b) - n_max + 1))}
    for i in range(len(a)):
        for n in range(min(n_max, len(a) - i), best, -1):
            if tuple(a[i:i + n]) in bset:
                best = n
                break
    return best


def _numbers(text: str) -> list[float]:
    out = []
    for raw in _NUM_RE.findall(text or ""):
        try:
            out.append(float(raw.replace(",", ".")))
        except ValueError:
            continue
    return out


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


def extract(rec: dict[str, Any]) -> dict[str, Any]:
    """Return {"bags": {channel: [idx...]}, "dense": [56 floats]} for one view record."""
    fields = view.view_text_fields(rec)
    cur = fields["current"]
    vis = fields["visible_so_far"] if rec["current"]["kind"] == "text" else ""
    user = fields["user_request"]
    sources = fields["sources"][:3]

    bags = {
        "cur": bag_indices((rec["current"]["tool"] or "") + " " + cur),
        "vis": bag_indices(vis),
        "user": bag_indices(user),
        "src": [bag_indices(s) for s in sources] or [[]],
    }

    cur_w = _content(cur)
    vis_w = _content(vis)
    user_w = _content(user)
    user_set = set(user_w)
    cur_set = set(cur_w)
    src_sets = [set(_content(s)) for s in sources]
    cur_or_vis = cur_w + vis_w

    def jacc(a: set[str], b: set[str]) -> float:
        return len(a & b) / (len(a | b) or 1)

    # text <-> user agreement
    f: list[float] = []
    f.append(jacc(cur_set, user_set))
    f.append(len(cur_set & user_set) / (len(cur_set) or 1))
    f.append(min(1.0, longest_common_ngram(cur_or_vis, user_w) / 6.0))
    # text <-> untrusted source agreement (max over retrieved sources)
    src_jacc = [jacc(set(cur_or_vis), s) for s in src_sets] or [0.0]
    src_lcn = [longest_common_ngram(cur_or_vis, _content(s)) for s in sources] or [0]
    f.append(max(src_jacc))
    f.append(min(1.0, max(src_lcn) / 6.0))
    f.append(max(src_jacc) - jacc(cur_set, user_set))

    # per-argument provenance
    vs = rec.get("value_sources") or []
    n_args = len(vs)
    n_exact_user = n_norm_user = n_in_src = n_no_src = 0
    n_num = n_eq = n_gt = n_lt = 0
    n_digit6 = n_at = n_upper = n_long = 0
    conflicts = 0
    user_nums = _numbers(user)
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

    # numeric agreement between the current text and the user request (generic relation)
    cur_nums = _numbers(cur + " " + vis)
    f.append(float(bool(cur_nums) and bool(user_nums)))
    f.append(float(bool(set(round(x, 6) for x in cur_nums) & set(round(x, 6) for x in user_nums))))
    f.append(float(bool(cur_nums) and bool(user_nums) and min(cur_nums) > max(user_nums)))

    # structure
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
    if len(f) != DENSE_DIM:
        raise RuntimeError(f"dense feature width {len(f)} != {DENSE_DIM}")
    return {"bags": bags, "dense": f}
