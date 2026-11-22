"""Strictly causal, deployment-reproducible parameter-level view of one CT event.

The only authoritative input is the *actual* prompt token-id prefix that the frozen model
consumed, up to the accepted endpoint token of the event::

    ids[round_offset : round_offset + token_end]

It is decoded with the frozen Qwen3-8B tokenizer that produced it (no re-serialisation, no
re-templating, no truncation of the endpoint).  From that decoded prefix we recover

  * the trusted original user request (first non-tool ``user`` block),
  * every earlier message block with its true provenance role -- a Qwen tool output is a
    ``user`` block wrapped in ``<tool_response>``; it is *data*, never an instruction,
  * the current action: for a ``tool_call`` event the call head + full argument JSON; for a
    ``content_segment`` event the exact text already emitted in the current block, plus the
    current sentence-sized segment,
  * a generic, GT-free value/source index: where each argument value literally appears
    (exact bytes, or whitespace/case-normalised) in the user request / an earlier tool output.

Nothing after the endpoint is read: the prefix simply does not contain it.  No ground-truth
label, checker goal, suite/task id, attack family, group id, attack type, future tool result
or later call is used to build a feature.  Those are carried only as metadata for accounting.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
_BLOCK_RE = re.compile(r"<\|im_start\|>([A-Za-z0-9_]+)\n")
_TOOL_RESP_OPEN = "<tool_response>"
_TOOL_RESP_CLOSE = "</tool_response>"

VIEW_VERSION = "v6param-view-4-callfix"

# Deterministic, deployment-legal truncation budgets (characters, not tokens).  Full block
# text is still used for the exact value/source index; only the stored/read digest is clipped.
MAX_USER_CHARS = 1600
HEAD_CHARS = 520
TAIL_CHARS = 260
MAX_PRIOR_BLOCKS = 24
MAX_CURRENT_CHARS = 2400
MAX_RELEVANT_BLOCKS = 3

_WORD_RE = re.compile(r"[A-Za-z0-9_@.\-]+")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


@lru_cache(maxsize=1024)
def words(text: str) -> tuple[str, ...]:
    return tuple(w.lower() for w in _WORD_RE.findall(str(text)))


@lru_cache(maxsize=4096)
def word_set(text: str) -> frozenset[str]:
    return frozenset(w for w in words(text) if len(w) >= 4)


@dataclass
class Block:
    role: str          # system | user | tool | assistant
    text: str          # full body
    closed: bool
    raw_role: str      # role token as it appeared in the template


def _classify(raw_role: str, body: str) -> str:
    if raw_role == "system":
        return "system"
    if raw_role in ("user", "tool"):
        return "tool" if _TOOL_RESP_OPEN in body else "user"
    return raw_role


def _tool_depth(text: str) -> int:
    """Unclosed ``<tool_response>`` markers in a fragment (tool data is quoted, not a turn)."""
    return max(0, text.count(_TOOL_RESP_OPEN) - text.count(_TOOL_RESP_CLOSE))


def _find_block_end(chunk: str, raw_role: str) -> int:
    """First ``<|im_end|>`` that is not inside quoted tool data.

    For ``user``/``tool`` blocks an open ``<tool_response>`` payload quotes the marker.
    For an ``assistant`` block an open ``<tool_call>`` JSON string quotes it as well; a
    forged ``<|im_end|>`` inside an argument value must not truncate the call.
    """
    if raw_role not in ("user", "tool", "assistant"):
        return chunk.find(IM_END)
    start = 0
    while True:
        idx = chunk.find(IM_END, start)
        if idx < 0:
            return -1
        if raw_role in ("user", "tool"):
            inside = _tool_depth(chunk[:idx]) > 0
        else:
            inside = _has_open_tool_call_json(chunk[:idx])
        if not inside:
            return idx
        start = idx + len(IM_END)


def _has_open_tool_call_json(text: str) -> bool:
    """True iff a ``<tool_call>`` in ``text`` starts a JSON object that is not closed yet.

    Used by the block boundary test: a template marker inside an unterminated tool-call
    argument string is data, not a new message.  This is the F1 repair for adversarial
    assistant arguments that contain a quoted ``<|im_start|>role`` sequence.
    """
    i = 0
    while True:
        j = text.find("<tool_call>", i)
        if j < 0:
            return False
        m = j + len("<tool_call>")
        while m < len(text) and text[m] in " \t\r\n":
            m += 1
        if m >= len(text) or text[m] != "{":
            i = j + len("<tool_call>")
            continue
        e = _scan_json_object(text, m)
        if e is None:
            return True
        i = e


def _true_starts(prefix_text: str) -> list[re.Match[str]]:
    """Real template block boundaries only.

    A role token inside tool output is untrusted data.  A candidate is a real boundary only if
    (a) it is at the beginning of the prompt, (b) the text immediately before it ends with the
    template's ``<|im_end|>`` (optionally a newline), and (c) the previous block has no open
    ``<tool_response>`` payload.  This stops the common injected
    ``...</tool output><|im_start|>user\n...`` from becoming a real user turn.
    """
    matches = list(_BLOCK_RE.finditer(prefix_text))
    starts: list[re.Match[str]] = []
    for m in matches:
        if not starts:
            if prefix_text[:m.start()].strip() == "":
                starts.append(m)
            continue
        prev = starts[-1]
        if prev.group(1) in ("user", "tool") and _tool_depth(prefix_text[prev.end():m.start()]) > 0:
            continue
        if prev.group(1) == "assistant" and _has_open_tool_call_json(
                prefix_text[prev.end():m.start()]):
            continue
        # avoid copying the whole prefix for every candidate (mechanical, same result):
        # true iff the text before the candidate, ignoring trailing whitespace, ends with IM_END
        k = m.start()
        while k > 0 and prefix_text[k - 1] in " \t\r\n":
            k -= 1
        if k >= len(IM_END) and prefix_text[k - len(IM_END):k] == IM_END:
            starts.append(m)
    return starts


def parse_blocks(prefix_text: str) -> list[Block]:
    """Split a (possibly truncated) decoded chat prefix into provenance-aware blocks.

    Only the first ``system`` and the first ``user`` block can carry trusted provenance.  Any
    later ``user``/``system`` role token that survives the boundary test is downgraded to data
    (``data_user``/``data_system``): in this template a later user block is a tool payload, and a
    forged ChatML role must never become a trusted instruction.
    """
    starts = _true_starts(prefix_text)
    blocks: list[Block] = []
    trusted_user = trusted_system = False
    for i, m in enumerate(starts):
        raw_role = m.group(1)
        body_start = m.end()
        body_end = starts[i + 1].start() if i + 1 < len(starts) else len(prefix_text)
        chunk = prefix_text[body_start:body_end]
        closed = False
        idx = _find_block_end(chunk, raw_role)
        if idx >= 0:
            chunk = chunk[:idx]
            closed = True
        if chunk.endswith("\n"):
            chunk = chunk[:-1]
        role = _classify(raw_role, chunk)
        if role == "system":
            if trusted_system:
                role = "data_system"
            else:
                trusted_system = True
        elif role == "user":
            if trusted_user:
                role = "data_user"
            else:
                trusted_user = True
        blocks.append(Block(role=role, text=chunk, closed=closed, raw_role=raw_role))
    return blocks


def strip_system_tools(text: str) -> str:
    """Drop the constant tool-signature catalogue from the system block.

    The catalogue is byte-identical across episodes and describes *available* tools, not the
    selected call; the selected tool name plus its arguments are in the current action.
    """
    cut = text.find("# Tools")
    if cut > 0:
        text = text[:cut]
    return norm_space(text)


def strip_tool_wrapper(text: str) -> str:
    t = text
    if t.lstrip().startswith(_TOOL_RESP_OPEN):
        t = t.lstrip()[len(_TOOL_RESP_OPEN):]
    end = t.rfind(_TOOL_RESP_CLOSE)
    if end >= 0:
        t = t[:end]
    return t


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)


def split_assistant(text: str) -> tuple[str, str]:
    reasoning = ""
    m = _THINK_RE.search(text)
    if m:
        reasoning = m.group(1)
        text = text[: m.start()] + text[m.end():]
    return reasoning.strip(), text.strip()


def clip_block(text: str) -> str:
    text = norm_space(text)
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return text[:HEAD_CHARS] + " [...] " + text[-TAIL_CHARS:]


def _reasoning_spans(text: str) -> list[tuple[int, int]]:
    """Character spans that are reasoning rather than emitted action.

    A closed ``<think>...</think>`` span is reasoning.  An unclosed ``<think>`` suppresses
    every following tag: with no visible closing marker the material is still reasoning.
    """
    spans = [(m.start(), m.end()) for m in _THINK_RE.finditer(text)]
    for m in re.finditer(r"<think>", text):
        if not any(a <= m.start() < b for a, b in spans):
            spans.append((m.start(), len(text)))
            break
    return spans


def _scan_json_object(text: str, start: int) -> int | None:
    """Index just past the matching ``}`` of the object starting at ``start``.

    Tracks JSON string/escape state, so braces and ``<tool_call>`` literals inside strings
    are not structural boundaries.  Returns ``None`` if the object is not closed.
    """
    depth = 0
    in_string = False
    escaped = False
    i = start
    while i < len(text):
        c = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


def scan_tool_calls(text: str) -> list[dict[str, Any]]:
    """All real tool-call emissions in ``text``, in order, with causal character spans.

    Rules (see PROTOCOL_CORRECTED_R3 C1):

    * a candidate must be ``<tool_call>`` followed (whitespace allowed) by a JSON object;
    * the JSON object is scanned with string/escape awareness, so nested JSON, escaped
      quotes and a literal ``<tool_call>`` inside a JSON string are handled;
    * calls inside reasoning spans are ignored;
    * the closing ``</tool_call>`` is optional (the CT endpoint may stop before it), so a
      complete object without a closer is a valid unclosed call;
    * an incomplete object at the end of the available text is recorded with ``obj=None``
      instead of being guessed.
    """
    spans = _reasoning_spans(text)
    out: list[dict[str, Any]] = []
    i = 0
    while True:
        j = text.find("<tool_call>", i)
        if j < 0:
            break
        if any(a <= j < b for a, b in spans):
            i = j + len("<tool_call>")
            continue
        m = j + len("<tool_call>")
        while m < len(text) and text[m] in " \t\r\n":
            m += 1
        if m >= len(text) or text[m] != "{":
            i = j + len("<tool_call>")
            continue
        e = _scan_json_object(text, m)
        if e is None:
            out.append({"start": j, "json_start": m, "json_end": None, "end": len(text),
                        "raw": text[j:], "closed": False, "obj": None})
            break
        close = text.find("</tool_call>", e)
        closed = False
        end = e
        if close >= 0 and text[e:close].strip() == "":
            end = close + len("</tool_call>")
            closed = True
        try:
            obj = json.loads(text[m:e])
        except json.JSONDecodeError:
            obj = None
        out.append({"start": j, "json_start": m, "json_end": e, "end": end,
                    "raw": text[j:end], "closed": closed, "obj": obj})
        i = end
    return out


def _call_payload(call: dict[str, Any] | None) -> tuple[Any, Any]:
    if not call or not isinstance(call.get("obj"), dict):
        return None, None
    obj = call["obj"]
    name = obj.get("name")
    args = obj.get("arguments")
    if isinstance(obj.get("function"), dict):
        name = name or obj["function"].get("name")
        args = args if args is not None else obj["function"].get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return name, args


def select_current_call(current_text: str, segment_text: str,
                        call_index: int | None) -> dict[str, Any] | None:
    """Select the call described by *this* CT row (never a later/earlier call).

    ``call_index`` is the event's ordinal among the calls of the current assistant block;
    it is used only to select a candidate.  The independently parsed CT ``segment`` must
    agree on both name and arguments.  If they disagree, the segment (the actual readout)
    wins when its payload is unique; otherwise the ordinal candidate is kept and the
    conflict is recorded.  All calls live in the already-decoded causal prefix.
    """
    calls = scan_tool_calls(current_text)
    seg_calls = scan_tool_calls(segment_text)
    seg = seg_calls[-1] if seg_calls else None
    seg_name, seg_args = _call_payload(seg)
    ci = int(call_index) if call_index is not None else None
    ordinal = calls[ci] if ci is not None and 0 <= ci < len(calls) else None
    ord_name, ord_args = _call_payload(ordinal)
    conflict = False
    selected = None
    method = "none"
    if seg_name is not None and ord_name is not None:
        if seg_name == ord_name and seg_args == ord_args:
            selected, method = ordinal, "ordinal+segment"
        else:
            conflict = True
            matches = [i for i, c in enumerate(calls)
                       if _call_payload(c) == (seg_name, seg_args)]
            if len(matches) == 1:
                selected, method = calls[matches[0]], "segment"
            elif matches and ci in matches:
                selected, method = ordinal, "segment+ordinal_ambiguous"
            elif matches:
                selected, method = calls[matches[0]], "segment_ambiguous"
            else:
                selected, method = seg, "segment_only"
    elif ordinal is not None:
        selected, method = ordinal, "ordinal"
    elif seg_name is not None:
        matches = [i for i, c in enumerate(calls) if _call_payload(c) == (seg_name, seg_args)]
        if len(matches) == 1:
            selected, method = calls[matches[0]], "segment"
        elif matches and ci in matches:
            selected, method = ordinal, "segment+ordinal_ambiguous"
        elif matches:
            selected, method = calls[matches[0]], "segment_ambiguous"
        else:
            selected, method = seg, "segment_only"
    # a selected call that is not the last one visible in the prefix would be suspicious,
    # but calls after the row endpoint cannot be in the prefix by construction.
    selected_pos = None
    if selected is not None:
        for i, c in enumerate(calls):
            if c is selected:
                selected_pos = i
                break
    name, args = _call_payload(selected)
    if selected is not None and not isinstance(selected.get("obj"), dict):
        return {"parse_ok": False, "raw": selected.get("raw"), "name": None,
                "arguments": None, "call_match": method, "call_binding_conflict": conflict,
                "call_ordinal": selected_pos, "calls_in_block": len(calls)}
    if selected is None:
        return None
    return {"parse_ok": True, "name": name, "arguments": args, "raw": selected.get("raw"),
            "call_match": method, "call_binding_conflict": conflict,
            "call_ordinal": selected_pos, "calls_in_block": len(calls),
            "call_closed": bool(selected.get("closed"))}


def _norm_value(value: Any) -> str:
    return norm_space(str(value)).lower()


def locate_value(value: str, text: str) -> dict[str, Any]:
    if not value:
        return {"exact": False, "normalized": False}
    exact = value in text
    if exact:
        return {"exact": True, "normalized": True, "offset": text.find(value)}
    v = _norm_value(value)
    t = text.lower()
    return {"exact": False, "normalized": bool(v) and v in norm_space(t)}


def arg_values(arguments: Any) -> list[tuple[str, str]]:
    """(json_path, string_value) pairs of the current call arguments (flattened)."""
    out: list[tuple[str, str]] = []

    def walk(prefix: str, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(val, list):
            for i, v in enumerate(val):
                walk(f"{prefix}[{i}]", v)
        elif isinstance(val, bool) or val is None:
            return
        elif isinstance(val, (int, float)):
            out.append((prefix, str(val)))
        elif isinstance(val, str):
            out.append((prefix, val))

    walk("", arguments)
    return out


def value_sources(values: list[tuple[str, str]], user_request: str,
                  prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for path, value in values:
        rec: dict[str, Any] = {"path": path, "value": value,
                               "user": locate_value(value, user_request), "sources": []}
        for block in prior:
            if block["role"] != "tool":
                continue
            hit = locate_value(value, block["full_text"])
            if hit["exact"] or hit["normalized"]:
                rec["sources"].append({"block_index": block["block_index"],
                                       "exact": bool(hit["exact"])})
        out.append(rec)
    return out


def retrieval(current_text: str, prior: list[dict[str, Any]], *, k: int = MAX_RELEVANT_BLOCKS) -> list[dict[str, Any]]:
    """Generic lexical retrieval of the tool-output blocks most related to the current action.

    Bag-of-words overlap over words of length >= 4, normalised by ``sqrt(len(cur))``; ties
    broken by recency.  A deployable index query: no label, no injection marker, no fixed
    entity constant.
    """
    cur = word_set(current_text)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for pos, block in enumerate(prior):
        if block["role"] != "tool":
            continue
        bw = word_set(block["full_text"])
        overlap = len(cur & bw)
        scored.append((overlap / (1.0 + len(cur) ** 0.5), pos, block))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    picked: list[dict[str, Any]] = []
    seen: set[int] = set()
    for score, _pos, block in scored:
        if len(picked) >= k:
            break
        if block["block_index"] in seen:
            continue
        seen.add(block["block_index"])
        picked.append({"block_index": block["block_index"], "lexical_score": round(float(score), 6)})
    recent = [b for b in prior if b["role"] == "tool"]
    if recent and recent[-1]["block_index"] not in seen:
        picked.append({"block_index": recent[-1]["block_index"], "lexical_score": 0.0})
    picked.sort(key=lambda r: r["block_index"])
    return picked


def build_view(
    *,
    episode_id: str,
    row: dict[str, Any],
    round_ids: list[int],
    round_seq_len: list[int],
    round_prompt_len: list[int],
    offsets: list[int],
    ids: Any,
    decode,
    row_label: bool,
    episode_label: bool,
    group_id: str,
    scope: str,
) -> dict[str, Any]:
    """One event -> one JSON-serialisable, strictly causal view record."""
    rid = int(row["row_round_id"])
    if rid not in round_ids:
        raise ValueError(f"{episode_id}: round {rid} missing from prepared rounds")
    ri = round_ids.index(rid)
    off = int(offsets[ri])
    seq_len = int(round_seq_len[ri])
    t_start, t_end = int(row["token_start"]), int(row["token_end"])
    if not (0 <= t_start <= t_end <= seq_len):
        raise ValueError(f"{episode_id}: bad span {t_start}-{t_end} for seq {seq_len}")
    prefix_text = decode(ids[off: off + t_end])
    segment_text = decode(ids[off + t_start: off + t_end])

    blocks = parse_blocks(prefix_text)
    if not blocks:
        raise ValueError(f"{episode_id}: decoded prefix has no chat block")
    current_kind = "tool_call" if str(row["event_type"]) == "tool_call" else "text"
    block_fallback = False
    current_block = blocks[-1]
    if current_block.role != "assistant":
        # Adversarial TRAIN rows can close a tool_response and then forge a user/assistant
        # marker before the real assistant tail; the accepted CT span still proves a real
        # emitted tool call.  Use that span as the current action (causal, no future) and
        # keep every parsed block as prior data.  Fail closed for text rows.
        if current_kind == "tool_call" and norm_space(segment_text):
            block_fallback = True
            current_full = segment_text
            _reasoning, current_visible = split_assistant(current_full)
            current_visible = current_visible[-1200:]
            prior_blocks_all = list(blocks)
        else:
            raise ValueError(f"{episode_id}: current block role {current_block.role!r} is not assistant")
    else:
        current_full = current_block.text
        _reasoning, current_visible = split_assistant(current_full)
        current_visible = current_visible[-1200:]
        prior_blocks_all = blocks[:-1]

    user_request = ""
    for block in blocks:
        if block.role == "user":
            user_request = norm_space(block.text)[:MAX_USER_CHARS]
            break

    prior_blocks = prior_blocks_all[-MAX_PRIOR_BLOCKS:]
    prior: list[dict[str, Any]] = []
    for block in prior_blocks:
        full = block.text
        if block.role == "system":
            full = strip_system_tools(full)
            stored = full
        elif block.role == "tool":
            full = strip_tool_wrapper(full)
            stored = clip_block(full)
        elif block.role == "assistant":
            reasoning, visible = split_assistant(full)
            stored = clip_block((visible + " " + reasoning).strip() if reasoning else visible)
            full = stored
        else:
            stored = clip_block(full)
        if not norm_space(stored):
            continue
        prior.append({"block_index": len(prior), "role": block.role, "text": stored,
                      "chars": len(stored), "full_text": norm_space(full)[:20000]})

    current_kind = "tool_call" if str(row["event_type"]) == "tool_call" else "text"
    empty_segment = int(t_start) == int(t_end)
    visible_norm = norm_space(current_visible)
    unknown_adapter = bool(empty_segment and current_kind == "tool_call")
    no_new_emission = bool(empty_segment and not unknown_adapter and not visible_norm)
    call = None
    if current_kind == "tool_call" and not unknown_adapter:
        call = select_current_call(current_full, segment_text, row.get("call_index"))
        if call is not None and not call.get("parse_ok"):
            raise ValueError(f"{episode_id}: current tool call JSON does not parse")
        if call is None:
            raise ValueError(f"{episode_id}: tool_call row has no visible current call")
        if call.get("raw") and call["raw"] not in current_full:
            # The accepted segment carries the real call but the decoded block parser could
            # not retain it (adversarial nested template text).  Treat the segment as the
            # current action; prior material is unchanged and causal.
            block_fallback = True
            current_full = segment_text
            _reasoning, current_visible = split_assistant(current_full)
            current_visible = current_visible[-1200:]
            visible_norm = norm_space(current_visible)

    values = [(p, v) for p, v in arg_values(call.get("arguments") if call else None) if len(v) >= 2][:12]
    sources = value_sources(values, user_request, prior) if values else []
    relevant = retrieval(current_full, prior)

    rec: dict[str, Any] = {
        "view_version": VIEW_VERSION,
        "episode_id": episode_id,
        "row_index": int(row["row_index"]),
        "event_id": str(row["event_id"]),
        "event_type": str(row["event_type"]),
        "scope": scope,
        "group_id": group_id,
        "round_id": rid,
        "token_start": t_start,
        "token_end": t_end,
        "round_seq_len": seq_len,
        "round_prompt_len": int(round_prompt_len[ri]),
        "row_label": bool(row_label),
        "episode_label": bool(episode_label),
        "user_request": user_request,
        "prior": [{k: v for k, v in b.items() if k != "full_text"} for b in prior],
        "relevant_tool_blocks": relevant,
        "current": {
            "kind": current_kind,
            "text": current_full[:MAX_CURRENT_CHARS],
            "segment": segment_text[:MAX_CURRENT_CHARS],
            "visible_so_far": current_visible,
            "tool": None if not call else call.get("name"),
            "arguments": None if not call else call.get("arguments"),
            "call_index": None if row.get("call_index") is None else int(row["call_index"]),
            "call_match": None if not call else call.get("call_match"),
            "call_ordinal": None if not call else call.get("call_ordinal"),
            "call_binding_conflict": False if not call else bool(call.get("call_binding_conflict")),
            "calls_in_block": None if not call else call.get("calls_in_block"),
        },
        "empty_segment": empty_segment,
        "no_new_emission": no_new_emission,
        "unknown_adapter": unknown_adapter,
        "block_fallback": block_fallback,
        "visible_chars": len(visible_norm),
        "value_sources": sources,
        "prefix_chars": len(prefix_text),
        "prefix_sha256": sha256_text(prefix_text),
        "current_sha256": sha256_text(segment_text),
        "has_reasoning_before": any("<think>" in b.text for b in blocks[:-1]),
        "counts": {
            "prior_blocks": len(prior),
            "tool_blocks": sum(1 for b in prior if b["role"] == "tool"),
            "user_blocks": sum(1 for b in prior if b["role"] == "user"),
            "args": len(values),
        },
    }
    return rec


def view_text_fields(rec: dict[str, Any], *, max_sources: int = MAX_RELEVANT_BLOCKS) -> dict[str, Any]:
    """The exact strings the student encoder is allowed to read (single source of truth)."""
    by_index = {b["block_index"]: b for b in rec["prior"]}
    picked = [by_index[r["block_index"]] for r in rec["relevant_tool_blocks"]
              if r["block_index"] in by_index][:max_sources]
    return {
        "user_request": rec["user_request"],
        "current": rec["current"]["segment"],
        "visible_so_far": rec["current"]["visible_so_far"],
        "sources": [b["text"] for b in picked],
        "tool_count": sum(1 for b in rec["prior"] if b["role"] == "tool"),
    }


def build_view_stream(
    *,
    episode_id: str,
    row_index: int,
    event_id: str,
    event_type: str,
    scope: str,
    group_id: str,
    round_id: int,
    user_request: str,
    prior: list[dict[str, Any]],
    current_segment_text: str,
    current_visible_so_far: str,
    current_tool: str | None,
    current_args: Any,
    token_start: int = 0,
    token_end: int = 0,
    row_label: bool = False,
    episode_label: bool = False,
) -> dict[str, Any]:
    """Same record as :func:`build_view`, but from an already-maintained incremental index.

    ``prior`` entries carry ``{"role", "text", "full_text"}`` (role already classified when the
    block was appended).  This is the deployment path: only the *new* segment text is decoded and
    every derived field (retrieval, value/source index, counts) is recomputed from strings, so the
    latency benchmark charges all of it.
    """
    prior_out: list[dict[str, Any]] = []
    for block in prior[-MAX_PRIOR_BLOCKS:]:
        stored = block["text"]
        full = block.get("full_text", stored)
        if not norm_space(stored):
            continue
        prior_out.append({"block_index": len(prior_out), "role": block["role"], "text": stored,
                          "chars": len(stored), "full_text": full})
    norm_user = norm_space(user_request)[:MAX_USER_CHARS]
    kind = "tool_call" if str(event_type) == "tool_call" else "text"
    call = {"parse_ok": True, "name": current_tool, "arguments": current_args} if kind == "tool_call" else None
    values = [(p, v) for p, v in arg_values(current_args) if len(v) >= 2][:12]
    sources = value_sources(values, norm_user, prior_out) if values else []
    relevant = retrieval(current_segment_text, prior_out)
    return {
        "view_version": VIEW_VERSION,
        "episode_id": episode_id,
        "row_index": int(row_index),
        "event_id": str(event_id),
        "event_type": str(event_type),
        "scope": scope,
        "group_id": group_id,
        "round_id": int(round_id),
        "token_start": int(token_start),
        "token_end": int(token_end),
        "round_seq_len": 0,
        "round_prompt_len": 0,
        "row_label": bool(row_label),
        "episode_label": bool(episode_label),
        "user_request": norm_user,
        "prior": [{k: v for k, v in b.items() if k != "full_text"} for b in prior_out],
        "relevant_tool_blocks": relevant,
        "current": {"kind": kind, "text": current_segment_text[:MAX_CURRENT_CHARS],
                    "segment": current_segment_text[:MAX_CURRENT_CHARS],
                    "visible_so_far": current_visible_so_far[-1200:],
                    "tool": current_tool, "arguments": current_args},
        "value_sources": sources,
        "prefix_chars": len(norm_user),
        "prefix_sha256": sha256_text(current_segment_text),
        "current_sha256": sha256_text(current_segment_text),
        "has_reasoning_before": False,
        "counts": {"prior_blocks": len(prior_out),
                   "tool_blocks": sum(1 for b in prior_out if b["role"] == "tool"),
                   "user_blocks": sum(1 for b in prior_out if b["role"] == "user"),
                   "args": len(values)},
    }
