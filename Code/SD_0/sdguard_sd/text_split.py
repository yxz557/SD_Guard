"""Split Qwen3 generated token IDs into thinking and final-answer regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class CompletionParts:
    thinking: str
    final_answer: str
    full_completion: str
    answer_start_generated_index: int | None
    parse_status: str


def _valid_special_id(tokenizer: Any, token: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        return None
    try:
        value = int(token_id)
    except (TypeError, ValueError):
        return None
    unknown = getattr(tokenizer, "unk_token_id", None)
    if unknown is not None and value == int(unknown):
        return None
    return value


def split_completion(
    tokenizer: Any,
    generated_ids: Sequence[int],
    *,
    thinking_expected: bool = True,
) -> CompletionParts:
    ids = [int(token_id) for token_id in generated_ids]
    if not thinking_expected:
        answer = tokenizer.decode(ids, skip_special_tokens=True).strip()
        return CompletionParts(
            thinking="",
            final_answer=answer,
            full_completion=answer,
            answer_start_generated_index=0,
            parse_status="thinking_disabled",
        )

    close_id = _valid_special_id(tokenizer, "</think>")
    close_index: int | None = None
    if close_id is not None:
        for index in range(len(ids)):
            if ids[index] == close_id:
                close_index = index
                break

    if close_index is None:
        raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
        return CompletionParts(
            thinking=raw,
            final_answer="",
            full_completion=f"<think>\n{raw}",
            answer_start_generated_index=None,
            parse_status="missing_close_think",
        )

    answer_start = close_index + 1

    thinking = tokenizer.decode(ids[:close_index], skip_special_tokens=True).strip()
    answer = tokenizer.decode(ids[answer_start:], skip_special_tokens=True).strip()
    full = f"<think>\n{thinking}\n</think>\n\n{answer}".strip()
    return CompletionParts(
        thinking=thinking,
        final_answer=answer,
        full_completion=full,
        answer_start_generated_index=answer_start,
        parse_status="ok",
    )


__all__ = ["CompletionParts", "split_completion"]
