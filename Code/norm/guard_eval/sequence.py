"""Chat-template construction and token-boundary discovery.

The released ``eval_stream.py`` first renders the Qwen chat template to text
and then tokenizes that text.  This module intentionally uses the same two-step
path and the same response-start rules so location annotations keep their
published token coordinate system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dataset import VALID_SPLITS


class SequenceBoundaryError(ValueError):
    """Raised when a required Qwen chat-template boundary cannot be located."""


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Comparison between freshly tokenized IDs and dataset annotation IDs."""

    checked: bool
    matches: bool | None
    generated_length: int
    annotated_length: int | None
    first_mismatch_index: int | None = None
    generated_token_id: int | None = None
    annotated_token_id: int | None = None

    @property
    def usable_for_location_metrics(self) -> bool:
        return self.checked and self.matches is True


@dataclass(frozen=True, slots=True)
class PreparedSequence:
    """One fully templated sample and its inclusive/exclusive boundaries.

    ``user_end_index`` is inclusive and points at the user's ``<|im_end|>``.
    ``eval_start_index`` is also inclusive and matches ``split_idx_eval`` in
    the official script. Tokens between those indices are template/control
    tokens: the streaming runner feeds them into the cache but does not time
    them as Assistant response tokens.
    """

    split: str
    rendered_text: str
    input_ids: tuple[int, ...]
    decoded_tokens: tuple[str, ...]
    user_start_index: int
    user_end_index: int
    eval_start_index: int
    alignment: AlignmentReport

    @property
    def prompt_ids(self) -> tuple[int, ...]:
        return self.input_ids[: self.user_end_index + 1]

    @property
    def bridge_ids(self) -> tuple[int, ...]:
        return self.input_ids[self.user_end_index + 1 : self.eval_start_index]

    @property
    def response_ids(self) -> tuple[int, ...]:
        return self.input_ids[self.eval_start_index :]

    @property
    def prompt_token_count(self) -> int:
        return self.user_end_index + 1

    @property
    def bridge_token_count(self) -> int:
        return self.eval_start_index - self.user_end_index - 1

    @property
    def response_token_count(self) -> int:
        return len(self.input_ids) - self.eval_start_index


def split_uses_thinking(split: str) -> bool:
    """Whether the official evaluator starts at the beginning of thinking."""

    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {VALID_SPLITS}.")
    return split in {"thinking", "thinking_loc"}


def _as_flat_ids(encoded: Any) -> tuple[int, ...]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids

    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise SequenceBoundaryError("Expected one unbatched token sequence.")
        encoded = encoded[0]
    try:
        ids = tuple(int(token_id) for token_id in encoded)
    except (TypeError, ValueError) as exc:
        raise SequenceBoundaryError("Tokenizer returned invalid input_ids.") from exc
    if not ids:
        raise SequenceBoundaryError("Chat template produced an empty token sequence.")
    return ids


def render_and_tokenize(
    tokenizer: Any, messages: Sequence[Mapping[str, Any]]
) -> tuple[str, tuple[int, ...]]:
    """Reproduce the official two-stage template/tokenizer invocation."""

    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise SequenceBoundaryError("apply_chat_template did not return text.")
    encoded = tokenizer(rendered, return_tensors=None)
    return rendered, _as_flat_ids(encoded)


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise SequenceBoundaryError(f"Tokenizer has no ID for required token {token!r}.")
    try:
        return int(token_id)
    except (TypeError, ValueError) as exc:
        raise SequenceBoundaryError(
            f"Tokenizer returned an invalid ID for required token {token!r}."
        ) from exc


def find_last_user_turn(
    tokenizer: Any, input_ids: Sequence[int]
) -> tuple[int, int]:
    """Return ``(<|im_start|> index, <|im_end|> index)`` for the last user turn."""

    im_start_id = _token_id(tokenizer, "<|im_start|>")
    user_id = _token_id(tokenizer, "user")
    im_end_id = _token_id(tokenizer, "<|im_end|>")

    user_start = next(
        (
            index
            for index in range(len(input_ids) - 2, -1, -1)
            if input_ids[index] == im_start_id and input_ids[index + 1] == user_id
        ),
        None,
    )
    if user_start is None:
        raise SequenceBoundaryError(
            "Could not find the last '<|im_start|>user' token pair."
        )

    user_end = next(
        (
            index
            for index in range(user_start + 2, len(input_ids))
            if input_ids[index] == im_end_id
        ),
        None,
    )
    if user_end is None:
        raise SequenceBoundaryError("Could not find '<|im_end|>' for the last user turn.")
    return user_start, user_end


def find_eval_start(
    tokenizer: Any,
    input_ids: Sequence[int],
    decoded_tokens: Sequence[str],
    *,
    thinking: bool,
    search_after: int,
) -> int:
    """Locate the response evaluation start using official ``eval_stream.py`` rules."""

    if thinking:
        im_start_id = _token_id(tokenizer, "<|im_start|>")
        assistant_id = _token_id(tokenizer, "assistant")
        assistant_role_index = next(
            (
                index
                for index in range(max(1, search_after + 1), len(input_ids))
                if input_ids[index - 1] == im_start_id
                and input_ids[index] == assistant_id
            ),
            None,
        )
        if assistant_role_index is None:
            raise SequenceBoundaryError(
                "Could not find the Assistant turn after the last user turn."
            )
        # This is the released evaluator's exact assistant_index + 4 rule. It
        # skips `assistant`, the following newline, and the opening think marker.
        eval_start = assistant_role_index + 4
    else:
        marker_index = next(
            (
                index
                for index in range(max(1, search_after + 1), len(decoded_tokens))
                if decoded_tokens[index - 1] == "</think>"
                and decoded_tokens[index] == "\n\n"
            ),
            None,
        )
        if marker_index is None:
            raise SequenceBoundaryError(
                "Could not find the official '</think>' + '\\n\\n' response boundary. "
                "Check that response_loc was templated with enable_thinking=False."
            )
        eval_start = marker_index + 1

    if not (search_after < eval_start < len(input_ids)):
        raise SequenceBoundaryError(
            f"Invalid response boundary {eval_start} for sequence length {len(input_ids)}."
        )
    return eval_start


def compare_input_ids(
    generated: Sequence[int], annotated: Sequence[int] | None
) -> AlignmentReport:
    """Build a diagnostic alignment report without silently accepting shifts."""

    if annotated is None:
        return AlignmentReport(
            checked=False,
            matches=None,
            generated_length=len(generated),
            annotated_length=None,
        )

    annotated_ids = tuple(int(token_id) for token_id in annotated)
    first_mismatch: int | None = None
    generated_token: int | None = None
    annotated_token: int | None = None
    for index, (left, right) in enumerate(zip(generated, annotated_ids)):
        if left != right:
            first_mismatch = index
            generated_token = int(left)
            annotated_token = int(right)
            break
    if first_mismatch is None and len(generated) != len(annotated_ids):
        first_mismatch = min(len(generated), len(annotated_ids))
        if first_mismatch < len(generated):
            generated_token = int(generated[first_mismatch])
        if first_mismatch < len(annotated_ids):
            annotated_token = int(annotated_ids[first_mismatch])

    return AlignmentReport(
        checked=True,
        matches=first_mismatch is None,
        generated_length=len(generated),
        annotated_length=len(annotated_ids),
        first_mismatch_index=first_mismatch,
        generated_token_id=generated_token,
        annotated_token_id=annotated_token,
    )


def prepare_sequence(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    split: str,
    *,
    annotated_input_ids: Sequence[int] | None = None,
    require_alignment: bool = False,
) -> PreparedSequence:
    """Template, tokenize, find boundaries, and optionally verify annotation IDs."""

    rendered, input_ids = render_and_tokenize(tokenizer, messages)
    decoded_tokens = tuple(
        tokenizer.decode([token_id], skip_special_tokens=False) for token_id in input_ids
    )
    user_start, user_end = find_last_user_turn(tokenizer, input_ids)
    eval_start = find_eval_start(
        tokenizer,
        input_ids,
        decoded_tokens,
        thinking=split_uses_thinking(split),
        search_after=user_end,
    )
    alignment = compare_input_ids(input_ids, annotated_input_ids)
    if require_alignment and alignment.matches is not True:
        if not alignment.checked:
            detail = "the sample has no annotated input_ids"
        else:
            detail = (
                f"first mismatch at index {alignment.first_mismatch_index}; "
                f"generated length={alignment.generated_length}, "
                f"annotated length={alignment.annotated_length}"
            )
        raise SequenceBoundaryError(f"Token alignment is required, but {detail}.")

    return PreparedSequence(
        split=split,
        rendered_text=rendered,
        input_ids=input_ids,
        decoded_tokens=decoded_tokens,
        user_start_index=user_start,
        user_end_index=user_end,
        eval_start_index=eval_start,
        alignment=alignment,
    )


__all__ = [
    "AlignmentReport",
    "PreparedSequence",
    "SequenceBoundaryError",
    "compare_input_ids",
    "find_eval_start",
    "find_last_user_turn",
    "prepare_sequence",
    "render_and_tokenize",
    "split_uses_thinking",
]
