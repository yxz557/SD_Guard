"""One-shot causal Qwen3Guard-Stream replay over a generated completion."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from .policies import evaluate_policies, normalize_label
from .runtime import import_runtime, synchronize_devices


@dataclass(frozen=True, slots=True)
class GuardResult:
    prompt_risk_level: str
    prompt_risk_probability: float
    token_records: tuple[dict[str, Any], ...]
    scopes: dict[str, Any]
    latency_ms: float


def _mapping(model: Any, attribute: str, fallback: dict[int, str]) -> dict[int, str]:
    raw = getattr(model, attribute, None)
    if raw is None:
        raw = getattr(getattr(model, "config", None), attribute, None)
    if raw is None:
        raw = fallback
    return {int(key): str(value) for key, value in raw.items()}


def _decode_head(torch: Any, logits: Any, mapping: dict[int, str]) -> tuple[list[str], list[float]]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    best_probability, best_index = probabilities.max(dim=-1)
    indices = best_index.detach().cpu().reshape(-1).tolist()
    values = best_probability.detach().cpu().reshape(-1).tolist()
    try:
        labels = [mapping[int(index)] for index in indices]
    except KeyError as exc:
        raise RuntimeError(f"Guard produced an unmapped head index: {exc.args[0]!r}") from exc
    return labels, [float(value) for value in values]


def _last_user_end(tokenizer: Any, prompt_ids: Sequence[int]) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    matches = [index for index, value in enumerate(prompt_ids) if int(value) == int(token_id)]
    if not matches:
        raise RuntimeError("Could not find the user <|im_end|> token in the generation prompt")
    return matches[-1]


def run_guard_full_forward(
    guard_model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    generated_ids: Sequence[int],
    *,
    answer_start_generated_index: int | None,
    guard_device: str,
) -> GuardResult:
    torch, _, _, _ = import_runtime()
    full_ids = [int(value) for value in prompt_ids] + [int(value) for value in generated_ids]
    max_positions = int(getattr(guard_model.config, "max_position_embeddings", len(full_ids)))
    if len(full_ids) > max_positions:
        raise RuntimeError(
            f"Guard input length {len(full_ids)} exceeds max_position_embeddings={max_positions}; "
            "reduce --max-new-tokens rather than silently truncating the safety context"
        )
    prompt_length = len(prompt_ids)
    user_end = _last_user_end(tokenizer, prompt_ids)
    input_ids = torch.tensor(full_ids, dtype=torch.long, device=guard_device).unsqueeze(0)
    sequence_length = len(full_ids)
    causal_mask = torch.tril(
        torch.ones(
            (sequence_length, sequence_length),
            dtype=torch.bool,
            device=guard_device,
        )
    ).unsqueeze(0).unsqueeze(0)

    synchronize_devices(guard_device)
    started_ns = time.perf_counter_ns()
    with torch.inference_mode():
        outputs = guard_model.forward(input_ids=input_ids, attention_mask=causal_mask)
    synchronize_devices(guard_device)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0

    response_risk_map = _mapping(
        guard_model,
        "response_risk_level_map",
        {0: "Safe", 1: "Unsafe", 2: "Controversial"},
    )
    response_category_map = _mapping(guard_model, "response_category_map", {})
    query_risk_map = _mapping(
        guard_model,
        "query_risk_level_map",
        {0: "Safe", 1: "Unsafe", 2: "Controversial"},
    )

    response_risk, response_risk_prob = _decode_head(
        torch,
        outputs.risk_level_logits[0, prompt_length:, :],
        response_risk_map,
    )
    response_category, response_category_prob = _decode_head(
        torch,
        outputs.category_logits[0, prompt_length:, :],
        response_category_map,
    )
    prompt_risk, prompt_risk_prob = _decode_head(
        torch,
        outputs.query_risk_level_logits[0, user_end : user_end + 1, :],
        query_risk_map,
    )

    token_records: list[dict[str, Any]] = []
    for index, token_id in enumerate(generated_ids):
        token_records.append(
            {
                "generated_token_index": index,
                "full_sequence_index": prompt_length + index,
                "token_id": int(token_id),
                "token_text": tokenizer.decode([int(token_id)], skip_special_tokens=False),
                "risk_level": normalize_label(response_risk[index]),
                "risk_probability": response_risk_prob[index],
                "category": response_category[index],
                "category_probability": response_category_prob[index],
                "in_answer_scope": (
                    answer_start_generated_index is not None
                    and index >= answer_start_generated_index
                ),
            }
        )

    full_labels = [record["risk_level"] for record in token_records]
    full_decisions = evaluate_policies(full_labels, generated_index_offset=0)
    for decision in full_decisions.values():
        trigger = decision["trigger_generated_index"]
        decision["trigger_token_text"] = (
            None if trigger is None else token_records[int(trigger)]["token_text"]
        )
        decision["trigger_category"] = (
            None if trigger is None else token_records[int(trigger)]["category"]
        )

    if answer_start_generated_index is None:
        answer_scope: dict[str, Any] = {
            "available": False,
            "start_generated_index": None,
            "token_count": 0,
            "decisions": None,
            "unavailable_reason": "missing_close_think",
        }
    else:
        answer_labels = full_labels[answer_start_generated_index:]
        answer_decisions = evaluate_policies(
            answer_labels,
            generated_index_offset=answer_start_generated_index,
        )
        for decision in answer_decisions.values():
            trigger = decision["trigger_generated_index"]
            decision["trigger_token_text"] = (
                None if trigger is None else token_records[int(trigger)]["token_text"]
            )
            decision["trigger_category"] = (
                None if trigger is None else token_records[int(trigger)]["category"]
            )
        answer_scope = {
            "available": True,
            "start_generated_index": answer_start_generated_index,
            "token_count": len(answer_labels),
            "decisions": answer_decisions,
            "unavailable_reason": None,
        }

    return GuardResult(
        prompt_risk_level=normalize_label(prompt_risk[-1]),
        prompt_risk_probability=prompt_risk_prob[-1],
        token_records=tuple(token_records),
        scopes={
            "full_trace": {
                "available": True,
                "start_generated_index": 0,
                "token_count": len(full_labels),
                "decisions": full_decisions,
                "unavailable_reason": None,
            },
            "answer_only": answer_scope,
        },
        latency_ms=elapsed_ms,
    )


__all__ = ["GuardResult", "run_guard_full_forward"]
