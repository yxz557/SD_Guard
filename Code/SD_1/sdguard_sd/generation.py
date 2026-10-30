"""Qwen3 speculative sampling through Transformers' lossless verifier."""

from __future__ import annotations

import copy
import hashlib
import time
import types
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .prompts import build_generation_messages
from .runtime import import_runtime, synchronize_devices


@dataclass(frozen=True, slots=True)
class SpeculativeRound:
    round_index: int
    prefix_length: int
    proposed_ids: tuple[int, ...]
    accepted_count: int
    committed_ids: tuple[int, ...]
    commit_kind: str

    def to_dict(self, tokenizer: Any | None = None) -> dict[str, Any]:
        value = asdict(self)
        value["proposed_ids"] = list(self.proposed_ids)
        value["committed_ids"] = list(self.committed_ids)
        if tokenizer is not None:
            value["proposed_text"] = tokenizer.decode(
                list(self.proposed_ids), skip_special_tokens=False
            )
            value["committed_text"] = tokenizer.decode(
                list(self.committed_ids), skip_special_tokens=False
            )
        return value


@dataclass(frozen=True, slots=True)
class GenerationResult:
    prompt_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    latency_ms: float
    rounds: tuple[SpeculativeRound, ...]
    stopped_on_eos: bool
    truncated: bool

    @property
    def generated_token_count(self) -> int:
        return len(self.generated_ids)

    @property
    def round_count(self) -> int:
        return len(self.rounds)

    @property
    def proposed_token_count(self) -> int:
        return sum(len(round_.proposed_ids) for round_ in self.rounds)

    @property
    def accepted_draft_token_count(self) -> int:
        return sum(round_.accepted_count for round_ in self.rounds)

    @property
    def acceptance_rate(self) -> float | None:
        if self.proposed_token_count == 0:
            return None
        return self.accepted_draft_token_count / self.proposed_token_count

    @property
    def tokens_per_second(self) -> float | None:
        if self.latency_ms <= 0:
            return None
        return self.generated_token_count / (self.latency_ms / 1000.0)


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def seed_everything(seed: int) -> None:
    torch, _, _, _ = import_runtime()
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _TraceSink:
    def __init__(self) -> None:
        self.rounds: list[SpeculativeRound] = []
        self.pending: dict[str, Any] | None = None


def _install_candidate_trace(target_model: Any, draft_model: Any, sink: _TraceSink) -> Any:
    """Instrument the stock candidate instance without changing its concrete type."""

    try:
        from transformers.generation.candidate_generator import AssistedCandidateGenerator
    except ImportError as exc:  # pragma: no cover - version guard runs on host
        raise RuntimeError("Transformers lacks AssistedCandidateGenerator") from exc

    original = target_model._get_candidate_generator

    def traced_factory(
        model_self: Any,
        generation_config: Any,
        input_ids: Any,
        inputs_tensor: Any,
        logits_processor: Any,
        model_kwargs: dict[str, Any],
        assistant_model: Any = None,
        target_tokenizer: Any = None,
        assistant_tokenizer: Any = None,
    ) -> Any:
        candidate_generator = original(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            assistant_model=assistant_model,
            target_tokenizer=target_tokenizer,
            assistant_tokenizer=assistant_tokenizer,
        )
        if assistant_model is not draft_model or type(candidate_generator) is not AssistedCandidateGenerator:
            return candidate_generator

        original_get_candidates = candidate_generator.get_candidates
        original_update_strategy = candidate_generator.update_candidate_strategy

        def traced_get_candidates(candidate_self: Any, current_ids: Any) -> tuple[Any, Any]:
            prefix_length = int(current_ids.shape[1])
            candidate_ids, candidate_logits = original_get_candidates(current_ids)
            proposed = tuple(
                int(value)
                for value in candidate_ids[0, prefix_length:].detach().cpu().tolist()
            )
            sink.pending = {
                "round_index": len(sink.rounds),
                "prefix_length": prefix_length,
                "proposed_ids": proposed,
            }
            return candidate_ids, candidate_logits

        def traced_update_strategy(
            candidate_self: Any,
            current_ids: Any,
            scores: Any,
            num_matches: Any,
        ) -> None:
            pending = sink.pending
            if pending is None:
                raise RuntimeError("Candidate trace lost the current speculative round")
            raw_matches = int(num_matches.item() if hasattr(num_matches, "item") else num_matches)
            committed = tuple(
                int(value)
                for value in current_ids[0, pending["prefix_length"] :].detach().cpu().tolist()
            )
            proposed = pending["proposed_ids"]
            terminal_candidate = bool(
                proposed
                and len(committed) == len(proposed)
                and committed == proposed
                and raw_matches == len(proposed) - 1
            )
            accepted = len(proposed) if terminal_candidate else min(raw_matches, len(proposed))
            if terminal_candidate:
                kind = "terminal_candidate"
            elif accepted == len(proposed):
                kind = "target_bonus"
            else:
                kind = "target_correction"
            sink.rounds.append(
                SpeculativeRound(
                    round_index=int(pending["round_index"]),
                    prefix_length=int(pending["prefix_length"]),
                    proposed_ids=tuple(proposed),
                    accepted_count=int(accepted),
                    committed_ids=committed,
                    commit_kind=kind,
                )
            )
            sink.pending = None
            original_update_strategy(current_ids, scores, num_matches)

        candidate_generator.get_candidates = types.MethodType(
            traced_get_candidates, candidate_generator
        )
        candidate_generator.update_candidate_strategy = types.MethodType(
            traced_update_strategy, candidate_generator
        )
        return candidate_generator

    target_model._get_candidate_generator = types.MethodType(traced_factory, target_model)
    return original


def prepare_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    messages = build_generation_messages(prompt, system_prompt=system_prompt)
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
        return_dict=True,
    )
    if not isinstance(encoded, dict) and not hasattr(encoded, "items"):
        raise RuntimeError("Tokenizer did not return a mapping for the generation prompt")
    return dict(encoded)


def prompt_token_ids(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool,
    system_prompt: str | None = None,
) -> tuple[int, ...]:
    """Tokenize a chat prompt without moving it to a model device."""

    encoded = prepare_prompt(
        tokenizer,
        prompt,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )
    return tuple(int(value) for value in encoded["input_ids"][0].detach().cpu().tolist())


def generate_speculatively(
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    target_device: str,
    draft_device: str,
    sample_seed: int,
    enable_thinking: bool,
    max_new_tokens: int,
    num_assistant_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    system_prompt: str | None = None,
) -> GenerationResult:
    torch, _, _, _ = import_runtime()
    if max_new_tokens <= 0 or num_assistant_tokens <= 0:
        raise ValueError("Token limits must be positive")

    seed_everything(sample_seed)
    inputs = prepare_prompt(
        tokenizer,
        prompt,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )
    inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_ids = tuple(int(value) for value in inputs["input_ids"][0].detach().cpu().tolist())

    config = copy.deepcopy(target_model.generation_config)
    config.max_new_tokens = max_new_tokens
    config.do_sample = do_sample
    config.num_beams = 1
    config.use_cache = True
    if do_sample:
        config.temperature = temperature
        config.top_p = top_p
        config.top_k = top_k
        if hasattr(config, "min_p"):
            config.min_p = None
    draft_model.generation_config.num_assistant_tokens = float(num_assistant_tokens)
    draft_model.generation_config.num_assistant_tokens_schedule = "constant"
    draft_model.generation_config.assistant_confidence_threshold = 0.0

    sink = _TraceSink()
    original_factory = _install_candidate_trace(target_model, draft_model, sink)
    try:
        synchronize_devices(target_device, draft_device)
        started_ns = time.perf_counter_ns()
        with torch.inference_mode():
            output = target_model.generate(
                **inputs,
                generation_config=config,
                assistant_model=draft_model,
                return_dict_in_generate=True,
                output_scores=False,
            )
        synchronize_devices(target_device, draft_device)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    finally:
        target_model._get_candidate_generator = original_factory

    sequence = output.sequences[0].detach().cpu().tolist()
    generated = tuple(int(value) for value in sequence[len(prompt_ids) :])
    eos_values = config.eos_token_id
    if eos_values is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_values, (list, tuple, set)):
        eos_ids = {int(value) for value in eos_values}
    else:
        eos_ids = {int(eos_values)}
    stopped_on_eos = bool(generated and generated[-1] in eos_ids)
    truncated = len(generated) >= max_new_tokens and not stopped_on_eos
    return GenerationResult(
        prompt_ids=prompt_ids,
        generated_ids=generated,
        latency_ms=elapsed_ms,
        rounds=tuple(sink.rounds),
        stopped_on_eos=stopped_on_eos,
        truncated=truncated,
    )


def generate_target_only_greedy(
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    target_device: str,
    enable_thinking: bool,
    max_new_tokens: int,
    system_prompt: str | None = None,
) -> tuple[int, ...]:
    torch, _, _, _ = import_runtime()
    inputs = prepare_prompt(
        tokenizer,
        prompt,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )
    inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_length = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output = target_model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return tuple(int(value) for value in output[0, prompt_length:].detach().cpu().tolist())


def generate_assisted_greedy_untraced(
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    target_device: str,
    enable_thinking: bool,
    max_new_tokens: int,
    num_assistant_tokens: int,
    system_prompt: str | None = None,
) -> tuple[int, ...]:
    """Run the stock Transformers assisted-greedy path without our tracer."""

    torch, _, _, _ = import_runtime()
    inputs = prepare_prompt(
        tokenizer,
        prompt,
        enable_thinking=enable_thinking,
        system_prompt=system_prompt,
    )
    inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_length = int(inputs["input_ids"].shape[1])
    config = copy.deepcopy(target_model.generation_config)
    config.max_new_tokens = max_new_tokens
    config.do_sample = False
    config.num_beams = 1
    config.use_cache = True
    draft_model.generation_config.num_assistant_tokens = float(num_assistant_tokens)
    draft_model.generation_config.num_assistant_tokens_schedule = "constant"
    draft_model.generation_config.assistant_confidence_threshold = 0.0
    with torch.inference_mode():
        output = target_model.generate(
            **inputs,
            generation_config=config,
            assistant_model=draft_model,
        )
    return tuple(int(value) for value in output[0, prompt_length:].detach().cpu().tolist())


__all__ = [
    "GenerationResult",
    "SpeculativeRound",
    "generate_assisted_greedy_untraced",
    "generate_speculatively",
    "generate_target_only_greedy",
    "prepare_prompt",
    "prompt_token_ids",
    "seed_everything",
    "stable_sample_seed",
]

