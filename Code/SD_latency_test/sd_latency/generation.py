"""SD_0 generation strategy with optional round tracing and phase timing."""

from __future__ import annotations

import copy
import random
import time
import types
from dataclasses import asdict, dataclass
from typing import Any

from .runtime import import_runtime, synchronize_devices


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_new_tokens: int = 96
    num_assistant_tokens: int = 5
    assistant_schedule: str = "constant"
    assistant_confidence_threshold: float = 0.4
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: bool = True


@dataclass(frozen=True, slots=True)
class SpeculativeRound:
    round_index: int
    prefix_length: int
    proposed_ids: tuple[int, ...]
    accepted_count: int
    committed_ids: tuple[int, ...]
    commit_kind: str
    draft_proposal_ms: float
    proposed_trace_ms: float
    verify_and_decide_ms: float
    target_forward_cuda_ms: float | None
    result_trace_ms: float
    strategy_update_ms: float

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
    preparation_ms: float
    generate_ms: float
    output_transfer_ms: float
    total_ms: float
    rounds: tuple[SpeculativeRound, ...]
    traced: bool
    stopped_on_eos: bool
    truncated: bool

    @property
    def generated_token_count(self) -> int:
        return len(self.generated_ids)

    @property
    def tokens_per_second(self) -> float | None:
        if self.total_ms <= 0:
            return None
        return self.generated_token_count / (self.total_ms / 1000.0)

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

    def phase_summary(self) -> dict[str, float | int | None]:
        draft = sum(round_.draft_proposal_ms for round_ in self.rounds)
        proposed_trace = sum(round_.proposed_trace_ms for round_ in self.rounds)
        verify = sum(round_.verify_and_decide_ms for round_ in self.rounds)
        target_cuda_values = [
            round_.target_forward_cuda_ms
            for round_ in self.rounds
            if round_.target_forward_cuda_ms is not None
        ]
        target_cuda = sum(target_cuda_values) if target_cuda_values else None
        result_trace = sum(round_.result_trace_ms for round_ in self.rounds)
        strategy = sum(round_.strategy_update_ms for round_ in self.rounds)
        accounted = draft + proposed_trace + verify + result_trace + strategy
        return {
            "round_count": len(self.rounds),
            "draft_proposal_ms": draft,
            "proposed_trace_ms": proposed_trace,
            "verify_and_decide_ms": verify,
            "target_forward_cuda_ms": target_cuda,
            "result_trace_ms": result_trace,
            "strategy_update_ms": strategy,
            "profiled_round_phases_ms": accounted,
            "generate_framework_other_estimate_ms": self.generate_ms - accounted,
        }


def seed_everything(seed: int) -> None:
    torch, _, _ = import_runtime()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_prompt(tokenizer: Any, prompt: str, *, enable_thinking: bool) -> dict[str, Any]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_tensors="pt",
        return_dict=True,
    )
    if not isinstance(encoded, dict) and not hasattr(encoded, "items"):
        raise RuntimeError("Tokenizer did not return a mapping for the generation prompt")
    return dict(encoded)


def _generation_config(target_model: Any, settings: GenerationSettings) -> Any:
    config = copy.deepcopy(target_model.generation_config)
    config.max_new_tokens = settings.max_new_tokens
    config.do_sample = settings.do_sample
    config.num_beams = 1
    config.use_cache = True
    # Keep the assisted-generation controls explicit on the config copied into
    # the internal draft generate() call as well as on the draft model itself.
    config.num_assistant_tokens = float(settings.num_assistant_tokens)
    config.num_assistant_tokens_schedule = settings.assistant_schedule
    config.assistant_confidence_threshold = settings.assistant_confidence_threshold
    if settings.do_sample:
        config.temperature = settings.temperature
        config.top_p = settings.top_p
        config.top_k = settings.top_k
        if hasattr(config, "min_p"):
            config.min_p = None
    return config


def _generation_overrides(settings: GenerationSettings) -> dict[str, Any]:
    """Return decoding kwargs that must win over model-specific defaults."""

    overrides: dict[str, Any] = {"do_sample": settings.do_sample}
    if settings.do_sample:
        overrides.update(
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
        )
        # Qwen generation configs may define min_p. Explicitly disable it so
        # sampling is controlled only by this harness's documented settings.
        overrides["min_p"] = None
    return overrides


def _configure_draft(draft_model: Any, settings: GenerationSettings) -> None:
    # AssistedCandidateGenerator invokes assistant_model.generate() internally.
    # Match the draft model's own defaults to the requested mode as well, or a
    # model-specific do_sample=True default can turn a greedy draft into sampling.
    draft_model.generation_config.do_sample = settings.do_sample
    if settings.do_sample:
        draft_model.generation_config.temperature = settings.temperature
        draft_model.generation_config.top_p = settings.top_p
        draft_model.generation_config.top_k = settings.top_k
        if hasattr(draft_model.generation_config, "min_p"):
            draft_model.generation_config.min_p = None
    draft_model.generation_config.num_assistant_tokens = float(
        settings.num_assistant_tokens
    )
    draft_model.generation_config.num_assistant_tokens_schedule = (
        settings.assistant_schedule
    )
    draft_model.generation_config.assistant_confidence_threshold = (
        settings.assistant_confidence_threshold
    )


def _finish_result(
    *,
    output: Any,
    prompt_ids: tuple[int, ...],
    config: Any,
    preparation_ms: float,
    generate_ms: float,
    overall_started_ns: int,
    rounds: tuple[SpeculativeRound, ...],
    traced: bool,
) -> GenerationResult:
    transfer_started_ns = time.perf_counter_ns()
    sequence = output.sequences[0].detach().cpu().tolist()
    output_transfer_ms = (time.perf_counter_ns() - transfer_started_ns) / 1_000_000.0
    generated = tuple(int(value) for value in sequence[len(prompt_ids) :])
    eos_values = config.eos_token_id
    if eos_values is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_values, (list, tuple, set)):
        eos_ids = {int(value) for value in eos_values}
    else:
        eos_ids = {int(eos_values)}
    stopped_on_eos = bool(generated and generated[-1] in eos_ids)
    return GenerationResult(
        prompt_ids=prompt_ids,
        generated_ids=generated,
        preparation_ms=preparation_ms,
        generate_ms=generate_ms,
        output_transfer_ms=output_transfer_ms,
        total_ms=(time.perf_counter_ns() - overall_started_ns) / 1_000_000.0,
        rounds=rounds,
        traced=traced,
        stopped_on_eos=stopped_on_eos,
        truncated=len(generated) >= config.max_new_tokens and not stopped_on_eos,
    )


class _TraceSink:
    def __init__(self, *, target_device: str, draft_device: str) -> None:
        self.target_device = target_device
        self.draft_device = draft_device
        self.rounds: list[SpeculativeRound] = []
        self.pending: dict[str, Any] | None = None


def _install_trace(target_model: Any, draft_model: Any, sink: _TraceSink) -> tuple[Any, Any]:
    """Trace the stock HF verifier; timing syncs are intentional and reported."""

    try:
        from transformers.generation.candidate_generator import AssistedCandidateGenerator
    except ImportError as exc:  # pragma: no cover - protected by the version check
        raise RuntimeError("Transformers lacks AssistedCandidateGenerator") from exc

    torch, _, _ = import_runtime()
    original_factory = target_model._get_candidate_generator
    original_forward = target_model.forward

    def traced_forward(model_self: Any, *args: Any, **kwargs: Any) -> Any:
        pending = sink.pending
        event_pair: tuple[Any, Any] | None = None
        if pending is not None and torch.cuda.is_available():
            with torch.cuda.device(sink.target_device):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            event_pair = (start_event, end_event)
        output = original_forward(*args, **kwargs)
        if event_pair is not None:
            with torch.cuda.device(sink.target_device):
                event_pair[1].record()
            pending["target_forward_events"].append(event_pair)
        return output

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
        candidate_generator = original_factory(
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
            draft_started_ns = time.perf_counter_ns()
            candidate_ids, candidate_logits = original_get_candidates(current_ids)
            synchronize_devices(sink.draft_device)
            draft_proposal_ms = (time.perf_counter_ns() - draft_started_ns) / 1_000_000.0

            trace_started_ns = time.perf_counter_ns()
            proposed = tuple(
                int(value)
                for value in candidate_ids[0, prefix_length:].detach().cpu().tolist()
            )
            proposed_trace_ms = (time.perf_counter_ns() - trace_started_ns) / 1_000_000.0
            sink.pending = {
                "round_index": len(sink.rounds),
                "prefix_length": prefix_length,
                "proposed_ids": proposed,
                "draft_proposal_ms": draft_proposal_ms,
                "proposed_trace_ms": proposed_trace_ms,
                "verify_started_ns": time.perf_counter_ns(),
                "target_forward_events": [],
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

            synchronize_devices(sink.target_device, sink.draft_device)
            verify_and_decide_ms = (
                time.perf_counter_ns() - int(pending["verify_started_ns"])
            ) / 1_000_000.0
            event_pairs = pending["target_forward_events"]
            target_forward_cuda_ms = (
                sum(float(start.elapsed_time(end)) for start, end in event_pairs)
                if event_pairs
                else None
            )

            trace_started_ns = time.perf_counter_ns()
            raw_matches = int(num_matches.item() if hasattr(num_matches, "item") else num_matches)
            committed = tuple(
                int(value)
                for value in current_ids[0, pending["prefix_length"] :].detach().cpu().tolist()
            )
            result_trace_ms = (time.perf_counter_ns() - trace_started_ns) / 1_000_000.0
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

            strategy_started_ns = time.perf_counter_ns()
            original_update_strategy(current_ids, scores, num_matches)
            strategy_update_ms = (time.perf_counter_ns() - strategy_started_ns) / 1_000_000.0
            sink.rounds.append(
                SpeculativeRound(
                    round_index=int(pending["round_index"]),
                    prefix_length=int(pending["prefix_length"]),
                    proposed_ids=tuple(proposed),
                    accepted_count=int(accepted),
                    committed_ids=committed,
                    commit_kind=kind,
                    draft_proposal_ms=float(pending["draft_proposal_ms"]),
                    proposed_trace_ms=float(pending["proposed_trace_ms"]),
                    verify_and_decide_ms=verify_and_decide_ms,
                    target_forward_cuda_ms=target_forward_cuda_ms,
                    result_trace_ms=result_trace_ms,
                    strategy_update_ms=strategy_update_ms,
                )
            )
            sink.pending = None

        candidate_generator.get_candidates = types.MethodType(
            traced_get_candidates, candidate_generator
        )
        candidate_generator.update_candidate_strategy = types.MethodType(
            traced_update_strategy, candidate_generator
        )
        return candidate_generator

    target_model._get_candidate_generator = types.MethodType(traced_factory, target_model)
    target_model.forward = types.MethodType(traced_forward, target_model)
    return original_factory, original_forward


def generate_target_only(
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: str,
    settings: GenerationSettings,
    seed: int,
) -> GenerationResult:
    torch, _, _ = import_runtime()
    seed_everything(seed)
    overall_started_ns = time.perf_counter_ns()
    preparation_started_ns = overall_started_ns
    inputs = prepare_prompt(tokenizer, prompt, enable_thinking=settings.enable_thinking)
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    synchronize_devices(device)
    preparation_ms = (time.perf_counter_ns() - preparation_started_ns) / 1_000_000.0
    prompt_ids = tuple(int(value) for value in inputs["input_ids"][0].detach().cpu().tolist())
    config = _generation_config(target_model, settings)
    synchronize_devices(device)
    generate_started_ns = time.perf_counter_ns()
    with torch.inference_mode():
        output = target_model.generate(
            **inputs,
            generation_config=config,
            use_model_defaults=False,
            return_dict_in_generate=True,
            output_scores=False,
            **_generation_overrides(settings),
        )
    synchronize_devices(device)
    generate_ms = (time.perf_counter_ns() - generate_started_ns) / 1_000_000.0
    return _finish_result(
        output=output,
        prompt_ids=prompt_ids,
        config=config,
        preparation_ms=preparation_ms,
        generate_ms=generate_ms,
        overall_started_ns=overall_started_ns,
        rounds=(),
        traced=False,
    )


def generate_assisted(
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    target_device: str,
    draft_device: str,
    settings: GenerationSettings,
    seed: int,
    trace: bool,
) -> GenerationResult:
    torch, _, _ = import_runtime()
    if settings.num_assistant_tokens <= 0 or settings.max_new_tokens <= 0:
        raise ValueError("Token limits must be positive")
    seed_everything(seed)
    overall_started_ns = time.perf_counter_ns()
    preparation_started_ns = overall_started_ns
    inputs = prepare_prompt(tokenizer, prompt, enable_thinking=settings.enable_thinking)
    inputs = {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    synchronize_devices(target_device, draft_device)
    preparation_ms = (time.perf_counter_ns() - preparation_started_ns) / 1_000_000.0
    prompt_ids = tuple(int(value) for value in inputs["input_ids"][0].detach().cpu().tolist())
    config = _generation_config(target_model, settings)
    _configure_draft(draft_model, settings)

    sink = _TraceSink(target_device=target_device, draft_device=draft_device)
    original_factory: Any | None = None
    original_forward: Any | None = None
    if trace:
        original_factory, original_forward = _install_trace(target_model, draft_model, sink)
    try:
        synchronize_devices(target_device, draft_device)
        generate_started_ns = time.perf_counter_ns()
        with torch.inference_mode():
            output = target_model.generate(
                **inputs,
                generation_config=config,
                assistant_model=draft_model,
                use_model_defaults=False,
                return_dict_in_generate=True,
                output_scores=False,
                **_generation_overrides(settings),
            )
        synchronize_devices(target_device, draft_device)
        generate_ms = (time.perf_counter_ns() - generate_started_ns) / 1_000_000.0
    finally:
        if trace:
            target_model._get_candidate_generator = original_factory
            target_model.forward = original_forward

    return _finish_result(
        output=output,
        prompt_ids=prompt_ids,
        config=config,
        preparation_ms=preparation_ms,
        generate_ms=generate_ms,
        overall_started_ns=overall_started_ns,
        rounds=tuple(sink.rounds),
        traced=trace,
    )
