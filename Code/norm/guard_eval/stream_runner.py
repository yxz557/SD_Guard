"""True incremental Qwen3Guard-Stream execution and latency measurement."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from .model_loader import model_cuda_devices, unwrap_model
from .policies import OnlinePolicyTracker, normalize_risk_label
from .sequence import PreparedSequence


OnlineUpdate = Callable[[str, int], Any]


@dataclass(frozen=True, slots=True)
class PromptModerationRecord:
    """Prompt-head output from the initial cached prefill."""

    model: str | None
    split: str
    sample_id: str
    prompt_token_count: int
    user_end_index: int
    risk_level: str
    risk_prob: float
    prompt_prefill_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TokenModerationRecord:
    """One measured Assistant token moderation call."""

    model: str | None
    split: str
    sample_id: str
    assistant_token_index: int
    full_sequence_index: int
    token_id: int
    context_length: int
    risk_level: str
    risk_prob: float
    latency_ms: float
    online_newly_triggered: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["online_newly_triggered"] = list(self.online_newly_triggered)
        return value


@dataclass(frozen=True, slots=True)
class StreamSampleResult:
    """All raw streaming outputs for one benchmark sample."""

    model: str | None
    split: str
    sample_id: str
    prompt: PromptModerationRecord
    token_records: tuple[TokenModerationRecord, ...]
    online_decisions: Mapping[str, Mapping[str, Any]]
    total_sequence_token_count: int
    prompt_token_count: int
    bridge_token_count: int
    response_token_count: int
    available_response_token_count: int
    truncated: bool

    @property
    def risk_levels(self) -> tuple[str, ...]:
        return tuple(record.risk_level for record in self.token_records)

    @property
    def mean_context_length(self) -> float | None:
        if not self.token_records:
            return None
        return sum(record.context_length for record in self.token_records) / len(
            self.token_records
        )

    @property
    def mean_token_latency_ms(self) -> float | None:
        if not self.token_records:
            return None
        return sum(record.latency_ms for record in self.token_records) / len(
            self.token_records
        )

    def to_sample_dict(self) -> dict[str, Any]:
        """Return the compact sample-level record (without repeated token rows)."""

        return {
            "model": self.model,
            "split": self.split,
            "sample_id": self.sample_id,
            "total_sequence_token_count": self.total_sequence_token_count,
            "prompt_token_count": self.prompt_token_count,
            "bridge_token_count": self.bridge_token_count,
            "response_token_count": self.response_token_count,
            "available_response_token_count": self.available_response_token_count,
            "truncated": self.truncated,
            "mean_context_length": self.mean_context_length,
            "mean_token_latency_ms": self.mean_token_latency_ms,
            "online_decisions": {
                name: dict(decision) for name, decision in self.online_decisions.items()
            },
        }


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("Streaming Qwen3Guard evaluation requires PyTorch.") from exc
    return torch


def _latest_result(result: Mapping[str, Any]) -> tuple[str, float]:
    try:
        levels = result["risk_level"]
        probabilities = result["risk_prob"]
        level = levels if isinstance(levels, str) else levels[-1]
        probability = probabilities if isinstance(probabilities, (int, float)) else probabilities[-1]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Unexpected stream_moderate_from_ids result; expected non-empty "
            "'risk_level' and 'risk_prob' fields."
        ) from exc
    return normalize_risk_label(level), float(probability)


def _sync_function(model: Any, torch: Any) -> Callable[[], None]:
    """Resolve devices once so bookkeeping does not pollute every measurement."""

    devices = model_cuda_devices(model)
    if not devices:
        return lambda: None

    def synchronize() -> None:
        for device in devices:
            torch.cuda.synchronize(device)

    return synchronize


def _close_stream(model: Any, stream_state: Any) -> None:
    if stream_state is None:
        return
    close = getattr(model, "close_stream", None)
    if callable(close):
        close(stream_state)
    else:  # Defensive fallback for early public checkpoint revisions.
        stream_state.close()


def run_stream_sample(
    model_or_loaded: Any,
    sequence: PreparedSequence,
    sample_id: str,
    *,
    model_name: str | None = None,
    online_update: OnlineUpdate | None = None,
    max_response_tokens: int | None = None,
) -> StreamSampleResult:
    """Run one sample through the official incremental API.

    Measurement order for every evaluated Assistant token is exactly::

        CUDA sync -> timer start -> stream_moderate_from_ids -> CUDA sync
        -> online policy update (and optional callback) -> timer end

    The scalar token tensor is prepared before the first synchronization. The
    prompt prefill is measured separately, while bridge/control tokens are fed
    one-by-one without response-latency records.
    """

    if max_response_tokens is not None and max_response_tokens < 0:
        raise ValueError("max_response_tokens must be non-negative or None.")

    torch = _import_torch()
    model = unwrap_model(model_or_loaded)
    synchronize = _sync_function(model, torch)
    tracker = OnlinePolicyTracker()
    stream_state: Any = None

    # The downloaded benchmark is a CPU-side replay. Keeping token IDs on CPU
    # matches the official example and includes the Guard API's device transfer
    # in the end-to-end overhead measurement.
    prompt_ids = torch.tensor(sequence.prompt_ids, dtype=torch.long, device="cpu")
    try:
        synchronize()
        prompt_start_ns = time.perf_counter_ns()
        prompt_result, stream_state = model.stream_moderate_from_ids(
            prompt_ids, role="user", stream_state=None
        )
        synchronize()
        prompt_risk, prompt_probability = _latest_result(prompt_result)
        prompt_end_ns = time.perf_counter_ns()
        prompt_record = PromptModerationRecord(
            model=model_name,
            split=sequence.split,
            sample_id=str(sample_id),
            prompt_token_count=sequence.prompt_token_count,
            user_end_index=sequence.user_end_index,
            risk_level=prompt_risk,
            risk_prob=prompt_probability,
            prompt_prefill_latency_ms=(prompt_end_ns - prompt_start_ns) / 1_000_000.0,
        )

        # Preserve the exact causal context while excluding fixed template and
        # control tokens from the response latency distribution.
        for token_id in sequence.bridge_ids:
            bridge_tensor = torch.tensor(token_id, dtype=torch.long, device="cpu")
            _, stream_state = model.stream_moderate_from_ids(
                bridge_tensor, role="assistant", stream_state=stream_state
            )

        available = sequence.response_token_count
        count = available if max_response_tokens is None else min(available, max_response_tokens)
        token_records: list[TokenModerationRecord] = []
        for assistant_index in range(count):
            full_index = sequence.eval_start_index + assistant_index
            token_id = int(sequence.input_ids[full_index])
            # Tensor construction and Python sequence indexing are intentionally
            # outside the measured Guard-call interval.
            current_token = torch.tensor(token_id, dtype=torch.long, device="cpu")

            synchronize()
            start_ns = time.perf_counter_ns()
            result, stream_state = model.stream_moderate_from_ids(
                current_token, role="assistant", stream_state=stream_state
            )
            synchronize()
            risk_level, risk_probability = _latest_result(result)
            # The latency endpoint is the moment the Guard label is available.
            # The tiny CPU-only pair-policy update is intentionally excluded.
            end_ns = time.perf_counter_ns()
            online_event = tracker.update(risk_level)
            if online_update is not None:
                online_update(risk_level, assistant_index)

            token_records.append(
                TokenModerationRecord(
                    model=model_name,
                    split=sequence.split,
                    sample_id=str(sample_id),
                    assistant_token_index=assistant_index,
                    full_sequence_index=full_index,
                    token_id=token_id,
                    # This is both the full templated prefix length and the KV
                    # cache length immediately after processing current_token.
                    context_length=full_index + 1,
                    risk_level=risk_level,
                    risk_prob=risk_probability,
                    latency_ms=(end_ns - start_ns) / 1_000_000.0,
                    online_newly_triggered=tuple(online_event["newly_triggered"]),
                )
            )

        return StreamSampleResult(
            model=model_name,
            split=sequence.split,
            sample_id=str(sample_id),
            prompt=prompt_record,
            token_records=tuple(token_records),
            online_decisions=tracker.decisions(),
            total_sequence_token_count=len(sequence.input_ids),
            prompt_token_count=sequence.prompt_token_count,
            bridge_token_count=sequence.bridge_token_count,
            response_token_count=len(token_records),
            available_response_token_count=available,
            truncated=len(token_records) != available,
        )
    finally:
        _close_stream(model, stream_state)


def warmup_stream_sample(
    model_or_loaded: Any,
    sequence: PreparedSequence,
    *,
    repeats: int = 1,
    response_tokens: int = 8,
) -> None:
    """Replay a small prefix to warm kernels/allocators without retaining output."""

    if repeats < 0:
        raise ValueError("repeats must be non-negative.")
    if response_tokens < 0:
        raise ValueError("response_tokens must be non-negative.")
    for repeat in range(repeats):
        run_stream_sample(
            model_or_loaded,
            sequence,
            sample_id=f"__warmup_{repeat}",
            max_response_tokens=response_tokens,
        )


__all__ = [
    "OnlineUpdate",
    "PromptModerationRecord",
    "StreamSampleResult",
    "TokenModerationRecord",
    "run_stream_sample",
    "warmup_stream_sample",
]
