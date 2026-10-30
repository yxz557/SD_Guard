"""Official-style full causal forward and streaming consistency checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .model_loader import model_input_device, unwrap_model
from .policies import normalize_risk_label
from .sequence import PreparedSequence
from .stream_runner import StreamSampleResult


@dataclass(frozen=True, slots=True)
class FullForwardResult:
    """Risk-head predictions aligned to the evaluated Assistant token range."""

    split: str
    input_ids: tuple[int, ...]
    eval_start_index: int
    risk_levels: tuple[str, ...]
    risk_probs: tuple[float, ...]
    prompt_risk_level: str
    prompt_risk_prob: float

    @property
    def response_token_count(self) -> int:
        return len(self.risk_levels)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["input_ids"] = list(self.input_ids)
        value["risk_levels"] = list(self.risk_levels)
        value["risk_probs"] = list(self.risk_probs)
        return value


@dataclass(frozen=True, slots=True)
class ConsistencyMismatch:
    assistant_token_index: int
    full_sequence_index: int
    reason: str
    stream_risk_level: str | None
    full_risk_level: str | None
    stream_risk_prob: float | None
    full_risk_prob: float | None


@dataclass(frozen=True, slots=True)
class ConsistencyReport:
    """Comparison report for a single full-forward/streaming sample pair."""

    sample_id: str
    split: str
    labels_match: bool
    probabilities_match: bool | None
    prompt_label_matches: bool
    stream_token_count: int
    full_token_count: int
    probability_tolerance: float | None
    mismatches: tuple[ConsistencyMismatch, ...]

    @property
    def matches(self) -> bool:
        probability_ok = self.probabilities_match in {None, True}
        return self.labels_match and self.prompt_label_matches and probability_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "matches": self.matches,
            "labels_match": self.labels_match,
            "probabilities_match": self.probabilities_match,
            "prompt_label_matches": self.prompt_label_matches,
            "stream_token_count": self.stream_token_count,
            "full_token_count": self.full_token_count,
            "probability_tolerance": self.probability_tolerance,
            "mismatches": [asdict(mismatch) for mismatch in self.mismatches],
        }


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise RuntimeError("Full-forward validation requires PyTorch.") from exc
    return torch


def _risk_map(model: Any, *, prompt: bool) -> dict[int, str]:
    attribute = "query_risk_level_map" if prompt else "response_risk_level_map"
    raw = getattr(model, attribute, None)
    if raw is None and getattr(model, "config", None) is not None:
        raw = getattr(model.config, attribute, None)
    if raw is None:
        raw = {0: "Safe", 1: "Unsafe", 2: "Controversial"}
    return {int(index): normalize_risk_label(label) for index, label in raw.items()}


def _decode_logits(torch: Any, logits: Any, mapping: dict[int, str]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    best_probabilities, best_indices = torch.max(probabilities, dim=-1)
    indices = best_indices.detach().cpu().reshape(-1).tolist()
    values = best_probabilities.detach().cpu().reshape(-1).tolist()
    try:
        labels = tuple(mapping[int(index)] for index in indices)
    except KeyError as exc:
        raise RuntimeError(f"Model produced an unmapped risk index {exc.args[0]!r}.") from exc
    return labels, tuple(float(value) for value in values)


def run_full_forward(
    model_or_loaded: Any,
    sequence: PreparedSequence,
) -> FullForwardResult:
    """Run the released evaluator's one-shot causal-mask computation.

    This path is for a small correctness check only. It deliberately constructs
    the quadratic four-dimensional boolean mask used in the official script and
    must not be used for the latency benchmark.
    """

    torch = _import_torch()
    model = unwrap_model(model_or_loaded)
    device = model_input_device(model)
    input_ids = torch.tensor(sequence.input_ids, dtype=torch.long, device=device).unsqueeze(0)
    sequence_length = input_ids.shape[1]
    causal_mask = torch.tril(
        torch.ones(
            (sequence_length, sequence_length),
            device=device,
            dtype=torch.bool,
        )
    ).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        outputs = model.forward(input_ids=input_ids, attention_mask=causal_mask)

    response_logits = outputs.risk_level_logits[0, sequence.eval_start_index :, :]
    prompt_logits = outputs.query_risk_level_logits[
        0, sequence.user_end_index : sequence.user_end_index + 1, :
    ]
    response_labels, response_probabilities = _decode_logits(
        torch, response_logits, _risk_map(model, prompt=False)
    )
    prompt_labels, prompt_probabilities = _decode_logits(
        torch, prompt_logits, _risk_map(model, prompt=True)
    )

    return FullForwardResult(
        split=sequence.split,
        input_ids=sequence.input_ids,
        eval_start_index=sequence.eval_start_index,
        risk_levels=response_labels,
        risk_probs=response_probabilities,
        prompt_risk_level=prompt_labels[-1],
        prompt_risk_prob=prompt_probabilities[-1],
    )


def compare_stream_and_full(
    stream: StreamSampleResult,
    full: FullForwardResult,
    *,
    probability_tolerance: float | None = None,
) -> ConsistencyReport:
    """Compare labels, and optionally probabilities, at aligned token positions.

    The public streaming method rounds probabilities to two decimals. If a
    probability check is desired, ``0.011`` is a sensible tolerance; label
    equality remains the primary correctness criterion.
    """

    if stream.split != full.split:
        raise ValueError(
            f"Cannot compare different splits: {stream.split!r} and {full.split!r}."
        )
    if probability_tolerance is not None and probability_tolerance < 0:
        raise ValueError("probability_tolerance must be non-negative or None.")

    mismatches: list[ConsistencyMismatch] = []
    stream_records = stream.token_records
    paired_count = min(len(stream_records), len(full.risk_levels))
    label_mismatch_found = False
    probability_mismatch_found = False

    for index in range(paired_count):
        record = stream_records[index]
        full_label = full.risk_levels[index]
        full_probability = full.risk_probs[index]
        label_differs = record.risk_level != full_label
        probability_differs = (
            probability_tolerance is not None
            and abs(record.risk_prob - full_probability) > probability_tolerance
        )
        if label_differs or probability_differs:
            label_mismatch_found = label_mismatch_found or label_differs
            probability_mismatch_found = probability_mismatch_found or probability_differs
            reasons: list[str] = []
            if label_differs:
                reasons.append("risk_label")
            if probability_differs:
                reasons.append("risk_probability")
            mismatches.append(
                ConsistencyMismatch(
                    assistant_token_index=index,
                    full_sequence_index=full.eval_start_index + index,
                    reason="+".join(reasons),
                    stream_risk_level=record.risk_level,
                    full_risk_level=full_label,
                    stream_risk_prob=record.risk_prob,
                    full_risk_prob=full_probability,
                )
            )

    if len(stream_records) != len(full.risk_levels):
        label_mismatch_found = True
        if probability_tolerance is not None:
            probability_mismatch_found = True
        for index in range(paired_count, max(len(stream_records), len(full.risk_levels))):
            record = stream_records[index] if index < len(stream_records) else None
            full_label = full.risk_levels[index] if index < len(full.risk_levels) else None
            full_probability = full.risk_probs[index] if index < len(full.risk_probs) else None
            mismatches.append(
                ConsistencyMismatch(
                    assistant_token_index=index,
                    full_sequence_index=full.eval_start_index + index,
                    reason="missing_token",
                    stream_risk_level=None if record is None else record.risk_level,
                    full_risk_level=full_label,
                    stream_risk_prob=None if record is None else record.risk_prob,
                    full_risk_prob=full_probability,
                )
            )

    return ConsistencyReport(
        sample_id=stream.sample_id,
        split=stream.split,
        labels_match=not label_mismatch_found,
        probabilities_match=(
            None if probability_tolerance is None else not probability_mismatch_found
        ),
        prompt_label_matches=stream.prompt.risk_level == full.prompt_risk_level,
        stream_token_count=len(stream_records),
        full_token_count=len(full.risk_levels),
        probability_tolerance=probability_tolerance,
        mismatches=tuple(mismatches),
    )


__all__ = [
    "ConsistencyMismatch",
    "ConsistencyReport",
    "FullForwardResult",
    "compare_stream_and_full",
    "run_full_forward",
]
