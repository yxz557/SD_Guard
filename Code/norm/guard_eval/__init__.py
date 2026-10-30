"""Reusable components for the Qwen3Guard-Stream reproduction scripts.

Heavy third-party libraries are imported lazily by the functions that need
them, so importing :mod:`guard_eval` itself is safe in inspection and analysis
jobs that do not load a model.
"""

from .dataset import (
    DatasetFormatError,
    GuardSample,
    LOCATION_SPLITS,
    VALID_SPLITS,
    iter_guard_samples,
    load_guard_samples,
    load_split,
    normalize_label,
)
from .full_forward import (
    ConsistencyMismatch,
    ConsistencyReport,
    FullForwardResult,
    compare_stream_and_full,
    run_full_forward,
)
from .location_metrics import (
    LOCATION_BINS,
    RELATIONS,
    classify_trigger_location,
    summarize_location_decisions,
)
from .model_loader import (
    LoadedGuard,
    collect_runtime_metadata,
    load_stream_guard,
    model_cuda_devices,
    model_input_device,
    synchronize_model,
    unwrap_model,
)
from .safe_trigger_metrics import (
    SAFE_TRIGGER_BINS,
    SAFE_TRIGGER_THRESHOLDS,
    classify_safe_trigger,
    summarize_safe_trigger_decisions,
)
from .sequence import (
    AlignmentReport,
    PreparedSequence,
    SequenceBoundaryError,
    compare_input_ids,
    prepare_sequence,
    split_uses_thinking,
)
from .stream_runner import (
    PromptModerationRecord,
    StreamSampleResult,
    TokenModerationRecord,
    run_stream_sample,
    warmup_stream_sample,
)


__all__ = [
    "AlignmentReport",
    "ConsistencyMismatch",
    "ConsistencyReport",
    "DatasetFormatError",
    "FullForwardResult",
    "GuardSample",
    "LOCATION_BINS",
    "LOCATION_SPLITS",
    "LoadedGuard",
    "PreparedSequence",
    "PromptModerationRecord",
    "RELATIONS",
    "SAFE_TRIGGER_BINS",
    "SAFE_TRIGGER_THRESHOLDS",
    "SequenceBoundaryError",
    "StreamSampleResult",
    "TokenModerationRecord",
    "VALID_SPLITS",
    "collect_runtime_metadata",
    "classify_trigger_location",
    "classify_safe_trigger",
    "compare_input_ids",
    "compare_stream_and_full",
    "iter_guard_samples",
    "load_guard_samples",
    "load_split",
    "load_stream_guard",
    "model_cuda_devices",
    "model_input_device",
    "normalize_label",
    "prepare_sequence",
    "run_full_forward",
    "run_stream_sample",
    "split_uses_thinking",
    "summarize_location_decisions",
    "summarize_safe_trigger_decisions",
    "synchronize_model",
    "unwrap_model",
    "warmup_stream_sample",
]
