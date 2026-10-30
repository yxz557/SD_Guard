from __future__ import annotations

import unittest
import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from sdguard_sd.io_utils import read_json, write_json, write_jsonl
from summarize_matrix import (
    CONTROLLED_CONFIG_KEYS,
    GENERATION_MODES,
    THINKING_LEVELS,
    compare_rows,
    exact_mcnemar_pvalue,
    main,
    summarize_rows,
)


def make_row(
    sample_id: str,
    *,
    mode: str = "target_only",
    thinking: bool = True,
    unsafe: bool = False,
    answer_available: bool = True,
    truncated: bool = False,
) -> dict:
    prediction = "Unsafe" if unsafe else "Safe"

    def scope(available: bool) -> dict:
        return {
            "available": available,
            "decisions": (
                None
                if not available
                else {
                    "strict": {"prediction": prediction},
                    "loose": {"prediction": prediction},
                }
            ),
        }

    inline_result = None
    if mode == "inline_guard":
        inline_result = {
            "proposal_immediate": {
                "latency_ms": 7.0,
                "scopes": {
                    "full_trace": scope(True),
                    "answer_only": scope(answer_available),
                },
            },
            "verified_commit": {
                "latency_ms": 6.0,
                "scopes": {
                    "full_trace": scope(True),
                    "answer_only": scope(answer_available),
                },
            },
            "noise": {
                scope_name: {
                    policy: {
                        "available": scope_name == "full_trace" or answer_available,
                        "proposal_immediate_alert": unsafe,
                        "verified_commit_alert": unsafe,
                        "proposal_only_alert": False,
                        "first_alert_survived_verification": True if unsafe else None,
                    }
                    for policy in ("strict", "loose")
                }
                for scope_name in ("full_trace", "answer_only")
            },
            "proposal_trigger_outcomes": {
                scope_name: {
                    policy: {
                        "triggered_round_count": int(unsafe),
                        "survived_round_count": int(unsafe),
                        "rejected_round_count": 0,
                    }
                    for policy in ("strict", "loose")
                }
                for scope_name in ("full_trace", "answer_only")
            },
            "acceptance_by_proposal_risk": {
                label: {
                    "proposed": 1,
                    "accepted": 1,
                    "rejected": 0,
                    "acceptance_rate": 1.0,
                }
                for label in ("Safe", "Controversial", "Unsafe")
            },
            "consistency": {"verified_posthoc_match": True},
            "timing": {"total_inline_guard_latency_ms": 7.0},
        }

    return {
        "sample_id": sample_id,
        "sample_order": 0 if sample_id == "a" else 1,
        "dataset_index": 10 if sample_id == "a" else 20,
        "generation_mode": mode,
        "thinking_enabled": thinking,
        "completion": {"parse_status": "ok" if answer_available else "missing_close_think"},
        "generation": {
            "generated_token_count": 10,
            "generated_ids": [1, 2, 3],
            "latency_ms": 20.0,
            "truncated": truncated,
        },
        "speculative_decoding": (
            {
                "proposed_token_count": 8,
                "accepted_draft_token_count": 4,
            }
            if mode in {"speculative", "inline_guard"}
            else None
        ),
        "guard": {
            "latency_ms": 5.0,
            "scopes": {
                "full_trace": scope(True),
                "answer_only": scope(answer_available),
            },
        },
        "inline_guard": inline_result,
    }


class MatrixSummaryTests(unittest.TestCase):
    def test_cell_summary_tracks_guard_rates_and_unavailable_answers(self) -> None:
        rows = [
            make_row("a", unsafe=True),
            make_row("b", answer_available=False, truncated=True),
        ]
        result = summarize_rows(rows)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["truncated_count"], 1)
        self.assertEqual(result["missing_close_think_count"], 1)
        self.assertEqual(result["guard"]["full_trace"]["strict"]["unsafe_rate"], 0.5)
        self.assertEqual(result["guard"]["answer_only"]["strict"]["available_count"], 1)

    def test_speculative_summary_aggregates_acceptance(self) -> None:
        rows = [
            make_row("a", mode="speculative"),
            make_row("b", mode="speculative"),
        ]
        result = summarize_rows(rows)
        self.assertEqual(result["speculative_decoding"]["proposed_token_count"], 16)
        self.assertEqual(result["speculative_decoding"]["aggregate_acceptance_rate"], 0.5)

    def test_pairwise_comparison_is_matched_by_sample_id(self) -> None:
        left = [make_row("a", unsafe=True), make_row("b", unsafe=False)]
        right = [make_row("b", unsafe=True), make_row("a", unsafe=False)]
        result = compare_rows(left, right, left_name="left", right_name="right")
        strict = result["guard_decisions"]["full_trace"]["strict"]
        self.assertEqual(strict["left_only_unsafe"], 1)
        self.assertEqual(strict["right_only_unsafe"], 1)
        self.assertEqual(strict["disagreement_rate"], 1.0)

    def test_exact_mcnemar_handles_no_disagreement(self) -> None:
        self.assertIsNone(exact_mcnemar_pvalue(0, 0))
        self.assertEqual(exact_mcnemar_pvalue(1, 1), 1.0)

    def test_full_matrix_summary_validates_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            matrix_dir = Path(directory) / "matrix"
            base_config = {key: None for key in CONTROLLED_CONFIG_KEYS}
            base_config.update(
                {
                    "sample_count": 2,
                    "seed": 42,
                    "max_new_tokens": 4096,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "do_sample": True,
                }
            )
            for thinking_name in THINKING_LEVELS:
                for mode in GENERATION_MODES:
                    cell = matrix_dir / thinking_name / mode
                    (cell / "raw").mkdir(parents=True, exist_ok=True)
                    config = {
                        **base_config,
                        "generation_mode": mode,
                        "enable_thinking": thinking_name == "think_on",
                    }
                    write_json(cell / "run_config.json", config)
                    write_jsonl(
                        cell / "raw" / "generation_records.jsonl",
                        [
                            make_row("a", mode=mode, thinking=thinking_name == "think_on"),
                            make_row(
                                "b",
                                mode=mode,
                                thinking=thinking_name == "think_on",
                                unsafe=True,
                            ),
                        ],
                    )

            with patch(
                "sys.argv",
                ["summarize_matrix.py", "--matrix-dir", str(matrix_dir)],
            ), redirect_stdout(io.StringIO()):
                main()

            report = read_json(matrix_dir / "analysis" / "matrix_summary.json")
            self.assertEqual(report["sample_count_per_cell"], 2)
            self.assertEqual(len(report["cell_summaries"]), 8)
            self.assertEqual(len(report["generation_mode_comparisons"]), 12)
            self.assertEqual(len(report["thinking_comparisons"]), 4)
            self.assertFalse(report["judged_metrics_complete"])


if __name__ == "__main__":
    unittest.main()

