from __future__ import annotations

import unittest
from types import SimpleNamespace

from run_vllm_latency_test import _build_report
from sd_latency.vllm_profile import classify_profile_event, summarize_trace_events
from sd_latency.vllm_worker import (
    _diagnose_profile,
    _partition,
    _subtract_metric_snapshots,
    derive_speculative_metrics,
)


class VllmWorkerPureHelperTests(unittest.TestCase):
    def test_profile_event_classification_uses_call_stack(self) -> None:
        self.assertEqual(
            classify_profile_event(
                {
                    "name": "aten::matmul",
                    "args": {
                        "Call stack": (
                            "/vllm/v1/worker/gpu/spec_decode/draft_model/"
                            "speculator.py(300): _multi_step_decode"
                        )
                    },
                }
            ),
            "draft_proposal",
        )

    def test_profile_summary_attributes_correlated_gpu_kernels(self) -> None:
        events = [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "draft_proposal",
                "pid": 1,
                "tid": 2,
                "ts": 0,
                "dur": 90,
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "pid": 1,
                "tid": 2,
                "ts": 10,
                "dur": 1,
                "args": {"correlation": 100},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "draft_kernel",
                "pid": 0,
                "tid": 7,
                "ts": 20,
                "dur": 30,
                "args": {"correlation": 100},
            },
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "gpu_model_runner: forward",
                "pid": 1,
                "tid": 2,
                "ts": 100,
                "dur": 90,
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaGraphLaunch",
                "pid": 1,
                "tid": 2,
                "ts": 110,
                "dur": 1,
                "args": {"correlation": 200},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "target_kernel",
                "pid": 0,
                "tid": 7,
                "ts": 120,
                "dur": 70,
                "args": {"correlation": 200},
            },
        ]
        summary = summarize_trace_events(events)
        self.assertTrue(summary["available"])
        self.assertEqual(summary["total_gpu_kernel_ms"], 0.1)
        self.assertEqual(summary["cuda_graph_launch_count"], 1)
        self.assertEqual(
            summary["stages"]["draft_proposal"]["share_of_total_gpu_kernel_time"],
            0.3,
        )
        self.assertEqual(
            summary["stages"]["target_verification"][
                "share_of_total_gpu_kernel_time"
            ],
            0.7,
        )

    def test_profile_diagnosis_combines_acceptance_and_runtime_facts(self) -> None:
        diagnosis = _diagnose_profile(
            {
                "attribution_coverage": 0.9,
                "stages": {
                    "draft_proposal": {
                        "gpu_kernel_ms": 80.0,
                        "share_of_total_gpu_kernel_time": 0.8,
                    },
                    "target_verification": {
                        "gpu_kernel_ms": 20.0,
                        "share_of_total_gpu_kernel_time": 0.2,
                    },
                    "other_or_unattributed": {
                        "gpu_kernel_ms": 0.0,
                        "share_of_total_gpu_kernel_time": 0.0,
                    },
                },
            },
            {"draft_acceptance_rate": 0.33},
            {"plain_draft_cuda_graph_supported": False},
        )
        self.assertTrue(diagnosis["stage_conclusion_supported"])
        self.assertEqual(
            diagnosis["dominant_attributed_gpu_stage"], "draft_proposal"
        )
        statuses = {signal["id"]: signal["status"] for signal in diagnosis["signals"]}
        self.assertEqual(statuses["draft_acceptance"], "low")
        self.assertEqual(statuses["plain_draft_cuda_graph"], "unsupported")

    def test_partition_preserves_records(self) -> None:
        records = [{"id": str(index)} for index in range(4)]
        self.assertEqual(
            [[row["id"] for row in batch] for batch in _partition(records, 2)],
            [["0", "1"], ["2", "3"]],
        )

    def test_counter_delta_excludes_warmup(self) -> None:
        before = {
            "vllm:spec_decode_num_drafts": 3.0,
            "vllm:spec_decode_num_accepted_tokens_per_pos": [2.0, 1.0],
        }
        after = {
            "vllm:spec_decode_num_drafts": 8.0,
            "vllm:spec_decode_num_accepted_tokens_per_pos": [6.0, 3.0],
        }
        self.assertEqual(
            _subtract_metric_snapshots(after, before),
            {
                "vllm:spec_decode_num_drafts": 5.0,
                "vllm:spec_decode_num_accepted_tokens_per_pos": [4.0, 2.0],
            },
        )

    def test_derived_acceptance_metrics(self) -> None:
        value = derive_speculative_metrics(
            {
                "vllm:spec_decode_num_drafts": 10.0,
                "vllm:spec_decode_num_draft_tokens": 40.0,
                "vllm:spec_decode_num_accepted_tokens": 20.0,
                "vllm:spec_decode_num_accepted_tokens_per_pos": [8.0, 6.0],
            }
        )
        self.assertTrue(value["available"])
        self.assertEqual(value["num_spec_steps"], 10)
        self.assertEqual(value["num_draft_tokens"], 40)
        self.assertEqual(value["num_accepted_draft_tokens"], 20)
        self.assertEqual(value["draft_acceptance_rate"], 0.5)
        self.assertEqual(value["mean_acceptance_length"], 3.0)
        self.assertEqual(value["acceptance_rate_per_position"], [0.8, 0.6])

    def test_parent_report_compares_two_worker_results(self) -> None:
        class FakeTokenizer:
            @staticmethod
            def decode(ids, skip_special_tokens=False):
                del skip_special_tokens
                return " ".join(str(value) for value in ids)

        def worker(mode: str, elapsed_ms: float):
            return {
                "runtime": {"vllm_version": "test"},
                "engine_load_ms": 1.0,
                "models": {"target": "target", "draft": "draft"},
                "revisions": {"target": None, "draft": None},
                "engine": {"attention_backend": "auto"},
                "settings": {"num_speculative_tokens": 5},
                "benchmark": {"modes": ["greedy"], "batch_size": 1},
                "modes": {
                    "greedy": {
                        "first_outputs": {"p": [1, 2]},
                        "runs": [
                            {
                                "run_key": "batch0:repeat0",
                                "batch_index": 0,
                                "repeat_index": 0,
                                "prompt_ids": ["p"],
                                "request_count": 1,
                                "preparation_ms": 0.0,
                                "generate_ms": elapsed_ms,
                                "output_transfer_ms": 0.0,
                                "total_ms": elapsed_ms,
                                "generated_token_count": 2,
                                "tokens_per_second": 2000.0 / elapsed_ms,
                                "output_token_ids": {"p": [1, 2]},
                            }
                        ],
                        "speculative_decoding_metrics": {
                            "available": mode == "speculative"
                        },
                    }
                },
            }

        report, pairs = _build_report(
            SimpleNamespace(engine_order="target-first"),
            [
                {
                    "id": "p",
                    "prompt": "hello",
                    "prompt_token_ids": [10],
                    "seed": 1,
                }
            ],
            FakeTokenizer(),
            worker("target_only", 100.0),
            worker("speculative", 50.0),
        )
        self.assertEqual(report["aggregate_latency"]["greedy"]["latency_speedup"], 2.0)
        self.assertTrue(
            report["aggregate_latency"]["greedy"][
                "all_paired_outputs_exactly_match"
            ]
        )
        self.assertIn("TARGET_ONLY", "\n".join(pairs))


if __name__ == "__main__":
    unittest.main()
