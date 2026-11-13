from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guard_training.qwen3_trace_features.inspect_tool_step_coverage import (
    anomaly_record,
    build_report,
    enrich_row,
    inspect_raw_trace,
    load_feature_rows,
    load_offline_annotations,
)


def metadata_row(sample_id: str, label: int, kinds: list[str]) -> dict:
    return {
        "sample_id": sample_id,
        "label": label,
        "suite_name": "travel" if sample_id != "neither" else "workspace",
        "attack_type": "direct",
        "user_task_id": "user_task_0",
        "injection_task_id": "injection_task_0",
        "group_id": '["travel","user_task_0","injection_task_0"]',
        "steps": [
            {"kind": kind, "token_index": index, "tool_call_id": f"call-{index}"}
            for index, kind in enumerate(kinds)
        ],
    }


class ToolStepCoverageTest(unittest.TestCase):
    def test_enrich_row_classifies_four_coverage_cases(self) -> None:
        cases = {
            "both": ["tool_call_end", "tool_result_end"],
            "call_only": ["tool_call_end"],
            "result_only": ["tool_result_end"],
            "neither": ["assistant_message_end"],
        }
        for expected, kinds in cases.items():
            with self.subTest(expected=expected):
                row = enrich_row(metadata_row(expected, 0, kinds), "local")
                self.assertEqual(row["tool_step_coverage"], expected)

    def test_report_compares_all_rows_with_offline_selection(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            feature_root = root / "features-local"
            feature_dir = feature_root / "features"
            feature_dir.mkdir(parents=True)
            rows = [
                metadata_row("both", 1, ["tool_call_end", "tool_result_end"]),
                metadata_row("call_only", 0, ["tool_call_end"]),
                metadata_row("result_only", 0, ["tool_result_end"]),
                metadata_row("neither", 0, []),
            ]
            with (feature_dir / "shard_00000.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            offline = root / "offline"
            offline.mkdir()
            with (offline / "train.metadata.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(json.dumps({"sample_id": "both", "data_source": "local"}) + "\n")
                handle.write(
                    json.dumps({"sample_id": "call_only", "data_source": "local"}) + "\n"
                )

            loaded = load_feature_rows([feature_root], ["local"])
            annotations = load_offline_annotations(offline)
            report = build_report(
                loaded,
                [feature_root],
                ["local"],
                annotations,
                offline,
                examples_per_category=1,
            )

            overall = report["all_feature_rows"]["overall"]
            self.assertEqual(overall["samples"], 4)
            self.assertEqual(overall["coverage"]["both"]["samples"], 1)
            self.assertEqual(overall["coverage"]["call_only"]["samples"], 1)
            self.assertEqual(overall["coverage"]["result_only"]["samples"], 1)
            self.assertEqual(overall["coverage"]["neither"]["samples"], 1)
            self.assertEqual(report["all_feature_rows"]["by_label"]["0"]["samples"], 3)
            self.assertEqual(report["all_feature_rows"]["by_label"]["1"]["samples"], 1)

            selected = report["offline_selection"]
            self.assertEqual(selected["selected_samples"], 2)
            self.assertEqual(selected["excluded_feature_rows"], 2)
            self.assertEqual(selected["overall"]["eligible_call_and_result"]["samples"], 1)
            self.assertEqual(selected["by_offline_split"]["train"]["samples"], 2)

    def test_raw_trace_errors_are_reported_in_anomaly_record(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            path = Path(temporary) / "raw_trace.json"
            raw = {
                "provenance": {"run_error": None},
                "trajectory": {
                    "messages": [
                        {
                            "role": "tool",
                            "tool_call_id": "call-1",
                            "tool_call": {"function": "search_files"},
                            "content": [{"type": "text", "content": ""}],
                            "error": "ValueError: no files found",
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call-2",
                            "tool_call": {"function": "list_files"},
                            "content": [{"type": "text", "content": "file-1"}],
                            "error": None,
                        },
                    ]
                },
            }
            path.write_text(json.dumps(raw), encoding="utf-8")
            diagnostics = inspect_raw_trace(path, max_error_chars=2000)
            self.assertEqual(diagnostics["tool_message_count"], 2)
            self.assertEqual(diagnostics["structured_tool_error_count"], 1)
            self.assertTrue(diagnostics["has_structured_tool_error"])
            self.assertFalse(diagnostics["all_tool_results_structured_errors"])
            self.assertFalse(diagnostics["last_tool_result_structured_error"])

            row = enrich_row(
                metadata_row("failed_then_recovered", 0, ["tool_call_end", "tool_result_end"]),
                "local",
            )
            row.update(diagnostics)
            record = anomaly_record(row)
            self.assertIsNotNone(record)
            assert record is not None
            self.assertIn("structured_tool_error", record["diagnostic_flags"])
            self.assertNotIn(
                "all_tool_results_structured_errors", record["diagnostic_flags"]
            )


if __name__ == "__main__":
    unittest.main()
