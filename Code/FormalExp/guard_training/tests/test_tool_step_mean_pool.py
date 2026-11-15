from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from guard_training.common.feature_loading import read_jsonl
from guard_training.common.feature_store import FeatureShardWriter, iter_feature_metadata
from guard_training.common.offline_dataset import load_offline_split
from guard_training.tool_step_mean_pool.build_offline_datasets import (
    build_offline_datasets,
)
from guard_training.tool_step_mean_pool.compare_experiments import normalized_metrics


class ToolStepMeanPoolTest(unittest.TestCase):
    def make_feature_root(self, root: Path) -> dict[str, dict]:
        root.mkdir(parents=True)
        config = {
            "schema_version": "test",
            "serialization_version": "test",
            "model_config_sha256": "model",
            "tokenizer_class": "test",
            "tokenizer_chat_template_sha256": "template",
            "saved_layer_count": 2,
            "hidden_size": 2,
            "storage_dtype": "float32",
            "readout_prompt": "judge safety",
            "global_features": ["readout_last"],
            "step_kinds": {
                "tool_call_end": 0,
                "assistant_decision_end": 1,
                "tool_result_end": 2,
                "tool_round_end": 3,
            },
        }
        (root / "feature_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        writer = FeatureShardWriter(root / "features", shard_size=20)
        samples = {
            "train-good": {
                "label": 1,
                "steps": [
                    ("tool_call_end", (1.0, 3.0)),
                    ("tool_result_end", (10.0, 12.0)),
                    ("tool_call_end", (5.0, 7.0)),
                ],
                "readout": (20.0, 22.0),
            },
            "train-no-tools": {"label": 0, "steps": [], "readout": (2.0, 4.0)},
            "validation-good": {
                "label": 0,
                "steps": [
                    ("tool_call_end", (2.0, 4.0)),
                    ("tool_result_end", (6.0, 8.0)),
                ],
                "readout": (12.0, 14.0),
            },
            "test-good": {
                "label": 1,
                "steps": [
                    ("tool_call_end", (3.0, 5.0)),
                    ("tool_result_end", (7.0, 9.0)),
                ],
                "readout": (13.0, 15.0),
            },
        }
        for index, (sample_id, value) in enumerate(samples.items()):
            metadata = {
                "sample_id": sample_id,
                "group_id": f"group-{sample_id}",
                "label": value["label"],
                "suite_name": "travel",
                "attack_type": "direct",
                "user_task_id": f"user-{index}",
                "injection_task_id": "injection-0",
                "data_source": "synthetic",
            }
            step_metadata = [
                {"kind": kind, "token_index": step_index}
                for step_index, (kind, _) in enumerate(value["steps"])
            ]
            step_features = torch.empty((len(step_metadata), 2, 2), dtype=torch.float32)
            for step_index, (_, final_layer) in enumerate(value["steps"]):
                step_features[step_index, 0] = torch.tensor((-1.0, -1.0))
                step_features[step_index, 1] = torch.tensor(final_layer)
            global_features = {
                "readout_last": torch.tensor(
                    [[-2.0, -2.0], list(value["readout"])], dtype=torch.float32
                )
            }
            writer.add(metadata, global_features, step_features, step_metadata)
        writer.flush()
        return {str(row["sample_id"]): row for row in iter_feature_metadata(root / "features")}

    def make_input_dataset(
        self, root: Path, feature_root: Path, feature_rows: dict[str, dict]
    ) -> None:
        root.mkdir(parents=True)
        (root / "dataset_config.json").write_text(
            json.dumps(
                {
                    "schema_version": "input",
                    "feature": "readout_last",
                    "layer": -1,
                    "local_feature_root": str(feature_root),
                }
            ),
            encoding="utf-8",
        )
        splits = {
            "train": ["train-good", "train-no-tools"],
            "validation": ["validation-good"],
            "test": ["test-good"],
        }
        for split, sample_ids in splits.items():
            with (root / f"{split}.metadata.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                for compact_row, sample_id in enumerate(sample_ids):
                    row = dict(feature_rows[sample_id])
                    row["compact_row"] = compact_row
                    handle.write(json.dumps(row) + "\n")

    def test_builds_separate_mean_pools_and_common_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature_root = root / "features-source"
            rows = self.make_feature_root(feature_root)
            input_dataset = root / "input"
            self.make_input_dataset(input_dataset, feature_root, rows)
            output = root / "output"
            report = build_offline_datasets(
                input_dataset=input_dataset,
                output_root=output,
                feature_roots=[feature_root],
                layer=-1,
                storage_dtype_name="float32",
                include_r0_common=True,
            )

            self.assertEqual(report["splits"]["train"]["before_filter"]["samples"], 2)
            self.assertEqual(report["splits"]["train"]["after_filter"]["samples"], 1)
            self.assertEqual(report["excluded_samples"], 1)
            excluded = read_jsonl(output / "excluded_by_eligibility.jsonl")
            self.assertEqual(excluded[0]["sample_id"], "train-no-tools")

            c0_x, _, c0_rows, c0_config = load_offline_split(output / "c0", "train")
            self.assertEqual(c0_rows[0]["sample_id"], "train-good")
            self.assertEqual(c0_config["input_dim"], 2)
            torch.testing.assert_close(c0_x[0], torch.tensor([3.0, 5.0]))

            rct_x, _, _, rct_config = load_offline_split(output / "rct", "train")
            self.assertEqual(rct_config["input_dim"], 6)
            torch.testing.assert_close(
                rct_x[0], torch.tensor([20.0, 22.0, 3.0, 5.0, 10.0, 12.0])
            )
            self.assertTrue((output / "r0_common" / "dataset_config.json").is_file())

    def test_normalized_metrics_derives_accuracy_fpr_and_fnr(self) -> None:
        view = normalized_metrics(
            {
                "confusion_matrix": [[8, 2], [1, 9]],
                "average_precision": 0.8,
                "roc_auc": 0.9,
                "balanced_accuracy": 0.85,
                "positive_precision": 9 / 11,
                "positive_recall": 0.9,
                "positive_f1": 0.85,
                "threshold": 0.4,
            }
        )
        self.assertAlmostEqual(view["accuracy"], 0.85)
        self.assertAlmostEqual(view["false_positive_rate"], 0.2)
        self.assertAlmostEqual(view["false_negative_rate"], 0.1)


if __name__ == "__main__":
    unittest.main()

