"""Focused contracts for the Lane T cached trainer, selector and no-scorer replay."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from sdsafe_sequence_explore.data import open_sequence_dataset
from sdsafe_sequence_explore.model import ModelConfig
from sdsafe_sequence_explore.objectives import ObjectiveConfig
from sdsafe_sequence_explore.train import (
    ModelSelector,
    TrainConfig,
    archive_source_snapshot,
    build_training_inputs,
    make_batches,
    preload_inputs,
    producer_fingerprint,
    sample_epoch_indices,
    train_one,
    validate_saved_run,
    verify_input_bindings,
    _scaler_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def metric(fnr: float, fpr: float, auc: float) -> dict:
    return {"primary": {"fnr": fnr, "fpr": fpr, "threshold": 0.0}, "auc": auc}


class ActualDataFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = open_sequence_dataset(ROOT, verify_integrity=False)

        def take(split: str, label: int, count: int, *, agentlab: bool | None = None) -> list[int]:
            out = []
            for index in cls.dataset.split_episode_indices(split):
                episode = cls.dataset.episodes[index]
                if int(episode["bag_label"]) != label:
                    continue
                if agentlab is not None and (str(episode["scope"]) == "agentlab") != agentlab:
                    continue
                out.append(index)
                if len(out) == count:
                    break
            return out

        cls.fit = take("fit", 1, 8) + take("fit", 0, 8)
        cls.select = take("select", 1, 8) + take("select", 0, 8)
        cls.cal = take("cal", 1, 6) + take("cal", 0, 6)
        cls.dev_common = take("dev", 1, 6, agentlab=False) + take("dev", 0, 6, agentlab=False)
        cls.dev_agentlab = take("dev", 1, 4, agentlab=True) + take("dev", 0, 4, agentlab=True)
        cls.subset = sorted(set(cls.fit + cls.select + cls.cal + cls.dev_common + cls.dev_agentlab))
        cls.inputs = build_training_inputs(cls.dataset, "cpu", episode_indices=cls.subset)

    def test_cached_batch_matches_D_collate_exactly(self) -> None:
        indices = self.fit[:4]
        batch = self.inputs.episode_batch(indices, scaler_name="fit")
        reference = self.dataset.collate_episodes(indices, dtype="float32", scaled=True, scaler="fit")
        self.assertTrue(torch.equal(batch.features, reference.features))
        self.assertTrue(torch.equal(batch.padding_mask, reference.padding_mask))
        self.assertTrue(torch.equal(batch.ct_mask, reference.ct_mask))
        self.assertTrue(torch.equal(batch.bag_label, reference.bag_label))
        self.assertTrue(torch.equal(batch.prefix_state, reference.prefix_state))
        self.assertTrue(torch.equal(batch.exact, reference.exact))
        for policy in ("L0", "L1", "L2", "L3", "L4"):
            self.assertTrue(torch.equal(batch.policy_labels[policy], reference.policy_labels[policy]))
            self.assertTrue(torch.equal(batch.policy_masks[policy], reference.policy_masks[policy]))

    def test_scaled_cache_identity_and_read_only_identity(self) -> None:
        first = self.inputs.scaled_features("fit")
        self.assertIs(first, self.inputs.scaled_features("fit"))
        self.assertEqual(tuple(first.shape), (self.inputs.num_rows, 4096))
        self.assertTrue(bool(torch.isfinite(first).all()))
        self.assertIn("manifest_sha256", self.inputs.identity)
        self.assertIn("source_ledger_content_sha256", self.inputs.identity)
        self.assertFalse(self.inputs.identity["bindings_verified"])

    def test_preload_subset_equals_build(self) -> None:
        other = preload_inputs(
            self.dataset, "cpu", verify_bindings=False, episode_indices=self.subset
        )
        self.assertEqual(other.identity["episode_global_indices_sha256"], self.inputs.identity["episode_global_indices_sha256"])
        self.assertEqual(other.identity["manifest_sha256"], self.inputs.identity["manifest_sha256"])
        self.assertTrue(torch.equal(other.features_bf16, self.inputs.features_bf16))

    def test_verify_bindings_without_hashing_all_inputs(self) -> None:
        report = verify_input_bindings(ROOT, hash_inputs=False)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["source_file_mismatches"], {})
        self.assertTrue(report["manifest_match"])
        self.assertTrue(report["source_ledger_content_match"])

    def test_producer_fingerprint_and_snapshot(self) -> None:
        fingerprint = producer_fingerprint(ROOT)
        self.assertNotIn("src/sdsafe_sequence_explore/cli.py", fingerprint["files"])
        self.assertIn("src/sdsafe_sequence_explore/model.py", fingerprint["files"])
        self.assertIn("src/sdsafe_sequence_explore/train.py", fingerprint["files"])
        self.assertNotIn("src/sdsafe_sequence_explore/report.py", fingerprint["files"])
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            snapshot = archive_source_snapshot(run_dir, fingerprint)
            self.assertEqual(snapshot["fingerprint_sha256"], fingerprint["fingerprint_sha256"])
            self.assertTrue((run_dir / "source_snapshot.json").is_file())
            with self.assertRaises(RuntimeError):
                archive_source_snapshot(run_dir, fingerprint)


class SamplerAndSelectorTests(ActualDataFixture):
    def test_sampler_is_deterministic_and_balanced(self) -> None:
        fit = self.inputs.episode_indices("fit")
        rng_a = np.random.default_rng(1234)
        rng_b = np.random.default_rng(1234)
        draws_a, stats_a = sample_epoch_indices(
            fit, self.inputs.episode_bag_label, self.inputs, rng_a, balanced=True, pos_neg_ratio=3.0
        )
        draws_b, stats_b = sample_epoch_indices(
            fit, self.inputs.episode_bag_label, self.inputs, rng_b, balanced=True, pos_neg_ratio=3.0
        )
        self.assertEqual(draws_a, draws_b)
        self.assertEqual(stats_a, stats_b)
        self.assertGreater(stats_a["positive_draws"], 0)
        self.assertEqual(stats_a["negative_draws"], 3 * stats_a["positive_draws"])
        self.assertEqual(stats_a["draws"], stats_a["unique"] + stats_a["repeats"])

    def test_bucket_batches_cover_each_episode_once(self) -> None:
        fit = self.inputs.episode_indices("fit")
        batches = make_batches(fit, 4, np.random.default_rng(9), inputs=self.inputs, bucket_by_length=True)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), sorted(fit))
        self.assertTrue(all(len(batch) <= 4 for batch in batches))

    def test_dual_selector_direction_and_stopping(self) -> None:
        selector = ModelSelector(patience=1)
        state = {"weight": torch.tensor([1.0])}
        selector.update(0, metric(0.50, 0.05, 0.80), state)
        selector.update(1, metric(0.50, 0.04, 0.90), state)
        self.assertEqual(selector.primary_epoch, 1)
        self.assertEqual(selector.auc_epoch, 1)
        selector.update(2, metric(0.60, 0.04, 0.70), state)
        self.assertEqual(selector.primary_epoch, 1)
        self.assertEqual(selector.auc_epoch, 1)
        self.assertTrue(selector.should_stop())
        restored = ModelSelector.from_state(selector.state_dict())
        self.assertEqual(restored.primary_epoch, 1)
        self.assertEqual(restored.auc_epoch, 1)


class TrainOneTests(ActualDataFixture):
    def make_config(self, *, mode: str = "fit", seed: int = 7) -> TrainConfig:
        model = ModelConfig(
            family="mlp",
            input_dim=4096,
            hidden=8,
            dropout=0.0,
            aux_head=True,
        )
        objective = ObjectiveConfig(
            endpoint_policy="L1",
            endpoint_weight=1.0,
            mil_weight=1.0,
            mil_reduction="max",
            aux_weight=0.1,
            aux_target="policy",
            aux_policy="L1",
        )
        return TrainConfig(
            run_id=f"cpu-{mode}-{seed}",
            mode=mode,
            seed=seed,
            epochs=2,
            max_epochs=2,
            patience=7,
            batch_size=4,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=1.0,
            balanced=True,
            pos_neg_ratio=3.0,
            bucket_by_length=True,
            scaler="fit" if mode != "full_train" else "full_train",
            selection_split="select",
            eval_dev=True,
            eval_cal=True,
            max_steps_per_epoch=2,
            eval_batch_size=8,
            tags=("synthetic", "cpu_focused_test"),
            model=model,
            objective=objective,
        )

    def test_train_complete_validate_and_noop_resume(self) -> None:
        config = self.make_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = train_one(config, self.inputs, run_dir, resume=True)
            self.assertEqual(result["status"], "complete")
            for name in ("run_meta.json", "history.json", "metrics.json", "best_primary.pt", "best_auc.pt", "last.pt"):
                self.assertTrue((run_dir / name).is_file(), name)
            meta = json.loads((run_dir / "run_meta.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["config_sha256"], config.config_sha256)
            self.assertTrue((run_dir / "dev_predictions.npz").is_file())
            self.assertTrue((run_dir / "select_predictions.npz").is_file())
            self.assertTrue((run_dir / "cal_predictions.npz").is_file())
            validation = validate_saved_run(run_dir, inputs=self.inputs)
            self.assertEqual(validation["status"], "PASS")
            self.assertTrue(all(item["pass"] for item in validation["checkpoints"].values()))
            # Completed no-op must verify the frozen recipe/input identity, not overwrite.
            noop = train_one(config, self.inputs, run_dir, resume=True)
            self.assertEqual(noop["status"], "complete_noop")
            changed = TrainConfig.from_dict(config.to_dict())
            object.__setattr__(changed, "seed", 8)
            with self.assertRaises(RuntimeError):
                train_one(changed, self.inputs, run_dir, resume=True)

    def test_validation_does_not_construct_or_forward_a_model(self) -> None:
        config = self.make_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            train_one(config, self.inputs, run_dir)
            with patch("sdsafe_sequence_explore.train.build_model", side_effect=AssertionError("model constructed")):
                with patch("sdsafe_sequence_explore.train.score_episodes", side_effect=AssertionError("rescored")):
                    validation = validate_saved_run(run_dir, inputs=self.inputs)
            self.assertEqual(validation["status"], "PASS")

    def test_failed_run_is_not_overwritten(self) -> None:
        config = self.make_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "run_meta.json").write_text(json.dumps({"status": "failed", "config": config.to_dict()}))
            with self.assertRaises(RuntimeError):
                train_one(config, self.inputs, run_dir, resume=True)

    def test_full_train_mode_uses_full_scaler_and_fixed_epochs(self) -> None:
        config = self.make_config(mode="full_train", seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = train_one(config, self.inputs, run_dir)
            self.assertEqual(result["status"], "complete")
            meta = json.loads((run_dir / "run_meta.json").read_text())
            self.assertEqual(meta["config"]["scaler"], "full_train")
            self.assertFalse(meta["metrics"]["cal_used_adaptively"])
            self.assertTrue(meta["metrics"]["cal_in_training"])
            self.assertFalse(meta["metrics"]["calibration_independent"])
            self.assertTrue(meta["metrics"]["select_is_in_sample"])
            self.assertTrue(meta["metrics"]["selection_rule"].startswith("FINAL_FIXED_REFIT"))
            self.assertEqual(meta["epochs_run"], 2)
            validation = validate_saved_run(run_dir, inputs=self.inputs)
            self.assertEqual(validation["status"], "PASS")
            self.assertFalse(validation["history"]["early_stopped"])

    @staticmethod
    def _refresh_prediction_record(meta: dict, run_dir: Path, name: str) -> None:
        path = run_dir / f"{name}_predictions.npz"
        meta["predictions"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        meta["predictions"][name]["bytes"] = path.stat().st_size

    @staticmethod
    def _rewrite_meta(run_dir: Path, meta: dict) -> None:
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")

    @staticmethod
    def _load_npz(path: Path) -> dict:
        with np.load(path, allow_pickle=False) as data:
            return {key: np.array(data[key]) for key in data.files}

    def test_validator_rejects_missing_prediction_file(self) -> None:
        config = self.make_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            train_one(config, self.inputs, run_dir)
            (run_dir / "dev_predictions.npz").unlink()
            with self.assertRaises(RuntimeError):
                validate_saved_run(run_dir, inputs=self.inputs)

    def test_validator_rejects_tampered_position_even_if_episode_max_unchanged(self) -> None:
        config = self.make_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            train_one(config, self.inputs, run_dir)
            path = run_dir / "dev_predictions.npz"
            arrays = self._load_npz(path)
            logits = arrays["logits"]
            valid = arrays["valid_mask"]
            ct = arrays["ct_mask"]
            found = False
            for row in range(logits.shape[0]):
                eligible = np.flatnonzero(valid[row] & ct[row])
                if len(eligible) >= 2:
                    values = logits[row, eligible]
                    max_position = int(eligible[int(np.argmax(values))])
                    other = int(next(position for position in eligible if position != max_position))
                    logits[row, other] = values.max() - 0.25
                    found = True
                    break
            self.assertTrue(found)
            np.savez_compressed(path, **arrays)
            with self.assertRaises(RuntimeError):
                validate_saved_run(run_dir, inputs=self.inputs)

    def test_validator_rejects_metadata_binding_even_after_file_record_refresh(self) -> None:
        config = self.make_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            train_one(config, self.inputs, run_dir)
            path = run_dir / "dev_predictions.npz"
            arrays = self._load_npz(path)
            arrays["episode_ids"][0] = "wrong::episode"
            np.savez_compressed(path, **arrays)
            meta = json.loads((run_dir / "run_meta.json").read_text())
            self._refresh_prediction_record(meta, run_dir, "dev")
            self._rewrite_meta(run_dir, meta)
            with self.assertRaises(RuntimeError):
                validate_saved_run(run_dir, inputs=self.inputs)

    def test_validator_rejects_wrong_scaler_even_after_declared_hash_edits(self) -> None:
        config = self.make_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            train_one(config, self.inputs, run_dir)
            checkpoint = torch.load(run_dir / "best_primary.pt", map_location="cpu", weights_only=False)
            scaler = checkpoint["scaler"]
            scaler["mean"][0] = scaler["mean"][0] + 1.0
            scaler["mean_sha256"] = hashlib.sha256(scaler["mean"].numpy().tobytes()).hexdigest()
            scaler["tensor_sha256"] = _scaler_sha256(scaler["mean"], scaler["std"])
            torch.save(checkpoint, run_dir / "best_primary.pt")
            meta = json.loads((run_dir / "run_meta.json").read_text())
            record = meta["checkpoints"]["best_primary"]
            record["sha256"] = hashlib.sha256((run_dir / "best_primary.pt").read_bytes()).hexdigest()
            record["bytes"] = (run_dir / "best_primary.pt").stat().st_size
            record["scaler_tensor_sha256"] = scaler["tensor_sha256"]
            record["scaler_mean_sha256"] = scaler["mean_sha256"]
            self._rewrite_meta(run_dir, meta)
            with self.assertRaises(RuntimeError):
                validate_saved_run(run_dir, inputs=self.inputs)

    def test_validator_rejects_history_tamper(self) -> None:
        config = self.make_config()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            train_one(config, self.inputs, run_dir)
            history_path = run_dir / "history.json"
            history = json.loads(history_path.read_text())
            history = history[:-1]
            history_path.write_text(json.dumps(history, indent=1, sort_keys=True) + "\n")
            meta = json.loads((run_dir / "run_meta.json").read_text())
            meta["epochs_run"] = len(history)
            meta["artifact_files"]["history"]["sha256"] = hashlib.sha256(history_path.read_bytes()).hexdigest()
            meta["artifact_files"]["history"]["bytes"] = history_path.stat().st_size
            self._rewrite_meta(run_dir, meta)
            with self.assertRaises(RuntimeError):
                validate_saved_run(run_dir, inputs=self.inputs)

    def test_config_guards(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig(mode="fit", scaler="full_train")
        with self.assertRaises(ValueError):
            TrainConfig(mode="full_train", scaler="fit")
        bad_model = ModelConfig(family="mlp", hidden=8, aux_head=False)
        with self.assertRaises(ValueError):
            TrainConfig(
                model=bad_model,
                objective=ObjectiveConfig(endpoint_policy=None, aux_weight=0.1),
            )
        with self.assertRaises(ValueError):
            TrainConfig(mode="full_train", max_train_episodes=10)
        with self.assertRaises(ValueError):
            TrainConfig(epochs=1, max_epochs=2)

    def test_mixed_partitions_require_explicit_full_train_refit(self) -> None:
        mixed = [self.fit[0], self.select[0], self.cal[0]]
        with self.assertRaises(ValueError):
            self.inputs.episode_batch(mixed, scaler_name="fit")
        with self.assertRaises(ValueError):
            self.inputs.episode_batch(mixed, scaler_name="full_train")
        allowed = self.inputs.episode_batch(
            mixed, scaler_name="full_train", allow_mixed_train_partitions=True
        )
        self.assertEqual(tuple(allowed.features.shape[:2]), (3, max(int(x) for x in allowed.lengths)))
        with self.assertRaises(ValueError):
            self.inputs.episode_batch([self.fit[0], self.dev_common[0]], scaler_name="fit")

    def test_gold_free_cached_batch(self) -> None:
        batch = self.inputs.episode_batch(self.fit[:2], include_labels=False)
        self.assertEqual(batch.policy_labels, {})
        self.assertEqual(batch.policy_masks, {})
        self.assertIsNone(batch.prefix_state)
        self.assertIsNone(batch.exact)
        self.assertIsNone(batch.bag_label)
        self.assertTrue(bool(batch.padding_mask.any()))


if __name__ == "__main__":
    unittest.main()
