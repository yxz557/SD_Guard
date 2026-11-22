"""Focused synthetic + materialised-contract tests for the sequence data seam."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sdsafe_sequence_explore import (
    DataContractError,
    DatasetError,
    EXPECTED_DEV_EPISODES,
    EXPECTED_DEV_ROWS,
    EXPECTED_POLICY_COUNTS,
    EXPECTED_TRAIN_CT_ROWS,
    EXPECTED_TRAIN_E_ROWS,
    EXPECTED_TRAIN_EPISODES,
    HIDDEN_SIZE,
    PARTITIONS,
    ROW_BYTES,
    SPLIT_DEV,
    SPLIT_TRAIN,
)
from sdsafe_sequence_explore import data as D
from sdsafe_sequence_explore import labels as L

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "artifacts/fine_sequence_explore_v1/data/sequence_v1"


def feature_ref(event_id: str, split: str, *, token_index: int = 10, seq: str = "seq") -> dict:
    return {
        "kind": "cached_canonical_bf16",
        "scope": "cot",
        "split": split,
        "global_feature_row": token_index,
        "local_feature_row": token_index,
        "feature_file": "matrix.bf16",
        "feature_byte_offset": token_index * ROW_BYTES,
        "feature_token_index": token_index,
        "sequence_sha256": seq,
        "event_id": event_id,
        "hidden_size": HIDDEN_SIZE,
        "dtype": "bfloat16",
        "status": "cached_read_only",
    }


def source_row(
    event_id: str,
    *,
    split: str = SPLIT_DEV,
    event_type: str = "tool_call",
    token_index: int = 10,
    round_id: int = 1,
    index: int = 0,
    prefix: bool | None = False,
    previous: bool | None = False,
    status: str = "exact",
    strict_final: bool = False,
) -> dict:
    exact = status == "exact"
    if not exact:
        prefix = None
        previous = None
    return {
        "episode_id": "cot32k::ep1",
        "namespaced_episode_id": "cot32k::ep1",
        "event_id": event_id,
        "source_tag": "cot32k",
        "scope": "cot",
        "split": split,
        "source_split": split,
        "group_id": "g1",
        "group_hash": "h1",
        "strict_final": strict_final,
        "event_type": event_type,
        "round_id": round_id,
        "message_index": 1,
        "index": index,
        "call_index": index if event_type == "tool_call" else None,
        "segment_index": index if event_type == "content_segment" else None,
        "feature_token_index": token_index,
        "token_start": token_index,
        "token_end": token_index,
        "prompt_length": 1,
        "sequence_length": token_index + 1,
        "sequence_sha256": "seq",
        "round_sequence_sha256": "seq",
        "terminal_token_id": 151645,
        "inference_position": True,
        "label_status": status,
        "label_value": bool(prefix) if exact else None,
        "prefix_security": prefix,
        "previous_security": previous,
        "security_flip": bool(prefix and not previous) if exact else None,
        "repair_status": "active_old",
        "repair_version": "fine_label_repair_v1",
        "supersedes_event_id": None,
        "feature_ref": feature_ref(event_id, split, token_index=token_index),
    }


class FakeProvider:
    """Synthetic stand-in for the actual read boundary.

    ``authority`` is an event_id -> split map.  ``resolve_ref`` runs before any vector
    is produced, exactly like ``FeatureStore.resolve_ref``.
    """

    def __init__(self, authority: dict[str, str]) -> None:
        self.authority = authority
        self.reads: list[tuple[str, ...]] = []

    def resolve_ref(self, ref: dict, expected_split: str) -> D.ResolvedFeature:
        event_id = str(ref["event_id"])
        actual = self.authority.get(event_id)
        if actual is None:
            raise DataContractError(f"{event_id}: absent from authority")
        if actual not in (SPLIT_TRAIN, SPLIT_DEV):
            raise DataContractError(f"{event_id}: refusing TEST authority row")
        if actual != expected_split:
            raise DataContractError(f"{event_id}: authority {actual!r} != requested {expected_split!r}")
        if str(ref.get("split")) != expected_split:
            raise DataContractError(f"{event_id}: forged ref split {ref.get('split')!r}")
        return D.ResolvedFeature(
            physical_key=("fake", event_id, 0),
            kind=str(ref["kind"]),
            version="fake",
            matrix_path=Path("/tmp/fake/matrix.bf16"),
            matrix_total_rows=100,
            feature_row=int(ref["feature_token_index"]),
            scope=str(ref["scope"]),
            event_id=event_id,
            sequence_sha256=str(ref["sequence_sha256"]),
            feature_token_index=int(ref["feature_token_index"]),
        )

    def read_refs(self, refs, expected_split: str) -> np.ndarray:
        self.reads.append(tuple(str(ref["event_id"]) for ref in refs))
        return np.ones((len(refs), HIDDEN_SIZE), dtype=np.float32)

    def round_end_metadata(self, split: str) -> list[dict]:
        return []


def fake_episode(*, strict_final: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        namespaced_episode_id="cot32k::ep1",
        scope="cot",
        source_tag="cot32k",
        group_id="g1",
        group_hash="h1",
        split=SPLIT_DEV,
        source_split=SPLIT_DEV,
        strict_final=bool(strict_final),
    )


class OrderingTests(unittest.TestCase):
    def test_chronological_identity_and_ties(self) -> None:
        rows = [
            source_row("cot32k::ep1::r1::round_end::0", event_type="round_end", token_index=10),
            source_row("cot32k::ep1::r1::content_segment::0", event_type="content_segment", token_index=10, index=0),
            source_row("cot32k::ep1::r1::tool_call::0", event_type="tool_call", token_index=10, index=0),
            source_row("cot32k::ep1::r2::tool_call::0", event_type="tool_call", token_index=9, round_id=2),
            source_row("cot32k::ep1::r1::content_segment::1", event_type="content_segment", token_index=5, index=1),
        ]
        ordered = sorted(rows, key=L.event_order_key)
        self.assertEqual(
            [row["event_id"] for row in ordered],
            [
                "cot32k::ep1::r1::content_segment::1",
                "cot32k::ep1::r1::tool_call::0",
                "cot32k::ep1::r1::content_segment::0",
                "cot32k::ep1::r1::round_end::0",
                "cot32k::ep1::r2::tool_call::0",
            ],
        )
        self.assertEqual(ordered, sorted(rows, key=L.event_order_key))


class GroupSplitTests(unittest.TestCase):
    @staticmethod
    def synthetic_episodes(groups: int = 20, per_group: int = 3) -> list[SimpleNamespace]:
        out = []
        for group_index in range(groups):
            for episode_index in range(per_group):
                out.append(
                    SimpleNamespace(
                        namespaced_episode_id=f"cot32k::ep{group_index}_{episode_index}",
                        scope="cot",
                        source_tag="cot32k",
                        group_id=f"g{group_index}",
                        group_hash=f"h{group_index}",
                        split=SPLIT_TRAIN,
                        source_split=SPLIT_TRAIN,
                        strict_final=(episode_index == 0),
                    )
                )
        return out

    def test_whole_group_disjoint_and_deterministic(self) -> None:
        episodes = self.synthetic_episodes()
        first, audit = D.assign_group_partitions(episodes, enforce_expected_counts=False)
        second, _ = D.assign_group_partitions(episodes, enforce_expected_counts=False)
        self.assertEqual(first, second)
        seen = set()
        for partition in PARTITIONS:
            groups = set(first[partition])
            self.assertFalse(groups & seen)
            seen |= groups
        self.assertEqual(seen, {f"g{index}" for index in range(20)})
        self.assertTrue(audit["group_disjoint"])
        self.assertFalse(audit["dev_used"])

    def test_group_split_does_not_cut_groups(self) -> None:
        episodes = self.synthetic_episodes(groups=10, per_group=4)
        partitions, _ = D.assign_group_partitions(episodes, enforce_expected_counts=False)
        for partition, group_ids in partitions.items():
            group_set = set(group_ids)
            selected = [episode for episode in episodes if episode.group_id in group_set]
            self.assertEqual(len(selected), 4 * len(group_set))


class SyntheticBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_root = D._ROOT_FOR_BUILD
        D._ROOT_FOR_BUILD = ROOT

    def tearDown(self) -> None:
        D._ROOT_FOR_BUILD = self._old_root

    def test_zero_ct_and_unknown_positions_are_retained(self) -> None:
        episode = fake_episode()
        ct = source_row("cot32k::ep1::r1::tool_call::0", event_type="tool_call", token_index=10, status="unknown")
        e = source_row("cot32k::ep1::r1::round_end::0", event_type="round_end", token_index=11, index=0)
        provider = FakeProvider({ct["event_id"]: SPLIT_DEV, e["event_id"]: SPLIT_DEV})
        build = D._build_split(
            split=SPLIT_DEV,
            cohort_episodes=[episode],
            repaired_by_episode={"cot32k::ep1": [ct, e]},
            e_by_episode={},
            provider=provider,
            cache_offset=0,
            row_offset=0,
            partition_by_episode=None,
            enforce_expected_counts=False,
        )
        self.assertEqual(len(build.rows), 2)
        self.assertEqual(len(build.episodes), 1)
        self.assertEqual(build.episodes[0]["ct_count"], 1)
        self.assertEqual(build.episodes[0]["e_count"], 1)
        self.assertFalse(build.episodes[0]["zero_ct"])
        self.assertEqual(int(build.label_arrays["prefix_state"][0]), L.MISSING)
        self.assertFalse(bool(build.label_arrays["exact"][0]))
        self.assertEqual(int(build.label_arrays["prefix_state"][1]), 0)

        only_e = source_row("cot32k::ep2::r1::round_end::0", event_type="round_end", token_index=5)
        only_e["namespaced_episode_id"] = "cot32k::ep2"
        only_e["episode_id"] = "cot32k::ep2"
        only_e["group_id"] = "g2"
        only_e["group_hash"] = "h2"
        only_e["feature_ref"]["event_id"] = only_e["event_id"]
        episode2 = SimpleNamespace(
            namespaced_episode_id="cot32k::ep2",
            scope="cot",
            source_tag="cot32k",
            group_id="g2",
            group_hash="h2",
            split=SPLIT_DEV,
            source_split=SPLIT_DEV,
            strict_final=False,
        )
        provider2 = FakeProvider({only_e["event_id"]: SPLIT_DEV})
        build2 = D._build_split(
            split=SPLIT_DEV,
            cohort_episodes=[episode2],
            repaired_by_episode={"cot32k::ep2": [only_e]},
            e_by_episode={},
            provider=provider2,
            cache_offset=0,
            row_offset=0,
            partition_by_episode=None,
            enforce_expected_counts=False,
        )
        self.assertEqual(len(build2.rows), 1)
        self.assertTrue(build2.episodes[0]["zero_ct"])
        self.assertEqual(build2.episodes[0]["ct_count"], 0)

    def test_forged_test_ref_rejected_before_read(self) -> None:
        episode = fake_episode()
        row = source_row("cot32k::ep1::r1::tool_call::0")
        provider = FakeProvider({row["event_id"]: "test"})
        with self.assertRaises(DataContractError):
            D._build_split(
                split=SPLIT_DEV,
                cohort_episodes=[episode],
                repaired_by_episode={"cot32k::ep1": [row]},
                e_by_episode={},
                provider=provider,
                cache_offset=0,
                row_offset=0,
                partition_by_episode=None,
                enforce_expected_counts=False,
            )
        self.assertEqual(provider.reads, [])

    def test_forged_split_tag_rejected_before_read(self) -> None:
        episode = fake_episode()
        row = source_row("cot32k::ep1::r1::tool_call::0")
        row["feature_ref"]["split"] = SPLIT_TRAIN
        provider = FakeProvider({row["event_id"]: SPLIT_DEV})
        with self.assertRaises(DataContractError):
            D._build_split(
                split=SPLIT_DEV,
                cohort_episodes=[episode],
                repaired_by_episode={"cot32k::ep1": [row]},
                e_by_episode={},
                provider=provider,
                cache_offset=0,
                row_offset=0,
                partition_by_episode=None,
                enforce_expected_counts=False,
            )
        self.assertEqual(provider.reads, [])

    def test_episode_identity_mismatch_rejected(self) -> None:
        episode = fake_episode()
        row = source_row("cot32k::ep1::r1::tool_call::0")
        row["group_id"] = "forged-group"
        provider = FakeProvider({row["event_id"]: SPLIT_DEV})
        with self.assertRaises(DataContractError):
            D._build_split(
                split=SPLIT_DEV,
                cohort_episodes=[episode],
                repaired_by_episode={"cot32k::ep1": [row]},
                e_by_episode={},
                provider=provider,
                cache_offset=0,
                row_offset=0,
                partition_by_episode=None,
                enforce_expected_counts=False,
            )


class ScalerLeakageTests(unittest.TestCase):
    def test_heldout_perturbation_cannot_change_fit_scaler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rng = np.random.default_rng(7)
            values = rng.normal(size=(8, HIDDEN_SIZE)).astype(np.float32)
            matrix_a = tmp_path / "a.bin"
            matrix_a.write_bytes(D._float32_to_bf16_bytes(values))
            fit_rows = [0, 1, 2, 3]
            mean_a, std_a, audit_a = D._fit_scaler_from_matrix(matrix_a, total_rows=8, cache_rows=fit_rows)

            heldout_perturbed = values.copy()
            heldout_perturbed[4:] += 1000.0
            matrix_b = tmp_path / "b.bin"
            matrix_b.write_bytes(D._float32_to_bf16_bytes(heldout_perturbed))
            mean_b, std_b, audit_b = D._fit_scaler_from_matrix(matrix_b, total_rows=8, cache_rows=fit_rows)
            self.assertTrue(np.array_equal(mean_a, mean_b))
            self.assertTrue(np.array_equal(std_a, std_b))
            self.assertEqual(audit_a["mean_sha256"], audit_b["mean_sha256"])
            self.assertEqual(audit_a["std_sha256"], audit_b["std_sha256"])

            mean_c, std_c, _ = D._fit_scaler_from_matrix(
                matrix_b, total_rows=8, cache_rows=list(range(8))
            )
            self.assertFalse(np.array_equal(mean_a, mean_c))
            self.assertFalse(np.array_equal(std_a, std_c))

    def test_scaler_std_floor_and_finiteness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.bin"
            values = np.zeros((2, HIDDEN_SIZE), dtype=np.float32)
            path.write_bytes(D._float32_to_bf16_bytes(values))
            mean, std, audit = D._fit_scaler_from_matrix(path, total_rows=2, cache_rows=[0, 1])
            self.assertTrue(np.isfinite(mean).all())
            self.assertTrue(np.isfinite(std).all())
            self.assertTrue(np.all(std >= D.STD_FLOOR))
            self.assertEqual(audit["nonfinite_values"], 0)


class ManifestCorruptionTests(unittest.TestCase):
    def test_manifest_rejects_corrupt_file_and_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data_file = directory / "a.txt"
            data_file.write_text("hello", encoding="utf-8")
            matrix_file = directory / "features_bf16.bin"
            matrix_file.write_bytes(b"\x00" * ROW_BYTES)
            manifest = {
                "schema_version": 1,
                "dataset_version": "sequence_v1",
                "status": "complete",
                "files": {
                    "a.txt": {
                        "sha256": D.sha256_file(data_file),
                        "bytes": data_file.stat().st_size,
                    }
                },
                "matrix": {
                    "file": "features_bf16.bin",
                    "sha256": D.sha256_file(matrix_file),
                    "shape": [1, HIDDEN_SIZE],
                },
            }
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded = D._load_manifest(directory, verify_integrity=True)
            self.assertEqual(loaded["status"], "complete")

            data_file.write_text("tampered", encoding="utf-8")
            with self.assertRaises(DatasetError):
                D._load_manifest(directory, verify_integrity=True)

    def test_manifest_rejects_corrupt_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            data_file = directory / "a.txt"
            data_file.write_text("hello", encoding="utf-8")
            matrix_file = directory / "features_bf16.bin"
            matrix_file.write_bytes(b"\x00" * ROW_BYTES)
            manifest = {
                "schema_version": 1,
                "dataset_version": "sequence_v1",
                "status": "complete",
                "files": {
                    "a.txt": {"sha256": D.sha256_file(data_file), "bytes": data_file.stat().st_size}
                },
                "matrix": {
                    "file": "features_bf16.bin",
                    "sha256": D.sha256_file(matrix_file),
                    "shape": [1, HIDDEN_SIZE],
                },
            }
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            matrix_file.write_bytes(b"\x01" * ROW_BYTES)
            with self.assertRaises(DatasetError):
                D._load_manifest(directory, verify_integrity=True)


class ActualReaderBoundaryTests(unittest.TestCase):
    def test_real_feature_store_rejects_forged_split_tag(self) -> None:
        provider = D.FeatureStoreProvider(ROOT)
        dev_rows = [json.loads(line) for line in (ROOT / "artifacts/fine_label_repair_v1/data/repaired_dev_rows.jsonl").read_text(encoding="utf-8").splitlines()[:1]]
        row = dev_rows[0]
        forged = dict(row["feature_ref"])
        forged["split"] = SPLIT_TRAIN
        with self.assertRaises(Exception):
            provider.resolve_ref(forged, SPLIT_TRAIN)
        resolved = provider.resolve_ref(row["feature_ref"], SPLIT_DEV)
        self.assertEqual(resolved.event_id, row["event_id"])


@unittest.skipUnless((DATASET / "manifest.json").is_file(), "materialised dataset is not built yet")
class MaterialisedContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = D.open_sequence_dataset(ROOT, verify_integrity=True)

    def test_populations_and_policy_arrays(self) -> None:
        dataset = self.dataset
        self.assertEqual(len(dataset.episodes_train), EXPECTED_TRAIN_EPISODES)
        self.assertEqual(len(dataset.episodes_dev), EXPECTED_DEV_EPISODES)
        self.assertEqual(len(dataset.rows_train), EXPECTED_TRAIN_CT_ROWS + EXPECTED_TRAIN_E_ROWS)
        self.assertEqual(len(dataset.rows_dev), EXPECTED_DEV_ROWS)
        self.assertEqual(tuple(dataset.matrix.shape), tuple(dataset.manifest["matrix"]["shape"]))
        self.assertEqual(dataset.matrix.shape[1], HIDDEN_SIZE)
        ct = np.asarray([bool(row["ct_mask"]) for row in dataset.rows_train])
        labels = dataset.labels_train
        for policy, expected in EXPECTED_POLICY_COUNTS.items():
            label = labels[f"label_{policy}"]
            mask = labels[f"mask_{policy}"]
            exact = labels["exact"]
            self.assertEqual(int(np.sum((label == 1) & ~mask & ct)), expected["positive"], policy)
            self.assertEqual(int(np.sum((label == 0) & ~mask & ct)), expected["negative"], policy)
            self.assertEqual(int(np.sum(mask & exact & ct)), expected["masked_exact"], policy)
            self.assertEqual(int(np.sum(mask & ~exact & ct)), expected["unknown"], policy)
        self.assertTrue(np.array_equal(labels["prefix_state"][ct], labels["label_L3"][ct]))
        self.assertTrue(np.all(labels["label_L0"][ct & ~labels["exact"]] == L.MISSING))

    def test_scalers_and_norms(self) -> None:
        dataset = self.dataset
        for name in ("fit", "full_train"):
            scaler = dataset.scaler(name)
            self.assertEqual(scaler.mean.shape, (HIDDEN_SIZE,))
            self.assertEqual(scaler.std.shape, (HIDDEN_SIZE,))
            self.assertTrue(np.isfinite(scaler.mean).all())
            self.assertTrue(np.isfinite(scaler.std).all())
            self.assertTrue(np.all(scaler.std >= scaler.eps))
        sample = dataset.rows[:: max(1, len(dataset.rows) // 64)][:64]
        values = dataset.read_rows([row["row_index"] for row in sample], dtype="float32").numpy()
        norms = np.linalg.norm(values.astype(np.float64), axis=1)
        self.assertTrue(np.isfinite(norms).all())
        self.assertGreater(float(norms.max()), 0.0)

    def test_collate_shapes_padding_and_labels(self) -> None:
        dataset = self.dataset
        indices = dataset.split_episode_indices("fit")[:2]
        batch = dataset.collate_episodes(indices, dtype="bfloat16", include_labels=True)
        self.assertEqual(batch.features.shape[0], len(indices))
        self.assertEqual(batch.features.shape[2], HIDDEN_SIZE)
        self.assertEqual(batch.features.dtype, __import__("torch").bfloat16)
        self.assertTrue(bool(batch.padding_mask[:, 0].all()))
        for batch_index, length in enumerate(batch.lengths.tolist()):
            self.assertTrue(bool(batch.padding_mask[batch_index, :length].all()))
            self.assertFalse(bool(batch.padding_mask[batch_index, length:].any()))
            for policy in EXPECTED_POLICY_COUNTS:
                self.assertTrue(bool((batch.policy_labels[policy][batch_index, length:] == L.MISSING).all()))
                self.assertTrue(bool(batch.policy_masks[policy][batch_index, length:].all()))
            self.assertTrue(bool((batch.prefix_state[batch_index, length:] == L.MISSING).all()))
        self.assertTrue(set(batch.policy_labels) == set(EXPECTED_POLICY_COUNTS))
        self.assertEqual(batch.bag_label.shape, (len(indices),))

    def test_dev_collate_uses_local_label_slices(self) -> None:
        dataset = self.dataset
        indices = dataset.split_episode_indices("dev")[:2]
        batch = dataset.collate_episodes(indices, dtype="float32", include_labels=True)
        self.assertEqual(batch.split, "dev")
        self.assertEqual(batch.policy_labels, {})
        self.assertEqual(batch.prefix_state.shape[0], len(indices))
        for batch_index, length in enumerate(batch.lengths.tolist()):
            start = int(dataset.episodes[indices[batch_index]]["row_start"]) - len(dataset.rows_train)
            end = int(dataset.episodes[indices[batch_index]]["row_end"]) - len(dataset.rows_train)
            self.assertTrue(
                np.array_equal(
                    batch.prefix_state[batch_index, :length].numpy(),
                    dataset.labels_dev["prefix_state"][start:end],
                )
            )
            self.assertTrue(bool((batch.prefix_state[batch_index, length:] == L.MISSING).all()))

    def test_reader_is_read_only(self) -> None:
        before = {
            path.relative_to(DATASET): path.stat().st_mtime_ns
            for path in sorted(DATASET.rglob("*"))
            if path.is_file()
        }
        dataset = D.open_sequence_dataset(ROOT, verify_integrity=False)
        dataset.summary()
        dataset.collate_episodes(dataset.split_episode_indices("fit")[:2])
        dataset.read_rows([0, 1, 2])
        after = {
            path.relative_to(DATASET): path.stat().st_mtime_ns
            for path in sorted(DATASET.rglob("*"))
            if path.is_file()
        }
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
