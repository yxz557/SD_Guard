"""Reusable, resumable training/validation harness for fine_sequence_explore_v1 (Lane T).

Design constraints deliberately encoded here:

* Cached feature rows are materialised once on the GPU (or CPU in tests); episode
  batches are padded only to the batch's longest episode, never to the corpus maximum.
* FIT internal runs use the FIT-only scaler and stop/select on SELECT empirical
  FNR at <=5% FPR.  Full-TRAIN refits use the separately named FULL-TRAIN scaler,
  fixed epochs and never early-stop on DEV.
* Sampler and model dropout use separate RNGs; paired FIT seeds therefore expose the
  same episode draws across architecture arms when sampler settings are unchanged.
* A completed run is immutable.  Its frozen config, source snapshot, input identity and
  checkpoint tensors can be validated without constructing a model or rescoring.
* DEV and AgentLAB are scored once for the selected FIT checkpoint.  CAL thresholds
  are reported separately and never masquerade as independent calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from . import evaluation
from .model import ModelConfig, build_model, count_parameters
from .objectives import ObjectiveConfig, compute_objective

ROOT = Path(__file__).resolve().parents[2]
HANDOFF_PATH = ROOT / "artifacts/fine_sequence_explore_v1/handoffs/data.json"
MISSING = -1
POLICIES = ("L0", "L1", "L2", "L3", "L4")
MODES = ("fit", "full_train", "smoke")
PRODUCER_FILES = (
    "src/sdsafe_features/common.py",
    "src/sdsafe_fine_confirm/__init__.py",
    "src/sdsafe_fine_confirm/sources.py",
    "src/sdsafe_fine_confirm/train_data.py",
    "src/sdsafe_fine_label_repair/__init__.py",
    "src/sdsafe_fine_label_repair/data.py",
    "src/sdsafe_fine_label_repair/features.py",
    "src/sdsafe_fine_label_repair/labels.py",
    "src/sdsafe_sequence_explore/__init__.py",
    "src/sdsafe_sequence_explore/data.py",
    "src/sdsafe_sequence_explore/labels.py",
    "src/sdsafe_sequence_explore/model.py",
    "src/sdsafe_sequence_explore/objectives.py",
    "src/sdsafe_sequence_explore/train.py",
    "src/sdsafe_sequence_explore/evaluation.py",
    "tests/test_sequence_explore_model.py",
    "tests/test_sequence_explore_objectives.py",
    "tests/test_sequence_explore_train.py",
    "tests/test_sequence_explore_evaluation.py",
)
FUTURE_CONSUMER_MODULES = (
    "src/sdsafe_sequence_explore/cli.py",
    "src/sdsafe_sequence_explore/__main__.py",
    "src/sdsafe_sequence_explore/search.py",
    "src/sdsafe_sequence_explore/report.py",
)


# --------------------------------------------------------------------------- utils


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256_bytes(payload)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _seed_from_text(text: str) -> int:
    return int(sha256_bytes(text.encode("utf-8"))[:16], 16) % (2**63 - 1)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TrainConfig:
    run_id: str = "run"
    mode: str = "fit"
    seed: int = 1
    epochs: int = 40
    max_epochs: int = 40
    patience: int = 7
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    balanced: bool = True
    pos_neg_ratio: float = 3.0
    bucket_by_length: bool = True
    scaler: str = "fit"
    selection_split: str = "select"
    eval_dev: bool = True
    eval_cal: bool = True
    max_train_episodes: int | None = None
    max_steps_per_epoch: int | None = None
    eval_batch_size: int = 32
    tags: tuple[str, ...] = ()
    model: ModelConfig = field(default_factory=ModelConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode == "fit" and self.scaler != "fit":
            raise ValueError("FIT internal exploration must use the FIT-only scaler")
        if self.mode == "full_train" and self.scaler != "full_train":
            raise ValueError("full-TRAIN refits must use the full_train scaler explicitly")
        if self.scaler not in ("fit", "full_train"):
            raise ValueError(self.scaler)
        if self.epochs < 1 or self.max_epochs < 1:
            raise ValueError("epochs must be positive")
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.max_epochs > self.epochs:
            raise ValueError("max_epochs cannot exceed planned epochs")
        if self.objective.aux_weight > 0 and not self.model.aux_head:
            raise ValueError("aux_weight > 0 requires model.aux_head=True (separate head)")
        if self.mode == "full_train" and self.max_train_episodes is not None:
            raise ValueError("full_train must use all authoritative TRAIN episodes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "seed": int(self.seed),
            "epochs": int(self.epochs),
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "batch_size": int(self.batch_size),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "grad_clip": float(self.grad_clip),
            "balanced": bool(self.balanced),
            "pos_neg_ratio": float(self.pos_neg_ratio),
            "bucket_by_length": bool(self.bucket_by_length),
            "scaler": self.scaler,
            "selection_split": self.selection_split,
            "eval_dev": bool(self.eval_dev),
            "eval_cal": bool(self.eval_cal),
            "max_train_episodes": self.max_train_episodes,
            "max_steps_per_epoch": self.max_steps_per_epoch,
            "eval_batch_size": int(self.eval_batch_size),
            "tags": list(self.tags),
            "model": self.model.to_dict(),
            "objective": self.objective.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
        values = dict(data)
        values["model"] = ModelConfig.from_dict(values["model"])
        values["objective"] = ObjectiveConfig.from_dict(values["objective"])
        values["tags"] = tuple(values.get("tags", ()))
        return cls(**values)

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


# --------------------------------------------------------------------------- inputs


@dataclass
class TrainingInputs:
    dataset_dir: str
    device: torch.device
    features_bf16: torch.Tensor
    row_ct_mask: torch.Tensor
    row_event_family: torch.Tensor
    row_round_id: torch.Tensor
    row_token_index: torch.Tensor
    row_token_gap: torch.Tensor
    row_round_gap: torch.Tensor
    row_policy_labels: dict[str, torch.Tensor]
    row_policy_masks: dict[str, torch.Tensor]
    row_prefix_state: torch.Tensor
    row_exact: torch.Tensor
    episode_row_start: torch.Tensor
    episode_length: torch.Tensor
    episode_bag_label: torch.Tensor
    episode_global_index: list[int]
    episode_ids: list[str]
    episode_group_id: list[str]
    episode_scope: list[str]
    episode_split: list[str]
    episode_partition: list[str]
    scaler_states: dict[str, dict[str, Any]]
    identity: dict[str, Any]
    _scaled: dict[str, torch.Tensor] = field(default_factory=dict)
    _global_to_local: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._global_to_local:
            self._global_to_local = {int(global_index): local for local, global_index in enumerate(self.episode_global_index)}

    @property
    def num_rows(self) -> int:
        return int(self.features_bf16.shape[0])

    @property
    def num_episodes(self) -> int:
        return len(self.episode_global_index)

    def scaler_tensors(self, name: str = "fit") -> tuple[torch.Tensor, torch.Tensor]:
        state = self.scaler_states[name]
        mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=self.device)
        std = torch.as_tensor(state["std"], dtype=torch.float32, device=self.device)
        return mean, std

    def scaled_features(self, name: str = "fit") -> torch.Tensor:
        if name not in self._scaled:
            mean, std = self.scaler_tensors(name)
            raw = self.features_bf16.to(torch.float32)
            self._scaled[name] = (raw - mean) / std
        return self._scaled[name]

    def episode_indices(self, name: str) -> list[int]:
        if name in ("train", "dev"):
            return [g for g, split in zip(self.episode_global_index, self.episode_split) if split == name]
        if name in ("fit", "cal", "select"):
            return [g for g, part in zip(self.episode_global_index, self.episode_partition) if part == name]
        raise ValueError(f"unknown split/partition {name!r}")

    def episode_batch(
        self,
        episode_indices: Sequence[int],
        *,
        scaler_name: str = "fit",
        include_labels: bool = True,
        allow_mixed_train_partitions: bool = False,
    ) -> SimpleNamespace:
        if not episode_indices:
            raise ValueError("empty episode batch")
        global_indices = [int(index) for index in episode_indices]
        local = [self._global_to_local[g] for g in global_indices]
        splits = {self.episode_split[index] for index in local}
        if len(splits) != 1:
            raise ValueError("cannot mix TRAIN and DEV episodes in one batch")
        if splits == {"train"}:
            partitions = {self.episode_partition[index] for index in local}
            if len(partitions) > 1 and not (
                allow_mixed_train_partitions and scaler_name == "full_train"
            ):
                raise ValueError(
                    "mixed FIT/CAL/SELECT batch requires explicit all-TRAIN full_train refit mode"
                )
        lengths = torch.tensor([int(self.episode_length[i].item()) for i in local], dtype=torch.int64)
        max_len = max(1, int(lengths.max().item()))
        batch_size = len(global_indices)
        feature_dim = int(self.features_bf16.shape[1])
        features = torch.zeros((batch_size, max_len, feature_dim), dtype=torch.float32, device=self.device)
        valid = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        ct = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        event_family = torch.zeros((batch_size, max_len), dtype=torch.int64, device=self.device)
        round_id = torch.zeros((batch_size, max_len), dtype=torch.int64, device=self.device)
        token_index = torch.zeros((batch_size, max_len), dtype=torch.int64, device=self.device)
        token_gap = torch.zeros((batch_size, max_len), dtype=torch.int64, device=self.device)
        round_gap = torch.zeros((batch_size, max_len), dtype=torch.int64, device=self.device)
        prefix_state = torch.full((batch_size, max_len), MISSING, dtype=torch.int8, device=self.device)
        exact = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        policy_labels = {
            policy: torch.full((batch_size, max_len), MISSING, dtype=torch.int8, device=self.device)
            for policy in self.row_policy_labels
        }
        policy_masks = {
            policy: torch.ones((batch_size, max_len), dtype=torch.bool, device=self.device)
            for policy in self.row_policy_masks
        }
        scaled = self.scaled_features(scaler_name)
        for row, local_index in enumerate(local):
            length = int(self.episode_length[local_index].item())
            if length == 0:
                continue
            start = int(self.episode_row_start[local_index].item())
            stop = start + length
            features[row, :length] = scaled[start:stop]
            valid[row, :length] = True
            ct[row, :length] = self.row_ct_mask[start:stop]
            event_family[row, :length] = self.row_event_family[start:stop]
            round_id[row, :length] = self.row_round_id[start:stop]
            token_index[row, :length] = self.row_token_index[start:stop]
            token_gap[row, :length] = self.row_token_gap[start:stop]
            round_gap[row, :length] = self.row_round_gap[start:stop]
            prefix_state[row, :length] = self.row_prefix_state[start:stop]
            exact[row, :length] = self.row_exact[start:stop]
            for policy in policy_labels:
                policy_labels[policy][row, :length] = self.row_policy_labels[policy][start:stop]
                policy_masks[policy][row, :length] = self.row_policy_masks[policy][start:stop]
        scalers = self.scaler_states[scaler_name]
        return SimpleNamespace(
            features=features,
            padding_mask=valid,
            ct_mask=ct,
            event_family=event_family,
            round_id=round_id,
            token_index=token_index,
            token_gap=token_gap,
            round_gap=round_gap,
            lengths=lengths.to(self.device),
            policy_labels=policy_labels if include_labels else {},
            policy_masks=policy_masks if include_labels else {},
            prefix_state=prefix_state if include_labels else None,
            exact=exact if include_labels else None,
            bag_label=self.episode_bag_label[local].to(self.device) if include_labels else None,
            episode_ids=[self.episode_ids[i] for i in local],
            group_ids=[self.episode_group_id[i] for i in local],
            scopes=[self.episode_scope[i] for i in local],
            split=self.episode_split[local[0]],
            partition=self.episode_partition[local[0]],
            global_indices=global_indices,
            row_start=torch.tensor([int(self.episode_row_start[i].item()) for i in local], dtype=torch.int64),
            row_end=torch.tensor(
                [int(self.episode_row_start[i].item() + self.episode_length[i].item()) for i in local],
                dtype=torch.int64,
            ),
            scaler_name=scaler_name,
            scaler_mean_sha256=scalers.get("mean_sha256"),
            scaler_std_sha256=scalers.get("std_sha256"),
        )


def _gather_bf16(dataset: Any, row_indices: Sequence[int], device: torch.device) -> torch.Tensor:
    if not row_indices:
        return torch.zeros((0, int(dataset.matrix.shape[1])), dtype=torch.bfloat16, device=device)
    cache_rows = [int(dataset.rows[int(index)]["cache_row"]) for index in row_indices]
    chunks: list[torch.Tensor] = []
    for start in range(0, len(cache_rows), 32768):
        block = np.asarray(dataset.matrix[np.asarray(cache_rows[start : start + 32768], dtype=np.int64)])
        chunks.append(torch.from_numpy(block).view(torch.bfloat16).to(device))
    return torch.cat(chunks, dim=0)


def _episode_order(dataset: Any, episode_indices: Sequence[int] | None) -> list[int]:
    if episode_indices is None:
        return list(range(len(dataset.episodes)))
    indices = [int(index) for index in episode_indices]
    if len(set(indices)) != len(indices):
        raise ValueError("episode subset contains duplicates")
    for index in indices:
        if index < 0 or index >= len(dataset.episodes):
            raise ValueError("episode index out of range")
    return indices


def build_training_inputs(
    dataset: Any,
    device: torch.device | str = "cpu",
    episode_indices: Sequence[int] | None = None,
    *,
    bindings: Mapping[str, Any] | None = None,
) -> TrainingInputs:
    """Materialise a read-only flat row cache and episode index for training.

    ``episode_indices=None`` caches every episode (TRAIN then DEV).  A subset is useful
    for focused tests and always keeps row/label alignment intact.
    """
    device = torch.device(device)
    order = _episode_order(dataset, episode_indices)
    global_rows: list[int] = []
    row_start: list[int] = []
    row_length: list[int] = []
    for episode_index in order:
        episode = dataset.episodes[episode_index]
        start, end = int(episode["row_start"]), int(episode["row_end"])
        row_start.append(len(global_rows))
        row_length.append(end - start)
        global_rows.extend(range(start, end))
    features = _gather_bf16(dataset, global_rows, device)
    offsets = {"train": 0, "dev": len(dataset.rows_train)}
    rows = [dataset.rows[index] for index in global_rows]
    n = len(rows)
    row_ct = np.zeros(n, dtype=np.bool_)
    row_family = np.zeros(n, dtype=np.int64)
    row_round = np.zeros(n, dtype=np.int64)
    row_token = np.zeros(n, dtype=np.int64)
    row_token_gap = np.zeros(n, dtype=np.int64)
    row_round_gap = np.zeros(n, dtype=np.int64)
    row_prefix = np.full(n, MISSING, dtype=np.int8)
    row_exact = np.zeros(n, dtype=np.bool_)
    policy_labels_np = {policy: np.full(n, MISSING, dtype=np.int8) for policy in POLICIES}
    policy_masks_np = {policy: np.ones(n, dtype=np.bool_) for policy in POLICIES}
    labels_by_split = {"train": dataset.labels_train, "dev": dataset.labels_dev}
    for position, row in enumerate(rows):
        split = str(row["split"])
        local_row = int(row["row_index"]) - offsets[split]
        labels = labels_by_split[split]
        row_ct[position] = bool(row["ct_mask"])
        row_family[position] = 0 if str(row["event_family"]) == "ct" else 1
        row_round[position] = int(row["round_id"])
        row_token[position] = int(row["feature_token_index"])
        row_token_gap[position] = int(row["token_gap"])
        row_round_gap[position] = int(row["round_gap"])
        row_prefix[position] = int(labels["prefix_state"][local_row])
        row_exact[position] = bool(labels["exact"][local_row])
        if split == "train":
            for policy in POLICIES:
                policy_labels_np[policy][position] = int(labels[f"label_{policy}"][local_row])
                policy_masks_np[policy][position] = bool(labels[f"mask_{policy}"][local_row])

    def tensor(values: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(values, dtype=dtype, device=device)

    scaler_states: dict[str, dict[str, Any]] = {}
    for name in ("fit", "full_train"):
        state = dataset.scaler(name).state_dict()
        scaler_states[name] = state
        scaler_states[name]["file_sha256"] = sha256_file(
            Path(dataset.dataset_dir) / "scalers" / ("fit_scaler.npz" if name == "fit" else "full_train_scaler.npz")
        )
    source_ledger_path = Path(dataset.dataset_dir) / "source_ledger.json"
    source_ledger = _read_json(source_ledger_path)
    manifest_path = Path(dataset.dataset_dir) / "manifest.json"
    identity = {
        "dataset_dir": _relative(Path(dataset.dataset_dir)),
        "dataset_version": str(dataset.manifest.get("dataset_version")),
        "manifest_sha256": sha256_file(manifest_path),
        "matrix_sha256": str(dataset.manifest["matrix"]["sha256"]),
        "matrix_shape": list(dataset.manifest["matrix"]["shape"]),
        "source_ledger_sha256": sha256_file(source_ledger_path),
        "source_ledger_content_sha256": str(source_ledger["ledger_sha256"]),
        "scaler_file_sha256": {name: scaler_states[name]["file_sha256"] for name in scaler_states},
        "scaler_state_sha256": {
            name: {
                "mean": str(scaler_states[name]["mean_sha256"]),
                "std": str(scaler_states[name]["std_sha256"]),
            }
            for name in scaler_states
        },
        "episodes": len(order),
        "rows": n,
        "episode_global_indices_sha256": canonical_sha256(order),
        "bindings_verified": bool(bindings and bindings.get("status") == "PASS"),
        "bindings": dict(bindings) if bindings is not None else {},
    }
    episode_bag = np.asarray([float(dataset.episodes[index]["bag_label"]) for index in order], dtype=np.float32)
    return TrainingInputs(
        dataset_dir=str(dataset.dataset_dir),
        device=device,
        features_bf16=features,
        row_ct_mask=tensor(row_ct, torch.bool),
        row_event_family=tensor(row_family, torch.int64),
        row_round_id=tensor(row_round, torch.int64),
        row_token_index=tensor(row_token, torch.int64),
        row_token_gap=tensor(row_token_gap, torch.int64),
        row_round_gap=tensor(row_round_gap, torch.int64),
        row_policy_labels={policy: tensor(values, torch.int8) for policy, values in policy_labels_np.items()},
        row_policy_masks={policy: tensor(values, torch.bool) for policy, values in policy_masks_np.items()},
        row_prefix_state=tensor(row_prefix, torch.int8),
        row_exact=tensor(row_exact, torch.bool),
        episode_row_start=torch.tensor(row_start, dtype=torch.int64),
        episode_length=torch.tensor(row_length, dtype=torch.int64),
        episode_bag_label=tensor(episode_bag, torch.float32),
        episode_global_index=list(order),
        episode_ids=[str(dataset.episodes[index]["episode_id"]) for index in order],
        episode_group_id=[str(dataset.episodes[index]["group_id"]) for index in order],
        episode_scope=[str(dataset.episodes[index]["scope"]) for index in order],
        episode_split=[str(dataset.episodes[index]["split"]) for index in order],
        episode_partition=[str(dataset.episodes[index].get("partition", "")) for index in order],
        scaler_states=scaler_states,
        identity=identity,
    )


def verify_input_bindings(root: Path | str = ROOT, *, hash_inputs: bool = True) -> dict[str, Any]:
    """Read-only check of accepted D manifest/source/input identities."""
    root = Path(root)
    handoff = _read_json(root / "artifacts/fine_sequence_explore_v1/handoffs/data.json")
    requirements = handoff["preload_requirements"]
    dataset_dir = root / handoff["dataset_dir"]
    manifest_sha = sha256_file(dataset_dir / "manifest.json")
    source_ledger_path = dataset_dir / "source_ledger.json"
    source_ledger = _read_json(source_ledger_path)
    source_ledger_sha = sha256_file(source_ledger_path)
    checks: dict[str, Any] = {
        "manifest_sha256": manifest_sha,
        "expected_manifest_sha256": requirements["final_manifest_sha256"],
        "manifest_match": manifest_sha == requirements["final_manifest_sha256"],
        "source_ledger_sha256": source_ledger_sha,
        "expected_source_ledger_sha256": requirements["source_ledger_sha256"],
        "source_ledger_match": source_ledger_sha == requirements["source_ledger_sha256"],
        "source_ledger_content_sha256": source_ledger["ledger_sha256"],
        "expected_source_ledger_content_sha256": requirements["source_ledger_content_sha256"],
        "source_ledger_content_match": source_ledger["ledger_sha256"] == requirements["source_ledger_content_sha256"],
    }
    source_mismatches: dict[str, Any] = {}
    for relative, expected in source_ledger["files"].items():
        path = root / relative
        if not path.is_file():
            source_mismatches[relative] = "missing"
            continue
        actual = sha256_file(path)
        if actual != expected["sha256"] or path.stat().st_size != int(expected["bytes"]):
            source_mismatches[relative] = {"actual": actual, "expected": expected["sha256"]}
    checks["source_file_mismatches"] = source_mismatches
    input_mismatches: dict[str, Any] = {}
    if hash_inputs:
        for name, path_value in handoff.get("inputs", {}).items():
            path = root / str(path_value["path"])
            expected = str(path_value["sha256"])
            if not path.is_file():
                input_mismatches[name] = "missing"
            else:
                actual = sha256_file(path)
                if actual != expected:
                    input_mismatches[name] = {"actual": actual, "expected": expected}
    checks["input_mismatches"] = input_mismatches
    checks["hash_inputs"] = bool(hash_inputs)
    checks["status"] = (
        "PASS"
        if checks["manifest_match"]
        and checks["source_ledger_match"]
        and checks["source_ledger_content_match"]
        and not source_mismatches
        and not input_mismatches
        else "FAIL"
    )
    return checks


def preload_inputs(
    dataset: Any,
    device: torch.device | str = "cpu",
    *,
    verify_bindings: bool = True,
    episode_indices: Sequence[int] | None = None,
) -> TrainingInputs:
    bindings = verify_input_bindings() if verify_bindings else {"status": "SKIPPED", "hash_inputs": False}
    if verify_bindings and bindings["status"] != "PASS":
        raise RuntimeError(f"accepted D input/source binding check failed: {bindings}")
    return build_training_inputs(dataset, device, episode_indices, bindings=bindings)


# --------------------------------------------------------------------------- sampler


def _episode_length(inputs: TrainingInputs, global_index: int) -> int:
    return int(inputs.episode_length[inputs._global_to_local[int(global_index)]].item())


def make_batches(
    episode_indices: Sequence[int],
    batch_size: int,
    rng: np.random.Generator | None = None,
    *,
    inputs: TrainingInputs | None = None,
    bucket_by_length: bool = False,
) -> list[list[int]]:
    order = [int(index) for index in episode_indices]
    if rng is not None:
        rng.shuffle(order)
    if bucket_by_length and inputs is not None and order:
        window = max(1, int(batch_size) * 8)
        bucketed: list[int] = []
        for start in range(0, len(order), window):
            chunk = order[start : start + window]
            chunk.sort(key=lambda index: (_episode_length(inputs, index), index))
            bucketed.extend(chunk)
        order = bucketed
    return [order[start : start + int(batch_size)] for start in range(0, len(order), int(batch_size))]


def sample_epoch_indices(
    train_indices: Sequence[int],
    bag_labels: torch.Tensor,
    inputs: TrainingInputs,
    rng: np.random.Generator,
    *,
    balanced: bool,
    pos_neg_ratio: float,
) -> tuple[list[int], dict[str, Any]]:
    indices = [int(index) for index in train_indices]
    positive = [index for index in indices if float(bag_labels[inputs._global_to_local[index]].item()) > 0]
    negative = [index for index in indices if float(bag_labels[inputs._global_to_local[index]].item()) <= 0]
    if balanced and positive:
        needed = int(round(len(positive) * float(pos_neg_ratio)))
        if needed <= len(negative):
            chosen_negatives = list(rng.choice(np.asarray(negative, dtype=np.int64), size=needed, replace=False))
        else:
            chosen_negatives = list(rng.choice(np.asarray(negative, dtype=np.int64), size=needed, replace=True))
        draws = [int(x) for x in positive] + [int(x) for x in chosen_negatives]
    else:
        draws = list(indices)
    rng.shuffle(draws)
    unique = len(set(draws))
    stats = {
        "balanced": bool(balanced),
        "draws": len(draws),
        "unique": unique,
        "repeats": len(draws) - unique,
        "positive_draws": sum(
            1 for index in draws if float(bag_labels[inputs._global_to_local[index]].item()) > 0
        ),
        "negative_draws": sum(
            1 for index in draws if float(bag_labels[inputs._global_to_local[index]].item()) <= 0
        ),
        "train_pool": len(indices),
        "nominal_epochs": len(draws) / max(1, len(indices)),
        "pos_neg_ratio": float(pos_neg_ratio),
    }
    return draws, stats


# --------------------------------------------------------------------------- selector


def _clone_state(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _file_record(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _atomic_torch_save(payload: Any, path: Path | str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _scaler_sha256(mean: torch.Tensor, std: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for name, value in (("mean", mean), ("std", std)):
        tensor = value.detach().cpu().contiguous().to(torch.float32)
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _scaler_payload(scaler_state: Mapping[str, Any], name: str) -> dict[str, Any]:
    mean = torch.as_tensor(scaler_state["mean"], dtype=torch.float32).cpu().contiguous().clone()
    std = torch.as_tensor(scaler_state["std"], dtype=torch.float32).cpu().contiguous().clone()
    if mean.ndim != 1 or std.shape != mean.shape:
        raise ValueError("scaler mean/std must be same-shape 1-D tensors")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise ValueError("scaler tensors are non-finite")
    eps = float(scaler_state.get("eps", 1e-6))
    if bool((std < eps).any()):
        raise ValueError("scaler std is below its declared eps floor")
    return {
        "name": str(name),
        "mean": mean,
        "std": std,
        "fit_rows": int(scaler_state.get("fit_rows", 0)),
        "eps": eps,
        "file_sha256": str(scaler_state.get("file_sha256", "")),
        "mean_sha256": str(scaler_state.get("mean_sha256", sha256_bytes(mean.numpy().tobytes()))),
        "std_sha256": str(scaler_state.get("std_sha256", sha256_bytes(std.numpy().tobytes()))),
        "tensor_sha256": _scaler_sha256(mean, std),
    }


class ModelSelector:
    """Dual selector: primary FNR@5%-FPR tuple, secondary AUC; shared stopping."""

    def __init__(self, patience: int = 7) -> None:
        self.patience = int(patience)
        self.primary_key: tuple[float, float, float, int] | None = None
        self.primary_epoch: int | None = None
        self.primary_metrics: dict[str, Any] | None = None
        self.primary_state: dict[str, torch.Tensor] | None = None
        self.auc_key: tuple[float, int] | None = None
        self.auc_epoch: int | None = None
        self.auc_metrics: dict[str, Any] | None = None
        self.auc_state: dict[str, torch.Tensor] | None = None
        self.last_epoch: int = -1

    def update(self, epoch: int, metrics: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]) -> dict[str, bool]:
        point = metrics["primary"]
        auc = float(metrics.get("auc", float("nan")))
        auc_for_key = auc if math.isfinite(auc) else -1.0
        key = (float(point["fnr"]), float(point["fpr"]), -auc_for_key, int(epoch))
        improved_primary = self.primary_key is None or key < self.primary_key
        if improved_primary:
            self.primary_key = key
            self.primary_epoch = int(epoch)
            self.primary_metrics = dict(metrics)
            self.primary_state = _clone_state(state_dict)
        auc_key = (-auc_for_key, int(epoch))
        improved_auc = self.auc_key is None or auc_key < self.auc_key
        if improved_auc:
            self.auc_key = auc_key
            self.auc_epoch = int(epoch)
            self.auc_metrics = dict(metrics)
            self.auc_state = _clone_state(state_dict)
        self.last_epoch = int(epoch)
        return {"primary": improved_primary, "auc": improved_auc}

    def should_stop(self) -> bool:
        if self.primary_epoch is None or self.auc_epoch is None:
            return False
        return (self.last_epoch - self.primary_epoch >= self.patience) and (
            self.last_epoch - self.auc_epoch >= self.patience
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience": self.patience,
            "primary_key": self.primary_key,
            "primary_epoch": self.primary_epoch,
            "primary_metrics": self.primary_metrics,
            "primary_state": self.primary_state,
            "auc_key": self.auc_key,
            "auc_epoch": self.auc_epoch,
            "auc_metrics": self.auc_metrics,
            "auc_state": self.auc_state,
            "last_epoch": self.last_epoch,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ModelSelector":
        selector = cls(int(state["patience"]))
        for key, value in state.items():
            if key != "patience":
                setattr(selector, key, value)
        return selector


# --------------------------------------------------------------------------- producer fingerprint


def producer_fingerprint(root: Path | str = ROOT) -> dict[str, Any]:
    root = Path(root)
    files: dict[str, Any] = {}
    missing: list[str] = []
    for relative in PRODUCER_FILES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "files": files,
        "missing": missing,
        "future_consumer_modules_excluded": list(FUTURE_CONSUMER_MODULES),
        "fingerprint_sha256": canonical_sha256(files),
    }


def archive_source_snapshot(run_dir: Path | str, fingerprint: Mapping[str, Any] | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    fingerprint = dict(fingerprint or producer_fingerprint())
    snapshot_root = run_dir / "source_snapshot"
    if snapshot_root.exists():
        raise RuntimeError("source snapshot already exists; refusing to overwrite a frozen run")
    snapshot_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Any] = {}
    for relative, record in fingerprint["files"].items():
        source = ROOT / relative
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied[relative] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    payload = {
        "files": copied,
        "missing": list(fingerprint.get("missing", [])),
        "future_consumer_modules_excluded": list(FUTURE_CONSUMER_MODULES),
        "fingerprint_sha256": canonical_sha256(copied),
    }
    _write_json(run_dir / "source_snapshot.json", payload)
    return payload


def _verify_source_snapshot(run_dir: Path) -> dict[str, Any]:
    payload = _read_json(run_dir / "source_snapshot.json")
    mismatches: dict[str, Any] = {}
    for relative, record in payload["files"].items():
        path = run_dir / "source_snapshot" / relative
        if not path.is_file():
            mismatches[relative] = "missing"
            continue
        actual = sha256_file(path)
        if actual != record["sha256"] or path.stat().st_size != int(record["bytes"]):
            mismatches[relative] = {"actual": actual, "expected": record["sha256"]}
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "fingerprint_sha256": canonical_sha256(payload["files"]),
        "recorded_fingerprint_sha256": payload.get("fingerprint_sha256"),
    }


# --------------------------------------------------------------------------- scoring


def _model_inputs(model: nn.Module, batch: SimpleNamespace) -> tuple[Any, ...]:
    return (
        batch.features,
        batch.padding_mask,
        {
            "ct_mask": batch.ct_mask,
            "token_gap": batch.token_gap,
            "round_gap": batch.round_gap,
            "token_index": batch.token_index,
        },
    )


def score_episodes(
    model: nn.Module,
    inputs: TrainingInputs,
    episode_indices: Sequence[int],
    *,
    scaler_name: str = "fit",
    batch_size: int = 32,
    bucket_by_length: bool = True,
) -> dict[str, Any]:
    """Deterministic eval-mode scoring; no labels are model inputs."""
    indices = [int(index) for index in episode_indices]
    model.eval()
    chunks: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_indices in make_batches(
            indices, batch_size, None, inputs=inputs, bucket_by_length=bucket_by_length
        ):
            batch = inputs.episode_batch(batch_indices, scaler_name=scaler_name, include_labels=True)
            features, valid, kwargs = _model_inputs(model, batch)
            logits = model(features, valid, **kwargs)
            chunks.append(
                {
                    "logits": logits.detach().to(torch.float32).cpu(),
                    "valid": batch.padding_mask.detach().cpu(),
                    "ct": batch.ct_mask.detach().cpu(),
                    "prefix": batch.prefix_state.detach().to(torch.int8).cpu(),
                    "exact": batch.exact.detach().cpu(),
                    "bag": batch.bag_label.detach().to(torch.float32).cpu(),
                    "episode_ids": batch.episode_ids,
                    "group_ids": batch.group_ids,
                    "scopes": batch.scopes,
                    "lengths": batch.lengths.detach().cpu(),
                    "row_start": batch.row_start.detach().cpu(),
                    "row_end": batch.row_end.detach().cpu(),
                }
            )
    if not chunks:
        raise ValueError("no episodes to score")
    max_len = max(int(chunk["logits"].shape[1]) for chunk in chunks)
    total = sum(int(chunk["logits"].shape[0]) for chunk in chunks)

    def padded(key: str, dtype: torch.dtype, shape_len: int, fill: float = 0.0) -> torch.Tensor:
        out = torch.full((total, max_len, shape_len), fill, dtype=dtype)
        position = 0
        for chunk in chunks:
            values = chunk[key]
            if values.ndim == 2:
                out[position : position + values.shape[0], : values.shape[1], 0] = values
            else:
                out[position : position + values.shape[0], : values.shape[1]] = values
            position += values.shape[0]
        return out

    logits = padded("logits", torch.float32, 1).squeeze(-1)
    valid = padded("valid", torch.bool, 1).squeeze(-1)
    ct = padded("ct", torch.bool, 1).squeeze(-1)
    prefix = padded("prefix", torch.int8, 1, fill=-1).squeeze(-1)
    exact = padded("exact", torch.bool, 1).squeeze(-1)
    bag = torch.cat([chunk["bag"].reshape(-1) for chunk in chunks])
    scores = evaluation.episode_scores_from_logits(logits, valid, ct)
    payload = evaluation.predictions_payload(
        logits=logits,
        valid_mask=valid,
        ct_mask=ct,
        prefix_state=prefix,
        exact=exact,
        bag_label=bag,
        episode_ids=[item for chunk in chunks for item in chunk["episode_ids"]],
        group_ids=[item for chunk in chunks for item in chunk["group_ids"]],
        scopes=[item for chunk in chunks for item in chunk["scopes"]],
        lengths=torch.cat([chunk["lengths"] for chunk in chunks]),
        row_start=torch.cat([chunk["row_start"] for chunk in chunks]),
        row_end=torch.cat([chunk["row_end"] for chunk in chunks]),
        scores=scores,
        split_name="eval",
    )
    payload["model_mode"] = np.asarray("eval")
    return payload


def _metrics_from_payload(payload: Mapping[str, Any], mask: Any | None = None) -> dict[str, Any]:
    scores = np.asarray(payload["episode_scores"], dtype=np.float64)
    labels = np.asarray(payload["bag_label"], dtype=np.int64)
    if mask is not None:
        scores, labels = scores[mask], labels[mask]
    return evaluation.evaluate_scores(scores, labels)


def _save_prediction_pack(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    arrays = {key: value for key, value in payload.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(path, **arrays)
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "keys": sorted(arrays),
        "episodes": int(len(arrays.get("episode_scores", []))),
    }


def _save_checkpoint(
    path: Path,
    *,
    state_dict: Mapping[str, torch.Tensor],
    epoch: int,
    config: TrainConfig,
    identity: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    scaler_state: Mapping[str, Any],
) -> dict[str, Any]:
    state = {key: value.detach().cpu().clone() for key, value in state_dict.items()}
    scaler = _scaler_payload(scaler_state, config.scaler)
    payload = {
        "state_dict": state,
        "epoch": int(epoch),
        "config": config.to_dict(),
        "config_sha256": config.config_sha256,
        "input_identity": dict(identity),
        "producer_fingerprint_sha256": fingerprint["fingerprint_sha256"],
        "scaler": scaler,
        "schema_version": 2,
    }
    _atomic_torch_save(payload, path)
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "tensor_sha256": _state_sha256(state),
        "scaler_tensor_sha256": scaler["tensor_sha256"],
        "scaler_name": config.scaler,
        "scaler_mean_sha256": scaler["mean_sha256"],
        "scaler_std_sha256": scaler["std_sha256"],
        "scaler_file_sha256": scaler["file_sha256"],
        "epoch": int(epoch),
        "bytes": path.stat().st_size,
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _rng_state() -> dict[str, Any]:
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# --------------------------------------------------------------------------- train / validate


def _best_history_metric(history: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, float, float, int] | None, int | None]:
    best_key: tuple[float, float, float, int] | None = None
    best_epoch: int | None = None
    for record in history:
        metrics = record.get("select")
        if not metrics:
            continue
        point = metrics["primary"]
        auc = float(metrics.get("auc", float("nan")))
        auc_for_key = auc if math.isfinite(auc) else -1.0
        key = (float(point["fnr"]), float(point["fpr"]), -auc_for_key, int(record["epoch"]))
        if best_key is None or key < best_key:
            best_key, best_epoch = key, int(record["epoch"])
    return best_key, best_epoch


def _identity_matches(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    keys = (
        "manifest_sha256",
        "matrix_sha256",
        "source_ledger_content_sha256",
        "episode_global_indices_sha256",
        "scaler_file_sha256",
        "scaler_state_sha256",
    )
    return all(a.get(key) == b.get(key) for key in keys)


def _metric_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return bool(a) == bool(b)
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    fa, fb = float(a), float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    if math.isinf(fa) or math.isinf(fb):
        return fa == fb
    return math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-12)


def _compare_metrics(label: str, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field in ("auc", "n_episodes", "n_positives", "n_negatives", "empty_bag_scores"):
        if field not in expected or not _metric_equal(actual[field], expected[field]):
            raise RuntimeError(f"{label} {field} does not replay")
    actual_points = actual["operating_points"]
    expected_points = expected["operating_points"]
    if set(actual_points) != set(expected_points):
        raise RuntimeError(f"{label} operating-point budgets differ")
    for budget, point in actual_points.items():
        other = expected_points[budget]
        for field in ("threshold", "fnr", "fpr", "tpr", "tp", "fp", "tn", "fn", "n_pos", "n_neg"):
            if not _metric_equal(point[field], other[field]):
                raise RuntimeError(f"{label} {budget} {field} does not replay")
    # Compare the full primary 5% point explicitly (it is already in operating_points,
    # but a direct check makes the selected metric contract unambiguous).
    for field in ("threshold", "fnr", "fpr", "tp", "fp", "tn", "fn"):
        if not _metric_equal(actual["primary"][field], expected["primary"][field]):
            raise RuntimeError(f"{label} primary {field} does not replay")


def _history_best_auc(history: Sequence[Mapping[str, Any]]) -> tuple[float, int] | tuple[None, None]:
    best_key: tuple[float, int] | None = None
    best_epoch: int | None = None
    for record in history:
        metrics = record.get("select")
        if not metrics:
            continue
        auc = float(metrics.get("auc", float("nan")))
        auc_value = auc if math.isfinite(auc) else -1.0
        key = (-auc_value, int(record["epoch"]))
        if best_key is None or key < best_key:
            best_key, best_epoch = key, int(record["epoch"])
    if best_key is None or best_epoch is None:
        return None, None
    return best_key, best_epoch


def _validate_history(
    raw_config: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    checkpoint_epochs: Mapping[str, int],
) -> dict[str, Any]:
    mode = str(raw_config["mode"])
    max_epochs = int(raw_config["max_epochs"])
    patience = int(raw_config["patience"])
    epochs_run = int(meta.get("epochs_run", -1))
    if epochs_run < 1 or epochs_run != len(history):
        raise RuntimeError("history length/epochs_run mismatch")
    indices = [int(record.get("epoch", -1)) for record in history]
    if indices != list(range(epochs_run)):
        raise RuntimeError("history epoch indices are not contiguous from zero")
    if epochs_run > max_epochs:
        raise RuntimeError("history exceeds frozen max_epochs")
    tags = {str(tag) for tag in raw_config.get("tags", [])}
    has_cap = raw_config.get("max_train_episodes") is not None or raw_config.get("max_steps_per_epoch") is not None
    if has_cap and mode in ("fit", "full_train") and not (tags & {"synthetic", "short_smoke"}):
        raise RuntimeError("budget cap in a formal mode requires a named synthetic/short_smoke run")
    last_epoch = epochs_run - 1
    if checkpoint_epochs.get("last") != last_epoch:
        raise RuntimeError("last checkpoint epoch does not match history")
    metrics = meta.get("metrics", {})
    result: dict[str, Any] = {"epochs_run": epochs_run, "mode": mode}
    if mode == "full_train":
        if epochs_run != max_epochs:
            raise RuntimeError("full_train fixed refit did not reach frozen max_epochs")
        if int(meta.get("selected_epoch", -1)) != last_epoch:
            raise RuntimeError("full_train must select the final epoch")
        if checkpoint_epochs.get("best_primary") != last_epoch:
            raise RuntimeError("full_train best_primary is not the final epoch")
        if checkpoint_epochs.get("best_auc") != last_epoch:
            raise RuntimeError("full_train best_auc is not the final fixture")
        if not str(metrics.get("selection_rule", "")).startswith("FINAL_FIXED_REFIT"):
            raise RuntimeError("full_train selection rule is not labelled FINAL_FIXED_REFIT")
        if not bool(metrics.get("cal_in_training")) or bool(metrics.get("calibration_independent")):
            raise RuntimeError("full_train CAL metadata is not in-sample/non-independent")
        if not bool(metrics.get("select_is_in_sample")):
            raise RuntimeError("full_train SELECT metadata is not marked in-sample")
        result.update({"best_primary_epoch": last_epoch, "best_auc_epoch": last_epoch, "early_stopped": False})
        return result
    best_key, best_epoch = _best_history_metric(history)
    auc_key, auc_epoch = _history_best_auc(history)
    if best_epoch is None or auc_epoch is None or best_key is None or auc_key is None:
        raise RuntimeError("history has no reconstructable SELECT best epoch")
    if int(meta.get("selected_epoch", -1)) != best_epoch:
        raise RuntimeError("selected epoch is not the history primary best")
    if checkpoint_epochs.get("best_primary") != best_epoch:
        raise RuntimeError("best_primary checkpoint epoch is not the history primary best")
    if checkpoint_epochs.get("best_auc") != auc_epoch:
        raise RuntimeError("best_auc checkpoint epoch is not the history AUC best")
    early_stopped = epochs_run < max_epochs
    if early_stopped and not (
        last_epoch - best_epoch >= patience and last_epoch - auc_epoch >= patience
    ):
        raise RuntimeError("early stop before max_epochs without both selectors stale >= patience")
    rule = str(metrics.get("selection_rule", ""))
    if mode == "smoke" and not rule.startswith("SMOKE_ONLY"):
        raise RuntimeError("smoke selection rule is not labelled SMOKE_ONLY")
    if mode == "fit" and not rule.startswith("minimum (FNR"):
        raise RuntimeError("fit selection rule is not the declared SELECT tuple")
    if bool(metrics.get("cal_in_training")) or bool(metrics.get("calibration_independent")):
        raise RuntimeError("fit/smoke CAL metadata falsely claims in-sample/independent calibration")
    result.update({"best_primary_epoch": best_epoch, "best_auc_epoch": auc_epoch, "early_stopped": early_stopped})
    return result


def _verify_checkpoint_scaler(
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    raw_config: Mapping[str, Any],
    meta: Mapping[str, Any],
    inputs: TrainingInputs | None,
) -> dict[str, Any]:
    scaler = payload.get("scaler")
    if not isinstance(scaler, Mapping):
        raise RuntimeError("checkpoint lacks actual scaler tensors")
    name = str(raw_config["scaler"])
    if str(scaler.get("name")) != name:
        raise RuntimeError("checkpoint scaler name differs from frozen config")
    mean = scaler.get("mean")
    std = scaler.get("std")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise RuntimeError("checkpoint scaler mean/std are not tensors")
    if tuple(mean.shape) != tuple(std.shape) or mean.ndim != 1 or mean.shape[0] != int(meta["input_identity"]["matrix_shape"][1]):
        raise RuntimeError("checkpoint scaler tensor shape is wrong")
    mean = mean.detach().cpu().to(torch.float32).contiguous()
    std = std.detach().cpu().to(torch.float32).contiguous()
    eps = float(scaler.get("eps", 0.0))
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
        raise RuntimeError("checkpoint scaler tensors are non-finite")
    if eps <= 0 or bool((std < eps).any()):
        raise RuntimeError("checkpoint scaler std violates eps floor")
    tensor_sha = _scaler_sha256(mean, std)
    if tensor_sha != str(scaler.get("tensor_sha256")) or tensor_sha != str(record.get("scaler_tensor_sha256")):
        raise RuntimeError("checkpoint scaler tensor SHA mismatch")
    mean_sha = sha256_bytes(mean.numpy().tobytes())
    std_sha = sha256_bytes(std.numpy().tobytes())
    if mean_sha != str(scaler.get("mean_sha256")) or std_sha != str(scaler.get("std_sha256")):
        raise RuntimeError("checkpoint scaler mean/std SHA mismatch")
    identity = meta["input_identity"]
    if str(scaler.get("file_sha256")) != str(identity["scaler_file_sha256"][name]):
        raise RuntimeError("checkpoint scaler file SHA differs from frozen input identity")
    if mean_sha != str(identity["scaler_state_sha256"][name]["mean"]) or std_sha != str(
        identity["scaler_state_sha256"][name]["std"]
    ):
        raise RuntimeError("checkpoint scaler state SHA differs from frozen input identity")
    if inputs is not None:
        expected = inputs.scaler_states[name]
        expected_mean = torch.as_tensor(expected["mean"], dtype=torch.float32)
        expected_std = torch.as_tensor(expected["std"], dtype=torch.float32)
        if not torch.equal(mean, expected_mean) or not torch.equal(std, expected_std):
            raise RuntimeError("checkpoint scaler tensors differ from expected inputs")
        if int(scaler.get("fit_rows", -1)) != int(expected.get("fit_rows", -2)):
            raise RuntimeError("checkpoint scaler fit_rows differ from expected inputs")
        if float(scaler.get("eps", -1.0)) != float(expected.get("eps", -2.0)):
            raise RuntimeError("checkpoint scaler eps differs from expected inputs")
    return {
        "name": name,
        "tensor_sha256": tensor_sha,
        "mean_sha256": mean_sha,
        "std_sha256": std_sha,
        "file_sha256": str(scaler.get("file_sha256")),
        "fit_rows": int(scaler.get("fit_rows", 0)),
        "eps": eps,
        "shape": list(mean.shape),
        "input_tensors_checked": inputs is not None,
    }


PREDICTION_REQUIRED_KEYS = frozenset(
    {
        "logits",
        "sigmoid_float32_diagnostic",
        "valid_mask",
        "ct_mask",
        "prefix_state",
        "exact",
        "bag_label",
        "episode_scores",
        "lengths",
        "row_start",
        "row_end",
        "episode_ids",
        "group_ids",
        "scopes",
        "is_agentlab",
        "is_common_dev",
        "split",
        "model_mode",
    }
)


def _input_episode_ids(inputs: TrainingInputs, split_name: str) -> set[str]:
    if split_name not in ("select", "cal", "dev"):
        return set()
    return {
        str(inputs.episode_ids[inputs._global_to_local[global_index]])
        for global_index in inputs.episode_indices(split_name)
    }


def _verify_prediction_pack(
    name: str,
    record: Mapping[str, Any],
    run_dir: Path,
    inputs: TrainingInputs | None,
) -> dict[str, Any]:
    path = Path(record["path"])
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise RuntimeError(f"declared prediction pack {name} is missing: {path}")
    try:
        path.resolve().relative_to(run_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"prediction pack {name} is outside its run directory") from exc
    actual_file = _file_record(path)
    if actual_file["sha256"] != record.get("sha256") or actual_file["bytes"] != int(record.get("bytes", -1)):
        raise RuntimeError(f"prediction pack {name} file SHA/bytes mismatch")
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(data.files)
        missing = sorted(PREDICTION_REQUIRED_KEYS - set(keys))
        if missing:
            raise RuntimeError(f"prediction pack {name} missing keys {missing}")
        if keys != sorted(record.get("keys", [])):
            raise RuntimeError(f"prediction pack {name} recorded key list mismatch")
        logits = np.asarray(data["logits"], dtype=np.float32)
        valid = np.asarray(data["valid_mask"], dtype=np.bool_)
        ct = np.asarray(data["ct_mask"], dtype=np.bool_)
        prefix = np.asarray(data["prefix_state"], dtype=np.int8)
        exact = np.asarray(data["exact"], dtype=np.bool_)
        bag = np.asarray(data["bag_label"], dtype=np.float32)
        scores = np.asarray(data["episode_scores"], dtype=np.float64)
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        row_start = np.asarray(data["row_start"], dtype=np.int64)
        row_end = np.asarray(data["row_end"], dtype=np.int64)
        episode_ids = [str(x) for x in data["episode_ids"].tolist()]
        group_ids = [str(x) for x in data["group_ids"].tolist()]
        scopes = [str(x) for x in data["scopes"].tolist()]
        is_agentlab = np.asarray(data["is_agentlab"], dtype=np.bool_)
        is_common = np.asarray(data["is_common_dev"], dtype=np.bool_)
        sigmoid = np.asarray(data["sigmoid_float32_diagnostic"], dtype=np.float32)
        episodes = logits.shape[0]
        if logits.ndim != 2 or valid.shape != logits.shape or ct.shape != logits.shape:
            raise RuntimeError(f"prediction pack {name} tensor shapes are inconsistent")
        if prefix.shape != logits.shape or exact.shape != logits.shape or sigmoid.shape != logits.shape:
            raise RuntimeError(f"prediction pack {name} position metadata shape mismatch")
        if bag.shape != (episodes,) or scores.shape != (episodes,) or lengths.shape != (episodes,):
            raise RuntimeError(f"prediction pack {name} episode-vector shape mismatch")
        if row_start.shape != (episodes,) or row_end.shape != (episodes,):
            raise RuntimeError(f"prediction pack {name} row binding shape mismatch")
        if len(episode_ids) != episodes or len(group_ids) != episodes or len(scopes) != episodes:
            raise RuntimeError(f"prediction pack {name} ID/metadata length mismatch")
        if is_agentlab.shape != (episodes,) or is_common.shape != (episodes,):
            raise RuntimeError(f"prediction pack {name} cohort flag shape mismatch")
        if len(set(episode_ids)) != episodes:
            raise RuntimeError(f"prediction pack {name} episode IDs are not unique")
        if int(record.get("episodes", -1)) != episodes:
            raise RuntimeError(f"prediction pack {name} episode count mismatch")
        if np.isnan(logits).any() or np.isposinf(logits).any() or not np.isfinite(logits[valid]).all():
            raise RuntimeError(f"prediction pack {name} has non-finite real logits")
        if bool((ct & ~valid).any()):
            raise RuntimeError(f"prediction pack {name} has CT on padding")
        if not np.array_equal(valid.sum(axis=1).astype(np.int64), lengths):
            raise RuntimeError(f"prediction pack {name} lengths do not match valid masks")
        if not np.array_equal(row_end - row_start, lengths):
            raise RuntimeError(f"prediction pack {name} row spans do not match lengths")
        for row, length in enumerate(lengths.tolist()):
            if length < 0 or length > logits.shape[1]:
                raise RuntimeError(f"prediction pack {name} bad length")
            if not bool(valid[row, :length].all()) or bool(valid[row, length:].any()):
                raise RuntimeError(f"prediction pack {name} valid mask is not a right-padded prefix")
            if not bool((prefix[row, length:] == -1).all()):
                raise RuntimeError(f"prediction pack {name} padded prefix truth is not -1")
            if bool(exact[row, length:].any()) or bool(ct[row, length:].any()):
                raise RuntimeError(f"prediction pack {name} padded metadata is not inert")
        replay_scores = evaluation.episode_scores_from_logits(
            torch.from_numpy(logits), torch.from_numpy(valid), torch.from_numpy(ct)
        )
        if not np.array_equal(replay_scores, scores):
            raise RuntimeError(f"prediction pack {name} episode scores do not replay from logits/masks")
        empty = ~np.any(valid & ct, axis=1)
        if np.isposinf(scores).any() or not np.array_equal(np.isneginf(scores), empty):
            raise RuntimeError(f"prediction pack {name} empty-bag score sentinel is wrong")
        replay_sigmoid = torch.sigmoid(torch.from_numpy(np.ascontiguousarray(logits))).numpy().astype(np.float32)
        if not np.array_equal(replay_sigmoid, sigmoid):
            raise RuntimeError(f"prediction pack {name} CPU float32 sigmoid diagnostic does not replay")
        if not np.array_equal(is_agentlab, np.asarray([scope == "agentlab" for scope in scopes], dtype=np.bool_)):
            raise RuntimeError(f"prediction pack {name} is_agentlab flag mismatch")
        if not np.array_equal(is_common, ~is_agentlab):
            raise RuntimeError(f"prediction pack {name} is_common_dev flag mismatch")
        binding_checked = False
        if inputs is not None:
            expected_ids = _input_episode_ids(inputs, name)
            if expected_ids and set(episode_ids) != expected_ids:
                raise RuntimeError(f"prediction pack {name} episode-ID set differs from accepted inputs")
            id_to_local = {str(episode_id): local for local, episode_id in enumerate(inputs.episode_ids)}
            for row, episode_id in enumerate(episode_ids):
                if episode_id not in id_to_local:
                    raise RuntimeError(f"prediction pack {name} episode {episode_id} is absent from accepted inputs")
                local = id_to_local[episode_id]
                length = int(inputs.episode_length[local].item())
                start = int(inputs.episode_row_start[local].item())
                if int(lengths[row]) != length:
                    raise RuntimeError(f"prediction pack {name} {episode_id} length mismatch")
                if int(row_start[row]) != start or int(row_end[row]) != start + length:
                    raise RuntimeError(f"prediction pack {name} {episode_id} row binding mismatch")
                if not np.array_equal(valid[row, :length], np.ones(length, dtype=np.bool_)):
                    raise RuntimeError(f"prediction pack {name} {episode_id} valid mask mismatch")
                if not np.array_equal(ct[row, :length], inputs.row_ct_mask[start : start + length].detach().cpu().numpy()):
                    raise RuntimeError(f"prediction pack {name} {episode_id} CT mask mismatch")
                if not np.array_equal(prefix[row, :length], inputs.row_prefix_state[start : start + length].detach().cpu().numpy()):
                    raise RuntimeError(f"prediction pack {name} {episode_id} prefix truth mismatch")
                if not np.array_equal(exact[row, :length], inputs.row_exact[start : start + length].detach().cpu().numpy()):
                    raise RuntimeError(f"prediction pack {name} {episode_id} exact truth mismatch")
                if not _metric_equal(bag[row], float(inputs.episode_bag_label[local].item())):
                    raise RuntimeError(f"prediction pack {name} {episode_id} bag truth mismatch")
                if str(group_ids[row]) != str(inputs.episode_group_id[local]) or str(scopes[row]) != str(inputs.episode_scope[local]):
                    raise RuntimeError(f"prediction pack {name} {episode_id} group/scope mismatch")
            binding_checked = True
        labels = bag.astype(np.int64)
        return {
            "name": name,
            "path": str(path),
            "sha256": actual_file["sha256"],
            "bytes": actual_file["bytes"],
            "episodes": episodes,
            "scores": scores,
            "labels": labels,
            "is_agentlab": is_agentlab,
            "binding_checked": binding_checked,
            "empty_bags": int(empty.sum()),
        }


def _compare_cal_frozen(metrics: Mapping[str, Any], packs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stored = metrics.get("cal_frozen_threshold")
    if not isinstance(stored, Mapping) or "cal" not in packs:
        raise RuntimeError("CAL frozen-threshold diagnostics are missing")
    cal = packs["cal"]
    threshold = evaluation.frozen_threshold(cal["scores"], cal["labels"], target_fpr=0.05)
    if not _metric_equal(threshold, stored["thresholds"]["5%"]):
        raise RuntimeError("CAL frozen threshold does not replay")
    checks: dict[str, Any] = {"threshold": threshold}
    targets = []
    if "select" in packs:
        targets.append(("select_5%", packs["select"]["scores"], packs["select"]["labels"]))
    if "dev" in packs:
        dev = packs["dev"]
        agent = dev["is_agentlab"]
        targets.extend(
            [
                ("dev_common_5%", dev["scores"][~agent], dev["labels"][~agent]),
                ("dev_all_5%", dev["scores"], dev["labels"]),
                ("dev_agentlab_diagnostic_5%", dev["scores"][agent], dev["labels"][agent]),
            ]
        )
    for key, scores, labels in targets:
        expected = stored.get(key)
        if not isinstance(expected, Mapping):
            raise RuntimeError(f"missing CAL frozen application {key}")
        actual = evaluation.apply_frozen_threshold(scores, labels, threshold)
        for field in ("threshold", "fnr", "fpr", "tp", "fp", "tn", "fn", "n_pos", "n_neg"):
            if not _metric_equal(actual[field], expected[field]):
                raise RuntimeError(f"CAL frozen {key} {field} does not replay")
        checks[key] = {field: actual[field] for field in ("tp", "fp", "tn", "fn", "fnr", "fpr")}
    return checks


def _train_epoch(
    model: nn.Module,
    inputs: TrainingInputs,
    draws: Sequence[int],
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> dict[str, Any]:
    model.train()
    batches = make_batches(
        draws,
        config.batch_size,
        rng,
        inputs=inputs,
        bucket_by_length=config.bucket_by_length,
    )
    accumulation: dict[str, float] = {
        "loss_sum": 0.0,
        "steps": 0.0,
        "positions": 0.0,
        "episodes_fed": 0.0,
        "positive_episodes_fed": 0.0,
        "negative_episodes_fed": 0.0,
        "endpoint_supervised_endpoints": 0.0,
        "endpoint_positive_mass": 0.0,
        "endpoint_negative_mass": 0.0,
        "aux_supervised_endpoints": 0.0,
        "aux_positive_mass": 0.0,
        "aux_negative_mass": 0.0,
        "mil_positive_mass": 0.0,
        "mil_negative_mass": 0.0,
        "empty_bags": 0.0,
        "trainable_bags": 0.0,
        "clipped_steps": 0.0,
    }
    grad_norms: list[float] = []
    unique_episodes: set[int] = set()
    last_batch_size = 0
    started = time.time()
    for step, batch_indices in enumerate(batches):
        if config.max_steps_per_epoch is not None and step >= int(config.max_steps_per_epoch):
            break
        batch = inputs.episode_batch(
            batch_indices,
            scaler_name=config.scaler,
            include_labels=True,
            allow_mixed_train_partitions=config.mode == "full_train",
        )
        batch_bags = batch.bag_label.detach().to(torch.float32)
        accumulation["episodes_fed"] += float(len(batch_indices))
        accumulation["positive_episodes_fed"] += float((batch_bags > 0).sum().item())
        accumulation["negative_episodes_fed"] += float((batch_bags <= 0).sum().item())
        unique_episodes.update(int(index) for index in batch_indices)
        features, valid, kwargs = _model_inputs(model, batch)
        return_aux = config.objective.aux_weight > 0
        output = model(features, valid, return_aux=return_aux, **kwargs)
        if return_aux:
            logits, aux_logits = output
        else:
            logits, aux_logits = output, None
        loss, stats = compute_objective(config.objective, logits, batch, aux_logits=aux_logits)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}: {stats}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip)).detach().item()
            if config.grad_clip > 0
            else 0.0
        )
        grad_norms.append(grad_norm)
        if config.grad_clip > 0 and grad_norm > float(config.grad_clip):
            accumulation["clipped_steps"] += 1.0
        optimizer.step()
        accumulation["loss_sum"] += float(stats["loss"])
        accumulation["steps"] += 1.0
        accumulation["positions"] += float(valid.sum().item())
        last_batch_size = len(batch_indices)
        terms = stats.get("terms", {})
        endpoint = terms.get("endpoint", {})
        aux = terms.get("aux", {})
        mil = terms.get("mil", {})
        accumulation["endpoint_supervised_endpoints"] += float(endpoint.get("supervised_endpoints", 0.0))
        accumulation["endpoint_positive_mass"] += float(endpoint.get("positive_mass", 0.0))
        accumulation["endpoint_negative_mass"] += float(endpoint.get("negative_mass", 0.0))
        accumulation["aux_supervised_endpoints"] += float(aux.get("supervised_endpoints", 0.0))
        accumulation["aux_positive_mass"] += float(aux.get("positive_mass", 0.0))
        accumulation["aux_negative_mass"] += float(aux.get("negative_mass", 0.0))
        accumulation["mil_positive_mass"] += float(mil.get("positive_mass", 0.0))
        accumulation["mil_negative_mass"] += float(mil.get("negative_mass", 0.0))
        accumulation["empty_bags"] += float(mil.get("empty_bags", 0.0))
        accumulation["trainable_bags"] += float(mil.get("trainable_bags", 0.0))
    steps = max(1.0, accumulation["steps"])
    observed = {
        "steps": int(accumulation["steps"]),
        "last_batch_size": int(last_batch_size),
        "positions": int(accumulation["positions"]),
        "episodes_fed": int(accumulation["episodes_fed"]),
        "unique_episodes_fed": len(unique_episodes),
        "positive_episodes_fed": int(accumulation["positive_episodes_fed"]),
        "negative_episodes_fed": int(accumulation["negative_episodes_fed"]),
        "endpoint_supervised_endpoints": int(accumulation["endpoint_supervised_endpoints"]),
        "endpoint_positive_weight_mass": float(accumulation["endpoint_positive_mass"]),
        "endpoint_negative_weight_mass": float(accumulation["endpoint_negative_mass"]),
        "aux_supervised_endpoints": int(accumulation["aux_supervised_endpoints"]),
        "aux_positive_weight_mass": float(accumulation["aux_positive_mass"]),
        "aux_negative_weight_mass": float(accumulation["aux_negative_mass"]),
        "mil_positive_bag_weight_mass": float(accumulation["mil_positive_mass"]),
        "mil_negative_bag_weight_mass": float(accumulation["mil_negative_mass"]),
        "empty_bags": int(accumulation["empty_bags"]),
        "trainable_bags": int(accumulation["trainable_bags"]),
        "clipped_steps": int(accumulation["clipped_steps"]),
        "mass_units": "endpoint/aux masses are positions; MIL masses are bags; do not sum across units",
    }
    return {
        "loss_mean": accumulation["loss_sum"] / steps,
        **observed,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
        "wall_seconds": time.time() - started,
    }


def _checkpoint_from_selector(selector: ModelSelector, name: str) -> dict[str, torch.Tensor] | None:
    if name == "primary":
        return selector.primary_state
    return selector.auc_state


def train_one(
    config: TrainConfig,
    inputs_or_dataset: TrainingInputs | Any,
    outdir: Path | str,
    *,
    device: torch.device | str | None = None,
    resume: bool = True,
    verify_bindings: bool = True,
) -> dict[str, Any]:
    """Train one frozen configuration and write an immutable run directory."""
    outdir = Path(outdir)
    if isinstance(inputs_or_dataset, TrainingInputs):
        inputs = inputs_or_dataset
        if device is not None and torch.device(device) != inputs.device:
            raise ValueError("device does not match cached TrainingInputs")
    else:
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        inputs = preload_inputs(inputs_or_dataset, target_device, verify_bindings=verify_bindings)
    device = inputs.device
    meta_path = outdir / "run_meta.json"
    config_payload = config.to_dict()
    config_sha = config.config_sha256
    if meta_path.exists():
        meta = _read_json(meta_path)
        if meta.get("status") == "complete":
            if not resume:
                raise RuntimeError(f"run {outdir} is complete; refusing force/reset")
            if meta.get("config") != config_payload or meta.get("config_sha256") != config_sha:
                raise RuntimeError("completed run config differs; use a new run id")
            if not _identity_matches(meta.get("input_identity", {}), inputs.identity):
                raise RuntimeError("completed run input identity differs; refusing no-op resume")
            source_check = _verify_source_snapshot(outdir)
            if source_check["status"] != "PASS":
                raise RuntimeError("completed run frozen source snapshot is corrupt")
            # Verify saved checkpoint tensors and replay saved metrics without scoring.
            replay = validate_saved_run(outdir, inputs=inputs)
            if replay["status"] != "PASS":
                raise RuntimeError(f"completed run validation failed: {replay}")
            return {
                "status": "complete_noop",
                "run_dir": str(outdir),
                "config_sha256": config_sha,
                "validation": replay,
            }
        if meta.get("status") == "failed":
            raise RuntimeError(f"run {outdir} failed; preserve it and use a new run id")
        if meta.get("status") != "running":
            raise RuntimeError(f"run {outdir} has unknown status {meta.get('status')!r}")
        if not resume:
            raise RuntimeError(f"run {outdir} is incomplete; pass resume=True or a new run id")
        if meta.get("config") != config_payload or meta.get("config_sha256") != config_sha:
            raise RuntimeError("resume config differs from frozen run config")
        if not _identity_matches(meta.get("input_identity", {}), inputs.identity):
            raise RuntimeError("resume input identity differs from frozen run identity")
        fingerprint = producer_fingerprint()
        if fingerprint["fingerprint_sha256"] != meta.get("producer_fingerprint", {}).get("fingerprint_sha256"):
            raise RuntimeError("resume producer fingerprint differs from frozen run")
        last_path = outdir / "last_state.pt"
        if not last_path.is_file():
            raise RuntimeError("incomplete run has no epoch-boundary checkpoint")
        resume_payload = _load_checkpoint(last_path)
        history = list(resume_payload.get("history", []))
        start_epoch = int(resume_payload["epoch"]) + 1
        selector = ModelSelector.from_state(resume_payload["selector"])
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        fingerprint = producer_fingerprint()
        snapshot = archive_source_snapshot(outdir, fingerprint)
        _write_json(
            meta_path,
            {
                "schema_version": 2,
                "status": "running",
                "run_id": config.run_id,
                "config": config_payload,
                "config_sha256": config_sha,
                "input_identity": inputs.identity,
                "producer_fingerprint": fingerprint,
                "source_snapshot": {
                    "fingerprint_sha256": snapshot["fingerprint_sha256"],
                    "files": len(snapshot["files"]),
                },
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_epoch": -1,
            },
        )
        history = []
        start_epoch = 0
        selector = ModelSelector(config.patience)
        resume_payload = None

    torch.manual_seed(int(config.seed))
    model = build_model(config.model).to(device)
    if config.objective.aux_weight > 0 and not config.model.aux_head:
        raise ValueError("aux_weight > 0 requires an auxiliary head")
    initial_parameter_sha256 = _state_sha256(model.state_dict())
    if resume_payload is not None:
        initial_parameter_sha256 = str(
            _read_json(meta_path).get("initial_parameter_sha256", initial_parameter_sha256)
        )
    if resume_payload is None:
        meta = _read_json(meta_path)
        meta["initial_parameter_sha256"] = initial_parameter_sha256
        _write_json(meta_path, meta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    sampler_rng = np.random.default_rng([int(config.seed), 20_261_110])
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        _restore_rng_state(resume_payload.get("rng_state", {}))
        sampler_rng.bit_generator.state = resume_payload["sampler_rng_state"]

    train_name = "train" if config.mode == "full_train" else "fit"
    train_indices = inputs.episode_indices(train_name)
    if config.max_train_episodes is not None:
        cap = int(config.max_train_episodes)
        if cap < len(train_indices):
            pick_rng = np.random.default_rng([int(config.seed), 7])
            permutation = pick_rng.permutation(len(train_indices))
            train_indices = [train_indices[int(position)] for position in permutation[:cap]]
    select_indices = inputs.episode_indices(config.selection_split)
    cal_indices = inputs.episode_indices("cal")
    dev_indices = inputs.episode_indices("dev")
    if not train_indices:
        raise ValueError("no training episodes selected")
    if config.mode in ("fit", "smoke") and not select_indices:
        raise ValueError("FIT/smoke selection requires SELECT episodes")

    started = time.time()
    for epoch in range(start_epoch, int(config.max_epochs)):
        draws, sample_stats = sample_epoch_indices(
            train_indices,
            inputs.episode_bag_label,
            inputs,
            sampler_rng,
            balanced=config.balanced,
            pos_neg_ratio=config.pos_neg_ratio,
        )
        train_stats = _train_epoch(model, inputs, draws, config, optimizer, sampler_rng)
        observed_keys = (
            "steps",
            "last_batch_size",
            "positions",
            "episodes_fed",
            "unique_episodes_fed",
            "positive_episodes_fed",
            "negative_episodes_fed",
            "endpoint_supervised_endpoints",
            "endpoint_positive_weight_mass",
            "endpoint_negative_weight_mass",
            "aux_supervised_endpoints",
            "aux_positive_weight_mass",
            "aux_negative_weight_mass",
            "mil_positive_bag_weight_mass",
            "mil_negative_bag_weight_mass",
            "empty_bags",
            "trainable_bags",
            "clipped_steps",
            "mass_units",
        )
        record: dict[str, Any] = {
            "epoch": int(epoch),
            "train": train_stats,
            "sample_plan": sample_stats,
            "observed": {key: train_stats[key] for key in observed_keys},
            "wall_seconds": train_stats["wall_seconds"],
        }
        if config.mode in ("fit", "smoke"):
            payload = score_episodes(
                model,
                inputs,
                select_indices,
                scaler_name=config.scaler,
                batch_size=config.eval_batch_size,
                bucket_by_length=config.bucket_by_length,
            )
            metrics = _metrics_from_payload(payload)
            record["select"] = metrics
            improved = selector.update(epoch, metrics, model.state_dict())
            record["improved"] = improved
            if improved["primary"] and selector.primary_state is not None:
                record["best_primary_epoch"] = selector.primary_epoch
            if improved["auc"] and selector.auc_state is not None:
                record["best_auc_epoch"] = selector.auc_epoch
        history.append(record)
        # Epoch-boundary checkpoint carries optimizer/RNG/selector for safe resume.
        _atomic_torch_save(
            {
                "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "selector": selector.state_dict(),
                "sampler_rng_state": sampler_rng.bit_generator.state,
                "rng_state": _rng_state(),
                "history": history,
                "epoch": int(epoch),
                "config_sha256": config_sha,
                "schema_version": 2,
            },
            outdir / "last_state.pt",
        )
        meta = _read_json(meta_path)
        meta["status"] = "running"
        meta["last_epoch"] = int(epoch)
        meta["history_epochs"] = len(history)
        _write_json(meta_path, meta)
        if config.mode in ("fit", "smoke") and selector.should_stop():
            break

    if config.mode in ("fit", "smoke"):
        if selector.primary_epoch is None or selector.primary_state is None:
            raise RuntimeError("premature completion: no selected primary checkpoint")
        history_key, history_epoch = _best_history_metric(history)
        if history_epoch != selector.primary_epoch or history_key != selector.primary_key:
            raise RuntimeError("selector/history global-best disagreement")
        selected_epoch = int(selector.primary_epoch)
        best_select_metrics = selector.primary_metrics
    else:
        selected_epoch = int(history[-1]["epoch"]) if history else -1
        best_select_metrics = None

    # Save both selected states with tensor hashes; no stale graph aliases.
    primary_state = selector.primary_state if config.mode in ("fit", "smoke") else _clone_state(model.state_dict())
    auc_state = selector.auc_state if selector.auc_state is not None else primary_state
    assert primary_state is not None and auc_state is not None
    checkpoints = {
        "best_primary": _save_checkpoint(
            outdir / "best_primary.pt",
            state_dict=primary_state,
            epoch=selected_epoch,
            config=config,
            identity=inputs.identity,
            fingerprint=fingerprint,
            scaler_state=inputs.scaler_states[config.scaler],
        ),
        "best_auc": _save_checkpoint(
            outdir / "best_auc.pt",
            state_dict=auc_state,
            epoch=int(selector.auc_epoch if selector.auc_epoch is not None else selected_epoch),
            config=config,
            identity=inputs.identity,
            fingerprint=fingerprint,
            scaler_state=inputs.scaler_states[config.scaler],
        ),
        "last": _save_checkpoint(
            outdir / "last.pt",
            state_dict=model.state_dict(),
            epoch=int(history[-1]["epoch"]) if history else -1,
            config=config,
            identity=inputs.identity,
            fingerprint=fingerprint,
            scaler_state=inputs.scaler_states[config.scaler],
        ),
    }

    # Reproduce the selected epoch from the saved checkpoint on a fresh model.
    selected_model = build_model(config.model).to(device)
    selected_model.load_state_dict(_load_checkpoint(outdir / "best_primary.pt")["state_dict"])
    select_payload = score_episodes(
        selected_model,
        inputs,
        select_indices,
        scaler_name=config.scaler,
        batch_size=config.eval_batch_size,
        bucket_by_length=config.bucket_by_length,
    ) if select_indices else None
    select_metrics = _metrics_from_payload(select_payload) if select_payload is not None else None
    if config.mode in ("fit", "smoke") and best_select_metrics is not None:
        expected = best_select_metrics["primary"]
        actual = select_metrics["primary"]
        for key in ("fnr", "fpr", "threshold"):
            if not math.isclose(float(expected[key]), float(actual[key]), rel_tol=1e-6, abs_tol=1e-7):
                raise RuntimeError(f"selected-epoch replay mismatch for {key}: {expected[key]} != {actual[key]}")
        if not math.isclose(
            float(best_select_metrics["auc"]), float(select_metrics["auc"]), rel_tol=1e-6, abs_tol=1e-9
        ):
            raise RuntimeError("selected-epoch AUC replay mismatch")

    if config.mode == "full_train":
        selection_rule = "FINAL_FIXED_REFIT: final epoch; SELECT/CAL are in-sample diagnostics only"
        select_is_in_sample = True
        cal_in_training = True
    elif config.mode == "smoke":
        selection_rule = "SMOKE_ONLY: SELECT empirical diagnostic; non-production short budget"
        select_is_in_sample = False
        cal_in_training = False
    else:
        selection_rule = "minimum (FNR, actual FPR, -AUC, earliest epoch) at empirical FPR<=5% on SELECT"
        select_is_in_sample = False
        cal_in_training = False
    metrics: dict[str, Any] = {
        "select": select_metrics,
        "selected_epoch": int(selected_epoch),
        "run_mode": config.mode,
        "selection_rule": selection_rule,
        "select_is_in_sample": bool(select_is_in_sample),
        "cal_in_training": bool(cal_in_training),
        "calibration_independent": False,
        "dev_early_stop": False,
        "dev_scored_once": False,
        "cal_used_adaptively": False,
    }
    prediction_files: dict[str, Any] = {}
    if select_payload is not None:
        prediction_files["select"] = _save_prediction_pack(outdir / "select_predictions.npz", select_payload)
    dev_payload = None
    if config.eval_dev and dev_indices:
        dev_payload = score_episodes(
            selected_model,
            inputs,
            dev_indices,
            scaler_name=config.scaler,
            batch_size=config.eval_batch_size,
            bucket_by_length=config.bucket_by_length,
        )
        prediction_files["dev"] = _save_prediction_pack(outdir / "dev_predictions.npz", dev_payload)
        is_agentlab = np.asarray(dev_payload["is_agentlab"], dtype=np.bool_)
        metrics["dev_all"] = _metrics_from_payload(dev_payload)
        metrics["dev_common"] = _metrics_from_payload(dev_payload, ~is_agentlab)
        metrics["dev_agentlab_diagnostic"] = _metrics_from_payload(dev_payload, is_agentlab)
        metrics["dev_by_source"] = {
            "common_dev": evaluation.summarise_by(
                dev_payload["episode_scores"][~is_agentlab],
                dev_payload["bag_label"][~is_agentlab],
                dev_payload["scopes"][~is_agentlab],
            ),
            "agentlab_diagnostic": evaluation.summarise_by(
                dev_payload["episode_scores"][is_agentlab],
                dev_payload["bag_label"][is_agentlab],
                dev_payload["scopes"][is_agentlab],
            ),
        }
        metrics["dev_by_group_common"] = evaluation.summarise_by(
            dev_payload["episode_scores"][~is_agentlab],
            dev_payload["bag_label"][~is_agentlab],
            dev_payload["group_ids"][~is_agentlab],
        )
        metrics["dev_scored_once"] = True
    if config.eval_cal and cal_indices:
        cal_payload = score_episodes(
            selected_model,
            inputs,
            cal_indices,
            scaler_name=config.scaler,
            batch_size=config.eval_batch_size,
            bucket_by_length=config.bucket_by_length,
        )
        prediction_files["cal"] = _save_prediction_pack(outdir / "cal_predictions.npz", cal_payload)
        cal_metrics = _metrics_from_payload(cal_payload)
        metrics["cal"] = cal_metrics
        threshold = evaluation.frozen_threshold(
            cal_payload["episode_scores"], cal_payload["bag_label"], target_fpr=0.05
        )
        frozen_source = (
            "IN-SAMPLE CAL empirical ROC (full_train contains CAL), diagnostic only"
            if config.mode == "full_train"
            else "CAL empirical ROC, frozen; calibration_independent=False"
        )
        frozen: dict[str, Any] = {"thresholds": {"5%": threshold}, "source": frozen_source}
        if select_payload is not None:
            frozen["select_5%"] = evaluation.apply_frozen_threshold(
                select_payload["episode_scores"], select_payload["bag_label"], threshold
            )
        if dev_payload is not None:
            is_agentlab = np.asarray(dev_payload["is_agentlab"], dtype=np.bool_)
            frozen["dev_common_5%"] = evaluation.apply_frozen_threshold(
                dev_payload["episode_scores"][~is_agentlab], dev_payload["bag_label"][~is_agentlab], threshold
            )
            frozen["dev_all_5%"] = evaluation.apply_frozen_threshold(
                dev_payload["episode_scores"], dev_payload["bag_label"], threshold
            )
            frozen["dev_agentlab_diagnostic_5%"] = evaluation.apply_frozen_threshold(
                dev_payload["episode_scores"][is_agentlab], dev_payload["bag_label"][is_agentlab], threshold
            )
        metrics["cal_frozen_threshold"] = frozen
        metrics["cal_used_adaptively"] = False

    summary = {
        "schema_version": 2,
        "run_id": config.run_id,
        "mode": config.mode,
        "config": config_payload,
        "config_sha256": config_sha,
        "input_identity": inputs.identity,
        "producer_fingerprint": fingerprint,
        "model": {
            "family": config.model.family,
            "parameters": count_parameters(model),
            "receptive_field": config.model.receptive_field,
            "initial_parameter_sha256": initial_parameter_sha256,
        },
        "training": {
            "train_episodes": len(train_indices),
            "selection_episodes": len(select_indices),
            "dev_episodes": len(dev_indices),
            "cal_episodes": len(cal_indices),
            "epochs_run": len(history),
            "selected_epoch": int(selected_epoch),
            "wall_seconds": time.time() - started,
            "history": history,
            "rng": {
                "model_seed": int(config.seed),
                "sampler_seed": [int(config.seed), 20_261_110],
            },
            "sample_units": {
                "sample_plan": "intended episode draws/classes before batching and max_steps caps",
                "observed": "actually fed episodes/positions/steps in history[epoch].observed",
                "note": "max_steps_per_epoch can make observed smaller than sample_plan; do not claim whole-plan exposure",
            },
            "gpu_memory": (
                {
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
                if torch.device(device).type == "cuda"
                else {}
            ),
        },
        "checkpoints": checkpoints,
        "predictions": prediction_files,
        "metrics": metrics,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(outdir / "history.json", history)
    _write_json(outdir / "metrics.json", metrics)
    artifact_files = {
        "history": _file_record(outdir / "history.json"),
        "metrics": _file_record(outdir / "metrics.json"),
    }
    summary["artifact_files"] = artifact_files
    _write_json(outdir / "run_summary.json", summary)
    meta = _read_json(meta_path)
    meta.update(
        {
            "schema_version": 2,
            "status": "complete",
            "completed_utc": summary["completed_utc"],
            "selected_epoch": int(selected_epoch),
            "checkpoints": checkpoints,
            "predictions": prediction_files,
            "metrics": metrics,
            "artifact_files": artifact_files,
            "model_parameters": summary["model"]["parameters"],
            "initial_parameter_sha256": initial_parameter_sha256,
            "epochs_run": len(history),
        }
    )
    _write_json(meta_path, meta)
    return {"status": "complete", "run_dir": str(outdir), "summary": summary}


def validate_saved_run(
    run_dir: Path | str,
    *,
    inputs: TrainingInputs | None = None,
    expected_config: Mapping[str, Any] | None = None,
    expected_input_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """No-scorer replay for the acceptance-v2 immutable payload.

    It never constructs a model or calls a scorer.  It binds raw frozen config JSON
    (without reconstructing typed dataclasses and silently adding future defaults),
    archived producer bytes, actual checkpoint tensors, the selected scaler tensors,
    history/selector/budget rules, and every declared prediction pack.
    """
    run_dir = Path(run_dir)
    meta = _read_json(run_dir / "run_meta.json")
    if meta.get("status") != "complete":
        return {"status": "INCOMPLETE", "run_dir": str(run_dir), "run_status": meta.get("status")}
    if int(meta.get("schema_version", 1)) < 2:
        return {
            "status": "LEGACY_V1_UNSUPPORTED_BY_V2_VALIDATOR",
            "run_dir": str(run_dir),
            "note": "Old v1 complete payloads remain byte-for-byte immutable; use their archived source evidence, not this v2 validator.",
        }
    raw_config = meta.get("config")
    if not isinstance(raw_config, Mapping):
        raise RuntimeError("frozen config is missing")
    raw_sha = canonical_sha256(raw_config)
    if raw_sha != meta.get("config_sha256"):
        raise RuntimeError("frozen raw-config canonical hash mismatch")
    if expected_config is not None and canonical_sha256(expected_config) != raw_sha:
        raise RuntimeError("expected config differs from frozen run config")
    source = _verify_source_snapshot(run_dir)
    if source["status"] != "PASS":
        raise RuntimeError(f"frozen source snapshot mismatch: {source}")
    if source["fingerprint_sha256"] != meta["producer_fingerprint"]["fingerprint_sha256"]:
        raise RuntimeError("source snapshot fingerprint does not match run metadata")
    identity = meta["input_identity"]
    if inputs is not None and not _identity_matches(identity, inputs.identity):
        raise RuntimeError("current input identity differs from frozen run identity")
    if expected_input_bindings is not None:
        for key, value in expected_input_bindings.items():
            if key in identity and canonical_sha256(identity[key]) != canonical_sha256(value):
                raise RuntimeError(f"expected input binding {key} differs from frozen run identity")

    checkpoints_meta = meta.get("checkpoints", {})
    required_checkpoints = {"best_primary", "best_auc", "last"}
    missing_checkpoints = sorted(required_checkpoints - set(checkpoints_meta))
    if missing_checkpoints:
        raise RuntimeError(f"required checkpoints missing: {missing_checkpoints}")
    checkpoint_checks: dict[str, Any] = {}
    checkpoint_epochs: dict[str, int] = {}
    for name, record in checkpoints_meta.items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise RuntimeError(f"checkpoint {name} is missing: {path}")
        try:
            path.resolve().relative_to(run_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"checkpoint {name} is outside its run directory") from exc
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != int(record.get("bytes", -1)):
            raise RuntimeError(f"checkpoint {name} file SHA/bytes mismatch")
        payload = _load_checkpoint(path)
        if int(payload.get("epoch", -1)) != int(record.get("epoch", -2)):
            raise RuntimeError(f"checkpoint {name} epoch differs from its record")
        if payload.get("config") != raw_config:
            raise RuntimeError(f"checkpoint {name} config differs from frozen run config")
        if payload.get("config_sha256") != meta.get("config_sha256"):
            raise RuntimeError(f"checkpoint {name} config SHA differs from frozen run")
        if canonical_sha256(payload.get("input_identity", {})) != canonical_sha256(identity):
            raise RuntimeError(f"checkpoint {name} input identity differs from frozen run")
        if payload.get("producer_fingerprint_sha256") != meta["producer_fingerprint"]["fingerprint_sha256"]:
            raise RuntimeError(f"checkpoint {name} producer fingerprint differs from frozen run")
        tensor_sha = _state_sha256(payload["state_dict"])
        if tensor_sha != record.get("tensor_sha256"):
            raise RuntimeError(f"checkpoint {name} model tensor SHA mismatch")
        scaler_check = _verify_checkpoint_scaler(payload, record, raw_config, meta, inputs)
        checkpoint_checks[name] = {
            "file_sha256": sha256_file(path),
            "expected_file_sha256": record.get("sha256"),
            "tensor_sha256": tensor_sha,
            "expected_tensor_sha256": record.get("tensor_sha256"),
            "epoch": int(record.get("epoch", -1)),
            "scaler": scaler_check,
            "pass": True,
        }
        checkpoint_epochs[name] = int(record.get("epoch", -1))

    artifact_files = meta.get("artifact_files", {})
    history_record = artifact_files.get("history")
    if not isinstance(history_record, Mapping):
        raise RuntimeError("frozen history.json file binding is missing")
    history_path = Path(history_record["path"])
    if not history_path.is_absolute():
        history_path = ROOT / history_path
    if not history_path.is_file():
        raise RuntimeError("frozen history.json is missing")
    if sha256_file(history_path) != history_record.get("sha256") or history_path.stat().st_size != int(
        history_record.get("bytes", -1)
    ):
        raise RuntimeError("history.json file SHA/bytes mismatch")
    history = _read_json(history_path)
    if not isinstance(history, list):
        raise RuntimeError("history.json is not a list")
    history_check = _validate_history(raw_config, history, meta, checkpoint_epochs)
    metrics_record = artifact_files.get("metrics")
    if not isinstance(metrics_record, Mapping):
        raise RuntimeError("frozen metrics.json file binding is missing")
    metrics_path = Path(metrics_record["path"])
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    if not metrics_path.is_file():
        raise RuntimeError("frozen metrics.json is missing")
    if sha256_file(metrics_path) != metrics_record.get("sha256") or metrics_path.stat().st_size != int(
        metrics_record.get("bytes", -1)
    ):
        raise RuntimeError("metrics.json file SHA/bytes mismatch")
    metrics = meta.get("metrics")
    if _read_json(metrics_path) != metrics:
        raise RuntimeError("metrics.json content differs from run_meta metrics")

    expected_predictions = {"select"}
    if bool(raw_config.get("eval_dev", True)):
        expected_predictions.add("dev")
    if bool(raw_config.get("eval_cal", True)):
        expected_predictions.add("cal")
    declared_predictions = meta.get("predictions", {})
    missing_predictions = sorted(expected_predictions - set(declared_predictions))
    if missing_predictions:
        raise RuntimeError(f"declared prediction packs missing: {missing_predictions}")
    packs: dict[str, dict[str, Any]] = {}
    prediction_checks: dict[str, Any] = {}
    for name, record in declared_predictions.items():
        pack = _verify_prediction_pack(name, record, run_dir, inputs)
        packs[name] = pack
        prediction_checks[name] = {key: value for key, value in pack.items() if key not in ("scores", "labels")}

    if "select" in packs:
        _compare_metrics(
            "select",
            evaluation.evaluate_scores(packs["select"]["scores"], packs["select"]["labels"]),
            metrics["select"],
        )
    if "dev" in packs:
        dev = packs["dev"]
        agent = dev["is_agentlab"]
        _compare_metrics(
            "dev_all", evaluation.evaluate_scores(dev["scores"], dev["labels"]), metrics["dev_all"]
        )
        _compare_metrics(
            "dev_common",
            evaluation.evaluate_scores(dev["scores"][~agent], dev["labels"][~agent]),
            metrics["dev_common"],
        )
        _compare_metrics(
            "dev_agentlab_diagnostic",
            evaluation.evaluate_scores(dev["scores"][agent], dev["labels"][agent]),
            metrics["dev_agentlab_diagnostic"],
        )
    if "cal" in packs:
        _compare_metrics(
            "cal", evaluation.evaluate_scores(packs["cal"]["scores"], packs["cal"]["labels"]), metrics["cal"]
        )
    cal_frozen_checks: dict[str, Any] = {}
    if bool(raw_config.get("eval_cal", True)):
        cal_frozen_checks = _compare_cal_frozen(metrics, packs)
    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "schema_version": 2,
        "config_sha256": meta["config_sha256"],
        "source_fingerprint_sha256": source["fingerprint_sha256"],
        "checkpoints": checkpoint_checks,
        "history": history_check,
        "history_file": history_record,
        "metrics_file": metrics_record,
        "predictions": prediction_checks,
        "cal_frozen_threshold": cal_frozen_checks,
        "input_binding_checked": inputs is not None,
    }

def evaluate_saved_predictions(run_dir: Path | str) -> dict[str, Any]:
    """Convenience read-only alias used by report/search consumers later."""
    return validate_saved_run(run_dir)


__all__ = [
    "TrainConfig",
    "TrainingInputs",
    "build_training_inputs",
    "preload_inputs",
    "verify_input_bindings",
    "producer_fingerprint",
    "archive_source_snapshot",
    "sample_epoch_indices",
    "make_batches",
    "ModelSelector",
    "score_episodes",
    "train_one",
    "validate_saved_run",
]
