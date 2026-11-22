"""End-to-end training smoke on CPU with a synthetic tiny dataset (no GPU, no capture).

Validates `load_split`, `_batch_matrix`, `Arm`, `train_arm`, `score_rows`, episode max and
the metric helpers on the real code path before the real fits run.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdsafe_v4 import train_eval as TE  # noqa: E402


def make_dataset(root: Path, rows: int = 600, episodes: int = 30) -> None:
    rng = np.random.default_rng(0)
    d = root / "dataset" / "dev"
    d.mkdir(parents=True, exist_ok=True)
    import torch

    early = (
        torch.from_numpy(rng.standard_normal((rows, 2 * 4096)).astype(np.float32))
        .to(torch.bfloat16)
        .view(torch.uint16)
        .numpy()
    )
    final = (
        torch.from_numpy(rng.standard_normal((rows, 4096)).astype(np.float32))
        .to(torch.bfloat16)
        .view(torch.uint16)
        .numpy()
    )
    np.save(d / "early_bf16.npy", early)
    np.save(d / "final_bf16.npy", final)
    ep = np.asarray([f"e{i % episodes}" for i in range(rows)], dtype=object)
    groups = np.asarray([f"g{i % episodes}" for i in range(rows)], dtype=object)
    np.savez(
        d / "meta.npz",
        row_index=np.arange(rows, dtype=np.int64),
        cache_row=np.arange(rows, dtype=np.int64),
        round_id=np.ones(rows, dtype=np.int32),
        message_index=np.ones(rows, dtype=np.int32),
        call_index=np.zeros(rows, dtype=np.int32),
        segment_index=-np.ones(rows, dtype=np.int32),
        token_start=np.zeros(rows, dtype=np.int32),
        token_end=np.ones(rows, dtype=np.int32),
        positions=np.ones(rows, dtype=np.int32),
        episode_id=ep,
        group_id=groups,
        episode_label=np.asarray([int(e[1:]) % 2 for e in ep], dtype=bool),
        scope=np.asarray(["cot"] * rows, dtype=object),
        row_label=np.where(np.arange(rows) % 4 == 0, 1, 0).astype(np.int8),
        in_loss=np.ones(rows, dtype=bool),
    )


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="v4train"))
    real_art, real_ds, real_dev = TE.ART, TE.DS, TE.DEVICE
    try:
        TE.ART = tmp
        TE.DS = tmp / "dataset"
        TE.DEVICE = "cpu"
        make_dataset(tmp)
        import shutil as _sh
        _sh.copytree(tmp / "dataset" / "dev", tmp / "dataset" / "train")
        run = TE.train_arm("F", 0, (1, 2, 3), "ho", epochs=1)
        assert run["params"] > 0 and np.isfinite(run["final_loss"]), run
        logits = TE.score_rows("F", 0, "ho", "dev")
        assert logits.shape == (600,), logits.shape
        assert np.isfinite(logits).all()
        data = TE.load_split("dev")
        out = TE.episode_scores(logits, data["episode_id"])
        assert out["score"].shape[0] == 30, out["score"].shape
        labels = np.asarray([int(e[1:]) % 2 for e in out["episode_id"]], dtype=bool)
        thr = TE.fpr_threshold(labels, out["score"])
        cm = TE.confusions(labels, out["score"], thr)
        assert sum(cm.values()) == 30, cm
        print("training smoke OK", {"loss": run["final_loss"], "params": run["params"],
                                    "episode_scores": out["score"].shape[0], "cm": cm})
    finally:
        TE.ART, TE.DS, TE.DEVICE = real_art, real_ds, real_dev
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
