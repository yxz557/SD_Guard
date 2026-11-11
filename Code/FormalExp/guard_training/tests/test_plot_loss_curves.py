from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guard_training.frozen_hidden_mlp.plot_loss_curves import (
    collect_histories,
    summarize_metric,
)


class PlotLossCurvesTest(unittest.TestCase):
    def test_collects_histories_and_summarizes_by_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for seed, losses in ((42, (1.0, 0.8)), (43, (1.2, 1.0))):
                run_directory = root / f"seed_{seed}"
                run_directory.mkdir()
                rows = [
                    {
                        "epoch": epoch,
                        "train_loss": loss,
                        "validation_loss": loss + 0.1,
                    }
                    for epoch, loss in enumerate(losses, start=1)
                ]
                (run_directory / "history.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            histories = collect_histories(root)
            epochs, means, deviations, counts = summarize_metric(
                histories, "train_loss"
            )

            self.assertEqual(len(histories), 2)
            self.assertEqual(epochs, [1, 2])
            self.assertEqual(counts, [2, 2])
            self.assertAlmostEqual(means[0], 1.1)
            self.assertAlmostEqual(means[1], 0.9)
            self.assertAlmostEqual(deviations[0], 0.1414213562)


if __name__ == "__main__":
    unittest.main()
