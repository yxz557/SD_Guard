from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from sd_latency.output_paths import default_output_root, resolve_experiment_output_dir


class OutputPathTests(unittest.TestCase):
    def test_default_root_is_anchored_at_sd_guard(self) -> None:
        expected_repository = next(
            parent for parent in Path(__file__).resolve().parents if parent.name == "SD_Guard"
        )
        self.assertEqual(
            default_output_root(),
            expected_repository / "SD_latency_test" / "outputs",
        )

    def test_default_path_uses_shared_root_and_experiment_descriptor(self) -> None:
        root = Path("repository") / "SD_latency_test" / "outputs"
        with (
            patch("sd_latency.output_paths.default_output_root", return_value=root),
            patch.object(Path, "exists", return_value=False),
        ):
            result = resolve_experiment_output_dir(
                None,
                timestamp="20261014_220000",
                descriptor="both_seed12345_tok256",
            )

        self.assertEqual(
            result,
            root / "20261014_220000_both_seed12345_tok256",
        )

    def test_default_path_adds_suffix_instead_of_reusing_an_experiment(self) -> None:
        root = Path("repository") / "SD_latency_test" / "outputs"
        with (
            patch("sd_latency.output_paths.default_output_root", return_value=root),
            patch.object(Path, "exists", side_effect=(True, False)),
        ):
            result = resolve_experiment_output_dir(
                None,
                timestamp="20261014_220000",
                descriptor="mode-check_seed12345_tok32",
            )

        self.assertEqual(
            result,
            root / "20261014_220000_mode-check_seed12345_tok32_01",
        )


if __name__ == "__main__":
    unittest.main()
