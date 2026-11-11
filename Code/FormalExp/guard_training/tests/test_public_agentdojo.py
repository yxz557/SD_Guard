from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from guard_training.public_agentdojo.prepare_traces import (
    build_sample,
    infer_defense,
    payload_was_observed,
)


class PublicAgentDojoPreparationTest(unittest.TestCase):
    def test_defense_inference(self) -> None:
        self.assertEqual(infer_defense("Meta-SecAlign-70B"), "none")
        self.assertEqual(
            infer_defense("Meta-SecAlign-70B-repeat_user_prompt"),
            "repeat_user_prompt",
        )

    def test_payload_observation(self) -> None:
        injections = {"placeholder": "INJECTED PAYLOAD"}
        self.assertTrue(
            payload_was_observed(
                injections,
                [{"role": "tool", "content": "prefix INJECTED PAYLOAD suffix"}],
            )
        )
        self.assertFalse(
            payload_was_observed(
                injections,
                [{"role": "tool", "content": "ordinary tool output"}],
            )
        )

    def test_build_sample_uses_security_as_hijack_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = (
                root
                / "Meta-SecAlign-70B"
                / "workspace"
                / "user_task_0"
                / "direct"
                / "injection_task_0.json"
            )
            source.parent.mkdir(parents=True)
            public = {
                "agentdojo_package_version": "unknown",
                "attack_type": "direct",
                "benchmark_version": "v1.2.1",
                "error": None,
                "evaluation_timestamp": "2026-01-01",
                "injection_task_id": "injection_task_0",
                "injections": {"slot": "ATTACK"},
                "messages": [
                    {"role": "user", "content": "do the task"},
                    {"role": "tool", "content": "ATTACK"},
                ],
                "pipeline_name": "local",
                "security": True,
                "suite_name": "workspace",
                "user_task_id": "user_task_0",
                "utility": False,
            }
            source.write_text(json.dumps(public), encoding="utf-8")
            sample_id, raw_bytes, truth_bytes, index = build_sample(
                source, root, "Meta-SecAlign-70B", public
            )
            raw = json.loads(raw_bytes)
            truth = json.loads(truth_bytes)
            self.assertTrue(sample_id.startswith("agentdojo-public-Meta-SecAlign-70B-"))
            self.assertEqual(raw["task"]["user_task_id"], "user_task_0")
            self.assertTrue(truth["hijacked"])
            self.assertTrue(truth["payload_present"])
            self.assertTrue(index["hijacked"])


if __name__ == "__main__":
    unittest.main()

