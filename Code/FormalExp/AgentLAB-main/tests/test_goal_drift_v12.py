from __future__ import annotations

import importlib.util
import shutil
import stat
import sys
import tempfile
import types
import unittest
from dataclasses import asdict
from pathlib import Path


AGENTLAB_ROOT = Path(__file__).resolve().parents[1]
if str(AGENTLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTLAB_ROOT))

try:
    import openai  # noqa: F401
except ImportError:
    stub = types.ModuleType("openai")
    stub.OpenAI = object
    sys.modules["openai"] = stub

MODULE_PATH = AGENTLAB_ROOT / "attacks" / "goal_drift" / "goal_drift_v12.py"
SPEC = importlib.util.spec_from_file_location("goal_drift_v12_under_test", MODULE_PATH)
goal_drift = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = goal_drift
SPEC.loader.exec_module(goal_drift)


class GoalDriftV12Tests(unittest.TestCase):
    ITEM_OBSERVATION = (
        "Instruction: [SEP] Find a black shirt in size xx-large [SEP] "
        "Back to Search [SEP] < Prev [SEP] color [SEP] a-yellow [SEP] "
        "d-black [SEP] navy [SEP] size [SEP] small [SEP] medium [SEP] "
        "xx-large [SEP] Example Shirt [SEP] Price: $19.99 [SEP] Buy Now"
    )

    def setUp(self) -> None:
        self.agent = object.__new__(goal_drift.FrugalShoppingAgent)
        temp_root = AGENTLAB_ROOT / "tests" / ".tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="goal-drift-test-", dir=temp_root))

    def tearDown(self) -> None:
        for path in sorted(self.temp_dir.rglob("*"), reverse=True):
            try:
                path.chmod(stat.S_IWRITE | stat.S_IREAD)
            except OSError:
                pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extracts_every_size_and_color_without_product_title(self) -> None:
        self.assertEqual(
            self.agent._extract_options(self.ITEM_OBSERVATION, "color"),
            ["a-yellow", "d-black", "navy"],
        )
        self.assertEqual(
            self.agent._extract_options(self.ITEM_OBSERVATION, "size"),
            ["small", "medium", "xx-large"],
        )

    def test_finds_task_matching_prefixed_color_and_size(self) -> None:
        colors = self.agent._extract_options(self.ITEM_OBSERVATION, "color")
        sizes = self.agent._extract_options(self.ITEM_OBSERVATION, "size")
        self.assertEqual(
            self.agent._task_matching_option(self.ITEM_OBSERVATION, colors),
            "d-black",
        )
        self.assertEqual(
            self.agent._task_matching_option(self.ITEM_OBSERVATION, sizes),
            "xx-large",
        )

    def test_detects_three_repetitions_of_two_action_cycle(self) -> None:
        self.agent.repeated_action_count = 0
        self.agent.action_history = ["search[x]", "click[a]"] * 3
        self.assertTrue(self.agent.should_terminate())

    def test_memory_upsert_is_idempotent_for_cumulative_counters(self) -> None:
        memory = goal_drift.InjectionMemorySystem(
            str(self.temp_dir / "memory.json"), verbose=False,
        )
        payload = goal_drift.InjectionPayload(
            "test injection", "test", "instruction", "aggressive",
            uses=7, successes=2,
        )
        memory.add_injection(payload)
        memory.add_injection(payload)
        self.assertEqual(memory.injections[0]["uses"], 7)
        self.assertEqual(memory.injections[0]["successes"], 2)

    def test_step_record_has_unambiguous_before_after_state(self) -> None:
        record = goal_drift.StepRecord(
            0, "results", "before", "", [], "click[item]", "", 0.0, False,
            current_product_price=0.0,
            observation_after="after",
            page_type_after="item",
            current_product_price_before=0.0,
            current_product_price_after=19.99,
        )
        data = asdict(record)
        self.assertEqual(data["page_type"], "results")
        self.assertEqual(data["page_type_after"], "item")
        self.assertEqual(data["current_product_price_before"], 0.0)
        self.assertEqual(data["current_product_price_after"], 19.99)


if __name__ == "__main__":
    unittest.main()
