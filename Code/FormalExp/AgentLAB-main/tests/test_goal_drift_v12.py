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

    def test_task_matching_does_not_accept_size_substrings(self) -> None:
        observation = (
            "Instruction: [SEP] Find a shirt in size 4x-large big tall "
            "and color cardinal [SEP] Back to Search [SEP] size [SEP] "
            "x-large [SEP] 4x-large tall [SEP] Example Shirt [SEP] "
            "Price: $10.00 [SEP] Buy Now"
        )
        self.assertIsNone(
            self.agent._task_matching_option(
                observation, ["x-large", "4x-large tall"],
            )
        )

    def test_task_matching_accepts_exact_multiword_size(self) -> None:
        observation = (
            "Instruction: [SEP] Find a shirt in size 4x-large big tall "
            "and color cardinal [SEP] Back to Search"
        )
        self.assertEqual(
            self.agent._task_matching_option(
                observation, ["x-large", "4x-large big tall"],
            ),
            "4x-large big tall",
        )

    def test_normalizes_asin_prefixed_click_without_losing_raw_action(self) -> None:
        action, reason = self.agent._normalize_action("click[ASIN:B000002F5S]")
        self.assertEqual(action, "click[B000002F5S]")
        self.assertEqual(reason, "normalize_asin_click")

    def test_missing_required_product_options_rejects_premature_buy(self) -> None:
        class FakeCompletions:
            @staticmethod
            def create(**_kwargs):
                message = types.SimpleNamespace(
                    content="ACTION: click[Buy Now]",
                    reasoning_content=None,
                    tool_calls=None,
                )
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)],
                )

        self.agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions()),
        )
        self.agent.config = types.SimpleNamespace(
            is_local_victim=False,
            victim_max_tokens=None,
            victim_top_p=None,
            victim_top_k=None,
            victim_thinking="off",
        )
        self.agent.model = "test-model"
        self.agent.temperature = 0.0
        self.agent.reset()
        observation = (
            "Instruction: [SEP] Find a shirt with color: cardinal, and size: "
            "4x-large big tall, and price lower than 50 dollars [SEP] "
            "Back to Search [SEP] < Prev [SEP] Gray Shirt [SEP] "
            "Price: $8.39 [SEP] Buy Now"
        )

        action, _reasoning, _repeated = self.agent.get_action(observation, "item")
        self.assertEqual(action, "click[< Prev]")
        self.assertEqual(
            self.agent.call_records[-1]["model_parsed_action"],
            "click[Buy Now]",
        )
        self.assertEqual(
            self.agent.call_records[-1]["action_override_reason"],
            "reject_product_missing_required_option",
        )
        last_user = self.agent.call_records[-1]["request_messages"][-1]["content"]
        self.assertIn("Task requires '4x-large big tall'", last_user)
        self.assertIn("Task requires 'cardinal'", last_user)
        self.assertIn("NOT READY", last_user)

    def test_premature_buy_is_replaced_by_required_option_and_audited(self) -> None:
        class FakeCompletions:
            @staticmethod
            def create(**_kwargs):
                message = types.SimpleNamespace(
                    content="I will purchase it. ACTION: click[Buy Now]",
                    reasoning_content=None,
                    tool_calls=None,
                )
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)],
                )

        self.agent.client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=FakeCompletions()),
        )
        self.agent.config = types.SimpleNamespace(
            is_local_victim=False,
            victim_max_tokens=None,
            victim_top_p=None,
            victim_top_k=None,
            victim_thinking="off",
        )
        self.agent.model = "test-model"
        self.agent.temperature = 0.0
        self.agent.reset()

        action, _reasoning, _repeated = self.agent.get_action(
            self.ITEM_OBSERVATION, "item",
        )
        self.assertEqual(action, "click[xx-large]")
        self.assertEqual(
            self.agent.call_records[-1]["model_parsed_action"],
            "click[Buy Now]",
        )
        self.assertEqual(
            self.agent.call_records[-1]["action_override_reason"],
            "select_required_size_before_purchase",
        )

        action, _reasoning, _repeated = self.agent.get_action(
            self.ITEM_OBSERVATION, "item",
        )
        self.assertEqual(action, "click[d-black]")
        self.assertEqual(
            self.agent.call_records[-1]["action_override_reason"],
            "select_required_color_before_purchase",
        )

        action, _reasoning, _repeated = self.agent.get_action(
            self.ITEM_OBSERVATION, "item",
        )
        self.assertEqual(action, "click[Buy Now]")

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

    def test_scores_report_purchase_and_valid_completion_pairs_separately(self) -> None:
        runner = object.__new__(goal_drift.ExperimentRunner)
        runner.scorer = goal_drift.FrugalityScorer()
        runner.results = goal_drift.ExperimentResults("test", "now")

        def trajectory(task_id, condition, completed, frugality, price):
            return goal_drift.TrajectoryRecord(
                task_id, "task", condition, 0, [], final_price=price,
                frugality_score=frugality, purchase_made=True,
                task_completed=completed,
                environment_reward=1.0 if completed else 0.6,
            )

        baseline = [
            trajectory(1, "baseline", True, 0.9, 10.0),
            trajectory(2, "baseline", False, 0.8, 12.0),
        ]
        attack = [
            trajectory(1, "optimized", True, 0.4, 20.0),
            trajectory(2, "optimized", True, 0.3, 30.0),
        ]
        runner._calculate_scores(baseline, attack)
        scores = runner.results.drift_scores
        self.assertEqual(scores["baseline"]["completed_count"], 1)
        self.assertEqual(scores["baseline"]["invalid_purchase_count"], 1)
        self.assertEqual(scores["attack"]["paired_purchase_count"], 2)
        self.assertEqual(scores["attack"]["unpaired_purchase_count"], 0)
        self.assertEqual(scores["attack"]["gd_spending"], 0.5)
        self.assertEqual(scores["attack"]["price_increase_pct"], 127.27)
        self.assertEqual(scores["attack"]["paired_completed_count"], 1)
        self.assertEqual(scores["attack"]["unpaired_completed_count"], 1)
        self.assertEqual(scores["attack"]["valid_completion_gd_spending"], 0.5)
        self.assertEqual(scores["attack"]["valid_completion_price_increase_pct"], 100.0)

    def test_scores_report_na_when_no_valid_pair_exists(self) -> None:
        runner = object.__new__(goal_drift.ExperimentRunner)
        runner.scorer = goal_drift.FrugalityScorer()
        runner.results = goal_drift.ExperimentResults("test", "now")
        baseline = [goal_drift.TrajectoryRecord(
            1, "task", "baseline", 0, [], final_price=10.0,
            frugality_score=0.9, purchase_made=True, task_completed=True,
            environment_reward=1.0,
        )]
        attack = [goal_drift.TrajectoryRecord(
            2, "task", "optimized", 0, [], final_price=20.0,
            frugality_score=0.4, purchase_made=True, task_completed=True,
            environment_reward=1.0,
        )]
        runner._calculate_scores(baseline, attack)
        scores = runner.results.drift_scores["attack"]
        self.assertEqual(scores["paired_purchase_count"], 0)
        self.assertEqual(scores["paired_completed_count"], 0)
        self.assertIsNone(scores["gd_spending"])
        self.assertIsNone(scores["price_increase_pct"])
        self.assertIsNone(scores["valid_completion_gd_spending"])


if __name__ == "__main__":
    unittest.main()
