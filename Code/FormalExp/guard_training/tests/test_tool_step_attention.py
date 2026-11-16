from __future__ import annotations

import unittest

import torch

from guard_training.tool_step_attention.build_offline_dataset import (
    align_individual_tools,
    pair_rounds,
)
from guard_training.tool_step_attention.dataset import collate_attention
from guard_training.tool_step_attention.model import (
    ToolStepAttentionClassifier,
    hierarchical_logits,
)
from guard_training.tool_step_mean_pool.run_parameter_ablation import parameter_count


class ToolStepAttentionTest(unittest.TestCase):
    def test_parameter_matched_hidden_dimensions(self) -> None:
        baseline = parameter_count(2560, 256)
        self.assertEqual(baseline, 661250)
        self.assertLess(abs(parameter_count(5120, 127) - baseline), 600)
        self.assertLess(abs(parameter_count(7680, 84) - baseline), 600)
        self.assertLess(abs(parameter_count(2560, 514) - parameter_count(5120, 256)), 1000)
        self.assertLess(abs(parameter_count(2560, 771) - parameter_count(7680, 256)), 1500)

    def test_round_pairing_keeps_unmatched_final_decision(self) -> None:
        steps = [
            {
                "kind": "assistant_decision_end",
                "round_index": 0,
                "message_index": 1,
            },
            {"kind": "tool_round_end", "round_index": 0, "message_index": 2},
            {
                "kind": "assistant_decision_end",
                "round_index": 1,
                "message_index": 3,
            },
        ]
        vectors = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        decisions, results, mask, pairs, unpaired = pair_rounds(steps, vectors)
        torch.testing.assert_close(decisions, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))
        torch.testing.assert_close(results, torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
        self.assertEqual(mask.tolist(), [True, False])
        self.assertIsNone(pairs[1]["result_step_index"])
        self.assertEqual(unpaired, 0)

    def test_shared_attention_reuses_decision_weights_with_result_mask(self) -> None:
        items = [
            {
                "index": 0,
                "readout": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "label": torch.tensor(1),
                "decisions": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                ),
                "round_results": torch.tensor(
                    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
                ),
                "round_result_mask": torch.tensor([True, False]),
            },
            {
                "index": 1,
                "readout": torch.tensor([0.4, 0.3, 0.2, 0.1]),
                "label": torch.tensor(0),
                "decisions": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
                "round_results": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                "round_result_mask": torch.tensor([True]),
            },
        ]
        batch = collate_attention(items, "rct_shared")
        model = ToolStepAttentionClassifier(
            hidden_size=4,
            variant="rct_shared",
            attention_dim=3,
            head_hidden_dim=5,
            dropout=0.0,
            attention_dropout=0.0,
        )
        logits, attention = model(batch, return_attention=True)
        self.assertEqual(tuple(logits.shape), (2, 2))
        torch.testing.assert_close(
            attention["decision_weights"].sum(dim=1), torch.ones(2)
        )
        torch.testing.assert_close(
            attention["result_weights"].sum(dim=1), torch.ones(2)
        )
        self.assertEqual(float(attention["result_weights"][0, 1]), 0.0)

    def test_individual_tool_alignment_uses_ids_and_round_contexts(self) -> None:
        steps = [
            {"kind": "tool_call_end", "round_index": 0, "tool_call_id": "c1"},
            {"kind": "tool_call_end", "round_index": 0, "tool_call_id": "c2"},
            {"kind": "assistant_decision_end", "round_index": 0},
            {"kind": "tool_result_end", "round_index": 0, "tool_call_id": "c2"},
            {"kind": "tool_result_end", "round_index": 0, "tool_call_id": "c1"},
            {"kind": "tool_round_end", "round_index": 0},
        ]
        vectors = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        aligned = align_individual_tools(steps, vectors, [0, 1], [3, 4])
        self.assertEqual(aligned.result_to_call.tolist(), [1, 0])
        torch.testing.assert_close(
            aligned.call_round_context,
            torch.stack([vectors[2], vectors[2]]),
        )
        torch.testing.assert_close(
            aligned.result_round_context,
            torch.stack([vectors[5], vectors[5]]),
        )
        self.assertEqual(aligned.call_round_context_mask.tolist(), [True, True])
        self.assertEqual(aligned.result_round_context_mask.tolist(), [True, True])
        self.assertEqual(aligned.call_round_group.tolist(), [0, 0])
        self.assertEqual(aligned.result_round_group.tolist(), [0, 0])
        self.assertEqual(aligned.diagnostics["tool_pairs_exact_id"], 2)

    def test_public_results_use_ordered_round_context_fallback(self) -> None:
        steps = [
            {"kind": "tool_call_end", "round_index": 0, "tool_call_id": None},
            {"kind": "tool_call_end", "round_index": 0, "tool_call_id": None},
            {"kind": "assistant_decision_end", "round_index": 0},
            {"kind": "tool_result_end", "round_index": None, "tool_call_id": None},
            {"kind": "tool_result_end", "round_index": None, "tool_call_id": None},
            {"kind": "tool_round_end", "round_index": None},
        ]
        vectors = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        aligned = align_individual_tools(steps, vectors, [0, 1], [3, 4])
        self.assertEqual(aligned.result_to_call.tolist(), [0, 1])
        torch.testing.assert_close(
            aligned.result_round_context,
            torch.stack([vectors[5], vectors[5]]),
        )
        self.assertEqual(aligned.result_round_context_mask.tolist(), [True, True])
        self.assertEqual(aligned.result_round_group.tolist(), [5, 5])
        self.assertEqual(
            aligned.diagnostics["result_round_context_order_fallback"], 2
        )
        self.assertEqual(aligned.diagnostics["results_without_round_context"], 0)

    def test_individual_shared_attention_maps_each_result_to_its_call(self) -> None:
        torch.manual_seed(7)
        items = [
            {
                "index": 0,
                "readout": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "label": torch.tensor(1),
                "calls": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                ),
                "results": torch.tensor(
                    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                ),
                "result_to_call": torch.tensor([1, 0]),
            }
        ]
        batch = collate_attention(items, "rct_tool_shared")
        model = ToolStepAttentionClassifier(
            hidden_size=4,
            variant="rct_tool_shared",
            attention_dim=3,
            head_hidden_dim=5,
            dropout=0.0,
            attention_dropout=0.0,
        )
        logits, attention = model(batch, return_attention=True)
        self.assertEqual(tuple(logits.shape), (1, 2))
        torch.testing.assert_close(
            attention["call_weights"].sum(dim=1), torch.ones(1)
        )
        torch.testing.assert_close(
            attention["result_weights"].sum(dim=1), torch.ones(1)
        )
        torch.testing.assert_close(
            attention["result_weights"][0],
            attention["call_weights"][0].flip(0),
        )
        independent_batch = collate_attention(items, "rct_tool_independent")
        independent_model = ToolStepAttentionClassifier(
            hidden_size=4,
            variant="rct_tool_independent",
            attention_dim=3,
            head_hidden_dim=5,
            dropout=0.0,
            attention_dropout=0.0,
        )
        independent_logits, independent_attention = independent_model(
            independent_batch, return_attention=True
        )
        self.assertEqual(tuple(independent_logits.shape), (1, 2))
        torch.testing.assert_close(
            independent_attention["result_weights"].sum(dim=1), torch.ones(1)
        )

    def test_round_tool_attention_keeps_tool_specific_weights(self) -> None:
        torch.manual_seed(11)
        shared_round = torch.tensor(
            [[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]]
        )
        items = [
            {
                "index": 0,
                "readout": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "label": torch.tensor(1),
                "calls": torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                ),
                "results": torch.tensor(
                    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                ),
                "result_to_call": torch.tensor([0, 1]),
                "call_round_context": shared_round,
                "call_round_context_mask": torch.tensor([True, True]),
                "call_round_group": torch.tensor([0, 0]),
                "result_round_context": shared_round,
                "result_round_context_mask": torch.tensor([True, True]),
                "result_round_group": torch.tensor([0, 0]),
            }
        ]
        batch = collate_attention(items, "rct_round_tool_independent")
        model = ToolStepAttentionClassifier(
            hidden_size=4,
            variant="rct_round_tool_independent",
            attention_dim=3,
            head_hidden_dim=5,
            dropout=0.0,
            attention_dropout=0.0,
        )
        _, attention = model(batch, return_attention=True)
        torch.testing.assert_close(
            attention["call_weights"].sum(dim=1), torch.ones(1)
        )
        self.assertGreater(
            abs(
                float(attention["call_weights"][0, 0])
                - float(attention["call_weights"][0, 1])
            ),
            1e-6,
        )
        shared_batch = collate_attention(items, "rct_round_tool_shared")
        shared_model = ToolStepAttentionClassifier(
            hidden_size=4,
            variant="rct_round_tool_shared",
            attention_dim=3,
            head_hidden_dim=5,
            dropout=0.0,
            attention_dropout=0.0,
        )
        shared_logits, shared_attention = shared_model(
            shared_batch, return_attention=True
        )
        self.assertEqual(tuple(shared_logits.shape), (1, 2))
        torch.testing.assert_close(
            shared_attention["result_weights"].sum(dim=1), torch.ones(1)
        )

    def test_hierarchical_round_weights_do_not_favor_more_tools(self) -> None:
        tool_logits = torch.zeros((1, 3))
        round_logits = torch.zeros((1, 3))
        groups = torch.tensor([[0, 0, 1]])
        mask = torch.tensor([[True, True, True]])
        combined = hierarchical_logits(tool_logits, round_logits, groups, mask)
        weights = torch.softmax(combined, dim=1)
        torch.testing.assert_close(
            weights, torch.tensor([[0.25, 0.25, 0.50]])
        )


if __name__ == "__main__":
    unittest.main()
