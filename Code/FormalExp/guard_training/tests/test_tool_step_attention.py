from __future__ import annotations

import unittest

import torch

from guard_training.tool_step_attention.build_offline_dataset import pair_rounds
from guard_training.tool_step_attention.dataset import collate_attention
from guard_training.tool_step_attention.model import ToolStepAttentionClassifier
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


if __name__ == "__main__":
    unittest.main()
