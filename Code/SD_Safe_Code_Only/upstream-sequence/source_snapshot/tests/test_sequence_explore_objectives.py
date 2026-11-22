"""Mask, gradient and target-semantics tests for Lane T objectives."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from sdsafe_sequence_explore.objectives import (
    ObjectiveConfig,
    aux_endpoint_loss,
    bag_ranking_loss,
    compute_objective,
    endpoint_bce,
    mil_bag_loss,
    mil_pool,
)


def batch(**overrides) -> SimpleNamespace:
    base = dict(
        padding_mask=torch.tensor(
            [[True, True, True, False], [True, True, True, True]], dtype=torch.bool
        ),
        ct_mask=torch.tensor(
            [[True, False, True, False], [True, True, False, False]], dtype=torch.bool
        ),
        event_family=torch.tensor([[0, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.int64),
        policy_labels={
            "L1": torch.tensor([[-1, 1, -1, -1], [0, 0, -1, -1]], dtype=torch.int8),
            "L3": torch.tensor([[0, 1, 0, -1], [0, 1, 0, -1]], dtype=torch.int8),
        },
        policy_masks={
            "L1": torch.tensor([[False, False, True, True], [False, False, True, True]], dtype=torch.bool),
            "L3": torch.tensor([[False, False, False, True], [False, False, False, True]], dtype=torch.bool),
        },
        prefix_state=torch.tensor([[0, 1, 1, -1], [0, 0, 0, -1]], dtype=torch.int8),
        exact=torch.tensor([[True, True, True, False], [True, True, True, False]], dtype=torch.bool),
        bag_label=torch.tensor([0.0, 0.0], dtype=torch.float32),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class EndpointLossTests(unittest.TestCase):
    def test_masks_unknown_padding_and_e_are_excluded(self) -> None:
        b = batch()
        logits = torch.tensor([[-1.0, 1.0, 99.0, 0.0], [2.0, -2.0, 0.0, 9.0]])
        labels = torch.tensor([[-1, 1, -1, -1], [0, 0, -1, -1]], dtype=torch.int8)
        policy_mask = torch.tensor([[False, False, True, True], [False, False, True, True]], dtype=torch.bool)
        loss, stats = endpoint_bce(logits, labels, b.padding_mask, policy_mask, length_weighting="endpoint")
        manual = (F.softplus(torch.tensor(-1.0)) + F.softplus(torch.tensor(2.0)) + F.softplus(torch.tensor(-2.0))) / 3
        self.assertTrue(math.isclose(float(loss), float(manual), rel_tol=1e-6))
        self.assertEqual(stats["supervised_endpoints"], 3)
        self.assertEqual(stats["positive_mass"], 1.0)
        self.assertEqual(stats["negative_mass"], 2.0)

    def test_unknown_logit_change_has_no_effect(self) -> None:
        b = batch()
        labels = torch.tensor([[-1, 0, -1, -1], [0, 0, -1, -1]], dtype=torch.int8)
        mask = torch.tensor([[False, False, True, True], [False, False, True, True]], dtype=torch.bool)
        first = torch.tensor([[-1.0, 0.5, 99.0, 0.0], [2.0, -2.0, 0.0, 9.0]])
        second = first.clone()
        second[0, 0] = 1e6
        second[0, 2] = -1e6
        loss_a, _ = endpoint_bce(first, labels, b.padding_mask, mask)
        loss_b, _ = endpoint_bce(second, labels, b.padding_mask, mask)
        self.assertTrue(torch.equal(loss_a, loss_b))

    def test_length_weighting_is_explicit(self) -> None:
        b = batch()
        labels = torch.tensor([[0, -1, -1, -1], [0, 0, 0, -1]], dtype=torch.int8)
        mask = torch.tensor([[False, True, True, True], [False, False, False, True]], dtype=torch.bool)
        logits = torch.tensor([[-1.0, 99.0, 99.0, 0.0], [2.0, -2.0, 0.0, 9.0]])
        endpoint, _ = endpoint_bce(logits, labels, b.padding_mask, mask, length_weighting="endpoint")
        episode, _ = endpoint_bce(logits, labels, b.padding_mask, mask, length_weighting="episode")
        first = F.softplus(torch.tensor(-1.0))
        second = (F.softplus(torch.tensor(2.0)) + F.softplus(torch.tensor(-2.0)) + F.softplus(torch.tensor(0.0))) / 3
        self.assertTrue(math.isclose(float(endpoint), float((first + 3 * second) / 4), rel_tol=1e-6))
        self.assertTrue(math.isclose(float(episode), float((first + second) / 2), rel_tol=1e-6))
        self.assertFalse(math.isclose(float(endpoint), float(episode), rel_tol=1e-4))

    def test_class_weight_applied_once_and_mass_reported(self) -> None:
        b = batch()
        labels = torch.zeros((1, 4), dtype=torch.int8)
        labels[0, 1] = 1
        mask = torch.zeros((1, 4), dtype=torch.bool)
        logits = torch.zeros((1, 4), dtype=torch.float32)
        loss, stats = endpoint_bce(
            logits, labels, torch.ones((1, 4), dtype=torch.bool), mask, pos_weight=3.0
        )
        expected = (3.0 * math.log(2.0) + 3.0 * math.log(2.0)) / 4.0
        self.assertTrue(math.isclose(float(loss), expected, rel_tol=1e-6))
        self.assertEqual(stats["positive_endpoints"], 1)
        self.assertEqual(stats["negative_endpoints"], 3)
        self.assertEqual(stats["positive_mass"], 3.0)
        self.assertEqual(stats["negative_mass"], 3.0)
        self.assertEqual(stats["class_weight_positive"], 3.0)
        self.assertTrue(stats["class_weight_applied"])

    def test_pos_weight_one_positive_one_negative_mass_regression(self) -> None:
        logits = torch.zeros((1, 2), dtype=torch.float32)
        labels = torch.tensor([[0, 1]], dtype=torch.int8)
        loss, stats = endpoint_bce(
            logits, labels, torch.ones((1, 2), dtype=torch.bool), pos_weight=3.0
        )
        self.assertTrue(math.isclose(float(loss), 2.0 * math.log(2.0), rel_tol=1e-6))
        self.assertEqual(stats["positive_endpoints"], 1)
        self.assertEqual(stats["negative_endpoints"], 1)
        self.assertEqual(stats["positive_mass"], 3.0)
        self.assertEqual(stats["negative_mass"], 1.0)
        self.assertTrue(math.isclose(stats["raw_bce_sum_unweighted"], 2.0 * math.log(2.0), rel_tol=1e-6))
        self.assertTrue(math.isclose(stats["class_weighted_bce_sum"], 4.0 * math.log(2.0), rel_tol=1e-6))

    def test_sample_weight_effective_mass_is_explicit(self) -> None:
        logits = torch.zeros((1, 2), dtype=torch.float32)
        labels = torch.tensor([[1, 0]], dtype=torch.int8)
        weights = torch.tensor([[2.0, 0.5]], dtype=torch.float32)
        loss, stats = endpoint_bce(
            logits,
            labels,
            torch.ones((1, 2), dtype=torch.bool),
            pos_weight=3.0,
            sample_weight=weights,
        )
        self.assertTrue(math.isclose(stats["positive_mass"], 6.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(stats["negative_mass"], 0.5, rel_tol=1e-6))
        self.assertTrue(math.isclose(stats["sample_weight_effective_mass"], 2.5, rel_tol=1e-6))
        self.assertTrue(math.isclose(float(loss), 3.25 * math.log(2.0), rel_tol=1e-6))
        self.assertTrue(stats["sample_weight_used"])

    def test_gradient_direction(self) -> None:
        logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        labels = torch.tensor([[1, 0]], dtype=torch.int8)
        loss, _ = endpoint_bce(logits, labels, torch.ones((1, 2), dtype=torch.bool))
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 1]), 0.0)

    def test_right_padding_has_no_loss(self) -> None:
        logits = torch.tensor([[0.0, 0.0, 1e6]], requires_grad=True)
        labels = torch.tensor([[0, 0, -1]], dtype=torch.int8)
        valid = torch.tensor([[True, True, False]])
        loss, stats = endpoint_bce(logits, labels, valid)
        loss.backward()
        self.assertEqual(stats["supervised_endpoints"], 2)
        self.assertEqual(float(logits.grad[0, 2]), 0.0)


class MilLossTests(unittest.TestCase):
    def test_pool_excludes_e_and_padding_and_empty_is_minus_inf(self) -> None:
        logits = torch.tensor([[0.5, 100.0, 2.0, 100.0], [-3.0, -1.0, 100.0, 100.0]])
        valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
        ct = torch.tensor([[True, False, True, False], [True, True, False, False]])
        pooled, eligible, empty = mil_pool(logits, valid, ct, reduction="max")
        self.assertTrue(torch.equal(pooled, torch.tensor([2.0, -1.0])))
        self.assertEqual(int(eligible.sum()), 4)
        self.assertFalse(bool(empty.any()))
        empty_logits = logits.clone()
        empty_ct = torch.zeros_like(ct)
        pooled_empty, _, empty_flags = mil_pool(empty_logits, valid, empty_ct)
        self.assertTrue(bool(empty_flags.all()))
        self.assertTrue(bool(torch.all(torch.isinf(pooled_empty) & (pooled_empty < 0))))

    def test_logsumexp_approaches_max(self) -> None:
        logits = torch.tensor([[0.5, 2.0, -1.0]])
        valid = torch.ones((1, 3), dtype=torch.bool)
        ct = torch.ones((1, 3), dtype=torch.bool)
        smooth, _, _ = mil_pool(logits, valid, ct, reduction="logsumexp", beta=1.0)
        sharp, _, _ = mil_pool(logits, valid, ct, reduction="logsumexp", beta=20000.0)
        hard, _, _ = mil_pool(logits, valid, ct, reduction="max")
        expected_smooth = float(torch.logsumexp(logits, dim=1) - math.log(3.0))
        self.assertTrue(math.isclose(float(smooth), expected_smooth, rel_tol=1e-6))
        self.assertLess(abs(float(sharp) - float(hard)), 1e-4)

    def test_normalized_logsumexp_constant_bag_and_beta_validation(self) -> None:
        logits = torch.full((1, 5), 2.0)
        valid = torch.ones((1, 5), dtype=torch.bool)
        ct = torch.tensor([[True, True, True, False, False]])
        pooled, _, empty = mil_pool(logits, valid, ct, reduction="logsumexp", beta=7.0)
        self.assertFalse(bool(empty.any()))
        self.assertTrue(math.isclose(float(pooled[0]), 2.0, rel_tol=1e-6))
        with self.assertRaises(ValueError):
            mil_pool(logits, valid, ct, reduction="logsumexp", beta=0.0)

    def test_logsumexp_empty_bag_backward_is_finite(self) -> None:
        logits = torch.zeros((2, 3), requires_grad=True)
        valid = torch.ones((2, 3), dtype=torch.bool)
        ct = torch.tensor([[True, False, False], [False, False, False]])
        loss, stats = mil_bag_loss(
            logits, valid, ct, torch.tensor([1.0, 0.0]), reduction="logsumexp", beta=2.0
        )
        self.assertTrue(math.isfinite(float(loss)))
        self.assertEqual(stats["empty_bags"], 1)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_empty_bag_has_no_loss_and_no_nan(self) -> None:
        logits = torch.tensor([[1.0, 0.0], [2.0, 3.0]], requires_grad=True)
        valid = torch.ones((2, 2), dtype=torch.bool)
        ct = torch.tensor([[True, False], [False, False]])
        loss, stats = mil_bag_loss(logits, valid, ct, torch.tensor([1.0, 0.0]))
        self.assertTrue(math.isfinite(float(loss)))
        self.assertEqual(stats["empty_bags"], 1)
        self.assertEqual(stats["trainable_bags"], 1)
        loss.backward()
        self.assertEqual(float(logits.grad[1].abs().sum()), 0.0)

    def test_all_empty_bags_are_finite(self) -> None:
        logits = torch.zeros((2, 3), requires_grad=True)
        valid = torch.ones((2, 3), dtype=torch.bool)
        ct = torch.zeros((2, 3), dtype=torch.bool)
        loss, stats = mil_bag_loss(logits, valid, ct, torch.tensor([1.0, 0.0]))
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(stats["empty_bags"], 2)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_mil_gradient_direction(self) -> None:
        logits = torch.tensor([[0.0]], requires_grad=True)
        valid = torch.ones((1, 1), dtype=torch.bool)
        ct = torch.ones((1, 1), dtype=torch.bool)
        loss, _ = mil_bag_loss(logits, valid, ct, torch.tensor([1.0]))
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0.0)


class AuxAndCombinedTests(unittest.TestCase):
    def test_aux_prefix_state_excludes_unknown_and_e(self) -> None:
        b = batch()
        aux = torch.zeros((2, 4), requires_grad=True)
        config = ObjectiveConfig(endpoint_policy=None, aux_weight=0.1, aux_target="prefix_state")
        loss, stats = aux_endpoint_loss(aux, b, config)
        self.assertTrue(math.isfinite(float(loss)))
        # b0 CT exact rows: positions 0,2; b1: 0,1 -> 4 supervised, 2 positive.
        self.assertEqual(stats["supervised_endpoints"], 4)
        self.assertEqual(stats["positive_mass"], 1.0)
        self.assertEqual(stats["conflict_episodes"], 1)

    def test_separate_aux_head_and_combined_terms(self) -> None:
        b = batch(bag_label=torch.tensor([1.0, 0.0]))
        logits = torch.tensor([[0.0, 1.0, -1.0, 0.0], [0.5, -0.5, 0.2, 0.1]], requires_grad=True)
        aux = torch.zeros((2, 4), requires_grad=True)
        config = ObjectiveConfig(
            endpoint_policy="L1",
            endpoint_weight=1.0,
            mil_weight=0.5,
            mil_reduction="logsumexp",
            mil_beta=2.0,
            aux_weight=0.1,
            aux_target="policy",
            aux_policy="L3",
        )
        loss, stats = compute_objective(config, logits, b, aux_logits=aux)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertIn("endpoint", stats["terms"])
        self.assertIn("mil", stats["terms"])
        self.assertIn("aux", stats["terms"])
        self.assertEqual(stats["terms"]["aux"]["target"], "policy_L3")
        self.assertIn("mass_summary", stats)
        self.assertIn("endpoint_positive_weight_mass", stats["mass_summary"])
        self.assertIn("mil_positive_bag_weight_mass", stats["mass_summary"])
        self.assertIn("do not sum", stats["mass_summary"]["note"])
        self.assertEqual(stats["terms"]["mil"]["coefficient"], 0.5)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool(torch.isfinite(aux.grad).all()))

    def test_aux_requires_head(self) -> None:
        b = batch()
        config = ObjectiveConfig(endpoint_policy=None, aux_weight=0.1)
        with self.assertRaises(ValueError):
            compute_objective(config, torch.zeros((2, 4)), b, aux_logits=None)

    def test_ranking_extension(self) -> None:
        logits = torch.tensor([[-2.0], [-1.0]], requires_grad=True)
        valid = torch.ones((2, 1), dtype=torch.bool)
        ct = torch.ones((2, 1), dtype=torch.bool)
        loss, stats = bag_ranking_loss(logits, valid, ct, torch.tensor([1.0, 0.0]), margin=1.0)
        self.assertTrue(math.isclose(float(loss), 2.0, rel_tol=1e-6))
        self.assertEqual(stats["pairs"], 1)
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[1, 0]), 0.0)

    def test_empty_bag_with_mil_keeps_finite_combined_loss(self) -> None:
        b = batch(
            bag_label=torch.tensor([1.0, 0.0]),
            ct_mask=torch.tensor([[True, False, True, False], [False, False, False, False]]),
        )
        logits = torch.zeros((2, 4), requires_grad=True)
        config = ObjectiveConfig(endpoint_policy="L1", mil_weight=1.0)
        loss, stats = compute_objective(config, logits, b)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertEqual(stats["terms"]["mil"]["empty_bags"], 1)


if __name__ == "__main__":
    unittest.main()
