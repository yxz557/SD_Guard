"""No-oracle metric and tie-handling tests for Lane T evaluation."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from sdsafe_sequence_explore import evaluation


def brute_auc(scores, labels) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    total = 0.0
    for positive in positives:
        for negative in negatives:
            total += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return total / (len(positives) * len(negatives))


class AucTests(unittest.TestCase):
    def test_tie_correct_auc_matches_bruteforce(self) -> None:
        cases = [
            ([0.5, 0.5, 0.2, 0.8], [1, 0, 0, 1]),
            ([1, 1, 0, 0], [1, 0, 1, 0]),
            ([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]),
            ([-np.inf, 0.0, 0.0, 2.0], [0, 1, 0, 1]),
        ]
        for scores, labels in cases:
            self.assertTrue(math.isclose(evaluation.roc_auc(scores, labels), brute_auc(scores, labels), rel_tol=1e-12))

    def test_perfect_and_reversed(self) -> None:
        self.assertEqual(evaluation.roc_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]), 1.0)
        self.assertEqual(evaluation.roc_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]), 0.0)

    def test_single_class_is_nan_not_fake(self) -> None:
        self.assertTrue(math.isnan(evaluation.roc_auc([0.1, 0.2], [0, 0])))


class OperatingPointTests(unittest.TestCase):
    def test_deterministic_threshold_and_confusion(self) -> None:
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        labels = [1, 0, 1, 0, 1, 0, 0, 0]
        first = evaluation.empirical_operating_point(scores, labels, 0.25)
        second = evaluation.empirical_operating_point(scores, labels, 0.25)
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertLessEqual(first["fpr"], 0.25 + 1e-12)
        point = evaluation.confusion_at(scores, labels, first["threshold"])
        for key in ("tp", "fp", "tn", "fn", "fnr", "fpr"):
            self.assertEqual(first[key], point[key])
        predicted = np.asarray(scores) >= first["threshold"]
        self.assertEqual(int(predicted.sum()), first["tp"] + first["fp"])

    def test_ties_are_deterministic_and_consistent(self) -> None:
        scores = [0.5, 0.5, 0.5, 0.2, 0.1]
        labels = [1, 0, 0, 0, 1]
        point = evaluation.empirical_operating_point(scores, labels, 0.5)
        repeated = evaluation.empirical_operating_point(scores, labels, 0.5)
        self.assertEqual(point["threshold"], repeated["threshold"])
        self.assertLessEqual(point["fpr"], 0.5 + 1e-12)
        self.assertEqual(point["tp"], int(np.sum((np.asarray(scores) >= point["threshold"]) & (np.asarray(labels) == 1))))
        self.assertEqual(point["fp"], int(np.sum((np.asarray(scores) >= point["threshold"]) & (np.asarray(labels) == 0))))

    def test_three_budgets_present(self) -> None:
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        labels = [1, 0, 1, 0, 1, 0, 0, 0]
        result = evaluation.evaluate_scores(scores, labels)
        self.assertEqual(set(result["operating_points"]), {"5%", "2%", "1%"})
        for point in result["operating_points"].values():
            self.assertLessEqual(point["fpr"], point["target_fpr"] + 1e-12)
            self.assertEqual(point["tp"] + point["fn"], 3)
            self.assertEqual(point["fp"] + point["tn"], 5)
            self.assertEqual(point["tp"] + point["fp"] + point["tn"] + point["fn"], len(scores))

    def test_frozen_threshold_application(self) -> None:
        cal_scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        cal_labels = [1, 0, 1, 0, 1, 0]
        threshold = evaluation.frozen_threshold(cal_scores, cal_labels, target_fpr=0.34)
        applied = evaluation.apply_frozen_threshold([0.95, 0.1], [1, 0], threshold)
        self.assertEqual(applied["n_pos"], 1)
        self.assertEqual(applied["n_neg"], 1)
        self.assertEqual(applied["tp"] + applied["fn"], 1)
        self.assertEqual(applied["fp"] + applied["tn"], 1)


class EpisodeScoreTests(unittest.TestCase):
    def test_ct_only_ignores_e_and_padding(self) -> None:
        logits = torch.tensor([[0.5, 100.0, 2.0, 100.0], [1.0, -1.0, 50.0, 50.0]])
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
        ct = torch.tensor([[True, False, True, False], [True, True, False, False]])
        scores = evaluation.episode_scores_from_logits(logits, valid, ct)
        self.assertTrue(np.array_equal(scores, [2.0, 1.0]))

    def test_empty_bag_is_minus_inf_and_stays_in_denominator(self) -> None:
        logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        valid = torch.ones((2, 2), dtype=torch.bool)
        ct = torch.tensor([[True, False], [False, False]])
        scores = evaluation.episode_scores_from_logits(logits, valid, ct)
        self.assertTrue(np.isfinite(scores[0]))
        self.assertTrue(np.isneginf(scores[1]))
        result = evaluation.evaluate_scores(scores, [1, 0])
        self.assertEqual(result["n_episodes"], 2)
        self.assertEqual(result["empty_bag_scores"], 1)
        self.assertEqual(result["auc"], 1.0)

    def test_missing_bag_never_alarms_even_at_threshold_minus_inf(self) -> None:
        scores = np.asarray([0.5, 0.1, -np.inf])
        labels = np.asarray([1, 0, 0])
        point = evaluation.confusion_at(scores, labels, float("-inf"))
        self.assertEqual(point["tp"], 1)
        self.assertEqual(point["fp"], 1)  # finite 0.1 >= -inf; missing -inf never alarms
        missing = evaluation.confusion_at(np.asarray([-np.inf]), np.asarray([0]), float("-inf"))
        self.assertEqual(missing["fp"], 0)
        with self.assertRaises(ValueError):
            evaluation.evaluate_scores(np.asarray([np.inf, 0.0]), np.asarray([1, 0]))

    def test_unknown_ct_positions_are_not_filtered(self) -> None:
        # Unknown rows are real inference positions; no truth filter is applied here.
        logits = torch.tensor([[0.1, 5.0, 0.2]])
        valid = torch.ones((1, 3), dtype=torch.bool)
        ct = torch.tensor([[True, True, True]])
        scores = evaluation.episode_scores_from_logits(logits, valid, ct)
        self.assertEqual(float(scores[0]), 5.0)

    def test_evaluate_episode_logits(self) -> None:
        logits = torch.tensor([[0.1, 0.2], [0.8, 0.9], [0.3, 0.4], [0.95, 0.1]])
        valid = torch.ones((4, 2), dtype=torch.bool)
        ct = torch.ones((4, 2), dtype=torch.bool)
        bag = torch.tensor([0.0, 1.0, 0.0, 1.0])
        result = evaluation.evaluate_episode_logits(logits, valid, ct, bag)
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["n_positives"], 2)
        self.assertEqual(result["n_negatives"], 2)


class GroupAndPayloadTests(unittest.TestCase):
    def test_summarise_by(self) -> None:
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [1, 0, 1, 0]
        sources = ["cot", "cot", "ctx40k", "ctx40k"]
        result = evaluation.summarise_by(scores, labels, sources)
        self.assertEqual(result["cot"]["own_roc_auc"], 1.0)
        self.assertEqual(result["ctx40k"]["own_roc_auc"], 1.0)
        self.assertEqual(result["cot"]["roc_scope"], "own_subgroup_roc")
        self.assertEqual(result["cot"]["n_positives"], 1)
        frozen = evaluation.summarise_by(scores, labels, sources, global_threshold=0.5)
        self.assertIn("global_threshold_confusion", frozen["cot"])

    def test_predictions_payload_roundtrip(self) -> None:
        logits = torch.zeros((2, 3))
        valid = torch.tensor([[True, True, False], [True, True, True]])
        ct = torch.tensor([[True, False, False], [True, True, True]])
        payload = evaluation.predictions_payload(
            logits=logits,
            valid_mask=valid,
            ct_mask=ct,
            prefix_state=torch.tensor([[0, 1, -1], [0, 0, 1]], dtype=torch.int8),
            exact=torch.tensor([[True, True, False], [True, True, True]]),
            bag_label=torch.tensor([0.0, 1.0]),
            episode_ids=["e0", "e1"],
            group_ids=["g0", "g1"],
            scopes=["cot", "agentlab"],
            lengths=torch.tensor([2, 3]),
            row_start=torch.tensor([0, 2]),
            row_end=torch.tensor([2, 5]),
            scores=np.asarray([0.0, 0.0]),
            split_name="dev",
        )
        self.assertTrue(np.array_equal(payload["is_common_dev"], [True, False]))
        self.assertTrue(np.array_equal(payload["is_agentlab"], [False, True]))
        self.assertEqual(payload["episode_ids"].tolist(), ["e0", "e1"])
        self.assertEqual(payload["valid_mask"].shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
