"""Focused causal/shape tests for the Lane T sequence model families."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sdsafe_sequence_explore.model import (
    FAMILIES,
    CausalTransformer,
    ModelConfig,
    build_model,
    count_parameters,
)

FAMILY_LIST = list(FAMILIES)
DROPOUT = 0.0
TOL = 1e-5


def tiny_config(family: str, **overrides) -> ModelConfig:
    values = dict(
        family=family,
        input_dim=16,
        hidden=16,
        proj_dim=16,
        num_layers=2,
        num_heads=4,
        dropout=DROPOUT,
        kernel_size=3,
        dilations=(1, 2, 4),
        transformer_ff=32,
    )
    values.update(overrides)
    return ModelConfig(**values)


def inputs(batch: int = 2, length: int = 9, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(batch, length, 16, generator=generator)
    valid = torch.ones(batch, length, dtype=torch.bool)
    if batch > 1 and length > 4:
        valid[1, length - 3 :] = False
    return features, valid


class ModelContractTests(unittest.TestCase):
    def test_all_families_shape_finite_and_padding(self) -> None:
        features, valid = inputs()
        for family in FAMILY_LIST:
            model = build_model(tiny_config(family)).eval()
            with torch.no_grad():
                logits = model(features, valid)
            self.assertEqual(tuple(logits.shape), tuple(valid.shape), family)
            self.assertTrue(bool(torch.isfinite(logits[valid]).all()), family)
            self.assertTrue(bool((logits[~valid] == 0).all()), family)
            self.assertGreater(count_parameters(model), 0)

    def test_mlp_is_positionwise(self) -> None:
        features, valid = inputs(batch=1)
        model = build_model(tiny_config("mlp")).eval()
        changed = features.clone()
        changed[:, 4] = torch.randn(1, 16)
        with torch.no_grad():
            before = model(features, valid)
            after = model(changed, valid)
        self.assertTrue(torch.allclose(before[:, :4], after[:, :4], atol=TOL, rtol=TOL))
        self.assertFalse(torch.allclose(before[:, 4], after[:, 4], atol=1e-7))

    def test_prefix_causality_each_family(self) -> None:
        features, valid = inputs(batch=1, length=10)
        for family in FAMILY_LIST:
            model = build_model(tiny_config(family)).eval()
            with torch.no_grad():
                full = model(features, valid)
                for prefix_length in (1, 3, 7, 9):
                    prefix = model(features[:, :prefix_length], valid[:, :prefix_length])
                    self.assertTrue(
                        torch.allclose(full[:, :prefix_length], prefix, atol=TOL, rtol=TOL),
                        f"{family} prefix {prefix_length}",
                    )

    def test_suffix_perturbation_does_not_change_prefix(self) -> None:
        features, valid = inputs(batch=1, length=10)
        changed = features.clone()
        changed[:, 6:] = torch.randn(1, 4, 16)
        for family in FAMILY_LIST:
            model = build_model(tiny_config(family)).eval()
            with torch.no_grad():
                before = model(features, valid)
                after = model(changed, valid)
            self.assertTrue(torch.allclose(before[:, :6], after[:, :6], atol=TOL, rtol=TOL), family)

    def test_padded_unpadded_batched_and_state_reset(self) -> None:
        features, _ = inputs(batch=1, length=10)
        other = torch.randn(1, 12, 16)
        for family in FAMILY_LIST:
            model = build_model(tiny_config(family)).eval()
            with torch.no_grad():
                sole = model(features, torch.ones(1, 10, dtype=torch.bool))
                padded = model(features, torch.ones(1, 10, dtype=torch.bool))
                self.assertTrue(torch.equal(sole, padded), family)
                # Episode alone vs in a batch with another episode and right padding.
                batch_features = torch.zeros(2, 12, 16)
                batch_features[0, :10] = features[0]
                batch_features[1, :12] = other[0]
                batch_valid = torch.zeros(2, 12, dtype=torch.bool)
                batch_valid[0, :10] = True
                batch_valid[1, :12] = True
                batched = model(batch_features, batch_valid)
                self.assertTrue(torch.allclose(sole[0], batched[0, :10], atol=TOL, rtol=TOL), family)
                self.assertTrue(torch.isfinite(batched[batch_valid]).all(), family)

    def test_transformer_corrupted_future_mask_is_detected(self) -> None:
        features, valid = inputs(batch=1, length=8)
        changed = features.clone()
        changed[:, 5:] = torch.randn(1, 3, 16)
        model = build_model(tiny_config("transformer")).eval()
        with torch.no_grad():
            before = model(features, valid)
            after = model(changed, valid)
        self.assertTrue(torch.allclose(before[:, :5], after[:, :5], atol=TOL, rtol=TOL))
        with patch.object(
            model,
            "_causal_attn_mask",
            new=lambda length, device: torch.zeros(length, length, dtype=torch.bool, device=device),
        ):
            with torch.no_grad():
                leaked_before = model(features, valid)
                leaked_after = model(changed, valid)
        self.assertFalse(
            torch.allclose(leaked_before[:, :5], leaked_after[:, :5], atol=1e-6, rtol=1e-6),
            "a deliberately bidirectional transformer must fail prefix invariance",
        )

    def test_meta_features_are_causal_and_toggleable(self) -> None:
        features, valid = inputs(batch=1, length=8)
        ct = torch.ones(1, 8, dtype=torch.bool)
        token_gap = torch.arange(8).unsqueeze(0)
        round_gap = torch.zeros(1, 8, dtype=torch.long)
        for family in FAMILY_LIST:
            config = tiny_config(family, use_meta_features=True)
            model = build_model(config).eval()
            with torch.no_grad():
                before = model(features, valid, ct_mask=ct, token_gap=token_gap, round_gap=round_gap)
            changed_gap = token_gap.clone()
            changed_gap[:, 6:] = 100
            with torch.no_grad():
                after = model(features, valid, ct_mask=ct, token_gap=changed_gap, round_gap=round_gap)
            self.assertTrue(torch.allclose(before[:, :6], after[:, :6], atol=TOL, rtol=TOL), family)

    def test_gru_meta_hidden_differs_from_projection(self) -> None:
        features, valid = inputs(batch=1, length=8)
        ct = torch.ones(1, 8, dtype=torch.bool)
        token_gap = torch.arange(8).unsqueeze(0)
        round_gap = torch.zeros(1, 8, dtype=torch.long)
        config = tiny_config(
            "gru", hidden=16, proj_dim=8, use_meta_features=True, meta_features=("ct", "token_gap")
        )
        model = build_model(config).eval()
        with torch.no_grad():
            full = model(features, valid, ct_mask=ct, token_gap=token_gap, round_gap=round_gap)
            for prefix_length in (1, 4, 7):
                prefix = model(
                    features[:, :prefix_length],
                    valid[:, :prefix_length],
                    ct_mask=ct[:, :prefix_length],
                    token_gap=token_gap[:, :prefix_length],
                    round_gap=round_gap[:, :prefix_length],
                )
                self.assertTrue(
                    torch.allclose(full[:, :prefix_length], prefix, atol=TOL, rtol=TOL),
                    f"GRU hidden!=proj meta prefix {prefix_length}",
                )
        self.assertEqual(model.meta_encoder.out_features, 8)

    def test_transformer_inactive_episode_is_finite_in_train_and_eval(self) -> None:
        features = torch.randn(2, 5, 16)
        valid = torch.tensor([[True, True, True, False, False], [False, False, False, False, False]])
        config = tiny_config("transformer")
        for training in (True, False):
            model = build_model(config)
            model.train(training)
            logits = model(features, valid)
            self.assertTrue(bool(torch.isfinite(logits[0, :3]).all()))
            self.assertFalse(bool(logits[1].any()))
            logits[0, :3].sum().backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        # Entirely inactive batch must also remain finite under native MHA.
        model = build_model(config).eval()
        all_pad = torch.zeros(1, 4, dtype=torch.bool)
        out = model(torch.randn(1, 4, 16), all_pad)
        self.assertFalse(bool(out.any()))

    def test_auxiliary_head_is_separate(self) -> None:
        features, valid = inputs(batch=2)
        model = build_model(tiny_config("gru", aux_head=True)).eval()
        with torch.no_grad():
            logits, aux = model(features, valid, return_aux=True)
        self.assertEqual(tuple(logits.shape), tuple(valid.shape))
        self.assertEqual(tuple(aux.shape), tuple(valid.shape))
        main_before = logits.clone()
        with torch.no_grad():
            for parameter in model.aux_head.parameters():
                parameter.add_(1.0)
            logits_after, _ = model(features, valid, return_aux=True)
        self.assertTrue(torch.equal(main_before, logits_after))

    def test_eval_mode_is_deterministic(self) -> None:
        features, valid = inputs()
        for family in FAMILY_LIST:
            model = build_model(tiny_config(family, dropout=0.5))
            model.eval()
            with torch.no_grad():
                first = model(features, valid)
                second = model(features, valid)
            self.assertTrue(torch.equal(first, second), family)
            model.train()
            self.assertTrue(model.training)

    def test_tcn_declared_receptive_field(self) -> None:
        config = tiny_config("tcn")
        self.assertEqual(config.receptive_field, 1 + (3 - 1) * (1 + 2 + 4))
        model = build_model(config).eval()
        self.assertIsInstance(model, torch.nn.Module)

    def test_transformer_rejects_bad_head_division(self) -> None:
        with self.assertRaises(ValueError):
            tiny_config("transformer", proj_dim=15, num_heads=4)


if __name__ == "__main__":
    unittest.main()
