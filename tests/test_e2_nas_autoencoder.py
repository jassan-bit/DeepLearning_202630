"""Synthetic mechanism checks for new SAE/TiDE NAS variants, never market fits."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from e2_nas_common import NASConfig
from e2_nas_autoencoder import SAENAS, TiDENAS, SAEAutoencoderNAS, build_auxiliary_stages


def config(widths=(16, 32, 64), bits=(1, 1, 1)):
    return NASConfig(widths, bits)


class AutoencoderNASMechanisms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(280301)
        self.x = torch.randn(4, 7, 5)

    def test_sae_asymmetric_groups_positive_output_and_all_gradients(self):
        model = SAENAS(config())
        output = model(self.x)
        self.assertEqual(output.shape, (4,))
        self.assertTrue(bool((output > 0).all()))
        self.assertEqual(model.encoder1.hidden_size, 16)
        self.assertEqual(model.encoder2.input_size, 16)
        self.assertEqual(model.encoder2.hidden_size, 32)
        self.assertEqual((model.dense.in_features, model.dense.out_features), (32, 64))
        output.square().mean().backward()
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(bool(torch.isfinite(value.grad).all()), name)
        self.assertFalse(any('decoder' in name for name, _ in model.named_parameters()))

    def test_sae_dropout_bits_have_exactly_declared_locations(self):
        model = SAENAS(config(bits=(1, 0, 1)))
        self.assertEqual([model.dropout1.p, model.dropout2.p, model.dropout3.p], [0.2, 0, 0.2])
        ae1, ae2 = SAEAutoencoderNAS(config(bits=(1, 0, 1)), 1), SAEAutoencoderNAS(config(bits=(1, 0, 1)), 2)
        self.assertEqual([ae1.encoder_dropout.p, ae1.decoder_dropout.p], [0.2, 0.2])
        self.assertEqual([ae2.encoder_dropout.p, ae2.decoder_dropout.p], [0, 0])

    def test_auxiliary_scale_uses_copied_training_inputs_and_never_refits(self):
        train = self.x.clone()
        train[0, 0, 0] = 8.0
        expected = train.clone()
        stages = build_auxiliary_stages(config(), train, fold_id='synthetic-train-1')
        train.fill_(999)
        inputs, target = stages.phase1_data()
        self.assertTrue(torch.equal(inputs, expected))
        self.assertTrue(torch.equal(target, expected / 8))
        inputs.zero_()
        self.assertTrue(torch.equal(stages.phase1_data()[0], expected))
        validation = self.x * 100
        before = stages.scale
        transformed = stages.reconstruction_target(validation)
        self.assertEqual(stages.scale, before)
        self.assertTrue(torch.equal(transformed, validation / 8))
        self.assertGreater(float(transformed.abs().max()), 1.0)
        seen = []
        hook = stages.ae1.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        stages.ae1(expected)
        hook.remove()
        self.assertTrue(torch.equal(seen[0], expected))  # X, never X/s, enters encoder.

    def test_auxiliary_decoders_and_targets_keep_original_five_channels(self):
        stages = build_auxiliary_stages(config((64, 16, 32)), self.x, fold_id='fold')
        x1, y1 = stages.phase1_data()
        self.assertEqual(stages.ae1(x1).shape, y1.shape)
        x2, y2 = stages.prepare_phase2()
        self.assertEqual(x2.shape, (4, 7, 64))
        self.assertTrue(torch.equal(y1, y2))
        self.assertEqual(stages.ae2(x2).shape, (4, 7, 5))
        for module, inputs, targets in ((stages.ae1, x1, y1), (stages.ae2, x2, y2)):
            module.requires_grad_(True)
            module.train()
            loss = torch.nn.functional.mse_loss(module(inputs), targets)
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in module.parameters()))

    def test_phase_two_freezes_eval_dropout_and_representations(self):
        stages = build_auxiliary_stages(config(), self.x, fold_id='fold')
        before = deepcopy(stages.ae1.state_dict())
        with self.assertRaises(RuntimeError):
            stages.phase2_data()
        x2, _ = stages.prepare_phase2()
        self.assertFalse(stages.ae1.training)
        self.assertFalse(stages.ae1.encoder_dropout.training)
        self.assertTrue(all(not p.requires_grad for p in stages.ae1.parameters()))
        self.assertFalse(x2.requires_grad)
        self.assertTrue(torch.equal(x2[:, 0], x2[:, -1]))
        stages.ae1.train()  # API reasserts evaluation mode even after external misuse.
        a, b = stages.phase2_inputs(self.x), stages.phase2_inputs(self.x)
        self.assertFalse(stages.ae1.training)
        self.assertTrue(torch.equal(a, b))
        for name, value in stages.ae1.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)
        x2.fill_(123)
        self.assertFalse(torch.equal(x2, stages.phase2_data()[0]))
        with self.assertRaises(RuntimeError):
            stages.prepare_phase2()

    def test_transfer_copies_only_encoders_reenables_training_and_has_no_aliases(self):
        stages = build_auxiliary_stages(config(), self.x, fold_id='fold')
        with self.assertRaises(RuntimeError):
            stages.transfer_predictor()
        stages.prepare_phase2()
        predictor = stages.transfer_predictor()
        self.assertTrue(predictor.training)
        self.assertTrue(predictor.dropout1.training)
        self.assertTrue(all(p.requires_grad for p in predictor.parameters()))
        self.assertFalse(any('decoder' in name or 'reconstruction' in name for name, _ in predictor.named_parameters()))
        for copied, original in ((predictor.encoder1, stages.ae1.encoder), (predictor.encoder2, stages.ae2.encoder)):
            for name, value in copied.state_dict().items():
                source = original.state_dict()[name]
                self.assertTrue(torch.equal(value, source), name)
                self.assertNotEqual(value.data_ptr(), source.data_ptr())
        predictor(self.x).sum().backward()
        self.assertIsNotNone(predictor.encoder1.weight_ih_l0.grad)
        self.assertTrue(all(p.grad is None for p in stages.ae1.parameters()))

    def test_tide_asymmetric_widths_scalar_positive_and_finite_gradients(self):
        model = TiDENAS(config((256, 16, 64)))
        self.assertEqual(model.encoder.linear_a.out_features, 256)
        self.assertEqual(model.decoder.linear_a.out_features, 16)
        self.assertEqual(model.temporal_decoder.linear_a.out_features, 64)
        self.assertEqual(model.temporal_decoder.norm.normalized_shape, (8,))
        self.assertEqual(model.latent_features(self.x).shape, (4, 5, 8))
        output = model(self.x)
        self.assertEqual(output.shape, (4,))
        self.assertTrue(bool((output > 0).all()))
        output.square().mean().backward()
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(bool(torch.isfinite(value.grad).all()), name)
        for group in (model.encoder, model.decoder, model.temporal_decoder, model.residual_projection):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in group.parameters()), 0.0)

    def test_tide_fixed_zero_future_sentinel_and_group_output_dropout(self):
        model = TiDENAS(config(bits=(1, 0, 1))).eval()
        covariates = []
        handle = model.feature_encoder.register_forward_pre_hook(lambda module, args: covariates.append(args[0].detach().clone()))
        model(self.x)
        handle.remove()
        self.assertEqual(len(covariates), 2)
        self.assertTrue(torch.equal(covariates[0], self.x))
        self.assertTrue(torch.equal(covariates[1], torch.zeros(4, 1, 5)))
        self.assertEqual([model.dropout1.p, model.dropout2.p, model.dropout3.p], [0.2, 0, 0.2])
        seen = []
        hooks = [drop.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))
                 for drop in (model.dropout1, model.dropout2, model.dropout3)]
        model(self.x)
        for hook in hooks:
            hook.remove()
        self.assertEqual(seen, [torch.Size((4, 5, 16)), torch.Size((4, 5, 8)), torch.Size((4, 5, 8))])

    def test_both_families_evaluate_singletons_independently_and_constant_gradients(self):
        for constructor in (SAENAS, TiDENAS):
            model = constructor(config()).eval()
            self.assertTrue(torch.allclose(model(self.x[:1]), model(self.x)[:1], atol=2e-6, rtol=1e-5))
            constant = torch.ones(1, 7, 5, requires_grad=True)
            output = model(constant)
            output.sum().backward()
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertTrue(bool(torch.isfinite(constant.grad).all()))

    def test_contract_rejects_invalid_windows_and_provenance(self):
        for cls in (SAENAS, TiDENAS):
            model = cls(config())
            for invalid in (torch.zeros(1, 8, 5), torch.zeros(1, 7, 4), torch.zeros(0, 7, 5), torch.ones(1, 7, 5, dtype=torch.long)):
                with self.assertRaises(ValueError):
                    model(invalid)
            with self.assertRaises(ValueError):
                model(torch.full((1, 7, 5), float('nan')))
        with self.assertRaises(ValueError):
            build_auxiliary_stages(config(), self.x, fold_id=' ')


if __name__ == '__main__':
    unittest.main()
