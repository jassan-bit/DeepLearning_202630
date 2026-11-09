"""Synthetic §22 checks; no author runtime, external weights or market data.

Controlled algebra references use float64 atol=1e-10/rtol=1e-9; evaluation
batch invariance float32 atol=2e-6/rtol=2e-5. These are not Keras integration
checks or evidence that the original experiments have been reproduced.
"""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import e2_recipe_models as m


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(731)

    def test_shapes_positive_finite_gradients_and_batch_one(self):
        for name in m.MODEL_NAMES:
            for batch in (1, 3):
                with self.subTest(name=name, batch=batch):
                    model = m.build_model(name)
                    x = torch.randn(batch, 7, 5, requires_grad=True)
                    y = model(x)
                    self.assertEqual(y.shape, (batch,))
                    self.assertTrue(bool((y > 0).all()))
                    y.square().mean().backward()
                    self.assertTrue(bool(torch.isfinite(x.grad).all()))
                    for key, p in model.named_parameters():
                        self.assertIsNotNone(p.grad, key)
                        self.assertTrue(bool(torch.isfinite(p.grad).all()), key)
                        if 'weight' in key:
                            self.assertGreater(float(p.grad.abs().sum()), 0, key)

    def test_bn_population_statistics_and_moving_update(self):
        bn = m.PopulationBatchNorm1d(2).double()
        x = torch.tensor([[[1., 2., 8.], [-4., 0., 1.]]], dtype=torch.double)
        mean = x.mean((0, 2))
        var = ((x - mean[None, :, None]) ** 2).mean((0, 2))
        expected = (x - mean[None, :, None]) / (var[None, :, None] + .001).sqrt()
        torch.testing.assert_close(bn(x), expected, atol=1e-10, rtol=1e-9)
        torch.testing.assert_close(bn.running_var, .99 + .01 * var)
        torch.testing.assert_close(bn.running_mean, .01 * mean)
        self.assertFalse(torch.allclose(bn.running_var, .99 + .01 * x.var((0, 2))))
        bn(x)
        torch.testing.assert_close(bn.running_var, .99 * (.99 + .01 * var) + .01 * var)

    def test_eval_frozen_bn_and_batch_invariance(self):
        for name in m.MODEL_NAMES:
            model = m.build_model(name)
            x = torch.randn(4, 7, 5)
            model(x)
            model.eval()
            state = deepcopy(model.state_dict())
            with torch.no_grad():
                y = model(x)
                alone = model(x[:1])
                augmented = model(torch.cat((x[:1], x[1:] * 50)))[:1]
            torch.testing.assert_close(y[:1], alone, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(alone, augmented, atol=2e-6, rtol=2e-5)
            for key, val in state.items():
                self.assertTrue(torch.equal(val, model.state_dict()[key]), key)

    def test_same_even_alignment_against_explicit_cross_correlation(self):
        for k in (1, 3, 5, 8, 10, 20, 40):
            layer = m.SameConv1d(2, 3, k).double()
            x = torch.randn(2, 2, 7, dtype=torch.double)
            out = torch.zeros(2, 3, 7, dtype=torch.double)
            for t in range(7):
                out[:, :, t] = layer.bias
                for j in range(k):
                    source = t + j - (k - 1) // 2
                    if 0 <= source < 7:
                        out[:, :, t] += x[:, :, source] @ layer.weight[:, :, j].T
            torch.testing.assert_close(layer(x), out, atol=1e-10, rtol=1e-9)

    def test_fcn_exact_architecture(self):
        model = m.FCNRecipe()
        convs = [x for x in model.modules() if isinstance(x, nn.Conv1d)]
        self.assertEqual([(x.in_channels, x.out_channels, x.kernel_size[0]) for x in convs],
                         [(5, 128, 8), (128, 256, 5), (256, 128, 3)])
        self.assertTrue(all(x.bias is not None for x in convs))
        self.assertFalse(any(isinstance(x, nn.Dropout) for x in model.modules()))
        self.assertEqual(model.head.in_features, 128)

    def test_resnet_three_projected_shortcuts_and_initial_bn_only_main(self):
        model = m.ResNetRecipe().eval()
        self.assertEqual(sum(isinstance(x, nn.Conv1d) for x in model.modules()), 12)
        self.assertEqual(sum(isinstance(x, m.PopulationBatchNorm1d) for x in model.modules()), 13)
        self.assertEqual([b.shortcut[0].out_channels for b in model.blocks], [64, 128, 128])
        self.assertEqual(model.blocks[2].shortcut[0].in_channels, 128)
        raw = torch.randn(2, 5, 7)
        seen = []
        hook = model.blocks[0].shortcut.register_forward_pre_hook(lambda mod, xs: seen.append(xs[0].clone()))
        with torch.no_grad():
            model.blocks[0].initial_bn.bias.fill_(9)
            model.blocks[0](raw)
        hook.remove()
        self.assertTrue(torch.equal(seen[0], raw))
        b = model.blocks[2]
        x = torch.ones(2, 128, 7)
        with torch.no_grad():
            for p in b.main.parameters():
                p.zero_()
            b.shortcut[0].weight.zero_()
            b.shortcut[0].bias.zero_()
            self.assertTrue(torch.equal(b(x), torch.zeros_like(x)))
            # Identity substitution would produce ones, not zero.

    def test_inception_linear_branches_six_modules_and_two_shortcuts(self):
        model = m.InceptionRecipe()
        self.assertEqual(len(model.blocks), 6)
        self.assertEqual(len(model.shortcuts), 2)
        for block in model.blocks:
            self.assertEqual([c.kernel_size[0] for c in block.branches], [40, 20, 10])
            self.assertEqual(block(torch.randn(2, block.bottleneck.in_channels, 7)).shape, (2, 128, 7))
            self.assertFalse(any(isinstance(v, nn.ReLU) for v in block.branches.modules()))
        self.assertTrue(all(v.bias is None for v in model.modules() if isinstance(v, nn.Conv1d)))
        self.assertFalse(any(isinstance(v, nn.Dropout) for v in model.modules()))

    def test_pool_excludes_absent_positions_even_when_negative(self):
        x = torch.tensor([[[-9., -4., -7., -6., -8., -3., -5.]]])
        torch.testing.assert_close(m.same_max_pool3(x), torch.tensor([[[-4., -4., -4., -6., -3., -3., -3.]]]))

    def test_reject_non_d7_invalid_inputs_and_unknown_model(self):
        for model in [m.build_model(name) for name in m.MODEL_NAMES]:
            for x in (torch.zeros(2, 8, 5), torch.zeros(2, 7, 6), torch.zeros(0, 7, 5),
                      torch.full((2, 7, 5), float('nan')), torch.zeros(2, 7, 5, dtype=torch.long)):
                with self.assertRaises(ValueError):
                    model(x)
        with self.assertRaises(ValueError):
            m.build_model('FCN1D')

    def test_lstm_gate_equations_initialization_and_zero_state(self):
        layer = m.recipe_lstm(5, 25).double()
        hsize = layer.hidden_size
        bias = layer.bias_ih_l0 + layer.bias_hh_l0
        expected = torch.zeros_like(bias)
        expected[hsize:2*hsize] = 1
        torch.testing.assert_close(bias, expected)
        torch.testing.assert_close(layer.weight_hh_l0.T @ layer.weight_hh_l0,
                                   torch.eye(hsize, dtype=torch.double), atol=3e-7, rtol=3e-7)
        x = torch.randn(2, 7, 5, dtype=torch.double)
        h = torch.zeros(2, hsize, dtype=torch.double)
        c = h.clone()
        manual = []
        for step in x.unbind(1):
            i, f, g, o = (step @ layer.weight_ih_l0.T + h @ layer.weight_hh_l0.T + bias).chunk(4, -1)
            c = f.sigmoid() * c + i.sigmoid() * g.tanh()
            h = o.sigmoid() * c.tanh()
            manual.append(h)
        torch.testing.assert_close(layer(x)[0], torch.stack(manual, 1), atol=1e-10, rtol=1e-9)
        first = layer(x)[0]
        layer(x * 10)
        self.assertTrue(torch.equal(first, layer(x)[0]))

    def test_sae_repeats_final_vector_not_stacked_sequence(self):
        model = m.SAERecipe().eval()
        seen = []
        hook = model.encoder2.register_forward_pre_hook(lambda mod, xs: seen.append(xs[0].clone()))
        x = torch.randn(2, 7, 5)
        model(x)
        hook.remove()
        torch.testing.assert_close(seen[0], m.repeat7(m.last_state(model.encoder1, x)))
        self.assertEqual(model.encoder2.hidden_size, 26)
        self.assertEqual([v.p for v in model.modules() if isinstance(v, nn.Dropout)], [.3, .3])
        self.assertFalse(any('decoder' in k for k in model.state_dict()))

    def test_sae_two_phases_targets_freezing_transfer_and_finetune(self):
        x = torch.randn(3, 7, 5) * 4
        prep = m.SAEPretraining(x, 'synthetic-fold-A', accept_auxiliary_scale=True)
        phase1_x, target1 = prep.phase1_data()
        self.assertTrue(torch.equal(phase1_x, x))
        opt1 = prep.new_optimizer(lr=.001)
        opt1.zero_grad()
        recon = prep.ae1(phase1_x)
        self.assertEqual(recon.shape, x.shape)
        F.mse_loss(recon, target1).backward()
        opt1.step()
        frozen = deepcopy(prep.ae1.state_dict())
        features, target2 = prep.prepare_phase2()
        torch.testing.assert_close(target1, target2, atol=0, rtol=0)
        self.assertEqual(features.shape, (3, 7, 25))
        self.assertFalse(features.requires_grad)
        self.assertFalse(prep.ae1.training)
        self.assertTrue(all(not p.requires_grad for p in prep.ae1.parameters()))
        opt2 = prep.new_optimizer(lr=.001)
        self.assertIsNot(opt1, opt2)
        self.assertEqual(len(opt2.state), 0)
        opt2.zero_grad()
        recon = prep.ae2(features)
        self.assertEqual(recon.shape, (3, 7, 5))
        F.mse_loss(recon, target2).backward()
        opt2.step()
        for key, val in frozen.items():
            self.assertTrue(torch.equal(val, prep.ae1.state_dict()[key]), key)
        model = prep.transfer_predictor()
        for src, dst in ((prep.ae1.encoder, model.encoder1), (prep.ae2.encoder, model.encoder2)):
            for p, q in zip(src.parameters(), dst.parameters()):
                self.assertTrue(torch.equal(p, q))
                self.assertNotEqual(p.data_ptr(), q.data_ptr())
                self.assertTrue(q.requires_grad)
        model(x).sum().backward()
        for enc in (model.encoder1, model.encoder2):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in enc.parameters()), 0)
        with self.assertRaises(RuntimeError):
            prep.prepare_phase2()

    def test_sae_scale_train_only_no_clipping_cache_and_fold_isolation(self):
        train = torch.ones(2, 7, 5) * 4
        prep = m.SAEPretraining(train, 'fold-A', accept_auxiliary_scale=True)
        self.assertEqual(prep.scale, 4)
        validation = torch.ones(1, 7, 5) * 20
        self.assertEqual(float(prep.reconstruction_target(validation).max()), 5)
        self.assertEqual(prep.scale, 4)
        train.fill_(99)  # Constructor detached/owned its data.
        self.assertEqual(float(prep.phase1_data()[0].max()), 4)
        features, target = prep.prepare_phase2()
        features.zero_(); target.zero_()
        self.assertGreater(float(prep.phase2_data()[0].abs().sum()), 0)
        self.assertEqual(float(prep.phase2_data()[1].max()), 1)
        other = m.SAEPretraining(torch.ones(1, 7, 5) * .2, 'fold-B', accept_auxiliary_scale=True)
        self.assertEqual(other.scale, 1)
        self.assertIsNone(other._phase2_x)
        self.assertNotEqual(other.fold_id, prep.fold_id)
        self.assertFalse(any(isinstance(v, nn.Dropout) for v in prep.ae1.modules()))
        self.assertFalse(any(isinstance(v, nn.Dropout) for v in prep.ae2.modules()))

    def test_sae_auxiliary_adaptation_and_phase_order_are_explicit(self):
        x = torch.zeros(1, 7, 5)
        with self.assertRaises(ValueError):
            m.SAEPretraining(x, 'fold')
        with self.assertRaises(ValueError):
            m.SAEPretraining(x, '', accept_auxiliary_scale=True)
        prep = m.SAEPretraining(x, 'fold', accept_auxiliary_scale=True)
        with self.assertRaises(RuntimeError):
            prep.transfer_predictor()
        with self.assertRaises(RuntimeError):
            prep.phase2_data()


if __name__ == '__main__':
    unittest.main()
