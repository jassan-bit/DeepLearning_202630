"""Algebraic/gradient/state tests using artificial tensors, never market files."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from e2_cpu_conv import (enable_cpu_crop, execution_report, cropped_conv2d,
                         _fused_crop)
from e2_nas_spectral import _OddKernelInception
from e2_nas_factory import build_nas_model
from e2_staged_training import state_sha256


class CPUConvolutionIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def _convolution_pair(self, height, width, kernel, dtype=torch.float64):
        generator = torch.Generator().manual_seed(101+height*100+width*10+kernel)
        x = torch.randn(2, 2, height, width, generator=generator, dtype=dtype).requires_grad_()
        weight = torch.randn(3, 2, kernel, kernel, generator=generator, dtype=dtype).requires_grad_()
        bias = torch.randn(3, generator=generator, dtype=dtype).requires_grad_()
        copied = [v.detach().clone().requires_grad_() for v in (x, weight, bias)]
        dense = F.conv2d(x, weight, bias, padding=kernel//2)
        cropped = cropped_conv2d(*copied, padding=kernel//2)
        tolerance = (2e-12, 2e-12) if dtype == torch.float64 else (5e-5, 2e-5)
        torch.testing.assert_close(cropped, dense, rtol=tolerance[0], atol=tolerance[1])
        cotangent = torch.randn(dense.shape, generator=generator, dtype=dtype)
        dense.backward(cotangent); cropped.backward(cotangent)
        for original, other in zip((x, weight, bias), copied):
            self.assertIsNotNone(other.grad)
            self.assertEqual(other.grad.shape, original.shape)
            torch.testing.assert_close(other.grad, original.grad,
                                       rtol=tolerance[0], atol=tolerance[1])
        if height == width == 1 and kernel > 1:
            gradient = copied[1].grad.clone()
            gradient[..., kernel//2, kernel//2] = 0
            self.assertEqual(torch.count_nonzero(gradient).item(), 0)

    def test_fp64_forward_input_weight_bias_gradients_for36_tiny_geometries(self):
        for height, width in ((1, 8), (2, 4), (4, 2), (1, 1), (3, 3), (7, 7)):
            for kernel in (1, 3, 5, 7, 9, 11):
                with self.subTest(height=height, width=width, kernel=kernel):
                    self._convolution_pair(height, width, kernel)

    def test_fp32_roundoff_bounds_cover_large_odd_kernels(self):
        for height, width, kernel in ((1, 8, 11), (2, 4, 9), (4, 2, 7),
                                      (1, 1, 11), (3, 3, 11), (7, 7, 11)):
            with self.subTest(shape=(height, width), kernel=kernel):
                self._convolution_pair(height, width, kernel, torch.float32)

    def test_fusion_crop_before_padding_preserves_every_kernel_gradient(self):
        for height, width in ((1, 8), (2, 4), (4, 2), (4, 8)):
            with self.subTest(shape=(height, width)), torch.random.fork_rng(devices=[]):
                torch.manual_seed(94)
                original = _OddKernelInception(2, 3, 6).double()
                other = deepcopy(original)
                x = torch.randn(2, 2, height, width, dtype=torch.float64, requires_grad=True)
                z = x.detach().clone().requires_grad_()
                dense = original(x)
                weight, bias, padding = _fused_crop(other, height, width)
                result = F.conv2d(z, weight, bias, padding=padding)
                torch.testing.assert_close(result, dense, rtol=2e-12, atol=2e-12)
                dense.square().mean().backward(); result.square().mean().backward()
                torch.testing.assert_close(z.grad, x.grad, rtol=2e-12, atol=2e-12)
                for first, second in zip(original.parameters(), other.parameters()):
                    self.assertIsNotNone(second.grad)
                    torch.testing.assert_close(second.grad, first.grad, rtol=2e-12, atol=2e-12)

    def test_unsupported_geometry_uses_the_ordinary_convolution(self):
        for options in ({"stride": 2, "padding": 2},
                        {"dilation": 2, "padding": 4},
                        {"groups": 2, "padding": 2},
                        {"padding": 0}, {"padding": "same"}):
            with self.subTest(options=options):
                groups = options.get("groups", 1)
                generator = torch.Generator().manual_seed(930)
                x = torch.randn(2, 2, 9, 9, generator=generator)
                w = torch.randn(4, 2//groups, 5, 5, generator=generator)
                b = torch.randn(4, generator=generator)
                torch.testing.assert_close(cropped_conv2d(x, w, b, **options),
                                           F.conv2d(x, w, b, **options), rtol=0, atol=0)
        x = torch.empty(2, 2, 1, 8, device="meta")
        w = torch.empty(3, 2, 11, 11, device="meta")
        self.assertEqual(tuple(cropped_conv2d(x, w, padding=5).shape), (2, 3, 1, 8))

    def test_even_kernel_and_invalid_inputs_preserve_pytorch_behaviour(self):
        x = torch.randn(2, 2, 4, 4)
        w = torch.randn(3, 2, 2, 2)
        torch.testing.assert_close(cropped_conv2d(x, w, padding=1),
                                   F.conv2d(x, w, padding=1), rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            cropped_conv2d(x, torch.randn(3, 4, 3, 3), padding=1)

    def _model_pair(self, widths, top_k, dtype, training):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(812)
            dense = build_nas_model("TimesNet", widths, (0, 1, 0),
                                    hpo={"top_k": top_k, "num_kernels": 6}).to(dtype=dtype)
            other = deepcopy(dense)
            enable_cpu_crop(other)
            dense.train(training); other.train(training)
            x = torch.randn(2, 7, 5, dtype=dtype, requires_grad=True)
            z = x.detach().clone().requires_grad_()
            torch.manual_seed(271)
            expected = dense(x)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(271)
            actual = other(z)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            tol = (2e-10, 2e-11) if dtype == torch.float64 else (3e-4, 4e-5)
            torch.testing.assert_close(actual, expected, rtol=tol[0], atol=tol[1])
            expected.square().sum().backward(); actual.square().sum().backward()
            torch.testing.assert_close(z.grad, x.grad, rtol=tol[0], atol=tol[1])
            for (name, first), (second_name, second) in zip(dense.named_parameters(), other.named_parameters()):
                self.assertEqual(name, second_name)
                self.assertIsNotNone(second.grad, name)
                torch.testing.assert_close(second.grad, first.grad, rtol=tol[0], atol=tol[1], msg=name)

    def test_timesnet_minimum_body_train_eval_and_frequency_choices_match(self):
        for top_k in (2, 3, 4):
            for training in (False, True):
                with self.subTest(top_k=top_k, training=training):
                    self._model_pair((16, 16, 16), top_k, torch.float64, training)

    def test_timesnet_reference_body_fp32_forward_and_all_gradients_match(self):
        self._model_pair((64, 64, 64), 4, torch.float32, False)

    def test_adamw_update_preserves_unused_weight_decay_and_full_optimizer_state(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            dense = build_nas_model("TimesNet", (16, 16, 16), (0, 1, 0),
                                    hpo={"top_k": 4, "num_kernels": 6}).double()
            other = enable_cpu_crop(deepcopy(dense))
            opts = [torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01, foreach=False)
                    for model in (dense, other)]
            x = torch.randn(2, 7, 5, dtype=torch.float64)
            target = torch.tensor([.1, .2], dtype=torch.float64)
            key = "blocks.0.inception_in.kernels.5.weight"
            before = dict(dense.named_parameters())[key][0, 0, 0, 0].item()
            for model, optimizer in zip((dense, other), opts):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(432)
                (model(x)-target).square().mean().backward()
                weight = dict(model.named_parameters())[key]
                self.assertEqual(weight.grad[0, 0, 0, 0].item(), 0.)
                self.assertTrue(all(p.grad is not None for p in model.parameters()))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                self.assertAlmostEqual(weight[0, 0, 0, 0].item(), before*(1-.001*.01), places=14)
            for first, second in zip(dense.parameters(), other.parameters()):
                torch.testing.assert_close(second, first, rtol=5e-10, atol=5e-11)
            states = [optimizer.state_dict() for optimizer in opts]
            self.assertEqual(states[0]["param_groups"], states[1]["param_groups"])
            for index, state in states[0]["state"].items():
                for key, expected in state.items():
                    torch.testing.assert_close(states[1]["state"][index][key], expected,
                                               rtol=5e-10, atol=5e-11)

    def test_installation_is_idempotent_preserves_parameters_metadata_and_rng(self):
        model = build_nas_model("TimesNet", (16, 32, 64), (1, 0, 1))
        metadata = deepcopy(model.architecture_metadata)
        keys = tuple(model.state_dict())
        parameters = tuple(model.parameters())
        initial = state_sha256(model)
        rng = torch.get_rng_state().clone()
        self.assertIs(enable_cpu_crop(model), model)
        installed = model.blocks[0].forward
        self.assertIs(enable_cpu_crop(model), model)
        self.assertIs(model.blocks[0].forward, installed)
        self.assertEqual(model.architecture_metadata, metadata)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertTrue(all(a is b for a, b in zip(parameters, model.parameters())))
        self.assertEqual(state_sha256(model), initial)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        report = execution_report(model)
        self.assertEqual(len(report["period_blocks"]), 3)
        self.assertEqual(len(report["inception_modules"]), 6)
        report["version"] = "mutated"
        self.assertNotEqual(execution_report(model)["version"], "mutated")

    def test_other_families_and_preexisting_execution_changes_are_rejected(self):
        with self.assertRaises(ValueError):
            enable_cpu_crop(nn.Linear(7, 1))
        with self.assertRaises(ValueError):
            enable_cpu_crop(build_nas_model("TimeMixer", (16, 16, 16), (0, 0, 0)))
        model = build_nas_model("TimesNet", (16, 16, 16), (0, 0, 0))
        with self.assertRaises(ValueError):
            enable_cpu_crop(model, name="TiDE")
        model.blocks[0].forward = lambda values: values
        with self.assertRaises(ValueError):
            enable_cpu_crop(model)

    def test_maximum_timesnet_parameter_count_is_preserved_without_allocating_weights(self):
        with torch.device("meta"):
            model = build_nas_model("TimesNet", (256, 256, 256), (1, 1, 1),
                                    hpo={"top_k": 4, "num_kernels": 6})
        count = sum(p.numel() for p in model.parameters())
        enable_cpu_crop(model)
        self.assertEqual(count, 225075755)
        self.assertEqual(sum(p.numel() for p in model.parameters()), count)


if __name__ == "__main__":
    unittest.main()
