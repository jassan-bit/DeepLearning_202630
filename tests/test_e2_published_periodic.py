"""Semantic/source parity checks for the prospective D7 periodic bodies."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_published_periodic as published


class FFTDomainTests(unittest.TestCase):
    def test_dc_is_excluded_even_when_every_frequency_ties_at_zero(self):
        values = torch.zeros(2, 8, 32)
        periods, weights = published.fft_for_period_non_dc(values, 3)
        self.assertTrue(np.isfinite(periods).all())
        self.assertTrue((periods >= 1).all())
        self.assertTrue(torch.equal(weights, torch.zeros(2, 3)))
        with self.assertRaises(ValueError):
            published.fft_for_period_non_dc(values, 5)

    def test_fft_keeps_the_author_batch_mean_selection_not_individual_topk(self):
        time = torch.arange(8, dtype=torch.float64)
        frequencies = torch.stack([torch.cos(2 * torch.pi * f * time / 8) for f in range(1, 5)])
        values = torch.stack((20 * frequencies[0] + frequencies[3],
                              30 * frequencies[1] + 5 * frequencies[2]))[:, :, None]
        periods, weights = published.fft_for_period_non_dc(values, 3)
        amplitude = torch.fft.rfft(values, dim=1).abs()
        indices = torch.topk(amplitude.mean(0).mean(-1)[1:], 3).indices + 1
        self.assertTrue(np.array_equal(periods, 8 // indices.numpy()))
        torch.testing.assert_close(weights, amplitude.mean(-1)[:, indices], rtol=0, atol=0)
        separate = torch.stack([published.fft_for_period_non_dc(row[None], 3)[1][0] for row in values])
        self.assertFalse(torch.equal(weights, separate))

    def test_invalid_name_dropout_and_missing_source_fail_before_construction(self):
        for name, dropout in (("Other", .1), ("TimesNet", -.1), ("TimeMixer", .6), ("TimesNet", float("nan"))):
            with self.subTest(name=name, dropout=dropout), self.assertRaises(ValueError):
                published.build_periodic(name, dropout, {})
        with self.assertRaises(ValueError):
            published.build_periodic("TimesNet", .1, {})


class PublishedPeriodicSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("E2_PUBLISHED_SOURCE_ROOTS")
        if not path:
            raise unittest.SkipTest("Pinned author trees not supplied; source integration remains pending")
        cls.roots = json.loads(Path(path).read_text(encoding="utf-8"))
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "previous_threads"):
            torch.set_num_threads(cls.previous_threads)

    def _model(self, name, dropout=0.):
        torch.manual_seed(20261112)
        return published.build_periodic(name, dropout, self.roots)

    def test_complete_forecast_and_gradients_match_pinned_source_at_shared_weights(self):
        for name in published.RECIPE_IDS:
            for training in (True, False):
                with self.subTest(model=name, training=training):
                    model = self._model(name).double().train(training)
                    root, _ = published.verify_source_tree(name, self.roots)
                    author = published._author_module(name, root).Model(published._args(name, 0.)).double().train(training)
                    author.load_state_dict(deepcopy(model.backbone.state_dict()))
                    values = torch.randn(2, 7, 5, dtype=torch.float64)
                    adapted_values = values.detach().clone().requires_grad_(True)
                    author_values = values.detach().clone().requires_grad_(True)
                    adapted = model.body_features(adapted_values)
                    if name == "TimesNet" and not training:
                        original = torch.cat([author(row[None], None, None, None)
                                              for row in author_values], dim=0)[:, 0, :]
                    else:
                        original = author(author_values, None, None, None)[:, 0, :]
                    torch.testing.assert_close(adapted, original, rtol=1e-10, atol=1e-10)
                    adapted.square().sum().backward()
                    original.square().sum().backward()
                    torch.testing.assert_close(adapted_values.grad, author_values.grad, rtol=1e-9, atol=1e-9)
                    official_parameters = dict(author.named_parameters())
                    for key, parameter in model.backbone.named_parameters():
                        if parameter.requires_grad:
                            self.assertIsNotNone(parameter.grad, key)
                            torch.testing.assert_close(parameter.grad, official_parameters[key].grad,
                                                       rtol=1e-8, atol=1e-9, msg=key)
                    torch.testing.assert_close(model(values), torch.nn.functional.softplus(model.readout(model.body_features(values))).squeeze(-1))

    def test_timesnet_eval_is_independent_of_later_companion_windows(self):
        model = self._model("TimesNet").double().eval()
        values = torch.randn(2, 7, 5, dtype=torch.float64)
        changed = values.clone()
        changed[1] = 100 * torch.randn(7, 5, dtype=torch.float64)
        torch.testing.assert_close(model(values)[0], model(changed)[0], rtol=0, atol=0)
        torch.testing.assert_close(model(values)[0], model(values[:1])[0], rtol=0, atol=0)
        self.assertTrue(model.architecture_metadata["inference_batch_size1_enforced"])

    def test_every_trainable_branch_gets_finite_gradients_and_a_tiny_optimizer_update(self):
        for name in published.RECIPE_IDS:
            with self.subTest(model=name):
                model = self._model(name).train()
                before = {key: value.detach().clone() for key, value in model.named_parameters()}
                optimizer = torch.optim.Adam([value for value in model.parameters() if value.requires_grad], lr=1e-3)
                values, target = torch.randn(2, 7, 5), torch.tensor([.4, .8])
                prediction = model(values)
                self.assertEqual(prediction.shape, (2,))
                self.assertTrue(bool((prediction > 0).all()))
                (prediction - target).square().mean().backward()
                for key, value in model.named_parameters():
                    if value.requires_grad:
                        self.assertIsNotNone(value.grad, key)
                        self.assertTrue(bool(torch.isfinite(value.grad).all()), key)
                    else:
                        self.assertIsNone(value.grad, key)
                optimizer.step()
                if name == "TimesNet":
                    branches = [f"backbone.model.{i}.conv" for i in range(2)]
                    branches += ["backbone.predict_linear", "backbone.projection", "readout"]
                else:
                    branches = [f"backbone.pdm_blocks.{i}" for i in range(4)]
                    branches += [f"backbone.predict_layers.{i}" for i in range(2)]
                    branches += ["backbone.projection_layer", "readout"]
                now = dict(model.named_parameters())
                for prefix in branches:
                    self.assertTrue(any(not torch.equal(before[key], value) for key, value in now.items()
                                        if key.startswith(prefix)), prefix)
                for key in model.architecture_metadata["frozen_unused_source_parameters"]:
                    self.assertTrue(torch.equal(before["backbone." + key], now["backbone." + key]), key)

    def test_constants_and_shapes_remain_finite_without_future_inputs_or_moments(self):
        for name in published.RECIPE_IDS:
            model = self._model(name).eval()
            for fill in (0., 1., -2.):
                with self.subTest(model=name, fill=fill):
                    result = model(torch.full((1, 7, 5), fill))
                    self.assertEqual(result.shape, (1,))
                    self.assertTrue(bool(torch.isfinite(result).all()))
            self.assertEqual(model.readout.in_features, 5)
            self.assertEqual(model.architecture_metadata["same_window_moments"], 0)
            self.assertFalse(model.architecture_metadata["future_covariates"])
            for values in (torch.ones(1, 8, 5), torch.ones(1, 7, 6), torch.ones(0, 7, 5),
                           torch.ones(1, 7, 5, dtype=torch.int64), torch.full((1, 7, 5), float("nan"))):
                with self.assertRaises(ValueError):
                    model(values)

    def test_time_mixer_uses_both_defined_scales_and_all_fmm_predictors(self):
        model = self._model("TimeMixer").eval()
        values = torch.randn(2, 7, 5)
        scales, unused_marks = model.backbone._Model__multi_scale_process_inputs(values, None)
        self.assertEqual([value.shape[1] for value in scales], [7, 3])
        self.assertIsNone(unused_marks)
        self.assertEqual(len(model.backbone.pdm_blocks), 4)
        self.assertEqual(len(model.backbone.predict_layers), 2)
        called = []
        handles = [layer.register_forward_hook(lambda unused_module, unused_input, output, index=index:
                                               called.append((index, tuple(output.shape))))
                   for index, layer in enumerate(model.backbone.predict_layers)]
        try:
            model(values)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual({index for index, _ in called}, {0, 1})
        self.assertTrue(all(shape[-1] == 1 for _, shape in called))

    def test_all_declared_dependencies_and_license_are_checked_before_import(self):
        for name, changed in (("TimeMixer", "utils/tools.py"), ("TimesNet", "LICENSE")):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as folder:
                copy = Path(folder) / name
                shutil.copytree(self.roots[name], copy)
                (copy / changed).write_bytes((copy / changed).read_bytes() + b"tampered\n")
                with self.assertRaisesRegex(ValueError, "Pinned author source missing or changed"):
                    published.build_periodic(name, 0., {name: str(copy)})
        manifest = json.loads(published.SOURCE_MANIFEST.read_text())
        for name in published.RECIPE_IDS:
            model = self._model(name)
            expected = [row for row in manifest["author_sources"] if row["model"] == name]
            self.assertEqual(len(model.architecture_metadata["source_verification"]["verified_files"]), len(expected))


if __name__ == "__main__":
    unittest.main()
