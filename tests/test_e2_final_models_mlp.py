"""Synthetic source/shape/gradient tests, never Bitcoin fitting or scoring."""
from __future__ import annotations

from pathlib import Path
import copy
import importlib
import os
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_final_models_mlp as models

SOURCES = {
    "TSMixer": Path(os.environ.get("E2_GOOGLE_SOURCE", "/workspace/e2_literature_update_20261105/primary_code/sources/google-research__google-research")),
    "TiDE": Path(os.environ.get("E2_GOOGLE_SOURCE", "/workspace/e2_literature_update_20261105/primary_code/sources/google-research__google-research")),
    "TimeMixer": Path(os.environ.get("E2_TIMEMIXER_SOURCE", "/workspace/e2_paper_guide_20261106/sources/TimeMixer-e24610583b36fdd8c76cc17a8df4e65759a5f460")),
    "TimesNet": Path(os.environ.get("E2_CONTROL_SOURCE", "/workspace/e2_paper_guide_20261106/sources/Time-Series-Library-4e938a1767106324dd753b2a44832bf870a0252e")),
    "PatchTST": Path(os.environ.get("E2_PATCH_SOURCE", "/workspace/e2_completion_20261106/author_sources/PatchTST-204c21efe0b39603ad6e2ca640ef5896646ab1a9")),
}


def make(name):
    return models.build_model(name, candidate={"model": name, "parameters": {"dropout": .1}}, source_roots=SOURCES)


class PurePortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_keras_axis_batchnorm_training_equations_and_singleton(self):
        norm = models.KerasAxisBatchNorm(shape=(2, 2))
        x = torch.tensor([[[1., 2.], [3., 4.]], [[3., 6.], [5., 8.]]])
        actual = norm(x)
        mean, variance = x.mean(0), x.var(0, unbiased=False)
        torch.testing.assert_close(actual, (x-mean)/torch.sqrt(variance+.001))
        torch.testing.assert_close(norm.running_mean, .01*mean)
        torch.testing.assert_close(norm.running_var, .99*torch.ones_like(variance)+.01*variance)
        self.assertTrue(torch.isfinite(norm(x[:1])).all())

    def test_keras_axis_batchnorm_eval_uses_frozen_training_statistics(self):
        norm = models.KerasAxisBatchNorm(shape=(2, 2))
        norm(torch.randn(4, 2, 2)); norm.eval()
        x = torch.randn(1, 2, 2)
        expected = norm(x)
        changed_companions = torch.cat((x, torch.full((7, 2, 2), 1e4)))
        torch.testing.assert_close(norm(changed_companions)[0], expected[0])

    def test_tsmixer_published_structure_not_pilot_dimensions(self):
        model = models.build_model("TSMixer", candidate={"dropout": .2})
        self.assertEqual(len(model.blocks), 6)
        self.assertEqual(model.blocks[0].feature_1.out_features, 512)
        self.assertEqual(model.readout.in_features, 45)
        self.assertEqual(model.architecture_metadata["source_commit"], models.GOOGLE_COMMIT)

    def test_tide_preset_layer_semantics_and_nonconstant_temporal_decoder(self):
        model = models.build_model("TiDE", candidate={"dropout": .1})
        self.assertEqual(model.encoder.linear_a.out_features, 256)
        self.assertEqual(model.feature_encoder.linear_a.out_features, 64)
        self.assertEqual(model.feature_encoder.linear_b.out_features, 4)
        self.assertEqual(model.decoder.linear_b.out_features, 8)
        self.assertEqual(model.temporal_decoder.linear_a.out_features, 128)
        self.assertEqual(tuple(model.temporal_decoder.norm.normalized_shape), (8,))
        x = torch.randn(5, 7, 5)
        output = model(x)
        output.square().mean().backward()
        for module in [model.feature_encoder, model.encoder, model.decoder, model.temporal_decoder, model.residual_projection]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        self.assertGreater(float(output.detach().std()), 0)

    def test_google_ports_keep_keras_initialization_defaults(self):
        for name in ("TSMixer", "TiDE"):
            model = models.build_model(name, candidate={"dropout": 0})
            for key, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and key != "readout":
                    torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
                    bound = (6 / (module.in_features + module.out_features))**.5
                    self.assertLessEqual(float(module.weight.detach().abs().max()), bound+1e-6)
        self.assertLessEqual(float(model.series_embedding.weight.detach().abs().max()), .05)

    def test_port_shape_input_validation_and_no_generic_fallback(self):
        for name in ("TSMixer", "TiDE"):
            model = models.build_model(name, candidate={"dropout": 0})
            for invalid in [torch.zeros(4, 8, 5), torch.zeros(0, 7, 5), torch.zeros(4, 7, 6), torch.full((2, 7, 5), float("nan"))]:
                with self.assertRaises(ValueError):
                    model(invalid)
        with self.assertRaises(ValueError):
            models.build_model("MLP", candidate={"dropout": 0})
        with self.assertRaises(ValueError):
            models.build_model("TSMixer", candidate={"width": 16, "dropout": 0})
        with self.assertRaises(ValueError):
            models.build_model("TiDE", candidate={"dropout": 1.0})
        with self.assertRaises(ValueError):
            models.build_model("TimesNet", candidate={"dropout": 0})

    def test_paper_source_dropout_is_admitted_without_resizing(self):
        model = models.build_model("TSMixer", candidate={"dropout": .9})
        self.assertEqual(len(model.blocks), 6)
        prediction = model(torch.randn(3, 7, 5))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))
        with self.assertRaises(ValueError):
            models.build_model("TiDE", candidate={"dropout": float("nan")})

    def test_pure_ports_batch_invariant_in_eval_with_zero_variance_inputs(self):
        for name in ("TSMixer", "TiDE"):
            model = models.build_model(name, candidate={"dropout": .1}); model.eval()
            x = torch.ones(1, 7, 5)
            one = model(x)
            batch = model(torch.cat((x, torch.randn(3, 7, 5))))[:1]
            torch.testing.assert_close(one, batch, rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.isfinite(one).all())


@unittest.skipUnless(all(SOURCES[n].exists() for n in SOURCES), "Pinned external author sources must be supplied for integration tests")
class PinnedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_five_registered_sources_verify_and_forward_backward(self):
        for i, name in enumerate(models.MODELS):
            with self.subTest(name=name):
                torch.manual_seed(901+i)
                model = make(name)
                x = torch.randn(4, 7, 5, requires_grad=True)
                y = model(x)
                self.assertEqual(tuple(y.shape), (4,))
                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue((y >= 0).all())
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
                self.assertEqual(unused, [])
                self.assertTrue(model.architecture_metadata["source_files_verified_at_build"])

    def test_all_five_eval_predictions_independent_of_batch_companions(self):
        for i, name in enumerate(models.MODELS):
            with self.subTest(name=name):
                torch.manual_seed(601+i)
                model = make(name); model.eval()
                x = torch.randn(1, 7, 5)
                own = model(x)
                together = model(torch.cat((x, torch.randn(5, 7, 5)*100), dim=0))[:1]
                torch.testing.assert_close(own, together, rtol=3e-5, atol=3e-6)

    def test_timemixer_uses_maximum_valid_published_pooling_depth(self):
        model = make("TimeMixer")
        self.assertEqual(len(model.backbone.predict_layers), 3)
        self.assertEqual([v.in_features for v in model.backbone.predict_layers], [7, 3, 1])
        self.assertEqual(len(model.backbone.pdm_blocks), 2)

    def test_timesnet_preserves_learned_extension_full_inception_and_frequency_limit(self):
        model = make("TimesNet")
        self.assertEqual((model.backbone.predict_linear.in_features, model.backbone.predict_linear.out_features), (7, 8))
        self.assertEqual(len(model.backbone.model), 2)
        self.assertEqual(len(model.backbone.model[0].conv[0].kernels), 6)
        self.assertEqual(model.backbone.model[0].conv[0].kernels[0].in_channels, 32)
        self.assertEqual(model.backbone.model[0].frequency_indices, (1, 2, 3, 4))
        self.assertEqual(model.backbone.model[0].periods, (8, 4, 2, 2))

    def test_fused_inception_matches_author_outputs_and_parameter_gradients(self):
        make("TimesNet")  # Sets the pinned source namespace for this import.
        source = importlib.import_module("layers.Conv_Blocks").Inception_Block_V1(3, 4, num_kernels=6).double()
        fused = models._EquivalentInception(copy.deepcopy(source)).double()
        x = torch.randn(2, 3, 3, 8, dtype=torch.float64)
        a, b = source(x), fused(x)
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
        weights = torch.randn_like(a)
        (a*weights).sum().backward(); (b*weights).sum().backward()
        for reference, actual in zip(source.parameters(), fused.parameters()):
            torch.testing.assert_close(reference.grad, actual.grad, rtol=1e-11, atol=1e-11)

    def test_patch_preserves_original_residual_attention_and_published_patch_dimensions(self):
        model = make("PatchTST")
        self.assertEqual(model.backbone.patch_len, 16)
        self.assertEqual(model.backbone.stride, 8)
        self.assertEqual(model.backbone.backbone.patch_num, 2)
        self.assertEqual(len(model.backbone.backbone.encoder.layers), 3)
        self.assertTrue(model.backbone.backbone.encoder.res_attention)
        x = torch.randn(2, 7, 5)
        latent = model.latent_features(x)
        self.assertEqual(tuple(latent.shape), (2, 5, 16, 2))
        torch.testing.assert_close(model.backbone.revin_layer.mean, x.mean(1, keepdim=True))

    def test_multiple_source_models_remain_valid_after_namespace_switches(self):
        x = torch.randn(2, 7, 5)
        built = [make(n).eval() for n in ("TimeMixer", "PatchTST", "TimesNet")]
        for model in built:
            self.assertTrue(torch.isfinite(model(x)).all())


if __name__ == "__main__":
    unittest.main()
