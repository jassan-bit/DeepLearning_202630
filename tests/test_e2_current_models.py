"""Synthetic-only checks of the provisional catalogue; never market fitting.

Set E2_CURRENT_SOURCE_ROOTS to a JSON mapping of the four pinned author roots,
or E2_CURRENT_AUTHOR_SOURCES to their common directory. Without either, the
four external integration cases explicitly skip; the local seven still run.
"""
from __future__ import annotations

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_current_models as models
import e2_proposed_models as proposed


def source_roots():
    if os.environ.get("E2_CURRENT_SOURCE_ROOTS"):
        return json.loads(Path(os.environ["E2_CURRENT_SOURCE_ROOTS"]).read_text())
    if os.environ.get("E2_CURRENT_AUTHOR_SOURCES"):
        base = Path(os.environ["E2_CURRENT_AUTHOR_SOURCES"])
        return {name: str(base / name) for name in models.EXTERNAL_NAMES}
    return {}


class CatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.roots = source_roots()

    def make(self, name, preset="compact_synthetic"):
        if name in models.EXTERNAL_NAMES and name not in self.roots:
            self.skipTest("Explicit pinned source root not supplied for " + name)
        return models.build_model(name, models.recipe_for(name, preset, self.roots))

    def test_exact_catalogue_and_no_ablation_aliases(self):
        expected = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
                    "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
                    "CNN-1D + LSTM + atención temporal")
        self.assertEqual(models.MODEL_NAMES, expected)
        self.assertEqual(tuple(models.CATALOGUE), expected)
        for name in ["HAR-residual", "FCN1D", "LSTM-AE", "MLP", *models.ABLATION_NAMES]:
            with self.assertRaises(ValueError):
                models.build_model(name)
        self.assertFalse(set(models.MODEL_NAMES) & set(models.ABLATION_NAMES))

    def test_compact_prototypes_are_explicitly_reused(self):
        for name, identifier in models.PROTOTYPE_NAMES.items():
            torch.manual_seed(17)
            current = self.make(name)
            torch.manual_seed(17)
            reference = proposed.build_prototype(identifier, width=8)
            x = torch.randn(2, 7, 5)
            torch.testing.assert_close(current(x), reference(x))
            self.assertEqual(type(current.core), type(reference))
            self.assertFalse(current.architecture_metadata["paper_replication_claimed"])

    def test_recipe_validation_and_defaults_are_explicit(self):
        self.assertEqual(models.recipe_for("FCN-1D")["width"], 16)
        self.assertEqual(models.recipe_for("FCN-1D", "compact_synthetic")["width"], 8)
        for recipe in [{"width": 0}, {"width": True}, {"width": 1.5}, {"dropout": .1},
                       {"preset": "author"}, {"new_parameter": 4}]:
            with self.assertRaises(ValueError):
                models.build_model("FCN-1D", recipe)
        for recipe in [{"dropout": float("nan")}, {"dropout": True}, {"dropout": .31}, {"d_model": 4}]:
            with self.assertRaises(ValueError):
                models.build_model("TSMixer", recipe)
        for invalid in [[], "compact_synthetic"]:
            with self.assertRaises(TypeError):
                models.build_model("FCN-1D", invalid)
        for name in models.EXTERNAL_NAMES:
            with self.assertRaisesRegex(ValueError, "Explicit hash-pinned"):
                models.build_model(name)

    def test_input_and_output_validation(self):
        invalid = [None, [], torch.ones(7, 5), torch.zeros(0, 7, 5), torch.zeros(2, 6, 5),
                   torch.zeros(2, 7, 6), torch.ones(2, 7, 5, dtype=torch.int64),
                   torch.full((2, 7, 5), float("nan")), torch.full((2, 7, 5), float("inf"))]
        for name in ("FCN-1D", models.AE_NAME, "TSMixer", "TiDE", models.HYBRID_NAME):
            model = self.make(name)
            for x in invalid:
                with self.subTest(model=name, shape=getattr(x, "shape", None)):
                    with self.assertRaises(ValueError):
                        model(x)
        model = self.make("FCN-1D")
        x = torch.zeros(2, 7, 5)
        for malformed in [torch.ones(2, 1), torch.ones(1), torch.tensor([1., float("nan")]), torch.tensor([-1., 1.])]:
            with mock.patch.object(model.core, "forward", return_value=malformed):
                with self.assertRaises(RuntimeError):
                    model(x)

    def test_metadata_recipe_independence_and_json_serialization(self):
        recipe = models.recipe_for(models.HYBRID_NAME, "compact_synthetic")
        snapshot = deepcopy(recipe)
        first = models.build_model(models.HYBRID_NAME, recipe)
        recipe["conv_width"] = 999
        self.assertEqual(first.recipe, snapshot)
        first.architecture_metadata["status"] = "changed externally"
        second = models.build_model(models.HYBRID_NAME, snapshot)
        self.assertNotEqual(second.architecture_metadata["status"], "changed externally")
        json.dumps(second.architecture_metadata, allow_nan=False)
        self.assertEqual(second.architecture_metadata["sequence_states"], 7)

    def test_ae_reconstruction_and_supervision_have_separate_gradient_paths(self):
        model = self.make(models.AE_NAME)
        x = torch.randn(4, 7, 5)
        reconstructed = model.reconstruct(x)
        self.assertEqual(reconstructed.shape, x.shape)
        (reconstructed-x).square().mean().backward()
        for name, p in model.named_parameters():
            self.assertEqual(p.grad is None, name.startswith("core.head."), name)
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        model.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        for name, p in model.named_parameters():
            self.assertEqual(p.grad is None, name.startswith(("core.decoder.", "core.reconstruction.")), name)
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_google_presets_and_moment_asymmetry_not_hidden(self):
        mixer, tide = self.make("TSMixer"), self.make("TiDE")
        self.assertEqual(len(mixer.core.blocks), 6)
        self.assertEqual(mixer.core.blocks[0].feature_1.out_features, 512)
        self.assertEqual(tide.core.encoder.linear_a.out_features, 256)
        self.assertEqual(mixer.architecture_metadata["same_window_moment_features"], 10)
        self.assertEqual(self.make("FCN-1D").architecture_metadata["same_window_moment_features"], 0)
        for model in (mixer, tide):
            self.assertFalse(model.architecture_metadata["structural_resize_for_smoke"])

    def test_dependency_manifest_rejects_changes_beyond_primary_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("primary")
            (root / "dependency.py").write_text("tampered")
            digest = models.sha256(b"primary").hexdigest()
            manifest = {"source_locations": {"TimesNet": {"repository": "owner/repo", "commit": "a"*40}},
                        "author_sources": [{"model": "TimesNet", "path": "model.py", "sha256": digest},
                                           {"model": "TimesNet", "path": "dependency.py", "sha256": digest}]}
            with mock.patch.object(models.Path, "read_text", return_value=json.dumps(manifest)):
                with self.assertRaisesRegex(ValueError, "dependency.py"):
                    models._verify_sources("TimesNet", {"TimesNet": root})


def _add_model_contract_test(name):
    def test(self):
        torch.manual_seed(2718)
        model = self.make(name)
        x = torch.randn(4, 7, 5, requires_grad=True)
        model.train()
        y = model(x)
        self.assertEqual(tuple(y.shape), (4,))
        self.assertTrue(torch.isfinite(y).all() and (y >= 0).all())
        loss = y.square().mean()
        if name == models.AE_NAME:
            # Exercise decoder separately; this is not a proposed joint loss.
            model.reconstruct(x).square().mean().backward(retain_graph=True)
        loss.backward()
        for key, p in model.named_parameters():
            if not p.requires_grad:
                # PatchTST registers its fixed attention scale as a Parameter.
                self.assertIsNone(p.grad, key)
                continue
            self.assertIsNotNone(p.grad, key)
            self.assertTrue(torch.isfinite(p.grad).all(), key)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0)
        model.eval()
        with torch.no_grad():
            expected = model(x)
            individual = model(x[:1])
            changed_batch = model(torch.cat((x[:1], torch.full((5, 7, 5), 20.))))[:1]
            torch.testing.assert_close(expected[:1], individual, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(expected[:1], changed_batch, rtol=2e-5, atol=2e-6)
            for constant in (0.0, 2.0):
                self.assertTrue(torch.isfinite(model(torch.full((1, 7, 5), constant))).all())
        # Real torch serialization round-trip, including normalizer buffers.
        checkpoint = io.BytesIO()
        torch.save({"state_dict": model.state_dict(), "recipe": model.recipe}, checkpoint)
        checkpoint.seek(0)
        payload = torch.load(checkpoint, weights_only=True)
        restored = models.build_model(name, payload["recipe"]).eval()
        restored.load_state_dict(payload["state_dict"], strict=True)
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(x), rtol=0, atol=0)
        json.dumps(model.architecture_metadata, allow_nan=False)
        if name in models.EXTERNAL_NAMES:
            self.assertGreater(len(model.architecture_metadata["source_verification"]["verified_files"]), 4)
    return test


for _index, _name in enumerate(models.MODEL_NAMES):
    setattr(CatalogueTests, "test_model_contract_%02d" % _index, _add_model_contract_test(_name))


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(9001)
        self.model = models.build_model(models.HYBRID_NAME, {"preset": "compact_synthetic"}).eval()
        self.x = torch.randn(4, 7, 5)

    def test_all_seven_states_and_normalized_additive_temporal_attention(self):
        core = self.model.core
        steps = core.encode_steps(self.x)
        self.assertEqual(tuple(steps.shape), (4, 7, 8))
        y, weights = core.forward_with_attention(self.x)
        self.assertEqual(tuple(weights.shape), (4, 7))
        self.assertTrue((weights > 0).all())
        torch.testing.assert_close(weights.sum(1), torch.ones(4))
        expected = core.attention_score(torch.tanh(core.attention_projection(steps))).squeeze(-1).softmax(1)
        torch.testing.assert_close(weights, expected)
        context = (steps * weights[:, :, None]).sum(1)
        torch.testing.assert_close(y, torch.nn.functional.softplus(core.head(context)).squeeze(-1))
        self.assertFalse(core.lstm.bidirectional)
        self.assertEqual(core.cnn.conv.padding, (0,))

    def test_prefix_causality_on_already_transformed_tensor(self):
        # Do not recompute any origin-relative transform for shortened raw data.
        full = self.model.core.encode_steps(self.x)
        conv_full = self.model.core.cnn(self.x)
        for prefix in range(1, 7):
            with self.subTest(prefix=prefix):
                short = self.model.core.encode_steps(self.x[:, :prefix])
                torch.testing.assert_close(full[:, :prefix], short, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(conv_full[:, :prefix], self.model.core.cnn(self.x[:, :prefix]))
                changed = self.x.clone()
                changed[:, prefix:] += 100
                torch.testing.assert_close(full[:, :prefix], self.model.core.encode_steps(changed)[:, :prefix])

    def test_states_reset_and_gradients_reach_every_component(self):
        first = self.model(self.x)
        self.model(torch.randn(2, 7, 5) * 100)
        torch.testing.assert_close(first, self.model(self.x), rtol=0, atol=0)
        first.square().mean().backward()
        for key in ("cnn", "lstm", "attention_projection", "attention_score", "head"):
            gradients = [p.grad for p in getattr(self.model.core, key).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients), key)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0, key)

    def test_isolated_ablations_remove_actual_components(self):
        for name in models.ABLATION_NAMES:
            with self.subTest(ablation=name):
                model = models.build_ablation(name, {"preset": "compact_synthetic"}).eval()
                self.assertEqual(hasattr(model.core, "cnn"), name != "LSTM-only")
                self.assertEqual(hasattr(model.core, "lstm"), name != "CNN-only")
                self.assertFalse(hasattr(model.core, "attention_projection"))
                y = model(self.x)
                self.assertEqual(tuple(y.shape), (4,))
                torch.testing.assert_close(y[:1], model(self.x[:1]))
                y.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                full = model.core.encode_steps(self.x)
                torch.testing.assert_close(full[:, :4], model.core.encode_steps(self.x[:, :4]))
                with self.assertRaises(ValueError):
                    model.core.forward_with_attention(self.x)
                self.assertNotIn(name, models.CATALOGUE)


if __name__ == "__main__":
    unittest.main()
