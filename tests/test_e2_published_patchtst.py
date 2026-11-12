"""Author parity and D7 semantics; synthetic tensors only, never market fitting."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e2_published_patchtst import AUTHOR_COMMIT, RECIPE_ID, build_patchtst

ROOTS_FILE = os.environ.get("E2_PUBLISHED_SOURCE_ROOTS")


class PatchTSTConstructionTests(unittest.TestCase):
    def test_invalid_dropout_fails_before_loading_author_sources(self):
        for value in (True, -0.1, 1, float("nan"), float("inf"), "0.1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_patchtst(dropout=value, source_roots={})

    def test_missing_pinned_sources_are_rejected_without_fallback(self):
        with self.assertRaises(ValueError):
            build_patchtst(dropout=0.1, source_roots={})


@unittest.skipUnless(ROOTS_FILE, "Set E2_PUBLISHED_SOURCE_ROOTS to verified source-roots JSON for author integration")
class PinnedPatchTSTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roots = json.loads(Path(ROOTS_FILE).read_text(encoding="utf-8"))
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make(self, dropout=0.0):
        torch.manual_seed(271401)
        return build_patchtst(dropout=dropout, source_roots=self.roots)

    def test_author_encoder_and_original_forecast_head_remain_intact(self):
        model = self.make(dropout=0.3)
        backbone = model.backbone
        self.assertEqual((backbone.patch_len, backbone.stride), (3, 2))
        self.assertEqual(backbone.backbone.patch_num, 4)
        self.assertEqual(len(backbone.backbone.encoder.layers), 3)
        self.assertTrue(backbone.backbone.encoder.res_attention)
        self.assertEqual(backbone.backbone.dropout.p, 0.3)
        self.assertEqual(backbone.head.dropout.p, 0.0)
        for layer in backbone.backbone.encoder.layers:
            self.assertEqual(layer.self_attn.n_heads, 4)
            self.assertEqual(layer.ff[0].in_features, 16)
            self.assertEqual(layer.ff[0].out_features, 128)
        self.assertEqual((backbone.head.linear.in_features, backbone.head.linear.out_features), (64, 1))
        self.assertEqual(model.readout.in_features, 5)
        metadata = model.architecture_metadata
        self.assertEqual(metadata["recipe_id"], RECIPE_ID)
        self.assertEqual(metadata["source_commit"], AUTHOR_COMMIT)
        self.assertEqual(metadata["same_window_moment_features"], 0)
        self.assertFalse(metadata["exact_published_long_context_preset"])
        self.assertEqual(metadata["published_source_preset"]["dropout"], 0.3)
        self.assertEqual(metadata["published_source_preset"]["fc_dropout"], 0.3)
        self.assertEqual(metadata["published_source_preset"]["head_dropout"], 0.0)
        self.assertIn("LICENSE", metadata["source_verification"]["verified_files"])
        self.assertEqual(len(metadata["source_verification"]["verified_files"]), 6)

    def test_four_patches_use_only_seven_observed_days_and_author_end_padding(self):
        model = self.make()
        x = torch.arange(35, dtype=torch.float32).reshape(1, 7, 5)
        patches = model.normalized_patches(x)
        mean, std = x.mean(1, keepdim=True), torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)
        normalized = ((x - mean) / std).transpose(1, 2)
        expected = torch.stack((normalized[:, :, [0, 1, 2]], normalized[:, :, [2, 3, 4]],
                                normalized[:, :, [4, 5, 6]], normalized[:, :, [6, 6, 6]]), dim=2)
        torch.testing.assert_close(patches, expected)
        torch.testing.assert_close(model.backbone.revin_layer.mean, mean)
        torch.testing.assert_close(model.backbone.revin_layer.stdev, std)

    def test_predictions_and_gradients_match_full_author_path_before_denormalization(self):
        model = self.make(dropout=0.3).double().eval()
        original = deepcopy(model.backbone).eval()
        scalar_head = deepcopy(model.readout)
        captured = []
        hook = original.head.register_forward_hook(lambda module, args, output: captured.append(output))
        x = torch.randn(3, 7, 5, dtype=torch.float64, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_(True)
        actual = model(x)
        original(reference_x.transpose(1, 2))
        hook.remove()
        expected = F.softplus(scalar_head(captured[0].squeeze(-1))).squeeze(-1)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        weights = torch.tensor([0.5, -0.25, 1.25], dtype=torch.float64)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-10, atol=1e-11)
        for (name, parameter), (reference_name, reference) in zip(model.backbone.named_parameters(), original.named_parameters()):
            self.assertEqual(name, reference_name)
            if not parameter.requires_grad:
                self.assertIsNone(parameter.grad, name)
                self.assertIsNone(reference.grad, name)
                continue
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-10, atol=1e-11)

    def test_attention_scores_are_passed_between_all_original_encoder_layers(self):
        model = self.make().eval()
        seen = []
        handles = []
        for layer in model.backbone.backbone.encoder.layers:
            def capture(module, args, kwargs):
                seen.append(kwargs["prev"])
            handles.append(layer.self_attn.sdp_attn.register_forward_pre_hook(capture, with_kwargs=True))
        model(torch.randn(2, 7, 5))
        for handle in handles:
            handle.remove()
        self.assertEqual(len(seen), 3)
        self.assertIsNone(seen[0])
        for previous in seen[1:]:
            self.assertIsInstance(previous, torch.Tensor)
            self.assertEqual(tuple(previous.shape), (10, 4, 4, 4))

    def test_scalar_output_never_applies_ohlcv_denormalization(self):
        model = self.make().eval()
        revin = model.backbone.revin_layer
        with patch.object(revin, "forward", wraps=revin.forward) as call:
            result = model(torch.randn(2, 7, 5))
        self.assertEqual([c.args[1] for c in call.call_args_list], ["norm"])
        self.assertEqual(tuple(result.shape), (2,))
        self.assertTrue(bool((result > 0).all()))

    def test_synthetic_optimizer_updates_both_author_body_and_scalar_readout(self):
        model = self.make(dropout=0.1)
        before_body = model.backbone.backbone.W_P.weight.detach().clone()
        before_head = model.readout.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(torch.randn(4, 7, 5)) - torch.tensor([0.1, 0.3, 0.2, 0.4])).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        optimizer.step()
        self.assertFalse(torch.equal(model.backbone.backbone.W_P.weight, before_body))
        self.assertFalse(torch.equal(model.readout.weight, before_head))

    def test_eval_is_batch_companion_independent_and_constants_are_finite(self):
        model = self.make().eval()
        x = torch.randn(1, 7, 5)
        torch.testing.assert_close(model(x), model(torch.cat((x, torch.randn(5, 7, 5) * 100)))[:1],
                                   rtol=1e-5, atol=1e-6)
        self.assertTrue(bool(torch.isfinite(model(torch.ones(2, 7, 5))).all()))
        for invalid in (torch.zeros(0, 7, 5), torch.zeros(1, 8, 5), torch.zeros(1, 7, 6),
                        torch.zeros(1, 7, 5, dtype=torch.int64), torch.full((1, 7, 5), float("nan"))):
            with self.assertRaises(ValueError):
                model(invalid)

    def test_small_fixed_synthetic_batch_can_be_overfit(self):
        model = self.make(dropout=0.0).train()
        x = torch.randn(8, 7, 5)
        targets = torch.linspace(0.08, 0.4, 8)
        with torch.no_grad():
            initial_loss = float((model(x) - targets).square().mean())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        for _ in range(50):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x) - targets).square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final_loss = float((model(x) - targets).square().mean())
        self.assertLess(final_loss, 0.05 * initial_loss)

    def test_any_declared_file_tampering_is_rejected_before_author_import(self):
        model = self.make()
        source_root = Path(self.roots["PatchTST"])
        if (source_root / "PatchTST_supervised").is_dir():
            source_root /= "PatchTST_supervised"
        for relative in ("LICENSE", "layers/PatchTST_layers.py"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory)
                for filename in model.architecture_metadata["source_verification"]["verified_files"]:
                    copied = destination / filename
                    copied.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_root / filename, copied)
                target = destination / relative
                target.write_bytes(target.read_bytes() + b"\nchanged\n")
                with self.assertRaisesRegex(ValueError, "Pinned author source missing or changed"):
                    build_patchtst(dropout=0, source_roots={"PatchTST": str(destination)})


if __name__ == "__main__":
    unittest.main()
