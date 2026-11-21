"""Software validation using generated fixtures only: no optimizer steps."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from seoi_v5 import data
from seoi_v5.model import RECIPES, build_model, discrete_crps

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


def fixture(n=12):
    rng = np.random.default_rng(716)
    close = 100 * np.exp(np.cumsum(rng.normal(.0003, .012, (n, 7)), axis=1))
    opening = close * np.exp(rng.normal(0, .003, (n, 7)))
    high = np.maximum(opening, close) * np.exp(rng.uniform(.001, .007, (n, 7)))
    low = np.minimum(opening, close) * np.exp(-rng.uniform(.001, .007, (n, 7)))
    volume = rng.uniform(1e3, 1e5, (n, 7))
    raw = np.stack([opening, high, low, close, volume], -1)
    A, _, _, _ = data.known_geometry(raw)
    y = np.sqrt(A + data.C * rng.uniform(.001, .04, n)**2)
    return raw, y


def tensors(raw, processor, recipe):
    X, A, scale, _ = data.transform(raw, processor, recipe)
    return torch.from_numpy(X), torch.from_numpy(A), torch.from_numpy(scale)


class ModelDataTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.y = fixture()
        self.processor = data.fit_preprocessor(self.raw)

    def model(self, recipe="seoi_mse"):
        return build_model(recipe, 17, .02, .012).eval()

    def test_capacity_and_seeded_backbone_shared_across_recipes(self):
        ref = self.model()
        for recipe in RECIPES:
            other = self.model(recipe)
            self.assertEqual(sum(p.numel() for p in other.parameters()), 1210 if recipe == "additive_mse" else 1262)
            for key, value in ref.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, other.encoder.state_dict()[key]))
            self.assertTrue(torch.equal(ref.query_raw, other.query_raw))

    def test_build_does_not_change_rng(self):
        before = torch.random.get_rng_state().clone()
        self.model()
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_support_and_exact_finite_law_expectation(self):
        for recipe in RECIPES:
            model = self.model(recipe)
            X, A, scale = tensors(self.raw, self.processor, recipe)
            output = model.forward_components(X, A, scale)
            self.assertTrue((output["prediction"] >= A.sqrt()).all())
            if recipe != "additive_mse":
                self.assertTrue((output["atoms"][:, 1:] > output["atoms"][:, :-1]).all())
                expected = (output["probabilities"] * (A[:, None] + data.C * output["atoms"].square()).sqrt()).sum(1)
                torch.testing.assert_close(output["prediction"], expected, rtol=1e-13, atol=1e-15)

    def test_local_log_affine_equivariance_with_fixed_fitted_references(self):
        transformed = self.raw.copy()
        k, shift = 1.7, .4
        transformed[..., :4] = np.exp(shift + k * np.log(self.raw[..., :4]))
        model = self.model()
        old = model.forward_components(*tensors(self.raw, self.processor, "seoi_mse"))
        new = model.forward_components(*tensors(transformed, self.processor, "seoi_mse"))
        # FP32 encoder, analytic readout FP64: tolerance acknowledges preprocessing roundoff.
        torch.testing.assert_close(new["prediction"], k * old["prediction"], rtol=3e-7, atol=1e-12)
        torch.testing.assert_close(new["atoms"], k * old["atoms"], rtol=3e-7, atol=1e-12)
        torch.testing.assert_close(new["probabilities"], old["probabilities"], rtol=3e-7, atol=1e-10)

    def test_price_level_invariance(self):
        shifted = self.raw.copy()
        shifted[..., :4] *= 31
        model = self.model()
        old = model(*tensors(self.raw, self.processor, "seoi_mse"))
        new = model(*tensors(shifted, self.processor, "seoi_mse"))
        torch.testing.assert_close(old, new, rtol=3e-7, atol=1e-12)

    def test_flat_window_is_finite_and_local_equivariance_not_claimed(self):
        raw = self.raw[:2].copy()
        raw[..., :4] = 100
        _, _, scale, binding = data.transform(raw, self.processor, "seoi_mse")
        self.assertTrue(binding.all())
        self.assertTrue(np.all(scale == data.EPSILON))
        out = self.model()(*tensors(raw, self.processor, "seoi_mse"))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out > 0).all())

    def test_joint_scale_flat_case_consistency(self):
        raw = self.raw[:2].copy()
        raw[..., :4] = 100
        k = 1.4
        scaled_fit = self.raw.copy()
        scaled_fit[..., :4] = np.exp(k * np.log(scaled_fit[..., :4]))
        new_processor = data.fit_preprocessor(scaled_fit)
        scaled_raw = raw.copy()
        scaled_raw[..., :4] = np.exp(k * np.log(scaled_raw[..., :4]))
        old = self.model()
        new = build_model("seoi_mse", 17, .02 * k, .012 * k).eval()
        a = old(*tensors(raw, self.processor, "seoi_mse"))
        b = new(*tensors(scaled_raw, new_processor, "seoi_mse"))
        torch.testing.assert_close(b, k * a, rtol=3e-7, atol=1e-12)

    def test_factorial_controls_remove_one_transform(self):
        common = deepcopy(self.model().state_dict())
        outs = {}
        for recipe in RECIPES[:4]:
            model = self.model(recipe)
            model.load_state_dict(common)
            outs[recipe] = model(*tensors(self.raw, self.processor, recipe))
        self.assertFalse(torch.allclose(outs["seoi_mse"], outs["input_only_mse"], rtol=1e-5, atol=1e-10))
        self.assertFalse(torch.allclose(outs["seoi_mse"], outs["output_only_mse"], rtol=1e-5, atol=1e-10))
        X1, *_ = tensors(self.raw, self.processor, "seoi_mse")
        X2, *_ = tensors(self.raw, self.processor, "input_only_mse")
        self.assertTrue(torch.equal(X1, X2))

    def test_no_label_or_later_window_enters_prediction(self):
        before = self.model()(*tensors(self.raw, self.processor, "seoi_mse"))
        changed = self.raw.copy()
        changed[-1, ..., :4] *= 2
        after = self.model()(*tensors(changed, self.processor, "seoi_mse"))
        self.assertTrue(torch.equal(before[:-1], after[:-1]))
        # No Y or U is accepted by forward_components or transform.
        self.assertNotIn("y", self.model().forward_components.__code__.co_varnames)

    def test_normalized_mse_same_raw_metric_and_finite_gradients(self):
        for recipe in RECIPES:
            model = self.model(recipe)
            X, A, scale = tensors(self.raw, self.processor, recipe)
            out = model.forward_components(X, A, scale)
            Y = torch.from_numpy(self.y)
            U = ((Y.square() - A).clamp_min(0) / data.C).sqrt()
            loss = model.loss_components(out, Y, U)
            torch.testing.assert_close(loss["mse_scaled"], ((out["prediction"] - Y) / .012).square().mean())
            loss["loss"].backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_state_serialization_round_trip(self):
        old, new = self.model(), self.model()
        new.load_state_dict(deepcopy(old.state_dict()))
        self.assertTrue(torch.equal(old(*tensors(self.raw, self.processor, "seoi_mse")), new(*tensors(self.raw, self.processor, "seoi_mse"))))

    def test_crps_point_mass_is_absolute_error(self):
        atoms = torch.tensor([[.1], [.3]], dtype=torch.float64)
        observed = torch.tensor([.2, .15], dtype=torch.float64)
        torch.testing.assert_close(discrete_crps(atoms, torch.ones_like(atoms), observed), (atoms[:, 0] - observed).abs())

    def test_invalid_input_or_geometry_is_rejected(self):
        bad = self.raw.copy()
        bad[0, 0, 1] = bad[0, 0, 2] / 2
        with self.assertRaises(ValueError):
            data.transform(bad, self.processor, "seoi_mse")
        X, A, scale = tensors(self.raw, self.processor, "seoi_mse")
        with self.assertRaises(ValueError):
            self.model()(X.to(torch.float64), A, scale)
        with self.assertRaises(ValueError):
            self.model()(X, -torch.ones_like(A), scale)

    def test_corrupt_npz_rejected_before_array_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "bad.npz"
            p.write_bytes(b"incorrect")
            with patch("numpy.load", side_effect=AssertionError("should not load")):
                with self.assertRaises(ValueError):
                    data.load_data(p)

    def test_only_training_arrays_loaded_and_fit_statistics_independent_of_later_labels(self):
        dates = np.arange("2023-01-16", "2024-11-01", dtype="datetime64[D]").astype(str)
        raw, y = fixture(len(dates))
        seen = []
        class Archive:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __getitem__(self, key):
                if key not in data.ALLOWED_KEYS: raise AssertionError("Forbidden key: " + key)
                seen.append(key)
                return {"training_raw": raw, "training_y_original": y, "training_dates": dates}[key]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "fixture.npz"
            p.write_bytes(b"\0" * data.NPZ_BYTES)
            with patch.object(data, "sha256", return_value=data.NPZ_SHA256), patch("numpy.load", return_value=Archive()):
                result = data.load_data(p)
        self.assertEqual(seen, list(data.ALLOWED_KEYS))
        self.assertEqual([len(result[k]["y"]) for k in ("train", "selection", "assessment")], [532, 54, 55])
        changed_y = y.copy()
        changed_y[dates >= "2024-07-08"] *= 3
        changed = data.prepare_arrays(raw, changed_y, dates)
        self.assertEqual(result["preprocessor"], changed["preprocessor"])
        self.assertEqual(result["scalers"], changed["scalers"])


if __name__ == "__main__":
    unittest.main()
