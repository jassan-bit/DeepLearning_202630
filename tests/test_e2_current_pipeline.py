"""Synthetic-only causal preprocessing and optimizer-contract regression tests."""
from __future__ import annotations

import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_current_pipeline as pipe
import e2_current_models as models
import smoke_e2_current_catalogue as cli


class SyntheticFixtureTests(unittest.TestCase):
    def setUp(self):
        self.data = pipe.make_synthetic_data()
        self.fold = pipe.make_synthetic_fold(self.data)

    def test_generator_is_deterministic_daily_and_aligned(self):
        other = pipe.make_synthetic_data()
        for key in ["features", "targets", "windows", "window_day_indices", "window_days_utc_ns"]:
            np.testing.assert_array_equal(self.data[key], other[key])
        self.assertEqual(str(self.data["calendar"].tz), "UTC")
        self.assertEqual(self.data["windows"].shape, (89, 7, 5))
        np.testing.assert_array_equal(self.data["windows"][0], self.data["features"][:7])
        np.testing.assert_array_equal(self.data["windows"][-1], self.data["features"][-8:-1])
        log_returns = pd.Series(np.log(self.data["features"][:, 3])).diff()
        expected = log_returns.rolling(7).std(ddof=0).to_numpy()[7:]
        np.testing.assert_allclose(self.data["targets"][7:], expected, rtol=1e-12, atol=1e-15)
        self.assertNotEqual(pipe.array_sha256(self.data["features"]),
                            pipe.array_sha256(pipe.make_synthetic_data(seed=1)["features"]))

    def test_no_market_loader_called(self):
        with mock.patch("dl_data.load_development", side_effect=AssertionError("market loader called")):
            data = pipe.make_synthetic_data()
            fold = pipe.make_synthetic_fold(data)
            preproc = pipe.fit_preprocessing(data, fold)
            pipe.transform_fold(data, fold, preproc, "validation")

    def test_invalid_calendars_are_rejected_instead_of_compressing_gaps(self):
        calendar = self.data["calendar"]
        invalid = [calendar.tz_localize(None), calendar.tz_convert("Europe/Madrid"),
                   calendar + pd.Timedelta(hours=1), calendar[::-1],
                   calendar[:20].append(calendar[21:]),
                   calendar[:20].append(calendar[19:])]
        for value in invalid:
            with self.subTest(value=str(value[:2])):
                bad = copy.deepcopy(self.data)
                bad["calendar"] = value
                with self.assertRaises(ValueError):
                    pipe.validate_synthetic_data(bad)

    def test_nonfinite_raw_values_are_rejected_even_outside_train(self):
        for key, index in [("features", (90, 0)), ("targets", 90), ("windows", (0, 0, 0))]:
            for value in [np.nan, np.inf, -np.inf]:
                with self.subTest(key=key, value=value):
                    bad = copy.deepcopy(self.data)
                    bad[key][index] = value
                    with self.assertRaises(ValueError):
                        pipe.fit_preprocessing(bad, self.fold)

    def test_window_and_outcome_dates_cannot_be_silently_shifted(self):
        for key in ["outcome_indices", "outcome_days_utc", "window_day_indices", "window_days_utc_ns", "windows"]:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.data)
                if key == "outcome_days_utc":
                    bad[key] += pd.Timedelta(days=1)
                elif key == "windows":
                    bad[key] = np.roll(bad[key], 1, axis=1)
                else:
                    bad[key][0] += 1
                with self.assertRaises(ValueError):
                    pipe.validate_synthetic_data(bad)
        for key in ["outcome_indices", "window_day_indices", "window_days_utc_ns"]:
            bad = copy.deepcopy(self.data)
            bad[key] = bad[key].astype(float)
            with self.assertRaises(ValueError):
                pipe.validate_synthetic_data(bad)

    def test_synthetic_marker_ohlcv_and_target_checks(self):
        changes = [("synthetic_only", False), ("generator", "market"),
                   ("targets", np.full(96, -1.)), ("features", np.zeros((96, 5)))]
        for key, value in changes:
            with self.subTest(key=key):
                bad = copy.deepcopy(self.data)
                bad[key] = value
                with self.assertRaises(ValueError):
                    pipe.validate_synthetic_data(bad)
        bad = copy.deepcopy(self.data)
        bad["features"][0, 1] = bad["features"][0, 2]
        with self.assertRaisesRegex(ValueError, "OHLC"):
            pipe.validate_synthetic_data(bad)

    def test_fold_origin_minus_eight_and_disjoint_day_support(self):
        train, validation = pipe.validate_fold(self.data, self.fold)
        self.assertEqual(train[-1], self.fold["origin_index"] - 8)
        self.assertEqual(validation[0], self.fold["origin_index"])
        self.assertEqual(len(train), 32)
        self.assertEqual(len(validation), 8)
        self.assertLess(train[-1], validation[0] - 7)
        train_input_days = self.data["window_day_indices"][train - 7].ravel()
        validation_input_days = self.data["window_day_indices"][validation - 7].ravel()
        self.assertFalse(np.intersect1d(train_input_days, validation_input_days).size)
        self.assertFalse(np.intersect1d(train, validation_input_days).size)

    def test_fold_overlap_embargo_and_wrong_dates_fail(self):
        for mutation in ["overlap", "embargo", "validation_past", "duplicate", "date", "origin", "timezone"]:
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.fold)
                if mutation == "overlap":
                    bad["validation_indices"][0] = bad["train_indices"][-1]
                elif mutation == "embargo":
                    bad["train_indices"][-1] += 1
                elif mutation == "validation_past":
                    bad["validation_indices"][0] -= 1
                elif mutation == "duplicate":
                    bad["train_indices"][1] = bad["train_indices"][0]
                elif mutation == "date":
                    bad["train_days_utc"] += pd.Timedelta(days=1)
                elif mutation == "origin":
                    bad["origin_day_utc"] += pd.Timedelta(days=1)
                else:
                    bad["origin_day_utc"] = bad["origin_day_utc"].tz_convert("Europe/Madrid")
                with self.assertRaises(ValueError):
                    pipe.validate_fold(self.data, bad)

    def test_out_of_range_and_invalid_integer_inputs_fail(self):
        for args in [{"seed": True}, {"days": 1000}, {"days": 3}, {"seed": -1}]:
            with self.assertRaises(ValueError):
                pipe.make_synthetic_data(**args)
        for args in [{"origin_index": 20}, {"origin_index": 95}, {"train_count": 65},
                     {"validation_count": 9}, {"origin_index": 72.0}]:
            with self.assertRaises(ValueError):
                pipe.make_synthetic_fold(self.data, **args)


class TrainingTransformTests(unittest.TestCase):
    def setUp(self):
        self.data = pipe.make_synthetic_data()
        self.fold = pipe.make_synthetic_fold(self.data)
        self.preproc = pipe.fit_preprocessing(self.data, self.fold)

    def test_scales_equal_only_admitted_train_statistics(self):
        train = self.fold["train_indices"]
        raw = self.data["windows"][train - 7]
        relative = pipe.relative_features(raw)
        pooled = relative.reshape(-1, 5)
        np.testing.assert_allclose(self.preproc["x_mean"], pooled.mean(axis=0), rtol=1e-14)
        np.testing.assert_allclose(self.preproc["x_scale"], pooled.std(axis=0), rtol=1e-14)
        self.assertEqual(self.preproc["y_scale"], float(self.data["targets"][train].mean()))
        self.assertFalse(self.preproc["target_centered"])
        self.assertEqual(self.preproc["x_fit_window_positions"], 32 * 7)
        np.testing.assert_array_equal(relative[:, -1, 3], np.zeros(32))
        x, y = pipe.transform_fold(self.data, self.fold, self.preproc, "train")
        self.assertEqual(x.shape, (32, 7, 5))
        self.assertEqual(x.dtype, np.float32)
        self.assertAlmostEqual(float(y.mean()), 1., places=6)
        np.testing.assert_allclose(y * self.preproc["y_scale"], self.data["targets"][train], rtol=1e-6)

    def test_future_changes_do_not_change_train_scale_or_train_arrays(self):
        bad = copy.deepcopy(self.data)
        # Last train label is 64, and the last train feature day is only 63.
        bad["features"][64:, :4] *= 1000
        bad["features"][64:, 4] += 1e8
        bad["targets"][65:] *= 100
        bad["windows"] = bad["features"][bad["window_day_indices"]].copy()
        after = pipe.fit_preprocessing(bad, self.fold)
        self.assertEqual(self.preproc, after)
        before_arrays = pipe.transform_fold(self.data, self.fold, self.preproc, "train")
        after_arrays = pipe.transform_fold(bad, self.fold, after, "train")
        for before, after_array in zip(before_arrays, after_arrays):
            np.testing.assert_array_equal(before, after_array)

    def test_frozen_transform_rejects_changed_training_provenance(self):
        bad = copy.deepcopy(self.data)
        bad["targets"][self.fold["train_indices"][0]] *= 2
        with self.assertRaisesRegex(ValueError, "provenance"):
            pipe.transform_fold(bad, self.fold, self.preproc, "validation")
        other = pipe.make_synthetic_fold(self.data, train_count=31)
        with self.assertRaisesRegex(ValueError, "provenance"):
            pipe.transform_fold(self.data, other, self.preproc, "validation")

    def test_zero_variance_and_zero_target_are_explicit(self):
        data = copy.deepcopy(self.data)
        data["features"][:] = [100., 100., 100., 100., 0.]
        data["windows"] = data["features"][data["window_day_indices"]].copy()
        preproc = pipe.fit_preprocessing(data, self.fold)
        self.assertEqual(preproc["x_scale"], [1.] * 5)
        data["targets"][self.fold["train_indices"]] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            pipe.fit_preprocessing(data, self.fold)

    def test_transform_rejects_nonfinite_scaling_and_float32_overflow(self):
        for key, value in [("x_scale", [0.] * 5), ("x_mean", [np.nan] * 5),
                           ("y_scale", -1.), ("x_mean", [1e300] * 5), ("target_centered", True)]:
            with self.subTest(key=key, value=value):
                preproc = copy.deepcopy(self.preproc)
                preproc[key] = value
                with self.assertRaises(ValueError):
                    pipe.transform_fold(self.data, self.fold, preproc, "validation")
        with self.assertRaises(ValueError):
            pipe.transform_fold(self.data, self.fold, self.preproc, "test")

    def test_original_units_prediction_and_training_mode_preserved(self):
        class Constant(nn.Module):
            def forward(self, x):
                return torch.full((len(x),), 2., dtype=x.dtype)
        model = Constant()
        model.train()
        x, _ = pipe.transform_fold(self.data, self.fold, self.preproc, "validation")
        predicted = pipe.predict_original_units(model, x, self.preproc)
        np.testing.assert_array_equal(predicted, np.full(8, 2. * self.preproc["y_scale"]))
        self.assertTrue(model.training)
        x[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            pipe.predict_original_units(model, x, self.preproc)

    def test_prediction_rejects_broadcastable_wrong_output_shape(self):
        class WrongShape(nn.Module):
            def forward(self, x):
                return torch.ones(len(x), 1)
        x, _ = pipe.transform_fold(self.data, self.fold, self.preproc, "validation")
        with self.assertRaisesRegex(ValueError, "Tensor\\[N\\]"):
            pipe.predict_original_units(WrongShape(), x, self.preproc)


class SyntheticOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_local_models_execute_finite_steps_without_data_loader(self):
        names = ["FCN-1D", "ResNet-1D", "Inception-1D", models.AE_NAME,
                 "TSMixer", "TiDE", models.HYBRID_NAME]
        with mock.patch("dl_data.load_development", side_effect=AssertionError("market loader called")):
            receipt = pipe.run_synthetic_smoke(model_names=names, config=pipe.SmokeConfig(supervised_steps=1))
        self.assertEqual(receipt["executed_model_count"], 7)
        self.assertEqual(receipt["skipped_model_count"], 0)
        self.assertFalse(receipt["market_data_read"])
        self.assertFalse(receipt["generalization_or_superiority_demonstrated"])
        for record in receipt["models"]:
            self.assertTrue(record["all_losses_and_predictions_finite"])
            self.assertNotEqual(record["initial_state_sha256"], record["final_state_sha256"])
            self.assertEqual(record["supervised_steps"], 1)
            self.assertEqual(record["supervised_gradient_audit"]["unexpected_missing"], [])
        ae = next(record for record in receipt["models"] if record["model"] == models.AE_NAME)
        self.assertEqual(ae["pretrain_steps"], 1)
        self.assertNotEqual(ae["initial_state_sha256"], ae["post_pretrain_state_sha256"])
        self.assertNotEqual(ae["post_pretrain_state_sha256"], ae["final_state_sha256"])
        self.assertTrue(all(name.startswith("core.head.") for name in
                            ae["pretrain_gradient_audit"]["expected_unused_parameter_names"]))
        self.assertTrue(all(name.startswith(("core.decoder.", "core.reconstruction.")) for name in
                            ae["supervised_gradient_audit"]["expected_unused_parameter_names"]))
        self.assertFalse(ae["ae_protocol"]["validation_used_for_optimization_or_selection"])
        json.dumps(receipt, allow_nan=False)

    def test_training_and_reconstruction_never_observe_validation(self):
        data = pipe.make_synthetic_data()
        fold = pipe.make_synthetic_fold(data)
        preproc = pipe.fit_preprocessing(data, fold)
        config = pipe.SmokeConfig(supervised_steps=1)
        trained, before = pipe.train_synthetic_model(models.AE_NAME, data, fold, preproc, config)
        changed = copy.deepcopy(data)
        changed["features"][65:, :4] *= 100
        changed["features"][65:, 4] += 1e5
        changed["targets"][65:] *= 100
        changed["windows"] = changed["features"][changed["window_day_indices"]].copy()
        retrained, after = pipe.train_synthetic_model(models.AE_NAME, changed, fold, preproc, config)
        self.assertEqual(before["final_state_sha256"], after["final_state_sha256"])
        self.assertEqual(before["pretrain_reconstruction_mse"], after["pretrain_reconstruction_mse"])
        self.assertEqual(before["supervised_scaled_target_mse"], after["supervised_scaled_target_mse"])
        for (name_a, value_a), (name_b, value_b) in zip(trained.state_dict().items(), retrained.state_dict().items()):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(value_a, value_b, rtol=0, atol=0)

    def test_unused_non_autoencoder_parameters_are_rejected(self):
        class Disconnected(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(35, 1)
                self.unused = nn.Linear(3, 3)
            def forward(self, x):
                return self.head(x.flatten(1)).squeeze(-1)
        with self.assertRaisesRegex(ValueError, "unused"):
            pipe._optimize(Disconnected(), torch.ones(8, 7, 5), torch.ones(8),
                           steps=1, config=pipe.SmokeConfig(), reconstruct=False)

    def test_caps_unknown_models_and_recipe_resizing_fail(self):
        for kwargs in [{"supervised_steps": 4}, {"pretrain_steps": 0}, {"days": 129},
                       {"batch_size": 1}, {"batch_size": 2}, {"learning_rate": 1.},
                       {"learning_rate": np.nan}, {"train_count": 65}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pipe.SmokeConfig(**kwargs).validate()
        for names in [[], ["fake"], ["FCN-1D", "FCN-1D"]]:
            with self.assertRaises(ValueError):
                pipe.run_synthetic_smoke(model_names=names)
        with self.assertRaisesRegex(ValueError, "dimensions are fixed"):
            pipe.run_synthetic_smoke(model_names=["FCN-1D"], recipe={"width": 100000})
        with self.assertRaisesRegex(ValueError, "compact preset"):
            pipe.run_synthetic_smoke(model_names=["FCN-1D"], recipe={"preset": "provisional_default"})

    def test_hash_includes_shape_and_bytes(self):
        array = np.arange(6, dtype=np.int64)
        self.assertNotEqual(pipe.array_sha256(array), pipe.array_sha256(array.reshape(2, 3)))
        self.assertNotEqual(pipe.array_sha256(array), pipe.array_sha256(array + 1))
        with self.assertRaises(ValueError):
            pipe.array_sha256(np.array([object()]))


class CommandLineTests(unittest.TestCase):
    def test_cli_has_no_data_file_argument_and_caps_steps(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            for args in [["--data", "market.csv"], ["--market-file", "market.csv"], ["--steps", "100"]]:
                with self.assertRaises(SystemExit) as error:
                    cli.main(args)
                self.assertEqual(error.exception.code, 2)

    def test_cli_does_not_overwrite_or_write_historical_locations(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic-existing.json"
            path.write_text("keep")
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                for target in [path, Path(temporary) / "receipt.json",
                               ROOT / "results/synthetic-receipt.json"]:
                    with self.assertRaises(SystemExit):
                        cli.main(["--output", str(target)])
            self.assertEqual(path.read_text(), "keep")

    def test_cli_writes_new_synthetic_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "synthetic-receipt.json"
            fake = {"synthetic_only": True, "models": []}
            with mock.patch.object(cli, "run_synthetic_smoke", return_value=fake) as run:
                self.assertEqual(cli.main(["--model", "FCN-1D", "--steps", "1", "--output", str(path)]), 0)
            self.assertEqual(json.loads(path.read_text()), fake)
            self.assertEqual(run.call_args.kwargs["model_names"], ["FCN-1D"])
            self.assertEqual(run.call_args.kwargs["config"].supervised_steps, 1)


if __name__ == "__main__":
    unittest.main()
