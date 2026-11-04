"""Causal data/preprocessing checks using hermetic development-calendar fixtures."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("dl_data", ROOT / "scripts/dl_data.py")
dl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dl)


def small_data(features, targets, history=2, calendar=None):
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if calendar is None:
        calendar = pd.date_range("2023-01-01", periods=len(features), tz="UTC")
    windows = dl.make_windows(features, history)
    return {
        "calendar": calendar,
        "features": features,
        "targets": targets,
        "windows_by_history": {history: windows},
        "eligible_by_history": {history: np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(targets)},
    }


class DevelopmentFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        folder = cls.root / "results/eda_development"
        folder.mkdir(parents=True)
        count = len(dl.CALENDAR)
        close = 20000 * np.exp(np.cumsum(.001 + .01 * np.sin(np.arange(count) / 5)))
        market = pd.DataFrame({
            "market_day_utc": dl.CALENDAR,
            "open": close * .999,
            "high": close * 1.01,
            "low": close * .99,
            "close": close,
            "volume": 100 + np.arange(count),
            "trades": np.repeat(999999, count),
        })
        hole = int(dl.CALENDAR.get_loc(pd.Timestamp("2023-03-24", tz="UTC")))
        market.loc[hole, list(dl.FEATURES)] = np.nan
        prices = market["close"].to_numpy()
        returns = np.full(count, np.nan)
        returns[1:] = np.diff(np.log(prices))
        # Direct population SD is independent of the loader's rolling algorithm.
        target = np.full(count, np.nan)
        for outcome in range(6, count):
            values = returns[outcome - 6:outcome + 1]
            if np.isfinite(values).all():
                target[outcome] = np.std(values, ddof=0)
        market.to_csv(folder / "daily_market_development.csv", index=False)
        pd.DataFrame({"market_day_utc": dl.CALENDAR, "sigma_7": target}).to_csv(
            folder / "daily_volatility_development.csv", index=False)
        cls.data = dl.load_development(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_loader_excludes_extra_feature_and_preserves_calendar(self):
        self.assertEqual(self.data["features"].shape, (731, 5))
        self.assertTrue(self.data["calendar"].equals(dl.CALENDAR))
        self.assertEqual(set(self.data["source_hashes"]), set(dl.SOURCE_PATHS))
        self.assertTrue(all(len(value) == 64 for value in self.data["source_hashes"].values()))
        self.assertTrue(np.isnan(self.data["persistence"][0]))
        np.testing.assert_allclose(self.data["persistence"][1:], self.data["targets"][:-1], equal_nan=True)

    def test_windows_never_include_outcome_or_bridge_missing_day(self):
        history = 7
        outcome = 60
        np.testing.assert_array_equal(
            self.data["windows_by_history"][history][outcome],
            self.data["features"][outcome - history:outcome])
        hole = int(dl.CALENDAR.get_loc(pd.Timestamp("2023-03-24", tz="UTC")))
        self.assertFalse(self.data["eligible_by_history"][history][hole + 1:hole + history + 1].any())
        self.assertTrue(np.isnan(self.data["windows_by_history"][history][:history]).all())

    def test_outer_and_inner_embargo_and_2024_cohorts(self):
        outer = int(dl.CALENDAR.get_loc(pd.Timestamp("2024-01-01", tz="UTC")))
        inner = int(dl.CALENDAR.get_loc(pd.Timestamp("2023-10-01", tz="UTC")))
        for history, expected in dl.EXPECTED_TRAIN_COUNTS.items():
            with self.subTest(history=history):
                split = dl.split_training(self.data, history)
                train = split["train_indices"]
                self.assertEqual(len(train), expected)
                self.assertEqual(train.max(), outer - history - 1)
                self.assertLessEqual(split["inner_train_indices"].max(), inner - history - 1)
                self.assertEqual(split["inner_validation_indices"].min(), inner)
                self.assertLessEqual(split["inner_validation_indices"].max(), train.max())
                self.assertFalse(np.intersect1d(split["inner_train_indices"], split["inner_validation_indices"]).size)
                self.assertEqual(len(split["selection_indices"]), 182)
                self.assertEqual(len(split["calibration_indices"]), 184)
                self.assertEqual(dl.CALENDAR[split["selection_indices"][-1]].strftime("%Y-%m-%d"), "2024-06-30")
                self.assertEqual(dl.CALENDAR[split["calibration_indices"]][0].strftime("%Y-%m-%d"), "2024-07-01")

    def test_loader_rejects_future_calendar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "results/eda_development"
            folder.mkdir(parents=True)
            frame = pd.read_csv(self.root / dl.SOURCE_PATHS[0])
            frame.loc[len(frame) - 1, "market_day_utc"] = "2025-01-01 00:00:00+00:00"
            frame.to_csv(folder / "daily_market_development.csv", index=False)
            (folder / "daily_volatility_development.csv").write_bytes((self.root / dl.SOURCE_PATHS[1]).read_bytes())
            with self.assertRaisesRegex(ValueError, "2023--2024"):
                dl.load_development(root)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        # Geometrically growing prices expose duplicate weighting of overlapping
        # window days: unique-day log means and window-weighted means differ.
        prices = np.exp(np.array([1., 2., 4., 8., 16., 32.]))
        features = np.column_stack([prices, prices * 1.1, prices * .9, prices, [0., 1., 3., 7., 15., 31.]])
        self.data = small_data(features, [1., 2., 3., 4., 5., 6.])

    def test_fit_uses_unique_input_days_and_no_future_data(self):
        pre = dl.fit_preprocessing(self.data, 2, np.array([2, 3]))
        self.assertEqual(pre["input_day_indices"], [0, 1, 2])
        self.assertAlmostEqual(pre["x_mean"][0], 7 / 3)
        self.assertNotAlmostEqual(pre["x_mean"][0], (1 + 2 + 2 + 4) / 4)
        self.assertAlmostEqual(pre["x_mean"][4], np.log(2))
        self.assertEqual(pre["y_scale"], 3.5)
        changed = small_data(self.data["features"].copy(), self.data["targets"].copy())
        changed["features"][3:] *= 1000
        changed["targets"][4:] = 1000
        changed["windows_by_history"][2] = dl.make_windows(changed["features"], 2)
        same = dl.fit_preprocessing(changed, 2, np.array([2, 3]))
        self.assertEqual(pre, same)
        json.dumps(pre, allow_nan=False)

    def test_transform_shape_dtype_target_inverse_and_training_parameters(self):
        pre = dl.fit_preprocessing(self.data, 2, np.array([2, 3]))
        x, y = dl.transform_windows(self.data, 2, np.array([4, 5]), pre)
        self.assertEqual(x.shape, (2, 2, 5))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)
        self.assertTrue((y > 0).all())
        np.testing.assert_allclose(y * pre["y_scale"], [5, 6], rtol=1e-6)
        self.assertFalse(pre["target_centered"])
        self.assertGreater(float(x[0, 0, 0]), 0)

    def test_zero_variance_and_invalid_outcomes(self):
        features = np.tile([10., 11., 9., 10., 0.], (6, 1))
        data = small_data(features, [1., 2., 3., 4., 5., 6.])
        pre = dl.fit_preprocessing(data, 2, np.array([2, 3]))
        self.assertEqual(pre["x_scale"], [1.] * 5)
        with self.assertRaises(ValueError):
            dl.fit_preprocessing(data, 2, np.array([1]))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            dl.fit_preprocessing(data, 2, np.array([2, 2]))
        data["targets"][:] = 0
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            dl.fit_preprocessing(data, 2, np.array([2, 3]))

    def test_mase_scale_never_compresses_holes_or_excluded_training_days(self):
        data = {"calendar": pd.date_range("2023-01-01", periods=6, tz="UTC"),
                "targets": np.array([1., 100., 4., 8., np.nan, 20.])}
        self.assertEqual(dl.mase_scale(data, np.array([0, 2, 3, 5])), 4.)
        self.assertTrue(np.isnan(dl.mase_scale(data, np.array([0, 2, 5]))))
        data["targets"][:] = 3
        self.assertTrue(np.isnan(dl.mase_scale(data, np.arange(6))))


if __name__ == "__main__":
    unittest.main()
