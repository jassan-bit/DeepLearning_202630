"""Checks for causal relative prices and expanding-origin purges; no fitting."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dl_data as v1
import dl_data_v2 as v2


def fixture(features, targets, history=2, calendar=None):
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if calendar is None:
        calendar = pd.date_range("2023-01-01", periods=len(features), tz="UTC")
    windows = v1.make_windows(features, history)
    return {
        "calendar": calendar,
        "features": features,
        "targets": targets,
        "windows_by_history": {history: windows},
        "eligible_by_history": {history: np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(targets)},
    }


class RelativeFeatureTests(unittest.TestCase):
    def setUp(self):
        close = np.array([10., 12., 9., 15., 18., 21.])
        self.features = np.column_stack([close * .98, close * 1.1, close * .9, close,
                                         [0., 1., 3., 7., 15., 31.]])
        self.data = fixture(self.features, [1., 2., 3., 4., 5., 6.])

    def test_prices_are_invariant_to_global_price_units(self):
        raw = self.data["windows_by_history"][2][[2, 3, 4]]
        changed = raw.copy()
        changed[..., :4] *= 1e6
        original = v2.relative_features(raw)
        np.testing.assert_allclose(v2.relative_features(changed), original, rtol=1e-12, atol=1e-14)
        np.testing.assert_array_equal(original[..., 4], np.log1p(raw[..., 4]))

    def test_anchor_is_each_windows_last_observed_close(self):
        raw = self.data["windows_by_history"][2][[2, 4]]
        original = raw.copy()
        result = v2.relative_features(raw)
        np.testing.assert_array_equal(result[:, -1, 3], [0., 0.])
        self.assertAlmostEqual(result[0, 0, 3], np.log(10 / 12))
        self.assertAlmostEqual(result[1, 0, 3], np.log(9 / 15))
        self.assertAlmostEqual(result[0, -1, 1], np.log(1.1))
        np.testing.assert_array_equal(raw, original)

    def test_training_scalers_ignore_unseen_rows_and_future_labels(self):
        train = np.array([2, 3])
        before = v2.fit_relative_preprocessing(self.data, 2, train)
        changed = copy.deepcopy(self.data)
        changed["features"][3:, :4] *= 1000
        changed["features"][3:, 4] += 1e9
        changed["targets"][4:] = 1000
        changed["windows_by_history"][2] = v1.make_windows(changed["features"], 2)
        after = v2.fit_relative_preprocessing(changed, 2, train)
        self.assertEqual(before, after)
        self.assertEqual(before["fit_window_positions"], 4)
        self.assertEqual(before["input_day_indices"], [0, 1, 2])
        self.assertEqual(before["last_anchor_day_utc"], "2023-01-03")
        self.assertEqual(before["y_scale"], 3.5)
        json.dumps(before, allow_nan=False)

    def test_transform_keeps_alignment_frozen_scales_and_original_y_units(self):
        pre = v2.fit_relative_preprocessing(self.data, 2, np.array([2, 3]))
        snapshot = copy.deepcopy(pre)
        x, y = v2.transform_relative_windows(self.data, 2, np.array([4, 5]), pre)
        self.assertEqual(x.shape, (2, 2, 5))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)
        np.testing.assert_allclose(y * pre["y_scale"], [5., 6.], rtol=1e-6)
        self.assertEqual(pre, snapshot)
        self.assertTrue(np.isfinite(x).all())
        # No outcome-day price can influence its own input window.
        changed = copy.deepcopy(self.data)
        changed["features"][4:] *= 100
        changed["windows_by_history"][2] = v1.make_windows(changed["features"], 2)
        same, _ = v2.transform_relative_windows(changed, 2, np.array([4]), pre)
        np.testing.assert_array_equal(same[0], x[0])

    def test_zero_variance_and_invalid_inputs_are_explicit(self):
        data = fixture(np.tile([10., 11., 9., 10., 0.], (6, 1)), [1., 2., 3., 4., 5., 6.])
        pre = v2.fit_relative_preprocessing(data, 2, np.array([2, 3]))
        self.assertEqual(pre["zero_variance_feature_indices"], [0, 1, 2, 3, 4])
        self.assertEqual(pre["x_scale"], [1.] * 5)
        for channel, bad in [(0, 0.), (3, -1.), (4, -1.), (1, np.nan)]:
            with self.subTest(channel=channel, bad=bad):
                raw = np.ones((1, 2, 5)); raw[0, 0, channel] = bad
                with self.assertRaises(ValueError):
                    v2.relative_features(raw)
        for bad in [np.array([2, 2]), np.array([1]), np.array([True]), np.array([2.])]:
            with self.subTest(indices=bad):
                with self.assertRaises(ValueError):
                    v2.fit_relative_preprocessing(data, 2, bad)
        wrong = copy.deepcopy(pre); wrong["history_days"] = 7
        with self.assertRaisesRegex(ValueError, "history"):
            v2.transform_relative_windows(data, 2, np.array([4]), wrong)
        wrong = copy.deepcopy(pre); wrong["x_scale"][0] = 0
        with self.assertRaisesRegex(ValueError, "parameters"):
            v2.transform_relative_windows(data, 2, np.array([4]), wrong)


class ExpandingOriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        n = len(v1.CALENDAR)
        close = 20000 * np.exp(np.cumsum(.001 + .01 * np.sin(np.arange(n) / 5)))
        features = np.column_stack([close * .999, close * 1.01, close * .99, close,
                                    100 + np.arange(n)])
        cls.hole = int(v1.CALENDAR.get_loc(pd.Timestamp("2023-03-24", tz="UTC")))
        features[cls.hole] = np.nan
        targets = pd.Series(np.log(features[:, 3])).diff().rolling(7, min_periods=7).std(ddof=0).to_numpy()
        cls.data = {"calendar": v1.CALENDAR, "features": features, "targets": targets,
                    "windows_by_history": {}, "eligible_by_history": {}}
        for d in v1.HISTORIES:
            windows = v1.make_windows(features, d)
            cls.data["windows_by_history"][d] = windows
            cls.data["eligible_by_history"][d] = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(targets)

    def test_each_origin_purges_inner_and_outer_cuts_before_fit(self):
        for date in ["2024-01-01", "2024-07-01", "2024-10-01"]:
            start = pd.Timestamp(date, tz="UTC")
            origin_index = int(v1.CALENDAR.get_loc(start))
            inner_index = int(v1.CALENDAR.get_loc(start - pd.Timedelta(days=90)))
            for d in v1.HISTORIES:
                with self.subTest(origin=date, history=d):
                    split = v2.split_for_origin(self.data, d, date)
                    train, inner, validation = [split[k] for k in
                                                ["train_indices", "inner_train_indices", "inner_validation_indices"]]
                    self.assertEqual(train.max(), origin_index - d - 1)
                    self.assertEqual(inner.max(), inner_index - d - 1)
                    self.assertEqual(validation.min(), inner_index)
                    self.assertEqual(validation.max(), train.max())
                    self.assertFalse(np.intersect1d(inner, validation).size)
                    self.assertTrue(np.isin(inner, train).all())
                    self.assertTrue(np.isin(validation, train).all())
                    pre = v2.fit_relative_preprocessing(self.data, d, train)
                    self.assertLess(pd.Timestamp(pre["last_input_day_utc"], tz="UTC"), start)
                    self.assertEqual(pre["fit_outcome_indices_sha256"], split["split_metadata"]["train_indices_sha256"])
                    json.dumps(split["split_metadata"], allow_nan=False)

    def test_missing_day_stays_in_calendar_without_compressed_windows(self):
        d = 7
        split = v2.split_for_origin(self.data, d, "2024-07-01")
        prohibited = np.arange(self.hole, self.hole + d + 1)
        self.assertFalse(np.intersect1d(split["train_indices"], prohibited).size)
        with self.assertRaisesRegex(ValueError, "Ineligible"):
            v2.fit_relative_preprocessing(self.data, d, np.array([self.hole + 2]))
        self.assertEqual(len(self.data["calendar"]), 731)
        self.assertEqual(self.data["calendar"][self.hole + 1], pd.Timestamp("2023-03-25", tz="UTC"))

    def test_future_calendars_origins_and_ill_defined_inner_periods_are_rejected(self):
        for origin in ["2025-01-01", "2023-12-01", "2024-07-01T05:00:00Z"]:
            with self.subTest(origin=origin):
                with self.assertRaises(ValueError):
                    v2.split_for_origin(self.data, 7, origin)
        with self.assertRaises(ValueError):
            v2.split_for_origin(self.data, 7, "2024-07-01", inner_days=7)
        broken = copy.deepcopy(self.data)
        broken["calendar"] = pd.date_range("2024-01-01", periods=731, tz="UTC")
        with self.assertRaisesRegex(ValueError, "2023--2024"):
            v2.split_for_origin(broken, 7, "2024-07-01")


if __name__ == "__main__":
    unittest.main()
