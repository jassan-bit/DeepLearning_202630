"""Synthetic contract checks; these fixtures do not certify market-data validity."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_current_data_closure as audit


class DailyFrameGuardTests(unittest.TestCase):
    def frame(self):
        dates = pd.date_range("2024-01-01", periods=5, tz="UTC")
        return pd.DataFrame(
            {"open": [10.] * 5, "high": [12.] * 5, "low": [9.] * 5,
             "close": [11.] * 5, "volume": [100.] * 5}, index=dates)

    def test_valid_calendar_is_preserved_and_missing_rows_are_not_compressed(self):
        frame = self.frame()
        result = audit.validate_daily_frame(frame, "2024-01-01", "2024-01-05")
        pd.testing.assert_frame_equal(result, frame)
        with self.assertRaises(ValueError):
            audit.validate_daily_frame(frame.drop(frame.index[2]), "2024-01-01", "2024-01-05")

    def test_only_an_explicitly_allowed_whole_day_can_be_missing(self):
        frame = self.frame()
        frame.loc[frame.index[2], :] = np.nan
        with self.assertRaises(ValueError):
            audit.validate_daily_frame(frame, "2024-01-01", "2024-01-05")
        result = audit.validate_daily_frame(
            frame, "2024-01-01", "2024-01-05", allowed_missing=("2024-01-03",))
        pd.testing.assert_frame_equal(result, frame)
        self.assertTrue(result.loc[frame.index[2]].isna().all())
        partial = self.frame()
        partial.loc[partial.index[2], "close"] = np.nan
        with self.assertRaises(ValueError):
            audit.validate_daily_frame(
                partial, "2024-01-01", "2024-01-05", allowed_missing=("2024-01-03",))

    def test_naive_or_duplicate_dates_do_not_satisfy_the_utc_calendar(self):
        naive = self.frame()
        naive.index = naive.index.tz_localize(None)
        with self.assertRaises(ValueError):
            audit.validate_daily_frame(naive, "2024-01-01", "2024-01-05")
        duplicated = self.frame()
        duplicated.index = duplicated.index[:2].append(duplicated.index[1:4])
        with self.assertRaises(ValueError):
            audit.validate_daily_frame(duplicated, "2024-01-01", "2024-01-05")

    def test_nonfinite_values_and_invalid_ohlcv_geometry_are_rejected(self):
        for field, value in (("high", 10.), ("low", 11.), ("close", 0.),
                             ("volume", -1.), ("open", np.inf)):
            with self.subTest(field=field, value=value):
                frame = self.frame()
                frame.loc[frame.index[1], field] = value
                with self.assertRaises(ValueError):
                    audit.validate_daily_frame(frame, "2024-01-01", "2024-01-05")


class CausalSupportTests(unittest.TestCase):
    def test_calibration_test_history_can_overlap_with_disjoint_outcome_ids(self):
        result = audit.support_intersections(["2024-12-30"], ["2025-01-01"])
        self.assertEqual(result["outcome_ID_intersection"], [])
        self.assertEqual(result["target_close_support_intersection"],
                         ["2024-12-25", "2024-12-26", "2024-12-27",
                          "2024-12-28", "2024-12-29", "2024-12-30"])
        self.assertEqual(result["D7_input_support_intersection"],
                         ["2024-12-25", "2024-12-26", "2024-12-27",
                          "2024-12-28", "2024-12-29"])
        self.assertEqual(result["target_return_support_intersection"],
                         ["2024-12-26", "2024-12-27", "2024-12-28",
                          "2024-12-29", "2024-12-30"])

    def test_training_calibration_embargo_separates_all_declared_supports(self):
        result = audit.support_intersections(["2024-10-31"], ["2024-11-08"])
        for field in ("outcome_ID_intersection", "target_close_support_intersection",
                      "D7_input_support_intersection", "target_return_support_intersection"):
            with self.subTest(field=field):
                self.assertEqual(result[field], [])

    def test_repeated_outcome_id_is_reported_separately_from_history_overlap(self):
        result = audit.support_intersections(["2024-12-30"], ["2024-12-30"])
        self.assertEqual(result["outcome_ID_intersection"], ["2024-12-30"])
        self.assertEqual(len(result["target_close_support_intersection"]), 8)
        self.assertEqual(len(result["D7_input_support_intersection"]), 7)
        self.assertEqual(len(result["target_return_support_intersection"]), 7)


class NeighborDiagnosticTests(unittest.TestCase):
    def test_distance_is_rms_of_35_standardized_coordinates(self):
        result = audit.nearest_rows(
            np.ones((1, 7, 5)), np.zeros((1, 7, 5)),
            ["2024-01-10"], ["2024-01-01"], "synthetic_cross_cohort")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["rms_standardized_distance"], 1., places=12)
        self.assertFalse(result[0]["below_fixed_threshold"])
        self.assertEqual(result[0]["query_outcome_day_utc"], "2024-01-10")
        self.assertEqual(result[0]["nearest_reference_outcome_day_utc"], "2024-01-01")

    def test_fixed_near_threshold_flags_similarity_without_changing_ids(self):
        query = np.stack((np.full((7, 5), .005), np.full((7, 5), .02)))
        result = audit.nearest_rows(
            query, np.zeros((1, 7, 5)), ["2024-01-10", "2024-01-11"],
            ["2024-01-01"], "synthetic_threshold")
        self.assertAlmostEqual(result[0]["rms_standardized_distance"], .005, places=12)
        self.assertAlmostEqual(result[1]["rms_standardized_distance"], .02, places=12)
        self.assertEqual([r["below_fixed_threshold"] for r in result], [True, False])
        self.assertTrue(all(not r["same_outcome_ID"] for r in result))

    def test_self_is_excluded_but_equal_content_with_distinct_ids_is_retained(self):
        windows = np.zeros((2, 7, 5))
        dates = ["2024-01-10", "2024-01-11"]
        result = audit.nearest_rows(windows, windows, dates, dates,
                                    "synthetic_within_cohort", within=True)
        self.assertEqual([r["nearest_reference_outcome_day_utc"] for r in result], dates[::-1])
        for row in result:
            self.assertEqual(row["rms_standardized_distance"], 0.)
            self.assertFalse(row["same_outcome_ID"])
            self.assertEqual(row["input_calendar_days_shared_with_nearest"], 6)
            self.assertEqual(row["target_closes_shared_with_nearest"], 7)

    def test_missing_distinct_reference_or_wrong_input_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.nearest_rows(np.zeros((1, 7, 5)), np.zeros((1, 7, 5)),
                               ["2024-01-10"], ["2024-01-10"], "self_only", within=True)
        with self.assertRaises(ValueError):
            audit.nearest_rows(np.zeros((1, 34)), np.zeros((1, 34)),
                               ["2024-01-10"], ["2024-01-01"], "wrong_geometry")
        with self.assertRaises(ValueError):
            audit.nearest_rows(np.zeros((1, 7, 5)), np.zeros((1, 7, 5)),
                               [], ["2024-01-01"], "missing_id")


class ScalerRecordGuardTests(unittest.TestCase):
    def record(self):
        return {"mean": [0., 1., 2., 3., 4.], "std": [1., 1., 1., 1., 1.],
                "target_scale": .02, "train_dates": ["2024-01-01", "2024-01-02"],
                "training_sha256": "a" * 64}

    def test_numerical_rounding_tolerance_is_absolute_and_bounded(self):
        expected, actual = self.record(), self.record()
        self.assertTrue(audit.compare_scaler_records(expected, actual)["matches"])
        actual["mean"][0] = 5e-13
        actual["target_scale"] += 5e-13
        self.assertTrue(audit.compare_scaler_records(expected, actual)["matches"])
        actual["mean"][0] = 2e-12
        self.assertFalse(audit.compare_scaler_records(expected, actual)["matches"])

    def test_same_numbers_cannot_mask_a_different_training_cohort(self):
        expected, actual = self.record(), self.record()
        actual["train_dates"][-1] = "2024-01-03"
        self.assertFalse(audit.compare_scaler_records(expected, actual)["matches"])
        actual = self.record()
        actual["training_sha256"] = "b" * 64
        self.assertFalse(audit.compare_scaler_records(expected, actual)["matches"])

    def test_malformed_or_nonfinite_scaler_does_not_match(self):
        expected, actual = self.record(), self.record()
        actual["std"] = actual["std"][:-1]
        self.assertFalse(audit.compare_scaler_records(expected, actual)["matches"])
        actual = self.record()
        actual["target_scale"] = float("nan")
        self.assertFalse(audit.compare_scaler_records(expected, actual)["matches"])


if __name__ == "__main__":
    unittest.main()
