"""Integration checks for the real-library monthly development CV adaptation."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CV_ENVIRONMENT = (np.__version__.startswith("1.26.")
                  and importlib.util.find_spec("tsxv") is not None)


@unittest.skipUnless(CV_ENVIRONMENT, "Run in the dedicated timeseries-cv0.1.5 / NumPy1.26 environment")
class MonthlyForwardCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        from prepare_dl_r2_cv import build_manifest
        cls.manifest = build_manifest(ROOT)
        cls.folds = cls.manifest["folds"]

    def test_covers_leap_year_once_with_daily_not_month_ahead_origins(self):
        outcomes = [i for fold in self.folds for i in fold["test_indices"]]
        self.assertEqual(outcomes, list(range(365, 731)))
        self.assertEqual(len(self.folds), 12)
        self.assertEqual(self.folds[1]["test"]["count"], 29)
        self.assertEqual(self.folds[1]["test"]["last_day"], "2024-02-29")
        self.assertEqual(sum(f["test"]["count"] for f in self.folds[6:]), 184)
        for fold in self.folds:
            self.assertEqual(len(fold["daily_prediction_origins_utc"]), fold["test"]["count"])
            for outcome, window in zip(fold["test_indices"], fold["test_input_indices"]):
                self.assertEqual(window, list(range(outcome - 7, outcome)))

    def test_explicit_embargo_removes_library_extra_candidate_at_both_cuts(self):
        january = self.folds[0]
        self.assertEqual(january["library_outer_train_candidate_last_index"], 358)
        self.assertEqual(january["outer_training_last_allowed_index"], 357)
        self.assertEqual(january["train"]["last_day"], "2023-12-24")
        self.assertNotIn(358, january["train_indices"])
        self.assertEqual(january["train"]["count"], 343)
        self.assertEqual(january["inner_validation_start_day_utc"], "2023-10-03")
        self.assertEqual(january["inner_train"]["last_day"], "2023-09-25")
        self.assertEqual(january["inner_train"]["count"], 253)
        self.assertEqual(january["inner_validation"]["count"], 83)
        for fold in self.folds:
            self.assertLessEqual(max(fold["train_indices"]), fold["fit_origin_index"] - 8)
            self.assertLessEqual(max(fold["inner_train_indices"]), fold["inner_validation_start_index"] - 8)

    def test_gap_propagation_excludes_invalid_labels_and_windows(self):
        # March24 is index82; its missing close invalidates two returns and
        # sigma7 through March31. April1 is again admissible without bridging.
        january = self.folds[0]
        train = set(january["train_indices"])
        self.assertIn(81, train)
        self.assertTrue(train.isdisjoint(range(82, 90)))
        self.assertIn(90, train)
        for fold in self.folds:
            for key in ("train_indices", "inner_train_indices", "inner_validation_indices"):
                for outcome in fold[key]:
                    self.assertNotIn(82, range(outcome - 7, outcome))
        self.assertEqual(self.manifest["library"]["version"], "0.1.5")
        self.assertEqual(self.manifest["models_fitted"], 0)
        self.assertEqual(self.manifest["test_files_opened"], 0)


if __name__ == "__main__":
    unittest.main()
