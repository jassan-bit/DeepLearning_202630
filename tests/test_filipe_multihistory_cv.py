"""Real-library checks for D14/D21/D28 with a fixed sigma7 target."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CV_ENVIRONMENT = (np.__version__.startswith("1.26.")
                  and importlib.util.find_spec("tsxv") is not None)


@unittest.skipUnless(CV_ENVIRONMENT, "Run with dedicated timeseries-cv0.1.5 / NumPy1.26")
class MultiHistoryCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        from prepare_filipe_multihistory import build_manifest
        cls.manifests = {history: build_manifest(history, ROOT) for history in (7, 14, 21, 28)}

    def test_every_history_has_the_same_daily_h1_evaluation_calendar(self):
        for history, manifest in self.manifests.items():
            self.assertEqual(manifest["target_returns_days"], 7)
            self.assertEqual(manifest["forecast_days"], 1)
            self.assertEqual(manifest["models_fitted"], 0)
            self.assertEqual(manifest["test_files_opened"], 0)
            outcomes = [i for fold in manifest["folds"] for i in fold["test_indices"]]
            self.assertEqual(outcomes, list(range(365, 731)))
            self.assertEqual(len(manifest["folds"]), 12)
            self.assertEqual(manifest["folds"][1]["test"]["count"], 29)
            for fold in manifest["folds"]:
                for outcome, window in zip(fold["test_indices"], fold["test_input_indices"]):
                    self.assertEqual(window, list(range(outcome-history, outcome)))

    def test_each_history_controls_both_embargoes_not_the_target_definition(self):
        expected_first_train = {7: 343, 14: 322, 21: 301, 28: 280}
        for history, manifest in self.manifests.items():
            self.assertEqual(manifest["folds"][0]["train"]["count"], expected_first_train[history])
            for fold in manifest["folds"]:
                self.assertEqual(fold["inner_validation"]["count"], 90-history)
                self.assertLessEqual(max(fold["train_indices"]), fold["fit_origin_index"]-history-1)
                self.assertLessEqual(max(fold["inner_train_indices"]), fold["inner_validation_start_index"]-history-1)

    def test_incomplete_day_never_enters_any_admitted_input(self):
        for history, manifest in self.manifests.items():
            train = set(manifest["folds"][0]["train_indices"])
            self.assertIn(81, train)
            self.assertNotIn(82+history, train)
            self.assertIn(83+history, train)
            for fold in manifest["folds"]:
                for cohort in ("train_indices", "inner_train_indices", "inner_validation_indices"):
                    for outcome in fold[cohort]:
                        self.assertNotIn(82, range(outcome-history, outcome))

    def test_d7_mapping_is_identical_to_preexisting_signed_generator(self):
        from prepare_dl_r2_cv import build_manifest
        original = build_manifest(ROOT)
        self.assertEqual(self.manifests[7]["folds"], original["folds"])
        self.assertEqual(self.manifests[7]["source_hashes"], original["source_hashes"])


if __name__ == "__main__":
    unittest.main()
