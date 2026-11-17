"""Read-only checks for the pinned E2/E3 statistical input panel.

These tests validate existing files and copies of their documents. They do not
run models, hypothesis tests, resampling, or the statistical analysis engine.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import inputs


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
JSON_FILES = {
    "combined": "combined_predictors.json", "old": "old_e2_predictions.json",
    "summary": "e2_summary.json", "e2_plan": "e2_plan.json", "e3_plan": "e3_plan.json",
    "e3_summary": "e3_summary.json", "snapshot": "e2_snapshot.json",
}
CSV_FILES = {
    "per_seed_metrics": "e2_per_seed_metrics.csv", "comparison": "e2_comparison.csv",
    "e3_metrics": "e3_per_seed_metrics.csv", "e3_predictions": "e3_evaluation_predictions.csv",
}
NEURAL = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "MLP16", "LSTM32", "GRU32")
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
E3 = "CNN-LSTM-atencion-aditiva E3"
SEEDS = list(range(271801, 271811))


def documents():
    result = {}
    for key, name in JSON_FILES.items():
        result[key] = json.loads((DATA / name).read_text())
    for key, name in CSV_FILES.items():
        with (DATA / name).open(newline="") as stream:
            result[key] = list(csv.DictReader(stream))
    return result


class InputPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = documents()

    def changed(self):
        return deepcopy(self.original)

    def rejects(self, changed):
        with self.assertRaises(ValueError):
            inputs.validate_documents(changed)

    def test_pinned_panel_contains_only_the_complete_17_model_family(self):
        import numpy as np
        panel = inputs.load_inputs(DATA)
        expected_order = [E3, *NEURAL, *CLASSICAL]
        self.assertEqual(list(panel["model_order"]), expected_order)
        self.assertEqual(list(panel["rivals"]), [*NEURAL, *CLASSICAL])
        self.assertEqual(list(panel["predictions"]), expected_order)
        self.assertEqual(list(panel["dates"]), self.original["combined"]["dates"])
        self.assertEqual(np.asarray(panel["targets"]).dtype, np.dtype("float64"))
        self.assertFalse(panel["targets"].flags.writeable)
        np.testing.assert_array_equal(panel["targets"], self.original["combined"]["y_true"])
        mask = np.asarray(panel["mask"])
        self.assertEqual(mask.dtype, np.dtype("bool"))
        self.assertEqual(mask.shape, (608,))
        self.assertTrue(mask.all())
        self.assertFalse(mask.flags.writeable)
        self.assertEqual(panel["train_counts"][E3], 532)
        self.assertTrue(all(panel["train_counts"][name] == 655 for name in (*NEURAL, *CLASSICAL)))
        for name in expected_order:
            with self.subTest(model=name):
                predictions = panel["predictions"][name]
                seeds = ("deterministic",) if name in CLASSICAL else tuple(SEEDS)
                self.assertEqual(tuple(panel["seeds"][name]), seeds)
                self.assertEqual(predictions.shape, (len(seeds), 608))
                self.assertEqual(predictions.dtype, np.dtype("float64"))
                self.assertTrue(np.isfinite(predictions).all())
                self.assertFalse(predictions.flags.writeable)
        self.assertIsInstance(panel["provenance"], dict)
        self.assertTrue(panel["provenance"])
        self.assertIs(panel["provenance"]["hypothesis_tests_performed"], False)
        self.assertIs(panel["provenance"]["legacy_original_analysis_ignored"], True)

    def test_semantic_validation_does_not_mutate_any_source_document(self):
        document = self.changed()
        untouched = deepcopy(document)
        inputs.validate_documents(document)
        self.assertEqual(document, untouched)

    def test_missing_or_duplicate_e2_predictor_is_rejected(self):
        for operation in (lambda rows: rows.pop(),
                          lambda rows: rows.append(deepcopy(rows[0])),
                          lambda rows: rows.__setitem__(82, deepcopy(rows[81]))):
            with self.subTest(operation=operation):
                changed = self.changed()
                operation(changed["combined"]["predictors"])
                self.rejects(changed)

    def test_e2_historical_seed_cannot_replace_a_registered_new_seed(self):
        changed = self.changed()
        changed["combined"]["predictors"][81]["seed"] = 20261116
        self.rejects(changed)

    def test_e3_historical_seed_is_not_an_eleventh_trajectory(self):
        changed = self.changed()
        changed["e3_predictions"][0]["seed"] = "20261116"
        self.rejects(changed)

    def test_missing_or_duplicate_e3_dated_row_is_rejected(self):
        changed = self.changed()
        changed["e3_predictions"].pop()
        self.rejects(changed)
        changed = self.changed()
        changed["e3_predictions"][1] = deepcopy(changed["e3_predictions"][0])
        self.rejects(changed)

    def test_e3_dates_must_equal_the_exact_common_calendar(self):
        changed = self.changed()
        changed["e3_predictions"][0]["date"] = "2024-12-31"
        self.rejects(changed)

    def test_e3_targets_must_match_e2_exactly(self):
        changed = self.changed()
        row = changed["e3_predictions"][0]
        row["target_sigma7"] = str(float(row["target_sigma7"]) + 1e-12)
        self.rejects(changed)

    def test_common_mask_cannot_silently_remove_dates(self):
        changed = self.changed()
        changed["combined"]["mask"][0] = False
        changed["old"]["mask"][0] = False
        self.rejects(changed)

    def test_all_original_81_predictor_objects_are_preserved(self):
        changed = self.changed()
        changed["combined"]["predictors"][0]["audit_only_added_field"] = "changed"
        self.rejects(changed)

    def test_nonfinite_or_negative_new_predictions_are_rejected(self):
        for invalid in (float("nan"), float("inf"), -1e-4):
            with self.subTest(invalid=invalid):
                changed = self.changed()
                changed["combined"]["predictors"][81]["prediction_original"][0] = invalid
                self.rejects(changed)

    def test_e2_per_seed_metric_mismatch_is_rejected(self):
        changed = self.changed()
        row = changed["per_seed_metrics"][0]
        row["RMSE"] = str(float(row["RMSE"]) * 1.1)
        self.rejects(changed)

    def test_e3_per_seed_metric_mismatch_is_rejected(self):
        changed = self.changed()
        row = changed["e3_metrics"][0]
        row["MAE"] = str(float(row["MAE"]) * 1.1)
        self.rejects(changed)

    def test_comparison_mean_and_classical_replication_count_are_checked(self):
        changed = self.changed()
        changed["comparison"][0]["RMSE_mean"] = "0.123"
        self.rejects(changed)
        changed = self.changed()
        classical = next(row for row in changed["comparison"] if row["model"] == "Persistence")
        classical["R"] = "10"
        self.rejects(changed)

    def test_incomplete_summary_is_rejected(self):
        changed = self.changed()
        changed["summary"]["complete"] = False
        self.rejects(changed)

    def test_e2_train_655_and_e3_train_532_cannot_be_relabelled_equal(self):
        changed = self.changed()
        changed["e2_plan"]["data"]["training_examples"] = 532
        self.rejects(changed)
        changed = self.changed()
        changed["e3_plan"]["data"]["fit_count"] = 655
        self.rejects(changed)
        changed = self.changed()
        e3_row = next(row for row in changed["comparison"] if row["model"] == E3)
        e3_row["train_n"] = "655"
        self.rejects(changed)

    def test_source_file_hash_tampering_is_rejected_before_semantic_loading(self):
        with tempfile.TemporaryDirectory(prefix="e3-input-hash-test-") as temporary:
            destination = Path(temporary) / "data"
            shutil.copytree(DATA, destination)
            # Whitespace preserves the JSON meaning but violates its pinned bytes.
            path = destination / "combined_predictors.json"
            with path.open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaises(ValueError):
                inputs.load_inputs(destination)


if __name__ == "__main__":
    unittest.main()
