"""Existing-evidence export checks; stdlib only, no training or market labels."""
from copy import deepcopy
import csv
import io
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_e2_delivery_snapshot as delivery


def fit(candidate=0, seed=271201, *, model="FCN-1D", origin="2024-01-01"):
    return {"model": model, "origin": origin, "candidate_id": "hpo:t%02d" % candidate,
            "job_id": "%s:%s:%s:%s" % (model, origin, candidate, seed),
            "seed": seed, "status": "completed", "optimizer_steps": 10}


def closure(model="FCN-1D", origin="2024-01-01"):
    return {"model": model, "origin": origin, "attempts": 36,
            "winner_id": "hpo:t00", "winner": {"trial": {"optimization_seeds": [271201, 271202]}},
            "fixed_epochs": {"supervised": 20}, "selection_signature": "a" * 64}


def complete(model="FCN-1D", origin="2024-01-01"):
    return [fit(candidate, seed, model=model, origin=origin)
            for candidate in range(18) for seed in (271201, 271202)]


class CoverageTests(unittest.TestCase):
    def test_reused_selection_is_supported_by_legacy_fits_without_double_count(self):
        rows = complete(model="TSMixer")
        reused = complete()
        summary = delivery.summarize_rows(rows, [closure(), closure(model="TSMixer")], reused_rows=reused)
        self.assertEqual(summary["new_fit_receipts"], 36)
        self.assertEqual(summary["legacy_reused_fit_receipts_in_current_selection"], 36)
        self.assertEqual(summary["local_hpo_selections_closed"], 2)
        self.assertEqual(summary["known_optimizer_steps_new_fits"], 360)
        coverage = summary["coverage_by_model_origin"]
        self.assertEqual(coverage[0]["new_fit_receipts"], 0)
        self.assertEqual(coverage[0]["legacy_reused_fit_receipts"], 36)

    def test_partial_origin_does_not_count_as_closed_selection(self):
        summary = delivery.summarize_rows(complete()[:13], [])
        self.assertEqual(summary["local_hpo_selections_closed"], 0)
        self.assertFalse(summary["coverage_by_model_origin"][0]["local_hpo_selection_closed"])
        with self.assertRaisesRegex(ValueError, "36_fit"):
            delivery.summarize_rows(complete()[:13], [closure()])

    def test_duplicate_jobs_and_fake_pair_or_winner_rejected(self):
        rows = complete()
        with self.assertRaisesRegex(ValueError, "duplicate_job"):
            delivery.summarize_rows([*rows, rows[0]], [])
        altered = deepcopy(rows)
        altered[1]["candidate_id"] = "alien"
        with self.assertRaisesRegex(ValueError, "18_paired"):
            delivery.summarize_rows(altered, [closure()])
        altered = deepcopy(rows)
        altered[1]["seed"] = 271203
        with self.assertRaisesRegex(ValueError, "registered_seed_pair"):
            delivery.summarize_rows(altered, [closure()])


class TrialMetadataTests(unittest.TestCase):
    def record(self):
        return {
            "job_id": "job1", "sha256": "a" * 64, "charged_wall_seconds": 12,
            "callback_wall_seconds": 11,
            "outcome": {
                "status": "completed", "prediction_dates": ["2024-01-01", "2024-01-02"],
                "predictions_original": [1e99, -1e99],
                "resource_use": {
                    "identity": {"model": "TSMixer", "origin": "2024-01-01", "stage": "hpo",
                                 "candidate_id": "hpo:t00", "seed": 271201, "payload_sha256": "b" * 64,
                                 "prepared_signature": "c" * 64},
                    "optimizer_steps": 10, "optimizer_steps_unknown": False, "model_parameters": 40,
                    "phase_records": {"supervised": {"best_epoch": 3, "epochs_completed": 5,
                                                         "selected_state_sha256": "d" * 64}},
                },
            },
        }

    def test_export_contains_fit_metadata_and_date_coverage_without_predictions(self):
        row = delivery.trial_row(self.record(), {"path": "campaigns/x/results/y.json", "git_blob_sha": "e" * 40})
        self.assertEqual(row["prediction_count"], 2)
        self.assertEqual(row["best_epoch"], 3)
        self.assertEqual(row["optimizer_steps"], 10)
        self.assertNotIn("predictions_original", row)
        self.assertNotIn("rmse", row)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=delivery.COLUMNS)
        writer.writeheader()
        writer.writerow(row)
        self.assertNotIn("1e+99", stream.getvalue())

    def test_unknown_steps_and_nonfinite_cost_are_not_replaced_by_zero(self):
        row = self.record()
        row["outcome"]["resource_use"]["optimizer_steps_unknown"] = True
        with self.assertRaisesRegex(ValueError, "unknown_optimizer"):
            delivery.trial_row(row, {"path": "x", "git_blob_sha": "e" * 40})
        row = self.record()
        row["charged_wall_seconds"] = math.nan
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            delivery.trial_row(row, {"path": "x", "git_blob_sha": "e" * 40})

    def test_misaligned_or_unsorted_prediction_dates_rejected(self):
        row = self.record()
        row["outcome"]["prediction_dates"].pop()
        with self.assertRaisesRegex(ValueError, "date_count"):
            delivery.trial_row(row, {"path": "x", "git_blob_sha": "e" * 40})
        row = self.record()
        row["outcome"]["prediction_dates"].reverse()
        with self.assertRaisesRegex(ValueError, "sorted"):
            delivery.trial_row(row, {"path": "x", "git_blob_sha": "e" * 40})


if __name__ == "__main__":
    unittest.main()
