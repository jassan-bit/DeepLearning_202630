"""Synthetic finite-search reports: no model or market observations."""
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from e2_hpo_report import build_hpo_report, write_hpo_report


def fixture(model="TSMixer"):
    candidates, rankings = [], []
    for i in range(12):
        if model == "Ridge":
            params = {"alpha": 10.**(-4+i/2)}
        elif model == "ARIMA":
            params = {"order": [i//4, (i//2)%2, i%2], "trend": "c" if (i//2)%2 == 0 else "n"}
        else:
            params = {"learning_rate": 10.**(-4+i/12), "weight_decay": 1e-4,
                      "dropout": i/40, "batch_size": (16, 32, 64)[i%3]}
        identity = model+f"-c{i+1:02d}"
        candidates.append({"model": model, "candidate_id": identity, "parameters": params,
                           "source_reference_point": i == 11, "assigned_compute_seconds": 172800})
        rankings.append({"candidate_id": identity, "status": "passed", "mean_seed_RMSE": .02-i*.0005,
                         "latency_ms": 1.+i, "parameters_total": 100, "dates_sha256": "a"*64})
    return candidates, rankings


class ReportTests(unittest.TestCase):
    def test_complete_keeps_twelve_and_correct_estimand(self):
        c, r = fixture(); report = build_hpo_report(c, r, model="TSMixer")
        self.assertEqual(len(report["all_candidates"]), 12)
        self.assertEqual(report["status"], "complete_registered_finite_search")
        self.assertEqual(report["best_observed"]["candidate_id"], "TSMixer-c12")
        self.assertFalse(report["inference"]["global_optimum_or_held_out_winner_performance_proved"])

    def test_failed_missing_and_nonfinite_scores_never_become_winners(self):
        c, r = fixture(); r[-1]["mean_seed_RMSE"] = math.nan
        r[-2].update(status="failed", mean_seed_RMSE=0., failures=[{"error": "divergence"}])
        report = build_hpo_report(c, r[:-3]+r[-2:], model="TSMixer")
        self.assertEqual(report["status"], "incomplete_or_failed_search")
        self.assertEqual(report["best_observed"]["candidate_id"], "TSMixer-c09")
        for row in report["all_candidates"][-3:]:
            self.assertIsNone(row["mean_seed_RMSE"])
        json.dumps(report, allow_nan=False)

    def test_source_outside_random_support_is_disclosed_without_clipping(self):
        c, r = fixture(); c[-1]["parameters"].update(learning_rate=1e-5, dropout=.9)
        best = build_hpo_report(c, r, model="TSMixer")["best_observed"]
        self.assertTrue(best["source_reference_point"])
        self.assertLess(best["positions"]["learning_rate"]["support_fraction"], 0)
        self.assertGreater(best["positions"]["dropout"]["support_fraction"], 1)
        self.assertFalse(best["positions"]["dropout"]["in_random_support"])

    def test_tie_break_matches_registered_order(self):
        c, r = fixture()
        for row in r: row.update(mean_seed_RMSE=.01, latency_ms=2., parameters_total=100)
        r[1]["latency_ms"] = 1.; r[2]["latency_ms"] = 1.; r[2]["parameters_total"] = 99
        self.assertEqual(build_hpo_report(c, r, model="TSMixer")["best_observed"]["candidate_id"], "TSMixer-c03")

    def test_monotonic_association_and_non_estimable_axes(self):
        c, r = fixture(); curves = build_hpo_report(c, r, model="TSMixer")["curves"]
        self.assertAlmostEqual(curves["learning_rate"]["Spearman_rho"], -1.)
        self.assertIsNone(curves["weight_decay"]["Spearman_rho"])
        self.assertEqual(curves["weight_decay"]["association_status"], "not_estimable")
        self.assertIsNone(build_hpo_report(c, r[:2], model="TSMixer")["curves"]["learning_rate"]["Spearman_rho"])
        self.assertNotIn("p", curves["learning_rate"])

    def test_categories_count_only_valid_scores_and_preserve_empty_levels(self):
        c, r = fixture(); report = build_hpo_report(c, r[:1], model="TSMixer")
        levels = report["curves"]["batch_size"]["level_summaries"]
        self.assertEqual([x["valid_candidates"] for x in levels], [1, 0, 0])
        self.assertIsNone(levels[1]["mean_candidate_RMSE"])

    def test_arima_is_categorical_and_ridge_alpha_logarithmic(self):
        c, r = fixture("ARIMA"); report = build_hpo_report(c, r, model="ARIMA")
        self.assertEqual(set(report["curves"]), {"order"})
        self.assertEqual(report["curves"]["order"]["association_status"], "categorical_descriptive_only")
        c, r = fixture("Ridge"); report = build_hpo_report(c, r, model="Ridge")
        self.assertEqual(report["curves"]["alpha"]["axis"]["scale"], "log10")
        self.assertAlmostEqual(report["best_observed"]["positions"]["alpha"]["support_fraction"], 1.)

    def test_duplicate_unknown_and_different_calendars_are_rejected(self):
        c, r = fixture()
        with self.assertRaises(ValueError): build_hpo_report(c, r+[r[0]], model="TSMixer")
        r[0]["candidate_id"] = "other"
        with self.assertRaises(ValueError): build_hpo_report(c, r, model="TSMixer")
        c, r = fixture(); r[0]["dates_sha256"] = "b"*64
        with self.assertRaises(ValueError): build_hpo_report(c, r, model="TSMixer")

    def test_outputs_are_aggregate_standalone_artifacts_and_not_overwritten(self):
        c, r = fixture(); report = build_hpo_report(c, r, model="TSMixer")
        with tempfile.TemporaryDirectory() as out:
            files = write_hpo_report(report, out)
            self.assertEqual([p.suffix for p in files], [".json", ".svg"])
            self.assertEqual(json.loads(files[0].read_text()), report)
            self.assertIn("<svg", files[1].read_text())
            with self.assertRaises(ValueError): write_hpo_report(report, out)


if __name__ == "__main__":
    unittest.main()
