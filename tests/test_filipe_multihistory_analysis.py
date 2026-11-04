"""Numerical and leakage-clock checks for the four-history forecast analysis."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import bds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_filipe_multihistory as analysis


def short_frame(histories=(7, 14), seeds=(4101, 4102), n=12):
    rows = []
    dates = pd.date_range("2024-01-01", periods=n, tz="UTC")
    target = np.linspace(.01, .03, n)
    for history in histories:
        for ordinal, model in enumerate(analysis.MODELS):
            for seed in seeds:
                for i, day in enumerate(dates):
                    rows.append({"history_days": history, "model": model, "seed": seed,
                                 "variant": "relative_window", "outcome_day": day.strftime("%Y-%m-%d"),
                                 "fold_id": day.strftime("%Y-%m"), "y_true": target[i],
                                 "y_pred": target[i]+.001*(ordinal+1)+.0001*(seed-4101),
                                 "persistence_pred": target[i]+.0005, "mase_train_scale": .002,
                                 "input_first_day": (day-pd.Timedelta(days=history)).strftime("%Y-%m-%d"),
                                 "input_last_day": (day-pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                 "prediction_origin_utc": day.isoformat(),
                                 "target_available_at_utc": (day+pd.Timedelta(days=1)).isoformat()})
    return pd.DataFrame(rows)


class MultihistoryAnalysisTests(unittest.TestCase):
    def validate_small(self, frame):
        return analysis.validate_predictions(frame, "cv", sorted(frame.outcome_day.unique()),
                                             histories=(7, 14), seeds=(4101, 4102))

    def test_complete_grid_and_causal_input_clocks(self):
        frame = short_frame()
        self.assertEqual(len(self.validate_small(frame)), 2*3*2*12)
        frame.loc[0, "input_last_day"] = frame.loc[0, "outcome_day"]
        with self.assertRaisesRegex(ValueError, "information clock"):
            self.validate_small(frame)

    def test_missing_duplicate_and_different_labels_rejected(self):
        frame = short_frame()
        with self.assertRaises(ValueError):
            self.validate_small(frame.iloc[1:])
        with self.assertRaises(ValueError):
            self.validate_small(pd.concat([frame, frame.iloc[:1]]))
        frame.loc[0, "y_true"] += .001
        with self.assertRaisesRegex(ValueError, "share labels"):
            self.validate_small(frame)

    def test_zero_volatility_is_legal_and_mape_reports_mask(self):
        frame = short_frame()
        frame.loc[frame.outcome_day.eq("2024-01-01"), "y_true"] = 0.
        self.validate_small(frame)
        first_path = frame.query("history_days == 7 and model == 'MLP' and seed == 4101")
        result = analysis.metric_values(first_path)
        self.assertAlmostEqual(result["mape_coverage"], 11/12)

    def test_metrics_are_original_scale_and_keep_historical_mase(self):
        frame = pd.DataFrame({"y_true": [1., 2., 4.], "y_pred": [2., 2., 2.],
                              "mase_train_scale": [1., 2., 4.]})
        result = analysis.metric_values(frame)
        self.assertAlmostEqual(result["mse"], 5/3)
        self.assertAlmostEqual(result["rmse"], np.sqrt(5/3))
        self.assertAlmostEqual(result["mae"], 1.)
        self.assertAlmostEqual(result["mape"], 50.)
        self.assertAlmostEqual(result["mase"], .5)

    def test_persistence_is_not_pseudoreplicated_across_seeds(self):
        annual, monthly = analysis.metric_tables(short_frame())
        self.assertEqual(len(annual.query("model == 'Persistencia'")), 2)
        self.assertTrue(annual.query("model == 'Persistencia'").seed.isna().all())
        summary = analysis.metric_summary(annual)
        deterministic = summary.query("model == 'Persistencia'")
        self.assertTrue(deterministic.sample_sd.isna().all())
        self.assertTrue(deterministic.ci95_lower.isna().all())
        self.assertTrue(deterministic.n_total.eq(1).all())
        neural = summary.query("model == 'MLP' and metric == 'rmse'")
        self.assertTrue(neural.n_total.eq(2).all())
        self.assertTrue(neural.sample_sd.gt(0).all())

    def test_seed_ci_uses_student_t_and_not_month_count(self):
        annual, _ = analysis.metric_tables(short_frame())
        part = annual.query("history_days == 7 and model == 'MLP'")
        result = analysis.metric_summary(part).query("metric == 'rmse'").iloc[0]
        width = stats.t.ppf(.975, 1)*part.rmse.std(ddof=1)/np.sqrt(2)
        self.assertAlmostEqual(result.ci95_upper, part.rmse.mean()+width)

    def test_holm_retains_undefined_planned_tests_and_log_underflow(self):
        rows = [analysis.probability_fields(np.log(.01)), analysis.probability_fields(np.nan),
                analysis.probability_fields(np.log(.03))]
        analysis.adjust_holm(rows, "planned-three")
        expected = multipletests([.01, 1., .03], method="holm")[1]
        self.assertAlmostEqual(rows[0]["p_holm"], expected[0])
        self.assertTrue(np.isnan(rows[1]["p_holm"]))
        self.assertEqual(rows[2]["family_size_planned"], 3)
        tiny = [analysis.probability_fields(-1000.)]
        analysis.adjust_holm(tiny, "tiny")
        self.assertTrue(tiny[0]["p_underflow"])
        self.assertEqual(tiny[0]["log_p_holm"], -1000.)
        self.assertIn("exp(", tiny[0]["p_value_display"])

    def test_bds_matches_independent_library_and_normal_tail(self):
        rng = np.random.default_rng(123)
        residual = rng.normal(size=31)
        rows = analysis.bds_for_residuals(residual)
        actual, probabilities = bds(residual, max_dim=3, epsilon=1.5*residual.std(ddof=1))
        for ordinal, row in enumerate(rows):
            self.assertAlmostEqual(row["statistic"], actual[ordinal], places=12)
            self.assertAlmostEqual(row["p_value"], probabilities[ordinal], places=12)
            self.assertEqual(row["n_dates"], 31)
            self.assertIn("approximate_short_block", row["status"])

    def test_bds_singular_and_short_rows_are_kept_undefined(self):
        for residual in [np.zeros(31), np.arange(4, dtype=float)]:
            rows = analysis.bds_for_residuals(residual)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(np.isnan(row["p_value"]) for row in rows))
            self.assertTrue(all("undefined" in row["status"] for row in rows))

    def test_bds_does_not_compress_internal_calendar_gaps(self):
        frame = short_frame(histories=(7,), seeds=(4101,), n=31)
        frame = frame.loc[~frame.outcome_day.eq("2024-01-15")]
        raw, _, _ = analysis.bds_tables(frame, "cv")
        january = raw.loc[raw.fold_id.eq("2024-01")]
        self.assertTrue(january.status.eq("undefined_noncontiguous_calendar").all())
        self.assertTrue(january.p_value.isna().all())
        self.assertEqual(len(raw), 3*12*2)  # Empty planned months retain hypotheses.

    def test_monthly_contiguity_is_independent_of_pandas_timestamp_unit(self):
        frame = short_frame(histories=(7,), seeds=(4101,), n=31)
        rng = np.random.default_rng(7)
        for model in analysis.MODELS:
            mask = frame.model.eq(model)
            frame.loc[mask, "y_pred"] = frame.loc[mask, "y_true"] + rng.normal(0, .001, 31)
        raw, _, _ = analysis.bds_tables(frame, "cv")
        january = raw.loc[raw.fold_id.eq("2024-01")]
        self.assertTrue(january.status.eq("calculated_approximate_short_block").all())
        self.assertTrue(january.p_value.notna().all())

    def test_cv_selection_cannot_use_test_and_breaks_exact_ties_by_history(self):
        rows = []
        for model in analysis.MODELS:
            for history in analysis.HISTORIES:
                rows.append({"model": model, "history_days": history, "metric": "rmse",
                             "mean": .1 if model == "MLP" else .2, "sample_sd": .01})
        selected = analysis.select_from_cv(pd.DataFrame(rows))
        self.assertEqual(selected["selected_history_by_model"]["MLP"]["history_days"], 7)
        self.assertEqual(selected["point_best_neural_candidate"]["model"], "MLP")
        self.assertFalse(selected["test_selection_allowed"])

    def test_cv_selection_signed_tolerance_then_history_then_parameters(self):
        rows = []
        for model in analysis.MODELS:
            for history in analysis.HISTORIES:
                rows.append({"model": model, "history_days": history, "metric": "rmse",
                             "mean": .1 + (5e-13 if history == 7 else 0.), "sample_sd": .01})
        selected = analysis.select_from_cv(pd.DataFrame(rows))
        self.assertEqual(selected["point_best_neural_candidate"]["history_days"], 7)
        self.assertEqual(selected["point_best_neural_candidate"]["model"], "MLP")


if __name__ == "__main__":
    unittest.main()
