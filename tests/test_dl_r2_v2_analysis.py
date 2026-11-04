"""Numerical and cohort checks for exploratory R2/CV reporting; no fitting."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_dl_r2_v2 import (prediction_metrics, mean_interval, validate_prediction_frame,
                             recompute_runs, compare_variants, per_fold_metrics, MODELS)


def small_predictions():
    rows = []
    for variant in ["absolute_log", "relative_window"]:
        for model in MODELS:
            for seed in [4101, 4102, 4103]:
                for i, date in enumerate(["2024-01-01", "2024-01-02", "2024-01-03"]):
                    y = .01+.002*i
                    rows.append(dict(variant=variant, model=model, seed=seed, history_days=7,
                                     fold_id="2024-01", outcome_day=date, y_true=y,
                                     y_pred=y+.001*(seed-4100)*(1 if variant == "absolute_log" else .5),
                                     persistence_pred=y-.001, mase_train_scale=.002))
    return pd.DataFrame(rows)


def summaries(frame):
    rows = []
    for key, group in frame.groupby(["variant", "model", "seed"]):
        rows.append({**dict(zip(["variant", "model", "seed"], key)),
                     **prediction_metrics(group.y_true, group.y_pred, group.mase_train_scale),
                     "parameters": 10, "training_seconds": 1, "inference_seconds": .01})
    return pd.DataFrame(rows)


class R2AndCVAnalysis(unittest.TestCase):
    def test_R2_identity_negative_score_and_bias_decomposition(self):
        y = np.array([1., 2., 3., 4.])
        p = y+2
        actual = prediction_metrics(y, p)
        self.assertAlmostEqual(actual["r2"], 1-actual["rmse"]**2/np.var(y))
        self.assertLess(actual["r2"], 0)
        self.assertAlmostEqual(actual["mse"], actual["squared_bias"]+actual["error_variance"])
        self.assertAlmostEqual(actual["bias"], 2)
        self.assertAlmostEqual(actual["squared_bias_fraction_of_mse"], 1)

    def test_constant_predictions_and_singleton_targets_are_explicit(self):
        actual = prediction_metrics(np.array([1., 2., 3.]), np.full(3, .1))
        self.assertEqual(actual["prediction_sd_ddof0"], 0)
        self.assertTrue(np.isnan(actual["pearson_correlation"]))
        for y in [np.full(8, .1), np.array([.1])]:
            actual = prediction_metrics(y, np.full(len(y), .2))
            self.assertTrue(np.isnan(actual["r2"]))

    def test_MASE_uses_calendar_specific_training_scale(self):
        result = prediction_metrics([1., 2., 3.], [2., 4., 6.], [.5, 2., np.nan])
        self.assertAlmostEqual(result["mase"], 1.5)
        self.assertAlmostEqual(result["mase_coverage"], 2/3)
        self.assertNotEqual(result["mase"], np.mean([1., 2.])/1.25)

    def test_mean_interval_matches_Student_reference(self):
        values = np.array([.2, .4, .5, .7, .9])
        actual = mean_interval(values)
        half = stats.t.ppf(.975, 4)*values.std(ddof=1)/np.sqrt(5)
        self.assertAlmostEqual(actual["ci95_lower"], values.mean()-half)
        self.assertAlmostEqual(actual["ci95_upper"], values.mean()+half)
        self.assertTrue(np.isnan(mean_interval(np.full(5, .1))["ci95_lower"]))

    def test_loader_checks_fold_dates_pairing_and_reserved_dates(self):
        frame = small_predictions()
        options = dict(variants=["absolute_log", "relative_window"], seeds=[4101, 4102, 4103],
                       dates=["2024-01-01", "2024-01-02", "2024-01-03"], require_monthly_folds=True)
        self.assertEqual(len(validate_prediction_frame(frame, **options)), 54)
        wrong = frame.copy()
        wrong.loc[0, "fold_id"] = "2024-02"
        with self.assertRaisesRegex(ValueError, "monthly CV"):
            validate_prediction_frame(wrong, **options)
        wrong = frame.copy()
        wrong.loc[0, "outcome_day"] = "2025-01-01"
        with self.assertRaisesRegex(ValueError, "Reserved test"):
            validate_prediction_frame(wrong, **options)
        with self.assertRaisesRegex(ValueError, "Duplicate/overlapping"):
            validate_prediction_frame(pd.concat([frame, frame.iloc[[0]]]), **options)
        with self.assertRaisesRegex(ValueError, "Incomplete common dates"):
            validate_prediction_frame(frame.iloc[1:], **options)
        missing = frame.loc[~(frame.variant.eq("absolute_log") & frame.model.eq("CNN1D"))]
        with self.assertRaisesRegex(ValueError, "Missing complete"):
            validate_prediction_frame(missing, **options)

    def test_stored_bad_R2_is_rejected_and_subcohort_is_recomputed(self):
        frame = small_predictions()
        runs = summaries(frame)
        actual = recompute_runs(frame, runs)
        self.assertEqual(len(actual), 18)
        bad = runs.copy()
        bad.loc[0, "r2"] += .5
        with self.assertRaisesRegex(ValueError, "Stored r2"):
            recompute_runs(frame, bad)
        subset = frame.loc[frame.outcome_day.ne("2024-01-01")]
        recalc = recompute_runs(subset, runs, validate_stored=False)
        self.assertTrue(recalc.training_seconds.isna().all())
        self.assertTrue(recalc.n_dates.eq(2).all())

    def test_Holm_family_contains_only_declared_CV_pairs(self):
        frame = small_predictions()
        metric_rows = recompute_runs(frame, summaries(frame))
        actual = compare_variants(metric_rows, ["absolute_log", "relative_window"])
        self.assertEqual(len(actual), 3)
        self.assertTrue(actual.family_size_planned.eq(3).all())
        for row in actual.itertuples():
            selected = metric_rows.loc[metric_rows.model.eq(row.model)]
            old = selected.loc[selected.variant.eq("absolute_log")].sort_values("seed")
            new = selected.loc[selected.variant.eq("relative_window")].sort_values("seed")
            reference = stats.ttest_rel(new.rmse, old.rmse)
            self.assertAlmostEqual(row.p_value, reference.pvalue)
            self.assertLess(row.mean_difference, 0)
            self.assertGreater(row.r2_mean_difference, 0)
            self.assertNotIn("original_static", [row.old_variant, row.new_variant])

    def test_per_fold_R2_uses_each_month_not_an_average_of_dates(self):
        frame = small_predictions()
        actual = per_fold_metrics(frame)
        self.assertEqual(len(actual), 18)
        self.assertTrue(actual.n_dates.eq(3).all())
        expected = prediction_metrics([.01, .012, .014], [.011, .013, .015])
        first = actual.loc[actual.variant.eq("absolute_log") & actual.model.eq("MLP") & actual.seed.eq(4101)].iloc[0]
        self.assertAlmostEqual(first.r2, expected["r2"])


if __name__ == "__main__":
    unittest.main()
