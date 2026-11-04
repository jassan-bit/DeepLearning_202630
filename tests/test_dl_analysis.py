"""Numerical invariants of analysis, using synthetic arrays only."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_dl_benchmark import (add_holm, paired_statistics, dm_hln,
                                  longest_contiguous_indices, bootstrap_scores,
                                  mean_seed_rmse, chi_square_logsf, block_indices,
                                  load_predictions)


class AnalysisInvariants(unittest.TestCase):
    def test_holm_known_fixture_and_missing_family_members(self):
        rows = [{"log_p_value": np.log(p)} for p in [.01, .04, .03]]
        add_holm(rows, "fixture")
        np.testing.assert_allclose([r["p_holm"] for r in rows], [.03, .06, .06])
        rows = [{"log_p_value": np.log(.01)}]+[{"log_p_value": np.nan} for _ in range(8)]
        add_holm(rows, "planned9")
        self.assertAlmostEqual(rows[0]["p_holm"], .09)
        self.assertTrue(all(np.isnan(r["p_holm"]) for r in rows[1:]))
        self.assertEqual(rows[0]["family_size_planned"], 9)

    def test_constant_differences_are_not_fabricated_significance(self):
        for number in [1.0, .1]:
            result = paired_statistics(np.full(10, number), np.zeros(10))
            self.assertIn("zero_pairwise", result["status"])
            self.assertTrue(np.isnan(result["p_value"]))
            self.assertTrue(np.isnan(result["paired_cohen_d_z"]))
            self.assertTrue(np.isnan(result["shapiro_p"]))

    def test_paired_t_matches_library_and_null_is_separate(self):
        x = np.array([.5, .9, .2, .4, .7, .8, .3, .1, .6, 1.0])
        y = np.linspace(.1, .4, len(x))
        actual = paired_statistics(x, y)
        expected = stats.ttest_rel(x, y)
        self.assertAlmostEqual(actual["t_statistic"], expected.statistic)
        self.assertAlmostEqual(actual["p_value"], expected.pvalue)
        shifted = paired_statistics(x, y, null_difference=.05, alternative="greater")
        expected = stats.ttest_1samp(x-y, .05, alternative="greater")
        self.assertAlmostEqual(shifted["p_value"], expected.pvalue)
        self.assertAlmostEqual(shifted["mean_difference"], actual["mean_difference"])

    def test_DM_lag_zero_HLN_h1_equals_one_sample_t(self):
        x = np.array([.2, -.1, .7, .4, -.2, .5, .1, -.4, .6, .3])
        actual = dm_hln(x, lag=0, horizon=1)
        expected = stats.ttest_1samp(x, 0)
        self.assertAlmostEqual(actual["DM_HLN"], expected.statistic)
        self.assertAlmostEqual(actual["p_value"], expected.pvalue)
        for number in [1.0, .1]:
            constant = dm_hln(np.full(40, number), lag=14)
            self.assertIn("constant", constant["status"])
            self.assertTrue(np.isnan(constant["p_value"]))

    def test_longest_segment_uses_calendar_and_earliest_tie(self):
        dates = pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-04", "2024-07-05"], utc=True)
        np.testing.assert_array_equal(longest_contiguous_indices(dates), [0, 1])

    def test_seed_RMSE_is_not_ensemble_error(self):
        # Seed forecasts y+1 and y-1 have RMSE1; their ensemble would have zero error.
        squared = np.ones((3, 2, 40))
        np.testing.assert_allclose(mean_seed_rmse(squared), 1)
        draws = bootstrap_scores(squared, length=7, replicates=100, seed=19)
        np.testing.assert_allclose(draws, 1)

    def test_block_pairs_share_indices_and_reverse_difference(self):
        errors = np.stack([np.linspace(.1, 1, 40), np.linspace(.3, .8, 40)])
        losses = np.stack([errors**2, errors**2, (errors*.5)**2])
        draws = bootstrap_scores(losses, length=7, replicates=250, seed=9)
        np.testing.assert_array_equal(draws[:, 0]-draws[:, 1], 0)
        diff = draws[:, 0]-draws[:, 2]
        ci = np.quantile(diff, [.025, .975])
        reverse = np.quantile(-diff, [.025, .975])
        np.testing.assert_allclose(reverse, -ci[::-1])

    def test_block_functional_matches_explicit_seed_then_date_calculation(self):
        losses = np.arange(1, 121, dtype=float).reshape(3, 2, 20)/100
        indices = block_indices(20, 4, 15, np.random.default_rng(12))
        expected = np.array([np.sqrt(losses[:, :, index].mean(axis=-1)).mean(axis=1)
                             for index in indices])
        actual = bootstrap_scores(losses, length=4, replicates=15, seed=12)
        np.testing.assert_allclose(actual, expected)

    def test_loader_rejects_reserved_year_and_duplicate_predictions(self):
        rows = [dict(phase="development_benchmark", run_id=f"{model}1001", model=model,
                     history_days=14, config_id=model, seed=1001, outcome_day=day,
                     y_true=.02, y_pred=.021, persistence_pred=.019)
                for model in ["MLP", "LSTM", "CNN1D"]
                for day in ["2024-07-01", "2024-07-02"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"synthetic.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            data = load_predictions(path)
            self.assertEqual(data.predictions.shape, (4, 1, 2))
            future = pd.DataFrame(rows)
            future.loc[:, "outcome_day"] = "2025-01-01"
            future.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "reserved test"):
                load_predictions(path)
            pd.DataFrame(rows+[rows[0]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_predictions(path)

    def test_stable_chi_square_tail(self):
        for degree in [7, 14, 28]:
            for statistic in [.1, 1, 10, 50, 200]:
                self.assertAlmostEqual(chi_square_logsf(statistic, degree), stats.chi2.logsf(statistic, degree), places=11)
        self.assertTrue(np.isfinite(chi_square_logsf(10000, 7)))


if __name__ == "__main__":
    unittest.main()
