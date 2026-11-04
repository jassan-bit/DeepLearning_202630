"""Numerical contracts and information-clock validation for the CV supplement."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_filipe_statistics as supplement


def synthetic_cube(perfect_persistence=False, shift_first=0.):
    dates = pd.date_range("2024-01-01", periods=40, tz="UTC")
    target = np.linspace(.1, .2, 40)
    errors = np.array([np.linspace(.01, .02, 10)+shift_first,
                       np.linspace(.03, .045, 10), np.linspace(.02, .04, 10),
                       np.repeat(0. if perfect_persistence else .005, 10)])
    predictions = target[None, None, :]+errors[:, :, None]
    return supplement.Cube(dates, target, predictions, pd.DataFrame())


class NumericalTests(unittest.TestCase):
    def test_paired_t_matches_one_sample_reference_and_reflects_order(self):
        x, y = np.array([2., 5., 8., 9.]), np.array([1., 3., 4., 8.])
        result = supplement.paired_statistics(x, y)
        reference = stats.ttest_1samp(x-y, popmean=0)
        reverse = supplement.paired_statistics(y, x)
        self.assertAlmostEqual(result["t_statistic"], reference.statistic, places=13)
        self.assertAlmostEqual(result["p_value"], reference.pvalue, places=13)
        self.assertAlmostEqual(reverse["p_value"], result["p_value"], places=13)
        self.assertAlmostEqual(reverse["ci95_lower"], -result["ci95_upper"], places=13)

    def test_constant_differences_are_undefined_not_fabricated_significant(self):
        result = supplement.paired_statistics(np.repeat(.1, 10), np.zeros(10))
        self.assertEqual(result["sd_difference"], 0)
        self.assertTrue(np.isnan(result["p_value"]))
        self.assertIn("undefined", result["status"])

    def test_holm_preserves_undefined_hypothesis_in_family(self):
        rows = [supplement.p_fields(np.log(.01)), supplement.p_fields(np.nan),
                supplement.p_fields(np.log(.03))]
        result = supplement.add_holm(rows, "three")
        reference = multipletests([.01, 1., .03], method="holm")[1]
        self.assertAlmostEqual(result[0]["p_holm"], reference[0])
        self.assertTrue(np.isnan(result[1]["p_holm"]))
        self.assertAlmostEqual(result[2]["p_holm"], reference[2])
        self.assertTrue(all(row["family_size_planned"] == 3 for row in result))

    def test_dm_zero_hac_h1_equals_one_sample_t(self):
        differences = np.array([1., 2., -1., 4., 3., -2., 1., 2.])
        result = supplement.dm_hln(differences, lag=0, horizon=1)
        reference = stats.ttest_1samp(differences, 0)
        self.assertAlmostEqual(result["DM_HLN"], reference.statistic, places=13)
        self.assertAlmostEqual(result["p_value"], reference.pvalue, places=13)
        scaled = supplement.dm_hln(-5*differences, lag=0, horizon=1)
        self.assertAlmostEqual(scaled["DM_HLN"], -result["DM_HLN"], places=13)
        self.assertAlmostEqual(scaled["p_value"], result["p_value"], places=13)

    def test_constant_dm_has_no_p_value(self):
        result = supplement.dm_hln(np.repeat(.001, 40), lag=14)
        self.assertTrue(np.isnan(result["p_value"]))
        self.assertEqual(result["long_run_variance"], 0)

    def test_bootstrap_uses_mean_of_roots_and_circular_contiguous_blocks(self):
        losses = np.array([[[1., 1., 1., 1.], [9., 9., 9., 9.]]])
        self.assertAlmostEqual(supplement.mean_seed_rmse(losses)[0], 2.)
        self.assertNotAlmostEqual(supplement.mean_seed_rmse(losses)[0], np.sqrt(5))
        scores = supplement.bootstrap_scores(losses, length=2, replicates=20, seed=7)
        np.testing.assert_array_equal(scores, np.full((20, 1), 2.))
        indices = supplement.block_indices(10, 4, 6, np.random.default_rng(7))
        self.assertEqual(indices.shape, (6, 10))
        for row in indices:
            for start in [0, 4, 8]:
                chunk = row[start:min(start+4, 10)]
                np.testing.assert_array_equal(np.diff(chunk) % 10, np.ones(len(chunk)-1))

    def test_global_holm_twelve_keeps_control_and_component_adjustments(self):
        controls = pd.DataFrame([{"model": model, "p_value": p, "log_p_value": np.log(p),
                                   "p_holm": min(1., p*3)}
                                  for model, p in zip(supplement.MODELS, [.01, .03, .2])])
        paired, global_rows, friedman = supplement.seed_tests(synthetic_cube(), controls)
        self.assertEqual((len(paired), len(global_rows)), (9, 12))
        reference = multipletests([r["p_value"] for r in global_rows], method="holm")[1]
        np.testing.assert_allclose([r["p_holm"] for r in global_rows], reference)
        self.assertEqual(global_rows[-3]["p_holm_component"], .03)
        self.assertTrue(all(r["family_size_planned"] == 12 for r in global_rows))
        self.assertEqual(friedman["n_seed_blocks"], 10)

    def test_perfect_persistence_keeps_gain_tests_and_bootstrap_undefined(self):
        controls = pd.DataFrame([{"model": m, "p_value": .5, "log_p_value": np.log(.5), "p_holm": 1.}
                                 for m in supplement.MODELS])
        paired, global_rows, _ = supplement.seed_tests(synthetic_cube(True), controls)
        self.assertTrue(all(np.isnan(r["p_value"]) for r in paired[-3:]))
        self.assertTrue(all(r["family_size_planned"] == 12 for r in global_rows))
        boot, _ = supplement.temporal_bootstrap(synthetic_cube(True), lengths=[7], replicates=20)
        against_p = [r for r in boot if r["model_b"] == "Persistencia"]
        self.assertTrue(all(np.isnan(r["ci95_relative_lower"]) for r in against_p))
        self.assertTrue(all(r["defined_relative_replicates"] == 0 for r in against_p))

    def test_design_sensitivity_ignores_observed_mean_effect(self):
        original, _ = supplement.sensitivity_and_superiority(synthetic_cube())
        shifted, _ = supplement.sensitivity_and_superiority(synthetic_cube(shift_first=.7))
        first = next(r for r in original if r["model_a"] == "MLP" and r["model_b"] == "Persistencia")
        second = next(r for r in shifted if r["model_a"] == "MLP" and r["model_b"] == "Persistencia")
        self.assertAlmostEqual(first["normal_approximation_MDE0"], second["normal_approximation_MDE0"], places=13)
        self.assertAlmostEqual(first["normal_approximation_R_for_delta"], second["normal_approximation_R_for_delta"])

    def test_superiority_exact_boundary_intervals_are_not_certainty(self):
        _, rows = supplement.sensitivity_and_superiority(synthetic_cube())
        winner = next(r for r in rows if r["model_a"] == "MLP" and r["model_b"] == "LSTM")
        self.assertEqual(winner["probability_superiority_half_ties"], 1.)
        self.assertAlmostEqual(winner["ci95_lower"], .025**.1, places=13)
        loser = next(r for r in rows if r["model_a"] == "MLP" and r["model_b"] == "Persistencia")
        self.assertEqual(loser["probability_superiority_half_ties"], 0.)
        self.assertAlmostEqual(loser["ci95_upper"], 1-.025**.1, places=13)

    def test_constant_residual_diagnostics_preserve_fifteen_undefined_tests(self):
        cube = supplement.Cube(pd.date_range("2024-01-01", periods=40, tz="UTC"),
            np.zeros(40), np.ones((4, 10, 40)), pd.DataFrame())
        diagnostics, acf_rows, _ = supplement.residual_diagnostics(cube)
        self.assertEqual(len(diagnostics), 15)
        self.assertTrue(all(np.isnan(row["p_value"]) for row in diagnostics))
        self.assertTrue(all(row["family_size_planned"] == 15 for row in diagnostics))
        self.assertTrue(all(np.isnan(row["acf"]) for row in acf_rows))


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        dates = pd.date_range("2024-01-01", "2024-12-31", tz="UTC")
        target = .02+.003*np.sin(np.arange(366)/8)
        persistence = np.concatenate(([target[0]+.001], target[:-1]))
        frames, runs = [], []
        for vi, variant in enumerate(["absolute_log", "relative_window"]):
            for mi, model in enumerate(supplement.MODELS):
                for si, seed in enumerate(supplement.SEEDS):
                    pred = target+.0001*(1+mi+vi)+.00001*si*np.cos(np.arange(366))
                    frames.append(pd.DataFrame({"variant": variant, "model": model, "seed": seed,
                        "fold_id": dates.strftime("%Y-%m"), "outcome_day": dates.strftime("%Y-%m-%d"),
                        "y_true": target, "y_pred": pred, "persistence_pred": persistence,
                        "prediction_origin_utc": dates.astype(str),
                        "input_last_day": (dates-pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        "target_available_at_utc": (dates+pd.Timedelta(days=1)).astype(str) }))
                    mse = np.mean((pred-target)**2)
                    runs.append({"variant": variant, "model": model, "seed": seed,
                                 "mse": mse, "rmse": np.sqrt(mse), "mae": np.mean(np.abs(pred-target)),
                                 "r2": 1-mse/np.var(target)})
        cls.frame, cls.runs = pd.concat(frames, ignore_index=True), pd.DataFrame(runs)
        cls.predictions_path, cls.runs_path = cls.directory/"predictions.csv", cls.directory/"runs.csv"
        cls.frame.to_csv(cls.predictions_path, index=False)
        cls.runs.to_csv(cls.runs_path, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_valid_complete_leap_year_grid_and_clock(self):
        cube = supplement.load_cube(self.predictions_path, self.runs_path)
        self.assertEqual(cube.predictions.shape, (4, 10, 366))
        self.assertIn(pd.Timestamp("2024-02-29", tz="UTC"), cube.dates)
        np.testing.assert_array_equal(cube.predictions[3, 0], cube.predictions[3, 9])

    def test_missing_path_reserved_year_wrong_month_and_metric_tampering_rejected(self):
        cases = []
        cases.append(self.frame.iloc[:-366].copy())
        future = self.frame.copy(); future.loc[0, "outcome_day"] = "2025-01-01"; cases.append(future)
        month = self.frame.copy(); month.loc[0, "fold_id"] = "2024-02"; cases.append(month)
        clock = self.frame.copy(); clock.loc[0, "input_last_day"] = "2024-01-01"; cases.append(clock)
        for i, frame in enumerate(cases):
            path = self.directory/f"invalid_{i}.csv"; frame.to_csv(path, index=False)
            with self.assertRaises(ValueError): supplement.load_cube(path, self.runs_path)
        runs = self.runs.copy(); runs.loc[runs.variant.eq("relative_window"), "rmse"] *= 2
        path = self.directory/"bad_runs.csv"; runs.to_csv(path, index=False)
        with self.assertRaises(ValueError): supplement.load_cube(self.predictions_path, path)


if __name__ == "__main__":
    unittest.main()
