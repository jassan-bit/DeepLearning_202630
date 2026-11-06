"""Analytical/synthetic checks for the assumption-bound predictability proxy."""
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2_predictability_reference import InnovationReference, permutation_entropy, predictability_summary


def law():
    dates = [date(2023, 11, 1)+timedelta(days=i) for i in range(54)]
    returns = .01*np.sin(np.arange(54))
    return InnovationReference.from_training_daily(dates, np.exp(4+np.cumsum(returns)))


class PredictabilityTests(unittest.TestCase):
    def test_sigma_matrix_is_exact_population_std_of_six_known_plus_one_return(self):
        ref = law()
        past = np.array([.01, -.02, .03, -.01, .005, .002])
        closes = np.exp(np.r_[4., 4.+np.cumsum(past)])[None, :]
        actual = ref.volatility_matrix(closes)[0]
        future = ref.daily_innovations[np.isfinite(ref.daily_innovations)]
        expected = np.array([np.std(np.r_[past, value], ddof=0) for value in future])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    def test_price_units_do_not_change_the_six_return_reference(self):
        ref = law(); windows = np.exp(4 + .01*np.arange(14).reshape(2, 7))
        np.testing.assert_allclose(ref.volatility_matrix(windows), ref.volatility_matrix(windows*10000), atol=1e-13)

    def test_innovation_law_excludes_later_dates_and_rejects_calendar_compression(self):
        dates = [date(2023, 11, 1)+timedelta(days=i) for i in range(55)]
        with self.assertRaises(ValueError):
            InnovationReference.from_training_daily(dates, np.ones(55))
        with self.assertRaises(ValueError):
            InnovationReference.from_training_daily(dates[:10]+dates[11:54], np.ones(53))
        prior = [date(2022, 11, 1)+timedelta(days=i) for i in range(54)]
        with self.assertRaisesRegex(ValueError, "2023-01-01"):
            InnovationReference.from_training_daily(prior, np.exp(.01*np.arange(54)))

    def test_missing_close_retains_missing_innovations_and_entropy_windows(self):
        dates = [date(2023, 11, 1)+timedelta(days=i) for i in range(54)]
        closes = np.ones(54)*100; closes[30] = np.nan
        ref = InnovationReference.from_training_daily(dates, closes)
        self.assertTrue(np.isnan(ref.daily_innovations[30:32]).all())
        self.assertGreater(ref.specification()["permutation_entropy"]["missing_windows_excluded"], 0)
        self.assertEqual(len(ref.dates), 54)

    def test_constant_future_law_has_no_artificial_positive_conditional_risk(self):
        dates = [date(2023, 11, 1)+timedelta(days=i) for i in range(54)]
        ref = InnovationReference.from_training_daily(dates, np.ones(54)*100)
        windows = np.array([[100., 101., 99., 103., 102., 100., 104.]])
        self.assertLess(ref.predict(windows)["assumption_bound_reference_RMSE"], 1e-12)

    def test_entropy_is_complexity_not_an_invented_accuracy(self):
        monotone = permutation_entropy(np.arange(100.))
        random = permutation_entropy(np.random.default_rng(12).normal(size=1000))
        self.assertAlmostEqual(monotone["normalized_entropy"], 0.)
        self.assertGreater(random["normalized_entropy"], .95)
        self.assertNotIn("forecast_accuracy", random)

    def test_gamma_uses_mean_seed_rmse_not_rmse_of_seed_average(self):
        ref = law(); dates = [date(2024, 1, 1)+timedelta(days=i) for i in range(60)]
        windows = np.exp(4+.01*np.sin(np.arange(60*7).reshape(60, 7)))
        truth = ref.predict(windows)["conditional_mean_sigma7"]
        base = np.stack((truth+.02, truth+.02))
        best = np.stack((truth+.01, truth+.02))
        summary = predictability_summary(ref, dates, windows, truth, base, best,
                    baseline_name="baseline", best_name="best", best_selection_scope="frozen_development_selection",
                    seed_ids=["s1", "s2"])
        self.assertAlmostEqual(summary["best_mean_seed_RMSE"], .015)
        self.assertGreaterEqual(summary["gamma"], 0.)
        self.assertLessEqual(summary["gamma"], 1.)
        self.assertEqual(summary["bootstrap"]["draws"], 2000)
        self.assertFalse(summary["universal_performance_ceiling_established"])

    def test_proxy_better_than_baseline_requirement_is_not_hidden_by_clipping(self):
        ref = law(); dates = [date(2024, 1, 1)+timedelta(days=i) for i in range(60)]
        windows = np.exp(4+.01*np.sin(np.arange(420).reshape(60, 7)))
        truth = ref.predict(windows)["conditional_mean_sigma7"]
        summary = predictability_summary(ref, dates, windows, truth, truth, truth,
                    baseline_name="synthetic perfect", best_name="synthetic perfect",
                    best_selection_scope="observed_test_minimum_descriptive")
        self.assertIsNone(summary["gamma"])
        self.assertIsNone(summary["gamma_ci95_nominal"])
        self.assertTrue(summary["reference_exceeded_by_observed_best"])
        self.assertFalse(summary["gamma_clipped_to_unit_interval"])

    def test_gamma_point_and_interval_use_the_same_small_denominator_threshold(self):
        ref = law(); dates = [date(2024, 1, 1)+timedelta(days=i) for i in range(60)]
        windows = np.exp(4+.01*np.sin(np.arange(420).reshape(60, 7)))
        predicted = ref.predict(windows); truth = predicted["conditional_mean_sigma7"]
        floor = predicted["assumption_bound_reference_RMSE"]
        result = predictability_summary(ref, dates, windows, truth, truth+floor+1e-15, truth+.02,
                    baseline_name="near boundary", best_name="best", best_selection_scope="frozen_development_selection")
        self.assertGreater(result["gamma_point_denominator"], 0.)
        self.assertLess(result["gamma_point_denominator"], result["gamma_minimum_positive_denominator"])
        self.assertIsNone(result["gamma"])
        self.assertIsNone(result["gamma_ci95_nominal"])
        self.assertEqual(result["gamma_point_status"], "not_estimable_unstable_denominator")

    def test_sparse_innovation_calendar_keeps_empty_bootstrap_laws_unestimable(self):
        dates = [date(2023, 1, 1)+timedelta(days=i) for i in range(358)]
        closes = np.full(358, np.nan); closes[:32] = np.exp(4+.01*np.sin(np.arange(32)))
        ref = InnovationReference.from_training_daily(dates, closes)
        future = [date(2024, 1, 1)+timedelta(days=i) for i in range(60)]
        windows = np.exp(4+.01*np.sin(np.arange(420).reshape(60, 7)))
        truth = ref.predict(windows)["conditional_mean_sigma7"]
        with np.errstate(invalid="raise", divide="raise"):
            result = predictability_summary(ref, future, windows, truth, truth+.02, truth+.01,
                    baseline_name="base", best_name="best", best_selection_scope="frozen_development_selection", block_length=28)
        self.assertGreater(result["risk_reference_unestimable_law_draws"], 100)
        self.assertEqual(result["risk_reference_estimable_draws"]+result["risk_reference_unestimable_law_draws"], 2000)
        self.assertIsNone(result["risk_reference_ci95_nominal"])
        self.assertIsNone(result["gamma_ci95_nominal"])
        self.assertEqual(result["risk_reference_interval_status"], "not_estimable_insufficient_valid_innovation_draws")
        self.assertTrue(result["bootstrap"]["unestimable_draws_not_silently_replaced_or_regenerated"])
        result = predictability_summary(ref, future, windows, truth, truth+.02, truth+.01,
                    baseline_name="base", best_name="best", best_selection_scope="frozen_development_selection", block_length=7)
        self.assertGreater(result["risk_reference_unestimable_law_draws"], 0)
        self.assertGreaterEqual(result["risk_reference_estimable_draws"], 1900)
        self.assertIsNotNone(result["risk_reference_ci95_nominal"])
        self.assertEqual(result["risk_reference_interval_status"], "estimated")

    def test_zero_finite_innovation_mass_in_all_bootstrap_laws_does_not_invent_zero_risk(self):
        ref = law(); dates = [date(2024, 1, 1)+timedelta(days=i) for i in range(60)]
        windows = np.exp(4+.01*np.sin(np.arange(420).reshape(60, 7)))
        truth = ref.predict(windows)["conditional_mean_sigma7"]
        import e2_predictability_reference as module
        original = module._block_counts
        def no_valid_law(length, block, repetitions, rng):
            if length != len(ref.dates):return original(length, block, repetitions, rng)
            counts = np.zeros((repetitions, length)); counts[:, 0] = length
            return counts
        with patch.object(module, "_block_counts", side_effect=no_valid_law), np.errstate(invalid="raise", divide="raise"):
            result = predictability_summary(ref, dates, windows, truth, truth+.02, truth+.01,
                    baseline_name="base", best_name="best", best_selection_scope="frozen_development_selection")
        self.assertEqual(result["risk_reference_estimable_draws"], 0)
        self.assertEqual(result["risk_reference_unestimable_law_draws"], 2000)
        self.assertIsNone(result["risk_reference_ci95_nominal"])
        self.assertIsNone(result["gamma_ci95_nominal"])
        self.assertGreater(result["estimated_conditional_risk_RMSE_under_training_law"], 0.)


if __name__ == "__main__":
    unittest.main()
