"""Formula, null-centering, dependence-unit and read-only guards for S8."""
import copy
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import e2_f2a_s8_comparisons as s8


class S8ComparisonsTests(unittest.TestCase):
    def test_daily_loss_preserves_individual_risk_not_ensemble_or_mean_rmse(self):
        y = np.zeros(2)
        predictions = np.array([[0., 2.], [2., 0.]])
        loss = s8.daily_loss(y, predictions)
        np.testing.assert_allclose(loss, [2., 2.])
        self.assertNotEqual(float(loss.mean()), float(np.mean(predictions.mean(axis=0)**2)))
        # Jensen distinction from the benchmark's mean seed RMSE.
        unequal = np.array([[0., 0.], [0., 4.]])
        self.assertNotEqual(float(np.sqrt(s8.daily_loss(y, unequal).mean())),
                            float(np.sqrt(np.mean(unequal**2, axis=1)).mean()))

    def test_hac_bartlett_formula_and_hln_interval_inversion(self):
        a = np.array([1., 4., 2., 6., 8., 3.])
        b = np.array([.5, 2., 1., 2., 4., 1.])
        d = a-b
        z = d-d.mean()
        long_run = (z @ z+2*(2/3)*(z[1:] @ z[:-1])+2*(1/3)*(z[2:] @ z[:-2]))/len(d)
        self.assertAlmostEqual(s8.hac_mean_variance(d, 2), long_run/len(d))
        result = s8.dm_hln(a, b, maxlags=2, horizon=1)
        hln = math.sqrt(5/6)
        expected = hln*d.mean()/math.sqrt(long_run/6)
        self.assertAlmostEqual(result["statistic"], expected)
        self.assertAlmostEqual(result["p_raw"], 2*stats.t.sf(abs(expected), 5))
        radius = stats.t.ppf(.975, 5)*math.sqrt(long_run/6)/hln
        self.assertAlmostEqual(result["CI95_high"], d.mean()+radius)
        self.assertAlmostEqual(result["CI95_low"], d.mean()-radius)

    def test_dm_symmetry_and_degenerate_variance_are_explicit(self):
        a = np.arange(30.)**2
        b = np.arange(30.)
        ab, ba = s8.dm_hln(a, b), s8.dm_hln(b, a)
        self.assertAlmostEqual(ab["statistic"], -ba["statistic"])
        self.assertAlmostEqual(ab["p_raw"], ba["p_raw"])
        same = s8.dm_hln(a, a)
        self.assertEqual(same["p_raw"], 1.)
        self.assertEqual(same["status"], "identical_loss_path")
        constant = s8.dm_hln(np.ones(30), np.zeros(30))
        self.assertIsNone(constant["p_raw"])
        self.assertEqual(constant["status"], "not_estimable_zero_variance")
        with self.assertRaises(ValueError):
            s8.dm_hln(np.arange(10.), np.arange(11.))

    def test_holm_family_and_missing_hypotheses(self):
        adjusted = s8.holm([.01, .04, None, .001])
        np.testing.assert_allclose([adjusted[0], adjusted[1], adjusted[3]], [.03, .08, .004])
        self.assertIsNone(adjusted[2])

    def test_circular_blocks_wrap_preserve_span_and_repeat(self):
        indices = s8.circular_indices(19, 7, 20, 72)
        np.testing.assert_array_equal(indices, s8.circular_indices(19, 7, 20, 72))
        self.assertTrue(np.all(indices < 19))
        for start, stop in [(0, 7), (7, 14), (14, 19)]:
            np.testing.assert_array_equal(np.diff(indices[:, start:stop], axis=1) % 19,
                                          np.ones((20, stop-start-1), dtype=int))
        self.assertFalse(np.array_equal(indices, s8.circular_indices(19, 7, 20, 73)))

    def test_mcs_centered_bootstrap_eliminates_clear_inferior_model(self):
        loss = np.tile([1., 2., 3.], (20, 1))
        sign = np.tile([-1., 1.], 100)
        means = loss.mean(axis=0)
        bootstrap = means+np.column_stack([.05*sign, .1*sign, .15*sign])
        result = s8.mcs_range(loss, ["best", "middle", "worst"], draws=200,
                              block_days=4, bootstrap_means=bootstrap)
        self.assertEqual(result["steps"][0]["candidate_e_R"], "worst")
        self.assertEqual(result["retained"], ["best"])
        self.assertAlmostEqual(result["steps"][0]["p_raw"], 1/201)
        self.assertTrue(result["null_centered"])

    def test_mcs_elimination_is_standardized_eR_not_largest_raw_mean(self):
        loss = np.tile([1., 2., 3.], (20, 1))
        signs_a = np.tile([-1., 1., -1., 1.], 50)
        signs_b = np.tile([-1., -1., 1., 1.], 50)
        bootstrap = loss.mean(axis=0)+np.column_stack([np.zeros(200), .01*signs_a, 100*signs_b])
        result = s8.mcs_range(loss, ["low", "precise_middle", "noisy_high"], draws=200,
                              block_days=4, bootstrap_means=bootstrap)
        self.assertEqual(result["steps"][0]["candidate_e_R"], "precise_middle")
        self.assertNotEqual(result["steps"][0]["candidate_e_R"], "noisy_high")

    def test_mcs_common_additive_path_and_permutation_invariance(self):
        rng = np.random.default_rng(24)
        losses = rng.normal(size=(80, 3))+.1*np.arange(3)
        result = s8.mcs_range(losses, ["a", "b", "c"], draws=199, block_days=5, seed=2)
        perm = [2, 0, 1]
        permuted = s8.mcs_range(losses[:, perm], ["c", "a", "b"], draws=199, block_days=5, seed=2)
        self.assertEqual(set(result["retained"]), set(permuted["retained"]))
        repeated = s8.mcs_range(losses, ["a", "b", "c"], draws=199, block_days=5, seed=2)
        self.assertEqual(result, repeated)
        pvalues = [r["p_monotone"] for r in result["steps"]]
        self.assertEqual(pvalues, sorted(pvalues))
        identical = s8.mcs_range(np.repeat(losses[:, :1], 3, axis=1), ["a", "b", "c"], draws=199, block_days=5)
        self.assertEqual(set(identical["retained"]), {"a", "b", "c"})
        self.assertTrue(all(r["p_raw"] == 1. for r in identical["steps"]))
        unavailable = s8.mcs_range(np.tile([1., 2.], (30, 1)), ["a", "b"], draws=199, block_days=5)
        self.assertEqual(unavailable["status"], "not_estimable_zero_pair_variance")

    def test_mcs_iid_null_smoke_does_not_systematically_delete_equal_models(self):
        # A fixed-seed size smoke check, not proof of finite-sample validity.
        rng = np.random.default_rng(512)
        false_exclusions = 0
        for repetition in range(40):
            equal_risk = rng.normal(size=(100, 3))**2
            result = s8.mcs_range(equal_risk, ["a", "b", "c"], alpha=.1,
                                  draws=399, block_days=1, seed=1000+repetition)
            false_exclusions += len(result["retained"]) < 3
        self.assertLessEqual(false_exclusions, 10)

    def test_seed_rank_friedman_and_small_n_holm_resolution(self):
        rng = np.random.default_rng(75)
        scores = np.tile(np.arange(13.), (6, 1))+.01*rng.normal(size=(6, 13))
        before = scores.copy()
        result, ranks, posthoc = s8.conditional_seed_ranks(scores, list(map(str, range(13))), draws=999, seed=30)
        self.assertAlmostEqual(result["Friedman_Q"], stats.friedmanchisquare(*scores.T).statistic)
        self.assertEqual(result["permutation_p"], 1/1000)
        self.assertTrue(result["posthoc_executed"])
        self.assertEqual(len(posthoc), 78)
        self.assertTrue(all(row["p_Holm78"] == 1. for row in posthoc))
        self.assertFalse(result["population_Friedman_over_independent_series_or_origins"])
        np.testing.assert_array_equal(scores, before)
        again = s8.conditional_seed_ranks(scores, list(map(str, range(13))), draws=999, seed=30)
        self.assertEqual((result, ranks, posthoc), again)
        with self.assertRaises(ValueError):
            s8.conditional_seed_ranks(np.ones((6, 13)), list(map(str, range(13))), draws=99)


if __name__ == "__main__":
    unittest.main()
