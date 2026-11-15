"""Analytic and invariance checks; no market outputs or fitted models."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_effects_rq2_closure as closure


class EffectsRQ2Tests(unittest.TestCase):
    def test_dz_uses_raw_paired_gains_and_is_scale_sign_invariant(self):
        gains = np.arange(1., 7.)
        result = closure.standardized_effect(gains, draws=2000, seed=125)
        self.assertAlmostEqual(result["mean_gain_RMSE"], 3.5)
        self.assertAlmostEqual(result["SD_seed_gains"], np.sqrt(3.5))
        self.assertAlmostEqual(result["paired_Cohen_dz"], np.sqrt(3.5))
        scaled = closure.standardized_effect(17*gains, draws=2000, seed=125)
        self.assertAlmostEqual(scaled["paired_Cohen_dz"], result["paired_Cohen_dz"])
        self.assertAlmostEqual(scaled["dz_CI95_lower"], result["dz_CI95_lower"])
        opposite = closure.standardized_effect(-gains, draws=2000, seed=125)
        self.assertAlmostEqual(opposite["dz_CI95_lower"], -result["dz_CI95_upper"])
        self.assertAlmostEqual(opposite["dz_CI95_upper"], -result["dz_CI95_lower"])

    def test_zero_variance_effect_is_not_zero_or_infinite(self):
        result = closure.standardized_effect(np.full(6, .1), draws=100, seed=11)
        self.assertEqual(result["status"], "not_estimable")
        self.assertIsNone(result["paired_Cohen_dz"])
        self.assertIsNone(result["dz_CI95_lower"])
        self.assertEqual(result["invalid_draws"], 100)

    def test_regime_uses_information_available_before_the_target_day(self):
        lag, high = closure.causal_regime([1., 10., 1.], 10., threshold=5.)
        np.testing.assert_array_equal(lag, [10., 1., 10.])
        np.testing.assert_array_equal(high, [True, False, True])
        _, changed = closure.causal_regime([1., 1., 1.], 10., threshold=5.)
        np.testing.assert_array_equal(changed[:2], high[:2])
        self.assertFalse(changed[2])

    def test_regime_statistic_matches_explicit_trajectory_recalculation(self):
        # Opposite signs cancel in an ensemble; their individual RMSE must survive.
        signs = np.array([-1., 1., -1., 1., -1., 1.])
        errors = signs[:, None] * np.arange(1., 7.)[:, None] * np.array([1., 1., 2., 2.])
        reference = np.full((6, 4), 10.)
        squared = np.stack([errors**2, reference**2])
        high = np.array([False, False, True, True])
        point, normal, elevated = closure.regime_statistic(squared, high, np.ones((1, 4)))
        self.assertAlmostEqual(normal[0, 0], 6.5)
        self.assertAlmostEqual(elevated[0, 0], 3.)
        self.assertAlmostEqual(point[0, 0], 3.5)
        weights = np.array([[1., 2., 1., 0.], [0., 2., 1., 3.]])
        batched = closure.regime_statistic(squared, high, weights)[0][:, 0]
        independent = []
        for counts in weights.astype(int):
            indices = np.repeat(np.arange(4), counts)
            sampled_high = high[indices]
            gain = []
            for regime in (False, True):
                chosen = indices[sampled_high == regime]
                rmse_candidate = [np.sqrt(np.mean(errors[s, chosen]**2)) for s in range(6)]
                rmse_reference = [np.sqrt(np.mean(reference[s, chosen]**2)) for s in range(6)]
                gain.append(np.mean(rmse_reference)-np.mean(rmse_candidate))
            independent.append(gain[0]-gain[1])
        np.testing.assert_allclose(batched, independent, rtol=1e-14)
        permuted = closure.regime_statistic(squared[:, ::-1], high, weights)[0]
        np.testing.assert_allclose(permuted[:, 0], batched, rtol=1e-14)

    def test_block_weights_preserve_calendar_length_and_rng_across_chunks(self):
        first = np.concatenate(list(closure.moving_block_counts(35, 7, 19, 17, chunk_size=4)))
        second = np.concatenate(list(closure.moving_block_counts(35, 7, 19, 17, chunk_size=13)))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first.sum(axis=1), np.full(19, 35))
        whole = np.concatenate(list(closure.moving_block_counts(35, 35, 3, 17)))
        np.testing.assert_array_equal(whole, np.ones((3, 35)))

    def test_null_frontier_is_delta_and_holm_family_excludes_sensitivity(self):
        samples = np.array([1., 2., 3., 4., 5., np.nan])
        result = closure.one_sided_summary(3., samples, delta=2., minimum_valid_fraction=.8)
        self.assertEqual(result["p_raw"], .5)
        self.assertEqual(result["valid_draws"], 5)
        self.assertEqual(result["invalid_draws"], 1)
        rows = [{"p_raw": .004}, *[{"p_raw": .9} for _ in range(9)]]
        closure.holm10(rows)
        self.assertEqual(rows[0]["p_Holm10"], .04)
        self.assertTrue(rows[0]["reject_FWER05"])
        self.assertEqual(rows[-1]["p_Holm10"], 1.)

    def test_absent_regime_remains_undefined_not_imputed(self):
        squared = np.ones((2, 6, 4))
        point = closure.regime_statistic(squared, np.zeros(4, dtype=bool), np.ones((1, 4)))[0]
        self.assertTrue(np.isnan(point).all())
        result = closure.one_sided_summary(np.nan, np.full(10, np.nan), delta=2., minimum_valid_fraction=.95)
        self.assertEqual(result["status"], "not_estimable")
        self.assertIsNone(result["p_raw"])
        self.assertEqual(result["invalid_draws"], 10)


if __name__ == "__main__":
    unittest.main()
