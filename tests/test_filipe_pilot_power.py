"""Independent numerical references for pilot dimensioning; never fits models."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from analyze_filipe_pilot import (
    PlanningLimitError, minimum_detectable_difference, minimum_repetitions,
    normal_approximation_R, normal_difference_power, upper_difference_sd,
)


class FilipePilotPowerTests(unittest.TestCase):
    def test_power_matches_independent_statsmodels_reference(self):
        reference = TTestPower()
        for n, effect, alpha in [(5, .5, .05/12), (25, .7, .05/12), (80, .35, .025)]:
            actual = normal_difference_power(n, effect*.002, .002, alpha)
            expected = reference.power(effect_size=effect, nobs=n, alpha=alpha, alternative="two-sided")
            self.assertAlmostEqual(actual, expected, places=11)

    def test_null_power_is_alpha_not_a_detection_claim(self):
        self.assertAlmostEqual(normal_difference_power(5, 0., 1., .05/12), .05/12, places=12)

    def test_minimal_integer_R_and_execution_floor(self):
        n = minimum_repetitions(.5, 1., .05/12)
        self.assertGreaterEqual(normal_difference_power(n, .5, 1., .05/12), .8)
        self.assertLess(normal_difference_power(n-1, .5, 1., .05/12), .8)
        self.assertEqual(minimum_repetitions(20., 1., .05/12), 5)

    def test_planning_bound_is_reported_not_silently_capped(self):
        with self.assertRaises(PlanningLimitError):
            minimum_repetitions(.0001, 1., .05/12, maximum=10)

    def test_MDE_inverts_the_same_noncentral_t_reference(self):
        for n in [5, 40, 100]:
            effect = minimum_detectable_difference(n, .002, .05/12)
            self.assertAlmostEqual(normal_difference_power(n, effect, .002, .05/12), .8, places=10)

    def test_SD_upper_bound_uses_chi_square_lower_tail(self):
        upper = upper_difference_sd(.003, 5, .95)
        expected = .003*np.sqrt(4/stats.chi2.ppf(.05, 4))
        self.assertAlmostEqual(upper, expected, places=13)
        self.assertGreater(upper, .003)
        self.assertGreater(upper_difference_sd(.003, 5, .99), upper)

    def test_normal_approximation_is_not_mislabelled_as_exact(self):
        approx = normal_approximation_R(.5, 1., .05/12, .8)
        exact = minimum_repetitions(.5, 1., .05/12)
        self.assertGreaterEqual(exact, approx)

    def test_invalid_and_degenerate_planning_inputs_rejected(self):
        cases = [(1, .5, 1., .05), (5.5, .5, 1., .05), (True, .5, 1., .05),
                 (5, -.5, 1., .05), (5, .5, 0., .05), (5, .5, np.nan, .05),
                 (5, .5, 1., 0.), (5, .5, 1., 1.)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                normal_difference_power(*args)
        with self.assertRaises(ValueError):
            minimum_repetitions(0., 1., .05)
        with self.assertRaises(ValueError):
            upper_difference_sd(0., 5)


if __name__ == "__main__":
    unittest.main()
