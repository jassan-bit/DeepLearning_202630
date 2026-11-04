import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from plan_e1_advanced_margin import paired_sd, normal_difference_power, minimum_repetitions, minimum_detectable_difference


class AdvancedPlanningTests(unittest.TestCase):
    def test_paired_dispersion_uses_covariance(self):
        self.assertAlmostEqual(paired_sd(2.,1.,.5),2.)
        self.assertAlmostEqual(paired_sd(2.,1.,0.),np.sqrt(8.))

    def test_invalid_and_degenerate_assumptions_rejected(self):
        for args in [(0,1,0),(1,-1,0),(1,1,1),(1,1,1.1)]:
            with self.assertRaises(ValueError):paired_sd(*args)

    def test_null_power_equals_alpha(self):
        self.assertAlmostEqual(normal_difference_power(10,0.,1.,.025),.025,places=10)

    def test_large_noncentrality_has_finite_probability(self):
        power=normal_difference_power(10,20.,1.,.025)
        self.assertTrue(np.isfinite(power))
        self.assertGreater(power,.999)

    def test_minimal_integer_repetitions_reaches_target(self):
        repetitions=minimum_repetitions(.5,1.,.025)
        self.assertGreaterEqual(normal_difference_power(repetitions,.5,1.,.025),.8)
        self.assertLess(normal_difference_power(repetitions-1,.5,1.,.025),.8)

    def test_larger_effect_needs_fewer_repetitions(self):
        self.assertGreater(minimum_repetitions(.25,1.,.025),minimum_repetitions(.5,1.,.025))

    def test_mde_inverts_specified_normal_difference_power(self):
        delta=minimum_detectable_difference(10,.002,.025)
        self.assertAlmostEqual(normal_difference_power(10,delta,.002,.025),.8,places=9)


if __name__ == "__main__":unittest.main()
