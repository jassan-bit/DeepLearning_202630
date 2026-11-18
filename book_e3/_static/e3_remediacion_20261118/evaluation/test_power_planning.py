"""Independent boundary, scaling and coordinate checks for planning arithmetic."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("e3_power_planning", Path(__file__).with_name("power_planning.py"))
power = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(power)


class PlanningContracts(unittest.TestCase):
    def test_null_boundary_has_test_size_not_observed_power(self):
        for pairs in (5, 10, 100):
            self.assertAlmostEqual(power.t_power(pairs, .003, 0), .05 / 48, places=12)

    def test_required_integer_is_minimal_and_respects_policy_minimum(self):
        n = power.required_pairs(.001, .0005, .9)
        self.assertGreaterEqual(power.t_power(n, .001, .0005), .9)
        self.assertLess(power.t_power(n - 1, .001, .0005), .9)
        self.assertEqual(power.required_pairs(.00001, .0005, .9), 10)

    def test_scale_units_cancel_and_stricter_family_needs_no_fewer_pairs(self):
        self.assertEqual(power.required_pairs(.001, .0005, .9), power.required_pairs(100, 50, .9))
        self.assertGreaterEqual(power.required_pairs(.001, .0005, .9, .05 / 48),
                                power.required_pairs(.001, .0005, .9, .05 / 16))

    def test_mde_inverts_noncentral_t_reference(self):
        mde = power.t_mde_separation(10, .001, .9)
        self.assertAlmostEqual(power.t_power(10, .001, mde), .9, places=10)
        self.assertGreater(mde, power.t_mde_separation(20, .001, .9))

    def test_uncertain_variance_and_joint_target_do_not_reduce_requirements(self):
        sd_hi = power.sigma_upper_bound(.001, 10)
        self.assertGreater(sd_hi, .001)
        joint_target = power.marginal_target_for_joint(.9, 48)
        self.assertAlmostEqual(1 - 48 * (1 - joint_target), .9)
        self.assertGreater(power.required_pairs(sd_hi, .0005, joint_target),
                           power.required_pairs(.001, .0005, .9))

    def test_seeds_align_and_deterministic_rival_has_one_real_trajectory(self):
        ids = tuple(range(271801, 271811))
        candidate = np.linspace(.001, .002, 10)
        gains = power.aligned_seed_gains(candidate, [.003], ids, ("deterministic",))
        self.assertEqual(gains.shape, (10,))
        np.testing.assert_allclose(gains, .003 - candidate)
        with self.assertRaises(ValueError):
            power.aligned_seed_gains(candidate, [.003] * 10, ids, ("deterministic",) * 10)
        with self.assertRaises(ValueError):
            power.aligned_seed_gains(candidate, candidate, ids, ids[::-1])

    def test_changing_observed_location_without_variance_does_not_change_design(self):
        ids = tuple(range(271801, 271811))
        candidate = np.linspace(.001, .002, 10)
        rival = np.linspace(.003, .005, 10)
        first = power.aligned_seed_gains(candidate, rival, ids, ids)
        shifted = power.aligned_seed_gains(candidate, rival + .1, ids, ids)
        self.assertAlmostEqual(np.std(first, ddof=1), np.std(shifted, ddof=1), places=14)
        self.assertEqual(power.required_pairs(float(np.std(first, ddof=1)), power.DELTA, .9),
                         power.required_pairs(float(np.std(shifted, ddof=1)), power.DELTA, .9))

    def test_temporal_normal_sensitivity_has_inverse_square_separation(self):
        n = power.normal_calendar_dates(.01, .001, .9)
        self.assertAlmostEqual(power.normal_calendar_dates(.01, .0005, .9), 4 * n)

    def test_invalid_variance_separation_and_coordinates_fail(self):
        for args in ((.001, 0, .9), (0, .001, .9), (.001, .001, 1), (float("nan"), .001, .9)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                power.required_pairs(*args)
        with self.assertRaises(ValueError):
            power.t_power(1, .001, .001)
        self.assertIsNone(power.required_pairs(.1, .000001, .9, maximum=10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
