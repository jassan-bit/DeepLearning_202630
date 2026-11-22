"""Independent checks for strict +1 Monte Carlo inversion in V2.

The reference grids and data fixtures are deterministic software inputs;
these tests are not new formal calibration replicas.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import primary_rmse as production
from test_native_kernel import direct_bartlett_covariance
import test_production_api as fixture_api


def direct_mc_p(value, reference):
    return (np.count_nonzero(reference >= abs(value)) + 1) / (len(reference) + 1)


class FixedBInversionV2Checks(unittest.TestCase):
    def test_critical_indices_match_literal_nominal_alpha(self):
        reference = np.arange(49999, dtype=float)
        for level, alpha, index in [(0.95, 0.05, 47500), (0.99, 0.01, 49500)]:
            value = production.mc_plus_one_critical(reference, level)
            self.assertEqual(value["alpha"], alpha)
            self.assertEqual(value["zero_based_order_index"], index)
            self.assertEqual(value["critical_value"], reference[index])

    def test_strict_plus_one_acceptance_at_critical_and_rejection_above(self):
        for reference in [np.arange(99, dtype=float), np.arange(100, dtype=float), np.repeat(np.arange(50, dtype=float), 2)]:
            value = production.mc_plus_one_critical(reference, 0.95)
            critical = value["critical_value"]
            self.assertGreaterEqual(direct_mc_p(critical, reference), 0.05)
            self.assertLess(direct_mc_p(np.nextafter(critical, np.inf), reference), 0.05)
            self.assertGreaterEqual(direct_mc_p(-critical, reference), 0.05)
            self.assertLess(direct_mc_p(np.nextafter(-critical, -np.inf), reference), 0.05)

    def test_insufficient_reference_resolution_cannot_give_finite_ci(self):
        reference = np.arange(99, dtype=float)
        value = production.mc_plus_one_critical(reference, 0.99)
        critical = value["critical_value"]
        self.assertTrue(critical is None or not math.isfinite(critical))
        self.assertIsNone(production.fixed_b_inverted_ci(1.0, 0.1, 0.9, value))
        self.assertEqual(direct_mc_p(1000, reference), 0.01)

    def test_hln_divides_the_inverted_interval_width(self):
        reference = np.arange(100, dtype=float)
        critical = production.mc_plus_one_critical(reference, 0.95)
        interval = production.fixed_b_inverted_ci(2.0, 0.1, 0.5, critical)
        width = critical["critical_value"] * 0.1 / 0.5
        np.testing.assert_allclose(interval, [2 - width, 2 + width], rtol=0, atol=1e-14)

    def test_operational_intervals_match_direct_delta_hac_and_log_gain(self):
        fixture = fixture_api.ProductionAPIChecks(); fixture.setUp()
        losses, z = fixture.losses, fixture.z
        settings = dict(bootstrap_reps=99, block_length=3, hac_lags=3, instruments=z,
                        reference_reps=999, seed=123, reference_seed=456, bootstrap_backend="native")
        result = production.analyze_losses(losses, ["candidate", "rival"], **settings)
        self.assertEqual(result["protocol_version"], "temporal-fixed-seeds-mean-rmse-v2")
        comparison = result["comparisons"][0]
        self.assertEqual(comparison["interval_method"], "scalar_fixed_b_mc_plus_one_inversion_v2")
        mu = losses.mean(axis=-1)
        rho = np.sqrt(mu)
        theta = rho.mean(axis=1)
        psi = ((losses - mu[..., None]) / (2 * rho[..., None])).mean(axis=1)
        n = losses.shape[-1]
        hln = math.sqrt((n - 1) / n)
        difference = theta[1] - theta[0]
        difference_se = math.sqrt(direct_bartlett_covariance(psi[1] - psi[0], 3)[0, 0] / n)
        logratio = math.log(theta[0] / theta[1])
        log_se = math.sqrt(direct_bartlett_covariance(psi[0] / theta[0] - psi[1] / theta[1], 3)[0, 0] / n)
        reference = production.fixed_b_reference(n, 3, 1, 1, 999, 456)[0]
        for confidence, alpha in [(95, 0.05), (99, 0.01)]:
            index = len(reference) - math.ceil(alpha * (len(reference) + 1)) + 1
            critical = reference[index]
            expected_difference = [difference - critical * difference_se / hln, difference + critical * difference_se / hln]
            expected_log = [logratio - critical * log_se / hln, logratio + critical * log_se / hln]
            expected_gain = [1 - math.exp(expected_log[1]), 1 - math.exp(expected_log[0])]
            np.testing.assert_allclose(comparison[f"ci{confidence}_rmse_difference"], expected_difference, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(comparison[f"ci{confidence}_relative_rmse_gain"], expected_gain, rtol=1e-11, atol=1e-13)

    def test_old_p_values_remain_identical_and_old_intervals_are_diagnostics(self):
        source = ROOT / "versions/mean_rmse_v1/methods/primary_rmse.py"
        spec = importlib.util.spec_from_file_location("immutable_mean_rmse_v1_for_review", source)
        previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)
        fixture = fixture_api.ProductionAPIChecks(); fixture.setUp()
        settings = dict(bootstrap_reps=99, block_length=3, hac_lags=3, instruments=fixture.z,
                        reference_reps=999, seed=123, reference_seed=456, bootstrap_backend="native")
        old = previous.analyze_losses(fixture.losses, ["candidate", "rival"], **settings)
        new = production.analyze_losses(fixture.losses, ["candidate", "rival"], **settings)
        before, after = old["comparisons"][0], new["comparisons"][0]
        for method in production.METHODS:
            for null in ["epa", "practical"]:
                for field in ["statistic", "raw_p", "adjusted_p", "status"]:
                    self.assertEqual(before["methods"][method][null][field], after["methods"][method][null][field])
        for field in [f"ci{confidence}_{kind}" for confidence in [95, 99] for kind in ["rmse_difference", "relative_rmse_gain"]]:
            self.assertEqual(before[field], after["diagnostic_bootstrap_intervals_v1"][field])


if __name__ == "__main__":
    unittest.main(verbosity=2)
