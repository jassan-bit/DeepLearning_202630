"""Independent production-API checks against direct functional arithmetic."""
from __future__ import annotations

from pathlib import Path
import copy
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import primary_rmse as production
from test_functional_derivatives import weighted_functional


class ProductionAPIChecks(unittest.TestCase):
    def setUp(self):
        t = np.arange(23, dtype=float)
        base = np.column_stack([
            0.03 + (0.4 + 0.3 * np.sin(t / 2)) ** 2,
            0.07 + (1.2 + 0.2 * np.cos(t / 3)) ** 2,
            0.1 + (2.1 + 0.4 * np.sin(t / 5)) ** 2,
        ])
        self.losses = np.stack([
            (0.7 * base + 0.05 * (1 + np.cos(t / 4))[:, None]).T,
            base.T,
        ])
        self.z = np.column_stack([np.ones(len(t)), 1.2 + np.sin(t / 3)])
        self.p = np.full(len(t), 1 / len(t))

    def test_exact_primary_and_temporal_if_match_independent_function(self):
        theta, psi, mu, rho = production.functional_and_influence(self.losses)
        for system in range(2):
            expected = weighted_functional(self.losses[system].T, self.z, self.p)
            self.assertAlmostEqual(theta[system], expected[0], places=14)
            np.testing.assert_allclose(psi[system], expected[1], rtol=1e-13, atol=1e-14)

    def test_gmm_with_estimated_risk_nuisance_matches_independent_function(self):
        candidate = weighted_functional(self.losses[0].T, self.z, self.p)
        rival = weighted_functional(self.losses[1].T, self.z, self.p)
        for factor in [1.0, 0.98]:
            actual = production.conditional_moments(self.losses, self.z, 1, factor)
            np.testing.assert_allclose(actual["moment_mean"], factor * rival[2] - candidate[2], rtol=2e-13, atol=2e-14)
            np.testing.assert_allclose(actual["influence"], factor * rival[4] - candidate[4], rtol=2e-12, atol=2e-14)
            np.testing.assert_allclose(actual["candidate_risk_derivative"], candidate[3], rtol=2e-12, atol=2e-14)
            np.testing.assert_allclose(actual["rival_risk_derivative"], rival[3], rtol=2e-12, atol=2e-14)

    def test_python_numpy_and_native_bootstrap_agree_for_fixed_dates(self):
        n = self.losses.shape[-1]
        indices = np.stack([np.arange(n), np.roll(np.arange(n), 4), np.arange(n)[::-1], (5 * np.arange(n) + 3) % n])
        settings = dict(reps=len(indices), block_length=3, lags=3, margin=0.02, seed=1, indices=indices)
        reference = production.bootstrap_functional(self.losses, backend="numpy", **settings)
        native = production.bootstrap_functional(self.losses, backend="native", **settings)
        for key in ["theta", "epa_t", "practical_t", "log_ratio_t"]:
            np.testing.assert_allclose(native[key], reference[key], rtol=2e-9, atol=2e-11)

    def test_nonregular_seed_risk_does_not_enable_a_comparison(self):
        losses = self.losses.copy()
        losses[0, 0] = 0
        result = production.analyze_losses(losses, ["candidate", "rival"], bootstrap_reps=99, block_length=3, hac_lags=3, instruments=self.z, reference_reps=999)
        self.assertFalse(result["regular_positive_seed_risks"])
        self.assertFalse(result["all_rivals_statistical_gate"])
        for method in result["comparisons"][0]["methods"].values():
            for test in method.values():
                self.assertEqual(test["raw_p"], 1)
                self.assertEqual(test["adjusted_p"], 1)
                self.assertEqual(test["status"], "nonpositive_seed_risk_nonsmooth_functional")


def logical_panel_fixture():
    """Pure rule fixture; its oracle zero p-values are not inference outputs."""
    comparisons = []
    for rival in production.EXPECTED_SYSTEMS[1:]:
        comparisons.append(dict(
            rival=rival, mean_rmse_difference=0.4, practical_mean=0.38,
            temporal_effect_cohen_d=0.5,
            ci95_rmse_difference=[0.2, 0.6], ci99_rmse_difference=[0.1, 0.7],
            ci95_relative_rmse_gain=[0.2, 0.4], ci99_relative_rmse_gain=[0.1, 0.5],
            interval_method="scalar_fixed_b_mc_plus_one_inversion_v2",
            methods={m: {null: dict(status="ok", raw_p=0.0, adjusted_p=0.0) for null in ["epa", "practical"]} for m in production.METHODS},
        ))
    return dict(
        protocol_version=production.PROTOCOL_VERSION,
        candidate=production.EXPECTED_SYSTEMS[0], systems=list(production.EXPECTED_SYSTEMS),
        n_dates=768, n_fixed_seeds=10,
        family=dict(alpha=0.025, methods=list(production.METHODS), nulls=["epa", "practical"], tests=120),
        settings=dict(bootstrap_reps=8191, block_length=32.0, hac_lags=48, horizon=1,
                      margin_relative_mean_seed_RMSE=0.02, alpha=0.025, reference_reps=49999,
                      reference_seed=1, operational_interval_method="scalar_fixed_b_mc_plus_one_inversion_v2"),
        comparisons=comparisons,
    )


class FullRuleContractChecks(unittest.TestCase):
    def test_incomplete_family_is_rejected(self):
        panel = logical_panel_fixture()
        panel["systems"] = panel["systems"][:2]
        panel["comparisons"] = panel["comparisons"][:1]
        panel["family"]["tests"] = 6
        with self.assertRaises(ValueError):
            production.statistical_rule([copy.deepcopy(panel) for _ in range(4)])

    def test_two_of_three_is_not_replaced_by_all_three(self):
        panels = [logical_panel_fixture() for _ in range(4)]
        for comparison in panels[3]["comparisons"]:
            for method in comparison["methods"].values():
                method["practical"]["raw_p"] = 1
                method["practical"]["adjusted_p"] = 1
        result = production.statistical_rule(panels)
        self.assertTrue(result["joint_statistical_rule_pass"])
        self.assertFalse(result["all_contrasts_pass"])
        self.assertTrue(result["full_family_enforced"])

    def test_two_of_three_is_per_rival(self):
        panels = [logical_panel_fixture() for _ in range(4)]
        for rival in range(20):
            comparison = panels[1 + rival % 3]["comparisons"][rival]
            for method in comparison["methods"].values():
                method["practical"]["raw_p"] = 1
                method["practical"]["adjusted_p"] = 1
        self.assertTrue(production.statistical_rule(panels)["joint_statistical_rule_pass"])

    def test_mixed_endpoint_version_is_rejected(self):
        panels = [logical_panel_fixture() for _ in range(4)]
        panels[2]["protocol_version"] = "temporal-fixed-seeds-mse-v4"
        with self.assertRaises(ValueError):
            production.statistical_rule(panels)

    def test_mixed_margin_is_rejected(self):
        panels = [logical_panel_fixture() for _ in range(4)]
        panels[2]["settings"]["margin_relative_mean_seed_RMSE"] = 0.01
        with self.assertRaises(ValueError):
            production.statistical_rule(panels)

    def test_old_bootstrap_intervals_cannot_be_used_as_v2_operational_intervals(self):
        panels = [logical_panel_fixture() for _ in range(4)]
        for comparison in panels[2]["comparisons"]:
            comparison["interval_method"] = "exact_functional_stationary_bootstrap_t_reestimated_roots_and_HAC"
        with self.assertRaises(ValueError):
            production.statistical_rule(panels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
