"""Formula and inference-scope checks using synthetic values only."""
import copy
from datetime import date, timedelta
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import e2_temporal_final_audit as audit


class TemporalFinalAuditTests(unittest.TestCase):
    def test_global_hac_matches_bartlett_denominator_T_and_batch(self):
        x = np.array([1., 3., -2., 4., 7., 2., 0.])
        z = x-x.mean()
        expected = ((z@z)+2*(2/3)*(z[1:]@z[:-1])+2*(1/3)*(z[2:]@z[:-2]))/7
        self.assertAlmostEqual(audit.hac_long_run_variance(x, 2), expected)
        np.testing.assert_allclose(audit.hac_long_run_variance(np.stack([x, 2*x]), 2),
                                   [expected, 4*expected])

    def test_GR_is_raw_local_sum_not_a_centered_bridge(self):
        x = np.array([1., -2., 3., 0., 5., 2., -1., 4.])
        m = 3
        result = audit.fluctuation_statistics(x, window=m, maxlags=2)
        variance = audit.hac_long_run_variance(x, 2)
        expected = np.array([x[i:i+m].sum() for i in range(len(x)-m+1)])/np.sqrt(m*variance)
        np.testing.assert_allclose(result["GR_path"], expected)
        self.assertAlmostEqual(result["GR_EPA"], np.abs(expected).max())
        # A persistent mean advantage can reject EPA without instability.
        shifted = audit.fluctuation_statistics(x+100, window=m, maxlags=2)
        self.assertGreater(shifted["GR_EPA"], 100)
        np.testing.assert_allclose(shifted["constancy_path"], result["constancy_path"], atol=1e-12)
        self.assertNotAlmostEqual(shifted["GR_EPA"], result["GR_EPA"])

    def test_two_sided_GR_is_sign_and_scale_invariant(self):
        x = np.random.default_rng(172).normal(size=100)
        r = audit.fluctuation_statistics(x, window=30, maxlags=4)
        opposite = audit.fluctuation_statistics(-7*x, window=30, maxlags=4)
        self.assertAlmostEqual(r["GR_EPA"], opposite["GR_EPA"])
        self.assertAlmostEqual(r["relative_mean_constancy"], opposite["relative_mean_constancy"])

    def test_null_bootstrap_is_reproducible_chunk_invariant_and_not_pointwise(self):
        x = np.random.default_rng(8).normal(size=90)
        a = audit.fluctuation_bootstrap(x, window=27, maxlags=3, block_days=5,
                                       draws=199, seed=18, chunk_size=19)
        b = audit.fluctuation_bootstrap(x, window=27, maxlags=3, block_days=5,
                                       draws=199, seed=18, chunk_size=81)
        self.assertEqual(a, b)
        self.assertEqual(len(a["GR_path"]), 64)
        self.assertGreater(a["critical_EPA_95"], 1.96)
        shifted = audit.fluctuation_bootstrap(x+10, window=27, maxlags=3,
                                             block_days=5, draws=199, seed=18)
        self.assertLess(shifted["p_EPA"], a["p_EPA"])
        self.assertAlmostEqual(shifted["p_constancy"], a["p_constancy"])
        self.assertAlmostEqual(shifted["critical_EPA_95"], a["critical_EPA_95"])

    def test_zero_long_run_variance_does_not_fabricate_significance(self):
        identical = audit.fluctuation_bootstrap(np.zeros(50), window=15, maxlags=3, draws=99)
        self.assertEqual(identical["status"], "identical_loss_path")
        self.assertEqual(identical["p_EPA"], 1.)
        constant = audit.fluctuation_bootstrap(np.ones(50), window=15, maxlags=3, draws=99)
        self.assertEqual(constant["status"], "not_estimable_zero_variance")
        self.assertIsNone(constant["p_EPA"])

    def test_clear_break_rejects_centered_constancy_diagnostic(self):
        rng = np.random.default_rng(84)
        x = rng.normal(size=240)+np.r_[np.zeros(120), np.full(120, 6.)]
        result = audit.fluctuation_bootstrap(x, window=72, maxlags=2,
                                            block_days=4, draws=399, seed=981)
        self.assertLess(result["p_constancy"], .05)

    def test_primary_requires_guide_minimum_and_plan_is_immutable(self):
        with self.assertRaises(ValueError):
            audit.primary_refinement({}, {}, np.zeros(10), np.ones(10, dtype=bool), 9999)
        plan = json.loads((Path(__file__).parents[1]/"experiments/e2_temporal_final_audit_20261114/analysis_plan.json").read_text())
        audit.validate_plan(plan)
        for group, key in (("primary", "draws"), ("primary", "block_days"),
                           ("primary", "delta"), ("fluctuation", "window_days")):
            changed = copy.deepcopy(plan)
            changed[group][key] += 1
            with self.assertRaises(ValueError):
                audit.validate_plan(changed)

    def test_saved_input_guard_preserves_true_calendar_and_seed_pairing(self):
        y = np.zeros(608)
        predictors = []
        for model in audit.SYSTEMS:
            for seed in (audit.SEEDS if model in audit.NEURAL else ("deterministic",)):
                predictors.append({"model": model, "seed": seed, "prediction_original": y.tolist()})
        payload = {"y_true": y.tolist(), "mask": [True]*608,
                   "dates": [(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(608)],
                   "predictors": predictors}
        _, values, _ = audit.predictions_by_model(payload)
        self.assertEqual(values["GRU32"].shape, (6, 608))
        self.assertEqual(values["Persistence"].shape, (1, 608))
        payload["mask"][10] = False
        with self.assertRaises(ValueError):
            audit.predictions_by_model(payload)
        payload["mask"][10] = True
        payload["dates"][10] = payload["dates"][11]
        with self.assertRaises(ValueError):
            audit.predictions_by_model(payload)


if __name__ == "__main__":
    unittest.main()
