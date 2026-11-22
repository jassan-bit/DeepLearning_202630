import json
import math
import unittest

import numpy as np

from engine.statistics import analyze, hac_covariance, holm, stationary_indices, _fixed_b_reference, _hac_windows, _bootstrap_studentized, _native_kernel


class TemporalStatisticsTests(unittest.TestCase):
    def fixture(self, sign=1., n=180):
        t = np.arange(n)
        d = sign*.9 + .15*np.sin(t*.73) + .1*np.cos(t*.19)
        q = np.array([np.full(n, 4.), 4.+d])
        predictions = np.sqrt(q[:, None, :] + np.array([0., .1])[None, :, None])
        z = np.column_stack([np.ones(n), np.sin(t*.21)])
        return predictions, np.zeros(n), z

    def run_analysis(self, sign=1., **kwargs):
        p, y, z = self.fixture(sign)
        settings = dict(bootstrap_reps=499, block_length=6, hac_lags=5, instruments=z)
        settings.update(kwargs)
        return analyze(p, y, ["candidate", "rival"], **settings)

    def test_holm_known_and_order(self):
        np.testing.assert_allclose(holm([.04, .001, .03, 1.]), [.09, .004, .09, 1.])
        np.testing.assert_allclose(holm([.01, .01, .03]), [.03, .03, .03])

    def test_hac_matches_bartlett_definition(self):
        x = np.array([1., 2., 4., 3., 0.])
        c = x-x.mean()
        expected = c@c/5 + 2*(2/3)*(c[1:]@c[:-1])/5 + 2*(1/3)*(c[2:]@c[:-2])/5
        self.assertAlmostEqual(hac_covariance(x, 2)[0, 0], expected)

    def test_dm_hln_formula_not_lag_zero(self):
        p, y, z = self.fixture()
        r = self.run_analysis()
        d = ((p[1]-y)**2).mean(0)-((p[0]-y)**2).mean(0)
        expected = d.mean()/math.sqrt(hac_covariance(d, 5)[0, 0]/len(d))*math.sqrt((len(d)-1)/len(d))
        test = r["comparisons"][0]["methods"]["dm_hln_hac"]["epa"]
        self.assertAlmostEqual(test["statistic"], expected)
        ref, _ = _fixed_b_reference(len(d), 5, 1, 1, 49999, 2026112103)
        expected_count = np.count_nonzero(ref >= abs(expected))
        self.assertEqual(test["raw_p"], (expected_count+1)/50000)
        self.assertEqual(test["mc_tail_count"], expected_count)
        self.assertEqual(test["mc_p_rule"], "absolute_tail_plus_one")

    def test_stationary_resampling_reproducible_and_joint(self):
        a = stationary_indices(40, 15, 4, np.random.default_rng(18))
        b = stationary_indices(40, 15, 4, np.random.default_rng(18))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (15, 40))
        self.assertTrue(np.all((a >= 0) & (a < 40)))
        p, y, z = self.fixture()
        r1 = analyze(p, y, ["c", "r"], bootstrap_reps=199, block_length=6, hac_lags=5, instruments=z)
        # Fixed seed replication changes no loss series and no sample size.
        r2 = analyze(np.repeat(p, 3, axis=1), y, ["c", "r"], bootstrap_reps=199, block_length=6, hac_lags=5, instruments=z)
        a, b = r1["comparisons"][0], r2["comparisons"][0]
        np.testing.assert_allclose(a["ci95_mse_difference"], b["ci95_mse_difference"])
        self.assertAlmostEqual(a["methods"]["dm_hln_hac"]["epa"]["raw_p"], b["methods"]["dm_hln_hac"]["epa"]["raw_p"])
        # An exactly deterministic reference is a single trace, irrespective of seed axis.
        self.assertEqual(r2["metrics"]["c"]["unique_prediction_seed_traces"], 2)
        p[1] = p[1, :1]
        deterministic = analyze(p, y, ["c", "deterministic_reference"], bootstrap_reps=199, block_length=6, hac_lags=5, instruments=z)
        self.assertEqual(deterministic["metrics"]["deterministic_reference"]["unique_prediction_seed_traces"], 1)

    def test_historical_rmse_pooled_rmse_and_ensemble_differ(self):
        p = np.tile(np.array([[0., 2.], [1.3, 1.3]])[:, :, None], (1, 1, 60))
        t = np.arange(60)
        r = analyze(p, np.zeros(60), ["c", "r"], bootstrap_reps=99, block_length=3, hac_lags=2,
                    instruments=np.column_stack([np.ones(60), np.sin(t)]))
        self.assertEqual(r["metrics"]["c"]["mean_seed_rmse"], 1.)
        self.assertAlmostEqual(r["metrics"]["c"]["rmse_pooled"], math.sqrt(2))
        self.assertGreater(r["metrics"]["c"]["rmse_pooled"], r["metrics"]["r"]["rmse_pooled"])
        self.assertLess(r["metrics"]["c"]["mean_seed_rmse"], r["metrics"]["r"]["mean_seed_rmse"])

    def test_practical_loss_exactly_corresponds_to_rmse_margin(self):
        r = self.run_analysis()
        c = r["comparisons"][0]
        qm = r["metrics"]["rival"]["mse"]
        qc = r["metrics"]["candidate"]["mse"]
        self.assertAlmostEqual(c["practical_mean"], .98**2*qm-qc)
        self.assertEqual(c["practical_mean"] > 0, c["relative_rmse_gain"] > .02)

    def test_conditional_gw_is_not_duplicate_of_dm_or_directional_evidence(self):
        n = 600
        t = np.arange(n)
        instrument = np.sin(2*np.pi*t/30)
        d = .3*instrument + .05*np.cos(2*np.pi*t/17)
        d -= d.mean()
        p = np.sqrt(np.array([np.full(n, 4.), 4+d]))[:, None, :]
        r = analyze(p, np.zeros(n), ["c", "r"], bootstrap_reps=199, block_length=8, hac_lags=8,
                    instruments=np.column_stack([np.ones(n), instrument]))
        m = r["comparisons"][0]["methods"]
        self.assertGreater(m["dm_hln_hac"]["epa"]["raw_p"], .9)
        self.assertLess(m["gw_conditional"]["epa"]["raw_p"], .001)
        self.assertFalse(r["all_rivals_statistical_gate"])

    def test_missing_or_constant_instrument_not_counted_as_gw(self):
        r = self.run_analysis(instruments=None)
        self.assertEqual(r["comparisons"][0]["methods"]["gw_conditional"]["epa"]["raw_p"], 1.)
        self.assertFalse(r["all_rivals_statistical_gate"])
        with self.assertRaises(ValueError):
            self.run_analysis(instruments=np.ones((180, 2)))

    def test_negative_raw_p_and_inferiority_preserved(self):
        r = self.run_analysis(sign=-1.)
        c = r["comparisons"][0]
        self.assertTrue(c["significant_inferiority"])
        self.assertFalse(c["provisional_statistical_gate"])
        for method in c["methods"].values():
            self.assertEqual(method["epa"]["direction"], "candidate_worse")
        self.assertEqual(c["methods"]["dm_hln_hac"]["epa"]["raw_p"], 1/50000)

    def test_degeneracy_is_conservative_and_json_strict(self):
        p = np.ones((3, 10, 60))
        z = np.column_stack([np.ones(60), np.sin(np.arange(60))])
        r = analyze(p, np.zeros(60), ["c", "a", "b"], bootstrap_reps=99, block_length=4, hac_lags=3, instruments=z)
        self.assertFalse(r["all_rivals_statistical_gate"])
        for c in r["comparisons"]:
            for method in c["methods"].values():
                self.assertEqual(method["epa"]["raw_p"], 1.)
        json.dumps(r, allow_nan=False)

    def test_constant_unequal_losses_do_not_create_infinite_certainty(self):
        p = np.stack([np.ones((10, 60)), np.full((10, 60), 1.1)])
        z = np.column_stack([np.ones(60), np.sin(np.arange(60))])
        r = analyze(p, np.zeros(60), ["c", "r"], bootstrap_reps=99, block_length=4, hac_lags=3, instruments=z)
        c = r["comparisons"][0]
        self.assertGreater(c["mean_mse_difference"], 0)
        self.assertFalse(c["provisional_statistical_gate"])
        for method in c["methods"].values():
            self.assertEqual(method["epa"]["raw_p"], 1.)

    def test_complete_family_and_bootstrap_resolution(self):
        p, y, z = self.fixture()
        p = np.concatenate([p[:1], np.repeat(p[1:], 20, axis=0)])
        r = analyze(p, y, ["c"]+[f"r{i}" for i in range(20)], bootstrap_reps=199, block_length=6, hac_lags=5, instruments=z)
        self.assertEqual(r["family"]["tests"], 120)
        self.assertFalse(r["family"]["bootstrap_resolution_sufficient_at_first_holm_step"])
        self.assertFalse(r["all_rivals_statistical_gate"])

    def test_invariance_to_loss_units(self):
        p, y, z = self.fixture()
        settings = dict(bootstrap_reps=499, block_length=6, hac_lags=5, instruments=z)
        reference = analyze(p, y, ["c", "r"], **settings)["comparisons"][0]
        for scale in [1e-6, 1e-3, 1e3, 1e6]:
            c = analyze(p*scale, y*scale, ["c", "r"], **settings)["comparisons"][0]
            self.assertAlmostEqual(c["relative_rmse_gain"], reference["relative_rmse_gain"])
            self.assertAlmostEqual(c["temporal_effect_cohen_d"], reference["temporal_effect_cohen_d"])
            for method in c["methods"]:
                for null in ["epa", "practical"]:
                    self.assertEqual(c["methods"][method][null]["status"], reference["methods"][method][null]["status"])
                    self.assertAlmostEqual(c["methods"][method][null]["raw_p"], reference["methods"][method][null]["raw_p"])

    def test_moving_window_hac_matrix_matches_lag_definition(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(37, 3))
        for lags in [0, 1, 7, 16]:
            centered = x-x.mean(0)
            expected = centered.T@centered/len(x)
            for lag in range(1, lags+1):
                gamma = centered[lag:].T@centered[:-lag]/len(x)
                expected += (1-lag/(lags+1))*(gamma+gamma.T)
            np.testing.assert_allclose(hac_covariance(x, lags), expected, atol=2e-15)

    def test_studentized_bootstrap_recomputes_replicate_variance(self):
        rng = np.random.default_rng(8)
        x = np.array([4+rng.uniform(-.4,.4,30), 4+rng.uniform(-.4,.4,30)])
        b = _bootstrap_studentized(x, 11, 4, 3, .02, 42)
        idx = stationary_indices(30, 11, 4, np.random.default_rng(42))
        observed = (x[1]-x[0]).mean()
        for k in range(11):
            d = x[1, idx[k]]-x[0, idx[k]]
            expected = (d.mean()-observed)/math.sqrt(hac_covariance(d, 3)[0,0]/30)
            self.assertAlmostEqual(b["epa_t"][k, 0], expected)

    def test_optional_native_acceleration_equals_numpy_at_extreme_scales(self):
        if _native_kernel() is None:
            self.skipTest("optional C++ compiler unavailable; numpy path remains executable")
        rng = np.random.default_rng(39)
        q = rng.uniform(1, 5, size=(5, 70))
        for scale in [1e-12, 1., 1e12]:
            a = _bootstrap_studentized(q*scale, 137, 7, 5, .02, 19, backend="numpy")
            b = _bootstrap_studentized(q*scale, 137, 7, 5, .02, 19, backend="native")
            for key in ["means", "epa_t", "practical_t", "log_ratio_t"]:
                np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-10*scale if key=="means" else 1e-10)
        p, y, z = self.fixture()
        settings = dict(bootstrap_reps=499, block_length=6, hac_lags=5, instruments=z, reference_reps=1999)
        a = analyze(p, y, ["c", "r"], bootstrap_backend="numpy", **settings)
        b = analyze(p, y, ["c", "r"], bootstrap_backend="native", **settings)
        for label in ["ci95_mse_difference", "ci99_mse_difference", "ci95_relative_rmse_gain", "ci99_relative_rmse_gain"]:
            np.testing.assert_allclose(a["comparisons"][0][label], b["comparisons"][0][label], atol=1e-12)
        for null in ["epa", "practical"]:
            aa=a["comparisons"][0]["methods"]["stationary_bootstrap"][null]
            bb=b["comparisons"][0]["methods"]["stationary_bootstrap"][null]
            self.assertEqual(aa["mc_tail_count"],bb["mc_tail_count"])
            self.assertEqual(aa["raw_p"],bb["raw_p"])

    def test_fixed_b_vector_pivot_invariant_under_nonsingular_transform(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(80, 2))
        transform = np.array([[2., .3], [-.4, 1.5]])
        def wald(a):
            return len(a)*a.mean(0)@np.linalg.solve(hac_covariance(a, 8), a.mean(0))
        self.assertAlmostEqual(wald(x), wald(x@transform))

    def test_v3_ci99_is_genuinely_wider_and_tail_metadata_present(self):
        c = self.run_analysis()["comparisons"][0]
        for metric in ["mse_difference", "relative_rmse_gain"]:
            self.assertLessEqual(c[f"ci99_{metric}"][0], c[f"ci95_{metric}"][0])
            self.assertGreaterEqual(c[f"ci99_{metric}"][1], c[f"ci95_{metric}"][1])
        for method in c["methods"].values():
            for test in method.values():
                self.assertEqual(test["raw_p"], (test["mc_tail_count"]+1)/(test["mc_reps"]+1))
                self.assertEqual(test["mc_p_rule"], "absolute_tail_plus_one")

    def test_v4_alpha_changes_decisions_not_statistics_or_intervals(self):
        p, y, z = self.fixture()
        settings=dict(bootstrap_reps=499, block_length=6, hac_lags=5,
                      instruments=z, reference_reps=1999)
        a=analyze(p,y,["c","r"],alpha=.05,**settings)
        b=analyze(p,y,["c","r"],alpha=.025,**settings)
        self.assertEqual(a["comparisons"][0]["methods"],b["comparisons"][0]["methods"])
        for field in ["ci95_mse_difference","ci99_mse_difference","ci95_relative_rmse_gain","ci99_relative_rmse_gain"]:
            self.assertEqual(a["comparisons"][0][field],b["comparisons"][0][field])
        self.assertEqual(b["family"]["alpha"],.025)
        with self.assertRaises(ValueError):analyze(p,y,["c","r"],alpha=.051,**settings)

    def test_gw_only_rejection_does_not_claim_negative_mean_inferiority(self):
        n=600;t=np.arange(n);instrument=np.sin(2*np.pi*t/30)
        d=.3*instrument+.05*np.cos(2*np.pi*t/17);d-=d.mean()+1e-6
        p=np.sqrt(np.array([np.full(n,4.),4+d]))[:,None,:]
        r=analyze(p,np.zeros(n),["c","r"],bootstrap_reps=199,block_length=8,
                  hac_lags=8,instruments=np.column_stack([np.ones(n),instrument]))
        c=r["comparisons"][0]
        self.assertLess(c["mean_mse_difference"],0)
        self.assertLess(c["methods"]["gw_conditional"]["epa"]["adjusted_p"],.025)
        self.assertTrue(c["gw_conditional_rejection_with_negative_sample_mean"])
        self.assertFalse(c["significant_inferiority"])

    def test_invalid_arrays_and_temporal_parameters(self):
        p, y, z = self.fixture()
        with self.assertRaises(ValueError):
            self.run_analysis(hac_lags=0)
        with self.assertRaises(ValueError):
            analyze(p, y[:-1], ["c", "r"], bootstrap_reps=99, block_length=4, hac_lags=2, instruments=z)
        with self.assertRaises(ValueError):
            analyze(p, y, ["same", "same"], bootstrap_reps=99, block_length=4, hac_lags=2, instruments=z)


if __name__ == "__main__":
    unittest.main()
