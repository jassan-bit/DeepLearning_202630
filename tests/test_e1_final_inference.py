"""Known-value contracts for the explicitly post-hoc final-test supplement."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_e1_final_inference as final


def synthetic_cube(n=45):
    dates = pd.date_range("2025-01-01", periods=n, tz="UTC")
    target = .04+.003*np.sin(np.arange(n)/5)
    errors = np.empty((4, 10, n))
    for i in range(3):
        for j in range(10):
            errors[i, j] = .001*(i+1)+.00003*j*(1+.2*np.cos(np.arange(n)/7))
    errors[3] = .0005*np.cos(np.arange(n)/4)
    return final.Cube(dates, target, target[None,None,:]+errors)


def frame_from_cube(cube):
    rows = []
    for i, model in enumerate(final.MODELS):
        for j, seed in enumerate(final.SEEDS):
            rows.append(pd.DataFrame({"model": model, "seed": seed, "history_days": 7,
                "variant": "relative_window", "outcome_day": cube.dates.strftime("%Y-%m-%d"),
                "fold_id": cube.dates.strftime("%Y-%m"), "y_true": cube.target,
                "y_pred": cube.predictions[i,j], "persistence_pred": cube.predictions[3,0],
                "input_first_day": (cube.dates-pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
                "input_last_day": (cube.dates-pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "prediction_origin_utc": cube.dates.astype(str),
                "target_available_at_utc": (cube.dates+pd.Timedelta(days=1)).astype(str)}))
    return pd.concat(rows, ignore_index=True)


class StatisticalContracts(unittest.TestCase):
    def test_paired_mean_sign_interval_effect_and_p_match_reference(self):
        a = np.array([1.,4.,6.,7.,10.])
        b = np.array([2.,3.,3.,8.,6.])
        row, reverse = final.paired_result(a,b), final.paired_result(b,a)
        ref = stats.ttest_1samp(a-b,0)
        self.assertAlmostEqual(row["t_statistic"], ref.statistic, places=13)
        self.assertAlmostEqual(row["p_value"], ref.pvalue, places=13)
        self.assertAlmostEqual(row["d_z"], np.mean(a-b)/np.std(a-b,ddof=1), places=13)
        self.assertAlmostEqual(reverse["ci95_lower"],-row["ci95_upper"],places=13)
        self.assertAlmostEqual(reverse["p_value"],row["p_value"],places=13)

    def test_constant_paired_dispersion_has_no_fabricated_significance(self):
        row = final.paired_result(np.ones(10),np.zeros(10))
        self.assertEqual(row["sd_difference"],0)
        self.assertTrue(np.isnan(row["p_value"]))
        self.assertTrue(np.isnan(row["d_z"]))
        self.assertIn("undefined",row["status"])

    def test_holm_log_adjustment_includes_undefined_hypotheses(self):
        rows = [final.probability(np.log(.01)),final.probability(np.nan),final.probability(np.log(.04))]
        final.holm(rows,"three")
        expected=multipletests([.01,1,.04],method="holm")[1]
        self.assertAlmostEqual(rows[0]["p_holm"],expected[0])
        self.assertTrue(np.isnan(rows[1]["p_holm"]))
        self.assertAlmostEqual(rows[2]["p_holm"],expected[2])
        self.assertTrue(all(r["family_size_planned"]==3 for r in rows))

    def test_extreme_probability_retains_log_and_does_not_print_false_zero(self):
        row=final.probability(-1000.)
        self.assertEqual(row["p_value"],0)
        self.assertTrue(row["p_value_underflow"])
        self.assertIn("exp(",row["p_value_display"])
        self.assertEqual(row["log_p_value"],-1000.)

    def test_dm_h1_zero_hac_matches_t_and_is_scale_equivariant(self):
        d=np.array([1.,2.,3.,-2.,1.,-1.,4.,3.])
        row=final.dm_hln(d,lag=0,horizon=1)
        ref=stats.ttest_1samp(d,0)
        self.assertAlmostEqual(row["DM_HLN"],ref.statistic,places=13)
        self.assertAlmostEqual(row["p_value"],ref.pvalue,places=13)
        reverse=final.dm_hln(-3*d,lag=0)
        self.assertAlmostEqual(reverse["DM_HLN"],-row["DM_HLN"],places=13)

    def test_constant_loss_difference_does_not_have_dm_p_value(self):
        row=final.dm_hln(np.ones(50),lag=14)
        self.assertTrue(np.isnan(row["p_value"]))
        self.assertIn("undefined",row["status"])

    def test_ljung_box_matches_independent_library(self):
        error=np.random.default_rng(2026).normal(size=100)
        error[1:]+=.5*error[:-1]
        row=final.ljung_box_result(error,14)
        ref=acorr_ljungbox(error,lags=[14],model_df=0,return_df=True).iloc[0]
        self.assertAlmostEqual(row["statistic"],ref.lb_stat,places=12)
        self.assertAlmostEqual(row["p_value"],ref.lb_pvalue,places=12)

    def test_chi2_extreme_even_df_tail_matches_exact_finite_polynomial(self):
        q,df=100000.,14
        x=q/2
        log_terms=np.array([k*np.log(x)-special_gammaln(k+1) for k in range(df//2)])
        expected=-x+stats_logsumexp(log_terms)
        self.assertTrue(np.isfinite(final.chi2_logsf(q,df)))
        self.assertAlmostEqual(final.chi2_logsf(q,df),expected,places=9)

    def test_exact_wilcoxon_resolution_and_unsupported_ties(self):
        row=final.wilcoxon_result(np.arange(1.,11),np.zeros(10))
        self.assertAlmostEqual(row["p_value"],2/2**10)
        self.assertAlmostEqual(row["minimum_possible_two_sided_p"],2/2**10)
        for d in [np.ones(10),np.arange(10.)]:
            r=final.wilcoxon_result(d,np.zeros(10))
            self.assertTrue(np.isnan(r["p_value"]))
            self.assertIn("unsupported",r["method"])

    def test_circular_blocks_are_contiguous_with_wrap_and_common_shape(self):
        values=final.circular_indices(10,4,8,np.random.default_rng(7))
        self.assertEqual(values.shape,(8,10))
        for row in values:
            for first in [0,4,8]:
                part=row[first:min(first+4,10)]
                np.testing.assert_array_equal(np.diff(part)%10,np.ones(len(part)-1))

    def test_date_bootstrap_means_seed_roots_not_root_of_mean_loss(self):
        cube=synthetic_cube(20)
        cube.target=np.ones(20)
        for method in range(4):
            for seed in range(10):cube.predictions[method,seed]=1+(1 if seed<5 else 3)
        values=final.bootstrap_date_scores(cube,length=7,replicates=30,seed=13)
        np.testing.assert_allclose(values["rmse"],2.,rtol=0,atol=1e-14)
        self.assertNotAlmostEqual(float(values["rmse"][0,0]),np.sqrt(5))

    def test_family_and_table_sizes_are_fixed_without_extra_hypotheses(self):
        cube=synthetic_cube()
        paired=final.paired_tests(cube,replicates=30)
        temporal=final.temporal_tests(cube)
        residual=final.residual_tests(cube)
        bootstrap=final.date_bootstrap(cube,replicates=20)
        self.assertEqual((len(paired),len(temporal),len(residual),len(bootstrap)),(6,18,9,36))
        self.assertTrue(all(r["family_size_planned"]==6 for r in paired))
        self.assertEqual(sum(r["primary"] for r in temporal),6)
        self.assertTrue(all(np.isnan(r["p_holm"]) for r in temporal if not r["primary"]))
        self.assertEqual(len(final.relative_differences(cube)),24)

    def test_retrospective_mde_uses_dispersion_not_observed_effect(self):
        cube=synthetic_cube(30)
        cube.target=np.ones(30)
        for method in range(4):
            for seed in range(10):cube.predictions[method,seed]=1+.01*(method+1)+seed*.001
        original=final.sensitivity(cube)
        cube.predictions[0]+=.5
        shifted=final.sensitivity(cube)
        for a,b in zip(original,shifted):
            self.assertAlmostEqual(a["normal_approximation_MDE0"],b["normal_approximation_MDE0"],places=12)


class SavedForecastValidation(unittest.TestCase):
    def setUp(self):
        self.cube=synthetic_cube(40)
        self.frame=frame_from_cube(self.cube)

    def test_complete_paths_and_real_datetime_unit_comparison(self):
        actual=final.cube_from_frame(self.frame,self.cube.dates)
        np.testing.assert_allclose(actual.predictions,self.cube.predictions)
        self.assertEqual(actual.predictions.shape,(4,10,40))

    def test_missing_duplicate_wrong_month_and_lookahead_rejected(self):
        cases=[self.frame.iloc[:-1].copy(),pd.concat([self.frame,self.frame.iloc[:1]],ignore_index=True)]
        wrong=self.frame.copy();wrong.loc[0,"fold_id"]="2025-02";cases.append(wrong)
        future=self.frame.copy();future.loc[0,"input_last_day"]="2025-01-01";cases.append(future)
        for bad in cases:
            with self.assertRaises(ValueError):final.cube_from_frame(bad,self.cube.dates)

    def test_sigma_zero_is_legal_but_negative_is_not(self):
        frame=self.frame.copy();frame.loc[frame.outcome_day.eq("2025-01-01"),"y_true"]=0.
        actual=final.cube_from_frame(frame,self.cube.dates)
        self.assertEqual(actual.target[0],0)
        frame.loc[0,"y_true"]=-.01
        with self.assertRaises(ValueError):final.cube_from_frame(frame,self.cube.dates)


def special_gammaln(value):
    from scipy.special import gammaln
    return gammaln(value)


def stats_logsumexp(value):
    from scipy.special import logsumexp
    return logsumexp(value)


if __name__=="__main__":
    unittest.main()
