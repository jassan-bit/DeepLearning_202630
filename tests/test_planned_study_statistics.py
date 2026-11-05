"""Numerical and cohort checks for the registered new study; no model fitting."""
from copy import deepcopy
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from planned_study_statistics import (
    MODELS, PAIRS, PlanningLimitError, analyze_predictions, circular_block_counts,
    detectable_difference, difference_power, minimum_runs, plan_sample_size,
)


def design():
    return {"statistics":{"delta":.00025772842664981106,"alpha":.05,"power":.8,
             "min_runs":10,"max_runs":10000,"sd_upper_simultaneous_confidence":.8,
             "bootstrap_draws":101,"bootstrap_seed":20261105,
             "sd_sensitivity_confidences":[.8,.9,.95]},
            "pilot":{"seeds":list(range(7101,7111))},
            "search":{"seeds":[6101,6102]},"definitive":{"seed_start":8101},
            "baseline_model":"GRU"}


def pilot_metrics():
    rng=np.random.default_rng(9283)
    common=rng.normal(0,.0002,10)
    rows=[]
    for index,model in enumerate(MODELS):
        noise=rng.normal(0,.0003,10)
        for seed,value in zip(range(7101,7111),.005+index*.0004+common+noise):
            rows.append({"model":model,"seed":seed,"rmse":value})
    return pd.DataFrame(rows)


def predictions():
    dates=pd.date_range("2026-09-01",periods=60)
    day=np.arange(60)
    target=.02+.003*np.sin(day/3.7)+.00003*day
    persistence=np.r_[.0192,target[:-1]]
    rows=[]
    for index,model in enumerate(MODELS):
        for seed in range(8101,8111):
            rng=np.random.default_rng(seed+100*index)
            error=.0004*(index+1)+rng.normal(0,.0002,60)*(1+(seed-8101)/20)
            for d,y,p,e in zip(dates,target,persistence,error):
                rows.append({"model":model,"seed":seed,"outcome_day":str(d.date()),
                             "y_true":y,"y_pred":y+e,"persistence_pred":p,
                             "mase_scale":.001})
    return pd.DataFrame(rows)


class PlannedPowerTests(unittest.TestCase):
    def test_noncentral_power_matches_independent_statsmodels(self):
        reference=TTestPower()
        for n,effect,alpha in ((10,.5,.05/3),(25,.7,.05/3),(100,.3,.025)):
            self.assertAlmostEqual(difference_power(n,effect*.002,.002,alpha),
                reference.power(effect_size=effect,nobs=n,alpha=alpha,alternative="two-sided"),places=11)
        self.assertAlmostEqual(difference_power(10,0,1,.05/3),.05/3,places=12)

    def test_minimum_integer_and_mde_invert_power(self):
        n=minimum_runs(.5,1,.05/3)
        self.assertGreaterEqual(difference_power(n,.5,1,.05/3),.8)
        self.assertLess(difference_power(n-1,.5,1,.05/3),.8)
        self.assertAlmostEqual(difference_power(n,detectable_difference(n,1,.05/3),1,.05/3),.8,places=10)
        with self.assertRaises(PlanningLimitError):
            minimum_runs(.00001,1,.05/3,maximum=10)

    def test_plan_uses_paired_dispersion_and_simultaneous_bound(self):
        source=pilot_metrics()
        result=plan_sample_size(source,design())
        panel=source.pivot(index="seed",columns="model",values="rmse")
        for row in result["pairs"]:
            differences=panel[row["model_a"]]-panel[row["model_b"]]
            expected_sd=differences.std(ddof=1)
            expected_upper=expected_sd*np.sqrt(9/stats.chi2.ppf(.2/3,9))
            self.assertAlmostEqual(row["paired_sd"],expected_sd,places=14)
            self.assertAlmostEqual(row["paired_sd_upper"],expected_upper,places=14)
            self.assertGreaterEqual(row["power_at_planned_R_upper_sd"],.8)
            self.assertLessEqual(row["mde_at_planned_R_upper_sd"],result["delta"]*(1+1e-9))
            bounds=[item["paired_sd_upper"] for item in row["sd_confidence_sensitivity"]]
            self.assertEqual(bounds,sorted(bounds))
        self.assertEqual(result["R"],max(r["upper_sd_required_R"] for r in result["pairs"]))
        self.assertEqual(result["seeds"],list(range(8101,8101+result["R"])))
        self.assertFalse(result["temporal_power_claimed"])

    def test_plan_rejects_seed_mismatch_duplicate_degenerate_and_overlap(self):
        frame=pilot_metrics()
        with self.assertRaises(ValueError):
            plan_sample_size(frame.iloc[:-1],design())
        with self.assertRaises(ValueError):
            plan_sample_size(pd.concat([frame,frame.iloc[:1]]),design())
        same=frame.copy()
        for model in MODELS:
            same.loc[same.model.eq(model),"rmse"]=np.arange(10)*.0001+.003
        with self.assertRaises(ValueError):
            plan_sample_size(same,design())
        overlap=design();overlap["definitive"]["seed_start"]=7101
        with self.assertRaises(ValueError):
            plan_sample_size(frame,overlap)

    def test_invalid_planning_inputs_are_rejected(self):
        for arguments in ((True,.1,1,.05),(3.5,.1,1,.05),(10,.1,0,.05),
                          (10,-.1,1,.05),(10,.1,1,0),(10,np.nan,1,.05)):
            with self.subTest(arguments=arguments),self.assertRaises(ValueError):
                difference_power(*arguments)


class PlannedAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design=design()
        cls.plan={"R":10,"seeds":list(range(8101,8111))}
        cls.forecasts=predictions()
        cls.output=analyze_predictions(cls.forecasts,cls.design,cls.plan)

    def test_primary_pairs_match_independent_t_and_holm(self):
        runs=self.output["run_metrics"]
        panel=runs.loc[runs.model.isin(MODELS)].pivot(index="seed",columns="model",values="rmse")
        expected=[]
        for row in self.output["pairwise_rmse"].itertuples():
            reference=stats.ttest_rel(panel[row.model_a],panel[row.model_b])
            expected.append(reference.pvalue)
            self.assertAlmostEqual(row.statistic,reference.statistic,places=10)
            self.assertAlmostEqual(row.p_value_unadjusted,reference.pvalue,places=12)
            self.assertLessEqual(row.ci_family_lower,row.ci95_lower)
            self.assertGreaterEqual(row.ci_family_upper,row.ci95_upper)
        adjusted=multipletests(expected,method="holm")[1]
        np.testing.assert_allclose(self.output["pairwise_rmse"].p_value_holm3,adjusted,rtol=1e-12)

    def test_metric_estimand_is_mean_seed_rmse_not_ensemble(self):
        group=self.forecasts.loc[self.forecasts.model.eq("GRU")]
        values=[]
        for _,part in group.groupby("seed"):
            values.append(np.sqrt(np.mean((part.y_true-part.y_pred)**2)))
        row=self.output["metric_summary"].query("model=='GRU' and metric=='rmse'").iloc[0]
        self.assertAlmostEqual(row["mean"],np.mean(values),places=14)
        ensemble=group.groupby("outcome_day")[["y_true","y_pred"]].mean()
        ensemble_rmse=np.sqrt(np.mean((ensemble.y_true-ensemble.y_pred)**2))
        self.assertGreater(abs(row["mean"]-ensemble_rmse),1e-6)
        self.assertEqual(len(self.output["metric_summary"]),24)
        persistent=self.output["metric_summary"].query("model=='Persistence'")
        self.assertTrue(persistent.sample_sd.isna().all())
        self.assertTrue(persistent.n_total.eq(1).all())

    def test_dm_is_daily_mean_squared_loss_with_six_fixed_pairs(self):
        group=self.forecasts.loc[self.forecasts.model.eq("GRU")]
        squared=(group.y_true-group.y_pred)**2
        row=self.output["dm_pairs"].query("model_a=='GRU' and model_b=='LSTM'").iloc[0]
        self.assertAlmostEqual(row.mean_squared_loss_a,squared.mean(),places=14)
        self.assertEqual(len(self.output["dm_pairs"]),6)
        self.assertTrue(self.output["dm_pairs"].multiplicity_family_size.eq(6).all())
        self.assertTrue(self.output["dm_pairs"].hac_lags.eq(6).all())
        self.assertAlmostEqual(row.hln_correction_factor,np.sqrt(59/60),places=13)
        adjusted=multipletests(self.output["dm_pairs"].p_value,method="holm")[1]
        np.testing.assert_allclose(self.output["dm_pairs"].p_value_holm6,adjusted,rtol=1e-12)

    def test_temporal_bootstrap_preserves_paired_estimand_and_reports_all_blocks(self):
        frame=self.output["temporal_bootstrap"]
        self.assertEqual(len(frame),18)
        self.assertEqual(set(frame.block_length),{3,7,14})
        for row in frame.itertuples():
            source=self.output["pairwise_rmse"]
            matching=source.loc[source.model_a.eq(row.model_a)&source.model_b.eq(row.model_b)]
            if len(matching):
                self.assertAlmostEqual(row.difference_a_minus_b,matching.difference_a_minus_b.iloc[0],places=14)
            self.assertLessEqual(row.ci_family_lower,row.ci95_lower)
            self.assertGreaterEqual(row.ci_family_upper,row.ci95_upper)
            self.assertEqual(row.multiplicity_family_size,6)
            self.assertEqual(row.n_dates,60)
        counts=circular_block_counts(60,7,300,np.random.default_rng(81))
        np.testing.assert_array_equal(counts.sum(axis=1),np.full(300,60))
        self.assertTrue((counts>=0).all())

    def test_missing_date_changed_outcome_and_duplicate_are_rejected(self):
        with self.assertRaises(ValueError):
            analyze_predictions(self.forecasts.iloc[:-1],self.design,self.plan)
        changed=self.forecasts.copy();changed.loc[0,"y_true"]+=.0001
        with self.assertRaises(ValueError):
            analyze_predictions(changed,self.design,self.plan)
        with self.assertRaises(ValueError):
            analyze_predictions(pd.concat([self.forecasts,self.forecasts.iloc[:1]]),self.design,self.plan)
        shifted=self.forecasts.copy()
        shifted["outcome_day"]=(pd.to_datetime(shifted.outcome_day)-pd.DateOffset(years=2)).dt.strftime("%Y-%m-%d")
        with self.assertRaises(ValueError):
            analyze_predictions(shifted,self.design,self.plan)

    def test_all_summary_outputs_exclude_individual_vectors(self):
        forbidden={"y_true","y_pred","prediction","residual","persistence_pred","mase_train_scale","mase_scale"}
        for name,output in self.output.items():
            if isinstance(output,pd.DataFrame):
                self.assertFalse(set(output)&forbidden,msg=name)
        self.assertFalse(self.output["summary"]["selection_using_holdout"])
        self.assertEqual(self.output["summary"]["baseline_selected_before_reserve"],"GRU")


if __name__=="__main__":
    unittest.main()
