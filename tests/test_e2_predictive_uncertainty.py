"""Synthetic scientific guards; no real forecasts, market data or model fits."""
from __future__ import annotations

import copy
import csv
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import numpy as np
import e2_predictive_uncertainty as u


def receipt(model="A"):
    seeds = [str(s) for s in range(660101,660106)]
    return {"model": model, "branch": "secondary_frozen_calibration_A", "primary_fit_replaced": False,
            "fit_origin": "2024-11-01", "fit_outcome_max": "2024-10-24", "scaler_fit_outcome_max": "2024-10-24",
            "configuration_selection_outcome_max": "2024-10-24", "branch_selection_outcome_max": "2024-10-24",
            "refit_during_calibration": False, "calibration_used_for_weight_selection": False, "test_used_for_selection": False,
            "point_predictor_kind": "mean_frozen_seed_members", "ensemble_seeds": seeds,
            "member_checkpoint_sha256": {s:format(i+1,"064x") for i,s in enumerate(seeds)},
            "member_scaler_sha256": {s:format(i+11,"064x") for i,s in enumerate(seeds)},
            "member_config_sha256": {s:format(i+21,"064x") for i,s in enumerate(seeds)},
            "member_run_ids": {s:f"branchA_{model}_{s}" for s in seeds},
            "checkpoint_closed_at_utc": "2026-11-06T12:00:00+00:00",
            "calibration_scoring_started_at_utc": "2026-11-06T12:00:01+00:00",
            "target_scale": "original_unannualized_daily_volatility"}


def calibration_rows(r):
    return [{"model": r["model"], "seed": seed, "outcome_day": d.isoformat(),
             "run_id": r["member_run_ids"][seed], "y_true": 1.+i/1000, "y_pred": 1.,
             "checkpoint_sha256": r["member_checkpoint_sha256"][seed]}
            for seed in r["ensemble_seeds"] for i,d in enumerate(u.calendar(u.CAL_START,u.CAL_END))]


def oof_rows(models=("A","B","C"), seeds=(1,2), *, stage="P0_diagnostic", deterministic=()):
    rows=[]
    for model in models:
        for seed in (["deterministic"] if model in deterministic else seeds):
            for i,d in enumerate(u.calendar(u.DEV_START,u.DEV_END)):
                cutoff=(d.replace(day=1)-timedelta(days=8)).isoformat()
                rows.append({"model": model, "seed": seed, "origin_utc": d.isoformat(), "eligible": True,
                             "y_true": 2.+i/1000, "y_pred": 2.+i/1000+(models.index(model)+1)*.1,
                             "fit_outcome_max": cutoff, "scaler_fit_outcome_max": cutoff,
                             "configuration_selection_outcome_max": "2023-12-24", "source_stage": stage})
    return rows


class PredictiveUncertaintyTests(unittest.TestCase):
    def test_full_primary_fit_cannot_be_calibrated_on_its_training_residuals(self):
        r=receipt();r["fit_outcome_max"]="2024-12-24"
        with self.assertRaisesRegex(ValueError,"2024-10-24"):
            u.validate_calibration_support(r,u.calendar(u.CAL_START,u.CAL_END))

    def test_scaler_configuration_and_branch_selection_respect_the_same_embargo(self):
        for key in ["scaler_fit_outcome_max","configuration_selection_outcome_max","branch_selection_outcome_max"]:
            r=receipt();r[key]="2024-10-31"
            with self.subTest(key=key),self.assertRaises(ValueError):
                u.validate_calibration_support(r,u.calendar(u.CAL_START,u.CAL_END))

    def test_nominal_calendar_does_not_fake_actual_historical_freeze_timestamps(self):
        r=receipt();result=u.validate_calibration_support(r,u.calendar(u.CAL_START,u.CAL_END),u.calendar(u.EVAL_START,u.EVAL_END))
        self.assertEqual(result["first_calibration_label_close_support"],["2024-10-25","2024-11-01"])
        self.assertFalse(result["overlap_with_fit_label_closes"])
        r["checkpoint_closed_at_utc"]=r["calibration_scoring_started_at_utc"]
        with self.assertRaises(ValueError):u.validate_calibration_support(r,u.calendar(u.CAL_START,u.CAL_END))

    def test_calibration_calendar_overlap_and_refit_are_rejected(self):
        dates=u.calendar(u.CAL_START,u.CAL_END)
        with self.assertRaises(ValueError):u.validate_calibration_support(receipt(),dates[:-1])
        with self.assertRaises(ValueError):u.validate_calibration_support(receipt(),dates,dates)
        r=receipt();r["refit_during_calibration"]=True
        with self.assertRaises(ValueError):u.validate_calibration_support(r,dates)

    def test_weighted_quantile_matches_unweighted_finite_order_statistic(self):
        # ceil((9+1)*.8)=8: the eighth order statistic, not interpolated percentile.
        self.assertEqual(u.weighted_split_quantile(np.arange(1,10),np.ones(9),.2),8.)
        self.assertEqual(u.weighted_split_quantile([1,2,3],[1,1,3],.5),3.)

    def test_zero_cases_invalid_levels_and_insufficient_mass_do_not_become_zero_radius(self):
        for scores,weights,alpha in [([],[],.1),([1],[1],.05),([1,2],[1,0],.5),([1,2],[1,1],0),([1,2],[1,1],1),([np.nan,2],[1,1],.5)]:
            with self.subTest(scores=scores,alpha=alpha),self.assertRaises(ValueError):
                u.weighted_split_quantile(scores,weights,alpha)

    def test_calibrator_uses_its_own_same_frozen_runs_not_OOF_substitutes(self):
        r=receipt();rows=calibration_rows(r);result=u.fit_temporal_calibrator(rows,r)
        self.assertEqual(result["calibration_origins"],60)
        self.assertFalse(result["exact_finite_sample_coverage_claimed"])
        q=result["radii_by_alpha"];self.assertLessEqual(q["0.2"],q["0.1"]);self.assertLessEqual(q["0.1"],q["0.05"])
        rows[0]["run_id"]="OOF_other_fit"
        with self.assertRaisesRegex(ValueError,"OOF"):u.fit_temporal_calibrator(rows,r)

    def test_calibrator_rejects_duplicate_missing_and_conflicting_seed_truth(self):
        r=receipt();rows=calibration_rows(r)
        for changed in [rows[:-1],rows+[rows[0]],copy.deepcopy(rows)]:
            if len(changed)==len(rows):changed[-1]["y_true"]+=1
            with self.assertRaises(ValueError):u.fit_temporal_calibrator(changed,r)

    def test_scaler_and_configuration_are_part_of_the_evaluated_predictor_identity(self):
        r=receipt();first=u.fit_temporal_calibrator(calibration_rows(r),r)
        changed=copy.deepcopy(r);changed["member_scaler_sha256"]["660101"]="a"*64
        second=u.fit_temporal_calibrator(calibration_rows(changed),changed)
        self.assertNotEqual(first["predictor_signature"],second["predictor_signature"])
        rows=calibration_rows(r);rows[0]["scaler_sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"scaler/configuration"):
            u.fit_temporal_calibrator(rows,r)

    def test_interval_scores_pinball_and_WIS_have_analytic_values(self):
        result=u.interval_scores([3.],[1.],{.2:([0.],[2.])})
        self.assertEqual(result["0.2"]["coverage"][0],0)
        self.assertEqual(result["0.2"]["width"][0],2)
        self.assertEqual(result["0.2"]["interval_score"][0],12)
        self.assertAlmostEqual(result["0.2"]["pinball_lower"][0],.3)
        self.assertAlmostEqual(result["0.2"]["pinball_upper"][0],.9)
        self.assertAlmostEqual(result["WIS"][0],(1+1.2)/1.5)

    def test_invalid_non_nested_bounds_and_nonfinite_radius_are_rejected(self):
        for bounds in [{.1:([2.],[1.])},{.2:([0.],[2.]),.1:([.5],[1.5])},{.1:([0.],[2.]),"0.1":([0.],[2.])}]:
            with self.assertRaises(ValueError):u.interval_scores([1.],[1.],bounds)
        for radius in [-1,np.inf,np.nan]:
            with self.assertRaises(ValueError):u.symmetric_intervals([1.],radius)

    def test_CRPS_requires_an_explicit_distribution_and_has_exact_two_atom_value(self):
        self.assertAlmostEqual(u.empirical_crps([1.],[[0.,2.]])[0],.5)
        self.assertAlmostEqual(u.empirical_crps([1.],[[0.,2.]],weights=[.25,.75])[0],.625)
        np.testing.assert_array_equal(u.signed_residual_distribution([.5],[-1,0,1]),[[0,.5,1.5]])
        with self.assertRaises(ValueError):u.empirical_crps([1.],[[1.]])

    def test_empty_conditional_group_is_not_zero_coverage(self):
        dates=u.calendar(u.EVAL_START,u.EVAL_END)
        result=u.temporal_mean_summary(np.ones(60),dates,eligible_mask=np.zeros(60,bool))
        self.assertIsNone(result["mean"]);self.assertIsNone(result["ci95_nominal"])
        self.assertEqual(result["status"],"not_estimable")

    def test_interval_evaluation_keeps_predictor_identity_primary_fit_and_empty_bins_explicit(self):
        dates=u.calendar(u.EVAL_START,u.EVAL_END)
        c={"model":"A","predictor_signature":"frozen-A","radii_by_alpha":{"0.1":.3}}
        with self.assertRaisesRegex(ValueError,"predictor differs"):
            u.evaluate_interval_forecasts(np.ones(60),np.ones(60),dates,c,evaluation_predictor_signature="primary-other-fit")
        result=u.evaluate_interval_forecasts(np.ones(60),np.ones(60),dates,c,
            evaluation_predictor_signature="frozen-A",primary_comparable_point=np.full(60,1.5),
            forecast_bin_edges=[.5,1.5],subgroup_receipt={"threshold_source":"training_only","threshold_outcome_max":"2024-10-24"})
        self.assertFalse(result["primary_fit_replaced"])
        self.assertAlmostEqual(result["point_performance_side_by_side"]["primary_comparable_fixed_ensemble_RMSE"],.5)
        self.assertEqual(result["CRPS"]["status"],"not_computed")
        self.assertIsNone(result["conditional_coverage"]["forecast_range_train_thresholds"]["0"]["0.1"]["mean"])
        self.assertEqual(result["conditional_coverage"]["forecast_range_train_thresholds"]["1"]["0.1"]["mean"],1.)

    def test_temporal_resampling_rejects_calendar_compression_and_too_few_draws(self):
        dates=u.calendar(u.EVAL_START,u.EVAL_END);dates.pop(10)
        with self.assertRaises(ValueError):u.temporal_mean_summary(np.ones(59),dates)
        with self.assertRaises(ValueError):u.temporal_mean_summary(np.ones(60),u.calendar(u.EVAL_START,u.EVAL_END),repetitions=1999)
        idx=u.block_indices(60,14,np.random.default_rng(2))
        for i in range(0,42,14):np.testing.assert_array_equal(np.diff(idx[i:i+14])%60,np.ones(13))

    def test_training_taxonomy_cannot_use_the_analyzed_2024_outcomes(self):
        dates=u.calendar("2023-01-08","2023-02-01")
        taxonomy=u.fit_error_taxonomy(np.linspace(0,1,len(dates)),dates,reference_name="fixed training persistence")
        self.assertFalse(taxonomy["primary_RMSE_delta_reused_as_case_threshold"])
        dates[-1]=date(2024,1,1)
        with self.assertRaises(ValueError):u.fit_error_taxonomy(np.ones(len(dates)),dates,reference_name="invalid")

    def test_MECE_ties_zero_and_mixed_seed_signs_do_not_cancel_away(self):
        t={"q50_abs_error":1.,"q90_abs_error":2.}
        expected={"low_reference_error":[0,0],"moderate_under":[-1.5,-1.5],"moderate_over":[1.5,1.5],
                  "moderate_mixed":[-1.5,1.5],"large_under":[-3,-3],"large_over":[3,3],"large_mixed":[-3,3]}
        for category,error in expected.items():self.assertEqual(u.category_for_case(error,t),category)
        self.assertEqual(u.category_for_case([1,1],t),"low_reference_error")
        self.assertEqual(u.category_for_case([2,2],t),"moderate_over")

    def test_unique_stratified_cases_are_reproducible_and_carry_weights(self):
        dates=u.calendar(u.DEV_START,u.DEV_END);cats=[u.CATEGORIES[i%7] for i in range(305)]
        a=u.stratified_case_sample(cats,dates);b=u.stratified_case_sample(cats,dates)
        self.assertEqual(a,b);self.assertEqual(len({r["origin_utc"] for r in a}),100)
        self.assertTrue(all(r["inverse_inclusion_weight"]>=1 for r in a))
        with self.assertRaises(ValueError):u.stratified_case_sample(cats[:99],dates[:99])

    def test_dense_OOF_cannot_replace_dates_by_more_seed_rows(self):
        rows=oof_rows();panel=u.validate_oof_panel(rows,["A","B","C"],[1,2],source_stage="P0_diagnostic")
        self.assertEqual(panel["unique_eligible_origins_per_model"],305)
        with self.assertRaises(ValueError):u.validate_oof_panel(rows[:-1],["A","B","C"],[1,2],source_stage="P0_diagnostic")
        for row in rows:
            if u._origin(row)>=date(2024,4,9):row.update(eligible=False,y_true="",y_pred="")
        with self.assertRaisesRegex(ValueError,"100 unique"):u.validate_oof_panel(rows,["A","B","C"],[1,2],source_stage="P0_diagnostic")

    def test_OOF_future_scaler_or_configuration_is_rejected(self):
        for field in ["fit_outcome_max","scaler_fit_outcome_max","configuration_selection_outcome_max"]:
            rows=oof_rows();rows[0][field]="2023-12-25"
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"embargo"):
                u.validate_oof_panel(rows,["A","B","C"],[1,2],source_stage="P0_diagnostic")

    def test_daily_refit_cannot_silently_replace_inherited_monthly_OOF_fits(self):
        rows=oof_rows();rows[30]["fit_outcome_max"]="2024-01-23"
        with self.assertRaisesRegex(ValueError,"monthly"):
            u.validate_oof_panel(rows,["A","B","C"],[1,2],source_stage="P0_diagnostic")
        rows=oof_rows();rows[0]["run_id"]="first";rows[1]["run_id"]="new_fit_same_month"
        with self.assertRaisesRegex(ValueError,"monthly"):
            u.validate_oof_panel(rows,["A","B","C"],[1,2],source_stage="P0_diagnostic")

    def test_E1_ETTh1_and_P0_cannot_be_silently_declared_final_Top3(self):
        with self.assertRaises(ValueError):u.validate_oof_panel(oof_rows(),["A","B","C"],[1,2],source_stage="E1")
        panel=u.validate_oof_panel(oof_rows(),["A","B","C"],[1,2],source_stage="P0_diagnostic")
        with self.assertRaises(ValueError):u.select_top3_from_development(panel,expected_candidate_models=["A","B","C"])

    def test_branch_A_selection_ignores_Oct25_31_without_changing_primary_calendar(self):
        models=["A","B","C","D"];rows=oof_rows(models=models,stage="E2_nested_development")
        for row in rows:
            if row["model"]=="A" and u._origin(row)>u.FIT_MAX:row["y_pred"]=row["y_true"]+10
        panel=u.validate_oof_panel(rows,models,[1,2],source_stage="E2_nested_development")
        selection=u.select_top3_from_development(panel,expected_candidate_models=models)
        self.assertEqual(selection["selected_models"],["A","B","C"])
        self.assertFalse(selection["primary_JanOct31_ranking_changed"]);self.assertEqual(len(panel["dates"]),305)
        with self.assertRaises(ValueError):u.select_top3_from_development(panel,expected_candidate_models=models[:-1])

    def test_deterministic_reference_is_stored_once_not_counted_as_randomized_fits(self):
        rows=oof_rows(models=("A","Persistence"),deterministic=("Persistence",))
        panel=u.validate_oof_panel(rows,["A","Persistence"],[1,2],source_stage="P0_diagnostic",deterministic_models=["Persistence"])
        self.assertEqual(len(rows),305*3)
        np.testing.assert_array_equal(panel["predictions"][1,0],panel["predictions"][1,1])

    def test_full_error_analysis_counts_origins_and_shared_failures_not_seed_rows(self):
        rows=oof_rows(models=("A","B"))
        for row in rows:
            i=(u._origin(row)-u.DEV_START).days;group=i%7
            error=[.1,-.3,.3,(-.3 if row["seed"]==1 else .3),-.6,.6,(-.6 if row["seed"]==1 else .6)][group]
            row["y_pred"]=row["y_true"]+error
        panel=u.validate_oof_panel(rows,["A","B"],[1,2],source_stage="P0_diagnostic")
        taxonomy={"q50_abs_error":.2,"q90_abs_error":.4,"threshold_source":"training_reference_absolute_errors_only","training_outcome_max":"2023-12-24"}
        result=u.analyze_oof_errors(panel,taxonomy)
        for model in ["A","B"]:
            self.assertEqual(sum(r["cases"] for r in result["models"][model]["categories"]),305)
            self.assertAlmostEqual(sum(r["frequency"] for r in result["models"][model]["categories"]),1.)
            self.assertEqual(len(result["models"][model]["private_stratified_audit_sample"]),100)
        self.assertEqual(result["pairwise_error_overlap"][0]["large_error_jaccard"],1.)
        self.assertAlmostEqual(result["complementarity_diagnostic"]["observed_RMSE_gain"],0.)
        self.assertFalse(result["final_top3_identified"])
        public=u.public_error_summary(result)
        self.assertNotIn("private_",json.dumps(public))
        self.assertEqual(public["models"]["A"]["unique_eligible_cases"],305)

    def test_default_command_is_plan_only_and_runs_without_numeric_site_packages(self):
        result=subprocess.run([sys.executable,"-S",str(ROOT/"scripts/e2_predictive_uncertainty.py")],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);output=json.loads(result.stdout)
        self.assertEqual(output["model_fits"],0);self.assertEqual(output["actual_error_cases_analyzed_by_this_command"],0)

    def test_explicit_private_diagnostic_CLI_is_reproducible_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="e2-uncertainty-fixture-") as directory:
            root=Path(directory);rows=oof_rows(models=("A","B"))
            for row in rows:
                row["run_id"]=f"{row['model']}_{row['seed']}_{row['origin_utc'][:7]}"
                row["checkpoint_sha256"]=sha256(row["run_id"].encode()).hexdigest()
                row["checkpoint_closed_at_utc"]="2026-11-06T12:00:00+00:00"
                row["scoring_started_at_utc"]="2026-11-06T12:00:01+00:00"
            path=root/"synthetic_OOF.csv"
            with path.open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            taxonomy=u.fit_error_taxonomy(np.linspace(0,.5,25),u.calendar("2023-01-08","2023-02-01"),reference_name="synthetic training reference")
            thresholds=root/"taxonomy.json";thresholds.write_text(json.dumps(taxonomy))
            command=[sys.executable,str(ROOT/"scripts/e2_predictive_uncertainty.py"),"--oof-csv",str(path),"--taxonomy-json",str(thresholds),
                     "--models","A","B","--seeds","1","2","--private-output",str(root/"output")]
            process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            summary=json.loads((root/"output/error_summary.json").read_text())
            self.assertNotIn("private_",json.dumps(summary));self.assertFalse(summary["final_top3_identified"])
            with (root/"output/private_audit_cases.csv").open() as stream:sample=list(csv.DictReader(stream))
            self.assertEqual(len(sample),200);self.assertEqual(len({r["origin_utc"] for r in sample if r["model"]=="A"}),100)
            again=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(again.returncode,1);self.assertIn("new empty private",again.stderr)


if __name__ == "__main__":
    unittest.main()
