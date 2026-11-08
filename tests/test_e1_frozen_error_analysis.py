"""Synthetic checks for descriptive diagnostics of a frozen forecast panel."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SOURCE=Path(__file__).resolve().parents[1]/"scripts/build_e1_frozen_error_analysis.py"
spec=importlib.util.spec_from_file_location("frozen_errors",SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def panel():
    days=pd.date_range("2025-01-01",periods=6).strftime("%Y-%m-%d").tolist()
    rows=[]
    observed=np.array([.005,.01,.015,.02,.025,.03])
    for model in m.MODELS:
        for seed in (18101,18102):
            for i,day in enumerate(days):
                rows.append({"model":model,"seed":seed,"outcome_day":day,"y_true":observed[i],
                    "y_pred":observed[i]+(seed-18100)*.001,"persistence_pred":observed[i]*.9,
                    "mase_scale":.002,"study_id":m.STUDY,"execution_id":"closed","source_signature":"source"})
    return pd.DataFrame(rows),days


def thresholds():
    return m.thresholds_from_development(pd.DataFrame({"market_day_utc":pd.date_range("2023-01-01",periods=100),"sigma_7":np.linspace(.001,.04,100)}))


class FrozenErrorTests(unittest.TestCase):
    def test_complete_matched_panel_adds_exactly_one_deterministic_reference(self):
        raw,days=panel();checked=m.validate_panel(raw,[18101,18102],days,"closed","source")
        self.assertEqual(len(checked),42)
        self.assertEqual(len(checked.loc[checked.model.eq("Persistence")]),6)
        self.assertEqual(checked.loc[checked.model.eq("Persistence"),"seed"].unique().tolist(),[0])

    def test_missing_seed_or_date_is_rejected(self):
        raw,days=panel()
        for bad in (raw.iloc[:-1],raw.loc[~(raw.model.eq("LSTM")&raw.seed.eq(18102))]):
            with self.assertRaises(ValueError):m.validate_panel(bad,[18101,18102],days,"closed","source")

    def test_mismatched_observed_values_or_persistence_are_rejected(self):
        raw,days=panel()
        for field in ("y_true","persistence_pred"):
            bad=raw.copy();bad.loc[0,field]+=.001
            with self.assertRaises(ValueError):m.validate_panel(bad,[18101,18102],days,"closed","source")

    def test_duplicate_predictions_and_mixed_execution_are_rejected(self):
        raw,days=panel();bad=raw.copy();bad.loc[0,"execution_id"]="other"
        for candidate in (bad,pd.concat([raw,raw.iloc[[0]]])):
            with self.assertRaises(ValueError):m.validate_panel(candidate,[18101,18102],days,"closed","source")

    def test_development_cutpoints_do_not_accept_test_dates(self):
        dev=pd.DataFrame({"market_day_utc":["2025-01-01","2025-01-02"],"sigma_7":[.01,.03]})
        with self.assertRaises(ValueError):m.thresholds_from_development(dev)

    def test_segment_cutpoint_ties_are_assigned_exactly_once(self):
        t=thresholds();q=t["quantiles"]
        frame=pd.DataFrame({"outcome_day":["2025-01-01"]*4,"y_true":[q["0.25"],q["0.25"]+1e-9,q["0.75"],q["0.75"]+1e-9]})
        self.assertEqual(m.add_segments(frame,t).target_regime.tolist(),["low","middle","middle","high"])

    def test_means_of_individual_rmse_cannot_become_ensemble_score(self):
        one=m.metric_values(np.array([1.,1.]),np.array([0.,0.]),.5)
        two=m.metric_values(np.array([1.,1.]),np.array([2.,2.]),.5)
        mean_individual=(one["rmse"]+two["rmse"])/2
        ensemble=m.metric_values(np.array([1.,1.]),np.array([1.,1.]),.5)["rmse"]
        self.assertEqual(mean_individual,1)
        self.assertEqual(ensemble,0)

    def test_zero_observed_mape_policy_and_residual_sign(self):
        result=m.metric_values([0,1],[1,2],.5)
        self.assertEqual(result["mape_defined_dates"],1)
        self.assertEqual(result["mape"],100)
        self.assertEqual(result["residual_bias"],1)
        self.assertEqual(result["mase"],2)

    def test_segment_metrics_reconcile_and_deterministic_ci_is_undefined(self):
        raw,days=panel();frame=m.add_segments(m.validate_panel(raw,[18101,18102],days,"closed","source"),thresholds())
        runs,summary=m.segment_metrics(frame)
        originals=runs.loc[runs.segment_dimension.eq("overall"),["model","seed",*m.METRICS]]
        self.assertEqual(m.verify_metrics(runs,originals)["status"],"PASS")
        reference=summary.loc[summary.model.eq("Persistence")]
        self.assertTrue(reference.sample_sd.isna().all())
        self.assertTrue(reference.ci95_lower.isna().all())
        modified=originals.copy();modified.loc[modified.index[0],"rmse"]+=1e-5
        with self.assertRaises(ValueError):m.verify_metrics(runs,modified)

    def test_bins_with_too_few_dates_do_not_publish_values(self):
        raw,days=panel();frame=m.add_segments(m.validate_panel(raw,[18101,18102],days,"closed","source"),thresholds())
        result=m.forecast_aggregates(frame,"target_decile",minimum_dates=5)
        self.assertTrue(result.observed_segment_mean.isna().all())
        self.assertTrue(result.forecast_segment_mean.isna().all())

    def test_monthly_means_keep_seed_interval_and_exact_date_support(self):
        raw,days=panel();frame=m.add_segments(m.validate_panel(raw,[18101,18102],days,"closed","source"),thresholds())
        result=m.forecast_aggregates(frame,"calendar_month")
        self.assertTrue(result.n_dates.eq(6).all())
        row=result.loc[result.model.eq("GRU")].iloc[0]
        self.assertAlmostEqual(row.forecast_segment_mean-row.observed_segment_mean,.0015)
        self.assertTrue(np.isfinite(row.forecast_mean_seed_ci95_lower))

    def test_density_grid_uses_all_pairs_and_has_fixed_common_bounds(self):
        raw,days=panel();frame=m.add_segments(m.validate_panel(raw,[18101,18102],days,"closed","source"),thresholds())
        density,support=m.forecast_density(frame,thresholds())
        self.assertEqual(len(density),3600)
        for model in m.ORDER:
            group=density.loc[density.model.eq(model)]
            self.assertEqual(int(group["count"].sum()),12 if model!="Persistence" else 6)
            self.assertAlmostEqual(group.percent_of_model_pairs.sum(),100)
            self.assertEqual(support[model]["dates"],6)

    def test_density_overflow_is_accounted_for_in_boundary_cells(self):
        raw,days=panel();frame=m.validate_panel(raw,[18101,18102],days,"closed","source")
        frame.loc[frame.model.eq("MLP"),"y_pred"]=1
        density,support=m.forecast_density(frame,thresholds())
        self.assertEqual(support["MLP"]["pairs_above_common_grid"],12)
        tail=density.loc[density.model.eq("MLP")&density.forecast_bin.eq(30)]
        self.assertEqual(int(tail["count"].sum()),12)

    def test_descriptive_variance_identity_is_exact_without_anova_claim(self):
        raw,days=panel();frame=m.add_segments(m.validate_panel(raw,[18101,18102],days,"closed","source"),thresholds())
        runs,_=m.segment_metrics(frame);result=m.variance_decomposition(runs)
        for item in result["metrics"].values():
            self.assertAlmostEqual(item["within_model_seed_sum_squares"]+item["between_model_means_sum_squares"],item["total_sum_squares"])
        self.assertIn("not ANOVA",result["scope"])


if __name__=="__main__":unittest.main()
