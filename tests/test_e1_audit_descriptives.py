"""Known-value checks for the post-evaluation descriptive supplement."""
import itertools
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import analyze_e1_audit_descriptives as descriptive


def saved_frame(y,pred,seed=4101,days=None):
    count=len(y)
    return pd.DataFrame({"history_days":7,"model":"MLP","seed":seed,
                         "outcome_day":days or ["2025-01-01"]*count,
                         "y_true":y,"y_pred":pred,"persistence_pred":pred,
                         "mase_train_scale":2.,"stochastic_training":True})


class DescriptiveChecks(unittest.TestCase):
    def test_metrics_keep_mape_mask_and_original_r2_denominator(self):
        value=descriptive.metrics(saved_frame([0.,2.,4.],[1.,1.,3.]))
        self.assertAlmostEqual(value["rmse"],1.)
        self.assertAlmostEqual(value["mae"],1.)
        self.assertAlmostEqual(value["mase"],.5)
        self.assertAlmostEqual(value["mape"],37.5)
        self.assertAlmostEqual(value["mape_coverage"],2/3)
        self.assertAlmostEqual(value["r2"],.625)

    def test_year_metrics_are_recomputed_within_each_period(self):
        frame=saved_frame([1.,3.,10.,14.],[2.,2.,10.,12.],days=["2025-01-01","2025-01-02","2026-01-01","2026-01-02"])
        runs,_=descriptive.annual_tables(frame)
        year=runs.set_index("period")
        self.assertEqual(set(year.index),{"2025","2026_Jan_Aug"})
        self.assertAlmostEqual(year.loc["2025","r2"],0.)
        self.assertAlmostEqual(year.loc["2026_Jan_Aug","r2"],.5)
        self.assertAlmostEqual(year.loc["2026_Jan_Aug","rmse"],np.sqrt(2.))

    def test_persistence_has_one_path_not_a_pseudo_seed_grid(self):
        a=saved_frame([1.,2.],[1.5,2.5],seed=4101,days=["2025-01-01","2025-01-02"])
        b=a.copy();b["seed"]=4102
        result=descriptive.add_reference(pd.concat([a,b],ignore_index=True))
        ref=result.loc[result.model.eq("Persistencia")]
        self.assertEqual(len(ref),2)
        self.assertEqual(set(ref.seed),{0})
        self.assertFalse(ref.stochastic_training.any())
        _,summary=descriptive.annual_tables(result)
        refsummary=summary.loc[summary.model.eq("Persistencia")]
        self.assertTrue(refsummary.sample_sd.isna().all())
        self.assertTrue(refsummary.ci95_lower.isna().all())

    def test_balanced_variance_identity_has_known_components(self):
        rows=[]
        for m,s,t in itertools.product([-1,1],repeat=3):
            rows.append({"model":m,"seed":s,"month":t,"rmse":20+m+2*s+3*t+m*s})
        result=descriptive.variance_components(pd.DataFrame(rows),["model","seed","month"]).set_index("factor")
        self.assertAlmostEqual(result.total_variance.iloc[0],15.)
        for key,expected in [("model",1.),("seed",4.),("month",9.),("interactions_residual",1.)]:
            self.assertAlmostEqual(result.loc[key,"component_variance"],expected)
        self.assertAlmostEqual(result.share.sum(),1.)

    def test_unbalanced_or_duplicate_variance_cells_rejected(self):
        frame=pd.DataFrame([{"seed":s,"month":m,"rmse":1+s+m} for s,m in itertools.product([1,2],[1,2])])
        with self.assertRaises(ValueError):descriptive.variance_components(frame.iloc[:-1],["seed","month"])
        with self.assertRaises(ValueError):descriptive.variance_components(pd.concat([frame,frame.iloc[:1]]),["seed","month"])

    def test_constant_panel_does_not_invent_variance_shares(self):
        frame=pd.DataFrame([{"seed":s,"month":m,"rmse":2.} for s,m in itertools.product([1,2],[1,2])])
        result=descriptive.variance_components(frame,["seed","month"])
        self.assertTrue(result.share.isna().all())
        self.assertTrue(result.status.eq("constant_panel").all())

    def test_severity_boundaries_are_unambiguous(self):
        self.assertEqual(descriptive.severity_label([1.,2.,3.,4.],[2.,3.]).tolist(),["low","middle","high","high"])

    def test_residual_cells_fewer_than_ten_suppress_values(self):
        frame=saved_frame([.01]*9,[.01]*9)
        _,summary=descriptive.residual_bins(frame,{"q25":.02,"q50":.03,"q75":.04})
        self.assertEqual(summary.n_defined_seed_records.sum(),0)
        self.assertTrue(summary.rmse.isna().all())

    def test_training_bins_exclude_incomplete_d28_windows_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/"results/eda_development";folder.mkdir(parents=True)
            days=pd.date_range("2023-01-01","2024-12-31",tz="UTC")
            prices=100+np.arange(len(days),dtype=float)
            market=pd.DataFrame({"market_day_utc":days,"open":prices,"high":prices+1,"low":prices-1,"close":prices,"volume":1000.})
            market.loc[days==pd.Timestamp("2023-03-24",tz="UTC"),["open","high","low","close","volume"]]=np.nan
            sigma=np.log(market.close).diff().rolling(7,min_periods=7).std(ddof=0)
            target=pd.DataFrame({"market_day_utc":days,"sigma_7":sigma})
            market.to_csv(folder/"daily_market_development.csv",index=False)
            target.to_csv(folder/"daily_volatility_development.csv",index=False)
            result=descriptive.training_thresholds(root)
            self.assertEqual(result["training_nonmissing_targets"],646)
            self.assertEqual(result["common_training_last_outcome_day"],"2024-12-03")
            # Altering the excluded final-development labels must not move bins.
            target.loc[days>pd.Timestamp("2024-12-03",tz="UTC"),"sigma_7"]=100.
            target.to_csv(folder/"daily_volatility_development.csv",index=False)
            second=descriptive.training_thresholds(root)
            self.assertEqual([result[k] for k in ["q25","q50","q75"]],[second[k] for k in ["q25","q50","q75"]])


if __name__=="__main__":unittest.main()
