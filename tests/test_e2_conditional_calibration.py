"""Independent formula, causality and degeneracy checks; synthetic data only."""
from datetime import date, timedelta
from pathlib import Path
import math
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e2_conditional_calibration as audit


class ConditionalCalibrationTests(unittest.TestCase):
    def test_original_rank49_and_extended_empirical_quantiles(self):
        x=np.arange(1.,54.)
        q,k=audit.quantile_radius(x,.1)
        self.assertEqual((q,k),(49.,49))
        self.assertEqual(audit.quantile_radius(x,.001),(math.inf,54))
        self.assertEqual(audit.quantile_radius(x,-.1),(math.inf,None))
        self.assertEqual(audit.quantile_radius(x,1.1),(-math.inf,None))
        with self.assertRaises(ValueError):audit.quantile_radius([-1.],.1)

    def test_future_and_current_targets_cannot_change_issued_interval(self):
        dates=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(12)]
        caldates=[(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]
        p=np.linspace(.1,.2,12);y=np.linspace(.2,.1,12)
        cp=np.zeros(53);cy=np.linspace(.001,.03,53)
        for policy in audit.POLICIES:
            original=audit.simulate(p,y,dates,cp,cy,caldates,policy,.03)
            mutated=y.copy();mutated[5:]=100
            changed=audit.simulate(p,mutated,dates,cp,cy,caldates,policy,.03)
            for i in range(6):
                for key in ("radius","alpha_used","lower","upper","history_latest_date"):
                    self.assertEqual(original[i][key],changed[i][key],(policy,i,key))
            if policy!="original_frozen":
                self.assertNotEqual(original[6]["alpha_used"] if policy.startswith("ACI") else original[6]["radius"],
                                    changed[6]["alpha_used"] if policy.startswith("ACI") else changed[6]["radius"])

    def test_ACI_feedback_identity_and_no_silent_alpha_clipping(self):
        dates=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(50)]
        caldates=[(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]
        trace=audit.simulate(np.zeros(50),np.arange(1.,51.),dates,np.zeros(53),np.zeros(53),
                             caldates,"ACI53_gamma005",0.)
        errors=sum(not r["hit"] for r in trace)
        self.assertAlmostEqual(trace[-1]["alpha_next"],.1+.005*(50*.1-errors),places=12)
        self.assertTrue(any(r["state"]=="full" for r in trace))
        self.assertTrue(all(-.005<=r["alpha_used"]<=1.005 for r in trace))

    def test_infinite_and_empty_sets_never_become_small_finite_intervals(self):
        full=audit.interval_metrics(.02,.05,math.inf)
        self.assertTrue(full["hit"]);self.assertEqual(full["width"],math.inf)
        empty=audit.interval_metrics(.02,.02,-math.inf)
        self.assertFalse(empty["hit"]);self.assertTrue(math.isnan(empty["width"]))
        self.assertEqual(audit.average_metric([1.,math.inf]),(None,"infinite_full_prediction_set"))
        self.assertEqual(audit.average_metric([1.,math.nan]),(None,"undefined_empty_prediction_set"))
        finite=audit.interval_metrics(.02,.005,.01)
        self.assertAlmostEqual(finite["interval_score"],.02+20*.005)

    def test_lag_is_previous_calendar_day_not_future_or_last_finite_fill(self):
        dev=pd.DataFrame({"market_day_utc":["2024-12-30","2024-12-31"],"sigma_7":[.01,.02]})
        dates=["2025-01-01","2025-01-02","2025-01-03"]
        np.testing.assert_allclose(audit.full_calendar_lag(dates,[.03,.04,.05],dev),[.02,.03,.04])
        missing=dev.iloc[:1]
        self.assertTrue(np.isnan(audit.full_calendar_lag(dates,[.03,.04,.05],missing)[0]))

    def test_ratio_bootstrap_preserves_calendar_and_reports_undefined_groups(self):
        n=19;den=np.zeros((n,3));num=np.zeros((n,3))
        den[:,0]=1;num[:,0]=np.arange(n)%2
        den[0,1]=1;num[0,1]=1
        a=audit.block_bootstrap_ratios(num,den,draws=10000,block_days=7,seed=8,chunk=79)
        b=audit.block_bootstrap_ratios(num,den,draws=10000,block_days=7,seed=8,chunk=127)
        self.assertEqual(a,b)
        self.assertGreater(a[1]["undefined_bootstrap_draws"],0)
        self.assertEqual(a[2]["CI_status"],"not_estimable")
        # Independent brute-force circular blocks reproduce the coverage CI.
        rng=np.random.default_rng(8);starts=rng.integers(0,n,size=(10000,math.ceil(n/7)))
        idx=((starts[:,:,None]+np.arange(7))%n).reshape(10000,-1)[:,:n]
        expected=np.quantile(num[idx,0].mean(axis=1),[.025,.975])
        np.testing.assert_allclose([a[0]["CI95_low"],a[0]["CI95_high"]],expected)
        with self.assertRaises(ValueError):audit.block_bootstrap_ratios(num,den,draws=9999)

    def test_ratio_HAC_does_not_compress_selected_dates_or_infer_from_zero_failures(self):
        den=np.tile([1.,0.,1.,0.],20);num=den*np.tile([1.,0.,0.,0.,1.,0.,1.,0.],10)
        r=audit.coverage_tests(num,den)
        theta=num.sum()/den.sum();z=num-theta*den;centered=z-z.mean();n=len(z)
        variance=float(centered@centered)/n
        for lag in range(1,15):variance+=2*(1-lag/15)*float(centered[lag:]@centered[:-lag])/n
        expected=(theta-.9)*den.mean()/math.sqrt(variance/n)
        self.assertAlmostEqual(r["z_HAC14"],expected)
        all_covered=audit.coverage_tests(den,den)
        self.assertFalse(all_covered["estimable"])
        self.assertEqual(all_covered["test_status"],"degenerate_variance_no_probability_claim")

    def test_holm_preserves_unestimable_opportunities_and_known_adjustment(self):
        rows=[{"p_raw_for_family":p,"estimable":True} for p in [.001,.03,.04]]
        rows.append({"p_raw_for_family":1.,"estimable":False})
        audit.holm_rows(rows)
        self.assertEqual([r["p_Holm1344"] for r in rows],[.004,.09,.09,None])
        self.assertIsNone(rows[-1]["reject_FWER05"])


if __name__=="__main__":unittest.main()
