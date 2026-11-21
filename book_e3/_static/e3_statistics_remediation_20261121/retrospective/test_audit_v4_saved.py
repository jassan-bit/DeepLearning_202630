"""Software guards with synthetic fixtures; not statistical calibration."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import stats
import audit_v4_saved as audit


class SoftwareGuards(unittest.TestCase):
    def test_holm_known_complete_family(self):
        np.testing.assert_allclose(audit.holm([.01,.04,.03,1.]),[.04,.09,.09,1.])

    def test_holm_preserves_invalid_reserved_slot(self):
        with_slot=audit.holm([.02,1.]);without=audit.holm([.02])
        self.assertGreater(with_slot[0],without[0]);self.assertEqual(with_slot[1],1.)

    def test_holm_bad_inputs(self):
        for value in [[],[float('nan')],[-.1],[1.1]]:
            with self.assertRaises(ValueError):audit.holm(value)

    def test_acf_known_and_scale_invariance(self):
        e=np.array([1.,0.,-1.,0.,1.])
        # centered=[.8,-.2,-1.2,-.2,.8]: lag1 numerator.16, denominator2.8.
        self.assertAlmostEqual(audit.acf(e,1),2/35)
        self.assertAlmostEqual(audit.acf(e,2),audit.acf(e*1e-7,2))

    def test_ljung_box_matches_definition(self):
        e=np.random.default_rng(111).normal(size=55);r=audit.ljung_box(e,7)
        centered=e-e.mean();ac=[float(centered[k:]@centered[:-k]/(centered@centered)) for k in range(1,8)]
        q=55*57*sum(v*v/(55-k) for k,v in enumerate(ac,1))
        self.assertAlmostEqual(r['statistic'],q);self.assertAlmostEqual(r['p'],stats.chi2.sf(q,7))

    def test_constant_diagnostics_do_not_manufacture_valid_p(self):
        for result in [audit.ljung_box(np.ones(55),7),audit.arch_lm(np.ones(55),7),audit.mz_hac(np.ones(55),np.ones(55),14)]:
            self.assertIsNone(result['p']);self.assertTrue(result['status'].startswith('not_computed'))

    def test_exact_MZ_fit_not_random_adequacy_test(self):
        p=np.linspace(.01,.04,55);self.assertIsNone(audit.mz_hac(p,p,14)['p'])

    def test_hac_mean_interval_scale_invariance(self):
        e=np.random.default_rng(112).normal(size=55)
        a=audit.mean_error_interval(e,14);b=audit.mean_error_interval(e*1e-7,14)
        for key in ['mean_error','standard_error','lower95','upper95']:
            self.assertAlmostEqual(a[key],b[key]/1e-7,places=11)

    def test_MZ_and_ARCH_scale_invariance(self):
        g=np.random.default_rng(113);p=.02+g.normal(0,.003,55);y=.001+.9*p+g.normal(0,.001,55)
        a=audit.mz_hac(y,p,14);b=audit.mz_hac(y*1e-7,p*1e-7,14)
        self.assertAlmostEqual(a['statistic'],b['statistic'],places=9)
        self.assertAlmostEqual(a['alpha'],b['alpha']/1e-7,places=10)
        self.assertAlmostEqual(a['beta'],b['beta'],places=10)
        self.assertAlmostEqual(audit.arch_lm(y-p,7)['statistic'],audit.arch_lm((y-p)*1e-7,7)['statistic'],places=9)

    def test_short_or_nonfinite_vectors_rejected(self):
        for fn in [lambda:audit.ljung_box([1.]*4,7),lambda:audit.arch_lm([1.]*9,7),lambda:audit.mean_error_interval([1.,float('nan')]*30,7)]:
            with self.assertRaises(ValueError):fn()

    def test_input_corruption_and_unsafe_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.csv';p.write_bytes(b'original')
            m={'files':[{'path':'data.csv','bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}]}
            self.assertEqual(audit.verify_inputs(d,m),1);p.write_bytes(b'changed!')
            with self.assertRaises(ValueError):audit.verify_inputs(d,m)
            m['files'][0]['path']='../data.csv'
            with self.assertRaises(ValueError):audit.verify_inputs(d,m)

    def test_synthetic_references_statsmodels(self):
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import acorr_ljungbox,het_arch
        rng=np.random.default_rng(114)
        for n in [55,128]:
            e=rng.normal(size=n);p=.02+rng.normal(0,.003,n);y=.002+.8*p+.001*e
            for lag in [7,14]:
                q=acorr_ljungbox(e,lags=[lag],return_df=True)
                self.assertAlmostEqual(audit.ljung_box(e,lag)['statistic'],float(q['lb_stat'].iloc[0]),places=10)
                self.assertAlmostEqual(audit.ljung_box(e,lag)['p'],float(q['lb_pvalue'].iloc[0]),places=12)
            h=het_arch(e/max(abs(e)),nlags=7,ddof=0)
            self.assertAlmostEqual(audit.arch_lm(e,7)['statistic'],float(h[0]),places=10)
            self.assertAlmostEqual(audit.arch_lm(e,7)['p'],float(h[1]),places=12)
            X=sm.add_constant((p-p.mean())/p.std())
            fit=sm.OLS(y,X).fit(cov_type='HAC',cov_kwds={'maxlags':14,'use_correction':True})
            test=fit.wald_test((np.eye(2),np.array([p.mean(),p.std()])),use_f=False,scalar=True)
            a=audit.mz_hac(y,p,14)
            self.assertAlmostEqual(a['statistic'],float(test.statistic),places=8)
            self.assertAlmostEqual(a['p'],float(test.pvalue),places=10)

    def _synthetic_grid(self, root):
        from datetime import date,timedelta
        names=['sigma_crps_v4','u_crps_control','mse_only_control','additive_positive_head']
        seeds=list(range(20261121,20261126)); rows=[]
        for name in names+['floor_only','floor_plus_mean_excess']:
            for seed in seeds if name in names else ['analytic']:
                for i in range(55):
                    rows.append({'name':name,'seed':seed,'date':str(date(2024,9,7)+timedelta(days=i)),
                                 'target':.02+i*.00001,'prediction':.02+i*.000009})
        audit.write_csv(Path(root)/'assessment_predictions.csv',rows)
        return rows,{'scope':{'expected_predictions':1210,'expected_trajectories':22,'expected_recipes':6,
                    'assessment_start':'2024-09-07','assessment_end':'2024-10-31','daily_dates':55,
                    'neural_recipes':names,'analytical_controls':['floor_only','floor_plus_mean_excess'],
                    'neural_seeds':seeds}}

    def test_alignment_and_full_grid_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            rows,plan=self._synthetic_grid(d);self.assertEqual(len(audit.load_trajectories(d,plan)),22)
            rows[0]['target']=.03;audit.write_csv(Path(d)/'assessment_predictions.csv',rows)
            with self.assertRaises(ValueError):audit.load_trajectories(d,plan)

    def test_duplicate_calendar_guard(self):
        with tempfile.TemporaryDirectory() as d:
            rows,plan=self._synthetic_grid(d);rows[0]['date']=rows[1]['date'];audit.write_csv(Path(d)/'assessment_predictions.csv',rows)
            with self.assertRaises(ValueError):audit.load_trajectories(d,plan)

    def test_seed_identity_guard(self):
        with tempfile.TemporaryDirectory() as d:
            rows,plan=self._synthetic_grid(d)
            for row in rows[:55]:row['seed']=20261126
            audit.write_csv(Path(d)/'assessment_predictions.csv',rows)
            with self.assertRaises(ValueError):audit.load_trajectories(d,plan)


if __name__=='__main__':unittest.main(verbosity=2)
