"""Synthetic fixtures: no actual model outputs or market source files are read."""
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_e2_remediation_pilot import (DELTA, MODELS, SEEDS, THRESHOLD,
    analyse_panel, block_indices, bootstrap_panel, calendar_for, centered_upper_p,
    conditional_power, holm_three, interval, mean_seed_rmse, region_gap,
    required_seeds, seed_power_report, validate_panel)


def fixture_rows(phase='pilot'):
    rows=[]
    for model in MODELS:
        for seed in SEEDS:
            for k,day in enumerate(calendar_for(phase)):
                origin='2024-11-01' if phase=='pilot' else day[:7]+'-01'
                cutoff=(date.fromisoformat(origin)-timedelta(days=8)).isoformat()
                rows.append({'run_id':f'{phase}_{origin}_{model}_s{seed}','model':model,'seed':str(seed),
                    'outcome_day':day,'y_true':'0.03','y_pred':str(.03+.001*(1+MODELS.index(model))),
                    'persistence_pred':str(.04 if k%3==0 else .01),'mase_scale':'0.002',
                    'high_volatility_known':str(k%3==0),'fit_origin':origin,'fit_outcome_max':cutoff,
                    'scaler_fit_outcome_max':cutoff,'configuration_selection_outcome_max':cutoff,
                    'checkpoint_closed_at_utc':'2026-11-06T12:00:00+00:00',
                    'scoring_started_at_utc':'2026-11-06T13:00:00+00:00','checkpoint_sha256':'a'*64,
                    'scope':'development_P0_not_final_benchmark','source_stage':'P0_diagnostic','eligible':'True'})
    return rows


class RemediationStatisticsTests(unittest.TestCase):
    def test_mean_seed_rmse_is_not_pooled_or_ensemble_rmse(self):
        sq=np.array([[[0.,0.],[4.,4.]]])
        self.assertEqual(mean_seed_rmse(sq)[0],1.)
        self.assertNotEqual(mean_seed_rmse(sq)[0],np.sqrt(sq.mean()))
        errors=np.array([[[2.,2.],[-2.,-2.]]])
        self.assertEqual(mean_seed_rmse(errors**2)[0],2.)
        self.assertEqual(np.sqrt(np.mean(errors.mean(axis=1)**2)),0.)

    def test_regime_gap_aggregates_seed_rmse_before_difference(self):
        sq=np.array([[[16.,16.,1.,1.],[0.,0.,1.,1.]]])
        self.assertEqual(region_gap(sq,[1,1,0,0])[0],1.)
        self.assertTrue(np.isnan(region_gap(sq,[1,1,1,1])[0]))

    def test_dense_calendar_complete_pairing_and_causal_mask(self):
        self.assertEqual(len(calendar_for('pilot')),60)
        self.assertEqual(len(calendar_for('oof')),305)
        rows=fixture_rows();panel=validate_panel(rows,'pilot')
        self.assertEqual(panel['errors'].shape,(4,5,60))
        self.assertEqual(panel['high'].sum(),20)
        self.assertTrue((panel['targets']>THRESHOLD).all())
        shuffled=list(reversed(rows))
        np.testing.assert_array_equal(panel['errors'],validate_panel(shuffled,'pilot')['errors'])
        for bad in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaises(ValueError):validate_panel(bad,'pilot')

    def test_future_day_rejected_before_numeric_decoding(self):
        rows=fixture_rows();rows[0]={**rows[0],'outcome_day':'2025-01-01','y_true':'must not decode'}
        with self.assertRaisesRegex(ValueError,'outside the fixed development'):validate_panel(rows,'pilot')

    def test_metadata_leakage_and_pairing_mismatches_fail(self):
        for key,value in [('high_volatility_known','False'),('y_true','0.031'),
                          ('scaler_fit_outcome_max','2024-12-01'),('checkpoint_sha256','b'*64),
                          ('checkpoint_closed_at_utc','2026-11-06T14:00:00+00:00')]:
            rows=fixture_rows();rows[0]={**rows[0],key:value}
            with self.assertRaises(ValueError,msg=key):validate_panel(rows,'pilot')

    def test_calendar_block_chunks_preserve_adjacent_days_and_seed_pairing(self):
        idx=block_indices(60,7,np.random.default_rng(10))
        for start in range(0,60,7):
            np.testing.assert_array_equal(np.diff(idx[start:start+7])%60,np.ones(len(idx[start:start+7])-1))
        rng=np.random.default_rng(1);sq=rng.uniform(.001,.01,(4,5,60));sq[3]=sq[0]
        boot=bootstrap_panel(sq,np.arange(60)%3==0,7,99)
        np.testing.assert_array_equal(boot['gain_vs_GRU'][:,0],np.zeros(99))
        np.testing.assert_array_equal(boot['RMSE'],bootstrap_panel(sq,np.arange(60)%3==0,7,99)['RMSE'])

    def test_empty_and_sparse_strata_are_not_false_zero_effects(self):
        sq=np.ones((4,5,60)); high=np.zeros(60,dtype=bool)
        result=bootstrap_panel(sq,high,7,99)
        self.assertTrue(np.isnan(result['gap']).all())
        self.assertEqual(interval(result['gap'][:,0])['status'],'not_estimable')
        self.assertIsNone(interval(result['gap'][:,0])['ci95'])
        high[0]=True; result=bootstrap_panel(sq,high,14,499)
        info=interval(result['gap'][:,0])
        self.assertLess(info['estimable_draws'],.95*499)
        self.assertEqual(info['status'],'not_estimable')
        self.assertEqual(interval(result['RMSE'][:,0])['status'],'estimable')

    def test_holm_keeps_three_hypotheses_with_missing_member(self):
        adjusted=holm_three([.003,None,.02])
        self.assertAlmostEqual(adjusted[0],.009)
        self.assertIsNone(adjusted[1])
        self.assertAlmostEqual(adjusted[2],.04)
        self.assertEqual(holm_three([None,None,None]),[None,None,None])
        with self.assertRaises(ValueError):holm_three([.01,.02])

    def test_centered_beyond_margin_tail_has_finite_draw_correction(self):
        self.assertEqual(centered_upper_p(2.,[1.,2.,3.,4.],1.),3/5)
        self.assertIsNone(centered_upper_p(float('nan'),[1.,2.],0.))

    def test_known_noncentral_t_planning_benchmark(self):
        # Classical one-sided one-sample t design: d=.5, alpha=.05,
        # 80% power needs 27 integer observations (26 remains below80%).
        r=required_seeds(1.,epsilon=.5,alpha=.05)
        self.assertEqual(r['required_R'],27)
        self.assertLess(conditional_power(26,1.,.5,.05),.8)
        self.assertGreaterEqual(conditional_power(27,1.,.5,.05),.8)

    def test_zero_sd_and_cap_do_not_claim_80percent_power(self):
        self.assertIsNone(required_seeds(0.)['required_R'])
        self.assertIsNone(conditional_power(5,0.))
        r=required_seeds(1.,epsilon=.001)
        self.assertEqual(r['status'],'required_R_exceeds_cap')
        self.assertEqual(r['required_R_lower_bound'],201)
        self.assertLess(r['power_at_cap'],.8)

    def test_sd_uncertainty_and_fixed_alternative_do_not_follow_observed_gain(self):
        gains=np.arange(-2,3)*.0001
        a,b=seed_power_report(gains),seed_power_report(gains+.1)
        self.assertEqual(a['planned_alternative_gain'],2*DELTA)
        self.assertEqual(a['point_SD_projection']['required_R'],b['point_SD_projection']['required_R'])
        self.assertEqual(a['variance_df'],4)
        low,high=a['SD_ci95_normal_seed_assumption'];sd=a['paired_difference_SD']
        self.assertTrue(.59*sd<low<.61*sd)
        self.assertTrue(2.85*sd<high<2.90*sd)
        self.assertFalse(a['final_post_HPO_R_certified'])

    def test_only_oof_block7_has_a_diagnostic_decision_p0_stays_descriptive(self):
        for phase in ['pilot','oof']:
            panel=validate_panel(fixture_rows(phase),phase)
            out=analyse_panel(panel,draws=99)
            principal=[r for r in out['gap'] if r['principal_diagnostic']]
            self.assertEqual(len(principal),3 if phase=='oof' else 0)
            if phase=='pilot':
                self.assertTrue(all(r['p_one_sided_beyond_delta'] is None and r['diagnostic_gap_supported'] is None for r in out['gap']))
                self.assertIsNotNone(out['power'])
            else:
                self.assertTrue(all(r['diagnostic_gap_supported'] is None for r in out['gap'] if r['block_days']!=7))
                self.assertIsNone(out['power'])


if __name__=='__main__':unittest.main()
