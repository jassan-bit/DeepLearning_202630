"""Deterministic V2 inversion checks; no outer Monte Carlo or market data."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np
import primary_rmse as m


class InversionTests(unittest.TestCase):
    def test_exact_nominal_ranks_do_not_subtract_binary_confidence(self):
        reference=np.arange(49999,dtype=float)
        for level,index,alpha in ((.95,47500,.05),(.99,49500,.01)):
            critical=m.mc_plus_one_critical(reference,level)
            self.assertEqual(critical['zero_based_order_index'],index)
            self.assertEqual(critical['alpha'],alpha)
            self.assertEqual(critical['critical_value'],reference[index])

    def test_closed_endpoint_and_just_outside_invert_strict_plus_one_p(self):
        for reference in (np.arange(1.,101.),np.repeat(np.arange(1.,26.),4)):
            for level in (.95,.99,.8):
                critical=m.mc_plus_one_critical(reference,level)
                q=critical['critical_value'];alpha=critical['alpha']
                self.assertGreaterEqual(m.mc_p(q,reference,'fixture')['raw_p'],alpha)
                self.assertLess(m.mc_p(np.nextafter(q,math.inf),reference,'fixture')['raw_p'],alpha)
                interval=m.fixed_b_inverted_ci(0.,1.,1.,critical)
                self.assertEqual(interval,[-q,q])

    def test_insufficient_mc_resolution_returns_unbounded_not_finite_interval(self):
        critical=m.mc_plus_one_critical(np.arange(1.,100.),.99)
        self.assertEqual(critical['status'],'unbounded_due_to_mc_resolution')
        self.assertIsNone(critical['critical_value'])
        self.assertIsNone(m.fixed_b_inverted_ci(2.,.1,.9,critical))
        self.assertGreaterEqual(m.mc_p(1e9,np.arange(1.,100.),'fixture')['raw_p'],.01)

    def test_hln_is_divided_not_multiplied(self):
        critical=m.mc_plus_one_critical(np.arange(1.,101.),.95)
        q=critical['critical_value'];point=.7;se=.03;hln=.8
        np.testing.assert_allclose(m.fixed_b_inverted_ci(point,se,hln,critical),
            [point-q*se/hln,point+q*se/hln],rtol=0,atol=0)
        self.assertIsNone(m.fixed_b_inverted_ci(point,0.,hln,critical))

    def test_invalid_reference_fails(self):
        for reference in ([],[2.,1.],[-1.,2.],[1.,math.nan]):
            with self.subTest(reference=reference),self.assertRaises(ValueError):
                m.mc_plus_one_critical(reference,.95)

    def test_v1_p_values_and_old_diagnostics_are_unchanged(self):
        archived=Path(__file__).resolve().parents[1]/'versions/mean_rmse_v1/methods/primary_rmse.py'
        spec=importlib.util.spec_from_file_location('archived_mean_rmse_v1',archived)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        rng=np.random.default_rng(673512)
        losses=np.exp(rng.normal(0,.25,(3,3,64)))
        z=np.column_stack([np.ones(64),np.sin(np.arange(64)/7)])
        args=dict(bootstrap_reps=199,block_length=5,hac_lags=5,instruments=z,
            reference_reps=999,bootstrap_backend='native')
        before=old.analyze_losses(losses,['c','a','b'],**args)
        after=m.analyze_losses(losses,['c','a','b'],**args)
        self.assertEqual(before['metrics'],after['metrics'])
        for previous,current in zip(before['comparisons'],after['comparisons']):
            self.assertEqual(previous['methods'],current['methods'])
            self.assertEqual(previous['temporal_effect_cohen_d'],current['temporal_effect_cohen_d'])
            for key in ('ci95_rmse_difference','ci99_rmse_difference','ci95_relative_rmse_gain','ci99_relative_rmse_gain'):
                self.assertEqual(previous[key],current['diagnostic_bootstrap_intervals_v1'][key])
            self.assertFalse(current['diagnostic_bootstrap_intervals_v1']['used_for_operational_gate'])
            delta=current['methods']['delta_hac_hln']['epa']
            q=after['settings']['interval_inversion_critical_values'][99]['critical_value']
            scale=math.sqrt(float(losses.max()))
            radius=q*delta['standard_error']/delta['hln_factor']*scale
            np.testing.assert_allclose(current['ci99_rmse_difference'],
                [current['mean_rmse_difference']-radius,current['mean_rmse_difference']+radius],rtol=1e-13)
        theta,psi,_,_=m.functional_and_influence(losses)
        log_point=math.log(theta[0]/theta[1]);log_if=psi[0]/theta[0]-psi[1]/theta[1]
        se=math.sqrt(m.hac_covariance(log_if,5)[0,0]/64)
        hln=math.sqrt(63/64);q=after['settings']['interval_inversion_critical_values'][99]['critical_value']
        expected=[1-math.exp(log_point+q*se/hln),1-math.exp(log_point-q*se/hln)]
        np.testing.assert_allclose(after['comparisons'][0]['ci99_relative_rmse_gain'],expected,rtol=1e-12,atol=1e-14)

    def test_joint_rule_rejects_v1_interval_as_operational(self):
        from test_primary_rmse import fixture_panel
        panels=[fixture_panel() for _ in range(4)]
        self.assertTrue(m.statistical_rule(panels)['joint_statistical_rule_pass'])
        panels[0]['comparisons'][0]['interval_method']=m.DIAGNOSTIC_INTERVAL_METHOD_V1
        with self.assertRaises(ValueError):m.statistical_rule(panels)


if __name__=='__main__':unittest.main(verbosity=2)
