"""Offline mathematical/software checks, not calibration or market evidence."""
import copy
import math
import unittest
import numpy as np

import primary_rmse as m


def fixture_panel():
    tests=lambda:{method:{null:dict(status='ok',raw_p=1e-8,adjusted_p=1e-6)
        for null in ('epa','practical')} for method in m.METHODS}
    return dict(protocol_version=m.PROTOCOL_VERSION,systems=list(m.EXPECTED_SYSTEMS),candidate=m.EXPECTED_SYSTEMS[0],
        n_dates=768,n_fixed_seeds=10,
        family=dict(alpha=.025,tests=120,methods=list(m.METHODS),nulls=['epa','practical']),
        settings=dict(bootstrap_reps=8191,block_length=32.,hac_lags=48,horizon=1,
            margin_relative_mean_seed_RMSE=.02,alpha=.025,reference_reps=49999,reference_seed=2026112202,
            operational_interval_method=m.OPERATIONAL_INTERVAL_METHOD),
        comparisons=[dict(rival=name,mean_rmse_difference=.1,practical_mean=.078,
            temporal_effect_cohen_d=.4,ci95_rmse_difference=[.05,.15],ci99_rmse_difference=[.03,.18],
            ci95_relative_rmse_gain=[.04,.12],ci99_relative_rmse_gain=[.03,.14],
            interval_method=m.OPERATIONAL_INTERVAL_METHOD,methods=tests())
            for name in m.EXPECTED_SYSTEMS[1:]])


class PrimaryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(5102341)
        self.losses=np.exp(self.rng.normal(0,.3,(3,3,96)))
        self.z=np.column_stack([np.ones(96),.6+.2*np.sin(np.arange(96)/8)])

    def test_primary_is_mean_of_individual_roots_not_pooled_root(self):
        risks=np.asarray([[1.21,1.21],[.01,4.]])
        values=np.repeat(risks[:,:,None],96,axis=2)
        theta,_,_,_=m.functional_and_influence(values)
        np.testing.assert_allclose(theta,[1.1,1.05],rtol=0,atol=1e-14)
        self.assertGreater(theta[0],theta[1])
        self.assertLess(np.sqrt(risks[0].mean()),np.sqrt(risks[1].mean()))

    def test_nuisance_derivative_uses_temporal_mean_not_sum(self):
        theta,psi,mu,rho=m.functional_and_influence(self.losses)
        actual=m.conditional_moments(self.losses,self.z,1)
        for j,key in ((0,'candidate_risk_derivative'),(1,'rival_risk_derivative')):
            expected=-(self.z.T@(self.losses[j]-mu[j,:,None]).T)/(4*96*3*mu[j][None,:]**1.5)
            np.testing.assert_allclose(actual[key],expected,rtol=1e-11,atol=1e-13)
        np.testing.assert_allclose(actual['influence'][:,0],psi[1]-psi[0],rtol=1e-11,atol=1e-13)

    def test_scalar_equality_does_not_imply_conditional_vector_null(self):
        losses=np.empty((2,2,64));pattern=np.tile([1.,4.],32)
        losses[0]=pattern;losses[1]=pattern[::-1]
        z=np.column_stack([np.ones(64),np.tile([0.,1.],32)])
        value=m.conditional_moments(losses,z,1)
        self.assertAlmostEqual(value['moment_mean'][0],0.,places=13)
        self.assertGreater(abs(value['moment_mean'][1]),.1)

    def test_native_and_numpy_recompute_roots_and_hac_on_same_indices(self):
        ix=m.stationary_indices(96,199,7,np.random.default_rng(440))
        a=m.bootstrap_functional(self.losses,199,7,7,.02,0,'native',ix,False)
        b=m.bootstrap_functional(self.losses,199,7,7,.02,0,'numpy',ix,False)
        for key in ('theta','epa_t','practical_t','log_ratio_t'):
            np.testing.assert_allclose(a[key],b[key],rtol=2e-10,atol=2e-10)

    def test_exact_channel_seed_compression_matches_uncompressed(self):
        base=np.exp(self.rng.normal(0,.2,(3,2,96)))
        values=np.repeat(np.concatenate([base[:2],np.repeat(base[2:],19,axis=0)]),5,axis=1)
        ix=m.stationary_indices(96,199,7,np.random.default_rng(45))
        a=m.bootstrap_functional(values,199,7,7,.02,0,'native',ix,False)
        b=m.bootstrap_functional(values,199,7,7,.02,0,'native',ix,True)
        self.assertEqual(b['exact_reduction']['computed_shape'],[3,2,96])
        self.assertEqual(b['theta'].shape,(199,21))
        for key in ('theta','epa_t','practical_t','log_ratio_t'):
            np.testing.assert_allclose(a[key],b[key],rtol=2e-10,atol=2e-10)

    def test_unequal_seed_class_sizes_are_not_reweighted_or_reduced(self):
        base=np.exp(self.rng.normal(0,.2,(3,2,96)))
        value=np.concatenate([np.repeat(base[:,:1],7,axis=1),np.repeat(base[:,1:],3,axis=1)],axis=1)
        reduced,_,metadata=m.exact_bootstrap_reduction(value)
        self.assertEqual(reduced.shape[1],10)
        self.assertFalse(metadata['equal_multiplicity_seed_reduction'])

    def test_cached_indices_exactly_match_fresh_generator_and_are_readonly(self):
        cached,identity=m._cached_stationary_indices(96,199,7,934)
        fresh=m.stationary_indices(96,199,7,np.random.default_rng(934))
        np.testing.assert_array_equal(cached,fresh)
        self.assertIs(cached,m._cached_stationary_indices(96,199,7,934)[0])
        with self.assertRaises(ValueError):cached[0,0]=13

    def test_primary_practical_margin_is_linear_in_theta(self):
        result=m.analyze_losses(self.losses,['candidate','a','b'],bootstrap_reps=199,block_length=7,
            hac_lags=7,instruments=self.z,reference_reps=999)
        c=result['comparisons'][0];theta=result['metrics']['candidate']['mean_seed_rmse'];rival=result['metrics']['a']['mean_seed_rmse']
        self.assertAlmostEqual(c['practical_mean'],.98*rival-theta,places=13)
        self.assertNotAlmostEqual(c['practical_mean'],.98**2*rival-theta,places=7)

    def test_zero_seed_risks_fail_closed_without_epsilon_clamping(self):
        values=self.losses.copy();values[0,0]=0.
        result=m.analyze_losses(values,['candidate','a','b'],bootstrap_reps=199,block_length=7,
            hac_lags=7,instruments=self.z,reference_reps=999)
        self.assertFalse(result['regular_positive_seed_risks'])
        self.assertFalse(result['all_rivals_statistical_gate'])
        self.assertEqual(result['metrics']['candidate']['seed_rmse'][0],0.)
        self.assertTrue(all(test['raw_p']==1. for c in result['comparisons'] for tests in c['methods'].values() for test in tests.values()))

    def test_formal_joint_rule_retains_two_of_three_not_all_three(self):
        panels=[fixture_panel() for _ in range(4)]
        for c in panels[3]['comparisons']:
            for tests in c['methods'].values():
                for test in tests.values():test.update(raw_p=1.,adjusted_p=1.)
        rule=m.statistical_rule(panels)
        self.assertTrue(rule['joint_statistical_rule_pass'])
        self.assertFalse(rule['all_contrasts_pass'])
        self.assertEqual(rule['additional_adverse_Holm_slots'],180)
        panels[2]=copy.deepcopy(panels[3])
        self.assertFalse(m.statistical_rule(panels)['joint_statistical_rule_pass'])

    def test_formal_family_and_version_guards(self):
        for defect in ('family','version','margin','CI','missing_method','slots'):
            panels=[fixture_panel() for _ in range(4)]
            if defect=='family':panels[0]['comparisons'].pop()
            elif defect=='version':panels[1]['protocol_version']='old-pooled-MSE'
            elif defect=='margin':panels[1]['settings']['margin_relative_mean_seed_RMSE']=.03
            elif defect=='CI':panels[1]['comparisons'][0]['interval_method']='old_MSE_CI'
            elif defect=='missing_method':panels[1]['comparisons'][0]['methods'].pop(m.METHODS[0])
            else:panels[1]['family']['tests']=114
            with self.subTest(defect=defect),self.assertRaises(ValueError):m.statistical_rule(panels)


if __name__=='__main__':unittest.main(verbosity=2)
