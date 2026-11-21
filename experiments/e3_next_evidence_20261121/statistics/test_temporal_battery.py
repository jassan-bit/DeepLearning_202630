"""Software guards only; these fixtures are not the controlled calibration run."""
import dataclasses
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import temporal_battery as b


class Guards(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4451)
        z=rng.normal(size=(12,48,1))
        self.q=z*z*np.array([1.,1.44,1.10])[None,None,:]+.05
        self.names=['candidate','rival1','rival2']
        self.settings=b.Settings(bootstrap_replicates=127,stationary_mean_block=6,hac_lag=4)

    def run_case(self,q=None,margin=0.,candidate=0):
        return b.run_battery(self.q if q is None else q,self.names,candidate,margin,
                             np.random.default_rng(774),self.settings)

    def test_mean_seed_RMSE_is_not_pooled_RMSE(self):
        q=np.ones((12,48,3));q[6:,:,0]=9;q[:,:,1]=4;q[:,:,2]=5
        result=self.run_case(q)
        self.assertAlmostEqual(result['pairwise'][0]['theta_hat'],0.)
        self.assertNotAlmostEqual(result['pairwise'][0]['theta_hat'],2-np.sqrt(5))

    def test_direction_and_margin(self):
        favorable=self.run_case()['pairwise'][0]
        self.assertGreater(favorable['theta_hat'],0)
        self.assertLess(favorable['methods'][b.P1]['p_raw'],.05)
        self.assertLess(favorable['methods'][b.P2]['p_raw'],.05)
        strict=self.run_case(margin=10.)['pairwise'][0]
        self.assertGreater(strict['methods'][b.P1]['p_raw'],.95)
        self.assertGreater(strict['methods'][b.P2]['p_raw'],.95)

    def test_seed_only_component_remains(self):
        q=np.broadcast_to(np.linspace(.5,1.5,12)[:,None,None]**2*self.q.mean(axis=(0,1))[None,None,:],self.q.shape).copy()
        result=self.run_case(q)['pairwise'][0]['variance_components']
        self.assertGreater(result['seed_variance'],0)
        self.assertLess(result['time_variance'],1e-25)
        self.assertLess(result['interaction_variance_diagnostic'],1e-25)

    def test_calendar_only_component_remains(self):
        q=np.broadcast_to(self.q[:1],self.q.shape).copy()
        result=self.run_case(q)['pairwise'][0]['variance_components']
        self.assertLess(result['seed_variance'],1e-25)
        self.assertGreater(result['time_variance'],0)
        self.assertLess(result['interaction_variance_diagnostic'],1e-25)

    def test_interaction_is_reported_and_not_silently_zeroed(self):
        rng=np.random.default_rng(13);q=.5+rng.uniform(size=self.q.shape)
        result=self.run_case(q)['pairwise'][0]['variance_components']
        self.assertGreater(result['interaction_variance_diagnostic'],0)
        self.assertAlmostEqual(result['variance_used'],result['seed_variance']+result['time_variance'])
        self.assertAlmostEqual(result['interaction_corrected_variance_diagnostic'],
             result['variance_used']-result['interaction_variance_diagnostic'])

    def test_bad_grids_rejected(self):
        for q in [self.q[:4],self.q[:,:12],-self.q,np.zeros_like(self.q),
                  np.full_like(self.q,np.nan),np.full_like(self.q,np.inf)]:
            with self.subTest(shape=q.shape):
                with self.assertRaises(ValueError):self.run_case(q)
        with self.assertRaises(ValueError):
            b.run_battery(self.q,['same','same','other'],0,0,np.random.default_rng(1),self.settings)

    def test_invalid_margins_and_resampling_rejected(self):
        for margin in [-1,float('nan'),[0,0,0]]:
            with self.subTest(margin=margin):
                with self.assertRaises((ValueError,TypeError)):self.run_case(margin=margin)
        with self.assertRaises(ValueError):
            b.run_battery(self.q,self.names,0,0,np.random.default_rng(1),
                          dataclasses.replace(self.settings,stationary_mean_block=49))
        with self.assertRaises(ValueError):
            b.run_battery(self.q,self.names,0,0,np.random.default_rng(1),
                          dataclasses.replace(self.settings,minimum_seeds=1))

    def test_identical_models_nonreject_and_MCS_retains(self):
        q=np.repeat(self.q[:,:,:1],3,axis=2);result=self.run_case(q)
        self.assertEqual(set(result['global_MCS']['retained']),set(self.names))
        self.assertEqual(result['global_MCS']['initial_global_p'],1.)
        for row in result['pairwise']:
            self.assertEqual(row['methods'][b.P1]['p_raw'],1.)
            self.assertEqual(row['methods'][b.P2]['p_raw'],1.)

    def test_global_MCS_does_not_depend_on_practical_margin(self):
        first=self.run_case(margin=0.)['global_MCS']
        second=self.run_case(margin=.5)['global_MCS']
        self.assertEqual(first,second)
        self.assertFalse(first['third_directional_p_provided'])

    def test_two_methods_are_not_reported_as_three_pairwise_tests(self):
        result=self.run_case()
        self.assertEqual(result['directional_pairwise_methods'],2)
        self.assertEqual(result['global_complement_methods'],1)
        self.assertFalse(result['three_tests_per_contrast_gate'])
        self.assertFalse(result['complete_three_method_power_computed'])
        self.assertFalse(result['confirmation_enabled'])
        for row in result['pairwise']:
            self.assertEqual(set(row['methods']),{b.P1,b.P2})

    def test_Holm_known_values_and_missing_reservation_conservative(self):
        np.testing.assert_allclose(b.holm([.01,.04,.03]),[.03,.06,.06])
        np.testing.assert_allclose(b.holm([.01,.04,.03],4),[.04,.09,.09])
        p=np.linspace(.0001,.8,38)
        self.assertTrue((b.holm(p,57)>=b.holm(p)).all())
        with self.assertRaises(ValueError):b.holm(p,19)

    def test_seeded_repeatability_and_input_read_only(self):
        q=self.q.copy();q.flags.writeable=False
        one=self.run_case(q);two=self.run_case(q)
        self.assertEqual(one,two);np.testing.assert_array_equal(q,self.q)

    def test_stationary_indices_circular_and_shared(self):
        indices=b.stationary_indices(127,48,6,np.random.default_rng(1))
        self.assertEqual(indices.shape,(127,48))
        self.assertTrue(((indices>=0)&(indices<48)).all())
        self.assertGreater(np.mean(indices[:,1:]==(indices[:,:-1]+1)%48),.65)


if __name__=='__main__':unittest.main()
