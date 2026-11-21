"""Software guards only; these are not Monte Carlo calibration evidence."""
import unittest
import numpy as np
from crossed_rmse_v2 import Settings, run, functional_and_covariance, pair_scales, holm, validate


def fixture():
    s=np.arange(12)[:,None,None];t=np.arange(48)[None,:,None];m=np.arange(4)[None,None,:]
    return (1+.015*s+.03*m+.12*np.sin((t+2*m+s)/7))**2


class Guards(unittest.TestCase):
    settings=Settings(B=63,hac_lag=4,mean_block=6,batch=8)
    names=['candidate','a','b','c']

    def call(self,q=None,seed=199):
        return run(fixture() if q is None else q,self.names,0,np.random.default_rng(seed),self.settings)

    def test_preserves_mean_individual_rmse(self):
        q=fixture();r=self.call(q)
        np.testing.assert_allclose(r['risks'],np.sqrt(q.mean(axis=1)).mean(axis=0),rtol=1e-14)

    def test_recomputes_all_studentizers(self):
        self.assertEqual(self.call()['bootstrap_studentizers_recomputed'],63)

    def test_deterministic_replay(self):
        self.assertEqual(self.call(),self.call())

    def test_unit_rescaling(self):
        a=self.call();b=self.call(fixture()*1e-20)
        self.assertEqual(a['MCS_bootstrap_t']['retained'],b['MCS_bootstrap_t']['retained'])
        self.assertAlmostEqual(a['MCS_bootstrap_t']['initial_global_p'],b['MCS_bootstrap_t']['initial_global_p'])
        np.testing.assert_allclose(np.array(b['risks'])/1e-10,a['risks'],rtol=1e-12)

    def test_equal_deterministic_models_retain_all(self):
        q=np.ones((12,48,4));r=self.call(q)
        self.assertEqual(r['MCS_bootstrap_t']['retained'],self.names)
        self.assertEqual(r['MCS_bootstrap_t']['initial_global_p'],1.)
        self.assertTrue(all(x['P1_equal_two_sided_p']==1 for x in r['pairwise']))

    def test_nonzero_zero_variance_is_unsupported(self):
        q=np.broadcast_to(np.array([1.,4.,9.,16.]),(12,48,4)).copy();r=self.call(q)
        self.assertEqual(r['MCS_bootstrap_t']['retained'],self.names)
        self.assertIn('unsupported',r['MCS_bootstrap_t']['status'])
        self.assertTrue(all(not x['nondegenerate'] for x in r['pairwise']))
        self.assertTrue(all(x['P1_CI95'] is None and x['P2_CI95'] is None for x in r['pairwise']))

    def test_requires_seed_calendar_grid(self):
        for q in [fixture()[:9],fixture()[:,:31],np.ones((12,48))]:
            with self.assertRaises(ValueError):self.call(q)

    def test_rejects_nonfinite_and_negative(self):
        for x in [np.nan,np.inf,-1.]:
            q=fixture();q[0,0,0]=x
            with self.assertRaises(ValueError):self.call(q)

    def test_rejects_zero_mse_and_duplicate_names(self):
        with self.assertRaises(ValueError):self.call(np.zeros((12,48,4)))
        with self.assertRaises(ValueError):run(fixture(),['x']*4,0,np.random.default_rng(1),self.settings)

    def test_covariance_is_symmetric_and_psd_fixture(self):
        _,c,_,_=functional_and_covariance(fixture()[None],4)
        np.testing.assert_allclose(c,c.transpose(0,2,1),atol=0)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(c[0]).min()),-1e-14)

    def test_holm_complete_family(self):
        np.testing.assert_allclose(holm([.01,.04,.03]),[.03,.06,.06])

    def test_scale_sign_swap(self):
        q=fixture();a=self.call(q);swap=q[:,:,[1,0,2,3]];b=self.call(swap)
        self.assertAlmostEqual(a['pairwise'][0]['theta_hat'],-b['pairwise'][0]['theta_hat'])
        self.assertAlmostEqual(a['pairwise'][0]['P2_equal_two_sided_p'],b['pairwise'][0]['P2_equal_two_sided_p'])

    def test_flags_cannot_accredit_confirmation(self):
        r=self.call();self.assertFalse(r['third_primary_method_accredited'])
        self.assertFalse(r['confirmed_valid_for_market']);self.assertFalse(r['R3_accredited'])
        self.assertFalse(r['MCS_bootstrap_t']['third_pairwise_p_provided'])

    def test_primary_ensemble_counterexample(self):
        a=np.sqrt(np.array([.01,8.41])).mean();b=np.sqrt(np.array([3.,3.])).mean()
        self.assertLess(a,b);self.assertGreater(np.mean([.01,8.41]),np.mean([3.,3.]))


if __name__=='__main__':unittest.main()
