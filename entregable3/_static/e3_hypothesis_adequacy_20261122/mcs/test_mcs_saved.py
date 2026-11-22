"""Generated software checks, not Monte Carlo validation or market evidence."""
from pathlib import Path
import hashlib
import importlib.util
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mcs_source',HERE/'mcs_tmax_reference.py')
mcs=importlib.util.module_from_spec(spec);spec.loader.exec_module(mcs)


class ExistingMCSTests(unittest.TestCase):
    def settings(self,alpha=.1,seed=71):
        return dict(mcs=dict(alpha=alpha,block_days=7,repetitions=399,seed=seed))

    def panel(self,losses,order=None):
        names=list(losses) if order is None else order
        n=len(next(iter(losses.values())))
        return dict(original_model='candidate',targets=np.zeros(n),
            predictions={k:np.sqrt(np.asarray(losses[k]))[None,:] for k in names})

    def varied(self):
        t=np.arange(140)
        a=1+.2*np.sin(t/4)
        return {'candidate':a,'control':a+.01*np.cos(t/3),'poor':a+3+.1*np.cos(t/7)}

    def test_reference_is_exact_existing_published_source(self):
        self.assertEqual(hashlib.sha256((HERE/'mcs_tmax_reference.py').read_bytes()).hexdigest(),
            'bb14d47277a06e4c59ac1145633dd1f5d6c23f065cac5885b5433324aeca6af4')

    def test_exact_ties_retain_all_and_have_EPA_p_one(self):
        a=np.linspace(1,2,140)
        result=mcs.run_mcs(self.panel({'candidate':a,'same':a.copy(),'also_same':a.copy()}),self.settings())
        self.assertEqual(set(result['included_models']),{'candidate','same','also_same'})
        self.assertEqual(result['steps'][0]['p'],1.)
        self.assertEqual(result['excluded_pvalues'],{})

    def test_varying_obviously_poor_rule_is_excluded(self):
        result=mcs.run_mcs(self.panel(self.varied()),self.settings())
        self.assertIn('poor',result['excluded_pvalues'])
        self.assertNotIn('poor',result['included_models'])
        self.assertIn('candidate',result['included_models'])

    def test_dictionary_permutation_changes_neither_stats_nor_membership(self):
        losses=self.varied()
        a=mcs.run_mcs(self.panel(losses),self.settings())
        b=mcs.run_mcs(self.panel(losses,['poor','candidate','control']),self.settings())
        self.assertEqual(a,b)

    def test_stepwise_elimination_p_is_monotone(self):
        result=mcs.run_mcs(self.panel(self.varied()),self.settings(.8))
        values=[s['p_stepwise'] for s in result['steps']]
        self.assertEqual(values,sorted(values))

    def test_higher_confidence_set_is_not_smaller(self):
        losses=self.varied()
        lower=mcs.run_mcs(self.panel(losses),self.settings(.8))
        higher=mcs.run_mcs(self.panel(losses),self.settings(.1))
        self.assertTrue(set(lower['included_models'])<=set(higher['included_models']))

    def test_common_additive_loss_path_is_contrast_invariant(self):
        losses=self.varied()
        offset=.3+.1*np.cos(np.arange(140)/8)
        a=mcs.run_mcs(self.panel(losses),self.settings())
        b=mcs.run_mcs(self.panel({k:v+offset for k,v in losses.items()}),self.settings())
        self.assertEqual(a['included_models'],b['included_models'])
        self.assertEqual(a['excluded_pvalues'],b['excluded_pvalues'])

    def test_primary_mean_RMSE_is_not_pool_MSE(self):
        candidate=np.array([1.21,1.21]); rival=np.array([0.,4.])
        self.assertLess(candidate.mean(),rival.mean())
        self.assertGreater(np.sqrt(candidate).mean(),np.sqrt(rival).mean())

    def test_zero_relative_variance_effect_is_disclosed_not_invented(self):
        panel=self.panel({'candidate':np.ones(140),'constant_poor':np.full(140,4.)})
        result=mcs.run_mcs(panel,self.settings())
        self.assertEqual(result['status'],'not_estimable_zero_relative_variance')
        self.assertIsNone(result['steps'][0]['p'])
        self.assertEqual(set(result['included_models']),{'candidate','constant_poor'})


if __name__=='__main__':
    unittest.main(verbosity=2)
