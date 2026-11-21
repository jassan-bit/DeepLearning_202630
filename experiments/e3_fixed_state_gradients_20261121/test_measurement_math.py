"""Deterministic software guards; no model forward, real data or optimizer step."""
import importlib.util
from pathlib import Path
import unittest
import torch

p=Path(__file__).with_name('measure_fixed_states.py')
spec=importlib.util.spec_from_file_location('measurement',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MathGuards(unittest.TestCase):
    def group(self,n):return [{'params':list(range(n)),'lr':.003,'betas':(.9,.999),'eps':1e-8,
                              'weight_decay':0.,'amsgrad':False,'maximize':False}]
    def states(self,params,t=0):return {i:{'step':torch.tensor(float(t)),
        'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)} for i,p in enumerate(params)}
    def test_first_step_matches_closed_form(self):
        p=[torch.zeros(2,dtype=torch.float64)];g=[torch.tensor([.2,-.3],dtype=torch.float64)]
        d,_=m.adam_direction(g,p,self.group(1),self.states(p),torch)
        expected=-.003*g[0]/(g[0].abs()+1e-8)
        torch.testing.assert_close(d[0],expected,rtol=1e-12,atol=1e-15)
    def test_global_clip_is_one_shared_coefficient(self):
        p=[torch.zeros(1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64)]
        _,r=m.adam_direction([torch.tensor([3.],dtype=torch.float64),torch.tensor([4.],dtype=torch.float64)],p,self.group(2),self.states(p),torch)
        self.assertAlmostEqual(r['global_gradient_norm_before_clip'],5.)
        self.assertAlmostEqual(r['clip_coefficient'],1/(5+1e-6))
    def test_None_is_not_zero_with_momentum(self):
        p=[torch.zeros(1),torch.zeros(1)];s=self.states(p,10)
        for v in s.values():v['exp_avg'].fill_(.2);v['exp_avg_sq'].fill_(.1)
        d,r=m.adam_direction([None,torch.zeros(1)],p,self.group(2),s,torch)
        self.assertEqual(float(d[0]),0.);self.assertNotEqual(float(d[1]),0.)
        self.assertEqual(r['omitted_None_tensors'],1)
        self.assertEqual(float(s[1]['step']),10.);self.assertEqual(float(s[1]['exp_avg']),float(torch.tensor(.2)))
    def test_zero_lambda_preserves_primary_mask(self):
        gm=(torch.tensor([1.]),None);ga=(torch.tensor([9.]),torch.tensor([7.]))
        combined=m.combine(gm,ga,0.)
        self.assertEqual(float(combined[0]),1.);self.assertIsNone(combined[1])
    def test_counterfactual_directions_do_not_mutate_moments(self):
        p=[torch.zeros(1,dtype=torch.float64)];s=self.states(p,5)
        s[0]['exp_avg'].fill_(.4);s[0]['exp_avg_sq'].fill_(.2)
        before={k:v.clone() for k,v in s[0].items()}
        m.adam_direction([torch.tensor([.3],dtype=torch.float64)],p,self.group(1),s,torch)
        m.adam_direction([torch.tensor([-.7],dtype=torch.float64)],p,self.group(1),s,torch)
        for k,v in before.items():self.assertTrue(torch.equal(v,s[0][k]))
    def test_unsupported_optimizer_not_silently_approximated(self):
        p=[torch.zeros(1)];g=self.group(1);g[0]['weight_decay']=.1
        with self.assertRaises(ValueError):m.adam_direction([torch.ones(1)],p,g,self.states(p),torch)
    def test_gradient_dtype_mismatch_rejected(self):
        p=[torch.zeros(1,dtype=torch.float64)]
        with self.assertRaises(ValueError):m.adam_direction([torch.ones(1,dtype=torch.float32)],p,self.group(1),self.states(p),torch)
    def test_finite_entries_with_overflowing_norm_rejected(self):
        p=[torch.zeros(2,dtype=torch.float32)]
        with self.assertRaises(ValueError):m.adam_direction([torch.tensor([1e38,1e38])],p,self.group(1),self.states(p),torch)
    def test_zero_mass_entropy_uses_zero_log_zero_convention(self):
        p=torch.tensor([1.,0.,0.],dtype=torch.float64)
        entropy=-torch.special.xlogy(p,p).sum()
        self.assertTrue(bool(torch.isfinite(entropy)));self.assertEqual(float(entropy),0.)

if __name__=='__main__':unittest.main()
