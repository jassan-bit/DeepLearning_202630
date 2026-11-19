"""Meaningful artificial checks; no market data or model fitting."""
import io
import math
import unittest
import torch

from src.e3_overlap_model_v3 import C, OverlapDistributionModel, discrete_crps, loss_components


class AtomicOverlapChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self):
        gen = torch.Generator().manual_seed(4701)
        return torch.randn(6, 7, 5, generator=gen), torch.linspace(0., .0009, 6, dtype=torch.float64)

    def test_counts_shared_initialization_support_and_probability_law(self):
        X, A = self.inputs()
        models = {name: OverlapDistributionModel(name, u_scale=.02, y_scale=.01, seed=83)
                  for name in ('atom_crps','atom_mse_only','fixed_probs')}
        self.assertEqual({name: sum(p.numel() for p in m.parameters()) for name,m in models.items()},
                         {'atom_crps':1262,'atom_mse_only':1262,'fixed_probs':1236})
        states = [m.forward_components(X, A) for m in models.values()]
        for result in states:
            self.assertTrue(bool((result['atoms'][:,1:] > result['atoms'][:,:-1]).all()))
            self.assertTrue(torch.allclose(result['probabilities'].sum(-1), torch.ones(6,dtype=torch.float64)))
            self.assertTrue(bool((result['prediction'] >= torch.sqrt(A)).all()))
        self.assertTrue(torch.equal(states[0]['atoms'], states[1]['atoms']))
        self.assertTrue(torch.equal(states[0]['atoms'], states[2]['atoms']))
        self.assertTrue(torch.equal(states[0]['prediction'], states[1]['prediction']))
        self.assertTrue(torch.equal(states[0]['prediction'], states[2]['prediction']))

    def test_crps_agrees_with_enumeration_and_point_mass_absolute_error(self):
        atoms = torch.tensor([[.1,.8,2.1]],dtype=torch.float64)
        probabilities = torch.tensor([[.2,.3,.5]],dtype=torch.float64)
        u = torch.tensor([1.3],dtype=torch.float64)
        brute = sum(float(probabilities[0,i])*abs(float(atoms[0,i])-1.3) for i in range(3))
        brute -= .5*sum(float(probabilities[0,i]*probabilities[0,j])*abs(float(atoms[0,i]-atoms[0,j])) for i in range(3) for j in range(3))
        self.assertAlmostEqual(float(discrete_crps(atoms, probabilities, u)),brute,places=14)
        point = discrete_crps(torch.tensor([[.7]],dtype=torch.float64),torch.ones(1,1,dtype=torch.float64),u)
        self.assertAlmostEqual(float(point),.6,places=14)

    def test_true_finite_law_minimizes_population_joint_score_on_grid(self):
        support = torch.tensor([.2,.8,1.7],dtype=torch.float64)
        true_p = torch.tensor([.2,.3,.5],dtype=torch.float64)
        A = .04
        truth_sigma = torch.sqrt(A+C*support.square())
        def risk(p):
            p=torch.as_tensor(p,dtype=torch.float64)
            pred=p@truth_sigma
            mse=(true_p*(pred-truth_sigma).square()).sum()
            crps=discrete_crps(support.repeat(3,1),p.repeat(3,1),support)
            return float(mse+.1*(true_p*crps).sum())
        baseline=risk(true_p)
        for i in range(1,9):
            for j in range(1,10-i):
                p=[i/10,j/10,1-(i+j)/10]
                self.assertGreaterEqual(risk(p)+1e-14,baseline)
        self.assertGreater(risk([.6,.2,.2]),baseline)

    def test_atom_crps_autograd_and_finite_network_gradients(self):
        atoms=torch.tensor([[.2,.8,1.7]],dtype=torch.float64,requires_grad=True)
        logits=torch.tensor([[.1,-.2,.4]],dtype=torch.float64,requires_grad=True)
        obs=torch.tensor([1.1],dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda a,l:discrete_crps(a,torch.softmax(l,-1),obs),(atoms,logits)))
        X,A=self.inputs();model=OverlapDistributionModel(u_scale=.02,y_scale=.01,seed=98)
        U=torch.linspace(.005,.05,6,dtype=torch.float64)
        Y=torch.sqrt(A+C*U.square())
        result=model.loss_components(model.forward_components(X,A),Y,U)
        result['loss'].backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        self.assertTrue(bool((model.probability_head.weight.grad.abs()>0).any()))

    def test_fixed_law_monotonicity_concavity_and_pushforward_not_rms(self):
        model=OverlapDistributionModel(u_scale=.02,y_scale=.01,seed=17)
        X=torch.zeros(1,7,5)
        grid=torch.linspace(0.,.01,201,dtype=torch.float64)
        with torch.no_grad():
            out=model.forward_components(X.repeat(len(grid),1,1),grid)
        predictions=out['prediction']
        self.assertTrue(bool((predictions[1:]>=predictions[:-1]).all()))
        self.assertTrue(bool((predictions[2:]-2*predictions[1:-1]+predictions[:-2]<=1e-14).all()))
        rms=torch.sqrt(grid+C*(out['probabilities']*out['atoms'].square()).sum(-1))
        self.assertTrue(bool((predictions<=rms+1e-14).all()))
        self.assertGreater(float((rms-predictions).max()),1e-6)

    def test_scale_invariance_and_single_normalization(self):
        X,A=self.inputs();k=7.0
        first=OverlapDistributionModel(u_scale=.02,y_scale=.01,seed=76)
        second=OverlapDistributionModel(u_scale=k*.02,y_scale=k*.01,seed=76)
        out1=first.forward_components(X,A);out2=second.forward_components(X,k*k*A)
        self.assertTrue(torch.allclose(out2['prediction'],k*out1['prediction'],rtol=1e-13,atol=1e-14))
        U=torch.linspace(.002,.05,6,dtype=torch.float64);Y=torch.sqrt(A+C*U.square())
        loss1=first.loss_components(out1,Y,U);loss2=second.loss_components(out2,k*Y,k*U)
        for key in loss1:self.assertAlmostEqual(float(loss1[key].detach()),float(loss2[key].detach()),places=12)

    def test_state_roundtrip_with_configuration_and_input_rejection(self):
        X,A=self.inputs();model=OverlapDistributionModel(u_scale=.02,y_scale=.01,seed=31).eval()
        before=model(X,A).detach()
        stream=io.BytesIO();torch.save({'configuration':model.configuration(),'state':model.state_dict()},stream);stream.seek(0)
        stored=torch.load(stream,weights_only=True)
        cfg=stored['configuration']
        restored=OverlapDistributionModel(cfg['variant'],u_scale=cfg['u_scale'],y_scale=cfg['y_scale'],seed=cfg['seed'],lambda_crps=cfg['lambda_crps']).eval()
        restored.load_state_dict(stored['state'])
        self.assertTrue(torch.equal(before,restored(X,A).detach()))
        with self.assertRaises(ValueError):model(X,-torch.ones(6))
        with self.assertRaises(ValueError):model(X.double(),A)
        with self.assertRaises(ValueError):model.loss_components(model.forward_components(X,A),torch.ones(6),-torch.ones(6))


if __name__=='__main__':
    unittest.main()
