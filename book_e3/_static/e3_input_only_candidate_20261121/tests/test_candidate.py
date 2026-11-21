"""Generated fixtures only; optimizer steps are forbidden throughout the suite."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from runtime import candidate as c
from runtime.vendor.seoi_v5.model import build_model

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


def fixture(n=12):
    rng = np.random.default_rng(721)
    close = 100*np.exp(np.cumsum(rng.normal(.0003,.012,(n,7)),axis=1))
    opening = close*np.exp(rng.normal(0,.003,(n,7)))
    high = np.maximum(opening,close)*np.exp(rng.uniform(.001,.007,(n,7)))
    low = np.minimum(opening,close)*np.exp(-rng.uniform(.001,.007,(n,7)))
    raw = np.stack([opening,high,low,close,rng.uniform(1e3,1e5,(n,7))],-1)
    A,*_ = c.data.known_geometry(raw)
    y = np.sqrt(A+c.data.C*rng.uniform(.001,.04,n)**2)
    return raw,y


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = patch.object(torch.optim.AdamW,"step",side_effect=AssertionError("Training forbidden"))
        cls.guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.guard.stop()

    def setUp(self):
        self.raw,self.y = fixture()
        self.fitted = c.fit_inputs(self.raw,self.y)
        self.model = c.make_model(23,self.fitted).eval()
        self.args = c.tensors(self.raw,self.fitted)[:3]

    def test_bitwise_original_recipe_state_prediction_and_loss(self):
        old = build_model("input_only_mse",23,self.fitted["u_scale"],self.fitted["y_scale"]).eval()
        self.assertEqual(set(old.state_dict()),set(self.model.state_dict()))
        for key,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,self.model.state_dict()[key]),key)
        a,b = old.forward_components(*self.args),self.model.forward_components(*self.args)
        for key in a:
            self.assertTrue(torch.equal(a[key],b[key]),key)
        Y = torch.from_numpy(self.y)
        U = torch.sqrt((Y.square()-self.args[1]).clamp_min(0)/c.data.C)
        self.assertTrue(torch.equal(old.loss_components(a,Y,U)["loss"],self.model.loss_components(b,Y,U)["loss"]))

    def test_identity_capacity_and_state_roundtrip(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),1262)
        self.assertEqual(self.model.configuration()["candidate_identity"],c.IDENTITY)
        new=c.make_model(99,self.fitted).eval()
        new.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.assertTrue(torch.equal(self.model(*self.args),new(*self.args)))

    def test_exact_finite_law_expectation_and_support(self):
        out=self.model.forward_components(*self.args)
        expected=(out["probabilities"]*torch.sqrt(self.args[1][:,None]+c.data.C*out["atoms"].square())).sum(-1)
        torch.testing.assert_close(out["prediction"],expected,atol=1e-15,rtol=1e-13)
        self.assertTrue((out["prediction"]>=self.args[1].sqrt()).all())

    def test_invariant_input_and_counterexample_to_output_equivariance(self):
        scaled=self.raw.copy();k=1.7
        scaled[...,:4]=np.exp(.4+k*np.log(scaled[...,:4]))
        newargs=c.tensors(scaled,self.fitted)[:3]
        np.testing.assert_allclose(newargs[0],self.args[0],rtol=1e-5,atol=1e-6)
        old=self.model.forward_components(*self.args)
        new=self.model.forward_components(*newargs)
        torch.testing.assert_close(old["atoms"],new["atoms"],atol=1e-10,rtol=1e-5)
        self.assertFalse(torch.allclose(new["prediction"],k*old["prediction"],rtol=1e-3,atol=1e-10))
        formula=(old["probabilities"]*torch.sqrt(k*k*self.args[1][:,None]+c.data.C*old["atoms"].square())).sum(-1)
        torch.testing.assert_close(new["prediction"],formula,rtol=1e-5,atol=1e-10)

    def test_price_level_invariance_and_later_window_independence(self):
        shifted=self.raw.copy();shifted[...,:4]*=30
        torch.testing.assert_close(self.model(*self.args),self.model(*c.tensors(shifted,self.fitted)[:3]),rtol=1e-6,atol=1e-10)
        changed=self.raw.copy();changed[-1,...,:4]*=2
        self.assertTrue(torch.equal(self.model(*self.args)[:-1],self.model(*c.tensors(changed,self.fitted)[:3])[:-1]))

    def test_flat_history_and_degenerate_fit(self):
        flat=self.raw.copy();flat[...,:4]=100
        args=c.tensors(flat,self.fitted)
        self.assertTrue(args[3].all())
        self.assertTrue(torch.isfinite(self.model(*args[:3])).all())
        with self.assertRaises(ValueError):c.fit_inputs(flat,np.ones(len(flat))*.01)

    def test_finite_gradient_and_mse_only(self):
        out=self.model.forward_components(*self.args)
        Y=torch.from_numpy(self.y)
        U=((Y.square()-self.args[1]).clamp_min(0)/c.data.C).sqrt()
        loss=self.model.loss_components(out,Y,U)
        self.assertEqual(float(loss["crps_u"]),0)
        torch.testing.assert_close(loss["loss"],((out["prediction"]-Y)/self.fitted["y_scale"]).square().mean())
        loss["loss"].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_rejects_wrong_identity_and_impossible_fit_labels(self):
        with self.assertRaises(ValueError):c.transform_known_history(self.raw,{"identity":"other"})
        with self.assertRaises(ValueError):c.fit_inputs(self.raw,np.zeros(len(self.raw)))


if __name__=="__main__":unittest.main()
