"""Independent arithmetic and estimand checks for the registered ensemble."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from e2_calibration_ensemble import fit_weights, combine, risk, gamma_values, resample_risks

class CalibrationEnsembleTests(unittest.TestCase):
    def test_simplex_fit_recovers_reliable_source_without_negative_leverage(self):
        y=np.linspace(.01,.1,53)
        p=np.stack([np.tile(y,(6,1)),np.tile(y+.5,(6,1))])
        w,receipt=fit_weights(p,y,.05)
        self.assertAlmostEqual(float(w.sum()),1.)
        self.assertTrue((w>=0).all())
        self.assertGreater(w[0],.9999)
        self.assertLess(receipt["calibration_scaled_objective"],receipt["uniform_scaled_objective"])

    def test_identical_sources_have_uniform_unique_regularized_solution(self):
        y=np.linspace(0,1,17);p=np.tile(y,(4,6,1))
        w,_=fit_weights(p,y,1.)
        np.testing.assert_allclose(w,np.full(4,.25),atol=1e-12)

    def test_evaluation_is_mean_individual_rmse_not_rmse_of_mean_forecast(self):
        p=np.array([[0.,0.],[2.,2.]])
        y=np.ones(2)
        self.assertEqual(risk(p,y),1.)
        self.assertEqual(risk(p.mean(0)[None,:],y),0.)
        np.testing.assert_allclose(combine([.25,.75],np.stack([p,2*p])),1.75*p)

    def test_invalid_empirical_reference_is_not_reported_as_ceiling_fraction(self):
        d=gamma_values(.0038,.0041,.0042)
        self.assertFalse(d["interpretable_empirical_reference_order"])
        self.assertIsNone(d["gamma_remaining_fraction"])
        self.assertAlmostEqual(gamma_values(.0038,.0041,.0035)["gamma_remaining_fraction"],.5)

    def test_bootstrap_joint_resampling_preserves_identical_risks(self):
        p=np.tile(np.arange(6,dtype=float),(2,1))
        forecasts=np.stack([p,p,p]);y=np.zeros(6)
        result=resample_risks(forecasts,y,draws=10000,block=2)
        np.testing.assert_array_equal(result[:,0],result[:,1])
        np.testing.assert_array_equal(result[:,0],result[:,2])
        with self.assertRaises(ValueError):resample_risks(forecasts,y,draws=9999)

if __name__=="__main__":unittest.main()
