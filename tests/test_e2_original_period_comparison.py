"""Analytic inference/unit checks on synthetic trajectories only."""
import copy
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_original_period_comparison as periods


class OriginalPeriodTests(unittest.TestCase):
    def fixture(self):
        dates = [(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(608)]
        calibration = [(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]
        predictors, freeze = [], []
        for model in (*periods.NEURAL, *periods.CLASSICAL):
            for seed in periods.SEEDS if model in periods.NEURAL else ("deterministic",):
                row = {"model":model,"seed":seed,"checkpoint_sha256":"a"*64,
                       "prediction_original":[1.]*608,"calibration_predictions":[1.]*53,
                       "calibration_dates":calibration,"calibration_targets":[0.]*53}
                predictors.append(row)
                freeze.append({k:row[k] for k in ("model","seed","checkpoint_sha256")})
        y = [0.]*608
        return {"dates":dates,"mask":[True]*608,"y_true":y,"predictors":predictors,
                "provenance":{"targets_sha256":periods.value_sha(y),
                              "calibration_targets_sha256":periods.value_sha([0.]*53),
                              "all_81_original_seed_RMSE_verified":True}}, {"models":freeze}

    def test_per_trajectory_risk_and_known_t(self):
        differences=[]
        for i, gain in enumerate(range(1,7)):
            sign = -1 if i%2 else 1
            row={"calibration_predictions":[sign]*53,"calibration_targets":[0.]*53,
                 "prediction_original":[sign*(1+gain)]*608}
            cal,test=periods.period_risks(row,np.zeros(608));differences.append(test-cal)
        np.testing.assert_array_equal(differences,range(1,7))
        result=periods.t_period(differences)
        self.assertAlmostEqual(result['difference_mean'],3.5)
        self.assertAlmostEqual(result['SD_seed_differences'],np.sqrt(3.5))
        self.assertAlmostEqual(result['t'],np.sqrt(21))

    def test_direction_and_scale_preserve_two_sided_p(self):
        d=np.array([1.,2.,3.,4.,5.,6.]);a=periods.t_period(d);b=periods.t_period(-d);c=periods.t_period(17*d)
        self.assertAlmostEqual(a['p_raw'],b['p_raw'])
        self.assertAlmostEqual(a['CI95_lower'],-b['CI95_upper'])
        self.assertAlmostEqual(a['p_raw'],c['p_raw'])
        self.assertAlmostEqual(17*a['CI95_lower'],c['CI95_lower'])

    def test_complete_identity_admission_rejects_corruption(self):
        payload,freeze=self.fixture();self.assertEqual(len(periods.admit(payload,freeze)),81)
        bad=copy.deepcopy(payload);bad['predictors'].pop()
        with self.assertRaises(ValueError):periods.admit(bad,freeze)
        bad=copy.deepcopy(payload);bad['predictors'][-1]=bad['predictors'][0]
        with self.assertRaises(ValueError):periods.admit(bad,freeze)
        bad=copy.deepcopy(payload);bad['predictors'][0]['checkpoint_sha256']='b'*64
        with self.assertRaises(ValueError):periods.admit(bad,freeze)
        bad=copy.deepcopy(payload);bad['predictors'][0]['prediction_original'][0]=float('nan')
        with self.assertRaises(ValueError):periods.admit(bad,freeze)

    def test_zero_variance_is_not_a_significant_t(self):
        result=periods.t_period(np.full(6,.1))
        self.assertIsNone(result['p_raw']);self.assertIsNone(result['CI95_lower'])

    def test_sign_flip_exact_assignments_not_Monte_Carlo(self):
        self.assertEqual(periods.sign_flip([1,2,3,4,5,6])['p_raw'],2/64)
        self.assertEqual(periods.sign_flip([-3,-2,-1,1,2,3])['p_raw'],1)
        d=np.array([1.,0.,-1.,3.,4.,0.]);a=periods.sign_flip(d);b=periods.sign_flip(d[::-1])
        self.assertEqual(a['p_raw'],b['p_raw'])

    def test_Holm_family_does_not_shrink_for_undefined(self):
        rows=[{'p_raw':.003}]+[{'p_raw':1.} for _ in range(12)]
        self.assertAlmostEqual(periods.holm13(rows)[0]['p_Holm13'],.039)
        rows=[{'p_raw':.004}]+[{'p_raw':None} for _ in range(12)]
        result=periods.holm13(rows);self.assertAlmostEqual(result[0]['p_Holm13'],.052)
        self.assertIsNone(result[-1]['p_Holm13'])


if __name__=='__main__':unittest.main()
