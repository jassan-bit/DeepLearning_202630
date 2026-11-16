"""Pure admission and estimand checks; no checkpoints or market forwards."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_cpu_test_iut as cpu


class CPUTestIUTTests(unittest.TestCase):
    def fixture(self):
        arrays={'training_X':np.zeros((655,7,5),dtype=np.float32),'calibration_X':np.zeros((53,7,5),dtype=np.float32),
                'training_y_original':np.ones(655),'calibration_y_original':np.ones(53),
                'training_dates':np.array(['2024-01-01']*655),'calibration_dates':np.array(['2024-12-01']*53),
                'training_target_scale':np.array(.1),'training_channel_mean':np.zeros(5),'training_channel_std':np.ones(5),
                'evaluation_X':np.zeros((608,7,5),dtype=np.float32),'evaluation_y_original':np.ones(608)}
        support={'train_x_fp32_sha256':cpu.fp32_tensor_sha(arrays['training_X']),'calibration_x_fp32_sha256':cpu.fp32_tensor_sha(arrays['calibration_X']),
                 'train_y_original_float64_sha256':cpu.scalar_sha(arrays['training_y_original']),'calibration_y_original_float64_sha256':cpu.scalar_sha(arrays['calibration_y_original']),
                 'train_dates':arrays['training_dates'].tolist(),'calibration_dates':arrays['calibration_dates'].tolist(),
                 'scaler_record':{'target_scale':.1,'mean':[0.]*5,'std':[1.]*5}}
        return arrays,support

    def test_reuse_requires_exact_input_target_and_scaler_bytes(self):
        arrays,support=self.fixture();self.assertTrue(cpu.reuse_admission(arrays,support)['previous_train655_cal53_reuse_admitted'])
        arrays['calibration_X'][0,0,0]=1
        with self.assertRaises(ValueError):cpu.reuse_admission(arrays,support)
        arrays,support=self.fixture();arrays['training_y_original'][0]=2
        with self.assertRaises(ValueError):cpu.reuse_admission(arrays,support)
        arrays,support=self.fixture();arrays['training_channel_std'][0]=2
        with self.assertRaises(ValueError):cpu.reuse_admission(arrays,support)

    def test_normalized_dtype_and_support_are_not_silently_changed(self):
        with self.assertRaises(ValueError):cpu.fp32_tensor_sha(np.zeros((2,7,5),dtype=np.float64))
        arrays,support=self.fixture();arrays['evaluation_X']=np.zeros((607,7,5),dtype=np.float32)
        with self.assertRaises(ValueError):cpu.reuse_admission(arrays,support)

    def test_CPU_estimand_is_mean_individual_RMSE_not_ensemble(self):
        predictions=np.stack([np.tile(np.array([1.,-1.,2.,-2.,3.,-3.])[:,None],(1,608)),np.ones((6,608))])
        risks=cpu.cpu_mean_seed_rmse(predictions,np.zeros(608))
        np.testing.assert_allclose(risks,[2.,1.])
        self.assertEqual(float(np.sqrt(np.mean(predictions[0].mean(axis=0)**2))),0.)
        with self.assertRaises(ValueError):cpu.cpu_mean_seed_rmse(predictions[:,:5],np.zeros(608))


if __name__=='__main__':unittest.main()
