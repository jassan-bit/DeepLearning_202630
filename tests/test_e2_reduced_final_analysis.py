import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import e2_reduced_final_analysis as analysis


class FinalAnalysisTests(unittest.TestCase):
    def test_primary_metric_is_mean_individual_seed_rmse(self):
        y=np.array([1.,1.]); p=np.array([[0.,0.],[2.,2.]])
        result=analysis.metrics(y,p)
        self.assertEqual(result['RMSE'],1.)
        self.assertFalse(result['primary_is_ensemble_RMSE'])
        self.assertEqual(float(np.sqrt(np.mean((p.mean(axis=0)-y)**2))),0.)

    def test_residual_quantile_and_predictor_binding(self):
        dates=[(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]
        calibrated=analysis.residual_calibrator(np.ones(53),np.arange(53)/100+1,dates=dates,predictor_sha256='a'*64)
        self.assertEqual(calibrated['rank'],49)
        self.assertAlmostEqual(calibrated['q'],.48)
        lo,hi=analysis.prediction_intervals(np.array([.1]),calibrated)
        self.assertEqual(lo[0],0.)
        self.assertAlmostEqual(hi[0],.58)
        changed=dict(calibrated,q=1.)
        with self.assertRaisesRegex(ValueError,'checksum'):
            analysis.prediction_intervals(np.array([.1]),changed)

    def test_calibration_refuses_reordered_or_final_dates(self):
        dates=[(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]
        with self.assertRaises(ValueError):
            analysis.residual_calibrator(np.ones(53),np.ones(53),dates=dates[::-1],predictor_sha256='a'*64)

    def test_holm_monotone_and_no_extra_test_count(self):
        rows=analysis.holm([{'p_one_sided':.01},{'p_one_sided':.04},{'p_one_sided':.2}])
        self.assertAlmostEqual(rows[0]['p_Holm'],.03)
        self.assertAlmostEqual(rows[1]['p_Holm'],.08)
        self.assertTrue(rows[0]['reject_at_FWER05'])
        self.assertFalse(rows[1]['reject_at_FWER05'])
        registration=analysis.analysis_registration()
        self.assertEqual(len(registration['diagnostics'])+1,3)
        self.assertEqual(registration['family_size'],20)

    def test_bootstrap_preserves_paired_seed_and_temporal_information(self):
        y=np.linspace(.01,.03,40)
        baseline=np.array([y+.008+i*.0001 for i in range(6)])
        good=np.array([y+.001+i*.0001 for i in range(6)])
        result=analysis.bootstrap_contrast(y,good,baseline,draws=99,seed=44)
        reverse=analysis.bootstrap_contrast(y,baseline,good,draws=99,seed=44)
        self.assertAlmostEqual(result['gain'],.007)
        self.assertLessEqual(result['p_one_sided'],.02)
        self.assertEqual(reverse['p_one_sided'],1.)
        self.assertEqual(result,analysis.bootstrap_contrast(y,good,baseline,draws=99,seed=44))

    def test_pilot_requires_two_representatives_and_exactly_five_seeds(self):
        y=np.linspace(.01,.03,53)
        data={'iTransformer':np.array([y+.001+i*.0001 for i in range(5)]),
              'GRU32':np.array([y+.002+i*.0003 for i in range(5)])}
        dates=[(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]
        result=analysis.pilot_power_mde(y,data,dates=dates)
        self.assertEqual(result['R_operative'],6)
        self.assertFalse(result['automatic_R_increase'])
        self.assertGreaterEqual(result['MDE_excess_above_delta_at_80pct'],0.)
        data['iTransformer']=data['iTransformer'][:4]
        with self.assertRaises(ValueError):analysis.pilot_power_mde(y,data,dates=dates)


if __name__=='__main__': unittest.main()
