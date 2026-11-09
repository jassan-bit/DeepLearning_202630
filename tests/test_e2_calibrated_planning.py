"""Small synthetic checks only; no model fits or production MC simulation."""
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import binom

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e2_calibrated_planning as planning


def artificial_pilot(models=2):
    time=np.arange(53)
    truth=.01+.001*np.sin(time/9)
    prediction=[]
    for model in range(models):
        rows=[]
        for seed in range(5):
            error=(.001+.00015*model+.000018*seed*(model+1)
                   +.00012*np.sin(time/(3+model)+seed*.11)
                   +.00004*np.cos(time/7+model))
            rows.append(truth+error)
        prediction.append(rows)
    dates=[(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]
    return np.asarray(prediction),truth,dates


def small_config(**overrides):
    values=dict(calibration_draws=700,validation_draws=800,alternative_draws=900,
                alpha=.2,R_values=(6,),batch_size=67)
    values.update(overrides)
    return planning.CalibrationConfig(**values)


class CalibratedPlanningTests(unittest.TestCase):
    def test_registered_binomial_rank_and_exact_cp_boundaries(self):
        count=200000
        target=.05/153/2
        # Deterministic order-statistic fixture, not 200000 DGP simulations.
        values=np.arange(count,dtype=np.float64)/count
        for pairs,k,rank,max_val,min_alt in ((1,21,199980,47,160447),
                                              (3,19,199982,44,160513)):
            gamma=.05/(4*pairs)
            result=planning.calibrate_abs_threshold(values,target,gamma)
            self.assertEqual(result["tail_order_k"],k)
            self.assertEqual(result["order_rank_1based"],rank)
            self.assertLessEqual(binom.cdf(k-1,count,target),gamma)
            self.assertGreater(binom.cdf(k,count,target),gamma)
            self.assertLessEqual(planning.clopper_pearson_upper(max_val,count,gamma),.05/153)
            self.assertGreater(planning.clopper_pearson_upper(max_val+1,count,gamma),.05/153)
            power_gamma=.05/(8*pairs)
            self.assertGreaterEqual(planning.clopper_pearson_lower(min_alt,count,power_gamma),.8)
            self.assertLess(planning.clopper_pearson_lower(min_alt-1,count,power_gamma),.8)

    def test_actual_uniform_null_type_i_and_independent_validation(self):
        # Uniform |noise| has exactly known tail probability. The threshold is
        # trained on one RNG and assessed on a second, never on its own draws.
        calibration=np.random.default_rng(7).uniform(-1,1,8000)
        result=planning.calibrate_abs_threshold(calibration,.025,.01)
        true_tail=1-result["critical_abs"]
        self.assertLessEqual(true_tail,.025)
        validation=np.random.default_rng(8).uniform(-1,1,9000)
        assessed=planning.validate_null(validation,result["critical_abs"],.05,.01)
        self.assertTrue(assessed["validated"])
        self.assertFalse(assessed["critical_value_retuned"])
        self.assertNotEqual(planning._array_sha(calibration),planning._array_sha(validation))

    def test_atoms_strict_rejection_and_insufficient_tail_draws(self):
        threshold=planning.calibrate_abs_threshold(np.ones(1000),.1,.01)
        result=planning.validate_null(np.ones(1000),threshold["critical_abs"],.1,.01)
        self.assertEqual(result["hits"],0)
        self.assertEqual(threshold["critical_abs"],1)
        missing=planning.calibrate_abs_threshold(np.arange(20.),.05/306,.0125)
        self.assertIsNone(missing["critical_abs"])
        self.assertEqual(missing["status"],"insufficient_calibration_tail_draws")

    def test_paired_streams_chunk_invariance_and_no_global_rng_mutation(self):
        prediction,truth,_=artificial_pilot()
        squared=(prediction-truth)**2
        score=np.sqrt(squared.mean(axis=-1))
        pair={"squared_errors":squared,"seed_residual_inflation":2.1,
              "original_mean_difference":float((score[0]-score[1]).mean())}
        np.random.seed(713)
        expected=np.random.random(4)
        np.random.seed(713)
        first=planning._sample_noise(pair,6,317,272401,0,17)
        second=planning._sample_noise(pair,6,317,272401,0,89)
        self.assertTrue(np.array_equal(first,second))
        self.assertTrue(np.array_equal(np.random.random(4),expected))
        for pool,pair_id,R in ((272402,0,6),(272403,0,6),(272401,1,6),(272401,0,10)):
            other=planning._sample_noise(pair,R,317,pool,pair_id,17)
            self.assertFalse(np.array_equal(first,other))
        # Identical models and corresponding seed IDs cancel exactly even
        # though their calendar errors and seed errors vary independently.
        pair["squared_errors"]=np.stack((squared[0],squared[0]))
        pair["original_mean_difference"]=0.
        self.assertTrue(np.array_equal(planning._sample_noise(pair,6,100,12,0,17),np.zeros(100)))

    def test_fixed_center_keeps_temporal_floor_as_R_grows(self):
        prediction,truth,_=artificial_pilot()
        squared=(prediction-truth)**2
        pair={"squared_errors":squared,"seed_residual_inflation":0.,
              "original_mean_difference":0.}
        # With zero seed residual contribution, R cannot divide the temporal
        # noise by sqrt(R). Same calendar namespace supplies the same draws.
        with patch.object(planning.np.random,"SeedSequence",side_effect=lambda entropy: entropy[:1]+entropy[2:]):
            six=planning._sample_noise(pair,6,100,17,0,13)
            thirty=planning._sample_noise(pair,30,100,17,0,13)
        self.assertTrue(np.array_equal(six,thirty))
        self.assertGreater(np.std(six),0)
        self.assertGreater(abs(np.mean(six)),1e-6)  # no per-stream recentering

    def test_complete_two_and_three_model_analysis_and_final_R_gate(self):
        for models in (2,3):
            prediction,truth,dates=artificial_pilot(models)
            result=planning.calibrated_power_analysis(prediction,truth,.01,m=1,dates=dates,
                         config=small_config(),seed_ids=[1,2,3,4,5])
            self.assertEqual(len(result["pairs"]),models*(models-1)//2)
            self.assertEqual(result["conditionally_supported_R"],6)
            self.assertIsNone(result["final_R"])
            self.assertIsNone(result["achieved_final_power"])
            self.assertFalse(result["registered_design_match"])
            self.assertFalse(result["joint_three_family_confidence_claimed"])
            for pair in result["pairs"]:
                cell=pair["cells"][0]
                self.assertTrue(cell["null_validated"])
                self.assertTrue(cell["supports_conditional_target"])
                self.assertGreater(cell["minimum_direction_CP_lower"],.8)
                self.assertNotEqual(cell["calibration_noise_sha256"],cell["validation_noise_sha256"])
                self.assertNotEqual(cell["validation_noise_sha256"],cell["alternative_noise_sha256"])
                self.assertEqual(cell["draws_executed"],{"calibration":700,"validation":800,"alternative":900})

    def test_validation_failure_never_tunes_threshold_or_admits_power(self):
        prediction,truth,dates=artificial_pilot()
        config=small_config()
        baseline=planning.calibrated_power_analysis(prediction,truth,.01,m=1,dates=dates,config=config)
        original=planning._sample_noise
        def poison_validation(pair,R,count,pool,pair_index,batch_size):
            if pool==config.validation_pool:return np.ones(count)
            return original(pair,R,count,pool,pair_index,batch_size)
        with patch.object(planning,"_sample_noise",side_effect=poison_validation):
            result=planning.calibrated_power_analysis(prediction,truth,.01,m=1,dates=dates,config=config)
        old=baseline["pairs"][0]["cells"][0]
        cell=result["pairs"][0]["cells"][0]
        self.assertEqual(old["calibration"],cell["calibration"])
        self.assertEqual(old["calibration_noise_sha256"],cell["calibration_noise_sha256"])
        self.assertFalse(cell["null_validated"])
        self.assertFalse(cell["supports_conditional_target"])
        self.assertIsNone(result["conditionally_supported_R"])
        self.assertEqual(result["resource_envelope_R"],30)
        self.assertFalse(cell["alternatives"]["plus_delta"]["interpretation_admitted_by_null_validation"])

    def test_seed_or_DGP_degeneracy_cannot_claim_perfect_power(self):
        prediction,truth,dates=artificial_pilot()
        same=np.stack((prediction[0],prediction[0]))
        result=planning.calibrated_power_analysis(same,truth,.01,m=1,dates=dates,config=small_config())
        self.assertEqual(result["pairs"][0]["status"],"not_estimable")
        self.assertIsNone(result["conditionally_supported_R"])
        with patch.object(planning,"_sample_noise",side_effect=lambda p,R,n,*args:np.zeros(n)):
            result=planning.calibrated_power_analysis(prediction,truth,.01,m=1,dates=dates,config=small_config())
        cell=result["pairs"][0]["cells"][0]
        self.assertEqual(cell["status"],"conditional_DGP_zero_or_unresolvable_variance")
        self.assertIsNone(cell["alternatives"])
        self.assertIsNone(result["conditionally_supported_R"])

    def test_invalid_geometry_dates_nonfinite_and_colliding_pools(self):
        prediction,truth,dates=artificial_pilot()
        with self.assertRaises(ValueError):planning.CalibrationConfig(validation_pool=272401)
        with self.assertRaises(ValueError):planning.CalibrationConfig(block_length=14)
        with self.assertRaises(ValueError):
            planning.calibrated_power_analysis(prediction[:,:4],truth,.01,dates=dates,config=small_config())
        bad=prediction.copy();bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            planning.calibrated_power_analysis(bad,truth,.01,dates=dates,config=small_config())
        for bad_dates in (dates[:1]+dates[:-1],["2025-01-01"]*53):
            with self.assertRaises(ValueError):
                planning.calibrated_power_analysis(prediction,truth,.01,dates=bad_dates,config=small_config())
        with self.assertRaises(ValueError):
            planning.calibrated_power_analysis(prediction,truth,.01,dates=dates,
                config=replace(small_config(),validation_draws=100000000))


if __name__=="__main__":unittest.main()
