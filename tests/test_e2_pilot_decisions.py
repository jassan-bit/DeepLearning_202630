"""Statistical pilot decisions, using deterministic small artificial score panels."""
from copy import deepcopy
from datetime import date,timedelta
import json
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import numpy as np
import e2_search_plan as planner
from e2_pilot_decisions import decide_k,decide_R,budget_report,conditional_power_analysis

PROTOCOL=Path(__file__).resolve().parents[1]/"book_e2/_static/search_protocol_20261109/protocol.json"


class PilotDecisionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads(PROTOCOL.read_text())
    def noise_rows(self):
        rows=[]
        for model in [row['model'] for row in self.protocol['models'] if row['nas']]:
            for index,pattern in enumerate(planner.noise_pilot_architectures(self.protocol)):
                for seed in self.protocol['budget']['noise_pilot_seed_values']:
                    rows.append({'model':model,'pattern_id':pattern['pattern_id'],'seed':seed,
                                 'status':'completed','validation_rmse':.001+index*.0001})
        return rows

    def test_complete_stable_registered_rank_rule_selects20(self):
        result=decide_k(self.protocol,self.noise_rows())
        self.assertEqual(result['k_recommendation'],20)
        self.assertTrue(result['full_rank_rule_evaluated'])
        self.assertEqual(result['completed_rows'],360)
        self.assertTrue(all(value['median_pairwise_spearman']==1 for value in result['per_model'].values()))

    def test_missing_or_censored_is_conservative40_without_faking_the_rank_rule(self):
        rows=self.noise_rows()
        result=decide_k(self.protocol,rows[:-1])
        self.assertEqual(result['k_recommendation'],40)
        self.assertFalse(result['full_rank_rule_evaluated'])
        self.assertEqual(result['missing_rows'],1)
        rows[-1]['status']='censored';rows[-1].pop('validation_rmse')
        result=decide_k(self.protocol,rows)
        self.assertEqual(result['unfinished_rows'],1)
        self.assertFalse(result['full_rank_rule_evaluated'])

    def test_nonestimable_constant_rank_is40_even_with_every_fit_complete(self):
        rows=self.noise_rows()
        model=rows[0]['model'];seed=rows[0]['seed']
        for row in rows:
            if row['model']==model and row['seed']==seed:row['validation_rmse']=.01
        result=decide_k(self.protocol,rows)
        self.assertTrue(result['full_rank_rule_evaluated'])
        self.assertEqual(result['k_recommendation'],40)
        self.assertIsNone(result['per_model'][model]['median_pairwise_spearman'])

    def test_noise_retries_foreign_seeds_and_nonfinite_success_are_rejected(self):
        rows=self.noise_rows()
        with self.assertRaises(ValueError):decide_k(self.protocol,rows+[deepcopy(rows[0])])
        row=deepcopy(rows[0]);row['seed']=42
        with self.assertRaises(ValueError):decide_k(self.protocol,[row])
        row=deepcopy(rows[0]);row['validation_rmse']=float('nan')
        with self.assertRaises(ValueError):decide_k(self.protocol,[row])

    def test_original_R_remains_blocked_when_reference_pilot_exists(self):
        result=decide_R(self.protocol,reference_analysis={'reference_pilot_R_recommendation':30})
        self.assertIsNone(result['R_selected'])
        self.assertIsNone(result['achieved_power'])
        self.assertEqual(result['reference_pilot_R_proposal'],30)
        self.assertEqual([row['R'] for row in result['R_scenarios']],[6,10,20,30])
        self.assertGreaterEqual(len(result['blocking_reasons']),3)

    def test_allowances_are_not_mislabeled_as_measured_total_hours(self):
        report=budget_report(self.protocol,40,30,resource_runs=[{'status':'success'},{'status':'timeout'}])
        self.assertEqual(report['phase_counts']['short_phase_total'],88096)
        self.assertEqual(report['phase_counts']['full_phase_total'],30159)
        self.assertEqual(report['attempt_time_allowances']['full_phase_allowance_seconds'],30159*7200)
        self.assertIsNone(report['total_campaign_estimated_hours'])
        self.assertIsNone(report['measured_conditional_projection'])
        self.assertFalse(report['attempt_time_allowances']['classical_timing_included'])

    def power_data(self,seeds=5,days=53):
        rng=np.random.default_rng(10)
        truth=.02+.003*np.sin(np.arange(days)/5)
        predictions=truth[None,None,:]+.001+rng.normal(0,.0006,size=(3,seeds,days))
        dates=[str(date(2024,11,8)+timedelta(days=index)) for index in range(days)]
        return predictions,truth,dates

    def test_conditional_power_reports_all_pairs_and_never_achieved_final_power(self):
        predictions,truth,dates=self.power_data()
        result=conditional_power_analysis(predictions,truth,.00025772842664981106,
            dates=dates,draws=1000,seed=99,seed_ids=list(range(271901,271906)))
        self.assertEqual(len(result['pairs']),3)
        self.assertIsNone(result['achieved_final_power'])
        self.assertEqual(result['multiplicity_family_size'],153)
        self.assertIn(result['reference_pilot_R_recommendation'],range(6,31))
        for pair in result['pairs']:
            self.assertEqual(pair['status'],'conditional_planning_only')
            self.assertGreater(pair['sigma_d_upper95_normal_assumption'],pair['sigma_d'])
            self.assertEqual([row['R'] for row in pair['scenarios']],[6,10,20,30])
            self.assertEqual(len(pair['all_R_simulation_results']),25)
            for row in pair['scenarios']:
                self.assertGreater(row['conditional_MDE_normal'],0)
                self.assertGreater(row['conditional_MDE_simulated'],0)
                self.assertTrue(0<=row['conditional_simulated_rejection_probability']<=1)
                self.assertTrue(0<=row['Monte_Carlo_CI95'][0]<=row['Monte_Carlo_CI95'][1]<=1)
        json.dumps(result,allow_nan=False)

    def test_simulation_is_reproducible_without_changing_global_numpy_RNG(self):
        predictions,truth,dates=self.power_data()
        before=np.random.get_state()
        one=conditional_power_analysis(predictions,truth,.00025,dates=dates,draws=1000,seed=42)
        two=conditional_power_analysis(predictions,truth,.00025,dates=dates,draws=1000,seed=42)
        self.assertEqual(one,two)
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])

    def test_zero_seed_variability_is_not_perfect_power(self):
        predictions,truth,dates=self.power_data()
        predictions[:]=predictions[:,:1,:]
        result=conditional_power_analysis(predictions,truth,.00025,dates=dates,draws=1000)
        self.assertEqual(result['reference_pilot_R_recommendation'],30)
        self.assertFalse(result['all_observed_pilot_pairs_meet_conditional_target_by30'])
        self.assertTrue(all(row['status']=='not_estimable' for row in result['pairs']))
        self.assertTrue(all(row['normal_R_approximation'] is None for row in result['pairs']))

    def test_incomplete_seed_support_is_reported_as_insufficient(self):
        predictions,truth,dates=self.power_data(seeds=4)
        result=conditional_power_analysis(predictions,truth,.00025,dates=dates,draws=1000)
        self.assertEqual(result['status'],'insufficient_pilot_support')
        self.assertIsNone(result['reference_pilot_R_recommendation'])

    def test_nonfinite_and_compressed_or_final_dates_are_rejected(self):
        predictions,truth,dates=self.power_data()
        bad=predictions.copy();bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):conditional_power_analysis(bad,truth,.00025,dates=dates,draws=1000)
        compressed=dates.copy();compressed[20]=str(date.fromisoformat(compressed[20])+timedelta(days=1))
        with self.assertRaises(ValueError):conditional_power_analysis(predictions,truth,.00025,dates=compressed,draws=1000)
        final=[str(date(2025,1,1)+timedelta(days=i)) for i in range(53)]
        with self.assertRaises(ValueError):conditional_power_analysis(predictions,truth,.00025,dates=final,draws=1000)
        with self.assertRaises(ValueError):conditional_power_analysis(predictions,truth,.00025,dates=None,draws=1000)


if __name__=='__main__':unittest.main()
