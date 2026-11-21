"""Generation/contract guards; not calibration estimates or market evidence."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import validate_controlled as v


class Guards(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads((Path(__file__).parent/'CONTROLLED_VALIDATION_PLAN.json').read_text())

    def test_truth_comes_from_population_bases_not_realization(self):
        case=self.plan['Monte_Carlo']['cases'][1]
        q,truth,bases=v.make_squared_errors(case,np.random.default_rng(43))
        np.testing.assert_allclose(truth,np.full(19,.03),atol=1e-15)
        estimate=np.sqrt(q.mean(axis=1)).mean(axis=0)
        self.assertGreater(np.max(np.abs(estimate[1:]-estimate[0]-truth)),.001)
        self.assertEqual(q.shape,(12,128,20))
        self.assertTrue(np.isfinite(q).all() and (q>=0).all())

    def test_same_stream_exactly_recreates_q_and_truth(self):
        case=self.plan['Monte_Carlo']['cases'][3]
        one=v.make_squared_errors(case,np.random.default_rng(73))
        two=v.make_squared_errors(case,np.random.default_rng(73))
        for a,b in zip(one,two):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(one[1][:9],.03)
        np.testing.assert_allclose(one[1][9:],.23)

    def test_seed_only_has_no_calendar_variability(self):
        case=next(x for x in self.plan['Monte_Carlo']['cases'] if x['id']=='seed_only')
        q,truth,_=v.make_squared_errors(case,np.random.default_rng(3))
        self.assertTrue(np.array_equal(q[:,0,:],q[:,127,:]))
        self.assertGreater(np.std(q[:,0,0]),0)

    def test_calendar_only_has_no_initialization_variability(self):
        case=next(x for x in self.plan['Monte_Carlo']['cases'] if x['id']=='calendar_only')
        q,truth,_=v.make_squared_errors(case,np.random.default_rng(3))
        self.assertTrue(np.array_equal(q[0],q[11]))
        self.assertGreater(np.std(q[0,:,0]),0)

    def test_CP_extremes_and_counts(self):
        self.assertEqual(v.CP(0,10)['CI95'][0],0)
        self.assertEqual(v.CP(10,10)['CI95'][1],1)
        self.assertIsNone(v.CP(0,0)['rate'])
        result=v.event_summary([False]*10,.05)
        self.assertFalse(result['simultaneous_validation_claimed'])

    def test_plan_count_and_three_test_gate(self):
        self.assertEqual(sum(c['mc_replicates'] for c in self.plan['Monte_Carlo']['cases']),1024)
        self.assertFalse(self.plan['three_tests_per_contrast_gate'])
        self.assertFalse(self.plan['Monte_Carlo']['final_market_power_certified'])


if __name__=='__main__':unittest.main()
