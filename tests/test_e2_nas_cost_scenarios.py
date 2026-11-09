import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e2_nas_cost_scenarios import scenario
class CostScenarios(unittest.TestCase):
 def test_guide_example(self):
  s=scenario(4);self.assertEqual(s['short_trajectories'],4000);self.assertEqual(s['full_trajectories'],220)
 def test_nested_and_reference(self):
  self.assertEqual(scenario(5,10)['full_trajectories'],2750)
  self.assertEqual(scenario(5,10,reference_already_selected=True)['full_trajectories'],2500)
 def test_no_time_claim(self):
  s=scenario(1);self.assertIsNone(s['measured_seconds']);self.assertIsNone(s['estimated_seconds']);self.assertFalse(s['includes_prior_HPO_final_refits_controls_or_ablation'])
 def test_invalid_parameters(self):
  for args in [(-1,), (True,), (1,0),(1,1,9),(1,1,10,4),(1,1,1001),(1,1,1000)]:
   with self.assertRaises(ValueError):scenario(*args)
 def test_full_enumeration(self):
  self.assertEqual(scenario(1,finalists=1000,reference_already_selected=True)['full_trajectories'],5000)

 def test_no_architecture_search_scenario(self):
  s=scenario(0);self.assertEqual(s['short_trajectories'],0);self.assertEqual(s['full_trajectories'],0)
  self.assertFalse(s['includes_prior_HPO_final_refits_controls_or_ablation'])
