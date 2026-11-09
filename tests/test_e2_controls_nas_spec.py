import hashlib,json,unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from e2_nas_cost_scenarios import scenario
D=ROOT/'references/e2_controls_nas_20261108'
class ControlsSpec(unittest.TestCase):
 def test_orders_and_budgets(self):
  c=json.loads((D/'controls.json').read_text());self.assertEqual(len({tuple(x) for x in c['ARIMA_orders']}),18)
  self.assertEqual(c['ARIMA_global_selection_fits']['total'],18*10+2)
  self.assertEqual(c['ARIMA_nested_selection_fits']['total'],(18+1)*12)
  self.assertTrue(c['scenarios_mutually_exclusive']);self.assertEqual(c['new_model_fits'],0);self.assertIsNone(c['measured_seconds'])
  self.assertFalse(c['seasonality_claimed']);self.assertFalse(c['common_HPO_budget_closed'])
 def test_cost_scenarios_regenerate(self):
  c=json.loads((D/'cost_scenarios.json').read_text());self.assertEqual(c['scenarios'],[scenario(m) for m in [0,1,4,5,11]]+[scenario(5,folds=10)])
  self.assertTrue(c['m_not_adjudicated']);self.assertFalse(c['timings_measured'])

 def test_nas_decisions_not_fabricated(self):
  a=json.loads((D/'nas_matrix.json').read_text());self.assertEqual(len(a['models']),11)
  self.assertEqual(len({x['model'] for x in a['models']}),11);self.assertIsNone(a['m_selected'])
  self.assertEqual(a['new_model_fits'],0)
  for row in a['models']:self.assertIsNone(row['NAS_applies']);self.assertFalse(row['exemption_granted'])
 def test_public_allowlist_mirror_and_hashes(self):
  a=json.loads((D/'manifest.json').read_text());expected={'controls.md','controls.json','nas.md','nas_matrix.json','cost_scenarios.json'}
  self.assertEqual(set(a['public_allowlist']),expected);self.assertEqual(set(a['files']),expected)
  dest=ROOT/'book_e2/_static/controls_nas_20261108'
  self.assertEqual({p.name for p in dest.iterdir()},expected|{'manifest.json'})
  for n,h in a['files'].items():
   self.assertEqual(hashlib.sha256((D/n).read_bytes()).hexdigest(),h)
   self.assertEqual((D/n).read_bytes(),(dest/n).read_bytes())
  self.assertEqual(a['cost_script_sha256'],hashlib.sha256((ROOT/'scripts/e2_nas_cost_scenarios.py').read_bytes()).hexdigest())
