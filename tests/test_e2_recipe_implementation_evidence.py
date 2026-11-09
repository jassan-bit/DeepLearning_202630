import hashlib,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'experiments/e2_recipe_synthetic_20261109'
class Evidence(unittest.TestCase):
 def test_hashes_mirror_and_scope(self):
  subprocess.run([sys.executable,str(ROOT/'scripts/publish_e2_recipe_implementation.py'),'--check'],check=True,capture_output=True)
 def test_preserve_historical_code(self):
  cards=json.loads((ROOT/'references/e2_recipe_cards_20261109/cards.json').read_text())
  for n,h in cards['preserved_code_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),h)
 def test_steps_phases_and_full_dimensions(self):
  r=json.loads((D/'synthetic_receipt.json').read_text());self.assertEqual(sum(p['steps'] for m in r['models'] for p in m['phases'].values()),13)
  self.assertEqual([x['parameter_count'] for x in r['models'][:4]],[268929,522955,490561,8739]);self.assertFalse(r['training_contract']['structural_resize_for_smoke'])
 def test_docs_test_counts_and_pending_scope(self):
  s=json.loads((D/'summary.json').read_text());self.assertEqual(sum(x['tests'] for x in s['test_suites'].values()),39);self.assertEqual(sum(x['skipped'] for x in s['test_suites'].values()),0)
  t=(ROOT/'book_e2/sections/23_implementacion_recetas.md').read_text();self.assertIn('13 pasos',t);self.assertIn('39 pruebas nuevas',t);self.assertIn('NAS',t)
if __name__=='__main__':unittest.main()
