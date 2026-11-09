import json,hashlib,re,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'references/e2_recipe_cards_20261109'
class RecipeCards(unittest.TestCase):
 def setUp(self):self.a=json.loads((D/'cards.json').read_text())
 def test_eleven_unique_concrete_proposals(self):
  rows=self.a['cards'];self.assertEqual(len(rows),11);self.assertEqual(len({x['recipe_id'] for x in rows}),11)
  self.assertEqual([x['head_input_features'] for x in rows],[128,128,128,26,42,45,50,90,170,1290,32])
  self.assertEqual([x['same_window_moments'] for x in rows],[0,0,0,0,10,10,10,10,10,10,0])
 def test_no_training_or_granted_exemption(self):
  self.assertEqual(self.a['new_model_fits'],0);self.assertFalse(self.a['model_implementation_changes']);self.assertIsNone(self.a['m_NAS_selected'])
  for row in self.a['cards']:self.assertIsNone(row['NAS_applies']);self.assertFalse(row['exemption_granted'])
 def test_code_preserved(self):
  for path,h in self.a['preserved_code_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),h)
 def test_public_allowlist_and_book(self):
  m=json.loads((D/'manifest.json').read_text());dest=ROOT/'book_e2/_static/recipe_cards_20261109'
  self.assertEqual(set(m['allowlist']),{'cards.json','cards.md'})
  self.assertEqual({p.name for p in dest.iterdir()},{'cards.json','cards.md','manifest.json'})
  for n,h in m['files'].items():
   self.assertEqual(hashlib.sha256((D/n).read_bytes()).hexdigest(),h);self.assertEqual((D/n).read_bytes(),(dest/n).read_bytes())
  self.assertEqual((D/'cards.md').read_bytes(),(ROOT/'book_e2/sections/22_fichas_recetas.md').read_bytes())
 def test_source_limits_and_clean_urls(self):
  rows=self.a['cards'];self.assertFalse(rows[3]['paper_body_reverified']);self.assertEqual(rows[3]['reconstruction_phases'],2)
  self.assertFalse(rows[10]['exact_combination_primary_source_verified'])
  for row in rows[:10]:
   self.assertTrue(row['source_urls'])
   for url in row['source_urls']:self.assertNotRegex(url,r'\s')
 def test_cost_is_illustrative(self):
  self.assertEqual(4*2*10*2*2,320);self.assertEqual(14*4*2*10*2+320,2560)
  self.assertIn('No es el presupuesto final conforme a la guía',(D/'cards.md').read_text())
