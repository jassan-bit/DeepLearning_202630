import json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class GuideAlignment(unittest.TestCase):
 def test_guide_record_and_mirror(self):
  p=ROOT/'references/e2_guide_alignment_20261108.json';a=json.loads(p.read_text())
  self.assertEqual(p.read_bytes(),(ROOT/'book_e2/_static/e2_guide_alignment_20261108.json').read_bytes())
  self.assertEqual(a['source']['pages'],65);self.assertEqual(a['catalogue_models'],10)
  self.assertEqual(a['current_route'],'F2a');self.assertEqual(a['minimum_comparison_systems_proposed_with_Ridge'],18)
  for key in ['original_reduced_grid_satisfies_guide','architecture_search_decisions_closed','full_guide_compliance_claimed']:self.assertFalse(a[key])
  self.assertEqual(a['new_model_fits'],0);self.assertEqual(a['new_HPO_trials'],0)
 def test_prior_budget_preserved_and_qualified(self):
  a=json.loads((ROOT/'experiments/e2_plan_audit_20261108/plan.json').read_text())
  self.assertEqual(a['total_comparators'],16)
  s=(ROOT/'book_e2/sections/19_plan_auditoria.md').read_text()
  self.assertIn('insuficiente para el espacio mínimo',s);self.assertIn('al menos 18 sistemas',s)
 def test_navigation_and_cautions(self):
  self.assertIn('sections/20_contraste_guia',(ROOT/'book_e2/_toc.yml').read_text())
  s=(ROOT/'book_e2/sections/20_contraste_guia.md').read_text()
  for t in ['p23','p47','p30','1.000','mejor modelo','no convierte retrospectivamente']:self.assertIn(t,s)
