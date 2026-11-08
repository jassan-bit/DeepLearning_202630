import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class AlignmentTests(unittest.TestCase):
 def setUp(self):self.p=ROOT/'references/books_alignment_20261108.json';self.s=json.loads(self.p.read_text())
 def test_candidate_is_proposed(self):
  self.assertEqual(self.s['candidate'],'CNN-1D + LSTM + atención temporal')
  for k in ['candidate_implemented','candidate_trained','superiority_demonstrated','new_experimental_design_registered','e3_official_guide_received']:self.assertFalse(self.s[k])
 def test_mirrors(self):
  for b in ['book','book_e2','book_e3']:self.assertEqual((ROOT/b/'_static/books_alignment_20261108.json').read_bytes(),self.p.read_bytes())
 def test_source_fingerprints(self):
  for f,h in self.s['source_fingerprints'].items():self.assertEqual(hashlib.sha256((ROOT/f).read_bytes()).hexdigest(),h,f)
 def test_e1_reference_not_transferred(self):
  self.assertEqual(self.s['e1_references'],['GRU32','LSTM32','MLP16']);self.assertFalse(self.s['e1_R_transferable_to_candidate']);self.assertEqual(self.s['reference_calendar']['evaluation_dates'],608)
 def test_ports_identify_candidate(self):
  for b in ['book','book_e2','book_e3']:self.assertIn(self.s['candidate'],(ROOT/b/'intro.md').read_text())
 def test_active_catalogue_matches(self):
  s=json.loads((ROOT/'references/e2_current_catalogue_20261108.json').read_text());self.assertEqual(s['candidate_11'],self.s['candidate']);self.assertEqual(len(s['models']),10)
if __name__=='__main__':unittest.main()
