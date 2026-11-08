import importlib.util,json,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('recovery',ROOT/'scripts/build_e2_recovery_second.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RecoveryTests(unittest.TestCase):
 def test_validate(self):self.assertEqual(m.validate()['examined_cumulative'],22)
 def test_versions(self):
  s=m.validate();self.assertEqual(s['preprint_fulltext_ids'],['E2-0285','E2-0867']);self.assertNotIn('E2-0867',s['editorial_fulltext_ids'])
 def test_denominator(self):
  s=m.validate();self.assertEqual(s['editorial_fulltext_cumulative']+s['preprint_fulltext_editorial_version_pending']+s['partial_cumulative']+s['unsearched'],40)
 def test_false_closure_rejected(self):
  original=Path.read_text
  def changed(p,*a,**kw):
   t=original(p,*a,**kw)
   if p==m.SOURCE/'summary.json':
    s=json.loads(t);s['global_review_complete']=True;t=json.dumps(s)
   return t
  with patch.object(Path,'read_text',changed):
   with self.assertRaises(AssertionError):m.validate()
if __name__=='__main__':unittest.main()
