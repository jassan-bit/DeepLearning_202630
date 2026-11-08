"""Aggregate-only progress checks; private source tables are not required."""
import importlib.util,json,tempfile,shutil,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('progress',ROOT/'scripts/build_e2_review_progress.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ProgressTests(unittest.TestCase):
 def test_counts_and_limits(self):
  s=m.validate();self.assertEqual(s['exclusion_audit']['not_individually_reviewed'],295);self.assertIsNone(s['final_fulltext_inclusions'])
 def test_false_closure_rejected(self):
  old=m.SOURCE
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'copy';shutil.copytree(old,p);s=json.loads((p/'summary.json').read_text());s['review_complete']=True;(p/'summary.json').write_text(json.dumps(s));m.SOURCE=p
   try:
    with self.assertRaises(AssertionError):m.validate()
   finally:m.SOURCE=old
 def test_extra_file_rejected(self):
  old=m.SOURCE
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'copy';shutil.copytree(old,p);(p/'abstracts.txt').write_text('not public');m.SOURCE=p
   try:
    with self.assertRaises(AssertionError):m.validate()
   finally:m.SOURCE=old
 def test_no_private_table_fields(self):
  s=json.loads((m.SOURCE/'summary.json').read_text());text=json.dumps(s)
  for field in ['evidencia_resumen','evidencia_titulo','razon_individual','abstract']:
   self.assertNotIn('"'+field+'"',text)
if __name__=='__main__':unittest.main()
