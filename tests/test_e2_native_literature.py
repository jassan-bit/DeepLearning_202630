"""Public bibliography checks, requiring no source exports or network."""
import importlib.util
import unittest
from pathlib import Path
import tempfile
import shutil
import csv

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e2_native',ROOT/'scripts/publish_e2_native_literature.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PublicEvidenceTests(unittest.TestCase):
 def test_counts_and_provisional_states(self):
  result=module.validate()
  self.assertFalse(result['final_eligibility_closed'])
  self.assertEqual(result['native_groups'],1078)
  self.assertEqual(result['arxiv_ids'],1004)
 def test_private_column_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'records.csv';p.write_text('title,abstract\na,b\n')
   with self.assertRaises(AssertionError):module.read_csv(p)
 def test_extra_file_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'copy';shutil.copytree(module.SOURCE,p)
   (p/'private.txt').write_text('unexpected')
   with self.assertRaises(AssertionError):module.validate(p)
 def test_source_scripts_guard_public_checkout(self):
  for name in ['cribar.py','crear_derivados_publicos.py']:
   self.assertIn('Run only from a private directory outside Git',(module.SOURCE/'native'/name).read_text())

if __name__=='__main__':unittest.main()
