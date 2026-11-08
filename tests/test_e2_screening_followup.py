import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('followup', ROOT/'scripts/build_e2_screening_followup.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FollowupTests(unittest.TestCase):
 def test_public_validation(self):
  self.assertFalse(m.validate()['global_review_complete'])
 def test_subset_counts(self):
  s=m.validate()
  self.assertEqual(sum(s['remaining_exclusions']['decisions'].values()),295)
  self.assertEqual(s['cumulative_exclusion_review']['initial_universe'],337)
 def test_recovery_denominator(self):
  f=m.validate()['fulltext_recovery']
  self.assertEqual((f['fulltexts_reviewed'],f['partial_or_not_recovered'],f['not_individually_searched']),(5,7,28))
 def test_false_completion_rejected(self):
  original=Path.read_text
  def altered(path,*args,**kwargs):
   value=original(path,*args,**kwargs)
   if path==m.SOURCE/'summary.json':
    s=json.loads(value);s['global_review_complete']=True;return json.dumps(s)
   return value
  with patch.object(Path,'read_text',altered):
   with self.assertRaises(AssertionError):m.validate()

if __name__=='__main__':unittest.main()
