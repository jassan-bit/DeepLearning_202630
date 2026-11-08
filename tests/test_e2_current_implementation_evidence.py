import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('evidence',ROOT/'scripts/publish_e2_current_implementation.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class EvidenceTests(unittest.TestCase):
 def setUp(self):self.s=json.loads((m.SOURCE/'synthetic_receipt.json').read_text())
 def test_valid_receipt(self):self.assertEqual(m.validate_receipt(self.s)['executed_model_count'],11)
 def test_market_claim_rejected(self):
  self.s['market_data_read']=True
  with self.assertRaises(AssertionError):m.validate_receipt(self.s)
 def test_skips_rejected(self):
  self.s['skipped_model_count']=1
  with self.assertRaises(AssertionError):m.validate_receipt(self.s)
 def test_local_roots_rejected(self):
  self.s['models'][0]['recipe']['source_roots']={'x':'/private'}
  with self.assertRaises(AssertionError):m.validate_receipt(self.s)
if __name__=='__main__':unittest.main()
