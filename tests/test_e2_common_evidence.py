import gzip,hashlib,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'experiments/e2_common_synthetic_20261109'
class CommonEvidence(unittest.TestCase):
 def test_chain_mirror_hashes_and_scope(self):
  subprocess.run([sys.executable,str(ROOT/'scripts/publish_e2_common_executor.py'),'--check'],check=True,capture_output=True)
 def test_dependency_spec_observed_only(self):
  s=json.loads((D/'summary.json').read_text());self.assertEqual(hashlib.sha256((ROOT/'requirements-e2-common-synthetic.lock.txt').read_bytes()).hexdigest(),s['dependency_spec_sha256']);self.assertEqual(s['functional_tests'],40)
 def test_preserve_previous_model_hashes(self):
  for d in ['e2_current_synthetic_20261108','e2_recipe_synthetic_20261109']:
   r=json.loads((ROOT/'experiments'/d/'synthetic_receipt.json').read_text())
   for n,h in r['source_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/'scripts'/n).read_bytes()).hexdigest(),h)
 def test_complete_compressed_evidence_and_final_pending(self):
  r=json.loads(gzip.decompress((D/'synthetic_receipt.json.gz').read_bytes()));a=json.loads((D/'availability.json').read_text());self.assertEqual(len(r['systems']),18);self.assertEqual(r['total_fit_stages'],36);self.assertIsNone(a['final_support']['common_count']);self.assertEqual(a['raw_archives_opened'],0)
if __name__=='__main__':unittest.main()
