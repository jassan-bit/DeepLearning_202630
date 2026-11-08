import hashlib,json,unittest,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class Evidence(unittest.TestCase):
 def test_public_manifest(self):
  d=ROOT/'experiments/e2_plan_audit_20261108'; mirror=ROOT/'book_e2/_static/plan_audit_20261108'
  manifest=json.loads((d/'manifest.json').read_text())
  self.assertEqual(set(manifest['files']),{'audit.json','plan.json','README.md'})
  for n,h in manifest['files'].items():
   self.assertEqual(hashlib.sha256((d/n).read_bytes()).hexdigest(),h)
   self.assertEqual((d/n).read_bytes(),(mirror/n).read_bytes())
 def test_no_training_and_code_hashes(self):
  a=json.loads((ROOT/'experiments/e2_plan_audit_20261108/audit.json').read_text())
  for k in ['model_fits','optimizer_steps','HPO_trials','downloads','market_rows_exported']:self.assertEqual(a[k],0)
  for n,h in a['code_and_contract_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),h)
 def test_proposed_budget(self):
  p=json.loads((ROOT/'experiments/e2_plan_audit_20261108/plan.json').read_text())
  self.assertEqual(p['search']['total_neural_stages_proposed'],14*4*2*10*2+4*2*10*2)
  self.assertEqual(p['pilot']['total_neural_stages_proposed'],14*10*2+10*2)
  self.assertIsNone(p['final_R']);self.assertFalse(p['real_training_pipeline_integrated']);self.assertTrue(p['test_608_already_observed'])

 def test_cli_rejects_overwrite_and_optimized_mode(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'existing.json';p.write_text('preserve')
   script=str(ROOT/'scripts/audit_e2_real_data.py')
   r=subprocess.run([sys.executable,script,'--output',str(p)],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertEqual(p.read_text(),'preserve')
   fresh=Path(tmp)/'never-created.json'
   r=subprocess.run([sys.executable,'-O',script,'--output',str(fresh)],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertFalse(fresh.exists())
