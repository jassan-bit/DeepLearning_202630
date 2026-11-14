from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import importlib.util,json,sys,tempfile,time,unittest
ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load(ROOT/'scripts/e2_posthoc_fits_only_resume.py','e2_fits_only_test')
class FitsOnlyTests(unittest.TestCase):
 def test_dry_run_does_not_fetch_or_start_any_job(self):
  n=load(ROOT/'notebooks/resume_e2_posthoc_fits_only_colab.py','e2_fits_only_notebook_test')
  with patch.object(n,'fetch',side_effect=AssertionError('No I/O')):
   self.assertFalse(n.resume_e2_posthoc_fits_only_colab('a'*40)['new_inference'])
 def test_failed_diagnostics_verification_prevents_any_controller_admission(self):
  old=SimpleNamespace(orchestrate=Mock());checker=SimpleNamespace(verify_completed=Mock(side_effect=ValueError('incomplete')))
  original=m.install_fits_only(old,checker,Path('/output'),Path('/extension'),{},lambda:3.)
  with self.assertRaisesRegex(ValueError,'incomplete'):old.orchestrate('original args')
  original.assert_not_called()
 def test_fits_only_keeps_training_enabled_and_accounts_launcher_once(self):
  expected=object();old=SimpleNamespace(orchestrate=Mock(return_value=expected))
  checker=SimpleNamespace(verify_completed=Mock(return_value={'completed':79}))
  original=m.install_fits_only(old,checker,Path('/output'),Path('/extension'),{'known':'state'},lambda:4.5)
  self.assertIs(old.orchestrate('original',run_inference=True),expected)
  self.assertEqual(original.call_args.args,('original',));self.assertFalse(original.call_args.kwargs['run_inference'])
  self.assertTrue(original.call_args.kwargs['run_training']);self.assertEqual(original.call_args.kwargs['launcher_elapsed_seconds'],4.5)
 def test_real_original_orchestrator_does_not_request_5400s_with_2600s_balance(self):
  sys.path.insert(0,str(ROOT/'scripts'))
  old=load(ROOT/'scripts/e2_posthoc_fit_reproduction.py','e2_actual_original_orchestrator_regression')
  class Budget:
   charges=0.
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def set_persistence(self,r):pass
   def charge_overhead(self,seconds,**kw):self.charges+=seconds
   def reserve(self,job,seconds,**kw):
    if seconds>2600:raise ValueError('unaffordable reservation')
   def charge(self,job,seconds,**kw):self.charges+=seconds
   def report(self):return {'remaining_seconds':2600-self.charges}
  budget=Budget();transport=SimpleNamespace(_branch_head=lambda:'a'*40)
  backup=SimpleNamespace(GitHubBackup=lambda *a,**kw:transport)
  remote=SimpleNamespace(restore=lambda:None,snapshot=lambda *a:{})
  frozen=SimpleNamespace(run=Mock(side_effect=AssertionError('Completed diagnostics must not be readmitted')))
  checker=SimpleNamespace(verify_completed=Mock(return_value={'completed':79}))
  original=m.install_fits_only(old,checker,Path('/output'),Path('/extension'),{},lambda:2.)
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);output=root/'posthoc';(output/'native_search').mkdir(parents=True)
   (output/'native_search/native_transverse_receipt.json').write_text('{}')
   registration=root/'extension/experiments/e2_posthoc_fit_reproduction_20261113/registration.json'
   registration.parent.mkdir(parents=True);registration.write_text(json.dumps({'scientific_commit':old.ENGINE_COMMIT,'files_sha256':{}}))
   with patch.dict(sys.modules,{'e2_reduced_github_backup':backup,'e2_frozen_posthoc_gpu':frozen}), \
       patch.object(old,'PosthocRemoteState',return_value=remote),patch.object(old,'PosthocLedger',return_value=budget), \
       patch.object(old,'validate_original_anchor',return_value={}),patch.object(old,'immutable_json'),patch.object(old,'run_fits',return_value={})as fits:
    old.orchestrate(root/'source',root/'original',output,root/'python',root/'extension','a'*40,'dummy-token')
   frozen.run.assert_not_called();fits.assert_called_once()
if __name__=='__main__':unittest.main()
