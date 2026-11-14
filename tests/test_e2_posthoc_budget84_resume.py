from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import base64,hashlib,importlib.util,json,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
m=load(ROOT/'scripts/e2_posthoc_budget84_resume.py','e2_budget84_resume_test')
import e2_reduced_github_backup as backup
class Budget84ResumeTests(unittest.TestCase):
 def test_entrypoint_dry_run_never_fetches_or_grants_credit(self):
  n=load(ROOT/'notebooks/resume_e2_posthoc_budget84_colab.py','e2_budget84_notebook_test')
  with patch.object(n,'fetch',side_effect=AssertionError('No I/O')):
   self.assertEqual(n.resume_e2_posthoc_budget84_colab('a'*40)['approved_total_hours'],84)
  with self.assertRaises(ValueError):n.resume_e2_posthoc_budget84_colab('main')
 def test_expected_authorization_manifest_matches_actual_backup_uploader(self):
  approval=b'{"user_statement":"si amplialo","assistant_selected_total_hours":84}\n'
  receipt,manifest,authority=m.authorization_record(backup,approval)
  client=backup.GitHubBackup('dummy-token');client._checked=True
  client._commit_files=Mock(return_value=('a'*40,False))
  with tempfile.TemporaryDirectory()as t:
   p=Path(t);(p/'authorization.json').write_bytes(approval)
   result=client.persist_result(p,receipt)
  self.assertEqual(client._commit_files.call_args.args[2],manifest)
  self.assertEqual(result['manifest_sha256'],authority['receipt_sha256'])
  files=client._commit_files.call_args.args[0]
  client._branch_head=Mock(return_value='b'*40)
  client._content=lambda path,head:{'content':base64.b64encode(files[path]).decode()}
  self.assertTrue(m.verify_authorization(client,manifest,authority,approval))
  files[manifest['artifacts'][0]['remote_path']]=b'changed authorization'
  with self.assertRaisesRegex(ValueError,'user-authorization evidence differs'):
   m.verify_authorization(client,manifest,authority,approval)
 def test_authority_is_stable_across_identical_upload_commit_changes(self):
  raw=b'{"assistant_selected_total_hours":84}\n'
  first=m.authorization_record(backup,raw);second=m.authorization_record(backup,raw)
  self.assertEqual(first,second);self.assertNotIn('receipt_remote_commit',first[2])
 def test_funding_reaches_fresh_controller_loaded_by_actual_fits_continuation(self):
  fits=load(ROOT/'scripts/e2_posthoc_fits_only_resume.py','e2_actual_fits_continuation_for_funding')
  old_before=object();old_fresh=object();authority={'registered':'authority'}
  adapter=SimpleNamespace(install=Mock());consistency=SimpleNamespace(install=Mock())
  actual_old=SimpleNamespace(PosthocRemoteState=lambda *a:SimpleNamespace(state=SimpleNamespace(_current=lambda head:{'state_id':'expected'})))
  fresh=SimpleNamespace(install_shutdown_handler=Mock(),load=Mock(return_value=old_fresh))
  def execute(base,expected,token):return fresh.load(Path('scripts/e2_posthoc_fit_reproduction.py'),'new-original')
  fresh.execute_continue=execute
  fake_backup=SimpleNamespace(GitHubBackup=lambda *a,**kw:SimpleNamespace(_branch_head=lambda:'a'*40))
  values=(Path('/base'),fresh,object(),actual_old,fake_backup,consistency,object(),Path('/extension'))
  fits.components=Mock(return_value=values)
  m.funded_components(fits,adapter,authority)
  with patch.object(fits,'install_fits_only'):
   result=fits.continue_child(Path('/existing'),'expected','dummy-token',3.)
  self.assertIs(result,old_fresh)
  self.assertEqual([c.args[0]for c in adapter.install.call_args_list],[actual_old,old_fresh])
  self.assertTrue(all(c.args[1]==authority for c in adapter.install.call_args_list))
if __name__=='__main__':unittest.main()
