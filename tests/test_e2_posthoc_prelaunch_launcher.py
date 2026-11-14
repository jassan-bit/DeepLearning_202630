import hashlib,importlib.util,io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load(ROOT/'notebooks/resume_e2_posthoc_prelaunch_colab.py','e2_prelaunch_notebook_test')
c=load(ROOT/'scripts/e2_posthoc_prelaunch_continue.py','e2_prelaunch_continue_test')
class PrelaunchLauncherTests(unittest.TestCase):
 def test_dry_run_never_downloads_or_reconciles(self):
  with patch.object(m,'fetch',side_effect=AssertionError('No I/O')):
   self.assertEqual(m.resume_e2_posthoc_prelaunch_colab('a'*40)['status'],'dry_run')
  with self.assertRaises(ValueError):m.resume_e2_posthoc_prelaunch_colab('main')
  with self.assertRaises(TypeError):m.resume_e2_posthoc_prelaunch_colab('a'*40,execute=1)
 def test_remote_journal_must_match_control_descriptor(self):
  import base64
  payload=b'{"kind":"closed"}\n';stream=io.BytesIO()
  with zipfile.ZipFile(stream,'w')as z:z.writestr('budget/posthoc_budget.jsonl',payload)
  raw=stream.getvalue();item={'path':'known.zip','bytes':len(raw),'sha256':m.sha(raw)}
  state={'files':{'budget/posthoc_budget.jsonl':{'bytes':len(payload),'sha256':m.sha(payload)}},'metadata_deltas':[item]}
  client=SimpleNamespace(_content=Mock(return_value={'content':base64.b64encode(raw).decode()}))
  self.assertEqual(m.read_remote_ledger(client,state,'head'),payload)
  item['sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'delta changed'):m.read_remote_ledger(client,state,'head')
 def test_continuation_checks_new_and_existing_sources_before_any_controller(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t)/'new';work=Path(t)/'workspace';base=work/('fresh-session-recovery-'+c.BASE_COMMIT)
   (root/'scripts').mkdir(parents=True);(base/'scripts').mkdir(parents=True)
   p=root/'scripts/adapter.py';p.write_bytes(b'new adapter')
   old=base/'scripts/controller.py';old.write_bytes(b'unchanged controller')
   r={'files_sha256':{'scripts/adapter.py':hashlib.sha256(p.read_bytes()).hexdigest()},
      'existing_launcher_files_sha256':{'scripts/controller.py':hashlib.sha256(old.read_bytes()).hexdigest()}}
   reg=root/c.REG;reg.parent.mkdir(parents=True);reg.write_text(json.dumps(r))
   with patch.object(c,'WORKSPACE',work):
    self.assertEqual(c.validate(root),(r,base))
    old.write_bytes(b'changed')
    with self.assertRaisesRegex(ValueError,'verified Colab session'):c.validate(root)
 def test_transport_installed_only_on_unchanged_controller_load(self):
  marker=object();consistency=SimpleNamespace(install=Mock())
  fresh=SimpleNamespace(load=Mock(return_value=marker),install_shutdown_handler=Mock())
  def continuation(base,expected,token):
   self.assertEqual(fresh.load(Path('scripts/e2_posthoc_fit_reproduction.py'),'old'),marker)
   self.assertEqual((expected,token),('expected','dummy-token'))
   return 'closed'
  fresh.execute_continue=continuation
  fake_backup=SimpleNamespace()
  original_load=fresh.load
  with patch.object(c,'validate',return_value=({},Path('/unchanged'))),patch.object(c,'load',side_effect=[fresh,consistency]), \
       patch.dict(sys.modules,{'e2_reduced_github_backup':fake_backup}):
   self.assertEqual(c.run(Path('/new'),'expected','dummy-token'),'closed')
  self.assertIs(fresh.load,original_load);consistency.install.assert_called_once_with(fake_backup,marker)
  fresh.install_shutdown_handler.assert_called_once()
if __name__=='__main__':unittest.main()
