import hashlib,json,subprocess,sys,tempfile,time,unittest,venv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_fresh_clean_runtime as m

def digest(v):return m.sha(m.canonical(v))
def checked(v):
    if v['sha256']!=digest({k:x for k,x in v.items() if k!='sha256'}):raise ValueError('checksum')

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);base=Path(self.tmp.name)
        self.posthoc=base/'posthoc';self.original=base/'github-primary-output-v1';self.original.mkdir()
        self.original_source=base/('reduced-source-'+m.ENGINE);self.original_source.mkdir()
        self.clean=self.posthoc/'reproduction/clean';self.source=self.clean/'source';self.python=self.clean/'venv/bin/python'
        self.existing=base/'base-python';self.existing.write_bytes(b'base profile might contain scipy1.17.1; never copied')
        self.existing.chmod(0o700)
        self.adapter=base/'environment-adapter';(self.adapter/'scripts').mkdir(parents=True)
        bootstrap=self.adapter/'scripts/e2_posthoc_clean_worker_bootstrap.py';bootstrap.write_bytes(b'exact historical bootstrap')
        env_adapter=self.adapter/'scripts/e2_posthoc_environment_adapter.py';env_adapter.write_bytes(b'exact historical adapter')
        controller=base/'old_controller.py';controller.write_bytes(b"VALUE='exact old controller'\n")
        self.controller_sha=m.sha(controller.read_bytes())
        self.profile={'python':[3,12,12],'packages':{'pip':'25.3','scipy':'1.17.0','torch':'2.8.0+cu126'}}
        self.runtime={'python':'3.12.12 full recorded build','torch':'2.8.0+cu126','cuda':'12.6',
                      'gpu':'NVIDIA A100-SXM4-80GB','total_memory_bytes':85094825984}
        lock=''.join(k+'=='+v+'\n' for k,v in sorted(self.profile['packages'].items())).encode()
        self.lock_sha=m.sha(lock)
        self.receipt={'version':'old-runtime-v1','clean_source':str(self.source),'clean_python':str(self.python),
            'scientific_commit':m.ENGINE,'new_venv_created':True,'system_site_packages':False,
            'package_lock_sha256':self.lock_sha,'package_profile':self.profile,'runtime':self.runtime,
            'environment_adapter_bootstrap_sha256':m.sha(bootstrap.read_bytes()),
            'environment_adapter_sha256':m.sha(env_adapter.read_bytes()),'setup_completed':True}
        self.receipt_raw=m.canonical(self.receipt)+b'\n';self.receipt_sha=m.sha(self.receipt_raw)
        self.acceptance={'version':'e2-posthoc-synthetic-acceptance-v1','status':'passed','optimizer_steps':0,
                         'training_performed':False,'market_data_read':False,'model_count':13}
        self.acceptance_raw=m.canonical(self.acceptance)+b'\n';self.acceptance_sha=m.sha(self.acceptance_raw)
        self.full={**self.receipt,'synthetic_acceptance_path':str(self.clean/'synthetic_acceptance.json'),
                   'synthetic_acceptance_scope':'historical-scope','synthetic_acceptance_sha256':self.acceptance_sha}
        self.job=self.posthoc/'reproduction/jobs/posthoc-reproduction-completed';self.job.mkdir(parents=True)
        self.spec={'job_id':self.job.name,'configuration_commit':m.PIN,'controller_source_sha256':self.controller_sha,
                   'clean_runtime':self.full,'model':'LSTM-autoencoder con predictor','seed':271801}
        self.result={'status':'completed','attempt_id':self.spec['job_id'],'configuration_commit':m.PIN,
                     'worker_spec_sha256':digest(self.spec),'new_clean_environment':self.full,'artifacts_sha256':{}}
        (self.job/'worker_spec.json').write_bytes(m.canonical(self.spec)+b'\n')
        self.result_raw=m.canonical(self.result)+b'\n';(self.job/'result.json').write_bytes(self.result_raw)
        self.rows=[]
        self.append('reserved',job_id='clean-setup-original',seconds=2400)
        self.append('charged',job_id='clean-setup-original',status='completed',evidence={'receipt_sha256':self.receipt_sha})
        self.append('reserved',job_id=self.spec['job_id'],seconds=5460)
        self.append('charged',job_id=self.spec['job_id'],status='completed',evidence={'result_sha256':m.sha(self.result_raw)})
        ledger=self.posthoc/'budget/posthoc_budget.jsonl';ledger.parent.mkdir();ledger.write_bytes(b''.join(m.canonical(r)+b'\n' for r in self.rows))
        self.development={'results/development/input.csv':m.sha(b'exact dev bytes')}
        p=self.original_source/'results/development/input.csv';p.parent.mkdir(parents=True);p.write_bytes(b'exact dev bytes')
        self.old=SimpleNamespace(__file__=str(controller),ZERO='0'*64,digest=digest,checked=checked,
            file_sha=lambda p:m.sha(Path(p).read_bytes()),clean_environment=lambda:{'NO_SECRET':'1'},verify_clean_runtime=Mock(return_value=True))
        self.old.run_bounded=Mock(side_effect=self.run_command)
        self.wrong_acceptance=False
    def append(self,kind,**fields):
        r={'kind':kind,'sequence':len(self.rows),'previous_sha256':self.rows[-1]['sha256'] if self.rows else '0'*64,**fields};r['sha256']=digest(r);self.rows.append(r)
    def run_command(self,command,**kw):
        c=[str(x) for x in command]
        if c[:3]==['git','init','--quiet']:(self.source/'.git/info').mkdir(parents=True)
        if '-m' in c and 'venv' in c:
            self.python.parent.mkdir(parents=True);self.python.write_bytes(b'new clean executable')
            (self.clean/'venv/pyvenv.cfg').write_text('include-system-site-packages = false\n')
        if '--requirement' in c:
            lock=Path(c[c.index('--requirement')+1]).read_text()
            self.assertIn('scipy==1.17.0\n',lock);self.assertNotIn('1.17.1',lock);self.assertIn('--no-deps',c)
            self.assertIn('https://pypi.org/simple',c);self.assertIn('https://download.pytorch.org/whl/cu126',c)
        if '--preflight' in c:(self.clean/'synthetic_acceptance.json').write_bytes(b'wrong' if self.wrong_acceptance else self.acceptance_raw)
        return .01
    def patches(self,**extra):
        return patch.multiple(m,RUNTIME_SHA=self.receipt_sha,LOCK_SHA=self.lock_sha,ACCEPTANCE_SHA=self.acceptance_sha,
                              CONTROLLER_SHA=self.controller_sha,DEVELOPMENT=self.development,**extra)
    def restore(self,probe=None):
        with self.patches(),patch.object(m,'_git_state',return_value={'head':m.ENGINE,'remote':m.REPOSITORY,'dirty':''}), \
             patch.object(m,'_boot_id',return_value='11111111-1111-1111-1111-111111111111'), \
             patch.object(m,'_probe',side_effect=probe or (lambda old,python,kind,*a:self.profile if kind=='profile' else self.runtime)) as call:
            value=m.restore_clean_runtime(self.old,self.original,self.posthoc,self.adapter,self.existing,time.monotonic()+100)
            self.probe_calls=call.call_args_list
            return value
    def test_exact_receipt_profile_and_acceptance_rebuilt_without_relabelling_history(self):
        value=self.restore();self.assertEqual(value,self.full)
        self.assertEqual((self.clean/'clean_runtime.json').read_bytes(),self.receipt_raw)
        self.assertEqual((self.clean/'synthetic_acceptance.json').read_bytes(),self.acceptance_raw)
        admission=json.loads(next((self.clean/'fresh_runtime_admissions').rglob('admission.json')).read_bytes())
        self.assertFalse(admission['historical_receipt_relabelled_as_new']);self.assertEqual(admission['optimizer_steps'],0)
        self.assertEqual(admission['actual_package_install_method'],'pip_exact_lock_official_indexes_with_default_cache')
        self.assertEqual([call.args[1] for call in self.probe_calls],[self.existing,self.existing,self.python,self.python])
        self.old.verify_clean_runtime.assert_called_once_with(self.full)
        self.assertFalse(any('--worker-spec' in [str(x) for x in call.args[0]] for call in self.old.run_bounded.call_args_list))
    def test_completed_worker_and_completed_setup_bindings_are_required(self):
        self.result['worker_spec_sha256']='0'*64;(self.job/'result.json').write_bytes(m.canonical(self.result)+b'\n')
        with self.patches(),self.assertRaisesRegex(ValueError,'bind'):m.select_runtime(self.old,self.posthoc)
        self.assertFalse(self.old.run_bounded.called)
    def test_historical_lock_sha_mismatch_fails_before_any_setup_command(self):
        with self.patches(),patch.object(m,'LOCK_SHA','0'*64),self.assertRaisesRegex(ValueError,'lock'):
            m.restore_clean_runtime(self.old,self.original,self.posthoc,self.adapter,self.existing,time.monotonic()+100)
        self.assertFalse(self.old.run_bounded.called)
    def test_observed_distribution_drift_stops_before_historical_receipt_publication(self):
        changed=json.loads(json.dumps(self.profile));changed['packages']['scipy']='1.17.1'
        with self.assertRaisesRegex(ValueError,'packages differ'):
            self.restore(probe=lambda old,python,kind,*a:(changed if python==self.python else self.profile) if kind=='profile' else self.runtime)
        self.assertFalse((self.clean/'clean_runtime.json').exists());self.old.verify_clean_runtime.assert_not_called()
    def test_observed_gpu_runtime_must_equal_original(self):
        changed=dict(self.runtime,gpu='Tesla T4')
        with self.assertRaisesRegex(ValueError,'A100 runtime differs'):
            self.restore(probe=lambda old,python,kind,*a:self.profile if kind=='profile' else changed)
        self.assertFalse((self.clean/'clean_runtime.json').exists())
    def test_real_symlink_venv_preserves_its_package_context(self):
        real=Path(self.tmp.name)/'real-prepared-venv';venv.EnvBuilder(with_pip=False,symlinks=True).create(real)
        python=real/'bin/python'
        self.assertTrue(python.is_symlink());self.assertEqual(m.prepared_python(python),python)
        value=json.loads(subprocess.check_output([str(m.prepared_python(python)),'-I','-c',
            'import json,sys;print(json.dumps({"prefix":sys.prefix,"python":list(sys.version_info[:3])}))'],text=True))
        self.assertEqual(value['prefix'],str(real));self.assertEqual(value['python'],list(sys.version_info[:3]))
        # The same symlink is accepted only for this executable, never an artifact.
        with self.assertRaisesRegex(ValueError,'regular'):m.regular(python)
        self.existing=python;self.assertEqual(self.restore(),self.full)
    def test_prepared_interpreter_requires_regular_parents_and_executable_target(self):
        linked_parent=Path(self.tmp.name)/'linked-parent';linked_parent.symlink_to(self.existing.parent,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'regular'):m.prepared_python(linked_parent/self.existing.name)
        self.existing.chmod(0o600)
        with self.assertRaisesRegex(ValueError,'executable'):m.prepared_python(self.existing)
        dangling=Path(self.tmp.name)/'missing-python';dangling.symlink_to('missing-target')
        with self.assertRaisesRegex(ValueError,'absent or invalid'):m.prepared_python(dangling)
    def test_wrong_prepared_interpreter_profile_stops_before_venv_creation(self):
        wrong={'python':[3,13,0],'packages':{'torch':'2.8.0+cu126'}}
        with self.assertRaisesRegex(ValueError,'Prepared base.*profile differs'):
            self.restore(probe=lambda *args:wrong)
        self.assertFalse(self.old.run_bounded.called);self.assertFalse((self.clean/'clean_runtime.json').exists())
    def test_synthetic_acceptance_must_be_regenerated_byte_identically(self):
        self.wrong_acceptance=True
        with self.assertRaisesRegex(ValueError,'byte-identical'):self.restore()
        self.assertFalse((self.clean/'fresh_runtime_admissions').exists());self.old.verify_clean_runtime.assert_not_called()
    def test_existing_environment_with_system_site_packages_is_rejected(self):
        self.python.parent.mkdir(parents=True);self.python.write_bytes(b'preexisting')
        (self.clean/'venv/pyvenv.cfg').write_text('include-system-site-packages = true\n')
        with self.assertRaisesRegex(ValueError,'system packages'):self.restore()
        self.assertFalse(any('pip' in [str(x) for x in call.args[0]] for call in self.old.run_bounded.call_args_list))
    def test_partial_venv_without_historical_receipt_retries_exact_package_install(self):
        self.python.parent.mkdir(parents=True);self.python.write_bytes(b'partial clean executable')
        (self.clean/'venv/pyvenv.cfg').write_text('include-system-site-packages = false\n')
        self.assertEqual(self.restore(),self.full)
        commands=[[str(x) for x in call.args[0]] for call in self.old.run_bounded.call_args_list]
        self.assertTrue(any('pip' in command and '--requirement' in command for command in commands))
        self.assertFalse(any('venv' in command for command in commands))
    def test_existing_historical_receipt_never_triggers_package_reinstallation(self):
        self.python.parent.mkdir(parents=True);self.python.write_bytes(b'existing clean executable')
        (self.clean/'venv/pyvenv.cfg').write_text('include-system-site-packages = false\n')
        (self.clean/'clean_runtime.json').write_bytes(self.receipt_raw)
        self.assertEqual(self.restore(),self.full)
        self.assertFalse(any('pip' in [str(x) for x in call.args[0]] for call in self.old.run_bounded.call_args_list))
    def test_deadline_and_symlink_guards_preserve_existing_inputs(self):
        with self.assertRaises(TimeoutError):m._remaining(time.monotonic()-1)
        with self.assertRaises(ValueError):m._remaining(float('nan'))
        p=self.original_source/'unsafe';p.symlink_to(self.existing)
        with self.assertRaisesRegex(ValueError,'regular'):m.once(p,b'changed')
        self.assertTrue(self.existing.read_bytes().startswith(b'base profile'))
    def test_cli_loads_exact_original_module_and_delegates_same_bound_arguments(self):
        (self.original_source/'scripts').mkdir()
        deadline=time.monotonic()+100
        args=['--old-controller',self.old.__file__,'--original-output',str(self.original),
              '--posthoc-output',str(self.posthoc),'--environment-adapter-root',str(self.adapter),
              '--existing-python',str(self.existing),'--deadline',str(deadline)]
        with self.patches(),patch.object(sys,'path',list(sys.path)),patch.object(m,'restore_clean_runtime') as restore:
            result=m.main(args)
        self.assertEqual(result,0);self.assertEqual(len(restore.call_args.args),6)
        self.assertEqual(Path(restore.call_args.args[0].__file__),Path(self.old.__file__))
        self.assertEqual(restore.call_args.args[1:],(self.original,self.posthoc,self.adapter,self.existing,deadline))
    def test_cli_original_source_tamper_fails_before_import_or_restoration(self):
        (self.original_source/'scripts').mkdir()
        with self.assertRaisesRegex(ValueError,'before import'):
            m.load_old_controller(Path(self.old.__file__),self.original)

if __name__=='__main__':unittest.main()
