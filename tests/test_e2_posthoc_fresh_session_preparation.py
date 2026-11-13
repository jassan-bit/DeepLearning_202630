import hashlib,importlib.util,io,json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e2_fresh_preparation',ROOT/'notebooks/prepare_e2_posthoc_fresh_session.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class PreparationTests(unittest.TestCase):
    def test_dry_run_cannot_download_or_start_a_controller(self):
        with patch.object(m,'module_for',side_effect=AssertionError('network forbidden')):
            r=m.prepare_fresh_posthoc_session(execute=False)
        self.assertFalse(r['training_started']);self.assertFalse(r['ledger_read_or_changed'])
        with self.assertRaises(TypeError):m.prepare_fresh_posthoc_session(execute=1)
    def test_source_download_requires_exact_immutable_bytes_and_size(self):
        raw=b'known source';checksum=hashlib.sha256(raw).hexdigest()
        opener=Mock(side_effect=lambda *a,**kw:io.BytesIO(raw))
        self.assertEqual(m.public_bytes('a'*40,'notebooks/helper.py',checksum,size=len(raw),opener=opener),raw)
        with self.assertRaisesRegex(ValueError,'mismatch'):m.public_bytes('a'*40,'x','0'*64,opener=opener)
        with self.assertRaisesRegex(ValueError,'mismatch'):m.public_bytes('a'*40,'x',checksum,size=2,opener=opener)
        with self.assertRaisesRegex(ValueError,'Immutable'):m.public_bytes('main','x',checksum,opener=opener)
    def test_existing_unequal_file_and_symlink_are_preserved(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'source.py';m.once(p,b'old')
            with self.assertRaisesRegex(ValueError,'Existing'):m.once(p,b'new')
            self.assertEqual(p.read_bytes(),b'old');m.once(p,b'old')
            link=Path(t)/'link';link.symlink_to(p)
            with self.assertRaisesRegex(ValueError,'regular'):m.once(link,b'old')
    def test_controller_detection_ignores_code_strings_but_recognizes_worker_paths(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for pid,args in [('7000001',b'python\0/content/ext/scripts/e2_posthoc_fit_reproduction.py\0--orchestrate\0'),
                             ('7000002',b'python\0-c\0print(" e2_posthoc_fit_reproduction.py ")\0')]:
                p=root/pid;p.mkdir();(p/'cmdline').write_bytes(args)
            self.assertEqual(m.active_pids(root),[7000001])
    def fixture(self,t,*,bad_metadata=False,bad_script=False):
        root=Path(t);source=root/('reduced-source-'+m.ENGINE_COMMIT);source.mkdir()
        for relative in m.FETCH_SCRIPTS:
            p=source/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/relative).read_bytes())
        if bad_script:(source/next(iter(m.FETCH_SCRIPTS))).write_bytes(b'changed source')
        metadata=source/m.METADATA_PATH;metadata.parent.mkdir(parents=True)
        metadata.write_bytes(b'changed metadata' if bad_metadata else (ROOT/m.METADATA_PATH).read_bytes())
        python=root/'fake-python';python.write_bytes(b'not executed')
        extension=root/('guide-closure-extension-'+m.ORIGINAL_POSTHOC_COMMIT)
        original=extension/m.PINS['posthoc'][1];original.parent.mkdir(parents=True)
        original.write_bytes((ROOT/m.PINS['posthoc'][1]).read_bytes())
        runtime_call=Mock(side_effect=lambda namespace,**kw:namespace.update(GPU_PYTHON=str(python)))
        modules={'runtime':{'prepare_e2_a100_runtime':runtime_call},
                 'checkout':{'_assert_idle':Mock(),'_checkout':Mock(return_value=source)},
                 'posthoc':{'materialize':Mock(return_value=extension),
                            'run_e2_posthoc_fit_reproduction_colab':Mock(side_effect=AssertionError('Controller forbidden'))}}
        return root,source,python,extension,modules
    def test_fresh_preparation_only_calls_bounded_pinned_acquisition_functions(self):
        with tempfile.TemporaryDirectory() as t:
            root,source,python,extension,modules=self.fixture(t);namespace={};calls=[]
            def run(command,*,env,cwd):
                self.assertNotIn('GITHUB_TOKEN',env);self.assertNotIn('GH_TOKEN',env);self.assertEqual(cwd,source)
                calls.append([str(x) for x in command])
            with patch.object(m,'active_pids',return_value=[]),patch.object(m,'module_for',side_effect=modules.__getitem__), \
                 patch.object(m,'run',side_effect=run),patch.dict(os.environ,{'GITHUB_TOKEN':'private-placeholder','GH_TOKEN':'private-placeholder'}):
                result=m.prepare_fresh_posthoc_session(namespace,workspace=root,execute=True)
            self.assertEqual(len(calls),2)
            self.assertTrue(calls[0][1].endswith('fetch_e2_development_sources.py'))
            self.assertTrue(calls[1][1].endswith('fetch_e2_published_sources.py'))
            modules['posthoc']['run_e2_posthoc_fit_reproduction_colab'].assert_not_called()
            self.assertFalse(result['training_started']);self.assertFalse(result['original_output_restored'])
            self.assertEqual(result['original_launcher'],str(extension/m.PINS['posthoc'][1]))
            self.assertEqual(namespace['GPU_PYTHON'],str(python))
    def test_changed_metadata_or_acquisition_script_stops_before_external_execution(self):
        for kind in ('metadata','script'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as t:
                root,source,python,extension,modules=self.fixture(t,bad_metadata=kind=='metadata',bad_script=kind=='script')
                with patch.object(m,'active_pids',return_value=[]),patch.object(m,'module_for',side_effect=modules.__getitem__),patch.object(m,'run') as run:
                    with self.assertRaises(ValueError):m.prepare_fresh_posthoc_session({},workspace=root,execute=True)
                    run.assert_not_called()
    def test_active_scientific_process_stops_before_runtime_or_files(self):
        with patch.object(m,'active_pids',return_value=[123]),patch.object(m,'module_for') as module:
            with self.assertRaisesRegex(RuntimeError,'controller'):m.prepare_fresh_posthoc_session({},execute=True)
            module.assert_not_called()
    def test_global_deadline_caps_nested_network_and_child_process_without_detaching(self):
        raw=b'pinned';expected=hashlib.sha256(raw).hexdigest()
        calls=[]
        def child(*args,**kwargs):calls.append(kwargs);return 'child-returned'
        local=Mock(side_effect=lambda *args,**kwargs:io.BytesIO(raw))
        remote=Mock(side_effect=lambda *args,**kwargs:io.BytesIO(raw))
        explicit=Mock(side_effect=lambda *args,**kwargs:io.BytesIO(raw))
        with patch.object(m.subprocess,'run',side_effect=child) as original_run, \
             patch.object(m,'urlopen',local),patch.object(m.urllib.request,'urlopen',remote), \
             patch.object(m.time,'monotonic',return_value=100.):
            with m.deadline_limits(107.):
                self.assertEqual(m.subprocess.run(['uv','install'],timeout=300,start_new_session=True),'child-returned')
                self.assertEqual(m.public_bytes('a'*40,'x',expected),raw)
                self.assertEqual(m.public_bytes('a'*40,'x',expected,opener=explicit),raw)
                with m.urllib.request.urlopen('https://example.test',timeout=45) as response:self.assertEqual(response.read(),raw)
            self.assertIs(m.subprocess.run,original_run);self.assertIs(m.urlopen,local)
        self.assertEqual(calls[0]['timeout'],7.);self.assertIs(calls[0]['start_new_session'],False)
        self.assertEqual(local.call_args.kwargs['timeout'],7.);self.assertEqual(remote.call_args.kwargs['timeout'],7.)
        self.assertEqual(explicit.call_args.kwargs['timeout'],7.)
    def test_deadline_expiry_never_starts_work_and_hooks_restore_after_failure(self):
        with patch.object(m.time,'monotonic',return_value=100.),patch.object(m,'module_for') as module:
            with self.assertRaises(TimeoutError):m.prepare_fresh_posthoc_session({},execute=True,deadline=99.)
            module.assert_not_called()
        original_run=m.subprocess.run;original_open=m.urllib.request.urlopen
        with self.assertRaisesRegex(RuntimeError,'setup failed'):
            with m.deadline_limits(time.monotonic()+10):raise RuntimeError('setup failed')
        self.assertIs(m.subprocess.run,original_run);self.assertIs(m.urllib.request.urlopen,original_open)
        with self.assertRaises(ValueError):m.timeout_left(float('nan'),30)
    def test_explicit_public_download_deadline_rejects_before_opening_network(self):
        opener=Mock()
        with self.assertRaises(TimeoutError):m.public_bytes('a'*40,'x','0'*64,opener=opener,deadline=time.monotonic()-1)
        opener.assert_not_called()

if __name__=='__main__':unittest.main()
