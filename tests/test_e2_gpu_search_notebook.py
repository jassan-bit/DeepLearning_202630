"""Execute notebook admission paths with fakes: no install, network, GPU or data."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import ast
import json
import os
import re
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
import subprocess

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/entregable2_busqueda_gpu.ipynb'


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))


def source(name):
    return next(c['source'] for c in notebook()['cells'] if c['id']=='search35-'+name)


def execute(code, namespace):
    output=StringIO()
    with redirect_stdout(output):
        exec(compile(code,str(NOTEBOOK),'exec'),namespace)
    return output.getvalue()


def configuration(directory):
    namespace={}
    with patch.dict(os.environ,{},clear=True):
        execute(source('config'),namespace)
    workspace=Path(directory)/'workspace'
    namespace.update(SOURCE_COMMIT='a'*40,WORKSPACE=workspace,REPO=workspace/'search-source',
                     VENV=workspace/'venv-py312-cu126',DATA_CACHE=workspace/'development_cache',
                     LOCAL_CAMPAIGN_CACHE=workspace/'search-campaigns')
    return namespace


class FakeResponse:
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,maximum):return b'{"full_name":"jassan-bit/DeepLearning_202630"}'


class SearchNotebookTests(unittest.TestCase):
    def test_notebook_is_unexecuted_valid_transport(self):
        value=notebook()
        self.assertEqual(value['nbformat'],4)
        ids=[c['id'] for c in value['cells']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertTrue(all(re.fullmatch(r'[A-Za-z0-9_-]{1,64}',i) for i in ids))
        code=[c for c in value['cells'] if c['cell_type']=='code']
        self.assertEqual(len(code),10)
        for cell in code:
            ast.parse(cell['source'])
            self.assertIsNone(cell['execution_count'])
            self.assertEqual(cell['outputs'],[])

    def test_published_source_is_an_immutable_commit(self):
        tree=ast.parse(source('config'))
        pin=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='SOURCE_COMMIT' for t in n.targets))
        if pin=='ROOT_CODE_COMMIT_TO_FREEZE':
            self.skipTest('Root is freezing the code commit before publishing the notebook')
        self.assertRegex(pin,r'^[0-9a-f]{40}$')
        self.assertNotIn(pin,{'main','master'})
        registration=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=='EXPECTED_SEARCH_REGISTRATION_SHA256' for t in n.targets))
        self.assertRegex(registration,r'^[0-9a-f]{64}$')

    def test_default_plan_executes_all_cells_without_gpu_auth_install_or_csv(self):
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            def run(args,**kwargs):
                args=[str(x) for x in args];calls.append(args)
                if args[:1]==['git']:
                    return SimpleNamespace(stdout='a'*40+'\n' if 'rev-parse' in args else '')
                if args[1:2]==[str(namespace['REPO']/'scripts/prepare_e2_pilot_snapshot.py')]:
                    return SimpleNamespace(stdout='')
                if '--plan-only' in args:
                    return SimpleNamespace(stdout='{"status":"planned","market_data_read":false}')
                raise AssertionError('Unexpected process in plan mode: '+repr(args))
            namespace['run']=run
            namespace['sha256_file']=lambda path:'e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe'
            output=''
            for name in ['source','plan','persistence','hardware','environment','checks','data','execute','report']:
                output+=execute(source(name),namespace)
            self.assertTrue(namespace['PLAN_READY'])
            for flag in ('RUN_SEARCH','REMOTE_READY','HARDWARE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY','EXECUTION_ATTEMPTED'):
                self.assertIs(namespace[flag],False)
            self.assertIsNone(namespace['GITHUB_ENV'])
            self.assertIn('MODO PLAN',output)
            self.assertEqual(sum('--plan-only' in c for c in calls),1)
            self.assertFalse(any('--execute' in c or 'nvidia-smi' in c or 'pip' in c or 'unittest' in c
                                 or any('fetch_e2_development_sources' in x for x in c) for c in calls))
            self.assertFalse(namespace['LOCAL_CAMPAIGN_CACHE'].exists())

    def test_invalid_budget_never_starts_planner_and_clears_ready_flag(self):
        for value in (False,0,-1,float('inf'),float('nan')):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as directory:
                namespace=configuration(directory)
                namespace.update(SOURCE_READY=True,PLAN_READY=True,SESSION_HOURS=value)
                namespace['run']=lambda *a,**k:self.fail('Invalid budget started a process')
                with self.assertRaises(ValueError):execute(source('plan'),namespace)
                self.assertIs(namespace['PLAN_READY'],False)

    def test_changed_source_is_rejected_before_planner(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory);namespace['SOURCE_READY']=True
            calls=[]
            def run(args,**kwargs):
                calls.append(args)
                return SimpleNamespace(stdout='scripts/e2_cuda_bindings.py\n')
            namespace['run']=run
            with self.assertRaisesRegex(RuntimeError,'fuente congelada'):
                execute(source('plan'),namespace)
            self.assertEqual(len(calls),1)
            self.assertIs(namespace['PLAN_READY'],False)

    def test_missing_secret_invalidates_previous_remote_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update(RUN_SEARCH=True,PLAN_READY=True,REMOTE_READY=True,GITHUB_ENV={'GITHUB_TOKEN':'old-secret'})
            google=ModuleType('google');google.__path__=[]
            colab=ModuleType('google.colab')
            def unavailable(name):raise RuntimeError('Secret unavailable')
            colab.userdata=SimpleNamespace(get=unavailable)
            with patch.dict(os.environ,{},clear=True),patch.dict(sys.modules,{'google':google,'google.colab':colab}):
                with self.assertRaisesRegex(RuntimeError,'Secretos de Colab'):
                    execute(source('persistence'),namespace)
            self.assertIs(namespace['REMOTE_READY'],False)
            self.assertIsNone(namespace['GITHUB_ENV'])

    def test_token_only_enters_search_environment_and_never_output_or_plan_command(self):
        secret='synthetic-token-not-a-real-credential'
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'GITHUB_TOKEN':secret},clear=True):
            namespace={};execute(source('config'),namespace)
            self.assertNotIn('GITHUB_TOKEN',namespace['CUDA_ENV'])
            namespace.update(RUN_SEARCH=True,PLAN_READY=True)
            requests=[]
            class Opener:
                def open(self,request,**kwargs):
                    requests.append(request)
                    return FakeResponse()
            with patch.object(urllib.request,'build_opener',return_value=Opener()):
                output=execute(source('persistence'),namespace)
            self.assertTrue(namespace['REMOTE_READY'])
            self.assertEqual(namespace['GITHUB_ENV']['GITHUB_TOKEN'],secret)
            self.assertNotIn(secret,output)
            self.assertEqual(requests[0].get_method(),'GET')
            self.assertEqual(requests[0].full_url,'https://api.github.com/repos/jassan-bit/DeepLearning_202630')
            self.assertIsNone(requests[0].data)
            self.assertNotIn('GITHUB_TOKEN',namespace['CUDA_ENV'])

    def test_http_auth_failure_omits_response_body_and_secret(self):
        secret='synthetic-token'
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'GITHUB_TOKEN':secret},clear=True):
            namespace=configuration(directory);namespace.update(RUN_SEARCH=True,PLAN_READY=True,REMOTE_READY=True)
            class Opener:
                def open(self,*args,**kwargs):
                    raise urllib.error.HTTPError('https://api.github.com/repos/example',403,'BODY '+secret,{},None)
            with patch.object(urllib.request,'build_opener',return_value=Opener()):
                with self.assertRaisesRegex(RuntimeError,'HTTP 403') as caught:
                    execute(source('persistence'),namespace)
            self.assertNotIn(secret,str(caught.exception))
            self.assertNotIn('BODY',str(caught.exception))
            self.assertIs(namespace['REMOTE_READY'],False)
            self.assertIsNone(namespace['GITHUB_ENV'])

    def test_failed_artificial_suite_clears_stale_data_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update(RUN_SEARCH=True,REMOTE_READY=True,GPU_READY=True,SYNTHETIC_READY=True,DATA_READY=True,
                             GPU_PYTHON='gpu-python',ensure_itransformer_sources=lambda:None)
            def fail(*args,**kwargs):raise RuntimeError('Synthetic suite failed')
            namespace['subprocess']=SimpleNamespace(run=fail)
            tree=ast.parse(source('checks'))
            nodes=[n for n in tree.body if not isinstance(n,ast.FunctionDef)]
            with self.assertRaisesRegex(RuntimeError,'Synthetic suite failed'):
                execute(ast.unparse(ast.Module(body=nodes,type_ignores=[])),namespace)
            self.assertIs(namespace['SYNTHETIC_READY'],False)
            self.assertIs(namespace['DATA_READY'],False)

    def test_data_requires_fresh_synthetic_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update(RUN_SEARCH=True,PLAN_READY=True,REMOTE_READY=True,GPU_READY=True,
                             SYNTHETIC_READY=False,DATA_READY=True)
            namespace['run']=lambda *a,**k:self.fail('Fetched market bytes before artificial checks')
            with self.assertRaises(RuntimeError):execute(source('data'),namespace)
            self.assertIs(namespace['DATA_READY'],False)

    def test_each_missing_execution_gate_blocks_child_process(self):
        gates=('SOURCE_READY','PLAN_READY','REMOTE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY')
        for gate in gates:
            with self.subTest(gate=gate),tempfile.TemporaryDirectory() as directory:
                namespace=configuration(directory)
                namespace.update({key:True for key in gates})
                namespace.update(RUN_SEARCH=True,EXECUTION_ATTEMPTED=True)
                namespace[gate]=False
                namespace['subprocess']=SimpleNamespace(run=lambda *a,**k:self.fail('Started with missing gate'))
                with self.assertRaises(RuntimeError):execute(source('execute'),namespace)
                self.assertIs(namespace['EXECUTION_ATTEMPTED'],False)

    def test_existing_campaign_cache_is_used_without_deletion_and_search_gets_scoped_token(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            output=Path(directory)/'existing-campaign';output.mkdir()
            sentinel=output/'previous.json';sentinel.write_text('{"preserve":true}')
            namespace.update({key:True for key in ('RUN_SEARCH','SOURCE_READY','PLAN_READY','REMOTE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY')})
            environment={'GITHUB_TOKEN':'synthetic-token'}
            namespace.update(CAMPAIGN_OUTPUT=output,GPU_PYTHON='gpu-python',GITHUB_ENV=environment,
                             verify_source_snapshot=lambda:None,search_command=lambda *a,**k:['gpu-python','runner','--execute'])
            calls=[]
            def run(args,**kwargs):calls.append((args,kwargs));return SimpleNamespace(returncode=0)
            namespace['subprocess']=SimpleNamespace(run=run)
            execute(source('execute'),namespace)
            self.assertEqual(sentinel.read_text(),'{"preserve":true}')
            self.assertIs(calls[0][1]['env'],environment)
            self.assertNotIn('synthetic-token',repr(calls[0][0]))
            self.assertTrue(namespace['EXECUTION_ATTEMPTED'])

    def test_final_cell_distinguishes_plan_from_a_missed_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            self.assertIn('MODO PLAN',execute(source('report'),namespace))
            namespace.update(RUN_SEARCH=True,EXECUTION_ATTEMPTED=False)
            self.assertIn('no se lanzó',execute(source('report'),namespace))

    def test_protocol_modified_after_planning_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update({key:True for key in ('RUN_SEARCH','SOURCE_READY','PLAN_READY','REMOTE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY')})
            namespace.update(PROTOCOL_RELATIVE='book_e2/_static/search_protocol_20261109/protocol.json',
                             NORMALIZED_PROTOCOL_SHA256='e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe',
                             run=lambda *a,**k:SimpleNamespace(stdout=''),sha256_file=lambda p:'0'*64)
            helper=next(n for n in ast.parse(source('plan')).body if isinstance(n,ast.FunctionDef) and n.name=='verify_source_snapshot')
            execute(ast.unparse(helper),namespace)
            namespace['subprocess']=SimpleNamespace(run=lambda *a,**k:self.fail('Started after protocol changed'))
            with self.assertRaisesRegex(RuntimeError,'protocolo cambió'):
                execute(source('execute'),namespace)
            self.assertIs(namespace['EXECUTION_ATTEMPTED'],False)

    def test_nonzero_search_labels_existing_report_as_potentially_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            output=Path(directory)/'campaign';output.mkdir()
            (output/'status.json').write_text('{"status":"previous_confirmed_state"}')
            namespace.update(RUN_SEARCH=True,EXECUTION_ATTEMPTED=True,LAST_SEARCH_RETURN_CODE=3,CAMPAIGN_OUTPUT=output)
            shown=execute(source('report'),namespace)
            self.assertIn('sesión anterior',shown)
            self.assertIn('previous_confirmed_state',shown)

    def test_bootstrap_failure_clears_all_runtime_admission_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update(RUN_SEARCH=True,HARDWARE_READY=True,GPU_READY=True,SYNTHETIC_READY=True,DATA_READY=True)
            def fail(*args,**kwargs):raise RuntimeError('Bootstrap unavailable')
            namespace['run']=fail
            with self.assertRaisesRegex(RuntimeError,'Bootstrap unavailable'):
                execute(source('environment'),namespace)
            for flag in ('GPU_READY','SYNTHETIC_READY','DATA_READY'):
                self.assertIs(namespace[flag],False)

    def test_notebook_plan_command_matches_real_cli_without_site_packages_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';scripts=root/'scripts';scripts.mkdir(parents=True)
            for name in ('run_e2_cuda_search.py','e2_cuda_search_plan.py','e2_search_plan.py'):
                (scripts/name).write_bytes((ROOT/'scripts'/name).read_bytes())
            protocol=root/'book_e2/_static/search_protocol_20261109/protocol.json'
            protocol.parent.mkdir(parents=True)
            data=(ROOT/'book_e2/_static/search_protocol_20261109/protocol.json').read_bytes()
            import hashlib
            expected='e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe'
            if hashlib.sha256(data).hexdigest()!=expected:
                data=data.rstrip(b'\n')+b'\n\n'
            self.assertEqual(hashlib.sha256(data).hexdigest(),expected)
            protocol.write_bytes(data)
            (root/'requirements-e2-training-gpu.lock.txt').write_bytes((ROOT/'requirements-e2-training-gpu.lock.txt').read_bytes())
            registration=root/'experiments/e2_cuda_search_20261109/registration.json'
            registration.parent.mkdir(parents=True)
            implementation={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(scripts.glob('*.py'))+[root/'requirements-e2-training-gpu.lock.txt']}
            registration.write_text(json.dumps({'version':'e2-cuda-search-admission-v1',
                'protocol_sha256':expected,'repository':'jassan-bit/DeepLearning_202630',
                'initial_budget':{'session_seconds':28800,'campaign_seconds':86400},
                'implementation_sha256':implementation}))
            namespace=configuration(directory)
            namespace.update(REPO=root,SOURCE_READY=True,
                EXPECTED_SEARCH_REGISTRATION_SHA256=hashlib.sha256(registration.read_bytes()).hexdigest(),
                sha256_file=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest())
            # Construct the actual notebook command while faking only git and normalization.
            def run(args,**kwargs):
                if '--plan-only' in args:return SimpleNamespace(stdout='{"planned":true}')
                return SimpleNamespace(stdout='')
            namespace['run']=run
            execute(source('plan'),namespace)
            command=[str(x) for x in namespace['search_command'](sys.executable,execute=False)]
            wrapper='''import builtins,runpy,socket,ssl,sys
from pathlib import Path
original_import=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','numpy','pandas','google'}:raise AssertionError('Heavy import in plan-only')
    return original_import(name,*args,**kwargs)
def forbidden(*args,**kwargs):raise AssertionError('Network in plan-only')
builtins.__import__=checked
socket.socket=forbidden;socket.create_connection=forbidden
sys.argv=sys.argv[1:];sys.path.insert(0,str(Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            completed=subprocess.run([command[0],'-S','-c',wrapper,*command[1:]],text=True,
                capture_output=True,check=True,env={'GITHUB_TOKEN':'synthetic-not-to-be-used','PYTHONNOUSERSITE':'1'})
            value=json.loads(completed.stdout)
            self.assertEqual(value['coverage']['selected_models'],14)
            self.assertEqual(value['coverage']['selected_origins'],11)
            self.assertEqual(value['hpo_attempts'],14*18*2*11)
            self.assertEqual(value['nas1_attempts'],6*1000*11)
            self.assertIs(value['scientific_HPO_or_NAS_executed'],False)
            self.assertNotIn('synthetic-not-to-be-used',completed.stdout+completed.stderr)
            self.assertFalse(namespace['CAMPAIGN_OUTPUT'].exists())
            self.assertFalse((root/'results').exists())


if __name__=='__main__':unittest.main()
