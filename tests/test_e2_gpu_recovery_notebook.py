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
NOTEBOOK = ROOT/'notebooks/entregable2_recuperacion_gpu.ipynb'


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))


def source(name):
    return next(c['source'] for c in notebook()['cells'] if c['id']=='recovery36-'+name)


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
    namespace.update(RECOVERY_COMMIT='b'*40,EXPECTED_RECOVERY_AUTHORIZATION_SHA256='c'*64,
                     WORKSPACE=workspace,REPO=workspace/'search-source',RECOVERY_REPO=workspace/'recovery-source',
                     VENV=workspace/'venv-py312-cu126',DATA_CACHE=workspace/'development_cache',
                     LOCAL_CAMPAIGN_CACHE=workspace/'search-campaigns')
    namespace['sha256_file']=lambda path: 'c'*64 if Path(path).name=='authorization.json' else 'e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe'
    return namespace


class FakeResponse:
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,maximum):return b'{"full_name":"jassan-bit/DeepLearning_202630"}'


class RecoveryNotebookTests(unittest.TestCase):
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

    def test_scientific_engine_pins_stay_exact_and_recovery_pins_are_frozen(self):
        tree=ast.parse(source('config'))
        values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                and n.targets[0].id in {'SOURCE_COMMIT','EXPECTED_SEARCH_REGISTRATION_SHA256','RECOVERY_COMMIT','EXPECTED_RECOVERY_AUTHORIZATION_SHA256'}}
        self.assertEqual(values['SOURCE_COMMIT'],'564ba4762f84a800b02fa4c734f5752a169f5223')
        self.assertEqual(values['EXPECTED_SEARCH_REGISTRATION_SHA256'],'78c3c5b8ca042a347aaa80d642f3affd8b5911e8697d8747a2de915d1c3194a9')
        if values['RECOVERY_COMMIT']=='ROOT_RECOVERY_COMMIT_TO_FREEZE':
            self.skipTest('Root is freezing the separate recovery controller and authorization')
        self.assertRegex(values['RECOVERY_COMMIT'],r'^[0-9a-f]{40}$')
        self.assertRegex(values['EXPECTED_RECOVERY_AUTHORIZATION_SHA256'],r'^[0-9a-f]{64}$')

    def test_default_inspection_executes_all_cells_without_gpu_auth_install_or_csv(self):
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            def run(args,**kwargs):
                args=[str(x) for x in args];calls.append(args)
                if args[:1]==['git']:
                    if 'rev-parse' in args:
                        pin=namespace['RECOVERY_COMMIT'] if str(namespace['RECOVERY_REPO']) in args else namespace['SOURCE_COMMIT']
                        return SimpleNamespace(stdout=pin+'\n')
                    return SimpleNamespace(stdout='')
                if args[1:2]==[str(namespace['REPO']/'scripts/prepare_e2_pilot_snapshot.py')]:
                    return SimpleNamespace(stdout='')
                if '--plan-only' in args:
                    return SimpleNamespace(stdout='{"status":"inspection_only","previous_completed_jobs":494,"original_unknown_attempt":{"optimizer_steps":null,"elapsed_seconds":null}}')
                raise AssertionError('Unexpected process in inspection mode: '+repr(args))
            namespace['run']=run
            output=''
            for name in ['source','plan','persistence','hardware','environment','checks','data','execute','report']:
                output+=execute(source(name),namespace)
            self.assertTrue(namespace['SOURCE_READY'])
            self.assertTrue(namespace['RECOVERY_SOURCE_READY'])
            self.assertTrue(namespace['PLAN_READY'])
            for flag in ('RUN_SEARCH','RETRY_LOST_ONCE','REMOTE_READY','HARDWARE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY','EXECUTION_ATTEMPTED'):
                self.assertIs(namespace[flag],False)
            self.assertIsNone(namespace['GITHUB_ENV'])
            self.assertIn('MODO INSPECCIÓN EN CPU',output)
            self.assertEqual(sum('--plan-only' in c for c in calls),1)
            self.assertFalse(any('--execute' in c or '--retry-lost-once' in c or 'nvidia-smi' in c or 'pip' in c or 'unittest' in c
                                 or any('fetch_e2_development_sources' in x for x in c) for c in calls))
            self.assertFalse(namespace['LOCAL_CAMPAIGN_CACHE'].exists())

    def test_invalid_budget_never_starts_planner_and_clears_ready_flag(self):
        for value in (False,0,-1,float('inf'),float('nan')):
            with self.subTest(value=value),tempfile.TemporaryDirectory() as directory:
                namespace=configuration(directory)
                namespace.update(SOURCE_READY=True,RECOVERY_SOURCE_READY=True,PLAN_READY=True,SESSION_HOURS=value)
                namespace['run']=lambda *a,**k:self.fail('Invalid budget started a process')
                with self.assertRaises(ValueError):execute(source('plan'),namespace)
                self.assertIs(namespace['PLAN_READY'],False)

    def test_changed_engine_source_is_rejected_before_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory);namespace.update(SOURCE_READY=True,RECOVERY_SOURCE_READY=True)
            calls=[]
            def run(args,**kwargs):
                calls.append([str(x) for x in args])
                if 'rev-parse' in args:return SimpleNamespace(stdout=namespace['RECOVERY_COMMIT']+'\n')
                if 'diff' in args and namespace['REPO'] in args:return SimpleNamespace(stdout='scripts/e2_cuda_training.py\n')
                return SimpleNamespace(stdout='')
            namespace['run']=run
            with self.assertRaisesRegex(RuntimeError,'fuente congelada'):
                execute(source('plan'),namespace)
            self.assertFalse(any('--plan-only' in command for command in calls))
            self.assertIs(namespace['PLAN_READY'],False)

    def test_missing_secret_invalidates_previous_remote_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update(RUN_SEARCH=True,RETRY_LOST_ONCE=True,PLAN_READY=True,REMOTE_READY=True,GITHUB_ENV={'GITHUB_TOKEN':'old-secret'})
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
            namespace.update(RUN_SEARCH=True,RETRY_LOST_ONCE=True,PLAN_READY=True)
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
            namespace=configuration(directory);namespace.update(RUN_SEARCH=True,RETRY_LOST_ONCE=True,PLAN_READY=True,REMOTE_READY=True)
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
            namespace.update(RUN_SEARCH=True,RETRY_LOST_ONCE=True,PLAN_READY=True,REMOTE_READY=True,GPU_READY=True,
                             SYNTHETIC_READY=False,DATA_READY=True)
            namespace['run']=lambda *a,**k:self.fail('Fetched market bytes before artificial checks')
            with self.assertRaises(RuntimeError):execute(source('data'),namespace)
            self.assertIs(namespace['DATA_READY'],False)

    def test_each_missing_execution_gate_blocks_child_process(self):
        gates=('SOURCE_READY','RECOVERY_SOURCE_READY','PLAN_READY','REMOTE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY')
        for gate in gates:
            with self.subTest(gate=gate),tempfile.TemporaryDirectory() as directory:
                namespace=configuration(directory)
                namespace.update({key:True for key in gates})
                namespace.update(RUN_SEARCH=True,RETRY_LOST_ONCE=True,EXECUTION_ATTEMPTED=True)
                namespace[gate]=False
                namespace['subprocess']=SimpleNamespace(run=lambda *a,**k:self.fail('Started with missing gate'))
                with self.assertRaises(RuntimeError):execute(source('execute'),namespace)
                self.assertIs(namespace['EXECUTION_ATTEMPTED'],False)

    def test_existing_campaign_cache_is_used_without_deletion_and_search_gets_scoped_token(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            output=Path(directory)/'existing-campaign';output.mkdir()
            sentinel=output/'previous.json';sentinel.write_text('{"preserve":true}')
            namespace.update({key:True for key in ('RUN_SEARCH','RETRY_LOST_ONCE','SOURCE_READY','RECOVERY_SOURCE_READY','PLAN_READY','REMOTE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY')})
            environment={'GITHUB_TOKEN':'synthetic-token'}
            namespace.update(CAMPAIGN_OUTPUT=output,GPU_PYTHON='gpu-python',GITHUB_ENV=environment,
                             verify_source_snapshot=lambda:None,verify_controller_snapshot=lambda:None,search_command=lambda *a,**k:['gpu-python','runner','--execute'])
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
            self.assertIn('MODO INSPECCIÓN EN CPU',execute(source('report'),namespace))
            namespace.update(RUN_SEARCH=True,EXECUTION_ATTEMPTED=False)
            self.assertIn('no se lanzó',execute(source('report'),namespace))

    def test_protocol_modified_after_planning_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update({key:True for key in ('RUN_SEARCH','RETRY_LOST_ONCE','SOURCE_READY','RECOVERY_SOURCE_READY','PLAN_READY','REMOTE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY')})
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
            self.assertIn('anteriores',shown)
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

    def test_retry_permission_is_required_before_auth_or_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update({key:True for key in ('RUN_SEARCH','SOURCE_READY','RECOVERY_SOURCE_READY','PLAN_READY','REMOTE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY')})
            namespace['RETRY_LOST_ONCE']=False
            namespace['subprocess']=SimpleNamespace(run=lambda *a,**k:self.fail('Executed without manual retry permission'))
            with patch.object(urllib.request,'build_opener',side_effect=AssertionError('Authenticated without manual retry permission')):
                with self.assertRaisesRegex(RuntimeError,'RETRY_LOST_ONCE'):
                    execute(source('persistence'),namespace)
            self.assertIs(namespace['REMOTE_READY'],False)
            with self.assertRaisesRegex(RuntimeError,'RETRY_LOST_ONCE'):
                execute(source('execute'),namespace)
            self.assertIs(namespace['EXECUTION_ATTEMPTED'],False)

    def test_controller_checkout_is_separate_and_has_only_seven_admitted_paths(self):
        tree=ast.parse(source('source'))
        paths=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='RECOVERY_SPARSE_PATHS' for t in n.targets))
        self.assertEqual(len(paths),7)
        self.assertEqual(sum(p.startswith('/scripts/') for p in paths),3)
        self.assertEqual(sum(p.startswith('/tests/') for p in paths),3)
        self.assertFalse(any(p.endswith('.csv') or p=='/scripts/' for p in paths))
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            scripts=namespace['REPO']/'scripts';scripts.mkdir(parents=True)
            sentinel=scripts/'original_engine.py';sentinel.write_text('DO_NOT_CHANGE = True\n')
            def run(args,**kwargs):
                if 'rev-parse' in args:
                    pin=namespace['RECOVERY_COMMIT'] if namespace['RECOVERY_REPO'] in args else namespace['SOURCE_COMMIT']
                    return SimpleNamespace(stdout=pin+'\n')
                return SimpleNamespace(stdout='')
            namespace['run']=run
            execute(source('source'),namespace)
            self.assertEqual(sentinel.read_text(),'DO_NOT_CHANGE = True\n')
            self.assertEqual([p.name for p in scripts.iterdir()],['original_engine.py'])
            self.assertNotEqual(namespace['REPO'],namespace['RECOVERY_REPO'])

    def test_modified_controller_or_authorization_blocks_before_child(self):
        for issue in ('head','tracked','untracked','authorization'):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as directory:
                namespace=configuration(directory)
                namespace.update({key:True for key in ('RUN_SEARCH','RETRY_LOST_ONCE','SOURCE_READY','RECOVERY_SOURCE_READY','PLAN_READY','REMOTE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY')})
                namespace.update(AUTHORIZATION=namespace['RECOVERY_REPO']/'experiments/e2_recovery_20261110/authorization.json',
                                 verify_source_snapshot=lambda:None)
                def run(args,**kwargs):
                    if 'rev-parse' in args:return SimpleNamespace(stdout=('f'*40 if issue=='head' else namespace['RECOVERY_COMMIT'])+'\n')
                    if 'diff' in args:return SimpleNamespace(stdout='scripts/run_e2_cuda_recovery.py\n' if issue=='tracked' else '')
                    if 'ls-files' in args:return SimpleNamespace(stdout='injected.py\n' if issue=='untracked' else '')
                    raise AssertionError('Unexpected command')
                namespace['run']=run
                if issue=='authorization':namespace['sha256_file']=lambda path:'0'*64
                helper=next(n for n in ast.parse(source('plan')).body if isinstance(n,ast.FunctionDef) and n.name=='verify_controller_snapshot')
                execute(ast.unparse(helper),namespace)
                namespace['subprocess']=SimpleNamespace(run=lambda *a,**k:self.fail('Launched an altered controller'))
                with self.assertRaises(RuntimeError):execute(source('execute'),namespace)
                self.assertIs(namespace['EXECUTION_ATTEMPTED'],False)

    def test_cpu_quota_environment_blocks_before_install_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update(RUN_SEARCH=True,RETRY_LOST_ONCE=True,REMOTE_READY=True,PLAN_READY=True,
                             GPU_READY=True,SYNTHETIC_READY=True,DATA_READY=True,
                             shutil=SimpleNamespace(which=lambda name:None),
                             platform=SimpleNamespace(system=lambda:'Linux',machine=lambda:'x86_64'))
            namespace['run']=lambda *a,**k:self.fail('Queried a GPU or claimed recovery without a visible card')
            with self.assertRaisesRegex(RuntimeError,'No hay GPU NVIDIA'):
                execute(source('hardware'),namespace)
            for flag in ('HARDWARE_READY','GPU_READY','SYNTHETIC_READY','DATA_READY'):
                self.assertIs(namespace[flag],False)

    def test_non_t4_card_is_rejected_before_cuda_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update(RUN_SEARCH=True,RETRY_LOST_ONCE=True,REMOTE_READY=True,PLAN_READY=True,
                             shutil=SimpleNamespace(which=lambda name:'nvidia-smi'),
                             platform=SimpleNamespace(system=lambda:'Linux',machine=lambda:'x86_64'),
                             run=lambda *a,**k:SimpleNamespace(stdout='NVIDIA L4, 23034 MiB, 550.54\n'))
            with self.assertRaisesRegex(RuntimeError,'requiere una Tesla T4'):
                execute(source('hardware'),namespace)
            self.assertIs(namespace['HARDWARE_READY'],False)
            self.assertIs(namespace['GPU_READY'],False)

    def test_failed_recovery_suite_clears_data_after_old_engine_suites_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update(RUN_SEARCH=True,RETRY_LOST_ONCE=True,REMOTE_READY=True,GPU_READY=True,
                             SYNTHETIC_READY=True,DATA_READY=True,GPU_PYTHON='gpu-python',
                             ensure_itransformer_sources=lambda:None)
            calls=[]
            def run(args,**kwargs):
                calls.append((args,kwargs))
                if kwargs['cwd']==namespace['RECOVERY_REPO']:
                    self.assertNotIn('GITHUB_TOKEN',kwargs['env'])
                    self.assertEqual(kwargs['env']['E2_ENGINE_ROOT'],str(namespace['REPO']))
                    raise RuntimeError('Recovery suite failed')
                return SimpleNamespace(returncode=0)
            namespace['subprocess']=SimpleNamespace(run=run)
            nodes=[n for n in ast.parse(source('checks')).body if not isinstance(n,ast.FunctionDef)]
            with self.assertRaisesRegex(RuntimeError,'Recovery suite failed'):
                execute(ast.unparse(ast.Module(body=nodes,type_ignores=[])),namespace)
            self.assertEqual(sum(k['cwd']==namespace['REPO'] for _,k in calls),4)
            self.assertIs(namespace['SYNTHETIC_READY'],False)
            self.assertIs(namespace['DATA_READY'],False)

    def test_final_view_keeps_original_unknowns_separate_from_new_physical_result(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            output=Path(directory)/'campaign';output.mkdir()
            logical={'completed_jobs':495,'known_optimizer_steps':12345}
            physical={'original_unknown_attempt':{'outcome':'unknown','elapsed_seconds':None,'optimizer_steps':None,'optimizer_steps_known_lower_bound':0},
                      'recovery_result':{'status':'observed','optimizer_steps':40},'physical_session_elapsed_seconds':13.0}
            (output/'status.json').write_text(json.dumps(logical))
            (output.parent/('recovery-'+namespace['EXPECTED_RECOVERY_AUTHORIZATION_SHA256'][:16]+'.json')).write_text(json.dumps(physical))
            namespace.update(RUN_SEARCH=True,EXECUTION_ATTEMPTED=True,LAST_SEARCH_RETURN_CODE=0,CAMPAIGN_OUTPUT=output)
            shown=json.loads(execute(source('report'),namespace))
            self.assertEqual(shown['logical_campaign_status'],logical)
            self.assertEqual(shown['external_physical_recovery_report'],physical)
            original=shown['external_physical_recovery_report']['original_unknown_attempt']
            self.assertIsNone(original['elapsed_seconds'])
            self.assertIsNone(original['optimizer_steps'])
            self.assertEqual(original['optimizer_steps_known_lower_bound'],0)
            self.assertIn('not an observed zero cost',shown['original_lost_attempt_interpretation'])

    def test_recovery_cannot_change_budget_or_domain(self):
        for changes in ({'CAMPAIGN_HOURS':48},{'SESSION_HOURS':9},{'MODELS':['MLP16']},{'ORIGINS':['2025-01-01']}):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as directory:
                namespace=configuration(directory)
                namespace.update(SOURCE_READY=True,RECOVERY_SOURCE_READY=True,**changes)
                namespace['run']=lambda *a,**k:self.fail('Started with an unauthorized recovery plan change')
                with self.assertRaises(ValueError):execute(source('plan'),namespace)
                self.assertIs(namespace['PLAN_READY'],False)

    def test_real_controller_inspection_uses_stdlib_without_network_gpu_or_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'controller';scripts=root/'scripts';scripts.mkdir(parents=True)
            (scripts/'run_e2_cuda_recovery.py').write_bytes((ROOT/'scripts/run_e2_cuda_recovery.py').read_bytes())
            authorization=root/'experiments/e2_recovery_20261110/authorization.json'
            authorization.parent.mkdir(parents=True)
            data=(ROOT/'experiments/e2_recovery_20261110/authorization.json').read_bytes()
            authorization.write_bytes(data)
            namespace=configuration(directory)
            namespace.update(RECOVERY_REPO=root,AUTHORIZATION=authorization,
                             EXPECTED_RECOVERY_AUTHORIZATION_SHA256=__import__('hashlib').sha256(data).hexdigest(),
                             CAMPAIGN_OUTPUT=Path(directory)/'no-campaign-created',
                             GITHUB_BRANCH='campaigns/e2-cuda-selection-v1',GITHUB_PREFIX='campaigns/e2-cuda-selection-v1')
            builder=next(n for n in ast.parse(source('plan')).body if isinstance(n,ast.FunctionDef) and n.name=='search_command')
            execute(ast.unparse(builder),namespace)
            command=[str(x) for x in namespace['search_command'](sys.executable,execute=False)]
            wrapper='''import builtins,runpy,socket,ssl,sys
from pathlib import Path
original_import=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','numpy','pandas','google'}:raise AssertionError('Heavy import in CPU inspection')
    return original_import(name,*args,**kwargs)
def forbidden(*args,**kwargs):raise AssertionError('Network in inspection')
builtins.__import__=checked
socket.socket=forbidden;socket.create_connection=forbidden
sys.argv=sys.argv[1:];sys.path.insert(0,str(Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0],run_name='__main__')
'''
            before={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
            completed=subprocess.run([command[0],'-S','-c',wrapper,*command[1:]],capture_output=True,text=True,check=True,
                                     env={'GITHUB_TOKEN':'synthetic-not-to-be-used','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'})
            value=json.loads(completed.stdout)
            self.assertEqual(value['status'],'inspection_only')
            self.assertEqual(value['previous_completed_jobs'],494)
            self.assertEqual(value['selection_attempts_upper_bound'],85074)
            self.assertEqual(value['session_hours'],8)
            self.assertEqual(value['campaign_hours'],24)
            self.assertFalse(value['recovery_enabled'])
            self.assertFalse(value['scientific_HPO_or_NAS_executed'])
            self.assertIsNone(value['original_unknown_attempt']['elapsed_seconds'])
            self.assertIsNone(value['original_unknown_attempt']['optimizer_steps'])
            self.assertNotIn('synthetic-not-to-be-used',completed.stdout+completed.stderr)
            self.assertFalse(namespace['CAMPAIGN_OUTPUT'].exists())
            self.assertEqual(before,{p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_inspection_never_adds_retry_permission_even_when_operator_flag_is_true(self):
        with tempfile.TemporaryDirectory() as directory:
            namespace=configuration(directory)
            namespace.update(SOURCE_READY=True,RECOVERY_SOURCE_READY=True,RETRY_LOST_ONCE=True)
            def run(args,**kwargs):
                if 'rev-parse' in args:return SimpleNamespace(stdout=namespace['RECOVERY_COMMIT']+'\n')
                if '--plan-only' in args:return SimpleNamespace(stdout='{"status":"inspection_only"}')
                return SimpleNamespace(stdout='')
            namespace['run']=run;execute(source('plan'),namespace)
            command=[str(x) for x in namespace['search_command']('host-python',execute=False)]
            self.assertIn('--plan-only',command)
            self.assertNotIn('--execute',command)
            self.assertNotIn('--retry-lost-once',command)
            self.assertIn(str(namespace['RECOVERY_REPO']/'scripts/run_e2_cuda_recovery.py'),command)
            self.assertIn(str(namespace['REPO']),command)
            self.assertNotIn('--extend-campaign-hours',command)
            shown=execute(source('report'),namespace)
            self.assertIn('RUN_SEARCH: False',shown)
            self.assertIn('RETRY_LOST_ONCE: True',shown)


if __name__=='__main__':unittest.main()
