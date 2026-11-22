"""Offline notebook and pre-credential final guards. No market fit or access."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from engine import cli, confirmatory_gate, data
from package import crypto_notebook_template as nb

ROOT=Path(__file__).resolve().parents[1]


def codes(book):
    return [c['source'] for c in book['cells'] if c['cell_type']=='code']


def generated():
    return nb.notebook('a'*40,'b'*64,'c'*64)


def test_source_pins_default_activations_and_embedded_programs_compile():
    book=generated()
    assert len([c for c in book['cells'] if c['cell_type']=='markdown' and re.match(r'## [1-8]\. ',c['source'])])==8
    assert book['metadata']['e3_final_confirmation_enabled_in_this_version'] is False
    assert book['metadata']['e3_account_Nasdaq_required'] is False
    found={}
    for source in codes(book):
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
                for target in node.targets:
                    if isinstance(target,ast.Name):
                        if target.id.startswith(('RUN_','PREPARE_','OPEN_','FREEZE_','REGISTER_','SEAL_','RESTORE_','REACQUIRE_','EXPORT_','ASSESS_','SYNC_','ARCHIVE_')) and isinstance(node.value.value,bool):
                            found[target.id]=node.value.value
                        if target.id.endswith('_program') and isinstance(node.value.value,str):
                            ast.parse(node.value.value)
    assert found and all(value is False for value in found.values())
    assert found['PREPARE_CRYPTO_DEVELOPMENT'] is False
    with pytest.raises(ValueError):nb.notebook('main','b'*64,'c'*64)


def test_default_notebook_executes_metadata_only_without_secret_or_input(tmp_path,monkeypatch):
    bootstrap=f'''
from bootstrap_colab import cpu_env
from types import SimpleNamespace
commands=[]
def setup(commit,manifest,cache,existing_python=None,run_tests=True):
    assert commit == {'a'*40!r} and manifest == {'b'*64!r} and run_tests
    return {{"python":{sys.executable!r},"source_root":{str(ROOT)!r},"source_commit":commit}}
def command(settings,args,credentials=None,allow_blocked=False):
    assert credentials is None
    commands.append(args)
    return SimpleNamespace(returncode=0,stdout='{{}}')
'''.encode()
    book=nb.notebook('a'*40,'b'*64,hashlib.sha256(bootstrap).hexdigest())
    original=Path.is_dir
    monkeypatch.setattr(Path,'is_dir',lambda self:False if str(self)=='/content' else original(self))
    monkeypatch.chdir(tmp_path)
    calls=[]
    def fetch(url,**kwargs):
        calls.append(str(url));return io.BytesIO(bootstrap)
    namespace={'__name__':'__synthetic_notebook_fixture__'}
    with patch('urllib.request.urlopen',side_effect=fetch), \
         patch('getpass.getpass',side_effect=AssertionError('NO_SECRET_DEFAULT')), \
         patch('builtins.input',side_effect=AssertionError('NO_OWNER_OR_BUDGET_ACCEPTANCE_DEFAULT')):
        for source in codes(book):exec(compile(source,'crypto_notebook_fixture','exec'),namespace)
    assert calls==['https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+'a'*40+'/'+nb.PREFIX+'/bootstrap_colab.py']
    assert namespace['bootstrap'].commands==[['check']]
    assert namespace['missing']==['F00','F01','F02','F03']
    assert not namespace['PLANNING_CASE'].exists() and not namespace['FINAL_CASE'].exists()
    assert 'crypto' in str(namespace['WORK'])
    assert not (namespace['PRIVATE']/'raw').exists()
    for panel in namespace['PANELS']:
        form=json.loads((namespace['PRIVATE']/'inputs'/panel/'owner-development.json').read_text())
        assert form['actor_name'] is None and form['authorize_test_preparation'] is False
        assert not (namespace['PRIVATE']/'inputs'/panel/'rights.json').exists()


def _first_cell_reuse_fixture(tmp_path,monkeypatch,previous_info):
    """No interpreter is run; capture the argument passed to verified setup."""
    bootstrap=f'''
received=[]
def setup(commit,manifest,cache,existing_python=None,run_tests=True):
    received.append(existing_python)
    return {{"python":"new isolated CPU fixture","source_root":{str(ROOT)!r},"source_commit":commit}}
'''.encode()
    book=nb.notebook('a'*40,'b'*64,hashlib.sha256(bootstrap).hexdigest())
    original=Path.is_dir
    monkeypatch.setattr(Path,'is_dir',lambda self:False if str(self)=='/content' else original(self))
    monkeypatch.chdir(tmp_path)
    namespace={'__name__':'__first_cell_reuse_software_fixture__','SETTINGS':previous_info}
    with patch('urllib.request.urlopen',return_value=io.BytesIO(bootstrap)),redirect_stdout(io.StringIO()):
        exec(compile(codes(book)[0],'first_cell_reuse_fixture','exec'),namespace)
    return namespace['bootstrap'].received


def test_unrelated_missing_or_nonmapping_settings_never_reuse_an_interpreter(tmp_path,monkeypatch):
    foreign_python=tmp_path/'foreign-existing-python';foreign_python.write_text('SOFTWARE FIXTURE ONLY')
    cases=[{'source_commit':'5'*40,'python':str(foreign_python)},
           {'source_commit':'2'*40,'python':str(foreign_python)},
           {'python':str(foreign_python)},None,'not a mapping',[],
           {'source_commit':'c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c','python':str(tmp_path/'missing')},
           {'source_commit':'a'*40,'python':object()}]
    for previous in cases:
        with monkeypatch.context() as context:
            assert _first_cell_reuse_fixture(tmp_path,context,previous)==[None]


def test_only_known_engine_A_or_current_source_can_reuse_existing_python(tmp_path,monkeypatch):
    cpu_python=tmp_path/'known-existing-python';cpu_python.write_text('SOFTWARE FIXTURE ONLY')
    for commit in ('c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c','a'*40):
        with monkeypatch.context() as context:
            assert _first_cell_reuse_fixture(tmp_path,context,{'source_commit':commit,'python':str(cpu_python)})==[str(cpu_python)]


def test_real_helper_module_is_loaded_only_after_explicit_acquisition_flag():
    stage=next(s for s in codes(generated()) if s.startswith('PREPARE_CRYPTO_DEVELOPMENT ='))
    tree=ast.parse(stage)
    sensitive=next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=='PREPARE_CRYPTO_DEVELOPMENT')
    assert 'prepare_development(globals())' in ast.get_source_segment(stage,sensitive)
    assert 'input(' not in stage[:sensitive.lineno-1]


def test_active_crypto_power_design_changes_only_permitted_metadata():
    base=json.loads((ROOT/'configs/power_design.json').read_text())
    active=json.loads((ROOT/'configs/power_design_crypto_metadata.json').read_text())
    metadata={'version','panel_instrument_proposals','data_design_amendment','status'}
    assert {k:v for k,v in base.items() if k not in metadata}=={k:v for k,v in active.items() if k not in metadata}
    assert hashlib.sha256((ROOT/'configs/power_design.json').read_bytes()).hexdigest()=='9a683a53d942f1153fe19ae3fff34a02cb8339edb9ca9d53d3c5fd84e269fee9'
    assert set(active['panel_instrument_proposals'])=={'F00','F01','F02','F03'}
    assert 'BTCUSD_PERP' in active['panel_instrument_proposals']['F02']
    stage=next(s for s in codes(generated()) if s.startswith('RUN_PLANNING_TRAINING ='))
    assert 'configs/power_design_crypto_metadata.json' in stage
    assert '"--design", str(POWER_DESIGN)' in stage


def test_planning_requires_admissions_and_literal_budget_activation_before_any_run(tmp_path):
    source=next(s for s in codes(generated()) if s.startswith('RUN_PLANNING_TRAINING ='))
    assert source.index('assert not missing') < source.index('acceptance = input(') < source.index('run_cli(["prepare"')
    assert '320' in source and '10h40' in source and 'PLANIFICAR 340' in source
    source=source.replace('RUN_PLANNING_TRAINING = False','RUN_PLANNING_TRAINING = True',1)
    touched=[]
    namespace={'PRIVATE':tmp_path,'SOURCE':ROOT,'PLANNING_CASE':tmp_path/'case','missing':[],
        'run_cli':lambda *a,**kw:touched.append(a),'read_json':lambda p:json.loads(Path(p).read_text())}
    with patch('builtins.input',return_value='NO'):
        with pytest.raises(AssertionError,match='no activada'):
            exec(compile(source,'inactive_budget_fixture','exec'),namespace)
    assert touched==[] and not (tmp_path/'case').exists()


@pytest.mark.parametrize('flag',[
    'FREEZE_FINAL_PROTOCOL','RUN_FINAL_TRAINING','SEAL_FINAL_TRAINING',
    'REGISTER_FINAL_EXPOSURE','PREPARE_RESERVED_TEST','OPEN_RESERVED_EVALUATION'])
def test_final_cell_preflight_blocks_before_any_inputs_or_credentials(flag):
    source=next(s for s in codes(generated()) if re.search(r'^'+flag+r' = False$',s,re.M))
    source=source.replace(flag+' = False',flag+' = True',1)
    def blocked():raise RuntimeError('PRIMARY_ENDPOINT_ALIGNMENT_NOT_ESTABLISHED JOINT_DECISION_VALIDATION_NOT_ESTABLISHED')
    with pytest.raises(RuntimeError,match='PRIMARY_ENDPOINT_ALIGNMENT'):
        exec(compile(source,'blocked_final_cell_fixture','exec'),{'require_final_execution':blocked})


def call(args):
    out=io.StringIO()
    with redirect_stdout(out):code=cli.main(args)
    return code,json.loads(out.getvalue().splitlines()[-1])


@pytest.mark.parametrize('command', ['freeze-protocol','prepare-final','run-final','seal-training','evaluate'])
def test_cli_final_preflight_precedes_credentials_network_training_and_test(tmp_path,command):
    case=tmp_path/'case'
    if command=='run-final':
        case.mkdir();(case/'plan.json').write_text('{"stage":"final","software_fixture":false}')
    base=['--case',str(case),'--source-commit','a'*40,'--panels',str(tmp_path/'unread-panels.json')]
    if command=='freeze-protocol':args=['freeze-protocol','--base','unread','--power','unread','--method-report','unread','--test-spec','unread','--output','unwritten']
    elif command=='prepare-final':args=['prepare',*base,'--stage','final']
    elif command=='run-final':args=['run',*base,'--activate']
    elif command=='evaluate':args=['evaluate',*base,'--activate']
    else:args=['seal-training',*base]
    with patch.object(cli,'private_credentials',side_effect=AssertionError('NO_CREDENTIAL_READ')), \
         patch('urllib.request.urlopen',side_effect=AssertionError('NO_NETWORK')):
        code,result=call(args)
    assert code==2 and result['status']=='confirmation_not_enabled'
    assert result['gate_codes']==list(confirmatory_gate.CODES)
    assert result['evaluation_executed'] is False and result['R3_result'] is None


def test_cli_planning_is_not_misclassified_as_final_and_reads_no_test(tmp_path):
    args=SimpleNamespace(command='run',activate=True,case=tmp_path,source_root=ROOT)
    (tmp_path/'plan.json').write_text('{"stage":"planning","software_fixture":false}')
    assert confirmatory_gate.preflight(args) is None
    assert confirmatory_gate.preflight(SimpleNamespace(command='prepare',stage='planning',source_root=ROOT)) is None


def test_editing_gate_approval_booleans_cannot_enable_the_source_version(tmp_path):
    (tmp_path/'configs').mkdir()
    value=json.loads((ROOT/'configs/confirmatory_gate.json').read_text())
    value.update(primary_endpoint_alignment_established=True,joint_decision_validation_established=True,final_market_execution_enabled=True)
    (tmp_path/'configs/confirmatory_gate.json').write_text(json.dumps(value))
    with pytest.raises(confirmatory_gate.ConfirmatoryGateError) as error:
        confirmatory_gate.block_final_operation(tmp_path,'evaluate')
    assert error.value.invalid_metadata is True


@pytest.mark.parametrize('activate',[False,True])
def test_data_cli_final_guard_precedes_reading_any_raw_or_authorization(tmp_path,activate):
    args=['prepare-final','--manifest','unread-manifest','--split-config','unread-split',
          '--rights','unread-rights','--exposure','unread-exposure','--authorization','unread-authorization',
          '--output',str(tmp_path/'not-created')]
    if activate:args.append('--enable-final-preparation')
    out=io.StringIO()
    with redirect_stdout(out),patch.object(data,'prepare_final_partition',side_effect=AssertionError('NO_FINAL_FUNCTION')), \
         patch('urllib.request.urlopen',side_effect=AssertionError('NO_NETWORK')):
        with pytest.raises(SystemExit) as code:data.main(args)
    result=json.loads(out.getvalue())
    assert code.value.code==2 and result['status']=='confirmation_not_enabled'
    assert result['gate_codes']==list(confirmatory_gate.CODES) and result['evaluation_executed'] is False
    assert result['R3_result'] is None and not (tmp_path/'not-created').exists()


def test_credentials_have_minimum_independent_secret_and_are_not_global(tmp_path):
    helpers=codes(generated())[1]
    namespace={'WORK':tmp_path,'PRIVATE':tmp_path/'private','Path':Path,'json':json,'hashlib':hashlib,
        'SOURCE_COMMIT':'a'*40,'SOURCE':ROOT,'SETTINGS':{},'bootstrap':SimpleNamespace(),'subprocess':__import__('subprocess')}
    exec(helpers,namespace)
    namespace['private_value']=lambda name,prompt:'short'
    with pytest.raises(ValueError,match='32 caracteres'):namespace['github_credentials']()
    namespace['private_value']=lambda name,prompt:'SOFTWARE_FIXTURE_TOKEN' if name.endswith('TOKEN') else 'x'*32
    got=namespace['github_credentials']()
    assert set(got)=={'token','backup_secret'} and len(got['backup_secret'])==32
    assert 'token' not in namespace and 'backup_secret' not in namespace
