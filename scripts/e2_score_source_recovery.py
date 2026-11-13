"""Continue a proved pre-data HTTP404 without repeating scientific scoring.

The old charged acquisition failure and its partial marker stay immutable. A
new isolated inference worker uses E's already registered raw-archive route.
No data are read, downloaded, fitted, or inferred by this parent adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

VERSION = 'e2-score-source-recovery-v1'
ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
PRIOR_ARIMA_COMMIT = 'f12cc0904efefde6854342ef18a6c447d5572b3e'
FUNDING_COMMIT = '41da88cf600aadc0319214465c94a85335e2b1b7'
REGISTRATION_PATH = 'experiments/e2_score_source_recovery_20261113/registration.json'
POLICY_PATH = 'experiments/e2_score_source_recovery_20261113/policy.json'
SOURCE_PATH = 'scripts/e2_score_source_recovery.py'
FAILED_SPEC = 'c1226c9eb398b44825fdc927cdf8a8edd904629749992fcdbf5696dbb883cc33'
FAILED_JOB = 'ff53264094dc1e03fe8586b6aaabeb7b91b951e4ed54b7d9b6e9c020a7c2435a'
FAILED_CHARGE = '9d9cd441d09b635b7049b3ac118a28ba27ea3d04b3b57adf35380eb38dcd643a'
FAILED_SECONDS = 18.68105469799957
FREEZE_FILE = 'c0e9e2589dca9fa03b341bb72f87d8b541d33b4721715622daeae4438dbd60b4'
FREEZE_DIGEST = '7d8de5d3b589de235f9c2a35a67210e79c63daf13160451c68f8108999fe1242'
OLD_FILES = {
    'worker_spec.json': '7b7a6b1a3d85b0c5c5312481549fe1b41088440b29e9e1054b31e59e1063a717',
    'worker_failure.json': '3f11f5ac3c0dbd2c002ce59c3ddd2222951d3cdcf44f32949a5dad9036e16e5c',
    'worker.log': '6d144ea0abe44bb059d8a8cc4fcc5371e5413b3692c86a9f573140b11e929c48',
    'arima_fix_admission.json': '153fccf65a54a9b2d08e66a7329a535ad2f81f8223031b039600ad5e4944813d',
    'execution_extension_admission.json': 'b752d0206243af49d831143d3d2e7ee01e64c1a58331f82a323f8325c4186f1b',
    'scientific_outputs/scoring_started.json': 'd8619e5d62679c846c74d851768d9bea3d8c4e58f20297ee980577d5ae2714e9',
}
RECOVERY_DIRECTORY = 'score-worker-source-recovery-v1'
ABSENT_DAILY = 'private-known-E1-source-absent/daily_market_test.csv'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def policy_template():
    value = {'version': VERSION, 'protocol_id': 'e2-reduced-delivery-v1',
        'engine_commit': ENGINE_COMMIT, 'prior_arima_commit': PRIOR_ARIMA_COMMIT,
        'status': 'registered_pre_scoring_acquisition_continuation',
        'failed_worker_spec_sha256': FAILED_SPEC, 'failed_logical_job_id': FAILED_JOB,
        'failed_files_sha256': dict(OLD_FILES), 'failed_charge_sha256': FAILED_CHARGE,
        'failed_charged_seconds': FAILED_SECONDS, 'final_freeze_file_sha256': FREEZE_FILE,
        'final_freeze_canonical_sha256': FREEZE_DIGEST,
        'source_route': 'Original frozen E loader; twenty hash-pinned Binance archives',
        'raw_source_identity_changed': False, 'raw_download_opt_in_preserved': True,
        'raw_root_unchanged': True, 'absent_daily_relative_path': ABSENT_DAILY,
        'new_scoring_directory': RECOVERY_DIRECTORY,
        'daily_serialization': 'Raw-route daily serialization may differ from the prior E1 CSV; archive bytes and aggregation remain pinned',
        'old_E1_daily_csv_hash_forced_on_raw_route': False,
        'scientific_scoring_already_performed': False, 'scientific_scoring_retry_allowed': False,
        'retraining_allowed': False, 'original_failed_attempt_preserved': True,
        'existing_total_budget_hours': 80, 'new_budget_authorization': False,
        'budget_reset': False, 'persistent_destination': 'github'}
    value['sha256'] = digest(value)
    return value


def _regular(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if (not path.is_relative_to(root) or not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
            or not path.is_file()):
        raise ValueError('A regular original scoring artifact is required')
    return path


def _registration(commit, root, engine):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('An immutable acquisition-continuation registration is required')
    root = Path(root).absolute()
    def published(relative):
        return subprocess.check_output(['git', '-C', str(engine), 'show', commit + ':' + relative], stderr=subprocess.DEVNULL)
    try:
        raw = _regular(root/'registration.json', root).read_bytes()
        if raw != published(REGISTRATION_PATH):
            raise ValueError('The local acquisition registration differs from Git')
        record = json.loads(raw)
        source = _regular(root/SOURCE_PATH, root).read_bytes()
        policy_raw = _regular(root/POLICY_PATH, root).read_bytes()
        policy = json.loads(policy_raw)
        expected = {'engine_commit': ENGINE_COMMIT, 'protocol_id': 'e2-reduced-delivery-v1',
            'prior_arima_commit': PRIOR_ARIMA_COMMIT, 'funding_continuation_commit': FUNDING_COMMIT}
        if (any(record.get(k) != v for k,v in expected.items())
                or policy != policy_template() or source != published(SOURCE_PATH)
                or source != Path(__file__).read_bytes() or policy_raw != published(POLICY_PATH)
                or record.get('policy_sha256') != policy['sha256']
                or record.get('policy_file_sha256') != hashlib.sha256(policy_raw).hexdigest()
                or any(record.get('files_sha256',{}).get(p) != hashlib.sha256(b).hexdigest()
                    for p,b in [(SOURCE_PATH,source),(POLICY_PATH,policy_raw)])):
            raise ValueError('The registered acquisition policy or executable differs')
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        raise ValueError('The published acquisition registration is unavailable') from None
    return {'registration_commit':commit, 'registration_path':REGISTRATION_PATH,
        'registration_file_sha256':hashlib.sha256(raw).hexdigest(),
        'policy_sha256':policy['sha256'], 'source_sha256':hashlib.sha256(source).hexdigest()}


def verify_failure_records(spec_raw, failure_raw, marker_raw, ledger_raw, freeze_raw):
    """Metadata-only verification, also used with the real original evidence."""
    for name,raw in [('worker_spec.json',spec_raw),('worker_failure.json',failure_raw),
            ('scientific_outputs/scoring_started.json',marker_raw)]:
        if hashlib.sha256(raw).hexdigest() != OLD_FILES[name]:
            raise ValueError('The original failed scoring evidence changed: '+name)
    spec,failure,marker,freeze = map(json.loads,(spec_raw,failure_raw,marker_raw,freeze_raw))
    unsigned = dict(spec); expected = unsigned.pop('worker_spec_sha256',None)
    if (expected != FAILED_SPEC or digest(unsigned) != FAILED_SPEC
            or digest({'version':'e2-reduced-final-executor-v1','spec_sha256':FAILED_SPEC}) != FAILED_JOB
            or spec.get('kind') != 'final-score' or spec.get('model') != 'all16'
            or spec.get('daily_csv') is not None or spec.get('allow_download') is not True
            or spec.get('source_identity',{}).get('source_commit') != ENGINE_COMMIT
            or failure.get('status') != 'failed' or failure.get('error_type') != 'HTTPError'
            or failure.get('reason') != 'HTTP Error 404: Not Found'
            or failure.get('automatic_retry') is not False
            or hashlib.sha256(freeze_raw).hexdigest() != FREEZE_FILE
            or digest(freeze) != FREEZE_DIGEST or spec.get('final_freeze_sha256') != FREEZE_FILE
            or marker != {'freeze_sha256':FREEZE_DIGEST,
                'analysis_registration_sha256':freeze['analysis_registration_sha256'],'retune_allowed':False}):
        raise ValueError('The exact pre-payload HTTP404 and frozen predictors are required')
    if not ledger_raw or not ledger_raw.endswith(b'\n'):
        raise ValueError('A complete original budget journal is required')
    previous='0'*64; reserves=[];charges=[]
    for index,line in enumerate(ledger_raw.splitlines()):
        row=json.loads(line)
        if (row.get('sequence') != index or row.get('previous_sha256') != previous
                or row.get('sha256') != digest({k:v for k,v in row.items() if k!='sha256'})):
            raise ValueError('The original scoring budget chain changed')
        previous=row['sha256']
        if row.get('attempt_id') == FAILED_JOB:
            if row.get('kind')=='reserved':reserves.append(row)
            elif row.get('kind')=='charged':charges.append(row)
    if len(reserves)!=1 or len(charges)!=1:
        raise ValueError('The original failed acquisition must remain reserved and charged exactly once')
    reserve,charge=reserves[0],charges[0];usage=charge.get('resource_use',{})
    if (charge.get('sha256')!=FAILED_CHARGE or charge.get('status')!='failed'
            or charge.get('elapsed_seconds')!=FAILED_SECONDS or charge.get('reservation_overrun') is not False
            or reserve.get('reservation_seconds')!=1860 or charge.get('reservation_seconds')!=1860
            or reserve.get('metadata',{}).get('worker_spec_sha256')!=FAILED_SPEC
            or usage.get('worker_spec_sha256')!=FAILED_SPEC or usage.get('failure')!='worker_exit_2'
            or usage.get('optimizer_steps_unknown') is not True or usage.get('automatic_retry') is not False):
        raise ValueError('The original failed acquisition cost or uncertainty changed')
    return {'spec':spec,'failure':failure,'marker':marker,'freeze':freeze,'charge':charge}


def _guard_private(output):
    private=Path(output)/'private-known-E1-data'
    if private.exists() or private.is_symlink():
        raise ValueError('reconciliation_required: a local final data admission or partial decode already exists')
    sentinel=Path(output)/ABSENT_DAILY
    if sentinel.exists() or any(p.is_symlink() for p in (sentinel,*sentinel.parents)):
        raise ValueError('The raw-route daily sentinel must remain absent and unlinked')


def make_run_final_scoring(final, *, registration_commit, registration_root, engine_root):
    proof=_registration(registration_commit,registration_root,engine_root)
    def run_final_scoring(project_root,data_root,output,budget,*,source_identity,runtime_identity,
            source_roots=None,device='cuda:0',daily_csv=None,raw_root=None,allow_download=False,
            persist_result=None,python=None,supervisor=None):
        from e2_reduced_final_data import validate_final_freeze
        output=Path(output).absolute(); old=output/'score-worker'
        ledger_path=_regular(Path(budget.root)/'pilot_budget.jsonl',Path(budget.root))
        old_paths={}
        for path in old.rglob('*'):
            if path.is_symlink():raise ValueError('A failed scoring artifact must not be a symlink')
            if path.is_file():old_paths[path.relative_to(old).as_posix()]=path
        if set(old_paths)!=set(OLD_FILES):
            raise ValueError('The original score output contains missing or unexpected scientific artifacts')
        for name,path in old_paths.items():
            if hashlib.sha256(_regular(path,output).read_bytes()).hexdigest()!=OLD_FILES[name]:
                raise ValueError('The original failed scoring artifact changed: '+name)
        ledger=ledger_path.read_bytes(); freeze_raw=_regular(output/'final_freeze.json',output).read_bytes()
        evidence=verify_failure_records(old_paths['worker_spec.json'].read_bytes(),
            old_paths['worker_failure.json'].read_bytes(),old_paths['scientific_outputs/scoring_started.json'].read_bytes(),ledger,freeze_raw)
        if ledger_path.read_bytes()!=ledger:raise ValueError('Budget changed during acquisition reconciliation')
        base=final._base_spec(data_root,source_identity,runtime_identity,source_roots,device)
        expected={**base,'kind':'final-score','model':'all16','project_root':str(Path(project_root).resolve()),
            'final_output':str(output.resolve()),'final_freeze_sha256':FREEZE_FILE,
            'daily_csv':None if daily_csv is None else str(Path(daily_csv).resolve()),
            'raw_root':None if raw_root is None else str(Path(raw_root).resolve()),'allow_download':allow_download,
            'output':str(old.resolve()),'fit_wall_seconds':budget.stage_wall_seconds}
        expected['worker_spec_sha256']=digest(expected)
        if evidence['spec']!=expected or Path(raw_root).resolve()!=output.parent/'private_raw_E1':
            raise ValueError('Preserve the original frozen scoring specification, raw root and download consent')
        validate_final_freeze(evidence['freeze'],output)
        recovery=output/RECOVERY_DIRECTORY
        completed=(recovery/'result.json').is_file()
        if not completed:
            _guard_private(output)
            if recovery.exists() and any(recovery.iterdir()):
                raise ValueError('reconciliation_required: partial acquisition continuation is preserved without retry')
            disposition={ 'version':VERSION,'status':'pre_payload_failure_reconciled',
                'original_failed_job_id':FAILED_JOB,'original_worker_spec_sha256':FAILED_SPEC,
                'original_failed_files_sha256':dict(OLD_FILES),'original_charge_sha256':FAILED_CHARGE,
                'original_charged_wall_seconds':FAILED_SECONDS,'optimizer_steps_unknown':True,
                'optimizer_steps':None,'original_attempt_preserved':True,
                'final_freeze_sha256':FREEZE_FILE,'final_data_admission_found_before_continuation':False,
                'scientific_score_repeated':False,'neural_refits_requested':False,
                'raw_root_unchanged':True,'allow_download':True,'policy_proof':proof}
            disposition_path=output/'scoring_source_recovery'/'pre_payload_failure_disposition.json'
            final.write_json(disposition_path,disposition)
            if persist_result is not None:
                persist_result(disposition_path.parent,{**disposition,
                    'attempt_id':'score-source-recovery-admission-'+FAILED_JOB[:32],
                    'artifacts_sha256':{disposition_path.name:hashlib.sha256(disposition_path.read_bytes()).hexdigest()}})
        spec={**base,'kind':'final-score','model':'all16','project_root':str(Path(project_root).resolve()),
            'final_output':str(output.resolve()),'final_freeze_sha256':FREEZE_FILE,
            'daily_csv':str((output/ABSENT_DAILY).resolve()),'raw_root':str(Path(raw_root).resolve()),
            'allow_download':True}
        return final._dispatch(spec,recovery,budget,persist_result=persist_result,python=python,supervisor=supervisor)
    return run_final_scoring
