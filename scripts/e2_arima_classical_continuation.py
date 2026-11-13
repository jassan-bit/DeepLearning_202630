"""Account for one sealed implementation failure without repeating its fit.

The failed ARIMA opportunity remains failed and charged. A separate disposition
records the missing CV score; the original successful-result path is never
created. Remaining registered opportunities use the unchanged E dispatcher.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

VERSION = 'e2-arima-classical-continuation-v1'
ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
REGISTRATION_PATH = 'experiments/e2_arima_compat_20261113/registration.json'
POLICY_PATH = 'experiments/e2_arima_compat_20261113/policy.json'
SOURCE_PATH = 'scripts/e2_arima_classical_continuation.py'
FAILED_JOB = 'bd7439962faa00ce5106965056819e0280c5727b2b7b5e90d8ff60aeaf34e9fb'
FAILED_SPEC = '539b3b89c470619daeccf312e4e88a283972ef07eb22b64a1a7746b4ffb26c02'
FAILED_SPEC_FILE = '4277979498072acfd3095533ee24931561826fa26ed531f7fd553541682a45fa'
FAILED_WORKER_FILE = '3d2a02d4194d3a1d36931c04bd78c04a2c9fc3e70bca8f38da702be624180718'
FAILED_CHARGE = '7952fead459a6db83b34418ce45392dcefdedbcb85da830845916735f8bfba55'
FAILED_SECONDS = 14.505011292001655
FAILED_RELATIVE = 'classical-cv/6c395251bcaf9062/1f8780bc6fc9c663/2024-01-01'
FAILURE_REASON = "'Prediction' object has no attribute 'raw_value'"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def policy_template():
    value = {'version': VERSION, 'engine_commit': ENGINE_COMMIT,
        'protocol_id': 'e2-reduced-delivery-v1', 'status': 'registered_implementation_repair',
        'failed_logical_job_id': FAILED_JOB, 'failed_worker_spec_sha256': FAILED_SPEC,
        'failed_worker_spec_file_sha256': FAILED_SPEC_FILE,
        'failed_worker_failure_file_sha256': FAILED_WORKER_FILE,
        'failed_charge_sha256': FAILED_CHARGE, 'failed_charged_seconds': FAILED_SECONDS,
        'failed_relative_directory_under_final': FAILED_RELATIVE,
        'failed_model': 'ARIMA', 'failed_mode': 'classical-cv',
        'failed_order': [0, 0, 0], 'failed_fold_origin': '2024-01-01',
        'failure_type': 'AttributeError', 'failure_reason': FAILURE_REASON,
        'diagnostic_raw_prediction_field': 'Prediction.raw',
        'numerical_prediction_field_unchanged': 'Prediction.value',
        'failed_fit_repeated': False, 'new_budget_authorization': False,
        'existing_total_budget_hours': 80, 'budget_reset': False,
        'planned_classical_cv_opportunities': 63,
        'missing_cv_score_policy': 'One implementation-failed opportunity has no RMSE; its order is eligible only with three valid folds',
        'scientific_retry_allowed': False, 'persistent_destination': 'github'}
    value['sha256'] = digest(value)
    return value


def validate_policy(value):
    if value != policy_template():
        raise ValueError('The exact registered ARIMA implementation-failure policy is required')
    return dict(value)


def _regular(path, root):
    path, root = Path(path), Path(root).absolute()
    if (not path.absolute().is_relative_to(root) or not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents) if p.absolute().is_relative_to(root))
            or not path.is_file()):
        raise ValueError('A regular bound classical artifact is required')
    return path


def _sha(path, root):
    return hashlib.sha256(_regular(path, root).read_bytes()).hexdigest()


def _registration(policy, commit, root):
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('An immutable classical-repair registration is required')
    def published(path):
        return subprocess.check_output(['git', '-C', str(root), 'show', commit + ':' + path],
            stderr=subprocess.DEVNULL)
    try:
        raw = published(REGISTRATION_PATH)
        record = json.loads(raw)
        policy_raw, source = published(POLICY_PATH), published(SOURCE_PATH)
    except (OSError, ValueError, subprocess.CalledProcessError):
        raise ValueError('The published classical-repair registration is unavailable') from None
    expected = record.get('files_sha256', {}).get(SOURCE_PATH)
    if (record.get('engine_commit') != ENGINE_COMMIT
            or record.get('protocol_id') != 'e2-reduced-delivery-v1'
            or json.loads(policy_raw) != policy
            or record.get('policy_sha256') != policy['sha256']
            or record.get('policy_file_sha256') != hashlib.sha256(policy_raw).hexdigest()
            or record.get('files_sha256', {}).get(POLICY_PATH) != hashlib.sha256(policy_raw).hexdigest()
            or expected != hashlib.sha256(source).hexdigest()
            or expected != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('The classical policy or executable source differs from its registration')
    return {'registration_commit': commit, 'registration_path': REGISTRATION_PATH,
        'registration_file_sha256': hashlib.sha256(raw).hexdigest(),
        'policy_sha256': policy['sha256'], 'source_sha256': expected}


def _charge_from_bytes(raw):
    if not raw or not raw.endswith(b'\n'):
        raise ValueError('The existing classical budget journal must have a complete tail')
    previous = '0' * 64
    reservations, charges = [], []
    for index, line in enumerate(raw.splitlines()):
        row = json.loads(line)
        if (row.get('sequence') != index or row.get('previous_sha256') != previous
                or row.get('sha256') != digest({k:v for k,v in row.items() if k != 'sha256'})):
            raise ValueError('The original classical budget chain changed')
        previous = row['sha256']
        if row.get('attempt_id') == FAILED_JOB:
            if row.get('kind') == 'reserved': reservations.append(row)
            elif row.get('kind') == 'charged': charges.append(row)
    if len(reservations) != 1 or len(charges) != 1:
        raise ValueError('The failed ARIMA opportunity must have exactly one reservation and charge')
    reserve, charge = reservations[0], charges[0]
    usage = charge.get('resource_use', {})
    if (charge.get('sha256') != FAILED_CHARGE or charge.get('status') != 'failed'
            or charge.get('elapsed_seconds') != FAILED_SECONDS
            or charge.get('reservation_overrun') is not False
            or charge.get('reservation_seconds') != 1860
            or reserve.get('reservation_seconds') != 1860
            or reserve.get('metadata', {}).get('worker_spec_sha256') != FAILED_SPEC
            or usage.get('worker_spec_sha256') != FAILED_SPEC
            or usage.get('failure') != 'worker_exit_2'
            or usage.get('optimizer_steps_unknown') is not True
            or usage.get('automatic_retry') is not False):
        raise ValueError('The original charged implementation-failure binding changed')
    return charge


def verify_failure_records(spec_raw, failure_raw, ledger_raw):
    """Verify the sealed original metadata without opening data or fitting."""
    if (hashlib.sha256(spec_raw).hexdigest() != FAILED_SPEC_FILE
            or hashlib.sha256(failure_raw).hexdigest() != FAILED_WORKER_FILE):
        raise ValueError('The sealed original ARIMA failure files changed')
    spec, failure = json.loads(spec_raw), json.loads(failure_raw)
    unsigned = dict(spec)
    expected = unsigned.pop('worker_spec_sha256', None)
    if (expected != FAILED_SPEC or digest(unsigned) != expected
            or digest({'version': 'e2-reduced-final-executor-v1', 'spec_sha256': expected}) != FAILED_JOB
            or spec.get('model') != 'ARIMA' or spec.get('mode') != 'classical-cv'
            or spec.get('parameters') != {'order': [0, 0, 0]}
            or spec.get('fold_origin') != '2024-01-01'
            or spec.get('source_identity', {}).get('source_commit') != ENGINE_COMMIT
            or failure.get('status') != 'failed' or failure.get('error_type') != 'AttributeError'
            or failure.get('reason') != FAILURE_REASON or failure.get('automatic_retry') is not False):
        raise ValueError('The exact original ARIMA implementation failure is required')
    return {'spec': spec, 'failure': failure, 'budget_charge': _charge_from_bytes(ledger_raw)}


def _implementation_failure(final, directory, base_spec, budget, proof, policy, persist_result):
    output = Path(budget.root).absolute().parent
    directory = Path(directory).absolute()
    if directory.relative_to(output / 'final').as_posix() != FAILED_RELATIVE:
        raise ValueError('The failed classical opportunity belongs to another directory')
    if (directory / 'result.json').exists() or (directory / 'predictor.pkl').exists():
        raise ValueError('A previous successful result or recovered model requires reconciliation')
    spec_path = _regular(directory / 'worker_spec.json', output)
    failure_path = _regular(directory / 'worker_failure.json', output)
    ledger_path = _regular(Path(budget.root) / 'pilot_budget.jsonl', Path(budget.root))
    ledger_raw = ledger_path.read_bytes()
    verified = verify_failure_records(spec_path.read_bytes(), failure_path.read_bytes(), ledger_raw)
    if ledger_path.read_bytes() != ledger_raw:
        raise ValueError('The classical journal changed during failure verification')
    spec, failure, charge = verified['spec'], verified['failure'], verified['budget_charge']
    expected_spec = {**base_spec, 'kind': 'classical', 'mode': 'classical-cv',
        'model': 'ARIMA', 'parameters': {'order': [0, 0, 0]},
        'fold_origin': '2024-01-01', 'output': str(directory.resolve()),
        'fit_wall_seconds': budget.stage_wall_seconds}
    expected_spec['worker_spec_sha256'] = digest(expected_spec)
    if (_sha(spec_path, output) != FAILED_SPEC_FILE
            or _sha(failure_path, output) != FAILED_WORKER_FILE
            or spec != expected_spec or spec.get('worker_spec_sha256') != FAILED_SPEC
            or digest({'version': final.VERSION, 'spec_sha256': FAILED_SPEC}) != FAILED_JOB
            or failure.get('status') != 'failed' or failure.get('error_type') != 'AttributeError'
            or failure.get('reason') != FAILURE_REASON or failure.get('automatic_retry') is not False):
        raise ValueError('The exact sealed ARIMA implementation failure is required')
    disposition = {'version': VERSION, 'status': 'failed',
        'candidate_status': 'implementation_failed',
        'attempt_id': 'arima-implementation-failure-' + FAILED_JOB[:48],
        'original_logical_job_id': FAILED_JOB, 'worker_spec_sha256': FAILED_SPEC,
        'worker_spec_file_sha256': FAILED_SPEC_FILE,
        'worker_failure_file_sha256': FAILED_WORKER_FILE,
        'budget_charge_sha256': charge['sha256'], 'charged_wall_seconds': FAILED_SECONDS,
        'optimizer_steps_unknown': True, 'optimizer_steps': None,
        'convergence_status': 'not_inferred_from_implementation_failure',
        'model': 'ARIMA', 'parameters': {'order': [0, 0, 0]}, 'fold': '2024-01-01',
        'planned_validation_count': 83, 'completed_validation_count': 0,
        'RMSE': None, 'scientific_fit_repeated': False, 'automatic_retry': False,
        'registration_proof': proof, 'policy_sha256': policy['sha256'],
        'original_failure_preserved': True}
    path = directory / 'failure_disposition.json'
    final.write_json(path, disposition)
    if persist_result is not None:
        artifacts = {p.name: _sha(p, output) for p in
            [path, spec_path, failure_path]}
        receipt = {**disposition, 'artifacts_sha256': artifacts}
        persist_result(directory, receipt)
    return {'parameters': {'order': [0, 0, 0]}, 'fold': '2024-01-01',
        'candidate_status': 'implementation_failed', 'reason': FAILURE_REASON,
        'failure_disposition_path': path.relative_to(output).as_posix(),
        'failure_disposition_sha256': _sha(path, output),
        'planned_validation_count': 83, 'completed_validation_count': 0,
        'result_sha256': None, 'RMSE': None}


def make_run_classical(final, policy, *, registration_commit,
        registration_git_root, supervisor=None):
    """Return E's classical orchestration with one terminal failure accounted."""
    policy = validate_policy(policy)
    proof = _registration(policy, registration_commit, registration_git_root)
    repaired_supervisor = supervisor
    def run_classical(data_root, output, budget, *, source_identity, runtime_identity,
            device='cuda:0', persist_result=None, python=None, supervisor=None):
        base = final._base_spec(data_root, source_identity, runtime_identity, None, device)
        known_failure = _implementation_failure(final, Path(output) / FAILED_RELATIVE,
            base, budget, proof, policy, persist_result)
        results = []
        dispatch = repaired_supervisor if supervisor is None else supervisor
        for name in final.CLASSICAL:
            if name == 'Persistence': chosen, ledger = {'lag': 1}, []
            else:
                configs = ([{'period': period} for period in (7, 14, 28)] if name == 'SeasonalNaive'
                    else [{'order': [p, d, q]} for p in range(3) for d in range(2) for q in range(3)])
                candidates, ledger = [], []
                for config in configs:
                    scores = []
                    for fold in final.FOLDS:
                        directory = Path(output) / 'classical-cv' / final.slug(name) / digest(config)[:16] / fold
                        if name == 'ARIMA' and config == {'order': [0, 0, 0]} and fold == '2024-01-01':
                            ledger.append(dict(known_failure))
                            continue
                        spec = {**base, 'kind': 'classical', 'mode': 'classical-cv',
                            'model': name, 'parameters': config, 'fold_origin': fold}
                        row = final._dispatch(spec, directory, budget, persist_result=persist_result,
                            python=python, supervisor=dispatch)
                        if row['validation_count'] != 83:
                            raise ValueError('Classical common CV support changed')
                        status = row.get('candidate_status', 'completed')
                        if status == 'completed': scores.append(row['rmse_original'])
                        ledger.append({'parameters': config, 'fold': fold, 'candidate_status': status,
                            'reason': row.get('reason'), 'result_sha256': final.file_sha(directory / 'result.json'),
                            'RMSE': row['rmse_original']})
                    if len(scores) == 3:
                        candidates.append((math.sqrt(sum(v*v for v in scores)/3), digest(config), config))
                if not candidates:
                    raise ValueError('No classical configuration completed all three folds; do not substitute')
                chosen = min(candidates, key=lambda item: (item[0], item[1]))[2]
            selection_path = Path(output) / 'classical-selections' / (final.slug(name) + '.json')
            final.write_json(selection_path, {'model': name, 'parameters': chosen,
                'ledger': ledger, 'selection_metric': 'pooled_original_unit_CV_RMSE',
                'calibration_used_for_selection': False})
            directory = Path(output) / 'classical-final' / final.slug(name)
            spec = {**base, 'kind': 'classical', 'mode': 'classical-final',
                'model': name, 'parameters': chosen}
            results.append(final._dispatch(spec, directory, budget, persist_result=persist_result,
                python=python, supervisor=dispatch))
        return results
    return run_classical
