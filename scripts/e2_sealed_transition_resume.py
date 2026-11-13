"""Read-only reuse of the Q-sealed T4/A100 transition after HPO is closed.

Later cache admissions must never choose a new historical provenance witness.
The original three transition artifacts remain byte-for-byte unchanged. A
fresh, strictly compatible A100 observation is disclosed in a separate return
receipt; it is never relabelled as the historical whole-file observation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
Q_COMMIT = '72c04635b7a746d06066273ce61e891975feb510'
FINAL_VERSION = 'e2-reduced-final-executor-v1'
CHECKS = {'candidate_payload', 'recipe_identity', 'complete_validation_calendar',
    'known_completed_updates', 'prepared_inputs', 'job_payload', 'worker_runtime'}


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _regular(path, *, within=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('A regular existing sealed artifact is required: ' + str(path))
    if within is not None:
        root = Path(within).absolute()
        if not path.absolute().is_relative_to(root) or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('A sealed artifact escaped its control root')
        for part in (path, *path.parents):
            if part.absolute().is_relative_to(root) and part.is_symlink():
                raise ValueError('A sealed artifact must not contain a symlink')
    return path


def _read(path, *, within=None):
    return json.loads(_regular(path, within=within).read_bytes())


def _sha(path, *, within=None):
    return hashlib.sha256(_regular(path, within=within).read_bytes()).hexdigest()


def _bound_path(output, proof, path_key, hash_key):
    relative = proof.get(path_key)
    part = Path(relative) if isinstance(relative, str) else None
    if part is None or part.is_absolute() or '..' in part.parts:
        raise ValueError('A safe historical provenance path is required')
    path = _regular(output / part, within=output)
    if _sha(path, within=output) != proof.get(hash_key):
        raise ValueError('Historical provenance artifact changed: ' + relative)
    return path


def _verify_final_results(output, actual_runtime, budget):
    from e2_reduced_supervision import verified_worker_result
    rows = []
    for result_path in sorted((output / 'final/final-fit').glob('*/*/result.json')):
        directory = result_path.parent
        spec_path = directory / 'worker_spec.json'
        spec = _read(spec_path, within=output)
        unsigned = dict(spec)
        expected = unsigned.pop('worker_spec_sha256', None)
        if (spec.get('version') != FINAL_VERSION or not isinstance(expected, str)
                or _digest(unsigned) != expected
                or spec.get('kind') != 'neural-final'
                or spec.get('source_identity', {}).get('source_commit') != ENGINE_COMMIT
                or spec.get('runtime_identity') != actual_runtime):
            raise ValueError('Current A100 identity differs from a sealed completed final fit')
        result = verified_worker_result(directory, expected)
        if (result is None or result.get('model') != spec.get('model')
                or result.get('seed') != spec.get('seed')
                or result.get('selection_sha256') != _digest(spec.get('selection'))):
            raise ValueError('A sealed final fit does not bind its original winning recipe')
        logical_id = _digest({'version': FINAL_VERSION, 'spec_sha256': expected})
        charge = budget.completed_receipt(logical_id, expected_resource_use={
            'worker_spec_sha256': expected, 'result_sha256': _sha(result_path, within=output)})
        if charge is None:
            raise ValueError('A sealed final fit lacks its completed budget charge')
        rows.append({'result_path': result_path.relative_to(output).as_posix(),
            'result_sha256': _sha(result_path, within=output),
            'worker_spec_sha256': expected, 'budget_charge_sha256': charge.get('sha256')})
    if not rows:
        raise ValueError('This final continuation requires already sealed A100 final fits')
    return rows


def prepare_sealed_historical_cache(args, budget, *, registration_commit,
        transition_directory=None, resume_registration_commit=None):
    """Validate recorded witnesses, never search again or rewrite old artifacts."""
    import e2_a100_cache_selection as q
    if registration_commit != Q_COMMIT:
        raise ValueError('The original Q hardware-transition authority must remain unchanged')
    if resume_registration_commit is not None and not re.fullmatch('[0-9a-f]{40}', resume_registration_commit):
        raise ValueError('An immutable runtime-resume registration is required')
    output = Path(args.output).absolute()
    transition = output / 'hardware_transition' if transition_directory is None else Path(transition_directory).absolute()
    if not transition.resolve().is_relative_to(output.resolve()):
        raise ValueError('The sealed transition must remain inside its control root')
    historical_path = transition / 'historical_t4_profile.json'
    audit_path = transition / 'historical_cache_audit.json'
    receipt_path = transition / 'transition.json'
    historical = _read(historical_path, within=output)
    report = _read(audit_path, within=output)
    receipt = _read(receipt_path, within=output)
    actual = _read(args.runtime_profile)
    actual_runtime = q._strict_current(actual)
    required = {'version': q.TRANSITION_VERSION, 'engine_commit': ENGINE_COMMIT,
        'registration_commit': Q_COMMIT, 'historical_scores_modified': False,
        'optimizer_retry_performed': False, 'budget_reset': False,
        'authorized_total_hours': 72}
    if (any(type(receipt.get(k)) is not type(v) or receipt.get(k) != v for k, v in required.items())
            or receipt.get('sha256') != _digest({k:v for k,v in receipt.items() if k != 'sha256'})
            or receipt.get('historical_runtime_profile_sha256') != _sha(historical_path, within=output)
            or receipt.get('cache_audit_sha256') != _sha(audit_path, within=output)
            or historical.get('runtime', {}).get('device_name') != 'Tesla T4'
            or historical.get('fit_execution_policy_verified') is not True
            or historical.get('fit_execution_policy') != actual.get('fit_execution_policy')):
        raise ValueError('The original sealed hardware-transition binding changed')
    reconciliation = _read(Path(args.reconciliation_root) / 'budget_evidence.json')
    if receipt.get('historical_heads') != reconciliation.get('historical_heads'):
        raise ValueError('Historical reconciliation heads changed')
    source = receipt.get('historical_runtime_source', {})
    attestation = receipt.get('historical_policy_attestation', {})
    if source != historical.get('historical_runtime_source') or attestation != historical.get('historical_policy_attestation'):
        raise ValueError('The original historical provenance witnesses changed')
    spec_path = _bound_path(output, source, 'worker_spec_path', 'worker_spec_file_sha256')
    result_path = _bound_path(output, source, 'result_path', 'result_file_sha256')
    admission_path = _bound_path(output, attestation, 'cache_admission_path', 'cache_admission_file_sha256')
    cached_path = _bound_path(output, attestation, 'cached_result_path', 'cached_result_file_sha256')
    spec, result = _read(spec_path), _read(result_path)
    if (spec.get('runtime_identity') != historical.get('runtime')
            or spec.get('source_sha256') != q.original.source_identity()
            or result.get('status') != 'completed' or result.get('cache_reuse') is True
            or spec.get('job') != result.get('job')
            or result.get('worker_spec_sha256') != _digest(spec)):
        raise ValueError('The sealed historical T4 fit identity changed')
    charge = budget.completed_receipt(spec['job']['logical_job_id'], expected_resource_use={
        'worker_spec_sha256': _digest(spec), 'result_sha256': _sha(result_path, within=output)})
    if charge is None or charge.get('sha256') != source.get('budget_charge_sha256'):
        raise ValueError('The original historical T4 completed budget charge changed')
    admission, cached = _read(admission_path), _read(cached_path)
    if (cached.get('status') != 'completed' or cached.get('cache_reuse') is not True
            or cached.get('artifacts_sha256', {}).get('cache_admission.json') != _sha(admission_path, within=output)
            or any(admission.get(k) is not True for k in ('reuse_approved', 'runtime_facts_match',
                'fit_execution_policy_attested', 'fit_execution_policy_match'))
            or set(admission.get('checks', {})) != CHECKS
            or not all(v is True for v in admission['checks'].values())):
        raise ValueError('The original historical cache-policy attestation changed')
    final_results = _verify_final_results(output, actual_runtime, budget)
    current_sha = _sha(args.runtime_profile)
    same_observation = current_sha == receipt.get('current_runtime_profile_sha256')
    if not same_observation and (resume_registration_commit is None or resume_registration_commit == Q_COMMIT):
        raise ValueError('A distinct registered receipt is required for a fresh compatible A100 observation')
    args.historical_runtime_profile = historical_path
    args.cache_audit = audit_path
    args.hardware_transition_receipt = receipt_path
    args.hardware_transition_registration_commit = Q_COMMIT
    value = {'version': 'e2-sealed-transition-resume-v1',
        'engine_commit': ENGINE_COMMIT, 'hardware_transition_registration_commit': Q_COMMIT,
        'resume_registration_commit': resume_registration_commit,
        'historical_profile_sha256': _sha(historical_path, within=output),
        'historical_cache_audit_sha256': _sha(audit_path, within=output),
        'original_transition_sha256': _sha(receipt_path, within=output),
        'original_A100_profile_sha256': receipt.get('current_runtime_profile_sha256'),
        'current_A100_profile_sha256': current_sha,
        'current_observation_is_original_profile_file': same_observation,
        'current_runtime_identity_sha256': _digest(actual_runtime),
        'all_saved_final_runtime_identities_match_current': True,
        'verified_completed_final_fits': final_results,
        'historical_metadata_regenerated': False, 'historical_files_modified': False,
        'market_training_started_here': False, 'scientific_fit_repeated': False,
        'budget_extended_here': False}
    value['sha256'] = _digest(value)
    return {'historical_runtime_profile': historical_path, 'cache_audit': audit_path,
        'hardware_transition_receipt': receipt_path,
        'historical_cache_reuse_approved': report.get('reuse_approved'), 'resume_admission': value}
