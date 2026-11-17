"""Synthetic artifact tests only: never construct or run a market model."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import pytest

import aggregate as subject


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(subject.canonical(value)+b'\n')


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    inputs, run = tmp_path/'inputs', tmp_path/'run'
    inputs.mkdir(); run.mkdir()
    y, calibration_targets = [.01]*608, [.01]*53
    calibration_dates = ['2024-11-%02d' % i for i in range(8, 31)]
    calibration_dates += ['2024-12-%02d' % i for i in range(1, 31)]
    def predictions(seed):
        i = seed-271800
        return [y[0]+(-1 if i % 2 else 1)*i*.0001]*608
    old_rows = [{'model': model, 'seed': seed, 'prediction_original': predictions(seed),
        'calibration_targets': calibration_targets, 'checkpoint_sha256': 'f'*64,
        'extra_original_field': {'nested': [None, False, 5.125], 'text': 'preserve me'}}
        for model in subject.NEURAL for seed in subject.OLD_SEEDS]
    old_rows += [{'model': model, 'seed': 'deterministic', 'prediction_original': [.011]*608,
                 'original_classical_field': {'not_repeated': True}} for model in subject.CLASSICAL]
    old = {'dates': subject.DATES, 'y_true': y, 'mask': [True]*608, 'predictors': old_rows,
           'provenance': {'keep_this': True}, 'unrecognized_old_root_field': ['also retained']}
    write(inputs/'old_e2_predictions.json', old)
    jobs = [{'id': f'j{i:02d}-{seed}', 'model': model, 'seed': seed,
             'trial': {'frozen': model}, 'fixed_epochs': {'supervised': 1}, 'parameter_count': 2}
            for i, model in enumerate(subject.NEURAL) for seed in subject.NEW_SEEDS]
    plan = {'version': 'e2-seed-extension-v1', 'jobs': jobs}
    fingerprint = hashlib.sha256(subject.canonical(plan)).hexdigest()
    identity = {'version': plan['version'], 'run_id': 'synthetic-run', 'plan_fingerprint': fingerprint}
    write(run/'run_manifest.json', {**identity, 'jobs': jobs})
    gate_rows = []
    for job in jobs:
        directory = run/'jobs'/job['id']; directory.mkdir(parents=True)
        (directory/'fit').mkdir(); (directory/'fit/selected_model.pt').write_bytes(b'opaque fixture weights')
        write(directory/'scaler.json', {'target_scale': .01})
        checkpoint_sha = subject.file_sha(directory/'fit/selected_model.pt')
        cal_prediction = [.009]*53
        write(directory/'calibrator.json', {'q': abs(.01-.009), 'predictor_sha256': checkpoint_sha})
        write(directory/'calibration_predictions.json', {'dates': calibration_dates,
            'target_original': calibration_targets, 'prediction_original': cal_prediction,
            'checkpoint_sha256': checkpoint_sha})
        artifacts = {str(p.relative_to(directory)): subject.file_sha(p) for p in directory.rglob('*') if p.is_file()}
        receipt = {**identity, 'status': 'completed', 'job_id': job['id'], 'model': job['model'],
            'seed': job['seed'], 'training_examples': 655, 'calibration_examples': 53,
            'trial': job['trial'], 'fixed_phase_epochs': job['fixed_epochs'], 'artifacts_sha256': artifacts}
        for key, path in [('checkpoint', 'fit/selected_model.pt'), ('scaler', 'scaler.json'),
                          ('calibrator', 'calibrator.json'), ('calibration_predictions', 'calibration_predictions.json')]:
            receipt.update({key: path, key+'_sha256': subject.file_sha(directory/path)})
        write(directory/'fit_receipt.json', receipt)
        receipt_sha = subject.file_sha(directory/'fit_receipt.json')
        ack = {'job_id': job['id'], 'fit_receipt_sha256': receipt_sha, 'status': 'acknowledged', 'ack_commit': 'a'*40}
        write(run/'remote_acks'/(job['id']+'.json'), ack)
        gate_rows.append({'job_id': job['id'], 'fit_receipt_sha256': receipt_sha,
                         'remote_ack_sha256': subject.file_sha(run/'remote_acks'/(job['id']+'.json'))})
    write(run/'all_fits_remote_ack.json', {**identity, 'fits': gate_rows})
    for job in jobs:
        directory = run/'jobs'/job['id']; receipt = subject.read_json(directory/'fit_receipt.json')
        write(directory/'evaluation.json', {**identity, 'status': 'completed', 'job_id': job['id'],
            'model': job['model'], 'seed': job['seed'], 'dates': subject.DATES,
            'prediction_original': predictions(job['seed']), 'target_original': y,
            'n_dates': 608, 'batch_size': 1, 'inverse_scaling_count': 1,
            'checkpoint_sha256': receipt['checkpoint_sha256'], 'scaler_sha256': receipt['scaler_sha256'],
            'calibrator_sha256': receipt['calibrator_sha256'], 'fit_receipt_sha256': subject.file_sha(directory/'fit_receipt.json'),
            'all_fits_remote_ack_sha256': subject.file_sha(run/'all_fits_remote_ack.json')})
    e3_predictions, e3_metrics = [], []
    for seed in subject.ALL_SEEDS:
        values = [.01 + (seed-271800)*.00005]*608
        e3_predictions.extend({'seed': seed, 'date': day, 'target_sigma7': target, 'prediction_sigma7': p}
            for day, target, p in zip(subject.DATES, y, values))
        e3_metrics.append({'seed': seed, **subject.metrics(y, values), 'evaluation_n': 608})
    (inputs/'e3_evaluation_predictions.csv').write_bytes(subject.csv_bytes(e3_predictions))
    (inputs/'e3_per_seed_metrics.csv').write_bytes(subject.csv_bytes(e3_metrics))
    errors = [r['RMSE'] for r in e3_metrics]
    write(inputs/'e3_summary.json', {'complete_training_seeds': 10, 'comparison_complete': True,
        'RMSE_mean': statistics.fmean(errors), 'RMSE_sample_SD': statistics.stdev(errors)})
    monkeypatch.setattr(subject, 'SOURCE_HASHES', {name: subject.file_sha(inputs/name) for name in subject.SOURCE_HASHES})
    return inputs, run, plan, old


def test_complete_preserves_all_original_fields_and_source_bytes(campaign):
    inputs, run, plan, old = campaign
    before = (inputs/'old_e2_predictions.json').read_bytes()
    summary = subject.aggregate(inputs, run, plan)
    combined = subject.read_json(run/'combined_predictors.json')
    assert summary['complete'] and summary['new_fits'] == 52 and summary['E2_predictors'] == 133
    assert combined['predictors'][:81] == old['predictors']
    assert {k: combined[k] for k in old if k != 'predictors'} == {k: v for k, v in old.items() if k != 'predictors'}
    assert (inputs/'old_e2_predictions.json').read_bytes() == before
    assert summary['old_source_sha256'] == hashlib.sha256(before).hexdigest()
    assert summary['old_predictor_sha256'] == [hashlib.sha256(subject.canonical(r)).hexdigest() for r in old['predictors']]
    assert len([r for r in combined['predictors'] if r['seed'] == 'deterministic']) == 3
    assert not summary['statistical_tests_performed'] and not summary['R1_R3_verified']
    assert subject.aggregate(inputs, run, plan) == summary


def test_primary_is_mean_of_seed_rmse_not_ensemble_or_pooled(campaign):
    inputs, run, plan, _ = campaign
    subject.aggregate(inputs, run, plan)
    rows = list(csv.DictReader((run/'comparison.csv').open()))
    assert len(rows) == 17
    row = next(r for r in rows if r['model'] == subject.NEURAL[0])
    mean = float(row['RMSE_mean'])
    assert math.isclose(mean, .00055)
    assert not math.isclose(mean, math.sqrt(statistics.fmean((i*.0001)**2 for i in range(1, 11))))
    assert not math.isclose(mean, abs(statistics.fmean((-1 if i % 2 else 1)*i*.0001 for i in range(1, 11))))
    assert row['train_n'] == '655' and row['R'] == '10'
    assert rows[-1]['train_n'] == '532' and rows[-1]['R'] == '10'
    assert all(r['equal_training_budget_claimed'] == 'False' for r in rows)


@pytest.mark.parametrize('mutation', ['date', 'target', 'duplicate_seed', 'nonfinite', 'negative', 'binding'])
def test_rejects_changed_evaluation(campaign, mutation):
    inputs, run, plan, _ = campaign
    path = run/'jobs'/plan['jobs'][0]['id']/'evaluation.json'; value = subject.read_json(path)
    if mutation == 'date': value['dates'] = value['dates'][::-1]
    elif mutation == 'target': value['target_original'][0] += .0001
    elif mutation == 'duplicate_seed': value['seed'] = 271808
    elif mutation == 'nonfinite':
        path.write_text(json.dumps({**value, 'prediction_original': [float('nan')]*608}))
    elif mutation == 'negative': value['prediction_original'][0] = -.1
    elif mutation == 'binding': value['checkpoint_sha256'] = '0'*64
    if mutation != 'nonfinite': write(path, value)
    with pytest.raises(ValueError): subject.aggregate(inputs, run, plan)
    assert not (run/'summary.json').exists()


def test_rejects_missing_new_seed(campaign):
    inputs, run, plan, _ = campaign
    (run/'jobs'/plan['jobs'][-1]['id']/'evaluation.json').unlink()
    with pytest.raises(ValueError, match='Missing'): subject.aggregate(inputs, run, plan)


def test_rejects_original_source_byte_mutation(campaign):
    inputs, run, plan, _ = campaign
    path = inputs/'old_e2_predictions.json'; path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='Pinned source'): subject.aggregate(inputs, run, plan)


def test_rejects_duplicate_original_coordinate_even_when_rebound(campaign, monkeypatch):
    inputs, run, plan, old = campaign
    old['predictors'][1] = old['predictors'][0]
    write(inputs/'old_e2_predictions.json', old)
    monkeypatch.setitem(subject.SOURCE_HASHES, 'old_e2_predictions.json', subject.file_sha(inputs/'old_e2_predictions.json'))
    with pytest.raises(ValueError, match='81 predictor'): subject.aggregate(inputs, run, plan)


@pytest.mark.parametrize('mutation', ['checkpoint', 'ack', 'calibrator'])
def test_rejects_changed_fitted_artifact_or_missing_ack(campaign, mutation):
    inputs, run, plan, _ = campaign
    job = plan['jobs'][0]
    if mutation == 'checkpoint': (run/'jobs'/job['id']/'fit/selected_model.pt').write_bytes(b'mutated')
    elif mutation == 'calibrator': (run/'jobs'/job['id']/'calibrator.json').write_text('{}')
    else: (run/'remote_acks'/(job['id']+'.json')).unlink()
    with pytest.raises(ValueError): subject.aggregate(inputs, run, plan)


@pytest.mark.parametrize('mutation', ['duplicate', 'missing'])
def test_rejects_bad_new_seed_schedule(campaign, mutation):
    inputs, run, plan, _ = campaign
    if mutation == 'duplicate': plan['jobs'][1] = plan['jobs'][0]
    else: plan['jobs'].pop()
    with pytest.raises(ValueError, match='52 new coordinates'): subject.aggregate(inputs, run, plan)


def test_claim_bound_cli_writes_hashed_receipt(campaign, monkeypatch):
    inputs, run, plan, _ = campaign
    write(run/'plan.json', plan)
    manifest = subject.read_json(run/'run_manifest.json')
    reservation = {'event': 'reserve', 'sequence': 0, 'previous_sha256': '0'*64,
        'run_id': manifest['run_id'], 'plan_fingerprint': manifest['plan_fingerprint'],
        'job_id': 'aggregate', 'phase': 'aggregate', 'reservation_id': 'aggregate-reservation', 'cap_seconds': 120.0}
    reservation['sha256'] = hashlib.sha256(subject.canonical(reservation)).hexdigest()
    write(run/'ledger.jsonl', reservation)
    claim = {k: reservation[k] for k in ('run_id', 'plan_fingerprint', 'job_id', 'phase', 'reservation_id', 'cap_seconds')}
    claim.update(version='e2-seed-extension-v1', ledger_event_sha256=reservation['sha256'], ack_commit='a'*40)
    claim_path = run.parent/'claim.json'; write(claim_path, claim)
    monkeypatch.setattr(sys, 'argv', ['aggregate.py', '--inputs', str(inputs), '--run-dir', str(run),
        '--plan', str(run/'plan.json'), '--claim-path', str(claim_path), '--phase', 'aggregate', '--job-id', 'aggregate'])
    subject.main()
    receipt = subject.read_json(run/'aggregate_receipt.json')
    assert receipt['job_id'] == receipt['phase'] == 'aggregate' and receipt['status'] == 'completed'
    assert len(receipt['artifacts']) == 4
    for artifact in receipt['artifacts']:
        path = run/artifact['path']
        assert artifact['sha256'] == subject.file_sha(path) and artifact['bytes'] == path.stat().st_size
    import bootstrap
    assert bootstrap.validate_receipt(receipt, manifest, 'aggregate', 'aggregate', directory=run) == receipt
    claim['cap_seconds'] = 119; write(claim_path, claim)
    with pytest.raises(ValueError, match='claim or acknowledgement'): subject.validate_claim(run, claim_path)


def test_real_plan_source_aliases_and_coordinator_output_allowlist():
    """Metadata only: no real prediction arrays are opened or evaluated."""
    import bootstrap
    import worker
    plan = subject.read_json(Path(__file__).with_name('plan.json'))
    assert plan['version'] == worker.VERSION == 'e2-seed-extension-v1'
    assert tuple(plan['models']) == worker.NEURAL == subject.NEURAL
    assert tuple(plan['new_neural_seeds']) == worker.NEW_SEEDS == subject.NEW_SEEDS
    assert all(plan['inputs'][name]['sha256'] == expected for name, expected in subject.SOURCE_HASHES.items())
    assert all(bootstrap.approved_path(name, {'jobs': plan['jobs']}) for name in
        ('combined_predictors.json', 'per_seed_metrics.csv', 'comparison.csv', 'summary.json', 'aggregate_receipt.json'))


def test_aggregate_claim_understands_audited_reopen_and_preserves_failed_charge(tmp_path):
    """Use archived failure metadata only; append synthetic authorization/claim."""
    import bootstrap
    archived = subject.read_json(Path(__file__).with_name('failed_remote_snapshot.json'))
    manifest = json.loads(archived['run_manifest.json'])
    manifest['repair_authorization_sha256'] = 'c'*64
    run = tmp_path/'run'; run.mkdir()
    write(run/'run_manifest.json', manifest)
    events = bootstrap.events_from(archived['ledger.jsonl'].encode())
    def append(body):
        value = {**body, 'version': 'e2-extension-event-v1', 'run_id': manifest['run_id'],
                 'plan_fingerprint': manifest['plan_fingerprint'], 'sequence': len(events),
                 'previous_sha256': events[-1]['sha256']}
        value['sha256'] = bootstrap.sha(bootstrap.canonical(value))
        events.append(value)
    append({'event': 'reopen', 'job_id': bootstrap.FAILED_JOB_ID, 'phase': 'fit',
            'repair_id': bootstrap.REPAIR_ID, 'failed_head': bootstrap.FAILED_HEAD,
            'charged_seconds_preserved': bootstrap.FAILED_CHARGED_SECONDS,
            'authorization_sha256': manifest['repair_authorization_sha256']})
    append({'event': 'reserve', 'job_id': 'aggregate', 'phase': 'aggregate',
            'reservation_id': 'synthetic-aggregate-after-reopen', 'cap_seconds': 120.0})
    (run/'ledger.jsonl').write_bytes(b''.join(bootstrap.canonical(event)+b'\n' for event in events))
    reservation = events[-1]
    claim = {key: reservation[key] for key in ('run_id', 'plan_fingerprint', 'job_id', 'phase', 'reservation_id', 'cap_seconds')}
    claim.update(version='e2-seed-extension-v1', ledger_event_sha256=reservation['sha256'], ack_commit='a'*40)
    write(tmp_path/'claim.json', claim)
    assert subject.validate_claim(run, tmp_path/'claim.json') == (manifest, claim)
    ledger = bootstrap.ledger_summary(events, manifest)
    assert ledger['charged_seconds'] == bootstrap.FAILED_CHARGED_SECONDS
    assert ledger['remaining_seconds'] == 7200-bootstrap.FAILED_CHARGED_SECONDS
    state = bootstrap.state_projection(events, manifest)
    assert state['status'] == 'running' and state['prior_failure']['outcome'] == 'failed' and not state['jobs']
    changed = {**events[2], 'charged_seconds_preserved': 0.0}
    with pytest.raises(RuntimeError, match='audited metadata-only'):
        bootstrap.ledger_summary([*events[:2], changed, events[3]], manifest)
