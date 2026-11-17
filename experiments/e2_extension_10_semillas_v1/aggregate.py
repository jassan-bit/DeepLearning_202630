"""Describe a complete posterior E2 seed extension; no fitting or inference."""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

NEURAL = ('FCN-1D', 'ResNet-1D', 'Inception-1D', 'LSTM-autoencoder con predictor',
          'TimesNet', 'TSMixer', 'TiDE', 'TimeMixer', 'PatchTST', 'iTransformer',
          'MLP16', 'LSTM32', 'GRU32')
CLASSICAL = ('Persistence', 'SeasonalNaive', 'ARIMA')
OLD_SEEDS, NEW_SEEDS = tuple(range(271801, 271807)), tuple(range(271807, 271811))
ALL_SEEDS = OLD_SEEDS + NEW_SEEDS
DATES = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608)]
SOURCE_HASHES = {
    'old_e2_predictions.json': '65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187',
    'e3_summary.json': '7c90c800adcca2218ea94553081cbcf744f3b9d9ff657ece17402df40be8f176',
    'e3_per_seed_metrics.csv': '2fd1f3c6ec918d0c4371a3e3cf25864c271fd5250d459c3ef99d96ab7d31b2ff',
    'e3_evaluation_predictions.csv': 'b82ba0a8a0371ea5643d9ff666a3ac5a1a4e91018b95a640bd88b1be50bb5133',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def numeric(values, n=608):
    require(isinstance(values, list) and len(values) == n, 'Incorrect numeric support')
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values),
            'Finite nonnegative numeric values required')
    return values


def metrics(y, prediction):
    errors = [p-t for p, t in zip(numeric(prediction), y, strict=True)]
    return {'RMSE': math.sqrt(statistics.fmean(e*e for e in errors)),
            'MAE': statistics.fmean(abs(e) for e in errors), 'bias': statistics.fmean(errors)}


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)


def e3_metrics(inputs, y):
    rows = list(csv.DictReader((inputs / 'e3_evaluation_predictions.csv').open()))
    groups = {seed: [] for seed in ALL_SEEDS}
    for row in rows:
        seed = int(row['seed'])
        require(seed in groups, 'Unexpected E3 seed')
        groups[seed].append(row)
    result = []
    for seed, group in groups.items():
        require([r['date'] for r in group] == DATES, 'E3 dates differ or duplicate')
        require([float(r['target_sigma7']) for r in group] == y, 'E3 targets differ')
        result.append({'model': 'CNN-LSTM-atencion-aditiva E3', 'seed': seed,
                       **metrics(y, [float(r['prediction_sigma7']) for r in group])})
    declared = list(csv.DictReader((inputs / 'e3_per_seed_metrics.csv').open()))
    require(len(declared) == 10 and {int(r['seed']) for r in declared} == set(ALL_SEEDS),
            'Incomplete or duplicate E3 metrics')
    by_seed = {int(r['seed']): r for r in declared}
    for row in result:
        require(int(by_seed[row['seed']]['evaluation_n']) == 608 and
                all(close(row[k], by_seed[row['seed']][k]) for k in ('RMSE', 'MAE', 'bias')),
                'E3 source metrics disagree with predictions')
    summary = read_json(inputs / 'e3_summary.json')
    values = [r['RMSE'] for r in result]
    require(summary['complete_training_seeds'] == 10 and summary['comparison_complete'] is True and
            close(summary['RMSE_mean'], statistics.fmean(values)) and
            close(summary['RMSE_sample_SD'], statistics.stdev(values)), 'E3 summary is incomplete or changed')
    return result


def write_once(path, raw):
    if path.exists():
        require(not path.is_symlink() and path.read_bytes() == raw, 'Existing aggregate output differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(raw)


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def validate_new(run_dir, job, y, manifest, gate_sha, calibration_targets):
    directory = run_dir / 'jobs' / job['id']
    receipt_path = directory / 'fit_receipt.json'
    receipt, evaluation = read_json(receipt_path), read_json(directory / 'evaluation.json')
    for value in (receipt, evaluation):
        require(value['status'] == 'completed' and value['job_id'] == job['id'] and
                (value['model'], value['seed']) == (job['model'], job['seed']) and
                all(value[k] == manifest[k] for k in ('run_id', 'plan_fingerprint')),
                'Predictor coordinate or run identity differs')
    require(receipt['training_examples'] == 655 and receipt['calibration_examples'] == 53 and
            receipt['trial'] == job['trial'] and receipt['fixed_phase_epochs'] == job['fixed_epochs'],
            'Fit recipe or cohort differs')
    require(evaluation['fit_receipt_sha256'] == file_sha(receipt_path) and
            evaluation['all_fits_remote_ack_sha256'] == gate_sha and
            all(evaluation[k+'_sha256'] == receipt[k+'_sha256'] for k in ('checkpoint', 'scaler', 'calibrator')),
            'Evaluation is not bound to the acknowledged fit')
    require(evaluation['dates'] == DATES and evaluation['target_original'] == y and
            evaluation['n_dates'] == 608 and evaluation['batch_size'] == 1 and
            evaluation['inverse_scaling_count'] == 1, 'Evaluation support or scaling differs')
    numeric(evaluation['prediction_original'])
    calibration = read_json(directory / receipt['calibration_predictions'])
    calibrator = read_json(directory / receipt['calibrator'])
    calibration_dates = [(date(2024, 11, 8)+timedelta(days=i)).isoformat() for i in range(53)]
    require(calibration['dates'] == calibration_dates and calibration['target_original'] == calibration_targets and
            calibration['checkpoint_sha256'] == receipt['checkpoint_sha256'] and
            calibrator['predictor_sha256'] == receipt['checkpoint_sha256'], 'Calibration predictor or support differs')
    prediction = numeric(calibration['prediction_original'], 53)
    q = sorted(abs(t-p) for t, p in zip(calibration_targets, prediction, strict=True))[48]
    require(close(calibrator['q'], q), 'Calibrator residual quantile differs')
    row = {**evaluation, 'calibration_dates': calibration_dates, 'calibration_predictions': prediction,
           'calibration_targets': calibration_targets, 'calibrator_q': calibrator['q']}
    return row, {'evaluation_sha256': file_sha(directory / 'evaluation.json'),
                 'fit_receipt_sha256': file_sha(receipt_path)}


def aggregate(inputs, run_dir, plan):
    inputs, run_dir = Path(inputs), Path(run_dir)
    plan = read_json(plan) if isinstance(plan, (str, Path)) else plan
    require(plan.get('version') == 'e2-seed-extension-v1', 'Unsupported extension plan')
    for name, expected in SOURCE_HASHES.items():
        path = inputs / name
        require(path.is_file() and not path.is_symlink() and file_sha(path) == expected,
                'Pinned source SHA256 differs: ' + name)
        if 'inputs' in plan:
            require(plan['inputs'][name]['sha256'] == expected, 'Plan source binding differs: ' + name)
    old = read_json(inputs / 'old_e2_predictions.json')
    require(old['dates'] == DATES and old['mask'] == [True]*608, 'Original dates or mask differ')
    y = numeric(old['y_true'])
    expected_old = {(m, s) for m in NEURAL for s in OLD_SEEDS} | {(m, 'deterministic') for m in CLASSICAL}
    old_coordinates = [(r['model'], r['seed']) for r in old['predictors']]
    require(len(old_coordinates) == 81 and set(old_coordinates) == expected_old,
            'Original 81 predictor coordinates differ or duplicate')
    for row in old['predictors']:
        numeric(row['prediction_original'])
    jobs = plan['jobs']
    coordinates = [(j['model'], j['seed']) for j in jobs]
    require(len(jobs) == 52 and set(coordinates) == {(m, s) for m in NEURAL for s in NEW_SEEDS},
            'Plan must contain exactly 52 new coordinates')
    require(len({j['id'] for j in jobs}) == 52 and all(str(j['id']).replace('-', '').replace('_', '').isalnum() for j in jobs),
            'Duplicate or unsafe job IDs')
    expected_files = {j['id'] for j in jobs}
    require({p.parent.name for p in (run_dir / 'jobs').glob('*/evaluation.json')} == expected_files,
            'Missing or unexpected evaluation file')
    manifest = read_json(run_dir / 'run_manifest.json')
    require(manifest['jobs'] == jobs and manifest['plan_fingerprint'] == hashlib.sha256(canonical(plan)).hexdigest(),
            'Run manifest does not bind the supplied plan')
    from worker import freeze_gate
    freeze_gate(run_dir, manifest)  # Pure file verification; no Torch or numerical data loader.
    gate_sha = file_sha(run_dir / 'all_fits_remote_ack.json')
    new, bindings = [], {}
    for job in jobs:
        row, binding = validate_new(run_dir, job, y, manifest, gate_sha, old['predictors'][0]['calibration_targets'])
        new.append(row); bindings[job['id']] = binding
    e3 = e3_metrics(inputs, y)
    all_predictors = old['predictors'] + new
    per_seed = [dict(model=r['model'], seed=r['seed'], **metrics(y, r['prediction_original'])) for r in all_predictors] + e3
    comparison = []
    for model in (*NEURAL, *CLASSICAL, e3[0]['model']):
        rows = [r for r in per_seed if r['model'] == model]
        values = [r['RMSE'] for r in rows]
        comparison.append({'model': model, 'R': len(rows), 'RMSE_mean': statistics.fmean(values),
            'RMSE_sample_SD': statistics.stdev(values) if len(values) > 1 else None,
            'train_n': 532 if model == e3[0]['model'] else 655,
            'validation_n': 116 if model == e3[0]['model'] else None,
            'calibration_n': 0 if model == e3[0]['model'] else 53, 'evaluation_n': 608,
            'posterior': True, 'equal_training_budget_claimed': False})
    provenance = {'source_sha256': dict(SOURCE_HASHES), 'old_e2_commit': '2121b048d3b379c2c28dfe366cba35a1c08a3617',
        'e3_commit': '510c3bddfdc9926322e3210ac61ee1f5eeffad2e', 'plan_sha256': hashlib.sha256(canonical(plan)).hexdigest(),
        'new_evaluation_bindings': bindings}
    combined = {**old, 'predictors': all_predictors, 'extension_provenance': provenance}
    summary = {'status': 'completed', 'complete': True, 'new_fits': 52, 'E2_predictors': 133,
        'E2_neural_seeds_per_model': {m: 10 for m in NEURAL}, 'E2_deterministic_R': 1,
        'E3_seeds': 10, 'old_predictions_preserved': combined['predictors'][:81] == old['predictors'],
        'old_source_sha256': SOURCE_HASHES['old_e2_predictions.json'], 'E3_source_sha256': SOURCE_HASHES['e3_summary.json'],
        'old_predictor_sha256': [hashlib.sha256(canonical(r)).hexdigest() for r in old['predictors']],
        'posterior': True, 'equal_training_budget_claimed': False, 'statistical_tests_performed': False,
        'historical_E3_closed_charged_seconds': 948.4396250420002,
        'new_separate_E2_extension_budget_seconds': 7200, 'old_counters_reset': False,
        'R1_R3_verified': False, 'winner_declared': False, 'primary': 'arithmetic_mean_of_individual_seed_RMSE',
        'scope': 'Known historical calendar; E2 train655 versus E3 train532; formal statistical stage remains separate',
        'source_provenance': provenance}
    outputs = {'combined_predictors.json': canonical(combined)+b'\n', 'per_seed_metrics.csv': csv_bytes(per_seed),
               'comparison.csv': csv_bytes(comparison), 'summary.json': canonical(summary)+b'\n'}
    # Validate every output before writing any, keeping reruns immutable.
    for name, raw in outputs.items():
        path = run_dir / name
        require(not path.exists() or (not path.is_symlink() and path.read_bytes() == raw), 'Aggregate output conflict: ' + name)
    for name, raw in outputs.items():
        write_once(run_dir / name, raw)
    return summary


def validate_claim(run_dir, claim_path):
    from bootstrap import events_from, ledger_summary
    manifest, claim = read_json(run_dir/'run_manifest.json'), read_json(claim_path)
    reservation = ledger_summary(events_from((run_dir/'ledger.jsonl').read_bytes()), manifest)['active_reservation']
    require(reservation is not None and claim.get('version') == 'e2-seed-extension-v1' and
            claim.get('job_id') == 'aggregate' and claim.get('phase') == 'aggregate', 'Active aggregate claim required')
    require(all(claim.get(k) == reservation.get(k) for k in
            ('run_id', 'plan_fingerprint', 'job_id', 'phase', 'reservation_id', 'cap_seconds')) and
            claim.get('ledger_event_sha256') == reservation['sha256'] and
            bool(re.fullmatch(r'[0-9a-f]{40}', claim.get('ack_commit', ''))), 'Aggregate claim or acknowledgement differs')
    return manifest, claim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'run-dir', 'plan'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--phase', choices=('aggregate',), default='aggregate')
    parser.add_argument('--job-id', choices=('aggregate',), default='aggregate')
    parser.add_argument('--claim-path', required=True, type=Path)
    args = parser.parse_args()
    if args.workspace:
        args.inputs = args.inputs or args.workspace/'inputs'
        args.run_dir = args.run_dir or args.workspace/'run'
        args.plan = args.plan or args.run_dir/'plan.json'
    require(all((args.inputs, args.run_dir, args.plan)), 'Supply --inputs --run-dir --plan or --workspace')
    manifest, claim = validate_claim(args.run_dir, args.claim_path)
    summary = aggregate(args.inputs, args.run_dir, args.plan)
    artifacts = [{'path': name, 'sha256': file_sha(args.run_dir/name), 'bytes': (args.run_dir/name).stat().st_size}
        for name in ('combined_predictors.json', 'per_seed_metrics.csv', 'comparison.csv', 'summary.json')]
    receipt = {'version': 'e2-seed-extension-v1', 'status': 'completed', 'phase': 'aggregate', 'job_id': 'aggregate',
        'run_id': manifest['run_id'], 'plan_fingerprint': manifest['plan_fingerprint'],
        'reservation_id': claim['reservation_id'], 'claim_sha256': hashlib.sha256(canonical(claim)).hexdigest(),
        'artifacts': artifacts, 'old_predictions_preserved': summary['old_predictions_preserved'],
        'new_fits': 52, 'E2_predictors': 133, 'statistical_tests_performed': False}
    write_once(args.run_dir/'aggregate_receipt.json', canonical(receipt)+b'\n')
    print(json.dumps({'status': 'completed', 'phase': 'aggregate', 'E2_predictors': 133, 'new_fits': 52}))


if __name__ == '__main__':
    main()
