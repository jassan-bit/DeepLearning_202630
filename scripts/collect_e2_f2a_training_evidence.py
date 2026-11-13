"""Collect original E2 fitting metadata; never load weights, market data or models.

This post hoc audit reuses saved epoch evaluation losses, calibration predictions,
and already computed final RMSEs. It does not establish absence of overfitting:
no overfitting margin was registered, and temporal windows are not paired dates.
Only Python's standard library is used, including for immutable GitHub retrieval.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'jassan-bit/DeepLearning_202630'
RESULTS_COMMIT = '50cb634e9eb163b09f9607982842144ba9f74c87'
FOLDS = ('2024-01-01', '2024-07-01', '2024-10-01')
SAE = 'LSTM-autoencoder con predictor'
VERSION = 'e2-f2a-original-training-metadata-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_bytes(raw, descriptor, label):
    if descriptor.get('bytes') is not None and len(raw) != descriptor['bytes']:
        raise ValueError(f'{label}: byte count mismatch')
    expected = descriptor.get('sha256', descriptor.get('manifest_sha256'))
    if not expected or sha(raw) != expected:
        raise ValueError(f'{label}: SHA256 mismatch')
    if descriptor.get('git_blob_sha'):
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if actual != descriptor['git_blob_sha']:
            raise ValueError(f'{label}: Git blob SHA mismatch')
    return raw


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label}: finite numeric value required')
    return float(value)


def _safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe metadata path')
    return str(path)


def extract_fit_rows(fit, request, *, stage, fold_origin=None, calibration=None, final_rmse=None):
    """Separate supervised target risk from auxiliary reconstruction histories."""
    if stage not in ('hpo', 'final') or fit.get('status') != 'completed':
        raise ValueError('Only completed original HPO/final fits may be audited')
    model, seed = fit['model_id'], fit['seed']
    if request['name'] != model or request['seed'] != seed:
        raise ValueError('Fit/request model or seed mismatch')
    if fit.get('request_sha256') and sha(canonical(request)) != fit['request_sha256']:
        raise ValueError('Fit/request identity mismatch')
    checkpoint = fit['selected_checkpoint']['sha256']
    common = {'stage': stage, 'model': model, 'seed': seed,
              'fold_origin': fold_origin, 'trial_id': request['trial']['trial_id'],
              'request_sha256': fit.get('request_sha256'),
              'selected_checkpoint_sha256': checkpoint,
              'train_signature': request.get('train_signature'),
              'validation_signature': request.get('validation_signature')}
    target, auxiliary = [], []
    phases = fit['phase_records']
    if set(phases) != ({'ae1', 'ae2', 'supervised'} if model == SAE else {'supervised'}):
        raise ValueError('Unexpected optimizer phases')
    for phase, record in phases.items():
        logs = record['logs']
        epochs = [row['epoch'] for row in logs]
        if not epochs or epochs != list(range(1, len(logs) + 1)) or record['best_epoch'] not in epochs:
            raise ValueError('Missing/duplicate epoch or absent best epoch')
        for row in logs:
            train = finite(row['train_rmse_original'], 'train RMSE')
            mse = finite(row['train_mse_scaled'], 'train scaled MSE')
            if min(train, mse) < 0:
                raise ValueError('Negative risk')
            validation = row.get('validation_score')
            if validation is not None:
                validation = finite(validation, 'validation RMSE')
                if validation < 0:
                    raise ValueError('Negative validation risk')
            if (stage == 'final' and validation is not None) or (stage == 'hpo' and validation is None):
                raise ValueError('Validation availability differs from original fitting role')
            entry = {**common, 'phase': phase, 'epoch': row['epoch'],
                     'selected_epoch': row['epoch'] == record['best_epoch'],
                     'best_epoch': record['best_epoch'],
                     'epochs_completed': len(logs), 'fixed_epochs': record.get('fixed_epochs'),
                     'selected_tensor_state_sha256': record.get('selected_state_sha256'),
                     'train_mse_scaled': mse, 'examples_seen': row.get('examples_seen'),
                     'optimizer_steps_total': row.get('optimizer_steps_total')}
            if phase == 'supervised':
                scale = fit.get('scaler', {}).get('target_scale')
                if scale is not None and not math.isclose(math.sqrt(mse)*scale, train, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError('Scaled MSE and original target RMSE disagree')
                target.append({**entry, 'metric_scope': 'supervised_target_original_units',
                    'train_rmse_original': train, 'validation_rmse_original': validation,
                    'validation_minus_train_rmse_original': None if validation is None else validation-train})
            else:
                auxiliary.append({**entry, 'metric_scope': 'auxiliary_reconstruction',
                    'train_reconstruction_rmse': train, 'validation_reconstruction_rmse': validation,
                    'metric_units': 'auxiliary_scaled_input_reconstruction_not_target_units'})
    supervised = phases['supervised']
    selected = next(row for row in target if row['selected_epoch'])
    terminal = target[-1]
    if stage == 'final' and (supervised.get('fixed_epochs') != len(target) or not terminal['selected_epoch']):
        raise ValueError('Final predictor must use terminal frozen epoch')
    summary = {**common, 'best_epoch': supervised['best_epoch'], 'epochs_completed': len(target),
        'curve_availability': 'complete_original_epoch_logs',
        'fixed_epochs': supervised.get('fixed_epochs'),
        'selected_tensor_state_sha256': supervised.get('selected_state_sha256'),
        'selected_train_rmse_original': selected['train_rmse_original'],
        'terminal_train_rmse_original': terminal['train_rmse_original'],
        'selected_validation_rmse_original': selected['validation_rmse_original'],
        'terminal_validation_rmse_original': terminal['validation_rmse_original'],
        'validation_minus_train_rmse_original': selected['validation_minus_train_rmse_original'],
        'train_count': selected['examples_seen'],
        'delta_g': None, 'absence_of_overfit_established': False,
        'inference_scope': 'posthoc_descriptive_different_temporal_windows_no_preregistered_margin',
        'train_risk_mode': 'epoch_boundary_model_eval_no_grad_full_training_data',
        'epoch_evaluation_batch_size': request.get('config', {}).get('batch_size'),
        'persisted_prediction_batch_size': 1,
        'batch_size_rounding_caveat': request.get('config', {}).get('batch_size') != 1,
        'calibration_rmse_original': None, 'final_rmse_original': final_rmse,
        'calibration_minus_train_rmse_original': None, 'final_minus_calibration_rmse_original': None,
        'calibration_to_train_rmse_ratio': None, 'final_to_calibration_rmse_ratio': None,
        'same_final_predictor_metadata_binding_verified': False}
    if calibration is not None:
        if (stage != 'final' or calibration['checkpoint_sha256'] != checkpoint
                or calibration['model'] != model or calibration['seed'] != seed
                or calibration.get('same_predictor_as_final') is not True
                or calibration.get('used_for_HPO') is not False):
            raise ValueError('Calibration predictor checkpoint/model/seed identity mismatch')
        prediction, truth, dates = (calibration[k] for k in ('prediction_original', 'target_original', 'dates'))
        if not dates or len(dates) != len(prediction) or len(dates) != len(truth) or len(set(dates)) != len(dates):
            raise ValueError('Calibration length/date mismatch')
        rmse = math.sqrt(sum((finite(p, 'calibration prediction')-finite(y, 'calibration truth'))**2
                             for p, y in zip(prediction, truth))/len(dates))
        train = selected['train_rmse_original']
        summary.update(calibration_rmse_original=rmse, calibration_count=len(dates),
            calibration_start=dates[0], calibration_end=dates[-1],
            calibration_minus_train_rmse_original=rmse-train,
            calibration_to_train_rmse_ratio=None if train == 0 else rmse/train,
            same_final_predictor_metadata_binding_verified=True)
        if final_rmse is not None:
            final_rmse = finite(final_rmse, 'original final RMSE')
            summary.update(final_rmse_original=final_rmse,
                final_minus_calibration_rmse_original=final_rmse-rmse,
                final_to_calibration_rmse_ratio=None if rmse == 0 else final_rmse/rmse)
    return target, auxiliary, summary


def expected_validation_dates(origin):
    start = date.fromisoformat(origin)-timedelta(days=90)
    return [(start+timedelta(days=i)).isoformat() for i in range(83)]


def extract_cached_hpo_summary(job, source, admission, truths_by_date):
    """Preserve missing historical curves honestly for metadata-only cached HPO."""
    if sha(canonical({k: v for k, v in source.items() if k != 'sha256'})) != source['sha256']:
        raise ValueError('Cached original record self-hash mismatch')
    outcome = source['outcome']
    if sha(canonical(outcome)) != source['outcome_sha256']:
        raise ValueError('Cached original outcome self-hash mismatch')
    if (admission['result_sha256'] != source['sha256'] or admission.get('reuse_approved') is not True
            or admission['model_id'] != job['model_id'] or admission['origin'] != job['fold_origin']
            or admission['seed'] != job['seed'] or admission['parent_trial_id'] != job['parent_trial_id']
            or admission.get('weights_reusable') is not False or outcome['status'] != 'completed'):
        raise ValueError('Cached source admission identity mismatch')
    identity = outcome['resource_use']['identity']
    if (identity['model'] != job['model_id'] or identity['origin'] != job['fold_origin']
            or identity['seed'] != job['seed'] or identity['stage'] != 'hpo'
            or identity['candidate_id'] != 'hpo:'+job['trial']['trial_id']
            or admission['job_id'] != source['job_id']):
        raise ValueError('Original cached source fit identity mismatch')
    dates = [value[:10] for value in outcome['prediction_dates']]
    values = outcome['predictions_original']
    for key, value in (('validation_prediction_sha256', values), ('validation_dates_sha256', outcome['prediction_dates'])):
        if key in admission and admission[key] != sha(canonical(value)):
            raise ValueError('Cached admission prediction/date digest mismatch')
    if ('candidate_payload_sha256' in admission and admission['candidate_payload_sha256']
            != outcome['resource_use']['identity']['payload_sha256']):
        raise ValueError('Cached admission candidate payload identity mismatch')
    if dates != expected_validation_dates(job['fold_origin']) or len(values) != len(dates):
        raise ValueError('Cached original validation calendar mismatch')
    phase = outcome['resource_use']['phase_records']['supervised']
    if admission['best_epoch'] != phase['best_epoch']:
        raise ValueError('Original cached selected epoch differs from admission')
    if 'logs' in phase:
        raise ValueError('Cached record unexpectedly contains curves; audit explicitly')
    rmse = math.sqrt(sum((finite(p, 'cached original prediction')-truths_by_date[d])**2
                         for d, p in zip(dates, values))/len(dates))
    return {'stage': 'hpo', 'model': job['model_id'], 'seed': job['seed'],
        'fold_origin': job['fold_origin'], 'trial_id': job['trial']['trial_id'],
        'configuration_id': job['configuration_id'],
        'selected_checkpoint_sha256': None, 'selected_tensor_state_sha256': phase['selected_state_sha256'],
        'original_cached_record_sha256': source['sha256'], 'best_epoch': phase['best_epoch'],
        'epochs_completed': phase['epochs_completed'], 'fixed_epochs': None,
        'curve_availability': 'unavailable_in_original_cache_metadata',
        'selected_train_rmse_original': None, 'terminal_train_rmse_original': None,
        'selected_validation_rmse_original': None, 'terminal_validation_rmse_original': None,
        'validation_minus_train_rmse_original': None, 'train_count': None,
        'validation_count': len(dates), 'validation_start': dates[0], 'validation_end': dates[-1],
        'saved_validation_prediction_rmse_original': rmse,
        'delta_g': None, 'absence_of_overfit_established': False,
        'inference_scope': 'posthoc_cached_original_validation_metadata_only_training_risk_unavailable',
        'train_risk_mode': 'unavailable_no_original_epoch_losses',
        'persisted_prediction_batch_size': 1, 'weights_reusable': False}


class MetadataCollector:
    """A bounded, cached HTTP reader that rejects weight/raw-data artifact requests."""
    def __init__(self, output, head, commit=RESULTS_COMMIT, repository=REPOSITORY):
        if not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError('An immutable 40-character results commit is required')
        self.output, self.head, self.commit, self.repository = Path(output), head, commit, repository
        self.records = []
        self.lock = threading.Lock()

    def _read(self, url, path, descriptor, label):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return verify_bytes(path.read_bytes(), descriptor, label)
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': VERSION})
                with urllib.request.urlopen(req, timeout=60) as response:
                    raw = response.read(32*1024*1024+1)
                if len(raw) > 32*1024*1024:
                    raise ValueError('Metadata response exceeds 32 MiB')
                verify_bytes(raw, descriptor, label)
                path.write_bytes(raw)
                return raw
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(min(10, float(error.headers.get('Retry-After', 2**attempt))))
            except (TimeoutError, urllib.error.URLError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError('Unreachable retrieval state')

    def raw_url(self, path):
        return f'https://raw.githubusercontent.com/{self.repository}/{self.commit}/{urllib.parse.quote(_safe_relative(path))}'

    def collect(self, directory, paths, *, allow_cached_hpo=False):
        backup = self.head['result_backups'][directory]
        manifest_path = backup['manifest_path']
        local = self.output/'sources'/directory
        raw = self._read(self.raw_url(manifest_path), local/'manifest.json',
                         {'sha256': backup['manifest_sha256']}, manifest_path)
        manifest = json.loads(raw)
        if manifest.get('raw_market_data_included') is not False or manifest.get('fit_status') != 'completed':
            raise ValueError('Only completed metadata-only original manifests are admitted')
        inventory = {item['path']: item for item in manifest['artifacts']}
        if allow_cached_hpo and 'job_result.json' not in inventory:
            if not directory.startswith('jobs/') or not {'cached_source_record.json', 'cache_admission.json'} <= inventory.keys():
                raise ValueError('Original HPO metadata missing without a cache provenance record')
            paths = ('cached_source_record.json', 'cache_admission.json')
        data, evidence = {}, []
        archive_raw = None
        for relative in paths:
            _safe_relative(relative)
            if not relative.endswith('.json') or any(x in relative.lower() for x in ('selected_model', '.pt', '.pkl', 'market')):
                raise ValueError('Only allowlisted fitting JSON metadata may be requested')
            descriptor = inventory[relative]
            expected_head = self.head['files'].get(directory+'/'+relative)
            if expected_head and (expected_head['sha256'] != descriptor['sha256'] or expected_head['bytes'] != descriptor['bytes']):
                raise ValueError('Control-head artifact binding differs from manifest')
            destination = local/relative
            storage = descriptor['storage']
            if storage == 'git_blob':
                artifact_raw = self._read(self.raw_url(descriptor['remote_path']), destination, descriptor, relative)
                transport = self.raw_url(descriptor['remote_path'])
            elif storage == 'release_archive_member':
                archive = manifest['archive_transport']
                if archive.get('kind') != 'deterministic_zip' or archive['sha256'] != descriptor['archive_sha256']:
                    raise ValueError('Unbound metadata ZIP member')
                if archive_raw is None:
                    archive_raw = self._read(archive['url'], self.output/'sources'/'archives'/(archive['sha256']+'.zip'), archive, 'metadata archive')
                with zipfile.ZipFile(io.BytesIO(archive_raw)) as zipped:
                    if len(zipped.infolist()) != archive['member_count'] or len(set(zipped.namelist())) != len(zipped.namelist()):
                        raise ValueError('Metadata archive inventory mismatch')
                    if any(not name.endswith(('.json', '.jsonl', '.csv', '.txt')) for name in zipped.namelist()):
                        raise ValueError('Metadata ZIP contains a non-metadata member')
                    artifact_raw = verify_bytes(zipped.read(_safe_relative(descriptor['archive_member'])), descriptor, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(artifact_raw)
                transport = archive['url']+'#'+descriptor['archive_member']
            else:
                raise ValueError('Weight/release asset retrieval is prohibited')
            data[relative] = json.loads(artifact_raw)
            evidence.append({'path': relative, 'bytes': len(artifact_raw), 'sha256': sha(artifact_raw),
                             'storage': storage, 'transport': transport})
        with self.lock:
            self.records.append({'directory': directory, 'attempt_id': manifest['attempt_id'],
                'manifest_path': manifest_path, 'manifest_sha256': sha(raw), 'artifacts': evidence})
        return data


def hpo_jobs(root):
    """Reconstruct registered logical IDs using JSON only; import no fitting code."""
    domains = json.loads((root/'experiments/e2_reduced_protocol_20261112/configurations.json').read_text())
    extension = root/'experiments/e2_tide_published_r0_20261112/configurations.json'
    tide = json.loads(extension.read_text())
    domains['TiDE'] = tide.get('TiDE', tide.get('models', {}).get('TiDE', tide)) if isinstance(tide, dict) else tide
    jobs = []
    for model, slots in domains.items():
        for fold in FOLDS:
            for slot in slots:
                if not slot.get('execution_allowed'):
                    continue
                body = {'version': 'e2-reduced-selection-executor-v1', 'stage': 'hpo', 'model_id': model,
                    'fold_origin': fold, 'configuration_id': slot['configuration_id'],
                    'configuration_sha256': slot['configuration_sha256'], 'parent_trial_id': slot['parent_trial_id'],
                    'trial': slot['trial'], 'seed': 271201}
                jobs.append({**body, 'logical_job_id': sha(canonical(body))})
    return jobs


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_evidence(head_path, metadata_dir, output, *, workers=8, results_commit=RESULTS_COMMIT, root=ROOT):
    output, metadata_dir = Path(output), Path(metadata_dir)
    output.mkdir(parents=True, exist_ok=True)
    head_raw = Path(head_path).read_bytes()
    head = json.loads(head_raw)
    collector = MetadataCollector(output, head, results_commit)
    freeze_raw = verify_bytes((metadata_dir/'final_freeze.json').read_bytes(), head['files']['final/final_freeze.json'], 'original freeze')
    freeze = json.loads(freeze_raw)
    analysis_key = next(k for k in head['files'] if k.endswith('/scientific_outputs/benchmark_analysis.json'))
    analysis_raw = verify_bytes((metadata_dir/'benchmark_analysis.json').read_bytes(), head['files'][analysis_key], 'original final RMSE summary')
    analysis = json.loads(analysis_raw)
    score_directory = analysis_key.rsplit('/scientific_outputs/', 1)[0]
    score_backup = head['result_backups'][score_directory]
    score_manifest_raw = verify_bytes((metadata_dir/'score_manifest.json').read_bytes(),
        {'sha256': score_backup['manifest_sha256']}, 'original score manifest')
    score_manifest = json.loads(score_manifest_raw)
    score_archive_raw = verify_bytes((metadata_dir/'score_metadata.zip').read_bytes(), score_manifest['archive_transport'], 'original score metadata ZIP')
    score_inventory = {r['path']: r for r in score_manifest['artifacts']}
    score_predictions = {}
    with zipfile.ZipFile(io.BytesIO(score_archive_raw)) as zipped:
        for name, desc in score_inventory.items():
            if name.startswith('scientific_outputs/predictions_'):
                prediction = json.loads(verify_bytes(zipped.read(name), desc, name))
                score_predictions[prediction['model'], prediction['seed']] = prediction
    local_inputs = output/'sources'/'control'
    local_inputs.mkdir(parents=True, exist_ok=True)
    for name, raw in [('completed_benchmark_head.json', head_raw), ('final_freeze.json', freeze_raw),
                      ('benchmark_analysis.json', analysis_raw), ('score_manifest.json', score_manifest_raw)]:
        (local_inputs/name).write_bytes(raw)
    seeds = freeze['final_seed_ids']
    rmses = {(row['model'], seed): rmse for row in analysis['table_A2'] if row['R'] == len(seeds)
             for seed, rmse in zip(seeds, row['RMSE_per_seed'])}
    final_models = [row for row in freeze['models'] if isinstance(row['seed'], int)]
    if len(final_models) != 78:
        raise ValueError('Expected exactly 78 frozen neural fits')
    print('Retrieving 78 final-fit metadata records and calibration predictions', flush=True)
    def final_collect(row):
        directory = 'final/'+str(PurePosixPath(row['checkpoint']).parent.parent)
        records = collector.collect(directory, ('fit/job_result.json', 'fit/request.json', 'calibration_predictions.json'))
        fit, request, calibration = (records[p] for p in ('fit/job_result.json', 'fit/request.json', 'calibration_predictions.json'))
        score = score_predictions[row['model'], row['seed']]
        if ((fit['model_id'], fit['seed']) != (row['model'], row['seed'])
                or fit['selected_checkpoint']['sha256'] != row['checkpoint_sha256']
                or score['checkpoint_sha256'] != row['checkpoint_sha256']):
            raise ValueError('Final fit/freeze/score checkpoint binding mismatch')
        calibration_start = date(2024, 11, 8)
        if calibration['dates'] != [(calibration_start+timedelta(days=i)).isoformat() for i in range(53)]:
            raise ValueError('Final calibration predictions lack registered 53-date support')
        return fit, request, calibration
    with ThreadPoolExecutor(max_workers=workers) as executor:
        finals = list(executor.map(final_collect, final_models))
    winners = {}
    for fit, request, _ in finals:
        model, trial = fit['model_id'], request['trial']
        if model in winners and winners[model] != trial:
            raise ValueError('Final seeds do not share one selected recipe')
        winners[model] = trial
    jobs = [j for j in hpo_jobs(root) if j['model_id'] in winners and j['trial'] == winners[j['model_id']]]
    if len(winners) != 13 or len(jobs) != 39:
        raise ValueError('Expected 13 recipes and 39 selected three-origin HPO fits')
    print('Retrieving 39 selected HPO metadata records and original validation predictions', flush=True)
    def hpo_collect(job):
        records = collector.collect('jobs/'+job['logical_job_id'], ('job_result.json', 'request.json', 'predictions.json'), allow_cached_hpo=True)
        return job, records
    with ThreadPoolExecutor(max_workers=workers) as executor:
        hpos = list(executor.map(hpo_collect, jobs))
    target, auxiliary, summaries = [], [], []
    truths_by_date = {}
    for _, records in hpos:
        if 'predictions.json' not in records:
            continue
        predictions = records['predictions.json']
        for d, y in zip(predictions['dates'], predictions['truths_original']):
            if d in truths_by_date and truths_by_date[d] != y:
                raise ValueError('Original saved validation targets disagree across selected fits')
            truths_by_date[d] = y
    for fit, request, calibration in finals:
        rows, aux, summary = extract_fit_rows(fit, request, stage='final', calibration=calibration,
            final_rmse=rmses[fit['model_id'], fit['seed']])
        target.extend(rows); auxiliary.extend(aux); summaries.append(summary)
    for job, records in hpos:
        if 'cached_source_record.json' in records:
            summaries.append(extract_cached_hpo_summary(job, records['cached_source_record.json'],
                records['cache_admission.json'], truths_by_date))
            continue
        fit, request, predictions = (records[k] for k in ('job_result.json', 'request.json', 'predictions.json'))
        if (request['trial'] != job['trial'] or request['seed'] != job['seed']
                or fit['model_id'] != job['model_id'] or predictions['scope'] != 'inner_validation_not_final'):
            raise ValueError('Selected HPO identity/prediction role mismatch')
        count = len(predictions['dates'])
        if (predictions['dates'] != expected_validation_dates(job['fold_origin'])
                or len(predictions['predictions_original']) != count or len(predictions['truths_original']) != count):
            raise ValueError('HPO predictions lack registered 83-date support')
        if fit['prediction_dates'] != predictions['dates'] or fit['predictions_original'] != predictions['predictions_original']:
            raise ValueError('HPO persisted predictions differ from original fit result')
        rmse = math.sqrt(sum((p-y)**2 for p, y in zip(predictions['predictions_original'], predictions['truths_original']))/count)
        if not math.isclose(rmse, fit['rmse_original'], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError('Saved HPO RMSE differs from saved validation predictions')
        rows, aux, summary = extract_fit_rows(fit, request, stage='hpo', fold_origin=job['fold_origin'])
        summary.update(validation_count=count, saved_validation_prediction_rmse_original=rmse,
            configuration_id=job['configuration_id'],
            epoch_vs_batch1_validation_rmse_difference=summary['selected_validation_rmse_original']-rmse,
            validation_start=predictions['dates'][0], validation_end=predictions['dates'][-1])
        target.extend(rows); auxiliary.extend(aux); summaries.append(summary)
    sort = lambda row: (row['model'], row['stage'], row.get('fold_origin') or '', row['seed'], row.get('phase') or '', row.get('epoch', 0))
    target.sort(key=sort); auxiliary.sort(key=sort); summaries.sort(key=sort)
    write_csv(output/'training_curves.csv', target)
    write_csv(output/'auxiliary_curves.csv', auxiliary)
    write_csv(output/'fit_overfit_summary.csv', summaries)
    outputs = {name: {'bytes': (output/name).stat().st_size, 'sha256': sha((output/name).read_bytes())}
               for name in ('training_curves.csv', 'auxiliary_curves.csv', 'fit_overfit_summary.csv')}
    report = {'version': VERSION, 'status': 'completed', 'posthoc': True,
        'repository': collector.repository, 'immutable_results_commit': results_commit,
        'scientific_training_commit': head['scientific_commit'], 'control_state_id': head['state_id'],
        'control_head_sha256': sha(head_raw), 'freeze_sha256': sha(freeze_raw),
        'original_benchmark_analysis_sha256': sha(analysis_raw), 'original_score_manifest_sha256': sha(score_manifest_raw),
        'original_score_archive_sha256': sha(score_archive_raw),
        'final_fit_count': len(finals), 'selected_hpo_fit_count': len(hpos), 'model_count': len(winners),
        'selected_hpo_full_curve_fit_count': sum('job_result.json' in records for _, records in hpos),
        'selected_hpo_cached_fit_without_curves_count': sum('cached_source_record.json' in records for _, records in hpos),
        'winner_recipe_source': 'Frozen final fit requests; exact recipe equality checked across all six final seeds',
        'winner_recipes': winners,
        'selected_hpo_curve_availability': [{key: row[key] for key in
            ('model', 'fold_origin', 'configuration_id', 'trial_id', 'curve_availability')}
            for row in summaries if row['stage'] == 'hpo'],
        'supervised_epoch_count': len(target), 'auxiliary_epoch_count': len(auxiliary),
        'delta_g': None, 'absence_of_overfit_established': False,
        'train_risk_calculation': 'Saved epoch-boundary model.eval/no_grad full training evaluation; not minibatch optimization loss',
        'target_metric': 'original-unit RMSE; gap positive when later-window error exceeds training error',
        'auxiliary_metric': 'SAE ae1/ae2 reconstruction errors in auxiliary scaled-input units; excluded from target gaps',
        'HPO_scope': 'three overlapping-training internal origins, optimization seed 271201, selected recipe; postselection descriptive curves',
        'final_scope': 'six optimization seeds conditional on one observed chronology; final fit validation intentionally absent',
        'calibration_final_comparison': 'same checkpoint file identity, different temporal windows; descriptive and not a paired-date test',
        'batch_rounding': 'epoch risks use configured training batch size; saved validation/calibration/final predictions use batch size 1',
        'hash_domains': {'selected_checkpoint_sha256': 'serialized selected_model.pt container with model/scaler/architecture/request',
            'selected_tensor_state_sha256': 'tensor-only model-state digest; distinct from checkpoint-container hashes',
            'phase_checkpoint_sha256': 'optimizer/RNG/model snapshot container; distinct from selected model container'},
        'weight_bytes_downloaded': False, 'weight_bytes_rehashed': False, 'market_payload_loaded': False,
        'model_inference_performed': False, 'training_performed': False, 'final_metrics_recomputed': False,
        'source_records': sorted(collector.records, key=lambda row: row['directory']), 'outputs': outputs}
    (output/'source_evidence.json').write_bytes(canonical(report)+b'\n')
    print(json.dumps({k: report[k] for k in ('status', 'final_fit_count', 'selected_hpo_fit_count', 'supervised_epoch_count', 'auxiliary_epoch_count')}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-head', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True, help='Verified original freeze/score metadata export directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results-commit', default=RESULTS_COMMIT)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    args = parser.parse_args(argv)
    collect_evidence(args.control_head, args.metadata_dir, args.output,
                     workers=args.workers, results_commit=args.results_commit)


if __name__ == '__main__':
    main()
