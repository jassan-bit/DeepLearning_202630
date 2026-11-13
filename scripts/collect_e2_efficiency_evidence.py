#!/usr/bin/env python3
"""Audit original E2 resource metadata without loading market data or weights.

Phase wall time includes training risk evaluation and durable checkpoints. It is
neither kernel-only training time nor inference latency. Selected HPO recipes are
only a subset of the search; their cost is not the cost of HPO selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS_COMMIT = '50cb634e9eb163b09f9607982842144ba9f74c87'
SCIENTIFIC_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
SEEDS = set(range(271801, 271807))
VERSION = 'e2-original-efficiency-metadata-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify(raw, descriptor, label):
    if len(raw) != descriptor['bytes'] or sha(raw) != descriptor['sha256']:
        raise ValueError(f'{label}: bytes/SHA256 mismatch')
    return raw


def safe_path(path):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe metadata path')
    return str(p)


def number(value, name, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name}: nonnegative finite number required')
    if integer and (int(value) != value):
        raise ValueError(f'{name}: integer required')
    return int(value) if integer else float(value)


def optional_number(value, name, *, integer=False):
    return None if value is None else number(value, name, integer=integer)


def strict_sum(values):
    """Never turn a partly unobserved resource into an apparent measured total."""
    values = list(values)
    return sum(values) if values and all(v is not None for v in values) else None


def archive_member(meta, name, path):
    manifest = json.loads((meta / f'{name}_manifest.json').read_bytes())
    raw = (meta / f'{name}_metadata.zip').read_bytes()
    verify(raw, manifest['archive_transport'], f'{name} archive')
    descriptors = {a['path']: a for a in manifest['artifacts']}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        return verify(z.read(path), descriptors[path], f'{name}/{path}')


def extract_fit(fit, request, stage, source_id, directory):
    if stage not in ('final', 'selected_hpo') or fit.get('status') != 'completed':
        raise ValueError('Only original completed fits are admissible')
    if fit.get('synthetic') or fit.get('test_only_cpu_execution') or not fit.get('training_performed'):
        raise ValueError('Synthetic or unexecuted fit')
    if request['name'] != fit['model_id'] or request['seed'] != fit['seed']:
        raise ValueError('Fit/request identity mismatch')
    if sha(canonical(request)) != fit['request_sha256']:
        raise ValueError('Request identity hash mismatch')
    if stage == 'final' and request.get('sources', {}).get('source_commit') != SCIENTIFIC_COMMIT:
        raise ValueError('Scientific source commit mismatch')
    phases = fit['phase_records']
    if set(phases) != ({'ae1', 'ae2', 'supervised'} if fit['model_id'] == 'LSTM-autoencoder con predictor' else {'supervised'}):
        raise ValueError('Unexpected phase set')
    resources = request['resources']
    common = {'stage': stage, 'model': fit['model_id'], 'seed': fit['seed'], 'source_attempt_id': source_id,
              'source_directory': directory, 'trial_id': request['trial']['trial_id'],
              'request_sha256': fit['request_sha256'], 'checkpoint_sha256': fit['selected_checkpoint']['sha256'],
              'actual_parameter_count': number(fit['architecture']['actual_parameter_count'], 'parameters', integer=True),
              'device_name': resources.get('device_name'), 'cpu_threads': resources.get('cpu_threads'),
              'torch': resources.get('torch'), 'precision': resources.get('dtype'),
              'amp': resources.get('amp'), 'tf32': resources.get('tf32'),
              'deterministic': resources.get('deterministic'), 'metadata_kind': 'original_durable_fit',
              'fold_origin': request.get('origin'), 'inference_seconds': None, 'FLOPs': None}
    rows = []
    for phase, record in phases.items():
        elapsed = number(record['elapsed_seconds'], 'elapsed seconds')
        known = number(record['known_cost_seconds'], 'known cost')
        if not math.isclose(elapsed, known, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('Elapsed and known cost disagree')
        if stage == 'final' and record.get('fixed_epochs') != record['epochs_completed']:
            raise ValueError('Final fitting must use frozen epoch counts')
        rows.append({**common, 'phase': phase, 'elapsed_seconds': elapsed,
            'optimizer_steps': number(record['optimizer_steps'], 'optimizer steps', integer=True),
            'epochs_completed': number(record['epochs_completed'], 'epochs', integer=True),
            'cuda_synchronized_timing': record.get('cuda_synchronized_timing'),
            'max_cuda_memory_allocated_bytes': optional_number(record.get('max_cuda_memory_allocated_bytes'), 'allocated bytes', integer=True),
            'max_cuda_memory_reserved_bytes': optional_number(record.get('max_cuda_memory_reserved_bytes'), 'reserved bytes', integer=True),
            'max_observed_rss_bytes': optional_number(record.get('max_observed_rss_bytes'), 'RSS bytes', integer=True),
            'memory_scope': record.get('runtime', {}).get('cuda_peak_scope'),
            'phase_timing_scope': 'synchronized_phase_wall_including_evaluation_and_durable_checkpoints',
            'phase_gpu_memory_limit_bytes': record.get('gpu_memory_limit_bytes')})
    if sum(r['optimizer_steps'] for r in rows) != fit['optimizer_steps']:
        raise ValueError('Phase optimizer steps do not reconcile')
    return rows


def extract_cached(source, admission, source_id, directory):
    if sha(canonical({k: v for k, v in source.items() if k != 'sha256'})) != source['sha256']:
        raise ValueError('Cached source self-hash mismatch')
    outcome = source['outcome']
    if (sha(canonical(outcome)) != source['outcome_sha256'] or admission['result_sha256'] != source['sha256']
            or admission.get('reuse_approved') is not True or outcome.get('status') != 'completed'):
        raise ValueError('Cached outcome/admission identity mismatch')
    resource = outcome['resource_use']
    identity = resource['identity']
    if (identity['model'] != admission['model_id'] or identity['seed'] != admission['seed']
            or identity['origin'] != admission['origin'] or identity['stage'] != 'hpo'
            or source['job_id'] != admission['job_id']):
        raise ValueError('Cached HPO role mismatch')
    rows = []
    for phase, record in resource['phase_records'].items():
        rows.append({'stage': 'selected_hpo', 'model': identity['model'], 'seed': identity['seed'],
            'source_attempt_id': source_id, 'source_directory': directory, 'trial_id': admission['parent_trial_id'],
            'request_sha256': None, 'checkpoint_sha256': None,
            'actual_parameter_count': optional_number(resource.get('model_parameters'), 'cached parameters', integer=True),
            'device_name': None, 'cpu_threads': None, 'torch': None, 'precision': None,
            'amp': None, 'tf32': None, 'deterministic': None,
            'metadata_kind': 'original_cached_HPO_outcome', 'fold_origin': identity['origin'],
            'inference_seconds': None, 'FLOPs': None, 'phase': phase,
            'elapsed_seconds': number(record['elapsed_seconds'], 'cached phase seconds'),
            'optimizer_steps': number(record['optimizer_steps'], 'cached optimizer steps', integer=True),
            'epochs_completed': number(record['epochs_completed'], 'cached epochs', integer=True),
            'cuda_synchronized_timing': None, 'max_cuda_memory_allocated_bytes': None,
            'max_cuda_memory_reserved_bytes': None, 'max_observed_rss_bytes': None,
            'memory_scope': None, 'phase_gpu_memory_limit_bytes': None,
            'phase_timing_scope': 'historical_cached_phase_wall_with_different_checkpoint_policy'})
    return rows


def pareto(rows, cost_key, quality_key='RMSE'):
    """Minimize both verified coordinates. Ties do not dominate each other."""
    eligible = [r for r in rows if r.get(cost_key) is not None and r.get(quality_key) is not None]
    for row in eligible:
        number(row[cost_key], cost_key)
        number(row[quality_key], quality_key)
    return {r['model']: not any(
        s[cost_key] <= r[cost_key] and s[quality_key] <= r[quality_key]
        and (s[cost_key] < r[cost_key] or s[quality_key] < r[quality_key])
        for s in eligible) for r in eligible}


def audit_ledger(raw, budget):
    verify(raw, {'bytes': budget['ledger_source_bytes'], 'sha256': budget['ledger_source_sha256']}, 'closed budget ledger')
    rows = [json.loads(line) for line in raw.splitlines()]
    previous, active = '0'*64, set()
    for i, row in enumerate(rows):
        digest = sha(canonical({k: v for k, v in row.items() if k != 'sha256'}))
        if row['sequence'] != i or row['previous_sha256'] != previous or row['sha256'] != digest:
            raise ValueError('Budget ledger hash chain mismatch')
        previous = digest
        if row['kind'] == 'reserved':
            if row['attempt_id'] in active:
                raise ValueError('Duplicate reservation')
            active.add(row['attempt_id'])
        elif row['kind'] == 'charged':
            if row['attempt_id'] not in active:
                raise ValueError('Unmatched charge')
            active.remove(row['attempt_id'])
    charged = sum(number(r['elapsed_seconds'], 'charge') for r in rows if r['kind'] == 'charged')
    overhead = sum(number(r['elapsed_seconds'], 'overhead') for r in rows if r['kind'] == 'overhead')
    total = charged + overhead
    if active or previous != budget['terminal_event_sha256'] or not math.isclose(total, budget['current_ledger_charged_and_overhead_seconds'], abs_tol=1e-8):
        raise ValueError('Closed ledger does not reconcile with published evidence')
    if 'total_charged_seconds' in budget:
        combined = total + number(budget['historical_charged_seconds'], 'historical charge')
        if (not math.isclose(combined, budget['total_charged_seconds'], abs_tol=1e-8)
                or not math.isclose(budget['effective_total_hours']*3600-combined, budget['remaining_seconds'], abs_tol=1e-8)
                or len(rows) != budget['records']):
            raise ValueError('Historical/current budget totals do not reconcile')
    return {**budget, 'independently_verified_hash_chain': True,
            'current_worker_charged_seconds': charged, 'current_controller_overhead_seconds': overhead,
            'reserved_attempt_count': sum(r['kind'] == 'reserved' for r in rows),
            'reservation_sum_seconds_not_actual_cost': sum(r['reservation_seconds'] for r in rows if r['kind'] == 'reserved'),
            'monetary_cost': None, 'energy_kWh': None,
            'scope': 'current_workers_and_controller_plus_inherited_historical_charges_not_GPU_active_hours'}


def summarize(phases, quality):
    by_model = defaultdict(list)
    for row in phases:
        by_model[row['model']].append(row)
    result = []
    for q in quality:
        model = q['model']
        rows = by_model[model]
        final = [r for r in rows if r['stage'] == 'final']
        hpo = [r for r in rows if r['stage'] == 'selected_hpo']
        seeds = {r['seed'] for r in final}
        if int(q['n_common_dates']) != 608 or int(q['R']) not in (1, 6):
            raise ValueError('Unexpected original A2 evaluation scope')
        if int(q['R']) == 6 and seeds != SEEDS:
            raise ValueError('Missing or extra final seed')
        parameters = {r['actual_parameter_count'] for r in final}
        if final and (len(parameters) != 1 or None in parameters):
            raise ValueError('Final parameter count differs across seeds')
        times = [sum(r['elapsed_seconds'] for r in final if r['seed'] == seed) for seed in sorted(seeds)]
        final_sum = strict_sum(times)
        steps_per_seed = {sum(r['optimizer_steps'] for r in final if r['seed'] == seed) for seed in seeds}
        if len(steps_per_seed) > 1:
            raise ValueError('Frozen final optimizer steps differ across seeds')
        if final and len({r['trial_id'] for r in final}) != 1:
            raise ValueError('Final seeds do not share one selected recipe')
        if hpo and ({r['fold_origin'] for r in hpo} != {'2024-01-01', '2024-07-01', '2024-10-01'}
                    or len({r['source_attempt_id'] for r in hpo}) != 3):
            raise ValueError('Selected HPO recipe lacks three registered origins')
        if hpo and {r['trial_id'] for r in hpo} != {r['trial_id'] for r in final}:
            raise ValueError('HPO resource records are not the frozen winning recipe')
        record = {'model': model, 'RMSE': number(float(q['RMSE']), 'RMSE'),
            'RMSE_seed_SD': float(q['RMSE_seed_SD']) if q['RMSE_seed_SD'] else None,
            'n_common_dates': int(q['n_common_dates']), 'R': int(q['R']),
            'model_scope': 'six_seed_neural_frozen_predictors' if final else 'single_deterministic_baseline',
            'actual_parameter_count': next(iter(parameters)) if final else None,
            'final_fit_count': int(q['R']), 'final_phase_seconds_sum': final_sum,
            'final_phase_seconds_mean_per_seed': statistics.mean(times) if times else None,
            'final_phase_seconds_SD_across_seeds': statistics.stdev(times) if len(times)>1 else None,
            'final_optimizer_steps_sum': strict_sum(r['optimizer_steps'] for r in final),
            'final_optimizer_steps_per_seed': strict_sum(r['optimizer_steps'] for r in final if r['seed'] == min(seeds)) if seeds else None,
            'final_device_names': ';'.join(sorted({r['device_name'] for r in final if r['device_name']})) or None,
            'selected_hpo_fit_count': len({r['source_attempt_id'] for r in hpo}),
            'selected_hpo_phase_seconds_sum': strict_sum(r['elapsed_seconds'] for r in hpo),
            'selected_hpo_optimizer_steps_sum': strict_sum(r['optimizer_steps'] for r in hpo),
            'selected_hpo_timing_scope': 'selected_recipe_three_internal_origins_only_mixed_hardware_and_checkpoint_policies' if hpo else None,
            'full_selection_seconds': None, 'inference_seconds': None, 'inference_latency_ms': None,
            'FLOPs': None, 'monetary_cost': None, 'energy_kWh': None,
            'final_training_peak_allocated_bytes_observed': max((r['max_cuda_memory_allocated_bytes'] for r in final if r['max_cuda_memory_allocated_bytes'] is not None), default=None),
            'final_memory_scope': 'process_highwater_since_callers_last_reset_not_isolated_model_memory' if final else None}
        for phase in ('ae1', 'ae2', 'supervised'):
            record[f'final_{phase}_seconds_sum'] = strict_sum(r['elapsed_seconds'] for r in final if r['phase'] == phase)
        result.append(record)
    for cost, key in (('final_phase_seconds_mean_per_seed', 'pareto_RMSE_vs_measured_final_phase_time'),
                      ('actual_parameter_count', 'pareto_RMSE_vs_parameter_count')):
        frontier = pareto(result, cost)
        for row in result:
            row[key] = frontier.get(row['model'])
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output, rows):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/e2-efficiency-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.hashsalt': VERSION, 'font.size': 9})
    labels = {'LSTM-autoencoder con predictor': 'SAE', 'Inception-1D': 'Inception', 'ResNet-1D': 'ResNet', 'FCN-1D': 'FCN'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, cost, flag, xlabel in (
        (axes[0], 'final_phase_seconds_mean_per_seed', 'pareto_RMSE_vs_measured_final_phase_time', 'Tiempo de fases finales medio/semilla (s; log)'),
        (axes[1], 'actual_parameter_count', 'pareto_RMSE_vs_parameter_count', 'Parámetros del predictor seleccionado (log)')):
        data = [r for r in rows if r[cost] is not None]
        for i, row in enumerate(data):
            color = '#c25a20' if row[flag] else '#31688e'
            axis.scatter(row[cost], row['RMSE'], color=color, zorder=3, s=35)
            right_edge = row[cost] > max(r[cost] for r in data)*.7
            axis.annotate(labels.get(row['model'], row['model']), (row[cost], row['RMSE']),
                          xytext=(-5 if right_edge else 4, 5 if i%2 else -11),
                          ha='right' if right_edge else 'left', textcoords='offset points', fontsize=7)
        frontier = sorted((r for r in data if r[flag]), key=lambda r:r[cost])
        axis.plot([r[cost] for r in frontier], [r['RMSE'] for r in frontier], color='#c25a20', alpha=.65, linewidth=1)
        axis.set_xscale('log'); axis.set_xlabel(xlabel); axis.set_ylabel('RMSE A2 original; 608 fechas comunes')
        axis.grid(True, alpha=.2)
    fig.suptitle('Pareto descriptivo: 13 modelos neuronales, 6 semillas\nFases con evaluación y checkpoints; sin selección HPO ni latencia de inferencia', fontsize=10)
    fig.tight_layout()
    fig.savefig(output/'quality_cost_pareto.svg', metadata={'Date': None})
    fig.savefig(output/'quality_cost_pareto.png', dpi=180, metadata={'Software': VERSION})
    plt.close(fig)


def collect(training, meta, ledger, benchmark_evidence, output):
    output.mkdir(parents=True, exist_ok=True)
    source_raw = (training/'source_evidence.json').read_bytes()
    source = json.loads(source_raw)
    if source['immutable_results_commit'] != RESULTS_COMMIT or source['scientific_training_commit'] != SCIENTIFIC_COMMIT:
        raise ValueError('Original evidence commit mismatch')
    freeze_raw = archive_member(meta, 'freeze', 'final_freeze.json')
    freeze = json.loads(freeze_raw)
    quality_raw = archive_member(meta, 'score', 'scientific_outputs/benchmark_A2.csv')
    quality = list(csv.DictReader(io.StringIO(quality_raw.decode())))
    frozen = {(r['model'], r['seed']): r for r in freeze['models']}
    from collect_e2_f2a_training_evidence import hpo_jobs
    registered_jobs = {j['logical_job_id']: j for j in hpo_jobs(ROOT)}
    phases, verified_inputs = [], []
    for record in source['source_records']:
        directory = training/'sources'/safe_path(record['directory'])
        manifest_raw = (directory/'manifest.json').read_bytes()
        if sha(manifest_raw) != record['manifest_sha256']:
            raise ValueError('Original manifest hash mismatch')
        manifest = json.loads(manifest_raw)
        descriptors = {a['path']:a for a in manifest['artifacts']}
        data = {}
        for artifact in record['artifacts']:
            name = safe_path(artifact['path'])
            raw = verify((directory/name).read_bytes(), artifact, name)
            verify(raw, descriptors[name], name)
            data[name] = json.loads(raw)
        if 'cached_source_record.json' in data:
            entries = extract_cached(data['cached_source_record.json'], data['cache_admission.json'], record['attempt_id'], record['directory'])
        else:
            prefix = 'fit/' if record['directory'].startswith('final/') else ''
            stage = 'final' if prefix else 'selected_hpo'
            fit = data[prefix+'job_result.json']
            request = data[prefix+'request.json']
            entries = extract_fit(fit, request, stage, record['attempt_id'], record['directory'])
            if stage == 'final' and frozen[(fit['model_id'], fit['seed'])]['checkpoint_sha256'] != fit['selected_checkpoint']['sha256']:
                raise ValueError('Final cost does not bind to frozen scored checkpoint')
        if record['directory'].startswith('jobs/'):
            job = registered_jobs[record['directory'].split('/')[-1]]
            if any((e['model'], e['seed'], e['trial_id']) != (job['model_id'], job['seed'], job['trial']['trial_id']) for e in entries):
                raise ValueError('Selected HPO registered job identity mismatch')
            for entry in entries:
                entry['fold_origin'] = job['fold_origin']
        phases.extend(entries)
        verified_inputs.append(record)
    final_ids = {(r['model'],r['seed']) for r in phases if r['stage']=='final'}
    hpo_ids = {r['source_attempt_id'] for r in phases if r['stage']=='selected_hpo'}
    if len(final_ids) != 78 or len(hpo_ids) != 39 or len(frozen) != 81:
        raise ValueError('Original fit inventory coverage mismatch')
    models = summarize(phases, quality)
    published = json.loads(benchmark_evidence.read_bytes())
    budget = audit_ledger(ledger.read_bytes(), published['budget'])
    write_csv(output/'phase_resources.csv', phases)
    write_csv(output/'model_efficiency.csv', models)
    (output/'benchmark_A2_original.csv').write_bytes(quality_raw)
    plot(output, models)
    (output/'model_efficiency.json').write_bytes(canonical(models)+b'\n')
    evidence = {'version': VERSION, 'status': 'verified_original_metadata_only',
        'immutable_results_commit': RESULTS_COMMIT, 'scientific_training_commit': SCIENTIFIC_COMMIT,
        'training_performed': False, 'model_inference_performed': False, 'weight_bytes_downloaded': 0,
        'final_metrics_recomputed': False, 'final_fit_count': len(final_ids), 'selected_hpo_fit_count': len(hpo_ids),
        'neural_model_count': 13, 'deterministic_baseline_count': 3, 'frozen_checkpoint_count': len(frozen),
        'actual_A2_common_dates': 608, 'final_phase_seconds_sum': sum(r['elapsed_seconds'] for r in phases if r['stage']=='final'),
        'selected_recipe_HPO_phase_seconds_sum': sum(r['elapsed_seconds'] for r in phases if r['stage']=='selected_hpo'),
        'full_selection_cost_per_model': None, 'budget': budget,
        'pareto_scope': 'posthoc_descriptive_neural_models_only_A2_mean_seed_RMSE_vs_mean_seed_final_phase_wall_or_parameter_count',
        'pareto_inference': 'observed_point_dominance_only_no_significance_or_uncertainty_claim',
        'baseline_exclusion': 'single_deterministic_predictor_resource_measurements_missing_not_zero',
        'timing_definition': 'sum_original_synchronized_phase_wall_includes_optimizer_train_evaluation_checkpointing_excludes_HPO_final_prediction_and_controller',
        'selected_hpo_scope': 'three_origins_for_the_winning_recipe_not_full_search_mixed_A100_T4_and_cache_policy',
        'memory_definition': 'observed_process_highwater_since_callers_last_reset_not_isolated_predictor_memory',
        'missing_measurements': ['isolated_inference_latency','FLOPs','energy','monetary_cost','per_model_full_selection_cost','classical_baseline_resource_measurements'],
        'original_archive_transports': {name: json.loads((meta/f'{name}_manifest.json').read_bytes())['archive_transport'] for name in ('freeze','score')},
        'input_hashes': {'collector_sha256':sha(Path(__file__).read_bytes()),
                        'HPO_identity_collector_sha256':sha((ROOT/'scripts/collect_e2_f2a_training_evidence.py').read_bytes()),
                        'registered_configurations_sha256':sha((ROOT/'experiments/e2_reduced_protocol_20261112/configurations.json').read_bytes()),
                        'TiDE_registered_configurations_sha256':sha((ROOT/'experiments/e2_tide_published_r0_20261112/configurations.json').read_bytes()),
                        'training_source_evidence_sha256': sha(source_raw), 'freeze_sha256':sha(freeze_raw),
                        'benchmark_A2_sha256':sha(quality_raw), 'budget_ledger_sha256':sha(ledger.read_bytes()),
                        'benchmark_source_evidence_sha256':sha(benchmark_evidence.read_bytes())},
        'source_records': verified_inputs,
        'outputs': {p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in sorted(output.iterdir()) if p.is_file() and p.name in ('phase_resources.csv','model_efficiency.csv','benchmark_A2_original.csv','quality_cost_pareto.svg','quality_cost_pareto.png','model_efficiency.json')}}
    (output/'source_evidence.json').write_bytes(canonical(evidence)+b'\n')
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-evidence', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--benchmark-evidence', type=Path, default=ROOT/'book_e2/_static/e2_benchmark_final_20261113/source_evidence.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    evidence = collect(args.training_evidence, args.metadata_dir, args.ledger, args.benchmark_evidence, args.output_dir)
    print(json.dumps({k:evidence[k] for k in ('status','final_fit_count','selected_hpo_fit_count','final_phase_seconds_sum','selected_recipe_HPO_phase_seconds_sum')}, indent=2))


if __name__ == '__main__':
    main()
