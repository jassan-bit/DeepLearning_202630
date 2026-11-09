"""Bounded synthetic-only common executor. No external training-data API.

The one fixed recipe per system exercises inner/refit plumbing, NOT HPO/NAS.
All 18 forecasts are fsynced before any evaluation label is revealed. Historical
modules are imported unchanged; no historic predictions or fitted weights load.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

import e1_core_models_mlp as e1
import e2_current_pipeline as plumbing
import e2_recipe_catalogue as catalogue
import e2_recipe_models as bodies
import e2_temporal_controls as controls
from dl_data_v2 import relative_features

VERSION = 'e2-common-synthetic-v1'
REFERENCE_SPECS = {'E1-GRU32-refit-E2': ('GRU', 32, .1, 3777),
                   'E1-LSTM32-refit-E2': ('LSTM', 32, 0., 5025),
                   'E1-MLP16-refit-E2': ('MLP', 16, .1, 593)}
CONTROL_NAMES = ('Persistence', 'Ridge', 'SeasonalNaive', 'ARIMA')
SYSTEM_NAMES = (*catalogue.MODEL_NAMES, *REFERENCE_SPECS, *CONTROL_NAMES)
DENIED_PARTS = {'data', 'results', 'experiments', 'references', 'book', 'book_e2', 'book_e3'}
SEED = 20261109


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _safe_path(value):
    raw = Path(value).absolute()
    resolved = raw.resolve()
    if DENIED_PARTS.intersection(raw.parts) or DENIED_PARTS.intersection(resolved.parts):
        raise ValueError('Synthetic files/sources cannot use market or historical artifact directories')
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError('Symlink paths are not permitted')
    return resolved


class DurableJournal:
    """Exclusive new directory, append-only hash-chained events, fsync per event.

    This is crash-auditable local persistence, not a tamper-proof remote service.
    A failed run is retained; retries require a distinct new directory.
    """
    def __init__(self, path):
        self.path = _safe_path(path)
        if 'synthetic' not in self.path.name.lower():
            raise ValueError('New output directory name must contain synthetic')
        if not self.path.parent.is_dir():
            raise ValueError('Output parent must already exist')
        self.path.mkdir(exist_ok=False)
        self.file = (self.path / 'events.jsonl').open('x', encoding='utf-8')
        self.sequence, self.previous = 0, '0' * 64
        self._sync_dir(self.path.parent)
        self._sync_dir(self.path)

    @staticmethod
    def _sync_dir(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def append(self, kind, **payload):
        event = {'sequence': self.sequence, 'previous_sha256': self.previous,
                 'kind': kind, **payload}
        event['event_sha256'] = _hash_json(event)
        self.file.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())
        self.sequence += 1
        self.previous = event['event_sha256']
        return event['event_sha256']

    def receipt(self, payload):
        with (self.path / 'synthetic_receipt.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self._sync_dir(self.path)

    def close(self):
        self.file.close()


def validate_journal(output_dir):
    """Verify a closed local journal's chain and complete-before-reveal ordering.

    Hash validation detects alteration, not authenticity. Fsync semantics belong
    to DurableJournal's writer and are not inferable from bytes after the fact.
    """
    path = _safe_path(output_dir)
    previous, count = '0' * 64, 0
    pending = None
    forecast_count = observation_count = 0
    scores, completed, stage = set(), False, None
    with (path / 'events.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            digest = event.pop('event_sha256')
            if (event['sequence'] != count or event['previous_sha256'] != previous
                    or _hash_json(event) != digest):
                raise ValueError('Journal hash chain or sequence mismatch')
            if completed:
                raise ValueError('Unexpected event after run completion')
            kind = event['kind']
            if pending is not None and kind != 'observation':
                raise ValueError('Prediction must immediately precede its observation')
            if kind == 'stage_plan':
                stage = event['stage']
            elif kind == 'predictions':
                if event['stage'] != stage or set(event['systems']) != set(SYSTEM_NAMES):
                    raise ValueError('Incomplete system forecast or incorrect stage')
                pending = (stage, event['index'], digest)
                forecast_count += 1
            elif kind == 'observation':
                if pending != (event['stage'], event['index'], event['prediction_event_sha256']):
                    raise ValueError('Observation without complete prior prediction')
                pending = None
                observation_count += 1
            elif kind == 'scores':
                if event['stage'] != stage or set(event['metrics_original_units']) != set(SYSTEM_NAMES):
                    raise ValueError('Incorrect score stage or incomplete systems')
                scores.add(stage)
            elif kind == 'run_completed':
                if scores != {'inner','refit'} or forecast_count != observation_count:
                    raise ValueError('Premature completion')
                completed = True
            previous, count = digest, count + 1
    if pending is not None or not completed:
        raise ValueError('Incomplete or failed run journal')
    with (path / 'synthetic_receipt.json').open(encoding='utf-8') as stream:
        receipt = json.load(stream)
    if receipt.pop('journal_final_sha256') != previous or _hash_json(receipt) != event['receipt_payload_sha256']:
        raise ValueError('Receipt not bound to completed journal')
    return {'validated': True, 'events': count, 'forecast_days': forecast_count,
            'observed_days': observation_count, 'journal_final_sha256': previous,
            'scores_stages': sorted(scores), 'writer_fsync_proven_from_bytes': False}


def _fixture():
    """Internal 320-day generator; no path, array, seed or data argument."""
    rng = np.random.default_rng(SEED)
    days = 320
    returns = rng.normal(.0002, .012, days + 7)
    close = (100 * np.exp(np.cumsum(returns)))[7:]
    opening = close * np.exp(rng.normal(0, .002, days))
    high = np.maximum(opening, close) * (1 + rng.uniform(.001, .01, days))
    low = np.minimum(opening, close) * (1 - rng.uniform(.001, .01, days))
    features = np.column_stack((opening, high, low, close, rng.lognormal(8, .3, days)))
    y = np.array([returns[i + 1:i + 8].std(ddof=0) for i in range(days)])
    dates = pd.date_range('2001-01-01', periods=days, freq='D', tz='UTC')
    # Eligibility is metadata fixed by construction, not model losses or y values.
    valid = np.ones(days, dtype=bool)
    return dates, features, y, valid


def _plan():
    return {'inner': {'origin': 200, 'train': np.arange(7, 193), 'evaluate': np.arange(200, 283)},
            'refit': {'origin': 290, 'train': np.arange(7, 283), 'evaluate': np.arange(290, 298)}}


def _validate_plan(plan, days):
    origin = plan['origin']
    if type(origin) is not int or not 8 <= origin < days:
        raise ValueError('Invalid integer forecast origin')
    for key in ('train', 'evaluate'):
        values = np.asarray(plan[key])
        if (values.ndim != 1 or not len(values) or values.dtype.kind not in 'iu'
                or np.any(values < 7) or np.any(values >= days)
                or np.any(np.diff(values.astype(np.int64)) != 1)):
            raise ValueError('Plan indices must be bounded contiguous ordered unique integers')
    if (len(plan['train']) < 90 or plan['train'][-1] != origin-8
            or plan['evaluate'][0] != origin):
        raise ValueError('Invalid causal stage plan: training cutoff or evaluation origin')


def _mask(valid, evaluation):
    """Availability-only common support, including all three seasonal candidates."""
    rows = []
    for t in evaluation:
        reasons = []
        if t < 7 or not valid[t-7:t].all():
            reasons.append('neural_support')
        for lag in (0, 1, 7, 14, 28):
            if t-lag < 7 or not valid[t-lag-7:t-lag+1].all():
                reasons.append('truth' if lag == 0 else f'lag_{lag}')
        rows.append({'index': int(t), 'eligible': not reasons, 'reasons': reasons})
    return rows


def _fit_scaler(features, y_train, train):
    windows = features[train[:, None] - np.arange(7, 0, -1)]
    relative = relative_features(windows)
    mean = relative.reshape(-1, 5).mean(axis=0)
    std = relative.reshape(-1, 5).std(axis=0, ddof=0)
    scale = np.where(std == 0, 1., std)
    y_scale = float(y_train.mean())
    if not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError('Invalid training target mean')
    preproc = {'x_mean': mean.tolist(), 'x_scale': scale.tolist(), 'y_scale': y_scale,
               'representation': 'relative_window', 'target_centered': False,
               'train_windows_sha256': plumbing.array_sha256(windows),
               'train_targets_sha256': plumbing.array_sha256(y_train),
               'train_indices_sha256': plumbing.array_sha256(train),
               'x_fit_window_positions': len(train) * 7}
    x = ((relative - mean) / scale).astype(np.float32)
    return preproc, x, (y_train / y_scale).astype(np.float32)


def _x_at(features, indices, scaler):
    """No label argument or lookup in the inference transform."""
    windows = features[np.asarray(indices)[:, None] - np.arange(7, 0, -1)]
    return ((relative_features(windows) - scaler['x_mean']) / scaler['x_scale']).astype(np.float32)


def _neural_fit(name, x, y, scaler, source_roots, journal, stage_name):
    torch.manual_seed(SEED)
    config = plumbing.SmokeConfig(seed=SEED, supervised_steps=1, pretrain_steps=1)
    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
    phases, audit = {}, None
    def step(phase, module, inputs, targets):
        journal.append('fit_phase_started', stage=stage_name, system=name, phase=phase)
        result = catalogue._step(module, inputs, targets, 1, config)
        journal.append('fit_phase_completed', stage=stage_name, system=name, phase=phase, record=result)
        return result
    if name == catalogue.current.AE_NAME:
        stage = bodies.SAEPretraining(inputs, scaler['train_windows_sha256'], accept_auxiliary_scale=True)
        phases['ae1'] = step('ae1', stage.ae1, inputs, stage.reconstruction_target(inputs))
        detached, target = stage.prepare_phase2()
        before = plumbing._state_sha256(stage.ae1.encoder)
        if detached.requires_grad or any(p.requires_grad for p in stage.ae1.encoder.parameters()):
            raise ValueError('SAE phase2 encoder not frozen')
        phases['ae2'] = step('ae2', stage.ae2, detached, target)
        after = plumbing._state_sha256(stage.ae1.encoder)
        if before != after:
            raise ValueError('SAE encoder1 changed during phase2')
        model = stage.transfer_predictor()
        audit = {'encoder1_frozen_phase2': True, 'encoder1_before': before, 'encoder1_after': after,
                 'encoder2_transfer_sha256': plumbing._state_sha256(stage.ae2.encoder),
                 'auxiliary_scale': catalogue.fit_auxiliary_scale(inputs), 'scale_fit': 'train_only',
                 'both_reconstruction_targets': 'original transformed five-channel X / s',
                 'decoder_in_predictor': any('decoder' in n for n, _ in model.named_parameters()),
                 'transfer_exact': plumbing._state_sha256(model.encoder1) == after and
                    plumbing._state_sha256(model.encoder2) == plumbing._state_sha256(stage.ae2.encoder)}
        if audit['decoder_in_predictor'] or not audit['transfer_exact']:
            raise ValueError('Invalid SAE transfer')
    elif name in REFERENCE_SPECS:
        family, width, dropout, expected = REFERENCE_SPECS[name]
        model = e1.build_model(family, 7, width, dropout)
        if sum(p.numel() for p in model.parameters()) != expected:
            raise ValueError('E1 architecture fingerprint changed')
    else:
        model = catalogue.build_model(name, source_roots=source_roots)
    phases['supervised'] = step('supervised', model, inputs, targets)
    model.eval()
    return model, {'phases': phases, 'ae_protocol': audit,
                   'architecture': e1.architecture_spec(model) if name in REFERENCE_SPECS else
                                   getattr(model, 'architecture_metadata', {}),
                   'parameter_count': sum(p.numel() for p in model.parameters()),
                   'final_state_sha256': plumbing._state_sha256(model),
                   'historical_E1_metrics_or_weights_reused': False}


def _metrics(truth, prediction, train_y):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if not len(truth) or truth.shape != prediction.shape:
        raise ValueError('Scoring requires nonempty identical common cohorts')
    error = prediction - truth
    mse = float(np.mean(error**2))
    centered = float(np.sum((truth-truth.mean())**2))
    positive = truth > 0
    denominator = float(np.mean(np.abs(np.diff(train_y))))
    return {'RMSE': float(np.sqrt(mse)), 'MAE': float(np.mean(np.abs(error))), 'MSE': mse,
            'R2': None if centered == 0 else float(1-np.sum(error**2)/centered),
            'MAPE_percent': None if not positive.any() else float(np.mean(np.abs(error[positive]/truth[positive]))*100),
            'MAPE_coverage': float(positive.mean()), 'n': len(truth),
            'MASE': None if denominator == 0 else float(np.mean(np.abs(error))/denominator),
            'MASE_train_adjacent_denominator': denominator}


def _run_stage(stage_name, plan, fixture, roots, journal):
    dates, features, hidden_y, valid = fixture
    train, evaluation, origin = plan['train'], plan['evaluate'], plan['origin']
    _validate_plan(plan, len(dates))
    mask = _mask(valid, evaluation)
    journal.append('stage_plan', stage=stage_name, origin=str(dates[origin]),
                   train_indices=train.tolist(), evaluation_indices=evaluation.tolist(), common_mask=mask)
    # The hidden fixture array is owned here, never supplied to any model.
    train_y = hidden_y[train].copy()
    scaler, x, scaled_y = _fit_scaler(features, train_y, train)
    journal.append('preprocessing', stage=stage_name, scaler=scaler)
    models, fit_records, ledger = {}, {}, []
    for name in SYSTEM_NAMES:
        start = time.perf_counter()
        journal.append('fit_started', stage=stage_name, system=name)
        if name in (*catalogue.MODEL_NAMES, *REFERENCE_SPECS):
            model, record = _neural_fit(name, x, scaled_y, scaler, roots, journal, stage_name)
            models[name] = model
            fits = len(record['phases'])
        elif name == 'Ridge':
            flat = x.astype(np.float64).reshape(len(x), -1)
            xm, ym = flat.mean(0), scaled_y.astype(np.float64).mean()
            centered = flat-xm
            coef = np.linalg.solve(centered.T @ centered + np.eye(flat.shape[1]),
                                   centered.T @ (scaled_y-ym))
            models[name] = (coef, float(ym-xm @ coef))
            record = {'alpha': 1., 'intercept_unpenalized': True,
                      'coefficient_sha256': plumbing.array_sha256(coef), 'intercept': models[name][1]}
            fits = 1
        elif name == 'ARIMA':
            model = controls.FrozenARIMA.fit(pd.Series(train_y, index=dates[train]), dates[origin],
                (1, 0, 0), source_commit=_hash_json(_source_hashes()))
            models[name] = model
            record = dict(model.receipt)
            fits = 1
            for t in range(int(train[-1])+1, origin):
                model.observe(dates[t], hidden_y[t])
            record['bridge_indices'] = list(range(int(train[-1])+1, origin))
            record['parameters_sha256'] = plumbing.array_sha256(model._result.params)
        else:
            record = {'period': 1 if name == 'Persistence' else 7, 'fit_required': False,
                      'seasonal_candidate_periods_not_separate_systems': [7, 14, 28]}
            fits = 0
        record['elapsed_seconds'] = time.perf_counter()-start
        record['fit_count'] = fits
        fit_records[name] = record
        ledger.append({'stage': stage_name, 'system': name, 'fit_count': fits,
                       'elapsed_seconds': record['elapsed_seconds']})
        journal.append('fit_completed', stage=stage_name, system=name, record=record)
    frozen_states = {n: plumbing._state_sha256(m) for n,m in models.items() if isinstance(m, torch.nn.Module)}
    # Only observed past labels go into the temporal history.
    past = pd.Series(hidden_y[:origin].copy(), index=dates[:origin])
    truths, predictions = [], {name: [] for name in SYSTEM_NAMES}
    for row, t in zip(mask, evaluation):
        values = {}
        inputs = _x_at(features, [t], scaler)
        for name in SYSTEM_NAMES:
            if name in frozen_states:
                raw = float(plumbing.predict_original_units(models[name], inputs, scaler)[0])
                values[name] = {'raw': raw, 'value': raw, 'truncated': False}
            elif name == 'Ridge':
                coef, intercept = models[name]
                raw = float((inputs.astype(np.float64).reshape(1, -1) @ coef + intercept)[0] * scaler['y_scale'])
                values[name] = asdict(controls.output(dates[t], raw))
            elif name == 'ARIMA':
                values[name] = asdict(models[name].forecast(dates[t]))
            else:
                values[name] = asdict(controls.seasonal_naive(past, dates[t], 1 if name == 'Persistence' else 7))
        if set(values) != set(SYSTEM_NAMES) or any(not np.isfinite(v['value']) for v in values.values()):
            raise ValueError('Incomplete/nonfinite common forecast; no date deletion or fallback')
        # Persist ALL systems atomically in one complete JSONL record, flush/fsync.
        token = journal.append('predictions', stage=stage_name, index=int(t), date=str(dates[t]),
                               eligible=row['eligible'], systems=values)
        # First access to this evaluation truth occurs strictly after successful fsync.
        observation = float(hidden_y[t])
        journal.append('observation', stage=stage_name, index=int(t), value=observation,
                       prediction_event_sha256=token)
        models['ARIMA'].observe(dates[t], observation)
        past.loc[dates[t]] = observation
        if row['eligible']:
            truths.append(observation)
            for name in SYSTEM_NAMES:
                predictions[name].append(values[name]['value'])
    for name, state in frozen_states.items():
        if plumbing._state_sha256(models[name]) != state:
            raise ValueError('Neural state changed during evaluation')
    if plumbing.array_sha256(models['ARIMA']._result.params) != fit_records['ARIMA']['parameters_sha256']:
        raise ValueError('ARIMA parameters changed during evaluation')
    metrics = {name: _metrics(truths, predictions[name], train_y) for name in SYSTEM_NAMES}
    journal.append('scores', stage=stage_name, metrics_original_units=metrics)
    return {'plan': {'origin': origin, 'train': train.tolist(), 'evaluate': evaluation.tolist()},
            'mask': mask, 'preprocessing': scaler, 'fit_records': fit_records, 'fit_ledger': ledger,
            'metrics_original_units': metrics, 'evaluation_weights_scalers_fixed': True,
            'arima_filter_updates': len(models['ARIMA'].observations),
            'arima_truncation_rate': models['ARIMA'].truncation_rate}


def _source_hashes():
    root = Path(__file__).resolve().parent
    names = ('e2_common_executor.py', 'smoke_e2_common_executor.py', 'e2_recipe_catalogue.py',
             'e2_recipe_models.py', 'e2_current_pipeline.py', 'e2_current_models.py',
             'e2_temporal_controls.py', 'e1_core_models_mlp.py', 'dl_models.py',
             'dl_models_gru_lstm_tcn.py', 'dl_data_v2.py', 'dl_data.py',
             'e2_proposed_models.py', 'e2_final_models_attention.py', 'e2_final_models_mlp.py')
    result = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    manifest = 'experiments/e2_completion_20261106/design.json'
    result[manifest] = hashlib.sha256((root.parent/manifest).read_bytes()).hexdigest()
    return result


def run_synthetic_common(output_dir, *, synthetic_only=True, source_roots=None):
    """Execute fixed internal fixture, 18 systems, one inner fit and fresh refit.

    No external data, feature, target, checkpoint, seed, model subset, training
    budget or market path can be passed. ``source_roots`` are hash-verified author
    code only. The destination must be new and outside historical artifact trees.
    """
    if synthetic_only is not True:
        raise ValueError('Only synthetic_only=True is authorized')
    roots = None
    if source_roots is not None:
        if set(source_roots) != {'TimesNet', 'TimeMixer', 'PatchTST', 'iTransformer'}:
            raise ValueError('Exactly four pinned author source roots required')
        roots = {n: str(_safe_path(p)) for n,p in source_roots.items()}
    journal = DurableJournal(output_dir)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        fixture, plans = _fixture(), _plan()
        dates, features, y, valid = fixture
        if plans['inner']['train'][-1] != plans['refit']['origin']-98 or len(plans['inner']['evaluate']) != 83:
            raise ValueError('Inner E1 calendar relationship not preserved')
        sources = _source_hashes()
        journal.append('run_started', version=VERSION, synthetic_only=True,
                       sources_sha256=sources, systems=list(SYSTEM_NAMES), fixed_seed=SEED,
                       features_sha256=plumbing.array_sha256(features),
                       targets_sha256=plumbing.array_sha256(y),
                       eligibility_sha256=plumbing.array_sha256(valid))
        inner = _run_stage('inner', plans['inner'], fixture, roots, journal)
        selection = {'kind': 'singleton_identity_fixture_only', 'candidates_per_system': 1,
                     'uses_outer_evaluation': False, 'search_executed': False,
                     'seasonal_period_fixed_without_comparison': 7, 'ARIMA_order_fixed': [1,0,0],
                     'no_statistical_model_selection_claim': True}
        journal.append('selection_frozen', selection=selection)
        refit = _run_stage('refit', plans['refit'], fixture, roots, journal)
        if inner['preprocessing']['train_indices_sha256'] == refit['preprocessing']['train_indices_sha256']:
            raise ValueError('Inner/refit preprocessing cohorts must differ')
        ledger = inner['fit_ledger'] + refit['fit_ledger']
        verification = {n: r['architecture']['source_verification']
                        for n,r in refit['fit_records'].items()
                        if 'source_verification' in r.get('architecture', {})}
        verified_count = sum(len(v['verified_files']) for v in verification.values())
        if verified_count != 70:
            raise ValueError('Full executor requires all 70 pinned author files verified')
        if _source_hashes() != sources:
            raise ValueError('Source files changed during the synthetic run')
        receipt = {'version': VERSION, 'status': 'synthetic_mechanisms_only', 'synthetic_only': True,
                   'market_data_read': False, 'market_training_executed': False,
                   'historical_metadata_read': ['experiments/e2_completion_20261106/design.json'],
                   'historical_metadata_purpose': 'Pinned author-source verification only, not fitted weights or market data', 'HPO_executed': False,
                   'NAS_executed': False, 'pilot_executed': False, 'final_benchmark_executed': False,
                   'historical_E1_metrics_reused': False, 'system_count': len(SYSTEM_NAMES),
                   'systems': list(SYSTEM_NAMES), 'catalogue_count': len(catalogue.MODEL_NAMES),
                   'recipe_ids': {**catalogue.RECIPE_IDS,
                                  **{n: n+'-synthetic-one-step' for n in REFERENCE_SPECS},
                                  'Persistence': 'lag1-fixed', 'Ridge': 'alpha1-fixed',
                                  'SeasonalNaive': 'lag7-fixed', 'ARIMA': 'order100-fixed'},
                   'training_contract': {'optimizer': 'Adam', 'learning_rate': .001,
                       'batch_size': 8, 'steps_per_neural_phase': 1, 'gradient_clip_norm': 1.,
                       'reference_architectures': 'E1 GRU32/LSTM32/MLP16 instantiated afresh for E2',
                       'historical_E1_optimization_schedule_inherited': False,
                       'target': 'sigma7 population SD, original units after inverse training-only mean',
                       'primary_metric': 'per-system single-seed RMSE, no ensemble or multi-seed inference'},
                   'E1_references_separate_from_catalogue': list(REFERENCE_SPECS),
                   'source_sha256': sources, 'selection': selection,
                   'verified_author_sources': verification, 'verified_author_file_count': verified_count,
                   'author_verification_scope': '70 declared files in four external adapters, not entire repositories',
                   'stages': {'inner': inner, 'refit': refit},
                   'total_fit_stages': sum(r['fit_count'] for r in ledger), 'fit_ledger': ledger,
                   'neural_optimizer_steps': sum(p['steps'] for s in (inner,refit)
                        for r in s['fit_records'].values() for p in r.get('phases',{}).values()),
                   'all_optimizers_fresh': True,
                   'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                               'numpy': np.__version__, 'pandas': pd.__version__,
                               'statsmodels': controls._installed_version('statsmodels'), 'torch_threads': 1},
                   'fixture': {'days': len(dates), 'calendar_start': str(dates[0]),
                               'calendar_end': str(dates[-1]), 'seed': SEED,
                               'generator': 'internal OHLCV and seven-return population SD',
                               'features_sha256': plumbing.array_sha256(features),
                               'targets_sha256': plumbing.array_sha256(y)},
                   'limits': ['One step per neural phase; not meaningful predictive training.',
                              'Singleton identity only; no full A/B selection, HPO or NAS.',
                              'Synthetic timing is not a market compute estimate.',
                              'Local fsync/hash chain is crash audit, not adversarial tamper proof.',
                              'No market generalization, superiority or confirmatory inference.',
                              'Complete finite synthetic fixture only; general missing-data execution is not certified.',
                              'MASE denominator uses adjacent labels in the contiguous training fixture.',
                              'Unavailable required seasonal lag fails explicitly; no fallback or score-driven mask changes.']}
        receipt['journal_before_completion_sha256'] = journal.previous
        journal.append('run_completed', receipt_payload_sha256=_hash_json(receipt))
        receipt['journal_final_sha256'] = journal.previous
        journal.receipt(receipt)
        return receipt
    except Exception as exc:
        journal.append('run_failed', error_type=type(exc).__name__, error=str(exc),
                       control_receipt=getattr(exc, 'receipt', None), retry_or_fallback=False)
        raise
    finally:
        journal.close()
        torch.set_num_threads(previous_threads)
