"""Publish aggregate evidence for the GRU/LSTM/TCN replacement, without fitting.

The stored experiment is immutable. Its CV-selected baseline precedes its new
scoring of an already-observed historical test. The three-model comparisons,
diagnostics and future operational margin implemented here were defined AFTER
that test and are exploratory. This is not a new pre-analysis registration.

Only allowlisted aggregate metrics, diagnostic statistics and configuration
metadata are exported. Weights, scalers, raw files and machine-readable daily
forecast/residual vectors are never exported. One explicitly fixed-seed PNG
illustration shows the observed/predicted trajectory without a daily CSV.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gru-lstm-tcn-public-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from gru_lstm_tcn_diagnostics import diagnose_phase


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path('/workspace/e1_lstm_gru_tcn_local_20261105')
PUBLIC_NAME = 'gru_lstm_tcn_20261105'
MODELS = ('GRU', 'LSTM', 'TCN')
HISTORIES = (7, 14, 21, 28)
SEEDS = tuple(range(4101, 4111))
METRICS = ('rmse', 'mae', 'mse', 'r2', 'mape', 'mase')
KEY = ['history_days', 'model', 'seed']
PAIRS = tuple(itertools.combinations(MODELS, 2))
SCOPE = ('Análisis del nuevo trío definido después de inspeccionar el test histórico; '
         'exploratorio y condicional al calendario y a la aleatoriedad de entrenamiento. '
         'No constituye confirmación temporal independiente.')
FIGURES = ('metrics_by_history.png', 'learning_curves_D7.png', 'residual_histograms_D7.png',
           'monthly_rmse_D7.png', 'seed_rmse_D7.png', 'predictions_D7.png')
PUBLIC_FILES = {
    'summary.json', 'protocols_manifest.json', 'source_manifest.json', 'model_specs.json',
    'runtime.json', 'annual_metric_summary_test.csv',
    'selected_by_cv.json', 'public_analysis_plan.json', 'future_margin.json',
    'configuration_catalog_cv.csv', 'configuration_catalog_final.csv',
    'metric_summary_cv.csv', 'metric_summary_test.csv',
    'run_metrics_cv.csv', 'run_metrics_test.csv', 'monthly_summary_cv.csv', 'monthly_summary_test.csv',
    'within_trio_pairs_cv.csv', 'within_trio_pairs_test.csv',
    'within_trio_pairs_allD_cv.csv', 'within_trio_pairs_allD_test.csv',
    'friedman_D7.csv',
    'bds_by_fold_cv.csv', 'bds_by_fold_test.csv', 'ljung_box_by_fold_cv.csv', 'ljung_box_by_fold_test.csv',
    'dm_pairs_cv.csv', 'dm_pairs_test.csv', 'diagnostic_summary_cv.csv', 'diagnostic_summary_test.csv',
    'cost_summary_cv.csv', 'cost_summary_test.csv', 'seed_variance.csv',
    'learning_curve_summary.csv', 'residual_histogram_D7.csv',
    'README.md', 'report.md', *FIGURES,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def calendar(phase):
    if phase == 'cv':
        return pd.date_range('2024-01-01', '2024-12-31', tz='UTC')
    if phase == 'test':
        return pd.date_range('2025-01-01', '2026-08-31', tz='UTC')
    raise ValueError('Unknown phase')


def validate_predictions(frame, phase):
    frame = frame.copy()
    required = set(KEY + ['fold_id', 'outcome_day', 'y_true', 'y_pred', 'persistence_pred', 'mase_train_scale'])
    if not required <= set(frame):
        raise ValueError('Missing forecast fields')
    frame['history_days'] = frame.history_days.astype(int)
    frame['seed'] = frame.seed.astype(int)
    frame['outcome_day'] = pd.to_datetime(frame.outcome_day, utc=True).dt.strftime('%Y-%m-%d')
    days = calendar(phase).strftime('%Y-%m-%d').tolist()
    expected = {(d, m, s) for d in HISTORIES for m in MODELS for s in SEEDS}
    observed = set(frame[KEY].drop_duplicates().itertuples(index=False, name=None))
    if (observed != expected or len(frame) != len(days) * 120
            or frame.duplicated(KEY + ['outcome_day']).any()):
        raise ValueError('Incomplete or duplicate stored experiment grid: ' + phase)
    if 'variant' in frame and set(frame.variant) != {'relative_window'}:
        raise ValueError('Changed preprocessing variant')
    if 'horizon' in frame and not frame.horizon.eq(1).all():
        raise ValueError('Changed forecast horizon')
    values = frame[['y_true', 'y_pred', 'persistence_pred', 'mase_train_scale']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, :3] < 0).any() or (values[:, 3] <= 0).any():
        raise ValueError('Invalid volatility forecast/target or historical MASE denominator')
    reference = None
    for (d, _, _), part in frame.groupby(KEY, sort=True):
        part = part.sort_values('outcome_day')
        if part.outcome_day.tolist() != days or part.fold_id.tolist() != part.outcome_day.str[:7].tolist():
            raise ValueError('Unequal calendar or monthly reporting interval')
        actual_days = pd.DatetimeIndex(pd.to_datetime(part.outcome_day, utc=True))
        for field, expected_time in [('input_first_day', actual_days - pd.Timedelta(days=d)),
                                     ('input_last_day', actual_days - pd.Timedelta(days=1)),
                                     ('prediction_origin_utc', actual_days),
                                     ('target_available_at_utc', actual_days + pd.Timedelta(days=1))]:
            if field in part and not np.array_equal(pd.to_datetime(part[field], utc=True), expected_time):
                raise ValueError('Incorrect information clock: ' + field)
        current = part[['y_true', 'persistence_pred']].to_numpy(float)
        if reference is None:
            reference = current
        elif not np.allclose(current, reference, rtol=1e-12, atol=1e-14):
            raise ValueError('Different common targets/persistence')
    for _, part in frame.groupby(['history_days', 'outcome_day'], sort=False):
        if not np.allclose(part.mase_train_scale, part.mase_train_scale.iloc[0], rtol=1e-12, atol=1e-14):
            raise ValueError('Inconsistent historical MASE scales')
    return frame.sort_values(KEY + ['outcome_day']).reset_index(drop=True)


def metric_values(frame, prediction='y_pred'):
    target, pred = frame.y_true.to_numpy(float), frame[prediction].to_numpy(float)
    err = target - pred
    mse = float(np.mean(err ** 2))
    sst = float(np.sum((target - target.mean()) ** 2))
    positive = target > 0
    scale = frame.mase_train_scale.to_numpy(float)
    return {'n_dates': len(frame), 'rmse': math.sqrt(mse), 'mae': float(np.mean(abs(err))), 'mse': mse,
            'r2': 1 - float(np.sum(err ** 2)) / sst if sst > 0 else np.nan,
            'mape': float(100 * np.mean(abs(err[positive] / target[positive]))) if positive.any() else np.nan,
            'mape_coverage': float(positive.mean()), 'mase': float(np.mean(abs(err) / scale)),
            'mase_coverage': 1., 'residual_mean': float(err.mean()),
            'residual_sample_sd': float(err.std(ddof=1))}


def metric_tables(predictions):
    annual, monthly = [], []
    for identity, part in predictions.groupby(KEY, sort=True):
        base = dict(zip(KEY, identity))
        annual.append({**base, 'stochastic_training': True, **metric_values(part)})
        for month, sub in part.groupby('fold_id'):
            monthly.append({**base, 'fold_id': month, 'stochastic_training': True, **metric_values(sub)})
    for d, group in predictions.groupby('history_days'):
        part = group.drop_duplicates('outcome_day').sort_values('outcome_day')
        base = {'history_days': int(d), 'model': 'Persistence', 'seed': -1, 'stochastic_training': False}
        annual.append({**base, **metric_values(part, 'persistence_pred')})
        for month, sub in part.groupby('fold_id'):
            monthly.append({**base, 'fold_id': month, **metric_values(sub, 'persistence_pred')})
    return pd.DataFrame(annual), pd.DataFrame(monthly)


def summarize(frame, *, monthly=False):
    keys = ['history_days', 'model'] + (['fold_id'] if monthly else [])
    rows = []
    for ident, part in frame.groupby(keys, sort=True):
        identity = dict(zip(keys, ident))
        for metric in METRICS:
            values = part[metric].to_numpy(float)
            finite = values[np.isfinite(values)]
            n = len(finite)
            avg = float(finite.mean()) if n else np.nan
            sd = float(finite.std(ddof=1)) if n > 1 else np.nan
            half = stats.t.ppf(.975, n - 1) * sd / math.sqrt(n) if n > 1 else np.nan
            rows.append({**identity, 'metric': metric, 'n_total': len(part), 'n_defined': n,
                         'n_dates': int(part.n_dates.iloc[0]), 'mean': avg, 'sample_sd': sd,
                         'ci95_lower': avg - half, 'ci95_upper': avg + half,
                         'stochastic_training': bool(part.stochastic_training.iloc[0]),
                         'ci_method': 'marginal Student t across training seeds, conditional on fixed dates' if n > 1 else 'deterministic reference, no seed confidence interval',
                         'scope': SCOPE})
    return pd.DataFrame(rows)


def holm(pvalues):
    values = np.asarray(pvalues, float)
    work = np.where(np.isfinite(values), values, 1.)
    order = np.argsort(work, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate((len(work) - np.arange(len(work))) * work[order]))
    result = np.empty(len(work))
    result[order] = adjusted
    result[~np.isfinite(values)] = np.nan
    return result


def pair_table(annual, phase, histories):
    family = len(histories) * 3
    rows = []
    for d in histories:
        for pair_index, (a, b) in enumerate(PAIRS):
            left = annual.loc[annual.history_days.eq(d) & annual.model.eq(a), ['seed', 'rmse']]
            right = annual.loc[annual.history_days.eq(d) & annual.model.eq(b), ['seed', 'rmse']]
            paired = left.merge(right, on='seed', suffixes=('_a', '_b'), validate='one_to_one').sort_values('seed')
            if paired.seed.tolist() != list(SEEDS):
                raise ValueError('Ten matched seeds are required for each trio contrast')
            differences = paired.rmse_a.to_numpy() - paired.rmse_b.to_numpy()
            n, avg, sd = len(differences), float(differences.mean()), float(differences.std(ddof=1))
            se = sd / math.sqrt(n)
            statistic = avg / se if se > 0 else (0. if avg == 0 else math.copysign(np.inf, avg))
            p = float(2 * stats.t.sf(abs(statistic), n - 1))
            half = stats.t.ppf(.975, n - 1) * se
            half_family = stats.t.ppf(1 - .05 / (2 * family), n - 1) * se
            rng = np.random.default_rng(20261105 + (0 if phase == 'cv' else 10000) + d * 10 + pair_index)
            boot = differences[rng.integers(0, n, size=(4999, n))]
            boot_sd = boot.std(axis=1, ddof=1)
            effect_boot = boot.mean(axis=1)[boot_sd > 0] / boot_sd[boot_sd > 0]
            lower_effect, upper_effect = np.quantile(effect_boot, [.025, .975]) if len(effect_boot) else (np.nan, np.nan)
            wins_a = int(np.sum(differences < -1e-12))
            wins_b = int(np.sum(differences > 1e-12))
            effective = wins_a + wins_b
            win_lower = float(stats.beta.ppf(.025, wins_a, effective - wins_a + 1)) if wins_a else 0.
            win_upper = float(stats.beta.ppf(.975, wins_a + 1, effective - wins_a)) if wins_a < effective else 1.
            rows.append({'phase': phase, 'history_days': d, 'model_a': a, 'model_b': b,
                         'n_paired': n, 'mean_a': float(paired.rmse_a.mean()), 'mean_b': float(paired.rmse_b.mean()),
                         'difference_a_minus_b': avg, 'ci95_lower': avg - half, 'ci95_upper': avg + half,
                         'ci_family_lower': avg - half_family, 'ci_family_upper': avg + half_family,
                         'mean_gain_b_minus_a': -avg, 'gain_ci95_lower': -avg - half, 'gain_ci95_upper': -avg + half,
                         'gain_ci_family_lower': -avg - half_family, 'gain_ci_family_upper': -avg + half_family,
                         't_statistic': statistic, 'degrees_of_freedom': n - 1,
                         'p_value_unadjusted': p, 'cohen_d_z': avg / sd if sd > 0 else np.nan,
                         'effect_bootstrap_ci95_lower': lower_effect, 'effect_bootstrap_ci95_upper': upper_effect,
                         'effect_bootstrap_n': 4999, 'effect_bootstrap_defined': len(effect_boot),
                         'wins_model_a': wins_a, 'wins_model_b': wins_b, 'ties': n - effective,
                         'n_pairs_effective': effective, 'win_fraction_model_a': wins_a / effective if effective else np.nan,
                         'win_fraction_ci95_lower': win_lower if effective else np.nan,
                         'win_fraction_ci95_upper': win_upper if effective else np.nan,
                         'win_fraction_method': 'Clopper-Pearson exact binomial interval excluding ties |RMSEdiff|<=1e-12; RNG-conditional, not future-market win probability',
                         'shapiro_p_diagnostic': float(stats.shapiro(differences).pvalue) if sd > 0 else np.nan,
                         'normality_changes_primary_test': False, 'family_size': family,
                         'family_confidence_level': 1 - .05 / family,
                         'posthoc_after_test': True, 'scope': SCOPE})
    result = pd.DataFrame(rows)
    result[f'p_value_holm{family}'] = holm(result.p_value_unadjusted)
    result['p_value_adjusted'] = result[f'p_value_holm{family}']
    result['correction'] = f'Holm{family}'
    result['significant_adjusted'] = result.p_value_adjusted.lt(.05)
    return result


def friedman_table(annual):
    rows = []
    for phase in ('cv', 'test'):
        arrays = [annual[phase].loc[annual[phase].history_days.eq(7) & annual[phase].model.eq(model)].sort_values('seed').rmse.to_numpy()
                  for model in MODELS]
        if any(len(values) != 10 for values in arrays):
            raise ValueError('Friedman requires the same ten seed blocks')
        if np.allclose(arrays, arrays[0], rtol=0, atol=1e-15):
            statistic, p = np.nan, np.nan
            status = 'all_models_tied_undefined'
        else:
            result = stats.friedmanchisquare(*arrays)
            statistic, p, status = float(result.statistic), float(result.pvalue), 'computed_chi_square_approximation'
        rows.append({'phase': phase, 'history_days': 7, 'n_seed_blocks': 10, 'n_models': 3,
                     'statistic': statistic, 'p_value_chi2_approx': p, 'status': status,
                     'degrees_of_freedom': 2, 'posthoc_after_test': True,
                     'scope': 'Seed-block omnibus comparison conditional on known dates; chi-square approximation with only ten blocks, not independent market evidence.'})
    frame = pd.DataFrame(rows)
    frame['p_value_holm2_sensitivity'] = holm(frame.p_value_chi2_approx)
    return frame


def source_record(experiment, path):
    path = Path(path)
    try:
        identifier = str(path.relative_to(experiment))
    except ValueError:
        identifier = path.name
    return {'source_id': identifier, 'sha256': sha(path)}


def validate_sources(experiment):
    cv = experiment / 'protocol.json'
    before = read_json(experiment / 'freeze_before_training.json')
    if sha(cv) != before['protocol_sha256']:
        raise ValueError('The original CV protocol has changed')
    freeze = read_json(experiment / 'final_evaluation/freeze.json')
    if sha(experiment / 'final_evaluation/protocol.json') != freeze['protocol_sha256']:
        raise ValueError('The original final protocol has changed')
    for family in ('code_sha256', 'source_cv_sha256', 'source_epoch_configuration_sha256'):
        for name, expected in freeze[family].items():
            if sha(name) != expected:
                raise ValueError('A signed source changed: ' + name)
    if not freeze.get('old_test_already_observed') or freeze.get('test_selection_allowed') is not False:
        raise ValueError('Do not relabel the observed test or select with test')
    return read_json(cv), freeze


def catalog(experiment, cv_runs, final_runs):
    cv_rows, final_rows, learning = [], [], []
    for row in cv_runs.to_dict('records'):
        directory = experiment / 'results/runs' / row['run_id']
        for month in range(1, 13):
            part = directory / f'2024-{month:02d}'
            path = part / 'configuration.json'
            config = read_json(path)
            training = config['training']
            cv_rows.append({'model': row['model'], 'history_days': int(row['history_days']), 'seed': int(row['seed']),
                            'fold_id': f'2024-{month:02d}', 'width': config['width'], 'learning_rate': config['learning_rate'],
                            'optimizer': training['optimizer'], 'weight_decay': training['weight_decay'],
                            'dropout': training['dropout'], 'batch_size': training['batch_size'],
                            'max_epochs': training['max_epochs'], 'patience': training['patience'], 'min_delta': training['min_delta'],
                            'gradient_clip_norm': training['gradient_clip_norm'], 'best_epoch': config['best_epoch'],
                            'initialization_seed_inner': config['inner_initialization_seed'],
                            'initialization_seed_refit': config['refit_initialization_seed'],
                            'parameters': int(config['architecture']['parameters']['trainable']),
                            'configuration_sha256': sha(path), 'code_signature': config['code_signature'],
                            'fold_manifest_sha256': config['fold_manifest_sha256'], 'test_data_used_for_fit': False})
            # Aggregate loss curves only; no scaler values or observations leave the source.
            for stage, filename in [('inner', 'history_inner.csv'), ('refit', 'history_refit.csv')]:
                curve = pd.read_csv(part / filename)
                for metric in ('train_rmse_original', 'validation_rmse_original'):
                    if metric in curve:
                        values = pd.to_numeric(curve[metric], errors='coerce')
                        for epoch, value in zip(curve.epoch, values):
                            if np.isfinite(value):
                                learning.append({'history_days': int(row['history_days']), 'model': row['model'], 'stage': stage,
                                                 'epoch': int(epoch), 'metric': metric, 'value': float(value)})
    for row in final_runs.to_dict('records'):
        path = experiment / 'final_evaluation/final_fit' / row['run_id'] / 'configuration.json'
        config = read_json(path)
        final_rows.append({'model': row['model'], 'history_days': int(row['history_days']), 'seed': int(row['seed']),
                           'width': config['width'], 'learning_rate': config['learning_rate'],
                           'epochs_fixed_before_test': config['epochs'], 'parameters': int(config['parameters']),
                           'training_samples': int(config['train_samples']), 'training_first_day': config['train_first_day_utc'],
                           'training_last_day': config['train_last_day_utc'], 'embargo_days': config['embargo_days'],
                           'initialization_seed': config['initialization_seed'], 'configuration_sha256': sha(path),
                           'freeze_sha256': config['freeze_sha256'], 'historical_model_reused': bool(config.get('historical_model_reused', False)),
                           'new_final_fit': int(config.get('incremental_model_fits', 1)),
                           'test_data_used_for_fit': False})
    if len(cv_rows) != 1440 or len(final_rows) != 120:
        raise ValueError('Configuration catalogs must have exactly 1440 CV and 120 final rows')
    curve_frame = pd.DataFrame(learning)
    curves = curve_frame.groupby(['history_days', 'model', 'stage', 'epoch', 'metric'], as_index=False).agg(
        n_fold_seed_runs=('value', 'size'), mean=('value', 'mean'), sample_sd=('value', 'std'))
    curves['scope'] = 'Aggregate over available monthly/seed runs at each epoch; late epochs retain only runs not early-stopped.'
    return pd.DataFrame(cv_rows), pd.DataFrame(final_rows), curves


def cost_summary(runs, phase):
    result = runs.groupby(['history_days', 'model'], as_index=False).agg(
        n_runs=('seed', 'size'), parameters=('parameters', 'first'),
        training_seconds_mean=('training_seconds', 'mean'), training_seconds_sd=('training_seconds', 'std'),
        inference_seconds_mean=('inference_seconds', 'mean'), inference_seconds_sd=('inference_seconds', 'std'))
    result['phase'] = phase
    result['timing_scope'] = 'Recorded wall time under documented CPU concurrency; not isolated architecture speed.'
    if phase == 'test':
        result['LSTM_final_models_and_forecasts_reused'] = result.model.eq('LSTM')
        result['LSTM_incremental_fit_and_inference_cost'] = np.where(result.model.eq('LSTM'), 0., np.nan)
    return result


def residual_histograms(predictions):
    # Histogram counts over seed/date errors are aggregate visualization, not
    # independent samples for inference. No residual vector is exported.
    rows = []
    values = np.concatenate([(frame.loc[frame.history_days.eq(7), 'y_true'] - frame.loc[frame.history_days.eq(7), 'y_pred']).to_numpy()
                             for frame in predictions.values()])
    edge = float(np.max(abs(values)))
    bins = np.linspace(-edge, edge, 61)
    for phase, frame in predictions.items():
        for model in MODELS:
            part = frame.loc[frame.history_days.eq(7) & frame.model.eq(model)]
            counts, _ = np.histogram(part.y_true - part.y_pred, bins=bins)
            for index, count in enumerate(counts):
                rows.append({'phase': phase, 'model': model, 'history_days': 7, 'bin_left': bins[index],
                             'bin_right': bins[index + 1], 'count_seed_date_errors': int(count),
                             'scope': 'Descriptive histogram; seed/date errors are dependent, not independent observations.'})
    return pd.DataFrame(rows)


def plots(staging, summaries, annual, monthly, curves, histograms, predictions):
    colors = {'GRU': '#087f5b', 'LSTM': '#1971c2', 'TCN': '#9c36b5', 'Persistence': '#495057'}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, phase in enumerate(('cv', 'test')):
        for col, metric in enumerate(('rmse', 'r2')):
            ax = axes[row, col]
            for model in (*MODELS, 'Persistence'):
                part = summaries[phase].loc[summaries[phase].model.eq(model) & summaries[phase].metric.eq(metric)].sort_values('history_days')
                ax.errorbar(part.history_days, part['mean'], yerr=part.sample_sd.fillna(0),
                            color=colors[model], fmt='o--' if model == 'Persistence' else 'o-', capsize=3, label=model)
            ax.set(title=f'{phase.upper()}: {metric.upper()}', xlabel='Días de entrada', xticks=HISTORIES)
            ax.grid(alpha=.25)
    axes[0, 0].legend()
    figure.suptitle('Media ± DE entre semillas; test histórico ya observado')
    figure.savefig(staging / 'metrics_by_history.png', dpi=165); plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        for metric, label in [('train_rmse_original', 'Entrenamiento'), ('validation_rmse_original', 'Validación interna')]:
            part = curves.loc[curves.history_days.eq(7) & curves.model.eq(model) & curves.stage.eq('inner') & curves.metric.eq(metric)]
            ax.plot(part.epoch, part['mean'], label=label)
        ax.set(title=model, xlabel='Época', ylabel='RMSE en escala original'); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    figure.suptitle('D7: curvas medias de CV; número de corridas disponibles variable por early stopping')
    figure.savefig(staging / 'learning_curves_D7.png', dpi=165); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, phase in zip(axes, ('cv', 'test')):
        for model in MODELS:
            part = histograms.loc[histograms.phase.eq(phase) & histograms.model.eq(model)]
            ax.stairs(part.count_seed_date_errors, np.r_[part.bin_left.to_numpy(), part.bin_right.iloc[-1]],
                      color=colors[model], label=model)
        ax.set(title=phase.upper(), xlabel='Residuo: observado − predicho', ylabel='Conteo descriptivo semilla × fecha'); ax.grid(alpha=.2)
    axes[0].legend()
    figure.savefig(staging / 'residual_histograms_D7.png', dpi=165); plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    for ax, phase in zip(axes, ('cv', 'test')):
        for model in (*MODELS, 'Persistence'):
            part = monthly[phase].loc[monthly[phase].history_days.eq(7) & monthly[phase].model.eq(model) & monthly[phase].metric.eq('rmse')]
            ax.plot(pd.to_datetime(part.fold_id + '-01'), part['mean'], 'o-', color=colors[model], label=model, markersize=3)
        ax.set(title=phase.upper(), ylabel='RMSE mensual'); ax.grid(alpha=.2)
    axes[0].legend(ncol=4, fontsize=8)
    figure.suptitle('D7: intervalos mensuales descriptivos; el test no tiene reajustes')
    figure.savefig(staging / 'monthly_rmse_D7.png', dpi=165); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, phase in zip(axes, ('cv', 'test')):
        for model in MODELS:
            part = annual[phase].loc[annual[phase].history_days.eq(7) & annual[phase].model.eq(model)].sort_values('seed')
            ax.plot(part.seed, part.rmse, 'o-', color=colors[model], label=model)
        ax.set(title=phase.upper(), xlabel='Semilla', ylabel='RMSE agrupado'); ax.grid(alpha=.2)
    axes[0].legend()
    figure.savefig(staging / 'seed_rmse_D7.png', dpi=165); plt.close(figure)
    figure, ax = plt.subplots(figsize=(12, 4.2), constrained_layout=True)
    for index, model in enumerate(MODELS):
        part = predictions['test'].loc[predictions['test'].history_days.eq(7) &
                                      predictions['test'].model.eq(model) & predictions['test'].seed.eq(4101)].sort_values('outcome_day')
        dates = pd.to_datetime(part.outcome_day)
        if index == 0:
            ax.plot(dates, part.y_true, color='#212529', label='Observado', linewidth=1.2)
        ax.plot(dates, part.y_pred, color=colors[model], label=model, linewidth=.8, alpha=.8)
    ax.set(xlabel='Fecha UTC', ylabel='Volatilidad σ7, sin anualizar',
           title='Test ya observado: D7, semilla 4101 fija como ilustración; selección no basada en su error')
    ax.grid(alpha=.2); ax.legend(ncol=4)
    figure.savefig(staging / 'predictions_D7.png', dpi=165); plt.close(figure)


def build(experiment=EXPERIMENT, root=ROOT):
    experiment, root = Path(experiment).resolve(), Path(root).resolve()
    plan, freeze = validate_sources(experiment)
    source_paths = [experiment / p for p in ['protocol.json', 'freeze_before_training.json',
        'results/runs.csv', 'results/predictions.csv', 'results/fold_metrics.csv', 'results/execution_summary.json',
        'results/runtime.json', 'implementation/architecture_examples.json',
        'final_evaluation/protocol.json', 'final_evaluation/freeze.json', 'final_evaluation/final_models_manifest.json',
        'final_evaluation/final_fit_summary.csv', 'final_evaluation/test/runs.csv',
        'final_evaluation/test/predictions.csv', 'final_evaluation/test/fold_metrics.csv',
        'final_evaluation/test/execution_summary.json']]
    before = {str(path): sha(path) for path in source_paths}
    data, run_inputs, annual, monthly, summaries, diagnostics = {}, {}, {}, {}, {}, {}
    for phase, relative in [('cv', 'results'), ('test', 'final_evaluation/test')]:
        data[phase] = validate_predictions(pd.read_csv(experiment / relative / 'predictions.csv'), phase)
        run_inputs[phase] = pd.read_csv(experiment / relative / 'runs.csv')
        annual[phase], by_month = metric_tables(data[phase])
        summaries[phase], monthly[phase] = summarize(annual[phase]), summarize(by_month, monthly=True)
        merged = annual[phase].loc[annual[phase].model.isin(MODELS)].merge(run_inputs[phase], on=KEY, suffixes=('_calc', '_stored'), validate='one_to_one')
        if len(merged) != 120:
            raise ValueError('Run summaries do not match all 120 trajectories')
        for metric in METRICS:
            if not np.allclose(merged[metric + '_calc'], merged[metric + '_stored'], rtol=1e-9, atol=1e-12, equal_nan=True):
                raise ValueError('Stored and independently recomputed metric differ: ' + phase + '/' + metric)
        diagnostics[phase] = diagnose_phase(data[phase], phase)
    baseline = freeze['selection']['baseline']
    if baseline['model'] != 'GRU' or baseline['history_days'] != 7:
        raise ValueError('Expected already-frozen GRU/D7 CV baseline; do not reselect using test')
    gru_cv = float(summaries['cv'].loc[summaries['cv'].history_days.eq(7) & summaries['cv'].model.eq('GRU') & summaries['cv'].metric.eq('rmse'), 'mean'].iloc[0])
    if not np.isclose(gru_cv, baseline['cv_rmse_mean'], rtol=1e-12):
        raise ValueError('The recomputed CV baseline differs from its pre-test freeze')
    future_delta = .05 * gru_cv
    catalog_cv, catalog_final, curves = catalog(experiment, run_inputs['cv'], pd.read_csv(experiment / 'final_evaluation/final_fit_summary.csv'))
    source_specs = read_json(experiment / 'implementation/architecture_examples.json')
    specs = {'models': sorted(source_specs['models'], key=lambda row: MODELS.index(row['family'])),
             'shape': 'batch × D × 5 OHLCV channels, oldest first', 'histories': list(HISTORIES),
             'target': 'populationSD(ddof0) of seven daily UTC Close log returns, no annualization',
             'forecast_horizon_days': 1, 'preprocessing': 'relative-window logOHLC plus log1pVolume; training-only channel standardization and positive mean target scaling',
             'reconstruction_scope': 'Compact project adaptations; not exact replications of every published architecture',
             'parameter_expressions': {'GRU': '3*(F*W+W^2+2*W)+(W+1)',
                                      'LSTM': '4*(F*W+W^2+2*W)+(W+1)',
                                      'TCN': '(K*F*W+W)+(K*W^2+W)+(F*W+W)+(B-1)*2*(K*W^2+W)+(W+1); B=3,K=3'},
             'training': plan['training'], 'pilot': {'new_prospective_pilot_for_GRU_TCN': False,
                'seed_count': 10, 'reason': 'Fixed resource-bounded screening budget, not pilot-derived 80% power'}}
    for model in specs['models']:
        if model['family'] == 'GRU':
            model.pop('gates', None)
            model['control_gates'] = ['reset', 'update']
            model['candidate_projection'] = 'new'
            model['gate_metadata_note'] = 'The candidate state n/new is not a third control gate; the signed source metadata is preserved privately.'
    recorded_runtime = read_json(experiment / 'results/runtime.json')
    runtime_fields = ('at_utc', 'python', 'platform', 'versions', 'device', 'cuda_available', 'cpu_count',
                      'cpu_affinity_count', 'cgroup_cpu_max', 'cgroup_memory_max', 'torch_threads_per_worker',
                      'torch_interop_threads', 'deterministic_algorithms', 'mkldnn_enabled', 'timing_scope')
    public_runtime = {'recorded_runtime': {key: recorded_runtime[key] for key in runtime_fields if key in recorded_runtime},
                      'source_id': 'results/runtime.json', 'source_sha256': sha(experiment / 'results/runtime.json'),
                      'scope': 'Curated execution runtime metadata only; no environment-variable dumps, credentials or observations',
                      'remeasured_during_publication': False}
    yearly_summaries = []
    test_years = pd.to_datetime(data['test'].outcome_day).dt.year
    for year in (2025, 2026):
        yearly_annual, _ = metric_tables(data['test'].loc[test_years.eq(year)])
        yearly_summary = summarize(yearly_annual)
        yearly_summary['year'] = year
        yearly_summary['calendar_scope'] = '2025 full year, 365 dates' if year == 2025 else '2026 January-August only, 243 dates'
        yearly_summary['aggregation_scope'] = 'Descriptive annual segment; do not average segment R2 or RMSE to recover the pooled test metric'
        yearly_summaries.append(yearly_summary)
    yearly_summary = pd.concat(yearly_summaries, ignore_index=True)
    if len(yearly_summary) != 192 or set(yearly_summary.loc[yearly_summary.year.eq(2025), 'n_dates']) != {365} or set(yearly_summary.loc[yearly_summary.year.eq(2026), 'n_dates']) != {243}:
        raise ValueError('Expected descriptive annual table: 192 rows, 365/243 dates')
    histograms = residual_histograms(data)
    target = root / 'results' / PUBLIC_NAME
    static = root / 'book/_static' / PUBLIC_NAME
    target.mkdir(parents=True, exist_ok=True)
    # Keep one explicit date for this post-test analysis declaration across rebuilds.
    existing_plan = target / 'public_analysis_plan.json'
    declared_at = read_json(existing_plan)['declared_at_utc'] if existing_plan.is_file() else datetime.now(timezone.utc).isoformat()
    posthoc_plan = {'declared_at_utc': declared_at, 'timing': 'AFTER historical test scoring and inspection',
        'publication_authorized_after_local_comparison': True, 'models': list(MODELS), 'histories': list(HISTORIES),
        'seeds': list(SEEDS), 'baseline': 'GRU/D7, chosen using CV before its new old-test evaluation',
        'primary_trio_report': {'history_days': 7, 'pairs': [list(p) for p in PAIRS], 'comparisons_per_phase': 3,
            'test': 'two-sided paired Student t on per-seed pooled RMSE', 'correction': 'Holm3 separately per phase',
            'intervals': '95% marginal and Bonferroni3 family intervals', 'effect': 'paired Cohen dz with4999 paired-seed bootstrap sensitivity'},
        'all_histories_sensitivity': {'comparisons_per_phase': 12, 'correction': 'Holm12 separately per phase',
            'intervals': '95% marginal and Bonferroni12 family intervals'},
        'normality': 'Shapiro diagnostic only; does not select a different test',
        'diagnostics': {'BDS': 'h1 monthly residuals, m2, epsilon1.5sampleSD, exploratory short-block iid diagnostic',
                        'LjungBox': 'h1 monthly residuals, lags1/5, model_df0; not ARMA df estimated for networks',
                        'DM': 'D7 three pairs, daily squared losses averaged over seeds; HAC Bartlett lag6, HLN h1, Holm3'},
        'margins': 'Prior frozen absolute delta retained as historical context; new5%GRU-CV margin declared now for future studies only',
        'fixed_seed_illustration': {'file': 'predictions_D7.png', 'seed': 4101, 'dates': 608,
            'format': 'PNG illustration of observed and predicted sigma7, not a machine-readable daily table',
            'seed_chosen_by_error': False, 'table_vectors_exported': False},
        'independent_confirmation_claimed': False, 'no_new_fits_or_checkpoint_inferences': True, 'scope': SCOPE}
    future = {'declared_at_utc': declared_at, 'timing': 'POSTTEST declaration for FUTURE advanced-model studies',
              'purpose': 'Academic operational relevance and sensitivity; no financial utility assertion',
              'new_baseline': {'model': 'GRU', 'history_days': 7, 'cv_mean_rmse': gru_cv},
              'relative_margin': .05, 'absolute_delta': future_delta,
              'sensitivity': [{'relative_margin': f, 'absolute_delta': f * gru_cv} for f in [.025, .05, .10]],
              'historical_original_delta': {'absolute': .0003024536888539928,
                  'scope': 'Original comparison protocol; historical reading only, not a pre-test margin for this new trio'},
              'future_requirements': ['Declare primary metric, new-study data split, contrasts and independent unit before future evaluation.',
                  'Run a prospective pilot for the future candidate and size the paired comparison.',
                  'Compare with both the frozen neural GRU baseline and persistence on the same future cohort.',
                  'Separate distinguishing zero, practical point gain and interval-supported practical gain.'],
              'applied_retroactively_to_current_test': False}
    source_manifest = {'sources': [source_record(experiment, p) for p in source_paths],
                       'source_protocols_code_and_results_modified': False, 'weights_exported': False,
                       'scalers_exported': False, 'raw_market_files_exported': False,
                       'machine_readable_daily_series_exported': False,
                       'individual_forecast_table_vectors_exported': False,
                       'individual_residual_table_vectors_exported': False,
                       'fixed_seed_illustration': posthoc_plan['fixed_seed_illustration'],
                       'builder': 'scripts/build_gru_lstm_tcn_evidence.py',
                       'builder_sha256': sha(__file__), 'diagnostics_sha256': sha(Path(__file__).with_name('gru_lstm_tcn_diagnostics.py'))}
    protocols = {'postpublication_analysis': posthoc_plan, 'signed_originals_preserved': True,
        'cv': {'registered_at_utc': plan['registered_at_utc'], 'protocol_sha256': sha(experiment / 'protocol.json'),
               'freeze_before_training_sha256': sha(experiment / 'freeze_before_training.json'),
               'calendar': '2023 initial fit; 2024 monthly blocked forward chaining', 'trajectories': 120,
               'inner_and_refit_fits': 2880, 'prediction_rows': 43920,
               'selection_and_comparisons_timing': 'Original recipes/24 historical-reference contrasts frozen before new runs; new within-trio analyses are post-test'},
        'final': {'closed_at_utc': freeze['closed_at_utc'], 'configuration_freeze_sha256': sha(experiment / 'final_evaluation/freeze.json'),
                  'protocol_sha256': freeze['protocol_sha256'], 'calendar': '2025-01-01..2026-08-31, already observed',
                  'new_model_fits': 80, 'LSTM_models_reused': 40, 'new_checkpoint_forward_calls': 80,
                  'LSTM_prediction_runs_reused': 40, 'prediction_rows': 72960,
                  'monthly_reporting_intervals': 20, 'test_period_refits': 0},
        'new_trio_analysis_predeclared_before_test': False, 'future_margin_predeclared_before_current_test': False,
        'prospective_pilot_for_new_GRU_TCN': False, 'scope': SCOPE}
    selected = {'selected_before_new_test_evaluation': True, 'selection_source': 'immutable final configuration freeze',
                'selection_rule': freeze['selection']['rule'], 'baseline': baseline,
                'by_model': freeze['selection']['by_model'], 'all_cv_settings': freeze['selection']['all_settings'],
                'test_selection_used': False, 'test_already_observed': True}
    with tempfile.TemporaryDirectory(prefix='gru-lstm-tcn-public-') as temporary:
        staging = Path(temporary)
        for phase in ('cv', 'test'):
            annual[phase].to_csv(staging / f'run_metrics_{phase}.csv', index=False)
            summaries[phase].to_csv(staging / f'metric_summary_{phase}.csv', index=False)
            monthly[phase].to_csv(staging / f'monthly_summary_{phase}.csv', index=False)
            pair_table(annual[phase], phase, (7,)).to_csv(staging / f'within_trio_pairs_{phase}.csv', index=False)
            pair_table(annual[phase], phase, HISTORIES).to_csv(staging / f'within_trio_pairs_allD_{phase}.csv', index=False)
            for name, table in diagnostics[phase].items():
                table.to_csv(staging / f'{name}_{phase}.csv', index=False)
            cost_summary(run_inputs[phase], phase).to_csv(staging / f'cost_summary_{phase}.csv', index=False)
        catalog_cv.to_csv(staging / 'configuration_catalog_cv.csv', index=False)
        catalog_final.to_csv(staging / 'configuration_catalog_final.csv', index=False)
        curves.to_csv(staging / 'learning_curve_summary.csv', index=False)
        histograms.to_csv(staging / 'residual_histogram_D7.csv', index=False)
        yearly_summary.to_csv(staging / 'annual_metric_summary_test.csv', index=False)
        friedman_table(annual).to_csv(staging / 'friedman_D7.csv', index=False)
        variance = pd.concat([summaries[p].assign(phase=p) for p in ('cv', 'test')])
        variance = variance.loc[variance.model.isin(MODELS)].copy()
        variance['sample_seed_variance'] = variance.sample_sd ** 2
        variance.to_csv(staging / 'seed_variance.csv', index=False)
        for filename, content in [('model_specs.json', specs), ('source_manifest.json', source_manifest),
                                   ('protocols_manifest.json', protocols), ('selected_by_cv.json', selected),
                                   ('public_analysis_plan.json', posthoc_plan), ('future_margin.json', future),
                                   ('runtime.json', public_runtime)]:
            write_json(staging / filename, content)
        baseline_metrics = {phase: {row['metric']: {key: row[key] for key in ['mean', 'sample_sd', 'ci95_lower', 'ci95_upper', 'n_defined']}
                              for row in summaries[phase].loc[summaries[phase].history_days.eq(7) & summaries[phase].model.eq('GRU')].to_dict('records')}
                            for phase in ('cv', 'test')}
        summary = {'public_analysis_declared_at_utc': declared_at, 'models': list(MODELS), 'histories': list(HISTORIES),
                   'seeds': list(SEEDS), 'seeds_per_model_history': 10,
                   'parameters': {'GRU': 3777, 'LSTM': 5025, 'TCN': 4289},
                   'cv': {'trajectories': 120, 'dates': 366, 'monthly_training_blocks': 12, 'fits': 2880, 'predictions': 43920},
                   'test': {'trajectories': 120, 'dates': 608, 'monthly_reporting_intervals': 20,
                            'new_final_fits': 80, 'historical_LSTM_reused': 40, 'new_inferences': 80,
                            'predictions': 72960, 'test_period_refits': 0, 'already_observed': True},
                   'baseline_selected_by_cv': baseline, 'baseline_metrics': baseline_metrics,
                   'historical_margin_absolute': .0003024536888539928,
                   'future_margin_absolute_posttest_declaration': future_delta,
                   'new_prospective_pilot_exists': False, 'number_of_seeds_pilot_derived': False,
                   'trio_pairwise_analysis_posthoc_after_test': True, 'independent_confirmation_claimed': False,
                   'scope': SCOPE, 'new_fits_or_checkpoint_inferences_by_this_builder': 0}
        write_json(staging / 'summary.json', summary)
        plots(staging, summaries, annual, monthly, curves, histograms, data)
        write_reports(staging, summary, posthoc_plan, future)
        actual = {p.name for p in staging.iterdir()}
        if actual != PUBLIC_FILES:
            raise ValueError('Public allowlist mismatch: ' + str(actual ^ PUBLIC_FILES))
        # Recheck the input snapshot before making public copies.
        if any(sha(path) != expected for path, expected in before.items()):
            raise ValueError('An immutable source changed during aggregation')
        manifest = {'declared_at_utc': declared_at, 'scope': 'This builder allowlist only; no private run artifacts',
                    'file_sha256': {name: sha(staging / name) for name in sorted(PUBLIC_FILES)},
                    'excluded': ['weights', 'scalers', 'raw market files', 'machine-readable daily series', 'per-date forecast CSV vectors', 'per-date residual CSV vectors'],
                    'fixed_seed_illustration': posthoc_plan['fixed_seed_illustration'],
                    'individual_configuration_metadata_allowed': True}
        write_json(staging / 'public_allowlist_manifest.json', manifest)
        for destination in (target, static):
            destination.mkdir(parents=True, exist_ok=True)
            for name in sorted(PUBLIC_FILES | {'public_allowlist_manifest.json'}):
                shutil.copyfile(staging / name, destination / name)
    print(json.dumps(clean({'output': str(target), 'static': str(static), 'files': len(PUBLIC_FILES) + 1,
                            'baseline': baseline, 'future_delta': future_delta,
                            'source_files_changed': False, 'fits': 0, 'checkpoint_inferences': 0}), indent=2))
    return summary


def write_reports(staging, summary, posthoc, future):
    readme = '''# Evidencia agregada de GRU, LSTM y TCN

Este paquete sustituye la terna del contenido experimental principal. Reúne
métricas agregadas, estadísticos diagnósticos y metadatos de configuración.
Se excluyen pesos, escaladores, archivos de mercado y series diarias o vectores
individuales en formatos legibles por máquina. Se publica una ilustración PNG
(`predictions_D7.png`) con σ7 observado frente a las predicciones de la semilla
4101 fija durante las 608 fechas del test conocido; no se entrega su CSV diario
ni se seleccionó esa semilla por tener el menor error.

Los ajustes originales y sus firmas permanecen intactos. La selección GRU/D7
se realizó en CV antes de ejecutar los modelos nuevos sobre el test histórico,
que ya se conocía. Los contrastes entre los tres modelos de este paquete y los
diagnósticos adicionales se definieron DESPUÉS del test: son exploratorios.

`protocols_manifest.json` y `source_manifest.json` conservan fechas y hashes;
`public_analysis_plan.json` declara el alcance posterior. Las tablas de
contrastes usan diferencia RMSE(A)−RMSE(B): un valor negativo favorece A.
Se ofrece también la ganancia B−A y sus intervalos con nombres explícitos.

No existe piloto prospectivo para dimensionar las diez semillas de GRU/TCN.
El presupuesto fijo no acredita potencia 0,80. Las fechas, folds y semillas
no son repeticiones independientes de mercados.

El nuevo margen operativo de futuros trabajos está en `future_margin.json`.
Se declara ahora, después del test, y no se aplica retrospectivamente para
convertir los resultados actuales en confirmatorios. Persistencia conserva
su papel de referencia explícita con la misma cohorte y métricas.

Reproducción de esta agregación: `scripts/build_gru_lstm_tcn_evidence.py`
con las predicciones y configuraciones privadas originales identificadas por
SHA256. El programa no entrena ni vuelve a inferir ningún checkpoint.
'''
    (staging / 'README.md').write_text(readme)
    cv = summary['baseline_metrics']['cv']['rmse']
    test = summary['baseline_metrics']['test']['rmse']
    lines = ['# Síntesis del nuevo trío', '',
             f'Baseline neural fijado en CV: **GRU/D7**, {summary["parameters"]["GRU"]} parámetros.', '',
             f'RMSE CV: **{cv["mean"]:.10g} ± {cv["sample_sd"]:.10g}**. '
             f'RMSE del test histórico ya observado: **{test["mean"]:.10g} ± {test["sample_sd"]:.10g}**.', '',
             'Estas cifras son medias y desviaciones estándar entre diez semillas. Los IC del paquete '
             'son condicionales al calendario, no incertidumbre de generalización a futuros mercados.', '',
             'Los tres contrastes D7 por fase usan t pareada y Holm3; los doce contrastes de todas '
             'las entradas usan Holm12 como sensibilidad. Ambos análisis se formularon después del test. '
             'Se conserva cada comparación, incluso si no es significativa. Ausencia de significancia '
             'no significa equivalencia.', '',
             'BDS y Ljung–Box se calculan por semilla e intervalo. Sus frecuencias de rechazo y medias '
             'de p-valores son descriptivas: no son pruebas combinadas. Un rechazo BDS no identifica '
             'por sí solo no linealidad explotable. Los bloques mensuales cortos limitan su calibración.', '',
             'Diebold–Mariano compara pérdidas cuadradas diarias promediadas sobre semillas, con HAC '
             'Bartlett de seis rezagos por el solapamiento σ7 y corrección HLN para h=1. Su estimando '
             'es diferente del promedio de RMSE anual por semilla. Ninguno elimina la observación previa del test.', '',
             f'Margen académico futuro: **δ = {future["absolute_delta"]:.12g}**, el 5 % del RMSE CV de GRU/D7. '
             'Se declara después del test para nuevos estudios; no es un margen previo del experimento actual '
             'ni una afirmación de utilidad financiera. Se conserva el margen original únicamente como lectura histórica.', '',
             'Los futuros modelos deben contrastarse tanto con GRU como con persistencia y declarar '
             'sus propios datos, comparaciones y piloto antes de la evaluación. Las ventajas pueden '
             'diferir entre RMSE, MAE y MAPE; las tablas publican las seis métricas.', '',
             'El entrenamiento inicial conserva 2023; la selección usa forward chaining en 2024; '
             'el test comprende enero de 2025–agosto de 2026. Septiembre–octubre de 2026 permanece sin asignar.']
    (staging / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=EXPERIMENT)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    build(args.experiment, args.root)
