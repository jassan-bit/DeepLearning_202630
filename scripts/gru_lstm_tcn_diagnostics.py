"""Post-test exploratory residual diagnostics for the published neural trio.

These functions consume stored forecasts; they perform no training, inference,
file access or publication. BDS and Ljung--Box operate on short monthly blocks
per training seed. Diebold--Mariano compares daily squared losses averaged over
the ten training seeds, not squared errors of an ensemble prediction.
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds


MODELS = ('GRU', 'LSTM', 'TCN')
MONTHLY_MIN_N = 10
BDS_DIMENSION = 2
BDS_EPSILON_SD_MULTIPLIER = 1.5
LB_LAGS = (1, 5)
DM_HORIZON = 1
DM_HAC_LAGS = 6
ALPHA = .05
SCOPE = ('Post-test exploratory diagnostics on stored forecasts; training seeds '
         'share market dates and monthly blocks are not independent markets.')
BDS_LIMIT = ('Asymptotic iid diagnostic on n28--31 monthly residuals with '
             'overlapping sigma7 targets and estimated networks. Rejection does '
             'not uniquely identify exploitable nonlinearity or establish that '
             'a more complex model will improve forecasts.')
LB_LIMIT = ('Short-block autocorrelation diagnostic; model_df=0 does not claim '
            'that neural fitting consumes zero degrees of freedom. No standard '
            'ARMA parameter-count correction is available for these networks. '
            'Rejection does not itself establish forecast improvement.')
DM_LIMIT = ('Daily mean squared loss across training seeds; inference is '
            'exploratory and conditional on the known calendar. HAC lag6 '
            'addresses the seven-return overlapping volatility target but '
            'cannot guarantee all dependence is captured. This estimand differs '
            'from the mean of per-seed RMSEs.')


def holm_with_undefined(pvalues):
    """Retain all planned rows; undefined tests keep family slots as p=1."""
    values = np.asarray(pvalues, dtype=float)
    size = len(values)
    if not size:
        return values.copy()
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError('A defined p-value lies outside [0,1]')
    work = np.where(finite, values, 1.)
    order = np.argsort(work, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate((size - np.arange(size)) * work[order]))
    result = np.empty(size, dtype=float)
    result[order] = adjusted
    result[~finite] = np.nan
    return result


def diebold_mariano_hln(loss_a, loss_b, *, horizon=DM_HORIZON, hac_lags=DM_HAC_LAGS):
    """Two-sided HLN-adjusted DM with a Bartlett/Newey--West long-run variance.

    Sign convention: negative mean(loss_a-loss_b) favors model A. Covariances
    use denominator n, preserving the positive-semidefinite Bartlett estimator.
    HAC lag6 is fixed before this post-test diagnostic, not optimized by p-value.
    """
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('DM requires equal, finite one-dimensional loss series')
    if (isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1
            or isinstance(hac_lags, bool) or int(hac_lags) != hac_lags or hac_lags < horizon - 1):
        raise ValueError('Invalid forecast horizon or dependence lag')
    n = len(a)
    result = {'n_dates': n, 'horizon': int(horizon), 'hac_lags': int(hac_lags),
              'hac_kernel': 'Bartlett/Newey-West; covariances divided by n',
              'mean_squared_loss_a': float(a.mean()) if n else np.nan,
              'mean_squared_loss_b': float(b.mean()) if n else np.nan,
              'mean_loss_a_minus_b': float((a-b).mean()) if n else np.nan,
              'long_run_variance': np.nan, 'variance_of_mean_loss_difference': np.nan,
              'dm_raw_statistic': np.nan, 'hln_correction_factor': np.nan,
              'dm_hln_statistic': np.nan, 'degrees_of_freedom': n - 1 if n else np.nan,
              'p_value': np.nan, 'status': 'insufficient_dates', 'limitations': DM_LIMIT}
    if n < max(10, hac_lags + 2, horizon + 1):
        return result
    difference = a - b
    if np.ptp(difference) == 0:
        result['long_run_variance'] = 0.
        result['variance_of_mean_loss_difference'] = 0.
        result['status'] = 'degenerate_loss_difference'
        return result
    centered = difference - difference.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0
    for lag in range(1, hac_lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run += 2 * (1 - lag / (hac_lags + 1)) * covariance
    result['long_run_variance'] = long_run
    result['variance_of_mean_loss_difference'] = long_run / n
    if not np.isfinite(long_run) or long_run <= 0 or gamma0 <= 0:
        result['status'] = 'degenerate_loss_difference'
        return result
    factor_squared = (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    if factor_squared <= 0:
        result['status'] = 'invalid_hln_factor'
        return result
    raw = float(difference.mean() / math.sqrt(long_run / n))
    factor = math.sqrt(factor_squared)
    statistic = raw * factor
    p_value = float(2 * stats.t.sf(abs(statistic), n - 1))
    result.update(dm_raw_statistic=raw, hln_correction_factor=factor,
                  dm_hln_statistic=statistic, p_value=p_value, status='defined')
    return result


def _monthly_diagnostics(frame, phase):
    bds_rows, lb_rows = [], []
    keys = ['history_days', 'model', 'seed', 'fold_id']
    for identity, group in frame.groupby(keys, sort=True):
        group = group.sort_values('outcome_day')
        residuals = group.y_true.to_numpy(float) - group.y_pred.to_numpy(float)
        sd = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan
        base = {'phase': phase, **dict(zip(keys, identity)), 'horizon': 1,
                'n_residuals': len(residuals), 'residual_sample_sd': sd, 'scope': SCOPE}
        bds_row = {**base, 'embedding_dimension': BDS_DIMENSION,
                   'epsilon_sd_multiplier': BDS_EPSILON_SD_MULTIPLIER,
                   'epsilon': BDS_EPSILON_SD_MULTIPLIER * sd,
                   'statistic': np.nan, 'p_value': np.nan, 'status': 'insufficient_residuals',
                   'warnings': '', 'limitations': BDS_LIMIT}
        lb_base = {**base, 'model_df': 0, 'statistic': np.nan, 'p_value': np.nan,
                   'status': 'insufficient_residuals', 'warnings': '', 'limitations': LB_LIMIT}
        if len(residuals) >= MONTHLY_MIN_N:
            if not np.isfinite(sd) or sd <= np.finfo(float).eps * max(1., np.max(abs(residuals))):
                bds_row['status'] = 'degenerate_residual_variance'
                for lag in LB_LAGS:
                    lb_rows.append({**lb_base, 'lag': lag, 'status': 'degenerate_residual_variance'})
            else:
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        statistic, p_value = bds(residuals, max_dim=BDS_DIMENSION,
                                                 epsilon=BDS_EPSILON_SD_MULTIPLIER * sd)
                    statistic = float(np.asarray(statistic).ravel()[0])
                    p_value = float(np.asarray(p_value).ravel()[0])
                    defined = np.isfinite(statistic) and np.isfinite(p_value)
                    bds_row.update(statistic=statistic if defined else np.nan,
                                   p_value=p_value if defined else np.nan,
                                   status='defined' if defined else 'nonfinite_result',
                                   warnings='; '.join(str(item.message) for item in caught))
                except Exception as exc:
                    bds_row.update(status='error', warnings=f'{type(exc).__name__}: {exc}')
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        lb = acorr_ljungbox(residuals, lags=list(LB_LAGS), model_df=0, return_df=True)
                    for lag in LB_LAGS:
                        statistic, p_value = float(lb.loc[lag, 'lb_stat']), float(lb.loc[lag, 'lb_pvalue'])
                        defined = np.isfinite(statistic) and np.isfinite(p_value)
                        lb_rows.append({**lb_base, 'lag': lag, 'statistic': statistic if defined else np.nan,
                                        'p_value': p_value if defined else np.nan,
                                        'status': 'defined' if defined else 'nonfinite_result',
                                        'warnings': '; '.join(str(item.message) for item in caught)})
                except Exception as exc:
                    for lag in LB_LAGS:
                        lb_rows.append({**lb_base, 'lag': lag, 'status': 'error',
                                        'warnings': f'{type(exc).__name__}: {exc}'})
        else:
            for lag in LB_LAGS:
                lb_rows.append({**lb_base, 'lag': lag})
        bds_rows.append(bds_row)
    outputs = []
    for records in (bds_rows, lb_rows):
        result = pd.DataFrame(records)
        result['p_value_holm_phase'] = holm_with_undefined(result.p_value)
        result['multiplicity_family_size'] = len(result)
        result['reject_raw_005'] = result.p_value.lt(ALPHA)
        result['reject_holm_phase_005'] = result.p_value_holm_phase.lt(ALPHA)
        outputs.append(result)
    return outputs


def _dm_pairs(frame, phase):
    d7 = frame.loc[frame.history_days.eq(7)].copy()
    if set(d7.model) != set(MODELS):
        raise ValueError('All three D7 families are required for the three DM comparisons')
    d7['squared_loss'] = (d7.y_true - d7.y_pred) ** 2
    daily = d7.groupby(['model', 'outcome_day'], as_index=False).agg(
        mean_squared_loss=('squared_loss', 'mean'), seed_count=('seed', 'nunique'))
    counts = daily.groupby('model').seed_count.agg(['min', 'max'])
    if (counts['min'] != counts['max']).any() or counts['min'].nunique() != 1:
        raise ValueError('Daily seed support differs across DM comparisons')
    panel = daily.pivot(index='outcome_day', columns='model', values='mean_squared_loss').sort_index()
    if panel.isna().any().any():
        raise ValueError('All D7 models must share the same complete DM date cohort')
    rows = []
    for model_a, model_b in itertools.combinations(MODELS, 2):
        result = diebold_mariano_hln(panel[model_a].to_numpy(), panel[model_b].to_numpy())
        rows.append({'phase': phase, 'history_days': 7, 'model_a': model_a, 'model_b': model_b,
                     'seeds_per_day': int(counts['min'].iloc[0]),
                     'loss': 'Daily mean across seeds of squared original-unit residuals',
                     'direction': 'Negative loss_a-loss_b favors model_a; positive favors model_b',
                     **result, 'scope': SCOPE})
    result = pd.DataFrame(rows)
    result['p_value_holm3'] = holm_with_undefined(result.p_value)
    result['multiplicity_family_size'] = 3
    result['reject_holm3_005'] = result.p_value_holm3.lt(ALPHA)
    return result


def _summary(bds_frame, lb_frame):
    summaries = []
    for label, frame in [('BDS', bds_frame), ('Ljung-Box', lb_frame)]:
        keys = ['phase', 'history_days', 'model'] + (['lag'] if label == 'Ljung-Box' else [])
        for identity, group in frame.groupby(keys, sort=True):
            finite = group.p_value.notna() & np.isfinite(group.p_value)
            valid = group.loc[finite]
            summaries.append({**dict(zip(keys, identity)), 'diagnostic': label,
                              'planned_rows': len(group), 'defined_rows': len(valid),
                              'undefined_rows': int((~finite).sum()),
                              'p_value_mean_descriptive': float(valid.p_value.mean()) if len(valid) else np.nan,
                              'p_value_sample_sd_descriptive': float(valid.p_value.std(ddof=1)) if len(valid) > 1 else np.nan,
                              'raw_rejection_fraction_defined': float(valid.reject_raw_005.mean()) if len(valid) else np.nan,
                              'holm_rejection_fraction_defined': float(valid.reject_holm_phase_005.mean()) if len(valid) else np.nan,
                              'scope': 'Descriptive rates/mean p-values across dependent short blocks; not combined tests or evidence of iid.'})
    return pd.DataFrame(summaries)


def diagnose_phase(predictions, phase):
    """Return aggregate-only BDS, Ljung--Box, DM and diagnostic summary frames."""
    if phase not in ('cv', 'test'):
        raise ValueError('phase must be cv or test')
    required = {'history_days', 'model', 'seed', 'fold_id', 'outcome_day', 'y_true', 'y_pred'}
    if not required <= set(predictions):
        raise ValueError('Missing diagnostic inputs: ' + str(required - set(predictions)))
    frame = predictions.copy()
    frame['outcome_day'] = pd.to_datetime(frame.outcome_day, utc=True).dt.strftime('%Y-%m-%d')
    if (frame.empty or not set(frame.model) <= set(MODELS)
            or frame.duplicated(['history_days', 'model', 'seed', 'outcome_day']).any()
            or not np.isfinite(frame[['y_true', 'y_pred']].to_numpy(float)).all()):
        raise ValueError('Diagnostic inputs must be finite, unique stored forecasts of the three models')
    if not frame.fold_id.eq(frame.outcome_day.str[:7]).all():
        raise ValueError('Monthly identifiers must match outcome dates')
    bds_frame, lb_frame = _monthly_diagnostics(frame, phase)
    dm_frame = _dm_pairs(frame, phase)
    return {'bds_by_fold': bds_frame, 'ljung_box_by_fold': lb_frame,
            'dm_pairs': dm_frame, 'diagnostic_summary': _summary(bds_frame, lb_frame)}
