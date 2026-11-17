"""Posterior numerical audit of frozen forecasts; no fitting or data acquisition.

Calendar resampling/HAC condition on fitted trajectories; seed t inference
conditions on the observed calendar. Their conjunction is exploratory and does
not complete the guide's homogeneous temporal battery, power or R3 requirements.
Only synthetic fixtures may be analyzed before the external plan is published.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
import numpy as np
from scipy import stats

ORIGINAL = 'CNN-LSTM-atencion-aditiva E3'
NEURAL = ('FCN-1D', 'ResNet-1D', 'Inception-1D', 'LSTM-autoencoder con predictor',
          'TimesNet', 'TSMixer', 'TiDE', 'TimeMixer', 'PatchTST', 'iTransformer',
          'MLP16', 'LSTM32', 'GRU32')
CLASSICAL = ('Persistence', 'SeasonalNaive', 'ARIMA')
SEEDS = tuple(range(271801, 271811))
DELTA = .00025772842664981106
METHODS = ('paired_circular_block_bootstrap', 'mean_RMSE_influence_Bartlett_HAC', 'paired_seed_t_fixed_calendar')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def mean_individual_rmse(squared):
    values = np.asarray(squared, dtype=np.float64)
    require(values.ndim == 2 and np.isfinite(values).all() and np.min(values) >= 0,
            'Expected finite nonnegative R by T squared errors')
    return float(np.sqrt(values.mean(axis=1)).mean())


def circular_counts(n_dates, draws, block, rng):
    require(type(block) is int and 1 <= block <= n_dates and draws >= 1,
            'Invalid circular bootstrap dimensions')
    starts = rng.integers(0, n_dates, (draws, math.ceil(n_dates/block)))
    indices = ((starts[:, :, None]+np.arange(block)) % n_dates).reshape(draws, -1)[:, :n_dates]
    counts = np.zeros((draws, n_dates), dtype=np.int16)
    np.add.at(counts, (np.arange(draws)[:, None], indices), 1)
    return counts


def temporal_draws(squared, absolute, *, repetitions, block, seed, chunk_size=64):
    """One identical dated index draw for every model and every seed trajectory."""
    n = next(iter(squared.values())).shape[1]
    rng = np.random.default_rng(np.random.SeedSequence([seed, block, 701]))
    names = list(squared)
    rmse = {m: np.empty(repetitions) for m in names}
    mae = {m: np.empty(repetitions) for m in names}
    for start in range(0, repetitions, chunk_size):
        stop = min(start+chunk_size, repetitions)
        counts = circular_counts(n, stop-start, block, rng)
        for model in names:
            rmse[model][start:stop] = np.sqrt(counts @ squared[model].T/n).mean(axis=1)
            mae[model][start:stop] = counts @ absolute[model].mean(axis=0)/n
    return rmse, mae


def rmse_influence(squared):
    """Exact first derivative of mean individual RMSE, one value per date."""
    squared = np.asarray(squared, dtype=float)
    mse = squared.mean(axis=1)
    if np.any(mse <= 0) or not np.isfinite(squared).all():
        return None
    return ((squared-mse[:, None])/(2*np.sqrt(mse[:, None]))).mean(axis=0)


def hac_standard_error(influence, lags):
    if influence is None:
        return None
    x = np.asarray(influence, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        return None
    require(type(lags) is int and 0 <= lags < len(x), 'Invalid HAC bandwidth')
    centered = x-x.mean(); n = len(x)
    variance = float(centered @ centered/n)
    for lag in range(1, lags+1):
        variance += 2*(1-lag/(lags+1))*float(centered[lag:] @ centered[:-lag]/n)
    return math.sqrt(variance/n) if variance > 0 else None


def bootstrap_result(observed, draws, delta, alpha=.05, family_size=48):
    draws = np.asarray(draws, dtype=float)
    require(np.isfinite(draws).all() and len(draws) >= 2, 'Nonfinite bootstrap draws')
    spread = float(np.std(draws, ddof=1))
    if spread <= np.finfo(float).eps*max(1., abs(observed)):
        return {'status': 'not_estimable', 'reason': 'degenerate_bootstrap_distribution', 'p_raw': None,
                'standard_error': None, 'gain_ci95': None, 'bonferroni_lower_bound': None}
    noise = draws-observed
    exceedances = int(np.count_nonzero(noise >= observed-delta))
    p = (1+exceedances)/(len(draws)+1)
    return {'status': 'estimable', 'reason': None, 'p_raw': p,
            'statistic': observed-delta, 'statistic_name': 'observed_gain_minus_null_boundary',
            'standard_error': spread, 'gain_ci95': [observed-np.quantile(noise, .975), observed-np.quantile(noise, .025)],
            'ci_method': 'basic_centered_block_bootstrap_nominal_95_percent',
            'bonferroni_lower_bound': observed-np.quantile(noise, 1-alpha/family_size),
            'exceedances': exceedances, 'bootstrap_repetitions': len(draws),
            'p_value_resolution': 1/(len(draws)+1),
            'monte_carlo_standard_error_approx': math.sqrt(p*(1-p)/(len(draws)+1))}


def hac_result(observed, influence, delta, lags, alpha=.05, family_size=48):
    se = hac_standard_error(influence, lags)
    if se is None or se <= np.finfo(float).eps*max(1., abs(observed)):
        return {'status': 'not_estimable', 'reason': 'undefined_RMSE_derivative_or_nonpositive_HAC_variance',
                'p_raw': None, 'standard_error': None, 'statistic': None, 'gain_ci95': None}
    statistic = (observed-delta)/se
    return {'status': 'estimable', 'reason': None, 'statistic': statistic,
            'p_raw': float(stats.norm.sf(statistic)), 'standard_error': se,
            'gain_ci95': [observed-stats.norm.ppf(.975)*se, observed+stats.norm.ppf(.975)*se],
            'ci_method': 'normal_delta_method_HAC_nominal_95_percent',
            'bonferroni_lower_bound': observed-stats.norm.ppf(1-alpha/family_size)*se,
            'HAC_lags': lags, 'HAC_kernel': 'Bartlett', 'reference_distribution': 'standard_normal_asymptotic'}


def seed_result(gains, delta, bootstrap_indices, alpha=.05, family_size=48):
    gains = np.asarray(gains, dtype=float); n = len(gains)
    mean, sd = float(gains.mean()), float(gains.std(ddof=1))
    if sd <= np.finfo(float).eps*max(1., abs(mean)):
        return {'status': 'not_estimable', 'reason': 'degenerate_seed_gain_distribution', 'p_raw': None,
                'standard_error': None, 'gain_ci95': None, 'paired_Cohen_dz': None,
                'paired_Cohen_dz_ci95': None, 'Shapiro_W': None, 'Shapiro_p_diagnostic': None}
    se = sd/math.sqrt(n); statistic = (mean-delta)/se
    draws = gains[bootstrap_indices]
    draw_sd = draws.std(axis=1, ddof=1)
    valid = draw_sd > np.finfo(float).eps*max(1., abs(mean))
    dz = draws[valid].mean(axis=1)/draw_sd[valid]
    shapiro = stats.shapiro(gains)
    return {'status': 'estimable', 'reason': None, 'statistic': statistic,
            'p_raw': float(stats.t.sf(statistic, n-1)), 'standard_error': se,
            'gain_ci95': [mean-stats.t.ppf(.975, n-1)*se, mean+stats.t.ppf(.975, n-1)*se],
            'ci_method': 'paired_seed_t_nominal_95_percent_conditional_calendar',
            'bonferroni_lower_bound': mean-stats.t.ppf(1-alpha/family_size, n-1)*se,
            'seed_n': n, 'degrees_of_freedom': n-1, 'paired_gain_sample_SD': sd,
            'paired_Cohen_dz': mean/sd,
            'paired_Cohen_dz_ci95': np.quantile(dz, [.025, .975]) if valid.mean() >= .95 else None,
            'Cohen_dz_reference': 'zero_gain_not_delta; nominal_seed_bootstrap_CI',
            'Cohen_dz_estimable_bootstrap_draws': int(valid.sum()),
            'Shapiro_W': float(shapiro.statistic), 'Shapiro_p_diagnostic': float(shapiro.pvalue),
            'Shapiro_use': 'diagnostic_only; does_not_validate_normality_or_select_a_different_method'}


def holm(rows, family_size=48, alpha=.05):
    """Missing hypotheses remain in the family as conservative internal p=1."""
    require(len(rows) == family_size, 'Fixed Holm family size differs')
    adjusted = [dict(row, p_holm=None, reject=False) for row in rows]
    values = []
    for index, row in enumerate(rows):
        p = row['p_raw']
        require(p is None or math.isfinite(p) and 0 <= p <= 1, 'Invalid raw p-value')
        values.append((1. if p is None else p, index))
    previous = 0.
    for rank, (p, index) in enumerate(sorted(values)):
        previous = max(previous, min(1., (family_size-rank)*p))
        if rows[index]['p_raw'] is not None:
            adjusted[index].update(p_holm=previous, reject=previous < alpha)
    return adjusted


def settings(plan):
    cfg = plan['analysis']
    result = {'delta': cfg.get('delta', DELTA), 'alpha': cfg.get('alpha', .05),
        'B': cfg.get('bootstrap_repetitions', 9999), 'seed': cfg.get('bootstrap_seed', 2026111701),
        'block': cfg.get('primary_block_days', 14), 'lags': cfg.get('hac_lags', 14),
        'sens_blocks': cfg.get('sensitivity_block_days', [7, 28]),
        'sens_lags': cfg.get('sensitivity_hac_lags', [7, 28]),
        'chunk': cfg.get('bootstrap_chunk_size', 64),
        'effect_B': cfg.get('seed_effect_bootstrap_repetitions', cfg.get('bootstrap_repetitions', 9999))}
    require(result['delta'] == DELTA and result['alpha'] == .05, 'Inherited margin or family alpha changed')
    require(type(result['B']) is int and result['B'] >= 99 and type(result['seed']) is int,
            'Explicit bootstrap count and integer RNG seed required')
    require(type(result['chunk']) is int and result['chunk'] > 0 and type(result['effect_B']) is int and result['effect_B'] >= 99,
            'Valid chunk and seed-effect bootstrap counts required')
    require(result['block'] == result['lags'] == 14 and result['sens_blocks'] == result['sens_lags'] == [7, 28],
            'The predeclared primary14/sensitivity7,28 policy changed')
    return result


def validate_panel(panel):
    names = list(panel['model_order']); rivals = list(panel['rivals'])
    require(names == [ORIGINAL, *NEURAL, *CLASSICAL] and rivals == [*NEURAL, *CLASSICAL],
            'The complete original plus16 rival roster must be preserved')
    y = np.asarray(panel['targets'], dtype=np.float64)
    require(y.shape == (608,) and np.isfinite(y).all() and np.min(y) >= 0, 'Exactly608 finite original targets required')
    expected_dates = [(date(2025, 1, 1)+timedelta(days=i)).isoformat() for i in range(608)]
    require(list(panel['dates']) == expected_dates and np.asarray(panel['mask']).shape == (608,) and
            np.asarray(panel['mask']).dtype == np.dtype(bool) and np.asarray(panel['mask']).all(),
            'Complete dense ordered common calendar required')
    squared, absolute, seed_rmse, metric_rows, subgroup = {}, {}, {}, [], []
    for model in names:
        p = np.asarray(panel['predictions'][model], dtype=np.float64)
        expected_seeds = ('deterministic',) if model in CLASSICAL else SEEDS
        require(tuple(panel['seeds'][model]) == expected_seeds and p.shape == (len(expected_seeds), 608),
                'Seed IDs, deterministic singleton or predictor shape differs: ' + model)
        require(np.isfinite(p).all() and np.min(p) >= 0, 'Predictions must be finite and nonnegative')
        with np.errstate(over='ignore', invalid='ignore'):
            error = p-y[None, :]; squared[model], absolute[model] = error**2, np.abs(error)
        require(np.isfinite(squared[model]).all(), 'Squared forecast errors overflowed')
        seed_rmse[model] = np.sqrt(squared[model].mean(axis=1))
        metric_rows.append({'model': model, 'R': len(p), 'RMSE_mean': seed_rmse[model].mean(),
            'RMSE_sample_SD': seed_rmse[model].std(ddof=1) if len(p) > 1 else None,
            'MAE_mean': absolute[model].mean(), 'train_n': panel.get('train_counts', {}).get(model),
            'calendar_n': 608, 'seed_values_are_not_independent_datasets': True})
        if model not in CLASSICAL:
            for label, slc in [('first5', slice(0, 5)), ('last5', slice(5, 10))]:
                subgroup.append({'model': model, 'subgroup': label, 'seed_ids': list(SEEDS[slc]),
                    'RMSE_mean': seed_rmse[model][slc].mean(), 'MAE_mean': absolute[model][slc].mean(),
                    'descriptive_only': True, 'no_hypothesis_test': True})
    return names, rivals, squared, absolute, seed_rmse, metric_rows, subgroup


def analyze(panel, plan):
    """Return JSON-safe aggregates. Publication and frozen-plan admission are external."""
    cfg = settings(plan)
    names, rivals, squared, absolute, seed_rmse, metric_rows, subgroup = validate_panel(panel)
    point = {m: float(seed_rmse[m].mean()) for m in names}
    mae_point = {m: float(absolute[m].mean()) for m in names}
    influence = {m: rmse_influence(squared[m]) for m in names}
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 907]))
    seed_indices = rng.integers(0, 10, (cfg['effect_B'], 10))
    seed_tests, gains, effects = {}, {}, []
    for model in rivals:
        gains[model] = seed_rmse[model]-seed_rmse[ORIGINAL]  # singleton broadcasts, never adds classical fits
        seed_tests[model] = seed_result(gains[model], cfg['delta'], seed_indices)
        effects.append({'rival': model, 'original': ORIGINAL, 'gain_RMSE': point[model]-point[ORIGINAL],
            'relative_RMSE_reduction_percent': 100*(point[model]-point[ORIGINAL])/point[model] if point[model] > 0 else None,
            'paired_Cohen_dz': seed_tests[model]['paired_Cohen_dz'],
            'paired_Cohen_dz_ci95': seed_tests[model]['paired_Cohen_dz_ci95'],
            'Cohen_dz_scope': 'initializations_conditional_fixed_calendar; nominal_CI; no_effect_threshold_claim'})
    primary, sensitivities, secondary = [], [], []
    for block in [cfg['block'], *cfg['sens_blocks']]:
        boot_rmse, boot_mae = temporal_draws(squared, absolute, repetitions=cfg['B'], block=block,
                                           seed=cfg['seed'], chunk_size=cfg['chunk'])
        rows = []
        for model in rivals:
            gain = point[model]-point[ORIGINAL]
            diff_influence = None if influence[model] is None or influence[ORIGINAL] is None else influence[model]-influence[ORIGINAL]
            results = [bootstrap_result(gain, boot_rmse[model]-boot_rmse[ORIGINAL], cfg['delta']),
                       hac_result(gain, diff_influence, cfg['delta'], block), seed_tests[model]]
            scopes = ['calendar_conditional_fitted_trajectories', 'calendar_conditional_fitted_trajectories',
                      'initializations_conditional_fixed_calendar']
            for method, result, scope in zip(METHODS, results, scopes):
                rows.append({'rival': model, 'original': ORIGINAL, 'method': method,
                    'gain_RMSE': gain, 'delta': cfg['delta'], 'null': 'gain_RMSE <= inherited_delta',
                    'alternative': 'gain_RMSE > inherited_delta', 'uncertainty_scope': scope,
                    'block_days': block if method == METHODS[0] else None,
                    'primary': block == cfg['block'], **result})
            if block == cfg['block']:
                mae_gain = mae_point[model]-mae_point[ORIGINAL]
                mae_result = bootstrap_result(mae_gain, boot_mae[model]-boot_mae[ORIGINAL], 0., family_size=16)
                secondary.append({'rival': model, 'metric': 'mean_individual_seed_MAE', 'gain_MAE': mae_gain,
                    'null': 'rival_minus_candidate_mean_MAE <= 0', 'alternative': 'MAE_gain > 0',
                    'null_boundary': 0., 'no_practical_MAE_margin_claimed': True,
                    'block_days': block, 'scope': 'calendar_conditional_fitted_trajectories', **mae_result})
        adjusted = holm(rows)
        if block == cfg['block']:
            primary = adjusted
        else:
            sensitivities.append({'block_days': block, 'HAC_lags': block, 'family_size': 48,
                'no_additional_primary_rejection_opportunity': True, 'contrasts': adjusted})
    verdicts = []
    for effect in effects:
        row = next(r for r in primary if r['rival'] == effect['rival'] and r['method'] == METHODS[0])
        effect['gain_ci95_temporal'] = row['gain_ci95']
    for model in rivals:
        rows = [r for r in primary if r['rival'] == model]
        verdicts.append({'rival': model, 'all_three_estimable': all(r['status'] == 'estimable' for r in rows),
            'all_three_Holm48_reject': all(r['reject'] for r in rows),
            'gain_above_delta_descriptive': point[model]-point[ORIGINAL] > cfg['delta'],
            'conditional_exploratory_verdict': 'all_three_reject' if all(r['reject'] for r in rows) else 'not_all_three_reject',
            'sensitivity_all_three_reject': {str(s['block_days']): all(r['reject'] for r in s['contrasts'] if r['rival'] == model) for s in sensitivities}})
    result = {'scope': 'posterior_frozen_forecast_analysis_of_known_BTC_calendar', 'metrics': metric_rows,
        'effects': effects, 'primary_contrasts': primary, 'primary_family_size': 48,
        'primary_family_alpha': .05, 'multiplicity': 'one_global_Holm48_across16rivals_by3procedures',
        'per_rival_verdicts': verdicts, 'calendar_sensitivities': sensitivities,
        'MAE_secondary_contrasts': holm(secondary, family_size=16),
        'MAE_secondary_family_size': 16, 'MAE_secondary_multiplicity': 'separate_Holm16',
        'seed_subgroups_descriptive': subgroup,
        'all_16_all_three_reject': all(r['all_three_Holm48_reject'] for r in verdicts),
        'bootstrap_repetitions': cfg['B'], 'bootstrap_seed': cfg['seed'], 'primary_block_days': 14,
        'seed_effect_bootstrap_repetitions': cfg['effect_B'],
        'observed_power': 'not_estimated; no_retrospective_observed_power', 'MDE': 'not_estimated_from_test',
        'R3_verified': False, 'confirmatory_preregistration': False, 'guide_three_valid_F2a_tests_verified': False,
        'new_model_fits': 0, 'new_model_inference': 0, 'training_budget_fairness_claimed': False,
        'assumptions_and_limits': [
            'Calendar bootstrap/HAC assume sufficiently weak dependence and stable enough loss behavior; rolling sigma7 overlap is retained by synchronous blocks.',
            'Temporal inference conditions on frozen fitted trajectories and does not resimulate training-data or HPO selection uncertainty.',
            'The seed t-test assumes independent paired initializations and approximately normal seed gains; n10 and Shapiro cannot certify that assumption.',
            'The three procedures have different uncertainty scopes and are not three homogeneous temporal confirmations.',
            'E2 uses655 supervised training examples; E3 uses532 plus116 stopping-validation examples. Historical data and previous results were exposed.',
            'Nominal95percent intervals are not simultaneous Holm confidence intervals; Bonferroni lower bounds are separately labeled.',
            'Ten initializations and one shared608-date calendar do not establish power, independent-market replication, novel mechanism or R3.'],
        'input_provenance': panel.get('provenance', {})}
    return clean(result)
