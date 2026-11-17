"""Synthetic numerical checks; this file never loads published market forecasts."""
from copy import deepcopy
from datetime import date, timedelta
import json
import math

import numpy as np
import pytest
from scipy import stats

import analysis as a


def synthetic_panel():
    t = np.arange(608)
    target = .02+.001*np.sin(t/30)
    base = np.array([(.001+.00002*r)*(1+.35*np.sin(t/13)+.2*np.cos(t/40))
                     +.0001*np.sin((r+1)*t/80) for r in range(10)])
    names = [a.ORIGINAL, *a.NEURAL, *a.CLASSICAL]
    predictions = {a.ORIGINAL: target[None, :]+base}
    for i, model in enumerate(a.NEURAL):
        predictions[model] = target[None, :]+(1.4+i*.07)*base+.00005*np.sin(t/17)[None, :]
    for i, model in enumerate(a.CLASSICAL):
        predictions[model] = target[None, :]+(.002+i*.0002)*(1+.3*np.sin(t/9))[None, :]
    return {'model_order': names, 'rivals': [*a.NEURAL, *a.CLASSICAL], 'predictions': predictions,
        'targets': target, 'dates': [(date(2025, 1, 1)+timedelta(days=i)).isoformat() for i in range(608)],
        'mask': np.ones(608, dtype=bool), 'seeds': {m: ('deterministic',) if m in a.CLASSICAL else a.SEEDS for m in names},
        'train_counts': {m: 532 if m == a.ORIGINAL else 655 for m in names}, 'provenance': {'synthetic_fixture': True}}


def synthetic_plan():
    return {'analysis': {'delta': a.DELTA, 'alpha': .05, 'bootstrap_repetitions': 99,
        'bootstrap_seed': 2026111701, 'primary_block_days': 14, 'sensitivity_block_days': [7, 28],
        'hac_lags': 14, 'sensitivity_hac_lags': [7, 28], 'bootstrap_chunk_size': 32,
        'seed_effect_bootstrap_repetitions': 199}}


def test_primary_metric_is_mean_of_individual_rmse():
    squared = np.array([[1., 1.], [9., 9.]])
    assert a.mean_individual_rmse(squared) == 2.
    assert not math.isclose(a.mean_individual_rmse(squared), math.sqrt(squared.mean()))
    # Errors +1 and -3 would instead give an ensemble RMSE of1.
    assert not math.isclose(a.mean_individual_rmse(squared), 1.)


def test_circular_counts_preserve608_positions_and_synchronous_models():
    counts = a.circular_counts(608, 7, 14, np.random.default_rng(1))
    assert counts.shape == (7, 608) and np.all(counts.sum(axis=1) == 608)
    squared = np.array([np.linspace(.1, 4, 608), np.linspace(2, .2, 608)])
    loss = {'first': squared, 'same': squared.copy(), 'classical': squared[:1]}
    rmse, mae = a.temporal_draws(loss, {k: np.sqrt(v) for k, v in loss.items()}, repetitions=99, block=14, seed=7)
    assert np.array_equal(rmse['first'], rmse['same'])
    assert np.array_equal(mae['first'], mae['same'])
    assert np.all(rmse['first']-rmse['same'] == 0)
    assert loss['classical'].shape[0] == 1


def test_rmse_influence_matches_finite_difference_and_hac_sign_swap():
    sq = np.array([[1., 4., 9., 16.], [4., 1., 9., 4.]])
    direction = np.array([-.1, .05, .03, .02])
    epsilon = 1e-6
    value = lambda weights: np.sqrt(sq @ weights).mean()
    derivative = (value(np.ones(4)/4+epsilon*direction)-value(np.ones(4)/4-epsilon*direction))/(2*epsilon)
    influence = a.rmse_influence(sq)
    assert math.isclose(derivative, float(influence @ direction), rel_tol=1e-8)
    assert math.isclose(a.hac_standard_error(influence, 1), a.hac_standard_error(-influence, 1))
    assert math.isclose(a.hac_standard_error(np.array([1., 2., 3., 4.]), 1), .625)


def test_bootstrap_null_centering_and_margin_monotonicity():
    draws = np.array([.5, 1., 1.5, 2.])
    small_margin = a.bootstrap_result(1., draws, .25)
    large_margin = a.bootstrap_result(1., draws, .75)
    assert small_margin['p_raw'] == .4
    assert large_margin['p_raw'] == .6
    assert large_margin['p_raw'] >= small_margin['p_raw']
    assert small_margin['gain_ci95'][0] < small_margin['gain_ci95'][1]


def test_hac_test_uses_observed_gain_not_zero_mean_influence():
    influence = np.sin(np.arange(608)/5)*.01
    row = a.hac_result(.005, influence, .001, 14)
    assert np.mean(influence) != .005
    assert math.isclose(row['statistic'], (.005-.001)/a.hac_standard_error(influence, 14))
    assert row['p_raw'] < .01
    assert a.hac_result(.005, influence, .002, 14)['p_raw'] >= row['p_raw']


def test_seed_test_matches_paired_gain_t_and_classical_stays_singleton():
    gains = .001+np.arange(10)*.0001
    indices = np.random.default_rng(2).integers(0, 10, (199, 10))
    row = a.seed_result(gains, a.DELTA, indices)
    reference = stats.ttest_1samp(gains, a.DELTA, alternative='greater')
    assert math.isclose(row['statistic'], reference.statistic)
    assert math.isclose(row['p_raw'], reference.pvalue)
    assert row['degrees_of_freedom'] == 9 and row['seed_n'] == 10
    assert math.isclose(row['paired_Cohen_dz'], gains.mean()/gains.std(ddof=1))
    assert row['Shapiro_use'].startswith('diagnostic_only')
    original = np.linspace(.001, .002, 10)
    classical = np.array([.003])
    assert (classical-original).shape == (10,) and classical.shape == (1,)


def test_holm48_and_missing_hypothesis_conservative_slot():
    rows = [{'p_raw': .0009}] + [{'p_raw': 1.} for _ in range(47)]
    rows[12]['p_raw'] = None
    adjusted = a.holm(rows)
    assert math.isclose(adjusted[0]['p_holm'], .0432)
    assert adjusted[0]['reject'] and not adjusted[12]['reject'] and adjusted[12]['p_holm'] is None
    with pytest.raises(ValueError, match='family size'): a.holm(rows[:16])


def test_degenerate_uncertainty_does_not_create_significant_pzero():
    boot = a.bootstrap_result(.01, np.full(99, .01), a.DELTA)
    hac = a.hac_result(.01, np.zeros(608), a.DELTA, 14)
    seed = a.seed_result(np.ones(10)*.01, a.DELTA, np.zeros((99, 10), dtype=int))
    assert all(row['p_raw'] is None and row['status'] == 'not_estimable' for row in (boot, hac, seed))
    assert a.rmse_influence(np.zeros((1, 608))) is None


def test_complete_synthetic_stage_keeps_full_families_and_metadata():
    panel, plan = synthetic_panel(), synthetic_plan()
    output = a.analyze(panel, plan)
    assert len(output['primary_contrasts']) == output['primary_family_size'] == 48
    assert len(output['per_rival_verdicts']) == 16 and len(output['effects']) == 16
    assert len(output['MAE_secondary_contrasts']) == 16 and output['MAE_secondary_family_size'] == 16
    assert all(r['no_practical_MAE_margin_claimed'] for r in output['MAE_secondary_contrasts'])
    assert len(output['seed_subgroups_descriptive']) == 28
    assert output['primary_block_days'] == 14 and [s['block_days'] for s in output['calendar_sensitivities']] == [7, 28]
    assert all(r['primary'] for r in output['primary_contrasts'])
    assert all(not r['primary'] for s in output['calendar_sensitivities'] for r in s['contrasts'])
    assert all(s['no_additional_primary_rejection_opportunity'] for s in output['calendar_sensitivities'])
    assert not output['R3_verified'] and not output['confirmatory_preregistration']
    assert not output['guide_three_valid_F2a_tests_verified'] and output['new_model_fits'] == output['new_model_inference'] == 0
    assert output['observed_power'].startswith('not_estimated')
    assert {r['model']: r['R'] for r in output['metrics']} == {m: 1 if m in a.CLASSICAL else 10 for m in panel['model_order']}
    assert {r['method'] for r in output['primary_contrasts']} == set(a.METHODS)
    assert all(r['uncertainty_scope'] == 'initializations_conditional_fixed_calendar'
               for r in output['primary_contrasts'] if r['method'] == a.METHODS[2])
    json.dumps(output, allow_nan=False)
    assert a.analyze(panel, plan) == output


@pytest.mark.parametrize('mutation', ['duplicate_seed', 'fake_classical', 'mask', 'date', 'negative', 'nonfinite', 'roster'])
def test_invalid_panel_is_rejected_without_dropping_data(mutation):
    panel = synthetic_panel()
    if mutation == 'duplicate_seed': panel['seeds'][a.ORIGINAL] = (*a.SEEDS[:-1], a.SEEDS[0])
    elif mutation == 'fake_classical': panel['predictions']['ARIMA'] = np.repeat(panel['predictions']['ARIMA'], 10, axis=0)
    elif mutation == 'mask': panel['mask'][3] = False
    elif mutation == 'date': panel['dates'][3] = panel['dates'][2]
    elif mutation == 'negative': panel['predictions'][a.ORIGINAL][0, 0] = -.1
    elif mutation == 'nonfinite': panel['targets'][0] = np.nan
    elif mutation == 'roster': panel['rivals'].pop()
    with pytest.raises(ValueError): a.validate_panel(panel)


def test_inherited_delta_and_primary_sensitivity_choices_cannot_change():
    plan = synthetic_plan(); plan['analysis']['delta'] = 0.
    with pytest.raises(ValueError, match='margin'): a.settings(plan)
    plan = synthetic_plan(); plan['analysis']['primary_block_days'] = 7
    with pytest.raises(ValueError, match='primary14'): a.settings(plan)
