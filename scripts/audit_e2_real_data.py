"""Read-only E1 §17 real-development audit. No models, optimizers or downloads.

Only aggregate checks and source fingerprints are written. This is not an E2
market training runner: the current synthetic pipeline's provenance gate stays
intact. Preprocessing statistics are permitted; no predictive model is fitted.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dl_data import CALENDAR, FEATURES, SOURCE_PATHS, load_development, make_windows
from dl_data_v2 import fit_relative_preprocessing, relative_features, split_for_origin, transform_relative_windows

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'experiments/dl_r2_cv_20261104/protocol.json'
CONTRACT = 'book_e2/sections/17_contrato_sincronizado.md'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root):
    root = Path(root)
    registered = json.loads((root / PROTOCOL).read_text())['sources']
    result = {}
    for name in SOURCE_PATHS:
        path = root / name
        if not path.is_file():
            raise ValueError(f'Missing established snapshot: {path}')
        actual = digest(path)
        if actual != registered[name]:
            raise ValueError(f'Snapshot SHA256 differs from registration: {name}')
        result[name] = {'sha256': actual, 'matches_registration': True}
    return result


def inspect_frame(frame):
    """Check raw date strings before UTC conversion can hide malformed dates."""
    dates = pd.DatetimeIndex(pd.to_datetime(frame['market_day_utc']))
    if dates.tz is None or str(dates.tz) != 'UTC':
        raise ValueError('Source dates must explicitly use UTC')
    if dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Duplicate, invalid or unordered source dates')
    if not dates.equals(dates.normalize()):
        raise ValueError('Source dates must be UTC midnight')
    if not dates.equals(CALENDAR):
        raise ValueError('Missing calendar rows; never compress gaps')
    return {'rows': len(frame), 'duplicate_dates': 0, 'missing_calendar_rows': 0,
            'utc_midnight': True}


def check_arrays(data):
    x = np.asarray(data['features']); y = np.asarray(data['targets'])
    complete = np.isfinite(x).all(axis=1)
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError('Infinite values')
    observed = x[complete]
    if (np.any(observed[:, :4] <= 0) or np.any(observed[:, 4] < 0)
        or np.any(observed[:, 1] < np.maximum(observed[:, 0], observed[:, 3]))
        or np.any(observed[:, 2] > np.minimum(observed[:, 0], observed[:, 3]))):
        raise ValueError('Invalid OHLCV bounds')
    expected = pd.Series(np.log(x[:, 3])).diff().rolling(7, min_periods=7).std(ddof=0).to_numpy()
    np.testing.assert_allclose(y, expected, rtol=1e-9, atol=1e-13, equal_nan=True)
    windows = data['windows_by_history'][7]
    for t in range(7, len(x)):
        np.testing.assert_array_equal(windows[t], x[t-7:t])
    eligible = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(expected)
    np.testing.assert_array_equal(eligible, data['eligible_by_history'][7])
    return {'incomplete_OHLCV_days': data['calendar'][~complete].strftime('%Y-%m-%d').tolist(),
            'invalid_target_count': int((~np.isfinite(y)).sum()),
            'eligible_D7_count': int(eligible.sum()), 'ineligible_D7_count': int((~eligible).sum()),
            'population7_labels_match': True, 'D7_is_t_minus7_through_t_minus1': True,
            'eligibility_matches_independent_reconstruction': True,
            'label_max_absolute_difference': float(np.nanmax(np.abs(y-expected)))}


def fold_audit(data, origin, end, split=None):
    calendar = data['calendar']; eligible = data['eligible_by_history'][7]
    origin = pd.Timestamp(origin, tz='UTC'); end = pd.Timestamp(end, tz='UTC')
    pos = np.arange(len(calendar))
    o = int((origin-calendar[0]).days)
    if split is None:
        split = split_for_origin(data, 7, origin)
    train, inner, validation = (split[k] for k in ['train_indices', 'inner_train_indices', 'inner_validation_indices'])
    evaluation = np.flatnonzero(eligible & (calendar >= origin) & (calendar <= end))
    assert train.max() <= o-8 and inner.max() <= o-98
    assert np.array_equal(validation, np.flatnonzero(eligible & (pos >= o-90) & (pos <= o-8)))
    assert not np.intersect1d(train, evaluation).size
    assert not np.intersect1d(inner, validation).size
    windows = data['windows_by_history'][7]
    fingerprints = lambda ix: {hashlib.sha256(windows[i].tobytes()).hexdigest() for i in ix}
    collisions = len(fingerprints(train) & fingerprints(evaluation))
    assert collisions == 0
    # Separate inner/refit scalers, independently reconstructed on N*7 positions.
    for cohort in (train, inner):
        scaler = fit_relative_preprocessing(data, 7, cohort)
        raw = windows[cohort]
        independent = np.empty_like(raw)
        independent[:, :, :4] = np.log(raw[:, :, :4]) - np.log(raw[:, -1:, 3:4])
        independent[:, :, 4] = np.log1p(raw[:, :, 4])
        flat = independent.reshape(-1, 5)
        np.testing.assert_allclose(scaler['x_mean'], flat.mean(axis=0), atol=1e-14)
        std = flat.std(axis=0, ddof=0)
        np.testing.assert_allclose(scaler['x_scale'], np.where(std == 0, 1, std), atol=1e-14)
        assert scaler['y_scale'] == float(data['targets'][cohort].mean())
        assert scaler['target_centered'] is False
        # Perturb every observation after the last allowed outcome, including
        # embargo and evaluation; rebuild windows, never reuse stale windows.
        changed = copy.deepcopy(data)
        cut = int(cohort.max()) + 1
        changed['features'][cut:, :4] *= 3.0
        changed['features'][cut:, 4] *= 5.0
        changed['targets'][cut:] *= 7.0
        changed['windows_by_history'][7] = make_windows(changed['features'], 7)
        assert fit_relative_preprocessing(changed, 7, cohort) == scaler
        original_x, original_y = transform_relative_windows(data, 7, cohort, scaler)
        changed_x, changed_y = transform_relative_windows(changed, 7, cohort, scaler)
        np.testing.assert_array_equal(original_x, changed_x)
        np.testing.assert_array_equal(original_y, changed_y)
    # At the prediction origin, future prices and labels cannot change input.
    if o < len(calendar) and eligible[o]:
        future = data['features'].copy(); future[o:] *= 2
        np.testing.assert_array_equal(make_windows(future, 7)[o], windows[o])
    return {'origin': origin.strftime('%Y-%m-%d'), 'evaluation_end': end.strftime('%Y-%m-%d'),
            'train_count': len(train), 'inner_train_count': len(inner), 'inner_validation_count': len(validation),
            'evaluation_count_available': len(evaluation), 'last_train_day': str(calendar[train[-1]].date()),
            'last_inner_train_day': str(calendar[inner[-1]].date()),
            'train_at_most_origin_minus8': True, 'future_perturbation_invariant': True,
            'independent_scaler_reconstruction_matches': True, 'exact_cross_partition_window_duplicates': collisions}


def audit(root=ROOT):
    if not __debug__:
        raise RuntimeError("Audit requires normal Python; optimized mode disables invariant assertions")
    root = Path(root)
    sources = verify_sources(root)
    frames = {name: inspect_frame(pd.read_csv(root/name)) for name in SOURCE_PATHS}
    data = load_development(root)
    quality = check_arrays(data)
    folds = [fold_audit(data, f'2024-{m:02d}-01', str(pd.Timestamp(f'2024-{m:02d}-01') + pd.offsets.MonthEnd(0))[:10]) for m in range(1,11)]
    folds.append(fold_audit(data, '2024-11-01', '2024-12-30'))
    proposed_pilot = fold_audit(data, '2024-11-08', '2024-12-30')
    assert proposed_pilot['evaluation_count_available'] == 53
    assert proposed_pilot['last_train_day'] == '2024-10-31'
    proposed_pilot['status'] = 'proposed_support_only_not_executed'
    proposed_pilot['historical_E1_pilot_unchanged'] = True
    positions = np.arange(len(CALENDAR)); origin = len(CALENDAR); eligible = data['eligible_by_history'][7]
    split = {'train_indices': np.flatnonzero(eligible & (positions <= origin-8)),
             'inner_train_indices': np.flatnonzero(eligible & (positions <= origin-98)),
             'inner_validation_indices': np.flatnonzero(eligible & (positions >= origin-90) & (positions <= origin-8))}
    final = fold_audit(data, '2025-01-01', '2026-08-31', split)
    assert [final[k] for k in ['train_count','inner_train_count','inner_validation_count']] == [709,619,83]
    assert sum(f['evaluation_count_available'] for f in folds[:10]) == 305
    assert folds[-1]['evaluation_count_available'] == 60
    import e2_current_pipeline as pipeline
    assert pipeline.relative_features is relative_features
    try:
        pipeline.validate_synthetic_data(data)
    except ValueError as exc:
        gate = str(exc)
    else:
        raise AssertionError('Synthetic-only pipeline unexpectedly admitted market snapshot')
    tracked = [PROTOCOL, CONTRACT, 'scripts/dl_data.py', 'scripts/dl_data_v2.py', 'scripts/e2_current_pipeline.py', 'scripts/audit_e2_real_data.py']
    return {'schema_version': 1, 'status': 'development_audit_passed_real_pipeline_not_integrated',
            'model_fits': 0, 'optimizer_steps': 0, 'HPO_trials': 0, 'downloads': 0,
            'market_rows_exported': 0, 'sources': sources, 'source_date_checks': frames,
            'quality': quality, 'CV_and_pilot_folds': folds, 'proposed_pilot_support': proposed_pilot,
            'final_training_support': final,
            'pipeline': {'shared_relative_features_identity_verified': True, 'real_input_rejected': gate,
                         'real_end_to_end_integration': 'not implemented; synthetic provenance gate retained'},
            'evaluation_2025_2026': 'not read or recovered; unnecessary for development audit; 608-date real evaluation not certified',
            'limits': ['Daily derivative identity checked against historical registered hashes, not raw minute archive reaggregation.',
                       'Final support independently constructed from contract, not an executable real E2 final runner.',
                       'No prediction, performance, model-level leakage or generalization claims.'],
            'code_and_contract_sha256': {name: digest(root/name) for name in tracked}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output must be a new file: " + str(args.output))
    report = audit(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'model_fits': 0, 'output': str(args.output)}))

if __name__ == '__main__':
    main()
