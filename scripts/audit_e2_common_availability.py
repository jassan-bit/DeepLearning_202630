"""Read-only support audit for §21, never a fit, score, or final-data loader.

Only registered development derivatives are decoded. Final archive metadata is
not evidence of target/window availability. Receipts contain counts, dates,
causes and source hashes, never market values or descriptive distributions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'experiments/dl_r2_cv_20261104/protocol.json'
SOURCES = ('results/eda_development/daily_market_development.csv',
           'results/eda_development/daily_volatility_development.csv')
RESERVE = 'data/manifests/test_raw_reserve_20250101_20260831.json'
FEATURES = ('open', 'high', 'low', 'close', 'volume')
LAGS = (1, 7, 14, 28)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dates(index):
    return pd.DatetimeIndex(index).strftime('%Y-%m-%d').tolist()


def validate_calendar(calendar):
    calendar = pd.DatetimeIndex(calendar)
    if (calendar.tz is None or str(calendar.tz) != 'UTC' or calendar.hasnans
            or not calendar.is_unique or not calendar.is_monotonic_increasing
            or not calendar.equals(calendar.normalize()) or not len(calendar)):
        raise ValueError('Require ordered unique UTC midnight dates')
    if not calendar.equals(pd.date_range(calendar[0], calendar[-1], tz='UTC')):
        raise ValueError('Retain calendar gaps as explicit rows; never compress time')
    return calendar


def lag_mask(valid, lag):
    result = np.zeros(len(valid), dtype=bool)
    if lag < len(valid):
        result[lag:] = valid[:-lag]
    return result


def support_masks(calendar, features, targets, quality=None):
    """Pure validity operations. No scaling, target reconstruction or prediction.

    quality is an explicit boolean daily source-quality mask when supplied.
    Otherwise the derivative's missing/invalid bars are the quality evidence;
    this does not independently reaggregate minute archives.
    """
    calendar = validate_calendar(calendar)
    x, y = np.asarray(features, float), np.asarray(targets, float)
    if x.shape != (len(calendar), 5) or y.shape != (len(calendar),):
        raise ValueError('Calendar, OHLCV and target shapes differ')
    q = np.ones(len(calendar), dtype=bool) if quality is None else np.asarray(quality)
    if q.shape != y.shape or q.dtype.kind != 'b':
        raise ValueError('Quality must be an aligned boolean daily mask')
    bar = (np.isfinite(x).all(axis=1) & (x[:, :4] > 0).all(axis=1)
           & (x[:, 4] >= 0) & (x[:, 1] >= np.maximum(x[:, 0], x[:, 3]))
           & (x[:, 2] <= np.minimum(x[:, 0], x[:, 3])) & (x[:, 1] >= x[:, 2]) & q)
    # Conservative support: a finite close in an otherwise invalid bar is
    # not sufficient source-quality evidence. Explicit quality can only narrow
    # this validity mask, never rehabilitate invalid OHLCV.
    close = bar.copy()
    eight = close.copy()
    window = np.ones(len(calendar), dtype=bool)
    for lag in range(1, 8):
        eight &= lag_mask(close, lag)
        window &= lag_mask(bar, lag)
    target = np.isfinite(y) & (y >= 0) & eight
    masks = {'D7_window': window, 'target_sigma7_quality': target,
             'D7_target': window & target}
    for lag in LAGS:
        masks[f'lag{lag}_target'] = lag_mask(target, lag)
    masks['common'] = np.logical_and.reduce([masks['D7_target']] +
                                            [masks[f'lag{lag}_target'] for lag in LAGS])
    return masks


def period_report(calendar, masks, start, end, expected):
    wanted = pd.date_range(start, end, tz='UTC')
    positions = calendar.get_indexer(wanted)
    present = positions >= 0
    selected = {}
    for name, mask in masks.items():
        ok = np.zeros(len(wanted), dtype=bool)
        ok[present] = mask[positions[present]]
        selected[name] = ok
    causes = {'missing_calendar_row': dates(wanted[~present])}
    for name in ('D7_window', 'target_sigma7_quality', 'lag1_target',
                 'lag7_target', 'lag14_target', 'lag28_target'):
        causes[name] = dates(wanted[present & ~selected[name]])
    return {'start': start, 'end': end, 'reference_count': expected,
            'calendar_count': len(wanted), 'common_count': int(selected['common'].sum()),
            'difference_from_reference': int(selected['common'].sum()) - expected,
            'support_counts': {k: int(v.sum()) for k, v in selected.items()},
            'excluded_dates': dates(wanted[~selected['common']]),
            'exclusion_causes_nonexclusive': causes,
            'new_exclusions_vs_D7_target': dates(wanted[selected['D7_target'] & ~selected['common']])}


def arima_support(calendar, masks, origin):
    """Count dated valid labels; missing labels remain on the daily grid.

    ARIMA uses valid univariate targets, NOT the more restrictive common mask.
    Bridges need date rows, not every observed value, as missing-measurement
    propagation is allowed. These checks cannot certify numerical convergence.
    """
    o = pd.Timestamp(origin, tz='UTC')
    result = {'origin': origin, 'minimum_valid_labels': 90, 'convergence_certified': False}
    for key, cutoff in [('outer_training', o-pd.Timedelta(days=8)),
                        ('inner_training', o-pd.Timedelta(days=98))]:
        cohort = calendar <= cutoff
        valid = cohort & masks['target_sigma7_quality']
        result[key] = {'cutoff': str(cutoff.date()), 'calendar_rows': int(cohort.sum()),
                       'valid_labels': int(valid.sum()), 'at_least_90': bool(valid.sum() >= 90),
                       'missing_or_invalid_label_dates': dates(calendar[cohort & ~masks['target_sigma7_quality']]),
                       'neural_D7_training_count': int((cohort & masks['D7_target']).sum())}
    for key, begin, end in [('outer_bridge', o-pd.Timedelta(days=7), o-pd.Timedelta(days=1)),
                            ('inner_bridge', o-pd.Timedelta(days=97), o-pd.Timedelta(days=91)),
                            ('inner_validation', o-pd.Timedelta(days=90), o-pd.Timedelta(days=8))]:
        wanted = pd.date_range(begin, end, tz='UTC'); ix = calendar.get_indexer(wanted)
        present = ix >= 0
        valid = np.zeros(len(wanted), dtype=bool)
        valid[present] = masks['target_sigma7_quality'][ix[present]]
        result[key] = {'start': str(begin.date()), 'end': str(end.date()),
                       'calendar_days': len(wanted), 'all_calendar_rows_present': bool(present.all()),
                       'valid_labels': int(valid.sum()), 'missing_label_dates': dates(wanted[~valid]),
                       'missing_measurement_propagation_allowed': key != 'inner_validation'}
    return result


def audit(root=ROOT):
    root = Path(root)
    registered = json.loads((root/PROTOCOL).read_text())['sources']
    hashes = {}
    for name in SOURCES:
        actual = digest(root/name)
        if actual != registered[name]:
            raise ValueError('Registered source SHA256 mismatch: '+name)
        hashes[name] = {'sha256': actual, 'matches_registration': True}
    frames = [pd.read_csv(root/name) for name in SOURCES]
    calendars = [validate_calendar(pd.to_datetime(f['market_day_utc'])) for f in frames]
    calendar = pd.date_range('2023-01-01', '2024-12-31', tz='UTC')
    if not all(c.equals(calendar) for c in calendars):
        raise ValueError('Registered development calendar differs')
    masks = support_masks(calendar, frames[0][list(FEATURES)], frames[1]['sigma_7'])
    periods = {'development': period_report(calendar, masks, '2023-01-01', '2024-12-31', 731),
               'CV': period_report(calendar, masks, '2024-01-01', '2024-10-31', 305),
               'historical_E1_pilot': period_report(calendar, masks, '2024-11-01', '2024-12-30', 60),
               'proposed_E2_pilot': period_report(calendar, masks, '2024-11-08', '2024-12-30', 53)}
    origins = [f'2024-{m:02d}-01' for m in range(1, 11)] + ['2024-11-01', '2024-11-08', '2025-01-01']
    reserve = json.loads((root/RESERVE).read_text())
    if reserve['calendar_days'] != 608 or reserve['test_start_day_utc'] != '2025-01-01' or reserve['test_end_day_utc_inclusive'] != '2026-08-31':
        raise ValueError('Final reference calendar differs')
    tracked = [PROTOCOL, RESERVE, 'book_e2/sections/17_contrato_sincronizado.md',
               'book_e2/sections/21_comparadores_nas.md', 'scripts/audit_e2_common_availability.py']
    return {'schema_version': 1, 'status': 'development_available_final_payload_not_inspected',
            'model_fits': 0, 'optimizer_steps': 0, 'scaler_fits': 0, 'forecasts': 0,
            'performance_metrics': 0, 'HPO_trials': 0, 'NAS_trials': 0,
            'raw_archives_opened': 0, 'market_values_exported': 0,
            'sources': hashes, 'source_and_contract_sha256': {n: digest(root/n) for n in tracked},
            'development_support': periods,
            'ARIMA_dated_training_and_bridge_availability': [arima_support(calendar, masks, o) for o in origins],
            'final_support': {'start': '2025-01-01', 'end': '2026-08-31', 'reference_count': 608,
                              'common_count': None, 'difference_from_reference': None,
                              'status': 'not_certified', 'reason': 'Only reservation metadata read; no verified final daily derivative supplied. Final payload requires its separate access gate.',
                              'archive_count_reserved': len(reserve['archives']), 'missing_dates': None},
            'limits': ['Source-quality evidence is invalid/missing bars in registered daily derivatives; minute completeness is not independently reaggregated.',
                       'All 1/7/14/28 lags enter the common evaluation mask before any candidate selection.',
                       'Neural training support is not reduced by comparator lag requirements.',
                       'ARIMA availability does not certify fit convergence or finite forecast output.',
                       'Historical contracts and 305/60/53/608 reference supports remain unchanged.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.project_root.resolve()):
        raise ValueError('Availability receipts must remain outside the publication repository')
    if args.output.exists():
        raise FileExistsError('Never overwrite an existing receipt')
    result = audit(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'model_fits': 0}))


if __name__ == '__main__':
    main()
