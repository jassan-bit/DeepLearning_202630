"""Adopt one existing posterior calibrator for its exact serving predictor.

No model fitting, calibration refit or changes to scientific point predictions.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'experiments/e3_remediacion_20261118/calibration'
OUTPUT = ROOT / 'artifacts/e3_serving_calibration_v2'
MODEL = 'CNN-LSTM-atencion-aditiva E3'
SEED = 271801


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def build():
    metadata_path = ROOT / 'artifacts/e3_dashboard_v1/cnn_lstm_attention_e3/metadata.json'
    metadata = json.loads(metadata_path.read_bytes())
    original_verification = json.loads((SOURCE / 'results/verification.json').read_bytes())
    assert original_verification['status'] == 'passed'
    for name, expected in original_verification['result_sha256'].items():
        assert sha(SOURCE / 'results' / name) == expected, name
    plan = json.loads((SOURCE / 'analysis_plan.json').read_bytes())
    assert plan['acceptance']['activate_in_dashboard'] is False  # historical scope remains intact
    with (SOURCE / 'results/per_seed_intervals.csv').open(newline='') as f:
        row = next(r for r in csv.DictReader(f) if int(r['seed']) == SEED)
    assert metadata['seed'] == SEED and row['checkpoint_sha256'] == metadata['scientific_source']['checkpoint_sha256']
    assert sha(metadata_path.parent / 'model.onnx') == metadata['model_sha256']
    with (SOURCE / 'results/calibration_predictions.csv').open(newline='') as f:
        residual_rows = [r for r in csv.DictReader(f) if int(r['seed']) == SEED]
    dates = [r['target_date'] for r in residual_rows]
    residuals = [float(r['absolute_residual_sigma7']) for r in residual_rows]
    assert dates == sorted(set(dates)) and len(dates) == int(row['calibration_n']) == 53
    assert dates[0] == plan['calibration']['first'] and dates[-1] == plan['calibration']['last']
    alpha = plan['calibration']['alpha']
    rank = math.ceil((len(residuals) + 1) * (1-alpha))
    assert rank == int(row['calibration_order_rank']) == 49
    q = sorted(residuals)[rank-1]
    assert q == float(row['q_sigma7'])
    with (SOURCE / 'results/subgroup_diagnostics.csv').open(newline='') as f:
        groups = [{k: (int(v) if k == 'n' else float(v) if k.endswith('percent') else v)
                   for k, v in r.items() if k in ('group_type','group','n','coverage_percent')}
                  for r in csv.DictReader(f) if int(r['seed']) == SEED]
    source_paths = ['analysis_plan.json','input_manifest.json','results/per_seed_intervals.csv',
                    'results/calibration_predictions.csv','results/subgroup_diagnostics.csv','results/verification.json']
    sidecar = {
        'schema_version': 1, 'model_id': MODEL, 'seed': SEED,
        'profile': 'posterior_absolute_residual_90_v2',
        'model_sha256': metadata['model_sha256'],
        'checkpoint_sha256': row['checkpoint_sha256'],
        'base_metadata_sha256': sha(metadata_path),
        'output_scale': metadata['output_scale'], 'output_unit': metadata['output_unit'],
        'nominal_coverage': 1-alpha, 'alpha': alpha, 'q_sigma7': q,
        'interval_rule': 'max(0, prediction - q), prediction + q',
        'calibration_n': len(dates), 'order_rank': rank,
        'calibration_first_target': dates[0], 'calibration_last_target': dates[-1],
        'applicable_after_target_date': dates[-1],
        'calibration_dates': dates, 'absolute_calibration_residuals_sigma7': residuals,
        'historical_evaluation': {'seed': SEED, 'n': int(row['n']),
            'first': '2025-01-01', 'last': '2026-08-31',
            'coverage_percent': float(row['coverage_percent']),
            'mean_width_sigma7': float(row['mean_width_sigma7']),
            'mean_interval_score_alpha_0_1': float(row['mean_interval_score_alpha_0_1'])},
        'historical_subgroups': groups,
        'chronology': {'fit_n': 532, 'fit_last_target': '2024-06-30',
            'validation_n': 116, 'validation_first_target': '2024-07-08',
            'validation_last_target': '2024-10-31',
            'calibration_targets_disjoint_from_fit_validation': True,
            'calibration_inputs_start_after_validation_targets': True,
            'known_input_overlap_with_evaluation': True},
        'adoption_scope': 'New serving adoption of an existing retrospective diagnostic; original analysis plan and campaign unchanged.',
        'coverage_statement': 'Intervalo posterior nominal del 90 % del mismo predictor. Cobertura histórica 94,08 % en 608 fechas; 85,56 % en 2026Q1. Sin garantía bajo cambio temporal.',
        'not_a_prospective_or_exchangeability_guarantee': True,
        'point_forecasts_changed': False, 'refit_model': False, 'refit_quantile': False,
        'source_commit': '1bae73b5373c11466b226fb3efab706237b5e68b',
        'source_files': {str((SOURCE / p).relative_to(ROOT)): {'bytes': (SOURCE / p).stat().st_size, 'sha256': sha(SOURCE / p)} for p in source_paths},
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    raw = canonical(sidecar)
    (OUTPUT / 'e3_additive_271801.json').write_bytes(raw)
    manifest = {'schema_version': 1, 'models': [{'model_id': MODEL,
        'file': 'e3_additive_271801.json', 'bytes': len(raw),
        'sha256': hashlib.sha256(raw).hexdigest()}]}
    (OUTPUT / 'manifest.json').write_bytes(canonical(manifest))
    print(json.dumps({'model': MODEL,'seed': SEED,'q_sigma7': q,
                      'representative_coverage_percent':float(row['coverage_percent']),
                      'source_quantile_refit':False,'point_forecasts_changed':False}))


if __name__ == '__main__':
    build()
