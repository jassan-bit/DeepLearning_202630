"""Verify the adopted q on the exact ONNX against frozen historical arrays.

This is an engineering parity check; it does not refit q, weights or scalers.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
from e3_dashboard import inference as serving


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cohort, predictions):
    model = 'CNN-LSTM-atencion-aditiva E3'
    meta, folder = serving._metadata(model)
    profile = meta['serving_calibration']
    assert sha(cohort) == meta['data_source']['npz_sha256']
    manifest = json.loads((ROOT/'experiments/e3_remediacion_20261118/calibration/input_manifest.json').read_bytes())
    expected = next(r for r in manifest['files'] if r['name'] == 'evaluation_predictions.csv')
    assert sha(predictions) == expected['sha256']
    with predictions.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['seed']) == meta['seed']]
    with np.load(cohort, allow_pickle=False) as saved:
        raw = np.asarray(saved['evaluation_raw'])
        targets = np.asarray(saved['evaluation_y_original'])
        dates = [str(v) for v in saved['evaluation_dates']]
        cal = np.asarray(saved['calibration_raw'])
        cal_targets = np.asarray(saved['calibration_y_original'])
    assert raw.shape == (608,7,5) and cal.shape == (53,7,5)
    assert dates == [r['date'] for r in rows]
    assert np.array_equal(targets, [float(r['target_sigma7']) for r in rows])
    frozen = np.asarray([float(r['prediction_sigma7']) for r in rows])
    session = serving._session(model, meta, folder)
    current = np.asarray([serving._run(window, meta, session) for window in raw])
    calibration = np.asarray([serving._run(window, meta, session) for window in cal])
    with (ROOT/'experiments/e3_remediacion_20261118/calibration/results/calibration_predictions.csv').open(newline='') as stream:
        cal_rows = [r for r in csv.DictReader(stream) if int(r['seed']) == meta['seed']]
    cal_reference = np.asarray([float(r['frozen_CPU_prediction_sigma7']) for r in cal_rows])
    maximum_point_error = float(np.max(np.abs(current-frozen)))
    maximum_cal_error = float(np.max(np.abs(calibration-cal_reference)))
    assert maximum_point_error <= 1e-6 and maximum_cal_error <= 1e-6
    q = meta['calibration_q']
    lower, upper = np.maximum(0.,current-q), current+q
    coverage = 100 * float(np.mean((lower<=targets)&(targets<=upper)))
    q1 = np.asarray([d.startswith(('2026-01','2026-02','2026-03')) for d in dates])
    subgroup_coverage = 100 * float(np.mean(((lower<=targets)&(targets<=upper))[q1]))
    assert coverage == profile['historical_evaluation']['coverage_percent']
    expected_q1 = next(r['coverage_percent'] for r in profile['historical_subgroups']
                       if r['group_type']=='calendar_quarter' and r['group']=='2026Q1')
    assert abs(subgroup_coverage-expected_q1) < 1e-12
    result = {'verified_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'passed_serving_parity_and_posterior_interval_only',
        'python':platform.python_version(),'model_id':model,'seed':meta['seed'],
        'model_sha256':meta['model_sha256'],'checkpoint_sha256':profile['checkpoint_sha256'],
        'calibration_q_sigma7':q,'nominal_coverage':profile['nominal_coverage'],
        'calibration_windows_verified':53,'evaluation_windows_verified':608,
        'new_ORT_forward_windows_in_this_check':661,
        'maximum_absolute_point_error_vs_frozen':maximum_point_error,
        'maximum_absolute_calibration_error_vs_frozen_CPU':maximum_cal_error,
        'declared_tolerance_sigma7':1e-6,
        'current_ORT_coverage_percent':coverage,'current_ORT_2026Q1_coverage_percent':subgroup_coverage,
        'current_ORT_mean_width_sigma7':float(np.mean(upper-lower)),
        'current_ORT_calibration_empirical_coverage_percent':100*float(np.mean(np.abs(cal_targets-calibration)<=q)),
        'source_scientific_mean_10_seed_coverage_not_used':True,
        'point_function_changed':False,'weights_refitted':False,'scaler_refitted':False,
        'quantile_refitted':False,'new_real_data_fits':0,'new_hypothesis_tests':0,
        'new_confirmatory_market_forecasts':0,
        'scope':'Historical engineering forwards; does not certify exchangeability, prospective coverage or R1/R3.'}
    serving.clear_cache()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--frozen-predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to replace an existing verification receipt')
    result = run(args.cohort,args.frozen_predictions)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
