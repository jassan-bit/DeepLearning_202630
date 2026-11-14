"""Bounded, read-only audit of saved E2 calendars and metadata. No model execution."""
import argparse
import csv
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def readj(name):
    return json.loads((ROOT / name).read_text())

def shafile(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()

def days(a, b):
    a, b = date.fromisoformat(a), date.fromisoformat(b)
    return [(a + timedelta(days=i)).isoformat() for i in range((b-a).days+1)]

def shift(day, n):
    return (date.fromisoformat(day) + timedelta(days=n)).isoformat()

def sig(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def raw_support(ids, start_offset, end_offset):
    return {d for t in ids for d in days(shift(t,start_offset),shift(t,end_offset))}

def describe(ids):
    return {'count':len(ids),'first':min(ids) if ids else None,
            'last':max(ids) if ids else None,'dates_sha256':sig(sorted(ids))}


def build_audit():
    protocol_path='experiments/e2_reduced_protocol_20261112/protocol.json'
    prepared_path='experiments/e2_reduced_protocol_20261112/prepared_context.json'
    admission_path='experiments/e2_benchmark_final_20261113/data_admission.json'
    freeze_path='experiments/e2_benchmark_final_20261113/final_freeze.json'
    input_path='book_e2/_static/e2_final_audit_inputs_20261114/e2_f2a_input.json'
    fit_path='book_e2/_static/e2_f2a_audit_20261113/fit_overfit_summary.csv'
    availability_path='book_e2/_static/common_executor_20261109/availability.json'
    protocol,prepared,admission,freeze,payload,availability = map(readj,
        [protocol_path,prepared_path,admission_path,freeze_path,input_path,availability_path])
    rows=list(csv.DictReader((ROOT / fit_path).open()))
    assert shafile(protocol_path)==freeze['protocol_sha256']
    assert freeze['protocol_id']==protocol['protocol_id']=='e2-reduced-delivery-v1'
    assert admission['development_bridge']['source_sha256']=={
        k:v['sha256'] for k,v in availability['sources'].items()}
    for path,expected in admission['development_bridge']['source_sha256'].items():
        assert shafile(path)==expected

    # This existing guard reads arrays and dates only. It does not score or invoke a model.
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT / 'scripts'))
    from e2_temporal_final_audit import predictions_by_model
    targets,predictions,mask=predictions_by_model(payload)
    test_dates=days('2025-01-01','2026-08-31')
    cal_dates=days('2024-11-08','2024-12-30')
    assert payload['dates']==admission['common_eligible_dates']==test_dates
    assert len(cal_dates)==53 and len(test_dates)==608
    assert all(type(v) is bool for v in payload['mask']) and all(payload['mask'])
    freeze_by_id={(r['model'],r['seed']):r for r in freeze['models']}
    saved_by_id={(r['model'],r['seed']):r for r in payload['predictors']}
    assert len(freeze_by_id)==len(saved_by_id)==81 and freeze_by_id.keys()==saved_by_id.keys()
    for key,row in saved_by_id.items():
        assert row['checkpoint_sha256']==freeze_by_id[key]['checkpoint_sha256']
        assert row['calibration_dates']==cal_dates
        assert len(row['calibration_targets'])==len(row['calibration_predictions'])==53
        assert row['calibration_targets']==payload['predictors'][0]['calibration_targets']

    metadata_by_origin={r['origin']:r for r in availability['ARIMA_dated_training_and_bridge_availability']}
    invalid=metadata_by_origin['2024-11-08']['outer_training']['missing_or_invalid_label_dates']
    train_dates=[d for d in days('2023-01-01','2024-10-31') if d not in invalid]
    assert len(train_dates)==655
    final_rows=[r for r in rows if r['stage']=='final']
    assert len(final_rows)==78 and {r['train_count'] for r in final_rows}=={'655'}
    assert len({(r['model'],int(r['seed'])) for r in final_rows})==78
    for row in final_rows:
        key=(row['model'],int(row['seed']))
        assert row['selected_checkpoint_sha256']==freeze_by_id[key]['checkpoint_sha256']
        assert row['calibration_count']=='53'
        assert row['calibration_start']==cal_dates[0] and row['calibration_end']==cal_dates[-1]

    folds=[]
    for f in protocol['hpo']['folds']:
        origin=f['fold_origin'];m=metadata_by_origin[origin]['inner_training']
        t=[d for d in days('2023-01-01',f['inner_train_label_end'])
           if d not in m['missing_or_invalid_label_dates']]
        v=prepared['folds'][origin]['validation_dates']
        assert v==days(f['inner_validation_start'],f['inner_validation_end'])
        assert len(v)==83 and len(t)==m['neural_D7_training_count']
        assert not set(t)&set(v)
        target_overlap=raw_support(t,-7,0)&raw_support(v,-7,0)
        input_overlap=raw_support(t,-7,-1)&raw_support(v,-7,-1)
        assert not target_overlap and not input_overlap
        observed=[r for r in rows if r['stage']=='hpo' and r['fold_origin']==origin]
        direct=[r for r in observed if r['train_count']]
        assert len(observed)==13 and {r['validation_count'] for r in observed}=={'83'}
        assert {int(r['train_count']) for r in direct}=={len(t)}
        assert all(r['train_signature']==prepared['folds'][origin]['training_signature'] for r in direct)
        folds.append({
            'fold_origin':origin,'role':f['role'],
            'inner_train':describe(t),'inner_train_calendar_rows':m['calendar_rows'],
            'inner_train_excluded_objective_dates':m['missing_or_invalid_label_dates'],
            'validation':describe(v),'validation_mask_true':83,'validation_mask_false':0,
            'embargo_objective_dates':days(shift(t[-1],1),shift(v[0],-1)),
            'validation_target_close_support':describe(sorted(raw_support(v,-7,0))),
            'validation_input_OHLCV_support':describe(sorted(raw_support(v,-7,-1))),
            'train_validation_objective_ID_intersection':0,
            'train_validation_target_close_intersection':len(target_overlap),
            'train_validation_input_OHLCV_intersection':len(input_overlap),
            'current_prepared_training_signature':prepared['folds'][origin]['training_signature'],
            'current_prepared_validation_signature':prepared['folds'][origin]['validation_signature'],
            'saved_selected_HPO_rows':len(observed),
            'rows_with_direct_train_count_and_matching_signature':len(direct),
            'cached_rows_without_direct_train_count':len(observed)-len(direct),
            'count_evidence':'current prepared dates/signatures + same-source dated availability metadata; direct selected-fit rows where present',
        })

    cohorts={'final_train':train_dates,'calibration':cal_dates,'evaluation':test_dates}
    intersections=[]
    for a,b in [('final_train','calibration'),('final_train','evaluation'),('calibration','evaluation')]:
        ids=sorted(set(cohorts[a])&set(cohorts[b]))
        close=sorted(raw_support(cohorts[a],-7,0)&raw_support(cohorts[b],-7,0))
        returns=sorted(raw_support(cohorts[a],-6,0)&raw_support(cohorts[b],-6,0))
        inputs=sorted(raw_support(cohorts[a],-7,-1)&raw_support(cohorts[b],-7,-1))
        assert not ids
        intersections.append({'pair':[a,b],'objective_ID_intersection_count':len(ids),
            'target_close_raw_support_intersection':close,
            'target_return_raw_support_intersection':returns,
            'D7_input_OHLCV_raw_support_intersection':inputs,
            'interpretation':'causal rolling-window dependence, no reused fit objective IDs' if close
              else 'disjoint raw target-close supports under the registered date rule'})
    assert intersections[0]['target_close_raw_support_intersection']==[]
    assert intersections[1]['target_close_raw_support_intersection']==[]
    assert intersections[2]['target_close_raw_support_intersection']==days('2024-12-25','2024-12-30')

    source_paths=[protocol_path,prepared_path,admission_path,freeze_path,input_path,fit_path,availability_path]
    audit={
     'version':'e2-eda-current-partition-metadata-audit-v1',
     'scope':'Bounded verification of current E2 partition dates, saved masks, predictor IDs and inherited support metadata; no model execution or raw archive access',
     'operations':{'model_fits':0,'model_inferences':0,'scaler_fits':0,'new_predictions':0,
                   'raw_archives_opened':0,'new_performance_metrics':0,'source_artifacts_modified':False},
     'current_protocol':{'protocol_id':protocol['protocol_id'],'protocol_sha256_matches_final_freeze':True,
       'hpo_folds':3,'validation_dates_per_fold':83,'concatenated_validation_dates':249,
       'configurations_per_neural_model':6,'hpo_seed_ids':protocol['hpo']['seed_ids'],
       'final_seed_ids':freeze['final_seed_ids'],'systems':16,'frozen_predictor_IDs':81},
     'supersession_note':'The EDA02 description of 10 monthly folds and a 60-date pilot belongs to the historical design. It does not describe the executed reduced campaign. Historical metadata is used here only for unchanged source-validity masks and exact matching cutoffs, never as the current protocol authority.',
     'hpo_fold_table':folds,
     'final_partition':{
       'training':{**describe(train_dates),'mask_rule':'D7_target[t] and t <= 2024-10-31',
         'calendar_rows':670,'excluded_dates':invalid,'saved_current_neural_fit_rows':78,
         'all_saved_current_neural_train_counts':655,'common_saved_train_signature':final_rows[0]['train_signature'],
         'training_mask_reconstructed_from_saved_validity_metadata':True,
         'actual_private_training_tensor_bytes_reopened':False,
         'training_sample_IDs':train_dates},
       'training_to_calibration_bridge':{'dates':days('2024-11-01','2024-11-07'),
         'fit_or_calibration_objective_IDs':False,'role':'causal observations only; no predictor refit'},
       'calibration':{**describe(cal_dates),'saved_predictors_with_identical_dates_and_targets':81,
         'mask_true':53,'mask_false':0,'sample_IDs':cal_dates,
         'role':'fit frozen residual calibrators only; not HPO or predictor/scaler refit'},
       'calibration_to_evaluation_bridge':{'dates':['2024-12-31'],
         'in_training_objective_IDs':False,'in_calibration_objective_IDs':False,
         'in_evaluation_objective_IDs':False,'role':admission['development_bridge']['bridge_role'],
         'first_evaluation_input_dates':days('2024-12-25','2024-12-31')},
       'evaluation':{**describe(test_dates),'mask_true':608,'mask_false':0,'sample_IDs':test_dates,
         'all_16_systems_same_saved_support':True,'saved_predictor_trajectories':81,
         'saved_guard_function':'scripts/e2_temporal_final_audit.py:52 predictions_by_model',
         'saved_guard_passed':True,'predictor_IDs_unique_and_match_freeze':True,
         'checkpoint_metadata_match_freeze':True,
         'admission_quality_records':len(admission['quality_by_archive']),
         'admission_reported_complete_days':sum(r['complete_days'] for r in admission['quality_by_archive']),
         'quality_records_recomputed_from_raw_archives':False,
         'known_historical_E1_calendar':True,'prospective_blinding_recovered':False}},
     'formal_support':{
       'sample_ID':'UTC objective date t; h=1 issued after completed day t-1',
       'input':'X_t = OHLCV[t-7,...,t-1], 7x5; log(OHLC_d / Close_(t-1)), log1p(volume_d)',
       'target':'y_t = sqrt((1/7) sum_(j=0..6) (r_(t-j)-mean(r_(t-6)..r_t))^2), r_d=log(C_d)-log(C_(d-1))',
       'target_close_support':'[t-7,t], eight daily closes; population ddof0; not annualized',
       'target_return_support':'[t-6,t], seven daily log returns',
       'target_information_at_issue':'six of seven target returns are already observed at t-1; only r_t is future',
       'bar_mask':'finite five channels, positive prices, nonnegative volume, valid OHLC geometry and daily quality',
       'target_mask':'finite nonnegative y_t and valid quality bars for [t-7,t]',
       'D7_target_mask':'D7_window[t] AND target_sigma7_quality[t]',
       'common_mask':'D7_target[t] AND target_quality[t-1] AND target_quality[t-7] AND target_quality[t-14] AND target_quality[t-28]',
       'scaler_scope':'neural fit uses D7_target; evaluation uses common; fold-local scalers use training dates only',
       'seasonal_input_support':'lag28 y_(t-28) requires closes [t-35,t-28], all historical at issue time; classical state may also reuse older observed history',
       'embargo_proof':'If last train objective <= s-8 and first validation/calibration objective is s, max train target close <= s-8 < s-7 = first later target close. Thus disjoint target-close supports for each HPO train/validation pair and final train/calibration.',
       'cohort_pair_intersections':intersections,
       'causal_overlap_interpretation':'Disjoint objective-date IDs do not imply independent examples or disjoint historical inputs. Rolling targets and inputs within a cohort overlap. Reusing already observed prices/returns or past labels as authorized causal inputs/state is distinct from fitting weights, scalers or recipes on calibration/evaluation objective IDs.',
       'calibration_evaluation_exception':'No seven-day embargo is registered between Dec30 calibration and Jan1 evaluation. Their target-close unions overlap Dec25-Dec30 (six closes; five return dates Dec26-Dec30), and D7 input unions overlap Dec25-Dec29 (five OHLCV dates). Dec31 stays a causal bridge. No iid/exchangeability or exact finite-sample conformal coverage is certified.'},
     'evidence':[{ 'path':p,'local_bytes_sha256':shafile(p)} for p in source_paths],
     'code_locators':[
      'scripts/e2_reduced_protocol.py:68 current three folds',
      'scripts/e2_reduced_final.py:66 final training mask and :76 calibration dates',
      'scripts/e2_temporal_training.py:48 relative transform, :106 raw D7 windows, :130 fold-local scaler',
      'scripts/audit_e2_common_availability.py:50 source-validity masks',
      'scripts/e2_real_executor.py:60 support_frame, :114 dated block plan',
      'scripts/e2_reduced_final_data.py:162 population target; :166 Dec31 causal bridge',
      'scripts/e2_temporal_final_audit.py:52 saved calendar/vector/predictor guard'],
     'limits':[
      'This audit verifies saved metadata/date-mask consistency and formal interval intersections. It does not independently verify the physical private fit tensors, checkpoints, scaler statistics, or chronology of every historical execution.',
      'Three-fold train counts 253/435/527 are supported by matching saved validity metadata and prepared signatures. Selected HPO direct train-count metadata exists for 5/11/13 of 13 neural winners per fold; cached rows retain 83 validation dates but omit direct train counts.',
      'All 78 current final neural rows directly report 655 training examples and match checkpoint metadata. Classical availability is inherited metadata, not a new coefficient or convergence audit.',
      'The saved 608-entry all-true mask and finite vectors establish the exported input support. Dates alone cannot independently prove OHLCV source quality, minute completeness, or correct target values.',
      'Admission reports for 20 source archives and 608 complete days are existing receipts. No ZIP was opened or minute data reaggregated during this audit.',
      'Local file hashes bind only the listed bytes to the metadata used here. They are not a global provenance, absence-of-leakage, or human noninspection certificate.',
      'The final calendar was known from E1; this audit cannot recover blinding or independent prospective confirmation.',
      'No new model fit, inference, recalibration, prediction, benchmark score, or inferential test was executed.']}

    return audit


def summary(report):
    return {
        'version': report['version'],
        'hpo_folds': [
            {'origin': row['fold_origin'], 'training': row['inner_train']['count'],
             'validation': row['validation']['count']}
            for row in report['hpo_fold_table']],
        'final_train': report['final_partition']['training']['count'],
        'calibration': report['final_partition']['calibration']['count'],
        'evaluation': report['final_partition']['evaluation']['count'],
        'saved_predictor_IDs': report['current_protocol']['frozen_predictor_IDs'],
        'calibration_evaluation_shared_close_dates': report['formal_support']['cohort_pair_intersections'][2]['target_close_raw_support_intersection'],
        'model_fits': 0,
        'model_inferences': 0,
        'raw_archives_opened': 0,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        help='Write a new audit JSON. Existing output files are never overwritten; omit to print only a summary.')
    args=parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('Existing audit output is immutable; choose a new output path.')
    report=build_audit()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary(report),ensure_ascii=False))


if __name__ == '__main__':
    main()
