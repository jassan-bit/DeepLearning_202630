"""Audit saved new diagnostic files; no model imports or predictions."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import math
import statistics

ROOT = Path('/workspace/e3_completion_delivery')
PAYLOAD = ROOT/'reproduction/payload'
OUT = PAYLOAD/'diagnostic_transfer'
QA = ROOT/'qa'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a,b,tol=1e-12):
    assert abs(a-b) <= tol * max(1,abs(a),abs(b)), (a,b)


def audit():
    decision_path = ROOT/'protocol_power/decision_zero_shot.json'
    decision = json.loads(decision_path.read_text())
    expected_sha = '6bda16b7e322e03a1494420f4bedae58a38cad0aed790a8c6493d5d15637df29'
    assert sha(decision_path) == sha(QA/'remote_registered_decision_snapshot.json') == expected_sha
    remote = json.loads((QA/'zero_shot_remote_registration_metadata.json').read_text())
    before = json.loads((OUT/'before_first_forward.json').read_text())
    receipt = json.loads((OUT/'diagnostic_receipt.json').read_text())
    assert remote['sha'] == before['registered_decision_commit'] == receipt['registered_decision_commit']
    assert datetime.fromisoformat(remote['created_at'].replace('Z','+00:00')) < datetime.fromisoformat(before['started_at_utc']) <= datetime.fromisoformat(receipt['finished_at_utc'])
    assert receipt['decision_sha256'] == before['decision_sha256'] == expected_sha
    for rel,digest in receipt['inference_code_sha256'].items():
        assert sha(PAYLOAD/rel) == digest == before['inference_code_sha256'][rel]
    assert receipt['new_model_fits'] == receipt['new_calibrations'] == receipt['HPO_trials'] == 0
    assert receipt['p_values_computed'] is receipt['R3_accredited'] is receipt['confirmatory_claim'] is False
    assert receipt['pilot_heuristic_unchanged_and_failed'] is True and receipt['winning_ablation_promoted'] is False
    meta = json.loads((PAYLOAD/'reference/additional_domain_admission_metadata.json').read_text())
    admission = ROOT/'reproduction/private_inputs/admission'
    assert sha(admission/'admission.json') == decision['target_panels']['admission_receipt_sha256']
    expected_targets = {}
    for dataset,entry in meta['datasets'].items():
        raw_path = admission/entry['path']
        assert sha(raw_path) == entry['sha256'] == decision['target_panels']['datasets'][dataset]['normalized_CSV_sha256']
        rows=list(csv.DictReader(raw_path.open(newline='')))
        assert len(rows) == entry['rows']
        logclose = [math.log(float(r['close'])) for r in rows]
        returns = [b-a for a,b in zip(logclose,logclose[1:])]
        target_map={}
        for t in range(entry['raw_target_start_index'],entry['raw_target_stop_exclusive']):
            known = returns[t-7:t-1]
            y = statistics.pstdev(returns[t-7:t])
            floor = math.sqrt(sum((x-statistics.mean(known))**2 for x in known)/7)
            target_map[rows[t]['date']] = (y,floor)
        assert len(target_map) == decision['target_panels']['datasets'][dataset]['expected_test_rows']
        expected_targets[dataset] = target_map
    metrics={}
    row_count=0
    for name,digest in receipt['payload_sha256'].items():
        path=OUT/name
        assert sha(path) == digest
        rows=list(csv.DictReader(path.open(newline='')))
        assert rows
        dataset,recipe=rows[0]['dataset'],rows[0]['name']
        seed=int(rows[0]['seed']) if rows[0]['seed'] else None
        key=(dataset,recipe,seed)
        assert key not in metrics
        assert [r['date'] for r in rows] == list(expected_targets[dataset])
        errors=[];absolute=[];crps=[]
        for r in rows:
            assert r['dataset']==dataset and r['name']==recipe
            assert (int(r['seed']) if r['seed'] else None)==seed
            y,pred,floor=map(float,(r['target_sigma7'],r['prediction_sigma7'],r['floor_sigma7']))
            assert all(math.isfinite(x) for x in (y,pred,floor))
            close(y,expected_targets[dataset][r['date']][0])
            close(floor,expected_targets[dataset][r['date']][1])
            assert pred+1e-12>=floor
            if recipe=='floor_only':close(pred,floor)
            if recipe=='floor_plus_mean_excess':close(pred,floor+decision['frozen_pilot']['source_BTC_d_fit'])
            atomic=recipe in {'atom_crps','atom_mse_only','fixed_probs','atom_plugin'}
            assert bool(r['CRPS_U']) == atomic
            if atomic:
                score=float(r['CRPS_U']); assert math.isfinite(score) and score >=-1e-12;crps.append(score)
            errors.append((pred-y)**2);absolute.append(abs(pred-y))
        row_count+=len(rows)
        metrics[key]={'RMSE':math.sqrt(statistics.mean(errors)), 'MAE':statistics.mean(absolute),
            'CRPS_U':statistics.mean(crps) if crps else None, 'rows':len(rows)}
    expected_keys={(d,n,s) for d in expected_targets for n in decision['frozen_pilot']['neural_recipes'] for s in decision['frozen_pilot']['final_seeds']}
    expected_keys |= {(d,n,None) for d in expected_targets for n in ('floor_only','floor_plus_mean_excess')}
    assert set(metrics)==expected_keys and len(metrics)==51 and row_count==20400
    summaries={}
    for row in receipt['summary']:
        dataset,recipe=row['dataset'],row['name']
        values=[v for (d,n,s),v in metrics.items() if (d,n)==(dataset,recipe)]
        expected=3 if recipe in decision['frozen_pilot']['neural_recipes'] else 1
        assert len(values)==expected==row['finite_trajectories']==row['expected_trajectories']
        close(statistics.mean(v['RMSE'] for v in values),row['primary_mean_individual_seed_RMSE'])
        close(statistics.mean(v['MAE'] for v in values),row['mean_individual_seed_MAE'])
        if all(v['CRPS_U'] is not None for v in values): close(statistics.mean(v['CRPS_U'] for v in values),row['mean_CRPS_U'])
        else: assert row['mean_CRPS_U'] is None
        summaries[(dataset,recipe)]=row['primary_mean_individual_seed_RMSE']
    adverse=[]
    for dataset in expected_targets:
        candidate=summaries[(dataset,'atom_crps')]
        controls=[(n,value) for (d,n),value in summaries.items() if d==dataset and n!='atom_crps']
        best_name,best_value=min(controls,key=lambda x:x[1])
        assert candidate>best_value
        adverse.append({'dataset':dataset,'candidate_mean_individual_RMSE':candidate,
            'best_control':best_name,'best_control_RMSE':best_value,
            'candidate_percent_worse_than_best_control':100*(candidate/best_value-1)})
    report={'status':'retrospective_transfer_files_and_recorded_decision_order_QA_passed',
        'AI_assistance_declared':True,'new_QA_model_fits':0,'new_QA_model_predictions':0,
        'registered_commit':remote['sha'],'Git_commit_created_at':remote['created_at'],
        'first_forward_gate_recorded_at':before['started_at_utc'],
        'time_evidence_scope':'Git commit metadata, root publication observation/Merkle receipt and local forward gate; not an independent institutional timestamp certification.',
        'decision_sha256':expected_sha,'receipt_sha256':sha(OUT/'diagnostic_receipt.json'),
        'saved_forecast_files_hashed':51,'saved_prediction_rows_verified':row_count,
        'dataset_CSV_bytes_hashed':3,'target_dates_and_sigma7_ddof0_independently_checked':True,
        'mean_individual_seed_RMSE_and_MAE_independently_recomputed_from_saved_predictions':True,
        'CRPS_values_finite_nonnegative_and_means_recomputed':True,
        'CRPS_limitation':'QA did not reopen models or atoms/probabilities to reexecute CRPS; source computes raw-U CRPS only after each full forward trajectory.',
        'analytic_control_BTC_d_fit_unchanged':True,'all_original_final_recipes_and_seeds_included':True,
        'candidate_beats_best_control_in_any_domain':False,'domain_results':adverse,
        'new_diagnostic_prediction_rows_report_separately_from_parity':20400,
        'new_diagnostic_model_fits':0,'new_diagnostic_calibrations':0,'new_diagnostic_resamples':0,
        'scientific_claim':'Transfer diagnostic on previously exposed panels; no per-domain fitting, independent confirmation, R3, promotion or revision of failed2percent pilot heuristic.'}
    (QA/'zero_shot_output_review.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    report=audit()
    print(json.dumps({k:v for k,v in report.items() if k!='domain_results'},indent=2))
