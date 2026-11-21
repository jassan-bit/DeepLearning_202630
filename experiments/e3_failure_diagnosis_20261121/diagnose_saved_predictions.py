"""Descriptive development diagnostics; no model imports, fits, p-values or RNG."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOURCE = '5f29ab51e1bb3e79d972377b6d682d1785bdc298'
COHORTS = {
    'V3_GPU': {
        'predictions': 'experiments/e3_evidence_close_20261120/pilot/archive/payload/assessment_predictions.csv',
        'scores': 'experiments/e3_evidence_close_20261120/pilot/archive/payload/final_assessment_scores.csv',
        'candidate': 'atom_crps', 'control': 'atom_mse_only', 'seeds': ['20261118','20261119','20261120']},
    'V4_CPU': {
        'predictions': 'experiments/e3_next_evidence_20261121/development/cpu_pilot_results/assessment_predictions.csv',
        'scores': 'experiments/e3_next_evidence_20261121/development/cpu_pilot_results/per_trajectory_metrics.csv',
        'candidate': 'sigma_crps_v4', 'control': 'mse_only_control', 'seeds': ['20261121','20261122','20261123','20261124','20261125']}
}

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)+'\n')

def read_csv(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))

def write_csv(p, rows):
    with p.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stats(y, pred):
    e=[p-t for p,t in zip(pred,y)]
    bias=statistics.mean(e)
    mse=statistics.mean(x*x for x in e)
    centered=statistics.mean((x-bias)**2 for x in e)
    assert math.isclose(mse,bias*bias+centered,rel_tol=1e-12,abs_tol=1e-20)
    return {'n_dates':len(y),'RMSE':math.sqrt(mse),'MAE':statistics.mean(abs(x) for x in e),
            'mean_error':bias,'squared_mean_error':bias*bias,'centered_error_variance_population':centered,
            'MSE':mse}

def balanced_decomposition(matrix):
    R=len(matrix);T=len(matrix[0]);grand=statistics.mean(x for row in matrix for x in row)
    seed=[statistics.mean(row)-grand for row in matrix]
    date=[statistics.mean(matrix[s][t] for s in range(R))-grand for t in range(T)]
    interaction=statistics.mean((matrix[s][t]-grand-seed[s]-date[t])**2 for s in range(R) for t in range(T))
    d={'global_bias_squared':grand*grand,'seed_main_component':statistics.mean(x*x for x in seed),
       'date_main_component':statistics.mean(x*x for x in date),'seed_date_interaction':interaction}
    mse=statistics.mean(x*x for row in matrix for x in row)
    assert math.isclose(sum(d.values()),mse,rel_tol=1e-12,abs_tol=1e-20)
    return d

def main():
    a=argparse.ArgumentParser();a.add_argument('--source-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    inputs=[]
    for cfg in COHORTS.values():
        for key in ['predictions','scores']:
            p=args.source_root/cfg[key];b=p.read_bytes()
            inputs.append({'path':cfg[key],'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
                           'git_blob_sha':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()})
    cpu_root=args.source_root/'experiments/e3_next_evidence_20261121/development/cpu_pilot_results'
    curve_inputs=sorted((cpu_root/'jobs').glob('seed*/*/supervised/curve.json'))
    assert len(curve_inputs)==20
    for p in [q for c in curve_inputs for q in [c,c.with_name('receipt.json')]]:
        b=p.read_bytes();inputs.append({'path':p.relative_to(args.source_root).as_posix(),'bytes':len(b),
            'sha256':hashlib.sha256(b).hexdigest(),
            'git_blob_sha':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()})
    code=Path(__file__).read_bytes()
    plan={'kind':'descriptive_analysis_specification_recorded_before_this_script_output',
          'not_preregistration_of_already_exposed_development_data':True,'source_commit':SOURCE,
          'code_sha256':hashlib.sha256(code).hexdigest(),'input_files':inputs,
          'primary_comparisons':{k:{x:v[x] for x in ['candidate','control','seeds']} for k,v in COHORTS.items()},
          'metric_scope':'Mean individual RMSE remains primary; squared-error decompositions are explicitly separate descriptive endpoints.',
          'decompositions':['Within-seed MSE=bias_squared+centered_error_variance','Balanced observed error table:global_bias_squared+seed_main+date_main+interaction; not inferential ANOVA or causal variance components'],
          'fixed_slices':['first27dates','last28dates','target<=sample_median','target>sample_median'],
          'slice_note':'Exposed development data only; target-based slices are descriptions, never an admissible deployment rule or test-selected redesign.',
          'top_dates':'Top5 signed contributions to excess MSE, plus top5 absolute contributions; report both beneficial and adverse dates.',
          'no_p_values_no_resampling_no_new_models':True,'AI_assistance_declared':True}
    dump(out/'ANALYSIS_PLAN.json',plan)
    all_trajectories=[];pairs=[];dates_out=[];slices=[];summary={}
    reference_errors=[]
    for cohort,cfg in COHORTS.items():
        rows=read_csv(args.source_root/cfg['predictions'])
        scores=read_csv(args.source_root/cfg['scores'])
        by=defaultdict(dict); truth={}
        for r in rows:
            key=(r['name'],r['seed']);d=r['date'];y=float(r['target']);p=float(r['prediction'])
            assert all(math.isfinite(x) for x in [y,p])
            assert d not in by[key],(cohort,key,d)
            if d in truth:assert truth[d]==y,(cohort,d)
            truth[d]=y;by[key][d]=(y,p)
        dates=sorted(truth);assert len(dates)==55
        assert dates[0]=='2024-09-07' and dates[-1]=='2024-10-31'
        score_map={(r['name'],r['seed']):r for r in scores}
        trajectory={}
        for (name,seed),t in sorted(by.items()):
            assert sorted(t)==dates,(cohort,name,seed)
            st=stats([t[d][0] for d in dates],[t[d][1] for d in dates])
            trajectory[(name,seed)]=st
            all_trajectories.append({'cohort':cohort,'name':name,'seed':seed,**st})
            if (name,seed) in score_map:
                for metric in ['RMSE','MAE']:
                    diff=abs(st[metric]-float(score_map[(name,seed)][metric]))
                    reference_errors.append(diff)
                    assert diff<=1e-12,(cohort,name,seed,metric,diff)
        cand=cfg['candidate'];ctrl=cfg['control'];seeds=cfg['seeds']
        assert sorted(s for (n,s) in by if n==cand)==seeds
        assert sorted(s for (n,s) in by if n==ctrl)==seeds
        local_pairs=[]
        for seed in seeds:
            c=trajectory[(cand,seed)];b=trajectory[(ctrl,seed)]
            row={'cohort':cohort,'seed':seed,'candidate':cand,'control':ctrl,
                 'RMSE_candidate':c['RMSE'],'RMSE_control':b['RMSE'],
                 'RMSE_difference':c['RMSE']-b['RMSE'],
                 'MSE_difference':c['MSE']-b['MSE'],
                 'squared_mean_error_difference':c['squared_mean_error']-b['squared_mean_error'],
                 'centered_error_variance_difference':c['centered_error_variance_population']-b['centered_error_variance_population'],
                 'bias_candidate':c['mean_error'],'bias_control':b['mean_error']}
            assert math.isclose(row['MSE_difference'],row['squared_mean_error_difference']+row['centered_error_variance_difference'],rel_tol=1e-10,abs_tol=1e-20)
            local_pairs.append(row);pairs.append(row)
        local_dates=[]
        for d in dates:
            e_c=[by[(cand,s)][d][1]-truth[d] for s in seeds]
            e_b=[by[(ctrl,s)][d][1]-truth[d] for s in seeds]
            dc=statistics.mean(x*x for x in e_c);db=statistics.mean(x*x for x in e_b)
            local_dates.append({'cohort':cohort,'date':d,'target':truth[d],
                'candidate_mean_error':statistics.mean(e_c),'control_mean_error':statistics.mean(e_b),
                'candidate_mean_squared_error':dc,'control_mean_squared_error':db,
                'mean_squared_error_difference':dc-db,
                'mean_prediction_shift':statistics.mean(e_c)-statistics.mean(e_b),
                'seeds_candidate_worse':sum(x*x>y*y for x,y in zip(e_c,e_b)),
                'n_seeds':len(seeds)})
        dates_out.extend(local_dates)
        med=statistics.median(truth.values())
        groups={'first27dates':dates[:27],'last28dates':dates[27:],
                'target<=sample_median':[d for d in dates if truth[d]<=med],
                'target>sample_median':[d for d in dates if truth[d]>med]}
        for group,ds in groups.items():
            rmses={name:[stats([truth[d] for d in ds],[by[(name,s)][d][1] for d in ds])['RMSE'] for s in seeds] for name in [cand,ctrl]}
            v=statistics.mean(rmses[cand]);b=statistics.mean(rmses[ctrl])
            slices.append({'cohort':cohort,'slice':group,'n_dates':len(ds),'n_seeds':len(seeds),
                           'candidate_mean_individual_RMSE':v,'control_mean_individual_RMSE':b,
                           'relative_RMSE_change_percent':100*(v/b-1),
                           'post_result_development_description':True})
        cm=statistics.mean(trajectory[(cand,s)]['RMSE'] for s in seeds)
        bm=statistics.mean(trajectory[(ctrl,s)]['RMSE'] for s in seeds)
        dec={name:balanced_decomposition([[by[(name,s)][d][1]-truth[d] for d in dates] for s in seeds]) for name in [cand,ctrl]}
        net=statistics.mean(r['mean_squared_error_difference'] for r in local_dates)
        assert math.isclose(net,statistics.mean(r['MSE_difference'] for r in local_pairs),rel_tol=1e-10,abs_tol=1e-20)
        adverse=sorted(local_dates,key=lambda r:r['mean_squared_error_difference'],reverse=True)[:5]
        beneficial=sorted(local_dates,key=lambda r:r['mean_squared_error_difference'])[:5]
        total=sum(r['mean_squared_error_difference'] for r in local_dates)
        summary[cohort]={'candidate':cand,'control':ctrl,'n_seeds':len(seeds),'n_dates':55,
            'candidate_mean_individual_RMSE':cm,'control_mean_individual_RMSE':bm,
            'candidate_mean_individual_MAE':statistics.mean(trajectory[(cand,s)]['MAE'] for s in seeds),
            'control_mean_individual_MAE':statistics.mean(trajectory[(ctrl,s)]['MAE'] for s in seeds),
            'relative_RMSE_change_percent':100*(cm/bm-1),
            'seeds_with_higher_RMSE':sum(r['RMSE_difference']>0 for r in local_pairs),
            'dates_with_positive_mean_squared_error_difference':sum(r['mean_squared_error_difference']>0 for r in local_dates),
            'mean_MSE_difference':net,
            'mean_squared_bias_difference':statistics.mean(r['squared_mean_error_difference'] for r in local_pairs),
            'mean_centered_error_variance_difference':statistics.mean(r['centered_error_variance_difference'] for r in local_pairs),
            'mean_bias_candidate':statistics.mean(r['bias_candidate'] for r in local_pairs),
            'mean_bias_control':statistics.mean(r['bias_control'] for r in local_pairs),
            'top5_adverse_dates':[r['date'] for r in adverse],
            'top5_adverse_share_of_net_signed_excess_MSE':sum(r['mean_squared_error_difference'] for r in adverse)/total if total else None,
            'share_note':'May exceed1 because beneficial dates offset adverse ones. Not a share of RMSE or an independent sample.',
            'top5_beneficial_dates':[r['date'] for r in beneficial]}
        summary[cohort]['top5_absolute_contribution_dates']=[r['date'] for r in sorted(local_dates,key=lambda r:abs(r['mean_squared_error_difference']),reverse=True)[:5]]
        summary[cohort]['balanced_MSE_decomposition_candidate_minus_control']={k:dec[cand][k]-dec[ctrl][k] for k in dec[cand]}
    curve_rows=[]
    for p in curve_inputs:
        curve=json.loads(p.read_text());receipt=json.loads(p.with_name('receipt.json').read_text())
        name=p.parent.parent.name;seed=p.parent.parent.parent.name.removeprefix('seed')
        minimum=min(curve,key=lambda x:x['validation_mse_scaled'])
        assert receipt['best_epoch']==minimum['epoch']
        curve_rows.append({'name':name,'seed':seed,'epochs':receipt['epochs'],
            'best_epoch':receipt['best_epoch'],'stop_reason':receipt['stop_reason'],
            'optimizer_steps':receipt['optimizer_steps'],
            'first_validation_MSE_scaled':curve[0]['validation_mse_scaled'],
            'best_validation_MSE_scaled':receipt['best_validation_mse_scaled'],
            'last_validation_MSE_scaled':curve[-1]['validation_mse_scaled'],
            'last_train_total_loss':curve[-1]['train_loss'],
            'train_loss_not_comparable_across_objectives':True,
            'elapsed_wall_seconds':receipt['elapsed_wall_including_setup_io_and_restart_seconds'],
            'budget_seconds':receipt['budget_seconds'],'resume_segments':receipt['resume_segments']})
    write_csv(out/'trajectory_diagnostics.csv',all_trajectories)
    write_csv(out/'paired_seed_decomposition.csv',pairs)
    write_csv(out/'date_error_contributions.csv',dates_out)
    write_csv(out/'development_slices.csv',slices)
    write_csv(out/'training_curve_diagnostics.csv',curve_rows)
    result={'kind':'retrospective_development_diagnostics_not_causal_or_confirmatory',
            'source_commit':SOURCE,'cohorts':summary,'source_metric_max_discrepancy':max(reference_errors),
            'cohorts_pooled':False,'p_values':0,'new_fits':0,'new_model_forwards':0,
            'new_market_predictions':0,'new_MC_replicates':0,'new_statistical_resamples':0,
            'training_curve_summary':{'trajectories':20,
                'stop_reason_counts':{k:sum(r['stop_reason']==k for r in curve_rows) for k in sorted({r['stop_reason'] for r in curve_rows})},
                'wall_budget_stops':sum('budget' in r['stop_reason'] for r in curve_rows),
                'resume_segments':sum(r['resume_segments'] for r in curve_rows),
                'min_elapsed_wall_seconds':min(r['elapsed_wall_seconds'] for r in curve_rows),
                'max_elapsed_wall_seconds':max(r['elapsed_wall_seconds'] for r in curve_rows)},
            'R3_accredited':False,'AI_assistance_declared':True}
    dump(out/'DIAGNOSTIC_RESULTS.json',result)
    for f in inputs:
        assert hashlib.sha256((args.source_root/f['path']).read_bytes()).hexdigest()==f['sha256']
    dump(out/'EXECUTION_RECEIPT.json',{'completed_at_UTC':datetime.now(timezone.utc).isoformat(),
          'operation':'Deterministic reaggregation and descriptive decomposition of saved development predictions',
          'input_files_unchanged':True,'trajectory_rows':len(all_trajectories),'paired_seed_rows':len(pairs),
          'date_rows':len(dates_out),'slice_rows':len(slices),'source_metric_max_discrepancy':max(reference_errors),
          'training_curve_rows':len(curve_rows),'input_files_checked':len(inputs),
          'code_sha256':hashlib.sha256(code).hexdigest(),'new_fits':0,'new_forwards':0,'new_predictions':0,
          'new_p_values':0,'new_MC_replicates':0,'new_statistical_resamples':0})
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
