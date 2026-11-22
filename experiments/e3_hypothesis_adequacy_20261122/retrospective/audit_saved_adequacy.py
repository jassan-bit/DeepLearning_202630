"""New V5/input-only adequacy diagnostics from immutable saved point forecasts.

AI assistance declared. No market I/O, model imports, inference or training.
Auxiliary OLS diagnostics and exploratory resampling are counted separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
from scipy.stats import norm
import scipy
import statsmodels
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

PIN = 'beabedeb63ff10324b310ba8b86105505c3be783216e38770a092a7be438a17c'
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[1]/'e3_input_only_candidate_20261121'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Holm requires all frozen family positions and valid p values')
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1., np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    result = np.empty_like(p); result[order] = adjusted
    return result


def load_saved(path):
    raw = path.read_bytes()
    if sha(raw) != PIN:
        raise ValueError('ZIP differs from immutable pilot evidence')
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive members')
    read = lambda name: json.loads(archive.read(name))
    plan, prepared, admission, receipt = [read(p) for p in ('plan.json','prepared.json','admission.json','pilot_result.json')]
    assert sha(archive.read('plan.json')) == prepared['plan_file_sha256']
    assert sha(canon(plan)) == prepared['plan_sha256'] == receipt['plan_sha256']
    assert sha(canon(admission)) == plan['admission_sha256']
    for name, record in receipt['files'].items():
        value = archive.read(name)
        assert sha(value) == record['sha256'] and len(value) == record['bytes']
    assert read('PUBLIC_SOURCE_ANCHOR.json')['source_commit'] == 'c4ca86f1431afeaa958ff0b544b453ab17a4d7b8'
    rows = list(csv.DictReader(io.StringIO(archive.read('predictions.csv').decode())))
    dates = admission['assessment_dates']
    assert len(dates) == 55 and dates[0] == '2024-09-07' and dates[-1] == '2024-10-31'
    assert all((dt.date.fromisoformat(b)-dt.date.fromisoformat(a)).days == 1 for a,b in zip(dates,dates[1:]))
    assert len(rows) == 1760
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['recipe'],int(row['seed']) if row['seed'] else None].append(row)
    expected = {(r,s) for r in plan['recipes'] for s in plan['seeds']} | {(r,None) for r in plan['analytical_controls']}
    assert set(grouped) == expected and len(grouped) == 32
    target = None; predictions = {}; errors = {}; trajectories = []
    for recipe in plan['recipes']+plan['analytical_controls']:
        seeds = plan['seeds'] if recipe in plan['recipes'] else [None]
        local_predictions, local_errors = [], []
        for seed in seeds:
            records = sorted(grouped[recipe,seed], key=lambda r:r['date'])
            assert [r['date'] for r in records] == dates
            y = np.asarray([float(r['target']) for r in records])
            p = np.asarray([float(r['prediction']) for r in records])
            assert np.isfinite(y).all() and np.isfinite(p).all() and (y >= 0).all() and (p >= 0).all()
            if target is None: target=y
            else: assert np.array_equal(y,target)
            local_predictions.append(p); local_errors.append(p-y)
            trajectories.append(dict(recipe=recipe, seed=seed, prediction=p, error=p-y))
        predictions[recipe] = np.stack(local_predictions); errors[recipe] = np.stack(local_errors)
    return archive, plan, admission, dates, target, predictions, errors, trajectories


def diagnostic_family(trajectories, target, scale):
    rows=[]; bias=[]; errors=[]
    for trajectory in trajectories:
        e=trajectory['error']; p=trajectory['prediction']
        identity=dict(recipe=trajectory['recipe'],seed=trajectory['seed'],n_dates=len(e))
        for lag in (7,14):
            fit=OLS(e,np.ones((len(e),1))).fit(cov_type='HAC',cov_kwds=dict(maxlags=lag,use_correction=True),use_t=False)
            se=float(fit.bse[0]); value=float(e.mean()); radius=float(norm.ppf(.975))*se
            bias.append(dict(**identity,HAC_bandwidth=lag,mean_prediction_minus_target=value,
                standard_error=se,CI95_low=value-radius,CI95_high=value+radius,
                interval_scope='pointwise asymptotic, fixed stored predictor, not simultaneous'))
        for test in ('LB7','LB14','ARCH7','MZ_HAC14'):
            try:
                if test.startswith('LB'):
                    lag=int(test[2:]); result=acorr_ljungbox(e,lags=[lag],model_df=0,return_df=True)
                    stat=float(result['lb_stat'].iloc[0]); pvalue=float(result['lb_pvalue'].iloc[0]); coefficients={}
                elif test=='ARCH7':
                    stat,pvalue,_,_=het_arch(e,nlags=7,ddof=0)
                    stat,pvalue=float(stat),float(pvalue); coefficients={}
                else:
                    x=add_constant(p/scale,has_constant='add')
                    if np.linalg.matrix_rank(x) != 2: raise ValueError('Degenerate prediction design')
                    fit=OLS(target/scale,x).fit(cov_type='HAC',cov_kwds=dict(maxlags=14,use_correction=True),use_t=False)
                    result=fit.wald_test((np.eye(2),np.asarray([0.,1.])),use_f=False,scalar=True)
                    stat,pvalue=float(result.statistic),float(result.pvalue)
                    coefficients=dict(intercept_sigma7=float(fit.params[0]*scale),slope=float(fit.params[1]))
                if not np.isfinite(stat) or not np.isfinite(pvalue) or not 0 <= pvalue <= 1:
                    raise ValueError('Nonfinite diagnostic')
                row=dict(**identity,test=test,statistic=stat,p_value=pvalue,supported=True,**coefficients)
            except (ValueError,ZeroDivisionError,np.linalg.LinAlgError) as exc:
                row=dict(**identity,test=test,statistic=None,p_value=1.,supported=False,
                    limitation=type(exc).__name__+'; retained conservative p=1 family position')
            rows.append(row)
    assert len(rows) == 128
    for row,adjusted in zip(rows,holm([r['p_value'] for r in rows])):
        row.update(p_Holm128=float(adjusted),reject_Holm128_05=bool(adjusted < .05),
                   scope='exploratory diagnostic; classical/asymptotic reference not finite-sample validated')
    return rows,bias


def circular_indices(rng,n,block,B):
    starts=rng.integers(0,n,size=(B,(n+block-1)//block))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(B,-1)[:,:n]


def mean_individual_metrics(errors, indices=None):
    # Averaging happens after each seed's root. Never pool seeds or average forecasts.
    e = errors if indices is None else np.take(errors,indices,axis=1)
    return np.sqrt(np.mean(e*e,axis=-1)).mean(axis=0),np.mean(np.abs(e),axis=-1).mean(axis=0)


def temporal_intervals(errors,plan):
    c=plan['candidate_recipe']; B=plan['temporal_intervals']['replicates_each']
    rng=np.random.default_rng(plan['temporal_intervals']['rng_seed']); n=errors[c].shape[1]
    points={r:mean_individual_metrics(e) for r,e in errors.items()}; rows=[]
    for length in plan['temporal_intervals']['block_lengths']:
        indices=circular_indices(rng,n,length,B)
        draws={r:mean_individual_metrics(e,indices) for r,e in errors.items()}
        for control in errors:
            if control==c:continue
            delta=draws[c][0]-draws[control][0]
            mae_delta=draws[c][1]-draws[control][1]
            relative=1-draws[c][0]/draws[control][0]
            lo,hi=np.quantile(delta,[.025,.975],method='linear')
            ml,mh=np.quantile(mae_delta,[.025,.975],method='linear')
            rl,rh=np.quantile(relative,[.025,.975],method='linear')
            rows.append(dict(candidate=c,control=control,block_length=length,bootstrap_replicates=B,
                dates=n,fixed_candidate_seeds=errors[c].shape[0],fixed_control_trajectories=errors[control].shape[0],
                delta_mean_individual_RMSE=float(points[c][0]-points[control][0]),CI95_pointwise_low=float(lo),CI95_pointwise_high=float(hi),
                relative_mean_RMSE_gain=float(1-points[c][0]/points[control][0]),relative_CI95_low=float(rl),relative_CI95_high=float(rh),
                delta_mean_individual_MAE=float(points[c][1]-points[control][1]),MAE_CI95_low=float(ml),MAE_CI95_high=float(mh),
                zero_in_RMSE_interval=bool(lo <= 0 <= hi),initialization_scope='conditional on stored seeds; no seed-population inference',
                temporal_scope='pointwise exploratory circular-block interval; coverage unvalidated; not confirmation',p_value=None))
    return rows


def run(input_path,reuse,out,plan_path):
    assert not out.exists(), 'Use a new output directory; result sets are immutable'
    plan=json.loads(plan_path.read_text()); plan_raw=plan_path.read_bytes()
    assert plan['source_zip_sha256']==PIN and plan['residual_family_size']==128
    assert plan['temporal_intervals']['replicates_each']==3999 and plan['temporal_intervals']['block_lengths']==[7,14]
    started=utc(); cpu=time.process_time(); wall=time.perf_counter()
    archive,source_plan,admission,dates,target,predictions,errors,trajectories=load_saved(input_path)
    assert set(errors)==set(plan['source_recipes']) and len(source_plan['seeds'])==5
    source_files=[input_path,plan_path,Path(__file__),reuse/'AUDIT_SUMMARY.json',reuse/'recipe_summary_recomputed.csv',reuse/'selection_assessment_summary.csv',reuse/'selection_assessment_recomputed.csv']
    inventory=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in source_files]
    old_metrics={r['recipe']:r for r in csv.DictReader((reuse/'recipe_summary_recomputed.csv').open())}
    for recipe,e in errors.items():
        rmse,mae=mean_individual_metrics(e)
        assert np.isclose(rmse,float(old_metrics[recipe]['mean_RMSE']),rtol=1e-12,atol=1e-15)
        assert np.isclose(mae,float(old_metrics[recipe]['mean_MAE']),rtol=1e-12,atol=1e-15)
    diagnostics,bias=diagnostic_family(trajectories,target,admission['y_scale'])
    intervals=temporal_intervals(errors,plan)
    error_rows=[dict(recipe=t['recipe'],seed=t['seed'],date=date,target=float(y),prediction=float(p),error=float(e),squared_error=float(e*e))
        for t in trajectories for date,y,p,e in zip(dates,target,t['prediction'],t['error'])]
    diagnostic_summary=[dict(test=test,positions=sum(r['test']==test for r in diagnostics),
        supported=sum(r['test']==test and r['supported'] for r in diagnostics),
        rejects_Holm128=sum(r['test']==test and r['reject_Holm128_05'] for r in diagnostics))
        for test in ('LB7','LB14','ARCH7','MZ_HAC14')]
    candidate_diagnostics=[r for r in diagnostics if r['recipe']==plan['candidate_recipe']]
    summary=dict(version=plan['version'],scope=plan['scope'],candidate_recipe=plan['candidate_recipe'],
        source_zip_sha256=PIN,source_code_commit=plan['source_code_commit'],source_result_commit=plan['source_result_commit'],
        analysis_plan_sha256=sha(plan_raw),date_start=dates[0],date_end=dates[-1],n_dates=len(dates),
        n_trajectories=len(trajectories),n_point_predictions=len(error_rows),neural_seeds=source_plan['seeds'],
        neural_recipes=source_plan['recipes'],analytical_controls=source_plan['analytical_controls'],
        diagnostics=diagnostic_summary,candidate_diagnostic_rejections={test:sum(r['test']==test and r['reject_Holm128_05'] for r in candidate_diagnostics) for test in ('LB7','LB14','ARCH7','MZ_HAC14')},
        temporal_intervals=intervals,old_analysis_reused=['source integrity','point metrics','selection/assessment gaps','seed sensitivity','posthoc month/regime/leaveout descriptions'],
        forecast_model_fits=0,new_forward_passes=0,new_forecast_points=0,new_market_downloads=0,
        auxiliary_OLS_fits=128,auxiliary_OLS_breakdown=dict(bias_HAC=64,ARCH_auxiliary=32,Mincer_Zarnowitz=32),
        temporal_resampling=dict(distinct_time_index_matrices=2,replicates_per_matrix=3999,paired_contrasts=7,seed_resampling=False),
        full_Tables_A_B_reconstructed=False,predictive_interval_coverage_reconstructed=False,
        coverage_missing_reason='Point forecasts only; no saved quantiles, calibration scores or full forecast law for this candidate.',
        full_fit_gap_missing_reason='No saved fit or selection point forecasts and no selected-checkpoint fit loss; training mean-batch losses are not final-fit risk. Selection used for early stopping and assessment calendar differs.',
        current_data_eligibility='Exposed pilot with only8 recipes, not21-model planning family, 55 dates and5 seeds; not confirmation or power-sizing input.',
        finite_sample_diagnostic_validation=False,simultaneous_interval_coverage_claimed=False,R3_accredited=False,
        AI_assistance_declared=True)
    out.mkdir(parents=True)
    write_csv(out/'residuals.csv',error_rows); write_csv(out/'diagnostics_Holm128.csv',diagnostics)
    write_csv(out/'bias_HAC_pointwise.csv',bias); write_csv(out/'temporal_RMSE_MAE_intervals.csv',intervals)
    write_csv(out/'diagnostic_summary.csv',diagnostic_summary)
    write_json(out/'input_manifest.json',dict(files=inventory,ZIP_members_hashed=len(archive.namelist()),
        ZIP_members={name:sha(archive.read(name)) for name in archive.namelist()}))
    write_json(out/'summary.json',summary)
    for item in inventory:assert sha(Path(item['path']).read_bytes())==item['sha256']
    receipt=dict(started_utc=started,finished_utc=utc(),wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,statsmodels=statsmodels.__version__,
        analysis_plan_sha256=sha(plan_raw),script_sha256=sha(Path(__file__).read_bytes()),status='EXPLORATORY_EXECUTION_COMPLETED',
        sources_unchanged=True,forecast_training=False,new_model_inference=False,test_opened=False,
        preregistration_claimed=False,AI_assistance_declared=True)
    write_json(out/'execution_receipt.json',receipt)
    write_json(out/'OUTPUT_MANIFEST.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(dict(diagnostics=diagnostic_summary,candidate=summary['candidate_diagnostic_rejections'],
        candidate_global_intervals=[r for r in intervals if r['control']=='global_mse'],cpu_seconds=receipt['cpu_seconds']),indent=2))


def verify(input_path,out,plan_path):
    assert sha(input_path.read_bytes())==PIN
    summary=json.loads((out/'summary.json').read_text())
    assert summary['analysis_plan_sha256']==sha(plan_path.read_bytes())
    manifest=json.loads((out/'OUTPUT_MANIFEST.json').read_text())
    for name,record in manifest['files'].items():
        raw=(out/name).read_bytes(); assert sha(raw)==record['sha256'] and len(raw)==record['bytes']
    print(json.dumps(dict(status='EXISTING_OUTPUTS_HASH_VERIFIED',files=len(manifest['files']),no_resampling=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',type=Path,default=ORIGINAL/'inputs/public_results.zip')
    parser.add_argument('--reuse',type=Path,default=ORIGINAL/'analysis')
    parser.add_argument('--plan',type=Path,default=HERE/'analysis_plan.json')
    parser.add_argument('--output',type=Path,default=HERE/'results')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:verify(args.input,args.output,args.plan)
    else:run(args.input,args.reuse,args.output,args.plan)
