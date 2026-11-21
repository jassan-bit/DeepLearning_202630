"""One predeclared, bounded, fresh synthetic validation; zero market execution."""
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy import stats
from crossed_rmse_v2 import Settings, run, holm, require
from dgp_reference import make_squared_errors, bases_for
from legacy_mcs import functional_mcs

RUNTIME_FILES={'crossed_rmse_v2.py','run_validation.py','dgp_reference.py','legacy_mcs.py',
               'VALIDATION_PLAN.json','LEGACY_REFERENCE.json','test_crossed_rmse_v2.py','test_validation_driver.py'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def preflight(root, anchor_path):
    root=Path(root);plan_path=root/'VALIDATION_PLAN.json';manifest_path=root/'CODE_MANIFEST.json'
    plan=json.loads(plan_path.read_text());manifest=json.loads(manifest_path.read_text())
    anchor=json.loads(Path(anchor_path).read_text())
    require(anchor.get('status')=='MCS_bootstrap_t_code_and_plan_published_before_validation'
            and anchor.get('observed_by_root_before_validation') is True,
            'Immutable public anchor must be observed before validation')
    require(isinstance(anchor.get('commit'),str) and len(anchor['commit'])==40
            and all(c in '0123456789abcdef' for c in anchor['commit'])
            and anchor['commit']!='0'*40, 'Exact public commit required')
    require(anchor.get('plan_SHA256')==sha(plan_path)
            and anchor.get('source_manifest_SHA256')==sha(manifest_path), 'Public plan/source hashes differ')
    require(manifest.get('kind')=='frozen_before_new_controlled_validation','Wrong source freeze')
    require({r['path'] for r in manifest['files']}==RUNTIME_FILES
            and len(manifest['files'])==len(RUNTIME_FILES),'Complete exact runtime source inventory required')
    for row in manifest['files']:
        p=Path(row['path']);require(not p.is_absolute() and '..' not in p.parts,'Unsafe source path')
        body=(root/p).read_bytes()
        require(len(body)==row['bytes'] and hashlib.sha256(body).hexdigest()==row['sha256'],
                'Frozen source changed: '+str(p))
    require(plan['planned_replicates']==sum(c['replicates'] for c in plan['cases'])==640,
            'Planned replicate count changed')
    require(plan['validation_seed']==2026112191 and plan['old_validation_seed_not_reused']==2026112107,
            'Fresh validation namespace changed')
    require(plan['settings']['B']==999 and plan['max_wall_seconds']==300,'Budget or B changed')
    require(os.environ.get('OPENBLAS_NUM_THREADS')=='2' and os.environ.get('OMP_NUM_THREADS')=='2',
            'Run with OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2')
    return plan,manifest,anchor


def CP(k,n):
    require(n>0 and 0<=k<=n,'Invalid binomial events')
    return {'events':int(k),'replicates':int(n),'rate':float(k/n),
            'CI95':[0. if k==0 else float(stats.beta.ppf(.025,k,n-k+1)),
                    1. if k==n else float(stats.beta.ppf(.975,k+1,n-k))],
            'simultaneous_interval':False}


def rate(values):
    a=np.asarray(values,dtype=bool);return CP(int(a.sum()),len(a))


def screen(summary, nominal):
    lo,hi=summary['CI95']
    status='inflation_detected' if lo>nominal else ('exploratory_tolerance_screen_passed' if hi<=nominal+.03 else 'inconclusive')
    return {**summary,'nominal':nominal,'exploratory_tolerance':.03,'screen':status,
            'exact_nominal_level_certified':False,'market_calibration_certified':False}


def summarize(case, records):
    names=['candidate']+[f'rival_{i:02d}' for i in range(1,case['M'])]
    bases=bases_for(case);truth=bases[1:]-bases[0];null=np.abs(truth)<=1e-12
    out={'id':case['id'],'planned_replicates':case['replicates'],'completed_replicates':len(records),
         'complete':len(records)==case['replicates'],'shape':[case['R'],case['T'],case['M']],
         'analytic_longrun_theta':truth.tolist(),'practical_delta':case['delta'],
         'primary_null':'theta=0, equality of mean individual-seed population RMSE',
         'global_EPA_null_true':bool(np.all(null)), 'per_rival':[],
         'third_primary_method_accredited':False,'final_market_power_certified':False}
    if not records:return out
    for variant in ['MCS_bootstrap_t','MCS_fixed_studentizer_development_control','MCS_legacy_frozen_control_same_draws']:
        mcs=[x['result'][variant] for x in records]
        rejected=[m.get('initial_global_p') is not None and m['initial_global_p']<.10 for m in mcs]
        best={names[i] for i in np.flatnonzero(np.isclose(bases,bases.min(),atol=1e-12,rtol=0))}
        initial=rate(rejected)
        unsupported=sum(m.get('initial_global_p') is None for m in mcs)
        initial=screen(initial,.10) if null.all() else initial
        if unsupported and null.all():initial['screen']='inconclusive_unsupported_replicates'
        out[variant]={'initial_global_rejection':initial,
                      'all_true_best_retained':rate([best<=set(m['retained']) for m in mcs]),
                      'candidate_retained':rate(['candidate' in m['retained'] for m in mcs]),
                      'unique_candidate_set':rate([m['retained']==['candidate'] for m in mcs]),
                      'unsupported_replicates':unsupported,
                      'unsupported_are_not_evidence_of_calibration':True,
                      'pairwise_p_values_inferred_from_exclusion':False}
    raw=np.array([[[x['result']['pairwise'][j]['P1_equal_two_sided_p'],x['result']['pairwise'][j]['P2_equal_two_sided_p']]
                   for j in range(case['M']-1)] for x in records])
    adjusted=np.array([[[x['result']['pairwise'][j]['P1_Holm'],x['result']['pairwise'][j]['P2_Holm']]
                        for j in range(case['M']-1)] for x in records])
    for j,name in enumerate(names[1:]):
        estimates=np.array([x['result']['pairwise'][j]['theta_hat'] for x in records])
        row={'rival':name,'true_theta':float(truth[j]),'bias_against_longrun_truth':float(estimates.mean()-truth[j]),
             'empirical_SD_theta':float(estimates.std(ddof=1)) if len(records)>1 else None,
             'methods':{}}
        for k,method in enumerate(['P1','P2']):
            cis=np.array([x['result']['pairwise'][j][method+'_CI95'] or [np.nan,np.nan] for x in records])
            supported=np.isfinite(cis).all(axis=1)
            row['methods'][method]={'raw_rejection':rate(raw[:,j,k]<.05),
                                   'Holm38_rejection':rate(adjusted[:,j,k]<.05),
                                   'pointwise_CI95_coverage':rate((cis[:,0]<=truth[j])&(truth[j]<=cis[:,1])),
                                   'unsupported_intervals':int((~supported).sum()),
                                   'missing_interval_counts_as_not_covering_in_unconditional_rate':True,
                                   'mean_CI_width':float(np.mean(cis[supported,1]-cis[supported,0])) if supported.any() else None,
                                   'positive_and_lowerCI_above_delta':rate((estimates>case['delta'])&(cis[:,0]>case['delta']))}
        row['two_method_intersection_Holm38']=rate(np.all(adjusted[:,j,:]<.05,axis=1))
        out['per_rival'].append(row)
    if null.any():
        out['known_two_method_subset_FWER_Holm38']=screen(rate(np.any(adjusted[:,null,:]<.05,axis=(1,2))),.05)
        if any(not row['nondegenerate'] for item in records for row in item['result']['pairwise']):
            out['known_two_method_subset_FWER_Holm38']['screen']='inconclusive_unsupported_replicates'
    if (~null).any():out['two_method_joint_alternative_rejection_Holm38']=rate(np.all(adjusted[:,~null,:]<.05,axis=(1,2)))
    out['two_method_subset_is_complete_three_method_battery']=False
    out['paired_variant_diagnostics']=[]
    if null.all():
        new=np.array([x['result']['MCS_bootstrap_t'].get('initial_global_p') is not None
                      and x['result']['MCS_bootstrap_t']['initial_global_p']<.10 for x in records])
        for variant in ['MCS_legacy_frozen_control_same_draws','MCS_fixed_studentizer_development_control']:
            old=np.array([x['result'][variant].get('initial_global_p') is not None
                          and x['result'][variant]['initial_global_p']<.10 for x in records])
            old_only=int((old&~new).sum());new_only=int((new&~old).sum());n=old_only+new_only
            unsupported=sum(x['result'][variant].get('initial_global_p') is None
                            or x['result']['MCS_bootstrap_t'].get('initial_global_p') is None for x in records)
            p=1. if n==0 or unsupported else float(stats.binomtest(old_only,n,.5).pvalue)
            out['paired_variant_diagnostics'].append({'comparison':'bootstrap_t_vs_'+variant,
                    'old_only_rejection':old_only,'new_only_rejection':new_only,'exact_McNemar_two_sided_p':p,
                    'unsupported_pairs':unsupported,
                    'status':'inconclusive_unsupported_pairs_reserved_p1' if unsupported else 'paired_diagnostic_computed',
                    'family':'6predeclared mechanism comparisons across3nullscenarios; Holm6 applied in final summary'})
    return out


def execute(root, output, anchor_path):
    root=Path(root);output=Path(output);plan,manifest,anchor=preflight(root,anchor_path)
    require(not output.exists(),'New output directory required; no overwrite or favorable rerun')
    output.mkdir(parents=True);started=utc();start=time.monotonic();settings=Settings(**plan['settings'])
    receipt={'status':'running','started_at_utc':started,'plan_SHA256':sha(root/'VALIDATION_PLAN.json'),
             'source_manifest_SHA256':sha(root/'CODE_MANIFEST.json'),'public_anchor_commit':anchor['commit'],
             'planned_replicates':640,'planned_crossed_bootstrap_draws':639360,'completed_replicates':0,
             'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                        'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS')},
             'errors':[],'new_market_fits':0,'new_market_predictions':0,'R3_accredited':False,
             'source_implementation_changed_after_results':False,'human_execution_claimed':False}
    write(output/'execution_receipt.json',receipt);summaries=[];stop=False
    with (output/'per_replicate.jsonl.gz').open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as log:
        for index,case in enumerate(plan['cases']):
            records=[]
            for replicate in range(case['replicates']):
                if time.monotonic()-start>=plan['max_wall_seconds']:stop=True;break
                dgp_rng=np.random.default_rng(np.random.SeedSequence([plan['validation_seed'],index,replicate,0]))
                test_rng=np.random.default_rng(np.random.SeedSequence([plan['validation_seed'],index,replicate,1]))
                item={'case':case['id'],'replicate':replicate,'stage':'fresh_controlled_validation'}
                try:
                    q,truth,bases=make_squared_errors(case,dgp_rng)
                    names=['candidate']+[f'rival_{i:02d}' for i in range(1,case['M'])]
                    item.update({'q_SHA256':hashlib.sha256(q.tobytes(order='C')).hexdigest(),
                                 'analytic_longrun_theta':truth.tolist(),'bases':bases.tolist(),
                                 'result':run(q,names,0,test_rng,settings,legacy_mcs=functional_mcs)})
                    records.append(item);receipt['completed_replicates']+=1
                except Exception as error:
                    item['error']={'type':type(error).__name__,'message':str(error)}
                    receipt['errors'].append({'case':case['id'],'replicate':replicate,**item['error']});stop=True
                log.write((json.dumps(item,sort_keys=True,allow_nan=False)+'\n').encode());log.flush()
                if stop:break
            summary=summarize(case,records);summaries.append(summary);write(output/(case['id']+'_summary.json'),summary)
            receipt['elapsed_seconds']=time.monotonic()-start;write(output/'execution_receipt.json',receipt)
            print(json.dumps({'case':case['id'],'completed':len(records),'planned':case['replicates'],
                              'total_completed':receipt['completed_replicates'],'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
            if stop:break
    comparisons=[(s,d) for s in summaries for d in s.get('paired_variant_diagnostics',[])]
    if comparisons:
        p=holm(np.array([d['exact_McNemar_two_sided_p'] for _,d in comparisons])) if len(comparisons)==6 else None
        for i,(s,d) in enumerate(comparisons):d['Holm6_p']=None if p is None else float(p[i])
        for s in summaries:write(output/(s['id']+'_summary.json'),s)
    complete=receipt['completed_replicates']==640 and not receipt['errors']
    receipt.update({'status':'completed' if complete else ('error' if receipt['errors'] else 'resource_stop'),
                    'finished_at_utc':utc(),'elapsed_seconds':time.monotonic()-start,
                    'actual_crossed_bootstrap_draws':receipt['completed_replicates']*settings.B,
                    'old_control_extra_bootstrap_draws':0,'source_hashes_still_match':all(sha(root/r['path'])==r['sha256'] for r in manifest['files'])})
    write(output/'execution_receipt.json',receipt)
    overall={'status':receipt['status'],'complete':complete,'cases':summaries,
             'source_manifest_SHA256':receipt['source_manifest_SHA256'],'public_anchor_commit':anchor['commit'],
             'three_primary_pairwise_tests_accredited':False,'complete_three_way_power_computed':False,
             'final_market_calibration_certified':False,'final_market_power_certified':False,
             'R3_accredited':False,'new_market_fits':0,'new_market_predictions':0,
             'bootstrap_variants_are_not_three_distinct_primary_tests':True}
    write(output/'validation_summary.json',overall)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--anchor',type=Path);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    anchor=a.anchor or a.root/'PUBLIC_ANCHOR.json'
    if a.execute:
        require(a.output is not None,'--output is required with --execute')
        print(json.dumps(execute(a.root,a.output,anchor),sort_keys=True),flush=True)
    else:
        plan,_,anchor_value=preflight(a.root,anchor)
        print(json.dumps({'status':'preflight_passed_without_execution','planned_replicates':plan['planned_replicates'],
                          'public_anchor_commit':anchor_value['commit'],'new_validation_replicates':0}),flush=True)
