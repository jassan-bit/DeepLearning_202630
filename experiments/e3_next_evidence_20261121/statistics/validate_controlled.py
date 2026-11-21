"""Bounded prospective controlled-DGP experiment; no market model is executed."""
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
from scipy import stats
import scipy
from temporal_battery import P1,P2,Settings,run_battery,file_sha,require

PLAN_SHA='20d2f73adff064d47d19b2394e708e5a84515eed410ee81b258c2a8226c710f5'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def bases_for(case):
    base=np.ones(case['M']);d=case['delta'];kind=case['theta']
    if kind=='equal':base[1:]=1.
    elif kind=='boundary':base[1:]=1+d
    elif kind=='interior':base[1:]=1+d-.05
    elif kind=='partial':base[1:10]=1+d;base[10:]=1+d+.20
    elif kind=='alternative':base[1:]=1+d+.20
    else:raise ValueError('Unknown declared truth scenario')
    return base


def unit_second_moment_AR(rng,phi,T,features,innovation,nu):
    """Unit variance at every step; t innovations use a 128-point burn.

    Gaussian initialization is exactly stationary for the Gaussian AR case.
    In the t5 case only the second moment is exactly stationary; the full
    stationary distribution is approximated by burn-in, explicitly not exact.
    """
    require(abs(phi)<1,'AR stationary second moment requires abs(phi)<1')
    state=rng.normal(size=features)
    out=np.empty((T,features));scale=np.sqrt(1-phi*phi)
    for t in range(T+128):
        if innovation=='gaussian':shock=rng.normal(size=features)
        elif innovation=='student_t':
            require(nu>4,'Declared heavy-tail fourth moment must exist')
            shock=rng.standard_t(nu,size=features)*np.sqrt((nu-2)/nu)
        else:raise ValueError('Unknown innovations')
        state=phi*state+scale*shock
        if t>=128:out[t-128]=state
    return out


def make_squared_errors(case,rng):
    S,T,M=case['R'],case['T'],case['M'];bases=bases_for(case)
    latent=np.sqrt(.5)*rng.normal(size=(S,1))+np.sqrt(.5)*rng.normal(size=(S,M))
    sigma=case['seed_sigma']
    amplitude=bases[None,:]*np.exp(sigma*latent-sigma*sigma/2)
    weights=np.asarray(case['weights'],dtype=float)
    require(weights.shape==(4,) and (weights>=0).all(),'Invalid factor weights')
    if weights.sum()==0:
        z=np.ones((S,T,M))
    else:
        require(abs(float(weights.sum())-1)<1e-12,'Factor variances must sum to one')
        values=unit_second_moment_AR(rng,case['phi'],T,1+M+S+S*M,
                                    case['innovations'],case.get('nu'))
        global_time=values[:,0]
        model_time=values[:,1:1+M]
        seed_time=values[:,1+M:1+M+S]
        idio=values[:,1+M+S:].reshape(T,S,M).transpose(1,0,2)
        z=(np.sqrt(weights[0])*global_time[None,:,None]
           +np.sqrt(weights[1])*model_time[None,:,:]
           +np.sqrt(weights[2])*seed_time.T[:,:,None]
           +np.sqrt(weights[3])*idio)
    q=(amplitude[:,None,:]*z)**2
    return q,bases[1:]-bases[0],bases


def CP(k,n,alpha=.05):
    if n==0:return {'events':0,'replicates':0,'rate':None,'CI95':None}
    low=0. if k==0 else float(stats.beta.ppf(alpha/2,k,n-k+1))
    high=1. if k==n else float(stats.beta.ppf(1-alpha/2,k+1,n-k))
    return {'events':int(k),'replicates':int(n),'rate':k/n,'CI95':[low,high]}


def event_summary(events,nominal=None):
    a=np.asarray(events,dtype=bool);x=CP(int(a.sum()),len(a))
    if nominal is not None and len(a):
        lo,hi=x['CI95']
        x.update(nominal=nominal,
                 exploratory_inflation_signal=bool(lo>nominal),
                 exploratory_tolerance_screen_passed=bool(hi<=nominal+.05),
                 simultaneous_validation_claimed=False)
    return x


def summarize_case(case,records):
    names=['candidate']+[f'rival_{i:02d}' for i in range(1,case['M'])]
    bases=bases_for(case);truth=bases[1:]-bases[0]
    null=truth<=case['delta']+1e-12
    out={'id':case['id'],'R':case['R'],'T':case['T'],'models':case['M'],
         'planned_MC':case['mc_replicates'],'completed_MC':len(records),
         'complete':len(records)==case['mc_replicates'],
         'analytic_truth':truth.tolist(),'margin':case['delta'],
         'null_rivals':[names[i+1] for i in np.flatnonzero(null)],
         'alternative_rivals':[names[i+1] for i in np.flatnonzero(~null)],
         'per_rival':[],'complete_three_method_power_computed':False,
         'final_market_calibration_or_power_certified':False}
    if not records:return out
    raw=np.array([[[r['methods'][method]['p_raw'] for method in (P1,P2)]
                  for r in item['result']['pairwise']] for item in records])
    adjusted=np.array([[[r['methods'][method]['p_Holm_two_method_family'] for method in (P1,P2)]
                  for r in item['result']['pairwise']] for item in records])
    reserved=np.array([[[r['methods'][method]['p_Holm_reserved_third_slot'] for method in (P1,P2)]
                  for r in item['result']['pairwise']] for item in records])
    theta=np.array([[r['theta_hat'] for r in item['result']['pairwise']] for item in records])
    for k,name in enumerate(names[1:]):
        row={'rival':name,'truth_theta':float(truth[k]),'primary_margin_null_true':bool(null[k]),
             'mean_estimated_theta':float(theta[:,k].mean()),
             'bias_against_analytic_truth':float(theta[:,k].mean()-truth[k]),
             'empirical_SD_theta':float(theta[:,k].std(ddof=1)) if len(records)>1 else None,
             'methods':{}}
        for j,method in enumerate((P1,P2)):
            ci=np.array([item['result']['pairwise'][k]['methods'][method]['CI95'] for item in records])
            row['methods'][method]={
                'unadjusted_rejection':event_summary(raw[:,k,j]<.05,.05 if null[k] else None),
                'Holm38_two_method_rejection':event_summary(adjusted[:,k,j]<.05),
                'Holm57_reserved_third_slot_rejection':event_summary(reserved[:,k,j]<.05),
                'pointwise_CI95_coverage':event_summary((ci[:,0]<=truth[k])&(truth[k]<=ci[:,1])),
                'mean_CI_width':float(np.mean(ci[:,1]-ci[:,0]))}
        row['two_method_intersection_Holm38']=event_summary(np.all(adjusted[:,k,:]<.05,axis=1))
        row['two_method_intersection_is_full_battery']=False
        row['mean_variance_components']={key:float(np.mean([item['result']['pairwise'][k]['variance_components'][key]
                                                    for item in records]))
             for key in ('seed_variance','time_variance','interaction_variance_diagnostic',
                         'interaction_corrected_variance_diagnostic','variance_used')}
        out['per_rival'].append(row)
    if null.any():
        out['known_two_method_subset_FWER_Holm38']=event_summary(np.any(adjusted[:,null,:]<.05,axis=(1,2)),.05)
        out['known_two_method_subset_FWER_57_reserved']=event_summary(np.any(reserved[:,null,:]<.05,axis=(1,2)),.05)
        out['two_method_IUT_false_claim_FWER_Holm38']=event_summary(np.any(np.all(adjusted[:,null,:]<.05,axis=2),axis=1),.05)
    if (~null).any():
        out['two_method_joint_alternative_rejection_Holm38']=event_summary(np.all(adjusted[:,~null,:]<.05,axis=(1,2)))
    mcs=[item['result']['global_MCS'] for item in records]
    true_best={names[i] for i in np.flatnonzero(np.isclose(bases,bases.min(),atol=1e-12,rtol=0))}
    out['MCS_global']={
        'scope':'GlobalEPA/modelset only; not third practical-margin p',
        'true_best_models':sorted(true_best),
        'all_true_best_retained':event_summary([true_best<=set(x['retained']) for x in mcs]),
        'at_least_one_true_best_retained':event_summary([bool(true_best&set(x['retained'])) for x in mcs]),
        'candidate_retained':event_summary(['candidate' in x['retained'] for x in mcs]),
        'global_equal_predictive_ability_null_true':bool(np.allclose(bases,bases[0],atol=1e-12,rtol=0)),
        'initial_global_rejection':event_summary([x['initial_global_p'] is not None and x['initial_global_p']<.10 for x in mcs],
                       .10 if np.allclose(bases,bases[0],atol=1e-12,rtol=0) else None)}
    return out


def run(plan_path,out,anchor_path,freeze_path):
    require(file_sha(plan_path)==PLAN_SHA,'Frozen plan changed')
    plan=json.loads(plan_path.read_text());anchor=json.loads(anchor_path.read_text())
    require(anchor.get('status')=='controlled_validation_plan_immutably_published_before_Monte_Carlo'
            and anchor.get('plan_SHA256')==PLAN_SHA,'Plan anchor absent')
    freeze=json.loads(freeze_path.read_text())
    require(freeze.get('kind')=='CONTROLLED_VALIDATION_IMPLEMENTATION_FROZEN_BEFORE_RESULTS',
            'Implementation freeze absent')
    for row in freeze['files']:
        require(file_sha(freeze_path.parent/row['path'])==row['sha256'],'Implementation changed: '+row['path'])
    require(not out.exists(),'New validation output directory required; no overwrite/rerun')
    out.mkdir(parents=True)
    start=time.monotonic();started=utc();settings=Settings(
        bootstrap_replicates=plan['algorithms']['bootstrap_replicates'],
        stationary_mean_block=plan['algorithms']['stationary_mean_block'],
        hac_lag=plan['algorithms']['HAC_Bartlett_lag'])
    runtime={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
             'OPENBLAS_NUM_THREADS':os.environ.get('OPENBLAS_NUM_THREADS'),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS')}
    receipt={'status':'running','started_at_utc':started,'plan_SHA256':PLAN_SHA,
             'plan_anchor_commit':anchor['commit'],'implementation_freeze_SHA256':file_sha(freeze_path),
             'runtime':runtime,'new_market_fits':0,'new_market_predictions':0,
             'synthetic_controlled_DGP':True,'three_tests_per_contrast_gate':False,
             'R3_accredited':False,'final_market_power_certified':False,
             'human_execution_claimed':False,'completed_replicates':0,'errors':[]}
    write_json(out/'execution_receipt.json',receipt)
    summaries=[];stop=False;completed=0
    raw_path=out/'per_replicate.jsonl.gz'
    with raw_path.open('wb') as raw_file,gzip.GzipFile(filename='',mode='wb',fileobj=raw_file,mtime=0) as log:
        for case_index,case in enumerate(plan['Monte_Carlo']['cases']):
            records=[]
            for replicate in range(case['mc_replicates']):
                if stop or time.monotonic()-start>=plan['Monte_Carlo']['max_wall_seconds']:
                    stop=True;break
                item={'case':case['id'],'replicate':replicate,'stage':'controlled_DGP_validation'}
                try:
                    dgp_rng=np.random.default_rng(np.random.SeedSequence([plan['Monte_Carlo']['master_seed'],case_index,replicate,0]))
                    test_rng=np.random.default_rng(np.random.SeedSequence([plan['Monte_Carlo']['master_seed'],case_index,replicate,1]))
                    q,truth,bases=make_squared_errors(case,dgp_rng)
                    item.update(q_SHA256=hashlib.sha256(q.tobytes(order='C')).hexdigest(),
                                analytic_truth=truth.tolist(),bases=bases.tolist(),
                                result=run_battery(q,['candidate']+[f'rival_{i:02d}' for i in range(1,case['M'])],
                                                   0,case['delta'],test_rng,settings))
                    records.append(item);completed+=1
                except Exception as error:
                    item['error']={'type':type(error).__name__,'message':str(error)}
                    receipt['errors'].append({'case':case['id'],'replicate':replicate,**item['error']})
                    stop=True
                log.write((json.dumps(item,sort_keys=True,allow_nan=False)+'\n').encode());log.flush()
                if stop:break
            summary=summarize_case(case,records);summaries.append(summary)
            write_json(out/(case['id']+'_summary.json'),summary)
            receipt.update(completed_replicates=completed,elapsed_seconds=time.monotonic()-start)
            write_json(out/'execution_receipt.json',receipt)
            print(json.dumps({'case':case['id'],'completed_MC':len(records),'planned_MC':case['mc_replicates'],
                              'total_completed':completed,'elapsed_seconds':round(time.monotonic()-start,3)}),flush=True)
    planned=plan['Monte_Carlo']['total_replicates_planned']
    receipt.update(status=('failed_preserved' if receipt['errors'] else 'complete_controlled_validation' if completed==planned
                           else 'resource_bounded_partial_validation_preserved'),
                   finished_at_utc=utc(),elapsed_seconds=time.monotonic()-start,
                   completed_replicates=completed,planned_replicates=planned,
                   raw_results={'file':raw_path.name,'bytes':raw_path.stat().st_size,'sha256':file_sha(raw_path)},
                   all_cases_complete=completed==planned,
                   final_calibration_approval=False,
                   limitation='Two proposed directional extensions plus globalMCS; finite controlled scenarios and individualMC intervals do not accredit a complete or universally calibrated R3 battery')
    write_json(out/'execution_receipt.json',receipt)
    write_json(out/'validation_summary.json',{'execution':receipt,'cases':summaries,
       'CP_intervals_are_individual_not_simultaneous':True,
       'market_planning_variances_filled':False,'market_final_R_or_T_selected':False,
       'full_three_method_power_computed':False,'confirmation_enabled':False})
    print('CONTROLLED_VALIDATION_RECEIPT_BEGIN');print(json.dumps(receipt,indent=2));print('CONTROLLED_VALIDATION_RECEIPT_END')
    return 0 if receipt['status']=='complete_controlled_validation' else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--anchor',type=Path,required=True);parser.add_argument('--freeze',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(args.plan,args.out,args.anchor,args.freeze))
