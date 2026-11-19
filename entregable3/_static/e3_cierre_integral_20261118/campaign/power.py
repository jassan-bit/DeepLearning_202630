"""Conditional prospective simulation from NEW common development predictions.

Three one-sided seed-level location procedures are modeled: paired t, Wilcoxon
signed rank and exact sign test. Under the explicit multivariate normal seed-
gain planning model, their location/mean/median alternatives agree. This is not
a temporal-independence test, a domain-shift guarantee or automatic R3 evidence.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import stats

_spec=importlib.util.spec_from_file_location('e3_common_pilot',Path(__file__).with_name('pilot.py'))
_pilot=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_pilot)
FAMILY=_pilot.FAMILY
CANDIDATE=_pilot.CANDIDATE
RIVALS=tuple(name for name in FAMILY if name!=CANDIDATE)
PROCEDURES=('paired_t_location','wilcoxon_signed_rank_location','exact_sign_location')


def holm(pvalues):
    """One global family, last axis; no per-rival multiplicity shortcut."""
    p=np.asarray(pvalues,dtype=np.float64)
    if p.shape[-1]<1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Finite probability values required')
    order=np.argsort(p,axis=-1,kind='stable')
    ranked=np.take_along_axis(p,order,axis=-1)
    adjusted=np.minimum(1.,np.maximum.accumulate(ranked*np.arange(p.shape[-1],0,-1),axis=-1))
    result=np.empty_like(p);np.put_along_axis(result,order,adjusted,axis=-1)
    return result


def _wilson(successes,n,z):
    p=np.asarray(successes,dtype=float)/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return np.maximum(0.,center-half),np.minimum(1.,center+half)


def pilot_gains(directory,minimum_pilot_seeds=10):
    """Return actual paired relative RMSE reductions, aligned by seed and date."""
    if type(minimum_pilot_seeds)is not int or minimum_pilot_seeds<10:
        raise ValueError('At least ten pilot seeds are required for this planner')
    directory=Path(directory);request=json.loads((directory/'request.json').read_text())
    receipt=json.loads((directory/'pilot_receipt.json').read_text())
    reasons=[]
    if request.get('test_only'):reasons.append('Synthetic fixtures cannot dimension a real campaign')
    if request.get('suite')!='full_family':reasons.append('The full 18-system common pilot is required')
    if request.get('family')!=list(FAMILY):reasons.append('The full family was changed or reordered')
    if receipt.get('status')!='completed':reasons.append('Every planned pilot job must complete')
    path=directory/'development_predictions.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt.get('predictions_sha256'):
        reasons.append('Prediction panel hash differs from receipt')
    seeds=request.get('seeds',[])
    if len(seeds)<minimum_pilot_seeds:
        reasons.append(f'Only {len(seeds)} paired pilot seeds; need at least {minimum_pilot_seeds}')
    if reasons:return None,{'status':'not_ready','blockers':reasons,'real_campaign_allowed':False}
    with path.open() as stream:records=list(csv.DictReader(stream))
    by={}
    for row in records:
        key=(row['model'],int(row['seed']))
        rows=by.setdefault(key,{})
        if row['date'] in rows:raise ValueError('Repeated model/seed/date in pilot panel')
        rows[row['date']]=(float(row['target']),float(row['prediction']))
    dates=receipt['development_dates'];truth=None;rmse={}
    for name in FAMILY:
        deterministic=name in _pilot.CLASSICAL
        for seed in ((-1,) if deterministic else seeds):
            panel=by.get((name,seed),{})
            if set(panel)!=set(dates):raise ValueError(f'Incomplete dates for {name}/{seed}')
            values=np.asarray([panel[d] for d in dates])
            if not np.isfinite(values).all():raise ValueError('Nonfinite pilot values')
            if truth is None:truth=values[:,0]
            elif not np.array_equal(truth,values[:,0]):raise ValueError('Targets differ between jobs')
            rmse[(name,seed)]=float(np.sqrt(np.mean((values[:,1]-truth)**2)))
    gains=np.empty((len(seeds),len(RIVALS)))
    for i,seed in enumerate(seeds):
        for j,name in enumerate(RIVALS):
            rival=rmse[(name,-1 if name in _pilot.CLASSICAL else seed)]
            if rival<=0:raise ValueError('A zero-RMSE rival prevents relative-gain dimensioning')
            gains[i,j]=(rival-rmse[(CANDIDATE,seed)])/rival
    if (gains.std(axis=0,ddof=1)<=0).any():raise ValueError('Zero estimated seed variance; gather more pilot seeds')
    return gains,{'status':'actual_pilot_loaded','pilot_source_commit':request['source_commit'],
        'pilot_predictions_sha256':receipt['predictions_sha256'],'pilot_seeds':seeds,
        'pilot_dates':dates,'pilot_count_per_rival':len(seeds),
        'gain_definition':'(rival_RMSE - candidate_RMSE) / rival_RMSE per paired training seed',
        'deterministic_rivals':'one actual forecast trajectory reused with each candidate seed; no fictitious fits'}


def procedure_pvalues(samples,margin):
    """samples shape simulation × rival × seed; upper-location alternative."""
    values=np.asarray(samples,dtype=np.float64)-margin
    n=values.shape[-1]
    if n<10:raise ValueError('At least ten final paired seeds required')
    mean=values.mean(-1);sd=values.std(-1,ddof=1)
    t=np.divide(mean,sd/np.sqrt(n),out=np.zeros_like(mean),where=sd>0)
    p_t=stats.t.sf(t,n-1)
    p_t=np.where(sd==0,np.where(mean>0,0.,1.),p_t)
    # Future samples are continuous under the declared Gaussian planning model.
    # Pratt handling is stated for numerically exact zeros; approximation is
    # frozen for this planner, not selected after observing a favorable test.
    p_w=stats.wilcoxon(values,axis=-1,alternative='greater',zero_method='pratt',
                       correction=True,method='approx').pvalue
    positive=(values>0).sum(-1);nonzero=(values!=0).sum(-1)
    p_s=stats.binom.sf(positive-1,nonzero,.5)
    return np.stack((p_t,p_w,p_s),axis=-1)


def simulate(gains,*,seeds,margin,excess,simulations=10000,alpha=.05,
             covariance_shrinkage=0.,random_seed=20261118,confidence_z=3.5):
    """Joint Gaussian model of actual paired seed gains; preserve rival dependence.

    shrinkage=0 retains empirical off-diagonal covariance; sensitivity cases
    progressively reduce it. This does not estimate power across unseen markets.
    """
    if seeds<10 or simulations<1000 or margin<0 or excess<=0 or margin+excess>=1 or not 0<alpha<1:
        raise ValueError('Need n>=10, >=1000 simulations, nonnegative margin and positive alternative excess')
    if not 0<=covariance_shrinkage<=1:raise ValueError('Invalid covariance sensitivity')
    cov=np.cov(gains,rowvar=False,ddof=1)
    cov=(1-covariance_shrinkage)*cov+covariance_shrinkage*np.diag(np.diag(cov))
    eigenvalues,eigenvectors=np.linalg.eigh(cov)
    root=eigenvectors@np.diag(np.sqrt(np.maximum(eigenvalues,0.)))
    physical_tail=float(np.max(stats.norm.sf((1-margin-excess)/np.sqrt(np.diag(cov)))))
    rng=np.random.default_rng(random_seed)
    success=np.zeros((len(RIVALS),len(PROCEDURES)),dtype=np.int64)
    contrast_success=np.zeros(len(RIVALS),dtype=np.int64);joint=0
    for offset in range(0,simulations,256):
        count=min(256,simulations-offset)
        normal=rng.standard_normal((count,seeds,len(RIVALS)))
        sample=np.einsum('bnj,kj->bnk',normal,root).transpose(0,2,1)+(margin+excess)
        p=procedure_pvalues(sample,margin)
        adjusted=holm(p.reshape(count,-1)).reshape(p.shape)
        rejected=adjusted<=alpha
        success+=rejected.sum(axis=0)
        rival=np.max(adjusted,axis=-1)<=alpha
        contrast_success+=rival.sum(axis=0);joint+=int(rival.all(axis=-1).sum())
    lo,hi=_wilson(success,simulations,confidence_z)
    rlo,rhi=_wilson(contrast_success,simulations,confidence_z)
    jlo,jhi=_wilson(joint,simulations,confidence_z)
    return {'paired_final_seeds':seeds,'covariance_shrinkage':covariance_shrinkage,
        'alpha':alpha,'global_family_size':len(RIVALS)*len(PROCEDURES),
        'simulations':simulations,'MC_confidence_z':confidence_z,
        'max_normal_mass_above_physical_gain_bound':physical_tail,
        'normal_planning_boundary_check_passed':physical_tail<=.001,
        'boundary_margin_relative_rmse':margin,'alternative_relative_gain':margin+excess,
        'excess_over_margin':excess,'minimum_procedure_power':float((success/simulations).min()),
        'minimum_rival_all_three_power':float((contrast_success/simulations).min()),
        'minimum_rival_all_three_lower_MC_bound':float(rlo.min()),
        'joint_all_51_power':joint/simulations,'joint_lower_MC_bound':float(jlo),
        'joint_upper_MC_bound':float(jhi),'rivals':{
            name:{'all_three_power':float(contrast_success[j]/simulations),
                  'all_three_MC_interval':[float(rlo[j]),float(rhi[j])],
                  'procedures':{method:{'power':float(success[j,k]/simulations),
                    'MC_interval':[float(lo[j,k]),float(hi[j,k])]} for k,method in enumerate(PROCEDURES)}}
            for j,name in enumerate(RIVALS)}}


def run(pilot_directory,output,*,margin,excess,seed_grid=(10,20,40,80),
        simulations=10000,target_power=.9,covariance_scenarios=(0.,.5,1.),
        minimum_pilot_seeds=10,random_seed=20261118):
    """Dimension conditional seed-location tests; never freezes the protocol."""
    gains,provenance=pilot_gains(pilot_directory,minimum_pilot_seeds)
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    if gains is None:
        report={**provenance,'kind':'e3_prospective_power','power_dimensioned':False,
                'R3_accredited':False};_pilot.write_json(directory/'power.json',report);return report
    if not seed_grid or any(type(n)is not int or n<10 for n in seed_grid):raise ValueError('Invalid final seed grid')
    if not covariance_scenarios:raise ValueError('At least one covariance sensitivity scenario required')
    if not 0<target_power<1:raise ValueError('Invalid power target')
    # Bonferroni simultaneous MC intervals over planned rows and all 69 reported
    # event probabilities per row (51 tests +17 rivals +joint), before selection.
    z=float(stats.norm.ppf(1-.05/(2*len(seed_grid)*len(covariance_scenarios)*69)))
    rows=[]
    for n in seed_grid:
        for shrinkage in covariance_scenarios:
            row=simulate(gains,seeds=n,margin=margin,excess=excess,simulations=simulations,
                covariance_shrinkage=shrinkage,random_seed=random_seed,confidence_z=z)
            rows.append(row)
    passing=[n for n in seed_grid if all(r['joint_lower_MC_bound']>=target_power and
        r['minimum_rival_all_three_lower_MC_bound']>=target_power and r['normal_planning_boundary_check_passed']
        for r in rows if r['paired_final_seeds']==n)]
    best=max(seed_grid)
    # A directly simulated bracket for detectable excess at the largest budget.
    # This is a coarse prespecified grid, not a claimed exact root/MDE estimate.
    mde_grid=sorted(set([float(excess),.01,.02,.05,.10,.20,.40,.80]))
    mde_rows=[]
    mz=float(stats.norm.ppf(1-.05/(2*len(mde_grid)*len(covariance_scenarios))))
    for candidate_excess in mde_grid:
        if candidate_excess<=0 or margin+candidate_excess>=1:continue
        results=[simulate(gains,seeds=best,margin=margin,excess=candidate_excess,
            simulations=simulations,covariance_shrinkage=s,random_seed=random_seed,
            confidence_z=mz) for s in covariance_scenarios]
        mde_rows.append({'excess':candidate_excess,
            'normal_planning_boundary_check_passed':all(r['normal_planning_boundary_check_passed'] for r in results),
            'worst_joint_lower_MC_bound':min(r['joint_lower_MC_bound'] for r in results)})
    detectable=[r['excess'] for r in mde_rows if r['worst_joint_lower_MC_bound']>=target_power and r['normal_planning_boundary_check_passed']]
    report={'kind':'e3_prospective_power',**provenance,
        'status':'conditional_simulation_complete','power_dimensioned':False,
        'conditional_seed_dimension_found':bool(passing),'proposed_paired_final_seeds':min(passing) if passing else None,
        'observed_pilot_gain_means':dict(zip(RIVALS,gains.mean(0).tolist())),
        'observed_pilot_gain_standard_deviations':dict(zip(RIVALS,gains.std(0,ddof=1).tolist())),
        'margin':margin,'planning_excess':excess,'target_joint_power':target_power,
        'family':list(FAMILY),'procedures':list(PROCEDURES),'global_Holm_family_size':51,
        'decision_rule':'Every one of three globally Holm-adjusted p-values must reject for each of 17 rivals; all 17 rivals for joint success',
        'alternative_not_selected_from_pilot_mean':True,'random_seed':random_seed,
        'MC_interval_family':'Bonferroni simultaneous over prespecified grid and reported probabilities',
        'simulation_rows':rows,'largest_seed_budget':best,'detectable_excess_grid':mde_rows,
        'smallest_passing_grid_excess':min(detectable) if detectable else None,
        'limitations':[
            'Conditional multivariate Gaussian model of seed-level relative RMSE gains on exposed BTC development only.',
            'Paired t, signed-rank and sign tests share a location alternative only under the stated distributional assumptions.',
            'Training-seed variability does not by itself quantify temporal uncertainty, future calibration, distribution shift or additional-dataset power.',
            'Pilot covariance has sampling uncertainty; covariance sensitivity is not a confidence guarantee for the unknown covariance.',
            'A separate frozen temporal/data-set analysis and its power remain necessary; no tag is created and no full campaign is authorized by this file.'
        ],'required_before_final_freeze':['Validate procedure assumptions and temporal/data-set battery',
            'Bind common HPO/compute/capacity budget and final admitted datasets',
            'Review actual timing, variance, effect margin and full battery joint power',
            'Create annotated dated protocol tag before new confirmation fits'],
        'real_campaign_allowed':False,'R1_accredited':False,'R3_accredited':False}
    _pilot.write_json(directory/'power.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',required=True)
    p.add_argument('--output',required=True);p.add_argument('--margin',type=float,required=True)
    p.add_argument('--excess',type=float,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[10,20,40,80])
    p.add_argument('--simulations',type=int,default=10000);args=p.parse_args()
    report=run(args.pilot,args.output,margin=args.margin,excess=args.excess,
               seed_grid=tuple(args.seeds),simulations=args.simulations)
    print(json.dumps({k:v for k,v in report.items() if k not in ('simulation_rows','observed_pilot_gain_means')},indent=2))


if __name__=='__main__':main()
