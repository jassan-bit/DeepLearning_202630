"""CPU-only future planning-contract checks; cannot train or enable confirmation.

The simulation reader aggregates pre-existing planning-simulation p-values; it
does not simulate markets, fit models, infer forecasts, or claim actual power.
Prepared with declared AI assistance. Software fixtures are not scientific runs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

PROCEDURES=('P1_FUNCTIONAL_BLOCK_BOOTSTRAP','P2_RMSE_DELTA_HAC','P3_FUNCTIONAL_SUBSAMPLING')


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def holm(values):
    """Strong FWER adjustment, with monotonic adjusted p-values."""
    ordered=sorted(enumerate(values),key=lambda pair:(pair[1],pair[0]))
    require(values and all(finite(p) and 0<=p<=1 for p in values),'Invalid p-values')
    adjusted=[None]*len(values);last=0.0
    for rank,(index,p) in enumerate(ordered):
        last=max(last,min(1.0,(len(values)-rank)*p));adjusted[index]=last
    return adjusted


def wilson_lower(successes,n,alpha):
    """One-sided Wilson lower bound for MC binomial proportions only."""
    require(type(n) is int and n>0 and type(successes) is int and 0<=successes<=n,'Invalid Monte Carlo counts')
    require(0<alpha<.5,'Invalid confidence allocation')
    z=NormalDist().inv_cdf(1-alpha);p=successes/n;z2=z*z
    return max(0.0,(p+z2/(2*n)-z*math.sqrt(p*(1-p)/n+z2/(4*n*n)))/(1+z2/n))


def clopper_pearson_lower(successes,n,alpha):
    """One-sided exact-binomial MC bound, by stable tail inversion.

    Assumes independent Monte Carlo replicates. It makes no independence
    assumption about models or procedures within the same replicate.
    """
    require(type(n) is int and n>0 and type(successes) is int and 0<=successes<=n,'Invalid Monte Carlo counts')
    require(0<alpha<.5,'Invalid confidence allocation')
    if successes==0:return 0.0
    if successes==n:return alpha**(1/n)
    def tail(p):
        if p==0:return 0.0
        if p==1:return 1.0
        lp=math.log(p);lq=math.log1p(-p);k=successes
        term=math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*lp+(n-k)*lq
        total=term
        for x in range(k,n):
            term+=math.log(n-x)-math.log(x+1)+lp-lq
            hi=max(total,term);total=hi+math.log1p(math.exp(-abs(total-term)))
        return min(1.0,math.exp(total))
    low=0.0;high=successes/n
    for _ in range(60):
        middle=(low+high)/2
        if tail(middle)<alpha:low=middle
        else:high=middle
    return (low+high)/2


def summarize_rows(rows,rivals,alpha=.05):
    require(len(rivals)==len(set(rivals))==19,'Exactly nineteen named rivals required')
    expected={(r,p) for r in rivals for p in PROCEDURES};groups={};scope=None
    for row in rows:
        require(row['stage']=='planning_simulation','Observed confirmation/test p-values are forbidden in this planning reader')
        this_scope=(row['dataset_id'],int(row['R']),int(row['T']))
        require(this_scope[0] and this_scope[1]>=10 and this_scope[2]>0,'Invalid panel or candidate planning budget')
        if scope is None:scope=this_scope
        require(scope==this_scope,'A summary file must contain one panel and one R/T budget')
        require(row['rival'] in rivals and row['procedure'] in PROCEDURES,'Unplanned rival/procedure')
        replicate=int(row['replicate']);require(replicate>=0,'Invalid replicate')
        key=(row['rival'],row['procedure']);group=groups.setdefault(replicate,{})
        require(key not in group,'Duplicate replicate/rival/procedure')
        p=float(row['p_raw']);require(finite(p) and 0<=p<=1,'Invalid p-value')
        group[key]=p
    require(groups,'No complete planning simulations')
    keys=[(r,p) for r in rivals for p in PROCEDURES]
    cell_hits={key:0 for key in keys};rival_hits={r:0 for r in rivals};joint=0
    for group in groups.values():
        require(set(group)==expected,'Incomplete 57-component family in a simulation replicate')
        adjusted=dict(zip(keys,holm([group[key] for key in keys])))
        passed={r:all(adjusted[(r,p)]<alpha for p in PROCEDURES) for r in rivals}
        for key in keys:cell_hits[key]+=adjusted[key]<alpha
        for rival in rivals:rival_hits[rival]+=passed[rival]
        joint+=all(passed.values())
    n=len(groups)
    # Simultaneous MC bounds across19 rival-intersection decisions; no products
    # of marginal probabilities and no independence assumption across rivals.
    rival_rows=[{'rival':r,'Monte_Carlo_replicates':n,'three_procedure_intersection_rejections':rival_hits[r],
                  'conditional_power_estimate':rival_hits[r]/n,
                  'simultaneous_one_sided_Clopper_Pearson_lower95_over19':clopper_pearson_lower(rival_hits[r],n,.05/19)} for r in rivals]
    return {'kind':'conditional_planning_simulation_summary_not_actual_power_certificate',
            'dataset_id':scope[0],'R':scope[1],'T':scope[2],
            'family_rivals':19,'pvalues_adjusted_per_replicate':57,'Holm_alpha':alpha,
            'Monte_Carlo_replicates':n,'rival_intersection_power':rival_rows,
            'all57_joint_rejections':joint,'conditional_joint_power_estimate':joint/n,
            'joint_one_sided_Clopper_Pearson_lower95':clopper_pearson_lower(joint,n,.05),
            'component_power_estimates':[{'rival':r,'procedure':p,'estimate':cell_hits[(r,p)]/n} for r,p in keys],
            'all_rival_MC_lower_bounds_at_least90':all(x['simultaneous_one_sided_Clopper_Pearson_lower95_over19']>=.9 for x in rival_rows),
            'guide_requires_joint90':False,'Monte_Carlo_confidence_does_not_validate_DGP_or_tests':True,
            'confirmation_enabled':False,'new_model_fits':0,'new_model_predictions':0}


def bound(config,root,name):
    item=config.get('evidence_bindings',{}).get(name)
    if not isinstance(item,dict):return False
    path,sha=item.get('path'),item.get('sha256')
    if not isinstance(path,str) or not isinstance(sha,str) or len(sha)!=64:return False
    p=(root/path).resolve()
    return p.is_relative_to(root) and p.is_file() and not p.is_symlink() and digest(p)==sha


def readiness(config,root):
    root=root.resolve();family=config.get('family',{});data=config.get('data',{})
    sampling=config.get('sampling',{});margins=config.get('margins',{});science=config.get('science',{})
    panels=data.get('future_panels',[]);rivals=family.get('rivals',[])
    checks={
        'original_NO_GO_and_adverse_pilot_preserved':science.get('original_pilot_rule_preserved') is True and science.get('automatic_ablation_promotion') is False,
        'renewed_scientific_rationale_review':science.get('renewed_rationale_sufficient') is True and bound(config,root,'renewed_scientific_rationale'),
        'nineteen_rivals_and57_components':len(rivals)==len(set(rivals))==19 and family.get('procedures')==list(PROCEDURES),
        'four_unexposed_panels_admitted':len(panels)==4 and all(isinstance(p,dict) and p.get('admitted_unexposed') is True and p.get('bytes_sha256') for p in panels) and bound(config,root,'future_data_admission_and_access_log'),
        'all_four_operational_margins_bound':len(margins.get('by_panel',{}))==4 and all(finite(v) and v>0 for v in margins.get('by_panel',{}).values()) and bound(config,root,'operational_margins_before_outcomes'),
        'all76_panel_rival_variances_bound':sampling.get('planning_seed_pairs',0)>=5 and sampling.get('complete_variances_for_four_panels_and19_rivals') is True and bound(config,root,'complete_planning_variance_and_dependence'),
        'three_temporal_methods_and_same_estimand_reviewed':config.get('battery',{}).get('reviewed_and_validated') is True and bound(config,root,'temporal_battery_size_estimand_validation'),
        'prospective_power_joint_MDE_bound':type(sampling.get('final_R')) is int and sampling['final_R']>=10 and type(sampling.get('final_T')) is int and sampling['final_T']>0 and bound(config,root,'prospective_power_joint_MDE_certificate'),
        'common_treatment_capacity_budget_bound':config.get('treatment',{}).get('fairness_review_complete') is True and bound(config,root,'fair_treatment_capacity_and_budget'),
        'protocol_frozen_before_future_outcomes':config.get('freeze',{}).get('annotated_tag_verified') is True and bound(config,root,'annotated_tag_and_outcome_access_order'),
    }
    return {'status':'STRUCTURAL_INPUTS_COMPLETE_REQUIRES_SEPARATE_REVIEW' if all(checks.values()) else 'BLOCKED_PREPARATION_ONLY',
            'checks':checks,'blockers':[k for k,v in checks.items() if not v],
            'confirmation_enabled':False,'this_program_can_train':False,
            'scientific_validity_not_inferred_from_JSON_booleans':True,'new_model_fits':0,'new_model_predictions':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    pre=sub.add_parser('preflight');pre.add_argument('--config',type=Path,required=True);pre.add_argument('--output',type=Path,required=True);pre.add_argument('--prepare-only',action='store_true')
    power=sub.add_parser('summarize-simulation');power.add_argument('--config',type=Path,required=True);power.add_argument('--pvalues',type=Path,required=True);power.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_bytes())
    if args.command=='preflight':report=readiness(config,args.config.parent)
    else:
        with args.pvalues.open(newline='') as stream:rows=list(csv.DictReader(stream))
        report=summarize_rows(rows,config['family']['rivals'])
        report.update(input_pvalues_sha256=digest(args.pvalues),input_config_sha256=digest(args.config),
                      method_validation_complete=config['battery']['reviewed_and_validated'],
                      DGP_contract=config['planning_simulation_contract'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
    return 2 if args.command=='preflight' and report['blockers'] and not args.prepare_only else 0


if __name__=='__main__':raise SystemExit(main())
