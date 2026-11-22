"""Evaluate the actual four-panel decision, retaining reproducible compact records."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from .dgps import SYSTEM_IDS,generate
from .logic import joint_rule,witness


def ready(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [ready(v) for v in value]
    return value


def canonical(value):
    return json.dumps(ready(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def seed_for(plan_seed,scenario_index,replicate_index):
    return int(np.random.SeedSequence([int(plan_seed),int(scenario_index),int(replicate_index)]).generate_state(1,dtype=np.uint64)[0])


def mask_hex(values):
    flags=np.asarray(values,dtype=bool).ravel()
    return np.packbits(flags,bitorder='little').tobytes().hex()


def unmask_hex(value,size=80):
    raw=bytes.fromhex(value)
    out=np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little')
    if len(out)<size:raise ValueError('Incomplete decision mask')
    return out[:size].astype(bool)


def one_replica(plan,scenario_index,replicate_index,*,full=False):
    # Importing this scientific implementation does not train or read market.
    from methods import primary_rmse as method
    cpu=time.process_time();wall=time.perf_counter()
    case=plan['scenarios'][scenario_index]
    cell=plan['cell'];seed=seed_for(plan['data_seed'],scenario_index,replicate_index)
    data=generate(case,cell['R'],cell['T'],seed,plan['margin'])
    results=[]
    for p in range(4):
        results.append(method.analyze_losses(data['losses'][p],SYSTEM_IDS,
            bootstrap_reps=cell['B'],block_length=cell['block_length'],hac_lags=cell['hac_lags'],
            instruments=data['instruments'][p],margin=plan['margin'],seed=plan['inner_bootstrap_seed'],
            horizon=cell['horizon'],reference_reps=cell['reference_reps'],reference_seed=plan['reference_seed'],
            bootstrap_backend='native',alpha=plan['panel_alpha']))
    decision=method.statistical_rule(results,margin=plan['margin'])
    flags=np.asarray(decision['contrast_pass'],dtype=bool)
    if flags.shape!=(4,20):raise ValueError('The complete 80-contrast decision is required')
    inferred=joint_rule(flags,decision['additional_significant_inferiority'])
    if inferred!=bool(decision['joint_statistical_rule_pass']):
        raise AssertionError('Scientific implementation and independent decision topology differ')
    truth=data['true_superiority'];true_global=joint_rule(truth)
    chosen=witness(truth) or [(0,0)]
    p,r=chosen[0];c=results[p]['comparisons'][r]
    true_difference=float(data['theta'][p,r+1]-data['theta'][p,0])
    true_gain=float(data['gains'][p,r])
    coverage={}
    for level in (95,99):
        for kind,actual in (('rmse_difference',true_difference),('relative_rmse_gain',true_gain)):
            interval=c.get(f'ci{level}_{kind}')
            coverage[f'CI{level}_{kind}']=bool(interval is not None and interval[0]<=actual<=interval[1])
    raw_p={m:{n:{k:c['methods'][m][n].get(k) for k in ('status','raw_p','adjusted_p')}
        for n in ('epa','practical')} for m in method.METHODS}
    equality_false={m:{n:[False]*4 for n in ('epa','practical')} for m in method.METHODS}
    superior_false={m:{n:[False]*4 for n in ('epa','practical')} for m in method.METHODS}
    for pi,result in enumerate(results):
        for ri,comparison in enumerate(result['comparisons']):
            d=float(data['theta'][pi,ri+1]-data['theta'][pi,0])
            g=float(data['gains'][pi,ri])
            for m in method.METHODS:
                for n,is_null,is_boundary,direction in (
                        ('epa',d<=1e-12,abs(d)<=1e-12,comparison['mean_rmse_difference']>0),
                        ('practical',g<=plan['margin']+1e-12,abs(g-plan['margin'])<=1e-12,comparison['practical_mean']>0)):
                    # GMM has a stronger conditional-moment null. Under the
                    # predeclared cancellation scenario that null is false for
                    # the witness even though the scalar mean-RMSE null is true.
                    if m=='conditional_functional_gmm' and case.get('conditional_cancellation') and pi==0 and ri==0:
                        continue
                    t=comparison['methods'][m][n]
                    rejected=t['status']=='ok' and t['adjusted_p']<plan['panel_alpha']
                    if is_boundary and rejected:equality_false[m][n][pi]=True
                    if is_null and direction and rejected:superior_false[m][n][pi]=True
    record=dict(scenario_index=scenario_index,scenario=case['id'],replicate=replicate_index,data_seed=seed,
        true_global=bool(true_global),true_contrasts=mask_hex(truth),
        joint_acceptance=bool(inferred),global_false_acceptance=bool(inferred and not true_global),
        contrast_pass=mask_hex(flags),additional_inferiority=bool(decision['additional_significant_inferiority']),
        witness=dict(panel=p,rival_index=r,rival=SYSTEM_IDS[r+1],true_difference=true_difference,
            true_relative_gain=true_gain,p_values=raw_p,
            intervals={f'ci{level}_{kind}':c.get(f'ci{level}_{kind}') for level in (95,99)
                for kind in ('rmse_difference','relative_rmse_gain')},coverage=coverage),
        equality_boundary_type_I_by_method_and_panel=equality_false,
        false_directional_superiority_by_method_and_panel=superior_false,
        full_generated_data_sha256=hashlib.sha256(data['losses'].astype('<f8').tobytes()+data['instruments'].astype('<f8').tobytes()).hexdigest(),
        full_statistical_reports_sha256=digest(results),
        bootstrap_indices_sha256=results[0].get('bootstrap_indices_sha256'),
        CPU_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall)
    if full:return ready(dict(record=record,results=results,decision=decision,
        true_theta=data['theta'],true_gains=data['gains']))
    return ready(record)
