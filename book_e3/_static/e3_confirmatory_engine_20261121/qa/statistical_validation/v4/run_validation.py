"""Predeclared V4 conditional synthetic validation; no market data or fitting.

Four bounded threads share one process CPU clock and cached deterministic
fixed-b references. Each replicate has an independent predeclared seed.
Evidence is exclusive-create and flushed per complete replicate.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from engine.statistics import analyze, METHODS, PROTOCOL_VERSION, _fixed_b_reference, _native_kernel

GENERATOR=ROOT/'qa/statistical_validation/runner_v1.py'
spec=importlib.util.spec_from_file_location('preserved_generator',GENERATOR)
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def binomial(count,n):
    return {'count':int(count),'n':int(n),'estimate':float(count/n)if n else None,
            'cp95':[float(beta.ppf(.025,count,n-count+1))if count else 0.,
                    float(beta.ppf(.975,count+1,n-count))if count<n else 1.]}

def one(plan,scenario_index,replicate,started_cpu):
    if time.process_time()-started_cpu>=plan['cpu_budget_seconds']-plan['inflight_cpu_reserve_seconds']:
        return None
    scenario=plan['scenarios'][scenario_index]
    p,y,z,truth_d,truth_gain=generator.generate(plan,scenario,scenario_index,replicate)
    result=analyze(p,y,['candidate']+[f'rival_{i:02d}'for i in range(plan['n_rivals'])],
                   bootstrap_reps=plan['bootstrap_reps'],block_length=plan['block_length'],
                   hac_lags=plan['hac_lags'],instruments=z,margin=plan['margin'],
                   seed=plan['seed']+scenario_index*100000+replicate,
                   reference_reps=plan['reference_reps'],reference_seed=plan['reference_seed'],
                   bootstrap_backend=plan['bootstrap_backend'],alpha=plan['alpha'])
    comparisons=result['comparisons'];method_false={m:False for m in METHODS}
    false_reject,false_favorable=False,False
    for c in comparisons[:scenario['null_rivals']]:
        for method,tests in c['methods'].items():
            test=tests[scenario['boundary_null']];reject=test['adjusted_p']<plan['alpha']
            false_reject|=reject;false_favorable|=reject and test['direction']=='candidate_better'
            method_false[method]|=reject
    c0=comparisons[0]
    row={'scenario':scenario['id'],'panel':'single_temporal_panel','replicate':replicate,
         'data_seed_identity':[plan['seed'],scenario_index,replicate],
         'bootstrap_seed':plan['seed']+scenario_index*100000+replicate,
         'false_reject_any_null':bool(false_reject),'false_favorable_any_null':bool(false_favorable),
         'method_false_reject':method_false,'per_rival_gate':[c['provisional_statistical_gate']for c in comparisons],
         'joint_gate':bool(result['all_rivals_statistical_gate']),'truth_mse':truth_d,'truth_gain':truth_gain,
         'ci95_mse':c0['ci95_mse_difference'],'ci99_mse':c0['ci99_mse_difference'],
         'ci95_gain':c0['ci95_relative_rmse_gain'],'ci99_gain':c0['ci99_relative_rmse_gain']}
    for label,truth in [('mse',truth_d),('gain',truth_gain)]:
        for confidence in [95,99]:
            interval=row[f'ci{confidence}_{label}']
            row[f'coverage_ci{confidence}_{label}']=bool(interval is not None and interval[0]<=truth<=interval[1])
    tests=[t for c in comparisons for m in c['methods'].values()for t in m.values()]
    row['test_vectors']={key:[t.get(key)for t in tests]for key in
                         ['statistic','raw_p','adjusted_p','direction','status','mc_tail_count','mc_reps','mc_p_rule']}
    return row

def summarize(plan,rows):
    items=[];nplanned=plan['replicates_per_stochastic_scenario']
    for scenario in plan['scenarios']:
        group=sorted((r for r in rows if r['scenario']==scenario['id']),key=lambda r:r['replicate'])
        n=len(group);item={'id':scenario['id'],'n':n,'complete':n==nplanned}
        if n:
            item['fwer']=binomial(sum(r['false_reject_any_null']for r in group),n)if scenario['null_rivals']else None
            item['false_favorable_fwer']=binomial(sum(r['false_favorable_any_null']for r in group),n)if scenario['null_rivals']else None
            item['fwer_by_method']={m:binomial(sum(r['method_false_reject'][m]for r in group),n)for m in METHODS}if scenario['null_rivals']else None
            for label in ['mse','gain']:
                for confidence in [95,99]:
                    key=f'coverage_ci{confidence}_{label}';item[key]=binomial(sum(r[key]for r in group),n)
            item['joint_power']=binomial(sum(r['joint_gate']for r in group),n)
            item['per_rival_power']=[binomial(sum(r['per_rival_gate'][j]for r in group),n)for j in range(plan['n_rivals'])]
            item['fwer_demonstrated']=bool(item['fwer']is not None and item['fwer']['cp95'][1]<=.05)
            item['operational_ci99_coverage_demonstrated']=all(item[f'coverage_ci99_{label}']['cp95'][0]>=.95 for label in ['mse','gain'])
            item['diagnostic_ci95_undercoverage_detected']=any(item[f'coverage_ci95_{label}']['cp95'][1]<.95 for label in ['mse','gain'])
            item['failure_detected']=bool((item['fwer']is not None and item['fwer']['cp95'][0]>.05)or any(item[f'coverage_ci99_{label}']['cp95'][1]<.95 for label in ['mse','gain']))
            item['strong_alternative_power_demonstrated']=bool(scenario['null_rivals']==0 and min(x['cp95'][0]for x in item['per_rival_power'])>=.9)
        items.append(item)
    return items

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();folder=Path(__file__).resolve().parent
    plan=json.loads((folder/'plan.json').read_text());freeze=json.loads((folder/'freeze.json').read_text())
    for relative,digest in freeze['sha256'].items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Frozen V4 identity mismatch: '+relative)
    if plan['protocol_version']!=PROTOCOL_VERSION:raise RuntimeError('Protocol source version differs')
    args.output_dir.mkdir(parents=True,exist_ok=True);raw_path=args.output_dir/'replicates.jsonl';report_path=args.output_dir/'report.json'
    if raw_path.exists()or report_path.exists():raise FileExistsError('Existing evidence must be preserved; reproduce into a fresh directory')
    cpu,wall=time.process_time(),time.monotonic();started_at=dt.datetime.now(dt.timezone.utc).isoformat()
    if plan['bootstrap_backend']=='native'and _native_kernel()is None:raise RuntimeError('Frozen native benchmark unavailable; do not silently alter budget/backend')
    # Shared deterministic references are computed before threads, within the
    # same CPU budget. No stochastic audit dataset has been generated yet.
    for k in [1,2]:_fixed_b_reference(plan['n_dates'],plan['hac_lags'],k,plan['horizon'],plan['reference_reps'],plan['reference_seed'])
    rows=[]
    with raw_path.open('x')as stream, ThreadPoolExecutor(max_workers=plan['max_workers'])as pool:
        futures=[pool.submit(one,plan,si,ri,cpu)for si in range(len(plan['scenarios']))for ri in range(plan['replicates_per_stochastic_scenario'])]
        for future in as_completed(futures):
            row=future.result()
            if row is None:continue
            rows.append(row);stream.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');stream.flush()
            if len(rows)%128==0:print(json.dumps({'completed':len(rows),'planned':len(futures),'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-wall}),flush=True)
        stream.flush();os.fsync(stream.fileno())
    items=summarize(plan,rows);complete=all(x['complete']for x in items)
    failures=[x['id']for x in items if x.get('failure_detected')]
    bounded=bool(complete and not failures and all(x['operational_ci99_coverage_demonstrated']for x in items)and all(x['fwer_demonstrated']for x in items[:-1])and items[-1]['strong_alternative_power_demonstrated'])
    raw=raw_path.read_bytes();gz=args.output_dir/'replicates.jsonl.gz';gz.write_bytes(gzip.compress(raw,mtime=0))
    settings={'R':plan['n_fixed_seeds'],'T':plan['n_dates'],'B':plan['bootstrap_reps'],
              **{k:plan[k]for k in ['block_length','hac_lags','horizon','margin','reference_reps','reference_seed','alpha']}}
    report={'version':plan['version'],'protocol_version':PROTOCOL_VERSION,'validation_scope':'single_panel_temporal_family',
            'panels_per_replicate':1,'final_panels_application':4,'family_rivals':20,
            'all_three_methods':True,'complete_and_partial_nulls':True,
            'extreme_tail_validated':bounded,'coverage_validated':bounded,
            'synthetic_validation_executed':True,'failure_detected':bool(failures),
            'statistics_source_sha256':sha(ROOT/'engine/statistics.py'),'statistics_sha256':sha(ROOT/'engine/statistics.py'),
            'plan_sha256':sha(folder/'plan.json'),'source_hashes':freeze['sha256'],
            'scope':plan['purpose'],'criteria':plan['acceptance_fixed'],'scenarios':items,
            'validated_settings':[settings]if bounded else[],
            'started_at_utc':started_at,'finished_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-wall,
            'cpu_budget_seconds':plan['cpu_budget_seconds'],'max_inflight_workers':plan['max_workers'],
            'all_scenarios_complete':complete,'failed_scenarios':failures,
            'status':'FAILED_FINITE_SAMPLE_VALIDATION'if failures else'PASSED_PREDECLARED_VALIDATION'if bounded else'INCONCLUSIVE_BOUNDED_VALIDATION',
            'admitted':bounded,'conditional_scope_admitted':bounded,'protocol_closure_permitted':bounded,
            'general_validation_established':False,'r3_accredited':False,
            'raw_results_sha256':sha(raw_path),'replicates_uncompressed_sha256':sha(raw_path),'replicates_uncompressed_bytes':len(raw),
            'evidence':{'plan':{'path':'plan.json','sha256':sha(folder/'plan.json')},
                        'freeze':{'path':'freeze.json','sha256':sha(folder/'freeze.json')},
                        'runner':{'path':'run_validation.py','sha256':sha(folder/'run_validation.py')},
                        'replicates':{'path':'replicates.jsonl.gz','sha256':sha(gz),'uncompressed_sha256':sha(raw_path)}},
            'four_panel_application_note':plan['four_panel_application_note'],
            'uncertainty':'CP95 intervals are pointwise across declared diagnostics, not simultaneous over MC scenarios or coverage outcomes. Finite DGP scope only.',
            'prior_studies':'V1/V2 failed and V3 inconclusive are retained separately; new independent seeds, not pooled or selected to improve a bound.'}
    report_path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    for item in items:print(json.dumps({'scenario':item['id'],'n':item['n'],'fwer':item.get('fwer'),'CI99mse':item.get('coverage_ci99_mse'),'CI99gain':item.get('coverage_ci99_gain'),'joint_power':item.get('joint_power')}),flush=True)
    print(json.dumps({'status':report['status'],'cpu_seconds':report['cpu_seconds'],'wall_seconds':report['wall_seconds'],'report':str(report_path)}),flush=True)

if __name__=='__main__':main()
