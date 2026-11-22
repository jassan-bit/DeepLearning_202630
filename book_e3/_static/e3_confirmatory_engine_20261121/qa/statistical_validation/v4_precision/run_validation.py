"""Independent precision replication and explicitly retrospective dossier.

Only the unresolved complete-null scenario generates new draws. Four other
scenarios are carried from V4 without pooling, relabeling or hiding its
inconclusive original complete-null result. No model fits or market access.
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

from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
PRESERVED=ROOT/'qa/statistical_validation/v4/run_validation.py'
spec=importlib.util.spec_from_file_location('preserved_v4_runner',PRESERVED)
v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def strict_bounds(items):
    complete=all(x.get('complete') for x in items)
    failures=[x['id'] for x in items if x.get('failure_detected')]
    bounded=bool(complete and not failures and all(x.get('operational_ci99_coverage_demonstrated') for x in items)
                 and all(x.get('fwer_demonstrated') for x in items[:-1])
                 and items[-1].get('strong_alternative_power_demonstrated'))
    return complete,failures,bounded

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();folder=Path(__file__).resolve().parent
    plan=json.loads((folder/'plan.json').read_text());freeze=json.loads((folder/'freeze.json').read_text())
    for name,digest in freeze['sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen focal identity mismatch: '+name)
    source_map=json.loads((folder/'SCENARIO_SOURCE_MAP.json').read_text())
    if source_map.get('frozen_before_focal_execution') is not True or source_map.get('no_pooling') is not True:
        raise RuntimeError('No prospective focal source-map or no-pooling identity')
    prior_dir=ROOT/'qa/statistical_validation/v4'
    prior_report=json.loads((prior_dir/'report.json').read_text());prior_plan=json.loads((prior_dir/'plan.json').read_text())
    if prior_report['status']!='INCONCLUSIVE_BOUNDED_VALIDATION' or prior_report['admitted'] is not False:
        raise RuntimeError('Original V4 disposition changed')
    old_raw=gzip.decompress((prior_dir/'replicates.jsonl.gz').read_bytes())
    if hashlib.sha256(old_raw).hexdigest()!=plan['prior_study']['uncompressed_sha256']:
        raise RuntimeError('Prior original ledger checksum differs')
    old_rows=[json.loads(line) for line in old_raw.splitlines()]
    old_items=v4.summarize(prior_plan,old_rows)
    if old_items!=prior_report['scenarios']:raise RuntimeError('Prior summary does not equal original durable rows')
    focal=plan['focal_scenario']
    kept={x['id']:x for x in old_items if x['id']!=focal}
    if len(kept)!=4 or not all(x['complete'] and x['operational_ci99_coverage_demonstrated'] and
                                (x['fwer_demonstrated'] if x['fwer'] is not None else x['strong_alternative_power_demonstrated'])
                                for x in kept.values()):
        raise RuntimeError('Mapped existing criteria are not actually established')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    raw_path=args.output_dir/'replicates.jsonl';report_path=args.output_dir/'report.json'
    if raw_path.exists() or report_path.exists():raise FileExistsError('Evidence exists; use fresh reproduction directory')
    affinity=sorted(os.sched_getaffinity(0));chosen=affinity[:plan['cpu_affinity_cores']]
    if len(chosen)!=plan['cpu_affinity_cores']:raise RuntimeError('Frozen two-core resource profile unavailable')
    os.sched_setaffinity(0,set(chosen))
    cpu,wall=time.process_time(),time.monotonic();started_at=dt.datetime.now(dt.timezone.utc).isoformat()
    if plan['bootstrap_backend']=='native' and v4._native_kernel() is None:raise RuntimeError('Frozen native backend unavailable')
    for k in [1,2]:v4._fixed_b_reference(plan['n_dates'],plan['hac_lags'],k,plan['horizon'],plan['reference_reps'],plan['reference_seed'])
    generation_plan={**plan,'scenarios':[s for s in plan['scenarios'] if s['id']==focal]}
    if len(generation_plan['scenarios'])!=1:raise RuntimeError('Exactly one focal scenario required')
    rows=[]
    with raw_path.open('x') as stream,ThreadPoolExecutor(max_workers=plan['max_workers']) as pool:
        futures=[pool.submit(v4.one,generation_plan,0,r,cpu) for r in range(plan['new_stochastic_replicates_planned'])]
        for future in as_completed(futures):
            row=future.result()
            if row is None:continue
            rows.append(row);stream.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');stream.flush()
            if len(rows)%128==0:print(json.dumps({'new_completed':len(rows),'new_planned':len(futures),'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-wall}),flush=True)
        stream.flush();os.fsync(stream.fileno())
    new_item=v4.summarize(generation_plan,rows)[0]
    items=[new_item if s['id']==focal else kept[s['id']] for s in plan['scenarios']]
    complete,failures,bounded=strict_bounds(items)
    raw=raw_path.read_bytes();gz=args.output_dir/'replicates.jsonl.gz';gz.write_bytes(gzip.compress(raw,mtime=0))
    def link(path,**extra):return {'path':os.path.relpath(path,args.output_dir),'sha256':sha(path),**extra}
    settings={'R':plan['n_fixed_seeds'],'T':plan['n_dates'],'B':plan['bootstrap_reps'],
              **{k:plan[k] for k in ['block_length','hac_lags','horizon','margin','reference_reps','reference_seed','alpha']}}
    report={'version':plan['version'],'report_type':'provenance_composite','protocol_version':v4.PROTOCOL_VERSION,
            'validation_scope':'single_panel_temporal_family','panels_per_replicate':1,'final_panels_application':4,
            'family_rivals':20,'all_three_methods':True,'complete_and_partial_nulls':True,
            'extreme_tail_validated':bounded,'coverage_validated':bounded,'synthetic_validation_executed':True,
            'failure_detected':bool(failures),'statistics_source_sha256':sha(ROOT/'engine/statistics.py'),
            'statistics_sha256':sha(ROOT/'engine/statistics.py'),'plan_sha256':sha(folder/'plan.json'),
            'source_hashes':freeze['sha256'],'scope':plan['purpose'],'criteria':plan['acceptance_fixed'],
            'scenarios':items,'validated_settings':[settings] if bounded else [],
            'all_scenarios_complete':complete,'failed_scenarios':failures,
            'status':'FAILED_FINITE_SAMPLE_VALIDATION' if failures else 'PASSED_PREDECLARED_VALIDATION' if bounded else 'INCONCLUSIVE_BOUNDED_VALIDATION',
            'admitted':bounded,'conditional_scope_admitted':bounded,'protocol_closure_permitted':bounded,
            'general_validation_established':False,'r3_accredited':False,
            'new_stochastic_replicates':len(rows),'reused_stochastic_replicates':2048,
            'original_V4_total_preserved':2560,'original_V4_null_excluded_from_new_null_counts':512,
            'replicates_per_scenario':plan['replicates_per_scenario'],
            'raw_results_sha256':sha(raw_path),'replicates_uncompressed_sha256':sha(raw_path),
            'replicates_uncompressed_bytes':len(raw),'prior_replicates_uncompressed_sha256':plan['prior_study']['uncompressed_sha256'],
            'evidence':{'plan':link(folder/'plan.json'),'freeze':link(folder/'freeze.json'),'runner':link(folder/'run_validation.py'),
                        'replicates':link(gz,uncompressed_sha256=sha(raw_path)),
                        'prior_replicates':link(prior_dir/'replicates.jsonl.gz',uncompressed_sha256=plan['prior_study']['uncompressed_sha256']),
                        'prior_report':link(prior_dir/'report.json'),'scenario_source_map':link(folder/'SCENARIO_SOURCE_MAP.json')},
            'started_at_utc':started_at,'finished_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-wall,'cpu_budget_seconds':plan['cpu_budget_seconds'],
            'resource_profile':{'allowed_cores_before':affinity,'used_cores':chosen,'max_worker_threads':plan['max_workers']},
            'prior_study_cpu_seconds':prior_report['cpu_seconds'],
            'four_panel_application_note':plan['four_panel_application_note'],
            'confidence_scope':source_map['confidence_scope'],
            'selection_timing':'Dossier source-map fixed after observing V4 and before generating the focal independent draws; not a preregistration of reused V4 results.',
            'no_pooling':True}
    if new_item.get('fwer'):
        k,n=new_item['fwer']['count'],new_item['fwer']['n']
        report['focal_fwer_cp99_diagnostic']=[float(beta.ppf(.005,k,n-k+1)) if k else 0.,float(beta.ppf(.995,k+1,n-k)) if k<n else 1.]
    report_path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'focal_scenario':focal,'new_n':new_item['n'],'fwer':new_item.get('fwer'),'CI99mse':new_item.get('coverage_ci99_mse'),'CI99gain':new_item.get('coverage_ci99_gain'),'cp99_fwer_diagnostic':report.get('focal_fwer_cp99_diagnostic')}),flush=True)
    print(json.dumps({'status':report['status'],'cpu_seconds':report['cpu_seconds'],'wall_seconds':report['wall_seconds'],'report':str(report_path)}),flush=True)

if __name__=='__main__':main()
