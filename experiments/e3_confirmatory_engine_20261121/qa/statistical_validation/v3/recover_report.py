"""Recover a summary from the complete frozen V3 ledger; no simulations.

The environment was replaced after all 960 durable rows had been written but
before report.json existed. The original runner and frozen sources are not
modified. This command verifies the complete Cartesian ledger and recreates
the prespecified summary. It never calls analyze/generate or fills missing rows.
"""
from __future__ import annotations
import collections
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
FOLDER=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    plan=json.loads((FOLDER/'plan.json').read_text())
    freeze=json.loads((FOLDER/'freeze.json').read_text())
    for name,digest in freeze['sha256'].items():
        if sha(ROOT/name)!=digest:raise RuntimeError('Frozen V3 source changed: '+name)
    path=FOLDER/'replicates.jsonl';raw=path.read_bytes()
    if not raw.endswith(b'\n'):raise RuntimeError('Incomplete final JSONL row; preserve evidence')
    rows=[json.loads(line)for line in raw.splitlines()]
    n=plan['replicates_per_stochastic_scenario']
    expected={(s['id'],r)for s in plan['scenarios']for r in range(n)}
    keys=[(r['scenario'],r['replicate'])for r in rows]
    if len(keys)!=len(set(keys))or set(keys)!=expected:raise RuntimeError('V3 ledger is not exactly the frozen Cartesian plan')
    methods=['dm_hln_hac','stationary_bootstrap','gw_conditional']
    for row in rows:
        if set(row['method_false_reject'])!=set(methods)or len(row['per_rival_gate'])!=plan['n_rivals']:raise RuntimeError('Ledger method/rival mismatch')
        for key in ['statistic','raw_p','adjusted_p','direction','status','mc_tail_count','mc_reps','mc_p_rule']:
            if len(row['test_vectors'][key])!=120:raise RuntimeError('Truncated test vector')
        v=row['test_vectors']
        for p,count,reps,rule in zip(v['raw_p'],v['mc_tail_count'],v['mc_reps'],v['mc_p_rule']):
            if rule!='absolute_tail_plus_one'or p!=(count+1)/(reps+1):raise RuntimeError('MC p/count inconsistency')
    spec=importlib.util.spec_from_file_location('frozen_v3_runner',FOLDER/'run_validation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    b=module.binomial
    report={'version':plan['version'],'protocol_version':plan['protocol_version'],
            'plan_sha256':sha(FOLDER/'plan.json'),'source_hashes':freeze['sha256'],
            'scope':plan['purpose'],'criteria':plan['acceptance_fixed'],'scenarios':[],
            'intervals':'Operational CI99 and diagnostic CI95 are named separately. MC CP95 intervals are pointwise, not simultaneous.',
            'recovery':{'status':'SUMMARY_RECONSTRUCTED_FROM_COMPLETE_DURABLE_LEDGER',
                        'new_stochastic_replicates':0,'missing_replicates':0,'repeated_replicates':0,
                        'reason':'Execution environment replaced after all960 ledger rows were durable; report not present.',
                        'original_cpu_seconds':None,'original_wall_seconds':None,
                        'budget_compliance':'NOT_RECONSTRUCTABLE_FROM_LEDGER; original ceiling900CPU seconds retained; no exact consumption invented',
                        'note':'Plan budget_rule textual300s conflicts with numeric900s; frozen runner actually enforces numeric cpu_budget_seconds900. Both preserved.'}}
    for scenario in plan['scenarios']:
        group=[r for r in rows if r['scenario']==scenario['id']]
        item={'id':scenario['id'],'n':len(group),'complete':len(group)==n}
        item['fwer']=b(sum(r['false_reject_any_null']for r in group),n)if scenario['null_rivals']else None
        item['false_favorable_fwer']=b(sum(r['false_favorable_any_null']for r in group),n)if scenario['null_rivals']else None
        item['fwer_by_method']={m:b(sum(r['method_false_reject'][m]for r in group),n)for m in methods}if scenario['null_rivals']else None
        for label in ['mse','gain']:
            for confidence in [95,99]:
                key=f'coverage_ci{confidence}_{label}';item[key]=b(sum(r[key]for r in group),n)
        item['joint_power']=b(sum(r['joint_gate']for r in group),n)
        item['per_rival_power']=[b(sum(r['per_rival_gate'][j]for r in group),n)for j in range(plan['n_rivals'])]
        item['fwer_demonstrated']=bool(item['fwer']is not None and item['fwer']['cp95'][1]<=.05)
        item['operational_ci99_coverage_demonstrated']=all(item[f'coverage_ci99_{label}']['cp95'][0]>=.95 for label in ['mse','gain'])
        item['diagnostic_ci95_undercoverage_detected']=any(item[f'coverage_ci95_{label}']['cp95'][1]<.95 for label in ['mse','gain'])
        item['failure_detected']=bool((item['fwer']is not None and item['fwer']['cp95'][0]>.05)or any(item[f'coverage_ci99_{label}']['cp95'][1]<.95 for label in ['mse','gain']))
        item['strong_alternative_power_demonstrated']=bool(scenario['null_rivals']==0 and min(v['cp95'][0]for v in item['per_rival_power'])>=.9)
        report['scenarios'].append(item)
    failures=[r['id']for r in report['scenarios']if r['failure_detected']]
    bounded=bool(not failures and all(r['operational_ci99_coverage_demonstrated']for r in report['scenarios'])and all(r['fwer_demonstrated']for r in report['scenarios'][:-1])and report['scenarios'][-1]['strong_alternative_power_demonstrated'])
    report.update(cpu_seconds=None,wall_seconds=None,all_scenarios_complete=True,failed_scenarios=failures,
                  status='FAILED_FINITE_SAMPLE_VALIDATION'if failures else'BOUNDED_DGP_SCOPE_VALIDATED'if bounded else'INCONCLUSIVE_BOUNDED_VALIDATION',
                  admitted=bounded,conditional_scope_admitted=bounded,general_validation_established=False,
                  protocol_closure_permitted=False,raw_results_sha256=sha(path),synthetic_validation_executed=True)
    compressed=FOLDER/'replicates.jsonl.gz';encoded=gzip.compress(raw,mtime=0)
    if compressed.exists()and compressed.read_bytes()!=encoded:raise RuntimeError('Existing compressed ledger differs')
    compressed.write_bytes(encoded)
    report['replicates_archive']={'path':str(compressed.relative_to(ROOT)),'sha256':sha(compressed),'bytes':len(encoded),
                                  'uncompressed_sha256':sha(path),'uncompressed_bytes':len(raw),'rows':len(rows)}
    target=FOLDER/'report.json';body=json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if target.exists()and target.read_text()!=body:raise RuntimeError('Report already differs; preserve it')
    target.write_text(body)
    receipt={'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'recovery_script_sha256':sha(__file__),
             'report_sha256':sha(target),'raw_sha256':sha(path),'rows':len(rows),'new_mc_replicates':0,
             'all_frozen_hashes_verified':True,'no_missing_or_duplicate_rows':True}
    (FOLDER/'RECOVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'rows':len(rows),'report':str(target),'gzip_bytes':len(encoded)}))

if __name__=='__main__':main()
