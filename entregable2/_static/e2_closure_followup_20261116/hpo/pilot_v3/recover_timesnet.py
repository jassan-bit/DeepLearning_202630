#!/usr/bin/env python3
"""Resume only the source-blocked TimesNet pilot under the original450s cap.

The two complete v2 fits and every previous charge are preserved. A new source
admission is mandatory; the excluded portable campaign launcher is unrelated to
the scientific worker's imports. This script does not repair historical costs.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path(__file__).with_name('plan.json'))
    p.add_argument('--python',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    recover=plan['source_blocked_recovery']
    if recover['recovery_launcher_sha256']!=sha(__file__):raise ValueError('Recovery launcher changed')
    out=args.plan.parent;source=Path(plan['source_root']);ledger=Path(plan['ledger_path'])
    if sha(source/'scripts/e2_hpo_followup_pilot.py')!=plan['script_sha256']:raise ValueError('Original pilot driver changed')
    prior=Path(recover['previous_output']);prior_summary=prior/'pilot_summary.json'
    if sha(prior_summary)!=recover['previous_pilot_summary_sha256']:raise ValueError('Previous pilot outcomes changed')
    old=json.loads(prior_summary.read_text())
    for row in recover['completed_fit_receipts_retained']:
        if sha(Path(row['path']))!=row['sha256']:raise ValueError('Previous completed fit changed')
    admission=json.loads((out/'admission.json').read_text())
    if not admission['source_admission_passed'] or admission['plan_sha256']!=sha(args.plan):raise ValueError('New admission required')
    spec=importlib.util.spec_from_file_location('_continued_pilot_budget',source/'scripts/e2_corrections_budget.py')
    budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)
    with budget.locked_ledger(ledger) as (rows,_):
        prior_group_charges=sum(float(r['elapsed_seconds']) for r in rows if r['event']=='finish' and r['phase'].startswith('followup-HPO-timing'))
    job=next(j for j in plan['main_jobs'] if j['id']=='pilot-03')
    if prior_group_charges+job['cap_seconds']>450:raise ValueError('Recovery would exceed the original450s group cap')
    if (out/job['id']).exists():raise ValueError('No automatic retry or replacement of a prior recovery attempt')
    phase='followup-HPO-timing:'+job['id']
    key=budget.reserve(phase,job['cap_seconds'],path=ledger)
    started,code,process_timeout=time.monotonic(),1,False
    try:
        command=[str(args.python),str(source/'scripts/e2_hpo_followup_pilot.py'),'--worker',
                 '--plan',str(args.plan),'--output',str(out),'--ledger',str(ledger),
                 '--job',job['id'],'--reservation-id',key]
        code=subprocess.run(command,timeout=job['cap_seconds'],check=False).returncode
    except subprocess.TimeoutExpired:
        code,process_timeout=124,True
    finally:
        elapsed=time.monotonic()-started;budget.finish(key,elapsed,code,path=ledger)
    target=out/job['id']/'outcome.json'
    outcome=json.loads(target.read_text()) if target.exists() else {'status':'censored_process_timeout' if process_timeout else 'failed_before_worker_receipt','full_HPO_policy_fit_completed':False}
    charge={'job':job['id'],'model':job['model'],'cap_seconds':job['cap_seconds'],'reservation_id':key,
            'observed_charged_worker_seconds':elapsed,'returncode':code,'outcome':outcome}
    write(out/'pilot-03_worker_charge.json',charge)
    completed_prior=[r for r in old['jobs'] if r['job'] in ['pilot-01','pilot-02']]
    with budget.locked_ledger(ledger) as (rows,_):
        group_rows=[r for r in rows if r['event']=='finish' and r['phase'].startswith('followup-HPO-timing')]
    summary={'version':'e2-three-full-HPO-timing-pilot-recovered-v3','plan_sha256':sha(args.plan),
             'canonical_jobs':completed_prior+[charge],
             'completed_full_policy_fits':sum(r['outcome']['status']=='completed' for r in completed_prior+[charge]),
             'prior_blocked_before_fit_attempt_retained':old['jobs'][2],
             'all_worker_charges_including_prior_source_block_seconds':old['worker_charges_seconds']+elapsed,
             'all_group_charges_including_admissions_and_failure_seconds':sum(r['elapsed_seconds'] for r in group_rows),
             'all_group_charge_rows':group_rows,'original_group_cap_seconds':450,
             'original_group_cap_reset':False,'two_completed_fits_rerun':False,
             'calibration_test_accessed':False,'historical_A2_changed':False,
             'full13model_HPO_forecast_established':False,
             'selection_variance_established':False,'independent_importance_established':False,
             'projection_limit':'Observed completed-job cost is conditional to exactmodel/C03/firstfold/seed/runtime; three representatives do not identify costs for all13bodies. Censored attempts never substitute full costs.'}
    write(out/'pilot_summary.json',summary)
    print(json.dumps({'recovery_job':job['id'],'status':outcome['status'],
                     'total_group_seconds_including_failures_admissions':summary['all_group_charges_including_admissions_and_failure_seconds']}))


if __name__=='__main__':main()
