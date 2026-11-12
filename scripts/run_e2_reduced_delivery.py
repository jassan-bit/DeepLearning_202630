"""One sequential, resumable reduced-E2 delivery controller.

Planning is read-only. --execute uses one persistent64-hour ledger for HPO,
pilots, final refits, classical comparators and final inference. Every closed
scientific result is backed up before another fit. A pause is never completion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
VERSION='e2-reduced-delivery-controller-v1'
STAGES=('all','selection','final-fit','classical','pilot','freeze','score')


def read_json(path):
    return json.loads(Path(path).read_text())


def _persist_snapshot(directory,relative,client,*,kind):
    from e2_reduced_fit import file_sha
    from e2_reduced_protocol import digest
    directory=Path(directory)
    files={str(p):file_sha(directory/p) for p in relative}
    if not files:raise ValueError('A delivery snapshot must contain verified artifacts')
    receipt={'version':VERSION,'status':'completed','phase':kind,
        'attempt_id':kind+'-'+digest(files)[:24],'artifacts_sha256':files,
        'snapshot_completion_is_not_benchmark_completion':True}
    client.persist_result(directory,receipt)
    return receipt


def _load_selections(output):
    path=Path(output)/'selection_complete.json'
    if not path.is_file():raise ValueError('Complete the reduced HPO selection stage first')
    value=read_json(path)
    selections=value.get('selections')
    if not isinstance(selections,dict) or not selections:
        raise ValueError('No audited winning recipes are available')
    return selections


def _source():
    from run_e2_reduced_selection import source_identity
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    return {'source_commit':commit,'files_sha256':source_identity()}


def execute_delivery(args,budget,client):
    """Injectable orchestration boundary; the same budget owns every stage."""
    from e2_reduced_protocol import NEURAL, SYSTEMS
    from e2_reduced_fit import write_json
    from run_e2_reduced_selection import run_selection
    from e2_reduced_final import (run_final_fits,run_classical,close_pilot,
        freeze_final,run_final_scoring)
    from e2_reduced_final_analysis import analysis_registration
    output=Path(args.output);final_output=output/'final'
    output.mkdir(parents=True,exist_ok=True)
    stage=args.stage
    source=_source();runtime=read_json(args.runtime_profile)['runtime'];roots=read_json(args.source_roots)
    # Freeze inferential settings before any fitting or final-market admission.
    write_json(final_output/'analysis_registration.json',analysis_registration())
    common={'source_identity':source,'runtime_identity':runtime,'source_roots':roots,
        'device':'cuda:0','persist_result':client.persist_result,'python':getattr(args,'python',None)}
    completed=[]
    if stage in ('all','selection'):
        print(json.dumps({'event':'stage_started','stage':'selection','remaining_seconds':budget.remaining_seconds}),flush=True)
        selection=run_selection(args,budget,read_json(args.cache_audit),args.reconciliation_root,
            persist_result=client.persist_result)
        files=['selection_complete.json']+[p.relative_to(output).as_posix() for p in (output/'selections').glob('*.json')]
        _persist_snapshot(output,files,client,kind='reduced-selection-closure')
        completed.append('selection')
        selections=selection['selections']
    else:
        selections=_load_selections(output) if stage in ('final-fit','pilot') else None
    if stage in ('all','final-fit','pilot'):
        requested=getattr(args,'models',None)
        print(json.dumps({'event':'stage_started','stage':'pilot_and_final_fit' if stage=='all' else stage,
            'remaining_seconds':budget.remaining_seconds}),flush=True)
        report=run_final_fits(args.data_root,selections,final_output,budget,models=requested,
            pilot=stage!='final-fit',final=stage!='pilot',**common)
        if report['pending']:
            raise RuntimeError('Reduced neural recipes remain pending: '+','.join(p['model'] for p in report['pending']))
        completed.append('pilot_and_final_fit' if stage=='all' else stage)
        if stage in ('all','pilot'):
            close_pilot(final_output,persist_result=client.persist_result)
            completed.append('pilot_power_MDE')
    if stage in ('all','classical'):
        print(json.dumps({'event':'stage_started','stage':'classical','remaining_seconds':budget.remaining_seconds}),flush=True)
        classical_common={k:v for k,v in common.items() if k!='source_roots'}
        run_classical(args.data_root,final_output,budget,**classical_common)
        completed.append('classical')
    if stage in ('all','freeze'):
        print(json.dumps({'event':'stage_started','stage':'freeze','remaining_seconds':budget.remaining_seconds}),flush=True)
        freeze_final(final_output,persist_result=client.persist_result)
        completed.append('freeze_all81_predictors')
    score=None
    if stage in ('all','score'):
        print(json.dumps({'event':'stage_started','stage':'score','remaining_seconds':budget.remaining_seconds}),flush=True)
        score=run_final_scoring(ROOT,args.data_root,final_output,budget,daily_csv=getattr(args,'daily_csv',None),
            raw_root=getattr(args,'raw_root',None) or output/'private_raw_E1',
            allow_download=getattr(args,'allow_final_download',False),**common)
        if score.get('status')!='completed' or score.get('systems')!=16:
            raise RuntimeError('The16-system final benchmark has not completed')
        completed.append('final_scoring_and_analysis')
    report={'version':VERSION,'stage':stage,
        'status':'final_benchmark_completed' if score is not None else 'requested_stage_completed',
        'completed_stages':completed,'all_sixteen_systems_benchmark_complete':score is not None,
        'final_market_values_read':score is not None,'no_automatic_retry':True,'budget':budget.report(),
        'remaining_requirements_scope':'A completed benchmark is not a claim that every guide narrative/diagnostic item is finished'}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    relative='delivery_status/status-'+stamp+'.json';write_json(output/relative,report)
    _persist_snapshot(output,[relative],client,kind='reduced-delivery-status')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--stage',choices=STAGES,default='all')
    for name in ('output','data-root','source-roots','runtime-profile','budget-evidence','migration','cache-audit','reconciliation-root'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--migration-registration-commit')
    p.add_argument('--inherited-ledger',type=Path,action='append',default=[])
    p.add_argument('--models',nargs='+');p.add_argument('--python',default=sys.executable)
    p.add_argument('--daily-csv',type=Path);p.add_argument('--raw-root',type=Path)
    p.add_argument('--allow-final-download',action='store_true',
        help='After the full predictor freeze only, acquire pinned known-E1 raw archives if no daily cache exists')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if not args.execute:
        from e2_reduced_protocol import SYSTEMS
        print(json.dumps({'version':VERSION,'stage':args.stage,'systems':list(SYSTEMS),
            'training_started':False,'final_market_values_read':False,'budget_extended':False}))
        return 0
    required=('output','data_root','source_roots','runtime_profile','budget_evidence','migration','cache_audit','reconciliation_root','migration_registration_commit')
    if any(getattr(args,key,None) is None for key in required):
        p.error('Execution requires the registered code, data, runtime, budget, cache and migration paths')
    try:
        from e2_reduced_budget import ReducedBudgetLedger
        from e2_reduced_github_backup import GitHubBackup
        client=GitHubBackup();client.preflight()  # GitHub writes checked before any optimizer admission.
        with ReducedBudgetLedger(args.output/'shared_budget',read_json(args.budget_evidence),
            migration=read_json(args.migration),migration_registration_commit=args.migration_registration_commit,
            inherited_ledgers=args.inherited_ledger) as budget:
            execute_delivery(args,budget,client)
        return 0
    except (KeyboardInterrupt,Exception) as error:
        # The bounded-worker supervisor terminates a live child and charges it.
        # The durable lease/fit is retained. A rerun may upload closed artifacts,
        # but an open/failed optimizer attempt still needs explicit reconciliation.
        report={'version':VERSION,'status':'paused_not_complete','stage':args.stage,
            'error_type':type(error).__name__,'reason':str(error),
            'automatic_retry':False,'budget_extended':False,
            'all_sixteen_systems_benchmark_complete':False,
            'preserved_output':str(args.output)}
        print(json.dumps(report,ensure_ascii=False),file=sys.stderr,flush=True)
        return 3


if __name__=='__main__':raise SystemExit(main())
