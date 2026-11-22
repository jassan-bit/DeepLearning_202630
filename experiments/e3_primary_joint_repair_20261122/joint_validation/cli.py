"""Pinned CPU Monte Carlo with hash-chained checkpoints and portable exports.

Only the explicit `run` command evaluates new synthetic replicas. No training,
market acquisition, private credentials, or final-market observations exist.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import sys
import tempfile
import time
import zipfile

# Restrict BLAS in this standalone subprocess before importing the numerical core.
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'

import numpy as np
from scipy.stats import beta

from .logic import cp_interval,cp_lower,cp_upper
from .validation import canonical,digest,one_replica,unmask_hex

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
ZERO='0'*64


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        tmp=Path(stream.name);stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    try:os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def cohort():
    return dict(python_minor=f'{sys.version_info.major}.{sys.version_info.minor}',
        numpy=importlib.metadata.version('numpy'),scipy=importlib.metadata.version('scipy'),
        system=platform.system(),machine=platform.machine(),byteorder=sys.byteorder)


def source_files():
    return {p.relative_to(SOURCE).as_posix():sha(p)
        for folder in ('methods','joint_validation') for p in sorted((SOURCE/folder).glob('*'))
        if p.suffix in ('.py','.cpp') and p.is_file()}


def check_source(identity):
    if source_files()!=identity['files_sha256']:
        raise ValueError('CODE_IDENTITY_MISMATCH: preserve the case; install its pinned source')


def append(path,payload,previous):
    value=dict(payload,previous_sha256=previous)
    value['sha256']=digest(value)
    with path.open('ab') as stream:
        stream.write(canonical(value)+b'\n');stream.flush();os.fsync(stream.fileno())
    return value['sha256']


def read_chain(path,genesis):
    rows=[];previous=genesis
    if not path.exists():return rows,previous
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('TRUNCATED_LEDGER: original bytes retained; no silent truncation or overwrite')
    for line in raw.splitlines():
        row=json.loads(line);claimed=row.pop('sha256',None)
        if row.get('previous_sha256')!=previous or claimed!=digest(row):
            raise ValueError('LEDGER_HASH_MISMATCH: original bytes retained')
        row['sha256']=claimed;rows.append(row);previous=claimed
    return rows,previous


def load_case(case):
    identity=read(case/'source_identity.json')
    claimed=identity.pop('identity_sha256',None)
    if claimed!=digest(identity):raise ValueError('SOURCE_IDENTITY_HASH_MISMATCH')
    identity['identity_sha256']=claimed;check_source(identity)
    plan=read(case/'protocol.json')
    if sha(case/'protocol.json')!=identity['protocol_sha256']:
        raise ValueError('PROTOCOL_HASH_MISMATCH')
    if sha(case/'protocol.json')!=sha(HERE/'protocol.json'):
        raise ValueError('PROTOCOL_IS_NOT_THE_PINNED_PUBLISHED_PLAN')
    rows,last=read_chain(case/'ledger.jsonl',identity['identity_sha256'])
    events,event_last=read_chain(case/'events.jsonl',identity['identity_sha256'])
    expected=[(i,r) for r in range(plan['replicates_per_scenario']) for i in range(len(plan['scenarios']))]
    keys=[(r['scenario_index'],r['replicate']) for r in rows]
    if keys!=expected[:len(keys)]:raise ValueError('REPLICA_ORDER_OR_DUPLICATE_MISMATCH')
    return plan,identity,rows,last,events,event_last


def freeze(case,protocol,source_commit):
    if case.exists():raise ValueError('IMMUTABLE_CASE_EXISTS: resume or choose a new case')
    default=HERE/'protocol.json'
    if sha(protocol)!=sha(default):raise ValueError('Protocol differs from the published fixed plan')
    plan=read(protocol)
    files=source_files()
    if not files or not (SOURCE/'methods/primary_rmse.py').is_file():raise ValueError('Scientific source is incomplete')
    if not source_commit:
        anchor=SOURCE/'PUBLIC_SOURCE_ANCHOR.json'
        source_commit=read(anchor).get('source_commit') if anchor.is_file() else 'local-unpublished'
    identity=dict(source_commit=source_commit,code_id=digest(files),files_sha256=files,
        protocol_sha256=sha(protocol),numerical_cohort=cohort(),created_utc=utc(),
        source_scope='Synthetic statistical validation only; source hashes are not proof of real-market superiority')
    identity['identity_sha256']=digest(identity)
    case.mkdir(parents=True)
    (case/'protocol.json').write_bytes(Path(protocol).read_bytes())
    atomic(case/'source_identity.json',identity)
    atomic(case/'state.json',summarize(plan,identity,[],[]))
    return dict(status='frozen_no_replicas_run',case=str(case),source_commit=source_commit,
        code_id=identity['code_id'],protocol_sha256=identity['protocol_sha256'],new_replicates=0,
        market_fits=0,test_opened=False)


def summarize(plan,identity,rows,events):
    requested=plan['replicates_per_scenario'];k=len(plan['scenarios'])
    alpha_mc=plan['criteria']['MC_family_confidence_error']
    simultaneous_error=alpha_mc/plan['criteria']['MC_total_claims']
    global_error=coverage_error=power_error=simultaneous_error
    scenarios=[];complete=len(rows)==requested*k
    charged=sum(float(r['CPU_seconds']) for r in rows)+sum(float(e.get('CPU_seconds_charged',0.)) for e in events)
    uncertain=any(e.get('cost_uncertain',False) for e in events)
    all_ok=complete
    for i,s in enumerate(plan['scenarios']):
        local=[r for r in rows if r['scenario_index']==i];n=len(local)
        false=sum(r['global_false_acceptance'] for r in local)
        positive=sum(r['joint_acceptance'] for r in local)
        error_upper=cp_upper(false,n,global_error) if n else None
        power_lower=cp_lower(positive,n,global_error) if n else None
        cover={}
        for level in (95,99):
            for kind in ('rmse_difference','relative_rmse_gain'):
                name=f'CI{level}_{kind}';count=sum(r['witness']['coverage'][name] for r in local)
                cover[name]=dict(covered=count,n=n,rate=count/n if n else None,
                    pointwise_CP95=cp_interval(count,n) if n else None,
                    lower_simultaneous_MC=cp_lower(count,n,coverage_error) if n and level==99 else None,
                    floor=plan['criteria']['CI99_coverage_floor'] if level==99 else None,
                    nominal=level/100)
        contrast_counts=np.stack([unmask_hex(r['contrast_pass']) for r in local]).sum(axis=0) if local else np.zeros(80,dtype=int)
        if n and s['kind']=='alternative':
            lower=np.zeros(80)
            valid=contrast_counts>0
            lower[valid]=beta.ppf(power_error,contrast_counts[valid],n-contrast_counts[valid]+1)
            min_power_lower=float(lower.min())
        else:min_power_lower=None
        null_ok=n==requested and error_upper is not None and error_upper<=plan['criteria']['global_type_I_ceiling']
        positive_ok=s['kind']=='alternative' and n==requested and power_lower is not None and power_lower>=plan['criteria']['strong_joint_power_floor'] and min_power_lower is not None and min_power_lower>=plan['criteria']['strong_per_contrast_power_floor']
        cover_ok=n==requested and all(cover[f'CI99_{kind}']['lower_simultaneous_MC']>=plan['criteria']['CI99_coverage_floor'] for kind in ('rmse_difference','relative_rmse_gain'))
        gate=bool((positive_ok if s['kind']=='alternative' else null_ok) and cover_ok)
        all_ok=all_ok and gate
        scenarios.append(dict(id=s['id'],kind=s['kind'],n=n,requested=requested,
            global_false_acceptances=false,global_false_acceptance_rate=false/n if n else None,
            global_type_I_upper_simultaneous_MC=error_upper if s['kind']=='null' else None,
            joint_acceptances=positive,joint_acceptance_rate=positive/n if n else None,
            strong_joint_power_lower_simultaneous_MC=power_lower if s['kind']=='alternative' else None,
            strong_minimum_80_contrast_power_lower_simultaneous_MC=min_power_lower if s['kind']=='alternative' else None,
            coverage=cover,criterion_pass=gate,global_criterion_pass=positive_ok if s['kind']=='alternative' else null_ok,
            CI99_floor_criterion_pass=cover_ok,
            stress_outside_stationarity=bool(s.get('variance_break')),
            statement='Stress must satisfy the same predeclared numerical criteria; it does not extend a stationarity theorem' if s.get('variance_break') else 'Conditional fixed-DGP validation'))
    return dict(status='completed' if complete else 'incomplete',accepted=bool(all_ok),
        completed_replicates=len(rows),requested_replicates=requested*k,
        protocol_sha256=identity['protocol_sha256'],source_commit=identity['source_commit'],code_id=identity['code_id'],
        numerical_cohort=identity['numerical_cohort'],scenarios=scenarios,
        charged_CPU_seconds=charged,CPU_budget_seconds=plan['CPU_budget_seconds'],
        recorded_CPU_remaining=max(0.,plan['CPU_budget_seconds']-charged),CPU_cost_uncertain=uncertain,
        MC_confidence=dict(global_family_error=alpha_mc,total_claims=plan['criteria']['MC_total_claims'],
            error_each=simultaneous_error,global_null_claims=8,CI99_coverage_claims=18,
            strong_joint_power_claims=1,strong_per_contrast_power_claims=80,
            scope='Bonferroni confidence simultaneously over exactly 107 fixed synthetic-DGP statements; not universal calibration'),
        R3_accredited=False,market_power_estimated=False,market_fits=0,test_opened=False)


def run(case,resume,max_replicates,seconds_budget,workers):
    lock=case/'.run.lock'
    if lock.exists():
        owner=read(lock)
        try:os.kill(int(owner['pid']),0)
        except ProcessLookupError:
            old=case/('.stale_run_lock_'+str(time.time_ns())+'.json')
            os.replace(lock,old)
        else:raise ValueError('CASE_ALREADY_RUNNING: concurrent ledger writers are forbidden')
    try:
        descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        raise ValueError('CASE_ALREADY_RUNNING: concurrent ledger writers are forbidden') from None
    with os.fdopen(descriptor,'w') as stream:json.dump(dict(pid=os.getpid(),started_utc=utc()),stream)
    try:return _run_locked(case,resume,max_replicates,seconds_budget,workers)
    finally:lock.unlink(missing_ok=True)


def _run_locked(case,resume,max_replicates,seconds_budget,workers):
    plan,identity,rows,last,events,event_last=load_case(case)
    if rows and not resume:raise ValueError('Existing replicas require --resume')
    if not re.fullmatch('[0-9a-f]{40}',identity['source_commit']):
        raise ValueError('PUBLIC_SOURCE_ANCHOR_REQUIRED_BEFORE_FORMAL_MC')
    state=summarize(plan,identity,rows,events)
    if state['status']=='completed':return state
    if cohort()!=identity['numerical_cohort']:
        raise ValueError('COHORT_INCOMPATIBLE: inspect/export is portable, but new replicas require matching numerical cohort or a new case')
    if not 1<=workers<=plan['max_workers']:raise ValueError('Worker count exceeds the frozen plan')
    marker=case/'SESSION_OPEN.json'
    if marker.exists():
        previous=read(marker)
        payload=dict(kind='previous_session_unclosed',recorded_utc=utc(),cost_uncertain=True,
            CPU_seconds_charged=0.,details=previous,
            interpretation='Observed replica CPU is retained as a lower bound; unknown interrupted work is not invented')
        event_last=append(case/'events.jsonl',payload,event_last);events.append(payload)
        marker.unlink()
    budget=min(float(seconds_budget),state['recorded_CPU_remaining'])
    if budget<=0:return dict(state,stop_reason='frozen_CPU_budget_exhausted')
    if max_replicates<1:raise ValueError('Positive chunk replica count required')
    atomic(marker,dict(started_utc=utc(),pid=os.getpid(),planned_chunk_CPU_budget=budget,
        starting_replicas=len(rows),workers=workers))
    todo=[(i,r) for r in range(plan['replicates_per_scenario']) for i in range(len(plan['scenarios']))][len(rows):]
    todo=todo[:max_replicates]
    child_start=resource.getrusage(resource.RUSAGE_CHILDREN)
    child_start=child_start.ru_utime+child_start.ru_stime
    chunk_cpu=0.;parent_start=time.process_time();reported=time.monotonic()
    try:
        # Fixed small batches bound surplus in-flight work and preserve ledger order.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for start in range(0,len(todo),workers):
                if chunk_cpu>=budget:break
                batch=todo[start:start+workers]
                futures=[pool.submit(one_replica,plan,i,r) for i,r in batch]
                for future in futures:
                    record=future.result()
                    last=append(case/'ledger.jsonl',record,last)
                    rows.append(record);chunk_cpu+=record['CPU_seconds']
                    if time.monotonic()-reported>=30:
                        print(json.dumps(dict(event='validation_progress',completed=len(rows),total=plan['replicates_per_scenario']*len(plan['scenarios']),chunk_CPU_seconds=chunk_cpu)),flush=True)
                        reported=time.monotonic()
                atomic(case/'state.json',summarize(plan,identity,rows,events))
        child_end=resource.getrusage(resource.RUSAGE_CHILDREN)
        total_cpu=time.process_time()-parent_start+child_end.ru_utime+child_end.ru_stime-child_start
        overhead=max(0.,total_cpu-chunk_cpu)
        event=dict(kind='chunk_finished',recorded_utc=utc(),CPU_seconds_charged=overhead,
            process_pool_and_setup_overhead_CPU_seconds=overhead,replica_CPU_seconds=chunk_cpu,
            total_observed_parent_and_children_CPU_seconds=total_cpu,
            actual_last_batch_can_exceed_soft_check_budget=True,cost_uncertain=False)
        append(case/'events.jsonl',event,event_last);events.append(event)
        marker.unlink()
    except BaseException:
        # Keep SESSION_OPEN so later recovery never reports unobserved work as exact.
        atomic(case/'state.json',summarize(plan,identity,rows,events))
        raise
    state=summarize(plan,identity,rows,events)
    atomic(case/'state.json',state)
    return state


def export(case,output):
    load_case(case)
    if output.exists():raise ValueError('Export already exists')
    if any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Export cannot traverse symlinks')
    output.parent.mkdir(parents=True,exist_ok=True)
    files=[p for p in case.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')]
    payload={p.name:p.read_bytes() for p in files}
    manifest=dict(files={n:hashlib.sha256(b).hexdigest() for n,b in payload.items()},
        scope='Synthetic statistical checkpoint, no market rows, weights or credentials')
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,body in payload.items():z.writestr(name,body)
        z.writestr('EXPORT_MANIFEST.json',canonical(manifest))
    return dict(status='exported',archive=str(output),bytes=output.stat().st_size,sha256=sha(output))


def restore(archive,case,source_commit):
    if case.exists():raise ValueError('Restore refuses to overwrite an existing directory')
    if any(p.is_symlink() for p in (case,*case.parents)):raise ValueError('Restore cannot traverse symlinks')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any(Path(n).name!=n or not n.endswith(('.json','.jsonl')) for n in names):
            raise ValueError('Unsafe or duplicate archive paths')
        if sum(x.file_size for x in z.infolist())>64*1024*1024:raise ValueError('Archive exceeds 64 MiB')
        manifest=json.loads(z.read('EXPORT_MANIFEST.json'))
        if set(names)!=set(manifest['files'])|{'EXPORT_MANIFEST.json'}:raise ValueError('Export inventory differs')
        payload={n:z.read(n) for n in manifest['files']}
        if any(hashlib.sha256(b).hexdigest()!=manifest['files'][n] for n,b in payload.items()):raise ValueError('Export file checksum mismatch')
    case.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=case.parent) as temporary:
        staging=Path(temporary)/'case';staging.mkdir()
        for name,body in payload.items():(staging/name).write_bytes(body)
        plan,identity,rows,last,events,event_last=load_case(staging)
        # Source commits may differ solely by added report artifacts; scientific
        # file identities, original protocol and original source event stay bound.
        append(staging/'events.jsonl',dict(kind='restored',recorded_utc=utc(),
            current_source_commit=source_commit,CPU_seconds_charged=0.,cost_uncertain=False,
            original_source_commit=identity['source_commit'],numerical_cohort=cohort()),event_last)
        os.replace(staging,case)
    return dict(status='restored_without_new_replicas',case=str(case),source_commit=identity['source_commit'],
        completed_replicates=len(rows),numerical_cohort_matches=cohort()==identity['numerical_cohort'])


def main(argv=None):
    p=argparse.ArgumentParser();commands=p.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze');f.add_argument('--case','--output',dest='case',type=Path,required=True)
    f.add_argument('--protocol',type=Path,default=HERE/'protocol.json');f.add_argument('--source-commit')
    r=commands.add_parser('run');r.add_argument('--case',type=Path,required=True);r.add_argument('--resume',action='store_true')
    r.add_argument('--max-replicates',type=int,default=16);r.add_argument('--seconds-budget',type=float,default=300)
    r.add_argument('--workers',type=int,default=4)
    s=commands.add_parser('status');s.add_argument('--case',type=Path,required=True)
    e=commands.add_parser('export');e.add_argument('--case',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    x=commands.add_parser('restore');x.add_argument('--archive',type=Path,required=True)
    x.add_argument('--case','--output',dest='case',type=Path,required=True);x.add_argument('--source-commit')
    a=p.parse_args(argv)
    if a.command=='freeze':result=freeze(a.case,a.protocol,a.source_commit)
    elif a.command=='run':result=run(a.case,a.resume,a.max_replicates,a.seconds_budget,a.workers)
    elif a.command=='status':
        plan,identity,rows,_,events,_=load_case(a.case);result=summarize(plan,identity,rows,events)
    elif a.command=='export':result=export(a.case,a.output)
    else:result=restore(a.archive,a.case,a.source_commit)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
