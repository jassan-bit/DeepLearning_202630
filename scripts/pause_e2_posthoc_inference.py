"""Explicit inference-only pause/reconciliation; never an optimizer retry."""
from __future__ import annotations
import argparse, fcntl, hashlib, importlib, io, json, math, os
from pathlib import Path
import signal, sys, time, zipfile
from datetime import datetime, timezone

PIN='ae1a2d4f96807a20ea92c313297a5c09512a4817'
NEW_RELEASE='e2-guide-closure-posthoc-20261113-v1'
VERSION='e2-inference-transport-pause-v1'
ZERO='0'*64

def canonical(v):
    return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(v): return hashlib.sha256(v).hexdigest()
def file_sha(p): return sha(Path(p).read_bytes())
def utc(): return datetime.now(timezone.utc).isoformat()
def stamp(v):
    x=datetime.fromisoformat(v.replace('Z','+00:00'))
    if x.tzinfo is None: raise ValueError('Timezone-aware observer times required')
    return x

def regular(p):
    p=Path(p).absolute()
    if any(x.is_symlink() for x in (p,*p.parents)): raise ValueError('Symlink path rejected')
    return p

def immutable(p,v):
    p=regular(p);raw=canonical(v)+b'\n';p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=raw: raise ValueError('Immutable proof differs')
        return
    with p.open('xb') as f: f.write(raw);f.flush();os.fsync(f.fileno())

def process(pid,proc=Path('/proc')):
    try:
        args=(proc/str(pid)/'cmdline').read_bytes().split(b'\0')
        args=[x.decode() for x in args if x]
        fields=(proc/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        if not args or fields[0]=='Z': return None
        return {'pid':int(pid),'argv':args,'ppid':int(fields[1]),'start_ticks':fields[19]}
    except (OSError,UnicodeError,IndexError,ValueError): return None

def processes(proc=Path('/proc')):
    return [p for x in proc.glob('[0-9]*') if (p:=process(x.name,proc)) is not None]

def option(args,key):
    if args.count(key)!=1: raise ValueError('Expected one '+key)
    i=args.index(key)
    if i+1==len(args): raise ValueError('Missing '+key)
    return args[i+1]

def matching_controller(p,destination,commit=PIN):
    args=p['argv']
    if '--orchestrate' not in args: return False
    try:
        extension=Path(option(args,'--extension-root')).absolute()
        return (option(args,'--configuration-commit')==commit and
                Path(option(args,'--destination')).absolute()==Path(destination).absolute() and
                extension.name=='guide-closure-extension-'+commit and
                str(extension/'scripts/e2_posthoc_fit_reproduction.py') in args)
    except ValueError: return False

def no_optimizer(destination,records=None):
    root=regular(destination)
    if list((root/'reproduction/jobs').glob('*/worker_spec.json')):
        raise RuntimeError('An optimizer was admitted; inference-only amendment prohibited')
    if (root/'reproduction/clean/setup_started.json').exists():
        raise RuntimeError('Clean-fit preparation began; inference-only amendment prohibited')
    for p in processes() if records is None else records:
        if '--worker-spec' in p['argv'] and any(str(root) in a for a in p['argv']):
            raise RuntimeError('A posthoc optimizer child is still present')

def ledger_rows(path):
    raw=regular(path).read_bytes()
    if not raw or not raw.endswith(b'\n'): raise ValueError('Complete ledger required')
    rows=[];active={};previous=ZERO
    for index,line in enumerate(raw.splitlines()):
        r=json.loads(line);body={k:v for k,v in r.items() if k!='sha256'}
        if r.get('sha256')!=sha(canonical(body)) or r.get('sequence')!=index or r.get('previous_sha256')!=previous:
            raise ValueError('Ledger hash chain differs')
        previous=r['sha256'];rows.append(r)
        if r['kind']=='reserved': active[r['job_id']]=r
        elif r['kind']=='charged': active.pop(r['job_id'],None)
        elif r['kind'] not in ('interrupted','overhead'): raise ValueError('Unknown ledger row')
    if any(r['kind']=='reserved' and (
            r.get('metadata',{}).get('kind') in {'reproduction','ablation'}
            or r.get('job_id','').startswith(('posthoc-reproduction-','posthoc-ablation-','clean-setup-')))
            for r in rows):
        raise RuntimeError('An earlier optimizer or clean setup was admitted; inference-only recovery prohibited')
    if len(active)!=1: raise RuntimeError('Exactly one unresolved inference reservation required')
    r=next(iter(active.values()))
    if not r['job_id'].startswith('frozen-posthoc-') or not 0<r['seconds']<=5400:
        raise RuntimeError('Reservation is not the bounded frozen-inference phase')
    if r.get('metadata',{}).get('training_performed') is not False:
        raise RuntimeError('Reservation does not explicitly exclude training')
    return raw,rows,r

def complete_predictors(destination):
    root=regular(destination);found={}
    for path in sorted((root/'robustness/predictors').glob('*/result.json')):
        r=json.loads(regular(path).read_bytes())
        if r.get('status')!='completed' or not r.get('artifacts_sha256'): raise ValueError('Invalid completed predictor')
        for relative,expected in r['artifacts_sha256'].items():
            item=regular(path.parent/relative)
            if not item.resolve().is_relative_to(path.parent.resolve()) or file_sha(item)!=expected:
                raise ValueError('Completed predictor artifact differs')
        backup=path.parent/'github_backup_receipt.json'
        if backup.exists() and json.loads(backup.read_bytes()).get('status')=='backed_up':
            found[str(path.relative_to(root))]=file_sha(path)
    return found

def remote_completed(root,records):
    from urllib.request import Request,urlopen
    url=('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'
         'results/e2-reduced-delivery-v1/control/e2-guide-closure-posthoc-v1/head.json?pause='+str(time.time_ns()))
    try:
        with urlopen(Request(url,headers={'Cache-Control':'no-cache'}),timeout=10) as response:
            raw=response.read(4*1024**2+1)
        if len(raw)>4*1024**2: return False
        value=json.loads(raw);body={k:v for k,v in value.items() if k!='state_id'}
        if value.get('state_id')!=sha(canonical(body)) or value.get('output_path')!=str(root): return False
        return all(value.get('files',{}).get(relative,{}).get('sha256')==expected
                   and str(Path(relative).parent) in value.get('result_backups',{})
                   for relative,expected in records.items())
    except (OSError,ValueError,TimeoutError): return False

def pause(destination,*,pid=None,commit=PIN,timeout=1200.,poll=.25):
    """Terminal-only observer. Signal only after a new durably backed predictor."""
    root=regular(destination);no_optimizer(root)
    candidates=[p for p in processes() if matching_controller(p,root,commit)]
    if pid is not None: candidates=[p for p in candidates if p['pid']==pid]
    if len(candidates)!=1: raise RuntimeError('Exactly one matching original controller required')
    p=candidates[0];initial=complete_predictors(root)
    path=root/'budget/posthoc_budget.jsonl';before,rows,reservation=ledger_rows(path)
    observed_at=utc();started=time.monotonic()
    print('Esperando un predictor nuevo respaldado antes de la pausa de transporte...',flush=True)
    while True:
        current=process(p['pid'])
        if current is None or current['start_ticks']!=p['start_ticks']: raise RuntimeError('Controller exited before controlled signal')
        no_optimizer(root)
        raw,rs,r=ledger_rows(path)
        if r['job_id']!=reservation['job_id']: raise RuntimeError('Active phase changed while observing')
        done=complete_predictors(root);new={k:v for k,v in done.items() if k not in initial}
        if len(done)>=79: raise RuntimeError('Inference is closing; no signal sent at a fitting-stage boundary')
        if new and remote_completed(root,new): break
        if time.monotonic()-started>timeout: raise TimeoutError('No new complete backed-up predictor; no signal sent')
        time.sleep(poll)
    current=process(p['pid'])
    if current is None or current['start_ticks']!=p['start_ticks'] or not matching_controller(current,root,commit):
        raise RuntimeError('Controller identity changed')
    no_optimizer(root);signal_at=utc();signal_clock=time.monotonic()
    os.kill(p['pid'],signal.SIGINT)
    while True:
        current=process(p['pid'])
        if current is None or current['start_ticks']!=p['start_ticks']: break
        if time.monotonic()-signal_clock>120: raise TimeoutError('Controller has not exited; do not reconcile')
        time.sleep(poll)
    exited_at=utc();stop_elapsed=time.monotonic()-signal_clock
    no_optimizer(root)
    if any(matching_controller(x,root,commit) for x in processes()): raise RuntimeError('Another controller remains')
    after,rs,r=ledger_rows(path)
    interrupted=[x for x in rs if x['kind']=='interrupted' and x.get('job_id')==r['job_id']]
    if not interrupted: raise RuntimeError('Durable interrupted row absent; retain full reservation')
    proof={'version':VERSION,'configuration_commit':commit,'destination':str(root),'pid':p['pid'],
        'pid_start_ticks':p['start_ticks'],'cmdline':p['argv'],'cmdline_sha256':sha(canonical(p['argv'])),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'observer_started_at_utc':observed_at,'signal_sent_at_utc':signal_at,'controller_exited_at_utc':exited_at,
        'signal_to_exit_monotonic_seconds':stop_elapsed,'signal':'SIGINT','no_optimizer_specs':True,
        'no_matching_controller_after_exit':True,'new_completed_predictors':new,
        'ledger_before_observer_sha256':sha(before),'ledger_paused_sha256':sha(after),
        'ledger_paused_bytes':len(after),'reservation':r,'interrupted_record':interrupted[-1]}
    proof['proof_id']=sha(canonical(proof))
    target=root/'transport_amendments'/'pause_observers'/(proof['proof_id']+'.json')
    immutable(target,proof)
    print(json.dumps({'status':'controller_exited_inference_only','proof':str(target),'reservation_retained':True}),flush=True)
    return proof

def load_extension(extension_root,destination):
    root=regular(extension_root)
    path=root/'experiments/e2_posthoc_fit_reproduction_20261113/registration.json'
    registration=json.loads(path.read_bytes())
    for relative,expected in registration['files_sha256'].items():
        if file_sha(regular(root/relative))!=expected: raise ValueError('Original registered source changed')
    local=json.loads((Path(destination)/'registration.json').read_bytes())
    if local.get('configuration_commit')!=PIN or local.get('registration_sha256')!=file_sha(path):
        raise ValueError('Original ae1 configuration identity changed')
    sys.path.insert(0,str(root/'scripts'))
    old=importlib.import_module('e2_posthoc_fit_reproduction')
    if Path(old.__file__).resolve()!=root/'scripts/e2_posthoc_fit_reproduction.py':
        raise ValueError('Original ledger module was shadowed')
    return old

def recovery_ledger(old,root,anchor):
    """Explicit two-lock replay; deliberately does not discard unresolved work."""
    value=object.__new__(old.PosthocLedger)
    value.root=old.regular(root);value.anchor=dict(anchor);value.clock=time.monotonic
    value.lock=None;value.original_lock=None;value.persistence=None
    value.events=[];value.active={};value.attempts=set();value.charges=0.
    value.path=old.regular(value.root/'posthoc_budget.jsonl')
    try:
        value.lock=(value.root/'.posthoc-budget.lock').open('a+b')
        fcntl.flock(value.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        original=old.regular(anchor['original_ledger_path'])
        value.original_lock=(original.parent/'.pilot-budget.lock').open('a+b')
        fcntl.flock(value.original_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if file_sha(original)!=old.LEDGER_SHA256: raise ValueError('Original ledger changed')
        if json.loads((value.root/'original_anchor.json').read_bytes())!=anchor:
            raise ValueError('Original anchor differs')
        raw=value.path.read_bytes()
        if not raw.endswith(b'\n'): raise ValueError('Partial ledger tail')
        for line in raw.splitlines(): value._apply(json.loads(line))
        return value
    except BaseException:
        value.close();raise

def restore_controlled_prefix(remote,root,proof):
    """Accept only the exact controlled interruption suffix after last CAS."""
    try:
        remote.restore()
    except Exception as error:
        if (str(error)=='Remote posthoc reservation unresolved; no lost fit is retried or credited'
                and remote.state._restored): return
        if 'local_control_diverges' not in str(error): raise
        state=remote.state;head=state.client._branch_head();current=state._current(head)
        if current is None or current.get('output_path')!=str(Path(root).absolute()):
            raise ValueError('Latest remote control identity differs')
        files,_=state._scan();ledger='budget/posthoc_budget.jsonl'
        if set(files)!=set(current['files']): raise ValueError('Unpublished metadata beyond interrupted ledger')
        if any(files[k]!=current['files'][k] for k in files if k!=ledger):
            raise ValueError('Local/remote metadata differs outside controlled interruption')
        remote_raw=None
        for descriptor in current['metadata_deltas']:
            raw=state._content_bytes(descriptor['path'],head)
            if raw is None or len(raw)!=descriptor['bytes'] or sha(raw)!=descriptor['sha256']:
                raise ValueError('Remote control archive differs')
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if ledger in archive.namelist(): remote_raw=archive.read(ledger)
        local=(Path(root)/ledger).read_bytes()
        if (remote_raw is None or sha(remote_raw)!=current['files'][ledger]['sha256']
                or not local.startswith(remote_raw)):
            raise ValueError('Remote ledger is not exact paused-history prefix')
        suffix=[json.loads(x) for x in local[len(remote_raw):].splitlines()]
        if suffix!=[proof['interrupted_record']]:
            raise ValueError('Only the proven controlled interrupted row may extend remote')
        state._state=current;state._files=current['files'];state._archives=current['metadata_deltas']
        state._backups=current.get('result_backups',{});state._restored=True
        return
    raise ValueError('No unresolved reservation was restored; do not reconcile twice')

def upper_charge(proof,reservation):
    lower=proof['interrupted_record']['elapsed_seconds_lower_bound']
    if not isinstance(lower,(int,float)) or not math.isfinite(lower) or lower<0:
        raise ValueError('Invalid durable elapsed lower bound')
    utc_elapsed=(stamp(proof['controller_exited_at_utc'])-stamp(reservation['utc'])).total_seconds()
    stop=proof['signal_to_exit_monotonic_seconds']
    if not isinstance(stop,(int,float)) or not math.isfinite(stop) or stop<0:
        raise ValueError('Invalid observer monotonic interval')
    upper=max(lower+stop,utc_elapsed)+5.
    if upper>reservation['seconds']:
        raise ValueError('Observed upper bound exceeds reservation; retain full cost and reconcile manually')
    return upper

def archive_partials(root,target):
    root=regular(root);target=regular(target);entries=[]
    folders=sorted((root/'robustness/predictors').glob('*'))
    for folder in folders:
        if not folder.is_dir() or (folder/'result.json').exists(): continue
        regular(folder);files={}
        for item in sorted(folder.rglob('*')):
            regular(item)
            if item.is_file():
                if item.suffix.lower() in {'.pt','.pth','.ckpt','.pkl','.safetensors'}:
                    raise ValueError('Weights in partial inference folder; no automatic archival')
                files[str(item.relative_to(folder))]=file_sha(item)
        destination=target/folder.name
        if destination.exists(): raise ValueError('Archive target already exists')
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.rename(folder,destination)
        if any(file_sha(destination/name)!=expected for name,expected in files.items()):
            raise ValueError('Archived inference bytes differ')
        entries.append({'original':str(folder.relative_to(root)),
                        'archived':str(destination.relative_to(root)),'files_sha256':files})
    return entries

def reconcile(proof_path,*,token,extension_root=None,original_output=None,transport_client=None):
    """Call from quiescent kernel with existing credential; never through CLI."""
    started=time.monotonic();proof=json.loads(regular(proof_path).read_bytes())
    if proof.get('version')!=VERSION or proof.get('configuration_commit')!=PIN:
        raise ValueError('Original ae1 controlled-pause proof required')
    if proof.get('proof_id')!=sha(canonical({k:v for k,v in proof.items() if k!='proof_id'})):
        raise ValueError('Observer proof hash differs')
    if proof['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise RuntimeError('Observer proof belongs to a different runtime; retain reservation')
    root=regular(proof['destination']);no_optimizer(root)
    if any(matching_controller(p,root) for p in processes()): raise RuntimeError('Controller still active')
    prior=process(proof['pid'])
    if prior is not None and prior['start_ticks']==proof['pid_start_ticks']:
        raise RuntimeError('Observed controller has not exited')
    raw,rows,reservation=ledger_rows(root/'budget/posthoc_budget.jsonl')
    if sha(raw)!=proof['ledger_paused_sha256'] or len(raw)!=proof['ledger_paused_bytes']:
        raise ValueError('Paused journal changed since observer exit')
    if reservation!=proof['reservation'] or proof['interrupted_record'] not in rows:
        raise ValueError('Proof is not bound to current inference reservation')
    command=proof['cmdline']
    if sha(canonical(command))!=proof['cmdline_sha256']: raise ValueError('Command proof differs')
    extension_root=regular(extension_root or option(command,'--extension-root'))
    original_output=regular(original_output or option(command,'--original-output'))
    old=load_extension(extension_root,root);anchor=old.validate_original_anchor(original_output)
    if not isinstance(token,str) or not token or any(c.isspace() for c in token):
        raise ValueError('Use the existing private kernel credential, never a CLI argument')
    backup=importlib.import_module('e2_reduced_github_backup')
    if Path(backup.__file__).resolve()!=extension_root/'scripts/e2_reduced_github_backup.py':
        raise ValueError('Transport module was shadowed')
    backup.RELEASE_TAG=NEW_RELEASE
    client=transport_client or backup.GitHubBackup(token,timeout=20,max_conflicts=2)
    client._branch_head();client._checked=True
    remote=old.PosthocRemoteState(root,client,extension_root)
    ledger=recovery_ledger(old,root/'budget',anchor)
    try:
        if ledger.active!={reservation['job_id']:reservation['seconds']}:
            raise ValueError('Recovered active phase differs')
        restore_controlled_prefix(remote,root,proof)
        ledger.set_persistence(remote)
        upper=upper_charge(proof,reservation)
        previous_charges=ledger.charges
        ledger.charge(reservation['job_id'],upper,status='failed',evidence={
            'controlled_inference_stop_proof_sha256':file_sha(proof_path),
            'observed_PID_exited':True,'no_optimizer_admitted':True,
            'charge_is_conservative_elapsed_upper_bound':True,'scientific_fit_retry':False})
        if not math.isclose(ledger.charges,previous_charges+upper,abs_tol=1e-9):
            raise ValueError('Earlier charges were not preserved')
        ledger.charge_overhead(time.monotonic()-started,description='explicit_inference_transport_reconciliation_prelude')
        amend='transport-amendment-'+proof['proof_id'][:24]
        reserved_at=time.monotonic();ledger.reserve(amend,240,metadata={'scope':'transport_only_archival_and_backup','training_performed':False})
        target=root/'transport_amendments'/'interrupted_inference'/proof['proof_id']
        no_optimizer(root);archived=archive_partials(root,target/'partials')
        report={'version':VERSION,'status':'completed','attempt_id':amend,
            'original_configuration_commit':PIN,'new_release_tag':NEW_RELEASE,
            'observer_proof':proof,'closed_inference_job_id':reservation['job_id'],
            'charged_inference_upper_bound_seconds':upper,'all_prior_charges_preserved':True,
            'paused_ledger_prefix_sha256':sha(raw),'archived_inference_partials':archived,
            'optimizer_fit_retry_authorized':False,'same_scientific_sources_and_run_ids':True}
        immutable(target/'amendment.json',report)
        receipt={'status':'completed','attempt_id':amend,'artifacts_sha256':{
            str(p.relative_to(target)):file_sha(p) for p in target.rglob('*') if p.is_file()}}
        remote.persist_result(target,receipt)
        ledger.charge(amend,time.monotonic()-reserved_at,status='completed',evidence={'amendment_sha256':file_sha(target/'amendment.json')})
        remote.snapshot('controlled_inference_transport_amendment_closed')
        return {'status':'inference_only_transport_amendment_closed','proof':str(target/'amendment.json'),
                'budget':ledger.report(),'resume_original_configuration_commit':PIN}
    finally: ledger.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--pid',type=int);p.add_argument('--timeout',type=float,default=1200)
    a=p.parse_args();pause(a.destination,pid=a.pid,timeout=a.timeout)
if __name__=='__main__': main()
