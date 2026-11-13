"""Restore an existing E2 campaign after VM loss; never initialize its budget."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,hashlib,importlib.util,io,json,os,subprocess,sys,threading,time

WORKSPACE=Path('/content/entregable2_gpu_workspace')
PIN='ae1a2d4f96807a20ea92c313297a5c09512a4817'
ENV_PIN='70316a2aa146a7f647ee36c88080022ec86f8005'
ENGINE='1e8745497b484518f766b6da5c19be8362d9add7'
REGISTRATION='experiments/e2_posthoc_fresh_session_20261113/registration.json'
FINAL_PREFIXES=('final/final-fit/','final/classical-final/','final/score-worker-source-recovery-v1/','final/scoring_source_recovery/')

def install_shutdown_handler():
    """Let the unchanged worker manager reap its separate process groups."""
    import signal
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Fresh-session controller interrupted; preserve unresolved reservations')
    return signal.signal(signal.SIGTERM,interrupted)

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def validate_bundle(root):
    root=Path(root);record=json.loads((root/REGISTRATION).read_bytes())
    for relative,expected in record['files_sha256'].items():
        p=root/relative
        if any(x.is_symlink() for x in (p,*p.parents)) or not p.resolve().is_relative_to(root.resolve()) or sha(p.read_bytes())!=expected:
            raise ValueError('Fresh-session recovery source changed: '+relative)
    return record
def materialize_old(preparation,workspace):
    notebook=preparation.module_for('posthoc');extension=notebook['materialize'](PIN,workspace)
    environment=workspace/('environment-recovery-'+ENV_PIN)
    path='experiments/e2_posthoc_environment_20261113/registration.json'
    from urllib.request import urlopen
    def fetch(relative):
        with urlopen('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+ENV_PIN+'/'+relative,timeout=40) as r:return r.read(4*1024**2+1)
    raw=fetch(path);registration=json.loads(raw)
    if registration['original_configuration_commit']!=PIN:raise ValueError('Keep original scientific configuration')
    for relative,expected in registration['files_sha256'].items():
        payload=fetch(relative)
        if sha(payload)!=expected:raise ValueError('Original environment adapter changed')
        preparation.once(environment/relative,payload)
    preparation.once(environment/path,raw)
    return extension,environment
def original_paths(state):
    return {p for p in state['files'] if p.startswith('shared_budget/') or p in {
        'final/final_freeze.json','final/analysis_registration.json','final/pilot_power_MDE.json','final/github_backup_receipt.json'}
        or any(p==prefix.rstrip('/') or p.startswith(prefix) for prefix in FINAL_PREFIXES)}
def original_backups(state):
    return {p:v for p,v in state['result_backups'].items() if p=='final' or any(p==prefix.rstrip('/') or p.startswith(prefix) for prefix in FINAL_PREFIXES)}
def attach(remote,state):
    # All controlled metadata have been verified before this is used to reserve
    # restoration work. Fitting remains unavailable until physical checks close.
    files,_=remote.state._scan()
    if files!=state['files']:raise ValueError('Restored posthoc metadata do not match the authoritative state')
    obj=remote.state;obj._state=state;obj._files=state['files'];obj._archives=state['metadata_deltas']
    obj._backups=state.get('result_backups',{});obj._restored=True
def install_runtime_metadata(old,full,receipt_sha,acceptance_sha):
    original_verify=old.verify_clean_runtime
    additions={k:v for k,v in full.items() if k.startswith('synthetic_acceptance_')}
    if additions.get('synthetic_acceptance_sha256')!=acceptance_sha:raise ValueError('Original synthetic acceptance binding differs')
    def verified(value):
        original_verify(value)
        base={k:v for k,v in value.items() if not k.startswith('synthetic_acceptance_')}
        if sha(canonical(base)+b'\n')!=receipt_sha:raise ValueError('Historical clean runtime metadata changed')
        if any(k in value and value[k]!=v for k,v in additions.items()):raise ValueError('Synthetic acceptance metadata differs')
        # Keep the immutable historical receipt on disk. Enrich only the dict
        # returned to the unchanged worker-spec producer, matching prior fits.
        value.update(additions)
        return True
    old.verify_clean_runtime=verified
    return original_verify
def restore_artifacts(root,client,backups,head,restore,deadline):
    def one(item):
        if time.monotonic()>=deadline:raise TimeoutError('Fresh-session restoration allocation reached')
        path,receipt=item;return restore.restore_results(root,client,{path:receipt},head)
    totals={'manifests':0,'files_restored':0,'existing_identical_files':0}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,item) for item in backups.items()]
        for f in as_completed(futures):
            value=f.result()
            for key in totals:totals[key]+=value.get(key,0)
            if totals['manifests']%10==0 or totals['manifests']==len(backups):
                print(json.dumps({'event':'fresh_artifact_restore','verified_manifests':totals['manifests'],'planned':len(backups)}),flush=True)
    return totals
def preparation_run(command,*,timeout,cwd=None,log=None,env=None):
    """Preparation only: descendants stay in the globally supervised group."""
    begin=time.monotonic()
    stream=open(log,'ab') if log else subprocess.DEVNULL
    try:
        subprocess.run([str(x) for x in command],cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                       stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=max(.1,timeout),
                       start_new_session=False)
    finally:
        if log:stream.close()
    return time.monotonic()-begin

def bounded_preparation(command,*,deadline,cwd,log,env,token):
    """Hard wall limit for downloads, threads, installers and all descendants."""
    if deadline<=time.monotonic():raise TimeoutError('Restoration deadline reached before launch')
    with Path(log).open('ab') as stream:
        child=subprocess.Popen([str(x) for x in command],cwd=cwd,env=env,stdin=subprocess.PIPE,
            stdout=stream,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        try:
            child.stdin.write(token+'\n');child.stdin.close()
            code=child.wait(timeout=max(.1,deadline-time.monotonic()))
            if code:raise RuntimeError('Fresh-session preparation failed; see its preserved log')
        except BaseException:
            # The preparation-only runner never starts nested process groups.
            # Kill the whole group even if its leader exits first.
            for sig in (15,9):
                try:os.killpg(child.pid,sig)
                except ProcessLookupError:pass
                if sig==15:
                    try:child.wait(timeout=2)
                    except subprocess.TimeoutExpired:pass
            child.wait(timeout=2)
            raise
        else:
            # A successful preparation must not leave detached work running.
            try:os.killpg(child.pid,9)
            except ProcessLookupError:pass

def execute_prepare(bundle,context_path,token):
    validate_bundle(bundle);context=json.loads(Path(context_path).read_bytes())
    deadline=context['deadline'];workspace=WORKSPACE
    extension=workspace/('guide-closure-extension-'+PIN);environment=workspace/('environment-recovery-'+ENV_PIN)
    sys.path[:0]=[str(extension/'scripts'),str(bundle/'scripts')]
    old=load(extension/'scripts/e2_posthoc_fit_reproduction.py','e2_fresh_restore_only')
    # This process can restore and probe only. The unchanged scientific
    # continuation runs later in a different process with its original runner.
    old.run_bounded=preparation_run
    import e2_reduced_github_backup as backup
    import e2_reduced_github_restore as restore
    client=backup.GitHubBackup(token,timeout=60,max_conflicts=3);client._checked=True
    original=workspace/'github-primary-output-v1';output=workspace/'e2-guide-closure-posthoc-v1'
    preparation=load(bundle/'notebooks/prepare_e2_posthoc_fresh_session.py','e2_fresh_prepare_child')
    metadata=load(bundle/'scripts/e2_posthoc_fresh_metadata.py','e2_fresh_metadata_child')
    def progress(stage):
        Path(context['progress']).write_text(stage)
        print(json.dumps({'event':'fresh_session_preparation_stage','stage':stage}),flush=True)
    progress('restore_original_metadata')
    metadata.restore_metadata(original,client,context['original_state'],context['head'],
        paths=original_paths(context['original_state']),cache_root=Path(context['cache']),parallel=6,deadline=deadline)
    progress('restore_frozen_weights_and_completed_results')
    original_restored=restore_artifacts(original,client,original_backups(context['original_state']),context['head'],restore,deadline)
    posthoc_restored=restore_artifacts(output,client,context['posthoc_state']['result_backups'],context['head'],restore,deadline)
    progress('prepare_A100_sources_and_runtime')
    prepared=preparation.prepare_fresh_posthoc_session({},execute=True,deadline=deadline)
    python=Path(prepared['GPU_PYTHON']);source=Path(prepared['engine_root'])
    progress('reconstruct_hash_identical_known_market_cache')
    preparation_run([python,bundle/'scripts/e2_posthoc_fresh_market_cache.py','--source-root',source,'--original-output',original],
        timeout=deadline-time.monotonic(),cwd=bundle,env=old.clean_environment(),log=output/'fresh_market_restore.log')
    progress('reconstruct_original_clean_environment')
    clean=load(bundle/'scripts/e2_posthoc_fresh_clean_runtime.py','e2_fresh_clean_child')
    clean.restore_clean_runtime(old,original,output,environment,python,deadline)
    completed={str(p.parent.relative_to(output)):sha(p.read_bytes()) for p in (output/'reproduction/jobs').glob('*/result.json')}
    proof={'version':'e2-fresh-session-admission-v1','status':'completed',
        'original_control_state_id':context['original_state']['state_id'],
        'restored_posthoc_state_id':context['posthoc_state']['state_id'],'completed_fit_receipts_sha256':completed,
        'original_artifacts':original_restored,'posthoc_artifacts':posthoc_restored,'budget_reset':False,
        'training_performed':False,'original_configuration_commit':PIN,
        'fresh_source_registration_sha256':sha((bundle/REGISTRATION).read_bytes()),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'clean_runtime_sha256':sha((output/'reproduction/clean/clean_runtime.json').read_bytes())}
    folder=output/'fresh_session_admissions'/proof['boot_id'];old.immutable_json(folder/'admission.json',proof)
    additions={
        'fresh_runtime_admission.json':output/'reproduction/clean/fresh_runtime_admissions'/proof['boot_id']/'admission.json',
        'synthetic_acceptance.json':output/'reproduction/clean/synthetic_acceptance.json',
        'historical_clean_runtime.json':output/'reproduction/clean/clean_runtime.json',
        'environment.lock.txt':output/'reproduction/clean/environment.lock.txt'}
    for name,path in additions.items():preparation.once(folder/name,path.read_bytes())
    receipt={'status':'completed','attempt_id':'fresh-session-'+proof['boot_id'],'scientific_result':False,
             'artifacts_sha256':{name:sha((folder/name).read_bytes()) for name in ['admission.json',*additions]}}
    Path(context['result']).write_bytes(canonical({'folder':str(folder),'receipt':receipt,'python':str(python),'completed':len(completed)})+b'\n')
    progress('restoration_finished_no_fit_started')

def execute_continue(bundle,expected_state,token):
    validate_bundle(bundle);workspace=WORKSPACE;extension=workspace/('guide-closure-extension-'+PIN)
    env_root=workspace/('environment-recovery-'+ENV_PIN)
    sys.path[:0]=[str(extension/'scripts'),str(bundle/'scripts')]
    old=load(extension/'scripts/e2_posthoc_fit_reproduction.py','e2_unchanged_fresh_controller')
    import e2_reduced_github_backup as backup
    backup.RELEASE_TAG='e2-guide-closure-posthoc-20261113-v1'
    adapter=load(env_root/'scripts/e2_posthoc_environment_adapter.py','e2_original_environment_adapter')
    source=workspace/('reduced-source-'+ENGINE);original=workspace/'github-primary-output-v1';output=workspace/'e2-guide-closure-posthoc-v1'
    adapter.install_environment_adapter(old,source,env_root,original_output=original)
    clean=load(bundle/'scripts/e2_posthoc_fresh_clean_runtime.py','e2_fresh_clean_metadata')
    full,_,_=clean.select_runtime(old,output)
    if sha(Path(full['synthetic_acceptance_path']).read_bytes())!=clean.ACCEPTANCE_SHA:
        raise ValueError('Restored synthetic acceptance file differs from completed workers')
    original_verify=install_runtime_metadata(old,full,clean.RUNTIME_SHA,clean.ACCEPTANCE_SHA)
    original_restore=old.PosthocRemoteState.restore
    def ready_restore(self):
        head=self.state.client._branch_head();state=self.state._current(head)
        if state is None or state['state_id']!=expected_state:raise RuntimeError('Another session advanced the restored campaign')
        attach(self,state)
        for directory in (output/'reproduction/jobs').glob('*'):
            if directory.is_dir() and old.verified_completed(directory,controller_source_sha256=sha(Path(old.__file__).read_bytes()),configuration_commit=PIN) is None:
                raise RuntimeError('An admitted fit lacks a completed result; preserve for reconciliation')
        old.validate_original_anchor(original)
        runtime=json.loads((output/'reproduction/clean/clean_runtime.json').read_bytes());old.verify_clean_runtime(runtime)
        return {'status':'verified_fresh_session_ready','state_id':expected_state,'budget_reset':False}
    old.PosthocRemoteState.restore=ready_restore
    try:return old.orchestrate(source,original,output,workspace/'reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python',extension,PIN,token)
    finally:
        old.PosthocRemoteState.restore=original_restore
        old.verify_clean_runtime=original_verify

def run_fresh_session(bundle,namespace,*,started=None,execute=False):
    if not execute:return {'status':'dry_run','budget_reset':False,'authorized_total_hours':80,'persistent_destination':'GitHub'}
    started=time.monotonic() if started is None else started;bundle=Path(bundle);record=validate_bundle(bundle)
    preparation=load(bundle/'notebooks/prepare_e2_posthoc_fresh_session.py','e2_fresh_prerequisites')
    if preparation.active_pids():raise RuntimeError('An existing E2 controller is active; no second controller is started')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if 'A100' not in gpu:raise RuntimeError('Selecciona GPU A100 en Colab antes de restaurar; GPU observada: '+gpu)
    token=str(namespace.get('GITHUB_TOKEN') or os.environ.get('GITHUB_TOKEN','')).strip()
    if not token:
        from google.colab import userdata
        token=(userdata.get('GITHUB_TOKEN') or '').strip()
    if not token or any(x.isspace() for x in token):raise ValueError('Use the existing GITHUB_TOKEN secret')
    WORKSPACE.mkdir(parents=True,exist_ok=True);extension,environment=materialize_old(preparation,WORKSPACE)
    sys.path[:0]=[str(extension/'scripts'),str(bundle/'scripts')]
    old=load(extension/'scripts/e2_posthoc_fit_reproduction.py','e2_fresh_accounting_only')
    import e2_reduced_github_backup as backup
    import e2_reduced_github_state as primary
    import e2_reduced_github_restore as restore
    backup.RELEASE_TAG='e2-guide-closure-posthoc-20261113-v1'
    client=backup.GitHubBackup(token,timeout=60,max_conflicts=3);client._checked=True
    original=WORKSPACE/'github-primary-output-v1';output=WORKSPACE/'e2-guide-closure-posthoc-v1'
    cache=WORKSPACE/'fresh-session-control-cache'
    metadata=load(bundle/'scripts/e2_posthoc_fresh_metadata.py','e2_fresh_metadata')
    head=client._branch_head();original_state=primary.GitHubState(original,client)._current(head)
    remote=old.PosthocRemoteState(output,client,extension);posthoc_state=remote.state._current(head)
    if original_state is None or posthoc_state is None:raise RuntimeError('Existing GitHub campaign controls are required; no budget is initialized')
    print('Restaurando los diarios y recibos existentes desde GitHub.',flush=True)
    required={'shared_budget/pilot_budget.jsonl','shared_budget/budget_continuation_80.json','shared_budget/budget_continuation_80_approval.json'}
    metadata.restore_metadata(original,client,original_state,head,paths=required,cache_root=cache,parallel=6,deadline=time.monotonic()+600)
    metadata.restore_metadata(output,client,posthoc_state,head,cache_root=cache,parallel=6,deadline=time.monotonic()+600)
    anchor=old.validate_original_anchor(original)
    # Constructing this existing ledger blocks all unknown/unclosed reservations.
    with old.PosthocLedger(output/'budget',anchor) as budget:
        attach(remote,posthoc_state);budget.set_persistence(remote)
        budget.charge_overhead(time.monotonic()-started,description='fresh_session_source_and_existing_control_restore')
        if budget.remaining_seconds<1800:raise RuntimeError('Existing balance cannot admit the bounded fresh-session preparation')
        job='fresh-session-restore-'+str(time.time_ns());begin=time.monotonic();deadline=begin+1790
        budget.reserve(job,1800,metadata={'scope':'restore_existing_artifacts_and_same_environments','training_performed':False})
        progress=cache/(job+'.progress');result_path=cache/(job+'.result.json');context_path=cache/(job+'.context.json')
        context={'head':head,'original_state':original_state,'posthoc_state':posthoc_state,
                 'cache':str(cache),'deadline':deadline,'progress':str(progress),'result':str(result_path)}
        context_path.write_bytes(canonical(context)+b'\n')
        stop=threading.Event()
        def heartbeat():
            while not stop.wait(30):
                stage=progress.read_text() if progress.exists() else 'starting_isolated_restoration'
                print(json.dumps({'event':'fresh_session_restoring','stage':stage,'elapsed_seconds':round(time.monotonic()-begin)}),flush=True)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        log=output/'fresh_session_restore.log'
        command=[sys.executable,bundle/'scripts/e2_posthoc_fresh_resume.py','--prepare-restoration','--bundle',bundle,'--context',context_path]
        try:
            bounded_preparation(command,deadline=deadline,cwd=bundle,log=log,env=old.clean_environment(),token=token)
            result=json.loads(result_path.read_bytes());python=Path(result['python']);completed=result['completed']
        except BaseException as error:
            if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-35:]).replace(token,'[REDACTED]'),flush=True)
            closing=time.monotonic()
            interrupted=isinstance(error,(KeyboardInterrupt,SystemExit))
            budget.charge(job,closing-begin,status='interrupted' if interrupted else 'failed',
                          evidence={'restored_files_preserved':True,'scientific_fit_repeated':False,'preparation_process_group_closed':True})
            if not interrupted:budget.charge_overhead(time.monotonic()-closing,description='fresh_failed_restoration_cost_publication')
            raise
        else:
            closing=time.monotonic()
            budget.charge(job,closing-begin,status='completed',evidence={'restored_completed_fits':completed,'preparation_process_group_closed':True})
        finally:stop.set();thread.join(timeout=1)
        # Publication is measured separately, after the bounded restoration
        # group has exited. It consumes the same existing remaining allowance.
        old.backup_completed(Path(result['folder']),result['receipt'],remote)
        budget.charge_overhead(time.monotonic()-closing,description='fresh_restoration_cost_and_admission_publication')
        snapshot=remote.snapshot('fresh_session_ready_existing_results_preserved');expected=snapshot['state_id']
        print(json.dumps({'event':'fresh_session_ready','completed_new_fits_preserved':completed,
                          'remaining_seconds':budget.remaining_seconds,'authorized_total_hours':80,'budget_reset':False}),flush=True)
    command=[python,bundle/'scripts/e2_posthoc_fresh_resume.py','--continue-controller','--bundle',bundle,'--expected-state',expected]
    env=old.clean_environment();child=subprocess.Popen([str(x) for x in command],cwd=bundle,env=env,
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    child.stdin.write(token+'\n');child.stdin.close()
    try:
        for line in child.stdout:print(line.replace(token,'[REDACTED]'),end='',flush=True)
        code=child.wait()
    except BaseException:
        os.killpg(child.pid,15)
        try:child.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait(timeout=3)
        raise
    if code:raise RuntimeError('Continuation stopped; preserve its receipts and last output')
    return {'status':'controller_closed','exit_code':code,'budget_reset':False}

if __name__=='__main__':
    install_shutdown_handler()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--continue-controller',action='store_true')
    p.add_argument('--prepare-restoration',action='store_true')
    p.add_argument('--context',type=Path)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--expected-state')
    a=p.parse_args()
    if a.continue_controller==a.prepare_restoration:p.error('Choose one verified execution stage')
    if a.prepare_restoration and a.context is None:p.error('Restoration context is required')
    token=sys.stdin.readline().strip()
    if not token or any(c.isspace() for c in token):raise ValueError('Existing credential must arrive privately through STDIN')
    if a.prepare_restoration:execute_prepare(a.bundle,a.context,token)
    else:execute_continue(a.bundle,a.expected_state,token)
