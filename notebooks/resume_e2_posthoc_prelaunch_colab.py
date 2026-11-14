"""Reconcile the verified, unpublished prelaunch reservation and continue once."""
from pathlib import Path
import base64,hashlib,importlib.util,io,json,os,re,subprocess,sys,time,zipfile
from urllib.request import urlopen
WORKSPACE=Path('/content/entregable2_gpu_workspace')
REG='experiments/e2_prelaunch_recovery_20261113/registration.json'
REQUIRED={'notebooks/resume_e2_posthoc_prelaunch_colab.py','scripts/e2_posthoc_prelaunch_continue.py',
          'scripts/e2_posthoc_control_consistency.py','scripts/e2_posthoc_prelaunch_reconcile.py'}
PIN='ae1a2d4f96807a20ea92c313297a5c09512a4817'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def fetch(commit,path):
    with urlopen('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+commit+'/'+path,timeout=30) as r:raw=r.read(2*1024**2+1)
    if len(raw)>2*1024**2:raise ValueError('Recovery source size limit')
    return raw

def materialize(commit):
    raw=fetch(commit,REG);record=json.loads(raw)
    if set(record['files_sha256'])!=REQUIRED or record['authorized_total_hours']!=80 or record['new_budget_credit']is not False:
        raise ValueError('Unexpected reconciliation scope')
    root=WORKSPACE/('prelaunch-recovery-'+commit)
    for relative,expected in record['files_sha256'].items():
        p=root/relative;data=fetch(commit,relative)
        if sha(data)!=expected:raise ValueError('Reconciliation source checksum differs')
        if any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('Recovery paths must be regular')
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.read_bytes()!=data:raise ValueError('Existing reconciliation source changed')
        if not p.exists():p.write_bytes(data)
    p=root/REG;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and p.read_bytes()!=raw:raise ValueError('Existing registration changed')
    if not p.exists():p.write_bytes(raw)
    return root

def read_remote_ledger(client,state,head):
    relative='budget/posthoc_budget.jsonl';expected=state['files'][relative]
    for item in reversed(state['metadata_deltas']):
        response=client._content(item['path'],head)
        if response is None:raise ValueError('Remote control delta missing')
        encoded=response.get('content')
        if not encoded or response.get('encoding')=='none':encoded=client._request('GET','/git/blobs/'+response['sha'])['content']
        raw=base64.b64decode(encoded)
        if len(raw)!=item['bytes'] or sha(raw)!=item['sha256']:raise ValueError('Remote control delta changed')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if relative not in archive.namelist():continue
            info=archive.getinfo(relative)
            if info.file_size!=expected['bytes']:continue
            payload=archive.read(relative)
            if sha(payload)==expected['sha256']:return payload
    raise ValueError('Exact authoritative closed journal not found')

def active_pids(preparation):
    found=set(preparation.active_pids())
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        if int(path.parent.name)==os.getpid():continue
        try:words=path.read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(w).name in {'e2_posthoc_fresh_resume.py','e2_posthoc_prelaunch_continue.py'} for w in words if w):found.add(int(path.parent.name))
    return sorted(found)

def resume_e2_posthoc_prelaunch_colab(commit,*,namespace=None,execute=False):
    if not isinstance(commit,str) or re.fullmatch('[0-9a-f]{40}',commit)is None:raise ValueError('Pinned reconciliation commit required')
    if type(execute)is not bool:raise TypeError('execute must be boolean')
    if not execute:return {'status':'dry_run','authorized_total_hours':80,'budget_reset':False}
    begin=time.monotonic();namespace=globals() if namespace is None else namespace;root=materialize(commit)
    continuation=load(root/'scripts/e2_posthoc_prelaunch_continue.py','e2_prelaunch_continuation_sources')
    record,base=continuation.validate(root)
    fresh=load(base/'scripts/e2_posthoc_fresh_resume.py','e2_prelaunch_original_launcher')
    fresh.validate_bundle(base)
    preparation=load(base/'notebooks/prepare_e2_posthoc_fresh_session.py','e2_prelaunch_idle_check')
    if active_pids(preparation):raise RuntimeError('An E2 controller is active; no duplicate launch')
    extension=WORKSPACE/('guide-closure-extension-'+PIN);sys.path.insert(0,str(extension/'scripts'))
    old=load(extension/'scripts/e2_posthoc_fit_reproduction.py','e2_prelaunch_original_accounting')
    import e2_reduced_github_backup as backup
    backup.RELEASE_TAG='e2-guide-closure-posthoc-20261113-v1'
    consistency=load(root/'scripts/e2_posthoc_control_consistency.py','e2_prelaunch_control_consistency')
    consistency.install(backup,old)
    token=str(namespace.get('GITHUB_TOKEN')or os.environ.get('GITHUB_TOKEN','')).strip()
    if not token:
        from google.colab import userdata
        token=(userdata.get('GITHUB_TOKEN')or'').strip()
    if not token or any(x.isspace()for x in token):raise ValueError('Existing GITHUB_TOKEN secret required')
    output=WORKSPACE/'e2-guide-closure-posthoc-v1';original=WORKSPACE/'github-primary-output-v1'
    client=backup.GitHubBackup(token,timeout=20,max_conflicts=2);client._checked=True
    remote=old.PosthocRemoteState(output,client,extension)
    head=client._branch_head();state=remote.state._current(head)
    if state is None or state['state_id']!=record['expected_remote_state_id']:
        raise RuntimeError('Remote campaign changed; preserve local evidence for reconciliation')
    remote_ledger=read_remote_ledger(client,state,head)
    reconcile=load(root/'scripts/e2_posthoc_prelaunch_reconcile.py','e2_prelaunch_reconciliation')
    def check_remote():return remote.state._current(client._branch_head())
    print('Verificando la reserva local no publicada; se conservan los 72 resultados.',flush=True)
    recovered=reconcile.reconcile_local_only(old,output,original,WORKSPACE/'posthoc-prelaunch-evidence',remote_ledger,state,
        lambda folder,receipt:old.backup_completed(folder,receipt,client),check_remote,lambda:not active_pids(preparation))
    fresh.attach(remote,state)
    anchor=old.validate_original_anchor(original)
    with old.PosthocLedger(output/'budget',anchor)as budget:
        budget.set_persistence(remote)
        budget._append('overhead',elapsed_seconds=time.monotonic()-begin+300.,
            description='verified_unpublished_prelaunch_reconciliation_and_300s_conservative_failed_transport_charge',
            evidence={'archive':recovered,'scientific_fit_repeated':False,'new_budget_credit':False,
                      'failed_transport_charge_seconds':300,'failed_transport_charge_is_conservative_not_measured':True})
        expected=remote.snapshot('prelaunch_reconciled_first_worker_launch_permitted')['state_id']
        print(json.dumps({'event':'prelaunch_reconciliation_completed','completed_fits_preserved':72,
            'completed_ablations':18,'remaining_seconds':budget.remaining_seconds,'authorized_total_hours':80,
            'scientific_fit_repeated':False,'persistent_destination':'GitHub'}),flush=True)
    python=WORKSPACE/'reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python'
    child=subprocess.Popen([str(python),str(root/'scripts/e2_posthoc_prelaunch_continue.py'),'--root',str(root),'--expected-state',expected],
        cwd=root,env=old.clean_environment(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,bufsize=1,start_new_session=True)
    child.stdin.write(token+'\n');child.stdin.close()
    try:
        for line in child.stdout:print(line.replace(token,'[REDACTED]'),end='',flush=True)
        code=child.wait()
    except BaseException:
        try:os.killpg(child.pid,15)
        except ProcessLookupError:pass
        try:child.wait(timeout=8)
        except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait(timeout=3)
        raise
    if code:raise RuntimeError('Controller stopped; keep its receipts and last output')
    return {'status':'controller_closed','exit_code':code,'budget_reset':False}

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
