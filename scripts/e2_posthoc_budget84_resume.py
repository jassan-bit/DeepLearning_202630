"""Register one authorized +4h amendment, then resume only pending fixed fits."""
from pathlib import Path
import argparse,base64,hashlib,importlib.util,json,os,subprocess,sys,time
WORKSPACE=Path('/content/entregable2_gpu_workspace')
REG='experiments/e2_budget84_extension_20261113/registration.json'
APPROVAL='experiments/e2_budget84_extension_20261113/authorization.json'
EXISTING='72ea0731d72a486970f8373638b36bd0c89174ae'
ATTEMPT='e2-budget84-approval-v1'

def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def validate(root):
    root=Path(root);record=json.loads((root/REG).read_bytes());existing=WORKSPACE/('fits-only-resume-'+EXISTING)
    for folder,entries in ((root,record['files_sha256']),(existing,record['existing_fits_only_files_sha256'])):
        for relative,expected in entries.items():
            p=folder/relative
            if any(x.is_symlink()for x in(p,*p.parents))or sha(p.read_bytes())!=expected:
                raise ValueError('Verified budget continuation source changed: '+relative)
    approval=(root/APPROVAL).read_bytes();value=json.loads(approval)
    if(value.get('user_statement')!='si amplialo'or value.get('previous_total_hours')!=80
       or value.get('assistant_selected_total_hours')!=84 or value.get('added_seconds')!=14400):
        raise ValueError('Only the authorized 80-to-84-hour continuation is allowed')
    return existing,approval

def authorization_record(backup,approval):
    receipt={'status':'completed','attempt_id':ATTEMPT,'scientific_result':False,
             'previous_total_hours':80,'total_hours':84,'artifacts_sha256':{'authorization.json':sha(approval)}}
    prefix='runs/'+ATTEMPT+'/'
    artifact={'path':'authorization.json','storage':'git_blob','remote_path':prefix+'artifacts/authorization.json',
              'sha256':sha(approval),'bytes':len(approval),'git_blob_sha':backup.git_blob_sha(approval)}
    manifest={'version':backup.VERSION,'attempt_id':ATTEMPT,'fit_status':'completed',
              'receipt_sha256':sha(backup.canonical(receipt)),'artifacts':[artifact],
              'retry_scientific_fit':False,'raw_market_data_included':False}
    authority={'amendment_id':'e2-posthoc-budget-84-v1','previous_total_hours':80,'total_hours':84,
               'added_seconds':14400,'user_statement':'si amplialo','persistent_destination':'GitHub',
               'receipt_manifest_path':prefix+'manifest.json','receipt_sha256':sha(backup.canonical(manifest)+b'\n'),
               'approval_sha256':sha(approval),'assistant_selected_total_hours':84}
    return receipt,manifest,authority

def content(client,path,head):
    value=client._content(path,head)
    if value is None:raise ValueError('Budget authorization is not present in GitHub')
    encoded=value.get('content')
    if not encoded or value.get('encoding')=='none':encoded=client._request('GET','/git/blobs/'+value['sha'])['content']
    return base64.b64decode(encoded)

def verify_authorization(client,manifest,authority,approval):
    head=client._branch_head();raw=content(client,authority['receipt_manifest_path'],head)
    if sha(raw)!=authority['receipt_sha256']or json.loads(raw)!=manifest:
        raise ValueError('Durable budget authorization manifest differs')
    row=manifest['artifacts'][0]
    if content(client,row['remote_path'],head)!=approval:raise ValueError('Durable user-authorization evidence differs')
    return True

def once(path,raw):
    path=Path(path)
    if any(x.is_symlink()for x in(path,*path.parents)):raise ValueError('Regular authorization paths required')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists()and path.read_bytes()!=raw:raise ValueError('Existing budget authorization changed; preserve it')
    if not path.exists():
        with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())

def funded_components(fits,adapter,authority):
    original=fits.components
    def components(root):
        values=original(root);old,consistency=values[3],values[5]
        adapter.install(old,authority)
        original_install=consistency.install
        def install(backup,module):
            result=original_install(backup,module);adapter.install(module,authority);return result
        consistency.install=install
        return values
    fits.components=components
    return original

def prepare(root):
    existing,approval=validate(root)
    fits=load(existing/'scripts/e2_posthoc_fits_only_resume.py','e2_budget84_verified_fits_launcher')
    values=fits.components(existing);backup=values[4]
    receipt,manifest,authority=authorization_record(backup,approval)
    adapter=load(Path(root)/'scripts/e2_posthoc_budget84.py','e2_budget84_accounting_adapter')
    return existing,approval,fits,values,receipt,manifest,authority,adapter

def run(root,namespace,*,started=None):
    started=time.monotonic()if started is None else started
    existing,approval,fits,values,receipt,manifest,authority,adapter=prepare(root)
    base,fresh,prep,old,backup,consistency,checker,extension=values
    if not fits.idle(prep):raise RuntimeError('An existing E2 controller is active; no second launch')
    token=str(namespace.get('GITHUB_TOKEN')or os.environ.get('GITHUB_TOKEN','')).strip()
    if not token:
        from google.colab import userdata
        token=(userdata.get('GITHUB_TOKEN')or'').strip()
    if not token or any(c.isspace()for c in token):raise ValueError('Existing GITHUB_TOKEN secret required')
    client=backup.GitHubBackup(token,timeout=20,max_conflicts=2);client._checked=True
    output=WORKSPACE/'e2-guide-closure-posthoc-v1';original=WORKSPACE/'github-primary-output-v1'
    remote=old.PosthocRemoteState(output,client,extension);state=remote.state._current(client._branch_head())
    if state is None:raise RuntimeError('Existing authoritative campaign required')
    fresh.attach(remote,state);checker.verify_completed(output,extension,state)
    folder=WORKSPACE/'e2-budget84-approval-v1';once(folder/'authorization.json',approval)
    print('Registrando en GitHub la ampliación autorizada: 84 horas totales (+4).',flush=True)
    backed=old.backup_completed(folder,receipt,client)
    if backed.get('manifest_path')!=authority['receipt_manifest_path']or backed.get('manifest_sha256')!=authority['receipt_sha256']:
        raise ValueError('Uploaded authorization receipt differs from its deterministic identity')
    verify_authorization(client,manifest,authority,approval)
    current=remote.state._current(client._branch_head())
    if current['state_id']!=state['state_id']:raise RuntimeError('Another session advanced the campaign during funding')
    # Receipt verification precedes the accounting policy installation. The
    # existing ledger constructor still rejects every unresolved reservation.
    adapter.install(old,authority)
    with old.PosthocLedger(output/'budget',old.validate_original_anchor(original))as budget:
        budget.set_persistence(remote)
        granted=adapter.apply_extension(budget,authority)
        charged_at=time.monotonic()
        budget.charge_overhead(charged_at-started,description='84_hour_authorization_registration_and_verified_launcher')
        expected=remote.snapshot('authorized_84_hour_continuation_ready')['state_id']
        print(json.dumps({'event':'budget84_extension_confirmed','status':granted['status'],'total_hours':84,
            'added_hours':4,'remaining_seconds':budget.remaining_seconds,
            'completed_fits_preserved':sum(p.startswith('reproduction/jobs/')and p.endswith('/result.json')for p in state['files']),
            'budget_reset':False,'diagnostics_repeated':False,'persistent_destination':'GitHub'}),flush=True)
    python=WORKSPACE/'reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python'
    command=[str(python),str(Path(root)/'scripts/e2_posthoc_budget84_resume.py'),'--continue-controller','--root',str(root),
             '--expected-state',expected,'--launcher-elapsed-seconds',str(time.monotonic()-charged_at)]
    child=subprocess.Popen(command,cwd=root,env=old.clean_environment(),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
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
    if code:raise RuntimeError('Controller stopped; retain its last output and receipts')
    return {'status':'controller_closed','exit_code':code,'approved_total_hours':84,'budget_reset':False}

def continue_child(root,expected,token,launcher_elapsed):
    child_start=time.monotonic()
    existing,approval,fits,values,receipt,manifest,authority,adapter=prepare(root)
    backup=values[4];client=backup.GitHubBackup(token,timeout=20,max_conflicts=2);client._checked=True
    verify_authorization(client,manifest,authority,approval)
    original=funded_components(fits,adapter,authority)
    try:return fits.continue_child(existing,expected,token,launcher_elapsed+time.monotonic()-child_start)
    finally:fits.components=original

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--continue-controller',action='store_true')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-state',required=True)
    p.add_argument('--launcher-elapsed-seconds',type=float,default=0.);a=p.parse_args()
    if not a.continue_controller:p.error('Use the verified same-session entrypoint')
    token=sys.stdin.readline().strip()
    if not token or any(c.isspace()for c in token):raise ValueError('Existing credential must arrive privately through STDIN')
    continue_child(a.root,a.expected_state,token,a.launcher_elapsed_seconds)
if __name__=='__main__':main()
