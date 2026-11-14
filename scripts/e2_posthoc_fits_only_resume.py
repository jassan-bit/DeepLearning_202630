"""Resume only frozen-recipe fits after verified diagnostics and reconciliation."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re,subprocess,sys,time
WORKSPACE=Path('/content/entregable2_gpu_workspace')
REG='experiments/e2_fits_only_resume_20261113/registration.json'
BASE='b2199a8bc5df0e392430343b7891b11bedcd5306'
PIN='ae1a2d4f96807a20ea92c313297a5c09512a4817'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def validate(root):
    root=Path(root);record=json.loads((root/REG).read_bytes());base=WORKSPACE/('fresh-session-recovery-'+BASE)
    for directory,entries in ((root,record['files_sha256']),(base,record['existing_launcher_files_sha256'])):
        for relative,expected in entries.items():
            p=directory/relative
            if any(x.is_symlink()for x in(p,*p.parents))or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                raise ValueError('Verified continuation source changed: '+relative)
    return record,base

def components(root):
    record,base=validate(root)
    fresh=load(base/'scripts/e2_posthoc_fresh_resume.py','e2_fits_only_preserved_launcher')
    preparation=load(base/'notebooks/prepare_e2_posthoc_fresh_session.py','e2_fits_only_idle_guard')
    extension=WORKSPACE/('guide-closure-extension-'+PIN);sys.path.insert(0,str(extension/'scripts'))
    old=load(extension/'scripts/e2_posthoc_fit_reproduction.py','e2_fits_only_original_accounting')
    if hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()!='5d8ce059f96f86a35b7e2d2530e3fb776827240fcb29df920b4d05374f139fe9':
        raise ValueError('Original controller source changed')
    import e2_reduced_github_backup as backup
    backup.RELEASE_TAG='e2-guide-closure-posthoc-20261113-v1'
    consistency=load(Path(root)/'scripts/e2_posthoc_control_consistency.py','e2_fits_only_control_consistency')
    consistency.install(backup,old)
    checker=load(Path(root)/'scripts/e2_posthoc_completed_diagnostics.py','e2_fits_only_completed_diagnostics')
    return base,fresh,preparation,old,backup,consistency,checker,extension

def idle(preparation):
    found=set(preparation.active_pids())
    names={'e2_posthoc_fresh_resume.py','e2_posthoc_prelaunch_continue.py','e2_posthoc_fits_only_resume.py'}
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name)==os.getpid():continue
        try:words=p.read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(Path(w).name in names for w in words if w):found.add(int(p.parent.name))
    return not found

def install_fits_only(old,checker,output,extension,state,launcher_elapsed):
    original=old.orchestrate
    def only_fits(*args,**kwargs):
        proof=checker.verify_completed(output,extension,state)
        print(json.dumps({'event':'completed_diagnostics_verified_reused','diagnostic_predictors':79,
                          'new_diagnostics_reservation':False}),flush=True)
        kwargs.update(run_inference=False,run_training=True,launcher_elapsed_seconds=launcher_elapsed())
        return original(*args,**kwargs)
    old.orchestrate=only_fits
    return original

def continue_child(root,expected,token,launcher_elapsed):
    child_start=time.monotonic();base,fresh,prep,old,backup,consistency,checker,extension=components(root)
    fresh.install_shutdown_handler()
    client=backup.GitHubBackup(token,timeout=20,max_conflicts=2);client._checked=True
    output=WORKSPACE/'e2-guide-closure-posthoc-v1'
    state=old.PosthocRemoteState(output,client,extension).state._current(client._branch_head())
    if state is None or state['state_id']!=expected:raise RuntimeError('Another session advanced the control state')
    original_load=fresh.load
    def adapted(path,name):
        module=original_load(path,name)
        if Path(path).name=='e2_posthoc_fit_reproduction.py':
            consistency.install(backup,module)
            install_fits_only(module,checker,output,extension,state,lambda:launcher_elapsed+time.monotonic()-child_start)
        return module
    fresh.load=adapted
    try:return fresh.execute_continue(base,expected,token)
    finally:fresh.load=original_load

def run(root,namespace,*,started=None):
    started=time.monotonic()if started is None else started
    base,fresh,prep,old,backup,consistency,checker,extension=components(root)
    if not idle(prep):raise RuntimeError('An existing E2 controller is active; no second launch')
    token=str(namespace.get('GITHUB_TOKEN')or os.environ.get('GITHUB_TOKEN','')).strip()
    if not token:
        from google.colab import userdata
        token=(userdata.get('GITHUB_TOKEN')or'').strip()
    if not token or any(c.isspace()for c in token):raise ValueError('Existing GITHUB_TOKEN secret required')
    output=WORKSPACE/'e2-guide-closure-posthoc-v1';original=WORKSPACE/'github-primary-output-v1'
    client=backup.GitHubBackup(token,timeout=20,max_conflicts=2);client._checked=True
    remote=old.PosthocRemoteState(output,client,extension)
    state=remote.state._current(client._branch_head())
    if state is None:raise RuntimeError('Existing authoritative campaign is required')
    fresh.attach(remote,state)
    proof=checker.verify_completed(output,extension,state)
    with old.PosthocLedger(output/'budget',old.validate_original_anchor(original))as budget:
        print(json.dumps({'event':'fits_only_ready','diagnostic_predictors_verified':79,
            'completed_fits_preserved':sum(p.startswith('reproduction/jobs/')and p.endswith('/result.json')for p in state['files']),
            'remaining_seconds':budget.remaining_seconds,'authorized_total_hours':80,'new_budget_credit':False,
            'reconciliation_repeated':False,'prior_300s_charge_repeated':False}),flush=True)
    python=WORKSPACE/'reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python'
    command=[str(python),str(Path(root)/'scripts/e2_posthoc_fits_only_resume.py'),'--continue-controller','--root',str(root),
             '--expected-state',state['state_id'],'--launcher-elapsed-seconds',str(time.monotonic()-started)]
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
    return {'status':'controller_closed','exit_code':code,'budget_reset':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--continue-controller',action='store_true')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-state',required=True)
    p.add_argument('--launcher-elapsed-seconds',type=float,default=0.);a=p.parse_args()
    if not a.continue_controller:p.error('Use the verified same-session entrypoint')
    token=sys.stdin.readline().strip()
    if not token or any(c.isspace()for c in token):raise ValueError('Existing credential must arrive privately through STDIN')
    continue_child(a.root,a.expected_state,token,a.launcher_elapsed_seconds)
if __name__=='__main__':main()
