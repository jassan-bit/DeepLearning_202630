"""One-cell continuation after the charged Git checkout setup failure."""
from pathlib import Path
import hashlib,json,os,re,runpy,time
from urllib.request import Request,urlopen

PIN='ae1a2d4f96807a20ea92c313297a5c09512a4817'
WORKSPACE=Path('/content/entregable2_gpu_workspace')
REGISTRATION='experiments/e2_posthoc_environment_20261113/registration.json'
REQUIRED={'scripts/e2_posthoc_transport_resume.py','scripts/e2_arima_prediction_compat.py',
          'scripts/e2_posthoc_environment_adapter.py','scripts/e2_posthoc_failed_setup_archive.py',
          'scripts/e2_posthoc_clean_worker_bootstrap.py','scripts/e2_posthoc_environment_resume.py',
          'notebooks/resume_e2_clean_setup_colab.py'}

def get(commit,path):
    with urlopen(Request('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+commit+'/'+path,
                         headers={'Cache-Control':'no-cache'}),timeout=40) as r:raw=r.read(4*1024**2+1)
    if len(raw)>4*1024**2:raise ValueError('Recovery metadata is unexpectedly large')
    return raw

def once(path,raw):
    if any(x.is_symlink() for x in (path,*path.parents)):raise ValueError('Regular recovery paths required')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing recovery bytes differ; preserve for review')
    if not path.exists():path.write_bytes(raw)

def resume_e2_clean_setup_colab(commit,*,namespace=None,execute=False):
    if not isinstance(commit,str) or re.fullmatch('[0-9a-f]{40}',commit) is None:raise ValueError('Use immutable recovery commit')
    if type(execute)is not bool:raise TypeError('execute must be boolean')
    if not execute:return {'status':'dry_run','reuse_completed_79_diagnostics':True,'retry_scientific_fit':False,'authorized_total_hours':80}
    started=time.monotonic();raw=get(commit,REGISTRATION);record=json.loads(raw)
    if record.get('original_configuration_commit')!=PIN or record.get('new_budget_authorization') is not False:
        raise ValueError('Recovery must preserve original configuration and authorization')
    if set(record.get('files_sha256',{}))!=REQUIRED:raise ValueError('Unexpected recovery file inventory')
    root=WORKSPACE/('environment-recovery-'+commit)
    for path,expected in record['files_sha256'].items():
        payload=get(commit,path)
        if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Recovery file SHA mismatch: '+path)
        once(root/path,payload)
    once(root/REGISTRATION,raw)
    wrapper=root/'python';once(wrapper,(root/'scripts/e2_posthoc_environment_resume.py').read_bytes());wrapper.chmod(0o700)
    original=WORKSPACE/('guide-closure-extension-'+PIN)/'notebooks/run_e2_posthoc_fit_reproduction_colab.py'
    if hashlib.sha256(original.read_bytes()).hexdigest()!='5407e40fcebe463bae25b78f4993db5ef13d567fc1fe31b0571c261a2c02fef8':
        raise ValueError('Original launcher changed')
    ns=dict(globals() if namespace is None else namespace);ns['GPU_PYTHON']=str(wrapper)
    key='E2_ENV_RECOVERY_HOST_SECONDS';previous=os.environ.get(key)
    os.environ[key]=str(time.monotonic()-started)
    try:
        return runpy.run_path(str(original))['run_e2_posthoc_fit_reproduction_colab'](PIN,namespace=ns,execute=True)
    finally:
        if previous is None:os.environ.pop(key,None)
        else:os.environ[key]=previous
