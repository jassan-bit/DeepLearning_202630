"""Materialize and run the verified restoration for a new Colab A100 session."""
from pathlib import Path
import hashlib,importlib.util,json,re,time
from urllib.request import Request,urlopen

WORKSPACE=Path('/content/entregable2_gpu_workspace')
REGISTRATION='experiments/e2_posthoc_fresh_session_20261113/registration.json'
REQUIRED={'notebooks/prepare_e2_posthoc_fresh_session.py','notebooks/resume_e2_posthoc_fresh_colab.py',
          'scripts/e2_posthoc_fresh_resume.py','scripts/e2_posthoc_fresh_metadata.py',
          'scripts/e2_posthoc_fresh_market_cache.py','scripts/e2_posthoc_fresh_clean_runtime.py'}
def fetch(commit,path):
    with urlopen(Request('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+commit+'/'+path,
                         headers={'Cache-Control':'no-cache'}),timeout=40) as r:raw=r.read(4*1024**2+1)
    if len(raw)>4*1024**2:raise ValueError('Recovery source exceeds its size limit')
    return raw
def once(path,raw):
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Regular recovery paths required')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing recovery file changed; preserve for review')
    if not path.exists():path.write_bytes(raw)
def resume_e2_posthoc_fresh_colab(commit,*,namespace=None,execute=False):
    if not isinstance(commit,str) or re.fullmatch('[0-9a-f]{40}',commit) is None:raise ValueError('Use the immutable recovery commit')
    if type(execute)is not bool:raise TypeError('execute must be boolean')
    if not execute:return {'status':'dry_run','authorized_total_hours':80,'budget_reset':False,'restore_completed_fits':True}
    started=time.monotonic();raw=fetch(commit,REGISTRATION);record=json.loads(raw)
    if (record.get('new_budget_authorization') is not False or record.get('authorized_total_hours')!=80
            or set(record.get('files_sha256',{}))!=REQUIRED):raise ValueError('Unexpected recovery source or authorization')
    root=WORKSPACE/('fresh-session-recovery-'+commit)
    for path,expected in record['files_sha256'].items():
        payload=fetch(commit,path)
        if hashlib.sha256(payload).hexdigest()!=expected:raise ValueError('Recovery source SHA mismatch: '+path)
        once(root/path,payload)
    once(root/REGISTRATION,raw)
    spec=importlib.util.spec_from_file_location('e2_verified_fresh_session',root/'scripts/e2_posthoc_fresh_resume.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.run_fresh_session(root,globals() if namespace is None else namespace,started=started,execute=True)
