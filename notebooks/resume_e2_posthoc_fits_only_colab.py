"""Continue pending fits; reuse verified diagnostics and an already closed repair."""
from pathlib import Path
import hashlib,importlib.util,json,re,time
from urllib.request import urlopen
WORKSPACE=Path('/content/entregable2_gpu_workspace')
REG='experiments/e2_fits_only_resume_20261113/registration.json'
REQUIRED={'notebooks/resume_e2_posthoc_fits_only_colab.py','scripts/e2_posthoc_fits_only_resume.py',
          'scripts/e2_posthoc_control_consistency.py','scripts/e2_posthoc_completed_diagnostics.py'}

def fetch(commit,path):
    with urlopen('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+commit+'/'+path,timeout=30)as r:raw=r.read(2*1024**2+1)
    if len(raw)>2*1024**2:raise ValueError('Continuation source size limit')
    return raw

def resume_e2_posthoc_fits_only_colab(commit,*,namespace=None,execute=False):
    if not isinstance(commit,str)or re.fullmatch('[0-9a-f]{40}',commit)is None:raise ValueError('Pinned continuation commit required')
    if type(execute)is not bool:raise TypeError('execute must be boolean')
    if not execute:return {'status':'dry_run','authorized_total_hours':80,'budget_reset':False,'new_inference':False}
    started=time.monotonic();record_raw=fetch(commit,REG);record=json.loads(record_raw)
    if set(record['files_sha256'])!=REQUIRED or record['authorized_total_hours']!=80 or record['new_budget_credit']is not False:
        raise ValueError('Unexpected fits-only continuation scope')
    root=WORKSPACE/('fits-only-resume-'+commit)
    for path,expected in record['files_sha256'].items():
        raw=fetch(commit,path);p=root/path
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Continuation source hash differs')
        if any(x.is_symlink()for x in(p,*p.parents)):raise ValueError('Regular continuation paths required')
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists()and p.read_bytes()!=raw:raise ValueError('Existing continuation source changed')
        if not p.exists():p.write_bytes(raw)
    p=root/REG;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists()and p.read_bytes()!=record_raw:raise ValueError('Existing continuation registration changed')
    if not p.exists():p.write_bytes(record_raw)
    spec=importlib.util.spec_from_file_location('e2_verified_fits_only',root/'scripts/e2_posthoc_fits_only_resume.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.run(root,globals()if namespace is None else namespace,started=started)
