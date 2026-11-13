"""Restore only pinned prerequisites in a fresh A100 session; never launch fits.

The caller owns the existing remote budget/lease restoration and charges setup
wall time. This helper does not read, amend or initialize a scientific ledger.
"""
from __future__ import annotations
import hashlib,json,math,os,re,subprocess,time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import urllib.request
from urllib.request import Request,urlopen

ENGINE_COMMIT='1e8745497b484518f766b6da5c19be8362d9add7'
ORIGINAL_POSTHOC_COMMIT='ae1a2d4f96807a20ea92c313297a5c09512a4817'
WORKSPACE=Path('/content/entregable2_gpu_workspace')
_DEADLINE=ContextVar('e2_fresh_preparation_deadline',default=None)
PINS={
 'runtime':('d62a616a576fca304dcf987efefbba29ab990541','notebooks/preparar_e2_a100_runtime_colab.py',
            '1b7d69c5ef568491067829652540bd6b0e992aa19526a1d3f5d2ac5f6cfd230b'),
 'checkout':('70316a2aa146a7f647ee36c88080022ec86f8005','notebooks/run_e2_reduced_github_colab.py',
             '47d43540a7345135e731fc831a603b939ba324f2072f7af929f84b50836d0e85'),
 'posthoc':(ORIGINAL_POSTHOC_COMMIT,'notebooks/run_e2_posthoc_fit_reproduction_colab.py',
            '5407e40fcebe463bae25b78f4993db5ef13d567fc1fe31b0571c261a2c02fef8'),
}
METADATA_PATH='book_e2/_static/current_implementation_20261108/source_design.json'
METADATA_SHA='ce0a76304186fabf922f8e1351ea2e9d44db49f1714f44cd1fe2df04baeb3dbe'
FETCH_SCRIPTS={
 'scripts/fetch_e2_development_sources.py':'e91ea09398bca0f44f29ffe9baf4cb723c273ccae88b243863c158b602a5becc',
 'scripts/fetch_e2_published_sources.py':'4e51d63ffcabb075377b137e68c20926648de0a5fce7e5115a5ce98ec0551e17',
}
CONTROLLERS={'e2_posthoc_fit_reproduction.py','e2_resource_worker.py','run_e2_reduced_github_delivery.py',
 'run_e2_reduced_github72_delivery.py','run_e2_a100_recovery_delivery.py','run_e2_reduced_selection.py',
 'run_e2_cuda_search.py','run_e2_cuda_recovery.py','run_e2_optimized_selection_v2.py',
 'e2_posthoc_environment_resume.py','e2_posthoc_clean_worker_bootstrap.py'}

def active_pids(proc_root=Path('/proc')):
    found=[]
    for p in Path(proc_root).glob('[0-9]*/cmdline'):
        if int(p.parent.name)==os.getpid():continue
        try:args=p.read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(a and not any(c.isspace() for c in a) and Path(a).name in CONTROLLERS for a in args):
            found.append(int(p.parent.name))
    return found

def regular(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Fresh preparation paths must be regular')
    return path

def timeout_left(deadline,requested):
    if deadline is None:return requested
    if type(deadline) not in (int,float) or not math.isfinite(deadline):raise ValueError('Finite preparation deadline required')
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Fresh preparation reached its global deadline')
    return remaining if requested is None else min(float(requested),remaining)

@contextmanager
def deadline_limits(deadline):
    """Bound nested pinned setup functions inside the caller's isolated process.

    Their immutable sources are unchanged. Standard-library hooks apply only
    during this synchronous preparation and are restored on every exit. All
    descendants remain in the outer process group, so its hard stop reaches
    them rather than leaving pip/uv grandchildren running after lease closure.
    """
    if deadline is None:
        yield
        return
    timeout_left(deadline,None)
    original_run=subprocess.run;original_open=urllib.request.urlopen;original_local=urlopen
    active=_DEADLINE.set(deadline)
    def bounded_run(*args,**kwargs):
        kwargs=dict(kwargs);kwargs['timeout']=timeout_left(deadline,kwargs.get('timeout'))
        kwargs['start_new_session']=False
        return original_run(*args,**kwargs)
    def cap_open(fetch):
        def bounded_open(*args,**kwargs):
            args=list(args);kwargs=dict(kwargs)
            if len(args)>=3:args[2]=timeout_left(deadline,args[2])
            else:kwargs['timeout']=timeout_left(deadline,kwargs.get('timeout',40))
            return fetch(*args,**kwargs)
        return bounded_open
    subprocess.run=bounded_run;urllib.request.urlopen=cap_open(original_open);globals()['urlopen']=cap_open(original_local)
    try:yield
    finally:
        subprocess.run=original_run;urllib.request.urlopen=original_open;globals()['urlopen']=original_local
        _DEADLINE.reset(active)

def public_bytes(commit,path,expected,*,size=None,opener=None,deadline=None):
    deadline=_DEADLINE.get() if deadline is None else deadline
    if re.fullmatch('[0-9a-f]{40}',commit) is None:raise ValueError('Immutable source commit required')
    fetch=urlopen if opener is None else opener
    url='https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+commit+'/'+path
    with fetch(Request(url,headers={'Cache-Control':'no-cache'}),timeout=timeout_left(deadline,40)) as response:
        raw=response.read((size if size is not None else 4*1024**2)+1)
    if len(raw)>4*1024**2 or (size is not None and len(raw)!=size) or hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('Pinned preparation source mismatch: '+path)
    return raw

def module_for(label):
    commit,path,expected=PINS[label];raw=public_bytes(commit,path,expected)
    namespace={'__name__':'e2_fresh_preparation_'+label,'__file__':path}
    exec(compile(raw,path,'exec'),namespace)
    return namespace

def once(path,raw):
    path=regular(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Existing source differs; retained: '+str(path))
    else:
        with path.open('xb') as f:f.write(raw)

def run(command,*,env,cwd,deadline=None):
    deadline=_DEADLINE.get() if deadline is None else deadline
    return subprocess.run([str(x) for x in command],env=env,cwd=cwd,check=True,timeout=timeout_left(deadline,1800))

def _prepare_fresh_posthoc_session(namespace=None,*,workspace=None,execute=False):
    if type(execute)is not bool:raise TypeError('execute must be boolean')
    if not execute:return {'status':'dry_run','scientific_commit':ENGINE_COMMIT,'training_started':False,
                          'old_controller_started':False,'ledger_read_or_changed':False,'Google_Drive_used':False}
    start=time.monotonic();namespace=globals() if namespace is None else namespace
    workspace=regular(workspace or WORKSPACE)
    if active_pids():raise RuntimeError('Another E2 scientific controller is active; preparation stopped')
    workspace.mkdir(parents=True,exist_ok=True)
    print('Restaurando requisitos de la nueva sesión; todavía sin entrenamiento.',flush=True)
    runtime=module_for('runtime')
    runtime['prepare_e2_a100_runtime'](namespace,workspace=workspace/'reduced-runtime-py312-cu126-v1')
    python=Path(namespace['GPU_PYTHON'])
    env=dict(os.environ)
    for key in ('GITHUB_TOKEN','GH_TOKEN','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    checkout=module_for('checkout');checkout['_assert_idle']()
    print('Recuperando el checkout científico exacto y sus entradas verificadas.',flush=True)
    source=checkout['_checkout'](ENGINE_COMMIT,workspace,env)
    metadata=regular(source/METADATA_PATH)
    raw=metadata.read_bytes() if metadata.exists() else public_bytes(ENGINE_COMMIT,METADATA_PATH,METADATA_SHA,size=18811)
    if len(raw)!=18811 or hashlib.sha256(raw).hexdigest()!=METADATA_SHA:raise ValueError('Architecture metadata differs')
    once(metadata,raw)
    for relative,expected in FETCH_SCRIPTS.items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:raise ValueError('Pinned acquisition script changed')
    run([python,source/'scripts/fetch_e2_development_sources.py','--output',source],env=env,cwd=source)
    print('Recuperando fuentes autorales verificadas; sin pesos ni ajustes nuevos.',flush=True)
    author_root=workspace/'reduced-author-sources'
    run([python,source/'scripts/fetch_e2_published_sources.py','--output',author_root],env=env,cwd=source)
    original=module_for('posthoc');extension=original['materialize'](ORIGINAL_POSTHOC_COMMIT,workspace)
    if hashlib.sha256((extension/PINS['posthoc'][1]).read_bytes()).hexdigest()!=PINS['posthoc'][2]:
        raise ValueError('Restored original posthoc launcher differs')
    result={'version':'e2-fresh-session-prerequisites-v1','GPU_PYTHON':str(python),'engine_root':str(source),
        'source_roots':str(author_root/'source_roots.json'),'original_extension_root':str(extension),
        'original_launcher':str(extension/PINS['posthoc'][1]),'scientific_commit':ENGINE_COMMIT,
        'original_configuration_commit':ORIGINAL_POSTHOC_COMMIT,'training_started':False,
        'old_controller_started':False,'ledger_read_or_changed':False,'Google_Drive_used':False,
        'setup_elapsed_seconds':time.monotonic()-start,'accounting_scope':'Caller must charge this setup wall time to existing remote budget',
        'original_output_restored':False,'posthoc_output_restored':False,'persistent_destination':'GitHub'}
    namespace['E2_FRESH_PREREQUISITES']=result
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result

def prepare_fresh_posthoc_session(namespace=None,*,workspace=None,execute=False,deadline=None):
    if type(execute)is not bool:raise TypeError('execute must be boolean')
    if not execute:return _prepare_fresh_posthoc_session(namespace,workspace=workspace,execute=False)
    with deadline_limits(deadline):
        value=_prepare_fresh_posthoc_session(namespace,workspace=workspace,execute=True)
        timeout_left(deadline,None)
        return value
