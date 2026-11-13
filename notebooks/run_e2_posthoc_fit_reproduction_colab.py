"""One-cell authorized posthoc closure: searches, frozen diagnostics, clean fits.

Use only the existing original A100 workspace and its already-approved 80-hour
balance. New clean sources/packages are prepared separately; old control stays
read only. GitHub is the sole persistent destination. No new consent is asked.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import threading
from urllib.request import Request, urlopen

ENGINE_COMMIT='1e8745497b484518f766b6da5c19be8362d9add7'
REGISTRATION='experiments/e2_posthoc_fit_reproduction_20261113/registration.json'
WORKSPACE=Path('/content/entregable2_gpu_workspace')
REQUIRED=(
    'scripts/e2_posthoc_fit_reproduction.py',
    'scripts/e2_frozen_posthoc_gpu.py',
    'scripts/export_e2_f2a_cached_data.py',
    'scripts/e2_native_transverse_colab.py',
    'scripts/e2_reduced_github_backup.py',
    'scripts/e2_reduced_github_state.py',
    'scripts/e2_reduced_github_restore.py',
    'notebooks/run_e2_posthoc_fit_reproduction_colab.py',
)


def download(commit,relative):
    url='https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+commit+'/'+relative
    with urlopen(Request(url,headers={'Cache-Control':'no-cache'}),timeout=35) as response:
        raw=response.read(4*1024**2+1)
    if len(raw)>4*1024**2:
        raise RuntimeError('Un archivo publicado supera su límite de tamaño')
    return raw


def materialize(commit,workspace):
    raw=download(commit,REGISTRATION);registration=json.loads(raw)
    if (registration.get('scientific_commit')!=ENGINE_COMMIT or registration.get('new_budget_authorization') is not False
            or not set(REQUIRED).issubset(registration.get('files_sha256',{}))):
        raise RuntimeError('El registro no corresponde al cierre post hoc sin ampliación')
    root=Path(workspace)/('guide-closure-extension-'+commit)
    for relative in REQUIRED:
        path=root/relative
        if any(p.is_symlink() for p in (path,*path.parents)) or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError('La ruta de la extensión debe ser regular')
        payload=download(commit,relative)
        if hashlib.sha256(payload).hexdigest()!=registration['files_sha256'][relative]:
            raise RuntimeError('Una fuente publicada difiere: '+relative)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.read_bytes()!=payload:
            raise RuntimeError('Una extensión local cambió; se conserva para revisión')
        if not path.exists():
            with path.open('xb') as stream:stream.write(payload)
    path=root/REGISTRATION;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=raw:
        raise RuntimeError('Un registro local cambió')
    if not path.exists():
        with path.open('xb') as stream:stream.write(raw)
    return root


def run_e2_posthoc_fit_reproduction_colab(extension_commit, *, namespace=None, execute=False):
    """Execute all requested work, without touching old frozen control pointers."""
    if not isinstance(extension_commit,str) or not re.fullmatch(r'[0-9a-f]{40}',extension_commit):
        raise ValueError('Usa el commit inmutable publicado del cierre')
    if type(execute)is not bool:
        raise TypeError('execute debe ser booleano')
    plan={'execute':execute,'scientific_commit':ENGINE_COMMIT,'original_total_hours':80,
          'new_budget_authorization':False,'original_remaining_seconds':18169.090972095262,
          'steps':['3 consultas nativas','diagnósticos de predictores congelados',
                   'entorno y checkout limpios','81 ajustes finales con recetas/épocas congeladas',
                   '18 ablaciones reales SAE y bootstrap pareado >=10000'],
          'original_freeze_read_only':True,'persistent_destination':'GitHub'}
    if not execute:
        return plan
    namespace=globals() if namespace is None else namespace
    existing_python=Path(namespace.get('GPU_PYTHON') or
        WORKSPACE/'reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python')
    original_source=WORKSPACE/('reduced-source-'+ENGINE_COMMIT)
    original_output=WORKSPACE/'github-primary-output-v1'
    if not existing_python.is_file() or not (original_source/'.git').exists() or not original_output.is_dir():
        raise RuntimeError('Conserva la sesión A100 y sus fuentes, entorno y resultados originales')
    # Credentials stay in the host kernel and a private STDIN pipe. Never put
    # tokens in source files, arguments, child environment variables or logs.
    token=str(namespace.get('GITHUB_TOKEN') or os.environ.get('GITHUB_TOKEN','')).strip()
    if not token:
        from google.colab import userdata
        token=(userdata.get('GITHUB_TOKEN') or '').strip()
    if not token or any(c.isspace() for c in token):
        raise RuntimeError('Activa el secreto GITHUB_TOKEN existente; no lo pegues en el chat')
    started=time.monotonic()
    extension=materialize(extension_commit,WORKSPACE)
    env=dict(os.environ)
    for key in ('GITHUB_TOKEN','GH_TOKEN','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    command=[existing_python,extension/'scripts/e2_posthoc_fit_reproduction.py','--orchestrate',
        '--original-source',original_source,'--original-output',original_output,
        '--destination',WORKSPACE/'e2-guide-closure-posthoc-v1','--existing-python',existing_python,
        '--extension-root',extension,'--configuration-commit',extension_commit,
        '--launcher-elapsed-seconds',str(time.monotonic()-started)]
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    print('Se conserva el límite total de80h; cada ajuste se admite solo si su reserva cabe en el saldo.',flush=True)
    child=subprocess.Popen([str(x) for x in command],cwd=extension,env=env,stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    child.stdin.write(token+'\n');child.stdin.close()
    def stop_at_budget_deadline():
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
    watchdog=threading.Timer(max(1.,18169.090972095262-(time.monotonic()-started)),stop_at_budget_deadline)
    watchdog.daemon=True;watchdog.start()
    try:
        for line in child.stdout:
            print(line.replace(token,'[REDACTED]'),end='',flush=True)
        code=child.wait()
    except BaseException:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
        raise
    finally:
        watchdog.cancel()
    result={**plan,'exit_code':code,'launcher_elapsed_seconds':time.monotonic()-started,
            'extension_commit':extension_commit,'actual_execution_results_require_completed_receipts':True}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if code:
        raise RuntimeError('El controlador se detuvo; conserva los recibos y no repitas ajustes perdidos')
    return result
