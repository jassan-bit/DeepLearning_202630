"""Prepare pinned Python/Torch on the selected A100; no market or budget work."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import urllib.request

ENV_COMMIT='79f4ee4719346a9c00d795ed6f030ec69a2f0203'
HELPER_PATH='notebooks/run_e2_pilots_colab.py'
HELPER_SHA256='bd65ecfe77ba017f5610404f038b2d6edf1dd3468163abb481c9cf355575551e'
LOCK_PATH='requirements-e2-training-gpu.lock.txt'
LOCK_SHA256='5201f687d8c600ae9458444dbc46e5bc97bdedf938ddb21801887f6f9d4508d9'
WORKSPACE=Path('/content/entregable2_gpu_workspace/reduced-runtime-py312-cu126-v1')
CONTROLLERS={'run_e2_reduced_github_delivery.py','run_e2_reduced_github72_delivery.py',
 'run_e2_a100_recovery_delivery.py','e2_resource_worker.py','run_e2_reduced_selection.py',
 'run_e2_cuda_search.py','run_e2_cuda_recovery.py','run_e2_optimized_selection_v2.py'}


def _bytes(path,expected):
    url='https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+ENV_COMMIT+'/'+path
    with urllib.request.urlopen(url,timeout=30) as response:raw=response.read(131073)
    if len(raw)>131072 or hashlib.sha256(raw).hexdigest()!=expected:
        raise RuntimeError('The pinned environment source differs: '+path)
    return raw


def _assert_idle():
    current=os.getpid();found=[]
    for item in Path('/proc').iterdir():
        if not item.name.isdigit() or int(item.name)==current:continue
        try:words=(item/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if any(word and Path(word).name in CONTROLLERS for word in words[1:]):found.append(int(item.name))
    if found:raise RuntimeError('A scientific controller is active; environment preparation stopped: '+str(found))


def prepare_e2_a100_runtime(namespace=None,*,workspace=None):
    namespace=globals() if namespace is None else namespace
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise RuntimeError('Use the allocated Linux x86_64 A100 Colab runtime')
    _assert_idle()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    if 'A100' not in gpu:raise RuntimeError('The actual allocated GPU is not A100: '+gpu)
    helper=_bytes(HELPER_PATH,HELPER_SHA256);lock=_bytes(LOCK_PATH,LOCK_SHA256)
    module={'__name__':'e2_environment_functions_only'}
    exec(compile(helper,'e2_pinned_environment_functions','exec'),module)
    workspace=Path(workspace or WORKSPACE)
    if any(p.is_symlink() for p in (workspace,*workspace.parents)):
        raise RuntimeError('The isolated environment must not resolve through a symlink')
    python=workspace/'venv-py312-cu126/bin/python'
    if not python.is_file() and python.parent.parent.exists():
        raise RuntimeError('An incomplete existing environment was preserved: '+str(python.parent.parent))
    workspace.mkdir(parents=True,exist_ok=True)
    source=workspace/('pinned-source-'+ENV_COMMIT);source.mkdir(exist_ok=True)
    locked=source/LOCK_PATH
    if locked.exists() and locked.read_bytes()!=lock:raise RuntimeError('The local dependency lock differs')
    if not locked.exists():locked.write_bytes(lock)
    env=dict(os.environ)
    for key in ('GITHUB_TOKEN','GH_TOKEN','PYTHONPATH'):env.pop(key,None)
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',UV_NATIVE_TLS='true',
        UV_CACHE_DIR=str(workspace/'uv-cache'),UV_PYTHON_INSTALL_DIR=str(workspace/'managed-python'),
        UV_PYTHON_BIN_DIR=str(workspace/'managed-python-bin'))
    print('Preparando Python3.12.12 y Torch2.8.0+cu126 para la A100; sin entrenamiento ni cambios de presupuesto.',flush=True)
    if not python.is_file():python=module['_bootstrap_python'](workspace,env)
    runtime=module['_gpu_environment'](python,source,env)
    if 'A100' not in runtime.get('device',''):raise RuntimeError('The pinned Torch runtime did not observe the A100')
    namespace['GPU_PYTHON']=str(python)
    result={'GPU_PYTHON':str(python),'runtime':runtime,'training_started':False,
        'budget_changed':False,'persistent_destination':'github','Google_Drive_used':False}
    namespace['E2_A100_RUNTIME_PREPARATION']=result
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result
