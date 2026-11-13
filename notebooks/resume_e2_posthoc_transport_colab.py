"""Resume the unchanged controller after a controlled inference-only pause."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import runpy
from urllib.request import Request, urlopen

ORIGINAL_COMMIT='ae1a2d4f96807a20ea92c313297a5c09512a4817'
WORKSPACE=Path('/content/entregable2_gpu_workspace')
REGISTRATION='experiments/e2_posthoc_transport_20261113/registration.json'
WRAPPER='scripts/e2_posthoc_transport_resume.py'
PAUSE='scripts/pause_e2_posthoc_inference.py'
REQUIRED=(WRAPPER,PAUSE,'notebooks/resume_e2_posthoc_transport_colab.py')


def _download(commit,path):
    with urlopen(Request('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+commit+'/'+path,
                         headers={'Cache-Control':'no-cache'}),timeout=35) as response:
        raw=response.read(4*1024**2+1)
    if len(raw)>4*1024**2:
        raise ValueError('El archivo supera el límite de metadatos')
    return raw


def _write_once(path,payload):
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('El adaptador requiere rutas regulares')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=payload:
        raise ValueError('Se conserva un adaptador diferente para revisión')
    if not path.exists():
        with path.open('xb') as stream:stream.write(payload)


def _active(destination,pause):
    active={};previous=pause.ZERO
    path=destination/'budget/posthoc_budget.jsonl'
    raw=path.read_bytes()
    if not raw.endswith(b'\n'):
        raise RuntimeError('Se conserva un diario incompleto para revisión')
    for sequence,line in enumerate(raw.splitlines()):
        row=json.loads(line)
        if (row.get('sequence')!=sequence or row.get('previous_sha256')!=previous
                or row.get('sha256')!=pause.sha(pause.canonical({k:v for k,v in row.items() if k!='sha256'}))):
            raise ValueError('El diario existente no coincide con su cadena')
        previous=row['sha256']
        if row['kind']=='reserved':active[row['job_id']]=row['seconds']
        elif row['kind']=='charged':active.pop(row['job_id'],None)
    return active


def resume_e2_posthoc_transport_colab(adapter_commit,*,namespace=None,execute=False):
    if not isinstance(adapter_commit,str) or re.fullmatch(r'[0-9a-f]{40}',adapter_commit) is None:
        raise ValueError('Usa el commit inmutable del adaptador')
    if type(execute) is not bool:
        raise TypeError('execute debe ser booleano')
    if not execute:
        return {'status':'dry_run','original_configuration_commit':ORIGINAL_COMMIT,
                'transport_only':True,'budget_reset':False,'release_tag':'e2-guide-closure-posthoc-20261113-v1'}
    ns=dict(globals() if namespace is None else namespace)
    registration=json.loads(_download(adapter_commit,REGISTRATION))
    if (registration.get('original_configuration_commit')!=ORIGINAL_COMMIT
            or not set(REQUIRED).issubset(registration.get('files_sha256',{}))):
        raise ValueError('El adaptador debe conservar la configuración original')
    root=WORKSPACE/('transport-adapter-'+adapter_commit)
    for relative in REQUIRED:
        payload=_download(adapter_commit,relative)
        if hashlib.sha256(payload).hexdigest()!=registration['files_sha256'][relative]:
            raise ValueError('El adaptador no coincide con su registro')
        _write_once(root/relative,payload)
    spec=importlib.util.spec_from_file_location('e2_controlled_transport_pause',root/PAUSE)
    pause=importlib.util.module_from_spec(spec);spec.loader.exec_module(pause)
    destination=WORKSPACE/'e2-guide-closure-posthoc-v1'
    if any(pause.matching_controller(p,destination) for p in pause.processes()):
        raise RuntimeError('El controlador sigue activo; ejecuta primero la pausa controlada de Terminal')
    pause.no_optimizer(destination)
    proofs=list((destination/'transport_amendments/pause_observers').glob('*.json'))
    if not proofs:
        raise RuntimeError('Falta el recibo de la pausa controlada de inferencia')
    proof_path=max(proofs,key=lambda p:p.stat().st_mtime_ns)
    proof=json.loads(proof_path.read_bytes())
    if _active(destination,pause):
        token=ns.get('GITHUB_TOKEN','')
        if not token:
            from google.colab import userdata
            token=userdata.get('GITHUB_TOKEN') or ''
        result=pause.reconcile(proof_path,token=token)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    else:
        amendment=destination/'transport_amendments/interrupted_inference'/proof['proof_id']/'amendment.json'
        if not amendment.exists() or json.loads(amendment.read_bytes()).get('observer_proof')!=proof:
            raise RuntimeError('La conciliación previa no tiene un recibo verificable')
    if _active(destination,pause):
        raise RuntimeError('Una reserva permanece abierta; no se inicia otro controlador')
    for folder in (destination/'robustness/predictors').glob('*'):
        if (folder/'started.json').exists() and not (folder/'result.json').exists():
            raise RuntimeError('Falta conservar el intento de inferencia interrumpido')
    path=root/'python'
    _write_once(path,(root/WRAPPER).read_bytes());path.chmod(0o700)
    extension=WORKSPACE/('guide-closure-extension-'+ORIGINAL_COMMIT)
    original=extension/'notebooks/run_e2_posthoc_fit_reproduction_colab.py'
    old_registration=json.loads((extension/'experiments/e2_posthoc_fit_reproduction_20261113/registration.json').read_bytes())
    if hashlib.sha256(original.read_bytes()).hexdigest()!=old_registration['files_sha256']['notebooks/run_e2_posthoc_fit_reproduction_colab.py']:
        raise ValueError('El lanzador original cambió')
    ns['GPU_PYTHON']=str(path)
    return runpy.run_path(str(original))['run_e2_posthoc_fit_reproduction_colab'](
        ORIGINAL_COMMIT,namespace=ns,execute=True)
