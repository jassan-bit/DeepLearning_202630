"""Isolated Colab GPU development stage with incremental GitHub persistence.

Call run(full_source_commit, token_in_memory). The default fits only the new
development candidate and its controls; it never reads old test/calibration
members and never enables the final campaign or asserts R1/R3.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request

REPO='https://github.com/jassan-bit/DeepLearning_202630.git'
SCOPE='experiments/e3_cierre_integral_20261118'
OLD='experiments/e3_remediacion_20261118/evaluation'
QUAD='experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification'
NPZ_COMMIT='af3f1223ad5899ed63016d2af2d09bae8470ab00'
NPZ_PATH='book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz'
NPZ_SHA256='186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683'


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,Path(path));module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def _environment():
    env=dict(os.environ)
    for key in list(env):
        if re.search(r'(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD)(?:_|$)',key.upper()):env.pop(key,None)
    env.update(GIT_TERMINAL_PROMPT='0',PYTHONNOUSERSITE='1')
    return env


def checkout(source,base,env):
    """A separate public sparse checkout; no token in git commands/config/env."""
    directory=Path(base)/('e3-closure-gpu-source-'+source)
    if not directory.exists():
        subprocess.run(['git','clone','--depth=1','--filter=blob:none','--no-checkout',REPO,str(directory)],
                       check=True,env=env)
        subprocess.run(['git','sparse-checkout','init','--cone'],cwd=directory,check=True,env=env)
        subprocess.run(['git','sparse-checkout','set','src','scripts',SCOPE+'/campaign',
                        SCOPE+'/orchestration',OLD,QUAD],cwd=directory,check=True,env=env)
        subprocess.run(['git','fetch','--depth=1','origin',source],cwd=directory,check=True,env=env)
        subprocess.run(['git','checkout','--detach',source],cwd=directory,check=True,env=env)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,env=env,text=True).strip()
    if actual!=source:raise RuntimeError('Preserved checkout has a different HEAD; it was not changed')
    subprocess.run(['git','diff','--exit-code','HEAD'],cwd=directory,env=env,check=True,capture_output=True)
    extra=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=directory,env=env,text=True).splitlines()
    if any('__pycache__' not in Path(p).parts and not p.endswith('.pyc') for p in extra):
        raise RuntimeError('Untracked files in code checkout require reconciliation')
    return directory


def admit_npz(base):
    path=Path(base)/('e3-training-source-'+NPZ_SHA256[:12]+'.npz')
    if not path.exists():
        url='https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+NPZ_COMMIT+'/'+NPZ_PATH
        with urllib.request.urlopen(url,timeout=60) as response:raw=response.read(2**20)
        if len(raw)!=460090 or hashlib.sha256(raw).hexdigest()!=NPZ_SHA256:
            raise RuntimeError('Pinned source NPZ bytes differ; nothing was fitted')
        with path.open('xb') as stream:stream.write(raw)
    if path.stat().st_size!=460090 or hashlib.sha256(path.read_bytes()).hexdigest()!=NPZ_SHA256:
        raise RuntimeError('Cached source NPZ changed; it was preserved without fitting')
    return path


def _fit_attempts(jobs):
    # Counts attempted model fits once per job, not epochs or seed replicas for
    # deterministic controls. SAE's three optimizer phases are reported apart.
    attempts=phases=0
    for job in jobs:
        phase_directories={p.parent for pattern in ('phase_started.json','checkpoint.pt')
                           for p in job.rglob(pattern)}
        phases+=len(phase_directories)
        attempts+=int(bool(phase_directories) or (job/'arima_fit_started.json').exists())
    return attempts,phases


def create_snapshot(output,destination,source,master_run_id,jobs,*,partial,prior_saves):
    """Raw-byte shards keep PyTorch ZIP checkpoints intact, without nesting ZIPs."""
    output=Path(output);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    paths=[]
    for path in output.iterdir():
        if path.is_file() and path.suffix in {'.json','.csv','.log','.txt'}:paths.append(path)
    for job in jobs:
        paths.extend(p for p in job.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
    total=0
    for path in paths:
        size=path.stat().st_size
        if size>48*1024**2:raise RuntimeError('One checkpoint exceeds the upload budget; preserve local output')
        total+=size
        target=destination/path.relative_to(output);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    if total>192*1024**2 or len(paths)>900:
        raise RuntimeError('Snapshot shard exceeds the conservative upload budget; preserve local output')
    fits,phases=_fit_attempts(jobs)
    receipt={'kind':'e3_closure_stage_receipt','profile':'pilot','source_commit':source,
             'master_run_id':master_run_id,'partial':partial,'new_real_data_fits':fits,
             'attempted_optimizer_phases':phases,'fit_counter':'unique attempted model jobs in this shard; not optimizer epochs',
             'available_phase_checkpoints':sum(len(list(job.rglob('checkpoint.pt'))) for job in jobs),
             'jobs_in_shard':[p.name for p in jobs],'prior_saved_parts':prior_saves,
             'R1_accredited':False,'R3_accredited':False,'real_campaign_allowed':False,
             'additional_datasets_evaluated':0,'new_test_predictions':0,
             'scope':'BTC exposed development only; all checkpoint and selected-model bytes retained'}
    (destination/'stage_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return destination


def run(source_commit,token,*,base='/content',suite='controls',
        seeds=(20261118,20261119,20261120),max_epochs=40,patience=8,
        resume_output=None,save_every_jobs=3):
    """GPU stage; token stays in this controller's memory and GitHub requests."""
    if re.fullmatch('[0-9a-f]{40}',source_commit) is None:raise ValueError('Full immutable source SHA required')
    if suite not in ('controls','full_family'):raise ValueError('Unknown development suite')
    if type(save_every_jobs)is not int or not 1<=save_every_jobs<=3:raise ValueError('Persist every one to three jobs')
    base=Path(base);base.mkdir(parents=True,exist_ok=True);env=_environment()
    source=checkout(source_commit,base,env)
    api=load(source/OLD/'colab_github.py','e3_gpu_storage_api')
    store=load(source/SCOPE/'orchestration/github_store.py','e3_gpu_closure_store')
    token=api._token(token);api.validate_credentials(token)
    runtime=load(source/OLD/'colab_runtime.py','e3_gpu_isolated_runtime')
    python,gpu_env,runtime_receipt=runtime.prepare_runtime(base/'e3-preflight-gpu-runtime-v1')
    npz=admit_npz(base);master=stamp()
    output=Path(resume_output) if resume_output else base/('e3-closure-pilot-'+master)
    output.mkdir(parents=True,exist_ok=resume_output is not None)
    if not (output/'isolated_runtime.json').exists():
        (output/'isolated_runtime.json').write_text(json.dumps(runtime_receipt,indent=2)+'\n')
    master_record=output/'master_run.json'
    if master_record.exists():
        previous=json.loads(master_record.read_text())
        if previous['source_commit']!=source_commit:raise ValueError('Resume master belongs to another source')
        master=previous['master_run_id']
    else:
        master_record.write_text(json.dumps({'source_commit':source_commit,'master_run_id':master})+'\n')
    snapshots=base/('e3-closure-pilot-snapshots-'+stamp());snapshots.mkdir()
    command=[str(python),str(source/SCOPE/'campaign/pilot.py'),'--source-commit',source_commit,
        '--output',str(output),'--npz',str(npz),'--sha256',NPZ_SHA256,'--suite',suite,
        '--max-epochs',str(max_epochs),'--patience',str(patience),'--seeds',*[str(s) for s in seeds]]
    seen=set();saves=[];storage_errors=[]

    def persist(jobs,partial=True):
        # Three consecutive large TimesNet jobs remain comfortably under192MiB.
        for offset in range(0,len(jobs),save_every_jobs):
            group=jobs[offset:offset+save_every_jobs];part=stamp()
            evidence=create_snapshot(output,snapshots/part,source_commit,master,group,
                                     partial=partial,prior_saves=saves)
            try:
                saved=store.save(evidence,token,source_commit,part,api,profile='pilot')
                saves.append({'part':part,'jobs':[p.name for p in group],**saved})
                seen.update(p.name for p in group)
                print('Evidencia guardada en GitHub:',saved['run_url'],flush=True)
            except api.StorageError as exc:
                # Keep the exact immutable snapshot for a token/network retry.
                storage_errors.append({'part':part,'local_snapshot':str(evidence),'error':str(exc)})
                print('Persistencia pendiente; se conservó el snapshot local:',evidence,flush=True)
            (snapshots/'storage_index.json').write_text(json.dumps({'master_run_id':master,'saved':saves,
                'pending':storage_errors},indent=2)+'\n')

    print('Etapa GPU: desarrollo BTC 532/116; no se consultan test ni conjuntos adicionales.',flush=True)
    process=subprocess.Popen(command,cwd=source,env=gpu_env,stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,text=True,bufsize=1)
    try:
        with (output/'gpu_worker.log').open('a') as log:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
                if ': completed;' in line:
                    ready=[p for p in sorted((output/'jobs').iterdir())
                           if p.name not in seen and (p/'result.json').is_file()]
                    if len(ready)>=save_every_jobs:persist(ready)
        code=process.wait()
    except BaseException:
        process.terminate()
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise
    finally:
        pending=[p for p in sorted((output/'jobs').iterdir()) if p.is_dir() and p.name not in seen] if (output/'jobs').exists() else []
        if pending:persist(pending,partial=not (output/'pilot_receipt.json').exists())
        # The final small index contains all shard URLs plus the complete panel.
        (output/'storage_parts.json').write_text(json.dumps({'master_run_id':master,'saved':saves,
            'pending':storage_errors},indent=2)+'\n')
        # Save one final metadata-only part with the complete panel and URLs.
        part=stamp();evidence=create_snapshot(output,snapshots/part,source_commit,master,[],
            partial=not (output/'pilot_receipt.json').exists(),prior_saves=saves)
        try:
            saved=store.save(evidence,token,source_commit,part,api,profile='pilot')
            saves.append({'part':part,'jobs':[],**saved});print('Índice final de evidencia:',saved['run_url'],flush=True)
        except api.StorageError as exc:
            storage_errors.append({'part':part,'local_snapshot':str(evidence),'error':str(exc)})
    return {'worker_exit_code':code,'output':str(output),'source_commit':source_commit,
            'saved_parts':saves,'pending_storage':storage_errors,'real_campaign_allowed':False,
            'R1_accredited':False,'R3_accredited':False}
