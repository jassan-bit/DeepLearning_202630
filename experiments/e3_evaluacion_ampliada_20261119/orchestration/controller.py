"""Colab controller for the actual extended audit and verified continuation.

The controller owns credentials; scientific subprocesses never receive them.
All historical runs stay unchanged. Completion is not R1/R3 accreditation.
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
import sys
import tempfile

REPOSITORY='jassan-bit/DeepLearning_202630'
REPO='https://github.com/'+REPOSITORY+'.git'
SCOPE='experiments/e3_evaluacion_ampliada_20261119'
OLD='experiments/e3_remediacion_20261118/evaluation'
SPARSE=['src','scripts',SCOPE,'deployment/science',
    'experiments/e3_cierre_integral_20261118/campaign',
    'experiments/e3_cierre_integral_20261118/data',
    'experiments/e3_cierre_integral_20261118/orchestration',OLD,
    'experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification',
    'experiments/e3_estadistica_posterior_20261117',
    'experiments/e3_dataset_admission_20261118/inputs',
    'experiments/e3_d01_admission_20261118']


def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def atomic(path,raw):
    path=Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Symlink path refused')
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.partial-',delete=False) as stream:
        temporary=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def environment():
    env=dict(os.environ)
    for name in list(env):
        if (re.search(r'(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD|CREDENTIAL)(?:_|$)',name.upper())
                or name in {'PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'} or name.startswith(('PIP_','UV_'))):
            env.pop(name,None)
    env.update(GIT_TERMINAL_PROMPT='0',PYTHONNOUSERSITE='1',PIP_CONFIG_FILE=os.devnull,
               PIP_DISABLE_PIP_VERSION_CHECK='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
               MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def checkout(source,base):
    if re.fullmatch('[0-9a-f]{40}',source or '') is None:raise ValueError('Full source SHA required')
    folder=Path(base)/('e3-extended-source-'+source)
    if any(p.is_symlink() for p in (folder,*folder.parents)):raise ValueError('Checkout symlink refused')
    env=environment()
    if not folder.exists():
        subprocess.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout',REPO,str(folder)],env=env,check=True)
        subprocess.run(['git','sparse-checkout','init','--cone'],cwd=folder,env=env,check=True)
        subprocess.run(['git','sparse-checkout','set',*SPARSE],cwd=folder,env=env,check=True)
        subprocess.run(['git','fetch','--depth','1','origin',source],cwd=folder,env=env,check=True)
        subprocess.run(['git','checkout','--detach',source],cwd=folder,env=env,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=folder,env=env,text=True).strip()
    if actual!=source:raise RuntimeError('Preserved checkout has a different source SHA')
    subprocess.run(['git','diff','--exit-code','HEAD'],cwd=folder,env=env,check=True,capture_output=True)
    return folder


def _main_ancestry(api,token,older,newer):
    result=api._request('GET','/repos/'+REPOSITORY+'/compare/'+older+'...'+newer,token)
    if (result.get('status') not in {'ahead','identical'}
            or result.get('base_commit',{}).get('sha')!=older
            or result.get('merge_base_commit',{}).get('sha')!=older):
        raise ValueError('A resumed source must be an ancestor of the requested published source')


def _select_run(rows,master_run_id=None):
    selected=[row for row in rows if master_run_id is None or row['master_run_id']==master_run_id]
    if master_run_id and len(selected)>1:raise ValueError('Ambiguous audit master identity')
    return selected[0] if selected else None


def _preprotocol_upgrade(source,base,local,previous,token,api):
    """Allow a collector fix only for a proved, never-started local audit.

    Once a protocol, a tag, jobs or durable audit state exists, source pinning
    remains absolute. A failed CPU admission is not a frozen model campaign.
    """
    if previous is not None or not local:return None
    master=local.get('master_run_id','');attempt=local.get('admission_run_id','')
    if (re.fullmatch('[0-9]{8}T[0-9]{12}Z',master) is None
            or re.fullmatch('[0-9]{8}T[0-9]{12}Z',attempt) is None):return None
    output=base/('e3-extended-audit-'+master)
    admitted=base/('e3-extended-admission-'+attempt)
    if local.get('output')!=str(output) or local.get('admission_output')!=str(admitted):return None
    if any(p.is_symlink() for p in (output,admitted,*output.parents,*admitted.parents)):
        raise ValueError('Preprotocol recovery symlink refused')
    state=base/('e3-extended-continuation-'+master+'.json')
    view=base/('e3-extended-persist-view-'+master)
    # At a blocked physical admission the audit output has no products. Reject
    # any unexpected files too, instead of inferring they are safe to discard.
    if (state.exists() or any(output.rglob('*')) or any(view.rglob('*'))):return None
    admission=load(source/SCOPE/'admission/colab.py','extended_controller_upgrade_admission')
    try:proof=admission.inspect_failed_stage(admitted)
    except (OSError,ValueError,KeyError):return None
    if proof.get('source_commit')!=local.get('source_commit'):return None
    tag='/repos/'+REPOSITORY+'/git/ref/tags/e3-extended-audit-protocol-'+master
    if api._request('GET',tag,token,allow_missing=True) is not None:return None
    return {**proof,'previous_admission_output':str(admitted),
            'master_run_id':master,'no_local_or_remote_protocol':True}


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:process.kill();process.wait()


def stream(command,cwd,env,log,on_job=None):
    """Every persistence failure stops this worker instead of burning GPU."""
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen([str(v) for v in command],cwd=cwd,env=env,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    completed=0
    try:
        with Path(log).open('a',encoding='utf-8') as output:
            for line in process.stdout:
                print(line,end='',flush=True);output.write(line);output.flush()
                if line.startswith('JOB_COMPLETED '):
                    completed+=1
                    if on_job and completed%3==0:on_job()
        code=process.wait()
    except BaseException:
        _stop(process);raise
    if code:raise RuntimeError('Worker exited with status '+str(code)+'; evidence retained')
    return completed


def _copy_once(source,target):
    source,target=Path(source),Path(target)
    if source.is_symlink():raise ValueError('Evidence symlink refused')
    raw=source.read_bytes()
    if not target.exists() or target.read_bytes()!=raw:atomic(target,raw)
    return raw


def stable_view(output,view,*,finished=False):
    """Materialize closed artifacts without pausing a time-budgeted GPU job.

    Existing closed files are retained while a resumed worker regenerates an
    intermediate, partial model panel. Only a full neural seed set replaces a
    previously complete panel during live execution.
    """
    output,view=Path(output),Path(view);view.mkdir(parents=True,exist_ok=True)
    terminal={'evaluation_receipt.json','run_manifest.json','failures.json'}
    for path in sorted(output.rglob('*')):
        if path.is_symlink():raise ValueError('Evidence symlink refused')
        if not path.is_file() or path.name.endswith('.tmp') or path.name.startswith('.partial-'):continue
        relative=path.relative_to(output)
        if relative.parts[0] in {'jobs','panels'} or relative.as_posix()=='panel_manifest.json':continue
        if not finished and (relative.parts[0] in {'analysis','figures'}
                             or (path.suffix=='.csv' and relative.parts[0] not in {'admission','inputs'})
                             or path.name in terminal):continue
        _copy_once(path,view/relative)
    jobs=output/'jobs'
    if jobs.exists():
        files=(jobs.rglob('*') if finished else
               (path for result in jobs.rglob('result.json') for path in result.parent.rglob('*')))
        for path in files:
            if path.is_symlink():raise ValueError('Evidence symlink refused')
            if path.is_file() and not path.name.endswith('.tmp') and not path.name.startswith('.partial-'):
                _copy_once(path,view/path.relative_to(output))
    manifest_path=output/'panel_manifest.json'
    if manifest_path.exists():
        original_manifest=manifest_path.read_bytes()
        current=json.loads(original_manifest)
        previous_path=view/'panel_manifest.json'
        previous=json.loads(previous_path.read_bytes()) if previous_path.exists() else {'files':[]}
        rows={row['path']:row for row in previous.get('files',[])}
        seeds=sorted(current['seeds'])
        for row in current['files']:
            observed=row.get('observed_seeds',[])
            if not finished and observed not in (seeds,[-1]):continue
            name=row['path'];path=Path(name)
            if path.is_absolute() or '..' in path.parts or path.parts[0]!='panels':
                raise ValueError('Panel path escapes its declared directory')
            raw=_copy_once(output/path,view/path)
            if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
                raise ValueError('A completed panel changed during persistence')
            rows[name]=row
        if finished:
            # run_manifest.json binds these EXACT producer bytes, including
            # formatting/order. A persister must not rewrite that artifact.
            atomic(previous_path,original_manifest)
        else:
            saved={**current,'files':[rows[k] for k in sorted(rows)],
                   'total_rows':sum(r['rows'] for r in rows.values()),
                   'persistence_complete_models_only':True}
            atomic(previous_path,canonical(saved))
    return view


def _gpu_profile(source,python,env):
    code=('import importlib.util,json,sys\n'
          's=importlib.util.spec_from_file_location("audit_profile",sys.argv[1])\n'
          'm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
          'print(json.dumps(m.compute_profile()))')
    raw=subprocess.check_output([str(python),'-c',code,str(source/SCOPE/'orchestration/protocol.py')],
        cwd=source,env=env,text=True)
    return json.loads(raw.strip().splitlines()[-1])


def _package_cache(source,base):
    """Verify vendor bytes; retain incomplete caches before retrying acquisition.

    This helper uses only standard-library inventory functions and is invoked
    under the isolated scientific Python by prepare_package, before any fit.
    """
    preflight=load(Path(source)/OLD/'colab_preflight.py','extended_package_preflight')
    destination=Path(base)/'e3-immutable-source-package-c4a4a1e68e98'
    if destination.is_symlink():raise ValueError('Immutable package cache symlink refused')

    def verify():
        archive=destination/'02_codigo_modelos.zip'
        if (archive.is_symlink() or archive.stat().st_size!=7839993
                or hashlib.sha256(archive.read_bytes()).hexdigest()!=preflight.PACKAGE_SHA256):
            raise ValueError('Immutable source archive differs from pinned size/hash')
        candidates=list((destination/'extracted').rglob('code/scripts/e2_standalone_code.py'))
        if len(candidates)!=1:raise ValueError('Immutable package extraction is incomplete')
        package=candidates[0].parents[2]
        inventory,_,_=preflight.portable_inventory(package)
        return package,inventory

    quarantine=None
    if destination.exists():
        try:package,inventory=verify()
        except (OSError,ValueError,KeyError):
            quarantine=destination.with_name(destination.name+'-quarantine-'+stamp())
            destination.rename(quarantine)
    if not destination.exists():
        preflight.acquire_package(destination)
        package,inventory=verify()
    return {'path':str(package),'archive_sha256':preflight.PACKAGE_SHA256,
            'archive_bytes':7839993,'inventory_sha256':hashlib.sha256(canonical(inventory)).hexdigest(),
            'quarantined_cache':None if quarantine is None else str(quarantine),
            'new_real_data_fits':0,'source_verified':True}


def prepare_package(source,base,python,env):
    code=('import importlib.util,json,sys\n'
          's=importlib.util.spec_from_file_location("audit_package_cache",sys.argv[1])\n'
          'm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
          'print(json.dumps(m._package_cache(sys.argv[2],sys.argv[3])))')
    raw=subprocess.check_output([str(python),'-c',code,str(source/SCOPE/'orchestration/controller.py'),
        str(source),str(base)],cwd=source,env=env,text=True)
    receipt=json.loads(raw.strip().splitlines()[-1])
    if receipt.get('source_verified') is not True:raise ValueError('Immutable package was not verified')
    return Path(receipt['path']),receipt


def analysis_runtime(source,base,log):
    """CPU plotting/statistics environment, separate from kernel and CUDA lock."""
    folder=Path(base)/'e3-extended-analysis-runtime-v1';folder.mkdir(parents=True,exist_ok=True)
    python=folder/'venv/bin/python';env=environment()
    env['MPLCONFIGDIR']=str(folder/'mpl-cache')
    lock=source/'deployment/science/requirements.lock.txt'
    expected={line.split('==')[0]:line.split('==')[1] for line in lock.read_text().splitlines()
              if line and not line.startswith('#')}
    if not python.is_file():
        stream([sys.executable,'-m','venv','--without-pip',folder/'venv'],source,env,log)
    probe=('import importlib.metadata as m\ne='+repr(expected)
           +'\nassert all(m.version(k)==v for k,v in e.items())\n')
    checked=subprocess.run([str(python),'-c',probe],env=env,capture_output=True,text=True)
    if checked.returncode:
        stream([sys.executable,'-m','pip','--python',python,'install','--index-url',
                'https://pypi.org/simple','-r',lock],source,env,log)
    subprocess.run([str(python),'-c',probe],env=env,check=True,capture_output=True,text=True)
    return python,env,{'packages':expected,'lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),
                       'kernel_changed':False,'GPU_runtime_changed':False}


def _public(row):return {k:v for k,v in row.items() if not k.startswith('_')}


def run(source_commit,token,*,base='/content',start_new=False,master_run_id=None,admission_reuse_url=None):
    if re.fullmatch('[0-9a-f]{40}',source_commit or '') is None:raise ValueError('Full source SHA required')
    base=Path(base).absolute();base.mkdir(parents=True,exist_ok=True)
    if any(p.is_symlink() for p in (base,*base.parents)):raise ValueError('Base symlink refused')
    source=checkout(source_commit,base)
    api=load(source/OLD/'colab_github.py','extended_controller_api')
    token=api._token(token);api.validate_credentials(token)
    continuation=load(source/SCOPE/'orchestration/continuation.py','extended_controller_continuation')
    session=base/'e3_extended_audit_session.json'
    local=json.loads(session.read_bytes()) if session.exists() and not start_new else None
    requested=master_run_id or (local.get('master_run_id') if local else None)
    previous=_select_run(continuation.discover_runs(token,api),requested) if not start_new else None
    known_local=(local is not None and local.get('master_run_id')==master_run_id
                 and local.get('output')==str(base/('e3-extended-audit-'+str(master_run_id)))
                 and Path(local['output']).is_dir())
    if master_run_id is not None and not start_new and previous is None and not known_local:
        raise ValueError('Requested audit master was not found; refuse to start an accidental new campaign')
    if previous and previous['stage_status']=='completed' and previous['analysis_complete']:
        print('Esta auditoría ya está terminada y guardada:',previous['run_url'],flush=True)
        return {'status':'already_completed',**_public(previous),'R3_accredited':False}
    old_source=previous['source_commit'] if previous else (local.get('source_commit') if local else source_commit)
    upgrade=None
    if old_source!=source_commit:
        _main_ancestry(api,token,old_source,source_commit)
        upgrade=_preprotocol_upgrade(source,base,local,previous,token,api)
        if upgrade is None:
            old_checkout=checkout(old_source,base)
            original=load(old_checkout/SCOPE/'orchestration/controller.py','extended_original_controller')
            return original.run(old_source,token,base=base,start_new=False,
                                master_run_id=previous['master_run_id'] if previous else requested)
        print('Admisión fallida sin protocolo ni ajustes: se conserva el master y se actualiza su recolector.',flush=True)
    master=requested or (previous['master_run_id'] if previous else stamp())
    if re.fullmatch('[0-9]{8}T[0-9]{12}Z',master) is None:raise ValueError('Invalid master run ID')
    output=base/('e3-extended-audit-'+master)
    view=base/('e3-extended-persist-view-'+master)
    statefile=base/('e3-extended-continuation-'+master+'.json')
    context={'source_commit':source_commit,'master_run_id':master,'output':str(output)}
    if local and local.get('master_run_id')==master:
        context.update({k:local[k] for k in ('admission_output','admission_run_id','admission_cache_source',
                                           'admission_reuse_url','source_upgrades') if k in local})
    if upgrade is not None:
        context['source_upgrades']=[*context.get('source_upgrades',[]),
            {'from_source_commit':old_source,'to_source_commit':source_commit,**upgrade}]
    if admission_reuse_url is not None:
        if not isinstance(admission_reuse_url,str) or not admission_reuse_url.strip():
            raise ValueError('An explicit nonempty GitHub admission URL is required for cache reuse')
        context['admission_reuse_url']=admission_reuse_url.strip()
    atomic(session,canonical(context))
    if previous and (not (output/'protocol.json').exists() or not statefile.exists()):
        print('Restaurando auditoría y checkpoints desde GitHub:',previous['run_url'],flush=True)
        continuation.restore(output,token,source_commit,previous['protocol_sha256'],api,
                             master_run_id=master,statefile=statefile)
    output.mkdir(parents=True,exist_ok=True)
    protocol_api=load(source/SCOPE/'orchestration/protocol.py','extended_controller_protocol')
    protocol_path=output/'protocol.json';freeze_path=output/'freeze_receipt.json'
    if not protocol_path.exists():
        admission=load(source/SCOPE/'admission/colab.py','extended_controller_admission')
        admission_run_id=context.get('admission_run_id',master)
        admitted=Path(context.get('admission_output',base/('e3-extended-admission-'+master)))
        old_receipt=admitted/'stage_receipt.json'
        if old_receipt.exists() and json.loads(old_receipt.read_bytes()).get('datasets_admitted')!=3:
            # A completed failed admission remains immutable in its own branch.
            # A later network retry is a NEW physical-admission attempt, never
            # an overwrite or a second logical market-training campaign.
            context['admission_cache_source']=str(admitted)
            admission_run_id=stamp();admitted=base/('e3-extended-admission-'+admission_run_id)
        context.update(admission_output=str(admitted),admission_run_id=admission_run_id)
        atomic(session,canonical(context))
        if not (admitted/'stage_receipt.json').exists():
            cached=context.get('admission_cache_source')
            reuse=context.get('admission_reuse_url')
            cache_provenance=admitted/'cache_reuse.json'
            if cache_provenance.is_file():
                proof=json.loads(cache_provenance.read_bytes())
                if proof.get('kind')!='e3_admission_cache_reuse' or proof.get('source_commit')!=source_commit:
                    raise ValueError('Preserved admission cache belongs to another source or scope')
                # The collector may have extended download_receipts.json since
                # seeding. Its official checksums are revalidated by run_stage;
                # never reseed the old receipt over newer local progress.
            elif (cached and Path(cached).is_dir()) or reuse:
                admission.seed_cache(source,admitted,source_commit,
                    previous_output=cached,reuse_url=reuse,token=token,api=api)
        receipt=admission.run_stage(source,admitted,source_commit)
        storage=load(source/'experiments/e3_cierre_integral_20261118/orchestration/github_store.py','extended_controller_admission_store')
        saved=storage.save(admitted,token,source_commit,admission_run_id,api,profile='cpu')
        print('Admisión guardada:',saved['run_url'],flush=True)
        if receipt['datasets_admitted']!=3:
            return {'status':'blocked_at_physical_admission','datasets_admitted':receipt['datasets_admitted'],
                    'admission':saved,'new_real_data_fits':0,'R3_accredited':False}
        for p in (admitted/'data').rglob('*'):
            if p.is_file() and (p.name=='admission.json' or p.relative_to(admitted/'data').parts[0] in {'datasets','sources'}):
                _copy_once(p,output/'admission'/p.relative_to(admitted/'data'))
        atomic(output/'admission_storage.json',canonical(saved))
        if context.get('source_upgrades'):
            atomic(output/'preprotocol_source_upgrades.json',canonical(context['source_upgrades']))
    runtime=load(source/OLD/'colab_runtime.py','extended_controller_runtime')
    gpu_python,gpu_env,runtime_receipt=runtime.prepare_runtime(base/'e3-preflight-gpu-runtime-v1')
    # Re-sanitize even an injected/setup environment before any scientific process.
    for key in list(gpu_env):
        if re.search(r'(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD|CREDENTIAL)(?:_|$)',key.upper()):gpu_env.pop(key,None)
    actual_profile=_gpu_profile(source,gpu_python,gpu_env)
    if not protocol_path.exists():
        p=protocol_api.build(output/'admission/admission.json',source_commit)
        p['compute_profile']=actual_profile
        atomic(protocol_path,protocol_api.canonical(p))
    else:p=json.loads(protocol_path.read_bytes())
    protocol_api.verify_compute_profile(p,actual_profile)
    print('Verificando el paquete fuente inmutable; una caché incompleta se conserva aparte.',flush=True)
    package,package_receipt=prepare_package(source,base,gpu_python,gpu_env)
    if not (output/'immutable_package.json').exists():
        atomic(output/'immutable_package.json',canonical(package_receipt))
    if not freeze_path.exists():
        atomic(freeze_path,canonical(protocol_api.freeze(protocol_path,source_commit,master,token,api)))
    protocol_api.verify_receipt(protocol_path,freeze_path,source_commit)
    if not (output/'isolated_runtime.json').exists():atomic(output/'isolated_runtime.json',canonical(runtime_receipt))
    old_gpu=load(source/'experiments/e3_cierre_integral_20261118/campaign/colab.py','extended_controller_npz')
    npz=old_gpu.admit_npz(base)
    history=load(source/SCOPE/'admission/d00_history.py','extended_controller_btc_history')
    history_paths,metadata=history.acquire(output/'inputs')
    atomic(output/'D00_history_sources.json',canonical(metadata))
    latest=None

    def persist(finished=False):
        nonlocal latest
        stable_view(output,view,finished=finished)
        latest=continuation.save_changed(view,token,source_commit,master,
            hashlib.sha256(protocol_path.read_bytes()).hexdigest(),api,statefile=statefile)
        url='https://github.com/'+REPOSITORY+'/tree/e3-extended-audit-'+latest['head_part']+'/runs/e3_extended_audit/'+latest['head_part']
        print('Evidencia verificada en GitHub:',url,flush=True)
        return url

    def status(value,analysis=False):
        atomic(output/'controller_status.json',canonical({'stage_status':value,
            'analysis_complete':analysis,'source_commit':source_commit,'master_run_id':master,
            'R1_accredited':False,'R3_accredited':False}))

    # This verified write happens BEFORE the first market model optimization.
    status('running');persist()
    command=[gpu_python,source/SCOPE/'evaluation/run.py','--output',output,
        '--source-commit',source_commit,'--admission',output/'admission/admission.json',
        '--protocol',protocol_path,'--freeze-receipt',freeze_path,'--npz',npz,
        '--btc-history',*history_paths,'--device','cuda:0','--package-path',package]
    try:
        print('Comienza la auditoría real: 19 sistemas, cuatro conjuntos y diez semillas neuronales.',flush=True)
        print('Presupuesto máximo por ajuste: 120 s; los checkpoints se guardan cada tres trabajos.',flush=True)
        stream(command,source,gpu_env,output/'worker.log',lambda:persist())
        evaluation=json.loads((output/'evaluation_receipt.json').read_bytes())
        if evaluation.get('status')!='completed':raise RuntimeError('Incomplete model family; no successful audit claim')
        persist(finished=True)
        analysis_python,analysis_env,analysis_receipt=analysis_runtime(source,base,output/'analysis_setup.log')
        atomic(output/'analysis_runtime.json',canonical(analysis_receipt))
        args=[analysis_python,source/SCOPE/'robustness/analyze.py','--panel',output/'panel_manifest.json',
              '--protocol',protocol_path,'--output',output/'analysis','--run-manifest',output/'run_manifest.json',
              '--resources',output/'resources.csv','--learning-panel',output/'sample_curve_predictions.csv',
              '--learning-index',output/'sample_curve_jobs.json']
        if (output/'HPO_sensitivity.csv').exists():args.extend(['--hpo-sensitivity',output/'HPO_sensitivity.csv'])
        stream(args,source,analysis_env,output/'analysis.log')
        stream([analysis_python,source/SCOPE/'robustness/plots.py','--analysis-dir',output/'analysis',
                '--output',output/'figures'],source,analysis_env,output/'figures.log')
        status('completed',True);url=persist(finished=True)
        return {'status':'completed','source_commit':source_commit,'master_run_id':master,
                'output':str(output),'run_url':url,'storage':latest,'R1_accredited':False,'R3_accredited':False}
    except BaseException as error:
        status('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed')
        atomic(output/'controller_failure.json',canonical({'error_type':type(error).__name__,
            'error':str(error),'local_output':str(output),'source_commit':source_commit,
            'model_state_preserved':True,'R3_accredited':False}))
        try:persist(finished=True)
        except Exception:
            print('Guardado pendiente. Se conservaron los archivos locales y el estado de continuación:',output,flush=True)
        raise
