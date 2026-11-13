"""Rebuild a receipt-bound clean environment after VM loss, without fitting.

The historical runtime receipt keeps its original bytes and meaning. A separate
current-boot admission records the actual fresh installation and synthetic probe.
The caller owns and charges the encompassing durable setup reservation.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,re,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone

ENGINE='1e8745497b484518f766b6da5c19be8362d9add7'
PIN='ae1a2d4f96807a20ea92c313297a5c09512a4817'
CONTROLLER_SHA='5d8ce059f96f86a35b7e2d2530e3fb776827240fcb29df920b4d05374f139fe9'
RUNTIME_SHA='54c175598ac1e32d391e5ca8160832e96fb689bdaf3897835567b64729b57f5a'
LOCK_SHA='6d02a90c8551905126417f6c92c5248b66d249c8b5930babb96c7f60e6f93716'
ACCEPTANCE_SHA='05a9f5a32c5abd76f67e3093b98c14989d1082df5bbbdcc2c5decc8e6f2a2134'
REPOSITORY='https://github.com/jassan-bit/DeepLearning_202630.git'
DEVELOPMENT={
 'experiments/dl_r2_cv_20261104/protocol.json':'e8da0f0098acd24a58664b83d0ba499b43aa9d597a36c52da6b3325c2c3c7bec',
 'results/eda_development/daily_market_development.csv':'bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171',
 'results/eda_development/daily_volatility_development.csv':'20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a',
}
SPARSE=('/scripts/','/experiments/','/requirements*.txt','/data/manifests/','/book_e2/_static/',
        '/results/filipe_multihistory_20261104/test/raw_test_audit.json')

def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def regular(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('Fresh runtime paths must be regular')
    return path
def prepared_python(path):
    """Accept the normal venv executable link, without relaxing artifact paths.

    Keep its lexical venv path: resolving it before execution would discard the
    interpreter's pyvenv.cfg and therefore change its installed package profile.
    """
    path=Path(path).absolute();regular(path.parent)
    try:target=path.resolve(strict=True)
    except (OSError,RuntimeError) as error:raise ValueError('Prepared base Python executable is absent or invalid') from error
    regular(target)
    if not target.is_file() or not os.access(target,os.X_OK):
        raise ValueError('Prepared base Python target must be a regular executable file')
    return path
def once(path,raw):
    path=regular(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Existing runtime bytes differ; preserved: '+str(path))
    else:
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def select_runtime(old,posthoc):
    """Bind the original runtime to an actually completed job and journal charge."""
    posthoc=regular(posthoc);ledger=regular(posthoc/'budget/posthoc_budget.jsonl');raw=ledger.read_bytes()
    if not raw.endswith(b'\n'):raise ValueError('Complete restored journal required')
    rows=[json.loads(x) for x in raw.splitlines()];previous=old.ZERO
    for i,row in enumerate(rows):
        old.checked(row)
        if row['sequence']!=i or row['previous_sha256']!=previous:raise ValueError('Restored budget chain differs')
        previous=row['sha256']
    setup=[r for r in rows if r['kind']=='charged' and r.get('status')=='completed'
           and r.get('job_id','').startswith('clean-setup-')
           and r.get('evidence',{}).get('receipt_sha256')==RUNTIME_SHA]
    if not setup:raise ValueError('No completed setup charge binds the historical runtime receipt')
    admitted=[]
    for path in sorted((posthoc/'reproduction/jobs').glob('*/worker_spec.json')):
        path=regular(path);result_path=path.parent/'result.json'
        if not result_path.exists():continue
        spec=json.loads(path.read_bytes());result_raw=regular(result_path).read_bytes();result=json.loads(result_raw)
        if result.get('status')!='completed':continue
        if (result.get('worker_spec_sha256')!=old.digest(spec) or result.get('attempt_id')!=spec.get('job_id')
                or spec.get('controller_source_sha256')!=CONTROLLER_SHA
                or spec.get('configuration_commit')!=PIN or result.get('configuration_commit')!=PIN
                or result.get('new_clean_environment')!=spec.get('clean_runtime')):
            raise ValueError('Completed result does not bind its original worker/runtime specification')
        charges=[r for r in rows if r['kind']=='charged' and r.get('status')=='completed'
                 and r.get('job_id')==spec['job_id'] and r.get('evidence',{}).get('result_sha256')==sha(result_raw)]
        if len(charges)!=1:raise ValueError('Completed worker result lacks its exact charged journal binding')
        full=spec['clean_runtime'];receipt={k:v for k,v in full.items() if not k.startswith('synthetic_acceptance_')}
        receipt_raw=canonical(receipt)+b'\n'
        if sha(receipt_raw)!=RUNTIME_SHA or full.get('synthetic_acceptance_sha256')!=ACCEPTANCE_SHA:
            raise ValueError('Historical runtime or synthetic acceptance bytes differ')
        admitted.append((full,receipt_raw,{'worker_spec_sha256':sha(path.read_bytes()),'result_sha256':sha(result_raw),
                                         'job_id':spec['job_id'],'setup_charge_sha256':setup[-1]['sha256']}))
    if not admitted:raise ValueError('A completed restored worker is required to reconstruct the clean runtime')
    if any(v[0]!=admitted[0][0] for v in admitted):raise ValueError('Completed workers name different runtime receipts')
    return admitted[0]

def _remaining(deadline,cap=None):
    if not isinstance(deadline,(int,float)) or not math.isfinite(deadline):raise ValueError('Finite setup deadline required')
    value=deadline-time.monotonic()
    if value<=0:raise TimeoutError('Caller clean-runtime restoration reservation reached its deadline')
    return value if cap is None else min(value,cap)

def _git_state(source,env,deadline):
    def query(args):return subprocess.check_output(['git','-C',str(source),*args],env=env,text=True,timeout=_remaining(deadline,20)).strip()
    return {'head':query(['rev-parse','HEAD']),'remote':query(['remote','get-url','origin']),
            'dirty':query(['status','--porcelain','--untracked-files=no'])}

def _probe(old,python,kind,target,env,deadline,log):
    if kind=='profile':
        expression='{"python":list(sys.version_info[:3]),"packages":{d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}}'
        imports='import json,sys,importlib.metadata as m;'
    else:
        expression='{"python":sys.version,"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"total_memory_bytes":torch.cuda.get_device_properties(0).total_memory}'
        imports='import json,sys,torch;'
    command=imports+'from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps('+expression+'))'
    old.run_bounded([python,'-I','-c',command,target],env=env,timeout=_remaining(deadline,90),log=log)
    return json.loads(regular(target).read_bytes())

def _boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def restore_clean_runtime(old_module,original_output,posthoc_output,environment_adapter_root,existing_python,deadline):
    """Rebuild identical runtime bytes/profile and a fresh current-boot admission."""
    old=old_module;posthoc=regular(posthoc_output);original_output=regular(original_output)
    adapter=regular(environment_adapter_root);existing_python=prepared_python(existing_python)
    if old.file_sha(Path(old.__file__))!=CONTROLLER_SHA:raise ValueError('Original posthoc controller bytes changed')
    full,receipt_raw,binding=select_runtime(old,posthoc);receipt=json.loads(receipt_raw)
    clean=regular(posthoc/'reproduction/clean');source=regular(receipt['clean_source']);python=regular(receipt['clean_python'])
    if source!=clean/'source' or python!=clean/'venv/bin/python' or Path(full['synthetic_acceptance_path'])!=clean/'synthetic_acceptance.json':
        raise ValueError('Historical runtime paths do not match the restored campaign')
    profile=receipt['package_profile']
    if profile.get('python')!=[3,12,12] or profile.get('packages',{}).get('torch')!='2.8.0+cu126':
        raise ValueError('Pinned clean package profile differs')
    lines=[]
    for name,version in sorted(profile['packages'].items()):
        if not re.fullmatch('[A-Za-z0-9_.-]+',name) or not re.fullmatch('[A-Za-z0-9_.+!-]+',version):raise ValueError('Exact package lock entries required')
        lines.append(name+'=='+version)
    lock_raw=('\n'.join(lines)+'\n').encode()
    if sha(lock_raw)!=LOCK_SHA or receipt.get('package_lock_sha256')!=LOCK_SHA:raise ValueError('Historical package lock differs')
    bootstrap=regular(adapter/'scripts/e2_posthoc_clean_worker_bootstrap.py')
    if old.file_sha(bootstrap)!=receipt['environment_adapter_bootstrap_sha256']:
        raise ValueError('Exact historical bootstrap bytes are required')
    if old.file_sha(adapter/'scripts/e2_posthoc_environment_adapter.py')!=receipt['environment_adapter_sha256']:
        raise ValueError('Exact historical environment adapter bytes are required')
    clean.mkdir(parents=True,exist_ok=True);lock=clean/'environment.lock.txt';once(lock,lock_raw)
    env=old.clean_environment();log=clean/'fresh_environment_install.log';checkout_log=clean/'fresh_checkout.log'
    boot=_boot_id()
    if re.fullmatch('[0-9a-f-]{36}',boot) is None:raise ValueError('Fresh runtime boot identity is unavailable')
    probes=clean/'fresh_runtime_probes'/boot;probes.mkdir(parents=True,exist_ok=True)
    base_profile=_probe(old,existing_python,'profile',probes/'prepared_packages.json',env,deadline,log)
    if base_profile.get('python')!=[3,12,12] or base_profile.get('packages',{}).get('torch')!='2.8.0+cu126':
        raise ValueError('Prepared base Python3.12.12/Torch2.8cu126 profile differs')
    base_runtime=_probe(old,existing_python,'runtime',probes/'prepared_runtime.json',env,deadline,log)
    if base_runtime!=receipt['runtime']:raise ValueError('Prepared base A100 runtime differs from the historical runtime')
    created_source=not source.exists()
    if created_source:
        for command in (['git','init','--quiet',source],['git','-C',source,'remote','add','origin',REPOSITORY],
                        ['git','-C',source,'fetch','--depth','1','origin',ENGINE],
                        ['git','-C',source,'sparse-checkout','init','--no-cone']):
            old.run_bounded(command,env=env,timeout=_remaining(deadline,180),log=checkout_log)
        (source/'.git/info/sparse-checkout').write_text('\n'.join(SPARSE)+'\n')
        old.run_bounded(['git','-C',source,'checkout','--detach',ENGINE],env=env,timeout=_remaining(deadline,180),log=checkout_log)
    state=_git_state(source,env,deadline)
    if state!={'head':ENGINE,'remote':REPOSITORY,'dirty':''} or (source/'.git/objects/info/alternates').exists():
        raise ValueError('Restored source must be a clean independent exact-commit checkout')
    original_source=regular(original_output.parent/('reduced-source-'+ENGINE))
    for relative,expected in DEVELOPMENT.items():
        raw=regular(original_source/relative).read_bytes()
        if sha(raw)!=expected:raise ValueError('Original development input differs')
        once(source/relative,raw)
    created_venv=not python.exists()
    if created_venv:
        old.run_bounded([existing_python,'-I','-m','venv','--copies',clean/'venv'],env=env,timeout=_remaining(deadline,180),log=log)
    cfg=regular(clean/'venv/pyvenv.cfg').read_text().lower()
    if 'include-system-site-packages = false' not in cfg:raise ValueError('Fresh runtime must not inherit system packages')
    install_packages=created_venv or not (clean/'clean_runtime.json').exists()
    if install_packages:
        if any((clean/'fresh_runtime_admissions').rglob('admission.json')):
            raise ValueError('Existing runtime admission without its historical receipt; preserved')
        # The base setup uses pip; retaining its default HTTP cache avoids
        # downloading the same CUDA wheels again into a second cache format.
        old.run_bounded([python,'-I','-m','pip','install','--disable-pip-version-check','--no-deps','--requirement',lock,
                         '--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cu126'],
                        env=env,timeout=_remaining(deadline-5),log=log)
    observed_profile=_probe(old,python,'profile',probes/'packages.json',env,deadline,log)
    if observed_profile!=profile:raise ValueError('Actual clean packages differ from the historical worker profile')
    observed_runtime=_probe(old,python,'runtime',probes/'runtime.json',env,deadline,log)
    if observed_runtime!=receipt['runtime']:raise ValueError('Actual clean A100 runtime differs from the historical runtime')
    once(clean/'clean_runtime.json',receipt_raw)
    acceptance=clean/'synthetic_acceptance.json'
    if acceptance.exists() and old.file_sha(acceptance)!=ACCEPTANCE_SHA:raise ValueError('Existing synthetic acceptance differs; preserved')
    old.run_bounded([python,bootstrap,'--preflight','--clean-runtime',clean/'clean_runtime.json','--original-output',original_output,
                     '--old-controller',Path(old.__file__),'--old-controller-sha256',CONTROLLER_SHA],
                    env=env,timeout=_remaining(deadline),log=clean/'fresh_synthetic_acceptance.log')
    if old.file_sha(acceptance)!=ACCEPTANCE_SHA:raise ValueError('Fresh synthetic acceptance is not byte-identical to its original binding')
    old.verify_clean_runtime(full)
    admission={'version':'e2-fresh-clean-runtime-admission-v1','status':'verified','observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'boot_id':boot,'historical_runtime_receipt_sha256':RUNTIME_SHA,'historical_package_lock_sha256':LOCK_SHA,
        'synthetic_acceptance_sha256':ACCEPTANCE_SHA,'fresh_synthetic_probe_executed':True,
        'new_source_checkout_created_this_call':created_source,'new_venv_created_this_call':created_venv,
        'actual_package_install_method':'pip_exact_lock_official_indexes_with_default_cache' if install_packages else 'existing_exact_environment_verified',
        'prepared_python':str(existing_python),'observed_prepared_package_profile':base_profile,'observed_prepared_runtime':base_runtime,
        'observed_package_profile':observed_profile,'observed_runtime':observed_runtime,'completed_worker_binding':binding,
        'helper_sha256':old.file_sha(Path(__file__)),'historical_receipt_relabelled_as_new':False,
        'original_benchmark_modified':False,'optimizer_steps':0,'training_performed':False,'ledger_modified_here':False}
    path=clean/'fresh_runtime_admissions'/boot/'admission.json'
    if path.exists():
        prior=json.loads(path.read_bytes())
        for key in ('boot_id','historical_runtime_receipt_sha256','synthetic_acceptance_sha256','observed_package_profile','observed_runtime'):
            if prior.get(key)!=admission[key]:raise ValueError('Existing current-boot admission differs; preserved')
    else:once(path,canonical(admission)+b'\n')
    print(json.dumps({'event':'fresh_clean_runtime_restored','historical_runtime_receipt_sha256':RUNTIME_SHA,
                      'synthetic_acceptance_sha256':ACCEPTANCE_SHA,'training_started':False,'admission':str(path)}),flush=True)
    return dict(full)

def load_old_controller(path,original_output):
    path=regular(path);original_output=regular(original_output)
    if sha(path.read_bytes())!=CONTROLLER_SHA:raise ValueError('Original posthoc controller bytes changed before import')
    engine=original_output.parent/('reduced-source-'+ENGINE)/'scripts'
    if not engine.is_dir():raise ValueError('Restore the exact original engine before its clean runtime')
    sys.path[:0]=[str(path.parent),str(engine)]
    spec=importlib.util.spec_from_file_location('e2_receipt_bound_original_controller',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-controller',type=Path,required=True)
    parser.add_argument('--original-output',type=Path,required=True)
    parser.add_argument('--posthoc-output',type=Path,required=True)
    parser.add_argument('--environment-adapter-root',type=Path,required=True)
    parser.add_argument('--existing-python',type=Path,required=True)
    parser.add_argument('--deadline',type=float,required=True)
    args=parser.parse_args(argv);_remaining(args.deadline)
    old=load_old_controller(args.old_controller,args.original_output)
    restore_clean_runtime(old,args.original_output,args.posthoc_output,args.environment_adapter_root,args.existing_python,args.deadline)
    return 0

if __name__=='__main__':raise SystemExit(main())
