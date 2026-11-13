#!/content/entregable2_gpu_workspace/reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python
"""Continue a closed failed environment setup; never retry an admitted fit."""
from pathlib import Path
import hashlib, importlib.util, io, json, os, sys, time

ROOT=Path('/content/entregable2_gpu_workspace')
REAL_PYTHON=ROOT/'reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python'
PIN='ae1a2d4f96807a20ea92c313297a5c09512a4817'
REGISTRATION='experiments/e2_posthoc_environment_20261113/registration.json'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def bundle_root(path):
    path=Path(path).resolve()
    return path.parent if path.name=='python' else path.parents[1]

def main(args=None):
    args=list(sys.argv[1:] if args is None else args)
    if '--orchestrate' not in args:
        os.execv(str(REAL_PYTHON),[str(REAL_PYTHON),*args])
        raise AssertionError('exec returned')
    started=time.monotonic();own=bundle_root(__file__)
    reg=json.loads((own/REGISTRATION).read_bytes())
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=reg['files_sha256']['scripts/e2_posthoc_environment_resume.py']:
        raise ValueError('Executing environment interpreter bytes differ')
    for relative,expected in reg['files_sha256'].items():
        p=own/relative
        if any(x.is_symlink() for x in (p,*p.parents)) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Environment recovery registration differs: '+relative)
    transport=load(own/'scripts/e2_posthoc_transport_resume.py','e2_registered_release_transport')
    extension,destination,original_registration=transport.validate_invocation(args)
    token=sys.stdin.readline().rstrip('\r\n')
    if not token or any(c.isspace() for c in token):raise ValueError('Existing credential must arrive through STDIN')
    sys.path[:0]=[str(extension/'scripts'),str(own/'scripts')]
    old=load(Path(args[0]),'e2_unchanged_posthoc_controller')
    from e2_reduced_github_backup import GitHubBackup
    import e2_reduced_github_backup as backup_module
    backup_module.RELEASE_TAG=transport.NEW_RELEASE
    client=GitHubBackup(token,timeout=20,max_conflicts=2);client._checked=True
    remote=old.PosthocRemoteState(destination,client,extension)
    remote.restore()
    anchor=old.validate_original_anchor(transport.argument(args,'--original-output'))
    launcher=float(transport.argument(args,'--launcher-elapsed-seconds'))
    launcher+=float(os.environ.pop('E2_ENV_RECOVERY_HOST_SECONDS','0'))
    with old.PosthocLedger(destination/'budget',anchor) as budget:
        budget.set_persistence(remote)
        budget.charge_overhead(time.monotonic()-started+launcher,
                              description='environment_recovery_launcher_source_and_remote_verification')
        job='environment-recovery-'+str(time.time_ns())
        begin=time.monotonic();budget.reserve(job,240,metadata={'scope':'failed_clean_setup_archive','training_performed':False})
        try:
            # The original helper was registered and remains unchanged; its
            # release amendment is backed before any new tensor uploads.
            backup_module.RELEASE_TAG=transport.OLD_RELEASE
            transport.install_release_amendment(extension,destination,original_registration,token)
            archiver=load(own/'scripts/e2_posthoc_failed_setup_archive.py','e2_failed_setup_archive')
            archiver.archive_failed_setup(old,destination,budget,remote,reg)
        except BaseException as error:
            budget.charge(job,time.monotonic()-begin,
                          status='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed')
            raise
        else:budget.charge(job,time.monotonic()-begin,status='completed')
    resumed_at=time.monotonic()
    adapter=load(own/'scripts/e2_posthoc_environment_adapter.py','e2_registered_environment_adapter')
    adapter.install_environment_adapter(old,Path(transport.argument(args,'--original-source')),own,
                                        original_output=Path(transport.argument(args,'--original-output')))
    index=args.index('--launcher-elapsed-seconds');args[index+1]=str(time.monotonic()-resumed_at)
    sys.stdin=io.StringIO(token+'\n')
    print(json.dumps({'event':'environment_recovery_ready','original_configuration_commit':PIN,
                      'completed_frozen_diagnostics_reused':True,'scientific_source_files_modified':False,
                      'budget_reset':False,'authorized_total_hours':80,'persistent_destination':'GitHub'}),flush=True)
    return old.main(args[1:])

if __name__=='__main__':main()
