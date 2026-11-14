"""Execute an unchanged posthoc controller with fresh control reads only."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
REG='experiments/e2_prelaunch_recovery_20261113/registration.json'
BASE_COMMIT='b2199a8bc5df0e392430343b7891b11bedcd5306'
WORKSPACE=Path('/content/entregable2_gpu_workspace')

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def validate(root):
    root=Path(root);record=json.loads((root/REG).read_bytes())
    for path,expected in record['files_sha256'].items():
        p=root/path
        if any(x.is_symlink() for x in (p,*p.parents)) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Recovery source changed: '+path)
    base=WORKSPACE/('fresh-session-recovery-'+BASE_COMMIT)
    for path,expected in record['existing_launcher_files_sha256'].items():
        p=base/path
        if any(x.is_symlink() for x in (p,*p.parents)) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Keep the existing verified Colab session and launcher: '+path)
    return record,base

def run(root,expected,token):
    _,base=validate(root);fresh=load(base/'scripts/e2_posthoc_fresh_resume.py','e2_preserved_prelaunch_launcher')
    fresh.install_shutdown_handler()
    consistency=load(Path(root)/'scripts/e2_posthoc_control_consistency.py','e2_prelaunch_fresh_control_reads')
    original_load=fresh.load
    def adapted(path,name):
        module=original_load(path,name)
        if Path(path).name=='e2_posthoc_fit_reproduction.py':
            import e2_reduced_github_backup as backup
            consistency.install(backup,module)
        return module
    fresh.load=adapted
    try:return fresh.execute_continue(base,expected,token)
    finally:fresh.load=original_load

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-state',required=True)
    args=p.parse_args();token=sys.stdin.readline().strip()
    if not token or any(c.isspace() for c in token):raise ValueError('Pass the existing credential privately through STDIN')
    run(args.root,args.expected_state,token)
if __name__=='__main__':main()
