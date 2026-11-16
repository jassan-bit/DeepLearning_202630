"""Traceable E2 analytical replay; omitted training/gates keep the result partial.

Check: python scripts/reproduce_e2_campaign.py --check
Replay: python scripts/reproduce_e2_campaign.py --replay --archives-dir PATH
No Git, network, fitting, automatic installation or historical-file mutation.
A partial result intentionally exits 2; a corrupt/failed stage exits 1.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'book_e2/_static/e2_pending_closure_20261116'
SOURCE = 'ad8f163c1bbed55348c4af3d3c71b71f57bc7c55'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def checked_file(root, record):
    relative = Path(record['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe bound input path')
    path = root/relative
    if path.is_symlink() or not path.is_file(): raise ValueError('Missing regular input: '+str(relative))
    if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
        raise ValueError('Input bytes changed: '+str(relative))
    return path


def requirements(text):
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(('#','--')): continue
        match = re.match(r'^([\w.-]+)==([^\s;]+)', line)
        if match: result[match[1]] = match[2]
    return result


def dependencies(python, expected):
    code = 'import importlib.metadata as m,json,sys;d={};\nfor n in json.loads(sys.argv[1]):\n try:d[n]=m.version(n)\n except m.PackageNotFoundError:d[n]=None\nprint(json.dumps(d))'
    try:
        process = subprocess.run([str(python), '-I', '-c', code, json.dumps(list(expected))], capture_output=True, text=True, timeout=30)
        if process.returncode: return {'ready':False, 'error':process.stderr[-1000:]}
        observed = json.loads(process.stdout)
        mismatch = {k:{'expected':v,'observed':observed.get(k)} for k,v in expected.items() if observed.get(k)!=v}
        return {'ready':not mismatch, 'observed':observed, 'mismatches':mismatch}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {'ready':False, 'error':str(error)}


def safe_extract(archive, destination):
    with zipfile.ZipFile(archive) as source:
        for item in source.infolist():
            path = Path(item.filename)
            mode = item.external_attr >> 16
            if path.is_absolute() or '..' in path.parts or (mode & 0o170000)==0o120000:
                raise ValueError('Unsafe archive member: '+item.filename)
        source.extractall(destination)


def verify_package(package):
    manifest = json.loads((package/'source_manifest.json').read_text())
    if manifest['source_commit'] != SOURCE: raise ValueError('Wrong scientific snapshot')
    for row in manifest['files']: checked_file(package, row)


def finish_status(stages, blockers):
    failed = any(row['status']=='failed' for row in stages)
    incomplete = blockers or any(row['status'] in {'blocked','not_executed','ready'} for row in stages)
    return 'failed' if failed else 'partial' if incomplete else 'completed_analytical_scope_only'


def stage_definitions():
    stages = []
    for n in range(1,13):
        key = f'D{n:02}'
        kind = ('implementation_fixture_forward' if key=='D02' else 'arithmetic_reconstruction' if key in {'D03','D06','D08','D10','D12'} else 'book_extraction' if key=='D01' else 'recorded_evidence_validation')
        stages.append({'id':key+'_default','package':key,'script':'run.py','args':[], 'kind':kind, 'environment':'model' if key=='D02' else 'stdlib'})
    stages.extend([
        {'id':'development_reaudit','package':'D04','script':'audit_data/run.py','args':[], 'kind':'real_development_data_reaudit','environment':'stdlib'},
        {'id':'effects_RQ2_CPU_statistics','package':'D11','script':'reanalysis/run.py','args':[], 'kind':'numerical_replay_from_saved_forecasts_and_CPU_timers','environment':'corrections'},
        {'id':'HPO_metadata','package':'D07','script':'hpo_reanalysis/run.py','args':[], 'kind':'numerical_metadata_reconciliation_and_figures','environment':'hpo'},
        {'id':'primary_inference','package':'D06','script':'run.py','args':['--full-primary','--draws','20000'], 'kind':'primary_bootstrap_t_signflip_from_saved_forecasts','environment':'analysis'},
        {'id':'conditional_calibration_error_analysis','package':'D08','script':'run.py','args':['--full-uncertainty'], 'kind':'conditional_calibration_and_error_bootstrap_from_saved_forecasts','environment':'analysis'},
        {'id':'book_rebuild','package':'D01','script':'run.py','args':['--build'], 'kind':'rebuild_book_from_included_sources','environment':'book'},
    ])
    return stages


def load_budget(root):
    script = root/'scripts/e2_corrections_budget.py'
    spec = importlib.util.spec_from_file_location('e2_replay_budget',script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if not module.LEDGER.is_file(): raise ValueError('Existing additive campaign ledger required; never reset or create a replacement budget')
    return module


def run(args):
    manifest = json.loads(args.inputs_manifest.read_text())
    if manifest['scientific_source_commit'] != SOURCE: raise ValueError('Wrong input catalogue snapshot')
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output/'reproduction_status.json'
    if args.replay and status_path.exists(): raise ValueError('Replay requires a fresh output directory')
    evidence_errors = []
    for item in manifest['evidence_files']:
        try: checked_file(args.root, item)
        except ValueError as error: evidence_errors.append(str(error))
    archives = {}
    archive_states = []
    for item in manifest['archives']:
        record = {'id':item['id'],'file':item['file'],'expected_sha256':item['sha256']}
        path = args.archives_dir/item['file']
        if not path.is_file(): record['status']='missing'
        elif digest(path)!=item['sha256'] or path.stat().st_size!=item['bytes']: record['status']='changed'
        else: record['status']='verified'; archives[item['id']]=path
        archive_states.append(record)
    envs = {
        'stdlib': {'ready':True,'observed':{'python':sys.version.split()[0]}},
        'model': dependencies(args.model_python, {'torch':'2.8.0+cpu','numpy':'2.3.5','pandas':'2.2.3','einops':'0.8.1'}),
        'corrections': dependencies(args.python, {'numpy':'2.3.5','scipy':'1.17.0'}),
        'hpo': dependencies(args.python, {'numpy':'2.3.5','matplotlib':'3.10.8'}),
        'analysis': dependencies(args.analysis_python, {'numpy':'2.3.5','pandas':'2.2.3','scipy':'1.16.3','statsmodels':'0.14.5','matplotlib':'3.10.8'}),
        'book': dependencies(args.book_python, {'jupyter-book':'1.0.4.post1'}),
    }
    blockers = list(manifest['mandatory_unexecuted_stages'])
    stages = stage_definitions()
    for stage in stages:
        problems = list(evidence_errors)
        if stage['package'] not in archives: problems.append('Verified independent archive required: '+stage['package'])
        if not envs[stage['environment']]['ready']: problems.append('Dependency versions unavailable: '+stage['environment'])
        stage['status']='blocked' if problems else 'ready'
        stage['blockers']=problems
    state = {'version':'e2-traceable-reproduction-v1','mode':'replay' if args.replay else 'check',
             'at_utc':datetime.now(timezone.utc).isoformat(),'scientific_source_commit':SOURCE,
             'orchestrator_sha256':digest(Path(__file__)),'input_catalogue_sha256':digest(args.inputs_manifest),
             'archives':archive_states,'environments':envs,'stages':stages,'mandatory_unexecuted_stages':blockers,
             'HPO_cost_scopes':manifest['HPO_cost_scopes'],'original_full_training_repeated':False,
             'full_campaign_passed':False,'NAS1000':'not_applicable_defined_current_bodies',
             'interpretation':'Analytical reconstruction is distinct from training and from passing the original CUDA gate. A partial result exits 2.'}
    budget = None; reservation = None; started = None; returncode = 2
    try:
        if args.replay:
            if evidence_errors or any(x['status']=='changed' for x in archive_states): raise ValueError('Changed/missing immutable evidence; replay rejected before computation')
            budget = load_budget(args.root)
            reservation = budget.reserve('e2_unified_analytical_replay', args.cap_seconds)
            started = time.monotonic()
            cache = {}
            for stage in stages:
                if stage['status']!='ready': continue
                remaining = args.cap_seconds-(time.monotonic()-started)
                if remaining<=0: stage['status']='blocked'; stage['blockers']=['Shared replay time cap exhausted']; continue
                try:
                    key = stage['package']
                    if key not in cache:
                        destination = args.output/'extracted'/key
                        safe_extract(archives[key],destination); cache[key]=destination/key; verify_package(cache[key])
                    package = cache[key]
                    python = {'model':args.model_python,'analysis':args.analysis_python,'book':args.book_python}.get(stage['environment'],args.python)
                    destination = args.output/'stages'/stage['id']
                    command = [str(python),'-I',str(package/stage['script']),*stage['args'],'--output',str(destination)]
                    stage['command']=command
                    log = args.output/'logs'/(stage['id']+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
                    tick=time.monotonic()
                    with log.open('w') as stream:
                        result=subprocess.run(command,cwd=package,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
                    stage.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode,wall_seconds=time.monotonic()-tick,log_sha256=digest(log))
                except (OSError,ValueError,subprocess.TimeoutExpired) as error:
                    stage.update(status='failed',error=str(error))
                state['status']=finish_status(stages,blockers);write(status_path,state)
        state['status']=finish_status(stages,blockers)
        if evidence_errors or any(x['status']=='changed' for x in archive_states): state['status']='failed'
        returncode = 1 if state['status']=='failed' else 2 if state['status']=='partial' else 0
    finally:
        if reservation:
            budget.finish(reservation,time.monotonic()-started,returncode)
            with budget.locked_ledger(budget.LEDGER) as (rows,_): state['additive_compute_budget']=budget.summarize(rows)
        state.setdefault('status','failed');write(status_path,state)
    print(json.dumps({'status':state['status'],'completed_stages':sum(x['status']=='completed' for x in stages),'blocked_stages':sum(x['status']=='blocked' for x in stages),'full_campaign_passed':False,'status_file':str(status_path)},ensure_ascii=False))
    return returncode


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--check',action='store_true');mode.add_argument('--replay',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--inputs-manifest',type=Path,default=BASE/'reproduction_inputs.json')
    parser.add_argument('--archives-dir',type=Path,default=ROOT/'book_e2/_static/entregables_independientes_v2_20261115')
    parser.add_argument('--output',type=Path,default=BASE)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--analysis-python',type=Path,default=Path(sys.executable))
    parser.add_argument('--model-python',type=Path,default=Path(sys.executable))
    parser.add_argument('--book-python',type=Path,default=Path(sys.executable))
    parser.add_argument('--cap-seconds',type=float,default=600)
    args=parser.parse_args(argv)
    for name in ['root','inputs_manifest','archives_dir','output','python','analysis_python','model_python','book_python']:
        setattr(args,name,getattr(args,name).absolute())
    if not 0<args.cap_seconds<=3600: parser.error('Use a finite cap in (0,3600]; it reserves from the existing campaign ledger')
    try: return run(args)
    except (OSError,ValueError,KeyError,zipfile.BadZipFile) as error:
        print('Reproduction rejected: '+str(error),file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
