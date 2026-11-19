"""Admit the three OHLCV files in Colab, independently of Render or GPU.

AI assistance declared. This stage performs no target construction, model fit,
market inference, statistical contrast or scientific accreditation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

REPO = 'https://github.com/jassan-bit/DeepLearning_202630.git'
REPOSITORY = 'jassan-bit/DeepLearning_202630'
ADMISSION = 'experiments/e3_evaluacion_ampliada_20261119/admission'
COLLECTOR = ADMISSION+'/collect_recovery.py'
GITHUB_API = 'experiments/e3_remediacion_20261118/evaluation/colab_github.py'
GITHUB_STORE = 'experiments/e3_cierre_integral_20261118/orchestration/github_store.py'
EXPECTED_IDS = ('D01', 'D02', 'D03')
MAX_REUSE_FILE = 48*1024**2
MAX_REUSE_TOTAL = 256*1024**2
RUN_ID = r'[0-9]{8}T[0-9]{12}Z'


def _source(value):
    if re.fullmatch(r'[0-9a-f]{40}', value or '') is None:
        raise ValueError('A full immutable source SHA is required')
    return value


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def _canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False)+'\n').encode()


def _regular_directory(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink evidence or checkout directories are refused')
    return path


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _safe(name):
    path=PurePosixPath(name)
    return (bool(name) and str(path)==name and not path.is_absolute()
            and '..' not in path.parts and '\\' not in name
            and not any(ord(c)<32 for c in name))


def _cache_name(name):
    return _safe(name) and len(PurePosixPath(name).parts)>2 and PurePosixPath(name).parts[:2] in {
        ('data','responses'),('data','datasets'),('data','sources')} and not any(
            p.startswith('.partial-') for p in PurePosixPath(name).parts)


def _failed_scope(stage,report):
    """An admission retry may reuse bytes only before any scientific campaign."""
    if not isinstance(stage,dict) or not isinstance(report,dict):
        raise ValueError('Admission scope requires object receipts')
    source=_source(stage.get('source_commit'))
    if (stage.get('kind')!='e3_closure_stage_receipt' or stage.get('profile')!='cpu'
            or stage.get('stage')!='additional_dataset_admission_only'
            or stage.get('status')!='incomplete_admission'
            or stage.get('admission_report')!='data/admission.json'
            or report.get('kind')!='e3_additional_OHLCV_admission'
            or report.get('source_commit')!=source):
        raise ValueError('Only a failed physical-admission stage can seed a retry')
    for obj in (stage,report):
        if (obj.get('datasets_required')!=3 or type(obj.get('datasets_admitted')) is not int
                or not 0<=obj['datasets_admitted']<3
                or any(type(obj.get(k)) is not int or obj[k]!=0 for k in
                       ('datasets_evaluated_by_models','new_real_data_fits','new_market_predictions','new_statistical_contrasts'))
                or obj.get('target_values_computed') is not False
                or obj.get('protocol_frozen') is not False
                or obj.get('real_campaign_allowed') is not False):
            raise ValueError('Cache source already evaluated targets, models or a protocol')
    if (stage.get('R1_accredited') is not False or stage.get('R3_accredited') is not False
            or stage['datasets_admitted']!=report['datasets_admitted']):
        raise ValueError('Cache source accreditation or admission count differs')
    rows=report.get('datasets',[])
    if (not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows)
            or [r.get('id') for r in rows]!=list(EXPECTED_IDS)
            or any(type(r.get('admitted')) is not bool for r in rows)
            or sum(r['admitted'] for r in rows)!=report['datasets_admitted']):
        raise ValueError('Cache source must preserve the complete admission inventory')
    for row in rows:
        if row['admitted'] and (row.get('path')!='datasets/'+row['id']+'.csv'
                               or row.get('target_values_computed') is not False):
            raise ValueError('An admitted source file has an unexpected path or target scope')
    return source


def _checked(raw,record):
    if (not isinstance(record,dict) or type(record.get('bytes')) is not int or not 0<=record['bytes']<=MAX_REUSE_FILE
            or len(raw)!=record['bytes'] or _sha(raw)!=record.get('sha256')):
        raise ValueError('Reusable cache file differs from its byte/hash manifest')
    return raw


def _publish_new(path,raw):
    """Publish complete immutable bytes; interrupted staging is never evidence."""
    path=_regular_directory(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.partial-cache-',delete=False) as stream:
        temporary=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:
        try:os.link(temporary,path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes()!=raw:raise ValueError('Existing cache evidence differs')
    finally:temporary.unlink(missing_ok=True)


def _check_reuse_inventory(stage,report,raw_manifest,files):
    source=_failed_scope(stage,report)
    if (not isinstance(raw_manifest,dict) or raw_manifest.get('kind')!='admission_raw_file_manifest'
            or raw_manifest.get('source_commit')!=source
            or not isinstance(raw_manifest.get('files'),dict)):
        raise ValueError('Reusable cache manifest belongs to another source or scope')
    expected={'data/'+name:record for name,record in raw_manifest['files'].items()
              if _cache_name('data/'+name)}
    if not expected or set(expected)!=set(files) or len(files)>1024:
        raise ValueError('Reusable cache inventory is incomplete')
    if sum(len(raw) for raw in files.values())>MAX_REUSE_TOTAL:
        raise ValueError('Reusable cache exceeds its total byte budget')
    for name,raw in files.items():_checked(raw,expected[name])
    for row in report['datasets']:
        if row['admitted']:_checked(files['data/'+row['path']],row)
    return source


def _local_cache(previous):
    previous=_regular_directory(previous)
    stage_raw=(previous/'stage_receipt.json').read_bytes()
    report_raw=(previous/'data/admission.json').read_bytes()
    manifest_raw=(previous/'data/raw_manifest.json').read_bytes()
    manifest=json.loads(manifest_raw);files={};all_files={}
    if not isinstance(manifest,dict):raise ValueError('Raw cache manifest must be an object')
    for path in (previous/'data').rglob('*'):
        if path.is_symlink():raise ValueError('Reusable cache cannot contain symlinks')
        if path.is_file():
            name=path.relative_to(previous).as_posix()
            if path.name!='raw_manifest.json' and not path.name.startswith('.partial-'):
                all_files[path.relative_to(previous/'data').as_posix()]=path.read_bytes()
            if _cache_name(name):files[name]=path.read_bytes()
    inventory=manifest.get('files',{})
    if not isinstance(inventory,dict) or set(all_files)!=set(inventory):
        raise ValueError('Local failed stage differs from its complete raw manifest')
    for name,raw in all_files.items():
        if not _safe(name):raise ValueError('Raw cache manifest has an unsafe path')
        _checked(raw,inventory[name])
    source=_check_reuse_inventory(json.loads(stage_raw),json.loads(report_raw),manifest,files)
    return files,{'kind':'local_failed_admission','source_commit':source,'previous_output':str(previous),
        'stage_receipt_sha256':_sha(stage_raw),'admission_report_sha256':_sha(report_raw),
        'raw_manifest_sha256':_sha(manifest_raw)}


def inspect_failed_stage(previous_output):
    """Prove a local stage is preprotocol and bind its reusable source bytes."""
    previous=_regular_directory(previous_output)
    _,origin=_local_cache(previous)
    receipt=json.loads((previous/'stage_receipt.json').read_bytes())
    return {**origin,'datasets_admitted':receipt['datasets_admitted'],
            'admission_sha256':origin['admission_report_sha256'],
            'new_real_data_fits':0,'new_market_predictions':0,
            'new_statistical_contrasts':0,'datasets_evaluated_by_models':0,
            'target_values_computed':False,'protocol_frozen':False}


def _remote_cache(url,token,api):
    parsed=urlsplit(url)
    pattern='/'+REPOSITORY+'/tree/(e3-closure-cpu-('+RUN_ID+r')|[0-9a-f]{40})/runs/e3_closure_cpu/('+RUN_ID+')'
    match=re.fullmatch(pattern,parsed.path)
    if (parsed.scheme!='https' or parsed.netloc!='github.com' or parsed.query or parsed.fragment
            or match is None or (match.group(2) and match.group(2)!=match.group(3))):
        raise ValueError('Use the exact GitHub URL of a failed CPU admission stage')
    if api.REPOSITORY!=REPOSITORY:raise ValueError('Cache repository differs')
    token=api._token(token);root='/repos/'+REPOSITORY;ref=match.group(1);run_id=match.group(3)
    if ref.startswith('e3-closure-cpu-'):
        response=api._request('GET',root+'/git/ref/heads/'+ref,token)
        if response.get('ref')!='refs/heads/'+ref or response.get('object',{}).get('type')!='commit':
            raise ValueError('Cache URL must resolve to an immutable commit')
        commit_sha=_source(response['object'].get('sha'))
    else:commit_sha=_source(ref)
    commit=api._request('GET',root+'/git/commits/'+commit_sha,token)
    if commit.get('sha')!=commit_sha:raise ValueError('Cache commit identity differs')
    tree_sha=_source(commit.get('tree',{}).get('sha'))
    for part in ('runs','e3_closure_cpu',run_id):
        tree=api._request('GET',root+'/git/trees/'+tree_sha,token)
        if tree.get('sha')!=tree_sha or tree.get('truncated') is not False:
            raise ValueError('A complete cache tree is required')
        found=[r for r in tree.get('tree',[]) if r.get('path')==part]
        if len(found)!=1 or found[0].get('type')!='tree' or found[0].get('mode')!='040000':
            raise ValueError('Cache results prefix is absent or unsafe')
        tree_sha=_source(found[0].get('sha'))
    tree=api._request('GET',root+'/git/trees/'+tree_sha+'?recursive=1',token)
    if tree.get('sha')!=tree_sha or tree.get('truncated') is not False:
        raise ValueError('A complete cache results tree is required')
    rows=tree.get('tree',[])
    if len(rows)>4096 or len({r.get('path') for r in rows})!=len(rows):
        raise ValueError('Cache results tree is oversized or has duplicate paths')
    blobs={r['path']:r for r in rows if r.get('type')=='blob'}
    if any(not _safe(name) or row.get('mode')!='100644' for name,row in blobs.items()):
        raise ValueError('Cache results contain an unsafe path or mode')

    def blob(name):
        row=blobs.get(name)
        if row is None or type(row.get('size')) is not int or not 0<=row['size']<=MAX_REUSE_FILE:
            raise ValueError('A required regular cache blob is absent or too large')
        sha=_source(row.get('sha'));response=api._request('GET',root+'/git/blobs/'+sha,token)
        if response.get('sha')!=sha or response.get('encoding')!='base64':raise ValueError('Cache blob identity differs')
        try:raw=base64.b64decode(response.get('content',''),validate=False)
        except (ValueError,TypeError):raise ValueError('Invalid cache blob encoding') from None
        if (len(raw)!=row['size'] or api._git_sha(raw)!=sha
                or token.encode() in raw):raise ValueError('Cache blob hash, size or credential scan failed')
        return raw

    storage_raw=blob('storage_manifest.json');storage=json.loads(storage_raw)
    if (storage.get('kind')!='e3_closure_storage_manifest' or storage.get('profile')!='cpu'
            or storage.get('run_id')!=run_id or not isinstance(storage.get('files'),dict)):
        raise ValueError('Cache storage manifest has another scope')
    source=_source(storage.get('source_commit'))
    if [p.get('sha') for p in commit.get('parents',[])]!=[source]:
        raise ValueError('Cache results commit does not bind its declared source')
    inventory=storage['files']
    if (set(blobs)!=set(inventory)|{'storage_manifest.json'} or len(inventory)>1024
            or any(not isinstance(record,dict) or type(record.get('bytes')) is not int
                   or not 0<=record['bytes']<=MAX_REUSE_FILE for record in inventory.values())
            or sum(row['bytes'] for row in inventory.values())>MAX_REUSE_TOTAL):
        raise ValueError('Cache storage manifest does not bind its exact results tree')
    for name,record in inventory.items():
        row=blobs[name]
        if record.get('git_blob_sha')!=row['sha'] or record.get('bytes')!=row['size']:
            raise ValueError('Cache tree and storage manifest differ')
    needed=['stage_receipt.json','data/admission.json','data/raw_manifest.json']
    payload={name:_checked(blob(name),inventory[name]) for name in needed}
    stage=json.loads(payload['stage_receipt.json']);report=json.loads(payload['data/admission.json'])
    if _failed_scope(stage,report)!=source:raise ValueError('Cache stage and storage source differ')
    raw_manifest=json.loads(payload['data/raw_manifest.json'])
    if not isinstance(raw_manifest,dict):raise ValueError('Raw cache manifest must be an object')
    _checked(payload['data/admission.json'],raw_manifest.get('files',{}).get('admission.json',{}))
    files={name:_checked(blob(name),record) for name,record in inventory.items() if _cache_name(name)}
    _check_reuse_inventory(stage,report,raw_manifest,files)
    return files,{'kind':'github_failed_admission','source_commit':source,'reuse_url':url,
        'results_commit':commit_sha,'storage_manifest_sha256':_sha(storage_raw),
        'stage_receipt_sha256':_sha(payload['stage_receipt.json']),
        'admission_report_sha256':_sha(payload['data/admission.json']),
        'raw_manifest_sha256':_sha(payload['data/raw_manifest.json'])}


def seed_cache(checkout,output,source_commit,*,previous_output=None,reuse_url=None,token=None,api=None):
    """Seed a new attempt from verified bytes; never replace the failed stage."""
    source=_source(source_commit);checkout=_regular_directory(checkout);output=_regular_directory(output)
    previous=None if previous_output is None else _regular_directory(previous_output)
    if previous is not None and previous==output:raise ValueError('A retry needs a new admission directory')
    provenance_path=output/'cache_reuse.json'
    if provenance_path.is_symlink():raise ValueError('Cache provenance symlink refused')
    if provenance_path.is_file():
        previous_receipt=json.loads(provenance_path.read_bytes())
        if (not isinstance(previous_receipt,dict) or previous_receipt.get('source_commit')!=source
                or previous_receipt.get('kind')!='e3_admission_cache_reuse'):
            raise ValueError('Existing cache provenance belongs to another source')
        origin=previous_receipt.get('origin',{})
        if (previous is not None and previous.is_dir() and origin.get('previous_output')!=str(previous)):
            raise ValueError('Existing cache provenance belongs to another local stage')
        if (not (previous is not None and previous.is_dir()) and reuse_url
                and origin.get('reuse_url')!=reuse_url):
            raise ValueError('Existing cache provenance belongs to another results URL')
        for name,record in previous_receipt.get('files',{}).items():
            if not _cache_name(name):raise ValueError('Existing cache provenance contains an unsafe path')
            _checked((output/name).read_bytes(),record)
        return previous_receipt
    if (output/'stage_receipt.json').exists():raise ValueError('A completed admission stage remains immutable')
    if any((output/'data'/name).exists() for name in ('run_config.json','admission.json','raw_manifest.json')):
        raise ValueError('Cache reuse requires a new attempt, without previous final reports')
    if previous is not None and previous.is_dir():files,origin=_local_cache(previous)
    elif reuse_url:
        if api is None or token is None:raise ValueError('GitHub cache recovery requires authenticated read access')
        files,origin=_remote_cache(reuse_url,token,api)
    else:raise ValueError('A local failed admission or exact GitHub results URL is required')
    if token and any(token.encode() in raw for raw in files.values()):raise ValueError('Credentials cannot be copied into evidence')
    for name,raw in files.items():
        path=_regular_directory(output/name)
        if path.exists() and (not path.is_file() or path.read_bytes()!=raw):
            raise ValueError('Existing retry evidence differs; no cache file was replaced')
    output.mkdir(parents=True,exist_ok=True)
    for name,raw in files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():continue
        _publish_new(path,raw)
    metadata={name:{'bytes':len(raw),'sha256':_sha(raw)} for name,raw in sorted(files.items())}
    receipt={'kind':'e3_admission_cache_reuse','source_commit':source,'origin':origin,
        'files':metadata,'files_manifest_sha256':_sha(_canonical(metadata)),
        'copied_files':len(files),'copied_bytes':sum(len(raw) for raw in files.values()),
        'new_real_data_fits':0,'new_market_predictions':0,'target_values_computed':False,
        'protocol_frozen':False,'old_stage_modified':False}
    _publish_new(provenance_path,_canonical(receipt))
    return receipt


def _validate_collector_result(report, folder, source):
    """Bind any positive admission to its actual normalized file bytes."""
    if (report.get('kind') != 'e3_additional_OHLCV_admission'
            or report.get('source_commit') != source
            or report.get('datasets_required') != 3
            or report.get('datasets_evaluated_by_models') != 0
            or report.get('new_real_data_fits') != 0
            or report.get('new_market_predictions') != 0
            or report.get('target_values_computed') is not False
            or report.get('real_campaign_allowed') is not False):
        raise ValueError('Collector report has a different scientific scope')
    datasets = report.get('datasets')
    if not isinstance(datasets, list) or [r.get('id') for r in datasets] != list(EXPECTED_IDS):
        raise ValueError('The complete D01/D02/D03 admission inventory is required')
    if any(type(r.get('admitted')) is not bool for r in datasets):
        raise ValueError('Explicit admission booleans are required')
    count = sum(r['admitted'] for r in datasets)
    if type(report.get('datasets_admitted')) is not int or report['datasets_admitted'] != count:
        raise ValueError('Admission count does not match the per-file inventory')
    for row in datasets:
        if not row['admitted']:
            continue
        relative = 'datasets/'+row['id']+'.csv'
        path = folder/relative
        if (row.get('path') != relative or path.is_symlink() or not path.is_file()
                or row.get('target_values_computed') is not False):
            raise ValueError('An admitted file is absent or has an unexpected scope/path')
        raw = path.read_bytes()
        if len(raw) != row.get('bytes') or hashlib.sha256(raw).hexdigest() != row.get('sha256'):
            raise ValueError('Admitted normalized bytes differ from the collector receipt')
    return count


def run_stage(checkout, output, source_commit, *, collector=None):
    """Run only file admission; a completed stage is immutable on save retry."""
    source = _source(source_commit)
    checkout, output = _regular_directory(checkout), _regular_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output/'stage_receipt.json'
    if receipt_path.is_file():
        saved = json.loads(receipt_path.read_bytes())
        if (saved.get('source_commit') != source or saved.get('profile') != 'cpu'
                or saved.get('stage') != 'additional_dataset_admission_only'):
            raise ValueError('Existing stage belongs to another source or scope')
        print('Admisión local terminada: se conserva para reintentar su guardado.', flush=True)
        return saved
    started = datetime.now(timezone.utc).isoformat()
    collect = collector or _load(checkout/COLLECTOR, 'e3_extended_pinned_collector').run
    print('Admisión D01/D02/D03: CPU, sin Render ni entrenamiento.', flush=True)
    report = collect(output/'data', source)
    count = _validate_collector_result(report, output/'data', source)
    receipt = {'kind':'e3_closure_stage_receipt', 'profile':'cpu',
        'stage':'additional_dataset_admission_only', 'source_commit':source,
        'started_at_utc':started, 'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'all_three_files_admitted' if count == 3 else 'incomplete_admission',
        'datasets_admitted':count, 'datasets_required':3,
        'datasets_evaluated_by_models':0, 'new_real_data_fits':0,
        'new_market_predictions':0, 'new_statistical_contrasts':0,
        'target_values_computed':False, 'protocol_frozen':False,
        'real_campaign_allowed':False, 'R1_accredited':False, 'R3_accredited':False,
        'Render_checked':False, 'GPU_required':False,
        'admission_report':'data/admission.json',
        'scope':'Physical byte/calendar admission only; model evaluation and its protocol are separate.'}
    _publish_new(receipt_path,_canonical(receipt))
    print(json.dumps({'datasets_admitted':count, 'datasets_required':3,
                      'new_real_data_fits':0, 'output':str(output)}, ensure_ascii=False), flush=True)
    return receipt


def _environment():
    env = dict(os.environ)
    for key in list(env):
        if re.search(r'(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD|CREDENTIAL)(?:_|$)', key.upper()):
            env.pop(key, None)
    env.update(GIT_TERMINAL_PROMPT='0', PYTHONNOUSERSITE='1')
    return env


def checkout_source(source_commit, base):
    source = _source(source_commit)
    directory = _regular_directory(Path(base)/('e3-extended-admission-source-'+source))
    env = _environment()
    if not directory.exists():
        subprocess.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout',
                        REPO,str(directory)], env=env, check=True)
        subprocess.run(['git','sparse-checkout','init','--cone'], cwd=directory, env=env, check=True)
        subprocess.run(['git','sparse-checkout','set',ADMISSION,
                        'experiments/e3_cierre_integral_20261118/data',
                        'experiments/e3_cierre_integral_20261118/orchestration',
                        'experiments/e3_remediacion_20261118/evaluation',
                        'experiments/e3_dataset_admission_20261118/inputs',
                        'experiments/e3_d01_admission_20261118'], cwd=directory, env=env, check=True)
        subprocess.run(['git','fetch','--depth','1','origin',source], cwd=directory, env=env, check=True)
        subprocess.run(['git','checkout','--detach',source], cwd=directory, env=env, check=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=directory, env=env, text=True).strip()
    if head != source:
        raise RuntimeError('Preserved checkout is incomplete or has a different source SHA')
    subprocess.run(['git','diff','--exit-code','HEAD'], cwd=directory, env=env,
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return directory


def run(source_commit, token, *, base='/content', resume_output=None):
    """Get exact code, check GitHub access and persist this CPU-only stage."""
    source = _source(source_commit)
    base = _regular_directory(base)
    base.mkdir(parents=True, exist_ok=True)
    checkout = checkout_source(source, base)
    api = _load(checkout/GITHUB_API, 'e3_extended_admission_github_api')
    store = _load(checkout/GITHUB_STORE, 'e3_extended_admission_github_store')
    token = api._token(token)
    api.validate_credentials(token)
    if resume_output is None:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = base/('e3-additional-data-'+run_id)
    else:
        output = _regular_directory(resume_output)
        context = output/'run_context.json'
        if context.is_file():
            saved = json.loads(context.read_bytes())
            if saved.get('source_commit') != source:
                raise ValueError('Preserved admission output belongs to a different source')
            run_id = saved['run_id']
        else:
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=True)
    context = output/'run_context.json'
    if not context.is_file():
        _publish_new(context,_canonical({'source_commit':source,'run_id':run_id,
                                        'stage':'additional_dataset_admission_only'}))
    receipt = run_stage(checkout, output, source)
    saved = store.save(output, token, source, run_id, api, profile='cpu')
    print('Admisión guardada en GitHub:',saved['run_url'],flush=True)
    print('Commit:',saved['commit_sha'],flush=True)
    return {'receipt':receipt, 'storage':saved, 'output':str(output)}
