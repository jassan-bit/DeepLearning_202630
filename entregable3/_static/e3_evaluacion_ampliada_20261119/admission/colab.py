"""Admit the three OHLCV files in Colab, independently of Render or GPU.

AI assistance declared. This stage performs no target construction, model fit,
market inference, statistical contrast or scientific accreditation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = 'https://github.com/jassan-bit/DeepLearning_202630.git'
REPOSITORY = 'jassan-bit/DeepLearning_202630'
ADMISSION = 'experiments/e3_evaluacion_ampliada_20261119/admission'
COLLECTOR = 'experiments/e3_cierre_integral_20261118/data/collect.py'
GITHUB_API = 'experiments/e3_remediacion_20261118/evaluation/colab_github.py'
GITHUB_STORE = 'experiments/e3_cierre_integral_20261118/orchestration/github_store.py'
EXPECTED_IDS = ('D01', 'D02', 'D03')


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
    with receipt_path.open('xb') as stream:
        stream.write(_canonical(receipt))
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
        with context.open('xb') as stream:
            stream.write(_canonical({'source_commit':source,'run_id':run_id,
                                     'stage':'additional_dataset_admission_only'}))
    receipt = run_stage(checkout, output, source)
    saved = store.save(output, token, source, run_id, api, profile='cpu')
    print('Admisión guardada en GitHub:',saved['run_url'],flush=True)
    print('Commit:',saved['commit_sha'],flush=True)
    return {'receipt':receipt, 'storage':saved, 'output':str(output)}
