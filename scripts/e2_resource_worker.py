"""Verified prospective resource admission around the unchanged E trainers.

Specifications truthfully carry the new 1800-second per-phase limit. Original
600-second outcomes stay terminal; this entrypoint never resumes or retries a
fit. Its own published bytes and policy are bound before opening model data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
REGISTRATION_PATH = 'experiments/e2_resource_continuation_20261112/registration.json'
POLICY_PATH = 'experiments/e2_resource_continuation_20261112/resource_policy.json'
SELF_PATH = 'scripts/e2_resource_worker.py'
SIDECAR = 'execution_extension_admission.json'
RUNTIME_KEYS = ('python', 'torch', 'torch_cuda', 'device_name', 'total_memory_bytes',
    'cpu_threads', 'interop_threads', 'deterministic', 'tf32', 'cudnn_tf32')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_policy(policy):
    required = {'version':'e2-registered-resource-extension-v1',
        'engine_commit':ENGINE_COMMIT, 'initial_phase_wall_seconds':600,
        'stage_wall_seconds':1800, 'worker_overhead_seconds':60,
        'max_optimization_phases':3, 'max_epochs':200, 'patience_epochs':20,
        'authorized_total_hours':72, 'budget_reset':False,
        'new_opportunities_only':True, 'scientific_retry_allowed':False,
        'models_seeds_data_unchanged':True, 'prior_failed_attempts_retried':False}
    if (not isinstance(policy, dict)
            or any(policy.get(k) != value for k, value in required.items())
            or policy.get('sha256') != digest({k:v for k,v in policy.items() if k!='sha256'})):
        raise ValueError('The published prospective resource policy differs')
    return policy


def verify_extension(engine, extension, commit):
    engine, extension = Path(engine).absolute(), Path(extension).absolute()
    if not isinstance(commit, str) or not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('An immutable resource-extension commit is required')
    head = subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'], text=True).strip()
    if head != ENGINE_COMMIT:
        raise ValueError('Keep the unchanged registered scientific engine E')
    raw = (extension/'registration.json').read_bytes()
    published = subprocess.check_output(['git','-C',str(engine),'show',commit+':'+REGISTRATION_PATH])
    if raw != published:
        raise ValueError('Local extension registration differs from its actual Git commit')
    record = json.loads(raw)
    if record.get('engine_commit') != ENGINE_COMMIT or record.get('protocol_id') != 'e2-reduced-delivery-v1':
        raise ValueError('Resource extension belongs to another scientific protocol')
    policy = verify_policy(record.get('resource_policy'))
    hashes = record.get('files_sha256', {})
    if SELF_PATH not in hashes or POLICY_PATH not in hashes:
        raise ValueError('The resource worker and policy must both be byte bound')
    # The launcher materializes every registered runtime script. Tests and
    # narrative files need not be present in the Colab extension directory.
    runtime_files = [p for p in hashes if p.startswith('scripts/') or p == POLICY_PATH]
    for relative in runtime_files:
        target = extension/relative
        if (not target.resolve().is_relative_to(extension.resolve())
                or any(p.is_symlink() for p in (target,*target.parents) if p.is_relative_to(extension))
                or file_sha(target) != hashes[relative]):
            raise ValueError('Published resource execution bytes differ: '+relative)
    if Path(__file__).resolve() != (extension/SELF_PATH).resolve():
        raise ValueError('Execute the byte-bound resource worker itself')
    if json.loads((extension/POLICY_PATH).read_bytes()) != policy:
        raise ValueError('The standalone resource policy differs from registration')
    return engine, extension, record, policy


def observed_platform():
    import torch
    if not torch.cuda.is_available():
        raise ValueError('The registered CUDA device is unavailable')
    return {'python':platform.python_version(), 'torch':str(torch.__version__),
        'torch_cuda':torch.version.cuda, 'device_name':torch.cuda.get_device_name(0),
        'total_memory_bytes':int(torch.cuda.get_device_properties(0).total_memory)}


def validate_hpo(spec, selection, *, observed=None):
    if (spec.get('version') != selection.VERSION
            or spec.get('job') not in selection.job_plan([spec['job']['model_id']])):
        raise ValueError('Worker lies outside the unchanged reduced HPO domain')
    if spec.get('source_sha256') != selection.source_identity():
        raise ValueError('Frozen E HPO source identity differs')
    if spec.get('fit_wall_seconds') != 1800:
        raise ValueError('New worker specifications must truthfully declare1800seconds')
    expected = spec.get('runtime_identity', {})
    observed = observed_platform() if observed is None else observed
    for key in ('python','torch','torch_cuda','device_name','total_memory_bytes'):
        if observed.get(key) != expected.get(key):
            raise ValueError('Actual HPO platform differs from registered runtime: '+key)
    strict = {'cpu_threads':2, 'interop_threads':1, 'deterministic':True,
        'tf32':False, 'cudnn_tf32':False}
    if any(expected.get(k) != v for k,v in strict.items()):
        raise ValueError('Retain the unchanged strict fitting policy')
    # E.run_worker itself applies2/1 threads, then the E trainer applies its
    # strict CUDA context. Calling E's preflight first would configure the
    # process-only interop setting twice and raise before fitting.
    selection.verify_reduced_registration(selection.ROOT, require_git=True)


def validate_final(spec, expected, final, selection):
    if digest(spec) != expected:
        raise ValueError('Worker specification checksum mismatch')
    if spec.get('fit_wall_seconds') != 1800 or spec.get('device') != 'cuda:0':
        raise ValueError('Keep the actual prospective1800-second CUDA worker policy')
    source = spec.get('source_identity', {})
    if source.get('source_commit') != ENGINE_COMMIT or not isinstance(source.get('files_sha256'),dict) or not source['files_sha256']:
        raise ValueError('Bind the unchanged E source commit and executable files')
    for relative, checksum in source['files_sha256'].items():
        target = final.ROOT/relative
        if (not target.resolve().is_relative_to(final.ROOT.resolve())
                or target.is_symlink() or file_sha(target) != checksum):
            raise ValueError('Frozen final worker executable bytes differ')
    actual = selection.worker_preflight_policy()['runtime']
    for key in RUNTIME_KEYS:
        if actual.get(key) != spec.get('runtime_identity',{}).get(key):
            raise ValueError('Actual final runtime differs: '+key)


def write_admission(output, commit, record, policy, mode, spec_sha):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    value={'version':'e2-resource-worker-admission-v1','engine_commit':ENGINE_COMMIT,
        'extension_commit':commit,'resource_policy_sha256':policy['sha256'],
        'resource_phase_wall_seconds':1800,'worker_spec_sha256':spec_sha,'mode':mode,
        'extension_runtime_files_sha256':{p:h for p,h in record['files_sha256'].items()
            if p.startswith('scripts/') or p == POLICY_PATH},
        'max_epochs':200,'patience_epochs':20,'authorized_total_hours':72,
        'optimizer_retry_performed':False,'scientific_training_started_here':False}
    target=output/SIDECAR
    if target.exists() and json.loads(target.read_bytes()) != value:
        raise ValueError('A previous extension admission belongs to another opportunity')
    # A deterministic byte representation makes a repeated parent invocation
    # verify the existing admission rather than rewriting its scientific ID.
    if not target.exists():
        target.write_bytes(canonical(value)+b'\n')
    return value


def run_selection(spec, selection, commit, record, policy):
    validate_hpo(spec, selection)
    output=Path(spec['output'])
    write_admission(output,commit,record,policy,'selection',digest(spec))
    approved=digest(spec)
    previous=selection._validate_spec
    def admitted(candidate):
        if digest(candidate) != approved:
            raise ValueError('The admitted prospective HPO specification changed')
    selection._validate_spec=admitted
    try:
        return selection.run_worker(spec)
    finally:
        selection._validate_spec=previous


def run_final(spec, final, selection, commit, record, policy):
    spec=dict(spec); expected=spec.pop('worker_spec_sha256')
    output=Path(spec['output'])
    if (output/'result.json').exists():
        raise RuntimeError('An existing result belongs to the parent supervisor')
    validate_final(spec,expected,final,selection)
    write_admission(output,commit,record,policy,'final',expected)
    kind=spec.get('kind')
    if kind == 'neural-final':
        result=final.neural_final_worker(spec,output)
    elif kind == 'classical':
        result=final.classical_worker(spec,output)
    elif kind == 'final-score':
        final_output=Path(spec['final_output'])
        if file_sha(final_output/'final_freeze.json') != spec['final_freeze_sha256']:
            raise ValueError('The complete predictor freeze changed before scoring')
        result=final.score_final(spec['project_root'],spec['data_root'],final_output,
            daily_csv=spec.get('daily_csv'),raw_root=spec.get('raw_root'),
            allow_download=spec.get('allow_download',False),device=spec['device'],
            source_roots=spec.get('source_roots'),scoring_output=output/'scientific_outputs')
        result={**result,'phase':'score','model':'all16','seed':'aggregate',
            'A2':'scientific_outputs/benchmark_A2.csv','artifacts_sha256':final._artifact_hashes(output)}
    else:
        raise ValueError('Unknown reduced final worker kind')
    result['worker_spec_sha256']=expected
    result['attempt_id']=digest({'version':final.VERSION,'worker_spec_sha256':expected})
    final.write_json(output/'result.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-root',type=Path,required=True)
    parser.add_argument('--extension-root',type=Path,required=True)
    parser.add_argument('--extension-commit',required=True)
    parser.add_argument('--mode',choices=('selection','final'),required=True)
    parser.add_argument('--worker-spec',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        engine,extension,record,policy=verify_extension(args.engine_root,args.extension_root,args.extension_commit)
        sys.path.insert(0,str(engine/'scripts'))
        selection=importlib.import_module('run_e2_reduced_selection')
        selection.verify_reduced_registration(engine,require_git=True)
        spec=json.loads(args.worker_spec.read_bytes())
        if args.mode == 'selection':
            result=run_selection(spec,selection,args.extension_commit,record,policy)
        else:
            final=importlib.import_module('e2_reduced_final')
            result=run_final(spec,final,selection,args.extension_commit,record,policy)
        print(json.dumps({'event':'worker_completed','mode':args.mode,
            'status':result['status'],'resource_phase_wall_seconds':1800}),flush=True)
        return 0
    except Exception as error:
        failure={'status':'failed','error_type':type(error).__name__,'reason':str(error),
            'resource_receipt':getattr(error,'receipt',{}),'automatic_retry':False}
        # Preserve the complete resource receipt in an artifact. The cell log
        # receives a concise summary instead of hundreds of epoch records.
        target=args.worker_spec.parent/'worker_failure.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix='.worker-failure-',dir=target.parent)
        try:
            with os.fdopen(descriptor,'wb') as stream:
                stream.write(canonical(failure)+b'\n');stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        resource=failure['resource_receipt']
        print(json.dumps({'event':'worker_failed','error_type':failure['error_type'],
            'reason':failure['reason'],'error_code':resource.get('error_code'),
            'optimizer_steps':resource.get('optimizer_steps'),
            'epochs_completed':resource.get('epochs_completed'),
            'automatic_retry':False}),flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
