"""Environment-only continuation for the immutable E2 posthoc controller.

No scientific source, specification, optimizer, result or ledger is rewritten.
The failed setup must be preserved by the caller before choosing a new clean
setup directory. This adapter replaces only the local shared-clone transport,
completes its sparse metadata, and routes actual workers through a bound shim.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time

ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
REPOSITORY = 'https://github.com/jassan-bit/DeepLearning_202630.git'
VERSION = 'e2-posthoc-environment-adapter-v1'
ACCEPTANCE_VERSION = 'e2-posthoc-synthetic-acceptance-v1'
EXTRA_SPARSE = ('/book_e2/_static/', '/results/filipe_multihistory_20261104/test/raw_test_audit.json')
PYPI = 'https://pypi.org/simple'
TORCH_INDEX = 'https://download.pytorch.org/whl/cu126'


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Environment adapter paths must not use symlinks')
    return path


def _sha(path):
    return hashlib.sha256(_regular(path).read_bytes()).hexdigest()


def _strings(command):
    return [os.fspath(value) for value in command]


def _remaining_timeout(deadline, requested):
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Original bounded clean setup deadline reached')
    return min(float(requested), remaining)


def _uv_exact_official(command, lock):
    """Change resolution only for the old exact lock and its two official indexes."""
    words = _strings(command)
    if len(words)<3 or Path(words[0]).name!='uv' or words[1:3]!=['pip','install']:
        return command
    def value(flag):
        return words[words.index(flag)+1] if words.count(flag)==1 and words.index(flag)+1<len(words) else None
    requirement = value('--requirements') or value('-r')
    if (value('--index-url')!=PYPI or value('--extra-index-url')!=TORCH_INDEX
            or requirement is None or Path(requirement).absolute()!=lock
            or any(flag in words for flag in ('--index','--default-index','--find-links'))):
        return command
    lines = [line.strip() for line in _regular(lock).read_text().splitlines() if line.strip()]
    if not lines or any(re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+',line) is None for line in lines):
        raise ValueError('The official-index resolution adapter requires the unchanged exact package lock')
    if '--index-strategy' in words:
        if value('--index-strategy')!='unsafe-best-match':
            raise ValueError('Unexpected existing uv index strategy')
        return command
    return [*command,'--index-strategy','unsafe-best-match']


def install_environment_adapter(old_module, original_source, adapter_root, *, original_output=None):
    """Install a permanent bounded-worker route and temporary setup hooks.

    The bootstrap CLI is: --old-controller PATH --old-controller-sha256 SHA
    followed by the original --worker-spec/--worker-output arguments. The
    bootstrap, supplied by the caller, installs the separately registered ARIMA
    naming shim and invokes the unchanged old worker after verifying its SHA.
    """
    original_source, adapter_root = _regular(original_source), _regular(adapter_root)
    original_output = None if original_output is None else _regular(original_output)
    old_path = _regular(old_module.__file__)
    bootstrap = _regular(adapter_root/'scripts/e2_posthoc_clean_worker_bootstrap.py')
    if (getattr(old_module,'ENGINE_COMMIT',None)!=ENGINE_COMMIT or not old_path.is_file()
            or not bootstrap.is_file() or not original_source.is_dir()):
        raise ValueError('Exact old scientific engine, original source and new bootstrap are required')
    adapter_sha, old_sha, bootstrap_sha = _sha(Path(__file__)), _sha(old_path), _sha(bootstrap)
    metadata = {'version':VERSION,'environment_adapter_sha256':adapter_sha,
                'old_controller_sha256':old_sha,'bootstrap_sha256':bootstrap_sha,
                'original_source':str(original_source),'adapter_root':str(adapter_root),
                'original_output':None if original_output is None else str(original_output),
                'source_checkout_method':'remote_exact_commit','scientific_sources_modified':False}
    previous = getattr(old_module,'_e2_installed_environment_adapter',None)
    if previous is not None:
        if previous!=metadata:
            raise ValueError('Another environment adapter is already installed')
        return dict(metadata)
    original_prepare = old_module.prepare_clean_runtime
    original_run = old_module.run_bounded
    original_immutable = old_module.immutable_json

    def worker_route(command, **kwargs):
        words = _strings(command)
        if (len(words)==6 and Path(words[1]).absolute()==old_path
                and words[2]=='--worker-spec' and words[4]=='--worker-output'):
            if _sha(old_path)!=old_sha or _sha(bootstrap)!=bootstrap_sha:
                raise ValueError('Bound controller or worker bootstrap bytes changed')
            routed = [command[0],bootstrap,'--old-controller',old_path,
                      '--old-controller-sha256',old_sha,*command[2:]]
            return original_run(routed,**kwargs)
        return original_run(command,**kwargs)

    def prepare(original_source_arg, existing_python, destination, *, deadline):
        if _regular(original_source_arg)!=original_source:
            raise ValueError('The environment adapter cannot change the original source input')
        destination = _regular(destination)
        clean_source = _regular(destination/'source')
        receipt_path = destination/'clean_runtime.json'
        state = {'remote_checkout':False,'sparse_extended':False,'receipt_adjusted':False}
        prior_run, prior_immutable = old_module.run_bounded, old_module.immutable_json

        def run(command, **kwargs):
            words = _strings(command)
            exact_clone = ['git','clone','--shared','--no-checkout',str(original_source),str(clean_source)]
            if words==exact_clone:
                started = time.monotonic()
                # Fetch independently from the public immutable remote. No
                # alternates, shallow local object sharing or original .git writes.
                for operation in (
                    ['git','init','--quiet',clean_source],
                    ['git','-C',clean_source,'remote','add','origin',REPOSITORY],
                    ['git','-C',clean_source,'fetch','--depth','1','origin',ENGINE_COMMIT],
                ):
                    options = dict(kwargs)
                    options['timeout'] = _remaining_timeout(deadline,kwargs['timeout'])
                    prior_run(operation,**options)
                state['remote_checkout'] = True
                return time.monotonic()-started
            exact_checkout = ['git','-C',str(clean_source),'checkout','--detach',ENGINE_COMMIT]
            if words==exact_checkout:
                if not state['remote_checkout']:
                    raise ValueError('Clean detached checkout was not acquired independently from the remote')
                sparse = _regular(clean_source/'.git/info/sparse-checkout')
                lines = sparse.read_text().splitlines()
                for pattern in EXTRA_SPARSE:
                    if pattern not in lines: lines.append(pattern)
                sparse.write_text('\n'.join(lines)+'\n')
                state['sparse_extended'] = True
            command = _uv_exact_official(command,destination/'environment.lock.txt')
            options = dict(kwargs)
            options['timeout'] = _remaining_timeout(deadline,kwargs['timeout'])
            return prior_run(command,**options)

        def immutable(path, value):
            if Path(path).absolute()==receipt_path:
                if not state['remote_checkout'] or not state['sparse_extended']:
                    raise ValueError('Cannot claim an independent clean source before verified remote checkout')
                # Mutate this receipt object only: the original prepare returns
                # it after writing, so returned and persisted provenance agree.
                value.update(immutable_git_objects_shared=False,
                    source_checkout_method='remote_exact_commit',
                    environment_adapter_sha256=adapter_sha,source_tracked_status='clean',
                    environment_adapter_bootstrap_sha256=bootstrap_sha)
                state['receipt_adjusted'] = True
            return prior_immutable(path,value)

        old_module.run_bounded, old_module.immutable_json = run, immutable
        try:
            result = original_prepare(original_source_arg,existing_python,destination,deadline=deadline)
            if (not isinstance(result,dict) or result.get('environment_adapter_sha256')!=adapter_sha
                    or result.get('immutable_git_objects_shared') is not False
                    or result.get('source_checkout_method')!='remote_exact_commit'):
                raise ValueError('Existing clean setup lacks this adapter provenance; preserve it for explicit reconciliation')
            if original_output is not None:
                # The immutable environment receipt is already verified and
                # persisted. Acceptance is a separate synthetic-only artifact;
                # do not retrospectively amend that receipt or claim a fit.
                acceptance = _regular(destination/'synthetic_acceptance.json')
                if _sha(old_path)!=old_sha or _sha(bootstrap)!=bootstrap_sha:
                    raise ValueError('Bound controller or worker bootstrap bytes changed')
                if not acceptance.exists():
                    command = [result['clean_python'],bootstrap,'--preflight','--clean-runtime',receipt_path,
                        '--original-output',original_output,'--old-controller',old_path,
                        '--old-controller-sha256',old_sha]
                    original_run(command,timeout=_remaining_timeout(deadline,deadline-time.monotonic()),
                                 cwd=adapter_root,log=destination/'synthetic_acceptance.log')
                if not acceptance.is_file():
                    raise ValueError('Synthetic preflight returned without its separate acceptance receipt')
                evidence = json.loads(acceptance.read_bytes())
                required = {'version':ACCEPTANCE_VERSION,'status':'passed',
                    'clean_runtime_sha256':_sha(receipt_path),'old_controller_sha256':old_sha,
                    'bootstrap_sha256':bootstrap_sha,'model_count':13,
                    'market_data_read':False,'training_performed':False,'optimizer_steps':0}
                if any(evidence.get(key)!=value for key,value in required.items()):
                    raise ValueError('Existing synthetic acceptance has different source, environment or execution scope')
                result = dict(result,synthetic_acceptance_path=str(acceptance),
                    synthetic_acceptance_sha256=_sha(acceptance),
                    synthetic_acceptance_scope='synthetic_B1_D7_13_factories_no_optimizer_no_market_data')
            return result
        finally:
            old_module.run_bounded, old_module.immutable_json = prior_run, prior_immutable

    # Save references for review without changing the immutable module on disk.
    old_module._e2_environment_original_prepare = original_prepare
    old_module._e2_environment_original_run_bounded = original_run
    old_module._e2_environment_original_immutable_json = original_immutable
    old_module.prepare_clean_runtime = prepare
    old_module.run_bounded = worker_route
    old_module._e2_installed_environment_adapter = dict(metadata)
    return dict(metadata)
