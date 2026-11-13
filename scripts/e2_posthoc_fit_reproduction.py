"""Budgeted, separate post hoc clean refits and three actual SAE ablations.

The old 81-predictor freeze, scores, selections and control pointer are read only.
Production fitting is only available inside newly prepared Python and source
checkouts. This module also owns the shared additive 80-hour posthoc journal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
VERSION = 'e2-posthoc-clean-fit-reproduction-v1'
LEDGER_SHA256 = '27aa2336518432d1d873098a3b2e0e5dc039dc501e15caa17599fce7d1cc6c30'
ORIGINAL_CHARGED_SECONDS = 269830.90902790474
TOTAL_SECONDS = 80 * 3600
REMAINING_SECONDS = TOTAL_SECONDS - ORIGINAL_CHARGED_SECONDS
FUNDING_COMMIT = '41da88cf600aadc0319214465c94a85335e2b1b7'
SAE = 'LSTM-autoencoder con predictor'
SEEDS = tuple(range(271801, 271807))
VARIANTS = ('remove_auxiliary_pretraining', 'remove_second_encoder', 'dropout_zero')
ZERO = '0' * 64


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink paths are not admitted')
    return path


def checked(value):
    if not isinstance(value, dict) or value.get('sha256') != digest({k:v for k,v in value.items() if k != 'sha256'}):
        raise ValueError('Bound JSON checksum differs')
    return value


def immutable_json(path, value):
    path = regular(path)
    raw = canonical(value) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Immutable posthoc artifact changed: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def validate_original_anchor(original_output):
    """Verify the exact closed original journal and persisted existing consent."""
    original = regular(original_output)
    budget = original / 'shared_budget'
    raw = regular(budget / 'pilot_budget.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != LEDGER_SHA256 or not raw.endswith(b'\n'):
        raise ValueError('Original completed ledger bytes differ; no reset or automatic migration')
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 1368:
        raise ValueError('Expected all 1368 original budget records')
    previous, active = ZERO, {}
    for i, row in enumerate(rows):
        checked(row)
        if row.get('sequence') != i or row.get('previous_sha256') != previous:
            raise ValueError('Original budget hash chain differs')
        previous = row['sha256']
        if row['kind'] == 'reserved':
            if row['attempt_id'] in active:
                raise ValueError('Duplicate original reservation')
            active[row['attempt_id']] = row['reservation_seconds']
        elif row['kind'] == 'charged':
            if active.pop(row['attempt_id'], None) != row['reservation_seconds']:
                raise ValueError('Unbound original budget charge')
        elif row['kind'] not in ('overhead', 'progress'):
            raise ValueError('Unknown original budget record')
    if active:
        raise ValueError('Original unresolved reservations must remain charged')
    approval = checked(json.loads(regular(budget/'budget_continuation_80_approval.json').read_bytes()))
    authority = checked(json.loads(regular(budget/'budget_continuation_80.json').read_bytes()))
    if (approval.get('confirmed_text') != '80' or approval.get('approved_total_hours') != 80
            or approval.get('source') != 'explicit_colab_confirmation'
            or approval.get('continuation_registration_commit') != FUNDING_COMMIT
            or authority.get('effective_total_hours') != 80 or authority.get('budget_reset') is not False
            or authority.get('approval_sha256') != approval['sha256']
            or authority.get('approval_receipt') != approval):
        raise ValueError('Persisted original 80-hour authority differs')
    return {'version': VERSION, 'original_ledger_sha256': LEDGER_SHA256,
            'original_last_record_sha256': previous, 'original_records': len(rows),
            'original_ledger_path':str(budget/'pilot_budget.jsonl'),
            'original_charged_seconds': ORIGINAL_CHARGED_SECONDS,
            'approved_total_seconds': TOTAL_SECONDS, 'initial_remaining_seconds': REMAINING_SECONDS,
            'original_approval_sha256': approval['sha256'], 'original_authority_sha256': authority['sha256'],
            'scientific_commit': ENGINE_COMMIT, 'budget_reset': False,
            'new_budget_authorization': False, 'scope': 'sequential_posthoc_inference_clean_reproduction_SAE_ablations'}


class PosthocLedger:
    """One append-only additive ledger. Unknown work retains its reservation.

    Hold this context across the complete invocation, including preparation and
    backups. Each supervised subprocess reserves its complete worst-case wall;
    time outside reserved jobs is charged as measured overhead by the caller.
    Reopening an unresolved journal fails closed; no job is silently repeated.
    """
    def __init__(self, root, anchor, *, clock=time.monotonic):
        self.root, self.anchor, self.clock = regular(root), dict(anchor), clock
        if (anchor.get('approved_total_seconds') != TOTAL_SECONDS
                or anchor.get('original_ledger_sha256') != LEDGER_SHA256
                or anchor.get('original_charged_seconds') != ORIGINAL_CHARGED_SECONDS
                or anchor.get('initial_remaining_seconds') != REMAINING_SECONDS
                or anchor.get('budget_reset') is not False):
            raise ValueError('Unchanged original 80-hour anchor required')
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = regular(self.root/'.posthoc-budget.lock').open('a+b')
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close(); raise RuntimeError('Another posthoc controller is active') from None
        self.original_lock=None
        if anchor.get('original_ledger_path'):
            original_path=regular(anchor['original_ledger_path'])
            self.original_lock=(original_path.parent/'.pilot-budget.lock').open('a+b')
            try:
                fcntl.flock(self.original_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                if file_sha(original_path)!=LEDGER_SHA256:
                    raise ValueError('Original ledger changed before acquiring its controller lock')
            except BaseException:
                self.original_lock.close();self.original_lock=None
                fcntl.flock(self.lock.fileno(),fcntl.LOCK_UN);self.lock.close();self.lock=None
                raise
        self.events, self.active, self.attempts, self.charges = [], {}, set(), 0.
        self.persistence = None
        self.path = regular(self.root/'posthoc_budget.jsonl')
        try:
            immutable_json(self.root/'original_anchor.json', self.anchor)
            if self.path.exists():
                raw = self.path.read_bytes()
                if raw and not raw.endswith(b'\n'):
                    raise RuntimeError('Partial ledger tail; preserve full unresolved costs')
                for line in raw.splitlines():
                    self._apply(json.loads(line))
            if self.active:
                raise RuntimeError('Unresolved posthoc reservation; automatic retry prohibited')
        except BaseException:
            self.close(); raise

    def _apply(self, row):
        checked(row)
        if (row.get('version') != VERSION or row.get('anchor_sha256') != digest(self.anchor)
                or row.get('sequence') != len(self.events)
                or row.get('previous_sha256') != (self.events[-1]['sha256'] if self.events else ZERO)):
            raise ValueError('Posthoc journal chain or anchor differs')
        kind = row['kind']
        if kind == 'reserved':
            job, amount = row['job_id'], row['seconds']
            if self.active or job in self.attempts or not math.isfinite(amount) or amount <= 0 or amount > self.remaining_seconds + 1e-9:
                raise ValueError('Duplicate, overlapping or unaffordable posthoc reservation')
            self.active[job] = amount; self.attempts.add(job)
        elif kind == 'charged':
            amount = row['elapsed_seconds']
            if row['job_id'] not in self.active or not math.isfinite(amount) or amount < 0:
                raise ValueError('Invalid posthoc charge')
            self.charges += amount; del self.active[row['job_id']]
        elif kind == 'overhead':
            amount = row['elapsed_seconds']
            if not math.isfinite(amount) or amount < 0 or self.active:
                raise ValueError('Invalid or double-counted controller overhead')
            self.charges += amount
        elif kind == 'interrupted':
            if row['job_id'] not in self.active:
                raise ValueError('Unbound interruption')
        else:
            raise ValueError('Unknown posthoc budget event')
        self.events.append(row)

    def _append(self, kind, **fields):
        if self.lock is None:
            raise RuntimeError('Ledger is closed')
        row = {'version': VERSION, 'anchor_sha256': digest(self.anchor), 'kind': kind,
               'sequence': len(self.events), 'previous_sha256': self.events[-1]['sha256'] if self.events else ZERO,
               'utc': datetime.now(timezone.utc).isoformat(), **fields}
        row['sha256'] = digest(row)
        # Validate on an isolated replay state before persisting.
        clone = object.__new__(PosthocLedger)
        clone.anchor, clone.events, clone.active = self.anchor, list(self.events), dict(self.active)
        clone.attempts, clone.charges = set(self.attempts), self.charges
        clone._apply(row)
        with self.path.open('ab') as stream:
            stream.write(canonical(row)+b'\n'); stream.flush(); os.fsync(stream.fileno())
        self._apply(row)
        if self.persistence is not None:
            self.persistence.snapshot('posthoc_budget_'+kind)
        return row

    def set_persistence(self, state):
        self.persistence=state
        state.snapshot('posthoc_budget_opened')

    @property
    def remaining_seconds(self):
        return max(0., REMAINING_SECONDS-self.charges-sum(self.active.values()))

    def reserve(self, job_id, seconds, *, metadata=None):
        if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', job_id):
            raise ValueError('Bounded safe job identifier required')
        return self._append('reserved', job_id=job_id, seconds=float(seconds), metadata=metadata or {})

    def charge(self, job_id, elapsed_seconds, *, status='completed', evidence=None):
        if status not in ('completed', 'failed', 'interrupted'):
            raise ValueError('Explicit terminal status required')
        if status == 'interrupted':
            return self._append('interrupted', job_id=job_id, elapsed_seconds_lower_bound=float(elapsed_seconds),
                                evidence=evidence or {}, full_reservation_retained=True)
        reserved = self.active.get(job_id)
        row = self._append('charged', job_id=job_id, elapsed_seconds=float(elapsed_seconds), status=status,
                           evidence=evidence or {}, reservation_seconds=reserved)
        if elapsed_seconds > reserved + 1e-6:
            raise RuntimeError('Reservation overrun fully charged; stop further computation')
        return row

    def charge_overhead(self, seconds, *, description='controller_overhead'):
        row = self._append('overhead', elapsed_seconds=float(seconds), description=description)
        if self.remaining_seconds <= 0:
            raise RuntimeError('Original 80-hour budget exhausted')
        return row

    def report(self):
        return {'original_charged_seconds': ORIGINAL_CHARGED_SECONDS, 'posthoc_charged_seconds': self.charges,
                'unresolved_reservations': dict(self.active), 'remaining_seconds': self.remaining_seconds,
                'last_record_sha256': self.events[-1]['sha256'] if self.events else ZERO,
                'approved_total_hours': 80, 'budget_reset': False}

    def close(self):
        if self.original_lock is not None:
            fcntl.flock(self.original_lock.fileno(),fcntl.LOCK_UN);self.original_lock.close();self.original_lock=None
        if self.lock is not None:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN); self.lock.close(); self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def clean_environment(env=None):
    result = dict(os.environ if env is None else env)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'GITHUB_TOKEN', 'GH_TOKEN'):
        result.pop(key, None)
    result.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    return result


def run_bounded(command, *, timeout, cwd=None, log=None, env=None):
    """Bound the whole child process group; kill/reap before any cost release."""
    started = time.monotonic()
    stream = open(log, 'ab') if log else subprocess.DEVNULL
    child = subprocess.Popen([str(x) for x in command], cwd=cwd, env=clean_environment(env),
                             stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True)
    try:
        code = child.wait(timeout=max(.1, timeout))
    except BaseException:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=2)
        raise
    finally:
        if log:
            stream.close()
    if code:
        raise RuntimeError('Bounded child failed (exit %d); inspect its preserved log' % code)
    return time.monotonic()-started


def prepare_clean_runtime(original_source, existing_python, destination, *, deadline):
    """New source checkout plus isolated package installation; no site reuse.

    The package cache may supply bytes, but every distribution is installed into
    a new no-system-sitepackages venv. Network wheel retrieval remains bounded.
    A partial setup is preserved and never presented as a completed clean run.
    """
    original_source, destination = regular(original_source), regular(destination)
    existing_python = Path(existing_python).absolute()
    if not existing_python.is_file() or not os.access(existing_python,os.X_OK):
        raise ValueError('Existing isolated Python executable is absent')
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination/'clean_runtime.json'
    if receipt_path.exists():
        value = json.loads(receipt_path.read_bytes())
        source, python = Path(value['clean_source']), Path(value['clean_python'])
        verify_clean_runtime(value)
        return value
    if (destination/'setup_started.json').exists():
        raise RuntimeError('Incomplete clean setup preserved; explicit reconciliation required')
    immutable_json(destination/'setup_started.json', {'version': VERSION, 'scientific_commit': ENGINE_COMMIT,
        'existing_python': str(existing_python), 'no_system_site_packages': True})
    from export_e2_f2a_cached_data import validate_source
    validate_source(original_source)
    source = destination/'source'
    # Immutable Git objects may be reused; working files and package installs are new.
    run_bounded(['git', 'clone', '--shared', '--no-checkout', original_source, source],
                timeout=min(120, deadline-time.monotonic()), log=destination/'checkout.log')
    run_bounded(['git','-C',source,'sparse-checkout','init','--no-cone'],
                timeout=min(60,deadline-time.monotonic()),log=destination/'checkout.log')
    (source/'.git/info/sparse-checkout').write_text('/scripts/\n/experiments/\n/requirements*.txt\n/data/manifests/\n')
    run_bounded(['git', '-C', source, 'checkout', '--detach', ENGINE_COMMIT],
                timeout=min(120, deadline-time.monotonic()), log=destination/'checkout.log')
    # Development derivatives are untracked local cache inputs. Copy only the
    # original loader's exact hash-pinned files, never the final test cache.
    sys.path.insert(0, str(original_source/'scripts'))
    development = importlib.import_module('e2_development_data')
    for relative, expected in {development.PROTOCOL_PATH: development.EXPECTED_PROTOCOL_SHA256,
                               **development.EXPECTED_SOURCES}.items():
        raw = regular(original_source/relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Original development cache differs')
        target = source/relative; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError('Checkout development derivative differs')
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(raw)
    query = ('import json,sys,importlib.metadata as m; '
             'print(json.dumps({"python":list(sys.version_info[:3]),"packages":'
             '{d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}}))')
    profile = json.loads(subprocess.check_output([str(existing_python), '-I', '-c', query], env=clean_environment(), text=True))
    if profile['python'] != [3,12,12] or profile['packages'].get('torch') != '2.8.0+cu126':
        raise ValueError('Existing original Python3.12.12/Torch2.8cu126 environment required')
    lock = destination/'environment.lock.txt'
    lines = []
    for name, version in sorted(profile['packages'].items()):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not re.fullmatch(r'[A-Za-z0-9_.+!-]+', version):
            raise ValueError('Only exact ordinary package versions can enter the clean lock')
        lines.append(name+'=='+version)
    lock.write_text('\n'.join(lines)+'\n')
    python = destination/'venv/bin/python'
    run_bounded([existing_python, '-I', '-m', 'venv', '--copies', destination/'venv'],
                timeout=min(180, deadline-time.monotonic()), log=destination/'environment_install.log')
    import shutil
    uv = shutil.which('uv')
    if uv:
        command = [uv,'pip','install','--python',python,'--no-deps','--requirements',lock,
                   '--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cu126']
        cache_method = 'new_venv_install_using_uv_cache_or_official_indexes'
    else:
        command = [python,'-I','-m','pip','install','--no-deps','-r',lock,
                   '--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cu126']
        cache_method = 'new_venv_install_using_pip_cache_or_official_indexes'
    run_bounded(command, timeout=deadline-time.monotonic()-5, log=destination/'environment_install.log')
    new_profile = json.loads(subprocess.check_output([str(python),'-I','-c',query], env=clean_environment(), text=True))
    if new_profile != profile:
        raise ValueError('Clean installed distribution versions differ from original runtime')
    runtime_query = ('import json,sys,torch; print(json.dumps({"python":sys.version,"torch":torch.__version__, '
                     '"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),'
                     '"total_memory_bytes":torch.cuda.get_device_properties(0).total_memory}))')
    runtime = json.loads(subprocess.check_output([str(python),'-I','-c',runtime_query], env=clean_environment(),text=True))
    if 'A100' not in runtime['gpu'] or runtime['cuda'] != '12.6':
        raise ValueError('Original A100/CUDA12.6 class required for clean reproducibility comparison')
    receipt = {'version':VERSION, 'clean_source':str(source), 'clean_python':str(python),
               'scientific_commit':ENGINE_COMMIT,'new_venv_created':True,'system_site_packages':False,
               'clean_source_checkout':True,'immutable_git_objects_shared':True,
               'package_install_method':cache_method,'package_cache_reuse_is_not_environment_reuse':True,
               'package_lock_sha256':file_sha(lock), 'package_profile':profile, 'runtime':runtime,
               'source_tracked_status':'clean','setup_completed':True}
    verify_clean_runtime(receipt)
    immutable_json(receipt_path, receipt)
    return receipt


def verify_clean_runtime(value):
    source, python = regular(value['clean_source']), regular(value['clean_python'])
    if (value.get('scientific_commit') != ENGINE_COMMIT or not value.get('new_venv_created')
            or value.get('system_site_packages') is not False or not value.get('setup_completed')):
        raise ValueError('A genuinely new isolated environment and clean source receipt is required')
    head = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
    cfg = (python.parent.parent/'pyvenv.cfg').read_text().lower()
    if head != ENGINE_COMMIT or dirty or 'include-system-site-packages = false' not in cfg:
        raise ValueError('Clean source checkout or isolated environment changed')
    if file_sha(source.parent/'environment.lock.txt') != value['package_lock_sha256']:
        raise ValueError('Clean runtime lock changed')
    query=('import json,sys,importlib.metadata as m; '
           'print(json.dumps({"python":list(sys.version_info[:3]),"packages":'
           '{d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}}))')
    current=json.loads(subprocess.check_output([str(python),'-I','-c',query],env=clean_environment(),text=True))
    if current != value['package_profile']:
        raise ValueError('Actual clean installed package profile changed')
    return True


def compare_states(actual, original):
    """Compare state tensors, not torch.save container bytes or new receipts."""
    import torch
    if set(actual) != set(original):
        raise ValueError('Reproduced state has different parameter/buffer names')
    maximum = 0.; exact = True
    for key in actual:
        a,b = actual[key].detach().cpu(),original[key].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Reproduced state tensor geometry/dtype/nonfinite mismatch')
        exact = exact and torch.equal(a,b)
        if a.numel():
            maximum = max(maximum,float((a.to(torch.float64)-b.to(torch.float64)).abs().max()))
    return {'state_tensor_bitwise_equal':bool(exact),'maximum_state_tensor_absolute_difference':maximum,
            'state_entries':len(actual),'torch_serialized_file_equality_required':False}


def ablation_design(selection):
    epochs = dict(selection['fixed_phase_epochs']); trial = dict(selection['winner_trial'])
    if set(epochs) != {'ae1','ae2','supervised'} or trial.get('dropout') != .3:
        raise ValueError('Expected original published SAE frozen three-stage dropout0.3 winner')
    return {name:{'condition':name,'fixed_epochs':({k:v for k,v in epochs.items() if k == 'supervised'}
                    if name == VARIANTS[0] else {k:v for k,v in epochs.items() if k != 'ae2'}
                    if name == VARIANTS[1] else epochs),
                 'trial':{**trial,'dropout':0. if name == VARIANTS[2] else trial['dropout']},
                 'HPO':False,'selection_performed':False,'baseline':'fresh_clean_full_SAE_same_seed'} for name in VARIANTS}


def one_encoder_model(dropout):
    import torch
    from torch import nn
    import e2_recipe_models as recipes
    class OneEncoderPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder1 = recipes.recipe_lstm(5,25)
            self.dropout1 = nn.Dropout(dropout)
            self.head = recipes.dense(25)
        def forward(self,x):
            recipes.check_window(x)
            return torch.nn.functional.softplus(self.head(self.dropout1(recipes.last_state(self.encoder1,x)))).squeeze(-1)
    return OneEncoderPredictor()


def fit_ablation(spec, train, destination, *, allow_cpu_for_tests=False):
    """Actual new weights, original durable AdamW trainer, no validation or HPO."""
    import torch
    import e2_staged_training as historical
    import e2_reduced_fit as fitting
    from e2_amended_pilot_artifacts import PilotArtifactStore
    from e2_recipe_models import SAEPretraining
    from e2_published_sae import build_sae, _annotate
    variant = spec['variant']
    design = ablation_design(spec['selection'])[variant]
    trial, epochs = design['trial'], design['fixed_epochs']
    seed, device = spec['seed'], spec.get('device','cuda:0')
    if seed not in SEEDS or (device != 'cuda:0' and not allow_cpu_for_tests):
        raise ValueError('Original six seeds and strict production CUDA required')
    config = historical.TrainConfig(learning_rate=trial['learning_rate'],weight_decay=trial['weight_decay'],
        batch_size=trial['batch_size'],max_epochs=200,patience=20,seed=seed,
        warmup_fraction=trial.get('warmup_fraction',0),wall_limit_seconds=spec.get('phase_wall_seconds',1800),
        memory_limit_bytes=12*1024**3)
    directory = regular(destination); directory.mkdir(parents=True,exist_ok=True)
    if (directory/'request.json').exists():
        raise RuntimeError('Ablation fit already admitted; never restart or retry automatically')
    request = {'version':VERSION,'seed':seed,'design':design,'train_signature':train.signature,
               'source_commit':ENGINE_COMMIT,'fixed_epochs':epochs,'validation_used':False,
               'original_selection_sha256':digest(spec['selection'])}
    immutable_json(directory/'request.json',request)
    phases = {}
    gpu_limit = None if allow_cpu_for_tests else int(spec['runtime_identity']['total_memory_bytes']*.8)
    def phase(name,model,x,y,scale,supervised):
        store = PilotArtifactStore(directory/'phases',name,{'request_sha256':digest(request),'phase':name,'seed':seed})
        try:
            from dataclasses import asdict
            store.prepare({'config':asdict(config),'source':{'source_commit':ENGINE_COMMIT},
                           'data':train.record(),'seed':seed,'resources':spec.get('runtime_identity',{'device':device,'test_only':allow_cpu_for_tests}),
                           'posthoc_ablation':design,'source_commit':ENGINE_COMMIT,'no_validation':True})
            outcome = fitting._phase_fit(model,x,y,None,None,config,store,supervised=supervised,
                target_scale=scale,fixed_epochs=epochs[name],device=device,
                allow_cpu_for_tests=allow_cpu_for_tests,gpu_limit=gpu_limit)
            store.complete({'phase':name,'optimizer_steps':outcome.optimizer_steps,
                            'selected_state_sha256':historical.state_sha256(model)})
            phases[name] = outcome.receipt
        finally:
            store.close()
        # Preserve all curves and hashes, plus initial/latest/best snapshots.
        phase_dir=directory/'phases'/name
        manifest=json.loads((phase_dir/'manifest.json').read_bytes())
        keep={item['path'] for item in manifest['checkpoints'].values()}; retired={}
        for path in (phase_dir/'checkpoints').glob('*.pt'):
            relative=str(path.relative_to(phase_dir))
            if relative not in keep:
                retired[relative]=file_sha(path); path.unlink()
        immutable_json(phase_dir/'superseded_checkpoint_hashes.json',
            {'reason':'Successful phase closure; preserve initial/latest/best and every epoch curve',
             'deleted_checkpoint_hashes':retired,'current_checkpoint_paths':sorted(keep)})
        fitting.verify_phase_archive(phase_dir)
    historical._seed(seed)
    auxiliary_scale = None
    if variant == 'remove_auxiliary_pretraining':
        model = build_sae(dropout=trial['dropout'],seed=seed)
    else:
        manager = SAEPretraining(train.x.to(device),train.signature,accept_auxiliary_scale=True)
        x,y = manager.phase1_data()
        auxiliary_scale = manager.scale
        phase('ae1',manager.ae1,x,y,1.,False)
        if variant == 'remove_second_encoder':
            historical._seed(seed)
            model = one_encoder_model(trial['dropout']).to(device)
            model.encoder1.load_state_dict(manager.ae1.encoder.state_dict())
        else:
            x2,y2 = manager.prepare_phase2()
            if not torch.equal(y,y2):
                raise RuntimeError('Original SAE auxiliary target changed')
            phase('ae2',manager.ae2,x2,y2,1.,False)
            historical._seed(seed)
            model = _annotate(manager.transfer_predictor(),dropout=trial['dropout'],seed=seed)
    phase('supervised',model,train.x,train.y,train.scaler.target_scale,True)
    architecture={'model':SAE,'condition':variant,'hidden_units':[25] if variant==VARIANTS[1] else [25,26],
                  'dropout':trial['dropout'],'pretraining_executed':variant!=VARIANTS[0],
                  'auxiliary_scale':auxiliary_scale,'actual_parameter_count':sum(p.numel() for p in model.parameters()),
                  'same_frozen_remaining_phase_epochs':True,'no_HPO':True}
    checkpoint=directory/'selected_model.pt'
    with checkpoint.open('xb') as stream:
        torch.save({'version':VERSION,'state_dict':historical.clone_state(model),'scaler':train.scaler.record(),
                    'architecture':architecture,'request_sha256':digest(request)},stream)
        stream.flush();os.fsync(stream.fileno())
    return model,{'status':'completed','checkpoint':str(checkpoint.relative_to(directory)),
                  'checkpoint_sha256':file_sha(checkpoint),'phases':phases,'design':design,
                  'architecture':architecture,'optimizer_steps':sum(r['optimizer_steps'] for r in phases.values()),
                  'training_label_end':'2024-10-31','validation_or_test_used_for_fit':False}


def worker(spec_path, output):
    """Run once under the clean Python, with the exact clean scientific modules."""
    spec=json.loads(regular(spec_path).read_bytes()); output=regular(output)
    if spec.get('controller_source_sha256')!=file_sha(Path(__file__)):
        raise ValueError('Actual executing controller bytes differ from the bound worker spec')
    runtime=spec['clean_runtime']; verify_clean_runtime(runtime)
    expected_python=Path(runtime['clean_python']).absolute()
    if Path(sys.executable).absolute()!=expected_python:
        raise ValueError('Fitting must execute inside the newly installed clean Python')
    clean_source=Path(runtime['clean_source'])
    sys.path.insert(0,str(clean_source/'scripts'))
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    from e2_frozen_posthoc_gpu import original_context
    context=original_context(clean_source,spec['original_output'])
    import numpy as np
    import torch
    import e2_reduced_final as final
    from e2_published_pilot_training import predict_pilot
    from e2_reduced_final_analysis import residual_calibrator
    if Path(final.__file__).resolve()!=clean_source/'scripts/e2_reduced_final.py':
        raise ValueError('The exact clean original scientific module was shadowed')
    row=next(r for r in context['freeze']['models'] if r['model']==spec['model'] and r['seed']==spec['seed'])
    original_checkpoint=Path(spec['original_output'])/'final'/row['checkpoint']
    is_classical=spec['seed']=='deterministic'
    original_directory=original_checkpoint.parent if is_classical else original_checkpoint.parent.parent
    original_worker=json.loads((original_directory/'worker_spec.json').read_bytes())
    original_saved=None if is_classical else torch.load(original_checkpoint,map_location='cpu',weights_only=False)
    new_spec={**original_worker,'data_root':str(clean_source),'device':'cuda:0',
              'runtime_identity':runtime['runtime'],'fit_wall_seconds':1800}
    output.mkdir(parents=True,exist_ok=True)
    if spec['kind']=='reproduction' and is_classical:
        import pickle
        result=final.classical_worker(new_spec,output/'fresh')
        checkpoint=output/'fresh'/result['checkpoint']
        history=final._history(context['panel'])
        if spec['model']=='ARIMA':
            with checkpoint.open('rb') as stream: fresh_classical=pickle.load(stream)
            with original_checkpoint.open('rb') as stream: old_classical=pickle.load(stream)
            parameter_equal=fresh_classical.receipt['parameters']==old_classical.receipt['parameters']
            forecasts=final._arima_walk(fresh_classical,history,context['bundle']['dates'][-1],output/'causal_test_predictions')
            predictions=np.asarray([forecasts[d] for d in context['bundle']['dates']])
        else:
            config=json.loads(checkpoint.read_bytes())
            predictions=np.asarray(final._naive_predictions(history,context['bundle']['dates'],config['lag']))
            parameter_equal=json.loads(original_checkpoint.read_bytes())==config
        old_cal=json.loads((original_directory/'predictions.json').read_bytes())['prediction_original']
        new_cal=json.loads((output/'fresh/predictions.json').read_bytes())['prediction_original']
        calibration_difference=float(np.max(np.abs(np.asarray(new_cal)-np.asarray(old_cal))))
        state_comparison={'classical_parameters_exactly_equal':bool(parameter_equal)}
        metadata={'state_comparison':state_comparison,'calibration53_maximum_absolute_difference':calibration_difference,
                  'fresh_fit':True,'source_and_environment_clean':True,'CV_or_selection_repeated':False}
    elif spec['kind']=='reproduction':
        result=final.neural_final_worker(new_spec,output/'fresh')
        checkpoint=output/'fresh'/result['checkpoint']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        trial=original_worker['selection']['winner_trial']
        model=final._trial_builder(spec['model'],trial,spec['seed'],original_worker.get('source_roots'))
        model.load_state_dict(saved['state_dict']);model.eval()
        state_comparison=compare_states(saved['state_dict'],original_saved['state_dict'])
        fresh_cal=json.loads((output/'fresh/calibration_predictions.json').read_bytes())['prediction_original']
        old_cal=json.loads((original_checkpoint.parent.parent/'calibration_predictions.json').read_bytes())['prediction_original']
        calibration_difference=float(np.max(np.abs(np.asarray(fresh_cal)-np.asarray(old_cal))))
        metadata={'state_comparison':state_comparison,'calibration53_maximum_absolute_difference':calibration_difference,
                  'fresh_fit':True,'source_and_environment_clean':True}
    elif spec['kind']=='ablation' and spec['model']==SAE:
        final.validate_selection(SAE,original_worker['selection'])
        model,result=fit_ablation({**spec,'selection':original_worker['selection'],
            'runtime_identity':runtime['runtime'],'device':'cuda:0'},context['training'],output/'fit')
        checkpoint=output/'fit'/result['checkpoint']
        calib=final._calibration_split(context['panel'],context['training'].scaler)
        cp=predict_pilot(model,calib.x,batch_size=1,device='cuda:0').numpy()*context['training'].scaler.target_scale
        immutable_json(output/'calibration_predictions.json',{'dates':final.CALIBRATION_DATES,
            'prediction_original':cp.tolist(),'target_original':list(calib.y_original),'used_for_HPO':False})
        immutable_json(output/'calibrator.json',residual_calibrator(cp,calib.y_original,
            dates=final.CALIBRATION_DATES,predictor_sha256=file_sha(checkpoint)))
        metadata={'fresh_fit':True,'condition':spec['variant'],'selection_performed':False}
    else:
        raise ValueError('Unknown controlled posthoc fit kind')
    scaler=context['training'].scaler
    if original_saved is not None and original_saved['scaler']!=scaler.record():
        raise ValueError('Original training-only scaler differs')
    if not is_classical:
        x=torch.as_tensor(scaler.transform(context['raw_windows']),dtype=torch.float32)
        predictions=predict_pilot(model,x,batch_size=1,device='cuda:0').numpy()*scaler.target_scale
    baseline=next(r for r in context['bundle']['predictors'] if r['model']==spec['model'] and r['seed']==spec['seed'])
    baseline_prediction=np.asarray(baseline['prediction_original'])
    max_difference=float(np.max(np.abs(predictions-baseline_prediction)))
    immutable_json(output/'test_predictions.json',{'dates':context['bundle']['dates'],
        'prediction_original':predictions.tolist(),'checkpoint_sha256':file_sha(checkpoint),
        'known_posthoc_calendar':True,'used_for_HPO':False})
    if spec['kind']=='reproduction':
        metadata.update(test608_maximum_absolute_difference=max_difference,
            predictions_equal_at_1e_10=bool(max_difference<=1e-10),
            clean_reproduction_passed=bool(max_difference<=1e-10 and calibration_difference<=1e-10
                                          and state_comparison.get('state_tensor_bitwise_equal',
                                              state_comparison.get('classical_parameters_exactly_equal',False))))
    result={'version':VERSION,'status':'completed','controller_source_sha256':spec['controller_source_sha256'],
            'configuration_commit':spec['configuration_commit'],'worker_spec_sha256':digest(spec),
            'attempt_id':spec['job_id'],'kind':spec['kind'],
            'model':spec['model'],'seed':spec['seed'],'condition':spec.get('variant','full'),
            'rmse_original':float(np.sqrt(np.mean((predictions-context['y'])**2))),
            'original_read_only_prediction_maximum_difference':max_difference,
            'fit_result':result,'metadata':metadata,'original_freeze_unchanged':True,
            'scientific_commit':ENGINE_COMMIT,'new_clean_environment':runtime,
            'original_test_labels_known_posthoc':True,'new_winner_selection_performed':False}
    result['artifacts_sha256']={str(p.relative_to(output)):file_sha(p) for p in output.rglob('*')
        if p.is_file() and not p.name.startswith('.') and p.name not in ('result.json','worker.log','worker_spec.json')}
    immutable_json(output/'result.json',result)
    return result


def compare_ablations(y, full, variants, *, draws=10000, block_days=14, seed=202611134):
    """Matched-seed/calendar-block bootstrap of original mean-seed RMSE.

    Positive delta means removing a component worsens error. Six seeds describe
    algorithm RNG variation conditional on this one already-known chronology.
    All contrasts share bootstrap indices. No new winner is chosen.
    """
    import numpy as np
    y=np.asarray(y,dtype=float); full=np.asarray(full,dtype=float)
    if (y.ndim!=1 or full.shape!=(6,len(y)) or not np.isfinite(y).all()
            or not np.isfinite(full).all() or set(variants)!=set(VARIANTS)
            or type(draws)is not int or draws<10000 or not 1<=block_days<=len(y)):
        raise ValueError('Aligned six-seed full baseline, three actual conditions and >=10000 draws required')
    variants={k:np.asarray(v,dtype=float) for k,v in variants.items()}
    if any(v.shape!=full.shape or not np.isfinite(v).all() for v in variants.values()):
        raise ValueError('Ablation prediction support differs')
    rng=np.random.default_rng(seed); n=len(y)
    loss=np.stack([(full-y)**2]+[(variants[k]-y)**2 for k in VARIANTS])
    original=np.sqrt(loss.mean(axis=2)).mean(axis=1)
    samples=np.empty((draws,3))
    for start in range(0,draws,128):
        size=min(128,draws-start)
        beginnings=rng.integers(0,n,(size,math.ceil(n/block_days)))
        dates=((beginnings[:,:,None]+np.arange(block_days))%n).reshape(size,-1)[:,:n]
        seeds=rng.integers(0,6,(size,6))
        for j in range(size):
            scores=np.sqrt(loss[:,seeds[j]][:,:,dates[j]].mean(axis=2)).mean(axis=1)
            samples[start+j]=scores[1:]-scores[0]
    rows=[]
    for i,name in enumerate(VARIANTS):
        effect=float(original[i+1]-original[0]); null=samples[:,i]-effect
        p=float((1+np.count_nonzero(np.abs(null)>=abs(effect)))/(draws+1))
        interval=np.quantile(samples[:,i],[.025,.975])
        paired=np.sqrt(loss[i+1].mean(axis=1))-np.sqrt(loss[0].mean(axis=1))
        rows.append({'condition':name,'delta_mean_seed_RMSE_ablated_minus_full':effect,
                     'CI95_percentile_unadjusted':interval.tolist(),'p_two_sided_centered':p,
                     'paired_seed_RMSE_differences':paired.tolist(),'seed_pairs':6,
                     'draws':draws,'block_days':block_days,'seed':seed,
                     'method':'paired_seed_and_calendar_circular_block_bootstrap',
                     'post_hoc':True,'new_winner_selected':False})
    previous=0.
    for rank,i in enumerate(sorted(range(3),key=lambda i:rows[i]['p_two_sided_centered'])):
        previous=max(previous,min(1.,(3-rank)*rows[i]['p_two_sided_centered']))
        rows[i]['p_Holm3']=previous;rows[i]['reject_Holm05']=previous<.05
    return rows


def verified_completed(directory, *, controller_source_sha256=None, configuration_commit=None):
    path=Path(directory)/'result.json'
    if not path.exists():
        return None
    result=json.loads(path.read_bytes())
    if (controller_source_sha256 is not None and result.get('controller_source_sha256')!=controller_source_sha256
            or configuration_commit is not None and result.get('configuration_commit')!=configuration_commit):
        raise RuntimeError('Existing completed fit belongs to different controller bytes or configuration commit')
    if result.get('status')!='completed' or not result.get('artifacts_sha256'):
        raise RuntimeError('Incomplete existing posthoc result requires reconciliation')
    for relative,expected in result['artifacts_sha256'].items():
        item=regular(Path(directory)/relative)
        if not item.resolve().is_relative_to(Path(directory).resolve()) or file_sha(item)!=expected:
            raise ValueError('Completed posthoc artifact changed')
    return result


def backup_completed(directory,result,client):
    """Three bounded upload attempts, zero repeated fitting attempts."""
    last=None
    for attempt in range(3):
        try:
            client._branch_head();client._checked=True
            return client.persist_result(directory,result)
        except Exception as error:
            last=error
            if attempt<2:
                time.sleep(1)
    raise RuntimeError('GitHub backup incomplete; completed computation preserved, retry upload only') from last


def run_fits(original_source, original_output, destination, existing_python, *, budget, backup_client,
             extension_root=None, configuration_commit=None, clean_setup_seconds=2400):
    """Sequential 81 fresh fixed fits and18actual SAE fits under one balance.

    The individual worst-case reservation is admitted immediately before each
    job; unstarted jobs are reported when the remaining budget is insufficient.
    Completed jobs may only be re-verified/re-backed-up, never fitted again.
    """
    destination=regular(destination);destination.mkdir(parents=True,exist_ok=True)
    extension_root=Path(extension_root or Path(__file__).resolve().parents[1])
    if not isinstance(configuration_commit,str) or not re.fullmatch(r'[0-9a-f]{40}',configuration_commit):
        raise ValueError('Actual published immutable configuration commit required')
    controller_sha=file_sha(Path(__file__))
    invocation_start=time.monotonic();gap=invocation_start
    setup_id='clean-setup-'+str(time.time_ns())
    if not (destination/'clean/clean_runtime.json').exists():
        reservation=min(float(clean_setup_seconds),budget.remaining_seconds)
        if reservation<120:
            raise RuntimeError('Insufficient unchanged budget for clean setup')
        start=time.monotonic()
        budget.charge_overhead(start-gap,description='clean_setup_admission')
        budget.reserve(setup_id,reservation,metadata={'scope':'source_checkout_and_actual_new_venv_install'})
        try:
            runtime=prepare_clean_runtime(original_source,existing_python,destination/'clean',deadline=start+reservation-5)
        except BaseException as error:
            budget.charge(setup_id,time.monotonic()-start,status='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed')
            raise
        else:
            finished=time.monotonic()
            budget.charge(setup_id,finished-start,status='completed',evidence={'receipt_sha256':file_sha(destination/'clean/clean_runtime.json')})
        gap=finished
    else:
        runtime=json.loads((destination/'clean/clean_runtime.json').read_bytes());verify_clean_runtime(runtime)
    # The old source can be read here; every training process later imports
    # clean_source and executes with clean_python, with original bytes checked.
    sys.path.insert(0,str(Path(original_source)/'scripts'))
    import e2_reduced_protocol as protocol
    from export_e2_f2a_cached_data import build_bundle
    bundle=build_bundle(original_source,original_output)
    freeze=json.loads((Path(original_output)/'final/final_freeze.json').read_bytes())
    original_snapshot={'freeze':file_sha(Path(original_output)/'final/final_freeze.json'),
                       'ledger':file_sha(Path(original_output)/'shared_budget/pilot_budget.jsonl')}
    # Reserve three-stage jobs early, while their conservative cap still fits.
    # This order is fixed before scoring; it never selects by observed outcomes.
    sae_rows=[r for r in freeze['models'] if r['model']==SAE]
    remaining_rows=[r for r in freeze['models'] if r['model']!=SAE]
    jobs=[{'kind':'reproduction','model':r['model'],'seed':r['seed'],'condition':'full'} for r in sae_rows]
    jobs += [{'kind':'ablation','model':SAE,'seed':seed,'variant':variant,'condition':variant}
             for variant in reversed(VARIANTS) for seed in SEEDS]
    jobs += [{'kind':'reproduction','model':r['model'],'seed':r['seed'],'condition':'full'} for r in remaining_rows]
    completed=[];unstarted=[]
    for index,job in enumerate(jobs):
        job_id='posthoc-'+job['kind']+'-'+hashlib.sha256((job['model']+':'+str(job['seed'])+':'+job['condition']).encode()).hexdigest()[:24]
        directory=destination/'jobs'/job_id
        result=verified_completed(directory,controller_source_sha256=controller_sha,configuration_commit=configuration_commit)
        if result is not None:
            backup_completed(directory,result,backup_client);completed.append(result)
            continue
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError('An admitted or interrupted fit exists; never retry automatically: '+job_id)
        row=next(r for r in freeze['models'] if r['model']==job['model'] and r['seed']==job['seed'])
        checkpoint=Path(original_output)/'final'/row['checkpoint']
        original_dir=checkpoint.parent if job['seed']=='deterministic' else checkpoint.parent.parent
        old_spec=json.loads((original_dir/'worker_spec.json').read_bytes())
        if job['kind']=='ablation':
            phase_count=len(ablation_design(old_spec['selection'])[job['variant']]['fixed_epochs'])
        else:
            phase_count=3 if job['model']==SAE else 1
        reservation=1800*phase_count+60
        boundary=time.monotonic()
        budget.charge_overhead(boundary-gap,description='fit_controller_verification_upload_and_manifest_overhead')
        gap=boundary
        if budget.remaining_seconds<reservation:
            unstarted=jobs[index:];break
        directory.mkdir(parents=True,exist_ok=False)
        spec={**job,'job_id':job_id,'original_output':str(Path(original_output).absolute()),'clean_runtime':runtime,
              'controller_source_sha256':controller_sha,'configuration_commit':configuration_commit}
        immutable_json(directory/'worker_spec.json',spec)
        started=boundary
        budget.reserve(job_id,reservation,metadata={'kind':job['kind'],'model':job['model'],'seed':job['seed'],
                                                   'phases':phase_count,'phase_cap':1800,'overhead_cap':60})
        try:
            run_bounded([runtime['clean_python'],Path(__file__).resolve(),'--worker-spec',directory/'worker_spec.json',
                         '--worker-output',directory],timeout=max(.1,started+reservation-5-time.monotonic()),cwd=extension_root,log=directory/'worker.log')
            result=verified_completed(directory,controller_source_sha256=controller_sha,configuration_commit=configuration_commit)
            if result is None:
                raise RuntimeError('Worker returned without its completed immutable receipt')
            backup_completed(directory,result,backup_client)
        except BaseException as error:
            budget.charge(job_id,time.monotonic()-started,status='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed',
                          evidence={'partial_artifacts_preserved':True,'scientific_retry':False})
            raise
        else:
            finished=time.monotonic()
            budget.charge(job_id,finished-started,status='completed',evidence={'result_sha256':file_sha(directory/'result.json')})
        gap=finished
        completed.append(result)
        print(json.dumps({'event':'posthoc_fit_completed','completed':len(completed),'planned':len(jobs),
                          'kind':job['kind'],'model':job['model'],'seed':job['seed'],
                          'remaining_seconds':budget.remaining_seconds}),flush=True)
    if original_snapshot!={'freeze':file_sha(Path(original_output)/'final/final_freeze.json'),
                           'ledger':file_sha(Path(original_output)/'shared_budget/pilot_budget.jsonl')}:
        raise ValueError('Original scientific freeze or ledger changed')
    repetitions=[r for r in completed if r['kind']=='reproduction']
    ablations=[r for r in completed if r['kind']=='ablation']
    comparisons=None
    if len(ablations)==18 and sum(r['model']==SAE for r in repetitions)==6:
        def predictions(row):
            return json.loads((destination/'jobs'/row['attempt_id']/'test_predictions.json').read_bytes())['prediction_original']
        full=[predictions(next(r for r in repetitions if r['model']==SAE and r['seed']==seed)) for seed in SEEDS]
        variants={variant:[predictions(next(r for r in ablations if r['condition']==variant and r['seed']==seed)) for seed in SEEDS]
                  for variant in VARIANTS}
        comparisons=compare_ablations(bundle['y_true'],full,variants)
        immutable_json(destination/'SAE_ablation_comparisons.json',comparisons)
    summary={'version':VERSION,'status':'completed' if len(completed)==99 else 'paused_clean',
             'attempt_id':'posthoc-clean-reproduction-summary-v1','planned_reproduction_fits':81,
             'planned_neural_reproduction_fits':78,'planned_classical_reproduction_fits':3,
             'completed_reproduction_fits':len(repetitions),'planned_ablation_fits':18,'completed_ablation_fits':len(ablations),
             'reproduction_passed':len(repetitions)==81 and all(r['metadata']['clean_reproduction_passed'] for r in repetitions),
             'reproduction_failures':[{'model':r['model'],'seed':r['seed'],'metadata':r['metadata']}
                                      for r in repetitions if not r['metadata']['clean_reproduction_passed']],
             'unstarted_jobs':unstarted,'ablation_comparisons':comparisons,'original_snapshot':original_snapshot,
             'budget':budget.report(),'known_posthoc_test_calendar':True,'retuning_performed':False,
             'new_original_winner_claimed':False,'clean_source_and_environment_receipt':runtime,
             'elapsed_current_invocation_seconds':time.monotonic()-invocation_start}
    # Partial progress has a distinct immutable summary identity; no overwriting.
    summary_id='summary-'+digest({'jobs':[r['attempt_id'] for r in completed],
                                  'configuration_commit':configuration_commit,'invocation':time.time_ns()})[:16]
    summary_dir=destination/summary_id
    summary['attempt_id']='posthoc-clean-'+summary_id
    immutable_json(summary_dir/'summary.json',summary)
    summary['artifacts_sha256']={'summary.json':file_sha(summary_dir/'summary.json')}
    backup_completed(summary_dir,summary,backup_client)
    budget.charge_overhead(time.monotonic()-gap,description='fit_final_summary_bootstrap_and_upload')
    return summary



class PosthocRemoteState:
    """Reuse the proven transport with isolated new control globals/pointer.

    The original control/github-primary-v1 pointer is never read or written by
    this state. Only the additive ledger and completed result/spec metadata are
    scanned. Fresh source trees, environments and raw market data are excluded.
    """
    def __init__(self, root, client, extension_root):
        import importlib.util
        module_path=Path(extension_root)/'scripts/e2_reduced_github_state.py'
        spec=importlib.util.spec_from_file_location('_e2_separate_posthoc_state',module_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.PREFIX='control/e2-guide-closure-posthoc-v1'
        module.POINTER=module.PREFIX+'/head.json'
        module.VERSION='e2-guide-closure-posthoc-state-v1'
        base=module.GitHubState
        class SeparateState(base):
            def _scan(self):
                selected=[self.output/'budget/original_anchor.json',self.output/'budget/posthoc_budget.jsonl',
                          self.output/'registration.json',self.output/'native_search/native_transverse_receipt.json',
                          self.output/'robustness/registration.json',self.output/'robustness/result.json']
                selected += list((self.output/'robustness/predictors').glob('*/result.json'))
                selected += list((self.output/'reproduction/jobs').glob('*/result.json'))
                selected += list((self.output/'reproduction/jobs').glob('*/worker_spec.json'))
                files={};data={}
                for path in sorted(selected):
                    if path.exists():
                        regular(path);raw=path.read_bytes();relative=str(path.relative_to(self.output))
                        files[relative]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};data[relative]=raw
                return files,data
            def restore(self):
                head=self.client._branch_head();current=self._current(head)
                if current is None:
                    files,_=self._scan()
                    if files:
                        raise RuntimeError('Local posthoc control has no remote anchor; preserve for reconciliation')
                    self.output.mkdir(parents=True,exist_ok=True);self._restored=True
                    return {'status':'new_additive_posthoc_campaign','budget_reset':False}
                result=super().restore()
                path=self.output/'budget/posthoc_budget.jsonl'
                active={}
                if path.exists():
                    for line in path.read_bytes().splitlines():
                        row=json.loads(line)
                        if row['kind']=='reserved':active[row['job_id']]=row['seconds']
                        elif row['kind']=='charged':active.pop(row['job_id'],None)
                if active:
                    raise RuntimeError('Remote posthoc reservation unresolved; no lost fit is retried or credited')
                return result
        self.state=SeparateState(regular(root),client)
    def __getattr__(self,name):
        return getattr(self.state.client,name)
    @property
    def _checked(self):
        return self.state.client._checked
    @_checked.setter
    def _checked(self,value):
        self.state.client._checked=value
    def restore(self):
        return self.state.restore()
    def snapshot(self,reason):
        return self.state.snapshot(reason)
    def persist_result(self,directory,receipt):
        return self.state.persist_result(directory,receipt)



def orchestrate(original_source, original_output, destination, existing_python, extension_root,
                configuration_commit, token, *, run_inference=True, run_training=True, launcher_elapsed_seconds=0.):
    """One user action: native searches, frozen diagnostics, then fresh fits."""
    invocation_start=time.monotonic();original_source=Path(original_source);destination=regular(destination)
    sys.path.insert(0,str(Path(extension_root)/'scripts'))
    sys.path.append(str(original_source/'scripts'))
    from e2_reduced_github_backup import GitHubBackup
    transport=GitHubBackup(token,timeout=20,max_conflicts=2)
    # Existing branch/release transport is established by the completed campaign.
    transport._branch_head();transport._checked=True
    remote=PosthocRemoteState(destination,transport,extension_root)
    remote.restore()  # Latest CAS state is authoritative, before new reservations.
    anchor=validate_original_anchor(original_output)
    registration=Path(extension_root)/'experiments/e2_posthoc_fit_reproduction_20261113/registration.json'
    published=json.loads(registration.read_bytes())
    if published.get('scientific_commit')!=ENGINE_COMMIT:
        raise ValueError('Posthoc registration points at a different scientific source')
    for relative,expected in published['files_sha256'].items():
        if file_sha(Path(extension_root)/relative)!=expected:
            raise ValueError('Published extension source bytes differ: '+relative)
    immutable_json(destination/'registration.json',{'configuration_commit':configuration_commit,
                    'registration_sha256':file_sha(registration),'record':published})
    with PosthocLedger(destination/'budget',anchor) as budget:
        starting_charges=budget.charges
        budget.set_persistence(remote)
        boundary=time.monotonic()
        budget.charge_overhead(boundary-invocation_start+launcher_elapsed_seconds,description='launcher_downloads_runtime_source_remote_restore_and_anchor_verification')
        gap=boundary
        search_path=destination/'native_search/native_transverse_receipt.json'
        if not search_path.exists():
            search_id='posthoc-native-search-'+str(time.time_ns())
            start=time.monotonic();budget.reserve(search_id,120,metadata={'scope':'three_native_API_queries_no_abstracts'})
            try:
                from e2_native_transverse_colab import retrieve
                retrieve(search_path)
                receipt={'status':'completed','attempt_id':'posthoc-native-search-v1',
                         'source_sha256':file_sha(Path(extension_root)/'scripts/e2_native_transverse_colab.py'),
                         'configuration_commit':configuration_commit,
                         'artifacts_sha256':{search_path.name:file_sha(search_path)}}
                backup_completed(search_path.parent,receipt,remote)
            except BaseException as error:
                budget.charge(search_id,time.monotonic()-start,
                              status='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed')
                raise
            finished=time.monotonic();budget.charge(search_id,finished-start,status='completed')
            gap=finished
        if run_inference:
            from e2_frozen_posthoc_gpu import run as frozen_run
            boundary=time.monotonic();budget.charge_overhead(boundary-gap,description='before_frozen_diagnostics')
            result=frozen_run(original_source,original_output,destination/'robustness',remote,
                              wall_seconds=5400,budget=budget,configuration_commit=configuration_commit)
            gap=time.monotonic()
            # run() closes its charge remotely through the shared persistence callback.
        if run_training:
            boundary=time.monotonic();budget.charge_overhead(boundary-gap,description='before_clean_fits')
            result=run_fits(original_source,original_output,destination/'reproduction',existing_python,
                            budget=budget,backup_client=remote,extension_root=extension_root,
                            configuration_commit=configuration_commit)
            gap=time.monotonic()
        actual_elapsed=time.monotonic()-invocation_start+launcher_elapsed_seconds
        accounted=budget.charges-starting_charges
        budget.charge_overhead(max(0.,actual_elapsed-accounted),description='all_invocation_elapsed_gap_reconciliation')
        # Conservatively precharge and hard-bound the final control publication.
        closure_id='posthoc-close-'+str(time.time_ns())
        previous_handler=signal.getsignal(signal.SIGALRM)
        def closure_timeout(signum,frame):
            raise TimeoutError('Final publication wall cap reached; durable reservation is preserved')
        signal.signal(signal.SIGALRM,closure_timeout);signal.setitimer(signal.ITIMER_REAL,30.)
        try:
            budget.reserve(closure_id,30,metadata={'scope':'final_publication_full_conservative_charge'})
            budget.charge(closure_id,30,status='completed',evidence={'charged_full_cap_conservatively':True})
            report=budget.report()
            remote.snapshot('posthoc_controller_closed_no_budget_reset')
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous_handler)
        print(json.dumps({'event':'posthoc_controller_closed','budget':report,
                          'actual_model_fits_only_if_completed_receipts':True}),flush=True)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-spec',type=Path)
    parser.add_argument('--worker-output',type=Path)
    parser.add_argument('--orchestrate',action='store_true')
    parser.add_argument('--original-source',type=Path)
    parser.add_argument('--original-output',type=Path)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--existing-python',type=Path)
    parser.add_argument('--extension-root',type=Path)
    parser.add_argument('--configuration-commit')
    parser.add_argument('--launcher-elapsed-seconds',type=float,default=0.)
    args=parser.parse_args(argv)
    if args.orchestrate:
        token=sys.stdin.readline().strip()
        if not token or any(c.isspace() for c in token):
            raise ValueError('GitHub credential must arrive only through controller STDIN')
        def terminate(signum,frame):
            raise KeyboardInterrupt('Controller interrupted; unknown reservations remain preserved')
        signal.signal(signal.SIGTERM,terminate)
        return orchestrate(args.original_source,args.original_output,args.destination,args.existing_python,
                           args.extension_root,args.configuration_commit,token,launcher_elapsed_seconds=args.launcher_elapsed_seconds)
    if args.worker_spec is None or args.worker_output is None:
        parser.error('Use the registered Colab launcher; CLI fitting is a bounded clean-worker entrypoint only')
    worker(args.worker_spec,args.worker_output)


if __name__=='__main__':
    main()
