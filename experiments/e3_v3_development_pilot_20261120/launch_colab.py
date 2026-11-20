"""Immutable-source Colab launcher for one V3 development pilot.

Call main(EXACT_40_HEX_SOURCE_COMMIT). After a Colab reset pass the printed
resume_run_id explicitly. This module performs no network or fitting on import.
Git credentials stay in this parent process; the scientific child receives none.
"""
from __future__ import annotations

from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import tempfile
import time
import urllib.request

VERSION = 'e3-v3-development-colab-launcher-1'
REPOSITORY = 'https://github.com/jassan-bit/DeepLearning_202630.git'
SCOPE = 'experiments/e3_v3_development_pilot_20261120'
RUNTIME = 'experiments/e3_remediacion_20261118/evaluation/colab_runtime.py'
PHASE = 'experiments/e3_evaluacion_ampliada_20261119/evaluation/phase.py'
QUADRATURE = 'experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json'
FROZEN_FILES = (RUNTIME, PHASE, QUADRATURE, QUADRATURE + '.sha256')
NPZ_COMMIT = 'af3f1223ad5899ed63016d2af2d09bae8470ab00'
NPZ_PATH = 'book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz'
NPZ_SHA256 = '186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683'
NPZ_BYTES = 460090
SHA = re.compile(r'[0-9a-f]{40}')
RUN = re.compile(r'[0-9]{8}T[0-9]{12}Z')
SECRET = re.compile(r'TOKEN|SECRET|PASSWORD|CREDENTIAL|(?:^|_)PAT(?:_|$)', re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(path, *, directory=False):
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), 'Symlink path refused')
    if path.exists():
        require(path.is_dir() if directory else path.is_file(), 'Unexpected filesystem object')
    return path


def atomic(path, value, *, immutable=False, raw=False):
    path = safe(path)
    content = value if raw else canonical(value)
    if immutable and path.exists():
        require(path.read_bytes() == content, 'Immutable launcher record changed: ' + path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-v3-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def environment(extra=None, token=''):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('GIT_', 'PIP_', 'UV_')) and not SECRET.search(key)
           and key not in {'SSH_ASKPASS', 'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'}
           and (not token or value != token)}
    env.update({key: value for key, value in (extra or {}).items()
                if not key.startswith('GIT_') and not SECRET.search(key)
                and (not token or value != token)})
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT='0', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    return env


def load(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Cannot load launcher dependency')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(args, *, cwd=None, data=None):
    return subprocess.run(['git', '-c', 'credential.helper=', '-c', 'core.hooksPath=/dev/null', *args],
                          cwd=cwd, input=data, env=environment(), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=180, check=True).stdout


def _read_retry(callback):
    for attempt in range(3):
        try:
            return callback()
        except (OSError, subprocess.SubprocessError):
            if attempt == 2:
                raise
            time.sleep(2 + attempt)


def checkout(source_commit, base):
    """Exact shallow metadata fetch, sparse source files, no model artifact checkout."""
    require(SHA.fullmatch(source_commit), 'Full immutable source SHA required')
    folder = safe(Path(base) / ('e3-v3-source-' + source_commit), directory=True)
    if not folder.exists():
        folder.mkdir(mode=0o700)
        _git(['init', '--quiet'], cwd=folder)
        _git(['remote', 'add', 'origin', REPOSITORY], cwd=folder)
        _git(['config', 'remote.origin.promisor', 'true'], cwd=folder)
        _git(['config', 'remote.origin.partialclonefilter', 'blob:none'], cwd=folder)
        _git(['sparse-checkout', 'init', '--no-cone'], cwd=folder)
        patterns = '/src/\n/' + SCOPE + '/\n' + ''.join('/' + name + '\n' for name in FROZEN_FILES)
        _git(['sparse-checkout', 'set', '--no-cone', '--stdin'], cwd=folder, data=patterns.encode())
        _read_retry(lambda: _git(['fetch', '--quiet', '--no-tags', '--depth=1', '--filter=blob:none',
                                 'origin', source_commit], cwd=folder))
        _read_retry(lambda: _git(['checkout', '--quiet', '--detach', source_commit], cwd=folder))
    require(_git(['rev-parse', 'HEAD'], cwd=folder).decode().strip() == source_commit,
            'Source checkout is incomplete or has another HEAD; preserve it')
    require(_git(['remote', 'get-url', 'origin'], cwd=folder).decode().strip() == REPOSITORY,
            'Source checkout remote changed')
    _git(['diff', '--exit-code', 'HEAD'], cwd=folder)
    extra = _git(['ls-files', '--others', '--exclude-standard'], cwd=folder).decode().splitlines()
    require(not any('__pycache__' not in Path(name).parts and not name.endswith('.pyc') for name in extra),
            'Unexpected files in source checkout; preserve and reconcile')
    return folder


def admit_npz(base):
    """Copy/hash the fixed archive; never load its calibration or test members."""
    path = safe(Path(base) / ('e3-training-source-' + NPZ_SHA256[:12] + '.npz'))
    if not path.exists():
        url = 'https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/' + NPZ_COMMIT + '/' + NPZ_PATH
        def read():
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read(NPZ_BYTES + 1)
        raw = _read_retry(read)
        require(len(raw) == NPZ_BYTES and hashlib.sha256(raw).hexdigest() == NPZ_SHA256,
                'Pinned NPZ archive differs; no fit was started')
        atomic(path, raw, immutable=True, raw=True)
    require(path.stat().st_size == NPZ_BYTES and sha(path) == NPZ_SHA256,
            'Cached NPZ archive changed; preserve it')
    return path


def credential():
    value = None
    try:
        from google.colab import userdata
        value = userdata.get('GITHUB_TOKEN')
    except Exception:
        pass
    if not value:
        value = getpass.getpass('GitHub token (Contents: Read and write; hidden): ')
    require(isinstance(value, str) and len(value) >= 20 and '\n' not in value and '\r' not in value,
            'A valid GitHub credential is required; it is never written to files')
    return value


def choose_run(base, source_commit, resume_run_id, *, new_run=False):
    session = safe(Path(base) / 'e3-v3-development-session.json')
    old = json.loads(session.read_text()) if session.exists() else None
    if resume_run_id is not None:
        require(RUN.fullmatch(resume_run_id), 'Exact printed resume_run_id required')
        if old:
            require(old == {'source_commit': source_commit, 'master_run_id': resume_run_id},
                    'Another local pilot is active; use its printed run identifier')
        run_id, restoring = resume_run_id, True
    elif old:
        require(not new_run and old['source_commit'] == source_commit,
                'An existing local pilot must be resumed with its original source')
        run_id, restoring = old['master_run_id'], False
    else:
        existing = _read_retry(lambda: _git(['ls-remote', '--heads', REPOSITORY,
                                             'refs/heads/e3-v3-development-*'])).decode().splitlines()
        require(new_run or not existing,
                'A V3 pilot already exists. Resume its printed run identifier; no new run was created. ' +
                ' '.join(row.rsplit('e3-v3-development-', 1)[-1] for row in existing[-8:]))
        run_id, restoring = stamp(), False
    require(RUN.fullmatch(run_id), 'Invalid session run identifier')
    atomic(session, {'source_commit': source_commit, 'master_run_id': run_id}, immutable=True)
    return run_id, restoring


PROFILE_CODE = '''import json,platform,subprocess,numpy,torch
assert torch.cuda.is_available()
torch.set_num_threads(2)
p=torch.cuda.get_device_properties(0)
x=torch.ones(2,device="cuda:0",dtype=torch.float64)
assert (x*x).sum().item()==2.
print(json.dumps({"python":platform.python_version(),"numpy":numpy.__version__,"torch":torch.__version__,
"cuda_runtime":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"cpu_threads":torch.get_num_threads(),
"GPU":{"name":p.name,"total_memory_bytes":p.total_memory,"capability":list(torch.cuda.get_device_capability(0))},
"driver":subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).strip(),
"GPU_execution_verified":True,"synthetic_tensor_check_only":True}))
'''


def hardware_profile(python, env):
    raw = subprocess.check_output([str(python), '-c', PROFILE_CODE], env=env, timeout=45)
    return json.loads(raw)


def launcher_lock(source, source_commit):
    names = [SCOPE + '/launch_colab.py', SCOPE + '/git_checkpoint_store.py',
             SCOPE + '/pilot_engine.py', SCOPE + '/plugin_control.py', *FROZEN_FILES,
             'src/original_model.py', 'src/e3_development.py', 'src/e3_innovation_model.py',
             'src/e3_overlap_model_v3.py']
    return {'version': VERSION, 'source_commit': source_commit,
            'source_files': {name: sha(source / name) for name in names},
            'archive': {'commit': NPZ_COMMIT, 'path': NPZ_PATH, 'bytes': NPZ_BYTES, 'sha256': NPZ_SHA256},
            'scope': 'exposed_development_only', 'old_796_master_mutated': False,
            'R1_accredited': False, 'R3_accredited': False}


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def run_worker(command, *, source, env, output, store, durability_dir, token):
    """Remote barrier before optimizer, each completed job, and first valid checkpoint."""
    process = subprocess.Popen(command, cwd=source, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, bufsize=0)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    buffer = b''
    try:
        with (output / 'worker.log').open('ab', buffering=0) as log:
            while selector.get_map():
                ready = selector.select(timeout=30)
                if not ready:
                    print('Piloto activo; el plazo de cada ajuste permanece fijado en 120 s.', flush=True)
                    continue
                block = os.read(process.stdout.fileno(), 65536)
                if not block:
                    selector.unregister(process.stdout)
                    break
                buffer += block
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    require(not token or token.encode() not in line, 'Credential unexpectedly reached worker output')
                    log.write(line + b'\n')
                    text = line.decode('utf-8', 'replace')
                    print(text, flush=True)
                    try:
                        event = json.loads(text)
                    except (ValueError, TypeError):
                        continue
                    if event.get('event') == 'durability_barrier':
                        barrier = event.get('barrier_id')
                        require(isinstance(barrier, str) and re.fullmatch('[0-9a-f]{64}', barrier),
                                'Malformed durability barrier; preserve evidence')
                        receipt = store.publish(output)
                        atomic(durability_dir / (barrier + '.json'),
                               {'barrier_id': barrier, 'remote_verified': True,
                                'git_commit': receipt['git_commit']})
                        print('Checkpoint remoto verificado:', receipt['run_url'], flush=True)
            if buffer:
                require(not token or token.encode() not in buffer, 'Credential unexpectedly reached worker output')
                log.write(buffer)
                print(buffer.decode('utf-8', 'replace'), flush=True)
        code = process.wait(timeout=30)
        if code:
            try:
                store.publish(output)
            except Exception:
                pass
            raise RuntimeError('Pilot worker exited with status ' + str(code) +
                               '; preserve this run and its original source. No fresh campaign was created.')
        return store.publish(output)
    except BaseException:
        _stop(process)
        raise
    finally:
        selector.close()
        process.stdout.close()


def main(source_commit, *, base='/content', resume_run_id=None, new_run=False, token=None):
    """Start/resume the declared development pilot; no change to the old campaign."""
    require(isinstance(source_commit, str) and SHA.fullmatch(source_commit), 'Full immutable source SHA required')
    base = safe(base, directory=True)
    base.mkdir(parents=True, exist_ok=True)
    source = checkout(source_commit, base)
    store_module = load(source / SCOPE / 'git_checkpoint_store.py', 'e3_v3_git_checkpoint_store')
    run_id, restoring = choose_run(base, source_commit, resume_run_id, new_run=new_run)
    output = safe(base / ('e3-v3-development-' + run_id), directory=True)
    token = credential() if token is None else token
    store = store_module.GitCheckpointStore(base / ('e3-v3-publisher-' + run_id), source_commit, run_id, token)
    if restoring and (not output.exists() or not any(output.iterdir())):
        store.restore(output)
    elif store.state['sequence'] > 0:
        require(output.exists() and any(output.iterdir()), 'Local evidence is missing; use explicit remote restore')
        store._inventory(output)
    output.mkdir(exist_ok=True)
    identity = {'source_commit': source_commit, 'master_run_id': run_id, 'scope': 'development_only'}
    atomic(output / 'launcher_identity.json', identity, immutable=True)
    atomic(output / 'launcher_lock.json', launcher_lock(source, source_commit), immutable=True)
    print('V3 desarrollo: mismo código y plan al reanudar; no modifica el master anterior.', flush=True)
    print('Para reanudar tras reiniciar Colab: resume_run_id=' + repr(run_id), flush=True)
    print('Rama de evidencia:', store.branch, flush=True)
    runtime = load(source / RUNTIME, 'e3_v3_existing_pinned_runtime')
    python, gpu_env, runtime_receipt = runtime.prepare_runtime(base / 'e3-preflight-gpu-runtime-v1')
    env = environment(gpu_env, token)
    profile = hardware_profile(python, env)
    atomic(output / 'hardware_profile.json', profile, immutable=True)
    if not (output / 'isolated_runtime.json').exists():
        atomic(output / 'isolated_runtime.json', runtime_receipt, immutable=True)
    npz = admit_npz(base)
    command = [str(python), str(source / SCOPE / 'pilot_engine.py'), '--source-root', str(source),
               '--source-commit', source_commit, '--plugin', str(source / SCOPE / 'plugin_control.py'),
               '--npz', str(npz), '--npz-sha256', NPZ_SHA256, '--output', str(output), '--device', 'cuda:0']
    subprocess.run(command + ['--prepare-only'], cwd=source, env=env, check=True, timeout=120)
    plan = json.loads((output / 'plan.json').read_text())
    admission = json.loads((output / 'admission.json').read_text())
    require(plan['master_run_id'] == run_id, 'Engine and publisher run identities differ; no fit started')
    receipt = store.publish(output)
    engine = load(source / SCOPE / 'pilot_engine.py', 'e3_v3_engine_functions_only')
    published = {**receipt, 'plan_sha256': engine.digest(plan), 'admission_sha256': engine.digest(admission)}
    receipt_path = output / 'publication_receipt.json'
    if not receipt_path.exists():
        atomic(receipt_path, published, immutable=True)
    else:
        previous = json.loads(receipt_path.read_text())
        require(all(previous[key] == published[key] for key in
                    ('source_commit', 'master_run_id', 'plan_sha256', 'admission_sha256')),
                'Original publication receipt identity differs')
    receipt = store.publish(output)
    print('Plan y procedencia guardados y verificados antes del primer ajuste:', receipt['run_url'], flush=True)
    durability = safe(base / ('e3-v3-durability-' + run_id), directory=True)
    durability.mkdir(exist_ok=True)
    result = run_worker(command + ['--run', '--publication-receipt', str(receipt_path),
                                   '--durability-dir', str(durability)], source=source, env=env,
                        output=output, store=store, durability_dir=durability, token=token)
    print('Piloto de desarrollo terminado; R1 y R3 no se acreditan automáticamente.', flush=True)
    print('Evidencia:', result['run_url'], flush=True)
    return result
