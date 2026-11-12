"""Whole-worker deadline and shared-budget closure for reduced E2 fits.

A completed result is reusable only with its matching persisted budget charge.
Unclosed leases and interrupted artifact attempts are never replayed here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


class ReducedWorkerFailure(RuntimeError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_worker_result(directory, spec_sha, *, result_filename='result.json'):
    root = Path(directory)
    path = root / result_filename
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get('status') != 'completed' or result.get('worker_spec_sha256') != spec_sha:
        raise ValueError('Saved worker result has another identity or noncompleted status')
    artifacts = result.get('artifacts_sha256')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('Completed worker must bind every scientific artifact')
    for relative, expected in artifacts.items():
        target = root / relative
        if (not target.resolve().is_relative_to(root.resolve()) or target.is_symlink()
                or not target.is_file() or file_sha(target) != expected):
            raise ValueError('Worker artifact is missing or changed: ' + relative)
    return result


def supervise_command(command, budget, logical_id, output, spec_sha, phases=1, *,
                      cwd=None, result_filename='result.json', env=None):
    """Reserve once, supervise an isolated child, persist actual elapsed charge.

    Returns a mapping with ``result`` and the matching charge receipt. The worker
    writes result.json with status/completed, worker_spec_sha256 and a nonempty
    artifacts_sha256 map. A deadline kills its process group before releasing
    the lease; it never changes a training configuration or retries an attempt.
    """
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError('Supply an explicit worker argument array')
    if not isinstance(spec_sha, str) or len(spec_sha) != 64:
        raise ValueError('A complete worker specification hash is required')
    output = Path(output)
    if output.is_symlink():
        raise ValueError('Worker directory must not be a symlink')
    output.mkdir(parents=True, exist_ok=True)
    prior = verified_worker_result(output, spec_sha, result_filename=result_filename)
    if prior is not None:
        result_sha = file_sha(output / result_filename)
        receipt = budget.completed_receipt(logical_id, expected_resource_use={
            'worker_spec_sha256': spec_sha, 'result_sha256': result_sha})
        if receipt is None:
            raise RuntimeError('reconciliation_required: completed worker lacks its matching budget charge')
        return {'result': prior, 'result_sha256': result_sha,
                'charged_wall_seconds': receipt.get('elapsed_seconds'), 'budget_receipt': receipt}
    reservation = budget.reservation_for(phases=phases)
    worker_timeout = budget.worker_timeout_for(phases=phases) if callable(getattr(budget, 'worker_timeout_for', None)) else max(.001, reservation - 5.)
    budget.reserve(logical_id, reservation, metadata={'logical_job_id': logical_id,
        'worker_spec_sha256': spec_sha, 'phases': phases})
    child_env = dict(os.environ if env is None else env)
    for key in ('GITHUB_TOKEN', 'GH_TOKEN', 'PYTHONPATH'):
        child_env.pop(key, None)
    child_env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONNOUSERSITE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    started = time.monotonic()
    process = None
    failure = None
    charge_attempted = False
    try:
        with (output / 'worker.log').open('ab') as log, (output / 'worker.log').open('rb') as tail:
            process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=child_env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                chunk = tail.read()
                if chunk:
                    print(chunk.decode('utf-8',errors='replace'),end='',flush=True)
                remaining = worker_timeout - (time.monotonic() - started)
                if remaining <= 0:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    failure = 'whole_worker_deadline_exceeded'
                    break
                try:
                    process.wait(timeout=min(1., remaining))
                except subprocess.TimeoutExpired:
                    pass
            chunk = tail.read()
            if chunk:
                print(chunk.decode('utf-8',errors='replace'),end='',flush=True)
        if failure is None and process.returncode != 0:
            failure = 'worker_exit_' + str(process.returncode)
        result = None if failure else verified_worker_result(output, spec_sha, result_filename=result_filename)
        if failure is None and result is None:
            failure = 'worker_exited_without_completed_result'
        elapsed = time.monotonic() - started
        if failure:
            usage = {'worker_spec_sha256': spec_sha, 'failure': failure,
                'optimizer_steps_unknown': True, 'automatic_retry': False}
            charge_attempted = True
            receipt = budget.charge(logical_id, elapsed, status='failed', resource_use=usage)
            raise ReducedWorkerFailure(failure, {'budget_receipt': receipt, **usage,
                'charged_wall_seconds': elapsed, 'log': str(output / 'worker.log')})
        result_sha = file_sha(output / result_filename)
        charge_attempted = True
        receipt = budget.charge(logical_id, elapsed, status='completed', resource_use={
            'worker_spec_sha256': spec_sha, 'result_sha256': result_sha,
            'optimizer_steps': result.get('optimizer_steps'), 'automatic_retry': False})
        return {'result': result, 'result_sha256': result_sha,
            'charged_wall_seconds': elapsed, 'budget_receipt': receipt}
    except ReducedWorkerFailure:
        raise
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        # Actual child termination is observed, so its measured cost is known.
        # A failure to persist this charge leaves the lease unresolved on disk.
        elapsed = time.monotonic() - started
        if not charge_attempted:
            budget.charge(logical_id, elapsed, status='failed', resource_use={
            'worker_spec_sha256': spec_sha, 'failure': type(error).__name__,
            'optimizer_steps_unknown': True, 'automatic_retry': False})
        raise
