"""Bounded optimizer phase for the common extended-domain audit.

Prepared with declared AI assistance. The caller divides ONE system budget
between SAE's three phases; this function never grants three full budgets.
The immutable wall deadline includes setup, checkpoint I/O and restart downtime.
A running GPU batch cannot be preempted here: overruns at batch/validation
boundaries are reported, and this is not a process hard timeout.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time


class PhaseIncomplete(RuntimeError):
    """No validated trained checkpoint exists; never return random weights."""


def allocate_phase_budgets(system_seconds, names):
    """Equal predeclared shares of ONE total, including all SAE optimizer phases."""
    if (type(system_seconds) not in (int, float) or not math.isfinite(system_seconds)
            or system_seconds <= 0 or not names or len(names) != len(set(names))
            or any(not isinstance(name, str) or not name for name in names)):
        raise ValueError('Positive system budget and distinct named phases required')
    return {name: float(system_seconds)/len(names) for name in names}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _directory_sync(directory):
    if hasattr(os, 'O_DIRECTORY'):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        _directory_sync(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _torch_save(path, value):
    import torch
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _directory_sync(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _clone(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _rng():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda']:
        torch.cuda.set_rng_state_all(value['cuda'])


def _sync(x):
    if x.device.type == 'cuda':
        import torch
        torch.cuda.synchronize(x.device)


def _check_config(config, budget_seconds):
    if not isinstance(config, dict):
        raise ValueError('Explicit common training configuration required')
    for key in ('max_epochs', 'patience', 'batch_size'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError('Positive integer training limit required: ' + key)
    for key in ('learning_rate', 'gradient_clip'):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite training setting required: ' + key)
    if type(budget_seconds) not in (int, float) or not math.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError('Positive explicit phase budget required')
    decay = config.get('weight_decay', 0.)
    if type(decay) not in (int, float) or not math.isfinite(decay) or decay < 0:
        raise ValueError('Finite nonnegative weight decay required')


def _receipt(directory, state, marker, checkpoint):
    """Keep the first completion ledger stable across a completed-run retry."""
    path = Path(directory) / 'receipt.json'
    fingerprint = sha(checkpoint)
    if path.exists():
        previous = json.loads(path.read_text())
        if (previous.get('checkpoint_sha256') == fingerprint
                and previous.get('request_sha256') == state['request_sha256']):
            return previous
    finished = time.time()
    result = {'status': 'completed', 'request_sha256': state['request_sha256'],
              'best_epoch': state['best_epoch'], 'epochs': state['epoch'],
              'optimizer_steps': state['steps'], 'best_validation_mse_scaled': state['best_score'],
              'checkpoint_sha256': fingerprint, 'stop_reason': state['stop_reason'],
              'budget_seconds': marker['budget_seconds'], 'started_unix': marker['started_unix'],
              'immutable_deadline_unix': marker['deadline_unix'], 'finalized_unix': finished,
              'elapsed_wall_including_setup_io_and_restart_seconds': finished-marker['started_unix'],
              'deadline_overrun_seconds': max(0., finished-marker['deadline_unix']),
              'measured_completed_epoch_seconds': state['measured_completed_epoch_seconds'],
              'resume_segments': state['resume_segments'],
              'active_compute_lost_in_an_unclosed_crash_segment_is_unknown': state['resume_segments'] > 0,
              'deadline_includes_restart_downtime': True,
              'wall_budget_is_process_hard_timeout': False,
              'budget_check_boundary': 'before each minibatch; a batch and its validation/checkpoint may overrun',
              'epoch_boundary_resume': True,
              'partial_epoch_policy': 'restore previous durable model/optimizer/RNG; deadline is never reset',
              'untrained_checkpoint_accepted': False}
    write_json(path, result)
    return result


def phase(model, x, y, vx, vy, directory, config, *, identity, loss_fn=None, budget_seconds):
    """Fit one bounded phase and restore its best validated trained weights.

    ``loss_fn(model, indices)`` trains; ``indices=None`` validates. If omitted,
    exact-shape MSE is used. Callers must supply one predeclared budget allocation
    per phase whose SUM does not exceed the common whole-system allowance.
    """
    import torch
    _check_config(config, budget_seconds)
    if not isinstance(identity, dict):
        raise ValueError('Explicit source/model/phase identity required')
    if (len(x) == 0 or len(vx) == 0 or len(x) != len(y) or len(vx) != len(vy)
            or x.device != y.device or vx.device != vy.device or x.device != vx.device):
        raise ValueError('Nonempty aligned training/validation tensors on one device required')
    if not all(bool(torch.isfinite(value).all()) for value in (x, y, vx, vy)):
        raise ValueError('Nonfinite phase input')
    directory = Path(directory).absolute()
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise ValueError('Phase directory cannot traverse a symlink')
    directory.mkdir(parents=True, exist_ok=True)
    request = {'identity': identity, 'config': config, 'budget_seconds': float(budget_seconds),
               'train_shape': list(x.shape), 'validation_shape': list(vx.shape),
               'torch': torch.__version__, 'device': str(x.device)}
    req = digest(request)
    marker_path = directory / 'phase_started.json'
    checkpoint = directory / 'checkpoint.pt'
    if marker_path.is_symlink() or checkpoint.is_symlink():
        raise ValueError('Phase markers and checkpoints must be regular files')
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if (marker.get('request_sha256') != req
                or marker.get('kind') != 'e3_extended_optimizer_phase_started'
                or marker.get('budget_seconds') != float(budget_seconds)
                or marker.get('source_identity') != identity):
            raise ValueError('Immutable phase request differs; old bytes preserved')
        started, deadline = marker.get('started_unix'), marker.get('deadline_unix')
        if (type(started) not in (int, float) or type(deadline) not in (int, float)
                or not math.isfinite(started) or not math.isfinite(deadline)
                or not math.isclose(deadline-started, float(budget_seconds), rel_tol=0., abs_tol=1e-5)):
            raise ValueError('Immutable phase deadline differs from its allocated budget')
    else:
        now = time.time()
        marker = {'kind': 'e3_extended_optimizer_phase_started', 'request_sha256': req,
                  'source_identity': identity, 'budget_seconds': float(budget_seconds),
                  'started_unix': now, 'deadline_unix': now+float(budget_seconds),
                  'counter_scope': 'attempted phase; no optimizer step claimed by marker alone'}
        with marker_path.open('x') as stream:
            stream.write(json.dumps(marker, indent=2, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        _directory_sync(directory)
        write_json(directory / 'request.json', request)
    resumed = checkpoint.exists()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False) if resumed else None
    if state is not None:
        if state.get('request_sha256') != req or state.get('phase_marker_sha256') != sha(marker_path):
            raise ValueError('Checkpoint request differs')
        if state['completed']:
            if state['best_score'] is None or state['steps'] < 1:
                raise PhaseIncomplete('Completed phase has no validated training update')
            # A completed AE phase may be reused before the next phase starts.
            # Restore the phase-exit RNG as well as weights so the following
            # reconstruction phase sees the uninterrupted minibatch stream.
            _restore_rng(state['rng'])
            model.load_state_dict(state['best'])
            state['resource_receipt'] = _receipt(directory, state, marker, checkpoint)
            return state
    if time.time() >= marker['deadline_unix'] and (state is None or state['best_score'] is None):
        raise PhaseIncomplete('Phase deadline expired before any validated trained checkpoint')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config.get('weight_decay', 0.))
    if state is None:
        state = {'request_sha256': req, 'phase_marker_sha256': sha(marker_path),
                 'epoch': 0, 'best_epoch': 0, 'best_score': None,
                 'bad_epochs': 0, 'steps': 0, 'best': _clone(model), 'latest': _clone(model),
                 'optimizer': optimizer.state_dict(), 'rng': _rng(), 'curve': [],
                 'completed': False, 'stop_reason': None,
                 'measured_completed_epoch_seconds': 0., 'resume_segments': 0}
        _torch_save(checkpoint, state)
    else:
        model.load_state_dict(state['latest'])
        optimizer.load_state_dict(state['optimizer'])
        for group in optimizer.state.values():
            for key, value in group.items():
                if torch.is_tensor(value):
                    group[key] = value.to(x.device)
        _restore_rng(state['rng'])
        state['resume_segments'] += 1
    with (directory / 'segments.log').open('a') as stream:
        stream.write(canonical({'event': 'segment_start', 'resume_epoch': state['epoch'],
                                'at_unix': time.time(), 'immutable_deadline_unix': marker['deadline_unix']})+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    if time.time() >= marker['deadline_unix']:
        state.update(completed=True, stop_reason='wall_budget_expired_on_resume')
        _torch_save(checkpoint, state)
    for epoch in range(state['epoch']+1, config['max_epochs']+1):
        if state['completed']:
            break
        _sync(x)
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(x), device=x.device)
        total = 0.
        samples = 0
        steps_this_epoch = 0
        budget_hit = False
        for offset in range(0, len(x), config['batch_size']):
            if time.time() >= marker['deadline_unix']:
                budget_hit = True
                break
            indexes = order[offset:offset+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            if loss_fn is None:
                prediction = model(x[indexes])
                if prediction.shape != y[indexes].shape:
                    raise ValueError('Prediction/target shape mismatch; broadcasting refused')
                loss = torch.mean((prediction-y[indexes])**2)
            else:
                loss = loss_fn(model, indexes)
            if not torch.is_tensor(loss) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError('Training loss must be a finite scalar tensor')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            _sync(x)
            total += float(loss.detach())*len(indexes)
            samples += len(indexes)
            steps_this_epoch += 1
            state['steps'] += 1
        if steps_this_epoch == 0:
            if state['best_score'] is None:
                raise PhaseIncomplete('Phase exhausted its budget without a validated training update')
            state.update(completed=True, stop_reason='wall_budget')
            _torch_save(checkpoint, state)
            break
        model.eval()
        with torch.no_grad():
            if loss_fn is None:
                prediction = model(vx)
                if prediction.shape != vy.shape:
                    raise ValueError('Validation prediction/target shape mismatch')
                score = float(torch.mean((prediction-vy)**2))
            else:
                score = float(loss_fn(model, None))
        _sync(x)
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation score')
        duration = time.perf_counter()-started
        if state['best_score'] is None or score < state['best_score']:
            state.update(best_score=score, best_epoch=epoch, best=_clone(model), bad_epochs=0)
        else:
            state['bad_epochs'] += 1
        state['curve'].append({'epoch': epoch, 'train_loss': total/samples,
                               'validation_mse_scaled': score, 'seconds': duration,
                               'samples_seen': samples, 'optimizer_steps': steps_this_epoch,
                               'full_epoch_completed': samples == len(x)})
        state.update(epoch=epoch, latest=_clone(model), optimizer=optimizer.state_dict(), rng=_rng())
        state['measured_completed_epoch_seconds'] += duration
        if budget_hit or time.time() >= marker['deadline_unix']:
            state.update(completed=True, stop_reason='wall_budget')
        elif state['bad_epochs'] >= config['patience']:
            state.update(completed=True, stop_reason='early_stopping')
        elif epoch == config['max_epochs']:
            state.update(completed=True, stop_reason='max_epochs')
        _torch_save(checkpoint, state)
        write_json(directory / 'curve.json', state['curve'])
    if not state['completed'] or state['best_score'] is None or state['steps'] < 1:
        raise PhaseIncomplete('No completed validated trained phase; no fallback')
    model.load_state_dict(state['best'])
    state['resource_receipt'] = _receipt(directory, state, marker, checkpoint)
    return state
