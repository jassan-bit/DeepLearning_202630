"""Durable reduced-recipe fits, including all three published SAE phases.

No market loader or budget extension is supplied. The caller owns admission,
shared-budget reservation and the hard whole-worker deadline. CPU execution is
an explicit synthetic-test route. Every completed phase saves weights, curves,
optimizer/RNG states and actual fitting-policy/resource receipts.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

VERSION = 'e2-reduced-durable-fit-v1'
SAE = 'LSTM-autoencoder con predictor'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value) + b'\n'
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError('Immutable reduced artifact changed: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())



def verify_phase_archive(directory):
    """Verify a completed compact archive; never claim checkpoint-resume support."""
    from e2_amended_pilot_artifacts import PilotArtifactStore
    directory=Path(directory)
    identity=json.loads((directory/'identity.json').read_text())
    reader=object.__new__(PilotArtifactStore)
    reader.attempt_id=identity['attempt_id'];reader.identity_sha256=identity['identity_sha256']
    if digest(identity['identity'])!=reader.identity_sha256:
        raise ValueError('Phase archive identity changed')
    state=reader._empty_state()
    for line in (directory/'events.jsonl').read_bytes().splitlines():
        row=json.loads(line);bound=row.pop('sha256')
        if (bound!=digest(row) or row['sequence']!=state['event_count']
                or row['previous_sha256']!=state['last_event_sha256']
                or row['identity_sha256']!=reader.identity_sha256):
            raise ValueError('Phase event chain changed')
        row['sha256']=bound;state=reader._advance(state,row)
    if state!=json.loads((directory/'manifest.json').read_text()) or state['state']!='completed':
        raise ValueError('Only a fully closed phase may use compact archival')
    if digest(json.loads((directory/'prepared.json').read_text()))!=state['prepared_metadata_sha256']:
        raise ValueError('Phase prepared metadata changed')
    retired=json.loads((directory/'superseded_checkpoint_hashes.json').read_text())
    history={r['path']:r for r in state['checkpoint_history']}
    keep={r['path'] for r in state['checkpoints'].values()}
    removed=retired['deleted_checkpoint_hashes']
    if set(removed)!=set(history)-keep or set(retired['current_checkpoint_paths'])!=keep:
        raise ValueError('Compact archive inventory differs from checkpoint history')
    for relative,receipt in history.items():
        if relative in removed:
            if removed[relative]!=receipt['sha256'] or (directory/relative).exists():
                raise ValueError('Superseded checkpoint disclosure changed')
        elif file_sha(directory/relative)!=receipt['sha256']:
            raise ValueError('Retained phase checkpoint changed')
    return True


def verified_fit(directory, request):
    directory = Path(directory)
    saved = directory / 'job_result.json'
    if not saved.exists():
        return None
    value = json.loads(saved.read_text())
    if value.get('request_sha256') != digest(request) or value.get('status') != 'completed':
        raise ValueError('Saved reduced fit has a different request or incomplete status')
    for relative, expected in value.get('artifacts_sha256', {}).items():
        path = directory / relative
        if (not path.resolve().is_relative_to(directory.resolve()) or path.is_symlink()
                or not path.is_file() or file_sha(path) != expected):
            raise ValueError('Reduced fit artifact changed: ' + relative)
    for phase in value.get('phase_records', {}):
        verify_phase_archive(directory / 'phases' / phase)
    if not value.get('artifacts_sha256'):
        raise ValueError('Reduced fit has no bound artifacts')
    return value


def _phase_fit(model, x, y, validation_x, validation_y, config, store, *, supervised,
               target_scale, fixed_epochs, device, allow_cpu_for_tests, gpu_limit):
    """Reuse the frozen durable optimizer math; admit auxiliary geometry explicitly."""
    import torch
    import e2_cuda_training as cuda
    import e2_staged_training as historical
    import e2_published_pilot_training as durable
    selected = cuda._resolve_device(device, allow_cpu_for_tests)
    cuda._validate_gpu_limit(gpu_limit)
    cuda._validate_model(model)
    x, y = historical._pairs(x, y, supervised=supervised, training=True)
    if fixed_epochs is None:
        if validation_x is None or validation_y is None:
            raise ValueError('HPO requires an explicit validation pair')
        validation_x, validation_y = historical._pairs(validation_x, validation_y,
            supervised=supervised, training=False)
    elif (type(fixed_epochs) is not int or not 1 <= fixed_epochs <= config.max_epochs
            or validation_x is not None or validation_y is not None):
        raise ValueError('Fixed refit accepts only admitted epoch counts and no validation')
    with cuda._execution_policy(selected, True) as runtime:
        durable._store_ready(store, {'trainer_version': VERSION, 'config': asdict(config),
            'fixed_epochs': fixed_epochs, 'target_scale': target_scale,
            'device': str(selected), 'test_only_cpu_execution': selected.type == 'cpu',
            'auxiliary_reconstruction': not supervised, 'retry_performed': False,
            'resume_exact_supported': False})
        return durable._fit(model, x, y, validation_x, validation_y, config, store,
            fixed_epochs=fixed_epochs, selected=selected, target_scale=target_scale,
            gpu_memory_limit_bytes=gpu_limit, runtime=runtime, step_events=False)


def _builder(name, trial, seed, roots):
    if name == 'TiDE' and trial.get('recipe_id') == 'TIDE-D7-PUBLISHED-R0-v2':
        import e2_published_tide as tide
        if trial not in tide.registered_trials():
            raise ValueError('Unregistered published TiDE point')
        return tide.build_tide(seed=seed, source_roots=roots, trial=trial)
    from e2_published_registry import build_train_model
    return build_train_model(name, trial, seed=seed, source_roots=roots)


def fit_reduced_job(name, trial, seed, train, validation, output, *, fixed_epochs=None,
                    source_roots=None, device='cuda:0', lease=None, source_identity=None,
                    runtime_identity=None, fit_wall_seconds=7200, allow_cpu_for_tests=False):
    import torch
    import e2_staged_training as historical
    from e2_amended_pilot_artifacts import PilotArtifactStore
    from e2_published_pilot_training import predict_pilot
    if type(seed) is not int or seed < 0:
        raise ValueError('Explicit nonnegative seed required')
    if device != 'cuda:0' and not allow_cpu_for_tests:
        raise ValueError('Production reduced fitting requires cuda:0')
    if type(fit_wall_seconds) not in (int, float) or not math.isfinite(fit_wall_seconds) or fit_wall_seconds <= 0:
        raise ValueError('Explicit positive registered per-phase wall limit required')
    phases = ('ae1', 'ae2', 'supervised') if name == SAE else ('supervised',)
    if (validation is None) != (fixed_epochs is not None):
        raise ValueError('Use either validation selection or a fully frozen epoch refit')
    if fixed_epochs is not None and (not isinstance(fixed_epochs, dict) or set(fixed_epochs) != set(phases)
            or any(type(v) is not int or not 1 <= v <= 200 for v in fixed_epochs.values())):
        raise ValueError('Frozen phase counts must account for every optimizer stage')
    config = historical.TrainConfig(learning_rate=trial['learning_rate'], weight_decay=trial['weight_decay'],
        batch_size=trial['batch_size'], max_epochs=200, patience=20, seed=seed,
        warmup_fraction=trial.get('warmup_fraction', 0), wall_limit_seconds=fit_wall_seconds,
        memory_limit_bytes=12 * 1024**3)
    if callable(getattr(train, 'validate', None)):
        train.validate()
    if validation is not None:
        validation.validate()
        if validation.scaler.record() != train.scaler.record():
            raise ValueError('Validation must use the training-only scaler')
    if name == 'TiDE':
        import e2_published_tide as tide
        allowed_trials = tide.registered_trials()
    else:
        from e2_published_registry import registered_trials
        allowed_trials = registered_trials(name)
    if trial not in allowed_trials:
        raise ValueError('Use an exact registered current-recipe optimizer point')
    if not allow_cpu_for_tests and (not source_identity or not runtime_identity):
        raise ValueError('Production fits need bound sources and actual runtime identity')
    sources = dict(source_identity or {'scope': 'caller_must_bind_sources_before_scientific_execution'})
    resources = dict(runtime_identity or {'device': device, 'test_only': allow_cpu_for_tests})
    request = {'version': VERSION, 'name': name, 'trial': trial, 'seed': seed,
        'train_signature': train.signature, 'validation_signature': None if validation is None else validation.signature,
        'fixed_epochs': fixed_epochs, 'config': asdict(config), 'sources': sources, 'resources': resources,
        'test_only_cpu_execution': allow_cpu_for_tests}
    directory = Path(output)
    if directory.is_symlink():
        raise ValueError('Fit directory must be regular')
    directory.mkdir(parents=True, exist_ok=True)
    previous = verified_fit(directory, request)
    if previous is not None:
        return previous
    if (directory / 'request.json').exists():
        raise RuntimeError('reconciliation_required: an interrupted/failed fit is never restarted automatically')
    write_json(directory / 'request.json', request)
    records = {}
    gpu_limit = None if allow_cpu_for_tests else int(resources['total_memory_bytes'] * .8)
    train_data = {'train': train.record(), 'validation': None if validation is None else validation.record()}
    identity = {'request_sha256': digest(request), 'seed': seed}

    def run_phase(phase, model, x, y, vx, vy, target_scale, supervised):
        store = PilotArtifactStore(directory / 'phases', phase, {**identity, 'phase': phase})
        try:
            store.prepare({'config': asdict(config), 'source': sources, 'data': train_data,
                'seed': seed, 'resources': resources})
            result = _phase_fit(model, x, y, vx, vy, config, store,
                supervised=supervised, target_scale=target_scale,
                fixed_epochs=None if fixed_epochs is None else fixed_epochs[phase],
                device=device, allow_cpu_for_tests=allow_cpu_for_tests, gpu_limit=gpu_limit)
            store.complete({'phase': phase, 'optimizer_steps': result.optimizer_steps,
                'selected_state_sha256': historical.state_sha256(model)})
            records[phase] = result.receipt
        finally:
            store.close()
        # Reserved snapshots are versioned for crash safety. Once a phase
        # closes, keep its current initial/latest/best snapshots and disclose
        # hashes of superseded epoch versions; curves and the event chain stay.
        manifest = json.loads((directory / 'phases' / phase / 'manifest.json').read_text())
        keep = {item['path'] for item in manifest['checkpoints'].values()}
        retired = {}
        for path in (directory / 'phases' / phase / 'checkpoints').glob('*.pt'):
            relative = str(path.relative_to(directory / 'phases' / phase))
            if relative not in keep:
                retired[relative] = file_sha(path)
                path.unlink()
        write_json(directory / 'phases' / phase / 'superseded_checkpoint_hashes.json',
            {'reason': 'After successful phase closure preserve initial/latest/best plus every epoch curve; older weight versions are not retained',
             'deleted_checkpoint_hashes': retired, 'current_checkpoint_paths': sorted(keep)})
        verify_phase_archive(directory / 'phases' / phase)
        return result

    vx = None if validation is None else validation.x
    vy = None if validation is None else validation.y
    if name == SAE:
        from e2_recipe_models import SAEPretraining, repeat7, last_state
        from e2_published_sae import _annotate, _verify_source_and_dropout
        _verify_source_and_dropout(trial['dropout'])
        historical._seed(seed)
        manager = SAEPretraining(train.x.to(device), train.signature, accept_auxiliary_scale=True)
        x, y = manager.phase1_data()
        validation_reconstruction = None if vx is None else manager.reconstruction_target(vx)
        run_phase('ae1', manager.ae1, x, y, vx, validation_reconstruction, 1., False)
        x2, y2 = manager.prepare_phase2()
        encoder_before = historical.state_sha256(manager.ae1.encoder)
        if not torch.equal(y, y2):
            raise RuntimeError('AE phases must reconstruct the same original inputs')
        validation_encoded = None
        if vx is not None:
            with torch.no_grad():
                validation_encoded = repeat7(last_state(manager.ae1.encoder, vx.to(device))).detach().cpu().clone()
        run_phase('ae2', manager.ae2, x2, y2, validation_encoded, validation_reconstruction, 1., False)
        if historical.state_sha256(manager.ae1.encoder) != encoder_before:
            raise RuntimeError('First encoder changed during AE2')
        historical._seed(seed)
        model = _annotate(manager.transfer_predictor(), dropout=trial['dropout'], seed=seed)
        if (historical.state_sha256(model.encoder1) != encoder_before
                or historical.state_sha256(model.encoder2) != historical.state_sha256(manager.ae2.encoder)):
            raise RuntimeError('SAE encoder transfer is not exact')
        run_phase('supervised', model, train.x, train.y, vx, vy, train.scaler.target_scale, True)
        model.architecture_metadata.update(pretraining_executed=True, training_recipe_id=trial['recipe_id'],
            registered_trial_id=trial['trial_id'], training_hpo_record=trial, auxiliary_scale=manager.scale,
            actual_parameter_count=sum(p.numel() for p in model.parameters()))
    else:
        model = _builder(name, trial, seed, source_roots)
        run_phase('supervised', model, train.x, train.y, vx, vy, train.scaler.target_scale, True)
    checkpoint = directory / 'selected_model.pt'
    if checkpoint.exists():
        raise RuntimeError('Unjournaled selected model requires reconciliation')
    with checkpoint.open('xb') as stream:
        torch.save({'version': VERSION, 'state_dict': historical.clone_state(model),
            'scaler': train.scaler.record(), 'architecture': model.architecture_metadata,
            'request_sha256': digest(request)}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    result = {'version': VERSION, 'status': 'completed', 'model_id': name, 'seed': seed,
        'request_sha256': digest(request), 'selected_checkpoint': {'path': checkpoint.name, 'sha256': file_sha(checkpoint)},
        'phase_best_epochs': {p: records[p]['best_epoch'] for p in phases},
        'phase_records': records, 'architecture': model.architecture_metadata, 'scaler': train.scaler.record(),
        'optimizer_steps': sum(records[p]['optimizer_steps'] for p in phases),
        'training_performed': True, 'synthetic': allow_cpu_for_tests, 'test_only_cpu_execution': allow_cpu_for_tests}
    if validation is not None:
        prediction = predict_pilot(model, validation.x, batch_size=1, device=device,
            allow_cpu_for_tests=allow_cpu_for_tests).numpy() * train.scaler.target_scale
        truth = validation.y_original
        dates = [str(d.date()) for d in validation.dates]
        result.update(prediction_dates=dates, predictions_original=prediction.tolist(),
            rmse_original=math.sqrt(sum((float(p)-float(y))**2 for p, y in zip(prediction, truth))/len(truth)),
            validation_count=len(truth))
        write_json(directory / 'predictions.json', {'dates': dates, 'predictions_original': prediction.tolist(),
            'truths_original': list(truth), 'scope': 'inner_validation_not_final'})
    result['artifacts_sha256'] = {str(p.relative_to(directory)): file_sha(p) for p in directory.rglob('*')
        if p.is_file() and p.name not in ('.writer.lock', 'job_result.json', 'worker_spec.json', 'worker.log')}
    write_json(directory / 'job_result.json', result)
    return result
