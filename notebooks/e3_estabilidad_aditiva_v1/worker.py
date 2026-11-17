"""Ten fixed, new additive-head seeds, with resumable training and a shared budget.

Evaluation opens only after every prescribed training trajectory is complete and
its checkpoints have an acknowledged remote snapshot. Results are retrospective.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

VERSION = 'e3-additive-seed-stability-v1'
PLAN_FINGERPRINT = '0b2dbfea9f14f1506f09fd42b8e74fc299a91438b2a6c01fec6e4181fa7afa8c'
SEEDS = tuple(range(271801, 271811))
MAX_EPOCHS, PATIENCE, BATCH = 40, 8, 32
LIMIT, PRIOR, GLOBAL_LIMIT = 500.0, 650.3773435349993, 1200.0
TERMINAL = ('completed', 'budget_exhausted', 'failed')
HEX40 = re.compile(r'[0-9a-f]{40}')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def js(path):
    return json.loads(Path(path).read_text())


def load_reference(manifest):
    path = Path(__file__).with_name('reference.py')
    require(sha(path) == manifest['reference_sha256'], 'Reference code changed')
    spec = importlib.util.spec_from_file_location('stability_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_state(state):
    for value in state.values():
        if isinstance(value, dict):
            finite_state(value)
        elif isinstance(value, (list, tuple)):
            finite_state(dict(enumerate(value)))
        elif isinstance(value, torch.Tensor):
            require(bool(torch.isfinite(value).all()), 'Nonfinite model or optimizer state')
        elif isinstance(value, float):
            require(math.isfinite(value), 'Nonfinite checkpoint scalar')


def progress(run):
    result = {}
    for seed in SEEDS:
        path = run / 'seeds' / str(seed) / 'checkpoint.meta.json'
        result[str(seed)] = js(path) if path.is_file() else dict(seed=seed, epoch=0, batch_cursor=0,
            epoch_examples=0, initialized=False, complete=False, best_epoch=None, best_validation_rmse=None)
    return result


def state(run, manifest, lib, status):
    exposure = js(run/'evaluation_exposure.json') if (run/'evaluation_exposure.json').is_file() else {}
    value = dict(version=VERSION, run_id=manifest['run_id'], plan_sha256=manifest['plan_sha256'],
                 status=status, seeds=progress(run), evaluation_members_opened=bool(exposure.get('members_opened')))
    lib._atomic_json(run / 'state.json', value)
    return value


def basic_summary(run, manifest, lib, status, error=None):
    ledger = js(run / 'ledger.json')
    p = progress(run)
    exposure = js(run/'evaluation_exposure.json') if (run/'evaluation_exposure.json').is_file() else {}
    opened = bool(exposure.get('members_opened'))
    result = dict(version=VERSION, status=status, run_id=manifest['run_id'], plan_sha256=manifest['plan_sha256'],
        seeds=p, prescribed_seeds=list(SEEDS), complete_training_seeds=sum(v['complete'] for v in p.values()),
        charged_seconds=ledger['charged_seconds'], additional_limit_seconds=LIMIT,
        prior_charged_seconds=PRIOR, global_charged_seconds=PRIOR + ledger['charged_seconds'],
        global_limit_seconds=GLOBAL_LIMIT, budget_overshoot_seconds=max(0., ledger['charged_seconds'] - LIMIT),
        evaluation_scored=bool(exposure.get('scored_seeds')), evaluation_members_opened=opened, comparison_complete=False, winner_declared=False,
        statistical_tests_performed=False, R1_R3_verified=False,
        scope='posthoc_stability_fixed_recipe_known_calendar',
        members_read=['training_dates','training_raw','training_y_original'] + (['evaluation_dates','evaluation_raw','evaluation_y_original'] if opened else []),
        partial_evaluation_scored_seeds=exposure.get('scored_seeds', []), calibration_members_opened=False, error=error)
    lib._atomic_json(run / 'summary.json', result)
    return result


def checkpoint_path(run, seed):
    return run / 'seeds' / str(seed) / 'checkpoint.pt'


def save_cp(run, cp, manifest, budget, lib):
    finite_state(cp['model_state'])
    finite_state(cp['optimizer_state'])
    path = checkpoint_path(run, cp['seed'])
    path.parent.mkdir(parents=True, exist_ok=True)
    lib._save_checkpoint(path, cp)
    meta = {k: cp[k] for k in ('seed', 'epoch', 'batch_cursor', 'epoch_examples', 'initialized',
            'complete', 'best_epoch', 'best_validation_rmse', 'run_id', 'plan_sha256')}
    meta.update(checkpoint_sha256=sha(path), checkpoint_bytes=path.stat().st_size,
                charged_seconds_checkpoint=budget.ledger['charged_seconds'])
    lib._atomic_json(path.with_name('checkpoint.meta.json'), meta)
    state(run, manifest, lib, 'paused')
    budget.tick()


def read_cp(run, seed, manifest, lib):
    path = checkpoint_path(run, seed)
    require(path.is_file() and path.with_suffix('.pt.sha256').is_file(), 'Missing checkpoint or digest')
    digest = sha(path)
    require(digest in path.with_suffix('.pt.sha256').read_text().splitlines(), 'Checkpoint hash mismatch')
    cp = torch.load(path, map_location='cpu', weights_only=True)
    require(cp['version'] == VERSION and cp['seed'] == seed, 'Checkpoint version/seed mismatch')
    for key in ('run_id', 'plan_sha256', 'worker_sha256', 'reference_sha256', 'source_hashes'):
        require(cp[key] == manifest[key], 'Checkpoint scientific identity changed: ' + key)
    require(0 <= cp['epoch'] <= MAX_EPOCHS and 0 <= cp['batch_cursor'] <= 532,
            'Checkpoint cursor outside frozen recipe')
    require(cp['epoch_examples'] == cp['batch_cursor'], 'Checkpoint batch accounting changed')
    if cp['active_permutation'] is not None:
        perm = cp['active_permutation']
        require(perm.dtype == torch.int64 and perm.shape == (532,)
                and torch.equal(torch.sort(perm).values, torch.arange(532)), 'Invalid saved permutation')
    else:
        require(cp['batch_cursor'] == 0, 'Cursor without active permutation')
    require(cp['epoch'] == len(cp['history']), 'Checkpoint history length changed')
    require(cp['best_epoch'] is None or 1 <= cp['best_epoch'] <= cp['epoch'], 'Invalid best epoch')
    require(not cp['complete'] or cp['best_model_state'] is not None, 'Completed seed lacks best weights')
    finite_state(cp['model_state'])
    finite_state(cp['optimizer_state'])
    if cp['best_model_state'] is not None:
        finite_state(cp['best_model_state'])
    meta_path = path.with_name('checkpoint.meta.json')
    if meta_path.is_file():
        old = js(meta_path)
        require((old['epoch'], old['batch_cursor']) <= (cp['epoch'], cp['batch_cursor']),
                'Metadata is ahead of checkpoint; acknowledged progress cannot be discarded')
        if old['checkpoint_sha256'] == digest:
            require(all(old[k] == cp[k] for k in ('epoch', 'batch_cursor', 'initialized', 'complete', 'best_epoch')),
                    'Checkpoint metadata differs')
    return cp


def reconcile_metadata(run, manifest, budget, lib):
    # A checkpoint may have committed before a VM stopped during its derived JSON receipt.
    # Never alter or rewind checkpoint bytes; only regenerate metadata from admitted state.
    for seed in SEEDS:
        path = checkpoint_path(run, seed)
        if not path.is_file():
            continue
        budget.tick()
        cp = read_cp(run, seed, manifest, lib)
        meta = {k: cp[k] for k in ('seed', 'epoch', 'batch_cursor', 'epoch_examples', 'initialized',
                'complete', 'best_epoch', 'best_validation_rmse', 'run_id', 'plan_sha256')}
        meta.update(checkpoint_sha256=sha(path), checkpoint_bytes=path.stat().st_size,
                    charged_seconds_checkpoint=budget.ledger['charged_seconds'])
        lib._atomic_json(path.with_name('checkpoint.meta.json'), meta)
    budget.tick()


def new_model(seed, scaler, source_root, Model, device):
    rule = source_root / 'experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json'
    model = Model('additive_positive_head', u_scale=scaler['u_scale'], y_scale=scaler['y_scale'],
                 seed=seed, rule_path=rule).to(device)
    require(sum(p.numel() for p in model.parameters()) == 1210, 'Model capacity changed')
    return model


def make_cp(seed, model, optimizer, manifest, lib):
    return dict(version=VERSION, seed=seed, initialized=False, epoch=0, batch_cursor=0,
        epoch_examples=0, epoch_loss_sum=0., active_permutation=None, complete=False,
        best_epoch=None, best_validation_rmse=None, best_model_state=None, no_improvement=0,
        history=[], model_state=deepcopy(model.state_dict()), optimizer_state=deepcopy(optimizer.state_dict()),
        rng=lib._rng(), generator_state=torch.Generator().manual_seed(seed).get_state(),
        **{k: manifest[k] for k in ('run_id', 'plan_sha256', 'worker_sha256', 'reference_sha256', 'source_hashes')})


def train_data(inputs, admission, reference):
    with np.load(inputs / 'current_cohort_arrays.npz', allow_pickle=False) as source:
        ids = reference.dates(source['training_dates'], 655, 'Training')
        raw = np.asarray(source['training_raw'], dtype=np.float64).copy()
        y = np.asarray(source['training_y_original'], dtype=np.float64).copy()
    require(raw.shape == (655, 7, 5) and y.shape == (655,), 'Training array shape changed')
    positions = {d: i for i, d in enumerate(ids)}
    fit_ids, val_ids = admission['fit_dates'], admission['validation_dates']
    require(len(fit_ids) == 532 and max(fit_ids) == '2024-06-30' and len(val_ids) == 116
            and val_ids[0] == '2024-07-08' and val_ids[-1] == '2024-10-31', 'Frozen cohorts changed')
    x, a = reference.transform(raw, admission['scaler'])
    y = reference.validate_target(y, 655, np.sqrt(a))
    fit = np.array([positions[d] for d in fit_ids])
    val = np.array([positions[d] for d in val_ids])
    return dict(train_x=x[fit], train_a=a[fit], train_y=y[fit], val_x=x[val], val_a=a[val], val_y=y[val])


def predict(model, x, a, device, budget):
    parts = []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(x), BATCH):
            budget.tick()
            result = model(torch.from_numpy(x[first:first+BATCH]).to(device),
                           torch.from_numpy(a[first:first+BATCH]).to(device)).cpu().numpy()
            require(np.isfinite(result).all() and (result >= 0).all(), 'Nonfinite prediction')
            parts.append(result)
    budget.tick()
    return np.concatenate(parts).astype(np.float64)


def train_epoch(seed, run, data, scaler, manifest, source_root, Model, lib, budget, device):
    budget.tick()
    path = checkpoint_path(run, seed)
    lib._determinism(seed)
    model = new_model(seed, scaler, source_root, Model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    cp = read_cp(run, seed, manifest, lib) if path.is_file() else make_cp(seed, model, optimizer, manifest, lib)
    require(not cp['complete'], 'Completed seed cannot be trained again')
    model.load_state_dict(cp['model_state'], strict=True)
    optimizer.load_state_dict(cp['optimizer_state'])
    lib._restore_rng(cp['rng'])
    generator = torch.Generator()
    generator.set_state(cp['generator_state'])
    budget.tick()
    if not cp['initialized']:
        save_cp(run, cp, manifest, budget, lib)
        model.calibrate_bias(data['train_x'], data['train_a'], scaler['d_fit'],
                             budget_callback=budget.tick, batch_size=BATCH)
        cp['initialized'] = True
        cp['model_state'], cp['rng'] = deepcopy(model.state_dict()), lib._rng()
        save_cp(run, cp, manifest, budget, lib)
    if cp['active_permutation'] is None:
        cp['active_permutation'] = torch.randperm(532, generator=generator)
        cp['generator_state'] = generator.get_state()
        save_cp(run, cp, manifest, budget, lib)
    model.train()
    while cp['batch_cursor'] < 532:
        budget.tick()
        first = cp['batch_cursor']
        indices = cp['active_permutation'][first:first+BATCH].numpy()
        x = torch.from_numpy(data['train_x'][indices]).to(device)
        a = torch.from_numpy(data['train_a'][indices]).to(device)
        target = torch.from_numpy(data['train_y'][indices]).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted = model(x, a)
        loss = ((predicted - target) / scaler['y_scale']).square().mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite training loss; seed cannot be replaced')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        cp['batch_cursor'] += len(indices)
        cp['epoch_examples'] += len(indices)
        cp['epoch_loss_sum'] += float(loss.detach()) * len(indices)
        cp['model_state'], cp['optimizer_state'], cp['rng'] = (
            deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), lib._rng())
        save_cp(run, cp, manifest, budget, lib)
    prediction = predict(model, data['val_x'], data['val_a'], device, budget)
    rmse = math.sqrt(math.fsum(float(e)**2 for e in prediction-data['val_y']) / len(prediction))
    require(math.isfinite(rmse), 'Nonfinite validation RMSE')
    cp['epoch'] += 1
    cp['history'].append(dict(epoch=cp['epoch'], training_loss=cp['epoch_loss_sum']/532,
                              validation_RMSE=rmse))
    if cp['best_validation_rmse'] is None or rmse < cp['best_validation_rmse']:
        cp['best_epoch'], cp['best_validation_rmse'] = cp['epoch'], rmse
        cp['best_model_state'], cp['no_improvement'] = deepcopy(model.state_dict()), 0
    else:
        cp['no_improvement'] += 1
    cp['complete'] = cp['epoch'] >= MAX_EPOCHS or cp['no_improvement'] >= PATIENCE
    cp['active_permutation'], cp['batch_cursor'], cp['epoch_examples'], cp['epoch_loss_sum'] = None, 0, 0, 0.
    cp['rng'] = lib._rng()
    save_cp(run, cp, manifest, budget, lib)
    print(f"Semilla {seed}: época {cp['epoch']}, RMSE validación {rmse:.9f}, completa={cp['complete']}", flush=True)


def evaluation(run, inputs, manifest, claim, admission, reference, source_root, Model, lib, budget, device):
    p = progress(run)
    require(all(p[str(s)]['complete'] for s in SEEDS), 'All ten seeds must finish before evaluation')
    require(HEX40.fullmatch(claim.get('training_snapshot_commit', '')) is not None,
            'Completed checkpoints need a remote training snapshot ACK')
    expected_hashes = claim.get('checkpoint_sha256')
    require(isinstance(expected_hashes, dict) and set(expected_hashes) == set(map(str, SEEDS)),
            'Evaluation ACK must bind all ten trained checkpoints')
    checkpoints = {}
    for seed in SEEDS:
        budget.tick()
        require(sha(checkpoint_path(run, seed)) == expected_hashes[str(seed)], 'ACK checkpoint changed')
        checkpoints[seed] = read_cp(run, seed, manifest, lib)
        require(checkpoints[seed]['complete'], 'Incomplete checkpoint cannot be evaluated')
    expected_dates = reference.dates(js(inputs / 'e2_partition.json')['final_partition']['evaluation']['sample_IDs'], 608, 'E2 test')
    e2_admission = js(inputs / 'e2_data_admission.json')
    require(e2_admission['common_eligible_dates'] == expected_dates
            and e2_admission['target'] == 'population sigma7 of daily Close log returns, ddof0, unannualized; h1',
            'E2 target or test calendar changed')
    exposure = dict(members_opened=True, scored_seeds=[], training_snapshot_commit=claim['training_snapshot_commit'])
    lib._atomic_json(run/'evaluation_exposure.json', exposure)
    with np.load(inputs / 'current_cohort_arrays.npz', allow_pickle=False) as source:
        ids = reference.dates(source['evaluation_dates'], 608, 'Evaluation')
        require(ids == expected_dates, 'Evaluation IDs differ from E2')
        raw = np.asarray(source['evaluation_raw'], dtype=np.float64).copy()
        y = np.asarray(source['evaluation_y_original'], dtype=np.float64).copy()
    x, a = reference.transform(raw, admission['scaler'])
    y = reference.validate_target(y, 608, np.sqrt(a))
    rows, predictions = [], []
    for seed in SEEDS:
        budget.tick()
        cp = checkpoints[seed]
        model = new_model(seed, admission['scaler'], source_root, Model, device)
        model.load_state_dict(cp['best_model_state'], strict=True)
        predicted = predict(model, x, a, device, budget)
        result = reference.metrics(y, predicted)
        rows.append(dict(seed=seed, best_epoch=cp['best_epoch'], training_epochs=cp['epoch'],
                         validation_RMSE=cp['best_validation_rmse'], **result, evaluation_n=608))
        exposure['scored_seeds'].append(seed)
        lib._atomic_json(run/'evaluation_exposure.json', exposure)
        lib._atomic_bytes(run/'per_seed_metrics.csv', reference.csv_bytes(rows))
        predictions.extend(dict(seed=seed, date=d, target_sigma7=float(t), prediction_sigma7=float(v),
                                known_floor=float(f)) for d,t,v,f in zip(ids,y,predicted,np.sqrt(a)))
    values = np.array([r['RMSE'] for r in rows])
    budget.tick()
    boot = np.random.default_rng(20261117).choice(values, size=(10000, 10), replace=True).mean(axis=1)
    ci = np.quantile(boot, [.025,.975]).tolist()
    control_predictions = {'floor_only': np.sqrt(a), 'floor_plus_fit_mean': np.sqrt(a)+admission['scaler']['d_fit']}
    controls = [dict(model=name, **reference.metrics(y, v), evaluation_n=608)
                for name,v in control_predictions.items()]
    a2 = reference.read_csv(inputs / 'e2_benchmark_A2.csv')
    require(len(a2) == 16 and len({r['model'] for r in a2}) == 16, 'E2 family changed')
    comparison = [dict(model=r['model'], source='E2_frozen', RMSE=float(r['RMSE']),
        MAE=float(r['MAE']), optimization_seeds=int(r['R']), evaluation_n=int(r['n_common_dates']),
        aggregation='mean_of_seed_metrics' if int(r['R'])>1 else 'deterministic_trajectory',
        scope='descriptive_different_tuning_and_training_budgets') for r in a2]
    comparison.append(dict(model='CNN-LSTM-atencion_aditiva_v2_10_nuevas_semillas', source='E3_posthoc_stability',
        RMSE=float(values.mean()), MAE=float(np.mean([r['MAE'] for r in rows])), optimization_seeds=10,
        evaluation_n=608, aggregation='mean_of_individual_seed_metrics', scope='descriptive_different_tuning_and_training_budgets'))
    # The old selected seed is kept separate; never add it to the ten-seed mean.
    historical = js(inputs / 'historical_summary.json')
    for filename, content in [('per_seed_metrics.csv', rows), ('evaluation_predictions.csv', predictions),
                              ('analytic_controls.csv', controls), ('comparison_descriptive.csv', comparison)]:
        budget.tick()
        lib._atomic_bytes(run / filename, reference.csv_bytes(content))
    result = basic_summary(run, manifest, lib, 'completed')
    result.update(evaluation_scored=True, comparison_complete=True, evaluation_n=608,
        per_seed_metrics=rows, RMSE_mean=float(values.mean()), RMSE_sample_SD=float(values.std(ddof=1)),
        RMSE_bootstrap_CI95=ci, bootstrap_resamples=10000, bootstrap_seed=20261117,
        bootstrap_scope='optimization_seeds_conditional_on_known_calendar_not_independent_dataset_uncertainty',
        historical_seed_excluded=20261116,
        historical_reference=historical.get('evaluation', historical.get('metrics')),
        training_snapshot_commit=claim['training_snapshot_commit'], checkpoints_acknowledged_sha256=expected_hashes,
        E3_fit_examples=532, E2_neural_fit_examples=655, E2_neural_seeds=6,
        equal_training_budget_claimed=False, target=e2_admission['target'],
        calibration_members_opened=False, rows_dropped=0,
        members_read=['training_dates','training_raw','training_y_original','evaluation_dates','evaluation_raw','evaluation_y_original'],
        limits=['Known historical evaluation, after previous development and test results.',
                'All ten fixed new seeds are retained; no best-seed selection.',
                'E2 has six neural seeds, not ten; this does not complete the course statistical requirements.',
                'The bootstrap interval measures optimization-seed variation conditional on this calendar.'])
    lib._atomic_json(run / 'summary.json', result)
    budget.tick()
    return result


def execute(source_root, inputs, run, claim_path, phase, max_epochs_this_call=50,
            device='cuda', clock=time.monotonic):
    source_root, inputs, run = Path(source_root).resolve(), Path(inputs), Path(run)
    manifest, plan, claim, ledger = js(run/'manifest.json'), js(run/'plan.json'), js(claim_path), js(run/'ledger.json')
    fingerprint = hashlib.sha256(canonical(plan)).hexdigest()
    require(fingerprint == PLAN_FINGERPRINT == manifest['plan_sha256'] == claim['plan_sha256'], 'Plan identity changed')
    for key in ('run_id', 'claim_id'):
        require(manifest[key] == claim[key], 'Claim identity changed: '+key)
    require(sha(__file__) == manifest['worker_sha256'], 'Worker source changed')
    require(sha(Path(__file__).with_name('bootstrap.py')) == manifest['bootstrap_sha256'], 'Coordinator source changed')
    require(manifest['source_hashes'] == plan['source']['files'], 'Original sources changed')
    for name, digest in manifest['source_hashes'].items():
        require(sha(source_root/name) == digest, 'Original source SHA changed: '+name)
    for name, receipt in plan['inputs'].items():
        require(sha(inputs/name) == receipt['sha256'], 'Input SHA changed: '+name)
    require(plan['training']['seeds'] == list(SEEDS) and plan['training']['max_epochs'] == 40
            and plan['training']['patience'] == 8 and plan['training']['batch_size'] == 32
            and plan['training']['learning_rate'] == .001 and plan['training']['weight_decay'] == 0
            and plan['training']['gradient_clip_norm'] == 1 and plan['training']['bias_matching_steps'] == 64
            and plan['training']['optimizer'] == 'AdamW' and plan['training']['AMP'] is False
            and plan['training']['TF32'] is False and plan['architecture']['variant'] == 'additive_positive_head'
            and plan['architecture']['trainable_parameters'] == 1210,
            'Frozen training recipe changed')
    for key, expected in [('additional_limit_seconds', LIMIT), ('prior_charged_seconds', PRIOR), ('global_limit_seconds', GLOBAL_LIMIT)]:
        require(claim[key] == expected == plan['budget'][key], 'Claim budget changed: '+key)
    require(ledger['run_id'] == manifest['run_id'] and ledger['limit_seconds'] == LIMIT
            and ledger['prior_charged_seconds'] == PRIOR and ledger['global_limit_seconds'] == GLOBAL_LIMIT
            and 0 <= ledger['charged_seconds'], 'Ledger identity or budget changed')
    require(HEX40.fullmatch(claim.get('ack_commit','')) is not None and claim['phase'] == phase,
            'An acknowledged phase claim is required before any initialization')
    require(ledger['active_session'] and ledger['active_session']['session_id'] == claim['active_session_id'],
            'Claim active session differs from reserved ledger')
    oldstate = js(run/'state.json') if (run/'state.json').is_file() else {'status':'new'}
    if oldstate['status'] in TERMINAL:
        return js(run/'summary.json')
    require(phase in ('train','evaluate') and 1 <= max_epochs_this_call <= 50, 'Invalid phase/chunk')
    require(phase != 'evaluate' or oldstate['status'] == 'trained', 'Training must complete and be ACKed before evaluation')
    require(phase != 'train' or oldstate['status'] != 'trained', 'Trained campaign cannot be extended')
    reference = load_reference(manifest)
    sys.path.insert(0, str(source_root))
    from src import e3_innovation as lib
    from src.e3_innovation_model import InnovationModel as Model
    from src.e3_development import runtime_info, run_lock
    require(Path(lib.__file__).resolve() == source_root/'src/e3_innovation.py', 'Wrong source module cached')
    prior = js(inputs/'prior_moment_summary.json')
    require(prior['status'] == 'completed' and prior['global_charged_seconds'] == PRIOR,
            'Prior development campaign is not the sealed predecessor')
    admission, v2manifest = js(inputs/'v2_data_admission.json'), js(inputs/'v2_manifest.json')
    require(manifest['npz_sha256'] == claim['npz_sha256'] == plan['inputs']['current_cohort_arrays.npz']['sha256'] == admission['npz_sha256'], 'NPZ scientific identity changed')
    require(hashlib.sha256(canonical(admission)).hexdigest() == v2manifest['data_fingerprint'], 'Saved scaler provenance changed')
    require(admission['scaler']['fit_dates'] == admission['fit_dates'], 'Saved scaler fit IDs changed')
    lib._determinism(SEEDS[0])
    runtime = runtime_info(device)
    runtime['torch'] = str(runtime['torch'])
    require(runtime['torch'].split('+')[0] == '2.8.0' and runtime['numpy'] == '2.3.5', 'Numerical runtime changed')
    if device == 'cuda':
        require(runtime['python'] == '3.12.12' and runtime['torch'] == '2.8.0+cu126', 'GPU runtime must match fixed lock')
    require(device in ('cpu','cuda') and (device!='cuda' or torch.cuda.is_available()), 'Requested device unavailable')
    if (run/'runtime.json').is_file():
        oldruntime = js(run/'runtime.json')
        require(all(runtime.get(k)==oldruntime.get(k) for k in ('python','torch','numpy','device_type','gpu','cuda_build','cpu_threads')),
                'Exact resume requires the same recorded numerical runtime/device')
    with run_lock(run):
        lib._atomic_json(run/'runtime.json', runtime)
        budget = lib._Budget(run, ledger, clock, device)
        # Preserve conservative recovery charges while binding computation to the remote ACK.
        budget.session_id = claim['active_session_id']
        ledger['active_session']['session_id'] = budget.session_id
        lib._atomic_json(run/'ledger.json', ledger)
        status, error, result = 'paused' if phase=='train' else 'trained', None, None
        try:
            budget.tick()
            if phase == 'train':
                reconcile_metadata(run, manifest, budget, lib)
                data = train_data(inputs, admission, reference)
                budget.tick()
                count = 0
                while count < max_epochs_this_call:
                    p = progress(run)
                    active = [seed for seed in SEEDS if not p[str(seed)]['complete']]
                    if not active:
                        status = 'trained'
                        break
                    seed = min(active, key=lambda s: (p[str(s)]['epoch'], s))
                    train_epoch(seed, run, data, admission['scaler'], manifest, source_root, Model, lib, budget, device)
                    count += 1
                if all(v['complete'] for v in progress(run).values()):
                    status = 'trained'
            else:
                result = evaluation(run, inputs, manifest, claim, admission, reference, source_root, Model, lib, budget, device)
                status = 'completed'
        except lib.BudgetExhausted:
            status = 'budget_exhausted'
        except KeyboardInterrupt as caught:
            status, error = 'interrupted', caught
        except BaseException as caught:
            status, error = 'failed', caught
        finally:
            budget.close(status)
            if ledger['charged_seconds'] >= LIMIT:
                status = 'budget_exhausted'
            state(run, manifest, lib, status)
            if status != 'completed':
                result = basic_summary(run, manifest, lib, status, str(error) if error else None)
            else:
                result['charged_seconds'] = ledger['charged_seconds']
                result['global_charged_seconds'] = PRIOR + ledger['charged_seconds']
                lib._atomic_json(run/'summary.json', result)
        if error:
            raise error
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--claim-path', type=Path, required=True)
    parser.add_argument('--phase', choices=('train','evaluate'), required=True)
    parser.add_argument('--max-epochs-this-call', type=int, default=50)
    parser.add_argument('--device', choices=('cuda','cpu'), required=True)
    args = parser.parse_args()
    result = execute(args.source_root, args.inputs, args.run_dir, args.claim_path,
                     args.phase, args.max_epochs_this_call, args.device)
    print(json.dumps({k:result.get(k) for k in ('status','complete_training_seeds','charged_seconds','global_charged_seconds','RMSE_mean','RMSE_sample_SD')}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
