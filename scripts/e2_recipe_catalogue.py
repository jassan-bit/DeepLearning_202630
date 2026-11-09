"""Section 22 recipe factory and bounded synthetic mechanism checks.

No market loader, model search, historic receipt reuse, or training authorization
is supplied by this module. Four new ports coexist with the frozen catalogue.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import platform
import numpy as np
import torch
from torch import nn

import e2_current_models as current
import e2_current_pipeline as pipeline
import e2_recipe_models as bodies

MODEL_NAMES = current.MODEL_NAMES
NEW_NAMES = tuple(current.PROTOTYPE_NAMES)
VERSION = 'e2-section22-synthetic-v1'
RECIPE_IDS = {
 'FCN-1D': 'fcn1d_wang_d7_v1_proposed',
 'ResNet-1D': 'resnet1d_wang_d7_v1_proposed',
 'Inception-1D': 'inception1d_single_d7_v1_proposed',
 current.AE_NAME: 'lstm_sae_25_26_d7_v1_proposed',
 'TimesNet': 'TN-D7-r1', 'TSMixer': 'TSM-D7-r1', 'TiDE': 'TIDE-D7-r1',
 'TimeMixer': 'TM-D7-r1', 'PatchTST': 'PT-D7-r1', 'iTransformer': 'IT-D7-r1',
 current.HYBRID_NAME: 'HIB-D7-r1',
}


def build_model(name, *, source_roots=None):
    """Use exact new bodies or unchanged provisional-default adapters, never compact."""
    if name not in MODEL_NAMES:
        raise ValueError('Unknown section 22 model')
    if name in NEW_NAMES:
        model = bodies.build_model(name)
    else:
        model = current.build_model(name, current.recipe_for(name, 'provisional_default', source_roots))
    return model


def fit_auxiliary_scale(x_train):
    """Scalar reconstruction target scale fitted to training inputs only."""
    current._check_input(x_train)
    return max(1., float(x_train.detach().abs().max()))


def _step(module, inputs, targets, steps, config):
    optimizer = torch.optim.Adam(module.parameters(), lr=config.learning_rate)
    module.train()
    losses = []
    start = time.perf_counter()
    initial = pipeline._state_sha256(module)
    gradient_names = []
    for step in range(steps):
        indices = (torch.arange(config.batch_size) + step * config.batch_size) % len(inputs)
        optimizer.zero_grad(set_to_none=True)
        prediction = module(inputs[indices])
        target = targets[indices]
        if prediction.shape != target.shape:
            raise ValueError('Prediction and target shapes must match exactly')
        loss = nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite synthetic loss')
        loss.backward()
        parameters = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
        if not parameters or any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in parameters):
            raise ValueError('Missing or nonfinite gradients')
        gradient_names = [name for name, _ in parameters]
        nn.utils.clip_grad_norm_(module.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in module.parameters()):
            raise ValueError('Nonfinite updated parameters')
        losses.append(float(loss.detach()))
    return {'steps': steps, 'mse': losses, 'fresh_optimizer': True,
            'parameter_count': sum(p.numel() for p in module.parameters()),
            'trainable_parameter_count': sum(p.numel() for p in module.parameters() if p.requires_grad),
            'optimizer': 'Adam', 'gradient_parameter_names': gradient_names,
            'all_gradients_finite': True, 'elapsed_seconds': time.perf_counter()-start,
            'initial_state_sha256': initial, 'final_state_sha256': pipeline._state_sha256(module)}


def run_synthetic_smoke(*, model_names=None, config=None, source_roots=None, synthetic_only=True):
    """Generate the sole admitted fixture internally. No external data API."""
    if synthetic_only is not True:
        raise ValueError('Only synthetic_only=True is permitted')
    config = config or pipeline.SmokeConfig(seed=20261109, supervised_steps=1)
    config.validate()
    names = list(MODEL_NAMES if model_names is None else model_names)
    if not names or len(set(names)) != len(names) or any(n not in MODEL_NAMES for n in names):
        raise ValueError('Choose a nonempty unique subset of the eleven recipes')
    data = pipeline.make_synthetic_data(seed=config.seed, days=config.days)
    fold = pipeline.make_synthetic_fold(data, origin_index=config.origin_index,
            train_count=config.train_count, validation_count=config.validation_count)
    preprocessing = pipeline.fit_preprocessing(data, fold)
    xt, yt = pipeline.transform_fold(data, fold, preprocessing, 'train')
    xv, _ = pipeline.transform_fold(data, fold, preprocessing, 'validation')
    x, y = torch.from_numpy(xt), torch.from_numpy(yt)
    records = []
    verified = {}
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for name in names:
            torch.manual_seed(config.seed)
            phases = {}
            ae_audit = None
            if name == current.AE_NAME:
                stage = bodies.SAEPretraining(x, preprocessing['train_windows_sha256'], accept_auxiliary_scale=True)
                ae1_target = stage.reconstruction_target(x)
                if not torch.equal(ae1_target, x / fit_auxiliary_scale(x)):
                    raise ValueError('AE1 must reconstruct X/s with train-only scale')
                phases['ae1'] = _step(stage.ae1, x, ae1_target, config.pretrain_steps, config)
                frozen_inputs, target = stage.prepare_phase2()
                if any(p.requires_grad for p in stage.ae1.encoder.parameters()) or stage.ae1.encoder.training:
                    raise ValueError('AE1 encoder must be frozen and in eval for phase 2')
                if frozen_inputs.requires_grad or target.shape != x.shape:
                    raise ValueError('AE2 requires detached features and five-channel targets')
                if not torch.equal(target, x / fit_auxiliary_scale(x)):
                    raise ValueError('AE2 must reconstruct X/s with train-only scale')
                before = pipeline._state_sha256(stage.ae1.encoder)
                phases['ae2'] = _step(stage.ae2, frozen_inputs, target, config.pretrain_steps, config)
                after = pipeline._state_sha256(stage.ae1.encoder)
                if before != after:
                    raise ValueError('Encoder1 changed during AE2')
                model = stage.transfer_predictor()
                ae_audit = {'auxiliary_scale': fit_auxiliary_scale(x), 'auxiliary_scale_fit': 'training_only',
                    'auxiliary_scale_train_sha256': pipeline.array_sha256(xt),
                    'target': 'original five-channel transformed X / s in both phases',
                    'encoder1_phase2_before_sha256': before, 'encoder1_phase2_after_sha256': after,
                    'encoder1_frozen_during_phase2': before == after,
                    'phase2_inputs_sha256': pipeline.array_sha256(frozen_inputs.detach().numpy()),
                    'phase2_target_sha256': pipeline.array_sha256(target.detach().numpy()),
                    'validation_auxiliary_fraction_outside_unit_range': float(np.mean(np.abs(xv/fit_auxiliary_scale(x)) > 1)),
                    'decoder_parameters_in_predictor': any('decoder' in n for n, _ in model.named_parameters()),
                    'transfer_encoder1_exact': pipeline._state_sha256(model.encoder1) == after,
                    'transfer_encoder2_exact': pipeline._state_sha256(model.encoder2) == pipeline._state_sha256(stage.ae2.encoder)}
                if ae_audit['decoder_parameters_in_predictor'] or not all(ae_audit[k] for k in ['transfer_encoder1_exact', 'transfer_encoder2_exact']):
                    raise ValueError('Invalid encoder transfer or decoder in predictor')
            else:
                model = build_model(name, source_roots=source_roots)
            count = sum(p.numel() for p in model.parameters())
            if not 0 < count <= pipeline.MAX_PARAMETERS:
                raise ValueError('Recipe exceeds bounded synthetic parameter cap')
            original_metadata = getattr(model, 'architecture_metadata', {})
            # Public receipt uses explicit fields, never arbitrary execution paths.
            metadata = {
                'model': name, 'recipe_id': RECIPE_IDS[name], 'input_shape': [7, 5],
                'output_link': 'Softplus', 'same_window_moment_features':
                    0 if name in NEW_NAMES or name == current.HYBRID_NAME else 10,
                'implementation': 'new_section22_port' if name in NEW_NAMES else 'unchanged_current_default',
                'source_module': 'scripts/e2_recipe_models.py' if name in NEW_NAMES else 'scripts/e2_current_models.py',
                'paper_replication_claimed': False, 'structural_resize_for_smoke': False,
            }
            if 'source_verification' in original_metadata:
                source = original_metadata['source_verification']
                verified[name] = {key: deepcopy(source[key]) for key in
                    ('repository', 'commit', 'verified_files', 'verification_scope')}
            phases['supervised'] = _step(model, x, y, config.supervised_steps, config)
            prediction = pipeline.predict_original_units(model, xv, preprocessing)
            if np.any(prediction < 0):
                raise ValueError('Negative Softplus output')
            records.append({'model': name, 'recipe_id': RECIPE_IDS[name], 'parameter_count': count,
                'architecture': metadata, 'phases': phases, 'ae_protocol': ae_audit,
                'synthetic_predictions_original_units': prediction.tolist(),
                'synthetic_validation_mse_original_units': float(np.mean((prediction-data['targets'][fold['validation_indices']])**2))})
    finally:
        torch.set_num_threads(old_threads)
    source_names = ['e2_recipe_catalogue.py', 'e2_recipe_models.py', 'smoke_e2_recipe_catalogue.py',
        'e2_current_models.py', 'e2_current_pipeline.py', 'e2_proposed_models.py',
        'e2_final_models_mlp.py', 'e2_final_models_attention.py', 'dl_data.py', 'dl_data_v2.py']
    root = Path(__file__).resolve().parent
    receipt = {'receipt_version': VERSION, 'synthetic_only': True, 'status': 'synthetic_mechanics_only',
        'market_data_read': False, 'market_training_executed': False, 'hpo_executed': False,
        'nas_executed': False, 'nas_adjudication': None, 'full_benchmark_executed': False,
        'generalization_or_superiority_demonstrated': False, 'historical_receipts_reused': False,
        'requested_models': names, 'executed_model_count': len(records), 'skipped_model_count': 0,
        'config': asdict(config), 'preprocessing': preprocessing,
        'timing_scope': 'local CPU synthetic phase checks only; not market-training resource estimates',
        'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': 'cpu', 'torch_threads': 1},
        'source_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in source_names},
        'verified_author_sources': verified,
        'verified_author_file_count': sum(len(v['verified_files']) for v in verified.values()),
        'author_verification_scope': '70 declared files for the four external adapters in a full run; not whole repositories, not evidence for the four new ports',
        'fixture': {'features_sha256': pipeline.array_sha256(data['features']), 'targets_sha256': pipeline.array_sha256(data['targets'])},
        'fold': {'train_indices': fold['train_indices'].tolist(), 'validation_indices': fold['validation_indices'].tolist(),
            'origin_day_utc': fold['origin_day_utc'].isoformat(), 'training_labels_at_or_before_origin_minus_8': True,
            'validation_used_for_optimization_or_selection': False},
        'training_contract': {'synthetic_optimizer': 'fresh Adam per phase', 'architecture_preset': 'section22_full_dimensions',
            'structural_resize_for_smoke': False, 'target': 'sigma7 / positive training-only mean', 'target_centered': False,
            'auxiliary_scale_adaptation_accepted_for_synthetic_check_only': True,
            'no_paper_weight_equivalence_claim': True, 'hyperparameter_search': False}, 'models': records}
    json.dumps(receipt, allow_nan=False)
    return receipt
