"""Prospective fixed two-encoder SAE body and its three-stage D7 training.

The 25/26-unit body is the Table 2 model of Sagheer/Kotb (2019), supported by
the author's pinned two-layer implementation. D7, five OHLCV channels, the
train-only auxiliary X/s target and the positive volatility head are explicit
task adaptations. No market loader, source download or NAS shortcut is supplied.
Historical recipe/training modules and receipts remain unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from pathlib import Path
import time

import torch
from torch import nn

try:
    from . import e2_recipe_models as recipes
    from . import e2_staged_training as training
except ImportError:
    import e2_recipe_models as recipes
    import e2_staged_training as training

RECIPE_ID = "SAE-D7-PUBLISHED-v1"
BODY_SOURCE_SHA256 = "47380cc87e2d4b4221ee4fc97da3286990470b3304dd1d9ef990ace51c338b75"
SOURCE = {
    "paper_doi": "10.1038/s41598-019-55320-6",
    "paper_url": "https://www.nature.com/articles/s41598-019-55320-6",
    "author_hosted_text_url": "https://www.academia.edu/111461009/Unsupervised_Pre_training_of_a_Deep_LSTM_based_Stacked_Autoencoder_for_Multivariate_Time_Series_Forecasting_Problems",
    "verified_at_date": "2026-11-12",
    "verified_scope": "Architecture/learning sections pp7-8 and Table2 p11, author-hosted article text; pinned author implementation",
    "pdf_downloaded_or_hashed": False,
    "author_repository": "https://github.com/2M-kotb/LSTM-based-Stacked-Autoencoder",
    "author_commit": "8668132ef6f8855dbab289c5a2b2478f871c4c4b",
    "author_file": "2layer_evaluate.py",
    "author_locators": "hidden_layers lines138; AE construction/targets lines211-224; encoder transfer/predictor lines232-260",
    "independent_port": "scripts/e2_recipe_models.py",
    "independent_port_sha256": BODY_SOURCE_SHA256,
    "author_weights_reused": False,
    "backend_or_original_experiment_replication_claimed": False,
}


def _verify_source_and_dropout(dropout):
    if (type(dropout) not in (int, float) or not math.isfinite(dropout)
            or not 0 <= dropout <= .5):
        raise ValueError("Use an explicit finite dropout rate within [0,.5]")
    if sha256(Path(recipes.__file__).read_bytes()).hexdigest() != BODY_SOURCE_SHA256:
        raise ValueError("The verified fixed SAE body source changed; adjudicate a new recipe")


def _metadata(dropout, seed):
    return {
        "recipe_id": RECIPE_ID, "model": "LSTM-autoencoder con predictor",
        "published_hidden_units": [25, 26], "input_shape": [7, 5],
        "auxiliary_decoder_output_channels": 5,
        "additional_nas_dense_group": False, "nas_required_for_this_fixed_body": False,
        "dropout": float(dropout), "published_reference_dropout": .3,
        "dropout_policy": "Existing supervised dropout modules only; no auxiliary dropout",
        "initialization_seed": seed, "source": deepcopy(SOURCE),
        "adaptations": ["lag25 to D7", "source eight variables to five OHLCV channels",
                        "training-only auxiliary reconstruction target X/s",
                        "scalar volatility output with Softplus and caller-owned target scaling"],
        "pretraining_required": True, "pretraining_executed": False,
        "target_history_added": False, "moment_features_added": False,
        "market_training_claimed": False,
    }


def _annotate(model, *, dropout, seed):
    model.dropout1.p = model.dropout2.p = float(dropout)
    model.architecture_metadata = _metadata(dropout, seed)
    return model


def build_sae(*, dropout=.3, seed):
    """Construct the fixed predictor only; proper use still requires both AEs.

    CPU construction preserves the caller's RNG stream. Dropout changes rates
    of the two existing modules, not hidden widths, depth or decoder geometry.
    """
    _verify_source_and_dropout(dropout)
    if type(seed) is not int or seed < 0:
        raise ValueError("An explicit nonnegative initialization seed is required")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = recipes.SAERecipe()
    return _annotate(model, dropout=dropout, seed=seed)


def _tensor_sha256(value):
    return training.state_sha256({"tensor": value.detach().cpu()})


@dataclass(frozen=True)
class PublishedSAEResult:
    model: nn.Module
    phases: dict
    auxiliary_scale: float
    fold_id: str
    audit: dict

    def to_dict(self):
        return {"recipe_id": RECIPE_ID, "fold_id": self.fold_id,
                "auxiliary_scale": self.auxiliary_scale,
                "phases": {key: value.to_dict() for key, value in self.phases.items()},
                "optimizer_steps": sum(value.optimizer_steps for value in self.phases.values()),
                "decoder_parameters_in_predictor": any("decoder" in key for key in self.model.state_dict()),
                "architecture": deepcopy(self.model.architecture_metadata),
                "audit": deepcopy(self.audit)}


def fit_published_sae(x_train, y_train_scaled, x_val, y_val_scaled, target_scale,
                      config, *, fold_id, dropout=.3, fixed_epochs=None,
                      device="cpu", gpu_memory_limit_bytes=None, deterministic=True):
    """AE1 -> frozen-encoder AE2 -> transferred supervised fit, without NAS.

    All epochs, stopping, learning rate, batch and resource caps come from the
    supplied historical TrainConfig. Selection uses validation; a fresh refit
    requires three explicit fixed epoch counts and accepts no validation pair.
    Each primitive creates a fresh optimizer and restores its selected state.
    The caller, not a provenance string, owns chronological split admissibility.

    device='cpu' selects the CPU primitives explicitly. An explicit CUDA device
    uses the existing strict FP32/no-AMP/no-TF32 CUDA primitives with no fallback.
    """
    _verify_source_and_dropout(dropout)
    if not isinstance(config, training.TrainConfig):
        raise TypeError("A validated TrainConfig is required")
    if not isinstance(fold_id, str) or not fold_id.strip():
        raise ValueError("A nonempty training-fold identifier is required")
    if (type(target_scale) not in (int, float) or not math.isfinite(target_scale)
            or target_scale <= 0):
        raise ValueError("Training-only target scale must be positive and finite")
    if type(deterministic) is not bool:
        raise ValueError("Deterministic execution must be an explicit boolean")
    if fixed_epochs is not None:
        if (not isinstance(fixed_epochs, dict)
                or set(fixed_epochs) != {"ae1", "ae2", "supervised"}
                or any(type(n) is not int or not 1 <= n <= config.max_epochs
                       for n in fixed_epochs.values())):
            raise ValueError("Fixed refit needs all three stage epoch counts within the supplied cap")
        if x_val is not None or y_val_scaled is not None:
            raise ValueError("Fixed refit accepts no validation inputs or labels")
    x_train, y_train_scaled = training._pairs(
        x_train, y_train_scaled, supervised=True, training=True)
    if fixed_epochs is None:
        x_val, y_val_scaled = training._pairs(x_val, y_val_scaled, supervised=True, training=False)

    selected = torch.device(device)
    backend, kwargs, policy = training, {}, nullcontext()
    if selected.type == "cpu":
        if selected.index not in (None, 0) or gpu_memory_limit_bytes is not None:
            raise ValueError("CPU execution cannot accept a CUDA memory threshold or indexed CPU")
        selected = torch.device("cpu")
    else:
        try:
            from . import e2_cuda_training as cuda
        except ImportError:
            import e2_cuda_training as cuda
        selected = cuda.validate_cuda_device(selected)
        cuda._validate_gpu_limit(gpu_memory_limit_bytes)
        backend = cuda
        policy = cuda._execution_policy(selected, deterministic)
        kwargs = {"device": selected, "gpu_memory_limit_bytes": gpu_memory_limit_bytes,
                  "deterministic": deterministic}

    started, phases, boundary = time.perf_counter(), {}, "construction"
    audit = {"auxiliary_scale_fit": "training_windows_only",
             "training_input_sha256": _tensor_sha256(x_train),
             "training_chronology_certified_by_this_module": False,
             "device": str(selected), "phase_sequence": ["ae1", "ae2", "supervised"],
             "full_config": asdict(config), "fixed_epochs": deepcopy(fixed_epochs),
             "fresh_optimizer_per_phase": True, "retry_performed": False}

    def stage(name, function, *args):
        nonlocal boundary
        boundary = name
        result = function(*args, config,
                          fixed_epochs=fixed_epochs[name] if fixed_epochs else None, **kwargs)
        phases[name] = result
        return result

    try:
        with policy as runtime:
            audit["runtime"] = deepcopy(runtime)
            training._seed(config.seed)
            x_train, y_train_scaled = x_train.to(selected), y_train_scaled.to(selected)
            if fixed_epochs is None:
                x_val, y_val_scaled = x_val.to(selected), y_val_scaled.to(selected)
            manager = recipes.SAEPretraining(x_train, fold_id, accept_auxiliary_scale=True)
            ax, ay = manager.phase1_data()
            validation_target = manager.reconstruction_target(x_val) if fixed_epochs is None else None
            audit["auxiliary_training_target_sha256"] = _tensor_sha256(ay)
            audit["auxiliary_target"] = "original X_train / s in both AE phases"
            audit["validation_auxiliary_fraction_outside_unit_range"] = (
                float((validation_target.abs() > 1).float().mean())
                if validation_target is not None else None)
            stage("ae1", backend.fit_reconstruction, manager.ae1, ax, ay, x_val, validation_target)

            boundary = "phase2_preparation"
            bx, by = manager.prepare_phase2()
            if not torch.equal(ay, by):
                raise RuntimeError("Both auxiliary phases must reconstruct the same original training inputs")
            encoder1_before = training.state_sha256(manager.ae1.encoder)
            val_encoded = None
            if fixed_epochs is None:
                with torch.no_grad():
                    val_encoded = recipes.repeat7(recipes.last_state(manager.ae1.encoder, x_val)).detach().clone()
            stage("ae2", backend.fit_reconstruction, manager.ae2, bx, by, val_encoded, validation_target)
            encoder1_after = training.state_sha256(manager.ae1.encoder)
            if encoder1_before != encoder1_after or any(p.requires_grad for p in manager.ae1.encoder.parameters()):
                raise RuntimeError("The first encoder changed during the second auxiliary phase")

            boundary = "predictor_transfer"
            training._seed(config.seed)
            predictor = _annotate(manager.transfer_predictor(), dropout=dropout, seed=config.seed)
            audit.update(encoder1_before_ae2_sha256=encoder1_before,
                         encoder1_after_ae2_sha256=encoder1_after,
                         encoder1_frozen_during_ae2=True,
                         encoder1_transfer_exact=training.state_sha256(predictor.encoder1) == encoder1_after,
                         encoder2_transfer_exact=training.state_sha256(predictor.encoder2)
                            == training.state_sha256(manager.ae2.encoder))
            if not audit["encoder1_transfer_exact"] or not audit["encoder2_transfer_exact"]:
                raise RuntimeError("The predictor must receive exactly the selected pretrained encoders")
            audit["encoder1_before_supervised_sha256"] = training.state_sha256(predictor.encoder1)
            audit["encoder2_before_supervised_sha256"] = training.state_sha256(predictor.encoder2)
            stage("supervised", backend.fit_supervised, predictor, x_train, y_train_scaled,
                  x_val, y_val_scaled, target_scale)
            predictor.architecture_metadata["pretraining_executed"] = True
            audit.update(encoder1_after_supervised_sha256=training.state_sha256(predictor.encoder1),
                         encoder2_after_supervised_sha256=training.state_sha256(predictor.encoder2),
                         elapsed_seconds_total=time.perf_counter()-started,
                         phase_elapsed_seconds=sum(value.elapsed_seconds for value in phases.values()))
            return PublishedSAEResult(predictor, phases, float(manager.scale), fold_id, audit)
    except Exception as error:
        original = deepcopy(error.receipt) if isinstance(error, training.TrainingFailure) else {}
        known = original.get("completed_optimizer_steps", original.get("optimizer_steps", 0))
        completed_steps = sum(value.optimizer_steps for value in phases.values())
        receipt = {**original, "recipe_id": RECIPE_ID, "status": "failed",
                   "error_type": type(error).__name__, "error_code": original.get("error_code", training._failure_code(error)),
                   "message": str(error), "sae_phase": boundary, "fold_id": fold_id,
                   "completed_phases": {key: value.to_dict() for key, value in phases.items()},
                   "known_optimizer_steps_all_sae_phases": completed_steps + known if type(known) is int else None,
                   "sae_elapsed_seconds": time.perf_counter()-started,
                   "retry_performed": False, "audit": deepcopy(audit)}
        raise training.TrainingFailure(str(error), receipt) from error
