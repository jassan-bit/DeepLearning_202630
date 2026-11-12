"""Training/selection callbacks for prospectively restored published D7 bodies.

The existing temporal, optimization and checkpoint primitives remain unchanged.
The adapter replaces construction and fixed-SAE staging under new recipe IDs;
it never edits sealed campaigns or adopts their differently specified weights.
"""
from __future__ import annotations

from copy import deepcopy
import time

import torch

try:
    from . import e2_training_bindings as previous
    from . import e2_published_registry as registry
    from .e2_published_sae import fit_published_sae
    from .e2_staged_training import TrainingFailure, fit_supervised
    from .e2_temporal_training import select_b
    from .e2_temporal_training import TrialRun, RefitRun
except ImportError:
    import e2_training_bindings as previous
    import e2_published_registry as registry
    from e2_published_sae import fit_published_sae
    from e2_staged_training import TrainingFailure, fit_supervised
    from e2_temporal_training import select_b
    from e2_temporal_training import TrialRun, RefitRun

VERSION = "e2-published-training-v1"


class PublishedTrainingAdapter(previous.NeuralTrainingAdapter):
    architecture_policy_id = registry.POLICY_ID

    def __init__(self, name, checkpoints, *, device="cpu", gpu_memory_limit_bytes=None, **kwargs):
        registry.model_policy(name)
        super().__init__(name, checkpoints, **kwargs)
        self.device = str(device)
        self.gpu_memory_limit_bytes = gpu_memory_limit_bytes
        if self.device != "cpu":
            raise ValueError("This prospective callback is CPU-verified only; CUDA fitting/inference requires the separate sealed-campaign migration")

    def _run_id(self, candidate, seed, train, fixed_epochs=None):
        return previous._json_hash({"model": self.name, "candidate": candidate.candidate_id,
            "split": train.signature, "seed": seed,
            "stage": candidate.payload["stage"] if fixed_epochs is None else "refit"})

    def _fit(self, candidate, seed, train, validation, fixed_epochs=None):
        if self.name not in registry.RESTORED:
            return super()._fit(candidate, seed, train, validation, fixed_epochs)
        candidate.validate()
        payload = deepcopy(candidate.payload)
        if payload.get("model") != self.name or payload.get("architecture") is not None:
            raise ValueError("Use this fixed published body's own candidate without NAS overrides")
        stage = payload["stage"] if fixed_epochs is None else "refit"
        run_id = self._run_id(candidate, seed, train, fixed_epochs)
        if run_id in self._seen:
            raise ValueError("A fit attempt cannot be repeated")
        checkpoint_path = self.store.directory / (previous._json_hash(run_id) + ".pt")
        if checkpoint_path.exists() or checkpoint_path.with_suffix(".pt.tmp").exists():
            raise ValueError("An existing or interrupted checkpoint needs reconciliation before any new fit")
        self._seen.add(run_id)
        trial = payload["trial"]
        config = self._config(trial, seed, stage)
        start = time.perf_counter()
        model = registry.build_train_model(self.name, trial, source_roots=self.roots, seed=seed)
        metadata = deepcopy(model.architecture_metadata)
        auxiliary_scale = None
        if self.name == "LSTM-autoencoder con predictor":
            del model
            result = fit_published_sae(train.x, train.y,
                None if validation is None else validation.x,
                None if validation is None else validation.y,
                train.scaler.target_scale, config, fold_id=train.signature,
                dropout=trial["dropout"], fixed_epochs=fixed_epochs,
                device=self.device, gpu_memory_limit_bytes=self.gpu_memory_limit_bytes)
            model, phases, auxiliary_scale = result.model, result.phases, result.auxiliary_scale
            metadata.update(result.model.architecture_metadata)
            metadata["published_sae_audit"] = deepcopy(result.audit)
        else:
            options = {"fixed_epochs": None if fixed_epochs is None else fixed_epochs["supervised"]}
            result = fit_supervised(model, train.x, train.y,
                None if validation is None else validation.x,
                None if validation is None else validation.y,
                train.scaler.target_scale, config, **options)
            phases = {"supervised": result}
        model.cpu().eval()
        metadata.update(architecture_policy_id=registry.POLICY_ID,
                        training_recipe_id=trial["recipe_id"], training_axes_applied=True)
        model.architecture_metadata = deepcopy(metadata)
        epochs = {key: value.best_epoch for key, value in phases.items()}
        records = {key: value.to_dict() for key, value in phases.items()}
        try:
            checkpoint = self.store.save(run_id, {"version": VERSION, "run_id": run_id,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "model_metadata": metadata, "scaler": train.scaler.record(),
                "phase_epochs": epochs, "phase_records": records,
                "auxiliary_scale": auxiliary_scale, "training_split_signature": train.signature,
                "optimizer_state_saved": False})
        except Exception as error:
            raise TrainingFailure("Checkpoint persistence failed: " + str(error),
                {"status": "failed", "error_code": "checkpoint_persistence", "run_id": run_id,
                 "seed": seed, "stage": stage, "optimizer_steps": sum(p.optimizer_steps for p in phases.values()),
                 "optimization_stages": len(phases), "phase_records": records,
                 "elapsed_seconds": time.perf_counter() - start, "retry_performed": False}) from error
        resources = {"model_parameters": sum(p.numel() for p in model.parameters()),
            "optimizer_steps": sum(p.optimizer_steps for p in phases.values()),
            "optimization_stages": len(phases), "elapsed_seconds": time.perf_counter() - start,
            "checkpoint": checkpoint, "phase_records": records, "training_axes_applied": True,
            "model_construction_metadata": metadata, "device": self.device}
        if any(p.optimizer_initial_state_entries != 0 for p in phases.values()):
            raise RuntimeError("Optimizer state was reused")
        self.run_records.append({"run_id": run_id, "seed": seed, "stage": stage,
            "status": "completed", "candidate_id": candidate.candidate_id,
            "phase_epochs": epochs, "training_split_signature": train.signature,
            "resource_use": resources})
        artifact = previous.FittedNeural(self.name, seed, model, train.scaler, checkpoint, records)
        return run_id, resources, epochs, artifact

    def _record_failure(self, error, candidate, seed, train, *, refit=False, known_resources=None):
        run_id = self._run_id(candidate, seed, train, {} if refit else None)
        known_training_failure = isinstance(error, TrainingFailure)
        if known_resources is not None:
            resources = deepcopy(known_resources)
            resources.update(error_type=type(error).__name__, reason=str(error),
                             error_code="prediction", systematic_failure=True,
                             stage_where_failure="validation_prediction", retry_performed=False)
        else:
            resources = self._failed_resources(error) if known_training_failure else {
                "error_type": type(error).__name__, "reason": str(error),
                "systematic_failure": True, "optimizer_steps": 0, "retry_performed": False}
        record = {"status": "failed", "run_id": run_id,
            "candidate_id": candidate.candidate_id, "seed": seed,
            "stage": "refit" if refit else candidate.payload["stage"],
            "training_split_signature": train.signature, "resource_use": resources}
        existing = next((i for i in range(len(self.run_records)-1, -1, -1)
                         if self.run_records[i]["run_id"] == run_id), None)
        if known_resources is not None and existing is not None:
            self.run_records[existing] = record
        else:
            self.run_records.append(record)
        if refit:
            return RefitRun("failed", run_id, resources, reason=str(error), fresh_initialization=True)
        fatal = (not known_training_failure or resources.get("error_code") in
                 ("geometry", "missing_gradient", "checkpoint", "checkpoint_persistence"))
        return TrialRun.failed(str(error), resources, run_id, fatal=fatal)

    def fit_candidate(self, candidate, seed, train, validation):
        if self.name not in registry.RESTORED:
            return super().fit_candidate(candidate, seed, train, validation)
        resources = None
        try:
            run_id, resources, epochs, artifact = self._fit(candidate, seed, train, validation)
            with torch.no_grad():
                values = artifact.model(validation.x)
                if tuple(values.shape) != (len(validation.dates),) or not bool(torch.isfinite(values).all()):
                    raise ValueError("Published validation predictions must be finite aligned scalars")
                predictions = values.detach().cpu().numpy() * train.scaler.target_scale
            return TrialRun("completed", run_id, resources, predictions_original=predictions,
                            prediction_dates=validation.dates, phase_epochs=epochs)
        except Exception as error:
            return self._record_failure(error, candidate, seed, train, known_resources=resources)

    def refit_candidate(self, candidate, seed, train, fixed_epochs):
        if self.name not in registry.RESTORED:
            return super().refit_candidate(candidate, seed, train, fixed_epochs)
        try:
            run_id, resources, epochs, artifact = self._fit(candidate, seed, train, None, fixed_epochs)
            return RefitRun("completed", run_id, resources, phase_epochs=epochs,
                            fresh_initialization=True, artifact=artifact)
        except Exception as error:
            return self._record_failure(error, candidate, seed, train, refit=True)


def run_neural_selection(prepared, name, adapter, *, hpo_trials, architectures,
                         hpo_seeds, phase1_seeds, phase2_seeds, k, delta,
                         scope="registered_domains"):
    """NAS is admitted only for TiDE's retained variant; fixed bodies get HPO."""
    registry.model_policy(name)
    if name not in registry.RESTORED:
        return previous.run_neural_selection(prepared, name, adapter,
            hpo_trials=hpo_trials, architectures=architectures, hpo_seeds=hpo_seeds,
            phase1_seeds=phase1_seeds, phase2_seeds=phase2_seeds, k=k, delta=delta, scope=scope)
    if (not isinstance(adapter, PublishedTrainingAdapter) or adapter.name != name
            or adapter.architecture_policy_id != registry.POLICY_ID):
        raise ValueError("Restored bodies require the versioned published adapter; migrate the sealed controller first")
    if tuple(architectures):
        raise ValueError("NAS is not admitted for this fixed published recipe")
    if scope not in ("registered_domains", "synthetic_fixture"):
        raise ValueError("Declare registered domains or a bounded synthetic fixture")
    trials = tuple(hpo_trials)
    if scope == "registered_domains":
        plan = registry.historical.protocol()
        if (list(trials) != registry.registered_trials(name)
                or tuple(hpo_seeds) != tuple(plan["common_hpo"]["optimization_seeds"])
                or delta != plan["contract"]["delta"]
                or (adapter.full_epochs, adapter.short_epochs, adapter.patience) != (200, 50, 20)
                or adapter.wall != plan["hardware"]["full_stage_wall_limit_seconds"]
                or adapter.short_wall != plan["hardware"]["short_stage_wall_limit_seconds"]
                or adapter.memory != plan["hardware"]["fit_process_memory_limit_bytes"]):
            raise ValueError("Keep the registered HPO effort, seeds, margin, epochs and runtime caps")
    if not trials or any(t not in registry.registered_trials(name) for t in trials):
        raise ValueError("Unknown current published-body HPO point")
    candidates = tuple(previous.candidate_for(name, trial, None, "hpo") for trial in trials)
    selection = select_b(prepared, candidates, hpo_seeds, adapter.fit_candidate)
    return selection, {"hpo": selection}


def run_selected_refits(prepared, selection, adapter, seeds, **kwargs):
    name = selection.winner.payload["model"]
    registry.model_policy(name)
    if name in registry.RESTORED:
        if (not isinstance(adapter, PublishedTrainingAdapter) or adapter.name != name
                or selection.winner.payload["trial"] not in registry.registered_trials(name)
                or selection.winner.payload.get("architecture") is not None):
            raise ValueError("Historical NAS winners cannot be relabelled as restored published recipes")
    return previous.run_selected_refits(prepared, selection, adapter, seeds, **kwargs)
