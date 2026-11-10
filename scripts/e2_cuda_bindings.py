"""Explicit CUDA callbacks for the frozen E2 selection/refit contracts.

Selection retains numerical records and predictions by default, not one large
weight file per candidate. Fresh final refits require an explicit checkpoint
store. This module supplies no market loader, search-domain pruning, retry or
CPU fallback; the CPU route is solely an explicitly enabled test fixture.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
import errno
from pathlib import Path
import time

import numpy as np
import torch

import e2_cuda_training as cuda
import e2_training_models as registry
from e2_nas_common import NASConfig
from e2_real_executor import FrozenSystem
from e2_staged_training import TrainConfig, TrainingFailure, state_sha256
from e2_temporal_training import TrialRun, RefitRun
from e2_training_bindings import candidate_for, run_neural_selection, run_selected_refits, selection_record

VERSION = "e2-cuda-selection-bindings-v1"


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _no_links(path):
    absolute = Path(path).absolute()
    if any(part.is_symlink() for part in (absolute, *absolute.parents)):
        raise ValueError("Checkpoint paths cannot contain symbolic links")


class _BudgetWriter:
    def __init__(self, handle, allowance):
        self.handle, self.allowance, self.written = handle, allowance, 0

    def write(self, value):
        if self.written+len(value) > self.allowance:
            raise OSError("Inference checkpoint exceeds the admitted storage budget")
        count = self.handle.write(value)
        self.written += count
        return count

    def flush(self):
        return self.handle.flush()


class CudaCheckpointStore:
    """Bounded, exclusive CPU inference checkpoints with verified commit receipts.

    No hard links are required. A weight file is complete only after its receipt
    has been flushed. Resuming rejects interrupted or corrupted writes, rather
    than silently adopting or deleting them. Sequential campaign writers only.
    """
    def __init__(self, directory, *, max_bytes, resume=False):
        if type(max_bytes) is not int or max_bytes < 1 or type(resume) is not bool:
            raise ValueError("An explicit positive storage budget and boolean resume flag are required")
        self.directory, self.max_bytes = Path(directory), max_bytes
        _no_links(self.directory)
        self.directory.mkdir(parents=True, exist_ok=resume)
        if not self.directory.is_dir():
            raise ValueError("Checkpoint destination must be a directory")
        self.directory_fsync_supported = self._flush_directory()
        if (self.directory/".checkpoint.lock").exists():
            raise RuntimeError("A checkpoint write needs reconciliation before resuming")
        for path in self.directory.iterdir():
            _no_links(path)
            if not path.is_file():
                raise ValueError("Checkpoint storage contains an unexpected directory")
            if path.suffix == ".pt":
                self.load_receipt(path.name)
            elif not path.name.endswith(".pt.receipt.json"):
                raise ValueError("Checkpoint storage contains an uncommitted file")
            elif not path.with_name(path.name.removesuffix(".receipt.json")).is_file():
                raise ValueError("Checkpoint receipt has no corresponding weights")
        if self._used_bytes() > self.max_bytes:
            raise OSError("Existing checkpoints exceed the admitted storage budget")

    def _used_bytes(self):
        return sum(p.stat().st_size for p in self.directory.iterdir()
                   if p.name != ".checkpoint.lock" and p.is_file())

    def _flush_directory(self):
        descriptor = None
        try:
            descriptor = os.open(self.directory, os.O_RDONLY)
            os.fsync(descriptor)
            return True
        except OSError as error:
            unsupported = {errno.EINVAL,errno.ENOTSUP,getattr(errno,"EOPNOTSUPP",errno.ENOTSUP)}
            if error.errno not in unsupported:
                raise
            return False
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def load_receipt(self, name):
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("Checkpoint filename must be a local .pt basename")
        target, receipt_path = self.directory/name, self.directory/(name+".receipt.json")
        _no_links(target); _no_links(receipt_path)
        if not target.is_file() or not receipt_path.is_file():
            raise ValueError("Inference checkpoint has no completed commit receipt")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("file") != name or receipt.get("bytes") != target.stat().st_size
                or receipt.get("sha256") != _file_sha(target)
                or receipt.get("optimizer_state_saved") is not False):
            raise ValueError("Inference checkpoint differs from its commit receipt")
        return receipt

    def load(self, receipt):
        verified = self.load_receipt(receipt["file"])
        if verified != receipt:
            raise ValueError("Caller checkpoint receipt differs from the committed receipt")
        return torch.load(self.directory/receipt["file"], weights_only=True, map_location="cpu")

    def save(self, run_id, payload):
        state = payload.get("state_dict")
        if (not isinstance(state, dict) or not state or
                any(not isinstance(v, torch.Tensor) or v.device.type != "cpu" for v in state.values())):
            raise ValueError("Persist explicit CPU inference tensors only")
        metadata = {k:v for k,v in payload.items() if k != "state_dict"}
        copied = {"state_dict":state, **json.loads(json.dumps(metadata, allow_nan=False))}
        name = _json_hash(run_id)+".pt"
        target, receipt_path = self.directory/name, self.directory/(name+".receipt.json")
        lock = self.directory/".checkpoint.lock"
        _no_links(self.directory)
        descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
        os.close(descriptor)
        created = []
        try:
            if target.exists() or receipt_path.exists() or target.is_symlink() or receipt_path.is_symlink():
                raise FileExistsError("Inference checkpoints are never replaced")
            # The bounded writer reserves room for the small commit receipt.
            allowance = self.max_bytes-self._used_bytes()-4096
            if allowance < 1:
                raise OSError("No checkpoint storage allowance remains")
            with target.open("xb") as handle:
                created.append(target)
                torch.save(copied, _BudgetWriter(handle, allowance))
                handle.flush(); os.fsync(handle.fileno())
            receipt = {"file":name, "bytes":target.stat().st_size,
                       "sha256":_file_sha(target), "optimizer_state_saved":False,
                       "weights_saved":True,"retention":"inference_weights", "storage_budget_bytes":self.max_bytes,
                       "directory_fsync_supported":self.directory_fsync_supported,
                       "durability_scope":"Committed file flushes and SHA256; underlying filesystem guarantees apply"}
            serialized = (json.dumps(receipt, sort_keys=True, allow_nan=False)+"\n").encode()
            if self._used_bytes()+len(serialized) > self.max_bytes:
                raise OSError("Checkpoint receipt exceeds the admitted storage budget")
            with receipt_path.open("xb") as handle:
                created.append(receipt_path)
                handle.write(serialized); handle.flush(); os.fsync(handle.fileno())
            if self.directory_fsync_supported and not self._flush_directory():
                raise OSError("Directory flush became unavailable during checkpoint commit")
            return receipt
        except BaseException:
            # Only files exclusively created by this call can be removed here.
            for path in reversed(created):
                path.unlink()
            raise
        finally:
            lock.unlink()


@dataclass
class CudaFittedNeural:
    name: str
    seed: int
    model: torch.nn.Module
    scaler: object
    checkpoint: dict
    phase_records: dict
    device: str
    allow_cpu_for_tests: bool = False
    deterministic: bool = True
    inference_batch_size: int = 32

    def frozen_system(self):
        model, scaler = self.model, self.scaler
        model.eval()
        def predict(target, raw_window):
            x = torch.as_tensor(scaler.transform(np.asarray(raw_window)[None]), dtype=torch.float32)
            values = cuda.predict_scaled(model, x, device=self.device,
                batch_size=self.inference_batch_size, allow_cpu_for_tests=self.allow_cpu_for_tests,
                deterministic=self.deterministic)
            if tuple(values.shape) != (1,):
                raise RuntimeError("Frozen CUDA predictor changed scalar geometry")
            return float(values[0])*scaler.target_scale
        def fingerprint():
            return {"model_state":state_sha256(model), "scaler":scaler.record(),
                    "module_modes":{name:module.training for name,module in model.named_modules()},
                    "dropout_rates":{name:float(module.p) for name,module in model.named_modules()
                                     if isinstance(module,torch.nn.Dropout)},
                    "device":self.device}
        return FrozenSystem(self.name+"/seed"+str(self.seed), "window",
                            predict=predict, fingerprint=fingerprint)


class CudaNeuralTrainingAdapter:
    def __init__(self, name, checkpoints=None, *, device="cuda:0", source_roots=None,
                 full_epochs=200, short_epochs=50, patience=20, wall_limit_seconds=7200,
                 short_wall_limit_seconds=1800, memory_limit_bytes=12*1024**3,
                 gpu_memory_limit_bytes=None, deterministic=True, allow_cpu_for_tests=False,
                 retain_selection_weights=False):
        if name not in registry.NEURAL_NAMES:
            raise ValueError("Unknown neural system")
        if any(type(v) is not bool for v in (deterministic, allow_cpu_for_tests, retain_selection_weights)):
            raise TypeError("Execution and retention controls must be explicit booleans")
        selected = cuda._resolve_device(device, allow_cpu_for_tests)
        if selected.type == "cuda" and not deterministic:
            raise ValueError("Scientific CUDA selection requires the registered deterministic policy")
        cuda._validate_gpu_limit(gpu_memory_limit_bytes)
        if retain_selection_weights and checkpoints is None:
            raise ValueError("Retaining selection weights requires an explicit checkpoint store")
        if checkpoints is not None and not callable(getattr(checkpoints, "save", None)):
            raise TypeError("An explicit inference checkpoint store is required")
        if checkpoints is not None and (type(getattr(checkpoints,"max_bytes",None)) is not int
                                         or checkpoints.max_bytes < 1):
            raise ValueError("Checkpoint persistence requires an explicit positive storage budget")
        self.name, self.store, self.roots = name, checkpoints, source_roots
        self.device = str(selected)
        self.full_epochs, self.short_epochs, self.patience = full_epochs, short_epochs, patience
        self.wall, self.short_wall, self.memory = wall_limit_seconds, short_wall_limit_seconds, memory_limit_bytes
        self.gpu_memory_limit_bytes, self.deterministic = gpu_memory_limit_bytes, deterministic
        self.allow_cpu_for_tests, self.retain_selection_weights = allow_cpu_for_tests, retain_selection_weights
        self.run_records, self._seen = [], set()
        # Reuse the validated config contract before any model construction.
        self._config({"learning_rate":.001,"weight_decay":.0001,"batch_size":32},0,"hpo")
        self._config({"learning_rate":.001,"weight_decay":.0001,"batch_size":32},0,"nas1")

    def _config(self, trial, seed, stage):
        return TrainConfig(learning_rate=trial["learning_rate"],weight_decay=trial["weight_decay"],
            batch_size=trial["batch_size"],max_epochs=self.short_epochs if stage=="nas1" else self.full_epochs,
            patience=self.patience,seed=seed,warmup_fraction=trial.get("warmup_fraction",0),
            wall_limit_seconds=self.short_wall if stage=="nas1" else self.wall,memory_limit_bytes=self.memory)

    def _run_id(self, candidate, seed, train, stage, validation=None):
        return _json_hash({"backend":VERSION,"model":self.name,"candidate":candidate.candidate_id,
                          "candidate_payload_sha256":candidate.payload_sha256,
                          "split":train.signature,"validation_split":None if validation is None else validation.signature,
                          "seed":seed,"stage":stage,"device":self.device})

    def _base_resources(self, run_id, stage, train, started):
        return {"version":VERSION,"run_id":run_id,"stage":stage,"device":self.device,
                "test_only_cpu_execution":self.device=="cpu", "training_split_signature":train.signature,
                "elapsed_seconds":time.perf_counter()-started,"retry_performed":False}

    def _fit(self, candidate, seed, train, validation, fixed_epochs=None):
        candidate.validate(); train.validate()
        if validation is not None: validation.validate()
        payload = deepcopy(candidate.payload)
        if payload.get("model") != self.name or payload.get("stage") not in ("hpo","nas1","nas2"):
            raise ValueError("Candidate does not belong to this model and scientific selection stage")
        stage = payload["stage"] if fixed_epochs is None else "refit"
        if fixed_epochs is not None and self.store is None:
            raise ValueError("A final refit requires an explicit bounded inference checkpoint store")
        run_id = self._run_id(candidate,seed,train,stage,validation)
        if run_id in self._seen:
            raise ValueError("A fit attempt cannot be repeated by the CUDA adapter")
        self._seen.add(run_id)
        trial, architecture = payload["trial"], payload.get("architecture")
        config = self._config(trial,seed,stage)
        if fixed_epochs is not None:
            expected_phases = {"ae1","ae2","supervised"} if self.name=="LSTM-autoencoder con predictor" else {"supervised"}
            if (not isinstance(fixed_epochs,dict) or set(fixed_epochs)!=expected_phases
                    or any(type(n) is not int or not 1<=n<=config.max_epochs for n in fixed_epochs.values())):
                raise ValueError("Fixed refit epochs must declare every stage within the registered horizon")
        started, phases, entered = time.perf_counter(), {}, False
        model = None
        try:
            model = registry.build_train_model(self.name,trial,architecture=architecture,
                                               source_roots=self.roots,seed=seed)
            construction = deepcopy(model.architecture_metadata)
            common = {"device":self.device,"allow_cpu_for_tests":self.allow_cpu_for_tests,
                      "gpu_memory_limit_bytes":self.gpu_memory_limit_bytes,"deterministic":self.deterministic}
            entered = True
            if self.name == "LSTM-autoencoder con predictor":
                del model
                model = None
                result = cuda.fit_sae(NASConfig(architecture["widths"],architecture["dropout_pattern"]),
                    train.x,train.y,None if validation is None else validation.x,
                    None if validation is None else validation.y,train.scaler.target_scale,config,
                    fold_id=train.signature,fixed_epochs=fixed_epochs,**common)
                model, phases, auxiliary_scale = result.model, result.phases, result.auxiliary_scale
            else:
                result = cuda.fit_supervised(model,train.x,train.y,
                    None if validation is None else validation.x,None if validation is None else validation.y,
                    train.scaler.target_scale,config,
                    fixed_epochs=None if fixed_epochs is None else fixed_epochs["supervised"],**common)
                phases, auxiliary_scale = {"supervised":result}, None
            model.eval()
            if any(p.optimizer_initial_state_entries != 0 for p in phases.values()):
                raise RuntimeError("A fresh CUDA fit reused optimizer state")
            epochs = {key:value.best_epoch for key,value in phases.items()}
            records = {key:value.to_dict() for key,value in phases.items()}
            steps = sum(p.optimizer_steps for p in phases.values())
            state_bytes = sum(t.numel()*t.element_size() for t in model.state_dict().values())
            checkpoint = {"retention":"metadata_only","weights_saved":False,"optimizer_state_saved":False,
                          "selected_state_sha256":phases["supervised"].selected_state_sha256,
                          "state_bytes":state_bytes}
            if fixed_epochs is not None or self.retain_selection_weights:
                try:
                    checkpoint = self.store.save(run_id,{"version":VERSION,"run_id":run_id,
                        "state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        "model_metadata":construction,"scaler":train.scaler.record(),
                        "candidate_id":candidate.candidate_id,"candidate_payload_sha256":candidate.payload_sha256,
                        "phase_epochs":epochs,"phase_records":records,"auxiliary_scale":auxiliary_scale,
                        "training_split_signature":train.signature,"optimizer_state_saved":False})
                    checkpoint = {**checkpoint,"weights_saved":True}
                except Exception as error:
                    resources = {**self._base_resources(run_id,stage,train,started),
                        "status":"failed","error_code":"checkpoint_persistence", "optimizer_steps":steps,
                        "optimizer_steps_known_lower_bound":steps,"partial_optimizer_update_may_have_occurred":False,
                        "optimization_stages":len(phases),"phase_records":records}
                    raise TrainingFailure("CUDA checkpoint persistence failed: "+str(error),resources) from error
            resources = {**self._base_resources(run_id,stage,train,started),
                "model_parameters":sum(p.numel() for p in model.parameters()),"state_bytes":state_bytes,
                "optimizer_steps":steps,"optimizer_steps_known_lower_bound":steps,
                "partial_optimizer_update_may_have_occurred":False,"optimization_stages":len(phases),
                "checkpoint":checkpoint,"phase_records":records,"training_axes_applied":True,
                "model_construction_metadata":construction,
                "max_cuda_memory_allocated_bytes":max(getattr(p,"max_cuda_memory_allocated_bytes",0) for p in phases.values()),
                "max_cuda_memory_reserved_bytes":max(getattr(p,"max_cuda_memory_reserved_bytes",0) for p in phases.values())}
            artifact = CudaFittedNeural(self.name,seed,model,train.scaler,checkpoint,records,
                self.device,self.allow_cpu_for_tests,self.deterministic,trial["batch_size"])
            return run_id,resources,epochs,artifact
        except TrainingFailure as error:
            receipt = {**error.receipt,"adapter_elapsed_seconds":time.perf_counter()-started,
                       "run_id":run_id,"stage":stage,"training_split_signature":train.signature}
            raise TrainingFailure(str(error),receipt) from error
        except Exception as error:
            completed = sum(p.optimizer_steps for p in phases.values())
            unknown = entered and not phases
            resources = {**self._base_resources(run_id,stage,train,started),
                "status":"failed","error_type":type(error).__name__,"error_code":"adapter_runtime" if entered else "construction",
                "optimizer_steps":None if unknown else completed,"optimizer_steps_known_lower_bound":completed,
                "partial_optimizer_update_may_have_occurred":unknown,"optimization_stages":len(phases) or 1,
                "phase_records":{k:v.to_dict() for k,v in phases.items()}}
            raise TrainingFailure(str(error),resources) from error

    def _failed_resources(self, error):
        resources = deepcopy(error.receipt)
        previous = resources.get("completed_phases",{})
        prior_steps = sum(p.get("optimizer_steps",0) for p in previous.values())
        current = resources.get("completed_optimizer_steps",resources.get("optimizer_steps"))
        explicit_lower = resources.get("optimizer_steps_known_lower_bound")
        phase_lower = (current if type(current) is int else explicit_lower if type(explicit_lower) is int
                       else resources.get("optimizer_steps") if type(resources.get("optimizer_steps")) is int else 0)
        if previous:
            lower = resources.get("known_optimizer_steps_all_sae_phases")
            if type(lower) is not int:
                lower = prior_steps+phase_lower
            resources["failed_phase_optimizer_steps"] = current
            resources["optimization_stages"] = len(previous)+1
        else:
            lower = explicit_lower if type(explicit_lower) is int else phase_lower
            resources.setdefault("optimization_stages",1)
        uncertain = resources.get("partial_optimizer_update_may_have_occurred",False) or current is None
        resources["optimizer_steps"] = None if uncertain else lower
        resources["optimizer_steps_known_lower_bound"] = lower
        resources["partial_optimizer_update_may_have_occurred"] = bool(uncertain)
        resources.update(device=self.device,test_only_cpu_execution=self.device=="cpu",retry_performed=False)
        return resources

    @staticmethod
    def _fatal(resources):
        return resources.get("error_code") not in ("resource_limit","nonfinite")

    def _failed_record(self, candidate, seed, train, stage, error, validation=None):
        resources = self._failed_resources(error)
        run_id = self._run_id(candidate,seed,train,stage,validation)
        self.run_records.append({"run_id":run_id,"seed":seed,"stage":stage,"status":"failed",
            "candidate_id":candidate.candidate_id,"training_split_signature":train.signature,"resource_use":resources})
        return run_id,resources

    def fit_candidate(self, candidate, seed, train, validation):
        artifact = None
        resources = None
        try:
            run_id,resources,epochs,artifact = self._fit(candidate,seed,train,validation)
            try:
                prediction = cuda.predict_scaled(artifact.model,validation.x,device=self.device,
                    batch_size=candidate.payload["trial"]["batch_size"],allow_cpu_for_tests=self.allow_cpu_for_tests,
                    deterministic=self.deterministic).numpy()*train.scaler.target_scale
                if prediction.shape != (len(validation.dates),):
                    raise cuda.cpu.TrainingGeometryError("Complete original-unit validation predictions are required")
                if not np.isfinite(prediction).all():
                    raise cuda.cpu.TrainingNumericalError("Nonfinite original-unit validation predictions")
                train.validate(); validation.validate()
                if state_sha256(artifact.model) != artifact.phase_records["supervised"]["selected_state_sha256"]:
                    raise cuda.cpu.TrainingCheckpointError("Validation inference changed the selected model state")
            except Exception as error:
                code = cuda.cpu._failure_code(error)
                receipt = {**resources,"error_code":code if code!="runtime_error" else "validation_inference","status":"failed",
                           "error_type":type(error).__name__}
                raise TrainingFailure("CUDA validation inference failed: "+str(error),receipt) from error
            self.run_records.append({"run_id":run_id,"seed":seed,"stage":candidate.payload["stage"],
                "status":"completed","candidate_id":candidate.candidate_id,"phase_epochs":epochs,
                "training_split_signature":train.signature,"resource_use":resources})
            return TrialRun("completed",run_id,resources,predictions_original=prediction,
                            prediction_dates=validation.dates,phase_epochs=epochs)
        except TrainingFailure as error:
            run_id,resources = self._failed_record(candidate,seed,train,candidate.payload["stage"],error,validation)
            return TrialRun.failed(str(error),resources,run_id,fatal=self._fatal(resources))
        except Exception as error:
            receipt = ({**resources,"error_code":"adapter_result","error_type":type(error).__name__}
                       if resources is not None else
                       {"optimizer_steps":0,"optimizer_steps_known_lower_bound":0,
                        "error_code":"admission","error_type":type(error).__name__,
                        "partial_optimizer_update_may_have_occurred":False})
            failure = TrainingFailure(str(error),receipt)
            run_id,resources = self._failed_record(candidate,seed,train,candidate.payload.get("stage","hpo"),failure,validation)
            return TrialRun.failed(str(error),resources,run_id,fatal=True)
        finally:
            # Selection returns numerical predictions/records, never an artifact.
            # Dropping these references releases the trial model after scoring.
            artifact = None

    def refit_candidate(self, candidate, seed, train, fixed_epochs):
        resources = None
        try:
            run_id,resources,epochs,artifact = self._fit(candidate,seed,train,None,fixed_epochs)
            self.run_records.append({"run_id":run_id,"seed":seed,"stage":"refit","status":"completed",
                "candidate_id":candidate.candidate_id,"phase_epochs":epochs,
                "training_split_signature":train.signature,"resource_use":resources})
            return RefitRun("completed",run_id,resources,phase_epochs=epochs,
                            fresh_initialization=True,artifact=artifact)
        except TrainingFailure as error:
            run_id,resources = self._failed_record(candidate,seed,train,"refit",error)
            return RefitRun("failed",run_id,resources,reason=str(error),fresh_initialization=True)
        except Exception as error:
            receipt = ({**resources,"error_code":"adapter_result","error_type":type(error).__name__}
                       if resources is not None else {"optimizer_steps":0,"optimizer_steps_known_lower_bound":0,
                       "partial_optimizer_update_may_have_occurred":False,"error_code":"admission","error_type":type(error).__name__})
            failure = TrainingFailure(str(error),receipt)
            run_id,resources = self._failed_record(candidate,seed,train,"refit",failure)
            return RefitRun("failed",run_id,resources,reason=str(error),fresh_initialization=True)


# Explicit aliases ease integration without modifying the historical CPU module.
NeuralTrainingAdapter = CudaNeuralTrainingAdapter
FittedNeural = CudaFittedNeural
CheckpointStore = CudaCheckpointStore
