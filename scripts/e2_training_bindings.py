"""Staged neural selection/refit bindings and durable synthetic integration.

No market loader or search launcher is supplied. Public callbacks consume only
prepared inner/refit splits. The CLI below generates its own artificial panel.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from e2_nas_common import NASConfig
from e2_nas_factory import MODEL_IDS
import e2_search_plan as planner
import e2_training_models as registry
from e2_staged_training import (TrainConfig, TrainingFailure, fit_supervised,
                                fit_sae, state_sha256)
from e2_temporal_training import (CandidateSpec, TrialRun, RefitRun, select_b,
                                 refit_b, top_k_with_reference)
from e2_real_executor import FrozenSystem

VERSION = "e2-training-bindings-v1"


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class CheckpointStore:
    """Atomic, exclusive inference checkpoints; never overwrite an existing run."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)

    def save(self, run_id, payload):
        state=payload.get("state_dict")
        if not isinstance(state,dict) or any(not isinstance(v,torch.Tensor) for v in state.values()):
            raise ValueError("Checkpoint state must be an explicit tensor dictionary")
        metadata={k:v for k,v in payload.items() if k!="state_dict"}
        payload={"state_dict":state,**json.loads(json.dumps(metadata,allow_nan=False))}
        name = _json_hash(run_id) + ".pt"
        target = self.directory/name
        temporary = self.directory/(name+".tmp")
        if target.exists():
            raise FileExistsError("A run checkpoint already exists")
        try:
            with temporary.open("xb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, target)  # Exclusive atomic link, no replacement.
            temporary.unlink()
            descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise
        return {"file": name, "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "optimizer_state_saved": False}


@dataclass
class FittedNeural:
    name: str
    seed: int
    model: torch.nn.Module
    scaler: object
    checkpoint: dict
    phase_records: dict

    def frozen_system(self):
        model, scaler = self.model, self.scaler
        model.eval()
        def predict(target, raw_window):
            x = torch.as_tensor(scaler.transform(np.asarray(raw_window)[None]), dtype=torch.float32)
            with torch.no_grad():
                prediction = model(x)
            if tuple(prediction.shape) != (1,):
                raise RuntimeError("Frozen neural predictor changed scalar shape")
            return float(prediction[0])*scaler.target_scale
        def fingerprint():
            return {"model_state": state_sha256(model), "scaler": scaler.record(),
                    "module_modes":{name:module.training for name,module in model.named_modules()},
                    "dropout_rates":{name:float(module.p) for name,module in model.named_modules()
                                     if isinstance(module,torch.nn.Dropout)}}
        return FrozenSystem(self.name+"/seed"+str(self.seed), "window",
                            predict=predict, fingerprint=fingerprint)


class NeuralTrainingAdapter:
    def __init__(self, name, checkpoints, *, source_roots=None, full_epochs=200,
                 short_epochs=50, patience=20, wall_limit_seconds=7200,
                 short_wall_limit_seconds=1800,memory_limit_bytes=12*1024**3):
        if name not in registry.NEURAL_NAMES:
            raise ValueError("Unknown neural system")
        self.name, self.store, self.roots = name, checkpoints, source_roots
        self.full_epochs, self.short_epochs, self.patience = full_epochs, short_epochs, patience
        self.wall, self.memory = wall_limit_seconds, memory_limit_bytes
        self.short_wall = short_wall_limit_seconds
        self.run_records = []
        self._seen = set()

    def _config(self, trial, seed, stage):
        return TrainConfig(learning_rate=trial["learning_rate"], weight_decay=trial["weight_decay"],
            batch_size=trial["batch_size"], max_epochs=self.short_epochs if stage=="nas1" else self.full_epochs,
            patience=self.patience, seed=seed, warmup_fraction=trial.get("warmup_fraction", 0),
            wall_limit_seconds=self.short_wall if stage=="nas1" else self.wall,
            memory_limit_bytes=self.memory)

    def _fit(self, candidate, seed, train, validation, fixed_epochs=None):
        candidate.validate()
        payload=deepcopy(candidate.payload)
        if payload.get("model") != self.name:
            raise ValueError("Candidate belongs to another neural family")
        stage=payload["stage"] if fixed_epochs is None else "refit"
        run_id=_json_hash({"model":self.name,"candidate":candidate.candidate_id,
                          "split":train.signature,"seed":seed,"stage":stage})
        if run_id in self._seen:
            raise ValueError("A fit attempt cannot be repeated")
        self._seen.add(run_id)
        trial, architecture = payload["trial"], payload.get("architecture")
        config=self._config(trial,seed,stage)
        start=time.perf_counter()
        model=registry.build_train_model(self.name,trial,architecture=architecture,
                                        source_roots=self.roots,seed=seed)
        construction_metadata=deepcopy(model.architecture_metadata)
        if self.name=="LSTM-autoencoder con predictor":
            del model
            result=fit_sae(NASConfig(architecture["widths"],architecture["dropout_pattern"]),
                train.x,train.y,None if validation is None else validation.x,
                None if validation is None else validation.y,train.scaler.target_scale,
                config,fold_id=train.signature,fixed_epochs=fixed_epochs)
            model=result.model
            phases=result.phases
            auxiliary_scale=result.auxiliary_scale
        else:
            result=fit_supervised(model,train.x,train.y,
                None if validation is None else validation.x,
                None if validation is None else validation.y,
                train.scaler.target_scale,config,
                fixed_epochs=None if fixed_epochs is None else fixed_epochs["supervised"])
            phases={"supervised":result}
            auxiliary_scale=None
        model.eval()
        phase_epochs={key:value.best_epoch for key,value in phases.items()}
        records={key:value.to_dict() for key,value in phases.items()}
        try:
            checkpoint=self.store.save(run_id,{"version":VERSION,"run_id":run_id,
                "state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                "model_metadata":construction_metadata,"scaler":train.scaler.record(),
                "phase_epochs":phase_epochs,"phase_records":records,
                "auxiliary_scale":auxiliary_scale,"training_split_signature":train.signature,
                "optimizer_state_saved":False})
        except Exception as error:
            raise TrainingFailure("Checkpoint persistence failed: "+str(error),
                {"status":"failed","error_code":"checkpoint_persistence",
                 "run_id":run_id,"seed":seed,"stage":stage,
                 "optimizer_steps":sum(p.optimizer_steps for p in phases.values()),
                 "optimization_stages":len(phases),"phase_records":records,
                 "elapsed_seconds":time.perf_counter()-start,
                 "training_split_signature":train.signature,
                 "retry_performed":False}) from error
        resource={"model_parameters":sum(p.numel() for p in model.parameters()),
                  "optimizer_steps":sum(p.optimizer_steps for p in phases.values()),
                  "optimization_stages":len(phases),"elapsed_seconds":time.perf_counter()-start,
                  "checkpoint":checkpoint,"phase_records":records,"training_axes_applied":True,
                  "model_construction_metadata":construction_metadata}
        if any(record.optimizer_initial_state_entries != 0 for record in phases.values()):
            raise RuntimeError("Optimizer state was reused")
        record={"run_id":run_id,"seed":seed,"stage":stage,"status":"completed",
                "candidate_id":candidate.candidate_id,"phase_epochs":phase_epochs,
                "training_split_signature":train.signature,"resource_use":resource}
        self.run_records.append(record)
        artifact=FittedNeural(self.name,seed,model,train.scaler,checkpoint,records)
        return run_id,resource,phase_epochs,artifact

    def _failed_resources(self,error):
        resources=deepcopy(error.receipt)
        if "completed_phases" in resources:
            resources["failed_phase_optimizer_steps"]=resources.get("optimizer_steps",0)
            resources["optimizer_steps"]=resources.get("optimizer_steps",0)+sum(
                p.get("optimizer_steps",0) for p in resources["completed_phases"].values())
            resources["optimization_stages"]=len(resources["completed_phases"])+1
        else:
            resources.setdefault("optimization_stages",1)
            resources.setdefault("optimizer_steps",0)
        return resources

    def fit_candidate(self,candidate,seed,train,validation):
        try:
            run_id,resources,epochs,artifact=self._fit(candidate,seed,train,validation)
            with torch.no_grad():
                prediction=artifact.model(validation.x).detach().cpu().numpy()*train.scaler.target_scale
            return TrialRun("completed",run_id,resources,predictions_original=prediction,
                            prediction_dates=validation.dates,phase_epochs=epochs)
        except TrainingFailure as error:
            run_id=_json_hash({"model":self.name,"candidate":candidate.candidate_id,
                              "seed":seed,"split":train.signature,"stage":candidate.payload["stage"]})
            resources=self._failed_resources(error)
            receipt={"status":"failed","run_id":run_id,"candidate_id":candidate.candidate_id,
                     "seed":seed,"stage":candidate.payload["stage"],"resource_use":resources}
            self.run_records.append(receipt)
            fatal=resources.get("error_code") in ("geometry","missing_gradient","checkpoint",
                                                   "checkpoint_persistence")
            return TrialRun.failed(str(error),resources,run_id,fatal=fatal)
        except Exception as error:
            run_id=_json_hash({"construction_failure":candidate.candidate_id,"seed":seed,
                               "split":train.signature})
            receipt={"status":"failed","run_id":run_id,"candidate_id":candidate.candidate_id,
                     "seed":seed,"error_type":type(error).__name__,"reason":str(error),
                     "systematic_failure":True,"retry_performed":False}
            self.run_records.append(receipt)
            return TrialRun.failed(str(error),receipt,run_id,fatal=True)

    def refit_candidate(self,candidate,seed,train,fixed_epochs):
        try:
            run_id,resources,epochs,artifact=self._fit(candidate,seed,train,None,fixed_epochs)
            return RefitRun("completed",run_id,resources,phase_epochs=epochs,
                            fresh_initialization=True,artifact=artifact)
        except TrainingFailure as error:
            run_id=_json_hash({"model":self.name,"candidate":candidate.candidate_id,
                              "seed":seed,"split":train.signature,"stage":"refit"})
            resources=self._failed_resources(error)
            self.run_records.append({"status":"failed","run_id":run_id,
                "candidate_id":candidate.candidate_id,"seed":seed,"stage":"refit",
                "training_split_signature":train.signature,"resource_use":resources})
            return RefitRun("failed",run_id,resources,reason=str(error),fresh_initialization=True)
        except Exception as error:
            run_id=_json_hash({"model":self.name,"candidate":candidate.candidate_id,
                              "seed":seed,"split":train.signature,"stage":"refit"})
            resources={"error_type":type(error).__name__,"reason":str(error),
                       "systematic_failure":True,"optimizer_steps":0,"retry_performed":False}
            self.run_records.append({"status":"failed","run_id":run_id,"stage":"refit",
                                     "training_split_signature":train.signature,"resource_use":resources})
            return RefitRun("failed",run_id,resources,reason=str(error),fresh_initialization=True)


def candidate_for(name,trial,architecture,stage):
    payload={"model":name,"trial":deepcopy(trial),"architecture":deepcopy(architecture),"stage":stage}
    identity=stage+":"+trial["trial_id"]+(":"+architecture["architecture_id"] if architecture else "")
    return CandidateSpec(identity,payload)


def run_neural_selection(prepared,name,adapter,*,hpo_trials,architectures,
                         hpo_seeds,phase1_seeds,phase2_seeds,k,delta,
                         scope="registered_domains"):
    """Full algorithm over explicitly supplied registered domains; no hidden pruning."""
    if scope not in ("registered_domains","synthetic_fixture"):
        raise ValueError("Declare registered domains or bounded synthetic fixture")
    hpo_trials, architectures = tuple(hpo_trials), tuple(architectures)
    if scope=="registered_domains":
        if list(hpo_trials)!=registry.registered_trials(name):
            raise ValueError("Registered search cannot shorten or alter its eighteen HPO rows")
        plan=registry.protocol()
        if (type(delta) not in (float,int) or float(delta)!=plan["contract"]["delta"]):
            raise ValueError("Registered selection cannot change the inherited practical margin")
        if (adapter.wall!=plan["hardware"]["full_stage_wall_limit_seconds"]
                or adapter.short_wall!=plan["hardware"]["short_stage_wall_limit_seconds"]
                or adapter.memory!=plan["hardware"]["fit_process_memory_limit_bytes"]):
            raise ValueError("Registered runtime caps cannot be silently changed")
        if tuple(hpo_seeds)!=tuple(plan["common_hpo"]["optimization_seeds"]):
            raise ValueError("Registered HPO seed pool differs")
        if name in MODEL_IDS:
            if (list(architectures)!=planner.enumerate_architectures(plan,name)
                    or k not in plan["nas"]["k_options"]
                    or tuple(phase1_seeds)!=(plan["nas"]["phase1_seed"],)
                    or tuple(phase2_seeds)!=tuple(plan["nas"]["phase2_seeds"])):
                raise ValueError("Registered NAS cannot prune domains, finalists or seed pools")
        if (adapter.full_epochs,adapter.short_epochs,adapter.patience)!=(200,50,20):
            raise ValueError("Registered epoch budgets cannot be silently shortened")
    reference=registry.reference_architecture(name) if name in MODEL_IDS else None
    hpo_candidates=tuple(candidate_for(name,trial,reference,"hpo") for trial in hpo_trials)
    hpo=select_b(prepared,hpo_candidates,hpo_seeds,adapter.fit_candidate)
    stages={"hpo":hpo}
    if name not in MODEL_IDS:
        return hpo,stages
    chosen_trial=hpo.winner.payload["trial"]
    if len(phase1_seeds)!=1 or len(phase2_seeds)<5 or set(phase1_seeds)&set(phase2_seeds):
        raise ValueError("NAS needs one screening seed and five distinct confirmation seeds")
    first_candidates=tuple(candidate_for(name,chosen_trial,a,"nas1") for a in architectures)
    reference_candidate=next((c for c in first_candidates
        if c.payload["architecture"]["architecture_id"]==reference["architecture_id"]),None)
    if reference_candidate is None:
        raise ValueError("Explicit architecture domain must include the reference")
    phase1=select_b(prepared,first_candidates,phase1_seeds,adapter.fit_candidate)
    finalists=top_k_with_reference(prepared,phase1,first_candidates,k,reference_candidate.candidate_id)
    second_candidates=tuple(candidate_for(name,chosen_trial,c.payload["architecture"],"nas2") for c in finalists)
    phase2=select_b(prepared,second_candidates,phase2_seeds,adapter.fit_candidate,
                    selection_rule="nas_parsimony",delta=delta)
    stages.update(nas_phase1=phase1,nas_phase2=phase2)
    return phase2,stages


def selection_record(selection):
    return {"winner_id":selection.winner.candidate_id,"winner":selection.winner.payload,
            "fixed_epochs":selection.fixed_epochs,"candidate_ledger":selection.candidate_ledger,
            "attempts":selection.attempts,"selection_rule":selection.selection_rule,
            "tie_candidates":selection.tie_candidates,"signature":selection.signature}


def run_selected_refits(prepared,selection,adapter,seeds,*,scope="registered_domains"):
    seeds=tuple(seeds)
    plan=registry.protocol()
    if scope not in ("registered_domains","synthetic_fixture"):
        raise ValueError("Declare registered domains or bounded synthetic fixture")
    all_selection_seeds=set(plan["common_hpo"]["optimization_seeds"])
    all_selection_seeds.add(plan["nas"]["phase1_seed"])
    all_selection_seeds.update(plan["nas"]["phase2_seeds"])
    if set(seeds)&all_selection_seeds or any(s not in plan["budget"]["final_seeds"] for s in seeds):
        raise ValueError("Refit seeds must belong to the separate final seed pool")
    if scope=="registered_domains":
        chosen_R=plan["budget"]["final_seed_count"]
        if chosen_R is None or seeds!=tuple(plan["budget"]["final_seeds"][:chosen_R]):
            raise ValueError("Registered final repetition count must be fixed by the power protocol first")
    return refit_b(prepared,selection,seeds,adapter.refit_candidate)

