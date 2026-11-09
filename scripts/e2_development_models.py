"""Fixed pre-2024 development pilot models, separate from registered HPO.

The source for the literal recipe is section-27's
budget.noise_and_resource_pilot_fixed_nonarchitecture. This module neither
fabricates an HPO row nor changes the registered-HPO model constructor. No
network request, market loader or training is triggered on import.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn

try:
    from . import e2_search_plan as planner
    from . import e2_training_models as registry
    from .e2_nas_common import NASConfig
    from .e2_nas_factory import build_nas_model, MODEL_IDS
    from .e2_staged_training import TrainConfig, fit_supervised, fit_sae, state_sha256
    from .e2_temporal_training import PreparedB
except ImportError:
    import e2_search_plan as planner
    import e2_training_models as registry
    from e2_nas_common import NASConfig
    from e2_nas_factory import build_nas_model, MODEL_IDS
    from e2_staged_training import TrainConfig, fit_supervised, fit_sae, state_sha256
    from e2_temporal_training import PreparedB

VERSION = "e2-development-models-v1"
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT/"book_e2/_static/search_protocol_20261109/protocol.json"
NEURAL_NAMES = registry.NEURAL_NAMES
AE_NAME = "LSTM-autoencoder con predictor"


def _protocol():
    value=json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    planner.validate_protocol(value)
    if value.get("protocol_id") != "E2-NAS-HPO-20261109-v1":
        raise ValueError("The fixed development pilot requires its versioned protocol")
    expected={"learning_rate":.001,"weight_decay":.0001,"batch_size":32,
              "warmup_fraction":0,"TimesNet":{"top_k":4,"num_kernels":6},
              "TimeMixer":{"moving_average":7},"hybrid":{"kernel":3},
              "dropout_non_nas":"existing dropout0.1 for TSMixer,iTransformer and neural references;0 for source CNNs",
              "dropout_nas":"solely declared architecture bits at p0.2"}
    baseline=value["budget"].get("noise_and_resource_pilot_fixed_nonarchitecture")
    if not isinstance(baseline,dict) or any(baseline.get(k)!=v for k,v in expected.items()):
        raise ValueError("The fixed development-pilot recipe changed")
    return value


def fixed_pilot_recipe():
    """Return a copy of the literal pilot recipe; no HPO candidate is selected."""
    return deepcopy(_protocol()["budget"]["noise_and_resource_pilot_fixed_nonarchitecture"])


def reference_pattern():
    """Canonical reference record using the common pilot-pattern identity."""
    return deepcopy(next(p for p in planner.noise_pilot_architectures(_protocol()) if p["is_reference"]))


def _seed(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("An explicit nonnegative initialization seed below2**63 is required")
    return seed


def _pattern(name,pattern,p):
    if name not in NEURAL_NAMES:
        raise ValueError("Unknown neural development-pilot system")
    if name not in MODEL_IDS:
        if pattern is not None:
            raise ValueError("Fixed published bodies/E1 references require pattern=None")
        return None,None
    if not isinstance(pattern,dict):
        raise ValueError("Provide an explicit registered noise-pilot pattern or reference")
    allowed={"pattern_id","widths","dropout_pattern","dropout_rate","is_reference",
             "architecture_id","recipe_id"}
    if set(pattern)-allowed:
        raise ValueError("Unexpected pilot architecture fields")
    config=NASConfig(pattern.get("widths"),pattern.get("dropout_pattern"))
    records=planner.noise_pilot_architectures(p)
    match=next((row for row in records if tuple(row["widths"])==config.widths
                and tuple(row["dropout_pattern"])==config.dropout_pattern),None)
    if match is None:
        raise ValueError("Architecture is not one of the twelve preregistered pilot patterns")
    if "dropout_rate" in pattern and pattern["dropout_rate"]!=.2:
        raise ValueError("Pilot NAS dropout is fixed at0.2")
    if "pattern_id" in pattern and pattern["pattern_id"]!=match["pattern_id"]:
        raise ValueError("Pattern identity does not encode supplied widths/bits")
    if "is_reference" in pattern and pattern["is_reference"] is not match["is_reference"]:
        raise ValueError("Reference flag does not match pilot architecture")
    if "recipe_id" in pattern and pattern["recipe_id"]!=MODEL_IDS[name]:
        raise ValueError("Cannot mix NAS family identities")
    expected_id=MODEL_IDS[name]+":"+match["pattern_id"]
    if "architecture_id" in pattern and pattern["architecture_id"]!=expected_id:
        raise ValueError("Architecture identity does not encode the pilot pattern")
    return config,deepcopy(match)


def _structural_options(name,baseline):
    if name in ("TimesNet","TimeMixer"):
        return deepcopy(baseline[name])
    if name=="CNN-1D + LSTM + atención temporal":
        return deepcopy(baseline["hybrid"])
    return {}


def _metadata(model,name,pattern,seed,baseline):
    count=sum(v.numel() for v in model.parameters())
    if name in registry.FIXED_COUNTS and count!=registry.FIXED_COUNTS[name]:
        raise ValueError("Fixed published/E1 body parameter count changed")
    if name in MODEL_IDS:
        expected=NASConfig(pattern["widths"],pattern["dropout_pattern"])
        if model.config!=expected:
            raise ValueError("Constructed NAS widths/pattern differ from fixed pilot request")
    original=deepcopy(getattr(model,"architecture_metadata",{}))
    original.update({
        "development_factory_version":VERSION,"model":name,
        "pilot_fixed_not_hpo":True,"registered_trial_id":None,"training_hpo_record":None,
        "pilot_recipe":deepcopy(baseline),"pilot_pattern":deepcopy(pattern),
        "initialization_seed":seed,"weights_pretrained":False,"weights_reused":False,
        "role":"reinitialized_E1_reference" if name in registry.E1_WIDTHS else "E2_candidate",
        "input_shape":[7,5],"output":"positive sigma7/a; scale inversion is outside factory",
        "actual_parameter_count":count,"training_axes_applied":False,
        "external_source_verified":name=="iTransformer",
    })
    if name not in MODEL_IDS:
        original["dropout"]=0.0 if name in ("FCN-1D","ResNet-1D","Inception-1D") else .1
    model.architecture_metadata=original
    return model


def build_pilot_model(name,pattern,*,seed,source_roots=None):
    """Construct a fresh literal pilot recipe without calling registered HPO.

    NAS families use the explicit public NAS constructor. Fixed-source and E1
    bodies reuse the same public builders used by the training registry.
    iTransformer still checks all eight pinned source files before import.
    """
    seed=_seed(seed)
    p=_protocol()
    config,canonical=_pattern(name,pattern,p)
    baseline=p["budget"]["noise_and_resource_pilot_fixed_nonarchitecture"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if config is not None:
            model=build_nas_model(name,config.widths,config.dropout_pattern,
                                  hpo=_structural_options(name,baseline))
        elif name in ("FCN-1D","ResNet-1D","Inception-1D"):
            model=registry.recipes.build_model(name)
        elif name=="TSMixer":
            model=registry.TSMixerAdapter(.1)
        elif name=="iTransformer":
            model=registry.AttentionVolatilityAdapter(name,{"parameters":{"dropout":.1}},source_roots)
        else:
            builders={"MLP16":registry.SmallMLP,"LSTM32":registry.SingleLayerLSTM,
                      "GRU32":registry.SingleLayerGRU}
            model=builders[name](7,registry.E1_WIDTHS[name],dropout=.1,input_features=5)
    return _metadata(model,name,canonical,seed,baseline)


def fixed_pilot_config(seed,epoch_cap=200,wallcap=None,memory_limit_bytes=None):
    """Common AdamW pilot controls; epoch overrides are explicit in receipts."""
    seed=_seed(seed)
    baseline=fixed_pilot_recipe()
    return TrainConfig(
        learning_rate=baseline["learning_rate"],weight_decay=baseline["weight_decay"],
        batch_size=baseline["batch_size"],max_epochs=epoch_cap,patience=20,seed=seed,
        warmup_fraction=baseline["warmup_fraction"],gradient_clip=1.,
        wall_limit_seconds=wallcap,memory_limit_bytes=memory_limit_bytes)


@dataclass(frozen=True)
class PilotFit:
    model: nn.Module
    predictions_original: tuple[float,...]
    rmse_original: float
    receipt: dict
    phases: dict

    def to_dict(self):
        return deepcopy(self.receipt)


def fit_pilot(name,pattern,seed,prepared,source_roots=None,epoch_cap=200,wallcap=None,
              memory_limit_bytes=None):
    """Fit only the sealed pre-2024 inner-development train/validation pair.

    There is no outer-label input, refit or HPO selection. wallcap is a
    per-optimization-stage boundary check, not a combined SAE/process sandbox.
    The caller's subprocess supervisor owns hard total-time/RSS enforcement.
    TrainingFailure propagates unchanged with its partial-stage receipt.
    """
    if not isinstance(prepared,PreparedB):
        raise TypeError("A sealed PreparedB is required")
    prepared.validate()
    p=_protocol()
    origin=prepared.origin.strftime("%Y-%m-%d")
    if origin!=p["budget"]["noise_pilot_origin"]:
        raise ValueError("Fixed noise/resource pilot is registered only at origin2023-10-01")
    config,canonical=_pattern(name,pattern,p)
    train=prepared.inner_train.clone()
    valid=prepared.inner_validation.clone()
    controls=fixed_pilot_config(seed,epoch_cap,wallcap,memory_limit_bytes)
    target_scale=train.scaler.target_scale
    if name==AE_NAME:
        result=fit_sae(config,train.x,train.y,valid.x,valid.y,target_scale,controls,
                       fold_id="development-pilot-"+origin)
        model=result.model
        phases=result.phases
        extra={"auxiliary_scale":result.auxiliary_scale,
               "decoder_parameters_in_predictor":any("decoder" in k for k in model.state_dict())}
        _metadata(model,name,canonical,seed,p["budget"]["noise_and_resource_pilot_fixed_nonarchitecture"])
    else:
        model=build_pilot_model(name,canonical,seed=seed,source_roots=source_roots)
        result=fit_supervised(model,train.x,train.y,valid.x,valid.y,target_scale,controls)
        phases={"supervised":result}
        extra={}
    before=state_sha256(model)
    model.eval()
    outputs=[]
    with torch.no_grad():
        for batch in valid.x.split(controls.batch_size):
            predicted=model(batch)
            if predicted.shape!=(len(batch),) or not bool(torch.isfinite(predicted).all()) or bool((predicted<0).any()):
                raise ValueError("Invalid frozen pilot prediction")
            outputs.append(predicted.detach().cpu().to(torch.float64)*target_scale)
    original=torch.cat(outputs)
    truth=torch.tensor(valid.y_original,dtype=torch.float64)
    rmse=float(torch.sqrt(torch.mean((original-truth).square())))
    if state_sha256(model)!=before:
        raise RuntimeError("Validation prediction changed the selected checkpoint")
    model.architecture_metadata.update({"training_stage_executed":True,
        "fixed_pilot_hyperparameters_applied":True})
    receipt={
        "version":VERSION,"status":"completed",
        "scope":"fixed pre-2024 development noise/resource pilot; not HPO",
        "pilot_fixed_not_hpo":True,"registered_trial_id":None,"model":name,
        "pattern":canonical,"seed":seed,"origin":origin,"prepared_signature":prepared.signature,
        "training_split":train.record(),"validation_split":valid.record(),
        "config":asdict(controls),"protocol_full_epoch_cap":epoch_cap==200,
        "protocol_short_epoch_cap":epoch_cap==50,
        "nonprotocol_epoch_override":epoch_cap not in (50,200),
        "rmse_original_units":rmse,"predictions_original":original.tolist(),
        "phases":{key:value.to_dict() for key,value in phases.items()},
        "optimizer_steps":sum(value.optimizer_steps for value in phases.values()),
        "selected_checkpoint_sha256":before,"outer_labels_used":False,
        "market_dataset_files_opened":False,"protocol_config_read":True,
        "development_training_executed":True,"market_campaign_executed":False,
        "wall_limit_scope":"per optimization stage; total process bound belongs to supervisor",
        "protocol_file_sha256":sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        **extra,
    }
    return PilotFit(model,tuple(original.tolist()),rmse,receipt,phases)

