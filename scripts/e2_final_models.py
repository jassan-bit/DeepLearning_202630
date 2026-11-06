"""Final E2 neural catalogue, separate from the immutable development pilot.

The classical estimators are handled by the experiment runner. This factory
preserves the three inherited E1 recipes and exposes their retuned E2 variants
under distinct identities; it never changes datasets or selects a candidate.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn

MLP_CONV = ("TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST")
ATTENTION_SSM = ("Informer", "Autoformer", "iTransformer", "Crossformer", "S-Mamba")
RETUNED = ("GRU-E2", "LSTM-E2", "TCN-E2")
ANCHORS = {
    "GRU/D7-c03": {"family": "GRU", "width": 16, "dropout": .1, "parameters": 1121},
    "LSTM/D7-c02": {"family": "LSTM", "width": 16, "dropout": 0., "parameters": 1489},
    "TCN/D7-c02": {"family": "TCN", "width": 8, "dropout": 0., "parameters": 1185},
}
NEURAL_MODELS = (*MLP_CONV, *ATTENTION_SSM, *RETUNED, *ANCHORS)
DETERMINISTIC_MODELS = ("Ridge", "ARIMA", "Persistence", "SeasonalNaive")


def parameters(candidate=None):
    if candidate is None:
        return {}
    if not isinstance(candidate, Mapping):
        raise TypeError("A candidate must be a parameter mapping or registry row")
    values = dict(candidate.get("parameters", candidate))
    if "dropout" in values and not 0 <= float(values["dropout"]) < 1:
        raise ValueError("Dropout must be in [0,1)")
    return values


def build_model(name, *, candidate=None, source_roots=None):
    if isinstance(candidate, Mapping) and candidate.get("model", name) != name:
        raise ValueError("Candidate belongs to another model")
    values = parameters(candidate)
    outer_ablation = candidate.get("ablation_id") if isinstance(candidate, Mapping) else None
    inner_ablation = values.pop("ablation_id", None)
    if outer_ablation and inner_ablation and outer_ablation != inner_ablation:
        raise ValueError("Conflicting ablation identities")
    ablation_id = outer_ablation or inner_ablation
    if name in MLP_CONV:
        from e2_final_models_mlp import build_model as construct
        model = construct(name, candidate=values, source_roots=source_roots)
        return _apply_registered_ablation(model, name, ablation_id)
    if name in ATTENTION_SSM:
        from e2_final_models_attention import build_model as construct
        model = construct(name, candidate=values, source_roots=source_roots)
        return _apply_registered_ablation(model, name, ablation_id)
    if name not in (*RETUNED, *ANCHORS):
        raise ValueError("Unknown neural recipe; classical references use the runner")
    if ablation_id:
        raise ValueError("The registered three-component ablation cohort is the Top10 only")
    from dl_models_gru_lstm_tcn import build_model as construct, architecture_spec
    if name in ANCHORS:
        recipe = ANCHORS[name]
        fixed = {"width": recipe["width"], "dropout": recipe["dropout"],
                 "learning_rate": .003, "weight_decay": .0001, "batch_size": 32}
        for key, value in fixed.items():
            if key in values and values[key] != value:
                raise ValueError("Cannot retune the inherited anchor " + name + ": " + key)
        family, width, dropout = recipe["family"], recipe["width"], recipe["dropout"]
    else:
        family = name.split("-")[0]
        width = 8 if family == "TCN" else 16
        if "width" in values and values["width"] != width:
            raise ValueError("Defined E2 base width is fixed; structural changes require an amendment")
        dropout = float(values.get("dropout", .1))
    model = construct(family, history_days=7, width=width, dropout=dropout)
    count = sum(p.numel() for p in model.parameters())
    if name in ANCHORS and count != ANCHORS[name]["parameters"]:
        raise ValueError("Inherited E1 architecture/parameter identity changed")
    model.architecture_metadata = {
        **architecture_spec(model), "model_id": name,
        "role": "inherited_fixed_anchor" if name in ANCHORS else "retuned_E2_base",
        "weights_pretrained": False, "target_lags": False, "calendar_channels": False,
        "defined_architecture": True, "NAS_required": False,
        "source": "scripts/dl_models_gru_lstm_tcn.py and pinned E1 recipe",
        "optimizer_parameter_policy": "respect explicit source _no_weight_decay flags; all other parameters use candidate weight decay",
    }
    return model


def _apply_registered_ablation(model, name, ablation_id):
    if ablation_id in (None, "full_control", "control", "full"):
        return model
    from e2_final_diagnostics import apply_ablation, get_ablation_definitions
    definitions = get_ablation_definitions(name)
    identities = set(definitions)
    if ablation_id not in identities:
        raise ValueError("Unknown predeclared component lesion for " + name)
    handle = apply_ablation(model, ablation_id)
    model._e2_ablation_handle = handle
    model.architecture_metadata = {**model.architecture_metadata,
                                   "ablation_id": ablation_id,
                                   "ablation_definition": handle.receipt,
                                   "ablation_requires_retraining": True}
    return model


def optimizer_groups(model: nn.Module, weight_decay: float):
    """Honor explicit source exclusions, retaining their names in provenance.

    Mamba's A_log and D carry the original implementation's no-decay flag.
    No name-based exemptions are invented for biases or normalization layers.
    """
    decay = float(weight_decay)
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("Finite nonnegative weight decay required")
    regular, excluded, excluded_names = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if getattr(parameter, "_no_weight_decay", False):
            excluded.append(parameter); excluded_names.append(name)
        else:
            regular.append(parameter)
    groups = []
    if regular:
        groups.append({"params": regular, "weight_decay": decay})
    if excluded:
        groups.append({"params": excluded, "weight_decay": 0.})
    if not groups:
        raise ValueError("A neural training recipe needs trainable parameters")
    return groups, {"candidate_weight_decay": decay, "source_excluded_names": excluded_names,
                    "exclusion_rule": "explicit _no_weight_decay flag only"}


def architecture_spec(model):
    metadata = getattr(model, "architecture_metadata", None)
    if not isinstance(metadata, dict):
        raise ValueError("Final adapters must expose their audited architecture metadata")
    return {**metadata,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
