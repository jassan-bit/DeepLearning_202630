"""Fourteen versioned neural constructors for E2 training integration.

Only architecture construction is performed. Market observations, optimizers,
selection, target scaling and fitting remain outside this registry. Every HPO
row must belong to the immutable section-27 generated candidate list. Source
trees for iTransformer are supplied explicitly and verified before import.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn

try:
    from . import e2_search_plan as planner
    from .e2_nas_factory import MODEL_IDS, build_from_plan
    from . import e2_recipe_models as recipes
    from .e2_final_models_mlp import TSMixerAdapter
    from .e2_final_models_attention import AttentionVolatilityAdapter, SOURCE_SPECS
    from .dl_models import SmallMLP, SingleLayerLSTM
    from .dl_models_gru_lstm_tcn import SingleLayerGRU
except ImportError:
    import e2_search_plan as planner
    from e2_nas_factory import MODEL_IDS, build_from_plan
    import e2_recipe_models as recipes
    from e2_final_models_mlp import TSMixerAdapter
    from e2_final_models_attention import AttentionVolatilityAdapter, SOURCE_SPECS
    from dl_models import SmallMLP, SingleLayerLSTM
    from dl_models_gru_lstm_tcn import SingleLayerGRU

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT/"book_e2/_static/search_protocol_20261109/protocol.json"
NEURAL_NAMES = (
    "FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
    "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
    "CNN-1D + LSTM + atención temporal", "MLP16", "LSTM32", "GRU32",
)
FIXED_COUNTS = {"FCN-1D":268929,"ResNet-1D":522955,"Inception-1D":490561,
                "TSMixer":35044,"iTransformer":795403,
                "MLP16":593,"LSTM32":5025,"GRU32":3777}
E1_WIDTHS = {"MLP16":16, "LSTM32":32, "GRU32":32}
VERSION = "e2-training-registry-v1"


def protocol():
    result = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    planner.validate_protocol(result)
    expected = tuple(p["model"] for p in result["models"]+result["neural_reference_profiles"])
    if expected != NEURAL_NAMES:
        raise ValueError("The training registry differs from the versioned fourteen-model catalogue")
    return result


def registered_trials(name):
    if name not in NEURAL_NAMES:
        raise ValueError("Unknown E2 neural system")
    return planner.generate_hpo(protocol(), name)


def reference_architecture(name):
    if name not in MODEL_IDS:
        raise ValueError("Only new NAS families have a searched architecture reference")
    return next(row for row in planner.enumerate_architectures(protocol(), name) if row["is_reference"])


def _validate_trial(name, trial):
    if not isinstance(trial, dict) or trial not in registered_trials(name):
        raise ValueError("Use one exact registered HPO configuration for the named model")
    return deepcopy(trial)


def build_train_model(name, trial, *, architecture=None, source_roots=None, seed=None):
    """Construct a fresh CPU predictor, including fresh E1-reference weights.

    An explicit initialization seed is mandatory; the caller's RNG stream is
    preserved. Training axes are recorded here but applied by the trainer.
    For iTransformer, dropout0.5 is enabled by changing only existing dropout
    rates after historical wrapper construction (which accepts at most0.3).
    No layer dimensions or source files are changed.
    """
    if name not in NEURAL_NAMES:
        raise ValueError("Unknown E2 neural system")
    if type(seed) is not int or seed < 0:
        raise ValueError("Provide an explicit nonnegative initialization seed")
    trial = _validate_trial(name, trial)
    if name in MODEL_IDS:
        if architecture is None:
            raise ValueError("NAS construction requires an explicit versioned architecture row")
    elif architecture is not None:
        raise ValueError("Published bodies and E1 references cannot receive NAS architecture overrides")
    p = float(trial.get("dropout", 0.0))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name in MODEL_IDS:
            model = build_from_plan(protocol(), name, deepcopy(architecture), trial)
        elif name in ("FCN-1D", "ResNet-1D", "Inception-1D"):
            if p != 0:
                raise ValueError("The published CNN bodies contain no adjustable dropout module")
            model = recipes.build_model(name)
        elif name == "TSMixer":
            model = TSMixerAdapter(p)
        elif name == "iTransformer":
            # The adapter verifies all eight immutable files (including license)
            # before import. Missing sources are an error, never a substitute.
            model = AttentionVolatilityAdapter(name, {"parameters":{"dropout":min(p, .3)}},
                                               source_roots)
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = p
        else:
            builders = {"MLP16":SmallMLP, "LSTM32":SingleLayerLSTM, "GRU32":SingleLayerGRU}
            model = builders[name](7, E1_WIDTHS[name], dropout=p, input_features=5)
    count = sum(value.numel() for value in model.parameters())
    if name in FIXED_COUNTS and count != FIXED_COUNTS[name]:
        raise ValueError("Frozen body parameter count changed for "+name)
    inherited = deepcopy(getattr(model, "architecture_metadata", {}))
    if name not in MODEL_IDS:
        inherited["dropout"] = p
    model.architecture_metadata = {
        **inherited,
        "training_registry_version":VERSION, "model":name,
        "training_recipe_id":trial["recipe_id"], "registered_trial_id":trial["trial_id"],
        "initialization_seed":seed, "weights_pretrained":False,
        "role":"reinitialized_E1_reference" if name in E1_WIDTHS else "E2_candidate",
        "input_shape":[7,5], "output":"positive sigma7/a; inversion belongs to caller",
        "nas":name in MODEL_IDS, "frozen_body_parameter_count":FIXED_COUNTS.get(name),
        "actual_parameter_count":count, "training_hpo_record":deepcopy(trial),
        "training_axes_applied":False, "weights_reused":False,
        "requested_existing_dropout":p if name not in MODEL_IDS else None,
        "iTransformer_existing_dropout_rate_bridge":name=="iTransformer" and p>.3,
        "source_roots_verified":bool(name=="iTransformer"),
        "market_data_read":False,
    }
    return model


def itransformer_source_manifest():
    """Public immutable-file manifest; no network action is performed."""
    return deepcopy(SOURCE_SPECS["iTransformer"])

