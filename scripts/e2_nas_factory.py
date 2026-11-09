"""Versioned factory for six section-27 NAS families; no fitting or data access.

Training axes are recorded but deliberately not applied here. Optimizers, early
stopping, fold-local preprocessing and the market runner remain separate.
"""
from __future__ import annotations
from copy import deepcopy
import math

try:
    from .e2_nas_common import NASConfig
except ImportError:
    from e2_nas_common import NASConfig

VERSION = "e2-nas-factory-v1"
MODEL_IDS = {
    "LSTM-autoencoder con predictor": "SAE-D7-NAS-v1",
    "TimesNet": "TN-D7-NAS-v1", "TiDE": "TIDE-D7-NAS-v1",
    "TimeMixer": "TM-D7-NAS-v1", "PatchTST": "PT-D7-NAS-v1",
    "CNN-1D + LSTM + atención temporal": "HIB-D7-NAS-v1",
}
MOMENTS = {name: 0 if name in ("LSTM-autoencoder con predictor",
           "CNN-1D + LSTM + atención temporal") else 10 for name in MODEL_IDS}
EXTRA_DOMAINS = {
    "TimesNet": {"top_k": (2, 3, 4), "num_kernels": (2, 4, 6)},
    "TimeMixer": {"moving_average": (3, 7, 25)},
    "CNN-1D + LSTM + atención temporal": {"kernel": (2, 3, 5)},
    "PatchTST": {}, "TiDE": {}, "LSTM-autoencoder con predictor": {},
}
DEFAULT_EXTRAS = {"TimesNet": {"top_k": 4, "num_kernels": 6},
                  "TimeMixer": {"moving_average": 7},
                  "CNN-1D + LSTM + atención temporal": {"kernel": 3}}
TRAINING_KEYS = {"trial_id", "recipe_id", "learning_rate", "weight_decay",
                 "batch_size", "optimization_seeds", "architecture_reference"}
WARMUP_MODELS = {"PatchTST"}


def _options(name, hpo):
    if name not in MODEL_IDS:
        raise ValueError("Only six explicitly versioned NAS families are accepted")
    if hpo is not None and not isinstance(hpo, dict):
        raise TypeError("HPO options must be a dictionary")
    given = deepcopy(hpo or {})
    allowed = TRAINING_KEYS | set(EXTRA_DOMAINS[name])
    if name in WARMUP_MODELS:
        allowed |= {"warmup_fraction"}
    unknown = set(given) - allowed
    if unknown:
        raise ValueError("Unsupported HPO/model options: " + str(sorted(unknown)))
    if "recipe_id" in given and given["recipe_id"] != MODEL_IDS[name]:
        raise ValueError("HPO recipe identity does not match NAS family")
    result = dict(DEFAULT_EXTRAS.get(name, {}))
    for key, domain in EXTRA_DOMAINS[name].items():
        if key in given:
            if type(given[key]) is not int or given[key] not in domain:
                raise ValueError("Value outside frozen HPO domain: " + key)
            result[key] = given[key]
    if "warmup_fraction" in given and (type(given["warmup_fraction"]) not in (int, float)
                                     or given["warmup_fraction"] not in (0, .05, .1)):
        raise ValueError("Warmup outside frozen HPO domain")
    for key, low, high in (("learning_rate", 1e-5, .003), ("weight_decay", 0, .01)):
        if key in given and (type(given[key]) not in (int, float)
                             or not math.isfinite(given[key])
                             or not low <= given[key] <= high):
            raise ValueError("Value outside numerical HPO range: " + key)
    if "weight_decay" in given and 0 < given["weight_decay"] < 1e-6:
        raise ValueError("Positive weight decay below frozen domain")
    if "batch_size" in given and (type(given["batch_size"]) is not int
                                  or given["batch_size"] not in (16, 32, 64)):
        raise ValueError("Batch size outside frozen HPO domain")
    if "optimization_seeds" in given and given["optimization_seeds"] != [271201, 271202]:
        raise ValueError("HPO seed pool differs from versioned protocol")
    if "architecture_reference" in given:
        expected = {"widths": [64, 64, 64], "dropout_pattern": [0, 1, 0], "dropout_rate": .2}
        if given["architecture_reference"] != expected:
            raise ValueError("HPO reference differs from versioned protocol")
    return result, given


def build_nas_model(name, widths, dropout_pattern, *, hpo=None):
    config = NASConfig(widths, dropout_pattern)
    options, given = _options(name, hpo)
    try:
        from .e2_nas_attention import PatchTSTNAS, HybridNAS
        from .e2_nas_autoencoder import SAENAS, TiDENAS
        from .e2_nas_spectral import TimesNetNAS, TimeMixerNAS
    except ImportError:
        from e2_nas_attention import PatchTSTNAS, HybridNAS
        from e2_nas_autoencoder import SAENAS, TiDENAS
        from e2_nas_spectral import TimesNetNAS, TimeMixerNAS
    builders = {"LSTM-autoencoder con predictor": SAENAS, "TimesNet": TimesNetNAS,
                "TiDE": TiDENAS, "TimeMixer": TimeMixerNAS, "PatchTST": PatchTSTNAS,
                "CNN-1D + LSTM + atención temporal": HybridNAS}
    model = builders[name](config, **options)
    model.architecture_metadata = {
        "factory_version": VERSION, "model": name, "recipe_id": MODEL_IDS[name],
        "architecture_id": MODEL_IDS[name] + ":" + config.pattern_id,
        "widths": list(config.widths), "dropout_pattern": list(config.dropout_pattern),
        "dropout_rate": .2, "input_shape": [7, 5], "output": "sigma7/a positive scalar",
        "same_window_moments": MOMENTS[name], "structure_hpo_options": options,
        "training_hpo_record": given, "training_axes_applied": False,
        "historical_recipe_replaced": False, "paper_replication_claimed": False,
    }
    return model


def build_from_plan(protocol, name, architecture, trial):
    """Bind model/architecture/trial IDs before construction; no training begins."""
    try:
        from . import e2_search_plan as planner
    except ImportError:
        import e2_search_plan as planner
    planner.validate_protocol(protocol)
    rows = [m for m in protocol["models"] if m["model"] == name]
    if len(rows) != 1 or not rows[0]["nas"] or rows[0]["search_recipe_id"] != MODEL_IDS.get(name):
        raise ValueError("Family must have the matching affirmative NAS decision")
    if not isinstance(architecture, dict) or not isinstance(trial, dict):
        raise TypeError("Explicit architecture and trial records required")
    if architecture.get("recipe_id") != MODEL_IDS[name] or trial.get("recipe_id") != MODEL_IDS[name]:
        raise ValueError("Cannot mix family/trial/architecture identities")
    if trial not in planner.generate_hpo(protocol, name):
        raise ValueError("Trial is not one of the eighteen registered configurations")
    config = NASConfig(architecture.get("widths"), architecture.get("dropout_pattern"))
    if architecture.get("architecture_id") != MODEL_IDS[name] + ":" + config.pattern_id:
        raise ValueError("Architecture ID does not encode supplied widths and pattern")
    if architecture.get("dropout_rate") != .2:
        raise ValueError("Architecture dropout rate must be0.2")
    return build_nas_model(name, config.widths, config.dropout_pattern, hpo=trial)

