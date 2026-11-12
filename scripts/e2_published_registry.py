"""Prospective D7 architecture adjudication, separate from sealed NAS variants.

The four restored recipes have new identities. Historical HPO observations must
not be relabelled as these bodies. No constructor reads market data or downloads
source code. TiDE retains its registered NAS domain until compatibility is closed.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

try:
    from . import e2_training_models as historical
    from .e2_published_periodic import build_periodic
    from .e2_published_patchtst import build_patchtst
    from .e2_published_sae import build_sae
except ImportError:
    import e2_training_models as historical
    from e2_published_periodic import build_periodic
    from e2_published_patchtst import build_patchtst
    from e2_published_sae import build_sae

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "experiments/e2_published_architectures_20261112/architecture_policy.json"
POLICY_ID = "e2-published-bodies-d7-v1"
NEURAL_NAMES = tuple(n for n in historical.NEURAL_NAMES
                     if n != "CNN-1D + LSTM + atención temporal")
RESTORED = {
    "LSTM-autoencoder con predictor": ("SAE-D7-PUBLISHED-v1", .3),
    "TimesNet": ("TN-D7-PUBLISHED-v1", .1),
    "TimeMixer": ("TM-D7-PUBLISHED-v1", .1),
    "PatchTST": ("PT-D7-PUBLISHED-v1", .3),
}
NAS_REQUIRED = ("TiDE",)


def policy():
    value = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    rows = value.get("models", [])
    historical_profiles = historical.protocol()
    expected = {row["model"]: row["search_recipe_id"] for row in
                historical_profiles["models"] + historical_profiles["neural_reference_profiles"]}
    expected.update({name: recipe_id for name, (recipe_id, _) in RESTORED.items()})
    if (value.get("policy_id") != POLICY_ID
            or tuple(row.get("model_id") for row in rows) != NEURAL_NAMES
            or tuple(row["model_id"] for row in rows if row.get("nas_required")) != NAS_REQUIRED
            or any(row.get("recipe_id") != expected.get(row.get("model_id")) for row in rows)
            or value.get("counts", {}).get("defined_top10") != 9
            or value.get("counts", {}).get("nas_top10") != 1):
        raise ValueError("The adjudicated architecture policy differs from its implementation")
    return deepcopy(value)


def model_policy(name):
    if name not in NEURAL_NAMES:
        raise ValueError("Unknown mandatory neural system")
    return next(row for row in policy()["models"] if row["model_id"] == name)


def registered_trials(name):
    """Keep eighteen LR/WD/batch points; fix restored structural source settings."""
    model_policy(name)
    rows = historical.registered_trials(name)
    if name not in RESTORED:
        return rows
    recipe_id, dropout = RESTORED[name]
    result = []
    keys = ("learning_rate", "weight_decay", "batch_size", "optimization_seeds", "warmup_fraction")
    for row in rows:
        copied = {k: deepcopy(row[k]) for k in keys if k in row}
        copied.update(trial_id=recipe_id + ":" + row["trial_id"].rsplit(":", 1)[-1],
                      recipe_id=recipe_id, dropout=dropout)
        result.append(copied)
    return result


def build_train_model(name, trial, *, architecture=None, source_roots=None, seed=None):
    adjudication = model_policy(name)
    if type(seed) is not int or seed < 0:
        raise ValueError("An explicit nonnegative initialization seed is required")
    if not isinstance(trial, dict) or trial not in registered_trials(name):
        raise ValueError("Use an exact HPO row of the current named recipe; historical identities cannot be reused")
    if name not in RESTORED:
        return historical.build_train_model(name, trial, architecture=architecture,
                                           source_roots=source_roots, seed=seed)
    if architecture is not None:
        raise ValueError("A fixed published body cannot receive a NAS architecture override")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name == "LSTM-autoencoder con predictor":
            model = build_sae(dropout=trial["dropout"], seed=seed)
        elif name == "PatchTST":
            model = build_patchtst(dropout=trial["dropout"], source_roots=source_roots)
        else:
            model = build_periodic(name, trial["dropout"], source_roots)
    model.architecture_metadata.update(
        model=name, architecture_policy_id=POLICY_ID,
        training_recipe_id=adjudication["recipe_id"], registered_trial_id=trial["trial_id"],
        training_hpo_record=deepcopy(trial), initialization_seed=seed,
        actual_parameter_count=sum(p.numel() for p in model.parameters()),
        nas=False, weights_reused=False, weights_pretrained=False,
        training_axes_applied=False, restored_published_body=True,
    )
    return model
