"""Provisional eleven-model D7 catalogue, separate from frozen E2 experiments.

The four compact prototypes are our own adaptations, not paper replicas. The
six literature adapters retain their historical structure and explicit scalar
bridge (including ten same-window moments). No author code is downloaded here.
External backbones require explicit source roots and the pinned completion-file
manifest. Synthetic recipes are NOT author presets or training registrations.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:  # Support both scripts-on-sys.path and package-style imports.
    from . import e2_proposed_models as proposed
    from . import e2_final_models_mlp as mlp
    from . import e2_final_models_attention as attention
except ImportError:
    import e2_proposed_models as proposed
    import e2_final_models_mlp as mlp
    import e2_final_models_attention as attention


HYBRID_NAME = "CNN-1D + LSTM + atención temporal"
AE_NAME = "LSTM-autoencoder con predictor"
MODEL_NAMES = ("FCN-1D", "ResNet-1D", "Inception-1D", AE_NAME, "TimesNet",
               "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer", HYBRID_NAME)
MODELS = MODEL_NAMES
PROTOTYPE_NAMES = {"FCN-1D": "FCN1D", "ResNet-1D": "ResNet1D",
                   "Inception-1D": "Inception1D", AE_NAME: "LSTM-AE"}
EXTERNAL_NAMES = ("TimesNet", "TimeMixer", "PatchTST", "iTransformer")
PRESETS = ("provisional_default", "compact_synthetic")
ABLATION_NAMES = ("CNN-only", "LSTM-only", "CNN-LSTM-no-attention")
_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_MANIFEST = _ROOT / "experiments/e2_completion_20261106/design.json"

CATALOGUE = {
    name: {"model": name, "status": "provisional_implementation_not_preregistered",
           "input_shape": [7, 5], "output_shape": ["batch"],
           "output_link": "Softplus", "full_benchmark_executed": False,
           "paper_replication_claimed": False,
           "implementation_kind": ("own_compact_adaptation" if name in PROTOTYPE_NAMES
                                   else "own_causal_hybrid" if name == HYBRID_NAME
                                   else "existing_defined_literature_adapter"),
           "same_window_moment_features": 0 if name in PROTOTYPE_NAMES or name == HYBRID_NAME else 10,
           "requires_explicit_source_root": name in EXTERNAL_NAMES,
           "presets": list(PRESETS)}
    for name in MODEL_NAMES
}


def _check_input(x, *, prefix=False):
    length_ok = (1 <= x.shape[1] <= 7) if isinstance(x, torch.Tensor) and x.ndim == 3 else False
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or len(x) < 1 or x.shape[-1] != 5
            or (not length_ok if prefix else x.shape[1] != 7)):
        raise ValueError("Expected nonempty [batch,7,5] tensor" if not prefix else
                         "Expected nonempty transformed prefix [batch,1..7,5]")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("D7 inputs must be finite real floating-point values")


def _check_output(y, x):
    if not isinstance(y, torch.Tensor) or y.shape != (len(x),):
        raise RuntimeError("Model must return exactly one scalar per window: [batch]")
    if not torch.isfinite(y).all() or (y < 0).any():
        raise RuntimeError("Model returned a nonfinite or negative scale prediction")
    return y


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def recipe_for(name, preset="provisional_default", source_roots=None):
    """Return a fresh, explicit recipe; no source discovery or network access."""
    if name not in MODEL_NAMES and name not in ABLATION_NAMES:
        raise ValueError("Unknown current model or isolated ablation: " + str(name))
    if preset not in PRESETS:
        raise ValueError("Unknown provisional preset")
    compact = preset == "compact_synthetic"
    recipe = {"preset": preset, "dropout": 0.0 if compact else 0.1}
    if name in PROTOTYPE_NAMES:
        recipe["width"] = 8 if compact else 16
        recipe["dropout"] = 0.0  # Existing prototypes do not use dropout.
    elif name == HYBRID_NAME or name in ABLATION_NAMES:
        recipe.update(conv_width=8 if compact else 32, lstm_width=8 if compact else 32,
                      attention_width=8 if compact else 16, kernel_size=3, lstm_layers=1)
    if source_roots is not None:
        if not isinstance(source_roots, dict):
            raise TypeError("source_roots must be a mapping")
        recipe["source_roots"] = {key: str(value) for key, value in source_roots.items()}
    return recipe


def _resolve_recipe(name, supplied):
    if supplied is not None and not isinstance(supplied, dict):
        raise TypeError("recipe must be a dictionary or None")
    supplied = deepcopy(supplied or {})
    result = recipe_for(name, supplied.get("preset", "provisional_default"))
    allowed = set(result) | {"source_roots"}
    if set(supplied) - allowed:
        raise ValueError("Unsupported recipe keys for " + name + ": " + ", ".join(sorted(set(supplied)-allowed)))
    result.update(supplied)
    for key in ("width", "conv_width", "lstm_width", "attention_width", "kernel_size", "lstm_layers"):
        if key in result:
            _positive_int(result[key], key)
    p = result["dropout"]
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= .3:
        raise ValueError("dropout must be finite in [0,.3]")
    if name in PROTOTYPE_NAMES and p != 0:
        raise ValueError("Reused compact prototypes do not implement dropout")
    if "source_roots" in result:
        roots = result["source_roots"]
        if not isinstance(roots, dict) or any(not isinstance(key, str) or not isinstance(value, (str, Path))
                                               for key, value in roots.items()):
            raise TypeError("source_roots must map model names to paths")
        result["source_roots"] = {key: str(value) for key, value in roots.items()}
    return result


def _verify_sources(name, roots):
    """Verify the full declared file subset, not only the primary model file.

    This is not a whole-repository integrity claim. No files are copied,
    downloaded, installed or modified. The historical manifest stays unchanged.
    """
    if not isinstance(roots, dict) or name not in roots:
        raise ValueError("Explicit hash-pinned source_roots['" + name + "'] is required")
    given = Path(roots[name])
    if given.is_symlink():
        raise ValueError("Author source root must not be a symlink")
    root = given.resolve()
    if name == "PatchTST" and (root / "PatchTST_supervised").is_dir():
        root = root / "PatchTST_supervised"
    design = json.loads(_SOURCE_MANIFEST.read_text())
    rows = [row for row in design["author_sources"] if row["model"] == name]
    verified = {}
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source manifest path")
        path = root / relative
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
            raise ValueError("Symlinks are not permitted within pinned author sources")
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Pinned author source missing or changed: " + name + "/" + row["path"])
        verified[row["path"]] = row["sha256"]
    if not verified:
        raise ValueError("No pinned source manifest for " + name)
    return {"repository": design["source_locations"][name]["repository"],
            "commit": design["source_locations"][name]["commit"],
            "verified_files": verified, "verification_scope": "declared file subset, not whole repository"}


class CausalConv1D(nn.Module):
    """Left-only zero padding; seven transformed days stay seven tokens."""
    def __init__(self, incoming, width, kernel_size=3):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(incoming, width, kernel_size, padding=0)

    def forward(self, x):
        return F.relu(self.conv(F.pad(x.transpose(1, 2), (self.left_padding, 0)))).transpose(1, 2)


class CausalCNNLSTMAttention(nn.Module):
    """CNN -> unidirectional LSTM -> additive temporal attention -> scalar.

    Hidden/cell states are omitted on every LSTM call, hence reset to zero.
    Prefix causality concerns CNN/LSTM states of an already transformed tensor.
    The final softmax attends to all seven *observed* steps and is not a
    streaming prefix forecast. No claim is made about raw-price transforms.
    """
    def __init__(self, recipe, ablation=None):
        super().__init__()
        self.ablation = ablation
        use_cnn, use_lstm = ablation != "LSTM-only", ablation != "CNN-only"
        if use_cnn:
            self.cnn = CausalConv1D(5, recipe["conv_width"], recipe["kernel_size"])
        if use_lstm:
            self.lstm = nn.LSTM(recipe["conv_width"] if use_cnn else 5, recipe["lstm_width"],
                                num_layers=recipe["lstm_layers"], batch_first=True,
                                bidirectional=False, dropout=0.0)
        width = recipe["lstm_width"] if use_lstm else recipe["conv_width"]
        self.dropout = nn.Dropout(recipe["dropout"])
        if ablation is None:
            self.attention_projection = nn.Linear(width, recipe["attention_width"])
            self.attention_score = nn.Linear(recipe["attention_width"], 1, bias=False)
        self.head = nn.Linear(width, 1)

    def encode_steps(self, x):
        _check_input(x, prefix=True)
        z = self.cnn(x) if hasattr(self, "cnn") else x
        if hasattr(self, "lstm"):
            z, _ = self.lstm(z)  # Zero initial state on every window batch.
        return z

    def forward_with_attention(self, x):
        _check_input(x)
        if self.ablation is not None:
            raise ValueError("Isolated ablations do not have temporal attention")
        steps = self.encode_steps(x)
        scores = self.attention_score(torch.tanh(self.attention_projection(steps))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(steps * weights.unsqueeze(-1), dim=1)
        prediction = F.softplus(self.head(self.dropout(context))).squeeze(-1)
        return _check_output(prediction, x), weights

    def forward(self, x):
        _check_input(x)
        if self.ablation is None:
            return self.forward_with_attention(x)[0]
        # Explicit last-observed-state readout for all three isolated ablations.
        context = self.encode_steps(x)[:, -1]
        return _check_output(F.softplus(self.head(self.dropout(context))).squeeze(-1), x)


class CurrentModel(nn.Module):
    def __init__(self, core, recipe, metadata):
        super().__init__()
        self.core = core
        self.recipe = deepcopy(recipe)
        self.architecture_metadata = deepcopy(metadata)

    def forward(self, x):
        _check_input(x)
        return _check_output(self.core(x), x)


class CurrentAutoencoder(CurrentModel):
    def reconstruct(self, x):
        _check_input(x)
        output = self.core.reconstruct(x)
        if output.shape != x.shape or not torch.isfinite(output).all():
            raise RuntimeError("AE reconstruction must be finite with the input shape")
        return output


def _metadata(name, recipe, core, source_verification=None):
    metadata = deepcopy(CATALOGUE.get(name, {"model": name, "status": "isolated_ablation_not_in_catalogue"}))
    metadata.update(recipe=deepcopy(recipe), pretrained_weights="none", market_training_executed=False,
                    target="sigma7 / training-only y_scale", hidden_state_carried_between_windows=False,
                    target_lags_added=False, calendar_covariates_added=False,
                    future_values_consumed=False, original_OHLCV_denormalization_used=False,
                    registered_parameters=sum(p.numel() for p in core.parameters()),
                    trainable_parameters=sum(p.numel() for p in core.parameters() if p.requires_grad))
    # Source locations are execution inputs, not portable public recipe facts.
    metadata["recipe"].pop("source_roots", None)
    if name in PROTOTYPE_NAMES:
        metadata.update(adapter_module="scripts/e2_proposed_models.py",
                        prototype_identifier=PROTOTYPE_NAMES[name],
                        width=recipe["width"], source_preset="own compact implementation; no author preset")
        if name == AE_NAME:
            metadata["training_stages"] = ["fold-training-only reconstruction pretraining",
                                           "supervised encoder/head finetuning; decoder unused"]
    elif name == HYBRID_NAME or name in ABLATION_NAMES:
        metadata.update(adapter_module="scripts/e2_current_models.py",
                        sequence_states=7, convolution_padding="left-only zeros",
                        recurrent_direction="unidirectional", normalization="none across time or batch",
                        attention="additive tanh scores; softmax over seven observed steps" if name == HYBRID_NAME else None,
                        readout="attention-weighted context" if name == HYBRID_NAME else "last observed state",
                        source_preset="own provisional architecture; no author preset")
    else:
        metadata["backbone_metadata"] = deepcopy(core.architecture_metadata)
        metadata["structural_resize_for_smoke"] = False
        metadata["source_preset"] = "existing defined literature adapter; task changes retained"
    if source_verification is not None:
        metadata["source_verification"] = source_verification
    metadata["compact_synthetic_is_author_preset"] = False
    return metadata


def build_model(name, recipe=None):
    """Build one of exactly eleven models. Unknown names never fall back."""
    if name not in MODEL_NAMES:
        raise ValueError("Unknown current model; isolated ablations use build_ablation")
    resolved = _resolve_recipe(name, recipe)
    verification = None
    if name in PROTOTYPE_NAMES:
        core = proposed.build_prototype(PROTOTYPE_NAMES[name], width=resolved["width"])
    elif name == HYBRID_NAME:
        core = CausalCNNLSTMAttention(resolved)
    else:
        roots = resolved.get("source_roots")
        if name in EXTERNAL_NAMES or (roots and name in roots):
            verification = _verify_sources(name, roots)
        try:
            if name == "iTransformer":
                core = attention.build_model(name, candidate={"dropout": resolved["dropout"]}, source_roots=roots)
            else:
                core = mlp.build_model(name, candidate={"dropout": resolved["dropout"]}, source_roots=roots)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Missing dependency for the real " + name + " adapter: " + str(exc) +
                                      "; install the pinned E2 requirements, never a dummy module") from exc
    wrapper = CurrentAutoencoder if name == AE_NAME else CurrentModel
    return wrapper(core, resolved, _metadata(name, resolved, core, verification))


def build_ablation(name, recipe=None):
    """Build an isolated mechanism check, never an additional catalogue entry."""
    if name not in ABLATION_NAMES:
        raise ValueError("Unknown isolated hybrid ablation")
    resolved = _resolve_recipe(name, recipe)
    core = CausalCNNLSTMAttention(resolved, ablation=name)
    return CurrentModel(core, resolved, _metadata(name, resolved, core))
