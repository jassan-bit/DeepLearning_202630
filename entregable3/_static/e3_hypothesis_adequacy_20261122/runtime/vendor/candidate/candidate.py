"""Frozen input-only architecture; the fitted map matches the V5 ablation.

This version names a prospective object selected from exposed development. It
does not turn the failed V5 candidate into a successful preregistered recipe.
No optimizer, downloader, market dataset or experiment runs on import.
"""
from __future__ import annotations
import numpy as np
import torch
from .vendor.seoi_v5 import data
from .vendor.seoi_v5.model import ScaleOverlapModel

IDENTITY = "input-only-overlap-mse-1"
HISTORICAL_RECIPE = "input_only_mse"
SOURCE_COMMIT = "c4ca86f1431afeaa958ff0b544b453ab17a4d7b8"
TRAINING = {"optimizer": "AdamW", "learning_rate": .003, "weight_decay": 0.,
            "batch_size": 32, "max_epochs": 40, "patience": 8,
            "gradient_clip_norm": 1., "selection": "minimum_selection_RMSE_earliest_tie"}


class InputOnlyOverlapModel(ScaleOverlapModel):
    """Same tensor operations and state keys as the saved input_only_mse code."""
    def __init__(self, seed: int, u_scale: float, y_scale: float):
        super().__init__(HISTORICAL_RECIPE, seed, u_scale, y_scale)

    def configuration(self):
        old = super().configuration()
        old.update(version=IDENTITY, candidate_identity=IDENTITY,
                   implementation_recipe_reference=HISTORICAL_RECIPE,
                   source_commit=SOURCE_COMMIT, selected_after_development=True,
                   output_scale_equivariance_claimed=False,
                   true_atomic_law_identified=False)
        return old


def fit_inputs(raw_fit, y_fit):
    """Fit only on explicitly supplied training rows, never selection or test.

    Date/partition admission belongs to the separate protocol admission step.
    Unlike the historical loader, this function has no fixed 2024 calendar.
    """
    raw_fit = data.validate_raw(raw_fit)
    y_fit = np.asarray(y_fit, dtype=np.float64)
    A, _, _, _ = data.known_geometry(raw_fit)
    if y_fit.shape != (len(raw_fit),) or not np.isfinite(y_fit).all():
        raise ValueError("Aligned finite fit labels required")
    excess = y_fit**2 - A
    if (y_fit < 0).any() or (excess < -1e-12).any():
        raise ValueError("Fit labels violate the overlap identity support")
    U = np.sqrt(np.maximum(excess, 0) / data.C)
    u_scale, y_scale = float(U.mean()), float(y_fit.mean())
    if not (u_scale > 0 and y_scale > 0):
        raise ValueError("Positive fit-only scales required")
    return {"identity": IDENTITY, "preprocessor": data.fit_preprocessor(raw_fit),
            "u_scale": u_scale, "y_scale": y_scale,
            "fit_rows": len(raw_fit), "label_source": "explicit_fit_rows_only"}


def transform_known_history(raw_history, fitted):
    """Forward features use only the seven available OHLCV rows and fit state."""
    if fitted.get("identity") != IDENTITY:
        raise ValueError("Candidate preprocessing identity mismatch")
    return data.transform(raw_history, fitted["preprocessor"], HISTORICAL_RECIPE)


def make_model(seed, fitted):
    if fitted.get("identity") != IDENTITY:
        raise ValueError("Candidate preprocessing identity mismatch")
    return InputOnlyOverlapModel(seed, fitted["u_scale"], fitted["y_scale"])


def tensors(raw_history, fitted):
    X, A, scale, binding = transform_known_history(raw_history, fitted)
    return torch.from_numpy(X), torch.from_numpy(A), torch.from_numpy(scale), binding
