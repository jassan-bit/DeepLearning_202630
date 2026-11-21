"""Development-only sigma-coordinate CRPS revision of the frozen V3 readout.

Prepared with declared AI assistance. This version changes the auxiliary score
coordinate AND its training-mean scale. It does not establish novelty, utility,
calibration, independent confirmation or a replacement for the adverse V3 pilot.
No data, optimizer, checkpoint or network operation is performed by this module.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import torch

from src.e3_overlap_model_v3 import (
    C, OverlapDistributionModel, discrete_crps,
    loss_components as original_v3_loss_components,
)

VERSION = "e3-overlap-sigma-crps-v4-development"
BASE_SOURCE_SHA256 = "30a0e4daa480e803932482c93928e4e250e338eb3daff3fc801e73ed2cec434b"
FIXED_LAMBDA = 0.1
VARIANTS = ("sigma_crps_v4", "u_crps_control", "mse_only_control",
            "sigma_crps_a0matched_control")


def verify_base_source(path=None):
    """Bind the reused architecture; reading bytes is not executing a model."""
    path = Path(path or inspect.getfile(OverlapDistributionModel))
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular frozen V3 source file is required")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != BASE_SOURCE_SHA256:
        raise ValueError("Frozen V3 source differs from the declared dependency")
    return actual


def _scale(value, name, device):
    if isinstance(value, bool) or (isinstance(value, torch.Tensor) and value.dtype == torch.bool):
        raise ValueError(name + " must be a positive fixed scalar")
    result = torch.as_tensor(value, dtype=torch.float64, device=device)
    if (result.numel() != 1 or result.requires_grad or not bool(torch.isfinite(result).all())
            or not bool((result > 0).all())):
        raise ValueError(name + " must be a positive fixed scalar")
    return result


def outcome_loss_components(outputs, Y, U, *, y_scale, u_scale,
                            variant="sigma_crps_v4"):
    """Point MSE with one fixed auxiliary objective, plus both score diagnostics.

    sigma_crps_v4 uses CRPS(T_A#G, Y)/sY. The U comparator is the original
    CRPS(G,U)/sU. The A0-matched comparator uses CRPS(T_A#G,Y)/(sqrt(C)*sU),
    which equals the normalized U score at A=0. It is a normalization diagnostic,
    not a fitted coefficient or another automatically selected candidate.
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown declared development variant")
    prediction = outputs.get("prediction")
    if not isinstance(prediction, torch.Tensor) or prediction.ndim != 1:
        raise ValueError("One prediction tensor per sample required")
    ys = _scale(y_scale, "y_scale", prediction.device)
    us = _scale(u_scale, "u_scale", prediction.device)
    if not bool(torch.isfinite(prediction).all()) or not bool((prediction >= 0).all()):
        raise ValueError("Finite nonnegative prediction required")
    # Retain the exact historical point-MSE and U-score definitions. Computing
    # that diagnostic does not add it to the V4 loss or supply U to forward().
    previous = original_v3_loss_components(outputs, Y, U, y_scale=ys,
                                          u_scale=us, lambda_crps=FIXED_LAMBDA)
    sigma_atoms = outputs.get("sigma_atoms")
    if (not isinstance(sigma_atoms, torch.Tensor)
            or sigma_atoms.shape != outputs["atoms"].shape
            or not bool(torch.isfinite(sigma_atoms).all())
            or not bool((sigma_atoms >= 0).all())):
        raise ValueError("Aligned finite nonnegative sigma atoms required")
    target = torch.as_tensor(Y, dtype=torch.float64, device=prediction.device)
    sigma_crps = discrete_crps(sigma_atoms / ys, outputs["probabilities"], target / ys).mean()
    matched_scale = math.sqrt(C) * us
    matched = discrete_crps(sigma_atoms / matched_scale, outputs["probabilities"],
                            target / matched_scale).mean()
    if not all(bool(torch.isfinite(value)) for value in
               (previous["mse_scaled"], previous["crps_u"], sigma_crps, matched)):
        raise ValueError("Finite scores and point risk required")
    auxiliary = {"sigma_crps_v4": sigma_crps,
                 "u_crps_control": previous["crps_u"],
                 "sigma_crps_a0matched_control": matched,
                 "mse_only_control": previous["mse_scaled"].new_zeros(())}[variant]
    return {"loss": previous["mse_scaled"] + FIXED_LAMBDA * auxiliary,
            "mse_scaled": previous["mse_scaled"], "crps_u": previous["crps_u"],
            "crps_sigma": sigma_crps, "crps_sigma_a0matched": matched}


class OutcomeCRPSDevelopmentModel(OverlapDistributionModel):
    """The identical 1262-parameter V3 architecture with a new loss identity.

    Initial values, exact finite-law mean, forward inputs and precision remain
    bound to V3. No V3 novelty verdict or pilot performance is inherited.
    """
    def __init__(self, variant="sigma_crps_v4", *, u_scale, y_scale, seed=20261121):
        if variant not in VARIANTS:
            raise ValueError("Unknown declared development variant")
        verify_base_source()
        old_variant = "atom_mse_only" if variant == "mse_only_control" else "atom_crps"
        super().__init__(old_variant, u_scale=u_scale, y_scale=y_scale,
                         seed=seed, lambda_crps=FIXED_LAMBDA)
        self.development_variant = variant
        self.name = self.model_name = variant

    def loss_components(self, outputs, Y, U):
        return outcome_loss_components(outputs, Y, U, y_scale=self.y_scale,
                                       u_scale=self.u_scale, variant=self.development_variant)

    def configuration(self):
        result = super().configuration()
        result.update(version=VERSION, variant=self.development_variant,
                      frozen_base_variant=self.variant, frozen_base_source_sha256=BASE_SOURCE_SHA256,
                      auxiliary_coefficient=0.0 if self.development_variant == "mse_only_control" else FIXED_LAMBDA,
                      scoring_coordinate_and_normalization_both_changed=self.development_variant == "sigma_crps_v4",
                      R1_inherited=False, R3_accredited=False, market_utility_verified=False,
                      original_V3_NO_GO_preserved=True,
                      scope="new development revision; not confirmation or automatic model promotion")
        return result
