"""Development-only atomic overlap head, prepared with declared AI assistance.

It learns a finite conditional law for the future return-deviation magnitude U,
then evaluates E[sqrt(A + (6/49) U**2)] exactly under that discrete law. The
atomic parametrization and CRPS are established methods. Their adaptation here
is a research candidate, not an accredited novelty or calibrated distribution.
No data are loaded and no model is fitted by importing/constructing this module.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from src.e3_development import DevelopmentModel

VERSION = "e3-overlap-atomic-v3-development"
VARIANTS = ("atom_crps", "atom_mse_only", "fixed_probs")
NEURAL_VARIANTS = VARIANTS
C = 6.0 / 49.0
MIN_NORMALIZED_INCREMENT = 1e-4


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def _nonnegative(value, name, *, device):
    result = torch.as_tensor(value, dtype=torch.float64, device=device)
    if not bool(torch.isfinite(result).all()) or not bool((result >= 0).all()):
        raise ValueError(name + " must be finite and nonnegative")
    return result


def discrete_crps(atoms, probabilities, observations):
    """Exact E|Z-u| - .5 E|Z-Z'| for a finite probability law, per sample.

    Atoms/observations must already share units; normalization belongs to the
    caller. Absolute-value ties use PyTorch's zero subgradient convention.
    """
    if not isinstance(atoms, torch.Tensor) or atoms.ndim != 2 or atoms.shape[1] < 1:
        raise ValueError("atoms must have shape batch x K")
    atoms = atoms.to(torch.float64)
    probabilities = torch.as_tensor(probabilities, dtype=torch.float64, device=atoms.device)
    observations = torch.as_tensor(observations, dtype=torch.float64, device=atoms.device)
    if probabilities.shape != atoms.shape or observations.shape != atoms.shape[:1]:
        raise ValueError("Aligned atom probabilities and one observation per sample required")
    if (not bool(torch.isfinite(atoms).all()) or not bool(torch.isfinite(probabilities).all())
            or not bool(torch.isfinite(observations).all()) or not bool((probabilities >= 0).all())
            or not torch.allclose(probabilities.sum(-1), torch.ones_like(observations), atol=1e-12, rtol=1e-12)):
        raise ValueError("Finite probability laws required")
    first = (probabilities * (atoms - observations[:, None]).abs()).sum(-1)
    pair = (atoms[:, :, None] - atoms[:, None, :]).abs()
    second = (probabilities[:, :, None] * probabilities[:, None, :] * pair).sum((1, 2))
    return first - .5 * second


def loss_components(outputs, Y, U, *, y_scale, u_scale, lambda_crps):
    """Normalized sigma MSE plus dimensionless CRPS of the magnitude target.

    The CRPS is evaluated at U/u_scale, so it is not divided by u_scale twice.
    U is a training label and is never an input to forward().
    """
    prediction = outputs["prediction"]
    Y = _nonnegative(Y, "sigma target", device=prediction.device)
    U = _nonnegative(U, "magnitude target", device=prediction.device)
    if Y.shape != U.shape or Y.shape != prediction.shape:
        raise ValueError("Aligned prediction and targets required")
    ys = torch.as_tensor(y_scale, dtype=torch.float64, device=prediction.device)
    us = torch.as_tensor(u_scale, dtype=torch.float64, device=prediction.device)
    if ys.numel() != 1 or us.numel() != 1 or not bool(torch.isfinite(ys)) or not bool(torch.isfinite(us)) or not bool(ys > 0) or not bool(us > 0):
        raise ValueError("Positive scalar training scales required")
    if isinstance(lambda_crps, bool) or not isinstance(lambda_crps, (int, float)) or not math.isfinite(lambda_crps) or lambda_crps < 0:
        raise ValueError("lambda_crps must be finite and nonnegative")
    mse = ((prediction - Y) / ys).square().mean()
    crps = discrete_crps(outputs["atoms"] / us, outputs["probabilities"], U / us).mean()
    return {"loss": mse + float(lambda_crps) * crps, "mse_scaled": mse, "crps_u": crps}


class OverlapDistributionModel(nn.Module):
    """1197 shared encoder/attention parameters plus an atomic-law readout.

    atom_crps/atom_mse_only: 1262 trainable parameters, three ordered atoms and
    two free logits with a fixed reference logit. fixed_probs: 1236 parameters,
    three atoms and uniform probabilities; it is not claimed capacity-matched.
    """
    def __init__(self, variant="atom_crps", *, u_scale, y_scale, seed=20261118,
                 lambda_crps=.1):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown overlap-model variant")
        if type(seed) is not int or not 0 <= seed < 2 ** 63:
            raise ValueError("seed must be an integer in [0,2**63)")
        self.variant = self.name = self.model_name = variant
        self.seed = seed
        if isinstance(lambda_crps, bool) or not isinstance(lambda_crps, (int, float)) or not math.isfinite(float(lambda_crps)) or lambda_crps < 0:
            raise ValueError("lambda_crps must be finite and nonnegative")
        self.lambda_crps = 0.0 if variant == "atom_mse_only" else float(lambda_crps)
        base = DevelopmentModel("M00", seed).model
        self.encoder = base.encoder
        self.query_raw = nn.Parameter(base.mechanism.query_raw.detach().clone())
        self.temperature_raw = nn.Parameter(base.mechanism.temperature_raw.detach().clone())
        self.register_buffer("u_scale", torch.tensor(_positive(u_scale, "u_scale"), dtype=torch.float64))
        self.register_buffer("y_scale", torch.tensor(_positive(y_scale, "y_scale"), dtype=torch.float64))
        # Canonical five-row initialization makes atom rows identical across all
        # controls. Separate atom/logit GEMMs also keep their arithmetic identical
        # across controls with different active capacity. CPU RNG is restored.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.head = nn.Linear(12, 3, dtype=torch.float64)
            self.probability_head = (None if variant == "fixed_probs" else
                                     nn.Linear(12, 2, dtype=torch.float64))
            generator = torch.Generator(device="cpu")
            generator.manual_seed((seed + 49157) % (2 ** 63))
            weight = torch.randn(5, 12, generator=generator, dtype=torch.float64) * .01
            bias = torch.zeros(5, dtype=torch.float64)
            increments = torch.tensor([.3, .5, .8], dtype=torch.float64)
            bias[:3] = torch.log(torch.expm1(increments - MIN_NORMALIZED_INCREMENT))
            with torch.no_grad():
                self.head.weight.copy_(weight[:3])
                self.head.bias.copy_(bias[:3])
                if self.probability_head is not None:
                    self.probability_head.weight.zero_()  # uniform initial probabilities
                    self.probability_head.bias.zero_()

    def _context(self, values):
        if not isinstance(values, torch.Tensor) or values.dtype != torch.float32:
            raise ValueError("Encoder requires standardized float32 input")
        states = self.encoder(values)
        query = self.query_raw / torch.sqrt(1.0 + self.query_raw.square().sum())
        temperature = .25 + F.softplus(self.temperature_raw)
        weights = torch.softmax((states @ query) / temperature, dim=1)
        return (weights[:, :, None] * states).sum(1)

    def forward_components(self, values, A):
        context = self._context(values)
        context = context.to(torch.float64)
        atom_raw = self.head(context)
        probability_raw = (None if self.probability_head is None else self.probability_head(context))
        raw = atom_raw if probability_raw is None else torch.cat([atom_raw, probability_raw], dim=-1)
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("Nonfinite head output")
        A = _nonnegative(A, "A", device=raw.device)
        if A.shape != raw.shape[:1]:
            raise ValueError("One raw variance floor per input window required")
        increments = self.u_scale * (MIN_NORMALIZED_INCREMENT + F.softplus(atom_raw))
        atoms = increments.cumsum(-1)
        if self.variant == "fixed_probs":
            probabilities = torch.full_like(atoms, 1.0 / 3.0)
        else:
            logits = torch.cat([probability_raw, raw.new_zeros((len(raw), 1))], dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
        # hypot avoids squaring an otherwise representable large atom. Its
        # derivative with respect to external A at exactly zero is not claimed.
        sigma_atoms = torch.hypot(torch.sqrt(A)[:, None], math.sqrt(C) * atoms)
        prediction = (probabilities * sigma_atoms).sum(-1)
        prediction = torch.maximum(prediction, torch.sqrt(A))
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("Nonfinite volatility prediction")
        return {"prediction": prediction, "atoms": atoms, "probabilities": probabilities,
                "sigma_atoms": sigma_atoms, "raw_head": raw, "scale": None}

    def forward(self, values, A):
        return self.forward_components(values, A)["prediction"]

    def loss_components(self, outputs, Y, U):
        return loss_components(outputs, Y, U, y_scale=self.y_scale,
                               u_scale=self.u_scale, lambda_crps=self.lambda_crps)

    def configuration(self):
        return {"version": VERSION, "variant": self.variant, "seed": self.seed,
                "backbone": "M00_encoder_and_content_attention_no_penalty_or_saturation",
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "atoms": 3, "learned_probabilities": self.variant != "fixed_probs",
                "lambda_crps": self.lambda_crps, "u_scale": float(self.u_scale),
                "y_scale": float(self.y_scale), "minimum_normalized_increment": MIN_NORMALIZED_INCREMENT,
                "encoder_attention_precision": "float32", "atomic_head_precision": "float64",
                "exact_mean_under_finite_atomic_law": True,
                "exact_true_conditional_law_claimed": False,
                "calibration_claimed": False, "R1_R3_verified": False,
                "scope": "new development candidate, not a frozen confirmatory protocol"}
