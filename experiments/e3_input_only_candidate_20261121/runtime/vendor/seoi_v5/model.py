"""Causal scale reconstruction inside an overlap-aware atomic readout.

Own adaptation of known reversible normalization; CNN/LSTM/attention and the
atomic law are inherited, attributed components. No fitting on import. The
candidate is seoi_mse, not an automatically promoted old MSE-only checkpoint.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from .frozen_original import OriginalModel

RECIPES = ("seoi_mse", "input_only_mse", "output_only_mse", "global_mse",
           "global_crps", "additive_mse")
LOCAL_INPUT = ("seoi_mse", "input_only_mse")
LOCAL_OUTPUT = ("seoi_mse", "output_only_mse")
C = 6.0 / 49.0
MIN_INCREMENT = 1e-4


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def discrete_crps(atoms, probabilities, observation):
    """Exact finite-law CRPS in original U units; established score, not new."""
    if atoms.ndim != 2 or probabilities.shape != atoms.shape or observation.shape != atoms.shape[:1]:
        raise ValueError("Aligned BxK atoms/probabilities and B observations required")
    if not (torch.isfinite(atoms).all() and torch.isfinite(probabilities).all() and torch.isfinite(observation).all()):
        raise ValueError("Nonfinite CRPS argument")
    if not ((probabilities >= 0).all() and torch.allclose(probabilities.sum(-1), torch.ones_like(observation), atol=1e-12, rtol=1e-12)):
        raise ValueError("A probability law is required")
    first = (probabilities * (atoms - observation[:, None]).abs()).sum(-1)
    pair = (atoms[:, :, None] - atoms[:, None, :]).abs()
    return first - 0.5 * (probabilities[:, :, None] * probabilities[:, None, :] * pair).sum((1, 2))


class ScaleOverlapModel(nn.Module):
    """M00 backbone + own scale-reconstructing head, FP32 encoder / FP64 head.

    scale is the dimensionless s_t/r_fit. It is never an encoder feature.
    Input-only/global controls ignore this argument in their readout.
    """
    def __init__(self, recipe, seed, u_scale, y_scale):
        super().__init__()
        if recipe not in RECIPES:
            raise ValueError("Unknown recipe")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Invalid seed")
        self.recipe = self.name = recipe
        self.seed = seed
        self.register_buffer("u_scale", torch.tensor(positive(u_scale, "u_scale"), dtype=torch.float64))
        self.register_buffer("y_scale", torch.tensor(positive(y_scale, "y_scale"), dtype=torch.float64))
        base = OriginalModel(seed=seed)
        self.encoder = base.encoder
        self.query_raw = nn.Parameter(base.mechanism.query_raw.detach().clone())
        self.temperature_raw = nn.Parameter(base.mechanism.temperature_raw.detach().clone())
        self.additive = recipe == "additive_mse"
        if self.additive:
            # Same bounded readout and raw initialization as the old additive.
            self.readout_raw = nn.Parameter(base.mechanism.readout_raw.detach().to(torch.float64))
            self.bias = nn.Parameter(base.mechanism.bias.detach().to(torch.float64))
        else:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.head = nn.Linear(12, 3, dtype=torch.float64)
                self.probability_head = nn.Linear(12, 2, dtype=torch.float64)
                generator = torch.Generator(device="cpu").manual_seed((seed + 49157) % (2**63))
                weight = torch.randn(5, 12, generator=generator, dtype=torch.float64) * .01
                bias = torch.log(torch.expm1(torch.tensor([.3, .5, .8], dtype=torch.float64) - MIN_INCREMENT))
                with torch.no_grad():
                    self.head.weight.copy_(weight[:3])
                    self.head.bias.copy_(bias)
                    self.probability_head.weight.zero_()
                    self.probability_head.bias.zero_()

    def forward_components(self, X, A, scale):
        if not isinstance(X, torch.Tensor) or X.dtype != torch.float32 or X.ndim != 3 or X.shape[1:] != (7, 5):
            raise ValueError("Expected float32 Bx7x5 causal features")
        A = torch.as_tensor(A, dtype=torch.float64, device=X.device)
        scale = torch.as_tensor(scale, dtype=torch.float64, device=X.device)
        if A.shape != (len(X),) or scale.shape != A.shape or not torch.isfinite(A).all() or not (A >= 0).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Aligned nonnegative A and positive dimensionless scales required")
        states = self.encoder(X)
        query = self.query_raw / torch.sqrt(1 + self.query_raw.square().sum())
        weights = torch.softmax((states @ query) / (.25 + F.softplus(self.temperature_raw)), dim=1)
        context = (weights[:, :, None] * states).sum(1).to(torch.float64)
        if self.additive:
            readout = 4.0 * self.readout_raw / torch.sqrt(1 + self.readout_raw.square().sum())
            score = context @ readout + self.bias
            prediction = torch.sqrt(A) + self.y_scale * (MIN_INCREMENT + F.softplus(score))
            return {"prediction": prediction, "atoms": None, "probabilities": None}
        increments = MIN_INCREMENT + F.softplus(self.head(context))
        factor = scale if self.recipe in LOCAL_OUTPUT else torch.ones_like(scale)
        atoms = self.u_scale * factor[:, None] * increments.cumsum(-1)
        logits = torch.cat([self.probability_head(context), context.new_zeros((len(X), 1))], dim=-1)
        probabilities = torch.softmax(logits, -1)
        sigma_atoms = torch.hypot(torch.sqrt(A)[:, None], math.sqrt(C) * atoms)
        prediction = (probabilities * sigma_atoms).sum(-1)
        # Floating arithmetic guard; support is already proved in real arithmetic.
        prediction = torch.maximum(prediction, torch.sqrt(A))
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite prediction")
        return {"prediction": prediction, "atoms": atoms, "probabilities": probabilities,
                "sigma_atoms": sigma_atoms}

    def forward(self, X, A, scale):
        return self.forward_components(X, A, scale)["prediction"]

    def loss_components(self, outputs, Y, U):
        prediction = outputs["prediction"]
        Y = torch.as_tensor(Y, dtype=torch.float64, device=prediction.device)
        U = torch.as_tensor(U, dtype=torch.float64, device=prediction.device)
        if Y.shape != prediction.shape or U.shape != Y.shape or not torch.isfinite(Y).all() or not torch.isfinite(U).all() or not (Y >= 0).all() or not (U >= 0).all():
            raise ValueError("Aligned nonnegative targets required")
        mse = ((prediction - Y) / self.y_scale).square().mean()
        crps = (discrete_crps(outputs["atoms"] / self.u_scale, outputs["probabilities"], U / self.u_scale).mean()
                if self.recipe == "global_crps" else mse.new_zeros(()))
        return {"loss": mse + (0.1 * crps if self.recipe == "global_crps" else 0),
                "mse_scaled": mse, "crps_u": crps}

    def configuration(self):
        return {"version": "e3-seoi-v5-development-1", "recipe": self.recipe,
                "seed": self.seed, "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "local_input": self.recipe in LOCAL_INPUT, "local_output": self.recipe in LOCAL_OUTPUT,
                "u_scale": float(self.u_scale), "y_scale": float(self.y_scale),
                "lambda_crps": .1 if self.recipe == "global_crps" else 0.0,
                "encoder_precision": "float32", "head_precision": "float64",
                "R1_automatically_accredited": False, "R3_accredited": False}


def build_model(recipe, seed, u_scale, y_scale):
    return ScaleOverlapModel(recipe, seed, u_scale, y_scale)
