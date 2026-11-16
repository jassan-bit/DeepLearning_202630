"""Exploratory moment heads on the unchanged causal CNN-LSTM-content attention.

The moment envelope and mean scores have known mathematical antecedents. This
implementation is not evidence of novelty, optimization fidelity, coverage or
superiority. The old M00 readout is discarded, rather than retained as unused
parameters: 1197 encoder/attention parameters plus 39 active readout parameters.

Inputs are already transformed FP32 OHLCV and the observable variance statistic
A. Future U, U**2 and U**4 are loss targets only. All new output algebra uses
FP64 without an undisclosed positive floor or clipping of reports.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.e3_development import DevelopmentModel
from src.e3_innovation_model import load_innovation_data


VERSION = "e3-moment-model-exploratory-v1"
NEURAL_NAMES = ("moment_joint", "moment_mean_only", "direct_moment_aux")
NEURAL_VARIANTS = NEURAL_NAMES
C = 6.0 / 49.0


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _positive(value, name):
    _require(not isinstance(value, bool) and isinstance(value, (int, float, np.floating))
             and math.isfinite(float(value)) and float(value) > 0,
             name + " must be finite and positive")
    return float(value)


def inverse_softplus(value):
    value = _positive(value, "inverse-softplus argument")
    return value + math.log(-math.expm1(-value))


def _double(value, *, device=None):
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _nonnegative(value, name, *, device=None, positive=False):
    value = _double(value, device=device)
    _require(bool(torch.isfinite(value).all())
             and bool((value > 0 if positive else value >= 0).all()),
             name + (" must be finite and positive" if positive else " must be finite and nonnegative"))
    return value


def moment_bounds(A, m, q):
    """Known bounds for the conditional MEAN; they are not prediction intervals.

    q=m² is an exact mathematical branch. Other finite reports can lose width
    or strict endpoint inequalities through floating-point rounding; diagnostics
    expose that limitation instead of silently changing moments or predictions.
    """
    device = m.device if isinstance(m, torch.Tensor) else (A.device if isinstance(A, torch.Tensor) else None)
    A, m, q = torch.broadcast_tensors(_nonnegative(A, "A", device=device),
                                     _nonnegative(m, "m", device=device, positive=True),
                                     _nonnegative(q, "q", device=device, positive=True))
    m_squared = m.square()
    _require(bool(torch.isfinite(m_squared).all()) and bool((m_squared > 0).all()),
             "m squared underflowed or overflowed in FP64")
    _require(bool((q >= m_squared).all()), "Moment reports require q >= m squared")
    ratio = q / m
    _require(bool(torch.isfinite(ratio).all()), "q/m is outside the FP64 operating range")
    floor = torch.sqrt(A)
    upper = torch.sqrt(A + C * m)
    lower = floor + C * m / (torch.sqrt(A + C * ratio) + floor)
    # The degenerate law V=m has exactly one mean. This is an algebraic branch,
    # not a tolerance, clipping rule or modification of the moment domain.
    lower = torch.where(q == m_squared, upper, lower)
    _require(bool(torch.isfinite(lower).all()) and bool(torch.isfinite(upper).all())
             and bool((lower <= upper).all()), "Moment envelope is not representable safely in FP64")
    return lower, upper


def moment_head(A, a, b, d, *, u_scale):
    """Pure differentiable FP64 head; a,b,d are linear readouts, not labels."""
    device = a.device if isinstance(a, torch.Tensor) else None
    A, a, b, d = torch.broadcast_tensors(_nonnegative(A, "A", device=device),
                                        _double(a, device=device), _double(b, device=device),
                                        _double(d, device=device))
    _require(all(bool(torch.isfinite(v).all()) for v in (a, b, d)), "Readouts must be finite")
    us = _nonnegative(u_scale, "u_scale", device=device, positive=True)
    _require(us.ndim == 0, "u_scale must be scalar")
    unit = us.square()
    _require(bool(torch.isfinite(unit)) and bool(unit > 0), "u_scale squared is outside FP64 range")
    m = unit * F.softplus(a)
    b_squared, d_squared = b.square(), d.square()
    _require(bool(torch.isfinite(b_squared).all()) and bool(torch.isfinite(d_squared).all()),
             "Squared shape readouts are outside the FP64 operating range")
    q = m.square() * (1.0 + b_squared)
    rho = d_squared / (1.0 + d_squared)
    lower, upper = moment_bounds(A, m, q)
    prediction = lower + rho * (upper - lower)
    _require(bool(torch.isfinite(prediction).all()), "Nonfinite moment-head prediction")
    return {"prediction": prediction, "m": m, "q": q, "rho": rho,
            "L": lower, "H": upper,
            "rho_rounded_to_one": rho == 1,
            "envelope_collapsed_by_roundoff": (q > m.square()) & (lower == upper),
            "upper_endpoint_reached_by_roundoff": (q > m.square()) & (prediction >= upper)}


def load_moment_data(npz_path, expected_sha256, **date_kwargs):
    """Reuse the audited three-training-member loader and add FIT-only statistics.

    The direct control matches the initial moment head's mean excess over the
    observable floor. Because all readout weights start at zero, this is an
    algebraic calculation using fit A/U alone, without evaluating an encoder or
    accessing validation statistics. Pointwise initial predictions can differ.
    """
    data = load_innovation_data(npz_path, expected_sha256, **date_kwargs)
    try:
        with np.errstate(over="raise", invalid="raise"):
            mean_v = _positive(float(np.mean(np.square(data["train_U"]))), "Fit mean V")
    except FloatingPointError:
        raise ValueError("Fit innovation moment is outside the FP64 operating range") from None
    with torch.no_grad():
        fit_A = _double(data["train_A"])
        unit = _positive(data["scaler"]["u_scale"] * data["scaler"]["u_scale"],
                         "Fit u_scale squared")
        a = _double(inverse_softplus(mean_v / unit))
        out = moment_head(fit_A, a, _double(math.sqrt(8)), _double(1.),
                          u_scale=data["scaler"]["u_scale"])
        initial_excess = _positive(float((out["prediction"] - torch.sqrt(fit_A)).mean()),
                                   "Initial fit moment-head mean excess")
    data["scaler"].update(mean_v=mean_v, initial_mean_excess=initial_excess,
                           moment_initial_mean_excess=initial_excess)
    data["admission"].update(version=VERSION, additional_statistics_fit_only=True,
                              moment_initialization_uses_validation=False,
                              moment_statistics={"mean_v": mean_v, "initial_mean_excess": initial_excess})
    return data


class MomentModel(nn.Module):
    """Same 1197-parameter M00 backbone and three active scalar readouts (1236).

    The first encoder gradient is zero because all readout weights start at zero;
    this is intentional and shared across controls, not unused backbone storage.
    b=0 and d=0 are flat-gradient points; initialize b=sqrt(8), d=1 for moment
    heads. No future optimization or finite-sample improvement is guaranteed.
    """
    def __init__(self, name, *, u_scale, y_scale, mean_v, initial_mean_excess=None,
                 seed=20261116):
        super().__init__()
        _require(name in NEURAL_NAMES, "Unknown moment model")
        u_scale, y_scale, mean_v = (_positive(u_scale, "u_scale"),
                                    _positive(y_scale, "y_scale"), _positive(mean_v, "mean_v"))
        unit = u_scale * u_scale
        _require(math.isfinite(unit) and unit > 0, "u_scale squared is outside FP64 range")
        initial_a = inverse_softplus(mean_v / unit)
        initial_d = 1.
        if name == "direct_moment_aux":
            initial_d = inverse_softplus(_positive(initial_mean_excess, "initial_mean_excess") / y_scale)
        self.name = self.variant = self.model_name = name
        self.seed = seed
        self.lambda_m = self.lambda_q = 0. if name == "moment_mean_only" else .1

        # The project constructor exactly preserves CPU RNG state and uses the
        # frozen M00 seed. Retain only ancestors used by this model; its old
        # readout/bias and disabled-penalty parameter are not registered here.
        original = DevelopmentModel("M00", seed).model
        self.encoder = original.encoder
        self.config = original.config
        self.query_raw = original.mechanism.query_raw
        self.temperature_raw = original.mechanism.temperature_raw
        self.readout_weight = nn.Parameter(torch.zeros(3, self.config.hidden_size, dtype=torch.float64))
        self.readout_bias = nn.Parameter(torch.tensor([initial_a, math.sqrt(8.), initial_d], dtype=torch.float64))
        self.register_buffer("u_scale", _double(u_scale))
        self.register_buffer("y_scale", _double(y_scale))
        self.initial_mean_v = mean_v
        self.initial_mean_excess = initial_mean_excess

    def context(self, values):
        _require(isinstance(values, torch.Tensor) and values.dtype == torch.float32,
                 "The M00 encoder and attention require FP32 input")
        states = self.encoder(values)
        query = self.config.query_bound * self.query_raw / torch.sqrt(1. + self.query_raw.square().sum())
        temperature = self.config.temperature_min + F.softplus(self.temperature_raw)
        weights = torch.softmax((states @ query) / temperature, dim=-1)
        return (weights.unsqueeze(-1) * states).sum(dim=1)

    def forward_components(self, values, A):
        context = self.context(values)
        # A is an observed statistic, not an optimization parameter. Detachment
        # also prevents the sqrt(A) derivative at A=0 entering a training graph.
        A = _nonnegative(A, "A", device=context.device).detach()
        _require(A.shape == (len(values),), "A must have one value per input window")
        raw = context.to(torch.float64) @ self.readout_weight.T + self.readout_bias
        a, b, d = raw.unbind(dim=-1)
        output = moment_head(A, a, b, d if self.name != "direct_moment_aux" else torch.ones_like(d),
                             u_scale=self.u_scale)
        if self.name == "direct_moment_aux":
            output["prediction"] = torch.sqrt(A) + self.y_scale * F.softplus(d)
            _require(bool(torch.isfinite(output["prediction"]).all()), "Nonfinite direct mean")
            output["rho"] = None
            output["rho_rounded_to_one"] = None
            output["upper_endpoint_reached_by_roundoff"] = None
        output["raw_scores"] = raw
        return output

    components = forward_components

    def forward(self, values, A):
        return self.forward_components(values, A)["prediction"]

    def loss_components(self, outputs, Y, U):
        return loss_components(outputs, Y, U, y_scale=self.y_scale, u_scale=self.u_scale,
                               lambda_m=self.lambda_m, lambda_q=self.lambda_q)

    def configuration(self):
        return {"version": VERSION, "variant": self.name, "seed": self.seed,
                "backbone": "unchanged_M00_CNN_LSTM_content_attention",
                "cnn_channels": self.config.cnn_channels, "lstm_hidden_size": self.config.hidden_size,
                "penalty": False, "radial_saturation": False,
                "backbone_parameters": sum(p.numel() for p in self.encoder.parameters())
                                       + self.query_raw.numel() + self.temperature_raw.numel(),
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "active_readouts": 3, "readout_precision": "float64",
                "encoder_attention_precision": "float32", "u_scale": float(self.u_scale),
                "y_scale": float(self.y_scale), "initial_mean_v": self.initial_mean_v,
                "initial_mean_excess": self.initial_mean_excess,
                "lambda_m": self.lambda_m, "lambda_q": self.lambda_q,
                "zero_readout_weights_initially": True, "no_hidden_positive_floor": True,
                "prediction_units": "sigma7_original", "prediction_interval_claimed": False,
                "floating_point_strict_endpoint_guaranteed": False,
                "moment_identification_claimed": False,
                "moment_supervision_enabled": self.name != "moment_mean_only",
                "R1_R3_verified": False}


def loss_components(outputs, Y, U, *, y_scale, u_scale, lambda_m=.1, lambda_q=.1):
    """MSE plus known mean scores; this is not a joint likelihood or density."""
    prediction = outputs["prediction"]
    _require(isinstance(prediction, torch.Tensor), "Prediction tensor required")
    Y = _nonnegative(Y, "Y", device=prediction.device)
    U = _nonnegative(U, "U", device=prediction.device)
    m = _nonnegative(outputs["m"], "m", device=prediction.device, positive=True)
    q = _nonnegative(outputs["q"], "q", device=prediction.device, positive=True)
    _require(prediction.dtype == torch.float64 and bool(torch.isfinite(prediction).all())
             and bool((prediction >= 0).all()), "Prediction must be finite nonnegative FP64")
    _require(prediction.shape == Y.shape == U.shape == m.shape == q.shape,
             "Aligned prediction, moment reports and targets required")
    ys = _nonnegative(y_scale, "y_scale", device=prediction.device, positive=True)
    us = _nonnegative(u_scale, "u_scale", device=prediction.device, positive=True)
    _require(ys.ndim == us.ndim == 0, "Normalization scales must be scalar")
    for weight in (lambda_m, lambda_q):
        _require(not isinstance(weight, bool) and isinstance(weight, (int, float))
                 and math.isfinite(weight) and weight >= 0, "Auxiliary weights must be finite nonnegative numbers")
    v = (U / us).square()
    mn, qn = m / us.square(), q / us.square().square()
    _require(bool(torch.isfinite(v).all()) and bool(torch.isfinite(mn).all())
             and bool(torch.isfinite(qn).all()) and bool((mn > 0).all()) and bool((qn > 0).all()),
             "Normalized moment reports are outside the FP64 operating range")
    point = ((prediction - Y) / ys).square().mean()
    m_score = (torch.log(mn) + v / mn).mean()
    q_score = (torch.log(qn) + v.square() / qn).mean()
    total = point + lambda_m * m_score + lambda_q * q_score
    _require(all(bool(torch.isfinite(value)) for value in (point, m_score, q_score, total)),
             "Nonfinite moment loss; no labels or reports were clipped")
    return {"loss": total, "mse_scaled": point, "m_score": m_score, "q_score": q_score}


def moment_loss(model, X, A, Y, U):
    return model.loss_components(model.forward_components(X, A), Y, U)["loss"]
