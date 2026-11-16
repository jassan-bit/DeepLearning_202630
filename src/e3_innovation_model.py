"""Frozen V2 development design: unchanged M00 backbone and conditional heads.

No model is fitted at import, construction or data admission. The auxiliary
half-t likelihood models a magnitude, never a signed return. Quadrature is a
fixed numerical approximation, and nominal intervals are not calibrated.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.e3_development import DevelopmentModel

VERSION = "e3-innovation-model-v2"
NEURAL_NAMES = ("additive_positive_head", "innovation_mean_only", "innovation_density_mean")
NEURAL_VARIANTS = NEURAL_NAMES
NU, C, MIN_NORMALIZED_SCALE = 5.0, 6.0 / 49.0, 1e-4
MARGIN_TOLERANCE = 1e-12
RULE_SHA256 = "c3ce43d3cb81767ee3fc0daa0ccfe8432a3326643e28e51273a49cc1fe01c2fe"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULE = ROOT / "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json"
LOG_HALF_T5_CONSTANT = math.log(16.0 / (3.0 * math.pi * math.sqrt(5.0)))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def _positive(value, name):
    _require(not isinstance(value, bool) and isinstance(value, (float, int, np.floating))
             and math.isfinite(float(value)) and float(value) > 0, name + " must be finite and positive")
    return float(value)


def _iso(value):
    _require(isinstance(value, str) and date.fromisoformat(value).isoformat() == value,
             "Dates must use canonical ISO YYYY-MM-DD")
    return value


def load_quadrature(rule_path=None):
    """Read and admit the byte-pinned N128 artifact, without regenerating nodes."""
    path = DEFAULT_RULE if rule_path is None else Path(rule_path)
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    receipt = path.with_suffix(path.suffix + ".sha256").read_text().strip()
    _require(actual == RULE_SHA256 and receipt == actual, "Frozen quadrature SHA-256 mismatch")
    record = json.loads(payload)
    for key, expected in {"N": 128, "nu": NU, "c": C,
                          "jacobi_alpha": 1.0, "jacobi_beta": -0.5}.items():
        _require(record.get(key) == expected, "Frozen quadrature parameter mismatch: " + key)
    a, z = np.asarray(record["a_i"], dtype=np.float64), np.asarray(record["z_i"], dtype=np.float64)
    _require(a.shape == z.shape == (128,) and np.isfinite(a).all() and np.isfinite(z).all()
             and (a > 0).all() and (z > 0).all() and abs(float(a.sum()) - 1) < 5e-13,
             "Invalid positive quadrature")
    bound = math.sqrt(C) * float(a @ z)
    _require(math.isclose(bound, record["L_s_discrete"], rel_tol=1e-12), "Quadrature bound mismatch")
    return {"a": a.copy(), "z": z.copy(), "sha256": actual, "L_s": bound, "path": str(path)}


def load_innovation_data(npz_path, expected_sha256, fit_end="2024-06-30",
                         validation_start="2024-07-08", validation_end="2024-10-31"):
    """Admit only three raw training members; compute all fitted statistics on fit.

    Every admitted row is checked against the algebraic variance floor, including
    purged rows. An incoherent label aborts admission; no row is silently dropped.
    This loader does not reconstruct an unavailable future close from D7 inputs.
    """
    _require(isinstance(expected_sha256, str) and len(expected_sha256) == 64
             and all(x in "0123456789abcdefABCDEF" for x in expected_sha256), "Explicit NPZ SHA-256 required")
    actual = _sha(npz_path)
    _require(actual == expected_sha256.lower(), "NPZ SHA-256 mismatch")
    fit_end, validation_start, validation_end = map(_iso, (fit_end, validation_start, validation_end))
    _require(date.fromisoformat(validation_start) - timedelta(days=7) > date.fromisoformat(fit_end),
             "Fit/validation raw supports overlap")
    _require(validation_start <= validation_end <= "2024-10-31", "Invalid training-era validation dates")
    keys = ("training_raw", "training_y_original", "training_dates")
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            raw = np.asarray(source[keys[0]], dtype=np.float64).copy()
            y = np.asarray(source[keys[1]], dtype=np.float64).copy()
            ids = np.asarray(source[keys[2]]).copy()
    except KeyError as error:
        raise ValueError("Required raw training-only NPZ member missing") from error
    _require(raw.ndim == 3 and raw.shape[1:] == (7, 5) and len(raw) >= 3
             and y.shape == ids.shape == (len(raw),), "Expected raw N x 7 x 5 and aligned targets/dates")
    ids = np.asarray([_iso(str(x)) for x in ids], dtype="U10")
    _require(ids.tolist() == sorted(set(ids.tolist())) and ids[-1] <= "2024-10-31",
             "Dates must be ordered, unique and training-era")
    _require(np.isfinite(raw).all() and np.isfinite(y).all() and (y >= 0).all()
             and (raw[..., :4] > 0).all() and (raw[..., 4] >= 0).all(), "Invalid raw OHLCV or target")
    _require((raw[..., 1] >= np.maximum(raw[..., 0], raw[..., 3])).all()
             and (raw[..., 2] <= np.minimum(raw[..., 0], raw[..., 3])).all()
             and (raw[..., 1] >= raw[..., 2]).all(), "Invalid OHLC geometry")
    fit = ids <= fit_end
    val = (ids >= validation_start) & (ids <= validation_end)
    _require(int(fit.sum()) >= 2 and int(val.sum()) >= 1, "Nonempty fit/validation cohorts required")
    known_returns = np.diff(np.log(raw[..., 3]), axis=1)
    mean6 = known_returns.mean(axis=1)
    A = ((known_returns - mean6[:, None]) ** 2).sum(axis=1) / 7.0
    margin = y ** 2 - A
    _require(np.isfinite(A).all() and np.isfinite(margin).all(), "Nonfinite algebraic variance margin")
    if np.any(margin < -MARGIN_TOLERANCE):
        bad = np.flatnonzero(margin < -MARGIN_TOLERANCE)
        raise ValueError(f"Target violates sigma7 floor on {len(bad)} row(s), first date {ids[bad[0]]}; no rows dropped")
    U = np.sqrt(np.maximum(margin, 0.0) / C)
    floor = np.sqrt(A)
    relative = np.empty_like(raw)
    relative[..., :4] = np.log(raw[..., :4]) - np.log(raw[:, -1, 3])[:, None, None]
    relative[..., 4] = np.log1p(raw[..., 4])
    mean, std = relative[fit].mean(axis=(0, 1)), relative[fit].std(axis=(0, 1), ddof=0)
    std = np.where(std == 0, 1.0, std)
    y_scale = _positive(float(y[fit].mean()), "Fit target mean")
    u_scale = _positive(float(U[fit].mean()), "Fit magnitude mean")
    d_fit = _positive(float((y[fit] - floor[fit]).mean()), "Fit mean excess")
    train_x = ((relative[fit] - mean) / std).astype(np.float32)
    val_x = ((relative[val] - mean) / std).astype(np.float32)
    _require(np.isfinite(train_x).all() and np.isfinite(val_x).all(), "Nonfinite normalized inputs")
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "ddof": 0,
              "target_scale": y_scale, "y_scale": y_scale, "u_scale": u_scale, "d_fit": d_fit,
              "fit_dates": ids[fit].tolist(), "fit_target_q75": float(np.quantile(y[fit], .75)),
              "relative_window_anchor": "last_known_close", "target_centered": False}
    result = {"train_X": train_x, "val_X": val_x, "scaler": scaler}
    for name, mask in (("train", fit), ("val", val)):
        result.update({name + "_A": A[mask].copy(), name + "_U": U[mask].copy(),
                       name + "_y_original": y[mask].copy(), name + "_dates": ids[mask].copy(),
                       name + "_floor": floor[mask].copy(),
                       name + "_y_scaled": (y[mask] / y_scale).astype(np.float64)})
    for suffix in ("X", "A", "U", "y_original", "dates", "floor", "y_scaled"):
        result["validation_" + suffix] = result["val_" + suffix]
    result["admission"] = {
        "version": VERSION, "npz_sha256": actual, "keys_read": list(keys),
        "forbidden_arrays_loaded": False, "training_scaled_arrays_reused": False,
        "n_training_source": len(raw), "n_fit": int(fit.sum()), "n_validation": int(val.sum()),
        "n_purged_or_unused": int((~(fit | val)).sum()), "rows_dropped": 0,
        "fit_dates": ids[fit].tolist(), "validation_dates": ids[val].tolist(),
        "fit_end": fit_end, "validation_start": validation_start, "validation_end": validation_end,
        "raw_supports_disjoint": True, "margin_tolerance": MARGIN_TOLERANCE,
        "minimum_variance_margin": float(margin.min()), "roundoff_margins_clamped": int((margin < 0).sum()),
        "U_is_target_not_input": True, "scaler": scaler,
        "arrays_sha256": {key: _array_sha(result[key]) for key in
                          ("train_X", "train_A", "train_U", "train_y_original",
                           "val_X", "val_A", "val_U", "val_y_original")},
        "interpretation": "Exposed development validation; not confirmation or R1/R3 evidence"}
    return result


def _double(value, *, device=None):
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _nonnegative(value, name, *, device=None, positive=False):
    value = _double(value, device=device)
    _require(bool(torch.isfinite(value).all()) and bool((value > 0 if positive else value >= 0).all()),
             name + (" must be finite and positive" if positive else " must be finite and nonnegative"))
    return value


def quadrature_mean(A, s, a=None, z=None):
    """Fixed positive g128; no claim of exact integration or global encoder bound."""
    device = s.device if isinstance(s, torch.Tensor) else (A.device if isinstance(A, torch.Tensor) else None)
    A, s = torch.broadcast_tensors(_nonnegative(A, "A", device=device), _nonnegative(s, "s", device=device))
    if a is None or z is None:
        rule = load_quadrature()
        a, z = _double(rule["a"], device=device), _double(rule["z"], device=device)
    value = (a * torch.sqrt(A[..., None] + C * s[..., None].square() * z.square())).sum(dim=-1)
    # The exact rule is above the floor; protect that invariant from last-bit
    # summation error when the innovation contribution is below machine precision.
    value = torch.maximum(value, torch.sqrt(A))
    _require(bool(torch.isfinite(value).all()), "Nonfinite quadrature mean")
    return value


def half_student_nll(u, s):
    """Half-t5 NLL with the fold factor included once; inputs share their units."""
    device = s.device if isinstance(s, torch.Tensor) else (u.device if isinstance(u, torch.Tensor) else None)
    u, s = torch.broadcast_tensors(_nonnegative(u, "U", device=device),
                                    _nonnegative(s, "scale", device=device, positive=True))
    log_odds = 2.0 * (torch.log(u) - torch.log(s)) - math.log(NU)
    return torch.log(s) - LOG_HALF_T5_CONSTANT + 3.0 * torch.logaddexp(torch.zeros_like(log_odds), log_odds)


def _student_t5_cdf(x):
    theta = math.atan(x / math.sqrt(5.0))
    return .5 + theta / math.pi + 2 * math.sin(2 * theta) / (3 * math.pi) + math.sin(4 * theta) / (12 * math.pi)


@lru_cache(maxsize=32)
def _student_t5_ppf(p):
    _require(math.isfinite(p) and 0 < p < 1, "Student quantile probability must lie in (0,1)")
    if p == .5:
        return 0.0
    if p < .5:
        return -_student_t5_ppf(1 - p)
    low, high = 0.0, 1.0
    while _student_t5_cdf(high) < p:
        high *= 2
        _require(math.isfinite(high), "Student quantile bracketing failed")
    for _ in range(80):
        middle = (low + high) / 2
        if _student_t5_cdf(middle) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def predictive_quantile(A, s, p):
    _require(isinstance(p, (int, float)) and not isinstance(p, bool) and 0 < p < 1,
             "Magnitude quantile probability must lie in (0,1)")
    device = s.device if isinstance(s, torch.Tensor) else (A.device if isinstance(A, torch.Tensor) else None)
    A, s = torch.broadcast_tensors(_nonnegative(A, "A", device=device),
                                    _nonnegative(s, "scale", device=device, positive=True))
    return torch.sqrt(A + C * s.square() * _student_t5_ppf((1 + p) / 2) ** 2)


def nominal_interval(A, s, coverage=.9):
    _require(isinstance(coverage, (float, int)) and not isinstance(coverage, bool) and 0 < coverage < 1,
             "Nominal coverage must lie in (0,1)")
    tail = (1 - coverage) / 2
    return predictive_quantile(A, s, tail), predictive_quantile(A, s, 1 - tail)


def analytic_predictions(A, d_fit):
    A = np.asarray(A, dtype=np.float64)
    _require(np.isfinite(A).all() and (A >= 0).all(), "Invalid analytic floor")
    d_fit = _positive(d_fit, "Fit mean excess")
    return {"floor_only": np.sqrt(A), "floor_plus_fit_mean": np.sqrt(A) + d_fit}


class InnovationModel(nn.Module):
    """1210 parameters: M00 encoder/attention FP32, bounded readout/bias FP64.

    The original M00's Softplus is not called. Its linear preactivation is
    evaluated in FP64 and receives exactly one Softplus in the proposed head.
    """
    def __init__(self, name, *, u_scale, y_scale, seed=20261116, rule_path=None):
        super().__init__()
        _require(name in NEURAL_NAMES, "Unknown innovation model")
        self.name = self.variant = self.model_name = name
        self.seed = seed
        self.base = DevelopmentModel("M00", seed).model
        mechanism = self.base.mechanism
        mechanism.readout_raw = nn.Parameter(mechanism.readout_raw.detach().to(torch.float64))
        mechanism.bias = nn.Parameter(mechanism.bias.detach().to(torch.float64))
        self.register_buffer("u_scale", torch.tensor(_positive(u_scale, "u_scale"), dtype=torch.float64))
        self.register_buffer("y_scale", torch.tensor(_positive(y_scale, "y_scale"), dtype=torch.float64))
        rule = load_quadrature(rule_path)
        self.register_buffer("quadrature_a", torch.tensor(rule["a"], dtype=torch.float64))
        self.register_buffer("quadrature_z", torch.tensor(rule["z"], dtype=torch.float64))
        self.rule_sha256, self.lipschitz_s = rule["sha256"], rule["L_s"]
        self.lambda_nll = .1 if name == "innovation_density_mean" else 0.0

    @property
    def bias(self):
        return self.base.mechanism.bias

    def _raw_score(self, values, *, include_bias=True):
        _require(isinstance(values, torch.Tensor) and values.dtype == torch.float32,
                 "The frozen M00 encoder/attention requires float32 inputs")
        states = self.base.encode(values)
        context = self.base.mechanism.attention_components(states)["context"]
        readout = self.base.mechanism.effective_parameters()["w"]
        score = context.to(torch.float64) @ readout
        return score + self.bias if include_bias else score

    def raw_score(self, values):
        return self._raw_score(values)

    def _components_from_score(self, score, A):
        A = _nonnegative(A, "A", device=score.device)
        _require(A.shape == score.shape, "A must have one value per input window")
        positive = MIN_NORMALIZED_SCALE + F.softplus(score)
        if self.name == "additive_positive_head":
            return {"prediction": torch.sqrt(A) + self.y_scale * positive,
                    "scale": None, "raw_score": score}
        s = self.u_scale * positive
        return {"prediction": quadrature_mean(A, s, self.quadrature_a, self.quadrature_z),
                "scale": s, "raw_score": score}

    def scale(self, values):
        _require(self.name != "additive_positive_head", "The additive control has no Student scale")
        return self.u_scale * (MIN_NORMALIZED_SCALE + F.softplus(self.raw_score(values)))

    def forward_components(self, values, A):
        return self._components_from_score(self.raw_score(values), A)

    def forward(self, values, A):
        return self.forward_components(values, A)["prediction"]

    def configuration(self):
        return {"version": VERSION, "variant": self.name, "seed": self.seed,
                "backbone": "unchanged_M00_CNN_LSTM_content_attention",
                "base": self.base.configuration(), "penalty": False, "radial_saturation": False,
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "readout_precision": "float64", "encoder_attention_precision": "float32",
                "u_scale": float(self.u_scale), "y_scale": float(self.y_scale),
                "minimum_normalized_scale": MIN_NORMALIZED_SCALE, "nu": NU,
                "lambda_nll": self.lambda_nll, "quadrature_sha256": self.rule_sha256,
                "prediction_units": "sigma7_original", "base_forward_not_called": True,
                "single_softplus_on_linear_score": True,
                "exact_mean_claimed": False, "calibration_claimed": False, "R1_R3_verified": False}

    @torch.no_grad()
    def calibrate_bias(self, X, A, target_d_fit, *, budget_callback=None, batch_size=32):
        """Match only the fit mean excess. No optimizer step or validation access.

        The callback is invoked before each model batch and bisection and may
        raise a deadline exception. Only a fully checked result commits the bias.
        """
        target = _positive(target_d_fit, "Fit mean excess")
        _require(type(batch_size) is int and batch_size > 0, "Positive batch size required")
        callback = budget_callback if budget_callback is not None else lambda: None
        old_bias, old_mode = self.bias.detach().clone(), self.training
        device = self.bias.device
        callback()
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
        A = _nonnegative(A, "A", device=device)
        _require(X.ndim == 3 and len(X) >= 2 and A.shape == (len(X),), "Aligned fit windows and A required")
        try:
            self.eval()
            scores = []
            for first in range(0, len(X), batch_size):
                callback()
                scores.append(self._raw_score(X[first:first + batch_size], include_bias=False))
            raw = torch.cat(scores)
            floor = torch.sqrt(A)

            def excess_at(bias):
                callback()
                score = raw + bias
                if self.name == "additive_positive_head":
                    return float((self.y_scale * (MIN_NORMALIZED_SCALE + F.softplus(score))).mean())
                scale = self.u_scale * (MIN_NORMALIZED_SCALE + F.softplus(score))
                squared = C * scale[:, None].square() * self.quadrature_z.square()
                # Rationalized subtraction avoids cancellation near the floor.
                excess = squared / (torch.sqrt(A[:, None] + squared) + floor[:, None])
                return float((excess * self.quadrature_a).sum(dim=1).mean())

            low, high = -40.0, 40.0
            at_low, at_high = excess_at(low), excess_at(high)
            _require(math.isfinite(at_low) and math.isfinite(at_high) and at_low < target < at_high,
                     "Fit mean excess is outside the fixed bias bracket or minimum-scale support")
            for _ in range(64):
                middle = (low + high) / 2.0
                if excess_at(middle) < target:
                    low = middle
                else:
                    high = middle
            solved = (low + high) / 2.0
            actual = excess_at(solved)
            tolerance = 1e-8 * float(self.y_scale)
            _require(abs(actual - target) <= tolerance, "Fit bias matching failed the frozen tolerance")
            callback()
            self.bias.copy_(self.bias.new_tensor(solved))
            return {"method": "64_step_fit_only_bisection", "bias": solved,
                    "target_d_fit": target, "matched_mean_excess": actual,
                    "absolute_error": abs(actual - target), "tolerance": tolerance,
                    "bracket": [-40.0, 40.0], "steps": 64, "fit_examples": len(X),
                    "fit_only": True, "complete": True}
        except BaseException:
            self.bias.copy_(old_bias)
            raise
        finally:
            self.train(old_mode)


def match_initial_biases(models, train_X, train_A, d_fit, y_scale, *, budget_callback=None):
    """Budgeted initial matching; the probability pair receives one identical bias."""
    _require(set(models) == set(NEURAL_NAMES), "All three fresh neural variants required")
    y_scale = _positive(y_scale, "y_scale")
    for model in models.values():
        _require(float(model.y_scale) == y_scale, "Bias matching target scale mismatch")
    original = {name: model.bias.detach().clone() for name, model in models.items()}
    try:
        reports = {}
        for name in ("additive_positive_head", "innovation_mean_only"):
            reports[name] = models[name].calibrate_bias(train_X, train_A, d_fit, budget_callback=budget_callback)
        first, second = models["innovation_mean_only"], models["innovation_density_mean"]
        for name, value in first.state_dict().items():
            if name != "base.mechanism.bias":
                _require(torch.equal(value, second.state_dict()[name]), "Probability pair must share exact fresh weights")
        if budget_callback is not None:
            budget_callback()
        with torch.no_grad():
            second.bias.copy_(first.bias)
        reports["innovation_density_mean"] = {**reports["innovation_mean_only"],
                                               "shared_with": "innovation_mean_only"}
        return reports
    except BaseException:
        with torch.no_grad():
            for name, model in models.items():
                model.bias.copy_(original[name])
        raise


def loss_components(outputs, Y, U, *, y_scale, u_scale, lambda_nll):
    """MSE on normalized sigma plus NLL of normalized U, never NLL/u_scale."""
    prediction = outputs["prediction"]
    Y = _nonnegative(Y, "sigma target", device=prediction.device)
    U = _nonnegative(U, "magnitude target", device=prediction.device)
    _require(Y.shape == U.shape == prediction.shape, "Aligned prediction/targets required")
    ys = _nonnegative(y_scale, "y_scale", device=prediction.device, positive=True)
    us = _nonnegative(u_scale, "u_scale", device=prediction.device, positive=True)
    _require(not isinstance(lambda_nll, bool) and lambda_nll in (0, 0.0, .1), "Frozen NLL weight must be 0 or 0.1")
    mse = ((prediction - Y) / ys).square().mean()
    scale = outputs.get("scale")
    if scale is None:
        _require(lambda_nll == 0, "Additive control cannot have a Student likelihood")
        nll = prediction.new_zeros(())
    else:
        nll = half_student_nll(U / us, scale / us).mean()
    return {"loss": mse + lambda_nll * nll, "mse_scaled": mse, "nll_u": nll}


def innovation_loss(model, X, A, Y, U, lambda_nll=None):
    weight = model.lambda_nll if lambda_nll is None else lambda_nll
    return loss_components(model.forward_components(X, A), Y, U,
                           y_scale=model.y_scale, u_scale=model.u_scale, lambda_nll=weight)["loss"]
