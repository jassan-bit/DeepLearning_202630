"""Synthetic verification of the proposed fixed-quadrature Student head.

This module implements a scalar/vector numerical operator, not a neural model
or training campaign. No dataset, checkpoint, validation metric or network is
read. N=128 is fixed; 64/256 are numerical comparisons only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit, gamma, roots_jacobi
from scipy.stats import t as student_t
import torch

NU, C, ORDER = 5.0, 6.0 / 49.0, 128
E_ABS_T5 = 4.0 * math.sqrt(5) / (3 * math.pi)
EXACT_L_S = math.sqrt(C) * E_ABS_T5
PDF_CONSTANT = gamma(3.0) / (math.sqrt(5 * math.pi) * gamma(2.5))
OUTPUT = Path("experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification")
RULE_NAME = "quadrature_k128.json"
RELATIVE_ERROR_CRITERION = 1e-4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class Rule:
    order: int
    a: np.ndarray
    z: np.ndarray

    @property
    def lipschitz_s(self):
        return float(math.sqrt(C) * np.dot(self.a, self.z))


@lru_cache(maxsize=3)
def quadrature_rule(order=ORDER):
    require(type(order) is int and order in (64, 128, 256), "Only fixed 128 and numerical comparison orders 64/256 are supported")
    x, w = roots_jacobi(order, 1.0, -0.5)
    v = (x + 1) / 2
    a = w * np.sqrt(1 - v)
    a = a / a.sum()
    z = np.sqrt(NU * v / (1 - v))
    a.setflags(write=False)
    z.setflags(write=False)
    return Rule(order, a, z)


def _nonnegative(value, name, *, positive=False):
    value = np.asarray(value, dtype=np.float64)
    require(np.isfinite(value).all() and (value > 0 if positive else value >= 0).all(),
            name + (" must be finite and positive" if positive else " must be finite and nonnegative"))
    return value


def head_mean(A, s, rule=None):
    """g_N=Σa_i sqrt(A+c*s²*z_i²), with positive a_i summing to one."""
    rule = quadrature_rule() if rule is None else rule
    A, s = np.broadcast_arrays(_nonnegative(A, "A"), _nonnegative(s, "s"))
    return np.sum(rule.a * np.sqrt(A[..., None] + C * s[..., None] ** 2 * rule.z ** 2), axis=-1)


def half_student_density(u, s):
    u, s = np.broadcast_arrays(_nonnegative(u, "u"), _nonnegative(s, "s", positive=True))
    return 2 * PDF_CONSTANT / s * (1 + (u / s) ** 2 / NU) ** -3


def half_student_nll(u, s):
    """Auxiliary NLL on U; changing units adds a scale-independent constant."""
    u, s = np.broadcast_arrays(_nonnegative(u, "u"), _nonnegative(s, "s", positive=True))
    with np.errstate(divide="ignore"):
        log_odds = 2 * (np.log(u) - np.log(s)) - math.log(NU)
    return np.log(s) - math.log(2 * PDF_CONSTANT) + 3 * np.logaddexp(0, log_odds)


def auxiliary_score_log_s(u, s):
    u, s = np.broadcast_arrays(_nonnegative(u, "u"), _nonnegative(s, "s", positive=True))
    with np.errstate(divide="ignore"):
        log_odds = 2 * (np.log(u) - np.log(s)) - math.log(NU)
    return 1 - 6 * expit(log_odds)


def sigma_density(y, A, s):
    """q_U(u)*y/(c*u), with its distinct A=0 and A>0 endpoint limits."""
    y, A, s = np.broadcast_arrays(_nonnegative(y, "y"), _nonnegative(A, "A"), _nonnegative(s, "s", positive=True))
    lower = np.sqrt(A)
    result = np.zeros(y.shape, dtype=np.float64)
    inside = y > lower
    u = np.sqrt((y[inside] - lower[inside]) * (y[inside] + lower[inside]) / C)
    result[inside] = half_student_density(u, s[inside]) * y[inside] / (C * u)
    result[(y == lower) & (A > 0)] = np.inf
    zero_endpoint = (y == 0) & (A == 0)
    result[zero_endpoint] = 2 * PDF_CONSTANT / (math.sqrt(C) * s[zero_endpoint])
    return result


def predictive_second_moment(A, s):
    return _nonnegative(A, "A") + (10.0 / 49.0) * _nonnegative(s, "s") ** 2


def central_interval(A, s, coverage=0.9):
    require(0 < coverage < 1, "Coverage must lie in (0,1)")
    A, s = np.broadcast_arrays(_nonnegative(A, "A"), _nonnegative(s, "s", positive=True))
    tail = (1 - coverage) / 2
    low_z, high_z = student_t.ppf((1 + tail) / 2, NU), student_t.ppf((2 - tail) / 2, NU)
    return np.sqrt(A + C * s ** 2 * low_z ** 2), np.sqrt(A + C * s ** 2 * high_z ** 2)


def rule_document(rule):
    return {"version": "e3-student-head-quadrature-v1", "N": rule.order, "nu": NU, "c": C,
            "jacobi_alpha": 1.0, "jacobi_beta": -0.5, "normalization": "a_i=w_i*sqrt(1-v_i)/sum_j w_j*sqrt(1-v_j)",
            "z_definition": "sqrt(nu*v_i/(1-v_i)); v_i=(x_i+1)/2",
            "a_i": rule.a.tolist(), "z_i": rule.z.tolist(), "L_s_discrete": rule.lipschitz_s}


def reference_mean(rho):
    scale = max(1.0, rho)
    result, error = quad(lambda t: math.hypot(1 / scale, rho * t / scale)
                         * float(half_student_density(t, 1)), 0, math.inf,
                         epsabs=2e-13, epsrel=2e-13, limit=500)
    return result * scale, error * scale


def density_normalization(A, s):
    # Independent endpoint substitution z=sqrt(y-sqrt(A)), rescaled to avoid
    # missing a narrow peak. No cancellation of y²-A is used at the boundary.
    lower = math.sqrt(A)
    if A == 0:
        return quad(lambda w: float(sigma_density(math.sqrt(C) * s * w, A, s)) * math.sqrt(C) * s,
                    0, math.inf, epsabs=2e-12, epsrel=2e-12, limit=500)
    scale_z = math.sqrt(C * s * s / (math.hypot(lower, math.sqrt(C) * s) + lower))
    def integrand(w):
        z = scale_z * w
        u = z * math.sqrt((2 * lower + z * z) / C)
        return scale_z * 2 * float(half_student_density(u, s)) * (lower + z * z) / (math.sqrt(C) * math.sqrt(2 * lower + z * z))
    return quad(integrand, 0, math.inf, epsabs=2e-12, epsrel=2e-12, limit=500)


def verify_numerically():
    rhos = np.logspace(-6, 6, 241)
    references = [reference_mean(float(rho)) for rho in rhos]
    reference = np.array([x[0] for x in references])
    report = {"version": "e3-student-head-synthetic-verification-v1",
        "scope": "Synthetic operator/quadrature checks only; no market data, neural model, training, HPO or R1/R3 evidence",
        "fixed_order": ORDER, "nu": NU, "c": C, "E_abs_T5": E_ABS_T5,
        "L_s_exact_mean": EXACT_L_S, "E_T5_squared": NU / (NU - 2),
        "beta_ratio": 32 / (9 * math.pi), "beta_E_sqrt_one_minus_V": 9 * math.pi / 32,
        "criterion_relative_error": RELATIVE_ERROR_CRITERION,
        "grid": {"rho_min": 1e-6, "rho_max": 1e6, "rho_count": 241, "gradient_count_per_order": 81},
        "reference_relative_quad_error_estimate_max": float(max(e / v for v, e in references)),
        "orders": {}, "density": [], "market_data_read": False, "neural_model_implemented": False,
        "recorded_environment": {"python": platform.python_version(), "numpy": np.__version__,
                                 "scipy": scipy.__version__, "torch": torch.__version__}}
    for n in (64, 128, 256):
        rule = quadrature_rule(n)
        values = head_mean(1, rhos / math.sqrt(C), rule)
        errors = np.abs(values / reference - 1)
        scales, lower_violations, upper_violations = [], 0, 0
        for A in (0.0, 1e-6, 1.0, 100.0):
            for s in (0.0, 1e-8, 1.0, 1e8):
                value = float(head_mean(A, s, rule))
                lower_violations += int(value < math.sqrt(A) - 2e-14 * max(1, value))
                upper_violations += int(value > math.sqrt(A) + rule.lipschitz_s * s + 2e-14 * max(1, value))
                for k in (1e-6, 1e-3, 1.0, 1e3, 1e6):
                    expected = k * value
                    actual = float(head_mean(k * k * A, k * s, rule))
                    scales.append(abs(actual / expected - 1) if expected else abs(actual))
        a, z = torch.tensor(rule.a), torch.tensor(rule.z)
        ds_ratios, da_ratios = [], []
        for A in np.logspace(-12, 12, 9):
            for s in np.logspace(-12, 12, 9):
                at, st = torch.tensor(float(A), requires_grad=True, dtype=torch.float64), torch.tensor(float(s), requires_grad=True, dtype=torch.float64)
                value = (a * torch.sqrt(at + C * st.square() * z.square())).sum()
                da, ds = torch.autograd.grad(value, (at, st))
                require(torch.isfinite(da).item() and torch.isfinite(ds).item(), "Nonfinite synthetic gradient")
                ds_ratios.append(float(ds) / rule.lipschitz_s)
                da_ratios.append(float(da) * 2 * math.sqrt(A))
        item = {"max_relative_error": float(errors.max()), "A0_relative_error": abs(rule.lipschitz_s / EXACT_L_S - 1),
                "positive_weights": bool((rule.a > 0).all()), "positive_nodes": bool((rule.z > 0).all()),
                "weight_sum": float(rule.a.sum()), "L_s_discrete": rule.lipschitz_s,
                "scale_equivariance_error_max": max(scales), "support_violation_count": lower_violations,
                "upper_envelope_violation_count": upper_violations, "finite_gradient_count": len(ds_ratios),
                "derivative_s_bound_ratio_max": max(ds_ratios), "derivative_A_bound_ratio_max_A_positive": max(da_ratios),
                "g_A1_s0": float(head_mean(1, 0, rule)), "g_A0_s0": float(head_mean(0, 0, rule))}
        require(item["positive_weights"] and item["positive_nodes"] and abs(item["weight_sum"] - 1) < 1e-12, "Invalid quadrature weights")
        require(not lower_violations and not upper_violations and max(scales) < 1e-12, "Support or scale property failed")
        require(max(ds_ratios) <= 1 + 1e-12 and max(da_ratios) <= 1 + 1e-12, "Gradient bound failed")
        report["orders"][str(n)] = item
    chosen = report["orders"][str(ORDER)]
    require(max(chosen["max_relative_error"], chosen["A0_relative_error"]) <= RELATIVE_ERROR_CRITERION,
            "Fixed N128 quadrature fails the numerical error criterion")
    for A in (0.0, 1.0, 1e6):
        for s in (1e-6, 1e-3, 1.0, 1e3, 1e6):
            value, error = density_normalization(A, s)
            require(abs(value - 1) < 1e-10, "Conditional density normalization failed")
            report["density"].append({"A": A, "s": s, "integral": value, "quad_absolute_error_estimate": error})
    ratio = np.r_[0.0, np.logspace(-12, 12, 97)]
    scores = auxiliary_score_log_s(ratio, 1)
    require(scores.min() >= -5 and scores.max() <= 1, "Auxiliary log-scale score bound failed")
    rule = quadrature_rule()
    low, high = central_interval(1, 1)
    u_low, u_high = math.sqrt((float(low) ** 2 - 1) / C), math.sqrt((float(high) ** 2 - 1) / C)
    model_coverage = 2 * (student_t.cdf(u_high, NU) - student_t.cdf(u_low, NU))
    misspecified_coverage = 2 * (student_t.cdf(u_high / 2, NU) - student_t.cdf(u_low / 2, NU))
    report["auxiliary_and_magnitude"] = {"score_variable": "log(s)", "score_min": float(scores.min()),
        "score_max": float(scores.max()), "score_count": len(scores),
        "MSE_gradient_example": {"A": 0, "y": 0, "y_scale": 1, "s": [1, 10, 100],
            "d_MSE_d_log_s": [2 * rule.lipschitz_s ** 2 * s ** 2 for s in (1, 10, 100)]},
        "predictive_second_moment": "A+(10/49)*s^2",
        "nominal_coverage_if_assumed_law_is_true": float(model_coverage),
        "coverage_if_true_scale_is_twice_assumed_scale": float(misspecified_coverage),
        "interpretation": "Only the auxiliary log-scale score is bounded. The point MSE, encoder gradients and empirical calibration have no such guarantee. No additional calibration loss is defined."}
    report["boundaries"] = {"s0": "g(A,0)=sqrt(A); a point mass, not an ordinary density",
        "A0": "g_N(0,s)=L_N*s; approximate the exact sqrt(c)*E|T5|*s",
        "origin": "g(0,0)=0 by continuity; no joint differentiability claim",
        "density_endpoint": "A=0 has finite density q(0)/sqrt(c); A>0 has an integrable singularity",
        "A_derivative": "No global Lipschitz bound in A: the exact derivative diverges at A=0,s>0"}
    return report


def verify_saved(directory, recomputed):
    directory = Path(directory)
    payload = (directory / RULE_NAME).read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    require((directory / (RULE_NAME + ".sha256")).read_text().strip() == actual_hash, "Published rule byte hash mismatch")
    stored = json.loads(payload)
    expected = rule_document(quadrature_rule())
    for key in set(expected) - {"a_i", "z_i", "L_s_discrete"}:
        require(stored.get(key) == expected[key], "Published rule parameter mismatch: " + key)
    a, z = np.asarray(stored["a_i"]), np.asarray(stored["z_i"])
    require(a.shape == z.shape == (ORDER,) and np.isfinite(a).all() and np.isfinite(z).all()
            and (a > 0).all() and (z > 0).all() and abs(a.sum() - 1) < 5e-13, "Published rule properties failed")
    require(np.allclose(a, expected["a_i"], rtol=2e-10, atol=2e-13)
            and np.allclose(z, expected["z_i"], rtol=2e-10, atol=2e-13), "Published nodes/weights differ numerically")
    require(math.isclose(stored["L_s_discrete"], math.sqrt(C) * float(np.dot(a, z)), rel_tol=1e-12), "Published Lipschitz constant mismatch")
    saved = json.loads((directory / "summary.json").read_text())
    require(set(saved) == set(recomputed) | {"rule_sha256"}, "Saved verification schema differs")
    require(saved["rule_sha256"] == actual_hash, "Summary does not bind to the published rule")
    for key in ("version", "scope", "fixed_order", "nu", "c", "criterion_relative_error", "grid", "market_data_read", "neural_model_implemented", "boundaries"):
        require(saved[key] == recomputed[key], "Saved verification metadata differs: " + key)
    for key in ("E_abs_T5", "L_s_exact_mean", "E_T5_squared", "beta_ratio", "beta_E_sqrt_one_minus_V"):
        require(math.isclose(saved[key], recomputed[key], abs_tol=1e-14, rel_tol=1e-12), "Saved theoretical constant differs: " + key)
    for n in ("64", "128", "256"):
        for key in ("positive_weights", "positive_nodes", "support_violation_count", "upper_envelope_violation_count", "finite_gradient_count"):
            require(saved["orders"][n][key] == recomputed["orders"][n][key], "Saved property/count differs")
        for key in ("max_relative_error", "A0_relative_error", "weight_sum", "L_s_discrete", "scale_equivariance_error_max", "derivative_s_bound_ratio_max", "derivative_A_bound_ratio_max_A_positive", "g_A1_s0", "g_A0_s0"):
            require(math.isclose(saved["orders"][n][key], recomputed["orders"][n][key], abs_tol=2e-12, rel_tol=1e-9), "Saved verification numeric value differs: " + key)
    # Runtime metadata is documentary, not a requirement for bit-identical hosts.
    require(len(saved["density"]) == len(recomputed["density"]) == 15, "Density check count differs")
    for old, new in zip(saved["density"], recomputed["density"]):
        require(old["A"] == new["A"] and old["s"] == new["s"] and abs(old["integral"] - 1) < 1e-10, "Saved density normalization differs")
    def compare_block(old, new):
        require(type(old) is type(new), "Saved auxiliary verification type differs")
        if isinstance(new, dict):
            require(old.keys() == new.keys(), "Saved auxiliary verification keys differ")
            for key in new:
                compare_block(old[key], new[key])
        elif isinstance(new, list):
            require(len(old) == len(new), "Saved auxiliary verification length differs")
            for a, b in zip(old, new):
                compare_block(a, b)
        elif isinstance(new, float):
            require(math.isfinite(old) and math.isclose(old, new, abs_tol=2e-12, rel_tol=1e-9), "Saved auxiliary numeric verification differs")
        else:
            require(old == new, "Saved auxiliary verification metadata differs")
    compare_block(saved["auxiliary_and_magnitude"], recomputed["auxiliary_and_magnitude"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = verify_numerically()
    directory = args.root / OUTPUT
    if args.check:
        verify_saved(directory, report)
    else:
        directory.mkdir(parents=True, exist_ok=True)
        payload = canonical_bytes(rule_document(quadrature_rule()))
        rule_hash = hashlib.sha256(payload).hexdigest()
        (directory / RULE_NAME).write_bytes(payload)
        (directory / (RULE_NAME + ".sha256")).write_text(rule_hash + "\n")
        report["rule_sha256"] = rule_hash
        (directory / "summary.json").write_bytes(canonical_bytes(report))
    print("OK: fixed N128 synthetic operator verified; no market data or neural model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
