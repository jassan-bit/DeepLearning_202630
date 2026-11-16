"""Synthetic properties of the proposed scalar operator, without market data."""
from pathlib import Path
import math
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t as student_t
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import e3_verify_innovation_design as design


def test_positive_quadrature_preserves_support_scale_equivariance_and_scale_lipschitz_bound():
    rule = design.quadrature_rule()
    assert rule.order == 128 and (rule.a > 0).all() and (rule.z > 0).all()
    assert rule.a.sum() == pytest.approx(1, abs=2e-15)
    rng = np.random.default_rng(517)
    A = 10 ** rng.uniform(-8, 8, 32)
    first, second = 10 ** rng.uniform(-8, 8, 32), 10 ** rng.uniform(-8, 8, 32)
    a, b = design.head_mean(A, first), design.head_mean(A, second)
    assert np.all(a >= np.sqrt(A) - 2e-14 * np.maximum(1, a))
    assert np.all(np.abs(a - b) <= rule.lipschitz_s * np.abs(first - second) + 2e-14 * np.maximum(1, a))
    for scale in (1e-6, .01, 100, 1e6):
        np.testing.assert_allclose(design.head_mean(scale ** 2 * A, scale * first), scale * a,
                                   rtol=4e-15, atol=2e-15)


def test_density_jacobian_matches_independent_student_U_and_normalizes_at_distinct_boundaries():
    for A, s in ((0.0, .3), (1.0, .3), (4.0, 2.0)):
        u = s * np.logspace(-3, 3, 31)
        y = np.sqrt(A + design.C * u ** 2)
        transformed_back = design.sigma_density(y, A, s) * design.C * u / y
        expected_q = 2 * student_t.pdf(u / s, df=5) / s
        np.testing.assert_allclose(transformed_back, expected_q, rtol=2e-8, atol=2e-11)
        integral, _ = design.density_normalization(A, s)
        assert integral == pytest.approx(1, abs=2e-11)
    assert np.isfinite(design.sigma_density(0, 0, 1))
    assert np.isinf(design.sigma_density(1, 1, 1))
    assert design.sigma_density(.5, 1, 1) == 0
    with pytest.raises(ValueError):
        design.sigma_density(1, 1, 0)  # s=0 is a point mass, not a density.


def test_only_auxiliary_logscale_score_is_bounded_while_mse_magnitude_and_calibration_are_not_guaranteed():
    ratios = torch.tensor(np.r_[0.0, np.logspace(-6, 6, 49)], dtype=torch.float64)
    log_scale = torch.zeros_like(ratios, requires_grad=True)
    # Independent differentiation of the half-Student5 NLL, dropping its constant.
    loss = log_scale + 3 * torch.log1p((ratios * torch.exp(-log_scale)) ** 2 / 5)
    gradient, = torch.autograd.grad(loss.sum(), log_scale)
    assert gradient.min() >= -5 and gradient.max() <= 1
    np.testing.assert_allclose(gradient.numpy(), design.auxiliary_score_log_s(ratios.numpy(), 1), atol=2e-15, rtol=2e-15)
    rule = design.quadrature_rule()
    a, z = torch.tensor(rule.a), torch.tensor(rule.z)
    mse_gradients = []
    for s in (1.0, 10.0, 100.0):
        log_s = torch.tensor(math.log(s), dtype=torch.float64, requires_grad=True)
        mean = (a * torch.sqrt(design.C * torch.exp(2 * log_s) * z ** 2)).sum()
        derivative, = torch.autograd.grad(mean ** 2, log_s)  # A=y=0, y_scale=1.
        mse_gradients.append(float(derivative))
    assert mse_gradients[1] / mse_gradients[0] == pytest.approx(100)
    assert mse_gradients[2] > 1000  # The score bound does not bound total MSE gradients.
    moment, _ = quad(lambda u: u ** 2 * 2 * student_t.pdf(u / 3, 5) / 3, 0, np.inf)
    assert design.predictive_second_moment(1, 3) == pytest.approx(1 + design.C * moment, rel=1e-12)
    low, high = design.central_interval(1, 1, coverage=.9)
    u_low, u_high = math.sqrt((float(low) ** 2 - 1) / design.C), math.sqrt((float(high) ** 2 - 1) / design.C)
    nominal = 2 * (student_t.cdf(u_high, 5) - student_t.cdf(u_low, 5))
    misspecified = 2 * (student_t.cdf(u_high / 2, 5) - student_t.cdf(u_low / 2, 5))
    assert nominal == pytest.approx(.9, abs=1e-13)
    assert misspecified < .85  # Nominal law coverage is not empirical calibration.


def test_zero_scale_limit_and_zero_A_linear_limit_are_explicit():
    A = np.array([0.0, .01, 1.0, 9.0])
    np.testing.assert_allclose(design.head_mean(A, 0), np.sqrt(A), atol=1e-14, rtol=2e-15)
    small = design.head_mean(A, 1e-8)
    assert np.all(np.abs(small - np.sqrt(A)) <= design.quadrature_rule().lipschitz_s * 1e-8 + 1e-14)
    scales = np.array([0.0, 1e-6, 1.0, 1e6])
    np.testing.assert_allclose(design.head_mean(0, scales), design.quadrature_rule().lipschitz_s * scales, rtol=2e-15, atol=1e-15)
    assert design.head_mean(0, 0) == 0
