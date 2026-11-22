"""Deterministic, independent checks of mean-RMSE/GMM derivatives.

These checks do not simulate statistical calibration or train a model.
The formulas are implemented independently of the production methods module.
"""
from __future__ import annotations

import unittest

import numpy as np


def weighted_functional(losses, z, probabilities):
    """One system, losses[T,R], instruments[T,K], probability measure[T]."""
    mu = probabilities @ losses
    rho = np.sqrt(mu)
    theta = rho.mean()
    psi = ((losses - mu) / (2 * rho)).mean(axis=1)
    h = theta + psi
    g = (probabilities[:, None] * z * h[:, None]).sum(axis=0)
    derivative_mu = -np.einsum(
        "t,tq,ts->qs", probabilities, z, losses - mu
    ) / (4 * losses.shape[1] * mu[None, :] ** 1.5)
    influence = z * h[:, None] - g + (losses - mu) @ derivative_mu.T
    return theta, psi, g, derivative_mu, influence


class FunctionalDerivativeChecks(unittest.TestCase):
    def setUp(self):
        t = np.arange(17, dtype=float)
        self.losses = np.column_stack(
            [0.07 + (1.1 + 0.3 * np.sin(t)) ** 2,
             0.01 + (0.2 + 0.1 * np.cos(t / 2)) ** 2,
             1.3 + (2.2 + 0.4 * np.sin(t / 3)) ** 2]
        )
        self.z = np.column_stack([np.ones(len(t)), 1.2 + np.cos(t / 2)])
        self.p = np.full(len(t), 1 / len(t))

    def test_nonconstant_instrument_nuisance_derivative(self):
        losses, z, p = self.losses, self.z, self.p
        mu = p @ losses
        _, _, _, derivative, _ = weighted_functional(losses, z, p)

        def g_with_external_mu(value):
            rho = np.sqrt(value)
            h = rho.mean() + ((losses - value) / (2 * rho)).mean(axis=1)
            return (p[:, None] * z * h[:, None]).sum(axis=0)

        for s in range(losses.shape[1]):
            step = np.zeros_like(mu)
            step[s] = 1e-5 * mu[s]
            numerical = (g_with_external_mu(mu + step) - g_with_external_mu(mu - step)) / (2 * step[s])
            np.testing.assert_allclose(derivative[:, s], numerical, atol=2e-9, rtol=2e-6)
        np.testing.assert_allclose(derivative[0], 0, atol=1e-14)
        self.assertGreater(np.max(np.abs(derivative[1])), 1e-4)

    def test_full_pathwise_derivative_includes_nuisance(self):
        _, _, g, _, influence = weighted_functional(self.losses, self.z, self.p)
        for row in [0, 5, 11, 16]:
            direction = -self.p.copy()
            direction[row] += 1
            step = 1e-5
            plus = weighted_functional(self.losses, self.z, self.p + step * direction)[2]
            minus = weighted_functional(self.losses, self.z, self.p - step * direction)[2]
            np.testing.assert_allclose(influence[row], (plus - minus) / (2 * step), rtol=1e-7, atol=2e-9)
        np.testing.assert_allclose(self.p @ influence, 0, atol=2e-14)

    def test_constant_instrument_equals_exact_primary_and_scalar_if(self):
        theta, psi, g, _, influence = weighted_functional(self.losses, self.z, self.p)
        self.assertAlmostEqual(theta, g[0], places=14)
        np.testing.assert_allclose(influence[:, 0], psi, atol=2e-14)

    def test_nuisance_omission_is_detectable(self):
        theta, psi, g, _, influence = weighted_functional(self.losses, self.z, self.p)
        uncorrected = self.z * (theta + psi)[:, None] - g
        self.assertGreater(np.max(np.abs(uncorrected[:, 1] - influence[:, 1])), 0.001)

    def test_mean_roots_and_root_mean_have_opposite_order(self):
        candidate_risks = np.array([1.21, 1.21])
        rival_risks = np.array([0.01, 4])
        self.assertGreater(np.sqrt(candidate_risks).mean(), np.sqrt(rival_risks).mean())
        self.assertLess(np.sqrt(candidate_risks.mean()), np.sqrt(rival_risks.mean()))

    def test_scale_homogeneity(self):
        original = weighted_functional(self.losses, self.z, self.p)
        scaled = weighted_functional(self.losses * 9, self.z, self.p)
        for slot in [0, 1, 2, 4]:
            np.testing.assert_allclose(scaled[slot], 3 * original[slot], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scaled[3], original[3] / 3, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
