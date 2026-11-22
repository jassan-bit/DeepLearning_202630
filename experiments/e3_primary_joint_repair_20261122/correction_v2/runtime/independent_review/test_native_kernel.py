"""Compare production native exact-root bootstrap to independent arithmetic.

The date index fixtures are deterministic. This is a software check, not a
Monte Carlo calibration of rejection rates or a market experiment.
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


def direct_bartlett_covariance(series, lags):
    values = np.asarray(series, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / len(values)
    for lag in range(1, lags + 1):
        product = centered[lag:].T @ centered[:-lag] / len(values)
        covariance += (1 - lag / (lags + 1)) * (product + product.T)
    return covariance


class NativeKernelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("C++ compiler unavailable; native kernel not checked")
        source = Path(__file__).resolve().parents[1] / "methods" / "primary_bootstrap.cpp"
        cls.folder = tempfile.TemporaryDirectory(prefix="independent-primary-check-")
        shared = Path(cls.folder.name) / "kernel.so"
        subprocess.run([compiler, "-O3", "-std=c++11", "-shared", "-fPIC", str(source), "-o", str(shared)], check=True, capture_output=True)
        cls.library = ctypes.CDLL(str(shared))
        cls.function = cls.library.bootstrap_primary_rmse
        f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        i32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        cls.function.argtypes = [f64, i32, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, f64, f64, f64, f64, f64]
        cls.function.restype = None

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def run_kernel(self, losses, indices, lags=3, factor=0.98):
        m, r, t = losses.shape
        b = len(indices)
        observed = np.sqrt(losses.mean(axis=-1)).mean(axis=-1)
        theta = np.empty((b, m))
        epa, practical, logratio = [np.empty((b, m - 1)) for _ in range(3)]
        self.function(np.ascontiguousarray(losses), np.ascontiguousarray(indices, dtype=np.int32), b, m, r, t, lags, factor, np.ascontiguousarray(observed), theta, epa, practical, logratio)
        return observed, theta, epa, practical, logratio

    def test_exact_roots_and_fresh_hac_match_direct_reference(self):
        m, r, n = 4, 3, 24
        t = np.arange(n, dtype=float)
        losses = np.empty((m, r, n))
        for j in range(m):
            for s in range(r):
                losses[j, s] = 0.01 + ((0.4 + 0.8 * s + 0.6 * j) * (1 + 0.2 * np.sin((j + 1) * t / (s + 1) + 0.3 * s) + 0.04 * j)) ** 2
        indices = np.stack([np.arange(n), np.arange(n)[::-1], np.roll(np.arange(n), 3), np.roll(np.arange(n), 9), (np.arange(n) * 5 + 2) % n])
        observed, theta, epa, practical, logratio = self.run_kernel(losses, indices)
        for b, ix in enumerate(indices):
            sample = losses[:, :, ix]
            rho = np.sqrt(sample.mean(axis=-1))
            exact_theta = rho.mean(axis=1)
            psi = ((sample - rho[:, :, None] ** 2) / (2 * rho[:, :, None])).mean(axis=1)
            np.testing.assert_allclose(theta[b], exact_theta, rtol=2e-14, atol=2e-14)
            for j in range(1, m):
                scores = np.column_stack([psi[j] - psi[0], 0.98 * psi[j] - psi[0], psi[0] / exact_theta[0] - psi[j] / exact_theta[j]])
                variance = np.diag(direct_bartlett_covariance(scores, 3))
                numerator = np.array([(exact_theta[j] - exact_theta[0]) - (observed[j] - observed[0]), (0.98 * exact_theta[j] - exact_theta[0]) - (0.98 * observed[j] - observed[0]), np.log(exact_theta[0] / exact_theta[j]) - np.log(observed[0] / observed[j])])
                expected = numerator / np.sqrt(variance / n)
                np.testing.assert_allclose([epa[b, j - 1], practical[b, j - 1], logratio[b, j - 1]], expected, rtol=1e-9, atol=2e-11)

    def test_identical_rival_zero_variance_fails_closed(self):
        t = np.arange(24, dtype=float)
        system = np.stack([(1 + 0.1 * np.sin(t)) ** 2, (2 + 0.2 * np.cos(t / 3)) ** 2])
        losses = np.stack([system, system])
        _, _, epa, _, logratio = self.run_kernel(losses, np.arange(24)[None, :])
        self.assertTrue(math.isinf(epa[0, 0]))
        self.assertTrue(math.isinf(logratio[0, 0]))

    def test_zero_risk_fails_closed(self):
        losses = np.ones((3, 2, 24))
        losses[1, 0] = 0
        _, _, epa, practical, logratio = self.run_kernel(losses, np.arange(24)[None, :])
        self.assertTrue(np.isinf(epa).all())
        self.assertTrue(np.isinf(practical).all())
        self.assertTrue(np.isinf(logratio).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
