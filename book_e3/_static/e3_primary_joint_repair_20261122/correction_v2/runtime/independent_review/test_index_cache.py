"""Check memoized indices preserve uncached draws and remain unmodified."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import primary_rmse as production
from test_exact_compression import fixture


class IndexCacheChecks(unittest.TestCase):
    def test_cached_indices_match_uncached_pcg_bytes_and_reject_assignment(self):
        for settings in [(29, 31, 4, 5423), (29, 31, 4, 5424), (47, 53, 7, 984)] :
            n, reps, length, seed = settings
            cached, identity = production._cached_stationary_indices(*settings)
            uncached = production.stationary_indices(n, reps, length, np.random.default_rng(seed))
            self.assertEqual(cached.tobytes(), uncached.tobytes())
            self.assertEqual(identity, hashlib.sha256(uncached.tobytes()).hexdigest())
            self.assertFalse(cached.flags.writeable)
            with self.assertRaises(ValueError):
                cached[0, 0] = 0
            same, repeated_identity = production._cached_stationary_indices(*settings)
            self.assertIs(cached, same)
            self.assertEqual(identity, repeated_identity)

    def test_bootstrap_does_not_mutate_cached_date_indices(self):
        losses = fixture()
        settings = dict(reps=31, block_length=4, lags=3, margin=0.02, seed=5423)
        indices, expected = production._cached_stationary_indices(losses.shape[-1], settings["reps"], settings["block_length"], settings["seed"])
        for backend in ["native", "numpy"]:
            result = production.bootstrap_functional(losses, backend=backend, **settings)
            self.assertEqual(expected, hashlib.sha256(indices.tobytes()).hexdigest())
            self.assertEqual(expected, result["bootstrap_indices_sha256"])
            self.assertFalse(indices.flags.writeable)

    def test_cached_and_explicit_uncached_indices_give_identical_results(self):
        losses = fixture()
        settings = dict(reps=31, block_length=4, lags=3, margin=0.02, seed=5423)
        indices = production.stationary_indices(losses.shape[-1], settings["reps"], settings["block_length"], np.random.default_rng(settings["seed"]))
        for backend in ["native", "numpy"]:
            cached = production.bootstrap_functional(losses, backend=backend, **settings)
            explicit = production.bootstrap_functional(losses, backend=backend, indices=indices, **settings)
            self.assertEqual(cached["bootstrap_indices_sha256"], explicit["bootstrap_indices_sha256"])
            for key in ["theta", "epa_t", "practical_t", "log_ratio_t"]:
                np.testing.assert_array_equal(cached[key], explicit[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
