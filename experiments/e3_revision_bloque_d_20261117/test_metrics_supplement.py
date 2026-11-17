"""Small independent examples for descriptive formulas and support rejection."""
import importlib.util
import hashlib
import math
from pathlib import Path
import unittest
import tempfile

import numpy as np

spec = importlib.util.spec_from_file_location("metrics_supplement", Path(__file__).with_name("metrics_supplement.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DescriptiveMetricTests(unittest.TestCase):
    def test_hand_calculated_point_metrics(self):
        result = m.vector_metrics([1., 3.], [1., 2.])
        self.assertAlmostEqual(result["RMSE"], math.sqrt(.5))
        self.assertEqual(result["MAE"], .5)
        self.assertEqual(result["R2"], .5)
        self.assertAlmostEqual(result["sMAPE_percent"], 20.)

    def test_zero_zero_smape_is_zero_and_constant_target_R2_undefined(self):
        self.assertEqual(m.vector_metrics([0., 0.], [0., 0.])["sMAPE_percent"], 0.)
        self.assertIsNone(m.vector_metrics([0., 0.], [0., 0.])["R2"])
        self.assertAlmostEqual(m.vector_metrics([0., 2.], [0., 4.])["sMAPE_percent"], 100. / 3.)

    def test_mean_individual_RMSE_is_not_ensemble_RMSE(self):
        target = [1., 3.]
        first = m.vector_metrics(target, [0., 4.])
        second = m.vector_metrics(target, [2., 2.])
        summary = m.summarize([first, second])
        self.assertEqual(summary["RMSE_mean"], 1.)
        self.assertEqual(m.vector_metrics(target, [1., 3.])["RMSE"], 0.)

    def test_single_predictor_has_no_seed_SD_or_seed_bootstrap(self):
        metric = m.vector_metrics([1., 3.], [1., 2.])
        self.assertIsNone(m.summarize([metric])["RMSE_sample_SD"])
        self.assertIsNone(m.seed_bootstrap([metric["RMSE"]], np.zeros((4, 1), dtype=int)))

    def test_frozen_interval_clips_lower_endpoint_without_recalibration(self):
        result = m.frozen_interval_metrics([0., 2.], [.2, 2.], .5)
        self.assertEqual(result["coverage90_percent"], 100.)
        self.assertAlmostEqual(result["mean_interval_width90"], .85)

    def test_known_bootstrap_means_and_quantiles(self):
        indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        low, high = m.seed_bootstrap([1., 3.], indices)
        self.assertAlmostEqual(low, 1.075)
        self.assertAlmostEqual(high, 2.925)

    def test_misaligned_or_nonfinite_point_inputs_are_rejected(self):
        for y, p in (([1., 2.], [1.]), ([], []), ([1.], [float("nan")])):
            with self.assertRaises(ValueError):
                m.vector_metrics(y, p)

    def test_invalid_intervals_and_bootstrap_coordinates_are_rejected(self):
        for q in (-1., float("inf")):
            with self.assertRaises(ValueError):
                m.frozen_interval_metrics([1.], [1.], q)
        with self.assertRaises(ValueError):
            m.seed_bootstrap([1., 3.], np.array([[0, 2]]))

    def test_inherited_source_gate_rejects_same_length_hash_tampering(self):
        lib = m.validator()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_bytes(b"frozen")
            expected = {"bytes": 6, "sha256": hashlib.sha256(b"frozen").hexdigest()}
            lib.verify_source(path, expected)
            path.write_bytes(b"broken")
            with self.assertRaises(ValueError):
                lib.verify_source(path, expected)


if __name__ == "__main__":
    unittest.main()
