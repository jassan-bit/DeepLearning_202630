"""Synthetic tests of the port-audit detector, not certification of real data."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_port_audit as audit
from e2_temporal_training import DatedPanel, FoldScaler


class ToyPort(torch.nn.Module):
    def __init__(self, batch_dependency=False, zero_head=False, mutate=False):
        super().__init__()
        self.head = torch.nn.Linear(5, 1)
        torch.nn.init.constant_(self.head.weight, 0. if zero_head else .1)
        torch.nn.init.constant_(self.head.bias, 1.)
        self.register_buffer("counter", torch.tensor(0))
        self.batch_dependency, self.mutate = batch_dependency, mutate

    def forward(self, x):
        if self.mutate:
            self.counter.add_(1)
        features = x.mean(1)
        if self.batch_dependency:
            features = features + features.mean(0, keepdim=True)
        return torch.nn.functional.softplus(self.head(features)).squeeze(-1)


class NumericalGuardTests(unittest.TestCase):
    def test_fixed_tolerance_is_fp32_engineering_threshold(self):
        self.assertEqual(audit.ATOL, 64 * np.finfo(np.float32).eps)
        self.assertEqual(audit.RTOL, audit.ATOL)
        self.assertTrue(audit.comparison(np.array([0.]), np.array([audit.ATOL]), np)["passed"])
        self.assertFalse(audit.comparison(np.array([0.]), np.array([audit.ATOL * 1.01]), np)["passed"])

    def test_nonfinite_and_shape_changes_fail(self):
        for a, b in (([1.], [np.nan]), ([np.inf], [np.inf]), ([1.], [[1.]]), ([], [])):
            with self.subTest(a=a, b=b):
                self.assertFalse(audit.comparison(a, b, np)["passed"])


class DetectorTests(unittest.TestCase):
    def inputs(self):
        return torch.arange(6 * 7 * 5, dtype=torch.float32).reshape(6, 7, 5) / 100.

    def test_fixed_eval_model_passes_and_uses_every_condition(self):
        x = self.inputs()
        result = audit.audit_model(ToyPort().eval(), x, x.clone(), x.clone(), "head", torch, np)
        self.assertTrue(result["passed"])
        self.assertEqual([c["condition"] for c in result["conditions"]], list(audit.CONDITIONS))
        self.assertEqual(result["head_probe_shape"], [6, 5])
        self.assertEqual(result["outer_model_forward_calls"], 26)

    def test_batch_dependency_is_detected(self):
        x = self.inputs()
        result = audit.audit_model(ToyPort(batch_dependency=True).eval(), x, x.clone(), x.clone(), "head", torch, np)
        self.assertFalse(result["passed"])
        self.assertFalse(result["conditions"][0]["scaled_output"]["passed"])
        self.assertFalse(result["conditions"][0]["head_features"]["passed"])

    def test_head_probe_detects_dependency_hidden_by_constant_prediction(self):
        x = self.inputs()
        result = audit.audit_model(ToyPort(batch_dependency=True, zero_head=True).eval(), x, x.clone(), x.clone(), "head", torch, np)
        self.assertFalse(result["passed"])
        self.assertTrue(result["conditions"][0]["scaled_output"]["passed"])
        self.assertFalse(result["conditions"][0]["head_features"]["passed"])

    def test_registered_buffer_mutation_fails(self):
        x = self.inputs()
        result = audit.audit_model(ToyPort(mutate=True).eval(), x, x.clone(), x.clone(), "head", torch, np)
        self.assertFalse(result["passed"])
        self.assertFalse(result["state_unchanged"])

    def test_target_or_future_input_injection_fails_exact_input_guard(self):
        x = self.inputs()
        leaked = x.clone()
        leaked[0, 0, 0] += .25
        result = audit.audit_model(ToyPort().eval(), x, leaked, x.clone(), "head", torch, np)
        self.assertFalse(result["passed"])
        self.assertFalse(result["inputs_exactly_unchanged"])

    def test_capture_removes_hook_even_on_wrong_output_shape(self):
        class WrongShape(ToyPort):
            def forward(self, x):
                return self.head(x.mean(1))
        model = WrongShape().eval()
        with self.assertRaises(ValueError):
            audit.capture(model, self.inputs(), "head", torch)
        self.assertEqual(len(model.head._forward_pre_hooks), 0)


class DataAndProvenanceTests(unittest.TestCase):
    def test_base_spec_matches_current_eager_helper_schema(self):
        base = audit.read_json(audit.DEFAULT_BASE)
        self.assertEqual(base["runner_sha256"], audit.file_sha(audit.ROOT / "scripts/e2_corrections_frozen_inference.py"))
        self.assertIn("compatibility_correction", base)
        self.assertFalse(base["prediction_policy"]["mkldnn_enabled"])

    def test_future_perturbation_leaves_strictly_past_actual_window_and_scaler(self):
        dates = pd.date_range("2024-01-01", periods=20, tz="UTC")
        price = 100. + np.arange(20)
        features = np.stack((price, price + 2, price - 2, price + 1, price * 10), axis=1)
        panel = DatedPanel(dates, features, np.linspace(.01, .03, 20))
        train_dates = dates[7:12]
        scaler = FoldScaler.fit(panel.raw_windows(train_dates), panel.targets_at(train_dates), train_dates)
        queries = tuple(dates[[7, 10, 19]])
        original = scaler.record()
        future, labels, records = audit.transformed_perturbations(panel, queries, scaler, torch, np)
        expected = torch.from_numpy(scaler.transform(panel.raw_windows(queries)).astype(np.float32))
        self.assertTrue(torch.equal(future, expected))
        self.assertTrue(torch.equal(labels, expected))
        self.assertEqual(original, scaler.record())
        self.assertEqual([r["raw_rows_perturbed_at_or_after_query"] for r in records], [13, 10, 1])

    def test_changed_bound_file_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_bytes(b"original")
            record = audit.binding(path)
            audit.verify_binding(record)
            path.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                audit.verify_binding(record)

    def test_registered_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            audit.write_new_json(path, {"version": "first"})
            with self.assertRaises(FileExistsError):
                audit.write_new_json(path, {"version": "second"})
            self.assertEqual(audit.read_json(path), {"version": "first"})


if __name__ == "__main__":
    unittest.main()
