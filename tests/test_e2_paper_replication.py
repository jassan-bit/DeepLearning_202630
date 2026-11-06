"""Numerical data-boundary and frozen-artifact checks for the external replica."""
import copy
from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_paper_data as data
import run_e2_paper_replication as runner


class DataBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dates = pd.date_range("2016-07-01", periods=17420, freq="h")
        generator = np.random.default_rng(23)
        cls.values = generator.normal(size=(17420, 7))

    def test_train_only_scaling_is_invariant_to_future_values(self):
        initial = data.fit_scaler(self.values)
        changed = self.values.copy(); changed[8640:] += 1_000_000
        self.assertEqual(initial, data.fit_scaler(changed))
        self.assertEqual(initial["n_samples"], 8640)

    def test_population_scale_and_constant_channel(self):
        values = self.values.copy(); values[:8640, 2] = 9
        scaler = data.fit_scaler(values)
        np.testing.assert_allclose(scaler["mean"], values[:8640].mean(axis=0), atol=0, rtol=0)
        self.assertEqual(scaler["scale"][2], 1)
        np.testing.assert_allclose(np.asarray(scaler["scale"])[[0, 1, 3]],
                                   values[:8640].std(axis=0, ddof=0)[[0, 1, 3]])

    def test_original_partition_window_boundaries_and_coverage(self):
        scaler = data.fit_scaler(self.values)
        expected = {"train": (8449, 96, 8544), "val": (2785, 8640, 11424),
                    "test": (2785, 11520, 14304)}
        for split, (count, first, last) in expected.items():
            ds = data.make_split(self.values, self.dates, split, scaler=scaler)
            self.assertEqual((len(ds), ds.forecast_origins[0], ds.forecast_origins[-1]),
                             (count, first, last))
            x, y, xm, ym = ds[0]
            self.assertEqual((x.shape, y.shape, xm.shape, ym.shape),
                             ((96, 7), (96, 7), (96, 4), (96, 4)))
        self.assertEqual(2785 // 128 * 128, 2688)
        self.assertEqual(2785 - 2688, 97)

    def test_hourly_marks_match_original_timeF_definitions(self):
        dates = pd.date_range("2016-07-01 00:00", periods=24, freq="h")
        marks = data.calendar_features(dates)[[0, -1]]
        expected = [[-.5, 4/6-.5, -.5, 182/365-.5],
                    [.5, 4/6-.5, -.5, 182/365-.5]]
        np.testing.assert_allclose(marks, expected, atol=1e-7)

    def test_microsecond_resolution_keeps_hourly_calendar(self):
        original = data.calendar_features(self.dates)
        micro = data.calendar_features(self.dates.as_unit("us"))
        np.testing.assert_array_equal(original, micro)

    def test_windows_return_copies_and_do_not_change_source(self):
        ds = data.make_split(self.values, self.dates, "train")
        x, y, xm, ym = ds[0]; before = ds[0][0].copy(); x[:] = 0
        np.testing.assert_array_equal(ds[0][0], before)


class RegistrationBoundaryTests(unittest.TestCase):
    def test_plan_rejects_another_dataset_and_duplicate_seeds(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "plan.json"
            plan = {"replication": {"seeds": [2019, 2020, 2021, 2022, 2023]},
                    "dataset": {"sha256": runner.EXPECTED_DATA_SHA}}
            path.write_text(json.dumps(plan)); runner.read_design(path)
            plan["dataset"]["sha256"] = "0" * 64; path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "identity"): runner.read_design(path)
            plan["dataset"]["sha256"] = runner.EXPECTED_DATA_SHA
            plan["replication"]["seeds"] = [2019] * 5; path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "distinct"): runner.read_design(path)

    def test_source_hash_tamper_is_detected_before_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path = root / "model.py"; path.write_text("original")
            plan = {"sources": {"TimeMixer": {"files": [
                {"path": "model.py", "sha256": sha256(path.read_bytes()).hexdigest()}]}}}
            self.assertEqual(len(runner.source_receipts(plan, root)), 1)
            path.write_text("modified")
            with self.assertRaisesRegex(ValueError, "Source hash"):
                runner.source_receipts(plan, root)

    def test_registered_scaler_cannot_include_validation_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "plan.json"
            plan = {"replication": {"seeds": [2019, 2020, 2021, 2022, 2023]},
                    "dataset": {"sha256": runner.EXPECTED_DATA_SHA,
                                "global_scaler_fit_rows": [0, 11520]}}
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "boundary"):
                runner.read_design(path)

    def test_registered_inputs_are_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "registration.json"
            p.write_text(json.dumps({"immutable_inputs": {"design_sha256": "a"}}))
            runner.verify_registration(td, {"design_sha256": "a"})
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.verify_registration(td, {"design_sha256": "b"})

    def test_scoring_refuses_missing_seeds_before_loading_evaluation_data(self):
        with tempfile.TemporaryDirectory() as td:
            inputs = {"design_sha256": "a"}
            (Path(td) / "registration.json").write_text(json.dumps({"immutable_inputs": inputs}))
            args = SimpleNamespace(output_dir=Path(td), data=Path(td) / "never_read.csv")
            plan = {"replication": {"seeds": [2019, 2020, 2021, 2022, 2023]}}
            with patch.object(runner, "load_etth1", side_effect=AssertionError("test data read")):
                with self.assertRaisesRegex(ValueError, "not complete"):
                    runner.score(args, plan, inputs)

    def test_frozen_checkpoint_and_scaler_cannot_be_changed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "checkpoint.pth").write_bytes(b"weights")
            (p / "setup.json").write_text("{}")
            (p / "scaler.json").write_text("{}")
            receipt = {"design_sha256": "a", "checkpoint_sha256": runner.digest(p / "checkpoint.pth"),
                       "setup_sha256": runner.digest(p / "setup.json"),
                       "scaler_sha256": runner.digest(p / "scaler.json")}
            (p / "completed.json").write_text(json.dumps(receipt))
            runner.check_completed(p, "a")
            (p / "scaler.json").write_text('{"future_mean":1}')
            with self.assertRaisesRegex(ValueError, "metadata"):
                runner.check_completed(p, "a")

    def test_primary_preset_does_not_silently_become_short_pilot(self):
        plan = {"replication": {"training": {"epochs": 2}}}
        with self.assertRaisesRegex(ValueError, "preset"):
            runner.training_arguments(plan)

    def test_architecture_and_scheduler_cannot_drift_from_primary_preset(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            runner.model_arguments({"replication": {"model_args": {"d_model": 32}}})
        with self.assertRaisesRegex(ValueError, "scheduler"):
            runner.training_arguments({"replication": {"training": {"scheduler": "constant"}}})


if __name__ == "__main__":
    unittest.main()
