"""Synthetic information-boundary and frozen-trajectory tests; no market fits."""
from __future__ import annotations

import copy
import json
import os
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
import e2_temporal_models as models
import run_e2_remediation_pilot as runner
from dl_data_v2 import fit_relative_preprocessing, split_for_origin

PRIMARY = Path(os.environ.get("E2_TIMEMIXER_SOURCE", "/workspace/e2_paper_guide_20261106/sources/TimeMixer-e24610583b36fdd8c76cc17a8df4e65759a5f460"))
CONTROL = Path(os.environ.get("E2_CONTROL_SOURCE", "/workspace/e2_paper_guide_20261106/sources/Time-Series-Library-4e938a1767106324dd753b2a44832bf870a0252e"))


def synthetic_files(root):
    """Write full calendar fixtures with the inherited incomplete-day position."""
    rng = np.random.default_rng(43)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .02, len(runner.CALENDAR))))
    opening = close * np.exp(rng.normal(0, .003, len(close)))
    values = np.column_stack((opening, np.maximum(opening, close) * 1.01,
                              np.minimum(opening, close) * .99, close,
                              rng.lognormal(8, .5, len(close))))
    values[runner.CALENDAR.get_loc(pd.Timestamp("2023-03-24", tz="UTC"))] = np.nan
    labels = pd.Series(np.log(values[:, 3]), index=runner.CALENDAR).diff().rolling(7, min_periods=7).std(ddof=0)
    market = pd.DataFrame(values, columns=runner.FEATURES)
    market.insert(0, "market_day_utc", runner.CALENDAR.strftime("%Y-%m-%d"))
    target = pd.DataFrame({"market_day_utc": runner.CALENDAR.strftime("%Y-%m-%d"), "sigma_7": labels.to_numpy()})
    for name, frame in zip(runner.SOURCE_PATHS, (market, target)):
        path = Path(root) / name; path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return market, target


class InformationBoundaryTests(unittest.TestCase):
    def test_prefix_never_parses_later_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            market, target = synthetic_files(temp)
            original = pd.read_csv
            with patch.object(runner.pd, "read_csv", wraps=original) as spy:
                first = runner.load_development_prefix(temp, "2024-10-24")
            n = int((pd.Timestamp("2024-10-24", tz="UTC") - runner.CALENDAR[0]).days) + 1
            self.assertEqual([call.kwargs["nrows"] for call in spy.call_args_list], [n, n])
            self.assertTrue(np.isnan(first["targets"][n:]).all())
            self.assertTrue(np.isnan(first["features"][n:]).all())
            # Poison all future fields with nonnumeric text. A whole-file parse
            # or evaluation-statistic computation here would now fail.
            for frame in (market, target):
                for col in frame.columns[1:]:
                    frame[col] = frame[col].astype(object)
                    frame.loc[n:, col] = "UNREAD_FUTURE"
            market.to_csv(Path(temp) / runner.SOURCE_PATHS[0], index=False)
            target.to_csv(Path(temp) / runner.SOURCE_PATHS[1], index=False)
            second = runner.load_development_prefix(temp, "2024-10-24")
            np.testing.assert_array_equal(first["targets"], second["targets"])
            np.testing.assert_array_equal(first["features"], second["features"])

    def test_gap_day_is_not_compressed_and_windows_are_past_only(self):
        with tempfile.TemporaryDirectory() as temp:
            synthetic_files(temp); data = runner.load_development_prefix(temp, "2024-10-24")
            self.assertEqual(len(data["calendar"]), 731)
            missing = runner.CALENDAR.get_loc(pd.Timestamp("2023-03-24", tz="UTC"))
            self.assertFalse(data["eligible_by_history"][7][missing:missing + 8].any())
            self.assertTrue(data["eligible_by_history"][7][missing + 8])
            t = runner.CALENDAR.get_loc(pd.Timestamp("2024-09-01", tz="UTC"))
            np.testing.assert_array_equal(data["windows_by_history"][7][t], data["features"][t - 7:t])
            self.assertNotEqual(data["windows_by_history"][7][t, -1, 3], data["features"][t, 3])

    def test_original_split_and_train_only_scaling_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            synthetic_files(temp); data = runner.load_development_prefix(temp, "2024-10-24")
            split = split_for_origin(data, 7, "2024-11-01", inner_days=90)
            dates = data["calendar"]
            self.assertEqual(dates[split["train_indices"][-1]].strftime("%Y-%m-%d"), "2024-10-24")
            self.assertEqual(dates[split["inner_train_indices"][-1]].strftime("%Y-%m-%d"), "2024-07-26")
            self.assertEqual(dates[split["inner_validation_indices"][0]].strftime("%Y-%m-%d"), "2024-08-03")
            self.assertEqual(len(split["inner_validation_indices"]), 83)
            inner = split["inner_train_indices"]
            original = fit_relative_preprocessing(data, 7, inner)
            changed = copy.deepcopy(data)
            changed["targets"][split["inner_validation_indices"]] *= 1000
            changed["windows_by_history"][7][split["inner_validation_indices"]] *= 1000
            self.assertEqual(original, fit_relative_preprocessing(changed, 7, inner))

    def test_outside_calendar_and_partial_day_are_rejected_before_read(self):
        for day in ("2024-12-31", "2025-01-01", "2026-09-01", "2024-11-01 12:00", "2022-12-31"):
            with self.subTest(day=day), patch.object(runner.pd, "read_csv") as read:
                with self.assertRaises(ValueError): runner.load_development_prefix("/nonexistent", day)
                read.assert_not_called()


class RegistryTests(unittest.TestCase):
    def test_contract_and_seed_mutations_are_refused(self):
        original = runner.read_plan(runner.PLAN)
        changes = [lambda p: p["seeds"].__setitem__(0, 6101),
                   lambda p: p["input_contract"].__setitem__("D", 14),
                   lambda p: p.__setitem__("margin_absolute", .0001),
                   lambda p: p["training"].__setitem__("learning_rate", .001),
                   lambda p: p["fresh_GRU_reference"].__setitem__("width", 32),
                   lambda p: p["calendar"].__setitem__("max_decoded_outcome", "2026-10-30"),
                   lambda p: p["development_sources"].__setitem__("results/test.csv", "0" * 64)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            for change in changes:
                plan = copy.deepcopy(original); change(plan); runner.write_json(path, plan)
                with self.assertRaises(ValueError): runner.read_plan(path)

    def test_fixed_counts_and_same_seed_pairing_do_not_invent_independent_series(self):
        plan = runner.read_plan(runner.PLAN)
        self.assertEqual(len(runner.scheduled_fits(plan, "pilot")), 20)
        self.assertEqual(len(runner.scheduled_fits(plan, "oof")), 200)
        self.assertEqual(set(seed for _, _, seed in runner.scheduled_fits(plan, "oof")), set(range(660101, 660106)))
        self.assertEqual(plan["budget"]["P0_optimizer_fits"], 40)
        self.assertEqual(plan["budget"]["optional_OOF_optimizer_fits"], 400)
        self.assertEqual(plan["OOF"]["independent_series_count"], 1)

    def test_registration_is_mandatory_and_hash_changes_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output_dir=Path(temp))
            with self.assertRaisesRegex(ValueError, "Register"): runner.registration(args, {"hash": "original"})
            runner.write_json(Path(temp) / "registration.json", {"immutable_inputs": {"hash": "original"}})
            with self.assertRaisesRegex(ValueError, "changed"): runner.registration(args, {"hash": "altered"})
            self.assertEqual(runner.registration(args, {"hash": "original"})["immutable_inputs"], {"hash": "original"})

    def test_all_checkpoints_must_close_before_any_external_label_decode(self):
        plan = runner.read_plan(runner.PLAN)
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output_dir=Path(temp), phase="score")
            with patch.object(runner, "registration"), patch.object(runner, "load_development_prefix") as load:
                with self.assertRaises(FileNotFoundError): runner.score(args, plan, {"plan_sha256": "x"})
                load.assert_not_called()

    def test_overwriting_incomplete_fits_and_repo_outputs_is_refused(self):
        with self.assertRaises(ValueError): runner.private_output(ROOT / "results/private")
        plan = runner.read_plan(runner.PLAN)
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(output_dir=Path(temp), phase="pilot", model="GRU", seed=660101, origin=None)
            directory = runner.directory_for(temp, "pilot", "GRU", 660101, runner.PILOT_ORIGIN)
            directory.mkdir(parents=True); (directory / "started.json").write_text("{}")
            with patch.object(runner, "registration"), patch.object(runner, "load_development_prefix") as load:
                with self.assertRaises(FileNotFoundError): runner.fit_one(args, plan, {"plan_sha256": "x"})
                load.assert_not_called()

    def test_late_registration_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); (out / "started.json").write_text("{}")
            args = SimpleNamespace(output_dir=out)
            with self.assertRaisesRegex(ValueError, "retrospectively"):
                runner.register(args, {}, {})

    def test_closed_trajectory_receipt_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            runner.write_json(path / "completion.json", {"plan_sha256": "p", "model": "GRU",
                "evaluation_predictions_computed": False, "optimizer_fits": 2, "artifact_sha256": {}})
            with self.assertRaisesRegex(ValueError, "another model"):
                runner.completed(path, {"plan_sha256": "p"}, {"model": "TimeMixer"})
            with self.assertRaisesRegex(ValueError, "manifest"):
                runner.completed(path, {"plan_sha256": "p"}, {"model": "GRU"})


@unittest.skipUnless(PRIMARY.is_dir() and CONTROL.is_dir(), "Fixed author source trees are separately acquired")
class AuthorAdapterTests(unittest.TestCase):
    def setUp(self):
        runner.configure(1337, 1)

    @staticmethod
    def model(name): return models.build_model(name, PRIMARY if name == "TimeMixer" else CONTROL)

    def test_scalar_positive_readout_has_no_covariate_denormalization(self):
        for name in models.TOP_MODELS:
            with self.subTest(model=name):
                model = self.model(name); model.eval()
                latent = model.latent_features(torch.randn(3, 7, 5))
                self.assertEqual(tuple(latent.shape), (3, 5, 16))
                self.assertEqual(model.head.in_features, 90)
                output = model(torch.randn(3, 7, 5))
                self.assertEqual(tuple(output.shape), (3,)); self.assertTrue(torch.isfinite(output).all())
                self.assertTrue((output > 0).all())
                self.assertFalse(models.architecture_spec(model)["original_covariate_projection_and_denormalization_used"])
                old = "projection_layer" if name == "TimeMixer" else "projection" if name == "iTransformer" else "head"
                self.assertFalse(hasattr(model.backbone, old))

    def test_eval_predictions_are_independent_of_other_batch_origins(self):
        x = torch.randn(8, 7, 5)
        for name in models.MODELS:
            with self.subTest(model=name):
                model = self.model(name); model.eval()
                with torch.no_grad():
                    a = model(x[:1]); b = model(x)[:1]
                    permuted = model(x.flip(0))[-1:]
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(a, permuted, atol=1e-6, rtol=1e-5)

    def test_constant_input_backward_is_finite_and_all_parameters_are_active(self):
        for name in models.MODELS:
            with self.subTest(model=name):
                model = self.model(name); model.train()
                x = torch.ones(3, 7, 5, requires_grad=True)
                model(x).sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                count = models.count_parameters(model)
                self.assertEqual(count["active_in_last_backward"], count["registered"])

    def test_head_derived_moments_preserve_same_window_scale_information(self):
        x = torch.randn(2, 7, 5)
        for name in models.TOP_MODELS:
            with self.subTest(model=name):
                model = self.model(name); model.eval()
                # Read only the first window mean. This proves that amplitude
                # survives author instance normalization without new data.
                with torch.no_grad():
                    model.head.weight.zero_(); model.head.bias.zero_(); model.head.weight[0, 80] = 1.
                    original = model(x); shifted = model(x + 1)
                torch.testing.assert_close(shifted, torch.nn.functional.softplus(x.mean(1)[:, 0] + 1))
                self.assertTrue((shifted > original).all())

    def test_nonfinite_and_wrong_information_shape_are_rejected(self):
        model = self.model("TimeMixer")
        for x in (torch.ones(2, 14, 5), torch.ones(2, 7, 6), torch.full((2, 7, 5), float("nan"))):
            with self.assertRaises(ValueError): model(x)


if __name__ == "__main__": unittest.main()
