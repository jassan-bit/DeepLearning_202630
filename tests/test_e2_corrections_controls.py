"""Pure guards and artificial fixtures; no model fit or real-data inference."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("controls_under_test", SCRIPTS / "e2_corrections_controls.py")
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)


class ControlGuardTests(unittest.TestCase):
    def test_unaudited_cohort_rejected_before_any_input_read(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.json"
            audit.write_text(json.dumps({"data_ready_for_bounded_controls": False}))
            args = SimpleNamespace(audit=audit, arrays=Path("missing.npz"), expected_sha256="missing")
            with mock.patch.object(controls, "sha", side_effect=AssertionError("input must not be read")):
                with self.assertRaisesRegex(ValueError, "Auditor has not admitted"):
                    controls.prepare(args)

    def test_budget_guard_requires_one_matching_reservation(self):
        @contextmanager
        def ledger(_):
            yield [], None
        fake = SimpleNamespace(LEDGER=Path("unused"), locked_ledger=ledger)
        with mock.patch.dict(sys.modules, {"e2_corrections_budget": fake}):
            for reservations in ({}, {"x": {"phase": "other"}},
                                 {"x": {"phase": "control"}, "y": {"phase": "control"}}):
                fake.summarize = lambda rows, r=reservations: {"open_reservations": r}
                with self.assertRaises(RuntimeError):
                    controls.require_budget("control")
            fake.summarize = lambda rows: {"open_reservations": {"x": {"phase": "control", "cap_seconds": 10}}}
            self.assertEqual(controls.require_budget("control")["cap_seconds"], 10)

    def test_block_shuffle_preserves_rows_and_never_joins_gaps(self):
        dates = np.r_[np.arange("2020-01-01", "2020-02-06", dtype="datetime64[D]"),
                      np.arange("2021-01-01", "2021-02-12", dtype="datetime64[D]")]
        values = np.arange(len(dates))
        shuffled, source = controls.permute_blocks(values, dates, 7, 271801)
        np.testing.assert_array_equal(np.sort(shuffled), values)
        np.testing.assert_array_equal(shuffled, values[source])
        self.assertFalse(np.array_equal(source, values))
        cuts = np.flatnonzero(np.diff(dates).astype(int) != 1) + 1
        for run in np.split(np.arange(len(dates)), cuts):
            for start in range(0, len(run), 7):
                destination = run[start:start+7]
                copied_dates = dates[source[destination]]
                self.assertTrue((np.diff(copied_dates).astype(int) == 1).all())
        self.assertEqual(int(source[35]), 35)  # The single one-day edge has no same-length partner.

    def test_domain_shuffle_crosses_cohorts_instead_of_permuting_constant_labels_locally(self):
        dates = np.r_[np.arange("2020-01-01", "2020-01-29", dtype="datetime64[D]"),
                      np.arange("2021-01-01", "2021-01-29", dtype="datetime64[D]")]
        labels = np.r_[np.zeros(28, dtype=int), np.ones(28, dtype=int)]
        shuffled, source = controls.permute_blocks(labels, dates, 7, 2026111503)
        self.assertGreater(np.count_nonzero(shuffled != labels), 0)
        self.assertEqual(shuffled.sum(), labels.sum())
        self.assertEqual(len(np.unique(source)), len(labels))

    def test_block_shuffle_is_seeded_and_rejects_bad_calendar(self):
        dates = np.arange("2020-01-01", "2020-02-12", dtype="datetime64[D]")
        a, index_a = controls.permute_blocks(np.arange(42), dates, 7, 53)
        b, index_b = controls.permute_blocks(np.arange(42), dates, 7, 53)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(index_a, index_b)
        with self.assertRaises(ValueError):
            controls.permute_blocks(np.arange(42), dates[::-1], 7, 53)

    def test_purged_folds_cover_each_validation_row_once_and_exclude_seven_calendar_days(self):
        dates = np.arange("2020-01-01", "2020-04-30", dtype="datetime64[D]")
        x = np.arange(len(dates)*3, dtype=float).reshape(-1, 3)
        folds = controls.purged_folds(x, dates)
        np.testing.assert_array_equal(np.sort(np.concatenate([f["validation"] for f in folds])), np.arange(len(x)))
        for fold in folds:
            distance = abs(dates[fold["train"]].astype(int)[:, None]-dates[fold["validation"]].astype(int)[None, :])
            self.assertGreater(distance.min(), 7)
            np.testing.assert_allclose(fold["x_train"].mean(axis=0), 0, atol=1e-14)

    def test_classifier_scaler_is_fitted_only_on_its_training_fold(self):
        dates = np.arange("2020-01-01", "2020-04-30", dtype="datetime64[D]")
        x = np.arange(len(dates)*2, dtype=float).reshape(-1, 2)
        original = controls.purged_folds(x, dates)[2]
        changed_x = x.copy()
        changed_x[original["validation"]] += 1e8
        changed = controls.purged_folds(changed_x, dates)[2]
        np.testing.assert_array_equal(original["mean"], changed["mean"])
        np.testing.assert_array_equal(original["std"], changed["std"])
        np.testing.assert_array_equal(original["x_train"], changed["x_train"])

    def test_auc_ties_and_one_class_are_explicit(self):
        self.assertEqual(controls.auc([0, 1], [4, 4]), .5)
        self.assertEqual(controls.auc([0, 1], [0, 1]), 1.)
        self.assertEqual(controls.auc([0, 1], [1, 0]), 0.)
        self.assertTrue(math.isnan(controls.auc([0, 0], [0, 1])))
        with self.assertRaises(ValueError):
            controls.auc([0, 2], [0, 1])

    def test_calendar_sampler_preserves_missing_calendar_positions(self):
        dates = np.array(["2020-01-01", "2020-01-03"], dtype="datetime64[D]")
        indices = controls.calendar_indices(dates, 8, 1, 71)
        full_calendar = np.array([0, -1, 1])
        expected = full_calendar[np.random.default_rng(71).integers(0, 3, size=(8, 3))]
        np.testing.assert_array_equal(indices, expected)
        self.assertIn(-1, indices)
        self.assertEqual(indices.shape, (8, 3))

    def test_holm_retains_the_full_declared_family(self):
        values = [.001, .02] + [1.]*11
        adjusted = controls.holm(values)
        self.assertAlmostEqual(adjusted[0], .013)
        self.assertAlmostEqual(adjusted[1], .24)
        self.assertEqual(adjusted[2:], [1.]*11)
        with self.assertRaises(ValueError):
            controls.holm([float("nan")])

    def test_centered_bootstrap_boundary_and_plus_one_are_correct(self):
        self.assertEqual(controls.centered_bootstrap_p([1.5, 2, 2.5], 2., 1.), .25)
        self.assertEqual(controls.centered_bootstrap_p([1.5, 2, 2.5], 2., 2.), .75)
        with self.assertRaises(ValueError):
            controls.centered_bootstrap_p([float("nan")], 2., 1.)

    def test_canonical_auditor_keys_are_respected_and_missing_test_is_rejected(self):
        arrays, keys = {}, {}
        for stage, count, start in (("training", 655, "2019-01-01"),
                                    ("calibration", 53, "2022-01-01"), ("evaluation", 608, "2023-01-01")):
            keys[stage] = {"dates": stage+"_dates", "X_scaled": stage+"_X", "X_relative": stage+"_relative",
                           "raw": stage+"_raw", "y_original": stage+"_y_original", "y_scaled": stage+"_y_scaled"}
            for field in ("X", "relative", "raw"):
                arrays[stage+"_"+field] = np.ones((count, 7, 5), dtype="float32")
            arrays[stage+"_dates"] = np.datetime64(start)+np.arange(count)
            arrays[stage+"_y_original"] = np.full(count, .25)
            arrays[stage+"_y_scaled"] = np.ones(count, dtype="float32")
        arrays["training_target_scale"] = np.array(.25)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"arrays.npz"
            np.savez(path, **arrays)
            data, mapping = controls.load_cohorts(path, keys)
            self.assertEqual(mapping["train"]["relative"], "training_relative")
            self.assertEqual(mapping["test"]["X"], "evaluation_X")
            self.assertEqual(data["target_scale"], .25)
            np.savez(path, **{key: value for key, value in arrays.items() if not key.startswith("evaluation_")})
            with self.assertRaisesRegex(ValueError, "Missing array"):
                controls.load_cohorts(path, keys)

    def test_control_runner_never_calls_the_production_cpu_test_escape(self):
        import ast
        tree = ast.parse(Path(controls.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.assertNotIn("allow_cpu_for_tests", [keyword.arg for keyword in node.keywords])

    def test_prepared_source_binding_checks_pinned_originals_without_model_execution(self):
        bindings = controls.source_bindings()
        self.assertGreaterEqual(len(bindings), 76+12+4)
        self.assertEqual(bindings[str(SCRIPTS/"e2_recipe_models.py")],
                         "47380cc87e2d4b4221ee4fc97da3286990470b3304dd1d9ef990ace51c338b75")


if __name__ == "__main__":
    unittest.main()
