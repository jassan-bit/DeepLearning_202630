"""Portable final-workflow contracts tested with synthetic metadata only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/gru_lstm_tcn_final.py"
spec = importlib.util.spec_from_file_location("public_gru_lstm_tcn_final_for_tests", SCRIPT)
final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)


class FinalContracts(unittest.TestCase):
    def setUp(self):
        self.plan = {"histories": [7, 14, 21, 28],
            "models": {"LSTM": {"width": 32, "learning_rate": .003},
                       "GRU": {"width": 32, "learning_rate": .003},
                       "TCN": {"width": 16, "learning_rate": .001}},
            "seeds": list(range(4101, 4111)), "training": {"max_epochs": 100}}
        self.jobs = [{"history_days": history, "model": family, "seed": seed,
                      "epochs": 12, "parameters": 4000, **recipe}
                     for history in self.plan["histories"]
                     for family, recipe in self.plan["models"].items()
                     for seed in self.plan["seeds"]]

    def test_complete_grid_and_fixed_recipes(self):
        final.check_jobs(self.jobs, self.plan)
        for modified in (self.jobs[:-1], self.jobs + [self.jobs[0]],
                         [{**j, "width": 99} if i == 0 else j for i, j in enumerate(self.jobs)]):
            with self.assertRaises(ValueError):
                final.check_jobs(modified, self.plan)
        for epoch in (0, True, 101, 1.5):
            modified = [{**j, "epochs": epoch} if i == 0 else j for i, j in enumerate(self.jobs)]
            with self.assertRaises(ValueError):
                final.check_jobs(modified, self.plan)

    def test_selection_uses_cv_and_stable_ties(self):
        rows = [{"model": "GRU", "history_days": 14, "parameters": 3777, "cv_rmse_mean": .01},
                {"model": "GRU", "history_days": 7, "parameters": 3777, "cv_rmse_mean": .01 + 1e-14}]
        self.assertEqual(final.choose_setting(rows)["history_days"], 7)
        rows[0]["cv_rmse_mean"] = .009
        self.assertEqual(final.choose_setting(rows)["history_days"], 14)

    def test_paths_cannot_escape_portable_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(final._safe(root, "metadata/a.json"), root / "metadata/a.json")
            for bad in ("../outside", "/absolute/outside"):
                with self.assertRaises(ValueError):
                    final._safe(root, bad)

    def reservation(self):
        return {"calendar_days": 608, "archives": [
            {"archive": f"BTCUSDT-1m-{month}.zip", "relative_path": f"archive/{month}.zip", "sha256": "a" * 64}
            for month in final.TEST_MONTHS]}

    def test_reservation_admits_only_the_twenty_fixed_months(self):
        reserve = self.reservation()
        self.assertEqual(len(final.validate_reservation(reserve)), 20)
        for changed in (reserve["archives"][:-1], reserve["archives"] + [
                {"archive": "BTCUSDT-1m-2026-09.zip", "relative_path": "archive/extra.zip", "sha256": "b" * 64}]):
            with self.assertRaises(ValueError):
                final.validate_reservation({"calendar_days": 608, "archives": changed})
        unsafe = self.reservation()
        unsafe["archives"][0]["relative_path"] = "../outside.zip"
        with self.assertRaises(ValueError):
            final.validate_reservation(unsafe)

    def test_cached_declaration_reads_only_metadata_not_the_daily_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # Deliberately do not create the daily CSV: closure must not open it.
            audit = root / "audit.json"
            audit.write_text(json.dumps({"daily_market_sha256": "b" * 64}))
            declaration = final.declared_test_source(root, test_daily_csv="not-opened.csv", test_audit_json="audit.json")
            self.assertEqual(declaration["expected_daily_sha256"], "b" * 64)
            self.assertEqual(declaration["audit_sha256"], hashlib.sha256(audit.read_bytes()).hexdigest())
            self.assertFalse((root / "not-opened.csv").exists())
            with self.assertRaises(ValueError):
                final.declared_test_source(root, test_daily_csv="not-opened.csv")

    def test_raw_declaration_requires_only_reservation_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reservation_path = root / final.RESERVE_RELATIVE
            reservation_path.parent.mkdir(parents=True)
            reservation_path.write_text(json.dumps(self.reservation()))
            declared = final.declared_test_source(root)
            self.assertEqual(declared["mode"], "reserved_archives")
            self.assertFalse((root / "archive").exists())

    def test_model_manifest_gate_precedes_any_test_payload_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(final, "validate_final_models", side_effect=ValueError("unfrozen")):
                with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("payload read")):
                    with self.assertRaisesRegex(ValueError, "unfrozen"):
                        final._test_data(root, root / "out", raw_root=root / "raw")
            self.assertFalse((root / "out").exists())

    def test_complete_job_labels_do_not_substitute_for_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, out = Path(temporary), Path(temporary) / "out"
            destination = out / "final_evaluation"
            destination.mkdir(parents=True)
            (destination / "freeze.json").write_text("synthetic freeze identity")
            manifest = {"freeze_sha256": hashlib.sha256((destination / "freeze.json").read_bytes()).hexdigest(),
                        "jobs": self.jobs, "artifact_sha256": {}}
            (destination / "final_models_manifest.json").write_text(json.dumps(manifest))
            with mock.patch.object(final, "validate_final_configuration", return_value=({"final_jobs": self.jobs}, self.plan)):
                with mock.patch.object(final, "_verify", side_effect=AssertionError("hashing should not start")):
                    with self.assertRaisesRegex(ValueError, "lacks a required"):
                        final.validate_final_models(root, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
