"""Synthetic public-boundary and chronology contracts for the D7 exporter."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd

import export_planned_study as export


class ExportBoundaryTests(unittest.TestCase):
    def test_learned_vectors_removed_but_hash_provenance_and_recipes_retained(self):
        value = {"models": ["GRU", "LSTM", "TCN"],
                 "jobs": [{"width": 16, "seed": 8101, "mase_scale": .004,
                           "preprocessing": {"x_mean": [1, 2], "y_scale": .01},
                           "train_indices": [1, 2, 3]}],
                 "artifact_sha256": {"final/GRU_s8101/model.pt": "a" * 64}}
        public = export.public_metadata(value)
        self.assertEqual(public["jobs"], [{"width": 16, "seed": 8101}])
        self.assertEqual(public["artifact_sha256"], value["artifact_sha256"])
        self.assertIn("preprocessing", value["jobs"][0])

    def test_boundary_rejects_raw_csv_json_and_weights(self):
        cases = (("predictions.csv", "model,rmse\nGRU,.1\n"),
                 ("aggregate.csv", "model,y_true\nGRU,.1\n"),
                 ("aggregate.json", json.dumps({"job": {"x_scale": [1, 2]}})),
                 ("model.pt", "not a public aggregate"))
        for name, content in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                (directory / name).write_text(content)
                with self.assertRaises(ValueError):
                    export.validate_payloads(directory)

    def test_aggregate_catalog_excludes_scaler_and_preserves_seed_identity(self):
        frame = pd.DataFrame([{"model": "GRU", "trajectory_seed": 8101, "seed": 8101,
                               "rmse": .01, "mase_scale": .004, "train_indices": "1,2,3",
                               "training_seconds": 2.5, "parameters": 3777}])
        public = export.public_catalog(frame)
        self.assertEqual(public.seed.tolist(), [8101])
        self.assertEqual(len(public.columns), len(set(public.columns)))
        self.assertFalse(set(public.columns) & export.PRIVATE_KEYS)
        self.assertEqual(float(public.training_seconds.iloc[0]), 2.5)

    def test_mirrors_identical_and_different_existing_snapshot_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            stage, one, two = (base / name for name in ("stage", "one", "two"))
            stage.mkdir()
            (stage / "summary.json").write_text('{"models":["GRU","LSTM","TCN"]}\n')
            (stage / "metric_summary.csv").write_text("model,metric,mean\nGRU,rmse,.01\n")
            export.publish_identical(stage, [one, two])
            for p in stage.iterdir():
                self.assertEqual(p.read_bytes(), (one / p.name).read_bytes())
                self.assertEqual(p.read_bytes(), (two / p.name).read_bytes())
            export.publish_identical(stage, [one, two])
            (one / "summary.json").write_text("changed")
            with self.assertRaises(ValueError):
                export.publish_identical(stage, [one, two])


class ChronologyTests(unittest.TestCase):
    @staticmethod
    def records():
        signature = "test-registration-only"
        records = []
        keys = ("registered_at_utc", "selected_at_utc", "completed_at_utc", "planned_at_utc",
                "all_weights_frozen_at_utc", "at_utc", "completed_at_utc", "analyzed_at_utc")
        for minute, key in enumerate(keys):
            record = {key: f"2026-11-05T18:{minute:02d}:00+00:00", "registration_signature": signature}
            if minute in (0, 1, 2, 4):
                record["reserve_price_payloads_read"] = 0
            records.append(record)
        return records

    def test_actual_stage_order_and_scope(self):
        result = export.chronology(*self.records())
        self.assertTrue(result["strict_stage_order_verified"])
        self.assertTrue(result["new_reserve_is_historical_not_live"])
        self.assertFalse(result["temporal_power_claimed"])

    def test_access_before_weight_freeze_signature_change_and_preaccess_fail(self):
        mutations = ((5, "at_utc", "2026-11-05T18:03:30+00:00"),
                     (6, "registration_signature", "another"),
                     (2, "reserve_price_payloads_read", 1))
        for index, key, value in mutations:
            with self.subTest(index=index, key=key):
                records = copy.deepcopy(self.records())
                records[index][key] = value
                with self.assertRaises(ValueError):
                    export.chronology(*records)


if __name__ == "__main__":
    unittest.main()
