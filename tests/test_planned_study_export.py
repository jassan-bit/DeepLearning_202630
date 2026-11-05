"""Synthetic public-boundary and chronology contracts for the D7 exporter."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd

import export_planned_study as export
import package_gru_lstm_tcn_downloads as package


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

    def test_missing_inference_timings_are_not_fabricated_as_zero(self):
        records = [{"model": model, "seed": seed, "training_seconds": duration}
                   for model in export.MODELS for seed, duration in ((8101, 1.), (8102, 3.))]
        result = export.cost_summary({"final": pd.DataFrame(records)})
        self.assertEqual(len(result), 3)
        self.assertTrue((result.training_seconds_total == 4.).all())
        self.assertTrue((result.training_seconds_mean == 2.).all())
        self.assertTrue((result.inference_seconds_n_defined == 0).all())
        self.assertTrue(result.inference_seconds_total.isna().all())


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

    @staticmethod
    def closed_records():
        registration = {"source_commit": "a" * 40, "registration_signature": "registered"}
        selection = {"recipes": {"GRU": {"width": 16}, "LSTM": {"width": 16}, "TCN": {"width": 8}},
                     "baseline_model": "GRU"}
        plan = {"R": 10, "seeds": list(range(8101, 8111)),
                "planned_at_utc": "2026-11-05T19:03:00+00:00"}
        hashes = {name: str(number) * 64 for number, name in enumerate(
            ("registration", "selection", "pilot", "plan", "closed_planning"), 1)}
        closed = {"registration_signature": registration["registration_signature"],
                  "registration_sha256": hashes["registration"], "hpo_selection_sha256": hashes["selection"],
                  "pilot_complete_sha256": hashes["pilot"], "dimensioning_sha256": hashes["plan"],
                  "dimensioning": copy.deepcopy(plan), "selected_recipes": copy.deepcopy(selection["recipes"]),
                  "baseline_model_selected_by_hpo": "GRU", "preregistration_commit": registration["source_commit"],
                  "planned_fresh_final_models": 30, "planned_final_fit_stages": 60,
                  "planned_reserve_prediction_rows": 1800, "new_reserve_payloads_read": 0,
                  "closed_before_definitive_fit_at_utc": "2026-11-05T19:04:00+00:00"}
        attestation = {"public_planning_sha256": hashes["closed_planning"], "planned_R_per_family": 10,
                       "definitive_fits_started_before_publication": 0, "new_reserve_payload_accessed": False,
                       "public_planning_verified_at_utc": "2026-11-05T19:05:00+00:00",
                       "public_planning_commit": "b" * 40, "public_planning_url": "https://example.invalid/closure"}
        return [closed, attestation, registration, selection, plan, hashes, "2026-11-05T19:06:00+00:00"]

    def test_public_dimensioning_matches_private_R_plan_hash_and_precompletion_time(self):
        records = self.closed_records()
        result = export.verify_closed_planning(*records)
        self.assertEqual(result["registered_R_matches_public_closure"], 10)
        self.assertTrue(result["private_dimensioning_sha256_matches_public_closure"])
        self.assertTrue(result["completion_time_is_not_claimed_as_fit_start"])
        self.assertTrue(result["public_receipt_records_zero_definitive_fits_started_at_verification"])

    def test_wrong_dimensioning_or_publication_after_weights_is_rejected(self):
        mutations = ((0, "dimensioning_sha256", "wrong"),
                     (1, "planned_R_per_family", 9),
                     (1, "public_planning_sha256", "wrong"),
                     (1, "public_planning_verified_at_utc", "2026-11-05T19:07:00+00:00"),
                     (1, "definitive_fits_started_before_publication", 1))
        for index, key, value in mutations:
            with self.subTest(index=index, key=key):
                records = copy.deepcopy(self.closed_records())
                records[index][key] = value
                with self.assertRaises(ValueError):
                    export.verify_closed_planning(*records)


class PlannedPackagingTests(unittest.TestCase):
    def test_code_zip_receives_canonical_registered_design_and_readme(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'experiments/gru_lstm_tcn_planned_20261105'
            directory.mkdir(parents=True)
            (directory / 'design.json').write_text('{"study_id":"synthetic-only"}\n')
            (directory / 'README.md').write_text('# Synthetic registration\n')
            (directory / 'public_preregistration.json').write_text('{"synthetic_preregistration":true}\n')
            (directory / 'closed_planning.json').write_text('{"synthetic_closure":true}\n')
            packages = {item: {} for item in ('8.2', '8.6', '8.7')}
            with mock.patch.object(package, 'ROOT', root):
                names = package.add_planned_registration(packages)
                expected = {f'experiments/gru_lstm_tcn_planned_20261105/{name}'
                            for name in ('design.json', 'README.md', 'public_preregistration.json', 'closed_planning.json')}
                self.assertEqual(set(names), expected)
                for item in packages:
                    self.assertEqual(set(packages[item]), expected)
                    for name in expected:
                        self.assertEqual(packages[item][name], (root / name).read_bytes())
                (directory / 'design.json').unlink()
                with self.assertRaises(ValueError):
                    package.add_planned_registration({item: {} for item in packages})

    def test_second_phase_is_optional_but_partial_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "planned"
            with mock.patch.object(package, "PLANNED", directory):
                self.assertIsNone(package.planned_public_record())
                directory.mkdir()
                with self.assertRaises(ValueError):
                    package.planned_public_record()
                (directory / "public_allowlist_manifest.json").write_text('{}')
                (directory / "summary.json").write_text(json.dumps({"models": list(export.MODELS), "export_complete": True}))
                self.assertTrue(package.planned_public_record()["export_complete"])
                (directory / "summary.json").write_text(json.dumps({"models": ["MLP"], "export_complete": True}))
                with self.assertRaises(ValueError):
                    package.planned_public_record()

    def test_new_raw_cache_name_is_never_packaged(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw = directory / "daily_market_reserved.csv"
            raw.write_text("market_day_utc,close\n2026-09-01,1\n")
            with self.assertRaises(ValueError):
                package.add({}, raw, "data/daily_market_reserved.csv")

    def test_readme_differentiates_original_exploration_and_new_planned_phase(self):
        old = package.common_readme("test-revision")
        new = package.common_readme("test-revision", {"export_complete": True})
        self.assertIn("primer benchmark", old)
        self.assertNotIn("Segunda fase D7 planificada", old)
        self.assertIn("Segunda fase D7 planificada", new)
        self.assertIn("no acredita potencia temporal", new)


if __name__ == "__main__":
    unittest.main()
