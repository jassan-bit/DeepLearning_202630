"""Negative controls for the offline E2 handoff audit."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e3_audit_e2_handoff as audit


class HandoffAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backup = audit.load_json(ROOT / audit.RECEIPTS / "audit_backup_manifest.json")
        cls.freeze = audit.load_json(ROOT / audit.PUBLIC / "final_freeze.json")
        cls.partition = audit.load_json(ROOT / audit.PARTITION)
        cls.raw_zip = (ROOT / audit.OUTPUT / "inputs/audit_metadata.zip").read_bytes()
        with (ROOT / audit.PUBLIC / "benchmark_A2.csv").open(newline="", encoding="utf-8") as stream:
            cls.a2 = list(csv.DictReader(stream))

    def test_saved_outputs_reproduce_offline_without_weight_downloads(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(audit, "urlopen", side_effect=AssertionError("network forbidden")):
            cache = Path(temporary)
            audit.extract_metadata(self.raw_zip, self.backup, cache)
            rows, summary, checks = audit.audit(ROOT, [cache])
        self.assertEqual(len(rows), 81)
        self.assertEqual(summary["checkpoint_bytes_hash_verified"], 2)
        self.assertEqual(summary["downloaded_metadata_hashes_verified"], 167)
        self.assertFalse(summary["block_A_closed"])
        self.assertEqual(summary, audit.load_json(ROOT / audit.OUTPUT / "audit_summary.json"))
        self.assertEqual(audit.csv_text(rows), (ROOT / audit.OUTPUT / "audit_predictors.csv").read_text())
        self.assertEqual(audit.csv_text(checks), (ROOT / audit.OUTPUT / "audit_checks.csv").read_text())

    def test_corrupt_zip_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            damaged = self.raw_zip[:-1] + bytes([self.raw_zip[-1] ^ 1])
            with self.assertRaisesRegex(ValueError, "archive hash"):
                audit.extract_metadata(damaged, self.backup, Path(temporary))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_zip_path_traversal_is_rejected_even_with_matching_hash(self):
        stream = io.BytesIO()
        data = b"not outside"
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("../escape.json", data)
        raw = stream.getvalue()
        backup = {"archive_transport": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
                  "artifacts": [{"storage": "release_archive_member", "archive_member": "../escape.json",
                                 "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "unsafe ZIP"):
                audit.extract_metadata(raw, backup, Path(temporary))

    def test_downloaded_bytes_do_not_pass_on_manifest_claim_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            (cache / "scaler.json").write_text('{"target_scale":999}')
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                audit.local_artifact("scaler.json", "0" * 64, [cache])
            self.assertIsNone(audit.local_artifact("missing.pt", "0" * 64, [cache]))

    def test_calibrator_for_another_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            audit.extract_metadata(self.raw_zip, self.backup, cache)
            model = self.freeze["models"][0]
            scaler = audit.load_json(cache / model["scaler"])
            calibrator = audit.load_json(cache / model["calibrator"])
            calibrator["predictor_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "binding mismatch"):
                audit.validate_metadata(model, scaler, calibrator, self.partition)

    def test_scaler_future_date_is_rejected_even_if_json_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            audit.extract_metadata(self.raw_zip, self.backup, cache)
            model = self.freeze["models"][0]
            scaler = audit.load_json(cache / model["scaler"])
            scaler["train_dates"][-1] = "2025-01-01"
            with self.assertRaisesRegex(ValueError, "training dates mismatch"):
                audit.validate_metadata(model, scaler, audit.load_json(cache / model["calibrator"]), self.partition)

    def test_boolean_is_not_a_numeric_scale_or_calibration_radius(self):
        self.assertFalse(audit.finite_positive(True))
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            audit.extract_metadata(self.raw_zip, self.backup, cache)
            model = self.freeze["models"][0]
            scaler = audit.load_json(cache / model["scaler"])
            calibrator = audit.load_json(cache / model["calibrator"])
            calibrator["q"] = True
            with self.assertRaisesRegex(ValueError, "calibration radius"):
                audit.validate_metadata(model, scaler, calibrator, self.partition)

    def test_duplicate_seed_cannot_inflate_predictor_count(self):
        freeze = copy.deepcopy(self.freeze)
        freeze["models"][1]["seed"] = freeze["models"][0]["seed"]
        with self.assertRaisesRegex(ValueError, "duplicate predictor"):
            audit.validate_catalog(freeze, self.a2)

    def test_declared_asset_absent_from_release_listing_is_not_verified(self):
        asset = next(a for a in self.backup["artifacts"] if a["storage"] == "release_asset")
        self.assertFalse(audit.validate_release_record(asset, {}))
        listing = audit.load_json(ROOT / audit.RECEIPTS / "audit_release_listing.json")
        records = {a["id"]: a for a in listing["assets"]}
        self.assertTrue(audit.validate_release_record(asset, records))
        records[asset["asset_id"]]["size"] += 1
        self.assertFalse(audit.validate_release_record(asset, records))

    def test_overlapping_partition_metadata_is_rejected(self):
        partition = copy.deepcopy(self.partition)
        partition["final_partition"]["calibration"]["sample_IDs"][0] = "2024-10-31"
        with self.assertRaisesRegex(ValueError, "IDs overlap"):
            audit.validate_partition(partition)

    def test_A2_hash_change_cannot_be_reported_as_same_benchmark(self):
        original = audit.sha256
        def tampered(path):
            return "0" * 64 if path.name == "benchmark_A2.csv" else original(path)
        with patch.object(audit, "sha256", side_effect=tampered):
            with self.assertRaisesRegex(ValueError, "public source hash mismatch"):
                audit.audit(ROOT, [])


if __name__ == "__main__":
    unittest.main()
