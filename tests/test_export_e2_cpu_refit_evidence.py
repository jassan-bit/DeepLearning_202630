"""Metadata gate tests only: no fitted fixtures, model imports or predictions."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/export_e2_cpu_refit_evidence.py"
SPEC = importlib.util.spec_from_file_location("cpu_evidence_export", SOURCE)
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def roster():
    return ["predictor-" + str(i) for i in range(81)] + list(export.SHARED)


def completed_metadata():
    rows = [{"id": key, "CPU_study_plan_sha256": export.PLAN_SHA,
             "status": "completed", "returncode": 0,
             "charged_worker_wall_seconds": 1.0,
             "parent_wall_seconds_added": False} for key in roster()]
    status = {"status": export.COMPLETE, "jobs": rows,
              "original_CUDA_parity": False, "full_original_campaign_passed": False,
              "HPO_repeated": False, "parent_wall_seconds_added": False,
              "charged_worker_seconds": 83.0, "global_cap_sum_worker_seconds": 100.0}
    identity = {"plan_sha256": export.PLAN_SHA, "CPU_study": True,
                "original_CUDA_parity": False, "old_weights_or_predictions_used": False,
                "global_cap_sum_worker_seconds": 100.0}
    return status, identity


def ledger_bytes(events):
    previous = None
    lines = []
    for value in events:
        row = dict(value, previous_record_sha256=previous)
        raw = json.dumps(row, sort_keys=True).encode() + b"\n"
        previous = hashlib.sha256(raw).hexdigest()
        lines.append(raw)
    return b"".join(lines)


def events():
    return [
        {"event": "anchor", "historical_budget_sha256": export.HISTORICAL_SHA,
         "approved_total_hours": 84},
        {"event": "reserve", "reservation_id": "one", "phase": "example",
         "cap_seconds": 10.0},
        {"event": "finish", "reservation_id": "one", "phase": "example",
         "cap_seconds": 10.0, "elapsed_seconds": 2.0, "returncode": 0},
    ]


class CompletionGates(unittest.TestCase):
    def test_registered_complete_roster_passes_metadata_gate(self):
        status, identity = completed_metadata()
        export.validate_status(status, identity, export.PLAN_SHA, roster())

    def test_partial_status_cannot_pass_even_with83_rows(self):
        status, identity = completed_metadata()
        for value in ["running", "partial_stopped"]:
            status["status"] = value
            with self.assertRaisesRegex(ValueError, "not complete"):
                export.validate_status(status, identity, export.PLAN_SHA, roster())

    def test_mixed_plan_rejected(self):
        status, identity = completed_metadata()
        status["jobs"][17]["CPU_study_plan_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "Mixed plan"):
            export.validate_status(status, identity, export.PLAN_SHA, roster())

    def test_missing_primary_stage_cannot_pass(self):
        status, identity = completed_metadata()
        status["jobs"][-1]["id"] = "unregistered-replacement"
        with self.assertRaisesRegex(ValueError, "registered83"):
            export.validate_status(status, identity, export.PLAN_SHA, roster())

    def test_double_counted_parent_total_rejected(self):
        status, identity = completed_metadata()
        status["charged_worker_seconds"] = 89.0
        with self.assertRaisesRegex(ValueError, "total differs"):
            export.validate_status(status, identity, export.PLAN_SHA, roster())

    def test_physical_receipts_required(self):
        status, _ = completed_metadata()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "accounting").mkdir()
            with self.assertRaisesRegex(ValueError, "physical accounting"):
                export.validate_receipts(root, status, export.PLAN_SHA, {})
            self.assertEqual(list(root.rglob("*.json")), [])

    def test_failed_worker_prevents_success(self):
        status, identity = completed_metadata()
        status["jobs"][0]["returncode"] = 1
        with self.assertRaisesRegex(ValueError, "Failed or unclosed"):
            export.validate_status(status, identity, export.PLAN_SHA, roster())

    def test_reused_receipt_origins_require_explicit_mapping(self):
        status, _ = completed_metadata()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "accounting").mkdir()
            (root / "logs").mkdir()
            finishes = {}
            origins = {}
            for i, row in enumerate(status["jobs"]):
                origin = "/preserved/v3" if i < 80 else str(root)
                if i < 80:
                    origins[row["id"]] = origin
                log = root / "logs" / (row["id"] + ".log")
                log.write_bytes(b"")
                row.update(reservation_id="reservation-" + str(i), reservation_cap_seconds=10,
                           log_sha256=export.sha(log), command=["/python", "/registered.py",
                           "--job", row["id"], "--output", origin,
                           "--reservation-id", "reservation-" + str(i)])
                (root / "accounting" / (row["id"] + ".json")).write_text(json.dumps(row))
                finishes[row["reservation_id"]] = {"phase": export.PHASE + row["id"],
                    "returncode": 0, "elapsed_seconds": 1.0, "cap_seconds": 10.0}
            with self.assertRaisesRegex(ValueError, "different output/job"):
                export.validate_receipts(root, status, export.PLAN_SHA, finishes)
            validated = export.validate_receipts(root, status, export.PLAN_SHA, finishes, origins)
            self.assertEqual(len(validated), 83)
            # Accepting provenance must never rewrite copied source receipts.
            self.assertTrue(all(export.read(root / "accounting" / (key + ".json"))["command"][5]
                                == "/preserved/v3" for key in origins))


class LedgerAndPathGates(unittest.TestCase):
    def test_closed_chain_matches_finish(self):
        rows, finishes = export.validate_ledger(ledger_bytes(events()))
        self.assertEqual(len(rows), 3)
        self.assertEqual(finishes["one"]["elapsed_seconds"], 2.0)

    def test_live_reservation_rejected(self):
        with self.assertRaisesRegex(ValueError, "open reservations"):
            export.validate_ledger(ledger_bytes(events()[:2]))

    def test_broken_chain_rejected(self):
        raw = ledger_bytes(events()).replace(b'"previous_record_sha256": null',
                                             b'"previous_record_sha256": "wrong"', 1)
        with self.assertRaisesRegex(ValueError, "Broken ledger"):
            export.validate_ledger(raw)

    def test_unmatched_finish_rejected(self):
        value = events()
        value[-1]["reservation_id"] = "other"
        with self.assertRaisesRegex(ValueError, "Unmatched"):
            export.validate_ledger(ledger_bytes(value))

    def test_artifact_escape_and_changed_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "receipt.json"
            path.write_text("{}\n")
            checksum = export.sha(path)
            export.verified(root, "receipt.json", checksum)
            path.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                export.verified(root, "receipt.json", checksum)
            with self.assertRaisesRegex(ValueError, "Unsafe artifact"):
                export.inside(root, "../receipt.json")


if __name__ == "__main__":
    unittest.main()
