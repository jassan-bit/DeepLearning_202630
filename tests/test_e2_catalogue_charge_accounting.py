"""Failure checks for receipt deduplication and honest unknown-time treatment."""
import importlib.util
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location("catalogue_accounting", Path(__file__).resolve().parents[1] /
                                             "scripts/e2_catalogue_charge_accounting.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(attempt, physical=None, status="completed"):
    return {"attempt_id": attempt, "physical_job_id": physical or attempt,
            "physical_payload_sha256": "payload-" + attempt,
            "model": "M", "trial_id": attempt, "fold_origin": "2024-01-01", "seed": "271201",
            "status": status, "observed_wall_charge_seconds": None if status == "outcome_unknown" else 1.0,
            "unmeasured_conservative_allocation_seconds": 1860.0 if status == "outcome_unknown" else None}


class ExclusiveAccountingTests(unittest.TestCase):
    def test_disjoint_receipts_and_unknown_allocation_are_accepted(self):
        MODULE.validate_exclusive_records([row("cached"), row("current"), row("lost", status="outcome_unknown")])

    def test_same_physical_job_across_two_sources_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Physical job charged more than once"):
            MODULE.validate_exclusive_records([row("cached", "same-physical"), row("current", "same-physical")])

    def test_duplicate_registered_opportunity_under_another_attempt_is_rejected(self):
        a, b = row("one"), row("two")
        b["trial_id"] = a["trial_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate registered opportunity"):
            MODULE.validate_exclusive_records([a, b])

    def test_unknown_allocation_cannot_be_promoted_to_observed_elapsed(self):
        item = row("lost", status="outcome_unknown")
        item["observed_wall_charge_seconds"] = 1860.0
        with self.assertRaisesRegex(ValueError, "Unknown elapsed time cannot become measured"):
            MODULE.validate_exclusive_records([item])

    def test_known_receipt_cannot_carry_both_measured_and_unmeasured_cost(self):
        item = row("known")
        item["unmeasured_conservative_allocation_seconds"] = 100.0
        with self.assertRaisesRegex(ValueError, "exclusive"):
            MODULE.validate_exclusive_records([item])


if __name__ == "__main__":
    unittest.main()
