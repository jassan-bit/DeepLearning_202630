"""Metadata fixtures only: no market payloads, model fits or real freeze claims."""
from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_amended_data_guard as guard

NOW = "2026-11-12T19:00:00+00:00"
LATER = "2027-02-21T00:00:00+00:00"
PROTOCOL = sha256(b"synthetic-test-protocol-not-an-actual-registration").hexdigest()


def synthetic_freeze():
    """Clearly artificial attestation used solely to exercise validation rules."""
    systems = []
    for name in guard.REQUIRED:
        row = {"model_id": name, "training_cutoff": "2026-08-24"}
        for field in ("recipe_sha256", "weights_sha256", "preprocessing_sha256", "calibration_sha256"):
            row[field] = sha256(("synthetic-metadata:" + name + ":" + field).encode()).hexdigest()
        systems.append(row)
    return {"version": guard.FREEZE_VERSION, "protocol_sha256": PROTOCOL,
            "amendment_registered_at_utc": "2026-11-12T12:00:00+00:00",
            "selection_closed_at_utc": "2026-11-12T17:00:00+00:00",
            "frozen_at_utc": "2026-11-12T18:00:00+00:00",
            "selection_closed": True, "all_systems_closed": True, "calibrators_closed": True,
            "hpo_nas_data_end": "2024-09-23", "power_pilot_data_end": "2024-12-30",
            "R": 6, "final_seed_ids": list(guard.FINAL_SEED_POOL[:6]),
            "power_result_sha256": sha256(b"synthetic-power-result").hexdigest(),
            "analysis_plan_sha256": sha256(b"synthetic-analysis-plan").hexdigest(),
            "calibration": {"start": "2026-09-01", "end_inclusive": "2026-10-30", "calendar_days": 60},
            "systems": systems}


class AmendedDataGuardTests(unittest.TestCase):
    def setUp(self):
        self.freeze = synthetic_freeze()
        self.binding = guard.bind_prospective_test(self.freeze, PROTOCOL, now_utc=NOW)

    def test_exposure_registry_does_not_relabel_either_known_test(self):
        record = json.loads((ROOT / "experiments/e2_protocol_amendment_20261112/exposure_registry.json").read_text())
        self.assertFalse(record["existing_virgin_holdout_accredited"])
        roles = {row["role"]: row for row in record["known_data"]}
        self.assertEqual(roles["historical_primary_608"]["calendar_days"], 608)
        self.assertEqual(roles["historical_secondary_60"]["calendar_days"], 60)
        self.assertFalse(roles["historical_primary_608"]["independent_test"])
        self.assertFalse(roles["historical_secondary_60"]["independent_test"])
        self.assertIsNone(record["prospective_test"]["start"])
        self.assertIsNone(record["prospective_test"]["end_inclusive"])
        manifest = ROOT / record["source_manifest"]["path"]
        self.assertEqual(sha256(manifest.read_bytes()).hexdigest(), record["source_manifest"]["sha256"])
        self.assertEqual(record["source_manifest"]["archive_count"], 75)
        self.assertEqual(record["market_payloads_read_by_this_audit"], 0)

    def test_first_binding_is_future_fixed_one_hundred_days_and_has_no_file_access(self):
        with patch.object(Path, "open", side_effect=AssertionError("No market or private file access")):
            bound = guard.bind_prospective_test(self.freeze, PROTOCOL, now_utc=NOW)
        self.assertEqual(bound["start"], "2026-11-13")
        self.assertEqual(bound["end_inclusive"], "2027-02-20")
        self.assertEqual(bound["calendar_days"], 100)
        self.assertFalse(bound["automatic_extension_allowed"])
        self.assertEqual(bound["minimum_eligible_cases_for_error_analysis"], 100)
        self.assertEqual(bound, guard.bind_new_cohort(self.freeze, PROTOCOL, now_utc=NOW))
        self.assertIn("not private artifact existence", bound["verification_scope"])

    def test_late_first_binding_is_rejected_but_a_prior_binding_can_be_read_after_maturity(self):
        for later in ("2026-11-13T00:00:00+00:00", LATER):
            with self.subTest(now=later), self.assertRaisesRegex(ValueError, "Late binding"):
                guard.bind_prospective_test(self.freeze, PROTOCOL, now_utc=later)
        self.assertEqual(guard.validate_prospective_binding(self.binding, PROTOCOL, now_utc=LATER), self.binding)

    def test_freeze_requires_registration_before_freeze_and_selection_before_freeze(self):
        changes = [
            {"frozen_at_utc": "2026-11-13T01:00:00+00:00"},
            {"amendment_registered_at_utc": "2026-11-12T18:30:00+00:00"},
            {"selection_closed_at_utc": "2026-11-12T18:30:00+00:00"},
            {"frozen_at_utc": "2026-11-12T18:00:00"},
            {"amendment_registered_at_utc": "2026-11-12T12:00:00-05:00"},
        ]
        for update in changes:
            value = {**self.freeze, **update}
            with self.subTest(update=update), self.assertRaises(ValueError):
                guard.bind_prospective_test(value, PROTOCOL, now_utc=NOW)

    def test_all_sixteen_exact_ids_closed_hash_bundles_and_nonplaceholder_R_are_required(self):
        bad = []
        value = deepcopy(self.freeze); value["systems"].pop(); bad.append(value)
        value = deepcopy(self.freeze); value["systems"][-1]["model_id"] = "Ridge"; bad.append(value)
        value = deepcopy(self.freeze); value["systems"][-1]["model_id"] = value["systems"][0]["model_id"]; bad.append(value)
        value = deepcopy(self.freeze); value["systems"][0]["weights_sha256"] = None; bad.append(value)
        value = deepcopy(self.freeze); value["systems"][0]["recipe_sha256"] = "not-a-hash"; bad.append(value)
        for flag in ("all_systems_closed", "selection_closed", "calibrators_closed"):
            value = deepcopy(self.freeze); value[flag] = False; bad.append(value)
        for R in (None, False, 5, 31, 6.0, "6"):
            value = deepcopy(self.freeze); value["R"] = R; bad.append(value)
        value = deepcopy(self.freeze); value["final_seed_ids"] = [271201, *guard.FINAL_SEED_POOL[1:6]]; bad.append(value)
        value = deepcopy(self.freeze); value["power_result_sha256"] = None; bad.append(value)
        value = deepcopy(self.freeze); value["analysis_plan_sha256"] = None; bad.append(value)
        for i, value in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                guard.bind_prospective_test(value, PROTOCOL, now_utc=NOW)

    def test_HPO_power_training_and_calibration_boundaries_are_not_interchangeable(self):
        bad = []
        value = deepcopy(self.freeze); value["hpo_nas_data_end"] = "2024-09-24"; bad.append(value)
        value = deepcopy(self.freeze); value["power_pilot_data_end"] = "2024-12-31"; bad.append(value)
        value = deepcopy(self.freeze); value["systems"][0]["training_cutoff"] = "2026-08-25"; bad.append(value)
        value = deepcopy(self.freeze); value["calibration"]["calendar_days"] = 59; bad.append(value)
        value = deepcopy(self.freeze); value["calibration"]["start"] = "2026-08-31"; bad.append(value)
        for i, value in enumerate(bad):
            with self.subTest(case=i), self.assertRaises(ValueError):
                guard.bind_prospective_test(value, PROTOCOL, now_utc=NOW)
        self.assertEqual(guard.require_data_role(["2024-09-23"], "hpo_nas"), ("2024-09-23",))
        self.assertEqual(guard.require_data_role(["2024-12-30"], "power_pilot"), ("2024-12-30",))
        self.assertEqual(guard.require_data_role(["2026-08-24"], "prospective_training"), ("2026-08-24",))
        self.assertEqual(guard.require_data_role(["2026-08-25", "2026-08-31"], "calibration_bridge"), ("2026-08-25", "2026-08-31"))
        for values, role in ((["2024-09-24"], "hpo_nas"), (["2024-12-31"], "power_pilot"),
                             (["2026-08-25"], "prospective_training"), (["2026-08-31"], "calibration"),
                             (["2026-10-31"], "calibration")):
            with self.subTest(role=role), self.assertRaises(ValueError):
                guard.require_data_role(values, role)

    def test_known_historical_cohorts_cannot_become_an_independent_test(self):
        for value in ("2025-01-01", "2026-08-31", "2026-09-01", "2026-10-30"):
            self.assertEqual(guard.require_data_role([value], "historical_exploratory"), (value,))
            with self.subTest(date=value), self.assertRaises(ValueError):
                guard.require_data_role([value], "independent_test", binding=self.binding,
                                        protocol_sha256=PROTOCOL, now_utc=NOW)
        with self.assertRaisesRegex(ValueError, "before global freeze"):
            guard.require_data_role(["2026-11-13"], "independent_test")

    def test_future_refit_weights_cannot_be_relabelled_as_a_historical_replay(self):
        for cutoff in (None, "2024-12-25", "2026-08-24"):
            with self.subTest(cutoff=cutoff), self.assertRaisesRegex(ValueError, "own weights/scalers"):
                guard.require_data_role(["2025-01-01"], "historical_replay", model_training_cutoff=cutoff)
        self.assertEqual(guard.require_data_role(["2025-01-01", "2026-08-31"], "historical_replay",
                                                model_training_cutoff="2024-12-24"),
                         ("2025-01-01", "2026-08-31"))

    def test_inference_bridge_requires_prior_binding_and_only_past_pretest_days(self):
        self.assertEqual(guard.require_data_role(["2026-10-31", "2026-11-11"], "inference_bridge",
            binding=self.binding, protocol_sha256=PROTOCOL, now_utc=NOW), ("2026-10-31", "2026-11-11"))
        for values, binding, now in ((["2026-10-31"], None, NOW),
                                     (["2026-10-30"], self.binding, NOW),
                                     (["2026-11-12"], self.binding, NOW),
                                     (["2026-11-13"], self.binding, LATER)):
            with self.subTest(dates=values), self.assertRaises(ValueError):
                guard.require_data_role(values, "inference_bridge", binding=binding,
                                        protocol_sha256=PROTOCOL, now_utc=now)

    def test_binding_tampering_protocol_mismatch_and_after_start_receipts_are_rejected(self):
        bad = deepcopy(self.binding); bad["start"] = "2026-11-14"
        with self.assertRaisesRegex(ValueError, "changed after sealing"):
            guard.validate_prospective_binding(bad, PROTOCOL, now_utc=LATER)
        with self.assertRaisesRegex(ValueError, "different registered protocol"):
            guard.validate_prospective_binding(self.binding, "a" * 64, now_utc=LATER)
        with self.assertRaisesRegex(ValueError, "different registered protocol"):
            guard.bind_prospective_test(self.freeze, "a" * 64, now_utc=NOW)
        late = deepcopy(self.binding); late["bound_at_utc"] = "2026-11-13T00:00:00+00:00"
        late.pop("binding_sha256"); late["binding_sha256"] = guard._metadata_sha256(late)
        with self.assertRaisesRegex(ValueError, "before its first day"):
            guard.validate_prospective_binding(late, PROTOCOL, now_utc=LATER)

    def test_quality_loss_reports_incomplete_without_extending_or_replacing_dates(self):
        start = date.fromisoformat(self.binding["start"])
        dates = [(start + timedelta(days=i)).isoformat() for i in range(100)]
        full = guard.audit_prospective_coverage(dates, self.binding, PROTOCOL, now_utc=LATER)
        self.assertEqual(full["status"], "complete")
        limited = guard.audit_prospective_coverage(dates[:-1], self.binding, PROTOCOL, now_utc=LATER)
        self.assertEqual(limited["status"], "incomplete_quality_coverage")
        self.assertEqual(limited["eligible_cases"], 99)
        self.assertFalse(limited["error_analysis_requirement_met"])
        self.assertFalse(limited["automatic_extension_allowed"])
        self.assertEqual(limited["fixed_end_inclusive"], self.binding["end_inclusive"])
        with self.assertRaisesRegex(ValueError, "entire fixed calendar matures"):
            guard.audit_prospective_coverage(dates, self.binding, PROTOCOL, now_utc=NOW)
        with self.assertRaises(ValueError):
            guard.audit_prospective_coverage([*dates[:-1], "2027-02-21"], self.binding, PROTOCOL, now_utc=LATER)

    def test_empty_unordered_duplicate_dates_and_unknown_roles_fail_explicitly(self):
        for dates in ([], "2024-01-01", ["2024-01-02", "2024-01-01"], ["2024-01-01", "2024-01-01"]):
            with self.subTest(dates=dates), self.assertRaises(ValueError):
                guard.require_data_role(dates, "development")
        with self.assertRaises(ValueError):
            guard.require_data_role(["2024-01-01"], "unregistered_role")


if __name__ == "__main__":
    unittest.main()
