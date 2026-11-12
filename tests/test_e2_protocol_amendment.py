"""Registered schedule, model scope, chronology and budget boundaries."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_protocol_amendment as amendment


class ProtocolAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.protocol = amendment.read_protocol()

    def test_registered_scope_and_temporal_workload_are_explicit(self):
        self.assertEqual(len(self.protocol["systems"]), 16)
        self.assertEqual(len(self.protocol["neural_systems"]), 13)
        self.assertEqual(self.protocol["nas"]["models"], ["TiDE"])
        self.assertEqual(self.protocol["workload_upper_bounds"]["neural_selection_logical_fit_upper_bound"], 2609)
        self.assertIsNone(self.protocol["power"]["R"])
        self.assertIsNone(self.protocol["budget"]["remaining_seconds"])
        self.assertEqual(self.protocol["data_roles"]["independent_final"]["calendar_days"], 100)

    def test_no_eleven_searches_budget_reset_historical_test_alias_or_copied_R(self):
        mutations = [
            lambda p: p["calendar"].update(hpo_anchors=p["calendar"]["outer_development_origins"] + ["2025-01-01"]),
            lambda p: p["nas"].update(models=["TiDE", "TimesNet"]),
            lambda p: p["hpo"].update(selection_scope="pool all anchors before earlier forecasts"),
            lambda p: p["budget"].update(authorization_history_total_hours=[64]),
            lambda p: p["budget"].update(remaining_seconds=230400),
            lambda p: p["power"].update(R=309),
            lambda p: p["data_roles"]["historical_608"].update(already_evaluated=False),
            lambda p: p["data_roles"]["historical_60"].update(already_evaluated=False),
            lambda p: p["data_roles"]["independent_final"].update(start="2025-01-01"),
            lambda p: p["inference"].update(family_size=30),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                candidate = deepcopy(self.protocol)
                mutate(candidate)
                with self.assertRaises(ValueError):
                    amendment.validate_protocol(candidate)

    def test_NAS_and_HPO_are_confined_to_the_declared_common_schedule(self):
        result = amendment.validate_stage_context("TiDE", "nas1", "2024-01-01")
        self.assertFalse(result["execution_authorized_by_this_metadata_check"])
        for args in [("TiDE", "nas2", "2024-07-01"), ("TimesNet", "nas1", "2024-01-01"),
                     ("TSMixer", "hpo", "2025-01-01"), ("Ridge", "classical_hpo", "2024-01-01")]:
            with self.subTest(context=args), self.assertRaises(ValueError):
                amendment.validate_stage_context(*args)

    def test_later_TiDE_HPO_requires_the_initial_architecture_receipt(self):
        with self.assertRaises(ValueError):
            amendment.validate_stage_context("TiDE", "hpo", "2024-07-01")
        valid = amendment.validate_stage_context("TiDE", "hpo", "2024-07-01",
            architecture_source_origin="2024-01-01", architecture_id="fixture-frozen-January-winner")
        self.assertTrue(valid["valid_context"])
        with self.assertRaises(ValueError):
            amendment.validate_stage_context("TiDE", "hpo", "2024-07-01",
                architecture_source_origin="2024-10-01", architecture_id="fixture-future-winner")

    def test_monthly_refits_cannot_use_future_selection_or_embargo_labels(self):
        amendment.validate_stage_context("PatchTST", "outer_refit", "2024-03-01",
            selected_at="2024-01-01", train_label_end="2024-02-22")
        with self.assertRaises(ValueError):
            amendment.validate_stage_context("PatchTST", "outer_refit", "2024-03-01",
                selected_at="2024-10-01", train_label_end="2024-02-22")
        with self.assertRaises(ValueError):
            amendment.validate_stage_context("PatchTST", "outer_refit", "2024-03-01",
                selected_at="2024-01-01", train_label_end="2024-02-23")
        amendment.validate_stage_context("GRU32", "historical_refit", "2025-01-01",
            selected_at="2024-10-01", train_label_end="2024-12-24")

    def test_primary_family_has_the_margin_null_and_no_pseudo_independent_baseline_seeds(self):
        contrasts = self.protocol["inference"]["contrasts"]
        self.assertEqual(len({r["contrast_id"] for r in contrasts}), 20)
        self.assertTrue(all(r["baseline"] in ("GRU32", "Persistence") for r in contrasts))
        self.assertTrue(all(r["null"] == "gain <= delta" for r in contrasts))
        self.assertEqual(self.protocol["inference"]["planning_true_gain"], 2*amendment.DELTA)
        self.assertTrue(self.protocol["metrics"]["deterministic_baselines"].startswith("One deterministic trajectory"))

    def test_registration_receipt_verifies_actual_source_bytes_but_does_not_start_execution(self):
        result = amendment.verify_registration()
        self.assertFalse(result["execution_ready"])
        self.assertFalse(result["market_training_started"])
        self.assertGreaterEqual(result["source_bytes_verified"], 8)
        self.assertTrue(any(g["id"] == "prospective_global_freeze" for g in result["pending_gates"]))

    def test_required_git_verification_refuses_a_missing_or_unavailable_registration(self):
        with patch.object(amendment, "_git", side_effect=subprocess.CalledProcessError(1, "git")):
            with self.assertRaises(ValueError):
                amendment.git_registration(required=True)


if __name__ == "__main__":
    unittest.main()
