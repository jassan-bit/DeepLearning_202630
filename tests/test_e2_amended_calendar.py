"""Chronology, common selection opportunities and one-time NAS dependency."""
from copy import deepcopy
from datetime import date, datetime, timezone
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_amended_calendar as amended


class AmendedCalendarTests(unittest.TestCase):
    def test_three_common_anchors_compile_exact_inner_boundaries_and_both_embargoes(self):
        expected = {
            "2024-01-01": ("2023-09-25", "2023-10-03", "2023-12-24"),
            "2024-07-01": ("2024-03-25", "2024-04-02", "2024-06-23"),
            "2024-10-01": ("2024-06-25", "2024-07-03", "2024-09-23"),
        }
        for origin, values in expected.items():
            with self.subTest(origin=origin):
                row = amended.boundaries(origin)
                self.assertEqual((row["inner_train_cutoff"], row["inner_validation_start"],
                                  row["inner_validation_end"]), values)
                self.assertEqual(row["inner_validation_calendar_days"], 83)
                self.assertEqual(row["refit_train_cutoff"], values[2])
                for prefix in ("inner_bridge", "bridge"):
                    start, end = (date.fromisoformat(row[prefix + suffix]) for suffix in ("_start", "_end"))
                    self.assertEqual((end - start).days + 1, 7)

    def test_latest_available_recipe_is_carried_forward_without_future_selection(self):
        for month in range(1, 11):
            expected = "2024-01-01" if month <= 6 else "2024-07-01" if month <= 9 else "2024-10-01"
            self.assertEqual(amended.selection_anchor(f"2024-{month:02d}-01"), expected)
        self.assertEqual(amended.selection_anchor("2025-01-01"), "2024-10-01")
        self.assertEqual(amended.selection_anchor("2026-11-21"), "2024-10-01")
        with self.assertRaises(ValueError):
            amended.selection_anchor("2023-12-31")
        with self.assertRaises(ValueError):
            amended.validate_forecast_context("2024-06-01", "2024-07-01", "2024-05-24")
        with self.assertRaises(ValueError):
            amended.validate_forecast_context("2024-10-01", "2024-07-01", "2024-09-23")

    def test_training_labels_must_precede_the_seven_day_target_embargo(self):
        accepted = amended.validate_forecast_context("2024-07-01", "2024-07-01", "2024-06-23")
        self.assertEqual(accepted["maximum_train_label_end"], "2024-06-23")
        with self.assertRaisesRegex(ValueError, "embargo"):
            amended.validate_forecast_context("2024-07-01", "2024-07-01", "2024-06-24")
        with self.assertRaisesRegex(ValueError, "embargo"):
            amended.validate_forecast_context("2024-07-01", "2024-07-01", "2024-07-01")

    def test_ten_monthly_outer_blocks_are_disjoint_and_preserve_leap_day(self):
        compiled = amended.compile_calendar()
        rows = compiled["outer_plan_rows"]
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[1]["end"], "2024-02-29")
        self.assertEqual(rows[1]["calendar_days"], 29)
        self.assertEqual(sum(row["calendar_days"] for row in rows), 305)
        for previous, current in zip(rows, rows[1:]):
            self.assertEqual((date.fromisoformat(current["origin"]) - date.fromisoformat(previous["end"])).days, 1)
        self.assertTrue(all(not row["repeat_HPO"] and not row["repeat_NAS"] for row in rows))

    def test_only_the_initial_tide_search_can_define_later_architecture(self):
        initial = amended.validate_nas_dependency("2024-01-01")
        self.assertEqual(initial["body_for_hpo"], "registered_reference_architecture")
        for anchor in ("2024-07-01", "2024-10-01"):
            with self.subTest(anchor=anchor):
                with self.assertRaises(ValueError):
                    amended.validate_nas_dependency(anchor)
                with self.assertRaises(ValueError):
                    amended.validate_nas_dependency(anchor, winner_origin=anchor,
                        winner_architecture_id="own-later-winner")
                result = amended.validate_nas_dependency(anchor, winner_origin="2024-01-01",
                    winner_architecture_id="TIDE-D7-NAS-v1:reference-or-valid-finalist")
                self.assertFalse(result["repeat_nas"])
        with self.assertRaises(ValueError):
            amended.validate_nas_dependency("2024-01-01", winner_origin="2024-01-01",
                winner_architecture_id="winner-before-reference-search")

    def test_power_pilot_is_disjoint_from_selection_and_excludes_the_removed_hybrid(self):
        compiled = amended.compile_calendar()
        pilot = compiled["power_pilot"]
        self.assertEqual((pilot["origin"], pilot["end"], pilot["calendar_days"]),
                         ("2024-11-08", "2024-12-30", 53))
        self.assertEqual(pilot["train_label_cutoff"], "2024-10-31")
        self.assertEqual(pilot["models"], ["TimeMixer", "PatchTST", "GRU32"])
        self.assertEqual(pilot["seeds"], list(range(271701, 271706)))
        self.assertEqual(pilot["logical_fits"], 15)
        self.assertGreater(pilot["origin"], max(row["inner_validation_end"] for row in compiled["hpo_plan_rows"]))
        self.assertFalse(pilot["labels_used_for_HPO_or_NAS_selection"])

    def test_inventory_is_not_eleven_searches_or_an_observed_progress_counter(self):
        counts = amended.selection_inventory()
        self.assertEqual(counts["neural_hpo_logical_fits"], 1404)
        self.assertEqual(counts["nas_phase1_logical_fits"], 1000)
        self.assertEqual(counts["nas_phase2_logical_fit_upper_bound"], 205)
        self.assertEqual(counts["neural_selection_logical_fit_upper_bound"], 2609)
        self.assertEqual(counts["SAE_additional_auxiliary_optimizer_stages"], 216)
        self.assertEqual(counts["neural_selection_optimizer_stage_upper_bound"], 2825)
        self.assertEqual(counts["ARIMA_candidate_fits"], 54)
        self.assertIsNone(counts["observed_completed_fits"])
        self.assertFalse(counts["compatible_reuse_deducted"])
        self.assertTrue(counts["inventory_does_not_authorize_k_change"])
        for invalid in (True, 9, 41, 40.0):
            with self.subTest(k=invalid), self.assertRaises(ValueError):
                amended.selection_inventory(invalid)

    def test_supplied_protocol_must_match_registered_calendar_before_compilation(self):
        self.assertEqual(amended.compile_calendar({"calendar": amended.DEFAULT_CALENDAR}), amended.compile_calendar())
        for mutate in (
            lambda c: c.update(hpo_anchors=list(amended.OUTER_ORIGINS) + ["2025-01-01"]),
            lambda c: c.update(embargo_days=1),
            lambda c: c["inner"].update(validation_end_offset_days=-1),
            lambda c: c["power_pilot"].update(origin="2024-09-01"),
            lambda c: c["power_pilot"].update(models=["TimeMixer", "CNN-1D + LSTM + atención temporal", "GRU32"]),
        ):
            config = deepcopy(amended.DEFAULT_CALENDAR)
            mutate(config)
            with self.assertRaises(ValueError):
                amended.compile_calendar({"calendar": config})

    def test_known_historical_evaluation_is_not_relabelled_as_independent_test(self):
        result = amended.compile_calendar()
        final = result["historical_final"]
        self.assertEqual((final["origin"], final["end"]), ("2025-01-01", "2026-08-31"))
        self.assertEqual(final["selection_anchor"], "2024-10-01")
        self.assertIn("exploratory", final["data_role"])
        self.assertFalse(result["known_dates_become_unseen_by_this_amendment"])
        self.assertFalse(result["final_independent_cohort_defined_here"])
        self.assertFalse(result["R_fixed_here"])
        self.assertFalse(result["market_data_read"])

    def test_strict_dates_package_import_and_cli_need_no_market_dependency(self):
        for invalid in ("2024-1-01", "2024-02-30", "2024-01-01T00:00:00Z", 20240101,
                        datetime(2024, 1, 1, tzinfo=timezone.utc)):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                amended.boundaries(invalid)
        package = importlib.import_module("scripts.e2_amended_calendar")
        self.assertEqual(package.selection_anchor("2024-08-15"), "2024-07-01")
        result = subprocess.run([sys.executable, str(ROOT / "scripts/e2_amended_calendar.py")],
                                check=True, capture_output=True, text=True)
        self.assertFalse(json.loads(result.stdout)["market_data_read"])


if __name__ == "__main__":
    unittest.main()
