"""Stage-prefix tests use artificial derivative files, never market observations."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_amended_pilot_data as pilot
import e2_development_data as historical
from e2_temporal_training import prepare_b
from test_e2_development_data import registered_cache


class AmendedPilotAdmissionTests(unittest.TestCase):
    def test_each_stage_ignores_poisoned_numerical_cells_after_its_cutoff(self):
        expected = {"noise": (266, 8), "hpo": (632, 8),
                    "power_fit": (670, 60), "power_score": (730, 0)}
        for stage, config in pilot.STAGES.items():
            with self.subTest(stage=stage), registered_cache(config["numerical_end"]) as (root, raw):
                panel, receipt = pilot.load_pilot_panel(root, stage)
                self.assertEqual((receipt["decoded_calendar_rows"], receipt["placeholder_rows"]), expected[stage])
                self.assertEqual(receipt["output_calendar_end"], config["calendar_end"])
                self.assertFalse(receipt["source_numerical_values_after_stage_cutoff_decoded"])
                self.assertTrue(receipt["source_bytes_hashed_in_full"])
                for parsing in receipt["parsing"].values():
                    self.assertEqual(parsing["date_fields_inspected"], 731)
                    self.assertFalse(parsing["numerical_values_after_admission_decoded"])
                count = receipt["placeholder_rows"]
                if count:
                    self.assertTrue(np.isnan(panel.features[-count:]).all())
                    self.assertTrue(np.isnan(panel.targets[-count:]).all())
                    self.assertFalse(panel.quality[-count:].any())
                for name, body in raw.items():
                    self.assertEqual((root/name).read_bytes(), body)

    def test_october_hpo_has_real_validation_and_no_outer_labels(self):
        with registered_cache("2024-09-23") as (root, _):
            panel, receipt = pilot.load_pilot_panel(root, "hpo")
            prepared = prepare_b(panel, "2024-10-01", "2024-10-01")
            self.assertEqual(str(prepared.inner_train.dates[-1].date()), "2024-06-25")
            self.assertEqual(str(prepared.inner_validation.dates[0].date()), "2024-07-03")
            self.assertEqual(str(prepared.inner_validation.dates[-1].date()), "2024-09-23")
            self.assertEqual(len(prepared.inner_validation.dates), 83)
            self.assertEqual(sum(prepared.outer_plan.eligible), 0)
            self.assertEqual(receipt["stage"], "hpo")

    def test_power_fit_prepares_all_training_tensors_without_any_pilot_values(self):
        with registered_cache("2024-10-31") as (root, _):
            panel, receipt = pilot.load_pilot_panel(root, "power_fit")
            prepared = prepare_b(panel, "2024-11-08", "2024-12-30")
            self.assertEqual(str(prepared.refit_train.dates[-1].date()), "2024-10-31")
            self.assertEqual(len(prepared.refit_train.dates), 655)
            self.assertTrue(np.isfinite(prepared.refit_train.x.numpy()).all())
            self.assertTrue(np.isfinite(prepared.refit_train.y.numpy()).all())
            self.assertEqual(len(prepared.outer_plan.evaluation_dates), 53)
            self.assertEqual(sum(prepared.outer_plan.eligible), 0)
            mask = panel.calendar >= pd.Timestamp("2024-11-01", tz="UTC")
            self.assertTrue(np.isnan(panel.targets[mask]).all())
            self.assertTrue(np.isnan(panel.features[mask]).all())
            self.assertEqual(receipt["numerical_end"], "2024-10-31")

    def test_scoring_admission_preserves_the_sealed_training_split_signature(self):
        with registered_cache("2024-12-30") as (root, _):
            fit_panel, _ = pilot.load_pilot_panel(root, "power_fit")
            score_panel, receipt = pilot.load_pilot_panel(root, "power_score")
            fit = prepare_b(fit_panel, "2024-11-08", "2024-12-30")
            score = prepare_b(score_panel, "2024-11-08", "2024-12-30")
            self.assertEqual(fit.refit_train.signature, score.refit_train.signature)
            self.assertEqual(fit.refit_train.scaler.record(), score.refit_train.scaler.record())
            self.assertNotEqual(fit.signature, score.signature)
            self.assertEqual(sum(score.outer_plan.eligible), 53)
            self.assertTrue(receipt["power_score_requires_prior_runner_freeze"])
            np.testing.assert_array_equal(fit.refit_train.x.numpy(), score.refit_train.x.numpy())

    def test_unknown_stage_fails_before_any_source_access(self):
        for stage in (None, True, "final", "2024-12-31", ""):
            with self.subTest(stage=stage), patch.object(Path, "read_bytes", side_effect=AssertionError("source accessed")):
                with self.assertRaises(ValueError):
                    pilot.load_pilot_panel("unused", stage)

    def test_changed_source_hashes_and_wrong_targets_are_not_admitted(self):
        with registered_cache("2024-09-23") as (root, _):
            path = root/historical.TARGET_PATH
            path.write_bytes(path.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                pilot.load_pilot_panel(root, "hpo")
        with registered_cache("2024-09-23", corrupt_target=True) as (root, _):
            with self.assertRaisesRegex(ValueError, "population SD"):
                pilot.load_pilot_panel(root, "hpo")

    def test_wrong_protocol_mapping_and_symlink_root_are_refused(self):
        with registered_cache("2024-09-23") as (root, _):
            with patch.object(historical, "_read_registered", return_value=(b'{"sources": {}}', "a"*64)):
                with self.assertRaisesRegex(ValueError, "exact pinned"):
                    pilot.load_pilot_panel(root, "hpo")
            alias = root.parent/(root.name+"-alias")
            alias.symlink_to(root, target_is_directory=True)
            try:
                with self.assertRaisesRegex(ValueError, "symlink"):
                    pilot.load_pilot_panel(alias, "hpo")
            finally:
                alias.unlink()


if __name__ == "__main__":
    unittest.main()
