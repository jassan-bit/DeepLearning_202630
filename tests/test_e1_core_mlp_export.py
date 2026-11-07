"""Synthetic integrity guards for public export; these tests never fit a model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / "scripts/export_e1_core_mlp_extension.py"
SPEC = importlib.util.spec_from_file_location("core_mlp_export_under_test", MODULE)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class CoreMLPPublicExportIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.private = self.root / "source", self.root / "private"
        self.source.mkdir()
        self.private.mkdir()
        self.revision = "a" * 40
        self.design = {"models": list(exporter.MODELS), "history_days": 7}
        self.write_source(exporter.DESIGN, self.design)
        self.write_source("scripts/scientific_fixture.py", "# Synthetic source, no model.\n")
        self.registration = {"study_id": exporter.STUDY, "source_commit": self.revision,
            "registered_at_utc": self.time(1), "design": self.design,
            "design_sha256": exporter.sha(self.source / exporter.DESIGN),
            "sources_sha256": {"scripts/scientific_fixture.py": exporter.sha(self.source / "scripts/scientific_fixture.py")},
            "publication_verification": {"sha": self.revision, "observed_at_utc": self.time(0)},
            "new_fits_before_registration": 0, "reserve_price_payloads_read": 0}
        self.signature = hashlib.sha256(exporter.encode(self.registration)).hexdigest()
        self.registration["registration_signature"] = self.signature
        self.write("registration.json", self.registration)
        self.selection = {"registration_signature": self.signature, "selected_at_utc": self.time(3),
            "recipes": {name: {"model": name} for name in exporter.MODELS}, "baseline_model": "GRU",
            "trajectories": 48, "fit_stages": 960, "reserve_price_payloads_read": 0,
            "artifact_sha256": self.payload("hpo")}
        self.write("hpo/selection.json", self.selection)
        self.pilot = {"registration_signature": self.signature, "completed_at_utc": self.time(5),
            "selection_sha256": exporter.sha(self.private / "hpo/selection.json"),
            "trajectories": 30, "fit_stages": 60, "reserve_price_payloads_read": 0,
            "artifact_sha256": self.payload("pilot")}
        self.write("pilot/complete.json", self.pilot)
        self.plan = {"registration_signature": self.signature, "planned_at_utc": self.time(6),
            "pilot_complete_sha256": exporter.sha(self.private / "pilot/complete.json"),
            "selection_sha256": exporter.sha(self.private / "hpo/selection.json"),
            "R": 465, "R_final": 465, "R_required": 10, "seeds": list(range(8101, 8566)),
            "reserve_price_payloads_read": 0}
        self.write("plan.json", self.plan)
        self.publication = {"registration_signature": self.signature, "registered_at_utc": self.time(8),
            "plan_sha256": exporter.sha(self.private / "plan.json"), "published_plan_commit": "b"*40,
            "publication_verification": {"sha": "b"*40, "observed_at_utc": self.time(7),
                "file_sha256": exporter.sha(self.private / "plan.json")},
            "new_definitive_fits_before_plan_publication": 0}
        self.publication["receipt_signature"] = hashlib.sha256(exporter.encode(self.publication)).hexdigest()
        self.write("plan_publication.json", self.publication)
        self.start = {"registration_signature": self.signature, "at_utc": self.time(9),
            "plan_sha256": exporter.sha(self.private / "plan.json"),
            "plan_publication_sha256": exporter.sha(self.private / "plan_publication.json"),
            "planned_trajectories": 1395}
        self.write("final_started.json", self.start)
        jobs = [{"model": model, "seed": seed, "study_id": exporter.STUDY, "execution_id": self.signature}
                for model in exporter.MODELS for seed in self.plan["seeds"]]
        self.final = {"registration_signature": self.signature, "all_weights_frozen_at_utc": self.time(11),
            "plan_sha256": exporter.sha(self.private / "plan.json"),
            "selection_sha256": exporter.sha(self.private / "hpo/selection.json"),
            "R": 465, "models": 1395, "fit_stages": 2790, "jobs": jobs,
            "reserve_price_payloads_read": 0, "artifact_sha256": self.payload("final")}
        self.write("final_models_manifest.json", self.final)
        self.gate = {"checked_at_utc": self.time(12), "all_final_models_verified": 1395,
            "plan_sha256": exporter.sha(self.private / "plan.json"),
            "final_models_manifest_sha256": exporter.sha(self.private / "final_models_manifest.json")}
        self.write("score/frozen_access_gate.json", self.gate)
        self.score = {"registration_signature": self.signature, "completed_at_utc": self.time(13),
            "final_models_manifest_sha256": exporter.sha(self.private / "final_models_manifest.json"),
            "trajectories": 1395, "dates": 60, "prediction_rows": 83700,
            "scoring_passes_per_checkpoint": 1, "historical_exploratory_not_confirmatory": True,
            "artifact_sha256": self.payload("score")}
        self.write("score/complete.json", self.score)
        self.analysis = {"registration_signature": self.signature, "analyzed_at_utc": self.time(14),
            "score_complete_sha256": exporter.sha(self.private / "score/complete.json"),
            "artifact_sha256": self.payload("analysis")}
        self.write("analysis/manifest.json", self.analysis)
        self.workers = []
        for phase, second in (("hpo", 2), ("pilot", 4), ("final", 10)):
            for index in range(3):
                worker = {"phase": phase, "start_method": "spawn", "runtime": {"at_utc": self.time(second),
                    "actual_threads": 1, "actual_interop_threads": 1, "deterministic_algorithms": True}}
                self.workers.append(worker)
                self.write(f"worker_runtime/{phase}_{index}.json", worker)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def time(second):
        return f"2020-01-01T00:00:{second:02d}+00:00"

    def write_source(self, name, value):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(exporter.encode(value) if not isinstance(value, str) else value.encode())

    def write(self, name, value):
        path = self.private / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(exporter.encode(value))

    def payload(self, label):
        path = self.private / "synthetic_payloads" / (label + ".txt")
        path.parent.mkdir(exist_ok=True)
        path.write_text("Synthetic test payload: never trained or evaluated.\n")
        return {path.relative_to(self.private).as_posix(): exporter.sha(path)}

    def verify(self):
        with patch.object(exporter, "ROOT", self.source), patch.object(exporter.subprocess, "check_output",
                side_effect=[self.revision + "\n", ""]):
            return exporter.verify(self.source, self.private)

    def test_complete_new_core_closure_and_observed_chronology_validate(self):
        record, stages, jobs, workers, chronology = self.verify()
        self.assertEqual(record["registration_signature"], self.signature)
        self.assertEqual(len(jobs), 1395)
        self.assertEqual(set(jobs.model), {"GRU", "LSTM", "MLP"})
        self.assertEqual(len(workers), 9)
        self.assertLess(exporter.moment(chronology["plan_publication_observed"]),
                        exporter.moment(chronology["final_started"]))

    def test_incomplete_final_closure_cannot_be_exported(self):
        self.final["models"] = 1394
        self.write("final_models_manifest.json", self.final)
        with self.assertRaises(ValueError):
            self.verify()

    def test_plan_floor_is_not_satisfied_by_an_earlier_smaller_experiment(self):
        self.plan["R"] = self.plan["R_final"] = 10
        self.write("plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "minimum465"):
            self.verify()

    def test_mixed_execution_and_tcn_rows_cannot_enter_the_new_core(self):
        frame = pd.DataFrame(self.final["jobs"])
        frame.loc[0, "execution_id"] = "historical-other-execution"
        with self.assertRaisesRegex(ValueError, "Mixed historical"):
            exporter.identity(frame, self.signature)
        frame.loc[0, "model"] = "TCN"
        with self.assertRaisesRegex(ValueError, "exact new core"):
            exporter.panel(frame, self.plan["seeds"])

    def test_retroactive_source_or_planning_publication_is_rejected(self):
        self.publication["publication_verification"]["observed_at_utc"] = self.time(15)
        with self.assertRaisesRegex(ValueError, "chronology"):
            exporter.verify_chronology(self.registration, self.selection, self.pilot, self.plan,
                self.publication, self.start, self.final, self.gate, self.score, self.analysis, self.workers)

    def test_private_numeric_payloads_are_removed_or_rejected(self):
        runs = pd.DataFrame([{"model": "MLP", "seed": 8101, "rmse": .1,
                              "mase_scale": .002, "y_scale": .03, "y_pred": .01}])
        safe = exporter.projected_runs(runs)
        self.assertEqual(set(safe), {"model", "seed", "rmse"})
        for key in ("y_true", "outcome_day", "weights", "mase_train_scale"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                exporter.public_frame(pd.DataFrame({key: [1]}))
        receipt = {**self.registration, "publication_verification": {
            **self.registration["publication_verification"], "arbitrary_api_response": {"email": "private@example.test"}}}
        projected = exporter.projected_record(receipt, self.private / "registration.json")
        self.assertNotIn("arbitrary_api_response", projected["publication_verification"])
        self.assertTrue(projected["public_projection"])
        self.assertEqual(projected["original_private_record_sha256"], exporter.sha(self.private / "registration.json"))

    def test_refit_cannot_gain_a_validation_curve_from_the_reserve(self):
        curve = pd.DataFrame({"epoch": [1], "stage": ["refit"], "optimization_mse_scaled": [.1],
            "train_mse_scaled_eval": [.1], "validation_mse_scaled_eval": [np.nan],
            "best_epoch_so_far": [1], "learning_rate": [.001]})
        exporter.validate_curve(curve, "refit", 1)
        curve.loc[0, "validation_mse_scaled_eval"] = .09
        with self.assertRaisesRegex(ValueError, "no reserve validation"):
            exporter.validate_curve(curve, "refit", 1)

    def test_tampered_closed_artifacts_and_unsafe_output_paths_fail(self):
        (self.private / "synthetic_payloads/final.txt").write_text("Changed after closure")
        with self.assertRaisesRegex(ValueError, "Closed private artifact changed"):
            self.verify()
        with self.assertRaises(ValueError):
            exporter.destination(self.private / "public", self.source, self.private)
        target = self.root / "empty_public"
        target.mkdir()
        link = self.root / "linked_public"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            exporter.destination(link, self.source, self.private)


if __name__ == "__main__":
    unittest.main()
