"""Metadata-only audit checks with synthetic records; no models or market data."""
from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
import io
import json
from pathlib import Path
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_e2_f2a_training_evidence as audit


CHECKPOINT = "a" * 64
STATE = "b" * 64


def phase_fixture(*, auxiliary=False, fixed=False):
    train = [0.6, 0.4, 0.3] if not auxiliary else [6.0, 4.0, 3.0]
    valid = [0.7, 0.5, 0.8] if not auxiliary else [7.0, 5.0, 8.0]
    return {
        "best_epoch": 3 if fixed else 2,
        "fixed_epochs": 3 if fixed else None,
        "epochs_completed": 3,
        "optimizer_steps": 9,
        "selected_state_sha256": STATE,
        "stop_reason": "fixed_epochs" if fixed else "patience",
        "selection_metric": "rmse_original",
        "checkpoint_receipts": {
            "best": {"sha256": "c" * 64, "path": "checkpoints/best.pt"},
            "latest": {"sha256": "d" * 64, "path": "checkpoints/latest.pt"},
        },
        "logs": [
            {
                "epoch": i + 1,
                "train_mse_scaled": (value / (1.0 if auxiliary else 0.1)) ** 2,
                "train_rmse_original": value,
                "validation_score": None if fixed else valid[i],
                "examples_seen": 32,
                "optimizer_steps_total": (i + 1) * 3,
                "batch_sizes": [16, 16],
            }
            for i, value in enumerate(train)
        ],
    }


def fit_fixture(*, sae=False, fixed=False):
    name = "LSTM-autoencoder con predictor" if sae else "TSMixer"
    phases = {"supervised": phase_fixture(fixed=fixed)}
    if sae:
        phases.update(ae1=phase_fixture(auxiliary=True, fixed=fixed),
                      ae2=phase_fixture(auxiliary=True, fixed=fixed))
    fit = {
        "status": "completed", "model_id": name, "seed": 271801 if fixed else 271201,
        "phase_records": phases,
        "phase_best_epochs": {key: value["best_epoch"] for key, value in phases.items()},
        "selected_checkpoint": {"path": "selected_model.pt", "sha256": CHECKPOINT},
        "request_sha256": "e" * 64,
        "scaler": {"target_scale": 0.1},
        "training_performed": True,
        "synthetic": False,
        "test_only_cpu_execution": False,
    }
    request = {
        "name": name, "seed": fit["seed"],
        "trial": {"trial_id": "synthetic:hpo-008", "recipe_id": "synthetic"},
        "train_signature": "f" * 64,
        "validation_signature": None if fixed else "0" * 64,
        "fixed_epochs": {key: 3 for key in phases} if fixed else None,
        "config": {"batch_size": 16},
    }
    fit["request_sha256"] = audit.sha(audit.canonical(request))
    return fit, request


def calibration_fixture(fit):
    return {
        "model": fit["model_id"], "seed": fit["seed"],
        "checkpoint_sha256": CHECKPOINT,
        "dates": ["2024-11-08", "2024-11-09"],
        "prediction_original": [0.3, 0.6],
        "target_original": [0.0, 0.2],
        "same_predictor_as_final": True, "used_for_HPO": False,
    }


def cached_fixture():
    trial = {"trial_id": "synthetic:hpo-008", "recipe_id": "synthetic"}
    job = {
        "model_id": "ResNet-1D", "seed": 271201, "fold_origin": "2024-07-01",
        "parent_trial_id": trial["trial_id"], "trial": trial, "configuration_id": "C03",
    }
    start = date.fromisoformat(job["fold_origin"]) - timedelta(days=90)
    days = [(start + timedelta(days=i)).isoformat() for i in range(83)]
    original_dates = [value + "T00:00:00+00:00" for value in days]
    predictions = [0.2 + i * 0.001 for i in range(83)]
    truths = {value: 0.1 + i * 0.001 for i, value in enumerate(days)}
    payload = {"model": job["model_id"], "trial": trial, "architecture": None, "stage": "hpo"}
    payload_sha = audit.sha(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    outcome = {
        "status": "completed", "fatal": False,
        "prediction_dates": original_dates, "predictions_original": predictions,
        "resource_use": {
            "identity": {"model": job["model_id"], "origin": job["fold_origin"],
                         "candidate_id": "hpo:" + trial["trial_id"], "payload_sha256": payload_sha,
                         "seed": job["seed"], "stage": "hpo"},
            "optimizer_steps": 144, "optimizer_steps_unknown": False,
            "partial_optimizer_update_may_have_occurred": False,
            "phase_records": {"supervised": {"best_epoch": 28, "epochs_completed": 48,
                                               "fixed_epochs": None, "selected_state_sha256": STATE}},
        },
    }
    source = {"job_id": "1" * 64, "outcome": outcome,
              "outcome_sha256": audit.sha(audit.canonical(outcome))}
    source["sha256"] = audit.sha(audit.canonical(source))
    admission = {
        "job_id": source["job_id"], "result_sha256": source["sha256"],
        "reuse_approved": True, "weights_reusable": False, "best_epoch": 28,
        "model_id": job["model_id"], "origin": job["fold_origin"], "seed": job["seed"],
        "parent_trial_id": trial["trial_id"], "configuration_id": job["configuration_id"],
        "candidate_payload_sha256": payload_sha,
        "validation_prediction_sha256": audit.sha(audit.canonical(predictions)),
        "validation_dates_sha256": audit.sha(audit.canonical(original_dates)),
    }
    return job, source, admission, truths


class ArtifactIntegrityTests(unittest.TestCase):
    def test_original_bytes_accept_matching_size_and_digest(self):
        raw = b'{"synthetic":true}\n'
        descriptor = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
        audit.verify_bytes(raw, descriptor, "synthetic metadata")

    def test_digest_and_size_changes_are_rejected(self):
        raw = b'{"synthetic":true}\n'
        descriptor = {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
        mutations = [
            (raw.replace(b"true", b"evil"), descriptor),
            (raw, {**descriptor, "bytes": len(raw) + 1}),
            (raw, {**descriptor, "sha256": "0" * 64}),
        ]
        for payload, expected in mutations:
            with self.subTest(payload=payload, descriptor=expected), self.assertRaises(ValueError):
                audit.verify_bytes(payload, expected, "synthetic metadata")


class OriginalCurveTests(unittest.TestCase):
    def test_gap_uses_selected_checkpoint_epoch_instead_of_terminal_epoch(self):
        fit, request = fit_fixture()
        curves, auxiliary, summary = audit.extract_fit_rows(
            fit, request, stage="hpo", fold_origin="2024-07-01")
        self.assertEqual([row["epoch"] for row in curves], [1, 2, 3])
        self.assertEqual(auxiliary, [])
        self.assertAlmostEqual(summary["selected_train_rmse_original"], 0.4)
        self.assertAlmostEqual(summary["terminal_train_rmse_original"], 0.3)
        self.assertAlmostEqual(summary["selected_validation_rmse_original"], 0.5)
        self.assertAlmostEqual(summary["validation_minus_train_rmse_original"], 0.1)
        self.assertEqual(summary["curve_availability"], "complete_original_epoch_logs")

    def test_sae_reconstruction_metrics_never_enter_target_gap(self):
        fit, request = fit_fixture(sae=True)
        curves, auxiliary, summary = audit.extract_fit_rows(
            fit, request, stage="hpo", fold_origin="2024-07-01")
        self.assertEqual(len(curves), 3)
        self.assertEqual(len(auxiliary), 6)
        self.assertEqual({row["metric_scope"] for row in auxiliary},
                         {"auxiliary_reconstruction"})
        self.assertAlmostEqual(summary["selected_train_rmse_original"], 0.4)
        self.assertAlmostEqual(summary["selected_validation_rmse_original"], 0.5)
        self.assertAlmostEqual(summary["validation_minus_train_rmse_original"], 0.1)

    def test_missing_best_epoch_is_rejected(self):
        fit, request = fit_fixture()
        fit["phase_records"]["supervised"]["best_epoch"] = 8
        with self.assertRaises(ValueError):
            audit.extract_fit_rows(fit, request, stage="hpo", fold_origin="2024-07-01")

    def test_request_metadata_must_match_original_fitting_identity(self):
        fit, request = fit_fixture()
        request["trial"]["trial_id"] = "different:selected-recipe"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            audit.extract_fit_rows(fit, request, stage="hpo", fold_origin="2024-07-01")

    def test_fit_and_request_model_or_seed_mismatch_is_rejected(self):
        original, request = fit_fixture()
        for change in ({"model_id": "GRU32"}, {"seed": 271202}):
            fit = {**original, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.extract_fit_rows(fit, request, stage="hpo", fold_origin="2024-07-01")

    def test_nonfinite_supervised_metric_is_rejected(self):
        original, request = fit_fixture()
        for field in ("train_mse_scaled", "train_rmse_original", "validation_score"):
            fit = deepcopy(original)
            fit["phase_records"]["supervised"]["logs"][1][field] = math.nan
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.extract_fit_rows(fit, request, stage="hpo", fold_origin="2024-07-01")


class FinalPredictorIdentityTests(unittest.TestCase):
    def test_fixed_epoch_fit_has_no_fabricated_validation_curve(self):
        fit, request = fit_fixture(fixed=True)
        curves, auxiliary, summary = audit.extract_fit_rows(fit, request, stage="final")
        self.assertEqual(auxiliary, [])
        self.assertTrue(all(row["validation_rmse_original"] is None for row in curves))
        self.assertIsNone(summary["selected_validation_rmse_original"])
        self.assertIsNone(summary["validation_minus_train_rmse_original"])
        self.assertAlmostEqual(summary["selected_train_rmse_original"], 0.3)

    def test_calibration_and_final_metrics_use_exact_predictor_identity(self):
        fit, request = fit_fixture(fixed=True)
        calibration = calibration_fixture(fit)
        _, _, summary = audit.extract_fit_rows(
            fit, request, stage="final", calibration=calibration, final_rmse=0.5)
        expected = math.sqrt((0.3 ** 2 + 0.4 ** 2) / 2)
        self.assertAlmostEqual(summary["calibration_rmse_original"], expected)
        self.assertAlmostEqual(summary["final_rmse_original"], 0.5)
        self.assertAlmostEqual(summary["final_minus_calibration_rmse_original"], 0.5 - expected)

    def test_calibration_predictor_or_role_mismatch_is_rejected(self):
        fit, request = fit_fixture(fixed=True)
        original = calibration_fixture(fit)
        mutations = [
            {"checkpoint_sha256": "9" * 64}, {"model": "GRU32"}, {"seed": 271802},
            {"same_predictor_as_final": False}, {"used_for_HPO": True},
        ]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.extract_fit_rows(fit, request, stage="final",
                                       calibration={**original, **change})

    def test_calibration_truncation_is_rejected(self):
        fit, request = fit_fixture(fixed=True)
        original = calibration_fixture(fit)
        for key in ("dates", "prediction_original", "target_original"):
            bad = deepcopy(original)
            bad[key] = bad[key][:-1]
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.extract_fit_rows(fit, request, stage="final", calibration=bad)


class CachedOriginalEvidenceTests(unittest.TestCase):
    def test_missing_curves_never_create_training_risk_or_gap(self):
        job, source, admission, truths = cached_fixture()
        summary = audit.extract_cached_hpo_summary(job, source, admission, truths)
        self.assertEqual(summary["curve_availability"], "unavailable_in_original_cache_metadata")
        self.assertIsNone(summary["selected_train_rmse_original"])
        self.assertIsNone(summary["terminal_train_rmse_original"])
        self.assertIsNone(summary["selected_validation_rmse_original"])
        self.assertIsNone(summary["validation_minus_train_rmse_original"])
        self.assertIsNone(summary["selected_checkpoint_sha256"])
        self.assertEqual(summary["selected_tensor_state_sha256"], STATE)
        self.assertEqual(summary["best_epoch"], 28)
        self.assertEqual(summary["epochs_completed"], 48)
        self.assertAlmostEqual(summary["saved_validation_prediction_rmse_original"], 0.1)
        self.assertFalse(summary["absence_of_overfit_established"])

    def test_changed_cache_admission_is_rejected(self):
        job, source, original, truths = cached_fixture()
        mutations = [
            {"result_sha256": "0" * 64}, {"reuse_approved": False}, {"weights_reusable": True},
            {"model_id": "GRU32"}, {"origin": "2024-10-01"}, {"seed": 271202},
            {"parent_trial_id": "different:trial"}, {"job_id": "9" * 64}, {"best_epoch": 29},
            {"validation_prediction_sha256": "9" * 64}, {"validation_dates_sha256": "9" * 64},
            {"candidate_payload_sha256": "9" * 64},
        ]
        for change in mutations:
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.extract_cached_hpo_summary(job, source, {**original, **change}, truths)

    def test_rehashed_source_still_requires_original_job_identity(self):
        job, original, original_admission, truths = cached_fixture()
        mutations = [
            {"model": "GRU32"}, {"origin": "2024-10-01"}, {"seed": 271202},
            {"candidate_id": "hpo:different:trial"}, {"stage": "final"},
        ]
        for change in mutations:
            source, admission = deepcopy(original), deepcopy(original_admission)
            source["outcome"]["resource_use"]["identity"].update(change)
            source["outcome_sha256"] = audit.sha(audit.canonical(source["outcome"]))
            source["sha256"] = audit.sha(audit.canonical({key: value for key, value in source.items() if key != "sha256"}))
            admission["result_sha256"] = source["sha256"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.extract_cached_hpo_summary(job, source, admission, truths)

    def test_cached_original_record_tampering_is_rejected(self):
        job, source, admission, truths = cached_fixture()
        source["outcome"]["predictions_original"][0] += 0.01
        with self.assertRaises(ValueError):
            audit.extract_cached_hpo_summary(job, source, admission, truths)


class MetadataOnlyRetrievalTests(unittest.TestCase):
    def collector_fixture(self, root, manifest):
        raw = audit.canonical(manifest)
        head = {"files": {}, "result_backups": {"jobs/synthetic": {
            "manifest_path": "runs/synthetic/manifest.json", "manifest_sha256": audit.sha(raw)}}}
        return audit.MetadataCollector(root, head), raw

    def test_release_weight_transport_is_rejected_even_with_json_name(self):
        manifest = {
            "attempt_id": "synthetic", "fit_status": "completed", "raw_market_data_included": False,
            "artifacts": [{"path": "request.json", "storage": "release_asset",
                           "bytes": 2, "sha256": audit.sha(b"{}"), "url": "https://example.invalid/weights.pt"}],
        }
        with tempfile.TemporaryDirectory() as root:
            collector, raw = self.collector_fixture(root, manifest)
            with patch.object(collector, "_read", return_value=raw) as reader:
                with self.assertRaisesRegex(ValueError, "retrieval is prohibited"):
                    collector.collect("jobs/synthetic", ("request.json",))
                self.assertEqual(reader.call_count, 1)

    def test_metadata_archive_containing_weights_is_rejected(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as zipped:
            zipped.writestr("request.json", b"{}")
            zipped.writestr("selected_model.pt", b"synthetic forbidden payload")
        archive_raw = stream.getvalue()
        archive = {"kind": "deterministic_zip", "url": "https://example.invalid/metadata.zip",
                   "bytes": len(archive_raw), "sha256": audit.sha(archive_raw), "member_count": 2}
        manifest = {
            "attempt_id": "synthetic", "fit_status": "completed", "raw_market_data_included": False,
            "archive_transport": archive,
            "artifacts": [{"path": "request.json", "storage": "release_archive_member",
                           "bytes": 2, "sha256": audit.sha(b"{}"), "archive_member": "request.json",
                           "archive_sha256": archive["sha256"]}],
        }
        with tempfile.TemporaryDirectory() as root:
            collector, manifest_raw = self.collector_fixture(root, manifest)
            with patch.object(collector, "_read", side_effect=[manifest_raw, archive_raw]) as reader:
                with self.assertRaisesRegex(ValueError, "non-metadata member"):
                    collector.collect("jobs/synthetic", ("request.json",))
                self.assertEqual(reader.call_count, 2)


if __name__ == "__main__":
    unittest.main()
