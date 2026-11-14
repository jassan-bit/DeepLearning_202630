"""Integrity failures and estimand checks for the read-only delivery auditor."""
import copy
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import audit_e2_delivery as audit


def synthetic_panel():
    rows = []
    for model in audit.NEURAL+audit.CLASSICAL:
        seeds = audit.SEEDS if model in audit.NEURAL else ("deterministic",)
        for index, seed in enumerate(seeds):
            rows.append({"model": model, "seed": seed,
                         "prediction_original": [2. if index % 2 == 0 else -2.]*608,
                         "calibration_dates": ["saved"]*53,
                         "calibration_predictions": [0.]*53, "calibration_targets": [0.]*53})
    return {"dates": [(date(2025, 1, 1)+timedelta(days=i)).isoformat() for i in range(608)],
            "mask": [True]*608, "y_true": [0.]*608, "predictors": rows}


class DeliveryAuditTests(unittest.TestCase):
    def test_original_mean_individual_RMSE_is_not_ensemble_RMSE(self):
        panel = synthetic_panel()
        _, means = audit.reconstruct_panel(panel)
        self.assertEqual(means["FCN-1D"], 2.)
        # Three positive and three negative forecasts cancel in the ensemble.
        rows = [{"model": m, "RMSE": "2", "R": "6" if m in audit.NEURAL else "1",
                 "n_common_dates": "608"} for m in means]
        audit.verify_A2(rows, means)
        rows[0]["RMSE"] = "0"
        with self.assertRaisesRegex(ValueError, "mean individual-seed"):
            audit.verify_A2(rows, means)

    def test_missing_or_reordered_calendar_day_is_rejected(self):
        panel = synthetic_panel()
        panel["dates"][100], panel["dates"][101] = panel["dates"][101], panel["dates"][100]
        with self.assertRaisesRegex(ValueError, "calendar"):
            audit.reconstruct_panel(panel)

    def test_duplicate_seed_or_classical_repetition_is_not_independence(self):
        panel = synthetic_panel()
        panel["predictors"].append(copy.deepcopy(panel["predictors"][-1]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.reconstruct_panel(panel)
        panel = synthetic_panel()
        panel["predictors"][0]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "seeds changed"):
            audit.reconstruct_panel(panel)

    def test_nonfinite_forecasts_are_rejected(self):
        panel = synthetic_panel()
        panel["predictors"][0]["prediction_original"][10] = float("nan")
        with self.assertRaisesRegex(ValueError, "forecasts"):
            audit.reconstruct_panel(panel)

    def test_manifest_detects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/"data.txt"
            data.write_bytes(b"observed\r\n")
            manifest = {"files": {"data.txt": {"sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                                                "bytes": data.stat().st_size}}}
            (root/"manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(audit.verify_manifest(root)["verified_files"], 1)
            data.write_bytes(b"observed\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                audit.verify_manifest(root)
            manifest["files"] = {"../outside.txt": "0"*64}
            (root/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "escapes"):
                audit.verify_manifest(root)

    def test_refinement_below_guide_minimum_is_rejected(self):
        source = ROOT/"book_e2/_static/e2_temporal_final_audit_20261114"
        rows = audit.read_csv(source/"primary_refinement_20000.csv")
        rows[0]["draws"] = "4999"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"primary_refinement_20000.csv"
            with p.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "settings changed"):
                audit.verify_primary_refinement(tmp)

    def test_complete_public_delivery_has_no_new_fit_claim(self):
        result = audit.audit(ROOT)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["systems"], 16)
        self.assertEqual(result["original_predictors"], 81)
        self.assertLessEqual(result["recomputed_max_A2_RMSE_difference"], 5e-15)
        self.assertFalse(result["methodological_approval_claimed"])
        self.assertFalse(result["new_model_inference"])
        self.assertEqual(result["new_model_fits"], 0)

    def test_missing_required_supplement_cannot_pass_release_audit(self):
        # A private staging tree may lack supplements; the public-release flag
        # must never silently turn a missing final manifest into an approval.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original_folders = audit.MANIFEST_FOLDERS
            try:
                audit.MANIFEST_FOLDERS = ()
                with self.assertRaisesRegex(ValueError, "supplements missing"):
                    audit.audit(root, require_supplements=True)
            finally:
                audit.MANIFEST_FOLDERS = original_folders


if __name__ == "__main__":
    unittest.main()
