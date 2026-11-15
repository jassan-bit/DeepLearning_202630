"""Guard against test-driven selection, incomplete CV and forged handoff evidence."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e3_selection", ROOT/"scripts/e3_select_e2_configurations.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def receipt():
    ledger = []
    for n in range(1, 7):
        values = [0.001*n, 0.002*n, 0.003*n]
        ledger.append({"configuration_id": f"C{n:02d}", "status": "completed",
                       "validation_count": 249, "fold_RMSE_original": dict(zip(m.FOLDS, values)),
                       "pooled_cv_RMSE_original": math.sqrt(sum(x*x for x in values)/3),
                       "mean_fold_RMSE_original_diagnostic": sum(values)/3,
                       "test_RMSE": 1/n})
    return {"criteria_only": False, "fold_count": 3, "validation_count": 249,
            "selection_scope": "one_global_recipe_for_post_HPO_final_cohort",
            "candidate_ledger": ledger, "winner_configuration_id": "C01",
            "winner_pooled_cv_RMSE_original": ledger[0]["pooled_cv_RMSE_original"],
            "winner_trial": {"trial_id": "test:C01"}, "fixed_phase_epochs": {"supervised": 10}}


class ValidationSelectionTests(unittest.TestCase):
    def test_test_rank_cannot_replace_validation_winner(self):
        data = receipt()
        selected = m.select_neural(data)
        self.assertEqual(selected["configuration_id"], "C01")
        self.assertEqual(min(data["candidate_ledger"], key=lambda r:r["test_RMSE"])["configuration_id"], "C06")
        for row in data["candidate_ledger"]:
            row["test_RMSE"] = 100 if row["configuration_id"] == "C01" else 0
        self.assertEqual(m.select_neural(data)["configuration_id"], "C01")

    def test_recorded_test_winner_is_rejected(self):
        data = receipt(); data["winner_configuration_id"] = "C06"
        with self.assertRaisesRegex(ValueError, "validation minimizer"):
            m.select_neural(data)

    def test_completed_candidate_with_missing_fold_rejected(self):
        data = receipt(); del data["candidate_ledger"][0]["fold_RMSE_original"][m.FOLDS[0]]
        with self.assertRaisesRegex(ValueError, "validation support"):
            m.select_neural(data)

    def test_pooling_cannot_silently_drop_failed_fold(self):
        data = receipt()
        data["candidate_ledger"][0] = {"configuration_id": "C01", "status": "failed", "pooled_cv_RMSE_original": 0}
        with self.assertRaisesRegex(ValueError, "validation minimizer"):
            m.select_neural(data)

    def test_nan_fold_rejected(self):
        data = receipt();data["candidate_ledger"][0]["fold_RMSE_original"][m.FOLDS[0]] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            m.select_neural(data)

    def test_tampered_pooled_score_rejected(self):
        data = receipt();data["candidate_ledger"][0]["pooled_cv_RMSE_original"] = 0
        with self.assertRaisesRegex(ValueError, "Pooled validation"):
            m.select_neural(data)

    def test_wrong_validation_scope_rejected(self):
        data = receipt();data["selection_scope"] = "best_final_test_seed"
        with self.assertRaisesRegex(ValueError, "selection scope"):
            m.select_neural(data)

    def test_calibration_cannot_select_classical(self):
        with self.assertRaisesRegex(ValueError, "Calibration"):
            m.select_classical({"calibration_used_for_selection": True})

    def test_classical_incomplete_candidate_cannot_win(self):
        data = {"model": "ARIMA", "calibration_used_for_selection": False,
                "selection_metric": "pooled_original_unit_CV_RMSE", "parameters": {"order": [1,0,0]}, "ledger": []}
        for params, n, value in [({"order": [1,0,0]},2,0.00001),({"order": [2,0,1]},3,0.1)]:
            for fold in m.FOLDS[:n]:
                data["ledger"].append({"parameters":params,"fold":fold,"candidate_status":"completed","RMSE":value})
        with self.assertRaisesRegex(ValueError, "validation minimizer"):
            m.select_classical(data)

    def test_fixed_persistence_does_not_invent_validation_score(self):
        row = m.select_classical({"model":"Persistence","parameters":{"lag":1},"ledger":[],
                                 "selection_metric":"pooled_original_unit_CV_RMSE","calibration_used_for_selection":False})
        self.assertIsNone(row["validation_pooled_rmse"])


class RealEvidenceTests(unittest.TestCase):
    def test_changed_selection_receipt_fails_provenance_check(self):
        _, report = m.build_table(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in report["sources"]:
                target=root/source["path"];target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/source["path"],target)
            receipt_path=next((root/m.RECEIPTS).glob("*.json"))
            receipt_path.write_bytes(receipt_path.read_bytes()+b" ")
            with self.assertRaisesRegex(ValueError,"receipt hash mismatch"):
                m.build_table(root)

    def test_real_table_retains_metric_scopes_and_frozen_seed(self):
        rows, report = m.build_table(ROOT)
        self.assertEqual(len(rows),16)
        self.assertEqual(report["neural_recipes_verified"],13)
        self.assertTrue(report["all_neural_pooled_and_arithmetic_winners_agree"])
        self.assertEqual(report["available_validation_leader"],"TimeMixer")
        self.assertIsNone(report["global_validation_winner_16_systems"])
        self.assertFalse(report["available_validation_leader_mcs_retained"])
        self.assertEqual(report["historical_test_descriptive_leader"],"LSTM-autoencoder con predictor")
        self.assertTrue(report["historical_test_leader_mcs_retained"])
        for row in rows:
            self.assertIsNone(row["validation_ci95_lower"])
            self.assertIsNone(row["cpu_latency_median_ms"])
            if row["final_seed_ids"]:
                self.assertEqual(row["representative_export_seed"],271801)

    def test_report_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/"a",Path(tmp)/"b"
            m.write_outputs(ROOT,a);m.write_outputs(ROOT,b)
            self.assertEqual(sorted(p.name for p in a.iterdir()),sorted(p.name for p in b.iterdir()))
            for p in a.iterdir():self.assertEqual(p.read_bytes(),(b/p.name).read_bytes())


if __name__ == "__main__":
    unittest.main()
