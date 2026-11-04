"""Resume-integrity and causality checks without fitting project models."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAINING_ENVIRONMENT = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TRAINING_ENVIRONMENT, "Run with the training environment")
class MultiHistoryRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "scripts"))
        import train_filipe_multihistory
        cls.module = train_filipe_multihistory

    def test_resume_checks_every_saved_artifact_before_using_a_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            artifact = path / "predictions.csv"
            artifact.write_text("y_true,y_pred\n1,1\n")
            completion = {"code_signature": "signature", "record": {"model": "MLP"},
                "artifact_sha256": {"predictions.csv": hashlib.sha256(artifact.read_bytes()).hexdigest()}}
            (path / "completion.json").write_text(json.dumps(completion))
            record = self.module.verify_completion(path, "signature", "completion.json")
            self.assertEqual(record, {"model": "MLP"})
            artifact.write_text("y_true,y_pred\n1,2\n")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                self.module.verify_completion(path, "signature", "completion.json")

    def test_resume_rejects_signature_mismatch_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertIsNone(self.module.verify_completion(path, "s", "completion.json"))
            payload = {"code_signature": "other", "record": {}, "artifact_sha256": {}}
            (path / "completion.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                self.module.verify_completion(path, "s", "completion.json")
            payload["code_signature"] = "s"
            payload["artifact_sha256"] = {"missing.pt": "not-a-hash"}
            (path / "completion.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                self.module.verify_completion(path, "s", "completion.json")

    def test_real_manifests_pass_and_one_day_purge_relaxation_fails(self):
        import copy
        from dl_data import load_development
        data = load_development(ROOT)
        for history in (14, 21, 28):
            path = ROOT / "results/filipe_multihistory_20261104" / f"D{history}" / "folds.json"
            if not path.exists():
                self.skipTest("Generate new index-only folds before the runner validation test")
            manifest = json.loads(path.read_text())
            for fold in manifest["folds"]:
                self.module.validate_fold(data, history, fold)
            broken = copy.deepcopy(manifest["folds"][0])
            broken["train_indices"].append(broken["fit_origin_index"]-history)
            with self.assertRaisesRegex(ValueError, "past-only"):
                self.module.validate_fold(data, history, broken)


if __name__ == "__main__":
    unittest.main()
