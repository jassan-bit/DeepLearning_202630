"""Evidence-integrity tests for E3 export; no market or training needed."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/e3_export_readiness.py"
spec = importlib.util.spec_from_file_location("e3_export_readiness", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExportIntegrityTests(unittest.TestCase):
    def test_missing_bytes_do_not_become_exported(self):
        sha="a"*64
        freeze={"systems":["Tiny"],"models":[{"model":"Tiny","seed":9,
                **{f:f+".bin" for f in ("checkpoint","scaler","calibrator")},
                **{f+"_sha256":sha for f in ("checkpoint","scaler","calibrator")}}]}
        release={"artifacts":[{"path":f+".bin","sha256":sha,"storage":"release_asset","url":"https://example.invalid/weights"}
                              for f in ("checkpoint","scaler","calibrator")]}
        with tempfile.TemporaryDirectory() as d:
            out=module.readiness(freeze,release,weights_root=d,metadata_root=d)
        row=out["rows"][0]
        self.assertEqual(row["artifacts"]["checkpoint"]["status"],"missing_locally")
        self.assertEqual(row["export_status"],"not_executed_by_inventory")
        self.assertFalse(out["inference_artifacts_created"])

    def test_freeze_release_hash_mismatch_is_rejected(self):
        freeze={"systems":["Tiny"],"models":[{"model":"Tiny","seed":9,"checkpoint":"x","checkpoint_sha256":"a"*64}]}
        release={"artifacts":[{"path":"x","sha256":"b"*64}]}
        with self.assertRaisesRegex(ValueError,"Freeze/release mismatch"):
            module.readiness(freeze,release)

    def test_local_corruption_is_not_treated_as_missing(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"x").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                module.validate_file(d,"x",hashlib.sha256(b"original").hexdigest())

    def test_pointer_is_not_a_model(self):
        pointer=b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 50\n"
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"x").write_bytes(pointer)
            with self.assertRaisesRegex(ValueError,"LFS pointer"):
                module.validate_file(d,"x",hashlib.sha256(pointer).hexdigest())

    def test_export_receipt_cannot_certify_other_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"Tiny";p.mkdir();(p/"model.torchscript.pt").write_bytes(b"changed")
            record={"seed":11,"checkpoint_sha256":"a"*64}
            report={"model":"Tiny","seed":11,"checkpoint_sha256":"a"*64,
                    "export_sha256":hashlib.sha256(b"original").hexdigest()}
            (p/"roundtrip_report.json").write_text(json.dumps(report))
            (p/"metadata.json").write_text('{}')
            with self.assertRaisesRegex(ValueError,"integrity mismatch"):
                module.verified_export(d,"Tiny",record)

    def test_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,"escapes"):
                module.safe_path(d,"../secret")

    def test_serving_guard_rejects_shape_drift_and_nonfinite(self):
        valid=[[[0.0]*5 for _ in range(7)]]
        self.assertIs(module.validate_windows(valid),valid)
        invalid=[[],[[[0.0]*5 for _ in range(6)]],[[[0.0]*4 for _ in range(7)]],
                 [[[float('nan')]*5 for _ in range(7)]],[[[True]*5 for _ in range(7)]]]
        for data in invalid:
            with self.assertRaises(ValueError):module.validate_windows(data)

    def test_seed_rule_does_not_use_test_metric(self):
        freeze={"models":[{"model":"Tiny","seed":22,"test_RMSE":0.001},
                          {"model":"Tiny","seed":11,"test_RMSE":99}]}
        self.assertEqual(module.select_seed(freeze,"Tiny")["seed"],11)

    def test_duplicate_seed_is_rejected(self):
        freeze={"models":[{"model":"Tiny","seed":11},{"model":"Tiny","seed":11}]}
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            module.select_seed(freeze,"Tiny")

    def test_roundtrip_rejects_prediction_disagreement(self):
        with self.assertRaisesRegex(ValueError,"tolerance exceeded"):
            module.roundtrip_metrics([1,2],[1,3],[1,2],tolerance=1e-6)

    def test_roundtrip_rejects_wrong_cohort(self):
        with self.assertRaisesRegex(ValueError,"lengths differ"):
            module.roundtrip_metrics([1,2],[1],[1,2],tolerance=1e-6)

    def test_roundtrip_rejects_nonfinite_output(self):
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            module.roundtrip_metrics([1,float("nan")],[1,2],[1,2],tolerance=1e-6)

    def test_exact_predictions_preserve_rmse(self):
        out=module.roundtrip_metrics([1,3],[1,3],[1,2],tolerance=1e-6)
        self.assertAlmostEqual(out["RMSE_export_original"],math.sqrt(.5))
        self.assertEqual(out["RMSE_export_minus_memory"],0)

    def test_roundtrip_rejects_undeclared_invalid_tolerance(self):
        for tolerance in (0,-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError):
                module.roundtrip_metrics([1],[1],[1],tolerance=tolerance)


if __name__=="__main__":
    unittest.main()
