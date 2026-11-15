"""Keep the public page and deployment table bound to actual artifact evidence."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e3_handoff_book',ROOT/'scripts/e3_build_handoff_book.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class EvidenceJoinTests(unittest.TestCase):
    def test_missing_metadata_evidence_cannot_leave_old_success_narrative(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/m.DATA,root/m.DATA)
            path=root/m.DATA/'audit_summary.json'
            report=json.loads(path.read_text())
            report['downloaded_metadata_hashes_verified']=0
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'source counts'):
                m.generate(root)

    def test_only_measured_models_get_cpu_results(self):
        outputs=m.generate(ROOT)
        rows=json.loads(outputs[m.DATA/'deployment_master.json'])
        measured=[r for r in rows if r['cpu_latency_median_ms'] is not None]
        self.assertEqual({r['model'] for r in measured},set(m.REFERENCES))
        self.assertEqual(len(rows),16)
        for r in rows:
            self.assertIsNone(r['exported_A2_test_RMSE'])
        for r in measured:
            self.assertEqual(r['calibration_RMSE_difference'],0)
            self.assertIn('includes',r['cpu_memory_scope'])

    def test_damaged_inference_file_cannot_retain_verified_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/m.DATA,root/m.DATA)
            shutil.copytree(ROOT/'artifacts/e3_handoff',root/'artifacts/e3_handoff')
            artifact=root/'artifacts/e3_handoff/MLP16/model.torchscript.pt'
            artifact.write_bytes(artifact.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                m.generate(root)

    def test_calibration_cannot_be_relabelled_as_test_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/m.DATA,root/m.DATA)
            shutil.copytree(ROOT/'artifacts/e3_handoff',root/'artifacts/e3_handoff')
            path=root/'artifacts/e3_handoff/MLP16/roundtrip_report.json'
            report=json.loads(path.read_text())
            report['cohort']['A2_test_608_reproduced']=True
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'roundtrip scope'):
                m.generate(root)


if __name__=='__main__':
    unittest.main()
