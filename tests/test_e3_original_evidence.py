"""Validate saved development evidence without importing the tensor framework."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e3_original_receipt',ROOT/'scripts/e3_verify_original.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def copy_evidence(root):
    for rel in [*m.SOURCE_PATHS,str(m.REPORT)]:
        destination=root/rel;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,destination)


class OriginalEvidenceTests(unittest.TestCase):
    def test_nonfinite_sampled_gain_cannot_pass_the_property_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);copy_evidence(root)
            path=root/m.REPORT;report=json.loads(path.read_text())
            report['latent_verification']['maximum_sampled_gain']=float('nan')
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'numeric verification'):
                m.check(root)

    def test_saved_scope_has_no_market_or_superiority_claim(self):
        report=m.check(ROOT)
        self.assertFalse(report['real_market_data_used'])
        self.assertFalse(report['novelty_verified'])
        self.assertFalse(report['statistical_superiority_verified'])
        self.assertEqual(report['latent_verification']['sampled_violations'],0)

    def test_changed_implementation_invalidates_old_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);copy_evidence(root)
            source=root/'src/original_model.py'
            source.write_bytes(source.read_bytes()+b'\n# changed\n')
            with self.assertRaisesRegex(ValueError,'source changed'):
                m.check(root)

    def test_successful_software_cannot_be_relabelled_as_R3(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);copy_evidence(root)
            path=root/m.REPORT;report=json.loads(path.read_text())
            report['statistical_superiority_verified']=True
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'scope changes'):
                m.check(root)

    def test_failed_or_skipped_pytest_run_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'results.xml'
            path.write_text('<testsuites><testsuite tests="3" failures="0" errors="0" skipped="1"/></testsuites>')
            with self.assertRaisesRegex(ValueError,'non-skipped'):
                m.parse_junit(path)


if __name__=='__main__':
    unittest.main()
