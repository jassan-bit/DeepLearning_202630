"""Software-only fixtures: no scientific inputs, model loading or predictions."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('completed_eval_verifier', HERE/'completed_eval_verifier.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def raw(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


class CompletedEvaluationGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory(prefix='e3-completion-software-fixture-')
        cls.seed = Path(cls.storage.name)/'seed'
        cls.seed.mkdir()
        cls.protocol = HERE/'fixtures/protocol.json'
        shutil.copyfile(cls.protocol, cls.seed/'protocol.json')
        cls.source = Path(cls.storage.name)/'source'
        target = cls.source/gate.SCOPE/'evaluation/run.py'
        target.parent.mkdir(parents=True)
        shutil.copyfile(Path(os.environ.get('E3_SCIENTIFIC_SOURCE', '/workspace/e3_review9'))/gate.SCOPE/'evaluation/run.py', target)
        receipt = json.loads((HERE/'fixtures/evaluation_receipt.json').read_bytes())
        manifest = json.loads((HERE/'fixtures/run_manifest.json').read_bytes())
        panel = json.loads((HERE/'fixtures/panel_manifest.json').read_bytes())
        for entry in panel['files']:
            # Deliberately minimal inert CSV bytes. Tests exercise completion
            # bindings, not scientific semantic validity; the real gate pins
            # the independently fetched original manifests and payload hashes.
            payload = b'fixture_only\n'+b'0\n'*entry['rows']
            target = cls.seed/entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            entry.update(bytes=len(payload), sha256=sha(payload))
        panel_raw = raw(panel)
        manifest['panel_sha256'] = sha(panel_raw)
        freeze = {'verified': True, 'source_commit': gate.SOURCE_COMMIT,
                  'protocol_sha256': gate.PROTOCOL_SHA256,
                  'annotated_tag': manifest['protocol_tag']}
        freeze_raw = raw(freeze)
        manifest['freeze_receipt_sha256'] = sha(freeze_raw)
        for name, value in {'evaluation_receipt.json': raw(receipt),
                            'run_manifest.json': raw(manifest),
                            'panel_manifest.json': panel_raw,
                            'freeze_receipt.json': freeze_raw,
                            'failures.json': b'[]\n'}.items():
            (cls.seed/name).write_bytes(value)
        cls.anchors = {name: sha((cls.seed/name).read_bytes())
                       for name in gate.FINAL_ARTIFACT_SHA256}

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='e3-completion-tamper-')
        self.output = Path(self.temp.name)/'output'
        shutil.copytree(self.seed, self.output)
        self.anchors_now = dict(self.anchors)
        self.patch = patch.object(gate, 'FINAL_ARTIFACT_SHA256', self.anchors_now)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def check(self):
        return gate.verify_completed_evaluation(self.output, self.output/'protocol.json', self.source)

    def edit(self, name, change, reanchor=False):
        path = self.output/name
        data = json.loads(path.read_bytes())
        change(data)
        path.write_bytes(raw(data))
        if reanchor:
            self.anchors_now[name] = sha(path.read_bytes())

    def panel_entry(self):
        return json.loads((self.output/'panel_manifest.json').read_bytes())['files'][0]

    def test_complete_published_identity_is_read_only(self):
        before = {str(p.relative_to(self.output)): sha(p.read_bytes())
                  for p in self.output.rglob('*') if p.is_file()}
        proof = self.check()
        after = {str(p.relative_to(self.output)): sha(p.read_bytes())
                 for p in self.output.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(proof['verified_panel_files'], 76)
        self.assertEqual(proof['verified_panel_rows'], 1874560)
        self.assertEqual(proof['completed_job_results'], 796)
        self.assertEqual(proof['new_model_fits'], 0)
        self.assertEqual(proof['new_model_inference'], 0)
        self.assertIs(proof['R3_accredited'], False)

    def test_changed_completed_count_is_rejected_by_published_anchor(self):
        self.edit('evaluation_receipt.json', lambda d: d.update(completed_job_results=795))
        with self.assertRaisesRegex(ValueError, 'published final artifact differs'):
            self.check()

    def test_even_rebound_incomplete_evaluation_is_rejected(self):
        self.edit('evaluation_receipt.json', lambda d: d.update(status='partial_failed'), True)
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            self.check()

    def test_even_rebound_claimed_accreditation_is_rejected(self):
        self.edit('evaluation_receipt.json', lambda d: d.update(R3_accredited=True), True)
        with self.assertRaisesRegex(ValueError, 'accreditation flags changed'):
            self.check()

    def test_modified_prediction_bytes_are_rejected(self):
        path = self.output/self.panel_entry()['path']
        path.write_bytes(path.read_bytes().replace(b'0', b'1', 1))
        with self.assertRaisesRegex(ValueError, 'panel payload differs'):
            self.check()

    def test_truncated_prediction_is_rejected(self):
        path = self.output/self.panel_entry()['path']
        path.write_bytes(path.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, 'panel payload differs'):
            self.check()

    def test_missing_panel_is_rejected(self):
        (self.output/self.panel_entry()['path']).unlink()
        with self.assertRaisesRegex(ValueError, 'regular file missing'):
            self.check()

    def test_symlinked_panel_is_rejected(self):
        path = self.output/self.panel_entry()['path']
        target = Path(self.temp.name)/'outside.csv'
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'shard escaped|symlink refused'):
            self.check()

    def test_nonempty_failure_list_is_rejected(self):
        (self.output/'failures.json').write_bytes(raw([{'dataset': 'D00'}]))
        with self.assertRaisesRegex(ValueError, 'failure list is nonempty'):
            self.check()

    def test_protocol_change_is_rejected(self):
        self.edit('protocol.json', lambda d: d.update(evidence_mode='other'))
        with self.assertRaisesRegex(ValueError, 'protocol differs'):
            self.check()

    def test_source_change_is_rejected(self):
        altered = Path(self.temp.name)/'source'
        shutil.copytree(self.source, altered)
        path = altered/gate.SCOPE/'evaluation/run.py'
        path.write_bytes(path.read_bytes()+b'\n# modified\n')
        with self.assertRaisesRegex(ValueError, 'worker source differs'):
            gate.verify_completed_evaluation(self.output, self.output/'protocol.json', altered)

    def test_bad_freeze_receipt_is_rejected(self):
        self.edit('freeze_receipt.json', lambda d: d.update(verified=False))
        with self.assertRaisesRegex(ValueError, 'freeze receipt binding differs'):
            self.check()

    def test_rebound_seed_roster_is_rejected(self):
        self.edit('run_manifest.json', lambda d: d['seeds'].pop(), True)
        with self.assertRaisesRegex(ValueError, 'seed roster differs'):
            self.check()

    def test_rebound_run_panel_binding_is_rejected(self):
        self.edit('run_manifest.json', lambda d: d.update(panel_sha256='0'*64), True)
        with self.assertRaisesRegex(ValueError, 'run/panel manifest binding differs'):
            self.check()

    def test_missing_completion_marker_is_rejected(self):
        (self.output/'evaluation_receipt.json').unlink()
        with self.assertRaisesRegex(ValueError, 'regular file missing'):
            self.check()


if __name__ == '__main__':
    unittest.main(verbosity=2)
