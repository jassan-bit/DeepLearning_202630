"""Integrity and scope guards using declared software fixtures only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import verify_packet as v


class Guards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / v.MANIFEST).write_text('{}')
        (self.root / 'example.txt').write_bytes(b'fixed bytes\n')
        self.entry = {'path': 'example.txt', 'bytes': 12,
                      'sha256': hashlib.sha256(b'fixed bytes\n').hexdigest()}

    def tearDown(self):
        self.temp.cleanup()

    def rejected(self, entries=None):
        with self.assertRaises(ValueError):
            v.verify_inventory(self.root, entries or [self.entry])

    def test_exact_inventory_passes_without_running_payload(self):
        result = v.verify_inventory(self.root, [self.entry])
        self.assertTrue(result['exact_member_set_verified'])
        self.assertEqual(result['files_verified'], 1)

    def test_same_length_tamper_is_rejected(self):
        (self.root / 'example.txt').write_bytes(b'wrong bytes\n')
        self.rejected()

    def test_size_change_is_rejected(self):
        (self.root / 'example.txt').write_bytes(b'longer bytes!!!!')
        self.rejected()

    def test_missing_file_is_rejected(self):
        (self.root / 'example.txt').unlink()
        self.rejected()

    def test_undeclared_hidden_and_pycache_files_are_not_ignored(self):
        for name in ('.hidden', '__pycache__/undeclared.pyc'):
            with self.subTest(name=name):
                target = self.root / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(b'unexpected')
                self.rejected()
                target.unlink()

    def test_duplicate_member_is_rejected(self):
        self.rejected([self.entry, dict(self.entry)])

    def test_manifest_cannot_declare_itself(self):
        self.rejected([dict(self.entry, path=v.MANIFEST)])

    def test_traversal_absolute_and_normalized_paths_are_rejected(self):
        for name in ('../outside', '/absolute', './example.txt', 'a/../b',
                     'a//b', 'a\\b', 'C:drive', 'a\x00b', 'a\nb'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                v.safe_name(name)

    def test_file_and_directory_symlinks_are_rejected(self):
        link = self.root / 'link'
        link.symlink_to(self.root / 'example.txt')
        self.rejected()
        link.unlink()
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            v.checked_path(self.root, 'link/example.txt')

    def test_duplicate_json_keys_are_rejected(self):
        (self.root / 'duplicate.json').write_text('{"claim":false,"claim":true}')
        with self.assertRaises(ValueError):
            v.read_json(self.root / 'duplicate.json')

    def test_bool_is_not_a_byte_count(self):
        with self.assertRaises(ValueError):
            v.validate_entry(dict(self.entry, bytes=True))

    def test_component_manifest_and_inner_member_are_independently_bound(self):
        component = self.root / 'component'
        component.mkdir()
        (component / 'source.py').write_bytes(b'never executed\n')
        data = {'files': {'source.py': {'bytes': 15,
                 'sha256': hashlib.sha256(b'never executed\n').hexdigest()}}}
        manifest = component / 'manifest.json'
        manifest.write_text(json.dumps(data))
        pin = v.sha_file(manifest)
        self.assertEqual(v.verify_component(self.root, 'component/manifest.json', pin)[1], 1)
        (component / 'source.py').write_bytes(b'altered source\n')
        with self.assertRaises(ValueError):
            v.verify_component(self.root, 'component/manifest.json', pin)
        manifest.write_text(json.dumps({'files': {}}))
        with self.assertRaises(ValueError):
            v.verify_component(self.root, 'component/manifest.json', pin)

    @staticmethod
    def science():
        return {'base_commit': v.BASE, 'scientific_decision': 'NO_GO_CONFIRMATION_CURRENT_RECIPE',
                'preserved_additive_jobs': 796, 'preserved_V3_fits': 30,
                'preserved_transfer_trajectories': 51, 'preserved_transfer_rows': 20400,
                'pilot_error_increase': 0.02704311988801411, 'R3_accredited': False,
                'rule_changed': False, 'ablation_promoted': False,
                'new_model_fits': 0, 'new_predictions': 0}

    def test_no_go_adverse_result_and_experiment_identity_are_required(self):
        data = self.science()
        v.check_science(data)
        for key, wrong in [('scientific_decision', 'GO'), ('R3_accredited', True),
                           ('preserved_V3_fits', 796), ('pilot_error_increase', -0.027),
                           ('new_model_fits', 1), ('ablation_promoted', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                v.check_science(dict(data, **{key: wrong}))

    def test_zero_and_false_scope_flags_are_required_and_typed(self):
        for wrong in (None, False, 1):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                v.zero_fields({'new_model_fits': wrong}, ['new_model_fits'])
        for wrong in (None, 0, True):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                v.false_fields({'R3_accredited': wrong}, ['R3_accredited'])


if __name__ == '__main__':
    unittest.main()
