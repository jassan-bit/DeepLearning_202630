"""Integrity boundary tests with documentary fixtures; no external requests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import build_candidate
import verify_candidate
import verify_zenodo_deposit as public


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.raw = b'reviewed documentary fixture\n'
        self.expected = {'bytes': len(self.raw), 'sha256': hashlib.sha256(self.raw).hexdigest()}
        self.name = 'reviewed.zip'
        self.manifest = {'record_id': 24356217, 'reserved_doi': '10.5281/zenodo.24356217',
                         'files': {self.name: self.expected}}

    def fake_get(self, url, limit):
        if '/api/records/' in url:
            value = {'id': 24356217, 'doi': self.manifest['reserved_doi'],
                     'metadata': {'title': 'Fixture only'},
                     'files': [{'key': self.name, 'size': len(self.raw),
                                'checksum': 'md5:' + hashlib.md5(self.raw).hexdigest(),
                                'links': {'self': 'https://zenodo.org/records/24356217/files/reviewed.zip'}}]}
            return json.dumps(value).encode(), url, 200
        if 'doi.org' in url:
            return b'', 'https://zenodo.org/records/24356217', 200
        self.assertLessEqual(len(self.raw), limit)
        return self.raw, url, 200

    def test_exact_public_download_and_resolver_fixture(self):
        result = public.verify_deposit(self.manifest, 1000, self.fake_get)
        self.assertEqual(result['status'], 'public_DOI_and_expected_candidate_bytes_verified')
        self.assertTrue(result['files'][self.name]['reported_MD5_verified'])
        self.assertFalse(result['rights_verified'])

    def test_changed_deposit_bytes_rejected(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            public.verify_bytes(self.raw + b'changed', self.expected, None)

    def test_budget_and_empty_inventory_rejected_before_network(self):
        with self.assertRaisesRegex(ValueError, 'budget'):
            public.verify_deposit(self.manifest, 1, lambda *args: self.fail('must not request'))
        self.manifest['files'] = {}
        with self.assertRaisesRegex(ValueError, 'empty'):
            public.verify_deposit(self.manifest, 1000, lambda *args: self.fail('must not request'))

    def test_credentials_http_and_external_urls_rejected(self):
        for url in ['https://attacker.example/x', 'http://zenodo.org/x',
                    'https://token@zenodo.org/x', 'https://zenodo.org:444/x']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                public.public_url(url, {'zenodo.org'})

    def test_record_and_doi_disagreement_rejected(self):
        self.manifest['reserved_doi'] = '10.5281/zenodo.1'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            public.verify_deposit(self.manifest, 1000, lambda *args: self.fail('must not request'))

    def test_modified_zip_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive, manifest = build_candidate.build(Path(directory))
            archive.write_bytes(archive.read_bytes() + b'altered')
            with self.assertRaisesRegex(ValueError, 'bytes'):
                verify_candidate.verify(archive, manifest)

    def test_zip_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'unsafe.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('../unsafe', b'x')
                output.writestr('MANIFEST.json', b'{}')
            data = archive.read_bytes()
            manifest = {'files': {archive.name: {'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(), 'members': 2}}}
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                verify_candidate.verify(archive, manifest)


if __name__ == '__main__':
    unittest.main()
