"""Synthetic, offline acquisition-integrity checks."""
from hashlib import sha256
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('acquisition', Path(__file__).resolve().parents[1] / 'scripts/download_e2_completion_sources.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class AcquisitionTests(unittest.TestCase):
    def fixture(self):
        return {'source_locations': {'Model': {'repository': 'author/repo', 'commit': 'a' * 40}},
                'author_sources': [{'model': 'Model', 'path': 'layers/core.py', 'repository_path': 'src/core.py',
                                    'sha256': sha256(b'code').hexdigest(), 'bytes': 4}]}

    def test_pinned_file_and_equal_cache_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            urls = []
            def fetch(url):
                urls.append(url)
                return b'code'
            receipt = M.acquire(self.fixture(), d, fetcher=fetch)
            M.acquire(self.fixture(), d, fetcher=lambda _: self.fail('Cached source must not download again'))
            self.assertEqual(urls, ['https://raw.githubusercontent.com/author/repo/' + 'a' * 40 + '/src/core.py'])
            self.assertEqual(len(receipt['verified_sources']), 1)

    def test_wrong_download_never_written(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                M.acquire(self.fixture(), d, fetcher=lambda _: b'evil')
            self.assertFalse((Path(d) / 'Model/layers/core.py').exists())

    def test_empty_python_package_initializer_is_valid(self):
        fixture = self.fixture()
        row = fixture['author_sources'][0]
        row.update(path='layers/__init__.py', repository_path='layers/__init__.py',
                   bytes=0, sha256=sha256(b'').hexdigest())
        with tempfile.TemporaryDirectory() as d:
            M.acquire(fixture, d, fetcher=lambda _: b'')
            self.assertEqual((Path(d) / 'Model/layers/__init__.py').read_bytes(), b'')

    def test_existing_unequal_file_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'Model/layers/core.py'
            p.parent.mkdir(parents=True)
            p.write_bytes(b'evil')
            with self.assertRaises(ValueError):
                M.acquire(self.fixture(), d, fetcher=lambda _: b'code')
            self.assertEqual(p.read_bytes(), b'evil')

    def test_reject_traversal_mutable_commit_and_symlink(self):
        for bad in ['../escape', '/abs', 'a/../b', 'a\\b', 'a//b']:
            fixture = self.fixture()
            fixture['author_sources'][0]['path'] = bad
            with self.assertRaises(ValueError):
                M.validate(fixture)
        fixture = self.fixture()
        fixture['source_locations']['Model']['commit'] = 'main'
        with self.assertRaises(ValueError):
            M.validate(fixture)
        with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as outside:
            (Path(d) / 'Model').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                M.acquire(self.fixture(), d, fetcher=lambda _: b'code')


if __name__ == '__main__':
    unittest.main()
