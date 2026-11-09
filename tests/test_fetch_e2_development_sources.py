import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import fetch_e2_development_sources as fetcher


class Response(io.BytesIO):
    def __init__(self, data, url):
        super().__init__(data)
        self.url = url

    def geturl(self):
        return self.url


class DevelopmentDownloadTests(unittest.TestCase):
    def sources(self):
        return tuple(fetcher.Source(path, len(raw), hashlib.sha256(raw).hexdigest())
                     for path, raw in [("meta/protocol.json", b'{"v":1}'),
                                       ("data/development.csv", b"date,value\n2024-01-01,1\n")])

    def opener(self, url, timeout):
        return Response(b'{"v":1}' if url.endswith("protocol.json")
                        else b"date,value\n2024-01-01,1\n", url)

    def test_pinned_download_reuse_without_network_and_no_numeric_parse(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fetcher,"SOURCES",self.sources()):
            report = fetcher.fetch_sources(directory, opener=self.opener)
            self.assertFalse(report["numerical_market_values_parsed"])
            self.assertFalse(report["final_market_files_requested"])
            self.assertEqual(report["source_commit"], fetcher.SOURCE_COMMIT)
            again = fetcher.fetch_sources(directory, opener=lambda *a, **k: self.fail("Network reuse"))
            self.assertTrue(all(row["reused"] for row in again["files"]))

    def test_corrupt_later_download_publishes_no_new_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fetcher,"SOURCES",self.sources()):
            def corrupted(url, timeout):
                return Response(b'{"v":1}' if url.endswith("protocol.json") else b"corrupted",url)
            with self.assertRaisesRegex(ValueError,"pinned"):
                fetcher.fetch_sources(directory,opener=corrupted)
            self.assertFalse(list(Path(directory).rglob("*.json")))

    def test_existing_wrong_bytes_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fetcher,"SOURCES",self.sources()):
            p=Path(directory)/"meta/protocol.json";p.parent.mkdir();p.write_bytes(b"different")
            with self.assertRaises(ValueError):fetcher.fetch_sources(directory,opener=self.opener)
            self.assertEqual(p.read_bytes(),b"different")

    def test_symlink_and_external_redirect_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fetcher,"SOURCES",self.sources()):
            root=Path(directory);(root/"cache").symlink_to(root,target_is_directory=True)
            with self.assertRaises(ValueError):fetcher.fetch_sources(root/"cache",opener=self.opener)
            with self.assertRaisesRegex(ValueError,"redirected"):
                fetcher.fetch_sources(root/"new",opener=lambda *a,**k:Response(b'{"v":1}',"https://example.com/data"))

    def test_only_known_development_payloads_and_immutable_source_commit(self):
        self.assertEqual(len(fetcher.SOURCE_COMMIT),40)
        self.assertEqual(len(fetcher.SOURCES),3)
        self.assertTrue(all("development" in s.path or s.path.endswith("protocol.json") for s in fetcher.SOURCES))
        self.assertEqual({s.size for s in fetcher.SOURCES},{5718,60427,76492})


if __name__ == "__main__":unittest.main()
