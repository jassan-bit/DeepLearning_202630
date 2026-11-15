"""Original-data admission, without any real network or market read."""
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import download_e2_known_market as acquisition


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.raw=b"fixture,not,market\n"
        self.row=dict(archive="fixture.zip",relative_path="test/fixture.zip",bytes=len(self.raw),
            sha256=hashlib.sha256(self.raw).hexdigest(),url="https://data.binance.vision/fixture.zip")

    def test_good_download_is_reused_without_second_network_call(self):
        with patch.object(acquisition.urllib.request,"urlopen",return_value=io.BytesIO(self.raw)) as opened:
            path,status=acquisition.obtain(self.row,self.root)
        self.assertEqual(status,"downloaded_verified")
        self.assertEqual(path.read_bytes(),self.raw)
        with patch.object(acquisition.urllib.request,"urlopen") as opened:
            _,status=acquisition.obtain(self.row,self.root)
            opened.assert_not_called()
        self.assertEqual(status,"reused_verified")

    def test_corrupt_existing_bytes_never_overwritten(self):
        path=self.root/self.row['relative_path'];path.parent.mkdir(parents=True)
        path.write_bytes(b"changed")
        with patch.object(acquisition.urllib.request,"urlopen") as opened:
            with self.assertRaises(ValueError):acquisition.obtain(self.row,self.root)
            opened.assert_not_called()
        self.assertEqual(path.read_bytes(),b"changed")

    def test_bad_download_is_not_admitted_and_is_preserved(self):
        with patch.object(acquisition.urllib.request,"urlopen",return_value=io.BytesIO(b"wrong")):
            with self.assertRaises(ValueError):acquisition.obtain(self.row,self.root)
        self.assertFalse((self.root/self.row['relative_path']).exists())
        self.assertEqual(len(list(self.root.rglob('*.partial'))),1)


if __name__=="__main__":unittest.main()
