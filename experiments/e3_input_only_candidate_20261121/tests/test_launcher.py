"""Pinned-download corruption and cache preservation, without network or fits."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import prepare_colab as launcher


class LauncherTests(unittest.TestCase):
    def manifest(self,name="runtime/sample.py",body=b"pass\n"):
        return json.dumps({"algorithm":"sha256","files":{name:hashlib.sha256(body).hexdigest()},"sizes":{name:len(body)}}).encode()

    def test_pinned_acquisition_and_verified_cache(self):
        raw=b"pass\n";manifest=self.manifest(body=raw)
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(launcher,"download",side_effect=[manifest,raw]):
                root=launcher.acquire("a"*40,launcher.sha(manifest),temp)
            self.assertEqual((root/"runtime/sample.py").read_bytes(),raw)
            with patch.object(launcher,"download",return_value=manifest) as remote:
                self.assertEqual(launcher.acquire("a"*40,launcher.sha(manifest),temp),root)
                self.assertEqual(remote.call_count,1)

    def test_rejects_bad_manifest_and_bad_source_hash(self):
        manifest=self.manifest()
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(launcher,"download",return_value=manifest):
                with self.assertRaises(ValueError):launcher.acquire("a"*40,"0"*64,temp)
            with patch.object(launcher,"download",side_effect=[manifest,b"bad\n"]):
                with self.assertRaises(ValueError):launcher.acquire("a"*40,launcher.sha(manifest),temp)

    def test_rejects_path_escape_before_fetch(self):
        manifest=self.manifest("../escape.py")
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(launcher,"download",return_value=manifest) as remote:
                with self.assertRaises(ValueError):launcher.acquire("a"*40,launcher.sha(manifest),temp)
                self.assertEqual(remote.call_count,1)

    def test_preserves_tampered_existing_cache(self):
        raw=b"pass\n";manifest=self.manifest(body=raw)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/("source-"+"a"*40)
            (root/"runtime").mkdir(parents=True)
            (root/"runtime/sample.py").write_bytes(b"old\n")
            with patch.object(launcher,"download",return_value=manifest):
                with self.assertRaises(ValueError):launcher.acquire("a"*40,launcher.sha(manifest),temp)
            self.assertEqual((root/"runtime/sample.py").read_bytes(),b"old\n")

    def test_expected_scientific_block_is_reported_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"source";(root/"audit").mkdir(parents=True)
            report={"gate_counts":{"integrity":0,"data":4,"methods":3},"admission_ready":False}
            (root/"audit/preflight.json").write_text(json.dumps(report))
            def run(args,**kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(returncode=2 if "runtime.preflight" in args else 0,stdout="",stderr="")
            output=io.StringIO()
            with patch.object(launcher,"acquire",return_value=root),patch.object(launcher.subprocess,"run",side_effect=run) as subprocess_run,patch.object(launcher,"software_tests") as tests,redirect_stdout(output):
                launcher.main(["--source-commit","a"*40,"--manifest-sha256","b"*64,"--cache",temp])
            tests.assert_not_called()
            self.assertEqual(subprocess_run.call_count,2)
            last=json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(last["admission_exit_code"],2)
            self.assertEqual(last["new_forecasting_model_fits"],0)


if __name__=="__main__":unittest.main()
