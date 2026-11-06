"""Public/privacy gates and a real extracted-ZIP no-data smoke check."""
from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import package_e2_completion as package


class PublicOnlyTests(unittest.TestCase):
    def test_individual_or_native_fields_fail_even_when_nested(self):
        for value in ({"models": [{"predictions": [1.2]}]},
                      {"abstract": "native licensed abstract"},
                      {"models": [{"y_true": [1, 2]}]},
                      {"receipt": {"location": "/workspace/private/run"}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                package.check_public("public/receipt.json", package.json_bytes(value))
        package.check_public("public/status.json", package.json_bytes({
            "predictions": None, "fits": 0, "author_sources": [{"path": "models/author.py", "sha256": "a" * 64}]}))

    def test_no_disguised_daily_values_or_case_ledgers(self):
        for text in ("outcome_day,y_pred\n2026-09-01,0.03\n",
                     "market_day_utc,close\n2026-09-01,100\n",
                     "case_id,category,error\ncase1,tail,0.2\n",
                     "date,value\n2026-09-01,100\n",
                     "model,model\na,b\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                package.check_public("public/aggregate.csv", text.encode())
        package.check_public("public/aggregate.csv", b"model,RMSE_mean,seed\nmodel,0.02,670101\n")

    def test_payload_extensions_and_names_are_constrained(self):
        for name in ("private_results.json", "predictions.csv", "savedrecs.txt", "paper.pdf", "checkpoint.pt", "../../outside.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                package.check_public(name, b"{}")

    def test_duplicate_keys_cannot_hide_private_payload_and_NaN_is_rejected(self):
        for text in ('{"y_true":[1,2],"y_true":null}', '{"score":NaN}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                package.check_public("public/status.json", text.encode())

    def test_input_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "checkout"
            root.mkdir()
            outside = base / "secret"
            outside.write_text("must not package")
            (root / "link").symlink_to(base, target_is_directory=True)
            with self.assertRaises(ValueError):
                package.file_payload(root, "link/secret")
            for name in ("/etc/passwd", "../secret", "link//secret", "a\\b"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    package.file_payload(root, name)

    def test_function_local_dependency_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts/start.py").write_text("def f():\n from helper import x\n return x\n")
            (root / "scripts/helper.py").write_text("import second\nx=1\n")
            (root / "scripts/second.py").write_text("import json\n")
            self.assertEqual(package.local_import_closure(root, {"scripts/start.py"}),
                             {"scripts/start.py", "scripts/helper.py", "scripts/second.py"})


class ArchiveTests(unittest.TestCase):
    def payload(self):
        contents = {"scripts/safe.py": b"print('synthetic only')\n", "status.json": b'{"fits":0}\n',
                    "LEEME.md": package.README.encode()}
        design = {"protocol_id": "synthetic_fixture", "development_sources": {name: "a" * 64 for name in package.OMITTED_DATA},
                  "author_sources": [{"sha256": "b" * 64}]}
        with patch.object(package, "source_contents", return_value=(contents, design, [], ["not_yet_present"])):
            return package.build_payload()

    def test_archive_deterministic_and_manifest_honest_about_exclusions(self):
        first, second = self.payload(), self.payload()
        self.assertEqual(first, second)
        manifest = json.loads(first["MANIFIESTO.json"])
        self.assertFalse(manifest["standalone_without_repository_checkout"])
        self.assertFalse(manifest["market_data_or_native_abstracts_included"])
        self.assertFalse(manifest["individual_predictions_or_error_cases_included"])
        self.assertEqual(manifest["new_fits_performed_by_packager"], 0)
        package.validate_archive(first[package.ARCHIVE], first["MANIFIESTO.json"])

    def test_tamper_or_duplicate_is_rejected(self):
        payload = self.payload()
        buffer = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(payload[package.ARCHIVE])) as source, \
                zipfile.ZipFile(buffer, "w") as destination:
            for info in source.infolist():
                destination.writestr(info, b"forged" if info.filename == "status.json" else source.read(info.filename))
        with self.assertRaises(ValueError):
            package.validate_archive(buffer.getvalue(), payload["MANIFIESTO.json"])
        buffer = io.BytesIO(payload[package.ARCHIVE])
        with zipfile.ZipFile(buffer, "a") as destination:
            with self.assertWarns(UserWarning):
                destination.writestr("status.json", b"duplicate")
        with self.assertRaises(ValueError):
            package.validate_archive(buffer.getvalue(), payload["MANIFIESTO.json"])

    def test_invalid_delivery_preserves_existing_files(self):
        payload = self.payload()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "download"
            package.deliver(payload, output)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            payload[package.ARCHIVE] = b"invalid ZIP"
            with self.assertRaises((ValueError, zipfile.BadZipFile)):
                package.deliver(payload, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_delivery_cannot_follow_a_symlink_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "outside"
            target.mkdir()
            (base / "redirect").symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                package.deliver(self.payload(), base / "redirect/download")
            self.assertEqual(list(target.iterdir()), [])

    def test_unexpected_existing_download_content_is_not_silently_published(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "old_private.csv").write_text("private fixture")
            with self.assertRaises(ValueError):
                package.deliver(self.payload(), output)
            self.assertEqual([p.name for p in output.iterdir()], ["old_private.csv"])


class RealPackageIsolationTests(unittest.TestCase):
    def test_real_extracted_zip_can_plan_and_validate_contract_without_data(self):
        # This test deliberately uses the real allowlist and archive, then an
        # interpreter without site-packages. It cannot pass via an installed
        # working-tree helper or NumPy/Torch dependency.
        required = [ROOT / package.EXPERIMENT / name for name in package.CONFIGS]
        required += [ROOT / "scripts/e2_leakage_controls.py", ROOT / "tests/test_e2_leakage_controls.py"]
        if not all(path.is_file() for path in required):
            self.skipTest("Master completion design/control files are still being integrated")
        payload = package.build_payload()
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory)
            with zipfile.ZipFile(io.BytesIO(payload[package.ARCHIVE])) as archive:
                archive.extractall(isolated)
            for name in package.OMITTED_DATA:
                self.assertFalse((isolated / name).exists())
            self.assertFalse((isolated / ".git").exists())
            result = subprocess.run([sys.executable, "-S", "scripts/run_e2_final_benchmark.py", "--phase", "plan"],
                                    cwd=isolated, capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["dataset_payloads_opened"], 0)
            self.assertEqual(plan["cache_unique_HPO_neural_trajectories"], 9984)
            script = ("import sys;sys.path.insert(0,'scripts');import run_e2_final_benchmark as e;"
                      "d,h,p,c=e.read_design();assert len(c)==180;assert len(d['catalogue'])==20;print('validated')")
            contract = subprocess.run([sys.executable, "-S", "-c", script], cwd=isolated,
                                      capture_output=True, text=True, check=True)
            self.assertEqual(contract.stdout.strip(), "validated")
            self.assertFalse((isolated / "fits").exists())
            manifest = json.loads((isolated / "MANIFIESTO.json").read_text())
            for name, record in manifest["files"].items():
                self.assertEqual(sha256((isolated / name).read_bytes()).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
