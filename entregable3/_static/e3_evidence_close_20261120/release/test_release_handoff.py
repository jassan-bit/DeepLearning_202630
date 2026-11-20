"""Synthetic checks only: no model fitting, market inference or public requests."""
import importlib.util
import gzip
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_documentary_archive as archive
import make_documentary_spec as spec
import render_public_acceptance_colab as handoff


class DocumentaryArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-e3-release-")
        self.root = Path(self.temporary.name)
        self.input = self.root / "source"
        self.input.mkdir()
        (self.input / "SYNTHETIC.md").write_text("Artificial documentation; no market inputs.\n")
        (self.input / "SYNTHETIC_summary.csv").write_bytes(b"item,value\r\nsynthetic,2\r\n")
        self.spec = self.root / "spec.json"
        self.value = spec.make(self.input, ["SYNTHETIC.md", "SYNTHETIC_summary.csv"])
        self.spec.write_text(json.dumps(self.value))
        self.output = self.root / "documentary.zip"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, **kwargs):
        return archive.build(self.input, self.spec, "a" * 40, self.output, **kwargs)

    def test_explicit_spec_and_zip_are_exactly_reproducible_and_preserve_scope(self):
        first = self.build()
        body = self.output.read_bytes()
        self.assertEqual(self.build(check=True), first)
        self.assertEqual(self.output.read_bytes(), body)
        with zipfile.ZipFile(self.output) as zipped:
            manifest = json.loads(zipped.read(archive.NAME + "/MANIFEST.json"))
            self.assertEqual(len(manifest["members"]), 2)
            self.assertFalse(manifest["DOI_publication_verified"])
            self.assertFalse(manifest["included_trained_weights"])
            self.assertFalse(manifest["license_selected"])
            self.assertEqual(zipped.read(archive.NAME + "/SYNTHETIC_summary.csv"), b"item,value\r\nsynthetic,2\r\n")

    def test_changed_source_is_rejected_without_replacing_previous_evidence(self):
        self.build()
        before = self.output.read_bytes()
        (self.input / "SYNTHETIC.md").write_text("Changed after review\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            self.build()
        self.assertEqual(self.output.read_bytes(), before)

    def test_path_traversal_weights_and_inferred_licenses_are_rejected(self):
        for name in ("../outside.md", "/outside.md", ".env", "weights.pt", "model.onnx", "data.npz", "paper.pdf", "nested.zip", "LICENSE"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.relative(name)

    def test_symlink_source_is_rejected(self):
        (self.input / "link.md").symlink_to(self.input / "SYNTHETIC.md")
        with self.assertRaisesRegex(ValueError, "symlink"):
            spec.make(self.input, ["link.md"])

    def test_pat_like_value_is_rejected_before_archiving(self):
        (self.input / "SYNTHETIC.md").write_text("ghp_" + "A" * 40)
        self.spec.write_text(json.dumps(spec.make(self.input, ["SYNTHETIC.md"])))
        with self.assertRaisesRegex(ValueError, "credential"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_unreviewed_rights_category_is_rejected(self):
        self.value["entries"][0]["rights_status"] = "automatically_open"
        self.spec.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(ValueError, "rights"):
            self.build()

    def compressed_spec(self, name, raw):
        path = self.input / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        value = spec.make(self.input, [name])
        self.spec.write_text(json.dumps(value))
        return value

    @staticmethod
    def text_zip(name, body):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr(name, body)
        return output.getvalue()

    def test_reviewed_frozen_input_paths_preserve_bytes_and_expanded_manifest(self):
        prefix = "experiments/e3_evidence_close_20261120/inventory/"
        name = prefix + "sources.zip"
        raw = self.text_zip("synthetic.json", b'{"synthetic_only":true}\n')
        value = self.compressed_spec(name, raw)
        self.assertEqual(value["entries"][0]["kind"], archive.FROZEN_INPUT_KIND)
        self.assertEqual(value["entries"][0]["rights_status"], "derived_results_author_review_required")
        self.build()
        with zipfile.ZipFile(self.output) as zipped:
            self.assertEqual(zipped.read(archive.NAME + "/" + name), raw)
            manifest = json.loads(zipped.read(archive.NAME + "/MANIFEST.json"))
            check = manifest["members"][0]["frozen_input_validation"]
            self.assertEqual(check["format"], "zip_text_only")
            self.assertTrue(check["expanded_text_and_credentials_checked"])
            self.assertFalse(manifest["included_trained_weights"])
        # The ledger gzip is also admitted with a bounded UTF-8 CSV header.
        gz = gzip.compress(b"design_id,attempt\r\nSYNTHETIC,1\r\n", mtime=0)
        check = archive.validate_frozen_input(Path(prefix + "attempt_ledger.csv.gz"), gz)
        self.assertEqual(check["header"], ["design_id", "attempt"])

    def test_compressed_input_requires_kind_and_exact_source_namespace(self):
        names = ["book_e3/_static/e3_evidence_close_20261120/inventory/sources.zip",
                 "experiments/other/inventory/sources.zip",
                 "experiments/e3_evidence_close_20261120/release/sources.zip",
                 "experiments/e3_evidence_close_20261120/inventory/nested.zip"]
        for name in names:
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.relative(name, kind=archive.FROZEN_INPUT_KIND)
        with self.assertRaises(ValueError):
            archive.relative("experiments/e3_evidence_close_20261120/inventory/sources.zip", kind="documentation")

    def test_frozen_text_zip_rejects_unsafe_binary_nested_and_secret_members(self):
        source = Path("experiments/e3_evidence_close_20261120/inventory/sources.zip")
        cases = [("../outside.txt", b"synthetic"), ("nested.zip", b"synthetic"),
                 ("model.onnx", b"synthetic"), ("paper.pdf", b"synthetic"),
                 ("binary.txt", b"synthetic\0binary"),
                 ("credentials.txt", ("ghp_" + "A" * 40).encode())]
        for name, raw in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                archive.validate_frozen_input(source, self.text_zip(name, raw))

    def test_frozen_csv_checks_expansion_header_raw_inputs_and_hidden_credentials(self):
        source = Path("experiments/e3_evidence_close_20261120/adequacy/archive/synthetic.csv.gz")
        cases = [b"\n", b"duplicate,duplicate\n1,2\n",
                 b"open,high,low,close,volume\n1,2,3,4,5\n",
                 b"field,value\ncredential," + ("github_pat_" + "A" * 40).encode() + b"\n"]
        for raw in cases:
            with self.subTest(raw=raw[:32]), self.assertRaises(ValueError):
                archive.validate_frozen_input(source, gzip.compress(raw, mtime=0))
        with patch.object(archive, "FROZEN_EXPANDED_LIMIT", 16), self.assertRaisesRegex(ValueError, "expanded"):
            archive.validate_frozen_input(source, gzip.compress(b"field,value\n" + b"synthetic,1\n" * 4, mtime=0))


class RenderHandoffTests(unittest.TestCase):
    def test_import_has_no_network_or_subprocess_action(self):
        source = Path(handoff.__file__)
        with patch("subprocess.run", side_effect=AssertionError("No subprocess on import")):
            loader = importlib.util.spec_from_file_location("synthetic_handoff_import", source)
            module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(module)

    def test_missing_operator_sha_stops_before_any_action(self):
        with patch.object(handoff, "call", side_effect=AssertionError("No checkout without operator SHA")):
            for value in (None, "", "main", "abc1234", "g" * 40):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    handoff.main(value, download=False)

    def test_credentials_are_removed_from_child_environment(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "synthetic-only", "AWS_SECRET_ACCESS_KEY": "synthetic-only",
                                    "CUSTOM_CREDENTIAL": "synthetic-only", "GIT_ASKPASS": "/do-not-use"}):
            env = handoff.clean_environment()
        self.assertNotIn("GITHUB_TOKEN", env)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
        self.assertNotIn("CUSTOM_CREDENTIAL", env)
        self.assertNotIn("GIT_ASKPASS", env)
        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")

    def test_unreviewed_source_stops_before_pip_or_browser(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-render-client-") as directory:
            root = Path(directory)
            commit = "b" * 40
            checkout = root / ("e3-render-client-" + commit)
            source = checkout / handoff.CLIENT
            source.parent.mkdir(parents=True)
            source.write_text("Artificial altered client\n")
            helper = checkout / handoff.HELPER
            helper.parent.mkdir(parents=True)
            helper.write_text("Artificial helper\n")
            def fake_call(command, **kwargs):
                if command[:3] == ["git", "rev-parse", "HEAD"]:
                    return (commit + "\n").encode()
                if command[:3] == ["git", "diff", "--exit-code"]:
                    return b""
                raise AssertionError("No package installation or network expected")
            with patch.object(handoff, "call", side_effect=fake_call), patch("subprocess.run", side_effect=AssertionError("No pip/browser expected")):
                with self.assertRaisesRegex(ValueError, "difiere"):
                    handoff.main(commit, base=root, download=False)

    def test_interrupted_source_cache_is_preserved_before_clean_retry(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-render-cache-") as directory:
            root = Path(directory)
            commit = "c" * 40
            source = root / ("e3-render-client-" + commit)
            source.mkdir()
            (source / "incomplete-fetch-sentinel.txt").write_text("Preserve these bytes\n")
            first_read = True
            def fake_call(command, **kwargs):
                nonlocal first_read
                if command[:3] == ["git", "rev-parse", "HEAD"]:
                    if first_read:
                        first_read = False
                        raise subprocess.CalledProcessError(128, command, stderr=b"No HEAD yet")
                    return (commit + "\n").encode()
                return b""
            with patch.object(handoff, "call", side_effect=fake_call):
                actual = handoff.checkout_source(commit, root, handoff.clean_environment())
            preserved = list(root.glob(source.name + ".incomplete-*"))
            self.assertEqual(actual, source)
            self.assertEqual(len(preserved), 1)
            self.assertEqual((preserved[0] / "incomplete-fetch-sentinel.txt").read_text(), "Preserve these bytes\n")
            self.assertFalse((source / "incomplete-fetch-sentinel.txt").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
