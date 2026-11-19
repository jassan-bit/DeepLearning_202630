"""Offline stage persistence contracts using a fake GitHub API, never real credentials."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import struct
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


target = load("e3_closure_stage_storage_under_test", HERE / "github_store.py")
fixtures = load("e3_closure_existing_github_fixtures", HERE.parents[1] / "e3_remediacion_20261118/evaluation/test_colab_github.py")
api = fixtures.storage


def zipped(name="daily.csv", raw=b"date,close\n2020-01-01,10\n", *, symlink=False):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        if symlink:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, raw)
        else:
            archive.writestr(name, raw)
    return output.getvalue()


def evidence(root, profile="cpu", fits=0):
    output = root / "output"
    output.mkdir()
    receipt = {"kind":"e3_closure_stage_receipt", "source_commit":fixtures.SOURCE,
               "profile":profile, "R1_accredited":False, "R3_accredited":False,
               "new_real_data_fits":fits, "new_market_predictions":0}
    (output / "stage_receipt.json").write_text(json.dumps(receipt) + "\n")
    (output / "datasets").mkdir()
    (output / "datasets/D01.csv").write_text("date,open,high,low,close,volume\n2020-01-01,10,12,9,11,4\n")
    (output / "download.zip").write_bytes(zipped())
    (output / "download.zip.CHECKSUM").write_text(hashlib.sha256((output / "download.zip").read_bytes()).hexdigest() + "  download.zip\n")
    (output / "checkpoint.pt").write_bytes(zipped("archive/data.pkl", b"artificial binary fixture; never deserialize"))
    (output / "checkpoint.npz").write_bytes(zipped("weights.npy", b"artificial array fixture"))
    (output / "worker.log").write_text("Artificial stage fixture.\n")
    (output / "NOTICE.md").write_text("AI-assisted development fixture; no scientific accreditation.\n")
    return output


class StagePersistenceContracts(unittest.TestCase):
    def save(self, output, fake, *, profile="cpu", workers=3):
        branch = target.PROFILES[profile][0] + fixtures.RUN_ID
        with mock.patch.object(fixtures, "BRANCH", branch), mock.patch.object(api, "_request", side_effect=fake.request):
            return target.save(output, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID, api,
                               profile=profile, workers=workers)

    def assert_no_reference_write(self, fake):
        self.assertFalse(any(method == "POST" and path.endswith("/git/refs")
                             for method, path, payload in fake.calls))
        self.assertFalse(any(method in {"PATCH", "DELETE", "PUT"} for method, path, payload in fake.calls))

    def test_exact_cpu_binary_snapshot_preserves_the_full_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = evidence(Path(tmp))
            expected = target.snapshot(output, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID, api, "cpu")
            fake = fixtures.FakeGitHub()
            fake.old.append({"path":"src/existing.py", "type":"blob", "mode":"100755",
                             "sha":api._git_sha(b"print('old')\n"), "size":13})
            result = self.save(output, fake)
            self.assertTrue(result["saved"])
            self.assertTrue(result["all_files_verified"])
            self.assertFalse(result["main_updated"])
            self.assertEqual(result["persisted_files"], len(expected))
            self.assertIn("/e3-closure-cpu-" + fixtures.RUN_ID + "/runs/e3_closure_cpu/", result["run_url"])
            prefix = target.PROFILES["cpu"][1] + fixtures.RUN_ID + "/"
            actual = {row["path"]: row for row in fake.tree}
            for row in fake.old:
                self.assertEqual(actual[row["path"]], row)
            for name, raw in expected.items():
                blob = api._git_sha(raw)
                self.assertEqual(actual[prefix + name]["sha"], blob)
                self.assertEqual(fake.blobs[blob], raw)
            manifest = json.loads(expected["storage_manifest.json"])
            for name in ("download.zip", "download.zip.CHECKSUM", "checkpoint.pt", "checkpoint.npz"):
                self.assertIn(name, manifest["files"])
                self.assertEqual(manifest["files"][name]["sha256"], hashlib.sha256((output/name).read_bytes()).hexdigest())
            self.assertNotIn(fixtures.TOKEN, json.dumps([payload for method, path, payload in fake.calls]))
            references = [i for i, (method, path, payload) in enumerate(fake.calls) if method == "POST" and path.endswith("/git/refs")]
            self.assertEqual(len(references), 1)
            self.assertTrue(any(i < references[0] and method == "GET" and path.endswith("/git/commits/" + fixtures.COMMIT)
                                for i, (method, path, payload) in enumerate(fake.calls)))
            self.assertFalse(any(method in {"PATCH", "DELETE", "PUT"} for method, path, payload in fake.calls))

    def test_identical_retry_and_lost_ack_only_read_the_existing_snapshot(self):
        for lost_ack in (False, True):
            with self.subTest(lost_ack=lost_ack), tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                fake = fixtures.FakeGitHub(lost_ack=lost_ack)
                first = self.save(output, fake)
                boundary, blobs = len(fake.calls), deepcopy(fake.blobs)
                second = self.save(output, fake)
                self.assertEqual(first["commit_sha"], second["commit_sha"])
                self.assertEqual(fake.blobs, blobs)
                self.assertTrue(all(method == "GET" for method, path, payload in fake.calls[boundary:]))
                self.assertEqual(sum(method == "POST" and path.endswith("/git/refs") for method, path, payload in fake.calls), 1)

    def test_changed_local_bytes_cannot_overwrite_an_existing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = evidence(Path(tmp))
            fake = fixtures.FakeGitHub()
            self.save(output, fake)
            prior_ref = deepcopy(fake.ref)
            (output / "worker.log").write_text("different result\n")
            boundary = len(fake.calls)
            with self.assertRaises(api.StorageConflict):
                self.save(output, fake)
            self.assertEqual(fake.ref, prior_ref)
            self.assertTrue(all(method == "GET" for method, path, payload in fake.calls[boundary:]))

    def test_existing_ref_commit_or_tree_tampering_is_refused_without_new_writes(self):
        for variant in ("tag", "parent", "extra_path", "changed_blob"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                fake = fixtures.FakeGitHub()
                self.save(output, fake)
                if variant == "tag":
                    fake.ref["object"]["type"] = "tag"
                elif variant == "parent":
                    fake.commit["parents"] = [{"sha":fixtures.BASE_TREE}]
                elif variant == "extra_path":
                    fake.tree.append({"path":"unexpected.json", "type":"blob", "mode":"100644", "sha":"f"*40})
                else:
                    next(r for r in fake.tree if r["path"].endswith("worker.log"))["sha"] = "f"*40
                prior_ref, boundary = deepcopy(fake.ref), len(fake.calls)
                with self.assertRaises(api.StorageError):
                    self.save(output, fake)
                self.assertEqual(fake.ref, prior_ref)
                self.assertTrue(all(method == "GET" for method, path, payload in fake.calls[boundary:]))

    def test_secret_in_bytes_or_filename_is_rejected_before_any_api_request(self):
        for variant in ("binary", "text", "name"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                if variant == "binary":
                    (output / "uncompressed.pt").write_bytes(b"header" + fixtures.TOKEN.encode())
                elif variant == "text":
                    (output / "worker.log").write_text(fixtures.TOKEN)
                else:
                    (output / (fixtures.TOKEN + ".json")).write_text("{}")
                fake = fixtures.FakeGitHub()
                with self.assertRaises(api.StorageError) as caught:
                    self.save(output, fake)
                self.assertNotIn(fixtures.TOKEN, str(caught.exception))
                self.assertEqual(fake.calls, [])

    def test_secret_in_compressed_zip_pt_or_npz_is_rejected_without_deserializing(self):
        for extension in ("zip", "pt", "npz"):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                (output / ("private." + extension)).write_bytes(zipped("metadata.json", fixtures.TOKEN.encode()))
                fake = fixtures.FakeGitHub()
                with self.assertRaises(api.StorageError) as caught:
                    self.save(output, fake)
                self.assertNotIn(fixtures.TOKEN, str(caught.exception))
                self.assertEqual(fake.calls, [])

    def test_archive_nesting_duplicates_encryption_and_expansion_are_bounded(self):
        variants = ("nested", "duplicate", "absolute", "encrypted", "members", "expanded_member", "expanded_total")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                output = evidence(Path(tmp))
                if variant == "nested":
                    raw = zipped("inner.zip", zipped("data.json", b"{}"))
                elif variant == "duplicate":
                    buffer = io.BytesIO()
                    with warnings.catch_warnings(), zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        warnings.simplefilter("ignore", UserWarning)
                        archive.writestr("same.json", "{}")
                        archive.writestr("same.json", "{}")
                    raw = buffer.getvalue()
                elif variant == "absolute":
                    raw = zipped("/outside.json", b"{}")
                elif variant == "encrypted":
                    # Flip both standard flag fields; refusal occurs before any decrypt/read.
                    encrypted = bytearray(zipped("data.json", b"{}"))
                    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
                        position = encrypted.index(signature) + offset
                        struct.pack_into("<H", encrypted, position, struct.unpack_from("<H", encrypted, position)[0] | 1)
                    raw = bytes(encrypted)
                elif variant == "members":
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        for index in range(2049):
                            archive.writestr(str(index) + ".json", b"")
                    raw = buffer.getvalue()
                elif variant == "expanded_member":
                    raw = zipped("large.txt", b"a" * 4097)
                    stack.enter_context(mock.patch.object(target, "MAX_FILE", 4096))
                else:
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr("one.txt", b"a" * 3000)
                        archive.writestr("two.txt", b"a" * 3000)
                    raw = buffer.getvalue()
                    stack.enter_context(mock.patch.object(target, "MAX_TOTAL", 4096))
                (output / "unsafe.zip").write_bytes(raw)
                fake = fixtures.FakeGitHub()
                with self.assertRaises(api.StorageError):
                    self.save(output, fake)
                self.assertEqual(fake.calls, [])

    def test_missing_malformed_or_nonobject_stage_receipt_has_a_fixed_local_error(self):
        for variant in ("missing", "malformed", "nonobject", "unicode"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                receipt = output / "stage_receipt.json"
                if variant == "missing":
                    receipt.unlink()
                elif variant == "malformed":
                    receipt.write_text("{invalid\n")
                elif variant == "nonobject":
                    receipt.write_text("[]\n")
                else:
                    receipt.write_bytes(b"\xff\xfe\xff")
                fake = fixtures.FakeGitHub()
                with self.assertRaises(api.StorageError):
                    self.save(output, fake)
                self.assertEqual(fake.calls, [])

    def test_filesystem_and_embedded_archive_paths_or_symlinks_are_refused(self):
        for variant in ("file_link", "directory_link", "backslash", "source_package", "reserved_manifest", "unsupported_suffix", "archive_parent", "archive_symlink"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = evidence(root)
                if variant == "file_link":
                    (root / "outside.json").write_text("{}")
                    (output / "linked.json").symlink_to(root / "outside.json")
                elif variant == "directory_link":
                    (root / "linked").symlink_to(output, target_is_directory=True)
                    output = root / "linked"
                elif variant == "backslash":
                    (output / "..\\outside.json").write_text("{}")
                elif variant == "source_package":
                    (output / "source_package").mkdir()
                    (output / "source_package/private.json").write_text("{}")
                elif variant == "reserved_manifest":
                    (output / "storage_manifest.json").write_text("{}")
                elif variant == "unsupported_suffix":
                    (output / "run.sh").write_text("echo not evidence\n")
                elif variant == "archive_parent":
                    (output / "unsafe.zip").write_bytes(zipped("../outside.json", b"{}"))
                else:
                    (output / "unsafe.zip").write_bytes(zipped("link.json", b"../outside", symlink=True))
                fake = fixtures.FakeGitHub()
                with self.assertRaises(api.StorageError):
                    self.save(output, fake)
                self.assertEqual(fake.calls, [])

    def test_false_scientific_accreditation_or_cpu_fits_fail_before_upload(self):
        mutations = ({"R1_accredited":True}, {"R3_accredited":True}, {"new_real_data_fits":1},
                     {"new_real_data_fits":-1}, {"new_real_data_fits":False},
                     {"profile":"pilot"}, {"source_commit":fixtures.BASE_TREE})
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                p = output / "stage_receipt.json"
                receipt = json.loads(p.read_text()); receipt.update(mutation); p.write_text(json.dumps(receipt))
                fake = fixtures.FakeGitHub()
                with self.assertRaises(api.StorageError):
                    self.save(output, fake)
                self.assertEqual(fake.calls, [])

    def test_development_pilot_can_persist_an_honest_fit_counter_without_certifying_R1_R3(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = evidence(Path(tmp), profile="pilot", fits=4)
            fake = fixtures.FakeGitHub()
            result = self.save(output, fake, profile="pilot")
            self.assertTrue(result["saved"])
            self.assertIn("e3-closure-pilot-", result["run_url"])
            self.assertIn("runs/e3_closure_pilot/", result["run_url"])
            self.assertFalse(result["main_updated"])

    def test_corrupt_uploaded_blob_or_missing_old_source_never_creates_a_ref(self):
        for options in ({"bad_blob":True}, {"drop_old":True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                fake = fixtures.FakeGitHub(**options)
                with self.assertRaises(api.StorageError):
                    self.save(output, fake)
                self.assertIsNone(fake.ref)
                self.assert_no_reference_write(fake)

    def test_new_tree_must_be_complete_and_exact_before_reference_publication(self):
        for variant in ("truncated", "changed_blob", "injected_path"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                fake = fixtures.FakeGitHub()
                request = fake.request

                def tampered(method, path, token, payload=None, **kwargs):
                    response = request(method, path, token, payload, **kwargs)
                    if method == "GET" and path.endswith("/git/trees/" + fixtures.NEW_TREE + "?recursive=1"):
                        response = deepcopy(response)
                        if variant == "truncated":
                            response["truncated"] = True
                        elif variant == "changed_blob":
                            next(row for row in response["tree"] if row["path"].endswith("worker.log"))["sha"] = "f" * 40
                        else:
                            response["tree"].append({"path":"injected.json", "type":"blob", "mode":"100644", "sha":"f"*40})
                    return response

                fake.request = tampered
                with self.assertRaises(api.StorageError):
                    self.save(output, fake)
                self.assertIsNone(fake.ref)
                self.assert_no_reference_write(fake)

    def test_invalid_profile_run_or_exceeded_budget_fails_before_any_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = evidence(Path(tmp))
            for profile, source, run in (("main",fixtures.SOURCE,fixtures.RUN_ID),
                                         ("cpu","bad",fixtures.RUN_ID),
                                         ("cpu",fixtures.SOURCE,"../../main")):
                with self.subTest(profile=profile, source=source, run=run):
                    fake = fixtures.FakeGitHub()
                    with mock.patch.object(api, "_request", side_effect=fake.request), self.assertRaises(api.StorageError):
                        target.save(output, fixtures.TOKEN, source, run, api, profile=profile)
                    self.assertEqual(fake.calls, [])
            for constant in ("MAX_FILE", "MAX_TOTAL"):
                fake = fixtures.FakeGitHub()
                with mock.patch.object(target, constant, 1), self.assertRaises(api.StorageError):
                    self.save(output, fake)
                self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
