"""Dataset-audit persistence, fake API only; no preflight or market fits."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("dataset_storage_fixtures", HERE / "test_colab_github.py")
fixtures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixtures)
storage = fixtures.storage
BRANCH = "e3-dataset-audit-" + fixtures.RUN_ID


def dataset_fixture(root):
    output, archive = fixtures.fixture(root)
    (output / "preflight_receipt.json").unlink()
    receipt = {"kind": "dataset_admission_audit", "repository_HEAD": fixtures.SOURCE,
               "datasets": [{"id": "synthetic_metadata_candidate", "admitted": False,
                             "license_status": "not yet verified"}],
               "datasets_admitted": 0, "new_real_data_fits": 0,
               "new_market_predictions": 0, "new_hypothesis_tests": 0,
               "R1_R3_verified": False}
    (output / "dataset_audit.json").write_text(json.dumps(receipt) + "\n")
    (output / "license_evidence.json").write_text(json.dumps({"license_verified": False}) + "\n")
    fixtures.make_zip(output, archive)
    return output, archive


class DatasetGitHubPersistenceContracts(unittest.TestCase):
    def save(self, output, archive, api):
        with mock.patch.object(fixtures, "BRANCH", BRANCH), mock.patch.object(
                storage, "_request", side_effect=api.request):
            return storage.save_dataset_audit(output, archive, fixtures.TOKEN,
                                              fixtures.SOURCE, fixtures.RUN_ID)

    def test_zero_admission_audit_has_distinct_branch_and_no_false_model_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, archive = dataset_fixture(Path(temporary))
            api = fixtures.FakeGitHub()
            result = self.save(output, archive, api)
            self.assertTrue(result["saved"])
            self.assertEqual(result["branch"], BRANCH)
            self.assertFalse(result["main_updated"])
            self.assertIn("/runs/dataset_admission/", result["run_url"])
            manifest = json.loads(next(raw for raw in api.blobs.values() if b'"schema_version"' in raw))
            self.assertEqual(manifest["kind"], "dataset_admission_audit")
            self.assertEqual(manifest["datasets_total"], 1)
            self.assertEqual(manifest["datasets_admitted_reported"], 0)
            self.assertFalse(manifest["R1_R3_verified"])
            self.assertNotIn("preflight_status", manifest)
            self.assertIn("files/license_evidence.json", manifest["files"])
            self.assertNotIn(fixtures.TOKEN, json.dumps(api.calls))
            self.assertTrue(all(method != "PATCH" for method, _, _ in api.calls))

    def test_profiles_cannot_be_substituted_and_scientific_contracts_are_enforced(self):
        changes = ({"kind": "synthetic_preflight_receipts"}, {"datasets": {}},
                   {"datasets_admitted": True}, {"datasets_admitted": 1},
                   {"R1_R3_verified": True}, {"new_real_data_fits": 1})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                output, archive = dataset_fixture(Path(temporary))
                p = output / "dataset_audit.json"
                receipt = json.loads(p.read_text());receipt.update(change)
                p.write_text(json.dumps(receipt) + "\n")
                fixtures.make_zip(output, archive)
                api = fixtures.FakeGitHub()
                with self.assertRaises(storage.StorageError):
                    self.save(output, archive, api)
                self.assertEqual(api.calls, [])
        with tempfile.TemporaryDirectory() as temporary:
            output, archive = dataset_fixture(Path(temporary))
            with mock.patch.object(storage, "_request") as request, self.assertRaises(storage.StorageError):
                storage.save_run(output, archive, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID)
            request.assert_not_called()
            (output / "preflight_receipt.json").write_text("{}")
            fixtures.make_zip(output, archive)
            with mock.patch.object(storage, "_request") as request, self.assertRaises(storage.StorageError):
                storage.save_dataset_audit(output, archive, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID)
            request.assert_not_called()

    def test_dataset_retry_lost_ack_and_changed_evidence_preserve_existing_branch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, archive = dataset_fixture(Path(temporary))
            api = fixtures.FakeGitHub(lost_ack=True)
            first = self.save(output, archive, api)
            boundary = len(api.calls)
            second = self.save(output, archive, api)
            self.assertTrue(second["already_saved"])
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertFalse(any(method != "GET" for method, _, _ in api.calls[boundary:]))
            (output / "license_evidence.json").write_text('{"license_verified":false,"note":"different"}\n')
            fixtures.make_zip(output, archive)
            before = api.ref.copy()
            with self.assertRaises(storage.StorageConflict):
                self.save(output, archive, api)
            self.assertEqual(api.ref, before)

    def test_raw_csv_or_credential_cannot_be_smuggled_in_dataset_zip(self):
        for name, data in (("raw_prices.csv", b"close\n12.3\n"),
                           ("hidden.json", fixtures.TOKEN.encode())):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                output, archive = dataset_fixture(Path(temporary))
                fixtures.make_zip(output, archive, extra=[(name, data)])
                api = fixtures.FakeGitHub()
                with self.assertRaises(storage.StorageError):
                    self.save(output, archive, api)
                self.assertEqual(api.calls, [])
        with tempfile.TemporaryDirectory() as temporary:
            output, archive = dataset_fixture(Path(temporary))
            (output / "license_evidence.json").write_text(json.dumps({"secret": fixtures.TOKEN}))
            fixtures.make_zip(output, archive)
            api = fixtures.FakeGitHub()
            with self.assertRaises(storage.StorageError):
                self.save(output, archive, api)
            self.assertEqual(api.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
