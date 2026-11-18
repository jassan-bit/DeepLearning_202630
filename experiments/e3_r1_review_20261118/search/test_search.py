"""Meaningful parser/transport/storage contracts; no network or real tokens."""
from pathlib import Path
from unittest import mock
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.error

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


collector = load("r1_native_collector_under_test", HERE / "collect.py")
storage = load("r1_native_storage_under_test", HERE / "github_store.py")
fixtures = load("r1_existing_storage_fixtures", HERE.parents[1] / "e3_remediacion_20261118/evaluation/test_colab_github.py")


class Response(io.BytesIO):
    status = 200


def evidence(root):
    output = root / "output"
    output.mkdir()
    receipt = {"kind": "e3_r1_native_bibliographic_collection", "source_commit": fixtures.SOURCE,
               "attempts": [], "R1_accredited": False, "human_external_review_completed": False,
               "new_real_data_fits": 0, "new_market_predictions": 0, "new_statistical_contrasts": 0}
    (output / "search_receipt.json").write_text(json.dumps(receipt))
    (output / "screening.csv").write_text("title,title_decision\n")
    return output


class BibliographicContracts(unittest.TestCase):
    def test_real_zero_is_distinct_from_failed_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zero = collector.attempt("Crossref", collector.QUERIES[0], root,
                opener=lambda *a, **k: Response(b'{"message":{"total-results":0,"items":[]}}'))
            self.assertEqual(zero["status"], "response_parsed_not_screened")
            self.assertEqual(zero["native_result_count"], 0)
            self.assertTrue((root / zero["response_file"]).is_file())
            with mock.patch.object(collector.urllib.request, "urlopen", side_effect=urllib.error.URLError("blocked")):
                failed = collector.attempt("Crossref", collector.QUERIES[0], root,
                    opener=collector.urllib.request.urlopen)
            self.assertEqual(failed["status"], "blocked_or_unavailable")
            self.assertIsNone(failed["native_result_count"])
            self.assertFalse(failed["screening_complete"])

    def test_metadata_normalization_and_missing_count_fail_closed(self):
        samples = {
            "Crossref": {"message": {"total-results": 1, "items": [{"title": ["A"], "DOI": "10/x"}]}},
            "DBLP": {"result": {"hits": {"@total": "1", "hit": [{"info": {"title": "A", "doi": "10/x"}}]}}},
            "OpenAlex": {"meta": {"count": 1}, "results": [{"title": "A", "doi": "10/x"}]},
            "SemanticScholar": {"total": 1, "data": [{"title": "A", "externalIds": {"DOI": "10/x"}}]},
        }
        for provider, data in samples.items():
            count, records = collector.normalized(provider, json.dumps(data))
            self.assertEqual((count, records[0]["title"], records[0]["doi"]), (1, "A", "10/x"))
            with self.assertRaises((KeyError, TypeError, ValueError)):
                collector.normalized(provider, '{}')
        xml = '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/"><o:totalResults>1</o:totalResults><entry><title>A</title><id>https://arxiv.org/abs/1</id></entry></feed>'
        self.assertEqual(collector.normalized("arXiv", xml)[0], 1)

    def test_snapshot_rejects_secret_changed_raw_and_false_scientific_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = evidence(root)
            receipt_path = output / "search_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["R1_accredited"] = True
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaises(fixtures.storage.StorageError):
                storage.snapshot(output, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID, fixtures.storage)
            receipt["R1_accredited"] = False
            receipt["attempts"] = [{"response_file": "responses/Crossref_Q1.json", "response_bytes": 2, "response_sha256": "0" * 64}]
            receipt_path.write_text(json.dumps(receipt))
            (output / "responses").mkdir()
            (output / "responses/Crossref_Q1.json").write_text('{}')
            with self.assertRaises(fixtures.storage.StorageError):
                storage.snapshot(output, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID, fixtures.storage)
            (output / "responses/Crossref_Q1.json").write_text(fixtures.TOKEN)
            with self.assertRaises(fixtures.storage.StorageError):
                storage.snapshot(output, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID, fixtures.storage)

    def test_atomic_save_retry_and_lost_ack_preserve_every_source_blob(self):
        for lost_ack in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                fake = fixtures.FakeGitHub(lost_ack=lost_ack)
                with mock.patch.object(fixtures, "BRANCH", "e3-r1-search-" + fixtures.RUN_ID), mock.patch.object(fixtures.storage, "_request", side_effect=fake.request):
                    first = storage.save(output, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID, fixtures.storage)
                    blobs_before = len(fake.blobs)
                    second = storage.save(output, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID, fixtures.storage)
                self.assertEqual(first["commit_sha"], second["commit_sha"])
                self.assertFalse(first["main_updated"])
                self.assertEqual(blobs_before, len(fake.blobs))
                self.assertTrue(any(r["path"] == "README.md" for r in fake.tree))

    def test_corrupt_blob_or_lost_source_never_creates_results_ref(self):
        for kwargs in ({"bad_blob": True}, {"drop_old": True}):
            with tempfile.TemporaryDirectory() as tmp:
                output = evidence(Path(tmp))
                fake = fixtures.FakeGitHub(**kwargs)
                with mock.patch.object(fixtures, "BRANCH", "e3-r1-search-" + fixtures.RUN_ID), mock.patch.object(fixtures.storage, "_request", side_effect=fake.request):
                    with self.assertRaises(fixtures.storage.StorageError):
                        storage.save(output, fixtures.TOKEN, fixtures.SOURCE, fixtures.RUN_ID, fixtures.storage)
                self.assertIsNone(fake.ref)


if __name__ == "__main__":
    unittest.main()
