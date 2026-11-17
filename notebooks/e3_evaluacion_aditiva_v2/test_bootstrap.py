"""Mocked coordinator checks: no network, CUDA, data admission or inference."""
from contextlib import redirect_stdout
import csv
from datetime import date, timedelta
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("frozen_bootstrap", HERE / "bootstrap.py")
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


def csv_payload(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


class FakeAPI(b.GitHub):
    def __init__(self, current="old"):
        super().__init__("hidden")
        self.current, self.calls = current, []

    def api(self, method, path, value=None, **kwargs):
        self.calls.append((method, path, value))
        if path == "git/blobs": return {"sha": "blob"}
        if path == "git/trees": return {"sha": "tree"}
        if path == "git/commits": return {"sha": "new"}
        if path.startswith("git/ref/heads/"): return {"object": {"sha": self.current}}
        if path == "git/refs" or path.startswith("git/refs/heads/"):
            return {"object": {"sha": "new"}}
        raise AssertionError(path)


class PublicationTests(unittest.TestCase):
    def test_sparse_initial_commit_with_parent_and_identity(self):
        client = FakeAPI()
        self.assertEqual(client.publish({b.PREFIX + "/plan.json": b"{}"}), "new")
        commit = client.calls[-2][2]
        self.assertEqual(commit["parents"], [b.RESULT_COMMIT])
        self.assertEqual(commit["author"], b.COMMIT_IDENTITY)
        self.assertEqual(commit["committer"], b.COMMIT_IDENTITY)
        self.assertNotIn("base_tree", next(v for _, p, v in client.calls if p == "git/trees"))

    def test_nonforced_update(self):
        client = FakeAPI()
        self.assertEqual(client.publish({b.PREFIX + "/summary.json": b"{}"}, "old", "base"), "new")
        commit = next(v for _, p, v in client.calls if p == "git/commits")
        self.assertEqual(commit["parents"], ["old"])
        self.assertEqual(commit["author"], b.COMMIT_IDENTITY)
        self.assertEqual(client.calls[-1][2], {"sha": "new", "force": False})

    def test_writer_race_preserves_branch(self):
        client = FakeAPI("changed")
        with self.assertRaisesRegex(RuntimeError, "rama cambió"):
            client.publish({b.PREFIX + "/summary.json": b"{}"}, "old", "base")
        self.assertFalse(any(method == "PATCH" for method, _, _ in client.calls))

    def test_api_validation_detail_redacts_credentials(self):
        token = "test-private-credential"
        body = json.dumps({"message": "Validation failed " + token,
            "errors": [{"resource": "Commit", "field": "author.email", "code": "invalid",
                        "private_payload": "must not appear"}],
            "unapproved": "must not appear"}).encode()
        error = HTTPError("https://api.github.com/", 422, "failed", {}, io.BytesIO(body))
        with patch.object(b, "build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(RuntimeError) as raised:
                b.GitHub(token).api("POST", "git/commits", {})
        message = str(raised.exception)
        self.assertIn("HTTP 422", message)
        self.assertIn("author.email", message)
        self.assertIn("[redacted]", message)
        self.assertNotIn(token, message)
        self.assertNotIn("must not appear", message)

    def test_non_json_validation_detail_preserves_safe_status(self):
        error = HTTPError("https://api.github.com/", 422, "failed", {}, io.BytesIO(b"not JSON"))
        with patch.object(b, "build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaisesRegex(RuntimeError, "HTTP 422$"):
                b.GitHub("hidden").api("POST", "git/commits", {})


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.directory.name)
        self.test_dates = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608)]
        self.original = [{"model": "E2_model_" + str(i), "RMSE": .004, "MAE": .003,
                          "R": 6 if i < 13 else 1, "n_common_dates": 608} for i in range(16)]
        self.source_manifest = {"run_id": "synthetic-test-only", "source_hashes": {}}
        self.snapshot = {"files": {n: {"sha256": "a" * 64} for n in (
            "data_admission.json", "manifest.json", "additive_positive_head/best.pt",
            "additive_positive_head/validation_predictions.csv")}}
        self.snapshot["files"]["additive_positive_head/best.pt"]["sha256"] = "b" * 64
        self.client = self.remote()
        self.worker_calls = 0

    def tearDown(self):
        self.directory.cleanup()

    def remote(self):
        class Remote:
            token = "hidden"
            head = None
            files = {}
            blobs = {}
            def branch(self): return self.head, "tree", dict(self.files)
            def blob(self, sha): return self.blobs[sha]
            def publish(self, entries, head=None, tree_sha=None):
                if head is not None and head != self.head: raise RuntimeError("race")
                for path, payload in entries.items():
                    sha = b.digest(payload)
                    self.files[path], self.blobs[sha] = sha, payload
                self.head = "commit" + str(len(self.blobs))
                return self.head
        return Remote()

    def download(self, commit, relative, target, expected):
        if relative.endswith("snapshot_manifest.json"): payload = b.canonical(self.snapshot)
        elif relative.endswith("/manifest.json"): payload = b.canonical(self.source_manifest)
        elif relative.endswith("partition_audit.json"):
            payload = b.canonical({"final_partition": {"evaluation": {"sample_IDs": self.test_dates}}})
        elif relative.endswith("benchmark_A2.csv"): payload = csv_payload(self.original)
        elif relative.endswith("preparar_e3_innovacion_runtime.py"):
            payload = b'def preparar_entorno(workspace, device):\n return "python", {"GITHUB_TOKEN":"hidden", "OMP_NUM_THREADS":"2"}, {}\n'
        else: payload = b"synthetic mocked input only\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return payload

    def results(self, plan):
        predictions = [{"date": d, "target_sigma7": .01, "prediction_sigma7": .011,
                        "known_floor": .009} for d in self.test_dates]
        error = .011 - .01
        comparison = [{"model": r["model"], "RMSE": r["RMSE"], "MAE": r["MAE"],
                       "predictors": r["R"], "evaluation_examples": 608} for r in self.original]
        comparison.append({"model": "CNN-LSTM-atencion_aditiva_v2", "RMSE": error,
                           "MAE": error, "predictors": 1, "evaluation_examples": 608})
        payloads = {"predictions.csv": csv_payload(predictions), "comparison.csv": csv_payload(comparison)}
        names = ("snapshot_manifest.json", "manifest.json", "data_admission.json",
            "additive_positive_head/best.pt", "additive_positive_head/validation_predictions.csv",
            "current_cohort_arrays.npz", *plan["e2_input_hashes"])
        summary = {"status": "completed", "scope": "historical_descriptive", "variant": plan["model"],
            "frozen_checkpoint_sha256": plan["checkpoint_sha256"], "checkpoint_sha256": plan["checkpoint_sha256"],
            "source_result_commit": b.RESULT_COMMIT, "result_source_commit": b.RESULT_COMMIT,
            "E2_source_commit": b.E2_COMMIT, "data_npz_sha256": b.NPZ_SHA, "evaluation_n": 608,
            "source_run_id": self.source_manifest["run_id"], "source_hashes": {}, "seed": 20261116,
            "no_training": True, "rows_dropped": 0, "statistical_tests_performed": False,
            "R1_R3_verified": False, "winner_declared": False, "best_epoch_frozen_before_evaluation": 38,
            "input_hashes": {n: b.digest((self.workspace / "inputs" / n).read_bytes()) for n in names},
            "artifact_sha256": {n: b.digest(p) for n, p in payloads.items()},
            "validation_roundtrip": {"examples": 116, "passed_before_test_members_opened": True,
                                      "maximum_absolute_error": 0},
            "training": {k: 0 for k in ("new_model_fits", "new_scaler_fits", "optimizer_steps",
                                        "bias_matching_calls", "calibration_fits")},
            "evaluation": {"RMSE": error, "MAE": error}}
        payloads["summary.json"] = b.canonical(summary)
        return payloads

    def worker(self, arguments, **kwargs):
        self.worker_calls += 1
        self.assertIsNotNone(self.client.head, "plan must be acknowledged before worker")
        self.assertNotIn("GITHUB_TOKEN", kwargs["env"])
        plan = json.loads(self.client.blob(self.client.files[b.PREFIX + "/plan.json"]))
        output = self.workspace / "output"
        output.mkdir()
        for name, payload in self.results(plan).items(): (output / name).write_bytes(payload)

    def run_coordinator(self):
        with patch.object(b, "download", self.download), patch.object(b, "_token", return_value="hidden"), \
             patch.object(b, "GitHub", return_value=self.client), \
             patch.object(b.subprocess, "run", side_effect=self.worker), redirect_stdout(io.StringIO()):
            return b.run(self.workspace)

    def test_prepare_then_complete_then_remote_idempotence(self):
        self.run_coordinator()
        self.assertIn(b.PREFIX + "/publication.json", self.client.files)
        self.run_coordinator()
        self.assertEqual(self.worker_calls, 1)

    def test_recover_local_after_upload_failure_without_worker_repeat(self):
        self.run_coordinator()
        before = {p.name: p.read_bytes() for p in (self.workspace / "output").iterdir()}
        self.client.files = {b.PREFIX + "/plan.json": self.client.files[b.PREFIX + "/plan.json"]}
        self.run_coordinator()
        self.assertEqual(self.worker_calls, 1)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.workspace / "output").iterdir()})
        self.assertIn(b.PREFIX + "/publication.json", self.client.files)

    def test_partial_local_preserved_without_worker(self):
        self.run_coordinator()
        self.client.files = {b.PREFIX + "/plan.json": self.client.files[b.PREFIX + "/plan.json"]}
        (self.workspace / "output" / "summary.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "parciales"):
            self.run_coordinator()
        self.assertEqual(self.worker_calls, 1)
        self.assertTrue((self.workspace / "output" / "predictions.csv").is_file())

    def test_local_hash_change_rejected_before_upload(self):
        self.run_coordinator()
        self.client.files = {b.PREFIX + "/plan.json": self.client.files[b.PREFIX + "/plan.json"]}
        with (self.workspace / "output" / "predictions.csv").open("ab") as stream: stream.write(b"changed")
        with self.assertRaisesRegex(RuntimeError, "distintos del plan"):
            self.run_coordinator()
        self.assertEqual(self.worker_calls, 1)
        self.assertNotIn(b.PREFIX + "/publication.json", self.client.files)


if __name__ == "__main__":
    unittest.main()
