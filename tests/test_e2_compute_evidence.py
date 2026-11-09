"""Test evidence integrity and published measurements without training or Torch."""
from copy import deepcopy
import base64
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import package_e2_compute_evidence as evidence


REGISTRATION = ROOT/"experiments/e2_compute_optimization_20261109/registration.json"
PARITY = ROOT/"experiments/e2_compute_optimization_20261109/cpu_conv_parity.json"
DATA = ROOT/"book_e2/_static/compute_20261109"


def phase_fixture(phase):
    count = {"kernels": 8, "threads": 12, "scaling": 86}[phase]
    rows = []
    for index in range(count):
        job = {"job_id": phase+"-"+str(index).zfill(3), "phase": phase,
               "model": "TimesNet", "epochs": 2, "seed": 272201,
               "batch_windows": None, "threads": 1, "backend": "dense"}
        rows.append({"job": job, "status": "success", "optimizer_steps": 2,
                     "elapsed_wall_seconds": 1., "replacement_attempt": False,
                     "worker_profile": {"sources_unchanged_during_fit": True,
                                        "final_market_observations_read": False, "job": job}})
    reg = json.loads(REGISTRATION.read_text())
    return {"phase": phase, "status": "completed_all_attempts", "all_requested_jobs_attempted": True,
            "no_final_market_data_read": True, "scientific_HPO_or_NAS_executed": False,
            "registration_sha256": evidence.REGISTRATION_SHA256,
            "registration_commit": evidence.REGISTRATION_COMMIT,
            "plan": {"registration_sha256": evidence.REGISTRATION_SHA256,
                     "registration_commit": evidence.REGISTRATION_COMMIT,
                     "source_protocol_sha256": reg["source_protocol_sha256"],
                     "jobs": [deepcopy(row["job"]) for row in rows]},
            "results": rows, "micro_fallback_results": [], "known_optimizer_steps": count*2,
            "unknown_optimizer_step_workers": 0, "elapsed_workers_sum_seconds": float(count)}


def variance_fixture():
    models = {}
    for name in evidence.MODELS:
        models[name] = {}
        for key, phase in (("selection", "epoch_selection"), ("refit", "fresh_refit")):
            models[name][key] = [{"model": name, "phase": phase, "seed": seed,
                                  "status": "completed", "optimizer_steps": 1,
                                  "elapsed_wall_seconds": 1., "replacement_attempt": False,
                                  "worker_profile": {}} for seed in evidence.SEEDS]
    return {"registration_file_sha256": evidence.REGISTRATION_SHA256,
            "registration_commit": evidence.REGISTRATION_COMMIT, "models": models,
            "supersedes_previous_attempts": False, "replacement_attempts": 0,
            "final_2025_2026_access": False, "final_R_set": False, "final_power_claimed": False,
            "HPO_executed": False, "NAS_search_executed": False, "blocked_refits": {},
            "paired_3x5_complete": True, "fixed_epoch_counts": {name: 5 for name in evidence.MODELS},
            "resource_costs": {"fit_attempts": 30, "known_optimizer_steps": 30,
                               "unknown_optimizer_step_attempts": 0,
                               "known_failed_optimizer_steps": 0, "elapsed_worker_wall_seconds": 30.}}


def chain_bytes(*, dated=False):
    row = {"sequence": 0, "previous_sha256": "0"*64, "kind": "résultat"}
    options = {"sort_keys": True, "allow_nan": False}
    if not dated:
        options.update(ensure_ascii=False, separators=(",", ":"))
    row["event_sha256" if dated else "sha256"] = hashlib.sha256(json.dumps(row, **options).encode()).hexdigest()
    return (json.dumps(row, ensure_ascii=False)+"\n").encode()


class ComputeEvidenceIntegrityTests(unittest.TestCase):
    def check_phase(self, value, phase):
        return evidence.validate_phase(value, phase, json.loads(REGISTRATION.read_text()),
                                       evidence.REGISTRATION_SHA256, evidence.REGISTRATION_COMMIT)

    def check_variance(self, value):
        return evidence.validate_variance(value, evidence.REGISTRATION_SHA256,
                                          evidence.REGISTRATION_COMMIT)

    def test_nonfinite_and_duplicate_json_cannot_enter_receipts(self):
        for raw in (b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1,"a":2}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                evidence.strict_json(raw)

    def test_failed_unknown_steps_remain_null_and_known_failed_cost_is_counted(self):
        rows = [{"status": "timeout", "optimizer_steps": None, "elapsed_wall_seconds": 4.},
                {"status": "failed", "optimizer_steps": 7, "elapsed_wall_seconds": 2.},
                {"status": "success", "optimizer_steps": 11, "elapsed_wall_seconds": 3.}]
        result = evidence.cost_inventory(rows, success_status="success")
        self.assertIsNone(result["total_optimizer_steps"])
        self.assertEqual(result["known_optimizer_steps"], 18)
        self.assertEqual(result["known_failed_optimizer_steps"], 7)
        self.assertEqual(result["unknown_optimizer_step_workers"], 1)
        self.assertEqual(result["elapsed_workers_sum_seconds"], 9.)

    def test_completed_steps_cannot_be_missing_negative_or_boolean(self):
        for bad in (None, -1, True):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                evidence.cost_inventory([{"status": "success", "optimizer_steps": bad,
                                          "elapsed_wall_seconds": 1}], success_status="success")

    def test_missing_duplicate_and_misbound_registered_attempts_are_rejected(self):
        for phase in ("kernels", "threads", "scaling"):
            good = phase_fixture(phase)
            self.assertEqual(self.check_phase(good, phase)["primary"]["attempts"], len(good["results"]))
            missing, duplicate, wrong_commit = deepcopy(good), deepcopy(good), deepcopy(good)
            missing["results"].pop()
            duplicate["results"][1] = deepcopy(duplicate["results"][0])
            wrong_commit["plan"]["registration_commit"] = "0"*40
            for bad in (missing, duplicate, wrong_commit):
                with self.subTest(phase=phase), self.assertRaises(ValueError):
                    self.check_phase(bad, phase)

    def test_separate_microprofile_preserves_original_failure_and_censored_cost(self):
        value = phase_fixture("scaling")
        parent = value["results"][0]
        parent.update(status="timeout", optimizer_steps=None, worker_profile=None)
        micro = deepcopy(value["results"][1])
        micro["job"] = deepcopy(parent["job"])
        micro["job"].update(phase="micro_fallback", job_id="scaling-000-micro",
                            parent_job_id="scaling-000", seed=272501, epochs=1, batch_windows=32)
        micro.update(separate_micro_profile_not_replacement=True)
        value.update(micro_fallback_results=[micro], known_optimizer_steps=172,
                     unknown_optimizer_step_workers=1, elapsed_workers_sum_seconds=87.)
        result = self.check_phase(value, "scaling")
        self.assertEqual(result["primary"]["failed_attempts"], 1)
        self.assertEqual(result["separate_microprofiles"]["attempts"], 1)
        self.assertIsNone(result["combined"]["total_optimizer_steps"])
        misbound = deepcopy(value)
        misbound["micro_fallback_results"][0]["job"]["parent_job_id"] = "scaling-001"
        with self.assertRaises(ValueError):
            self.check_phase(misbound, "scaling")

    def test_five_seed_gate_blocks_all_refits_and_forbids_seed_replacement(self):
        complete = variance_fixture()
        self.assertEqual(self.check_variance(complete)["attempts"], 30)
        bad_seed = deepcopy(complete)
        bad_seed["models"][evidence.MODELS[0]]["selection"][0]["seed"] = 999
        with self.assertRaises(ValueError):
            self.check_variance(bad_seed)
        blocked = deepcopy(complete)
        blocked["models"][evidence.MODELS[0]]["selection"][0].update(status="failed", optimizer_steps=None)
        blocked["models"][evidence.MODELS[0]]["refit"] = []
        blocked["blocked_refits"][evidence.MODELS[0]] = "Missing fifth complete seed"
        blocked["paired_3x5_complete"] = False
        blocked["resource_costs"].update(fit_attempts=25, known_optimizer_steps=24,
                                       unknown_optimizer_step_attempts=1, elapsed_worker_wall_seconds=25.)
        result = self.check_variance(blocked)
        self.assertEqual(result["blocked_refits_not_attempted"], 5)
        self.assertIsNone(result["total_optimizer_steps"])
        blocked["models"][evidence.MODELS[0]]["refit"] = complete["models"][evidence.MODELS[0]]["refit"]
        with self.assertRaises(ValueError):
            self.check_variance(blocked)

    def test_both_hash_chain_conventions_and_incomplete_tail_are_preserved(self):
        for dated in (False, True):
            raw = chain_bytes(dated=dated)
            self.assertEqual(evidence.verify_chain(raw)["events_verified"], 1)
            self.assertTrue(evidence.verify_chain(raw+b'{"partial":')["incomplete_tail"])
            with self.assertRaises(ValueError):
                evidence.verify_chain(raw.replace("résultat".encode(), b"changed"))

    def test_recorded_source_hash_detects_postfit_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"scripts").mkdir()
            source = root/"scripts/model.py"
            source.write_bytes(b"original\n")
            record = {"source_sha256_before": {"scripts/model.py": evidence.digest(source.read_bytes())}}
            observed = {}
            self.assertEqual(evidence.check_sources(record, root, observed), 1)
            self.assertIn("scripts/model.py", observed)
            source.write_bytes(b"modified\n")
            with self.assertRaises(ValueError):
                evidence.check_sources(record, root)

    def test_explicit_archived_version_keeps_legitimate_source_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"scripts").mkdir()
            (root/"archives").mkdir()
            (root/"scripts/runner.py").write_bytes(b"new serializer\n")
            (root/"archives/registered.py").write_bytes(b"original serializer\n")
            old_sha = evidence.digest((root/"archives/registered.py").read_bytes())
            new_sha = evidence.digest((root/"scripts/runner.py").read_bytes())
            historical = {("scripts/runner.py", old_sha): "archives/registered.py"}
            observed = {}
            with patch.dict(evidence.HISTORICAL_SOURCES, historical, clear=True):
                old = {"source_sha256_before": {"scripts/runner.py": old_sha},
                       "source_sha256_after": {"scripts/runner.py": old_sha}}
                new = {"source_sha256_before": {"scripts/runner.py": new_sha},
                       "source_sha256_after": {"scripts/runner.py": new_sha}}
                evidence.check_sources([old, new], root, observed)
                self.assertEqual(set(observed["scripts/runner.py"]), {old_sha, new_sha})
                self.assertEqual(observed["scripts/runner.py"][old_sha], "archives/registered.py")
                old["source_sha256_after"]["scripts/runner.py"] = new_sha
                with self.assertRaises(ValueError):
                    evidence.check_sources(old, root)

    def test_lazy_import_accepts_verified_addition_but_preserves_every_before_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"scripts").mkdir()
            (root/"scripts/base.py").write_bytes(b"base module\n")
            (root/"scripts/lazy.py").write_bytes(b"lazy architecture module\n")
            base_sha = evidence.digest((root/"scripts/base.py").read_bytes())
            lazy_sha = evidence.digest((root/"scripts/lazy.py").read_bytes())
            record = {"sources_unchanged_during_fit": True,
                      "source_sha256_before": {"scripts/base.py": base_sha},
                      "source_sha256_after": {"scripts/base.py": base_sha, "scripts/lazy.py": lazy_sha}}
            observed = {}
            self.assertEqual(evidence.check_sources(record, root, observed), 3)
            self.assertEqual(set(observed), {"scripts/base.py", "scripts/lazy.py"})
            for change in ("missing", "changed", "unverified_lazy"):
                bad = deepcopy(record)
                if change == "missing":
                    del bad["source_sha256_after"]["scripts/base.py"]
                elif change == "changed":
                    bad["source_sha256_after"]["scripts/base.py"] = "0"*64
                else:
                    bad["source_sha256_after"]["scripts/lazy.py"] = "0"*64
                with self.subTest(change=change), self.assertRaises(ValueError):
                    evidence.check_sources(bad, root)
            fixed = deepcopy(record)
            fixed["version"] = evidence.VARIANCE_VERSION
            fixed["source_hashes_unchanged"] = True
            with self.assertRaises(ValueError):
                evidence.check_sources(fixed, root)

    def test_only_exact_registered_protocol_json_is_admitted_as_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root/evidence.PROTOCOL_SOURCE_PATH
            protocol.parent.mkdir(parents=True)
            protocol.write_bytes((ROOT/evidence.PROTOCOL_SOURCE_PATH).read_bytes())
            record = {"source_sha256_before": {evidence.PROTOCOL_SOURCE_PATH: evidence.PROTOCOL_SOURCE_SHA256}}
            observed = {}
            self.assertEqual(evidence.check_sources(record, root, observed), 1)
            self.assertIn(evidence.PROTOCOL_SOURCE_PATH, observed)
            protocol.write_bytes(protocol.read_bytes()+b"\n")
            with self.assertRaises(ValueError):
                evidence.check_sources(record, root)
            other = root/"other.json"
            other.write_bytes(b"{}\n")
            with self.assertRaises(ValueError):
                evidence.check_sources({"source_sha256": {"other.json": evidence.digest(other.read_bytes())}}, root)

    def test_ledger_utf8_newlines_exact_and_checkpoint_payload_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b'{"label":"caf\xc3\xa9"}\r\n\n'
            (root/"receipt.json").write_bytes(raw)
            (root/"events.jsonl").write_bytes(chain_bytes())
            (root/"checkpoints").mkdir()
            (root/"checkpoints/private.pt").write_bytes(b"private-weight-bytes")
            result = evidence.collect_ledgers({"test": root})
            receipt = result["files"]["test/receipt.json"]
            self.assertEqual(receipt["text"].encode(), raw)
            self.assertEqual(receipt["sha256"], evidence.digest(raw))
            self.assertEqual(set(result["files"]), {"test/receipt.json", "test/events.jsonl"})
            self.assertFalse(result["checkpoint_payloads_included"])

    def test_killed_multibyte_and_json_tails_are_retained_as_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = chain_bytes()+b'{"label":"caf\xc3'
            (root/"events.jsonl").write_bytes(raw)
            (root/"fit_receipt.json").write_bytes(b'{"optimizer_steps":')
            result = evidence.collect_ledgers({"failed": root})
            record = result["files"]["failed/events.jsonl"]
            self.assertEqual(record["encoding"], "base64")
            self.assertEqual(base64.b64decode(record["raw_base64"], validate=True), raw)
            self.assertTrue(record["event_chain"]["incomplete_tail"])
            self.assertEqual(result["files"]["failed/fit_receipt.json"]["json_parse_status"],
                             "unparseable_bytes_retained")

    def test_symlink_receipts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"real.json").write_text("{}\n")
            (root/"alias.json").symlink_to(root/"real.json")
            with self.assertRaises(ValueError):
                evidence.collect_ledgers({"test": root})

    def test_artifact_bit_corruption_and_decoded_hash_corruption_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b'{"a":1}\n'
            compressed = gzip.compress(raw, mtime=0)
            (root/"data.json.gz").write_bytes(compressed)
            manifest = {"artifacts": {"data.json.gz": evidence._artifact(compressed, compressed=True)}}
            (root/"manifest.json").write_bytes(evidence.json_bytes(manifest))
            evidence.verify_package(root)
            manifest["artifacts"]["data.json.gz"]["uncompressed_sha256"] = "0"*64
            (root/"manifest.json").write_bytes(evidence.json_bytes(manifest))
            with self.assertRaises(ValueError):
                evidence.verify_package(root)
            (root/"data.json.gz").write_bytes(compressed[:-1]+bytes([compressed[-1]^1]))
            with self.assertRaises((ValueError, gzip.BadGzipFile, EOFError)):
                evidence.verify_package(root)


@unittest.skipUnless((DATA/"manifest.json").is_file(), "Published chapter31 assets not generated yet")
class PublishedComputeEvidenceTests(unittest.TestCase):
    def load(self, name):
        return evidence.strict_json(gzip.decompress((DATA/(name+"_full.json.gz")).read_bytes()))

    def test_all_published_artifacts_and_recorded_sources_are_bound(self):
        manifest = evidence.verify_package(DATA)
        self.assertEqual(manifest["registration_commit"], evidence.REGISTRATION_COMMIT)
        self.assertEqual(manifest["registration_sha256"], evidence.REGISTRATION_SHA256)
        self.assertEqual(evidence.digest((DATA/"registration.json").read_bytes()), evidence.REGISTRATION_SHA256)
        self.assertEqual(evidence.digest((DATA/"cpu_conv_parity.json").read_bytes()), evidence.PARITY_SHA256)
        for version in manifest["recorded_source_versions"]:
            with self.subTest(source=version["source_path"], version=version["sha256"]):
                self.assertEqual(evidence.digest((ROOT/version["verification_path"]).read_bytes()),
                                 version["sha256"])
        self.assertIn("scripts/e2_cpu_conv.py", manifest["recorded_source_sha256"])
        if manifest["source_amendment_commit"] is not None:
            self.assertEqual(manifest["source_amendment_commit"], evidence.AMENDMENT_COMMIT)
            kernel_raw = gzip.decompress((DATA/"kernels_full.json.gz").read_bytes())
            thread_raw = gzip.decompress((DATA/"threads_full.json.gz").read_bytes())
            evidence.validate_amendment((DATA/"source_amendment.json").read_bytes(),
                                        (DATA/"thread_summary_repair.json").read_bytes(),
                                        (DATA/"scaling_launch_receipt.json").read_bytes(),
                                        amendment_commit=evidence.AMENDMENT_COMMIT,
                                        kernel_sha=evidence.digest(kernel_raw),
                                        thread_sha=evidence.digest(thread_raw))

    def test_registered_counts_failure_costs_and_final_gate_remain_explicit(self):
        reg = evidence.strict_json((DATA/"registration.json").read_bytes())
        for phase in ("kernels", "threads", "scaling"):
            result = evidence.validate_phase(self.load(phase), phase, reg,
                                             evidence.REGISTRATION_SHA256, evidence.REGISTRATION_COMMIT)
            self.assertEqual(result["primary"]["attempts"], {"kernels": 8, "threads": 12, "scaling": 86}[phase])
        evidence.validate_variance(self.load("variance"), evidence.REGISTRATION_SHA256,
                                   evidence.REGISTRATION_COMMIT)
        summary = evidence.strict_json((DATA/"summary.json").read_bytes())
        self.assertIsNone(summary["final_R"])
        self.assertIsNone(summary["achieved_final_power"])
        self.assertFalse(summary["full_HPO_or_NAS_executed"])
        self.assertFalse(summary["final_2025_2026_values_read"])
        self.assertFalse(summary["weights_published"])

    def test_exported_journals_exact_bytes_and_chains_survive_download(self):
        ledgers = evidence.strict_json(gzip.decompress((DATA/"ledgers.json.gz").read_bytes()))
        chains = 0
        for name, record in ledgers["files"].items():
            raw = (record["text"].encode("utf-8") if record["encoding"] == "utf-8"
                   else base64.b64decode(record["raw_base64"], validate=True))
            with self.subTest(ledger=name):
                self.assertEqual(evidence.digest(raw), record["sha256"])
                self.assertEqual(len(raw), record["bytes"])
                self.assertNotIn("checkpoints", Path(name).parts)
                if "event_chain" in record:
                    self.assertEqual(evidence.verify_chain(raw), record["event_chain"])
                    chains += 1
        self.assertEqual(chains, ledgers["event_chains_verified"])
        # A worker killed before its journal opens may legitimately have no
        # event file; its parent attempt and unknown cost remain in summaries.
        for name in ("kernels/events.jsonl", "threads/events.jsonl", "scaling/events.jsonl",
                     "variance/pilot_journal/events.jsonl"):
            self.assertIn(name, ledgers["files"])
            self.assertIn("event_chain", ledgers["files"][name])
        self.assertGreaterEqual(chains, 4)


if __name__ == "__main__":
    unittest.main()
