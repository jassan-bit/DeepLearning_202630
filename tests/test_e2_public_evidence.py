"""Audit scientific-only public evidence without exporting private receipts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import export_e2_public_evidence as public

WORK = ROOT/"work"
PUBLIC = ROOT/"book_e2/_static/compute_public_20261109"
INPUTS = {
    "kernels": WORK/"kernels_published/summary.json",
    "threads": WORK/"threads_published/summary.json",
    "scaling": WORK/"scaling_published/summary.json",
    "variance": WORK/"variance_published/summary.json",
    "calibration": WORK/"calibration_final/report.json",
    "budget": WORK/"cpu_budget/scenarios_k40_final/report.json",
    "registration": ROOT/"registration.json",
}
FILES = {"profiles.json", "variance.json", "calibration.json", "budget.json",
         "registration_scientific.json", "summary.json", "manifest.json"}
FORBIDDEN_KEYS = {"__private_extra__", "__unlisted_number__", "command", "commands", "environment", "env", "hostname",
                  "stdout", "stderr", "error_message", "worker_result_file", "supervisor_result_file",
                  "worker_profile", "durable_events", "events", "source_sha256_before",
                  "source_sha256_after", "data_root", "source_roots", "raw_reports"}


def load_json(path):
    def reject(value):
        raise ValueError("Nonfinite public JSON: "+value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def identity(row):
    return row.get("job", row)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


class PublicStringAuditTests(unittest.TestCase):
    def test_recursive_audit_rejects_private_paths_and_credential_markers(self):
        strings = ["/workspace/private/result.json", "/home/user/secrets", "/tmp/token.txt",
                   "sediment://file_private", "ghp_"+"a"*36, "github_pat_"+"a"*36,
                   "Authorization: Bearer synthetic-token"]
        for text in strings:
            with self.subTest(text=text), self.assertRaises(ValueError):
                public.audit_public({"nested": [1, {"value": text}]})


class ScientificPublicExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [str(path) for path in INPUTS.values() if not path.is_file()]
        if missing:
            raise unittest.SkipTest("Cloud-only export inputs are absent; published-asset checks remain independent")
        cls.inputs = {key: load_json(path) for key, path in INPUTS.items()}

    def test_whitelist_excludes_extra_fields_and_nested_private_metadata(self):
        marker = "PRIVATE_SENTINEL_4b75"
        values = deepcopy(self.inputs)
        for key, value in values.items():
            if key == "registration":
                continue
            value["__private_extra__"] = {"command": marker+" /workspace/private", "env": marker,
                                          "hostname": marker+".internal.example", "stdout": marker}
            for node in list(walk(value)):
                if isinstance(node, dict) and node is not value and "__private_extra__" not in node:
                    node["__private_extra__"] = marker+" /home/private"
                    node["__unlisted_number__"] = 987654321
        # Existing private input keys must be excluded as carefully as unknown
        # keys: adding a whitelist only at the top level is insufficient.
        for phase in ("kernels", "threads", "scaling"):
            for row in values[phase]["results"]+values[phase].get("micro_fallback_results", []):
                row["stdout"] = marker
                row["stderr"] = marker
                row["job"]["command"] = marker
                if isinstance(row.get("worker_profile"), dict):
                    row["worker_profile"]["error_message"] = marker+" /tmp/private"
        for model in values["variance"]["models"].values():
            if not isinstance(model, dict):
                continue
            for phase_name in ("selection", "refit"):
                phase = model[phase_name]
                for row in phase:
                    if isinstance(row, dict):
                        row["worker_result_file"] = marker+" /workspace/private"
                        row["supervisor_result_file"] = marker+" /workspace/private"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for key, value in values.items():
                if key == "registration":
                    paths[key] = INPUTS[key]
                    continue
                path = root/(key+".json")
                write_json(path, value)
                paths[key] = path
            output = root/"public"
            public.export_public(**paths, output=output)
            self.assertEqual({path.name for path in output.iterdir()}, FILES)
            for path in output.iterdir():
                raw = path.read_text(encoding="utf-8")
                self.assertNotIn(marker, raw)
                obj = load_json(path)
                public.audit_public(obj)
                for node in walk(obj):
                    if isinstance(node, dict):
                        self.assertFalse(set(node)&FORBIDDEN_KEYS, (path.name, set(node)&FORBIDDEN_KEYS))

    def test_private_content_in_an_allowed_scientific_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kernels = deepcopy(self.inputs["kernels"])
            kernels["results"][0]["job"]["model"] = "/workspace/private/model"
            path = root/"kernels.json"
            write_json(path, kernels)
            paths = dict(INPUTS)
            paths["kernels"] = path
            with self.assertRaises(ValueError):
                public.export_public(**paths, output=root/"public")

    def test_export_preserves_every_attempt_and_cost_without_raw_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"public"
            public.export_public(**INPUTS, output=output)
            profiles = load_json(output/"profiles.json")
            self.assertEqual(len(profiles["primary"]), 106)
            self.assertEqual(len(profiles["micro"]), 1)
            original = {row["job"]["job_id"]: row
                        for phase in ("kernels", "threads", "scaling")
                        for row in self.inputs[phase]["results"]}
            exported = {identity(row)["job_id"]: row for row in profiles["primary"]}
            self.assertEqual(set(original), set(exported))
            for job_id, row in original.items():
                with self.subTest(job=job_id):
                    self.assertEqual(exported[job_id]["optimizer_steps"], row["optimizer_steps"])
                    self.assertEqual(exported[job_id]["status"], row["status"])
            failed = [row for row in profiles["primary"] if row["status"] != "success"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(identity(failed[0])["job_id"], "scaling-014")
            self.assertEqual(failed[0]["optimizer_steps"], 38)
            variance = load_json(output/"variance.json")
            self.assertEqual(len(variance["attempts"]), 30)
            self.assertEqual(sum(row["optimizer_steps"] for row in variance["attempts"]), 51474)
            self.assertTrue(all(row["status"] == "completed" for row in variance["attempts"]))
            scenarios = load_json(output/"budget.json")["scenarios"]
            self.assertEqual(len(scenarios), 16)
            self.assertTrue(all(row["core_conditional_compute_hours_excluding_disk"] is None
                                for row in scenarios))


@unittest.skipUnless((PUBLIC/"manifest.json").is_file(), "Public chapter31 assets not generated yet")
class PublishedScientificAssetsTests(unittest.TestCase):
    def test_only_audited_utf8_json_and_exact_manifest_digests_are_published(self):
        self.assertEqual({path.name for path in PUBLIC.iterdir()}, FILES)
        manifest = load_json(PUBLIC/"manifest.json")
        self.assertEqual(set(manifest["artifacts"]), FILES-{"manifest.json"})
        for name, expected in manifest["artifacts"].items():
            raw = (PUBLIC/name).read_bytes()
            self.assertEqual(len(raw), expected["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), expected["sha256"])
        for name in FILES:
            value = load_json(PUBLIC/name)
            public.audit_public(value)
            for node in walk(value):
                if isinstance(node, dict):
                    self.assertFalse(set(node)&FORBIDDEN_KEYS)

    def test_all_attempts_failed_steps_and_separate_microprofile_are_retained(self):
        profiles = load_json(PUBLIC/"profiles.json")
        self.assertEqual(len(profiles["primary"]), 106)
        self.assertEqual(len(profiles["micro"]), 1)
        jobs = [identity(row)["job_id"] for row in profiles["primary"]]
        self.assertEqual(len(set(jobs)), 106)
        failed = [row for row in profiles["primary"] if row["status"] != "success"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["optimizer_steps"], 38)
        self.assertEqual(identity(failed[0])["job_id"], "scaling-014")
        self.assertEqual(identity(profiles["micro"][0])["parent_job_id"], "scaling-014")

    def test_variance_predictions_budget_and_final_gate_preserve_scientific_limits(self):
        variance = load_json(PUBLIC/"variance.json")
        self.assertEqual(len(variance["attempts"]), 30)
        self.assertEqual(sum(row["optimizer_steps"] for row in variance["attempts"]), 51474)
        identities = {(row["model"], row["seed"], row["phase"]) for row in variance["attempts"]}
        self.assertEqual(len(identities), 30)
        evaluations = [node for node in walk(variance)
                       if isinstance(node, dict) and isinstance(node.get("predictions"), list)
                       and isinstance(node.get("dates"), list)]
        self.assertEqual(len(evaluations), 15)
        for result in evaluations:
            self.assertEqual(len(result["predictions"]), 53)
            self.assertEqual(len(result["dates"]), 53)
            self.assertTrue(result["dates"][0].startswith("2024-11-08"))
            self.assertTrue(result["dates"][-1].startswith("2024-12-30"))
        budget = load_json(PUBLIC/"budget.json")
        self.assertEqual(len(budget["scenarios"]), 16)
        self.assertTrue(all(row["core_conditional_compute_hours_excluding_disk"] is None
                            for row in budget["scenarios"]))
        summary = load_json(PUBLIC/"summary.json")
        self.assertIsNone(summary["final_R"])
        calibration = load_json(PUBLIC/"calibration.json")
        cells = [cell for group in calibration["groups"].values()
                 for pair in group["pairs"] for cell in pair["cells"]]
        self.assertEqual(len(cells), 16)
        self.assertTrue(all(cell["null_validated"] for cell in cells))
        self.assertFalse(any(cell["supports_conditional_target"] for cell in cells))
        self.assertTrue(all(group["conditionally_supported_R"] is None
                            for group in calibration["groups"].values()))
        for node in walk(calibration):
            if isinstance(node, dict):
                if "final_R" in node:
                    self.assertIsNone(node["final_R"])
                if "achieved_final_power" in node:
                    self.assertIsNone(node["achieved_final_power"])


if __name__ == "__main__":
    unittest.main()
