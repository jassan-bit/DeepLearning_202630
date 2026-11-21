"""Export security fixtures only; no raw market arrays or trained model used."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cpu_pilot_driver as d
import export_cpu_pilot as e
from test_cpu_pilot_driver import fixed_plan


def fixture(root, *, completed_first=False):
    case = root / "v4-cpu-dev-fixture"; case.mkdir()
    plan = fixed_plan(); plan["admission_sha256"] = d.digest({"fixture_only": True})
    d.write_json(case / "cpu_pilot_plan.json", plan)
    d.write_json(case / "admission.json", {"fixture_only": True})
    d.write_json(case / "prepared_receipt.json", {"plan_sha256": d.digest(plan),
                  "plan_file_sha256": d.sha(case / "cpu_pilot_plan.json")})
    private = case / "inputs"; private.mkdir(); (private / "current_cohort_arrays.npz").write_bytes(b"PRIVATE RAW FIXTURE NEVER EXPORT")
    job = plan["jobs"][0]; directory = d.job_directory(case, job); (directory / "supervised").mkdir(parents=True)
    checkpoint = directory / "supervised/checkpoint.pt"; checkpoint.write_bytes(b"PRIVATE WEIGHT FIXTURE NEVER EXPORT")
    d.write_json(directory / "supervised/phase_started.json", {"fixture_started": True})
    (directory / "worker.log").write_text("owned software fixture log\n")
    if completed_first:
        d.write_json(directory / "job.json", {"plan_sha256": d.digest(plan), "job": job,
                     "status": "completed_validated_new_fit", "optimizer_steps": 1,
                     "checkpoint_sha256": d.sha(checkpoint)})
    return case, checkpoint


class Guards(unittest.TestCase):
    def test_only_allowlisted_metadata_is_copied_and_no_binary_is_deserialized(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); case, checkpoint = fixture(root, completed_first=True)
            (case / "unexpected.txt").write_text("not in allowlist")
            with mock.patch.object(d, "verify_sources", return_value="a" * 64):
                result = e.export_case(root, case, root / "public")
            self.assertEqual(result["validated_completed_jobs"], 1)
            names = {row["path"] for row in result["files"]}
            self.assertIn("LOCAL_CHECKPOINT_HASHES_ONLY.json", names)
            self.assertNotIn("unexpected.txt", names)
            self.assertFalse(any(name.endswith((".pt", ".npz", ".npy")) for name in names))
            inventory = json.loads((root / "public/LOCAL_CHECKPOINT_HASHES_ONLY.json").read_text())
            self.assertEqual(inventory["checkpoints"][0]["sha256"], d.sha(checkpoint))
            self.assertFalse(result["model_or_checkpoint_deserialized_by_exporter"])

    def test_partial_case_cannot_be_claimed_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); case, _ = fixture(root)
            with mock.patch.object(d, "verify_sources", return_value="a" * 64):
                with self.assertRaises(ValueError): e.export_case(root, case, root / "public", require_completed=True)
            self.assertFalse((root / "public").exists())

    def test_changed_checkpoint_refuses_completed_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); case, checkpoint = fixture(root, completed_first=True)
            checkpoint.write_bytes(b"tampered fixture")
            with mock.patch.object(d, "verify_sources", return_value="a" * 64):
                with self.assertRaises(ValueError): e.export_case(root, case, root / "public")

    def test_log_alias_and_preexisting_or_private_destination_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); case, _ = fixture(root)
            log = next((case / "jobs").rglob("worker.log")); log.unlink()
            secret = root / "unrelated.txt"; secret.write_text("unrelated fixture")
            log.symlink_to(secret)
            with mock.patch.object(d, "verify_sources", return_value="a" * 64):
                with self.assertRaises(ValueError): e.export_case(root, case, root / "public")
            self.assertEqual(secret.read_text(), "unrelated fixture")
            with self.assertRaises(ValueError): e.export_case(root, case, case / "public")
            (root / "old").mkdir()
            with self.assertRaises(ValueError): e.export_case(root, case, root / "old")


if __name__ == "__main__":
    unittest.main()
