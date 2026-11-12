"""Scientific domain, chronology and admission tests; no market fitting."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_e2_amended_pilots as runner
from e2_amended_pilot_budget import VERSION as BUDGET_VERSION, digest, BudgetPause


def reconciliation():
    row = {"version": BUDGET_VERSION, "status": "reconciled_before_new_pilot_reservations",
        "authorized_combined_hours": 64, "authorized_combined_seconds": 230400,
        "historical_charged_seconds": 222927.98951942776,
        "remaining_seconds_before_pilots": 7472.010480572237,
        "budget_reset": False, "historical_heads": {}, "snapshot_evidence_sha256": {}}
    return row | {"sha256": digest(row)}


class PilotRunnerTests(unittest.TestCase):
    def test_full_domains_without_optional_systems_or_extra_seeds(self):
        jobs = runner.job_plan()
        self.assertEqual(len(jobs), 183)
        self.assertEqual({s: sum(j["stage"] == s for j in jobs) for s in ("noise", "hpo", "power")},
                         {"noise": 60, "hpo": 108, "power": 15})
        self.assertEqual({j["model"] for j in jobs}, {"TiDE", "TimeMixer", "PatchTST", "GRU32"})
        self.assertEqual(len(runner.job_plan("power")), 123)
        self.assertEqual(len(runner.job_plan("noise")), 60)

    def test_complete_october_domain_before_freeze_and_median_epochs(self):
        hpo = [j for j in runner.job_plan("power") if j["stage"] == "hpo"]
        rows = [{"job": job, "validation_rmse": 1 + int(job["seed"] == 271202),
                 "best_epoch": 3 if job["seed"] == 271201 else 6} for job in hpo]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with self.assertRaises(ValueError):
                runner.freeze_october(rows[:-1], output)
            self.assertFalse((output / "october_winners.json").exists())
            freeze = runner.freeze_october(rows, output)
            self.assertTrue(all(v["fixed_epochs"] == 5 for v in freeze["winners"].values()))
            self.assertFalse(freeze["power_pilot_labels_used"])
            self.assertFalse(freeze["common_HPO_complete"])

    def test_partial_noise_cannot_choose_k(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(runner.close_noise([], Path(folder)))
            self.assertFalse((Path(folder) / "k_decision.json").exists())

    def test_complete_noise_uses_full_five_by_twelve_score_table(self):
        jobs = runner.job_plan("noise")
        patterns = []
        for job in jobs:
            if job["architecture"] not in patterns:
                patterns.append(job["architecture"])
        rows = [{"job": j, "validation_rmse": .01 + patterns.index(j["architecture"]) / 1000} for j in jobs]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                runner.close_noise(rows, Path(folder))
            decision = runner.close_noise(rows, Path(folder), scope="synthetic_fixture")
            self.assertIsNone(decision["k"])
            self.assertEqual(decision["fixture_k_candidate"], 20)
            self.assertEqual(decision["actual_completed_fits"], 0)
            self.assertEqual(len(decision["scores"]["TiDE"]), 5)

    def test_actual_remaining_allowance_blocks_before_any_market_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(output=Path(folder), source_roots=Path(folder) / "roots.json",
                data_root=Path(folder) / "no_market_here", stage="all")
            readiness = {"budget": reconciliation(), "device": "cuda:0", "device_name": "Tesla T4",
                         "total_memory_bytes": 15637086208}
            with patch.object(runner.subprocess, "Popen") as launch:
                with self.assertRaises(BudgetPause):
                    runner.execute(args, readiness)
            launch.assert_not_called()
            self.assertFalse((args.output / "attempts").exists())

    def test_saved_result_rejects_missing_tampered_or_fixture_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            artifact = directory / "selected.pt"
            artifact.write_bytes(b"synthetic test fixture, not real weights")
            spec = {"unit": "fixture"}
            result = {"status": "completed", "worker_spec_sha256": runner.sha(spec),
                "synthetic": False, "test_only_cpu_execution": False,
                "artifacts_sha256": {"selected.pt": runner.file_sha(artifact)}}
            (directory / "result.json").write_text(json.dumps(result))
            self.assertEqual(runner.verified_result(directory, spec), result)
            artifact.write_bytes(b"altered")
            with self.assertRaises(ValueError):
                runner.verified_result(directory, spec)
            result["synthetic"] = True
            (directory / "result.json").write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                runner.verified_result(directory, spec)

    def test_power_cannot_accept_unfrozen_recipes_or_wrong_epoch_count(self):
        with tempfile.TemporaryDirectory() as folder:
            job = runner.job_plan("power")[-1]
            spec = {"version": runner.VERSION, "spec_sha256": runner.sha(runner.execution_spec()),
                "source_sha256": {}, "job": job, "output": folder}
            with patch.object(runner, "source_identity", return_value={}):
                with self.assertRaises(ValueError):
                    runner._validate_worker_spec(spec)


if __name__ == "__main__":
    unittest.main()
