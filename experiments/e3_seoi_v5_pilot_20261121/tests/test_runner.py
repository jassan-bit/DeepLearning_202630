"""Synthetic/software tests only: never train or call optimizer.step()."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seoi_v5 import runner


class SourceAndProtocolTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        (root / "seoi_v5").mkdir()
        spec = {"recipes": list(runner.RECIPES), "seeds": list(runner.SEEDS),
                "training": runner.TRAINING, "budget": {**runner.BUDGET, "documentation": "cooperative"},
                "continuation": {**runner.CONTINUATION, "meaning": "development only"}}
        for name in ("seoi_v5/runner.py", "seoi_v5/model.py", "seoi_v5/data.py"):
            (root / name).write_text("# synthetic source placeholder\n")
        runner.write_json(root / "protocol.json", spec)
        files = [{"path": n, "bytes": (root/n).stat().st_size, "sha256": runner.sha(root/n)}
                 for n in ("seoi_v5/runner.py", "seoi_v5/model.py", "seoi_v5/data.py", "protocol.json")]
        runner.write_json(root / "SOURCE_MANIFEST.json", {"files": files})
        return root

    def test_manifest_detects_code_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(folder)
            self.assertEqual(len(runner.verify_sources(root)["files"]), 4)
            (root/"seoi_v5/model.py").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "identity differs"):
                runner.verify_sources(root)

    def test_manifest_rejects_traversal_and_self(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(folder)
            for name in ("../outside.py", "SOURCE_MANIFEST.json"):
                runner.write_json(root/"SOURCE_MANIFEST.json", {"files": [{"path": name}]})
                with self.assertRaises(ValueError):
                    runner.verify_sources(root)

    def test_protocol_keeps_fixed_thresholds_and_allows_explanation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.fixture(folder)
            self.assertEqual(runner.protocol(root)["budget"]["active_seconds_per_fit"], 120)
            bad = runner.read_json(root/"protocol.json")
            bad["continuation"]["minimum_relative_mean_RMSE_gain"] = 0
            runner.write_json(root/"protocol.json", bad)
            with self.assertRaisesRegex(ValueError, "fixed configuration"):
                runner.protocol(root)

    def test_immutable_record_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"plan.json"
            runner.write_json(path, {"version": 1}, immutable=True)
            runner.write_json(path, {"version": 1}, immutable=True)
            with self.assertRaisesRegex(ValueError, "Immutable"):
                runner.write_json(path, {"version": 2}, immutable=True)
            self.assertEqual(runner.read_json(path), {"version": 1})

    def test_jobs_are_thirty_repeats_of_six_designs(self):
        jobs = runner.jobs()
        self.assertEqual(len(jobs), 30)
        self.assertEqual(len({j["job_id"] for j in jobs}), 30)
        self.assertEqual({j["recipe"] for j in jobs}, set(runner.RECIPES))
        self.assertEqual({j["seed"] for j in jobs}, set(runner.SEEDS))


class SnapshotTests(unittest.TestCase):
    def test_snapshot_digest_and_rng_restoration_without_optimizer_step(self):
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            runner.seed_all(41)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
            with mock.patch.object(optimizer, "step", side_effect=AssertionError("No optimizer steps allowed")) as step:
                state = {"request_sha256": "bound-request", "current": runner.clone_weights(model),
                         "optimizer": optimizer.state_dict(), "rng": runner.capture_rng(), "epoch": 0}
                pointer = runner.save_snapshot(directory, state)
                expected = (random.random(), np.random.rand(), torch.rand(1))
                loaded, read_pointer = runner.load_snapshot(directory, "bound-request")
                runner.restore_rng(loaded["rng"])
                actual = (random.random(), np.random.rand(), torch.rand(1))
                self.assertEqual(expected[:2], actual[:2]); self.assertTrue(torch.equal(expected[2], actual[2]))
                self.assertEqual(pointer, read_pointer)
                self.assertEqual(loaded["optimizer"], optimizer.state_dict())
                step.assert_not_called()
                with self.assertRaisesRegex(ValueError, "identity/hash"):
                    runner.load_snapshot(directory, "other-request")
                (directory/pointer["file"]).write_bytes(b"altered checkpoint")
                with self.assertRaisesRegex(ValueError, "identity/hash"):
                    runner.load_snapshot(directory, "bound-request")

    def test_interrupted_epoch_debits_reservation_once(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            state = {"request_sha256": "r", "epoch": 3, "active_seconds_charged": 1.25,
                     "interrupted_epochs": 0, "settled_segments": []}
            marker = {"request_sha256": "r", "segment_id": "s", "epoch": 4, "reserved_seconds": 5.0}
            runner.write_json(directory/"epoch_pending.json", marker)
            with mock.patch.object(runner, "save_snapshot") as save:
                updated, recovered = runner.settle_interrupted_epoch(directory, state)
                self.assertTrue(recovered); self.assertEqual(updated["active_seconds_charged"], 6.25)
                self.assertEqual(updated["interrupted_epochs"], 1)
                self.assertIsNone(updated["lost_partial_epoch_optimizer_steps"])
                save.assert_called_once()
                runner.write_json(directory/"epoch_pending.json", marker)
                twice, _ = runner.settle_interrupted_epoch(directory, updated)
                self.assertEqual(twice["active_seconds_charged"], 6.25)
                self.assertEqual(save.call_count, 1)

    def test_known_overrun_is_charged_not_hidden(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            state = {"request_sha256": "r", "epoch": 0, "active_seconds_charged": 0,
                     "interrupted_epochs": 0, "settled_segments": []}
            runner.write_json(directory/"epoch_pending.json", {"request_sha256": "r", "segment_id": "s", "epoch": 1, "reserved_seconds": 5.0})
            runner.write_json(directory/"attempts.json", [{"segment_id": "s", "aborted_active_seconds_observed": 6.25}])
            with mock.patch.object(runner, "save_snapshot"):
                updated, _ = runner.settle_interrupted_epoch(directory, state)
            self.assertEqual(updated["active_seconds_charged"], 6.25)

    def test_completed_checkpoint_without_receipt_is_recovered_without_training(self):
        import types
        import torch
        with tempfile.TemporaryDirectory() as folder:
            case = Path(folder); job = runner.jobs()[0]; plan = {"cohort_binding": {}}
            request_hash = runner.digest(runner.job_request(plan, job))
            directory = case / "jobs" / job["job_id"]; directory.mkdir(parents=True)
            model = torch.nn.Linear(2, 1)
            state = {"request_sha256": request_hash, "completed": True,
                     "current": runner.clone_weights(model), "best": runner.clone_weights(model),
                     "best_score": .1, "best_epoch": 39, "epoch": 40,
                     "steps_committed": 680, "interrupted_epochs": 0, "stale": 0,
                     "active_seconds_charged": .5, "checkpoint_io_seconds": .1,
                     "stop_reason": "max_epochs",
                     "curve": [{"epoch": 40, "selection_mse_scaled": .1}]}
            runner.save_snapshot(directory, state)
            factory = types.SimpleNamespace(build_model=lambda *a, **k: torch.nn.Linear(2, 1))
            data = {"scalers": {job["recipe"]: {"u_scale": 1., "y_scale": 1.}}}
            with mock.patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("No training")) as step:
                receipt = runner.fit_job(case, plan, data, factory, job)
                step.assert_not_called()
            self.assertEqual(receipt["status"], "completed_validated_fit")
            self.assertIsNotNone(runner.complete_job(case, plan, job))
            self.assertEqual(runner.read_json(directory / "curve.json"), state["curve"])


class DecisionTests(unittest.TestCase):
    def fixture(self):
        summary = {r: {"mean_RMSE": 1., "mean_MAE": 1.} for r in runner.RECIPES + runner.ANALYTICAL}
        summary["seoi_mse"] = {"mean_RMSE": .97, "mean_MAE": 1.}
        by_seed = {r: {s: {"RMSE": .97 if r == "seoi_mse" else 1.} for s in runner.SEEDS} for r in runner.RECIPES}
        return summary, by_seed

    def test_fixed_continuation_is_not_R3(self):
        summary, by_seed = self.fixture()
        result = runner.continuation(summary, by_seed, complete_fits=True)
        self.assertEqual(result["decision"], "CONTINUE_DEVELOPMENT_ONLY")
        self.assertEqual(result["best_neural_control_by_mean_RMSE"], "input_only_mse")
        self.assertFalse(result["R3_accredited"])
        self.assertFalse(result["confirmation_allowed"])
        self.assertFalse(result["control_automatically_promoted"])

    def test_each_gate_can_fail_and_no_promotion(self):
        for kind in ("gain", "wins", "mae", "completion", "analytical"):
            summary, by_seed = self.fixture(); complete = True
            if kind == "gain":
                summary["seoi_mse"]["mean_RMSE"] = .99
            elif kind == "wins":
                for s in runner.SEEDS[:2]:
                    by_seed["seoi_mse"][s]["RMSE"] = 1.1
            elif kind == "mae":
                summary["seoi_mse"]["mean_MAE"] = 1.051
            elif kind == "completion":
                complete = False
            else:
                summary["floor_only"]["mean_RMSE"] = .8
            result = runner.continuation(summary, by_seed, complete_fits=complete)
            self.assertEqual(result["decision"], "ABANDON_CURRENT_SEOI_RECIPE", kind)
            self.assertFalse(result["control_automatically_promoted"])

    def test_missing_seed_rejected(self):
        summary, by_seed = self.fixture()
        del by_seed["seoi_mse"][runner.SEEDS[0]]
        with self.assertRaisesRegex(ValueError, "seed"):
            runner.continuation(summary, by_seed, complete_fits=True)


class AssessmentAndExportTests(unittest.TestCase):
    def test_assessment_blocked_before_any_model_load_when_fits_incomplete(self):
        plan = {"jobs": runner.jobs()}
        with mock.patch.object(runner, "load_case", return_value=(plan, None, None)) as load:
            with mock.patch.object(runner, "complete_job", return_value=None):
                with mock.patch.object(runner, "_cached_assessment", side_effect=AssertionError("No inference allowed")) as predict:
                    with self.assertRaisesRegex(ValueError, "All30"):
                        runner.assess("synthetic", "synthetic")
                    predict.assert_not_called()
            self.assertEqual(load.call_count, 1)
            self.assertFalse(load.call_args.kwargs["load_arrays"])

    def test_export_excludes_raw_weights_rng_and_all_nonallowlisted_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); case = root/"private"; case.mkdir(); output = root/"public"
            plan = {"case_id": "fixture", "jobs": runner.jobs()}
            runner.write_json(case/"plan.json", plan)
            runner.write_json(case/"failures.json", [])
            (case/"inputs").mkdir(); (case/"inputs/cohort.npz").write_bytes(b"SECRET RAW FIXTURE")
            job = case/"jobs"/plan["jobs"][0]["job_id"]; job.mkdir(parents=True)
            (job/"snapshot_fixture.pt").write_bytes(b"SECRET WEIGHTS AND RNG FIXTURE")
            runner.write_json(job/"curve.json", [{"epoch": 1, "selection_mse_scaled": .2}])
            runner.write_json(job/"request.json", {"not": "allowlisted"})
            with mock.patch.object(runner, "load_case", return_value=(plan, None, None)):
                receipt = runner.export_case(root, case, output)
            exported = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
            self.assertIn("plan.json", exported)
            self.assertIn("jobs/"+plan["jobs"][0]["job_id"]+"/curve.json", exported)
            self.assertFalse(any(n.endswith((".npz", ".pt")) or "inputs/" in n or n.endswith("request.json") for n in exported))
            self.assertFalse(receipt["weights_optimizer_or_RNG_exported"])
            self.assertFalse(receipt["raw_arrays_exported"])


if __name__ == "__main__":
    unittest.main()
