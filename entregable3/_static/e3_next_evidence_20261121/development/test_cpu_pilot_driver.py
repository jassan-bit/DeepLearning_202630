"""Software guards only: no market input is loaded and no optimizer is run."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cpu_pilot_driver as d


def fixed_plan():
    return {"version": d.VERSION, "case_id": "v4-cpu-dev-fixture",
            "scope": "new_exposed_CPU_development_not_confirmation_or_reproduction",
            "jobs": d.jobs(), "recipes": list(d.RECIPES), "analytical_controls": list(d.ANALYTICAL),
            "analytical_parameters": d.ANALYTICAL_PARAMETERS,
            "seeds": list(d.SEEDS), "config": d.CONFIG, "parameters": d.PARAMETERS,
            "R1_inherited": False, "R3_accredited": False, "HPO_fits": 0,
            "fit_rows": 532, "stop_rows": 54, "purged_rows": 7, "assessment_rows": 55,
            "assessment_predictions_expected": 1210, "neural_assessment_predictions_expected": 1100,
            "analytical_assessment_predictions_expected": 110, "assessment_trajectories_expected": 22,
            "analytical_controls_evaluated_once_not_repeated_per_seed": True,
            "continuation_rule": d.RULE, "capacity_matched": False,
            "complete_family_planning_pilot": False, "statistical_significance_or_power_claimed": False,
            "old_GPU_weights_used_for_fair_new_comparison": False,
            "original_V3_NO_GO_preserved": True,
            "budget": {"phase_seconds_per_fit": 120.0, "process_seconds_per_fit": 150.0,
                       "total_optimizer_nominal_seconds": 2400.0, "end_to_end_max_seconds": 3300.0,
                       "max_epochs_per_fit": 40, "deadline_includes_restart_downtime": True},
            "source_manifest_sha256": "a" * 64}


def anchor(plan):
    return {"public_anchor_commit": "b" * 40, "plan_sha256": d.digest(plan),
            "source_manifest_sha256": plan["source_manifest_sha256"],
            "development_only_not_confirmation": True, "software_QA_passed": True}


class Guards(unittest.TestCase):
    def test_import_is_stdlib_only_and_bare_CLI_does_not_train(self):
        root = str(Path(d.__file__).parent)
        code = "import sys;sys.path.insert(0," + repr(root) + ");import cpu_pilot_driver;print('torch' in sys.modules,'numpy' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0); self.assertEqual(result.stdout.strip(), "False False")
        result = subprocess.run([sys.executable, d.__file__], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2); self.assertIn("required", result.stderr)

    def test_fixed_inventory_is20_fits_not1210_fits_or_seed_designs(self):
        jobs = d.jobs()
        self.assertEqual(len(jobs), 20); self.assertEqual(len({j["job_id"] for j in jobs}), 20)
        self.assertEqual({j["seed"] for j in jobs}, set(range(20261121, 20261126)))
        self.assertEqual([j["name"] for j in jobs[:4]], list(d.RECIPES))
        self.assertEqual(d.PARAMETERS["sigma_crps_v4"], 1262)
        self.assertEqual(d.PARAMETERS["additive_positive_head"], 1210)
        d.check_plan(fixed_plan())

    def test_mutated_rule_budget_geometry_and_scope_are_blocked(self):
        for key, value in (("assessment_predictions_expected", 1100), ("R3_accredited", True),
                           ("HPO_fits", 1), ("original_V3_NO_GO_preserved", False),
                           ("capacity_matched", True), ("analytical_controls", [])):
            plan = fixed_plan(); plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): d.check_plan(plan)
        plan = fixed_plan(); plan["budget"] = dict(plan["budget"], phase_seconds_per_fit=121)
        with self.assertRaises(ValueError): d.check_plan(plan)
        plan = fixed_plan(); plan["continuation_rule"] = dict(d.RULE, minimum_relative_mean_RMSE_gain_against_best_all_five_controls=0)
        with self.assertRaises(ValueError): d.check_plan(plan)

    def test_public_anchor_and_QA_gate_both_hashes_and_development_scope(self):
        plan = fixed_plan(); valid = anchor(plan); d.verify_anchor(plan, valid)
        for key, value in (("public_anchor_commit", "main"), ("plan_sha256", "c" * 64),
                           ("source_manifest_sha256", "c" * 64), ("software_QA_passed", False),
                           ("development_only_not_confirmation", False)):
            wrong = dict(valid); wrong[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): d.verify_anchor(plan, wrong)

    def test_unanchored_worker_fails_before_scientific_import_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp) / "v4-cpu-dev-fixture"; case.mkdir()
            plan = fixed_plan(); d.write_json(case / "cpu_pilot_plan.json", plan)
            receipt = case / "anchor.json"; d.write_json(receipt, dict(anchor(plan), software_QA_passed=False))
            with mock.patch.object(d, "scientific_runtime", side_effect=AssertionError("must not import")) as runtime:
                with self.assertRaises(ValueError): d.worker(case, case / "absent.json", case, receipt, 0)
                runtime.assert_not_called()

    def test_source_hashes_duplicates_traversal_and_symlinks_are_guarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); file = root / "toy.py"; file.write_bytes(b"# fixture\n")
            item = {"path": "toy.py", "bytes": file.stat().st_size, "sha256": d.sha(file)}
            manifest = root / "manifest.json"
            with mock.patch.object(d, "REQUIRED_SOURCE", {"toy.py"}):
                d.write_json(manifest, {"files": [item]}); self.assertEqual(d.verify_sources(root, manifest), d.sha(manifest))
                file.write_bytes(b"# changed\n")
                with self.assertRaises(ValueError): d.verify_sources(root, manifest)
                file.write_bytes(b"# fixture\n")
                d.write_json(manifest, {"files": [item, item]})
                with self.assertRaises(ValueError): d.verify_sources(root, manifest)
                d.write_json(manifest, {"files": [dict(item, path="../toy.py")]})
                with self.assertRaises(ValueError): d.verify_sources(root, manifest)
                file.unlink(); (root / "other").write_bytes(b"# fixture\n"); file.symlink_to(root / "other")
                d.write_json(manifest, {"files": [item]})
                with self.assertRaises(ValueError): d.verify_sources(root, manifest)

    def test_bad_NPZ_transport_rejected_without_opening_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.npz"; path.write_bytes(b"not a market NPZ")
            with self.assertRaises(ValueError): d.validate_npz(path)

    def test_symlink_ancestor_and_immutable_record_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); real = root / "real"; real.mkdir(); (root / "link").symlink_to(real)
            with self.assertRaises(ValueError): d.no_symlink_path(root / "link/new.json")
            record = real / "fixed.json"; d.write_json(record, {"a": 1}, immutable=True)
            with self.assertRaises(ValueError): d.write_json(record, {"a": 2}, immutable=True)
            self.assertEqual(d.read_json(record), {"a": 1})

    def test_resume_log_CSV_and_temporary_aliases_cannot_escape_owned_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); outside = root / "outside"; outside.write_text("unchanged")
            case = root / "v4-cpu-dev-fixture"; case.mkdir()
            (case / "worker.log").symlink_to(outside)
            with self.assertRaises(ValueError): d.run_process(["never"], case / "worker.log", 1)
            with self.assertRaises(ValueError): d.guard_case_tree(case)
            (case / "worker.log").unlink()
            (case / "table.csv.tmp").symlink_to(outside)
            with self.assertRaises(ValueError): d.write_csv(case / "table.csv", [{"a": 1}])
            (case / "table.csv.tmp").unlink(); os.link(outside, case / "worker.log")
            with self.assertRaises(ValueError): d.guard_case_tree(case)
            with self.assertRaises(ValueError): d.run_process(["never"], case / "worker.log", 1)
            self.assertEqual(outside.read_text(), "unchanged")

    def test_global_deadline_is_not_reset_by_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp); plan = fixed_plan()
            with mock.patch.object(d.time, "time", return_value=1000.0):
                first = d.execution_marker(case, plan, create=True)
            with mock.patch.object(d.time, "time", return_value=2000.0):
                second = d.execution_marker(case, plan, create=True)
            self.assertEqual(first, second); self.assertEqual(first["deadline_unix"], 4300.0)
            self.assertEqual(d.remaining_seconds(first, now=2000), 2300)
            with self.assertRaises(ValueError): d.remaining_seconds(first, now=4300)
            tampered = dict(first, deadline_unix=4301); d.write_json(case / "execution_started.json", tampered)
            with self.assertRaises(ValueError): d.execution_marker(case, plan)

    def test_completed_job_requires_hash_identity_and_optimizer_update(self):
        with tempfile.TemporaryDirectory() as temp:
            case = Path(temp); plan = fixed_plan(); job = d.jobs()[0]
            self.assertIsNone(d.completed_job(case, plan, job))
            folder = d.job_directory(case, job); (folder / "supervised").mkdir(parents=True)
            checkpoint = folder / "supervised/checkpoint.pt"; checkpoint.write_bytes(b"synthetic opaque checkpoint fixture; never unpickled")
            record = {"status": "completed_validated_new_fit", "job": job, "plan_sha256": d.digest(plan),
                      "optimizer_steps": 1, "checkpoint_sha256": d.sha(checkpoint)}
            d.write_json(folder / "job.json", record)
            self.assertEqual(d.completed_job(case, plan, job), record)
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError): d.completed_job(case, plan, job)
            record["optimizer_steps"] = 0; record["checkpoint_sha256"] = d.sha(checkpoint)
            d.write_json(folder / "job.json", record)
            with self.assertRaises(ValueError): d.completed_job(case, plan, job)

    def test_cached_worker_does_not_repeat_construction_training_or_assessment(self):
        plan = fixed_plan()
        with mock.patch.object(d, "load_case", return_value=plan), \
             mock.patch.object(d, "execution_marker", return_value={"deadline_unix": 1e30}), \
             mock.patch.object(d, "completed_job", return_value={"completed": True}), \
             mock.patch.object(d, "scientific_runtime", side_effect=AssertionError("not needed")) as runtime:
            out = d.worker(Path("."), Path("manifest"), Path("case"), Path("anchor"), 0)
        self.assertTrue(out["reused_completed_job"]); self.assertFalse(out["new_fit_this_worker"])
        runtime.assert_not_called()

    def test_assessment_gate_requires_all20_jobs_before_any_forward(self):
        with mock.patch.object(d, "load_case", return_value=fixed_plan()), \
             mock.patch.object(d, "completed_job", return_value=None), \
             mock.patch.object(d, "scientific_runtime", side_effect=AssertionError("must not import")) as runtime:
            with self.assertRaises(ValueError): d.assess(Path("."), Path("manifest"), Path("case"), Path("anchor"))
            runtime.assert_not_called()

    def test_candidate_beating_neural_but_losing_floor_must_abandon(self):
        values = {"sigma_crps_v4": .97, "u_crps_control": 1.1, "mse_only_control": 1.,
                  "additive_positive_head": 1.2, "floor_only": .9, "floor_plus_mean_excess": 1.05}
        decision = d.continuation(values)
        self.assertGreater(decision["relative_mean_RMSE_gain_vs_best_neural"], .02)
        self.assertLess(decision["relative_mean_RMSE_gain_vs_best_all"], 0)
        self.assertEqual(decision["decision"], "ABANDON_CURRENT_V4_RECIPE")
        self.assertFalse(decision["control_automatically_promoted"])

    def test_fixed_two_percent_rule_requires_every_control_and_handles_zero(self):
        values = {"sigma_crps_v4": .98, "u_crps_control": 1.1, "mse_only_control": 1.,
                  "additive_positive_head": 1.2, "floor_only": 1.5, "floor_plus_mean_excess": 1.3}
        self.assertEqual(d.continuation(values)["decision"], "CONTINUE_DEVELOPMENT_ONLY")
        values["sigma_crps_v4"] = .99
        self.assertEqual(d.continuation(values)["decision"], "ABANDON_CURRENT_V4_RECIPE")
        values["floor_only"] = 0
        self.assertIsNone(d.continuation(values)["relative_mean_RMSE_gain_vs_best_all"])
        del values["floor_only"]
        with self.assertRaises(ValueError): d.continuation(values)

    def test_U_control_full_loss_gradients_match_original_and_atomic_initialization(self):
        # Synthetic inputs only. This imports the pinned model classes but never
        # loads NPZ data, creates an optimizer or executes a training phase.
        root = Path(d.__file__).parent
        sys.path.insert(0, str(root / "pilot_sources")); sys.path.insert(1, str(root))
        import torch
        from outcome_crps_v4 import OutcomeCRPSDevelopmentModel
        from src.e3_overlap_model_v3 import loss_components
        from src.e3_innovation_model import InnovationModel
        torch.set_num_threads(2)
        models = [OutcomeCRPSDevelopmentModel(name, u_scale=.03, y_scale=.02, seed=20261121)
                  for name in d.RECIPES[:3]]
        for name, value in models[0].state_dict().items():
            for other in models[1:]: self.assertTrue(torch.equal(value, other.state_dict()[name]), name)
        rule = root / "pilot_sources/experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json"
        additive = InnovationModel("additive_positive_head", u_scale=.03, y_scale=.02, seed=20261121, rule_path=rule)
        self.assertEqual(sum(p.numel() for p in additive.parameters()), 1210)
        for name, value in models[0].encoder.state_dict().items():
            self.assertTrue(torch.equal(value, additive.base.encoder.state_dict()[name]), name)
        self.assertTrue(torch.equal(models[0].query_raw, additive.base.mechanism.query_raw))
        self.assertTrue(torch.equal(models[0].temperature_raw, additive.base.mechanism.temperature_raw))
        control = models[1]; x = torch.zeros((3, 7, 5), dtype=torch.float32)
        A = torch.tensor([0, .001, .1], dtype=torch.float64)
        Y = torch.tensor([.02, .05, .4], dtype=torch.float64); U = torch.tensor([.05, .1, .7], dtype=torch.float64)
        out = control.forward_components(x, A)
        new_loss = control.loss_components(out, Y, U)["loss"]
        old_loss = loss_components(out, Y, U, y_scale=control.y_scale,
                                   u_scale=control.u_scale, lambda_crps=.1)["loss"]
        self.assertTrue(torch.equal(new_loss, old_loss))
        parameters = tuple(control.parameters())
        new_grad = torch.autograd.grad(new_loss, parameters, retain_graph=True)
        old_grad = torch.autograd.grad(old_loss, parameters)
        for a, b in zip(new_grad, old_grad): self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
