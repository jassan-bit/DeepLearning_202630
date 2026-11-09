"""Arithmetic and censoring checks; no profiling, model allocation or training."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_cpu_budget as cost


def rate():
    return {"profile_train_n": 64, "profile_validation_n": 83,
            "profile_batch_size": 32, "profile_epochs": 2,
            "optimizer_steps": 4, "update_loop_seconds": 8,
            "train_evaluation_seconds": 4, "validation_seconds_per_pass": 3,
            "setup_seconds": 5, "finalization_seconds": 7,
            "checkpoint_clone_seconds": 2}


def toy_frame():
    return [{"pattern_id": "a"+str(i), "role": "anchor", "stratum_index": None,
             "stratum_size": 1} for i in range(3)] + [
        {"pattern_id": "s0", "role": "sample", "stratum_index": 0, "stratum_size": 2},
        {"pattern_id": "s1", "role": "sample", "stratum_index": 1, "stratum_size": 3}]


def complete_fake_profiles(p):
    frame = cost.profile_sampling_frame(p)
    profiles = []
    for spec in p["models"]+p["neural_reference_profiles"]:
        phases = ["ae1", "ae2", "supervised"] if spec["auxiliary_phases"] else ["supervised"]
        for pattern in frame if spec["nas"] else [None]:
            profiles.append({"job": {"job_id": "scaling-"+str(len(profiles)), "phase": "scaling",
                                      "model": spec["model"], "cohort": "large", "pattern": pattern,
                                      "threads": 1, "backend": "cropped"},
                             "status": "success", "worker_profile": {
                                 "train_n": 64, "validation_n": 83, "fixed_epoch_count": 2,
                                 "build_seconds": 0, "between_stage_setup_seconds": 0,
                                 "budget_phase_rates": {phase: rate() for phase in phases}}})
    return profiles


class WeightingTests(unittest.TestCase):
    def test_anchor_and_stratum_expansion_not_biased_mean_of_samples(self):
        observations = {name: {"status": "success", "value": value}
                        for name, value in zip(["a0", "a1", "a2", "s0", "s1"], [1, 2, 3, 10, 20])}
        result = cost.ht_total(toy_frame(), observations, population_size=8)
        self.assertEqual(result["estimated_population_total"], 1+2+3+2*10+3*20)
        self.assertNotEqual(result["estimated_population_total"], sum([1, 2, 3, 10, 20])/5*8)
        self.assertIsNone(result["confidence_interval"])
        self.assertIsNone(result["within_stratum_variance_estimate"])

    def test_censored_or_missing_sample_does_not_create_survivor_estimate(self):
        obs = {r["pattern_id"]: {"status": "success", "value": 1} for r in toy_frame()}
        obs["s1"] = {"status": "timeout", "elapsed_seconds": 30}
        result = cost.ht_total(toy_frame(), obs, population_size=8)
        self.assertIsNone(result["estimated_population_total"])
        self.assertEqual(result["observed_weighted_subtotal_not_population_bound"], 5)
        self.assertEqual(result["incomplete_samples"][0]["status"], "timeout")
        del obs["s1"]
        self.assertEqual(cost.ht_total(toy_frame(), obs, population_size=8)["incomplete_samples"][0]["status"], "missing")

    def test_invalid_frame_or_success_value_is_rejected(self):
        frame = toy_frame()
        frame[-1]["stratum_index"] = 0
        with self.assertRaises(ValueError):
            cost.ht_total(frame, {}, population_size=8)
        with self.assertRaises(ValueError):
            cost.ht_total(toy_frame(), {"a0": {"status": "success", "value": float("nan")}}, population_size=8)

    def test_registered_frame_covers_1000_with_three_certainty_anchors(self):
        p = json.loads((ROOT/"book_e2/_static/search_protocol_20261109/protocol.json").read_text())
        frame = cost.profile_sampling_frame(p)
        self.assertEqual(len(frame), 12)
        self.assertEqual(sum(r["expansion_weight"] for r in frame), 1000)
        self.assertEqual([r["expansion_weight"] for r in frame[:3]], [1, 1, 1])
        self.assertEqual(sum(r["expansion_weight"] for r in frame[3:]), 997)
        self.assertEqual(len({r["pattern_id"] for r in frame}), 12)


class ProjectionTests(unittest.TestCase):
    def test_setup_once_and_selection_validation_each_epoch(self):
        result = cost.project_phase(rate(), train_n=96, validation_n=83, epochs=10,
                                    batch_size=32, selection=True)
        self.assertEqual(result["target_optimizer_steps"], 30)
        self.assertEqual(result["conditional_compute_seconds_excluding_disk"], 150)
        self.assertEqual(result["components"]["setup_seconds_once"], 5)
        self.assertEqual(result["components"]["finalization_seconds_once"], 7)
        self.assertEqual(result["components"]["additional_in_memory_checkpoint_seconds"], 18)
        self.assertIsNone(result["disk_checkpoint_seconds"])

    def test_refit_has_no_outer_validation_or_extra_best_checkpoint_clones(self):
        result = cost.project_phase(rate(), train_n=96, validation_n=0, epochs=10,
                                    batch_size=32, selection=False)
        self.assertEqual(result["conditional_compute_seconds_excluding_disk"], 102)
        self.assertEqual(result["components"]["validation_seconds"], 0)
        self.assertEqual(result["components"]["additional_in_memory_checkpoint_seconds"], 0)

    def test_unmeasured_batch_transfer_is_labeled_as_assumption(self):
        results = [cost.project_phase(rate(), train_n=96, validation_n=83, epochs=10,
                                     batch_size=16, selection=True, batch_assumption=assumption)
                   for assumption in ("constant_step", "linear_batch")]
        self.assertEqual([r["conditional_compute_seconds_excluding_disk"] for r in results], [270, 150])
        self.assertTrue(all(r["batch_cost_transfer_unvalidated"] for r in results))
        self.assertEqual(cost.batch_count(65, 32), 3)
        self.assertEqual(cost.batch_count(33, 32), 2)
        with self.assertRaises(ValueError):
            cost.batch_count(1, 32)

    def test_inconsistent_profile_steps_or_unknown_timing_is_not_invented(self):
        bad = rate(); bad["optimizer_steps"] = 3
        with self.assertRaises(ValueError):
            cost.project_phase(bad, train_n=96, validation_n=83, epochs=10, batch_size=32, selection=True)
        result = cost.project_trajectory({"status": "success", "worker_profile": {"elapsed_seconds": 9}},
                                         train_n=96, validation_n=83, epochs=10, batch_size=32, selection=True)
        self.assertIsNone(result["value"])
        self.assertEqual(result["status"], "timing_decomposition_missing")
        failed = cost.project_trajectory({"status": "timeout"}, train_n=96, validation_n=83,
                                         epochs=10, batch_size=32, selection=True)
        self.assertIsNone(failed["value"])

    def test_sae_phases_and_shared_build_overhead_are_counted_separately(self):
        profile = {"status": "success", "worker_profile": {"build_seconds": 11,
                   "between_stage_setup_seconds": 13,
                   "budget_phase_rates": {name: rate() for name in ("ae1", "ae2", "supervised")}}}
        result = cost.project_trajectory(profile, train_n=96, validation_n=83,
                                         epochs=10, batch_size=32, selection=True)
        self.assertEqual(result["value"], 11+13+3*150)
        self.assertEqual(set(result["phases"]), {"ae1", "ae2", "supervised"})

    def test_duplicate_worker_results_do_not_select_fastest_result(self):
        row = {"job": {"model": "X", "cohort": "large", "pattern": None}, "status": "success"}
        with self.assertRaises(ValueError):
            cost.index_profiles([row, deepcopy(row)])

    def test_worker_timer_adapter_keeps_auxiliary_proxy_explicit(self):
        timing = {"setup_seconds": 5, "loop_seconds": 12, "finalization_seconds": 7,
                  "update_loop_seconds": 8, "train_evaluation_seconds": 4,
                  "checkpoint_clone_seconds": 2}
        worker = {"train_n": 64, "validation_n": 83, "batch_size": 32,
                  "build_seconds": 11, "fit_management_seconds": 13,
                  "stage_timing": {name: timing for name in ("ae1", "supervised")},
                  "phase_receipts": {name: {"epochs_completed": 2, "optimizer_steps": 4}
                                     for name in ("ae1", "supervised")},
                  "validation_by_stage_seconds": {"ae1": None, "supervised": 3}}
        normalized = cost.normalize_profile_timing({"status": "success", "worker_profile": worker})
        rates = normalized["worker_profile"]["budget_phase_rates"]
        self.assertEqual(rates["ae1"]["validation_seconds_per_pass"], 3)
        self.assertIn("proxy", rates["ae1"]["validation_cost_basis"])
        self.assertIn("measured", rates["supervised"]["validation_cost_basis"])
        self.assertNotIn("budget_phase_rates", worker)  # Caller-owned evidence stays untouched.

    def test_complete_search_formula_and_censoring_one_required_architecture(self):
        p = json.loads((ROOT/"book_e2/_static/search_protocol_20261109/protocol.json").read_text())
        frame = cost.profile_sampling_frame(p)
        origins = [{"origin": str(i), "inner_train_n": 64, "refit_train_n": 64,
                    "validation_n": 83} for i in range(11)]
        profiles = []
        for spec in p["models"]+p["neural_reference_profiles"]:
            patterns = frame if spec["nas"] else [None]
            phases = ["ae1", "ae2", "supervised"] if spec["auxiliary_phases"] else ["supervised"]
            for pattern in patterns:
                profiles.append({"job": {"model": spec["model"], "cohort": "large", "pattern": pattern},
                                 "status": "success", "worker_profile": {
                                     "build_seconds": 0, "between_stage_setup_seconds": 0,
                                     "budget_phase_rates": {phase: rate() for phase in phases}}})
        result = cost.project_search(p, profiles, origins, k=20, R=30)
        totals = {component: sum(m["components_seconds"][component] for m in result["models"].values())
                  for component in ("hpo", "nas_phase1", "nas_phase2_conditional_finalists",
                                    "selected_refits_conditional_architecture")}
        self.assertEqual(totals, {"hpo": 16114560, "nas_phase1": 49280000,
                                  "nas_phase2_conditional_finalists": 20420400,
                                  "selected_refits_conditional_architecture": 6399360})
        self.assertEqual(result["core_conditional_compute_seconds_excluding_disk"], 92214320)
        self.assertIsNone(result["full_campaign_wall_hours"])
        target = next(row for row in profiles if row["job"]["model"] == "TimesNet"
                      and row["job"]["pattern"]["pattern_id"] == frame[-1]["pattern_id"])
        target["status"] = "timeout"
        failed = cost.project_search(p, profiles, origins, k=20, R=30)
        self.assertIsNone(failed["core_conditional_compute_hours_excluding_disk"])
        self.assertIsNone(failed["models"]["TimesNet"]["conditional_compute_seconds_excluding_disk"])
        self.assertGreater(failed["known_family_subtotal_seconds_not_total"], 0)


class StorageTests(unittest.TestCase):
    def test_exact_phase1_storage_and_finalist_ranges_use_all_architectures(self):
        ref = "w64-64-64:b010"
        rows = [{"pattern_id": ref if i == 17 else "p"+str(i), "state_bytes": i+1,
                 "parameter_count": 1, "auxiliary_state_bytes": []} for i in range(1000)]
        result = cost.checkpoint_storage_inventory(
            [{"model": "X", "nas": True, "reference_state_bytes": 18, "architectures": rows}],
            selection_origins=1, trials=1, hpo_seeds=1, finalists=2, phase2_seeds=1, final_seeds=1)
        parts = result["components_payload_bytes"]
        self.assertEqual(parts["nas_phase1_model_states"], sum(range(1, 1001)))
        self.assertEqual(parts["nas_phase2_min"], 1+2+18)
        self.assertEqual(parts["nas_phase2_max"], 999+1000+18)
        self.assertEqual(result["retain_only_selected_refits_payload_bytes_range"], [1, 1000])
        self.assertFalse(result["optimizer_states_included"])
        self.assertFalse(result["deletion_executed_or_authorized"])

    def test_auxiliary_checkpoint_payloads_are_not_free(self):
        result = cost.checkpoint_storage_inventory(
            [{"model": "AE", "nas": False, "reference_state_bytes": 30,
              "architectures": [{"pattern_id": "fixed", "state_bytes": 10,
                                  "parameter_count": 1, "auxiliary_state_bytes": [8, 12]}]}],
            selection_origins=1, trials=2, hpo_seeds=2, finalists=1, phase2_seeds=1, final_seeds=3)
        self.assertEqual(result["components_payload_bytes"]["hpo_model_states"], 120)
        self.assertEqual(result["retain_only_selected_refits_payload_bytes_range"], [90, 90])


class MicroSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((ROOT/"book_e2/_static/search_protocol_20261109/protocol.json").read_text())
        self.profiles = complete_fake_profiles(self.p)
        self.origins = [{"origin": str(i), "inner_train_n": 64, "refit_train_n": 64,
                         "validation_n": 83} for i in range(11)]
        self.primary = next(row for row in self.profiles if row["job"]["model"] == "TimesNet"
                            and row["job"]["pattern"]["widths"] == [256, 256, 256])
        self.primary.update(status="timeout", elapsed_wall_seconds=360)
        self.micro = deepcopy(self.primary)
        self.micro["status"] = "success"
        self.micro["job"].update(phase="micro_fallback", epochs=1, batch_windows=32,
                                  parent_job_id=self.primary["job"]["job_id"], job_id="micro-1")
        self.micro["worker_profile"].update(train_n=32, fixed_epoch_count=1)
        for phase_rate in self.micro["worker_profile"]["budget_phase_rates"].values():
            phase_rate.update(profile_train_n=32, profile_epochs=1, optimizer_steps=1,
                               update_loop_seconds=2, train_evaluation_seconds=1)

    def test_micro_is_separate_conditional_proxy_and_never_erases_censoring(self):
        summary = {"results": self.profiles, "micro_fallback_results": [self.micro]}
        before = deepcopy(summary)
        observed = cost.project_search(self.p, summary, self.origins, k=20, R=30)
        proxy = cost.micro_fallback_sensitivity(self.p, summary, self.origins, k=20, R=30)
        self.assertIsNone(observed["core_conditional_compute_hours_excluding_disk"])
        self.assertGreater(proxy["core_conditional_compute_hours_excluding_disk"], 0)
        self.assertFalse(proxy["full_cohort_fit_completion_established"])
        self.assertTrue(proxy["weighted_surrogate_costs_not_unbiased_full_size_HT_estimate"])
        self.assertTrue(proxy["micro_linkage"][0]["proxy_used"])
        self.assertIsNone(proxy["full_campaign_wall_hours"])
        self.assertEqual(summary, before)
        self.assertEqual(self.primary["status"], "timeout")

    def test_bad_parent_or_duplicate_micro_cannot_be_used_as_a_fast_survivor(self):
        self.micro["job"]["parent_job_id"] = "unknown"
        with self.assertRaises(ValueError):
            cost.micro_fallback_sensitivity(self.p, {"results": self.profiles, "micro_fallback_results": [self.micro]},
                                            self.origins, k=20, R=30)
        self.micro["job"]["parent_job_id"] = self.primary["job"]["job_id"]
        failed = deepcopy(self.micro); failed["status"] = "timeout"
        with self.assertRaises(ValueError):
            cost.micro_fallback_sensitivity(self.p, {"results": self.profiles, "micro_fallback_results": [failed, self.micro]},
                                            self.origins, k=20, R=30)


if __name__ == "__main__":
    unittest.main()
