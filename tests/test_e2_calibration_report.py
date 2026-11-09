"""Pure metadata fixtures for the two separate calibration groups; no simulation."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_e2_calibration_report as report


NEW_REGISTRATION_SHA256 = report.COMPUTE_REGISTRATION_SHA256
NEW_REGISTRATION_COMMIT = report.COMPUTE_SOURCE_COMMIT


def evaluation_dates():
    start = datetime(2024, 11, 8, tzinfo=timezone.utc)
    return [(start + timedelta(days=index)).isoformat() for index in range(53)]


def source_hashes(group):
    name = "run_e2_reference_variance.py" if group == "old" else "run_e2_optimized_variance.py"
    return {**report.CORE_SOURCE_SHA256,
            "scripts/" + name: report.RUNNER_SOURCE_SHA256[group]}


def runtime_policy():
    return {"registration_commit": report.COMPUTE_SOURCE_COMMIT,
            "registration_sha256": report.COMPUTE_REGISTRATION_SHA256,
            "threads": 2}


def campaign_fixture(group):
    """Small, self-contained valid receipts; prediction values are invented."""
    models = report.OLD_MODELS if group == "old" else report.NEW_MODELS
    seeds = report.OLD_SEEDS if group == "old" else report.NEW_SEEDS
    version = ("e2-preselection-reference-variance-v1" if group == "old"
               else "e2-optimized-preselection-reference-variance-v1")
    commit = (report.OLD_REGISTRATION_COMMIT if group == "old"
              else NEW_REGISTRATION_COMMIT)
    registration_sha = (report.OLD_REGISTRATION_SHA256 if group == "old"
                        else NEW_REGISTRATION_SHA256)
    truth = [.01 + index / 1_000_000 for index in range(53)]
    sources = source_hashes(group)
    summary = {
        "version": version,
        "status": "incomplete" if group == "old" else "complete",
        "registration_commit": commit,
        "registration_file_sha256": registration_sha,
        "preselection_reference_only": True,
        "HPO_executed": False,
        "NAS_search_executed": False,
        "replacement_attempts": 0,
        "final_2025_2026_access": False,
        "final_R_set": False,
        "final_power_claimed": False,
        "fixed_epoch_counts": {name: 3 for name in models},
        "blocked_refits": {"TimeMixer": "Old selection attempts incomplete"} if group == "old" else {},
        "models": {},
        "source_sha256": deepcopy(sources),
        "source_sha256_before": deepcopy(sources),
        "source_sha256_after": deepcopy(sources),
        "source_hashes_unchanged": True,
        "supersedes_previous_attempts": False,
        "paired_3x5_complete": group == "new",
        "runtime_policy": runtime_policy(),
    }
    for model_index, name in enumerate(models):
        records = {"selection": [], "refit": []}
        for phase, key in [("epoch_selection", "selection"), ("fresh_refit", "refit")]:
            for seed_index, seed in enumerate(seeds):
                selection = phase == "epoch_selection"
                worker = {
                    "version": version, "status": "completed", "model": name,
                    "seed": seed, "phase": phase,
                    "registration_commit": commit,
                    "registration_file_sha256": registration_sha,
                    "fresh_optimizer": True,
                    "hpo_executed": False, "nas_search_executed": False,
                    "selected_hpo_recipe": False, "weights_reused": False,
                    "final_2025_2026_access": False, "replacement_attempt": False,
                    "best_epoch": 3,
                    "optimizer_steps": 9,
                    "training_split": {
                        "dates": ["2023-01-08T00:00:00+00:00",
                                  ("2024-06-25T00:00:00+00:00" if selection
                                   else "2024-10-31T00:00:00+00:00")]},
                    "validation_split": {
                        "dates": ["2024-07-03T00:00:00+00:00",
                                  "2024-09-23T00:00:00+00:00"]} if selection else None,
                    "checkpoint": {"safe_load_weights_only": True,
                                   "optimizer_state_saved": False,
                                   "sha256": "c" * 64, "state_sha256": "d" * 64},
                    "data_receipt": {
                        "admitted_end": "2024-12-30",
                        "source_protocol_sha256": report.HISTORICAL_PROTOCOL_SHA256,
                        "source_sha256": deepcopy(report.DATA_SOURCE_SHA256),
                        "source_values_parsed_only_through_admission": True,
                        "external_2025_2026_files_opened": 0,
                    },
                    "model_metadata": {
                        "model": name, "initialization_seed": seed,
                        "pilot_fixed_not_hpo": True, "weights_reused": False,
                        "input_shape": [7, 5],
                        "pilot_pattern": None if name == "GRU32" else {
                            "widths": [64, 64, 64], "dropout_pattern": [0, 1, 0],
                            "is_reference": True},
                    },
                    "source_sha256": deepcopy(sources),
                    "source_sha256_before": deepcopy(sources),
                    "source_sha256_after": deepcopy(sources),
                    "source_hashes_unchanged": True,
                    "runtime_policy": runtime_policy(),
                    "torch_num_threads": 2,
                    "torch_num_interop_threads": 1,
                }
                if not selection:
                    worker.update(
                        fixed_epoch_count=3,
                        evaluation={"dates": evaluation_dates(), "truths": list(truth),
                                    "predictions": [value + (model_index + 1) / 100_000
                                                    + seed_index / 1_000_000 for value in truth]},
                        frozen_fingerprint_invariant=True,
                        observation_after_durable_prediction=True,
                        epoch_selection_inputs_passed_to_refit=False)
                else:
                    worker["outer_labels_used_for_epoch_selection"] = False
                records[key].append({
                    "model": name, "seed": seed, "phase": phase, "status": "completed",
                    "supervisor_status": "success", "best_epoch": 3,
                    "optimizer_steps": 9, "replacement_attempt": False,
                    "worker_profile": worker,
                })
        summary["models"][name] = records
    if group == "old":
        summary["models"]["TimeMixer"] = {"selection": [], "refit": []}
    return summary


class CalibrationPanelMetadata(unittest.TestCase):
    def extract(self, summary, group):
        return report.extract_panel(
            summary, group=group, registration_sha256=NEW_REGISTRATION_SHA256,
            registration_commit=NEW_REGISTRATION_COMMIT)

    def test_old_and_new_groups_keep_distinct_models_and_seed_pools(self):
        old = self.extract(campaign_fixture("old"), "old")
        new = self.extract(campaign_fixture("new"), "new")
        self.assertEqual(old["status"], "ready")
        self.assertEqual(new["status"], "ready")
        self.assertEqual(old["model_names"], list(report.OLD_MODELS))
        self.assertEqual(new["model_names"], list(report.NEW_MODELS))
        self.assertEqual(old["seed_ids"], list(report.OLD_SEEDS))
        self.assertEqual(new["seed_ids"], list(report.NEW_SEEDS))
        self.assertTrue(set(old["seed_ids"]).isdisjoint(new["seed_ids"]))
        self.assertEqual(old["excluded_models"], ["TimeMixer"])
        self.assertEqual(new["excluded_models"], [])
        self.assertEqual(old["dates"], evaluation_dates())
        self.assertEqual(new["dates"], evaluation_dates())
        self.assertEqual(len(old["predictions"]), 2)
        self.assertEqual(len(new["predictions"]), 3)
        self.assertTrue(all(len(model) == 5 for model in new["predictions"]))
        self.assertTrue(all(len(seed) == 53 for model in new["predictions"] for seed in model))

    def test_missing_seed_blocks_whole_group_without_survivor_panel(self):
        for group in ("old", "new"):
            value = campaign_fixture(group)
            names = report.OLD_MODELS if group == "old" else report.NEW_MODELS
            value["models"][names[0]]["refit"].pop()
            with self.subTest(group=group):
                panel = self.extract(value, group)
                self.assertEqual(panel["status"], "blocked_incomplete_panel")
                self.assertFalse(panel["predictions"])

    def test_failed_attempt_blocks_whole_group_without_survivor_panel(self):
        value = campaign_fixture("new")
        row = value["models"][report.NEW_MODELS[0]]["selection"][0]
        row["status"] = "failed"
        row["worker_profile"]["status"] = "failed"
        panel = self.extract(value, "new")
        self.assertEqual(panel["status"], "blocked_incomplete_panel")
        self.assertFalse(panel["predictions"])

    def test_wrong_seed_is_invalid_instead_of_a_missing_survivor(self):
        value = campaign_fixture("new")
        row = value["models"][report.NEW_MODELS[0]]["refit"][0]
        row["seed"] = report.OLD_SEEDS[0]
        row["worker_profile"]["seed"] = report.OLD_SEEDS[0]
        with self.assertRaises(ValueError):
            self.extract(value, "new")

    def test_wrong_worker_model_is_rejected(self):
        value = campaign_fixture("new")
        value["models"][report.NEW_MODELS[0]]["refit"][0]["worker_profile"]["model"] = "TimesNet"
        with self.assertRaises(ValueError):
            self.extract(value, "new")

    def test_reordered_or_replaced_evaluation_date_is_rejected(self):
        for changed in ("order", "outside"):
            value = campaign_fixture("new")
            dates = value["models"][report.NEW_MODELS[0]]["refit"][0]["worker_profile"]["evaluation"]["dates"]
            if changed == "order":
                dates[0], dates[1] = dates[1], dates[0]
            else:
                dates[0] = "2025-01-01T00:00:00+00:00"
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.extract(value, "new")

    def test_data_source_hash_mismatch_is_rejected(self):
        value = campaign_fixture("new")
        hashes = value["models"][report.NEW_MODELS[0]]["refit"][0]["worker_profile"]["data_receipt"]["source_sha256"]
        hashes[next(iter(hashes))] = "e" * 64
        with self.assertRaises(ValueError):
            self.extract(value, "new")

    def test_truth_must_be_identical_for_every_model_and_seed(self):
        value = campaign_fixture("new")
        value["models"][report.NEW_MODELS[-1]]["refit"][-1]["worker_profile"]["evaluation"]["truths"][5] += .001
        with self.assertRaises(ValueError):
            self.extract(value, "new")

    def test_nonfinite_predictions_are_rejected(self):
        value = campaign_fixture("new")
        value["models"][report.NEW_MODELS[0]]["refit"][0]["worker_profile"]["evaluation"]["predictions"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.extract(value, "new")

    def test_new_worker_source_invariance_requires_equal_hash_maps(self):
        value = campaign_fixture("new")
        worker = value["models"][report.NEW_MODELS[0]]["refit"][0]["worker_profile"]
        worker["source_sha256_after"][next(iter(report.CORE_SOURCE_SHA256))] = "e" * 64
        with self.assertRaises(ValueError):
            self.extract(value, "new")

    def test_new_summary_source_invariance_is_required(self):
        value = campaign_fixture("new")
        value["source_hashes_unchanged"] = False
        with self.assertRaises(ValueError):
            self.extract(value, "new")

    def test_group_version_and_registration_cannot_be_mixed(self):
        for field, changed in [("version", "e2-preselection-reference-variance-v1"),
                               ("registration_file_sha256", "e" * 64),
                               ("registration_commit", "f" * 40)]:
            value = campaign_fixture("new")
            value[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.extract(value, "new")

    def test_analysis_receives_separate_two_and_three_model_groups(self):
        panels = [self.extract(campaign_fixture(group), group) for group in ("old", "new")]
        config = object()
        fake = SimpleNamespace(
            CalibrationConfig=Mock(return_value=config),
            calibrated_power_analysis=Mock(return_value={"registered_design_match": True}))
        with patch.object(report, "load_frozen_analysis", return_value=fake) as load:
            outputs = [report.run_group(panel, .0001) for panel in panels]
        self.assertEqual(load.call_count, 2)
        self.assertEqual(fake.CalibrationConfig.call_count, 2)
        self.assertEqual(fake.calibrated_power_analysis.call_count, 2)
        for index, (panel, call) in enumerate(zip(panels, fake.calibrated_power_analysis.call_args_list)):
            self.assertEqual(len(call.args[0]), 2 + index)
            self.assertEqual(call.args, (panel["predictions"], panel["truths"], .0001))
            self.assertEqual(call.kwargs["m"], 153)
            self.assertEqual(call.kwargs["dates"], panel["dates"])
            self.assertEqual(call.kwargs["model_names"], panel["model_names"])
            self.assertEqual(call.kwargs["seed_ids"], panel["seed_ids"])
            self.assertIs(call.kwargs["config"], config)
            self.assertTrue(outputs[index]["simulations_executed"])
            self.assertNotIn("predictions", outputs[index])

    def test_blocked_group_never_loads_or_runs_simulation_code(self):
        value = campaign_fixture("new")
        value["models"][report.NEW_MODELS[0]]["refit"].pop()
        panel = self.extract(value, "new")
        with patch.object(report, "load_frozen_analysis") as load:
            result = report.run_group(panel, .0001)
        load.assert_not_called()
        self.assertIsNone(result["analysis"])
        self.assertFalse(result["simulations_executed"])

    def test_validate_only_hashes_panel_values_without_loading_simulation(self):
        panel = self.extract(campaign_fixture("new"), "new")
        changed = deepcopy(panel)
        changed["predictions"][0][0][0] += .001
        with patch.object(report, "load_frozen_analysis") as load:
            result = report.run_group(panel, .0001, validate_only=True)
            other = report.run_group(changed, .0001, validate_only=True)
        load.assert_not_called()
        self.assertTrue(result["validation_only"])
        self.assertFalse(result["simulations_executed"])
        self.assertIsNone(result["analysis"])
        self.assertRegex(result["panel_values_json_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(result["panel_values_json_sha256"], other["panel_values_json_sha256"])
        self.assertNotIn("predictions", result)

    def test_wrong_analysis_source_bytes_fail_before_module_import(self):
        with tempfile.TemporaryDirectory() as root:
            scripts = Path(root) / "scripts"
            scripts.mkdir()
            (scripts / "e2_calibrated_planning.py").write_text("import numpy\n")
            with patch.object(report, "ROOT", Path(root)), \
                 patch.object(report.importlib.util, "spec_from_file_location") as source_loader:
                with self.assertRaises(ValueError):
                    report.load_frozen_analysis()
            source_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
