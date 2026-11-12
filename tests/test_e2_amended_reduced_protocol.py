"""Reduced protocol domain, global selection and registration tests; no Torch."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_reduced_protocol as reduced


def parent_domains():
    """Explicit artificial domains; never current market results or weights."""
    domains = {}
    for index, model in enumerate(reduced.NEURAL):
        recipe, dropout = reduced.RESTORED.get(model, ("fixture-recipe-%02d" % index, .1))
        domains[model] = [{"trial_id": recipe + ":hpo-%03d" % i, "recipe_id": recipe,
            "learning_rate": .00001 * 1.3**i, "weight_decay": 0 if i % 4 == 0 else .0001,
            "batch_size": (16, 32, 64)[i % 3], "optimization_seeds": [271201, 271202],
            "dropout": dropout} for i in range(18)]
    return domains


def run_ledger(domains, model="GRU32"):
    rows = []
    for index, configured in enumerate(domains[model]):
        for fold_index, fold in enumerate(reduced.FOLDS):
            rows.append({"model_id": model, "configuration_id": configured["configuration_id"],
                "fold_origin": fold, "seed": reduced.HPO_SEED,
                "parent_trial_id": configured["parent_trial_id"],
                "configuration_sha256": configured["configuration_sha256"],
                "run_receipt_sha256": hashlib.sha256((model + str(index) + fold).encode()).hexdigest(),
                "status": "completed", "validation_count": 83,
                "rmse_original": 1. + index + fold_index / 10,
                "best_epoch": [3, 6, 9][fold_index]})
    return rows


def rehash(row):
    row["configuration_sha256"] = reduced.digest({k: v for k, v in row.items() if k != "configuration_sha256"})


class ReducedDomainTests(unittest.TestCase):
    def setUp(self):
        self.parents = parent_domains()
        self.domains = reduced.planning_six_domains(self.parents)

    def test_six_predetermined_slots_per_model_preserve_parent_bytes(self):
        self.assertEqual(len(self.domains), 13)
        for model, rows in self.domains.items():
            self.assertEqual(len(rows), 6)
            self.assertEqual([r["configuration_id"] for r in rows], list(reduced.SLOT_IDS))
            self.assertEqual([r["parent_trial_index"] for r in rows], [5, 7, 8, 9, 10, 15])
            for row in rows:
                self.assertEqual(row["trial"], self.parents[model][row["parent_trial_index"]])
                self.assertEqual(row["execution_seed_ids"], [271201])
                self.assertEqual(row["trial"]["optimization_seeds"], [271201, 271202])
        self.assertEqual(self.parents, parent_domains())

    def test_common_slots_are_not_misrepresented_as_identical_numeric_architectures(self):
        self.assertNotEqual(self.domains["FCN-1D"][0]["trial"]["recipe_id"],
                            self.domains["GRU32"][0]["trial"]["recipe_id"])
        protocol = reduced.build_reduced_protocol(self.domains)
        self.assertFalse(protocol["hpo"]["identical_numeric_optimizer_tuples_across_models"])
        self.assertEqual(protocol["hpo"]["logical_fit_upper_bound_before_reuse"], 234)

    def test_TiDE_is_blocked_and_cannot_be_relabelled_as_published(self):
        self.assertTrue(all(not row["execution_allowed"] for row in self.domains["TiDE"]))
        changed = deepcopy(self.domains)
        changed["TiDE"][0]["execution_allowed"] = True
        rehash(changed["TiDE"][0])
        with self.assertRaises(ValueError):
            reduced.validate_domains(changed)
        with self.assertRaisesRegex(ValueError, "TiDE remains blocked"):
            reduced.select_reduced_hpo([], "TiDE", self.domains)

    def test_foreign_parent_index_duplicate_ids_and_changed_rows_are_rejected(self):
        changed = deepcopy(self.domains)
        changed["GRU32"][0]["trial"]["learning_rate"] = .003
        with self.assertRaises(ValueError):
            reduced.validate_domains(changed)
        changed = deepcopy(self.domains)
        changed["GRU32"][0]["parent_trial_index"] = 1
        rehash(changed["GRU32"][0])
        with self.assertRaises(ValueError):
            reduced.validate_domains(changed)
        parents = parent_domains()
        parents["GRU32"][1]["trial_id"] = parents["GRU32"][0]["trial_id"]
        with self.assertRaises(ValueError):
            reduced.planning_six_domains(parents)

    def test_published_dropout_and_recipe_stay_fixed(self):
        for model, (recipe, dropout) in reduced.RESTORED.items():
            self.assertTrue(all(row["trial"]["recipe_id"] == recipe and row["trial"]["dropout"] == dropout
                                for row in self.domains[model]))
        changed = deepcopy(self.domains)
        changed["PatchTST"][0]["trial"]["dropout"] = .5
        rehash(changed["PatchTST"][0])
        with self.assertRaises(ValueError):
            reduced.validate_domains(changed)

    def test_slot_coverage_requires_all_batches_and_zero_and_positive_decay(self):
        for field, value in (("batch_size", 32), ("weight_decay", .0001)):
            changed = deepcopy(self.domains)
            for row in changed["GRU32"]:
                row["trial"][field] = value
                rehash(row)
            with self.subTest(field=field), self.assertRaises(ValueError):
                reduced.validate_domains(changed)

    def test_calendar_embargo_and_inherited_partition_are_preserved(self):
        protocol = reduced.build_reduced_protocol(self.domains)
        first = protocol["hpo"]["folds"][0]
        self.assertEqual(first["inner_train_label_end"], "2023-09-25")
        self.assertEqual(first["inner_validation_start"], "2023-10-03")
        self.assertEqual(first["inner_validation_end"], "2023-12-24")
        self.assertEqual(protocol["evaluation"]["training_label_end"], "2024-10-31")
        self.assertEqual(protocol["evaluation"]["calibration_start"], "2024-11-08")
        self.assertEqual(protocol["evaluation"]["calibration_end"], "2024-12-30")
        self.assertEqual(protocol["evaluation"]["causal_bridge_after_calibration"], ["2024-12-31"])
        self.assertEqual(protocol["evaluation"]["calendar_days"], 608)
        self.assertFalse(protocol["evaluation"]["blindness_recovered"])
        self.assertFalse(protocol["evaluation"]["new_100_day_cohort_required_for_this_delivery"])
        self.assertFalse(protocol["execution"]["ready"])
        self.assertFalse(protocol["replications"]["power_80_percent_claimed"])


class ReducedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.domains = reduced.planning_six_domains(parent_domains())
        self.rows = run_ledger(self.domains)

    def test_one_global_three_fold_recipe_not_local_winner_or_test_score(self):
        # C01 wins two folds but loses badly on the third; C02 wins the global
        # mean. This distinguishes the intended criterion from local selection.
        for row in self.rows:
            if row["configuration_id"] == "C01":
                row["rmse_original"] = 100. if row["fold_origin"] == reduced.FOLDS[-1] else 0.
        result = reduced.select_reduced_hpo(self.rows, "GRU32", self.domains)
        self.assertEqual(result["winner_configuration_id"], "C02")
        self.assertAlmostEqual(result["pooled_cv_RMSE"], ((2**2 + 2.1**2 + 2.2**2) / 3)**.5)
        self.assertAlmostEqual(result["winner_mean_fold_RMSE_original_diagnostic"], 2.1)
        self.assertEqual(result["validation_count"], 249)
        self.assertEqual(result["fixed_epochs"], 6)
        self.assertTrue(result["all_eighteen_slots_accounted_for"])
        self.assertTrue(result["criteria_only"])
        self.assertFalse(result["run_artifact_bytes_verified"])
        self.assertFalse(result["new_training_performed"])

    def test_tie_uses_slot_id_and_all_three_winner_epochs(self):
        for row in self.rows:
            row["rmse_original"] = 1.
        result = reduced.select_reduced_hpo(list(reversed(self.rows)), "GRU32", self.domains["GRU32"])
        self.assertEqual(result["winner_configuration_id"], "C01")
        self.assertEqual(result["fixed_epochs"], 6)

    def test_concatenated_RMSE_not_mean_fold_RMSE_controls_selection(self):
        for row in self.rows:
            row["rmse_original"] = 10.
            if row["configuration_id"] == "C01":
                row["rmse_original"] = 3. if row["fold_origin"] == reduced.FOLDS[-1] else 0.
            elif row["configuration_id"] == "C02":
                row["rmse_original"] = 1.5
        result = reduced.select_reduced_hpo(self.rows, "GRU32", self.domains)
        # Arithmetic mean would choose C01:1<1.5; concatenated RMSE chooses
        # C02:1.5<sqrt(3). Both use the same249 validation predictions.
        self.assertEqual(result["winner_configuration_id"], "C02")
        self.assertEqual(result["pooled_cv_RMSE"], 1.5)
        c01 = result["candidate_ledger"][0]
        self.assertEqual(c01["mean_fold_RMSE_original_diagnostic"], 1.)
        self.assertAlmostEqual(c01["pooled_cv_RMSE_original"], 3**.5)

    def test_changed_or_missing_validation_support_rejected(self):
        for value in (82, 84, 83.0, True, None):
            rows = deepcopy(self.rows)
            rows[0]["validation_count"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "validation dates"):
                reduced.select_reduced_hpo(rows, "GRU32", self.domains)

    def test_incomplete_duplicate_or_foreign_slots_cannot_select(self):
        for rows in (self.rows[:-1], [*self.rows[:-1], self.rows[0]]):
            with self.assertRaises(ValueError):
                reduced.select_reduced_hpo(rows, "GRU32", self.domains)
        rows = deepcopy(self.rows)
        rows[0]["configuration_id"] = "C07"
        with self.assertRaises(ValueError):
            reduced.select_reduced_hpo(rows, "GRU32", self.domains)

    def test_failed_attempt_is_retained_without_replacement_or_survivor_mean(self):
        self.rows[0].update(status="failed", reason="registered wall timeout")
        result = reduced.select_reduced_hpo(self.rows, "GRU32", self.domains)
        self.assertEqual(result["winner_configuration_id"], "C02")
        self.assertEqual(result["candidate_ledger"][0]["status"], "failed")
        self.assertFalse(result["candidate_ledger"][0]["replacement_performed"])
        with self.assertRaises(ValueError):
            self.rows[0]["reason"] = ""
            reduced.select_reduced_hpo(self.rows, "GRU32", self.domains)

    def test_fatal_systematic_and_unclassified_failures_abort(self):
        for field in ("fatal", "systematic_failure"):
            rows = deepcopy(self.rows)
            rows[0][field] = True
            with self.assertRaisesRegex(ValueError, "Fatal or systematic"):
                reduced.select_reduced_hpo(rows, "GRU32", self.domains)
        rows = deepcopy(self.rows)
        rows[0]["status"] = "paused"
        with self.assertRaises(ValueError):
            reduced.select_reduced_hpo(rows, "GRU32", self.domains)

    def test_wrong_seed_receipt_hash_recipe_or_earlier_origin_rejected(self):
        for field, value in (("seed", 271202), ("run_receipt_sha256", "missing"),
                             ("parent_trial_id", "old-recipe"), ("configuration_sha256", "0" * 64),
                             ("rmse_original", float("nan")), ("best_epoch", 0)):
            rows = deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                reduced.select_reduced_hpo(rows, "GRU32", self.domains)
        with self.assertRaisesRegex(ValueError, "earlier outer"):
            reduced.select_reduced_hpo(self.rows, "GRU32", self.domains, final_origin="2024-07-01")

    def test_SAE_requires_and_freezes_all_three_optimizer_phase_epochs(self):
        model = "LSTM-autoencoder con predictor"
        rows = run_ledger(self.domains, model)
        with self.assertRaisesRegex(ValueError, "optimizer phase"):
            reduced.select_reduced_hpo(rows, model, self.domains)
        for index, row in enumerate(rows):
            row["phase_best_epochs"] = {"ae1": [2, 4, 5][index % 3],
                "ae2": [9, 12, 15][index % 3], "supervised": row["best_epoch"]}
        result = reduced.select_reduced_hpo(rows, model, self.domains)
        self.assertEqual(result["fixed_phase_epochs"], {"ae1": 4, "ae2": 12, "supervised": 6})


class ReducedRegistrationTests(unittest.TestCase):
    def make_files(self, root):
        domains = reduced.planning_six_domains(parent_domains())
        protocol = reduced.build_reduced_protocol(domains)
        content = {reduced.PROTOCOL_PATH: protocol, reduced.CONFIGURATIONS_PATH: domains}
        for relative, value in content.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        script = root / "scripts/e2_reduced_protocol.py"
        script.parent.mkdir(parents=True)
        script.write_text("# explicit synthetic source fixture\n")
        files = {relative: reduced.file_digest(root / relative)
                 for relative in [*content, "scripts/e2_reduced_protocol.py"]}
        registration = {"protocol_id": reduced.PROTOCOL_ID, "parent_source_commit": reduced.PARENT,
            "protocol_sha256": files[reduced.PROTOCOL_PATH], "files_sha256": files}
        (root / reduced.REGISTRATION_PATH).write_text(json.dumps(registration))
        return protocol, domains

    def test_frozen_bytes_verify_without_claiming_GPU_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_files(root)
            result = reduced.verify_reduced_registration(root)
            self.assertEqual(result["source_bytes_verified"], 3)
            self.assertFalse(result["execution_ready"])
            self.assertFalse(result["market_training_started"])
            self.assertFalse(result["private_colab_GPU_verified"])
            self.assertFalse(result["git_registration"]["verified"])

    def test_source_tampering_budget_reset_or_changed_contract_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol, domains = self.make_files(root)
            (root / "scripts/e2_reduced_protocol.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "source bytes changed"):
                reduced.verify_reduced_registration(root)
            for branch, key, value in (("budget", "budget_reset", True),
                                       ("budget", "authorized_total_hours", 72),
                                       ("contract", "delta", .01)):
                changed = deepcopy(protocol)
                changed[branch][key] = value
                with self.assertRaises(ValueError):
                    reduced.validate_reduced_protocol(changed, domains)


if __name__ == "__main__":
    unittest.main()
