"""Prospective policy tests: synthetic numbers/Git fixtures, no market files or fits."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import e2_protocol_plan as p


class ProtocolPolicyTests(unittest.TestCase):
    def setUp(self):
        self.hpo, self.prereg, self.arch = p.read_documents()

    def test_closed_policy_has_twenty_models_fifteen_tunables_without_execution(self):
        r = p.validate_documents(self.hpo, self.prereg, self.arch)
        self.assertEqual((r["catalogue_size"], r["tunable_recipes"], r["trials_each"]), (20, 15, 12))
        self.assertFalse(r["ready_to_train_by_this_validation"])
        self.assertFalse(r["registered_by_this_validation"])
        self.assertEqual(r["new_Bitcoin_results"], 0)

    def test_inherited_margin_partition_and_primary_reference_are_protected(self):
        for key, value in [("absolute_delta", .01), ("history_days", 30),
                           ("baseline", "GRU-E2"), ("new_unseen_test_assigned", True)]:
            changed = copy.deepcopy(self.prereg); changed["inherited_contract"][key] = value
            with self.assertRaises(ValueError): p.validate_documents(self.hpo, changed, self.arch)
        changed = copy.deepcopy(self.prereg); changed["inherited_contract"]["partition"] = {}
        with self.assertRaises(ValueError): p.validate_documents(self.hpo, changed, self.arch)

    def test_retuned_and_historical_references_cannot_be_merged(self):
        changed = copy.deepcopy(self.prereg)
        changed["families"][1]["hypotheses"][0]["reference"] = "GRU/D7-c03"
        with self.assertRaises(ValueError): p.validate_documents(self.hpo, changed, self.arch)

    def test_family_members_and_alpha_cannot_shrink_after_missing_results(self):
        changed = copy.deepcopy(self.prereg); changed["families"][0]["hypotheses"].pop()
        with self.assertRaises(ValueError): p.validate_documents(self.hpo, changed, self.arch)
        changed = copy.deepcopy(self.prereg); changed["families"][0]["alpha"] = .05
        with self.assertRaises(ValueError): p.validate_documents(self.hpo, changed, self.arch)
        changed = copy.deepcopy(self.prereg); changed["diagnostic_family"]["hypotheses"].pop()
        with self.assertRaises(ValueError): p.validate_documents(self.hpo, changed, self.arch)

    def test_p0_cannot_be_reported_as_power_of_hpo_selected_recipes(self):
        changed = copy.deepcopy(self.prereg)
        changed["pilots"]["P0"]["not_final_HPO_recipe_power"] = False
        with self.assertRaises(ValueError): p.validate_documents(self.hpo, changed, self.arch)
        self.assertTrue(self.prereg["power"]["P0_not_valid_power_evidence"])

    def test_primary_p0_gap_calendar_is_oof_not_chosen_between_two_families(self):
        family = self.prereg["pilots"]["P0"]["development_gap_family"]
        self.assertEqual(family["potential_calendar_origins"],305)
        self.assertFalse(family["secondary_NovDec_inferential_family"])
        changed=copy.deepcopy(self.prereg)
        changed["pilots"]["P0"]["development_gap_family"]["primary_dataset"]="NovDec"
        with self.assertRaises(ValueError):p.validate_documents(self.hpo,changed,self.arch)

    def test_power_target_is_above_null_boundary_and_delta_g_is_operational(self):
        self.assertEqual(self.prereg["alternatives_for_planning"]["RQ1_gain"], 2*p.DELTA)
        self.assertEqual(self.prereg["generalization_gap"]["delta_g"], p.DELTA)
        changed = copy.deepcopy(self.prereg)
        changed["alternatives_for_planning"]["RQ1_gain"] = p.DELTA
        with self.assertRaises(ValueError): p.validate_documents(self.hpo, changed, self.arch)

    def test_phase_randomizations_are_disjoint_and_fresh_p0_ids_fixed(self):
        changed = copy.deepcopy(self.prereg)
        changed["pilots"]["P1"]["seed_ids"][0] = changed["pilots"]["P0"]["seed_ids"][0]
        with self.assertRaises(ValueError): p.validate_documents(self.hpo, changed, self.arch)
        self.assertEqual(self.prereg["pilots"]["P0"]["seed_ids"], list(range(660101,660106)))

    def test_equal_assigned_budget_does_not_silently_accept_more_trials(self):
        changed = copy.deepcopy(self.hpo); changed["tunable_models"][0]["trials"] = 13
        with self.assertRaises(ValueError): p.validate_documents(changed, self.prereg, self.arch)
        changed = copy.deepcopy(self.hpo); changed["tunable_models"][-1]["assigned_compute_seconds"] = 10
        with self.assertRaises(ValueError): p.validate_documents(changed, self.prereg, self.arch)

    def test_minimum_neural_axes_and_conditional_nas_are_not_interchangeable(self):
        changed = copy.deepcopy(self.hpo)
        del changed["tunable_models"][0]["space"]["weight_decay"]
        with self.assertRaises(ValueError): p.validate_documents(changed, self.prereg, self.arch)
        changed = copy.deepcopy(self.arch); changed["models"][0]["nas_required"] = True
        with self.assertRaises(ValueError): p.validate_documents(self.hpo, self.prereg, changed)

    def test_architecture_case_needs_evidence_and_cannot_claim_runtime_from_spec(self):
        for field, value in [("evidence", []), ("runtime_validated_by_this_record", True)]:
            changed = copy.deepcopy(self.arch); changed["models"][0][field] = value
            with self.assertRaises(ValueError): p.validate_documents(self.hpo, self.prereg, changed)

    def test_candidate_design_is_precomputed_reproducible_and_has_twelve_each(self):
        a = p.candidate_design(self.hpo, self.arch); b = p.candidate_design(self.hpo, self.arch)
        self.assertEqual(a, b); self.assertEqual(len(a), 180)
        self.assertEqual(len({r["candidate_id"] for r in a}), 180)
        for model in p.TUNABLES:
            self.assertEqual(sum(r["model"] == model for r in a), 12)
        lr = [r["parameters"]["learning_rate"] for r in a if r["model"] == "TimeMixer"]
        self.assertTrue(all(.0001 <= x <= .003 for x in lr))

    def test_arima_design_is_full_registered_twelve_order_space_with_valid_trends(self):
        rows = [r for r in p.candidate_design(self.hpo, self.arch) if r["model"] == "ARIMA"]
        self.assertEqual(len({tuple(r["parameters"]["order"]) for r in rows}), 12)
        for row in rows:
            d = row["parameters"]["order"][1]
            self.assertEqual(row["parameters"]["trend"], "c" if d == 0 else "n")

    def test_nas_catalogue_has_1000_unique_cases_including_control_and_no_pruning(self):
        rows = p.nas_candidates()
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({r["architecture_id"] for r in rows}), 1000)
        self.assertEqual(sum(r["dropout_bits"] == [0,0,0] for r in rows), 125)
        self.assertTrue(any(r["widths"] == [64,64,64] and r["dropout_bits"] == [0,1,0] for r in rows))
        self.assertTrue(all(r["status"] == "conditional_catalogue_not_executed" for r in rows))

    def test_noise_informed_k_is_fixed_before_screening_and_zero_between_is_conservative(self):
        self.assertEqual(p.choose_nas_k(.1,.2), 20)
        self.assertEqual(p.choose_nas_k(.2,.1), 40)
        self.assertEqual(p.choose_nas_k(0.,0.), 40)
        with self.assertRaises(ValueError): p.choose_nas_k(float("nan"), .1)

    def test_nas_phase2_duplicates_cannot_impersonate_independent_seeds(self):
        changed = copy.deepcopy(self.arch)
        changed["nas_protocol"]["stage2"]["seed_ids"] = [6501]*5
        with self.assertRaisesRegex(ValueError,"independent"): p.validate_documents(self.hpo,self.prereg,changed)

    def test_published_source_recipes_are_separate_from_tiny_p0_and_final_review(self):
        entries = {m["id"]:m for m in self.arch["models"]}
        self.assertEqual(entries["PatchTST"]["P0_fixed_recipe"]["e_layers"], 2)
        self.assertEqual(entries["PatchTST"]["fixed_source_recipe"]["e_layers"], 3)
        self.assertEqual(entries["PatchTST"]["fixed_source_recipe"]["d_ff"], 128)
        self.assertFalse(entries["PatchTST"]["final_architecture_exception_verified"])

    def test_normal_reference_R_scales_quadratically_but_does_not_certify_temporal_power(self):
        a = p.approximate_paired_R(.001); b = p.approximate_paired_R(.002)
        self.assertTrue(4*a-4 <= b <= 4*a)
        with self.assertRaises(ValueError): p.approximate_paired_R(0.)
        with self.assertRaises(ValueError): p.approximate_paired_R(.1, excess=0.)

    def small_power_fixture(self):
        plan = {"families": [{"hypotheses": [{"id": "synthetic:rq1"}]}],
                "power": {"candidate_R": [5,10], "maximum_R": 10, "sources": ["conditional","temporal"],
                          "target": .8, "mc_repetitions": 10000, "mc_seed": 123}}
        rows = [{"R": r, "hypothesis_id": "synthetic:rq1", "source": s,
                 "successes": 7000 if r==5 else 9000, "repetitions": 10000, "mc_seed": 123,
                 "planning_kind": "conservative_representative_proxy", "pilot_receipt_sha256": "a"*64}
                for r in [5,10] for s in ["conditional","temporal"]]
        return plan, rows

    def test_power_selection_needs_both_sources_and_mc_uncertainty_not_just_point_estimate(self):
        plan, rows = self.small_power_fixture(); r = p.select_final_R(rows, plan)
        self.assertEqual(r["selected_R"], 10); self.assertTrue(r["target_met_in_planning_simulation"])
        self.assertFalse(r["unpiloted_architectures_empirical_power_certified"])
        rows[-1]["successes"] = 8000
        r = p.select_final_R(rows, plan)
        self.assertFalse(r["target_met_in_planning_simulation"])
        self.assertEqual(r["status"], "cap_selected_target_not_met"); self.assertTrue(r["MDE_report_required"])

    def test_missing_duplicate_or_unknown_power_cells_do_not_certify_power(self):
        plan, rows = self.small_power_fixture()
        for bad in [rows[:-1], rows+rows[:1]]:
            with self.assertRaises(ValueError): p.select_final_R(bad, plan)
        rows[0]["mc_seed"] = 1
        with self.assertRaises(ValueError): p.select_final_R(rows, plan)

    def test_default_validation_uses_standard_library_and_writes_no_benchmark_results(self):
        run = subprocess.run([sys.executable, "-S", str(ROOT/"scripts/e2_protocol_plan.py"), "validate"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)["new_Bitcoin_results"], 0)

    def test_exports_only_static_policy_and_unevaluated_catalogues(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-protocol-export-",dir="/tmp") as tmp:
            r = p.export(output=tmp)
            self.assertFalse(r["ready_to_train_by_this_validation"])
            provenance = json.loads((Path(tmp)/"provenance.json").read_text())
            self.assertEqual(provenance["training_fits"], 0)
            self.assertEqual(provenance["market_or_prediction_files_opened"], 0)
            self.assertTrue(all(x["status"].endswith("not_evaluated") for x in
                                json.loads((Path(tmp)/"candidate_registry.json").read_text())))

    def git_fixture(self, directory):
        root = Path(directory)
        watched = self.prereg["registration"]["tracked_paths"] + ["book_e2/_static/e1_reference.json"]
        watched += [self.arch["literature_registry"]["path"]]
        watched += [e["path"] for model in self.arch["models"] for e in model["evidence"]
                    if e["kind"] == "explicit_local_architecture"]
        for relative in watched:
            dest = root/relative; dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,dest)
        commands = [["git","init","-q"], ["git","config","user.name","Synthetic Test"],
                    ["git","config","user.email","synthetic@example.invalid"], ["git","add","."],
                    ["git","commit","-qm","Register synthetic prospective policy fixture"]]
        for command in commands: subprocess.run(command,cwd=root,check=True,capture_output=True)
        return root

    def receipts(self, root, phase):
        evidence = root/"synthetic_preflight.txt"; evidence.write_text("SYNTHETIC receipt evidence; no real experiment")
        receipts = {name:{"status":"passed","preregistration_sha256":p.digest(root/p.PLAN_PATHS[1]),
                          "evidence_path":str(evidence),"evidence_sha256":p.digest(evidence)}
                    for name in p.PHASE_REQUIREMENTS[phase]}
        return receipts

    def test_registration_requires_exact_committed_bytes_then_detects_changes(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-protocol-git-",dir="/tmp") as tmp:
            root = self.git_fixture(tmp); receipt = p.registration_receipt(root)
            self.assertTrue(p.check_registration(receipt,root=root))
            (root/"scripts/e2_protocol_plan.py").write_text("# changed after registration\n")
            with self.assertRaisesRegex(ValueError,"modified"): p.check_registration(receipt,root=root)
            with self.assertRaisesRegex(ValueError,"Commit"): p.registration_receipt(root)

    def test_p0_phase_requires_registration_and_every_hash_bound_preflight_receipt(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-protocol-gate-",dir="/tmp") as tmp:
            root = self.git_fixture(tmp); registration = p.registration_receipt(root)
            with self.assertRaisesRegex(ValueError,"environment"): p.require_phase("P0",registration,{},root=root)
            receipts = self.receipts(root,"P0")
            r = p.require_phase("P0",registration,receipts,root=root)
            self.assertTrue(r["receipt_checks_passed"]); self.assertFalse(r["fit_or_score_performed"])
            Path(receipts["environment"]["evidence_path"]).write_text("tampered fixture")
            with self.assertRaisesRegex(ValueError,"changed"): p.require_phase("P0",registration,receipts,root=root)

    def test_nas_cannot_be_run_automatically_for_architectures_declared_defined(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-protocol-nas-",dir="/tmp") as tmp:
            root = self.git_fixture(tmp); registration = p.registration_receipt(root)
            with self.assertRaisesRegex(ValueError,"No registered"):p.require_phase("NAS_stage1",registration,{},root=root)

    def test_final_hpo_requires_casewise_exception_review_but_p0_still_has_own_gate(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-final-review-",dir="/tmp") as tmp:
            root = self.git_fixture(tmp); registration = p.registration_receipt(root)
            with self.assertRaisesRegex(ValueError,"Final architecture review"):
                p.require_phase("HPO",registration,{},root=root)
            self.assertTrue(p.require_phase("P0",registration,self.receipts(root,"P0"),root=root)["receipt_checks_passed"])

    def test_score_gate_rejects_second_scoring_phase_and_final_fit_needs_R(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-protocol-score-",dir="/tmp") as tmp:
            root = self.git_fixture(tmp)
            path = root/p.PLAN_PATHS[2]; architecture=json.loads(path.read_text())
            for model in architecture["models"]:model["final_architecture_exception_verified"]=True
            path.write_bytes(p.json_data(architecture))
            subprocess.run(["git","add","."],cwd=root,check=True,capture_output=True)
            subprocess.run(["git","commit","-qm","Synthetic final review fixture only"],cwd=root,check=True,capture_output=True)
            registration = p.registration_receipt(root)
            receipts = self.receipts(root,"score"); receipts["prior_E2_scoring_events"] = 1
            with self.assertRaisesRegex(ValueError,"Only one"):p.require_phase("score",registration,receipts,root=root)
            receipts = self.receipts(root,"final_fit")
            with self.assertRaisesRegex(ValueError,"concrete"):p.require_phase("final_fit",registration,receipts,root=root)
            receipts["power_selection"]["selection"] = {"selected_R":200,"target_met_in_planning_simulation":False}
            r = p.require_phase("final_fit",registration,receipts,root=root)
            self.assertFalse(r["fit_or_score_performed"])


if __name__ == "__main__":
    unittest.main()
