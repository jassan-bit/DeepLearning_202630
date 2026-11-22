"""Bounded synthetic fixtures; no market data, training or model forwards."""
from copy import deepcopy
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from engine.assessment import (REQUIRED_CRITERIA, _adequacy, _matrix, _method_admission, _power_evidence,
                               _panel_rule, _robustness, _sealed_evaluation, _threshold_admission, assess_case,
                               between_panel_tests, decide_r3, fit_calibrators,
                               residual_diagnostics, score_intervals)
from engine.common import digest, read_json, sha, write_csv, write_json
from engine.models import CANDIDATE, DETERMINISTIC, METHODS
from engine.statistics import PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[1]


def analysis_fixture(negative_panel=False, raw_p=1e-9):
    comparisons = []
    metrics = {name: {"mse": 1. if name == CANDIDATE else 2.,
                      "mean_seed_rmse": 1. if name == CANDIDATE else 1.5} for name in METHODS}
    for ri, name in enumerate(METHODS[1:]):
        sign = -1 if negative_panel and ri == 0 else 1
        methods = {m: {null: {"status": "ok", "raw_p": raw_p,
                              "adjusted_p": raw_p*120,
                              "direction": "candidate_better" if sign > 0 else "candidate_worse"}
                        for null in ("epa", "practical")}
                   for m in ("dm_hln_hac", "stationary_bootstrap", "gw_conditional")}
        comparisons.append({"rival": name, "methods": methods, "mean_mse_difference": float(sign),
                            "practical_mean": float(sign), "temporal_effect_cohen_d": .5*sign,
                            "ci95_mse_difference": [.5*sign, 1.5*sign],
                            "ci95_relative_rmse_gain": [.03*sign, .08*sign],
                            "historical_mean_seed_rmse_favorable": sign > 0})
        if sign < 0:
            metrics[name]["mean_seed_rmse"] = .5
    return {"candidate": CANDIDATE, "metrics": metrics, "comparisons": comparisons,
            "family": {"rivals": 20}, "systems": list(METHODS)}


class AssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_json(ROOT/"configs/protocol.json")
        cls.seeds = [str(x) for x in read_json(ROOT/"configs/FINAL_SEED_BANK.json")["seed_ids"][:10]]
        cls.protocol["statistical_settings"] = {"bootstrap_reps": 99, "block_length": 3,
            "hac_lags": 2, "horizon": 1, "margin": .02, "seed": 191, "reference_reps": 999}
        cls.protocol["final"].update(R=10, T=72, B=99, seed_ids=cls.seeds)
        cls.config = read_json(ROOT/"configs/adequacy.json")
        cls.config["primary_estimand_version"] = PROTOCOL_VERSION
        cls.config["robustness"]["temporal_alternatives"] = [{"id": "longer", "hac_lags": 3, "block_length": 4}]
        cls.protocol["adequacy"]["config_sha256"] = digest(cls.config)
        cls.rows = {split: [] for split in ("fit", "selection", "calibration", "test")}
        offset = 0
        for split, n in (("fit", 24), ("selection", 24), ("calibration", 12), ("test", 72)):
            for panel in cls.protocol["panel_ids"]:
                for mi, method in enumerate(METHODS):
                    seeds = ["deterministic"] if method in DETERMINISTIC else cls.seeds
                    for ri, seed in enumerate(seeds):
                        for t in range(n):
                            y = .01+.0007*np.sin(t*.2)
                            error = .0001*np.sin(t*.71+ri*.1) if method == CANDIDATE else .0009+.0002*np.sin(t*.4+mi*.1)
                            cls.rows[split].append({"panel": panel, "system": method, "seed": seed,
                                "date": str(dt.date(2031, 1, 1)+dt.timedelta(days=offset+t)),
                                "y": float(y), "prediction": float(y+error),
                                "known_floor_A": float((.002+.0001*np.sin(t*.17))**2), "rfit": .01})
            offset += n+8
        cls.calibrators = fit_calibrators(cls.rows["calibration"])

    def simple_rows(self, n, start=0):
        return [{"panel": "F00", "system": CANDIDATE, "seed": self.seeds[0],
                 "date": str(dt.date(2030, 1, 1)+dt.timedelta(days=start+i)),
                 "y": .006+i*.001, "prediction": .006,
                 "known_floor_A": .005**2, "rfit": .01} for i in range(n)]

    def test_finite_conformal_order_statistic_and_small_n(self):
        cal = fit_calibrators(self.simple_rows(9))
        self.assertEqual(cal["calibrators"][0]["rank_one_based"], 9)
        self.assertAlmostEqual(cal["calibrators"][0]["q"], .008)
        small = fit_calibrators(self.simple_rows(8))["calibrators"][0]
        self.assertIsNone(small["q"])
        self.assertEqual(small["status"], "insufficient_calibration_sample")

    def test_calibrator_content_and_file_idempotence(self):
        rows = self.simple_rows(10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"calibrators.json"
            first = fit_calibrators(rows, path)
            original = path.read_bytes()
            second = fit_calibrators(list(reversed(rows)), path)
            self.assertEqual(first, second)
            self.assertEqual(original, path.read_bytes())

    def test_interval_floor_uses_sqrt_A_and_actual_clipped_width(self):
        cal = fit_calibrators(self.simple_rows(9))
        test = self.simple_rows(12, start=30)
        for r in test:
            r["y"] = .007
        result = score_intervals(test, cal, settings=self.protocol["statistical_settings"])[0]
        self.assertAlmostEqual(result["minimum_lower"], .005)
        self.assertAlmostEqual(result["mean_width"], .006+.008-.005)
        self.assertEqual(result["floor_clipped_endpoints"], 12)
        self.assertEqual(result["coverage"], 1.)
        self.assertFalse(result["exchangeable_coverage_guarantee"])

    def test_calibration_cannot_use_test_dates(self):
        cal = fit_calibrators(self.simple_rows(9))
        with self.assertRaisesRegex(ValueError, "precede test"):
            score_intervals(self.simple_rows(12), cal, settings=self.protocol["statistical_settings"])

    def test_truth_conditions_can_pass_and_every_requirement_can_block(self):
        criteria = dict.fromkeys(REQUIRED_CRITERIA, True)
        self.assertTrue(decide_r3(criteria))
        for key in criteria:
            changed = dict(criteria, **{key: False})
            self.assertFalse(decide_r3(changed), key)
        with self.assertRaises(ValueError):
            decide_r3({"adequacy": True})

    def test_additional_inferiority_holm180_preserves_all_slots(self):
        results = [analysis_fixture() for _ in range(4)]
        results[3] = analysis_fixture(negative_panel=True)
        rule = _panel_rule(results, .02)
        self.assertEqual(rule["additional_inferiority_family"]["size"], 180)
        self.assertTrue(rule["additional_significant_inferiority"])
        self.assertFalse(rule["joint_statistical_rule_pass"])
        bad = [s for s in rule["additional_inferiority_family"]["slots"] if s["significant_inferiority"]]
        self.assertEqual(len(bad), 2)
        self.assertTrue(all(s["direction"] == "candidate_worse" for s in bad))

    def test_unsigned_gw_rejection_alone_is_not_mean_inferiority(self):
        results = [analysis_fixture() for _ in range(4)]
        c = results[3]["comparisons"][0]
        c["mean_mse_difference"] = -1e-12
        for method in c["methods"]:
            c["methods"][method]["epa"]["raw_p"] = 1e-9 if method == "gw_conditional" else 1.
        rule = _panel_rule(results, .02)
        self.assertFalse(rule["additional_significant_inferiority"])
        self.assertTrue(rule["additional_conditional_adverse_signal"])
        self.assertEqual(rule["additional_inferiority_family"]["size"], 180)

    def test_no_inferiority_uses_global_family_not_panelwise(self):
        results = [analysis_fixture(raw_p=1.) for _ in range(4)]
        results[3] = analysis_fixture(negative_panel=True, raw_p=1.)
        c = results[3]["comparisons"][0]
        c["methods"]["dm_hln_hac"]["epa"]["raw_p"] = .0006
        rule = _panel_rule(results, .02)
        slot = next(s for s in rule["additional_inferiority_family"]["slots"] if s["panel_index"] == 3 and s["rival"] == c["rival"] and s["method"] == "dm_hln_hac")
        self.assertAlmostEqual(slot["global_additional_epa_adjusted_p"], .108)
        self.assertFalse(rule["additional_significant_inferiority"])

    def test_operational_alpha025_is_preserved_in_additional_family(self):
        results = [analysis_fixture(raw_p=1.) for _ in range(4)]
        results[3] = analysis_fixture(negative_panel=True, raw_p=1.)
        for result in results:
            result["family"]["alpha"] = .025
        c = results[3]["comparisons"][0]
        c["methods"]["dm_hln_hac"]["epa"]["raw_p"] = .0002
        rule = _panel_rule(results, .02)
        slot = next(s for s in rule["additional_inferiority_family"]["slots"]
            if s["panel_index"] == 3 and s["rival"] == c["rival"] and s["method"] == "dm_hln_hac")
        self.assertAlmostEqual(slot["global_additional_epa_adjusted_p"], .036)
        self.assertEqual(rule["additional_inferiority_family"]["alpha"], .025)
        self.assertFalse(slot["significant_inferiority"])
        self.assertFalse(rule["additional_significant_inferiority"])

    def test_additional_two_of_three_is_per_rival(self):
        results = [analysis_fixture() for _ in range(4)]
        results[1] = analysis_fixture(negative_panel=True, raw_p=1.)
        results[2] = analysis_fixture(negative_panel=True, raw_p=1.)
        rule = _panel_rule(results, .02)
        self.assertFalse(rule["historical_additional_mean_rmse_two_of_three"])
        self.assertFalse(rule["additional_mean_pooled_mse_two_of_three"])

    def test_complete_family_and_single_deterministic_trace_enforced(self):
        rows = [r for r in self.rows["test"] if r["system"] != "global_mse"]
        with self.assertRaisesRegex(ValueError, "21 systems"):
            _matrix(rows, "F00", self.seeds)
        rows = deepcopy(self.rows["test"])
        extra = next(deepcopy(r) for r in rows if r["system"] == "Persistence")
        extra["seed"] = self.seeds[0]
        rows.append(extra)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            _matrix(rows, "F00", self.seeds)

    def test_degenerate_residuals_do_not_certify_adequacy(self):
        y = np.linspace(.01, .02, 48)
        result = residual_diagnostics(y, y, fit_scale=.01, hac_lags=2)
        self.assertEqual(result["tests"]["mz_hac"]["raw_p"], 1.)
        self.assertNotEqual(result["tests"]["mz_hac"]["status"], "ok")
        self.assertFalse(result["nonrejection_certifies_calibration"])
        json.dumps(result, allow_nan=False)

    def test_gap_has_absolute_and_relative_cap_and_no_loss_substitution(self):
        matrices = {p: _matrix(self.rows["test"], p, self.seeds) for p in self.protocol["panel_ids"]}
        result = _adequacy(self.rows, matrices, self.calibrators, self.config, self.protocol["statistical_settings"])
        for row in result["candidate_by_panel"]:
            self.assertLess(row["gap_limit_rmse"], .00025772842664981106)
            self.assertTrue(row["gap_below_margin"])
        missing = dict(self.rows); missing.pop("fit")
        result = _adequacy(missing, matrices, None, self.config, self.protocol["statistical_settings"])
        self.assertTrue(all(r["selection_minus_fit_rmse"] is None and not r["gap_below_margin"] for r in result["candidate_by_panel"]))
        zero = dict(self.rows, fit=deepcopy(self.rows["fit"]))
        for row in zero["fit"]:
            if row["system"] == CANDIDATE:
                row["prediction"] = row["y"]
        result = _adequacy(zero, matrices, None, self.config, self.protocol["statistical_settings"])
        self.assertTrue(all(r["gap_limit_rmse"] is None and not r["gap_below_margin"] for r in result["candidate_by_panel"]))

    def test_thresholds_need_bound_configuration_not_human_permission(self):
        value = _threshold_admission(self.config, {"sha256": digest(self.config)}, self.protocol)
        self.assertTrue(value["passed"])
        changed = deepcopy(self.protocol); changed["adequacy"]["config_sha256"] = "changed"
        self.assertFalse(_threshold_admission(self.config, {"sha256": digest(self.config)}, changed)["passed"])

    def test_actual_failed_finite_sample_evidence_never_admitted(self):
        matrices = {p: _matrix(self.rows["test"], p, self.seeds) for p in self.protocol["panel_ids"]}
        path = ROOT/"qa/statistical_validation/verdict_v2.json"
        report = _method_admission({"path": str(path), "sha256": sha(path)}, matrices, self.protocol["statistical_settings"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["evidence"]["status"], "FAILED_FINITE_SAMPLE_VALIDATION")

    def test_passing_method_claim_without_archived_replicates_is_rejected(self):
        """A plausible JSON certificate is not executed calibration evidence."""
        matrices = {p: _matrix(self.rows["test"], p, self.seeds) for p in self.protocol["panel_ids"]}
        settings = self.protocol["statistical_settings"]
        scope = {"R": 10, "T": 72, "B": 99, "block_length": 3, "hac_lags": 2,
            "horizon": 1, "margin": .02, "reference_reps": 999, "reference_seed": 2026112103}
        claim = {"status": "PASSED_PREDECLARED_VALIDATION", "admitted": True,
            "protocol_closure_permitted": True, "family_rivals": 20, "panels": 4,
            "all_three_methods": True, "complete_and_partial_nulls": True,
            "extreme_tail_validated": True, "coverage_validated": True,
            "statistics_source_sha256": sha(ROOT/"engine/statistics.py"),
            "validated_settings": [scope]}
        # Include any new optional settings exposed by the statistics API.
        import inspect
        from engine.statistics import analyze
        for key in ("alpha", "reference_reps", "reference_seed"):
            if key in inspect.signature(analyze).parameters:
                scope[key] = settings.get(key, inspect.signature(analyze).parameters[key].default)
        if "alpha" in scope:
            claim["operational_alpha"] = scope["alpha"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"false_claim.json"
            write_json(path, claim)
            result = _method_admission({"path": str(path), "sha256": sha(path)}, matrices, settings)
        self.assertFalse(result["passed"], "Bare passing claims must not bypass the real archived Monte Carlo ledger")

    def test_passing_power_claim_without_counters_and_planning_artifacts_is_rejected(self):
        cell = {"R": 10, "T": 72, "B": 99}
        fixed = {"cell": cell}
        contrasts = [{"panel": p, "rival": r, "simultaneous_mc_lower_bound": .95}
            for p in self.protocol["panel_ids"] for r in METHODS[1:]]
        claim = {"status": "DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP", "temporal_calibration_verified": True,
            **{"final_"+k: v for k, v in cell.items()}, "fixed_before_validation": fixed,
            "fixed_before_validation_sha256": digest(fixed),
            "validation_result": {"cell": cell, "meets_power_target": True,
                "minimum_per_contrast_simultaneous_mc_lower_bound": .95,
                "scenarios": [{"contrasts": contrasts, "joint_statistical_rule_power": .95}]}}
        result = _power_evidence(claim, cell, self.protocol["panel_ids"])
        self.assertFalse(result["passed"], "Asserted lower bounds need actual counter and planning evidence")

    def test_current_power_format_recounts_real_counters_after_separate_method_stub(self):
        """Only the orthogonal method certificate is stubbed, not CP validation.

        These manufactured inputs test contract wiring; they are not a real
        development-power study and cannot admit a final campaign.
        """
        from scipy.stats import beta
        from engine.controller import default_stress_plans
        design = read_json(ROOT/"configs/power_design.json")
        cell = {"R": 10, "T": 768, "B": 8191}
        trials = design["mc_validation_reps"]
        alternatives = design["alternatives_relative_rmse_gain"]
        lower = float(beta.ppf(design["mc_confidence_error"]/(2*80*len(alternatives)), trials, 1))
        self.assertGreater(lower, .9)
        fixed = {"cell": cell}
        rule = "finite-B-count-plus-one-actual-Holm-v1"
        scenarios = [{"alternative_relative_rmse_gain": gain, "joint_statistical_rule_power": 1.,
            "contrasts": [{"panel": p, "rival": r, "successes": trials, "trials": trials,
                "simultaneous_mc_lower_bound": lower} for p in self.protocol["panel_ids"] for r in METHODS[1:]]}
            for gain in alternatives]
        payload = {"status": "DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP", "temporal_calibration_verified": True,
            **{"final_"+k: v for k, v in cell.items()}, "fixed_before_validation": fixed,
            "fixed_before_validation_sha256": digest(fixed), "operational_rule": rule,
            "methods_sha256": sha(ROOT/"engine/statistics.py"), "design_sha256": digest(design),
            "validation_result": {"cell": cell, "meets_power_target": True, "operational_rule": rule,
                "minimum_per_contrast_simultaneous_mc_lower_bound": lower, "scenarios": scenarios}}
        protocol = deepcopy(self.protocol)
        protocol["robustness_stress_plans"] = default_stress_plans()
        protocol["adequacy"] = {"config_path": "configs/adequacy.json", "config_sha256": sha(ROOT/"configs/adequacy.json")}
        protocol["statistical_settings"] = {"bootstrap_reps": cell["B"], "block_length": design["inner_block_length"],
            "hac_lags": design["hac_lags"], "horizon": design["horizon"], "margin": design["margin_relative_rmse"],
            **{k: design[k] for k in ("reference_reps", "reference_seed", "alpha")}}
        protocol["final"].update(cell, closed=True,
            test_panels={p: {"count": cell["T"], "start": "2032-01-01", "end": "2035-01-01"} for p in self.protocol["panel_ids"]})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture = directory/"planning_fixture.json"
            write_json(fixture, {"software_fixture": True, "market_predictions": 0})
            payload["planning_evidence"] = {"verified_files": [{"path": str(fixture), "sha256": sha(fixture)}]}
            path = directory/"power_fixture.json"
            write_json(path, payload)
            spec = {"path": str(path), "sha256": sha(path)}
            protocol["final"]["power_receipt"] = spec
            with patch("engine.controller._method_admission", return_value={"software_fixture_only": True}):
                result = _power_evidence(spec, protocol["final"], self.protocol["panel_ids"],
                    protocol=protocol, protocol_evidence={"sha256": digest(protocol)})
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["verified_final_admission"]["R"], 10)
            # Keep every asserted bound favorable while falsifying the raw
            # count. The unmodified controller must reject it on CP recount.
            payload["validation_result"]["scenarios"][0]["contrasts"][0]["successes"] = 0
            write_json(path, payload)
            spec = {"path": str(path), "sha256": sha(path)}
            protocol["final"]["power_receipt"] = spec
            with patch("engine.controller._method_admission", return_value={"software_fixture_only": True}):
                result = _power_evidence(spec, protocol["final"], self.protocol["panel_ids"],
                    protocol=protocol, protocol_evidence={"sha256": digest(protocol)})
            self.assertFalse(result["passed"])
            self.assertEqual(result["reason"], "power_raw_counter_does_not_establish_point90")

    def test_between_panel_tests_are_separate_and_finite(self):
        result = between_panel_tests([analysis_fixture() for _ in range(4)])
        self.assertEqual(len(result["wilcoxon_signed_rank_sign_enumeration_holm"]), 20)
        self.assertTrue(all(r["raw_p"] >= .125 for r in result["wilcoxon_signed_rank_sign_enumeration_holm"]))
        self.assertFalse(result["independent_panel_reference_established"])
        json.dumps(result, allow_nan=False)

    def test_saved_stress_requires_exact_frozen_plan_and_unchanged_checkpoint(self):
        matrices = {p: _matrix(self.rows["test"], p, self.seeds) for p in self.protocol["panel_ids"]}
        cfg = deepcopy(self.config)
        cfg["robustness"].update(temporal_alternatives=[], seed_subsets=[], temporal_blocks=[], required_saved_stress_scenarios=["input_noise"])
        protocol = deepcopy(self.protocol)
        plan = {"frozen_before_test": True, "seed": 191, "sigma": .01}
        plan["scenario_sha256"] = digest(plan)
        protocol["robustness_stress_plans"] = {"input_noise": plan}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"saved_stress.csv"
            write_csv(path, self.rows["test"])
            saved = {"path": str(path), "sha256": sha(path), "scenario_sha256": plan["scenario_sha256"],
                "selected_manifest_sha256": "selected", "eval_id": "fixture-eval", "protocol_sha256": digest(protocol),
                "no_refit": True, "completed": True}
            receipt = {"status": "completed", "stress_prediction_files": {"input_noise": saved},
                "stress_plans_sha256": digest(protocol["robustness_stress_plans"]),
                "selected_manifest_sha256": "selected", "eval_id": "fixture-eval", "protocol_sha256": digest(protocol)}
            with patch("engine.assessment._analyze_panel", return_value=analysis_fixture()):
                result = _robustness(matrices, cfg, self.protocol["statistical_settings"], {"input_noise": {"path": str(path), "sha256": sha(path)}}, protocol, receipt)
            self.assertEqual(result["saved_stress_scenarios"][0]["status"], "executed_on_saved_predictions")
            self.assertTrue(result["saved_stress_scenarios"][0]["passed"])
            saved["no_refit"] = False
            result = _robustness(matrices, cfg, self.protocol["statistical_settings"], {"input_noise": {"path": str(path), "sha256": sha(path)}}, protocol, receipt)
            self.assertEqual(result["saved_stress_scenarios"][0]["status"], "invalid_saved_stress_evidence")

    def test_seal_checks_saved_artifact_bytes_and_deterministic_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            for split, rows in self.rows.items():
                write_csv(case/"predictions"/(split+"_predictions.csv"), rows)
            write_json(case/"metrics/calibrators.json", self.calibrators)
            keys = sorted({(r["panel"], r["system"], r["seed"]) for r in self.rows["test"]})
            selected = {"protocol_sha256": digest(self.protocol), "jobs": [{"panel": p, "model": m,
                "seed": int(self.seeds[0] if s == "deterministic" else s)} for p, m, s in keys]}
            write_json(case/"selected_manifest.json", selected)
            selected_sha = sha(case/"selected_manifest.json")
            write_json(case/"training_seal.json", {"protocol_sha256": digest(self.protocol),
                "selected_manifest_sha256": selected_sha, "calibrators_sha256": sha(case/"metrics/calibrators.json"), "test_opened": False})
            for split in ("fit", "selection", "calibration"):
                write_json(case/"receipts"/(split+"_prediction_identity.json"), {
                    "sha256": sha(case/"predictions"/(split+"_predictions.csv")), "selected_manifest_sha256": selected_sha})
            write_json(case/"receipts/final_admissions.json", {"fixture": True})
            identity = {"case_id": "fixture", "plan_sha256": "fixture-plan", "protocol_sha256": digest(self.protocol),
                "selected_manifest_sha256": selected_sha, "calibrators_file_sha256": sha(case/"metrics/calibrators.json"),
                "calibrators_sha256": self.calibrators["sha256"], "calibrators_frozen_before_test": True,
                "selected_checkpoints_verified_before_test": True, "final_admissions_file_sha256": sha(case/"receipts/final_admissions.json"),
                "stress_plans_sha256": digest(self.protocol.get("robustness_stress_plans", {})),
                "panels": {p: {} for p in self.protocol["panel_ids"]}}
            receipt = identity | {"eval_id": digest(identity), "status": "completed", "sealed": True,
                "predictions_sha256": sha(case/"predictions/test_predictions.csv")}
            checked = _sealed_evaluation(case, receipt, {"sha256": digest(self.protocol)}, self.calibrators, self.rows["test"])
            self.assertTrue(checked["artifact_bindings_passed"], checked)
            self.assertFalse(checked["passed"])
            self.assertFalse(checked["final_data_admission_passed"])
            with (case/"predictions/fit_predictions.csv").open("a") as handle:
                handle.write("\n")
            checked = _sealed_evaluation(case, receipt, {"sha256": digest(self.protocol)}, self.calibrators, self.rows["test"])
            self.assertFalse(checked["passed"])

    def test_assessment_consumes_saved_only_is_deterministic_and_false_actual_method(self):
        # Full real temporal procedures, all21 systems, four panels, two seed
        # subsets, two temporal blocks and MAE. Never calls a forecasting model.
        with tempfile.TemporaryDirectory() as tmp:
            report = assess_case(tmp, self.protocol, predictions=self.rows,
                calibrators=self.calibrators, adequacy_config=self.config)
            second = assess_case(tmp, self.protocol, predictions=self.rows,
                calibrators=self.calibrators, adequacy_config=self.config)
        self.assertEqual(digest(report), digest(second))
        self.assertFalse(report["r3_scientific_criteria_satisfied"])
        self.assertFalse(report["criteria"]["method_calibration_admitted"])
        self.assertEqual(report["new_model_forwards"], 0)
        self.assertEqual(len(report["panel_statistics"]), 4)
        self.assertEqual(report["joint_rule"]["additional_inferiority_family"]["size"], 180)
        self.assertEqual(len(report["robustness"]["sensitivities"]), 5)
        self.assertTrue(all(r["status"].startswith("unexecuted") for r in report["robustness"]["saved_stress_scenarios"]))
        self.assertFalse(report["institutional_approval_claimed"])
        json.dumps(report, allow_nan=False)

    def test_tampered_calibrator_and_partition_reuse_are_refused(self):
        cal = deepcopy(self.calibrators); cal["calibrators"][0]["q"] = .3
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "checksum"):
                assess_case(tmp, self.protocol, predictions=self.rows, calibrators=cal, adequacy_config=self.config)
            overlap = dict(self.rows, selection=self.rows["fit"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                assess_case(tmp, self.protocol, predictions=overlap, adequacy_config=self.config)

    def test_pipeline_flag_can_be_true_with_every_scientific_criterion_met(self):
        # Isolate conjunction wiring. The scientific procedures themselves are
        # exercised above; these explicit stubs are not validation evidence.
        protocol = deepcopy(self.protocol); protocol["final"]["closed"] = True
        receipt = {"status": "completed", "evaluation_sealed": True,
            "selected_checkpoints_verified_before_test": True, "calibrators_frozen_before_test": True,
            "calibrators_sha256": self.calibrators["sha256"], "protocol_sha256": digest(protocol)}
        with patch("engine.assessment._analyze_panel", return_value=analysis_fixture()), \
             patch("engine.assessment._adequacy", return_value={"passed": True}), \
             patch("engine.assessment._robustness", return_value={"passed": True}), \
             patch("engine.assessment._method_admission", return_value={"passed": True}), \
             patch("engine.assessment._power_evidence", return_value={"passed": True}), \
             patch("engine.assessment._sealed_evaluation", return_value={"passed": True}), \
             tempfile.TemporaryDirectory() as tmp:
            report = assess_case(tmp, protocol, predictions=self.rows, calibrators=self.calibrators,
                adequacy_config=self.config, evaluation_receipt=receipt)
        self.assertTrue(report["r3_scientific_criteria_satisfied"])
        self.assertFalse(report["institutional_approval_claimed"])


if __name__ == "__main__":
    unittest.main()
