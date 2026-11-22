"""Software/finite synthetic checks; these fixtures are not market planning."""
from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from engine import power
from engine.statistics import analyze


ROOT = Path(__file__).resolve().parents[1]


def design():
    return json.loads((ROOT / "configs" / "power_design.json").read_text())


def planning_fixture(panels=2, systems=3, seeds=5, dates=36):
    rng = np.random.default_rng(20261121)
    common = rng.normal(0, .1, dates)
    shared_seed = rng.normal(0, .05, seeds)
    losses = np.exp(rng.normal(0, .08, (panels, systems, seeds, dates))
                    + common[None, None, None, :] + shared_seed[None, None, :, None])
    y = np.full((panels, dates), 2.)
    z = np.ones((panels, dates, 2))
    z[:, :, 1] = np.linspace(.3, 1.1, dates)
    return {"losses": losses, "y": y, "instruments": z,
            "systems": ["input_only_mse"] + [f"rival_{i}" for i in range(1, systems)],
            "panels": [f"fixture_panel_{i}" for i in range(panels)],
            "planning_seed_ids": [str(x) for x in range(seeds)], "dates": list(range(dates)),
            "fit_receipts": [], "evidence": {"kind": "UNIT_FIXTURE_NOT_CALIBRATION_OR_PLANNING"}}


def result_fixture(rivals=1, raw=.001, negative=False):
    comparisons = []
    for rival in range(rivals):
        comparisons.append({"rival": f"rival_{rival}", "mean_mse_difference": -1. if negative else 1.,
                            "practical_mean": 1., "temporal_effect_cohen_d": .4,
                            "ci95_mse_difference": [.1, 1.5], "ci95_relative_rmse_gain": [.025, .06],
                            "methods": {m: {n: {"status": "ok", "raw_p": raw, "adjusted_p": raw}
                                                   for n in ("epa", "practical")} for m in power.METHODS}})
    return {"comparisons": comparisons, "candidate": "input_only_mse",
            "metrics": {"input_only_mse": {"mean_seed_rmse": 1.},
                        **{f"rival_{i}": {"mean_seed_rmse": 1.1} for i in range(rivals)}}}


def write_calendar_fixture(root, config):
    """Full-family schema fixture explicitly barred from final dimensioning."""
    root = Path(root)
    metadata = root / "fixture_state.json"
    metadata.write_text('{"kind":"software_fixture_no_training"}')
    ref = {"path": metadata.name, "sha256": power.sha256(metadata)}
    csv_path = root / "predictions.csv"
    jobs = []
    seeds = [str(x) for x in power.planning_seed_ids(config, 5)]
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["panel", "system", "seed", "date", "y", "prediction", "known_floor_A", "rfit"])
        for pi, panel in enumerate(config["panels"]):
            start = dt.date(2020 if pi == 0 else 2000, 1, 1)
            for si, system in enumerate(config["systems"]):
                for ri, seed in enumerate(seeds if system["kind"] == "neural" else ["deterministic"]):
                    jobs.append({"panel": panel, "system": system["id"], "seed": seed,
                                 "phase": "planning", "success": True, "active_seconds": 0., "checkpoint": ref})
                    for ti in range(80):
                        writer.writerow([panel, system["id"], seed, (start+dt.timedelta(days=ti)).isoformat(),
                                         10., 11+si*.01+ri*.001+ti*.0001, 1+ti*.01, 2.])
    receipt = {"kind": "development_planning_predictions", "partition": "planning",
               "final_test_opened": False, "synthetic": True, "is_fixture": True,
               "predictions_sha256": power.sha256(csv_path), "seed_bank_sha256": config["seed_bank"]["sha256"],
               "protocol_sha256": ref["sha256"], "methods_sha256": ref["sha256"],
               **{key: ref for key in ("protocol_file", "methods_file", "source_manifest_file", "planning_data_file")},
               "planning_partition_purged": True, "instruments_available_at_origin": True, "fit_receipts": jobs}
    csv_path.with_suffix(".receipt.json").write_text(json.dumps(receipt))
    return csv_path


class PowerContracts(unittest.TestCase):
    def test_exact_reference_budget_and_actual_receipts_are_distinct(self):
        result = power.budget(design(), R=10, T=252,
                              fit_receipts=[{"active_seconds": 3.5, "success": False},
                                            {"active_seconds": 7., "success": True}])
        self.assertEqual(result["neural_fits"], 640)
        self.assertEqual(result["arima_fits"], 4)
        self.assertEqual(result["analytic_trajectories"], 16)
        self.assertEqual(result["nominal_neural_active_seconds"], 76800)
        self.assertEqual(result["measured_planning_active_seconds"], 10.5)
        self.assertEqual(result["deterministic_independent_seed_replicates"], 0)
        self.assertIsNone(result["nominal_compute_money"])

    def test_bank_prefix_is_exact_and_checksum_bound(self):
        config = design()
        bank_path = Path(config["seed_bank"]["path"])
        if not bank_path.is_absolute():
            bank_path = ROOT / bank_path
        bank = json.loads(bank_path.read_text())["seed_ids"]
        self.assertEqual(power.seed_prefix(config, 10), bank[:10])
        self.assertEqual(power.planning_seed_ids(config, 5), bank[-5:])
        self.assertFalse(set(power.seed_prefix(config, 4091)) & set(power.planning_seed_ids(config, 5)))
        with self.assertRaises(ValueError):
            power.seed_prefix(config, 4092)
        config["seed_bank"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            power.seed_prefix(config, 10)

    def test_resolution_and_precision_do_not_claim_power(self):
        result = power.resampling_budget()
        self.assertEqual(result["positive_slots_per_panel"], 120)
        self.assertEqual(result["additional_adverse_slots"], 180)
        self.assertEqual(result["minimum_B_for_two_sided_resolution"], 3599)
        self.assertEqual(result["B_for_requested_expected_tail_exceedances_diagnostic_only"], 360000)
        legacy = power.resampling_budget(mc_p_rule="doubled_smaller_tail_plus_one")
        self.assertEqual(legacy["minimum_B_for_two_sided_resolution"], 7199)
        self.assertNotIn("power", result)

    def test_missing_planning_never_makes_up_final_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "missing.json"
            result = power.dimension([], design(), output=output)
            self.assertEqual(result["status"], "NOT_DIMENSIONED")
            for key in ("final_R", "final_T", "final_B", "final_seed_ids"):
                self.assertIsNone(result[key])
            self.assertFalse(result["r3_accredited"])
            self.assertEqual(json.loads(output.read_text())["status"], "NOT_DIMENSIONED")

    def test_synthetic_receipt_is_rejected_even_for_full_family(self):
        config = design()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "fixture.csv"
            csv_path.write_text("panel,system,seed,date,y,prediction,known_floor_A,rfit\n")
            receipt_path = csv_path.with_suffix(".receipt.json")
            receipt_path.write_text(json.dumps({"kind": "development_planning_predictions", "partition": "planning",
                                                "final_test_opened": False, "synthetic": True,
                                                "predictions_sha256": power.sha256(csv_path)}))
            result = power.dimension([csv_path], config)
            self.assertEqual(result["status"], "NOT_DIMENSIONED")
            self.assertIn("synthetic", result["reasons"][0])
            self.assertNotIn("selection_results", result)

    def test_partial_family_cannot_qualify_for_dimensioning(self):
        config = design()
        config["systems"] = config["systems"][:2]
        with self.assertRaisesRegex(ValueError, "complete 21-system"):
            power.load_planning(["fixture_that_is_never_read.csv"], config)

    def test_source_file_modification_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text("one")
            entry = {"path": "source.json", "sha256": power.sha256(path)}
            self.assertEqual(power._verify_file(entry, Path(tmp))["sha256"], entry["sha256"])
            path.write_text("two")
            with self.assertRaisesRegex(ValueError, "checksum"):
                power._verify_file(entry, Path(tmp))

    def test_alternative_targets_rmse_transform_without_negative_losses(self):
        planning = planning_fixture()
        alternative, factors = power.alternative_losses(planning["losses"], .03)
        means = alternative.mean(axis=(2, 3))
        np.testing.assert_allclose(1-np.sqrt(means[:, :1]/means[:, 1:]), .03, atol=1e-14)
        np.testing.assert_array_equal(alternative[:, 0], planning["losses"][:, 0])
        self.assertTrue(np.all(alternative >= 0))
        # Fixed scaling preserves the whole surface, rather than standardizing
        # each future sample so that it wins by construction.
        expected = np.broadcast_to(np.asarray(factors)[:, 1, None, None], alternative[:, 1].shape)
        np.testing.assert_allclose(alternative[:, 1] / planning["losses"][:, 1], expected)

    def test_crossed_resampling_preserves_pairing_and_deterministic_trace(self):
        p = planning_fixture(systems=3)
        p["losses"][:, 1] = 2 * p["losses"][:, 0]
        p["losses"][:, 2] = p["losses"][:, 2, :1]
        pred, target, z = power.crossed_draw(p, p["losses"], R=10, T=100,
                                           block_length=6, rng=np.random.default_rng(42))
        squared = (pred-target[:, None, None, :])**2
        np.testing.assert_allclose(squared[:, 1], 2*squared[:, 0])
        np.testing.assert_allclose(squared[:, 2], squared[:, 2, :1].repeat(10, axis=1))
        self.assertEqual(z.shape, (2, 100, 2))
        self.assertEqual(pred.shape, (2, 3, 10, 100))
        # All panels share exact calendar indices, detectable through instrument.
        np.testing.assert_array_equal(z[0], z[1])

    def test_joint_covariance_records_seed_and_calendar_variation(self):
        p = planning_fixture()
        summary = power.dependence_summary(p)
        self.assertEqual(np.asarray(summary["calendar_covariance_of_seed_mean_loss_differences"]).shape, (4, 4))
        self.assertEqual(np.asarray(summary["training_seed_covariance_of_calendar_mean_loss_differences"]).shape, (4, 4))
        self.assertFalse(summary["stationarity_verified"])

    def test_statistical_conjunction_and_additional_adverse_screen(self):
        good = [result_fixture() for _ in range(4)]
        self.assertTrue(power.statistical_rule(good)["joint_statistical_rule_pass"])
        bad = copy.deepcopy(good)
        bad[0]["comparisons"][0]["methods"]["gw_conditional"]["practical"]["adjusted_p"] = .5
        self.assertFalse(power.statistical_rule(bad)["joint_statistical_rule_pass"])
        bad = [good[0], result_fixture(raw=1e-9, negative=True), good[2], good[3]]
        result = power.statistical_rule(bad)
        self.assertTrue(result["additional_significant_inferiority"])
        self.assertFalse(result["joint_statistical_rule_pass"])

    def test_tail_guard_never_overstates_precision_or_changes_raw_p(self):
        result = result_fixture(raw=2/8192)
        guarded = power.conservative_bootstrap_guard(result, B=8191, family_error=.01)
        before = result["comparisons"][0]["methods"]["stationary_bootstrap"]["epa"]
        after = guarded["comparisons"][0]["methods"]["stationary_bootstrap"]["epa"]
        self.assertEqual(before["raw_p"], after["raw_p"])
        self.assertGreater(after["mc_p_upper"], before["raw_p"])
        self.assertEqual(after["inferred_minimum_tail_exceedances"], 0)
        self.assertNotIn("mc_p_upper", before)

    def test_historical_mean_rmse_guard_does_not_infer_its_order_from_pooled_mse(self):
        good = [result_fixture() for _ in range(4)]
        # Candidate seed RMSE=[1.2,1.2] has lower pooled risk than rival
        # [0,2], yet its historical mean RMSE is larger (1.2 > 1.0).
        good[0]["metrics"]["input_only_mse"]["mean_seed_rmse"] = 1.2
        good[0]["metrics"]["rival_0"]["mean_seed_rmse"] = 1.
        rule = power.statistical_rule(good)
        self.assertFalse(rule["historical_main_all_mean_rmse_favorable"])
        self.assertTrue(rule["historical_additional_mean_rmse_two_of_three"])
        self.assertFalse(rule["contrast_pass"][0, 0])
        self.assertFalse(rule["joint_statistical_rule_pass"])
        good = [result_fixture() for _ in range(4)]
        for pi in (1, 2):
            good[pi]["metrics"]["rival_0"]["mean_seed_rmse"] = .9
        rule = power.statistical_rule(good)
        self.assertFalse(rule["historical_additional_mean_rmse_two_of_three"])
        self.assertFalse(rule["joint_statistical_rule_pass"])

    def test_small_actual_monte_carlo_calls_the_real_battery(self):
        p = planning_fixture(panels=1, systems=2, dates=36)
        config = design()
        # Exercise legacy/current statistical APIs with their own fixed alpha;
        # explicit-family alpha propagation has a separate deterministic test.
        config.pop("alpha", None)
        config.update(alternatives_relative_rmse_gain=[.03], inner_block_length=4, outer_block_length=4, hac_lags=3, reference_reps=999)
        result = power.estimate_power(p, config, {"R": 10, "T": 36, "B": 99}, reps=2, seed=81, analyzer=analyze)
        self.assertEqual(result["mc_reps_per_alternative"], 2)
        self.assertEqual(len(result["scenarios"][0]["contrasts"]), 1)
        self.assertFalse(result["meets_power_target"])
        self.assertLess(result["minimum_per_contrast_simultaneous_mc_lower_bound"], .9)

    def test_disjoint_calendars_are_accepted_without_inventing_joint_shocks(self):
        config = design()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_calendar_fixture(tmp, config)
            p = power.load_planning([path], config, allow_synthetic_fixture=True)
            self.assertEqual(p["calendar_groups"], [[0], [1, 2, 3]])
            self.assertFalse(set(p["panel_dates"]["F00"]) & set(p["panel_dates"]["F01"]))
            summary = power.dependence_summary(p)
            covariance = summary["calendar_covariance_of_seed_mean_loss_differences"]
            self.assertIsNone(covariance[0][20])
            self.assertIsNotNone(covariance[20][40])
            _, _, instruments = power.crossed_draw(p, p["losses"], R=10, T=80, block_length=4,
                                                   rng=np.random.default_rng(8201))
            np.testing.assert_array_equal(instruments[1], instruments[2])
            self.assertFalse(np.array_equal(instruments[0], instruments[1]))
            self.assertEqual(power.dimension([path], config)["status"], "NOT_DIMENSIONED")

    def test_frechet_all_positive_upper_is_not_used_as_full_rule_upper(self):
        contrasts = [{"power": .9, "simultaneous_mc_interval": [.8, .95],
                      "tail_guard_error_adjusted_mc_lower_bound": .79} for _ in range(2)]
        bounds = power.joint_power_bounds(contrasts)
        self.assertAlmostEqual(bounds["all_positive_contrast_intersection"]["frechet_union_lower"], .8)
        self.assertAlmostEqual(bounds["all_positive_contrast_intersection"]["frechet_upper"], .9)
        self.assertAlmostEqual(bounds["full_statistical_rule"]["lower_from_all_positive_subset"], .8)
        self.assertEqual(bounds["full_statistical_rule"]["upper"], 1.)

    def test_v3_tail_guard_uses_actual_absolute_tail_counts_for_all_methods(self):
        result = result_fixture()
        for method in power.METHODS:
            for null in ("epa", "practical"):
                test = result["comparisons"][0]["methods"][method][null]
                test.update(raw_p=1/1000, mc_tail_count=0, mc_reps=999, mc_p_rule="absolute_tail_plus_one")
        guarded = power.conservative_bootstrap_guard(result, B=8191, family_error=.01)
        for method in power.METHODS:
            test = guarded["comparisons"][0]["methods"][method]["epa"]
            self.assertGreater(test["mc_p_upper"], test["raw_p"])
            self.assertEqual(test["mc_reps"], 999)
        bad = copy.deepcopy(result)
        bad["comparisons"][0]["methods"]["dm_hln_hac"]["epa"]["mc_tail_count"] = 9
        with self.assertRaisesRegex(ValueError, "tail count"):
            power.conservative_bootstrap_guard(bad, B=8191, family_error=.01)

    def test_gw_unsigned_rejection_alone_does_not_certify_mean_inferiority(self):
        panels = [result_fixture() for _ in range(4)]
        extra = panels[1]["comparisons"][0]
        extra["mean_mse_difference"] = -1e-14
        for method in power.METHODS:
            extra["methods"][method]["epa"]["raw_p"] = 1e-10 if method == "gw_conditional" else 1.
            extra["methods"][method]["epa"]["adjusted_p"] = 1e-10 if method == "gw_conditional" else 1.
        rule = power.statistical_rule(panels)
        self.assertFalse(rule["additional_significant_inferiority"])
        self.assertTrue(rule["additional_gw_conditional_rejection_with_negative_sample_mean"])
        self.assertTrue(rule["joint_statistical_rule_pass"])

    def test_uncertain_reserved_seconds_are_charged_not_called_measured(self):
        b = power.budget(design(), R=10, T=252, fit_receipts=[
            {"active_seconds": 120., "compute_uncertain": True, "active_seconds_is_charge_not_exact_elapsed": True},
            {"active_seconds": 5., "compute_uncertain": False},
            {"active_seconds": 0., "aborted_seconds_observed": 2.}])
        self.assertEqual(b["planning_active_seconds_charged_total"], 125.)
        self.assertEqual(b["measured_planning_active_seconds"], 5.)
        self.assertEqual(b["planning_cost_uncertain_attempts"], 1)
        self.assertEqual(b["planning_observed_interruption_partial_seconds"], 2.)

    def test_statistical_conjunction_obeys_prespecified_family_alpha(self):
        legacy = [result_fixture(raw=.03) for _ in range(4)]
        self.assertTrue(power.statistical_rule(legacy)["joint_statistical_rule_pass"])
        operational = copy.deepcopy(legacy)
        for panel in operational:
            panel["family"] = {"alpha": .025}
        self.assertFalse(power.statistical_rule(operational)["joint_statistical_rule_pass"])
        self.assertEqual(power.statistical_rule(operational)["family_alpha"], .025)
        operational[-1]["family"]["alpha"] = .05
        with self.assertRaisesRegex(ValueError, "same prespecified"):
            power.statistical_rule(operational)

    def test_predeclared_wall_cap_preserves_partial_inconclusive_study(self):
        config = design()
        result = power.estimate_power(planning_fixture(), config, {"R": 10, "T": 36, "B": 99},
                                      reps=2, seed=81, analyzer=lambda *a, **k: self.fail("must not run"),
                                      deadline=time.monotonic()-1)
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["completed_reps_current_alternative"], 0)
        self.assertFalse(result["meets_power_target"])

    def test_fixed_winner_validation_failure_is_retained_without_reselection(self):
        config = design()
        config["grid"] = {"R": [10, 20], "T": [252], "B": [8191]}
        p = planning_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "study.json"
            calls = []
            def estimate(*args, **kwargs):
                calls.append((dict(args[2]), kwargs["seed"]))
                if len(calls) == 2:
                    self.assertTrue(Path(str(output)+".winner_before_validation.json").is_file())
                return {"cell": dict(args[2]), "meets_power_target": len(calls) == 1}
            with patch.object(power, "load_planning", return_value=p), patch.object(power, "estimate_power", side_effect=estimate):
                result = power.dimension([], config, output=output)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][0], calls[1][0])
            self.assertNotEqual(calls[0][1], calls[1][1])
            self.assertEqual(result["status"], "NOT_DIMENSIONED")
            self.assertIsNone(result["final_R"])
            self.assertIn("fixed winner failed", " ".join(result["reasons"]))
            # A subsequent identical call returns the recorded adverse result.
            with patch.object(power, "load_planning", return_value=p), patch.object(power, "estimate_power", side_effect=AssertionError("repeat forbidden")):
                replay = power.dimension([], config, output=output)
            self.assertEqual(replay["fixed_before_validation_sha256"], result["fixed_before_validation_sha256"])

    def test_full_family_validation_cannot_be_claimed_from_fixture_or_failed_report(self):
        config = design()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed.json"
            path.write_text(json.dumps({"family_rivals": 20, "panels": 4, "all_three_methods": True,
                                        "complete_and_partial_nulls": True, "extreme_tail_validated": False,
                                        "coverage_validated": False, "status": "FAILED"}))
            config["temporal_validation_report"] = {"path": str(path), "sha256": power.sha256(path)}
            approved, reason = power._calibration(config, analyze)
            self.assertFalse(approved)
            self.assertIn("does not establish", reason)

    def test_bare_success_booleans_cannot_replace_validation_ledger(self):
        config = design()
        cell = {"R": 10, "T": 768, "B": 8191}
        settings = {**cell, "block_length": config["inner_block_length"],
                    "hac_lags": config["hac_lags"], "horizon": config["horizon"],
                    "margin": config["margin_relative_rmse"]}
        settings.update({k: config[k] for k in ("reference_reps", "reference_seed", "alpha") if k in config})
        report = {"status": "PASSED_PREDECLARED_VALIDATION", "admitted": True,
                  "protocol_closure_permitted": True, "family_rivals": 20,
                  "all_three_methods": True, "complete_and_partial_nulls": True,
                  "extreme_tail_validated": True, "coverage_validated": True,
                  "statistics_source_sha256": power.sha256(ROOT/"engine/statistics.py"),
                  "validated_settings": [settings]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"unsupported_success.json"
            path.write_text(json.dumps(report))
            config["temporal_validation_report"] = {"path": str(path), "sha256": power.sha256(path)}
            approved, reason = power._calibration(config, analyze, cell)
            self.assertFalse(approved)
            self.assertIn("evidence bundle rejected", reason)

    def test_validation_bundle_receives_exact_source_settings_and_family_alpha(self):
        config = design()
        config["alpha"] = .025
        cell = {"R": 10, "T": 768, "B": 8191}
        settings = {**cell, "block_length": config["inner_block_length"],
                    "hac_lags": config["hac_lags"], "horizon": config["horizon"],
                    "margin": config["margin_relative_rmse"], "reference_reps": config["reference_reps"],
                    "reference_seed": config["reference_seed"], "alpha": .025}
        report = {"status": "PASSED_PREDECLARED_VALIDATION", "family_rivals": 20,
                  "all_three_methods": True, "complete_and_partial_nulls": True,
                  "extreme_tail_validated": True, "coverage_validated": True,
                  "statistics_source_sha256": power.sha256(ROOT/"engine/statistics.py"),
                  "validated_settings": [settings]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"schema_fixture_only.json"
            path.write_text(json.dumps(report))
            spec = {"path": str(path), "sha256": power.sha256(path)}
            config["temporal_validation_report"] = spec
            # Only isolates the caller contract. Real admission rejects this
            # schema fixture, as the preceding test checks without a mock.
            with patch("engine.controller._method_admission", return_value={}) as validator:
                approved, reason = power._calibration(config, analyze, cell)
            self.assertTrue(approved)
            validator.assert_called_once_with(ROOT, spec, settings)
            report["validated_settings"][0]["alpha"] = .05
            path.write_text(json.dumps(report))
            config["temporal_validation_report"]["sha256"] = power.sha256(path)
            approved, reason = power._calibration(config, analyze, cell)
            self.assertFalse(approved)
            self.assertIn("exact selected", reason)

    def test_actual_finite_B_rule_can_pass_while_tail_precision_sensitivity_fails(self):
        config = design()
        config["alternatives_relative_rmse_gain"] = [.03]
        p = planning_fixture(panels=4, systems=21)
        p["systems"] = ["input_only_mse"] + [f"rival_{i}" for i in range(20)]
        good = result_fixture(rivals=20)
        good["family"] = {"alpha": .025}
        for comparison in good["comparisons"]:
            for method in power.METHODS:
                trials = 8191 if method == "stationary_bootstrap" else 49999
                for null in ("epa", "practical"):
                    comparison["methods"][method][null].update(
                        raw_p=1/(trials+1), adjusted_p=120/(trials+1),
                        mc_tail_count=0, mc_reps=trials,
                        mc_p_rule="absolute_tail_plus_one")
        # This fixture isolates which rule the planner counts. It is not
        # empirical power evidence or an admitted market planning artifact.
        result = power.estimate_power(p, config, {"R": 10, "T": 768, "B": 8191},
                                      reps=128, seed=81, analyzer=lambda *a, **kw: good)
        self.assertEqual(result["operational_rule"], power.OPERATIONAL_RULE)
        self.assertTrue(result["meets_power_target"])
        self.assertGreater(result["minimum_per_contrast_simultaneous_mc_lower_bound"], .90)
        scenario = result["scenarios"][0]
        self.assertTrue(all(c["successes"] == 128 for c in scenario["contrasts"]))
        diagnostic = scenario["finite_mc_tail_diagnostics"]
        self.assertFalse(diagnostic["used_for_continuation"])
        self.assertEqual(diagnostic["strict_tail_upper_joint_power"], 0.)
        self.assertTrue(np.array_equal(diagnostic["strict_tail_upper_contrast_successes"], np.zeros((4, 20), int)))
        self.assertEqual(len(diagnostic["replicate_tail_count_ledger"]), 128)
        self.assertEqual(diagnostic["replicate_tail_count_ledger"][0][0][2], [0, 8191, "ok"])

    def test_current_analyzer_source_identity_is_emitted_without_default_substitution(self):
        result = power.dimension([], design())
        self.assertEqual(result["methods_sha256"], power.sha256(ROOT/"engine/statistics.py"))
        self.assertEqual(result["source_identity"]["power_source_sha256"], power.sha256(ROOT/"engine/power.py"))
        self.assertEqual(result["operational_rule"], power.OPERATIONAL_RULE)

    def test_planning_from_another_temporal_version_is_rejected_before_simulation(self):
        config = design()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_calendar_fixture(tmp, config)
            receipt_path = path.with_suffix(".receipt.json")
            receipt = json.loads(receipt_path.read_text())
            receipt.update(synthetic=False, is_fixture=False)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "methods differ"):
                power.load_planning([path], config)


if __name__ == "__main__":
    unittest.main()
