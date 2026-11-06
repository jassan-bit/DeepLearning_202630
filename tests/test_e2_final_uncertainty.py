"""Synthetic A53/OOF provenance and numerical tests; no market fitting/scoring."""
from __future__ import annotations

from collections import defaultdict
import copy
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import e2_final_uncertainty as module
import e2_predictive_uncertainty as old
import run_e2_final_benchmark as engine


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(module.canonical_bytes(value))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def descriptor(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": module.digest_file(path)}


def make_A_fixture(root):
    now = datetime.now(timezone.utc)
    fmt = lambda hours: (now+timedelta(hours=hours)).isoformat()
    receipts, calibration, evaluation, primary = {}, [], [], []
    caldates = old.calendar(module.CAL_START, module.CAL_END)
    evaldates = old.calendar(old.EVAL_START, old.EVAL_END)
    for index, model in enumerate(module.CATALOGUE):
        seeds = ("deterministic",) if model in module.DETERMINISTIC else module.A_SEEDS
        r = {"branch": module.A_BRANCH, "model": model, "fit_origin": "2024-11-08",
             "fit_outcome_max": "2024-10-31", "scaler_fit_outcome_max": "2024-10-31",
             "configuration_selection_outcome_max": "2024-10-31", "branch_selection_outcome_max": "2024-10-31",
             "primary_fit_replaced": False, "point_predictor_kind": "deterministic_frozen_predictor" if model in module.DETERMINISTIC else "mean_frozen_seed_members",
             "ensemble_seeds": list(seeds), "checkpoint_closed_at_utc": fmt(-3),
             "calibration_scoring_started_at_utc": fmt(-2), "target_scale": "original_unannualized_daily_volatility",
             "refit_during_calibration": False, "calibration_used_for_weight_selection": False, "test_used_for_selection": False,
             **{key: {} for key in ["member_run_ids", "member_checkpoint_sha256", "member_scaler_sha256", "member_config_sha256",
                                    "member_files", "member_completion_sha256", "member_checkpoint_closed_at_utc"]}}
        for seedindex, seed in enumerate(seeds):
            folder = root/"members"/(model.replace("/", "_")+"_"+seed); folder.mkdir(parents=True)
            checkpoint = folder/"checkpoint.fixture"; checkpoint.write_bytes((model+seed).encode())
            setup = folder/"setup.json"; preprocessing = folder/"preprocessing.json"
            write_json(setup, {"model": model, "seed": seed, "fit_origin": "2024-11-08", "fit_outcome_max": "2024-10-31",
                               "scaler_fit_outcome_max": "2024-10-31", "configuration_selection_outcome_max": "2024-10-31",
                               "epoch_selection_outcome_max": "2024-10-31"})
            write_json(preprocessing, {"last_fit_outcome_day_utc": "2024-10-31", "y_scale": .02})
            completion = folder/"completion.json"
            write_json(completion, {"status": "passed", "completed_at_utc": fmt(-3),
                                   "artifacts": {p.name: module.digest_file(p) for p in [checkpoint, setup, preprocessing]}})
            r["member_run_ids"][seed] = sha256((model+"_"+seed).encode()).hexdigest()
            for key, path in [("member_checkpoint_sha256", checkpoint), ("member_scaler_sha256", preprocessing),
                              ("member_config_sha256", setup), ("member_completion_sha256", completion)]:
                r[key][seed] = module.digest_file(path)
            r["member_files"][seed] = {k: p.relative_to(root).as_posix() for k, p in [("checkpoint", checkpoint), ("scaler", preprocessing), ("config", setup), ("completion", completion)]}
            r["member_checkpoint_closed_at_utc"][seed] = fmt(-3)
            member_offset = 0 if model in module.DETERMINISTIC else (seedindex-2)*.0002
            for stage, dates, output, started in [(module.CAL_STAGE, caldates, calibration, fmt(-1)),
                                                   (module.EVAL_STAGE, evaldates, evaluation, fmt(3))]:
                for i, date in enumerate(dates):
                    truth = .03+.003*np.sin(i/5)
                    signed = .004*np.sin(i/3) if stage == module.CAL_STAGE else .001
                    pred = truth-signed+member_offset+index*1e-6
                    persistence = .03 if i%2 else .01
                    output.append({"model": model, "seed": seed, "outcome_day": date.isoformat(),
                        "y_true": truth, "y_pred": pred, "persistence_pred": persistence,
                        "high_volatility_known": persistence > module.HIGH_THRESHOLD, "eligible": True,
                        "run_id": r["member_run_ids"][seed], "checkpoint_sha256": r["member_checkpoint_sha256"][seed],
                        "scaler_sha256": r["member_scaler_sha256"][seed], "config_sha256": r["member_config_sha256"][seed],
                        "checkpoint_closed_at_utc": fmt(-3), "scoring_started_at_utc": started,
                        "fit_origin": "2024-11-08", "fit_outcome_max": "2024-10-31",
                        "scaler_fit_outcome_max": "2024-10-31", "configuration_selection_outcome_max": "2024-10-31", "source_stage": stage})
        receipts[model] = r
        for j, seed in enumerate(("deterministic",) if model in module.DETERMINISTIC else [str(s) for s in range(670101, 670106)]):
            for i, date in enumerate(evaldates):
                truth = .03+.003*np.sin(i/5)
                offset = 0 if model in module.DETERMINISTIC else (j-2)*.0002
                primary.append({"model": model, "seed": seed, "outcome_day": date.isoformat(), "y_true": truth, "y_pred": truth+offset,
                    "run_id": sha256(("primary"+model+seed).encode()).hexdigest(), "checkpoint_sha256": "a"*64,
                    "scaler_sha256": "b"*64, "config_sha256": "c"*64,
                    "checkpoint_closed_at_utc": fmt(-3), "scoring_started_at_utc": fmt(3), "source_stage": module.PRIMARY_STAGE})
    calpath, evalpath, primarypath = [root/n for n in ["calibration.csv", "evaluation.csv", "primary.csv"]]
    write_csv(calpath, calibration); write_csv(evalpath, evaluation); write_csv(primarypath, primary)
    manifest_A = root/"A_manifest.json"; manifest_P = root/"primary_manifest.json"
    write_json(manifest_A, {"synthetic_fixture_only": True, "members": 84})
    write_json(manifest_P, {"synthetic_fixture_only": True, "members": 84})
    cohort = {"all_primary_and_A_weights_closed_at_utc": fmt(-3), "calibration_scoring_started_at_utc": fmt(-2),
              "calibration_closed_at_utc": fmt(1), "evaluation_scoring_started_at_utc": fmt(2),
              "primary_fit_manifest": descriptor(manifest_P, root), "branch_A_fit_manifest": descriptor(manifest_A, root),
              "calibration_predictions": descriptor(calpath, root), "evaluation_predictions": descriptor(evalpath, root),
              "primary_predictions": descriptor(primarypath, root)}
    subgroup = {"threshold_source": "training_only", "threshold_outcome_max": "2023-12-24", "test_or_calibration_used": False,
                "high_volatility_threshold": module.HIGH_THRESHOLD, "forecast_bin_edges": [.02, .03], "frozen_at_utc": fmt(-4)}
    subgroup_path = root/"subgroup.json"; write_json(subgroup_path, subgroup)
    cohort["subgroup_receipt"] = descriptor(subgroup_path, root)
    return {"root": root, "receipts": receipts, "calibration": calibration, "evaluation": evaluation,
            "primary": primary, "cohort": cohort, "subgroup": subgroup}


def make_oof_fixture(root):
    rows, records = [], {}; inputs = "d"*64
    for model in module.CATALOGUE:
        seeds = ("deterministic",) if model in module.DETERMINISTIC else module.OOF_SEEDS
        inherited = model in module.LEGACY_CANDIDATES
        candidate = engine.fixed_candidate(model) if inherited else {"model": model, "candidate_id": "c01", "parameters": {}}
        for month in range(1, 11):
            origin = old.day(f"2024-{month:02d}-01")
            end = old.day("2024-10-31") if month == 10 else old.day(f"2024-{month+1:02d}-01")-timedelta(days=1)
            cutoff = (origin-timedelta(days=8)).isoformat()
            for j, seed in enumerate(seeds):
                selection = "2024-10-31" if inherited else cutoff
                claims = {"historical_selection_exposure_disclosed": inherited,
                    "configuration_selection_kind": "inherited_fixed_E1_recipe" if inherited else "registered_nested_selection_or_rule",
                    "E2_new_configuration_selection": not inherited, "unbiased_nested_selection_claimed": not inherited,
                    "configuration_selection_outcome_max": selection, "epoch_selection_outcome_max": cutoff}
                identity = {"model": model, "seed": seed, "origin": origin.isoformat(), "fit_outcome_max": cutoff,
                            "candidate_id": candidate["candidate_id"], "candidate_sha256": engine.hash_json(candidate)}
                run = engine.hash_json({"fit": identity})
                folder = root/"fits"/model.replace("/", "_")/candidate["candidate_id"]/origin.isoformat()/("seed_"+seed)
                folder.mkdir(parents=True)
                setup, preprocessing, checkpoint = folder/"setup.json", folder/"preprocessing.json", folder/"checkpoint.pt"
                write_json(setup, {"model": model, "candidate": candidate, "seed": seed, "fit_origin": origin.isoformat(),
                                   "fit_outcome_max": cutoff, "scaler_fit_outcome_max": cutoff, **claims})
                write_json(preprocessing, {"last_fit_outcome_day_utc": cutoff})
                checkpoint.write_bytes((run+"_synthetic_not_weights").encode())
                completion = folder/"completion.json"
                write_json(completion, {"status": "passed", "identity": identity, "registration_inputs_sha256": inputs,
                    "completed_at_utc": "2026-01-01T00:00:00+00:00",
                    "artifacts": {p.name: module.digest_file(p) for p in [setup, preprocessing, checkpoint]}})
                local_rows = []
                for k, date in enumerate(old.calendar(origin, end)):
                    truth = .02+.002*np.sin(date.timetuple().tm_yday)
                    local_rows.append({"model": model, "candidate_id": candidate["candidate_id"], "seed": seed,
                        "outcome_day": date.isoformat(), "y_true": truth,
                        "y_pred": truth+.001*(j+1), "run_id": run, "checkpoint_sha256": module.digest_file(checkpoint),
                        "scaler_sha256": module.digest_file(preprocessing), "config_sha256": module.digest_file(setup),
                        "checkpoint_closed_at_utc": "2026-01-01T00:00:00+00:00", "scoring_started_at_utc": "2026-01-02T00:00:00+00:00",
                        "fit_origin": origin.isoformat(), "fit_outcome_max": cutoff, "scaler_fit_outcome_max": cutoff,
                        **claims, "eligible": True, "source_stage": "E2_nested_development"})
                scorefolder = root/"scores"/run; predictions = scorefolder/"predictions.csv"
                write_csv(predictions, local_rows); score = scorefolder/"completion.json"
                write_json(score, {"status": "passed", "registration_inputs_sha256": inputs,
                    "fit_completion_sha256": module.digest_file(completion), "artifacts": {predictions.name: module.digest_file(predictions)}})
                records[run] = {"folder": folder.relative_to(root).as_posix(), "scoring_receipt": score.relative_to(root).as_posix(),
                                "predictions_file": predictions.relative_to(root).as_posix()}
                if model == "TimeMixer" and month == 1 and j == 0:
                    write_json(folder/"checkpoint_pruned_after_development_scoring.json", {
                        "original_fit_receipt_sha256": module.digest_file(completion), "registration_inputs_sha256": inputs,
                        "scoring_receipt_relative_to_output": score.relative_to(root).as_posix(),
                        "closed_scoring_receipt_sha256": module.digest_file(score),
                        "pruned_artifact_sha256": {checkpoint.name: module.digest_file(checkpoint)}})
                    checkpoint.unlink()
                rows += local_rows
    taxonomy = {"threshold_source": "training_reference_absolute_errors_only", "training_outcome_max": "2023-12-24",
                "training_cases": 100, "reference_name": "synthetic training reference", "q50_abs_error": .0005,
                "q90_abs_error": .0015, "created_at_utc": "2025-01-01T00:00:00+00:00"}
    return rows, records, inputs, taxonomy


class NumericTests(unittest.TestCase):
    def test_policy_closes_A53_gap_without_changing_old_A60(self):
        p = module.policy()
        self.assertEqual(p["calibration_origins"], 53)
        self.assertEqual(p["A_fit_outcome_max"], "2024-10-31")
        self.assertEqual(p["nominal_fit_origin"], "2024-11-08")
        self.assertEqual(old.CAL_START.isoformat(), "2024-11-01")
        self.assertEqual(old.FIT_MAX.isoformat(), "2024-10-24")

    def test_signed_CDF_quantiles_and_randomized_atom_PIT_are_consistent(self):
        points = np.array([.001, .005]); residuals = np.array([-.002, .003])
        samples = old.signed_residual_distribution(points, residuals)
        self.assertEqual(samples[0, 0], 0.)
        np.testing.assert_allclose(module.empirical_quantile_samples(samples, [.75, .25], .5), [0., .003])
        pit = module.randomized_empirical_pit([0., .003], samples, [.75, .25], seed=71)
        self.assertTrue(((pit >= 0)&(pit <= .75)).all())
        np.testing.assert_array_equal(pit, module.randomized_empirical_pit([0., .003], samples, [.75, .25], seed=71))

    def test_weighted_CRPS_matches_direct_pairwise_definition(self):
        draws = np.array([[0., .01, .03], [.02, .04, .06]])
        y, p = np.array([.015, .03]), np.array([.2, .5, .3])
        direct = (np.abs(draws-y[:, None])*p).sum(1)-.5*(np.abs(draws[:, :, None]-draws[:, None, :])*p[None, :, None]*p[None, None, :]).sum((1, 2))
        np.testing.assert_allclose(old.empirical_crps(y, draws, weights=p), direct)

    def test_calendar_bootstrap_keeps_empty_and_small_groups_not_estimable(self):
        values = np.arange(60)/60; draws = module._draws(60, 14)
        empty = module._summary(values, np.zeros(60, bool), draws, block=14, conditional=True)
        self.assertIsNone(empty["mean"]); self.assertIsNone(empty["ci95_nominal"])
        mask = np.arange(60)%2 == 0
        result = module._summary(values, mask, draws, block=14, conditional=True)
        hand = np.array([values[idx][mask[idx]].mean() for idx in draws])
        np.testing.assert_allclose(result["ci95_nominal"], np.quantile(hand, [.025, .975]))
        small = module._summary(values, np.arange(60)<3, draws, block=14, conditional=True)
        self.assertIsNotNone(small["mean"]); self.assertIsNone(small["ci95_nominal"])

    def test_PIT_uses_centered_temporal_null_instead_of_iid_KS(self):
        draws = module._draws(60, 14)
        uniform = module.pit_diagnostic((np.arange(60)+.5)/60, draws, block=14)
        biased = module.pit_diagnostic(np.full(60, .99), draws, block=14)
        self.assertGreater(uniform["p_temporal_centered_bootstrap"], .05)
        self.assertLess(biased["p_temporal_centered_bootstrap"], .01)
        self.assertFalse(biased["finite_sample_exact_pvalue_claimed"])

    def test_training_group_thresholds_reject_calibration_data(self):
        targets = np.linspace(.01, .04, 30)
        dates = ["2023-12-"+f"{i:02d}" for i in range(1, 25)]
        receipt = module.build_subgroup_receipt(targets[:24], dates, source_sha256="d"*64)
        self.assertEqual(receipt["forecast_bin_quantiles"], [1/3, 2/3])
        with self.assertRaises(ValueError):
            module.build_subgroup_receipt(targets[:24], ["2024-11-"+f"{i:02d}" for i in range(1, 25)], source_sha256="d"*64)

    def test_subgroup_calibration_slope_matches_known_relation_and_constant_is_not_zero(self):
        point = np.linspace(.01, .03, 60); truth = .001+2*point
        result = module.subgroup_calibration((np.arange(60)+.5)/60, point, truth, np.arange(60)%2 == 0, module._draws(60, 14))
        self.assertAlmostEqual(result["point_calibration_OLS"]["slope"], 2.)
        self.assertAlmostEqual(result["point_calibration_OLS"]["intercept"], .001)
        np.testing.assert_allclose(result["point_calibration_OLS"]["slope_ci95_nominal"], [2., 2.], atol=1e-11)
        constant = module.subgroup_calibration(np.linspace(0, 1, 60), np.full(60, .01), truth, np.ones(60, bool), module._draws(60, 14))
        self.assertIsNone(constant["point_calibration_OLS"]["slope"])

    def test_empty_subgroup_PIT_calibration_is_not_perfect_zero(self):
        result = module.subgroup_calibration(np.linspace(0, 1, 60), np.linspace(.01, .02, 60), np.linspace(.01, .03, 60),
                                             np.zeros(60, bool), module._draws(60, 14))
        self.assertIsNone(result["PIT_calibration_L1_grid"]["value"])
        self.assertIsNone(result["PIT_calibration_L1_grid"]["ci95_nominal"])

    def test_default_CLI_describes_plan_without_loading_data(self):
        with patch.object(module, "_read_csv", side_effect=AssertionError("must not read forecasts")), patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(module.main([]), 0)
            self.assertEqual(json.loads(output.getvalue())["calibration_origins"], 53)


class A53Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.f = make_A_fixture(cls.root)
        cls.calibrators = module.fit_calibration_cohort(cls.f["calibration"], cls.f["receipts"], artifact_root=cls.root, cohort_fit_receipt=cls.f["cohort"])
        path = cls.root/"calibrators.json"; write_json(path, cls.calibrators)
        cls.f["cohort"]["calibrators_file"] = descriptor(path, cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all20_A53_models_fit_residual_calibrators_with_84_members(self):
        self.assertEqual(set(self.calibrators["models"]), set(module.CATALOGUE))
        self.assertEqual(len(self.f["calibration"]), 84*53)
        for value in self.calibrators["models"].values():
            self.assertEqual(value["calibration_origins"], 53)
            self.assertEqual(len(value["signed_calibration_residuals"]), 53)
            self.assertFalse(value["exact_finite_sample_coverage_claimed"])

    def test_new_guard_rejects_old_A60_or_later_fit(self):
        r = copy.deepcopy(self.f["receipts"][module.CATALOGUE[0]])
        r["branch"] = "secondary_frozen_calibration_A"
        with self.assertRaises(ValueError):
            module.validate_calibration_support(r, old.calendar(module.CAL_START, module.CAL_END))
        r["branch"] = module.A_BRANCH; r["fit_outcome_max"] = "2024-11-01"
        with self.assertRaises(ValueError):
            module.validate_calibration_support(r, old.calendar(module.CAL_START, module.CAL_END))

    def test_changed_real_checkpoint_bytes_reject_calibration(self):
        r = self.f["receipts"][module.CATALOGUE[0]]; path = self.root/r["member_files"][module.A_SEEDS[0]]["checkpoint"]
        content = path.read_bytes()
        try:
            path.write_bytes(content+b"modified")
            with self.assertRaises(ValueError):
                module.fit_calibration_cohort(self.f["calibration"], self.f["receipts"], artifact_root=self.root, cohort_fit_receipt=self.f["cohort"])
        finally:
            path.write_bytes(content)

    def test_missing_member_date_rejects_complete_cohort_requirement(self):
        with self.assertRaises(ValueError):
            module._validate_A_rows(self.f["calibration"][:-1], self.f["receipts"], module.CAL_STAGE)

    def test_joint_order_requires_primary_and_A_closed_before_calibration(self):
        c = copy.deepcopy(self.f["cohort"])
        c["all_primary_and_A_weights_closed_at_utc"] = c["evaluation_scoring_started_at_utc"]
        with self.assertRaises(ValueError):
            module.validate_joint_order(c, artifact_root=self.root)

    def test_A_actual_setup_cannot_hide_postcutoff_recipe_selection(self):
        receipt = copy.deepcopy(self.f["receipts"][module.CATALOGUE[0]])
        seed = module.A_SEEDS[0]
        path = self.root/receipt["member_files"][seed]["config"]
        completion_path = self.root/receipt["member_files"][seed]["completion"]
        original, original_completion = path.read_bytes(), completion_path.read_bytes()
        try:
            setup = json.loads(original); setup["configuration_selection_outcome_max"] = "2024-11-01"
            write_json(path, setup)
            completion = json.loads(original_completion)
            completion["artifacts"][path.name] = module.digest_file(path)
            write_json(completion_path, completion)
            receipt["member_config_sha256"][seed] = module.digest_file(path)
            receipt["member_completion_sha256"][seed] = module.digest_file(completion_path)
            with self.assertRaisesRegex(ValueError, "recipe and epoch selection"):
                module._verify_member_files(receipt, artifact_root=self.root)
        finally:
            path.write_bytes(original); completion_path.write_bytes(original_completion)

    def test_full_evaluation_reports_all20_distinct_distribution_scores_and_estimands(self):
        result = module.evaluate_final_uncertainty(self.f["calibration"], self.f["evaluation"], self.f["receipts"],
            primary_rows=self.f["primary"], subgroup_receipt=self.f["subgroup"], top3=list(module.TOP10[:3]),
            artifact_root=self.root, cohort_receipt=self.f["cohort"], frozen_calibrators=self.calibrators)
        public = result["summary"]
        self.assertEqual(len(public["models"]), 20)
        self.assertTrue(public["intervals_and_CDF_are_distinct_procedures"])
        example = public["models"]["TSMixer"]
        comparison = example["point_performance_side_by_side"]
        self.assertGreater(comparison["primary_mean_seed_RMSE_distinct_estimand"], comparison["primary_fixed_ensemble_RMSE"])
        self.assertLess(comparison["primary_fixed_ensemble_RMSE"], 1e-12)
        pit = example["block_summaries"]["14"]["PIT"]
        self.assertIn("p_Holm_exploratory_family20", pit)
        self.assertNotIn("signed_calibration_residuals", json.dumps(public))
        self.assertEqual(len(result["private_evaluation_frontier"]["TSMixer"]["high_loss_origins"]), 10)
        self.assertNotIn("high_loss_origins", json.dumps(public))

    def test_changed_frozen_radii_cannot_be_selected_after_test(self):
        changed = copy.deepcopy(self.calibrators); changed["models"]["TSMixer"]["radii_by_alpha"]["0.2"] = .1
        with self.assertRaises(ValueError):
            module.evaluate_final_uncertainty(self.f["calibration"], self.f["evaluation"], self.f["receipts"],
                primary_rows=self.f["primary"], subgroup_receipt=self.f["subgroup"], top3=list(module.TOP10[:3]),
                artifact_root=self.root, cohort_receipt=self.f["cohort"], frozen_calibrators=changed)

    def test_thresholds_frozen_after_calibration_reject(self):
        receipt = copy.deepcopy(self.f["subgroup"]); receipt["frozen_at_utc"] = self.f["cohort"]["evaluation_scoring_started_at_utc"]
        with self.assertRaises(ValueError):
            module.evaluate_final_uncertainty(self.f["calibration"], self.f["evaluation"], self.f["receipts"],
                primary_rows=self.f["primary"], subgroup_receipt=receipt, top3=list(module.TOP10[:3]),
                artifact_root=self.root, cohort_receipt=self.f["cohort"], frozen_calibrators=self.calibrators)

    def test_threshold_values_cannot_change_after_sealing_even_with_old_timestamp(self):
        changed = copy.deepcopy(self.f["subgroup"]); changed["forecast_bin_edges"] = [.001, .002]
        with self.assertRaises(ValueError):
            module.evaluate_final_uncertainty(self.f["calibration"], self.f["evaluation"], self.f["receipts"],
                primary_rows=self.f["primary"], subgroup_receipt=changed, top3=list(module.TOP10[:3]),
                artifact_root=self.root, cohort_receipt=self.f["cohort"], frozen_calibrators=self.calibrators)

    def test_private_outputs_cannot_be_written_inside_the_repository(self):
        with self.assertRaises(ValueError):
            module._destination(ROOT/"book_e2/_static/private_cases")

    def test_calibration_CLI_roundtrips_csv_and_keeps_residuals_private(self):
        receipts = self.root/"cli_receipts.json"; cohort = self.root/"cli_cohort.json"
        write_json(receipts, self.f["receipts"]); write_json(cohort, self.f["cohort"])
        output = self.root/"new_CLI_calibration_output"
        with patch("sys.stdout", new_callable=io.StringIO):
            code = module.main(["calibrate", "--artifact-root", str(self.root), "--calibration-csv", str(self.root/"calibration.csv"),
                "--receipts-json", str(receipts), "--cohort-receipt-json", str(cohort), "--output-dir", str(output)])
        self.assertEqual(code, 0)
        receipt = json.loads((output/"analysis_receipt.json").read_text())
        self.assertEqual(set(receipt["input_sha256"]), {"calibration_csv", "receipts_json", "cohort_receipt_json"})
        self.assertTrue((output/"calibrators_private.json").exists())
        self.assertNotIn("signed_calibration_residuals", (output/"summary.json").read_text())


class FinalOOFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.rows, cls.records, cls.inputs, cls.taxonomy = make_oof_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_twenty_model_OOF_has100_unique_cases_each_and_one_pruned_chain(self):
        result = module.analyze_final_oof(self.rows, self.taxonomy, artifact_root=self.root, fit_records=self.records,
                                          registration_inputs_sha256=self.inputs, top3=list(module.TOP10[:3]))
        self.assertEqual(result["summary"]["checkpoint_evidence"]["runs_with_explicit_checkpoint_pruning"], 1)
        self.assertEqual(result["summary"]["checkpoint_evidence"]["closed_runs_verified"], 360)
        self.assertEqual(len(result["private_audit_cases"]), 20)
        for records in result["private_audit_cases"].values():
            self.assertEqual(len(records), 100)
            self.assertEqual(len({r["origin_utc"] for r in records}), 100)
        self.assertFalse(result["summary"]["postOOF_calibration_used_for_2024_confidence"])
        self.assertNotIn("private_stratified_audit_sample", json.dumps(result["summary"]))
        self.assertEqual(len(result["summary"]["nested_configuration_selection_models"]), 17)
        self.assertEqual(set(result["summary"]["conditional_inherited_references"]), set(module.LEGACY_CANDIDATES))
        self.assertFalse(result["summary"]["all_models_unbiased_nested_configuration_selection"])

    def test_inherited_selection_exposure_is_preserved_without_rewriting_dates(self):
        before = copy.deepcopy(self.rows)
        panel = module.validate_final_oof_panel(self.rows)
        self.assertEqual(self.rows, before)
        self.assertEqual(len(panel["dates"]), 305)
        inherited = next(r for r in self.rows if r["model"] in module.LEGACY_CANDIDATES)
        self.assertEqual(inherited["configuration_selection_outcome_max"], "2024-10-31")
        self.assertGreater(old.day(inherited["configuration_selection_outcome_max"]), old._origin(inherited))

    def test_inherited_hyperparameters_cannot_be_claimed_as_unbiased_nested_selection(self):
        rows = copy.deepcopy(self.rows)
        inherited = next(r for r in rows if r["model"] in module.LEGACY_CANDIDATES)
        inherited["unbiased_nested_selection_claimed"] = True
        with self.assertRaisesRegex(ValueError, "claims must be explicit and truthful"):
            module.validate_final_oof_panel(rows)

    def test_inherited_recipe_does_not_receive_a_weight_scaler_or_epoch_cutoff_exception(self):
        for field in ("fit_outcome_max", "scaler_fit_outcome_max", "epoch_selection_outcome_max"):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                inherited = next(r for r in rows if r["model"] in module.LEGACY_CANDIDATES)
                inherited[field] = "2024-10-31"
                with self.assertRaisesRegex(ValueError, "outer-origin embargo"):
                    module.validate_final_oof_panel(rows)

    def test_retuned_model_cannot_use_the_historical_selection_exception(self):
        rows = copy.deepcopy(self.rows)
        row = next(r for r in rows if r["model"] == "GRU-E2")
        row["configuration_selection_outcome_max"] = "2024-10-31"
        row["historical_selection_exposure_disclosed"] = True
        with self.assertRaisesRegex(ValueError, "monthly outer-origin embargo"):
            module.validate_final_oof_panel(rows)

    def test_exact_legacy_candidate_identity_is_required(self):
        rows = copy.deepcopy(self.rows)
        row = next(r for r in rows if r["model"] in module.LEGACY_CANDIDATES)
        row["candidate_id"] = "different-fixed-recipe"
        with self.assertRaisesRegex(ValueError, "exact three fixed E1 recipes"):
            module.validate_final_oof_panel(rows)

    def test_P0_relabeling_is_not_final_nested_OOF(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["source_stage"] = "P0_diagnostic"
        with self.assertRaises(ValueError):
            module.analyze_final_oof(rows, self.taxonomy, artifact_root=self.root, fit_records=self.records, registration_inputs_sha256=self.inputs)

    def test_modified_original_closed_score_csv_is_rejected(self):
        key = next(iter(self.records)); record = self.records[key]; path = self.root/record["predictions_file"]
        original = path.read_bytes()
        try:
            path.write_bytes(original+b"\n")
            with self.assertRaises(ValueError):
                module._verify_oof_prediction_chain([r for r in self.rows if r["run_id"] == key], {key: record},
                    artifact_root=self.root, registration_inputs_sha256=self.inputs)
        finally:
            path.write_bytes(original)

    def test_current_OOF_support_dates_cannot_be_changed_relative_to_closed_source(self):
        key = next(iter(self.records)); rows = [copy.deepcopy(r) for r in self.rows if r["run_id"] == key]
        rows[0]["fit_outcome_max"] = "2022-01-01"
        with self.assertRaises(ValueError):
            module._verify_oof_prediction_chain(rows, {key: self.records[key]}, artifact_root=self.root,
                                                 registration_inputs_sha256=self.inputs)


if __name__ == "__main__":
    unittest.main()
