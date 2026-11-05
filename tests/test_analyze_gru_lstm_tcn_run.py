"""Fresh public-analysis contracts with complete synthetic CV/test cohorts.

No recorded project observations, fitted networks or test archives are read.
Residual diagnostics are mocked here; their mathematics have separate tests.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/analyze_gru_lstm_tcn_run.py"
spec = importlib.util.spec_from_file_location("public_reproduction_analysis_for_tests", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

MODELS = ("LSTM", "GRU", "TCN")
HISTORIES = (7, 14, 21, 28)
SEEDS = tuple(range(4101, 4111))
PARAMETERS = {"LSTM": 5025, "GRU": 3777, "TCN": 4289}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def synthetic_predictions(phase):
    dates = pd.date_range("2024-01-01", "2024-12-31", tz="UTC") if phase == "cv" else pd.date_range("2025-01-01", "2026-08-31", tz="UTC")
    t = np.arange(len(dates), dtype=float)
    truth = .012 + .003 * np.sin(t / 31) + .001 * np.cos(t / 11)
    persistence = np.r_[truth[0], truth[:-1]]
    rows = []
    bases = {"LSTM": .0009, "GRU": .0005, "TCN": .0013}
    for history in HISTORIES:
        for index, model in enumerate(MODELS):
            for seed in SEEDS:
                # CV favors D7; test deliberately favors D28. Test analysis must
                # nevertheless report the history already fixed in the freeze.
                history_factor = 1 + (history - 7) / 100 if phase == "cv" else 1 + (28 - history) / 30
                amplitude = bases[model] * history_factor * (1 + .035 * np.sin(seed - 4101 + index / 3))
                prediction = truth + amplitude * (1 + .08 * np.sin(t / 19 + index / 4))
                rows.append(pd.DataFrame({"variant": "relative_window",
                    "run_id": f"synthetic_{phase}_{model}_D{history}_s{seed}",
                    "history_days": history, "model": model, "seed": seed, "horizon": 1,
                    "fold_id": dates.strftime("%Y-%m"), "outcome_day": dates.strftime("%Y-%m-%d"),
                    "prediction_origin_utc": [date.isoformat() for date in dates],
                    "target_available_at_utc": [(date + pd.Timedelta(days=1)).isoformat() for date in dates],
                    "input_first_day": (dates - pd.Timedelta(days=history)).strftime("%Y-%m-%d"),
                    "input_last_day": (dates - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "y_true": truth, "y_pred": prediction, "persistence_pred": persistence,
                    "mase_train_scale": .001 + history * .000001}))
    return pd.concat(rows, ignore_index=True)


def fake_diagnostics(frame, phase):
    return {name: pd.DataFrame({"phase": [phase], "status": ["synthetic_diagnostic_mock"]})
            for name in ("bds_by_fold", "ljung_box_by_fold", "dm_pairs", "diagnostic_summary")}


class SyntheticFreshAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictions = {phase: synthetic_predictions(phase) for phase in ("cv", "test")}
        cls.tables = {phase: analysis.metric_tables(frame) for phase, frame in cls.predictions.items()}

    def fixture(self, root, *, include_test=True):
        root = Path(root)
        project, out = root / "project", root / "reproduction"
        project.mkdir()
        scripts = project / "scripts"
        scripts.mkdir()
        for name in ("analyze_gru_lstm_tcn_run.py", "build_gru_lstm_tcn_evidence.py",
                     "gru_lstm_tcn_diagnostics.py", "run_gru_lstm_tcn.py"):
            (scripts / name).write_bytes((ROOT / "scripts" / name).read_bytes())
        plan = {"adapter_kind": "public_post_training_reproduction_adapter",
            "models": {"LSTM": {"width": 32, "learning_rate": .003},
                       "GRU": {"width": 32, "learning_rate": .003},
                       "TCN": {"width": 16, "learning_rate": .001}},
            "histories": list(HISTORIES), "seeds": list(SEEDS), "variant": "relative_window",
            "code_sha256": {}, "sources": {}, "training": {"max_epochs": 100},
            "pilot_status": "Synthetic fixture; no actual fitting or power claim"}
        write_json(out / "protocol.json", plan)
        for phase in (("cv", "test") if include_test else ("cv",)):
            destination = out / ("results" if phase == "cv" else "final_evaluation/test")
            destination.mkdir(parents=True, exist_ok=True)
            frame = self.predictions[phase]
            annual, monthly = self.tables[phase]
            runs = annual.loc[annual.model.isin(MODELS)].copy()
            runs["run_id"] = [f"synthetic_{phase}_{row.model}_D{row.history_days}_s{row.seed}" for row in runs.itertuples()]
            runs["parameters"] = runs.model.map(PARAMETERS)
            runs["training_seconds"], runs["inference_seconds"] = 1., .01
            runs["checkpoint_forward_calls"] = 1 if phase == "test" else 0
            runs["folds_fitted"], runs["model_fits"] = (12, 24) if phase == "cv" else (0, 1)
            runs.to_csv(destination / "runs.csv", index=False)
            monthly.loc[monthly.model.isin(MODELS)].to_csv(destination / "fold_metrics.csv", index=False)
            frame.to_csv(destination / "predictions.csv", index=False)
            execution = {"trajectories": 120, "completed_project_fits": 2880,
                "test_files_opened": 0, "all_registered_trajectories_complete": True,
                "model_runs": 120, "common_prediction_dates": 608, "predictions": 72960,
                "monthly_reporting_blocks": 20, "test_period_model_refits": 0,
                "actual_checkpoint_forward_calls": 120, "test_already_observed": True,
                "test_used_for_configuration_selection": False}
            write_json(destination / "execution_summary.json", execution)
        if include_test:
            final = out / "final_evaluation"
            final_plan = {**plan, "test_already_observed": True,
                          "test_selects_configuration": False,
                          "test_start": "2025-01-01", "test_end_inclusive": "2026-08-31"}
            write_json(final / "protocol.json", final_plan)
            frozen = {"protocol_sha256": digest(final / "protocol.json"),
                "selection": {"by_model": {model: {"model": model, "history_days": 7} for model in MODELS}},
                "test_already_observed": True, "test_selection_allowed": False,
                "automatic_publication_allowed": False, "code_sha256": {},
                "source_cv_sha256": {}, "epoch_source_sha256": {}}
            write_json(final / "freeze.json", frozen)
            write_json(final / "final_models_manifest.json", {"freeze_sha256": digest(final / "freeze.json"),
                "jobs": [], "artifact_sha256": {}, "scope": "Synthetic metadata only; no actual checkpoint"})
            execution_path = final / "test/execution_summary.json"
            execution = json.loads(execution_path.read_text())
            execution.update(freeze_sha256=digest(final / "freeze.json"),
                             final_models_manifest_sha256=digest(final / "final_models_manifest.json"))
            write_json(execution_path, execution)
        return project, out

    def test_full_synthetic_analysis_writes_reports_and_holm_families(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, out = self.fixture(temporary)
            with mock.patch.object(analysis, "diagnose_phase", side_effect=fake_diagnostics):
                result = analysis.analyze(project, out, phase="all")
            self.assertEqual(set(result["phases"]), {"cv", "test"})
            for phase in ("cv", "test"):
                destination = out / "reproduction_analysis" / phase
                for name in ("metric_summary.csv", "run_metrics.csv", "monthly_summary.csv",
                             "pairs_D7.csv", "pairs_allD.csv", "friedman_D7.csv",
                             "selected_by_cv_metrics.csv", "source_manifest.json", "summary.json",
                             "output_manifest.json", "bds_by_fold.csv", "ljung_box_by_fold.csv",
                             "dm_pairs.csv", "diagnostic_summary.csv"):
                    self.assertTrue((destination / name).is_file(), name)
                runs = pd.read_csv(destination / "run_metrics.csv")
                self.assertEqual(len(runs.loc[runs.model.isin(MODELS)]), 120)
                self.assertEqual(set(runs.loc[runs.model.isin(MODELS), "n_dates"]), {366 if phase == "cv" else 608})
                for name, size in (("pairs_D7.csv", 3), ("pairs_allD.csv", 12)):
                    pairs = pd.read_csv(destination / name)
                    self.assertEqual(len(pairs), size)
                    self.assertEqual(set(pairs.family_size), {size})
                    adjusted = multipletests(pairs.p_value_unadjusted, method="holm")[1]
                    np.testing.assert_allclose(pairs.p_value_adjusted, adjusted, rtol=1e-12, atol=1e-15)
                summary = json.loads((destination / "summary.json").read_text())
                self.assertFalse(summary["winner_selected_on_test"])
                self.assertFalse(summary["training_performed"])
                self.assertFalse(summary["checkpoint_inference_performed"])
                source_manifest = json.loads((destination / "source_manifest.json").read_text())
                for relative, expected in source_manifest["source_sha256"].items():
                    self.assertFalse(Path(relative).is_absolute())
                    self.assertEqual(digest(out / relative), expected)
                artifact_manifest = json.loads((destination / "output_manifest.json").read_text())
                for relative, expected in artifact_manifest["artifact_sha256"].items():
                    self.assertEqual(digest(destination / relative), expected)
            test_destination = out / "reproduction_analysis/test"
            selected = pd.read_csv(test_destination / "selected_by_cv_metrics.csv")
            self.assertEqual(set(selected.loc[selected.model.eq("GRU"), "history_days"]), {7})
            metrics = pd.read_csv(test_destination / "metric_summary.csv")
            gru = metrics.loc[metrics.model.eq("GRU") & metrics.metric.eq("rmse")]
            self.assertEqual(int(gru.loc[gru["mean"].idxmin(), "history_days"]), 28)
            summary = json.loads((test_destination / "summary.json").read_text())
            self.assertEqual(summary["cv_selected_by_model"]["GRU"]["history_days"], 7)

    def test_cv_analysis_can_run_before_any_final_freeze_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, out = self.fixture(temporary, include_test=False)
            self.assertFalse((out / "final_evaluation").exists())
            with mock.patch.object(analysis, "diagnose_phase", side_effect=fake_diagnostics):
                result = analysis.analyze(project, out, phase="cv")
            self.assertEqual(set(result["phases"]), {"cv"})
            self.assertFalse((out / "final_evaluation").exists())

    def test_incomplete_horizon_and_information_clock_are_rejected(self):
        frame = self.predictions["cv"]
        with self.assertRaises(ValueError):
            analysis.validate_predictions(frame.iloc[:-1], "cv")
        changed = frame.copy()
        changed.loc[0, "horizon"] = 2
        with self.assertRaises(ValueError):
            analysis.validate_predictions(changed, "cv")
        changed = frame.copy()
        changed.loc[0, "prediction_origin_utc"] = "2030-01-01T00:00:00+00:00"
        with self.assertRaises(ValueError):
            analysis.validate_predictions(changed, "cv")

    def test_incomplete_test_execution_is_rejected_before_predictions_are_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, out = self.fixture(temporary)
            path = out / "final_evaluation/test/execution_summary.json"
            execution = json.loads(path.read_text())
            execution["model_runs"] = 119
            write_json(path, execution)
            with mock.patch.object(analysis.pd, "read_csv", side_effect=AssertionError("premature prediction read")):
                with self.assertRaises(ValueError):
                    analysis.analyze(project, out, phase="test")

    def test_stored_metrics_must_match_the_forecasts(self):
        with tempfile.TemporaryDirectory() as temporary:
            project, out = self.fixture(temporary, include_test=False)
            path = out / "results/runs.csv"
            runs = pd.read_csv(path)
            runs.loc[0, "rmse"] += .01
            runs.to_csv(path, index=False)
            with mock.patch.object(analysis, "diagnose_phase", side_effect=fake_diagnostics):
                with self.assertRaisesRegex(ValueError, "Stored metric differs"):
                    analysis.analyze(project, out, phase="cv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
