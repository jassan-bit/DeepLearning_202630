"""Boundary/registration tests; no actual market fitting or reserve access."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_gru_lstm_tcn_planned as runner


def fixture_design():
    candidates = {}
    for model in runner.FAMILIES:
        widths = [8, 16] if model == "TCN" else [16, 32]
        candidates[model] = [{"candidate_id": f"c{i:02d}", "width": w,
            "learning_rate": lr, "dropout": d}
            for i, (w, lr, d) in enumerate((w, lr, d) for w in widths
                                           for lr in [.001, .003] for d in [0., .1])]
    return {"models": list(runner.FAMILIES), "history_days": 7,
        "target_returns_days": 7, "forecast_days": 1,
        "training": dict(runner.TRAINING),
        "search": {"method": "exhaustive_grid", "expected_new_fits": 960,
            "folds_source": runner.FOLDS, "candidates": candidates, "seeds": [6101, 6102],
            "months": [f"2024-{month:02d}" for month in range(1, 11)],
            "evaluation": "mean of pooled seed RMSE", "tie_tolerance": 1e-12},
        "pilot": {"seeds": list(range(7101, 7111)), "origin": "2024-11-01",
            "start": "2024-11-01", "end": "2024-12-30", "calendar_days": 60},
        "definitive": {"origin": "2025-01-01", "seed_start": 8101},
        "reserve": {"start": "2026-09-01", "end": "2026-10-30", "calendar_days": 60},
        "failure_policy": {"technical_failures": "stop and preserve"}}


class RegisteredSearchTests(unittest.TestCase):
    def test_explicit_grid_and_candidate_identifiers_are_preserved(self):
        design = fixture_design()
        registration = {"settings": runner.effective_settings(design)}
        grid = runner.candidate_grid(registration)
        self.assertEqual(len(grid), 24)
        for family in runner.FAMILIES:
            actual = [row for row in grid if row["model"] == family]
            self.assertEqual([row["candidate_id"] for row in actual], [f"c{i:02d}" for i in range(8)])
            self.assertEqual([{k: v for k, v in row.items() if k not in {"model", "history_days"}}
                              for row in actual], design["search"]["candidates"][family])

    def test_unregistered_search_or_period_change_is_rejected(self):
        for mutation in ("grid", "fits", "months", "pilot", "reserve"):
            design = fixture_design()
            if mutation == "grid": design["search"]["candidates"]["TCN"][0]["learning_rate"] = .2
            if mutation == "fits": design["search"]["expected_new_fits"] = 1200
            if mutation == "months": design["search"]["months"].append("2024-11")
            if mutation == "pilot": design["pilot"]["end"] = "2024-12-31"
            if mutation == "reserve": design["reserve"]["end"] = "2026-10-31"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.effective_settings(design)

    def test_equal_candidate_selection_with_ties_and_complete_seeds(self):
        candidates = runner.candidate_grid({"settings": runner.effective_settings(fixture_design())})
        rows = [{**candidate, "trajectory_seed": seed, "rmse": .01,
                 "parameters": candidate["width"] * 100}
                for candidate in candidates for seed in (6101, 6102)]
        recipes, baseline, summary = runner.select_candidates(rows, candidates)
        self.assertEqual(len(summary), 24)
        self.assertEqual({row["candidate_id"] for row in recipes.values()}, {"c00"})
        self.assertEqual(baseline, "TCN")  # All errors tie; fewer parameters wins.
        with self.assertRaises(ValueError): runner.select_candidates(rows[:-1], candidates)
        with self.assertRaises(ValueError): runner.select_candidates(rows + [rows[0]], candidates)


class CausalSplitTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.date_range("2023-01-01", "2024-12-31", tz="UTC")
        eligible = np.ones(len(self.calendar), dtype=bool)
        eligible[:8] = False
        self.data = {"calendar": self.calendar, "eligible_by_history": {7: eligible}}

    def test_fixed_pilot_stays_before_november_with_two_embargoes(self):
        split = runner.split_at(self.data, "2024-11-01")
        self.assertEqual(str(self.calendar[split["train_indices"][-1]].date()), "2024-10-24")
        self.assertEqual(str(self.calendar[split["inner_validation_indices"][0]].date()), "2024-08-03")
        self.assertEqual(len(split["inner_validation_indices"]), 83)
        self.assertLessEqual(split["inner_train_indices"][-1], split["inner_validation_indices"][0] - 8)

    def test_final_fit_uses_december24_boundary_and_no_reserve_data(self):
        split = runner.split_at(self.data, "2025-01-01")
        self.assertEqual(str(self.calendar[split["train_indices"][-1]].date()), "2024-12-24")
        self.assertEqual(str(self.calendar[split["inner_validation_indices"][0]].date()), "2024-10-03")
        self.assertEqual(str(self.calendar[split["inner_train_indices"][-1]].date()), "2024-09-25")
        with self.assertRaises(ValueError): runner.split_at(self.data, "2026-09-01")


class FrozenArtifactsTests(unittest.TestCase):
    def test_cached_jobs_reject_changed_bytes_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "artifact.txt").write_text("original")
            record = {"registration_signature": "sig", "record": {"model": "GRU"},
                "artifact_sha256": {"artifact.txt": runner.digest(folder / "artifact.txt")}}
            runner.write_json(folder / "completion.json", record)
            self.assertEqual(runner.cached_completion(folder, "sig"), {"model": "GRU"})
            (folder / "artifact.txt").write_text("changed")
            with self.assertRaises(ValueError): runner.cached_completion(folder, "sig")
            with self.assertRaises(ValueError): runner.safe_path(folder, "../artifact.txt")

    def test_incomplete_jobs_are_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "incomplete"
            folder.mkdir()
            (folder / "attempt.txt").write_text("preserve me")
            with self.assertRaises(RuntimeError): runner.prepare_directory(folder)
            self.assertEqual((folder / "attempt.txt").read_text(), "preserve me")

    def test_source_change_invalidates_registration_before_any_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "out"
            out.mkdir()
            (root / "source.py").write_text("original")
            registration = {"sources_sha256": {"source.py": runner.digest(root / "source.py")}, "reservation": {"id": 1}}
            signature = hashlib.sha256(json.dumps(registration, sort_keys=True).encode()).hexdigest()
            registration["registration_signature"] = signature
            runner.write_json(out / "registration.json", registration)
            runner.write_json(out / "reservation.json", registration["reservation"])
            self.assertEqual(runner.load_registration(root, out)["registration_signature"], signature)
            (root / "source.py").write_text("modified after registration")
            with self.assertRaises(ValueError): runner.load_registration(root, out)

    def test_reserve_score_fails_before_loader_import_if_weights_are_not_frozen(self):
        with patch.object(runner, "validate_final", side_effect=ValueError("not all weights frozen")) as gate:
            with self.assertRaisesRegex(ValueError, "not all weights"):
                runner.score(Path("/unused"), Path("/unused"), {}, None)
            gate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
