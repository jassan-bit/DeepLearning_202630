"""Synthetic mechanism tests only: no training, HPO/NAS or final data access."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_real_executor as e


def fixture(days=400):
    calendar = pd.date_range("2023-01-01", periods=days, tz="UTC")
    x = np.tile([10., 12., 9., 11., 20.], (days, 1))
    y = 1 + .1 * np.sin(np.arange(days) / 3) + np.arange(days) / 10000
    return calendar, x, y


class MemoryJournal:
    def __init__(self, fail_kind=None, fail_date=None):
        self.events = []
        self.fail_kind, self.fail_date = fail_kind, fail_date

    def append(self, kind, **payload):
        if kind == self.fail_kind and (self.fail_date is None or payload.get("date") == self.fail_date):
            raise OSError("injected persistence failure")
        token = str(len(self.events))
        self.events.append({"kind": kind, "token": token, **copy.deepcopy(payload)})
        return token


class StatefulDouble:
    """Fixed deterministic clock, not ARIMA and never estimated from data."""
    def __init__(self, cutoff, journal):
        self.current, self.journal = cutoff, journal
        self.pending = None
        self.forecasts, self.observations = [], []
        self.parameters = "fixed-test-double"

    def predict(self, day, x):
        assert x is None
        assert day == self.current + e.DAY and self.pending is None
        self.pending = day
        self.forecasts.append(day)
        return 1.

    def observe(self, day, value):
        assert day == self.current + e.DAY
        if self.pending is not None:
            assert self.pending == day
            assert self.journal.events[-1]["kind"] == "observation"
        else:
            assert self.journal.events[-1]["kind"] == "bridge_observation"
        self.current, self.pending = day, None
        self.observations.append((day, value))

    def system(self):
        return e.FrozenSystem("ARIMA-double", "state", self.predict, self.observe,
                              lambda: self.parameters, clip_nonnegative=True)


class CalendarAndPlanning(unittest.TestCase):
    def setUp(self):
        self.calendar, self.x, self.y = fixture(731)
        self.support = e.support_frame(self.calendar, self.x, self.y)

    def test_utc_dates_and_equivalent_offset(self):
        self.assertEqual(e.utc_day("2024-01-01"), e.utc_day("2023-12-31T19:00-05:00"))
        for value in (1, True, np.int64(3), "2024-01-01T00:00-05:00", "2024-01-01 12:00",
                      pd.Timestamp("2024-01-01"), pd.NaT):
            with self.subTest(value=value), self.assertRaises(ValueError):
                e.utc_day(value)

    def test_calendar_rejects_compression_duplicates_and_disorder(self):
        for index in ([], self.calendar.delete(20), self.calendar[::-1],
                      self.calendar.insert(2, self.calendar[2]), self.calendar.tz_localize(None)):
            with self.subTest(index=str(index)[:30]), self.assertRaises(ValueError):
                e.dated_index(index, consecutive=True)

    def test_B_exact_dates_include_leap_day_and_boundaries(self):
        plan = e.temporal_plan(self.support, "2024-04-01", "2024-04-30", scenario="B")
        inner, outer = plan["inner"], plan["outer"]
        self.assertEqual(inner.cutoff, e.utc_day("2023-12-25"))
        self.assertEqual(inner.origin, e.utc_day("2024-01-02"))
        self.assertEqual(inner.evaluation_dates[-1], e.utc_day("2024-03-24"))
        self.assertEqual(len(inner.evaluation_dates), 83)
        self.assertIn(e.utc_day("2024-02-29"), inner.evaluation_dates)
        self.assertEqual(len(inner.bridge_dates), 7)
        self.assertEqual(outer.cutoff, e.utc_day("2024-03-24"))
        self.assertEqual(outer.bridge_dates[-1], e.utc_day("2024-03-31"))

    def test_A_has_no_nested_selection_and_no_implicit_scenario(self):
        plan = e.temporal_plan(self.support, "2024-01-01", "2024-01-31", scenario="A")
        self.assertIsNone(plan["inner"])
        self.assertIn("not_nested_OOF", plan["selection_kind"])
        with self.assertRaises(ValueError):
            e.temporal_plan(self.support, "2024-01-01", "2024-01-31", scenario=None)

    def test_gaps_do_not_move_cuts_or_reduce_neural_for_comparator_lags(self):
        self.x[80] = np.nan
        support = e.support_frame(self.calendar, self.x, self.y)
        plan = e.block_plan(support, "2024-01-01", "2024-01-31", name="january")
        self.assertEqual(plan.cutoff, e.utc_day("2023-12-24"))
        self.assertIn(self.calendar[110], plan.neural_training_dates)
        self.assertFalse(support.common.iloc[110])
        self.assertNotIn(self.calendar[80], plan.target_training_dates)

    def test_unsupported_final_never_becomes_zero_count_certification(self):
        with self.assertRaisesRegex(ValueError, "not certified"):
            e.block_plan(self.support, "2025-01-01", "2026-08-31", name="final")

    def test_support_lag_and_common_mutation_rejected(self):
        for column in ("D7_target", "lag28_target", "common"):
            support = self.support.copy()
            support.loc[self.calendar[200], column] = False
            with self.assertRaises(ValueError):
                e.block_plan(support, "2024-01-01", "2024-01-31", name="bad")

    def test_scenario_A_fold_names_are_unique(self):
        plans = [e.temporal_plan(self.support, f"2024-{month:02d}-01", f"2024-{month:02d}-20", scenario="A")
                 for month in range(1, 11)]
        self.assertEqual(len({p["outer"].name for p in plans}), 10)


class DatedMASE(unittest.TestCase):
    def test_missing_calendar_pair_not_compressed(self):
        y = pd.Series([1., 3., 103.], index=[e.utc_day(d) for d in ("2024-01-01", "2024-01-02", "2024-01-04")])
        self.assertEqual(e.mase_scale(y, "2024-01-04"), {"value": 2., "pairs": 1, "reason": None})

    def test_missing_invalid_and_zero_denominator(self):
        index = pd.date_range("2024-01-01", periods=6, tz="UTC")
        y = pd.Series([1., 3., np.nan, -1., 10., 14.], index=index)
        self.assertEqual(e.mase_scale(y, index[-1])["value"], 3.)
        self.assertEqual(e.mase_scale(y, index[-1])["pairs"], 2)
        self.assertEqual(e.mase_scale(y * 0, index[-1])["reason"], "zero_training_scale")
        self.assertEqual(e.mase_scale(y.iloc[::2], index[-1])["reason"], "no_consecutive_training_pairs")

    def test_future_labels_never_change_scale(self):
        index = pd.date_range("2024-01-01", periods=6, tz="UTC")
        y = pd.Series([1., 2., 4., 8., 16., 32.], index=index)
        before = e.mase_scale(y, index[2])
        y.iloc[3:] = [1e20, np.inf, -1]
        self.assertEqual(e.mase_scale(y, index[2]), before)
        object_y = y.astype(object)
        object_y.iloc[3:] = "not a training label"
        self.assertEqual(e.mase_scale(object_y, index[2]), before)

    def test_shared_scale_and_undefined_metric(self):
        a = e.metrics([1, 2], [2, 4], {"value": 2., "pairs": 10, "reason": None})
        self.assertEqual(a["MASE"], .75)
        b = e.metrics([1], [2], {"value": None, "pairs": 0, "reason": "no_consecutive_training_pairs"})
        self.assertIsNone(b["MASE"])
        json.dumps(b, allow_nan=False)
        for prediction in ([np.nan], [np.inf], []):
            with self.assertRaises(ValueError):
                e.metrics([1], prediction, b["MASE_scale"])


class ReplayTests(unittest.TestCase):
    def run_case(self, *, bad_bar=None, bad_target=None, journal=None, neural=None, end=240):
        calendar, x, y = fixture()
        if bad_bar is not None:
            x[bad_bar] = np.nan
        if bad_target is not None:
            y[bad_target] = np.nan
        replay = e.DatedReplay(calendar, x, y)
        plan = e.block_plan(replay.support, calendar[200], calendar[end], name="test")
        journal = journal or MemoryJournal()
        state = StatefulDouble(plan.cutoff, journal)
        systems = [e.FrozenSystem("Neural-double", "window", neural or (lambda day, x: 1.),
                                  fingerprint=lambda: "fixed"), state.system()]
        systems += [e.FrozenSystem(f"Naive{lag}", "naive", lag=lag) for lag in (1, 7, 14, 28)]
        result = e.run_frozen_block(plan, replay, systems, journal)
        return result, replay, state, journal, plan

    def test_every_day_preserved_when_inputs_and_targets_are_missing(self):
        result, replay, state, journal, plan = self.run_case(bad_bar=204)
        self.assertEqual(len(state.forecasts), 41)
        self.assertEqual(len(state.observations), 48)
        self.assertEqual(state.current, plan.evaluation_dates[-1])
        self.assertEqual(len(replay.revealed), 48)
        self.assertLess(result["scored_days"], result["calendar_days"])
        observed = dict(state.observations)
        for i in range(204, 212):
            self.assertIsNone(observed[replay.calendar[i]])
        predictions = [row for row in journal.events if row["kind"] == "predictions"]
        unavailable = predictions[5]["systems"]["Neural-double"]
        self.assertEqual(unavailable["reason"], "D7_window_unavailable")
        self.assertIsNone(unavailable["value"])
        for i, event in enumerate(journal.events):
            if event["kind"] == "predictions":
                self.assertEqual(journal.events[i+1]["kind"], "observation")
                self.assertEqual(journal.events[i+1]["prediction_event_sha256"], event["token"])

    def test_lag_absence_does_not_prevent_valid_state_observation(self):
        result, replay, state, journal, plan = self.run_case(bad_target=180)
        event = next(row for row in journal.events if row.get("date") == replay.calendar[208].isoformat()
                     and row["kind"] == "predictions")
        self.assertFalse(event["eligible"])
        self.assertIsNone(event["systems"]["Naive28"]["value"])
        self.assertIsNotNone(dict(state.observations)[replay.calendar[208]])

    def test_forecast_write_failure_never_reveals_or_updates_day(self):
        calendar, x, y = fixture()
        replay = e.DatedReplay(calendar, x, y)
        plan = e.block_plan(replay.support, calendar[200], calendar[202], name="test")
        journal = MemoryJournal("predictions")
        state = StatefulDouble(plan.cutoff, journal)
        with self.assertRaisesRegex(OSError, "injected"):
            e.run_frozen_block(plan, replay, [state.system()], journal)
        self.assertEqual(replay.revealed[-1], calendar[199])
        self.assertEqual(state.current, calendar[199])
        self.assertFalse(any(row["kind"] == "block_completed" for row in journal.events))

    def test_observation_write_failure_prevents_state_update(self):
        calendar, x, y = fixture()
        replay = e.DatedReplay(calendar, x, y)
        plan = e.block_plan(replay.support, calendar[200], calendar[202], name="test")
        journal = MemoryJournal("observation")
        state = StatefulDouble(plan.cutoff, journal)
        with self.assertRaises(OSError):
            e.run_frozen_block(plan, replay, [state.system()], journal)
        self.assertEqual(state.current, calendar[199])
        self.assertEqual(replay.revealed[-1], calendar[200])

    def test_nonfinite_prediction_fails_even_on_excluded_day(self):
        journal = MemoryJournal()
        with self.assertRaises(e.ExecutionFailure):
            self.run_case(bad_target=200, journal=journal, neural=lambda d,x: np.inf)
        self.assertFalse(any(row["kind"] == "observation" for row in journal.events))

    def test_replay_single_use_and_observation_order(self):
        calendar, x, y = fixture()
        replay = e.DatedReplay(calendar, x, y)
        with self.assertRaises(ValueError):
            replay.reveal(calendar[200])
        replay.begin(calendar[192])
        for bad in (calendar[192], calendar[194]):
            with self.assertRaises(ValueError):
                replay.reveal(bad)
        replay.reveal(calendar[193])
        with self.assertRaises(ValueError):
            replay.begin(calendar[192])
        with self.assertRaises(ValueError):
            replay.reveal(calendar[193])

    def test_window_is_past_only_and_callbacks_cannot_modify_source(self):
        calendar, x, y = fixture()
        x[:, 4] = np.arange(len(x))
        replay = e.DatedReplay(calendar, x, y)
        replay.begin(calendar[199])
        window = replay.window(calendar[200])
        np.testing.assert_array_equal(window[:, 4], np.arange(193, 200))
        window[:] = 0
        self.assertTrue((replay.window(calendar[200])[:, 0] == 10).all())
        with self.assertRaises(ValueError):
            replay.window(calendar[201])

    def test_future_magnitudes_do_not_change_earlier_predictions(self):
        calendar, x, y = fixture()
        a = e.DatedReplay(calendar, x, y)
        x[210:] *= 2
        y[210:] *= 10
        b = e.DatedReplay(calendar, x, y)
        journals = []
        for replay in (a, b):
            plan = e.block_plan(replay.support, calendar[200], calendar[209], name="test")
            journal = MemoryJournal()
            e.run_frozen_block(plan, replay, [e.FrozenSystem("P", "naive", lag=1)], journal)
            journals.append(journal.events)
        self.assertEqual(journals[0], journals[1])

    def test_frozen_parameters_cannot_change(self):
        calendar, x, y = fixture()
        replay = e.DatedReplay(calendar, x, y)
        plan = e.block_plan(replay.support, calendar[200], calendar[201], name="test")
        journal = MemoryJournal()
        state = StatefulDouble(plan.cutoff, journal)
        def mutated(day, value):
            state.observe(day, value)
            state.parameters = str(day)
        system = replace(state.system(), observe=mutated)
        with self.assertRaisesRegex(e.ExecutionFailure, "Frozen"):
            e.run_frozen_block(plan, replay, [system], journal)

    def test_raw_negative_and_clipping_are_prespecified(self):
        calendar, x, y = fixture()
        replay = e.DatedReplay(calendar, x, y)
        plan = e.block_plan(replay.support, calendar[200], calendar[201], name="test")
        journal = MemoryJournal()
        system = e.FrozenSystem("Ridge-double", "window", lambda d,x: -1.,
                                fingerprint=lambda: "fixed", clip_nonnegative=True)
        e.run_frozen_block(plan, replay, [system], journal)
        value = next(row for row in journal.events if row["kind"] == "predictions")["systems"][system.name]
        self.assertEqual((value["raw"], value["value"], value["truncated"]), (-1., 0., True))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        calendar, x, y = fixture()
        support = e.support_frame(calendar, x, y)
        self.plans = [e.block_plan(support, calendar[200], calendar[201], name="first"),
                      e.block_plan(support, calendar[210], calendar[210], name="second")]
        self.outcomes = {}
        for candidate, errors in (("c1", [[1., 1.], [9.]]), ("c2", [[4., 4.], [4.]])):
            self.outcomes[candidate] = {"status": "completed", "folds": {
                plan.name: {"support_sha256": plan.support_sha256, "plan_sha256": e.canonical_hash(plan.record()),
                            "dates": e.date_strings(plan.evaluation_dates), "squared_errors": values}
                for plan, values in zip(self.plans, errors)}}

    def test_date_weighted_pooling_not_mean_of_fold_rmse(self):
        result = e.select_candidate(["c1", "c2"], self.plans, self.outcomes, scenario="A")
        self.assertEqual(result["winner"], "c1")
        self.assertAlmostEqual(result["candidate_ledger"]["c1"]["RMSE"], np.sqrt(11/3))

    def test_fixed_tie_order(self):
        self.outcomes["c2"] = copy.deepcopy(self.outcomes["c1"])
        self.assertEqual(e.select_candidate(["c2", "c1"], self.plans, self.outcomes, scenario="A")["winner"], "c2")

    def test_incomplete_duplicate_or_changed_support_rejected(self):
        for mutation in ("dates", "support_sha256", "squared_errors"):
            outcomes = copy.deepcopy(self.outcomes)
            outcomes["c1"]["folds"]["first"][mutation] = [] if mutation != "support_sha256" else "wrong"
            with self.assertRaises(ValueError):
                e.select_candidate(["c1", "c2"], self.plans, outcomes, scenario="A")
        with self.assertRaises(ValueError):
            e.select_candidate(["c1", "c1"], self.plans, self.outcomes, scenario="A")
        with self.assertRaises(ValueError):
            e.select_candidate(["c1", "c2"], [self.plans[0], self.plans[0]], self.outcomes, scenario="A")

    def test_failed_candidate_retained_without_date_rescue(self):
        self.outcomes["c1"] = {"status": "failed", "reason": "nonconvergence", "resource_use": {"seconds": 2}}
        result = e.select_candidate(["c1", "c2"], self.plans, self.outcomes, scenario="A")
        self.assertEqual(result["winner"], "c2")
        self.assertEqual(result["candidate_ledger"]["c1"], self.outcomes["c1"])
        self.outcomes["c2"] = copy.deepcopy(self.outcomes["c1"])
        with self.assertRaisesRegex(ValueError, "All candidates failed"):
            e.select_candidate(["c1", "c2"], self.plans, self.outcomes, scenario="A")

    def test_scenarios_cannot_be_mixed_or_chosen_implicitly(self):
        for scenario in (None, "C", "B"):
            with self.assertRaises(ValueError):
                e.select_candidate(["c1", "c2"], self.plans, self.outcomes, scenario=scenario)

    def test_plan_seal_prevents_same_length_mask_changes(self):
        modified = replace(self.plans[0], eligible=(False, True))
        with self.assertRaises(ValueError):
            e.select_candidate(["c1", "c2"], [modified, self.plans[1]], self.outcomes, scenario="A")


class ExistingArimaBoundary(unittest.TestCase):
    def test_existing_filter_api_with_untrained_backend_double(self):
        import e2_temporal_controls as controls
        calendar, x, y = fixture()
        y[201] = np.nan
        replay = e.DatedReplay(calendar, x, y)
        plan = e.block_plan(replay.support, calendar[200], calendar[210], name="test")

        class BackendDouble:
            params = np.array([.1, .2])
            def __init__(self):
                self.observations = []
            def forecast(self, steps):
                assert steps == 1
                return np.array([-.2])
            def extend(self, observation):
                self.observations.append(observation.copy())
                return self

        model = controls.FrozenARIMA.__new__(controls.FrozenARIMA)
        model._result, model.scale = BackendDouble(), 2.
        model.origin, model.current = plan.origin.tz_localize(None), plan.cutoff.tz_localize(None)
        model._pending, model._failure = None, None
        model.receipt, model.predictions, model.observations = {"status": "fitted", "warnings": []}, [], []
        system = e.frozen_arima_system(model, plan)
        journal = MemoryJournal()
        result = e.run_frozen_block(plan, replay, [system], journal)
        self.assertEqual(result["model_fits"], 0)
        self.assertEqual(len(model.predictions), 11)
        self.assertEqual(len(model.observations), 18)
        self.assertEqual(model.current, calendar[210].tz_localize(None))
        self.assertIsNone(model.observations[8]["value"])
        self.assertEqual(model.predictions[0]["raw"], -.4)
        self.assertEqual(model.predictions[0]["value"], 0.)
        for observation in model._result.observations:
            self.assertIsNone(observation.index.tz)
        with self.assertRaises(ValueError):
            e.frozen_arima_system(model, plan)  # Cannot reuse a consumed state for refit.


class DevelopmentReadBoundary(unittest.TestCase):
    def make_root(self, path):
        calendar, x, y = fixture(731)
        root = Path(path)
        market = pd.DataFrame(x, columns=e.availability.FEATURES)
        market.insert(0, "market_day_utc", calendar.astype(str))
        targets = pd.DataFrame({"market_day_utc": calendar.astype(str), "sigma_7": y})
        hashes = {}
        for name, frame in zip(e.availability.SOURCES, (market, targets)):
            source = root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(source, index=False)
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        protocol = root / e.availability.PROTOCOL
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text(json.dumps({"sources": hashes}))
        return root

    def test_preflight_reads_only_registered_inputs_and_performs_no_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            actual_csv, actual_digest = e.pd.read_csv, e.availability.digest
            reads = []
            allowed = {root / n for n in (e.availability.PROTOCOL, *e.availability.SOURCES)}
            def digest(path):
                self.assertIn(Path(path), allowed)
                reads.append(Path(path))
                return actual_digest(path)
            def csv(path):
                self.assertIn(Path(path), {root / n for n in e.availability.SOURCES})
                reads.append(Path(path))
                return actual_csv(path)
            with patch.object(e.pd, "read_csv", side_effect=csv), patch.object(e.availability, "digest", side_effect=digest):
                result = e.development_preflight(root)
            self.assertEqual(set(reads), allowed)
            self.assertEqual(result["final_files_opened"], 0)
            self.assertIsNone(result["scenario_selected"])
            self.assertIsNone(result["final_support"]["common_count"])
            self.assertEqual(result["model_fits"], 0)
            self.assertEqual(sum(sum(p["outer"]["eligible"]) for p in result["scenarios"]["A"]), 305)

    def test_hash_mismatch_stops_before_decoding_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            source = root / e.availability.SOURCES[0]
            source.write_text(source.read_text() + "\n")
            with patch.object(e.pd, "read_csv") as reader, self.assertRaisesRegex(ValueError, "SHA256"):
                e.development_preflight(root)
            reader.assert_not_called()

    def test_registered_source_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            source = root / e.availability.SOURCES[0]
            renamed = source.with_name("synthetic-copy.csv")
            source.rename(renamed)
            source.symlink_to(renamed)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                e.development_preflight(root)


class DurableAndSafetyTests(unittest.TestCase):
    def test_durable_journal_chain_exclusive_output_and_fsync(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic"
            journal = e.DatedJournal(path)
            with patch.object(e.os, "fsync", wraps=e.os.fsync) as sync:
                token = journal.append("test", payload=1)
                self.assertEqual(sync.call_count, 1)
            journal.close()
            row = json.loads((path / "events.jsonl").read_text())
            self.assertEqual(row.pop("event_sha256"), token)
            self.assertEqual(e.canonical_hash(row), token)
            with self.assertRaises(FileExistsError):
                e.DatedJournal(path)

    def test_real_fsync_failure_stops_journal_and_reveal(self):
        calendar, x, y = fixture()
        replay = e.DatedReplay(calendar, x, y)
        plan = e.block_plan(replay.support, calendar[200], calendar[201], name="test")
        with tempfile.TemporaryDirectory() as directory:
            journal = e.DatedJournal(Path(directory) / "synthetic")
            original_append = journal.append
            def append(kind, **payload):
                if kind == "predictions":
                    with patch.object(e.os, "fsync", side_effect=OSError("fsync injected")):
                        return original_append(kind, **payload)
                return original_append(kind, **payload)
            journal.append = append
            with self.assertRaisesRegex(OSError, "fsync injected"):
                e.run_frozen_block(plan, replay, [e.FrozenSystem("P", "naive", lag=1)], journal)
            self.assertEqual(replay.revealed[-1], calendar[199])
            with self.assertRaises(e.ExecutionFailure):
                original_append("retry")
            journal.close()

    def test_no_training_or_final_loader_imports(self):
        import ast
        source = Path(e.__file__).read_text()
        tree = ast.parse(source)
        names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        names += [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
        self.assertFalse(any((name or "").startswith(("torch", "statsmodels", "e2_final", "e2_common_executor")) for name in names))
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and node.func.attr in ("fit", "backward", "step") for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
