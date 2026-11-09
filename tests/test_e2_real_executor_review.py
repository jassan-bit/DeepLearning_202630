"""Independent synthetic adversarial QA; no fits or data-file access."""
from dataclasses import replace
import copy
import unittest
import numpy as np
from test_e2_real_executor import fixture, MemoryJournal, StatefulDouble, e


class IndependentReview(unittest.TestCase):
    def setup_case(self):
        calendar, x, y = fixture()
        replay = e.DatedReplay(calendar, x, y)
        plan = e.block_plan(replay.support, calendar[200], calendar[202], name='test')
        return calendar, replay, plan, MemoryJournal()

    def test_bridge_parameter_mutation_stops_before_first_forecast(self):
        calendar, replay, plan, journal = self.setup_case()
        state = StatefulDouble(plan.cutoff, journal)
        def observe(day, value):
            state.observe(day, value)
            state.parameters = 'changed'
        with self.assertRaises(e.ExecutionFailure):
            e.run_frozen_block(plan, replay, [replace(state.system(), observe=observe)], journal)
        self.assertEqual(state.forecasts, [], 'Changed parameters must be caught before inference')
        self.assertNotIn(calendar[200], replay.revealed)

    def test_mutable_fingerprint_does_not_alias_snapshot(self):
        calendar, replay, plan, journal = self.setup_case()
        state = StatefulDouble(plan.cutoff, journal)
        parameters = {'weight': 1.0}
        def observe(day, value):
            state.observe(day, value)
            parameters['weight'] += 1.0
        system = replace(state.system(), observe=observe, fingerprint=lambda: parameters)
        with self.assertRaises(e.ExecutionFailure):
            e.run_frozen_block(plan, replay, [system], journal)

    def test_selection_rejects_truncated_plan_mask(self):
        _, _, plan, _ = self.setup_case()
        broken = replace(plan, eligible=(True,))
        outcome = {'c': {'status': 'completed', 'folds': {'test': {
            'support_sha256': plan.support_sha256, 'plan_sha256': e.canonical_hash(plan.record()),
            'dates': e.date_strings(plan.evaluation_dates[:1]),
            'squared_errors': [1.0]}}}}
        with self.assertRaises(ValueError):
            e.select_candidate(['c'], [broken], outcome, scenario="A")

    def test_mase_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            e.metrics([0.0], [1.0], {'value': np.nextafter(0., 1.), 'pairs': 1, 'reason': None})

    def test_input_arrays_are_copied(self):
        calendar, x, y = fixture()
        replay = e.DatedReplay(calendar, x, y)
        x[:] = np.nan
        y[:] = np.nan
        self.assertTrue(replay.support.common.iloc[200])
        replay.begin(calendar[199])
        self.assertIsNotNone(replay.window(calendar[200]))
        self.assertIsNotNone(replay.reveal(calendar[200]))

    def test_selection_rejects_reordered_dates(self):
        _, _, plan, _ = self.setup_case()
        outcome = {'c': {'status': 'completed', 'folds': {'test': {
            'support_sha256': plan.support_sha256, 'plan_sha256': e.canonical_hash(plan.record()),
            'dates': e.date_strings(plan.evaluation_dates[::-1]),
            'squared_errors': [1.0]*3}}}}
        with self.assertRaises(ValueError):
            e.select_candidate(['c'], [plan], outcome, scenario="A")


if __name__ == '__main__':
    unittest.main()
