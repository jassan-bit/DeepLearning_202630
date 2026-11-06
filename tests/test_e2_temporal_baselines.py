"""Causality and calendar tests on synthetic sigma labels; no market I/O."""
from datetime import date, timedelta
from dataclasses import replace
import importlib.util
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2_temporal_baselines import (ArimaConfig, FitWindow, FixedParameterARIMA,
                                  benchmark_nonnegative, describe, local_arima_grid,
                                  naive_forecast)

DAY = timedelta(days=1)
START = date(2020, 1, 1)


class PastOnlyHistory(dict):
    def __init__(self, data, forbidden_from):
        super().__init__(data)
        self.forbidden_from = forbidden_from
        self.accessed = []

    def get(self, key, default=None):
        if key >= self.forbidden_from:
            raise AssertionError("Attempt to access an unavailable outcome")
        self.accessed.append(key)
        return super().get(key, default)


def synthetic_history(n=180):
    rng = random.Random(811)
    data, current = {}, .02
    for i in range(n):
        current = .02 + .75 * (current - .02) + rng.gauss(0, .0006)
        data[START + i * DAY] = current
    return data


class TrivialReferencesTests(unittest.TestCase):
    def test_actual_support_and_calendar_span_are_distinct(self):
        origin = START + 30 * DAY
        history = PastOnlyHistory(synthetic_history(), origin)
        a, b = naive_forecast(history, origin), naive_forecast(history, origin, 7)
        self.assertEqual(a.effective_close_start, origin - 8 * DAY)
        self.assertEqual(a.effective_close_end, origin - DAY)
        self.assertEqual(a.history_span_days, 8)
        self.assertEqual(b.effective_close_start, origin - 14 * DAY)
        self.assertEqual(b.effective_close_end, origin - 7 * DAY)
        self.assertEqual(b.history_span_days, 14)
        self.assertEqual((b.effective_close_end - b.effective_close_start).days + 1, 8)
        self.assertEqual(history.accessed, [origin - DAY, origin - 7 * DAY])

    def test_missing_calendar_label_is_not_previous_retained_row(self):
        origin = START + 30 * DAY
        history = synthetic_history()
        del history[origin - 7 * DAY]
        out = naive_forecast(history, origin, 7)
        self.assertFalse(out.eligible)
        self.assertIsNone(out.value)
        self.assertTrue(naive_forecast(history, origin).eligible)

    def test_future_values_have_no_effect_and_nan_stays_missing(self):
        origin = START + 40 * DAY
        history = synthetic_history()
        before = naive_forecast(history, origin)
        history[origin] = -99999
        history[origin + DAY] = float("inf")
        self.assertEqual(before, naive_forecast(history, origin))
        history[origin - DAY] = float("nan")
        self.assertFalse(naive_forecast(history, origin).eligible)

    def test_invalid_labels_lags_and_unregistered_embargo_rejected(self):
        origin = START + 40 * DAY
        with self.assertRaises(ValueError): naive_forecast({}, origin, 0)
        with self.assertRaises(ValueError): naive_forecast({origin - DAY: -1}, origin)
        with self.assertRaises(ValueError): FitWindow(START, origin - 7 * DAY, origin)
        with self.assertRaises(ValueError): FitWindow(START, origin - 8 * DAY, origin, 1)

    def test_grid_is_configuration_only_with_no_selection_claim(self):
        self.assertEqual(len(local_arima_grid()), 12)
        self.assertEqual(len({c.order for c in local_arima_grid()}), 12)
        self.assertFalse(describe()["common_HPO_budget_registered_by_this_module"])
        self.assertEqual(describe()["real_Bitcoin_fits_executed_by_this_module"], 0)

    def test_registered_zero_link_preserves_raw_and_ineligible_status(self):
        origin = START + 40 * DAY
        raw = replace(naive_forecast(synthetic_history(), origin), model="ARIMA",
                      value=-.002, raw_value=-.002)
        out = benchmark_nonnegative(raw)
        self.assertEqual(out.value, 0.)
        self.assertEqual(out.raw_value, -.002)
        self.assertTrue(out.eligible)
        self.assertEqual(benchmark_nonnegative(out), out)
        missing = benchmark_nonnegative(replace(raw, value=None, eligible=False,
                                                reason="latest_past_label_missing"))
        self.assertFalse(missing.eligible)
        self.assertIsNone(missing.value)
        self.assertEqual(missing.raw_value, -.002)
        self.assertEqual(missing.reason, "latest_past_label_missing")

    def test_zero_link_rejects_nonfinite_failures_instead_of_creating_zero(self):
        raw = naive_forecast(synthetic_history(), START + 40 * DAY)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    benchmark_nonnegative(replace(raw, value=invalid))
                with self.assertRaises(ValueError):
                    benchmark_nonnegative(replace(raw, raw_value=invalid))
        with self.assertRaises(ValueError):
            benchmark_nonnegative(replace(raw, value=None, eligible=True))


@unittest.skipUnless(importlib.util.find_spec("statsmodels") is not None,
                     "Run ARIMA tests in the pinned EDA environment")
class ArimaCalendarTests(unittest.TestCase):
    def setUp(self):
        self.origin = START + 128 * DAY
        self.window = FitWindow(START, self.origin - 8 * DAY, self.origin)
        self.history = synthetic_history()

    def test_fit_and_states_only_read_past_fixed_parameters_never_refit(self):
        history = PastOnlyHistory(self.history, self.origin)
        model = FixedParameterARIMA(history, self.window, ArimaConfig((1, 0, 0)))
        self.assertTrue(all(d <= self.window.fit_cutoff for d in history.accessed))
        params = model._result.params.copy()
        out = model.predict(history, self.origin)
        self.assertTrue(out.eligible)
        self.assertTrue(all(d < self.origin for d in history.accessed))
        history.forbidden_from = self.origin + DAY
        next_out = model.predict(history, self.origin + DAY)
        self.assertTrue(next_out.eligible)
        self.assertEqual(model.fit_count, 1)
        self.assertEqual(model.state_updates, 8)
        self.assertEqual(list(params), list(model._result.params))
        self.assertFalse(model.receipt()["refit_after_cutoff"])
        with self.assertRaises(ValueError): model.predict(history, self.origin)

    def test_future_mutation_cannot_change_fit_or_forecast(self):
        altered = dict(self.history)
        for day in list(altered):
            if day >= self.origin:
                altered[day] = float("inf")
        a = FixedParameterARIMA(self.history, self.window)
        b = FixedParameterARIMA(altered, self.window)
        self.assertAlmostEqual(a.predict(self.history, self.origin).value,
                               b.predict(altered, self.origin).value, places=13)

    def test_calendar_holes_preserve_training_and_state_lengths(self):
        history = dict(self.history)
        del history[START + 50 * DAY]
        del history[self.origin - 4 * DAY]
        model = FixedParameterARIMA(history, self.window, ArimaConfig((0, 0, 0)))
        self.assertEqual(model.missing_training_labels, 1)
        self.assertEqual(len(model.training_dates), 121)
        self.assertTrue(model.predict(history, self.origin).eligible)
        self.assertEqual(model.state_updates, 7)
        self.assertEqual(len(model._result.model.endog), 128)

    def test_latest_past_hole_is_ineligible_not_silently_imputed(self):
        history = dict(self.history)
        del history[self.origin - DAY]
        model = FixedParameterARIMA(history, self.window, ArimaConfig((0, 0, 0)))
        out = model.predict(history, self.origin)
        self.assertFalse(out.eligible)
        self.assertIsNone(out.value)
        self.assertEqual(out.reason, "latest_past_label_missing")

    def test_insufficient_training_rejected_before_fitting(self):
        with self.assertRaises(ValueError): FixedParameterARIMA({}, self.window)
        with self.assertRaises(ValueError): ArimaConfig((0, 2, 0))


if __name__ == "__main__":
    unittest.main()
