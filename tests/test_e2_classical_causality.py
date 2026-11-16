"""Synthetic detectors/forecast ordering; no fitted or actual classical model."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_classical_causality as audit
import e2_reduced_final as final
import e2_temporal_controls as controls
from e2_arima_prediction_compat import install_prediction_raw_alias


class PrefixVariantTests(unittest.TestCase):
    def history(self):
        return pd.Series([.01, .02, np.nan, .04, .05, .06],
                         index=pd.date_range("2025-01-01", periods=6))

    def test_future_change_starts_at_unobserved_same_day_and_preserves_past(self):
        history = self.history()
        original = history.copy()
        target = pd.Timestamp("2025-01-04")
        result = audit.prefix_variant(history, target, "future_only_perturbed")
        pd.testing.assert_series_equal(result.loc[:"2025-01-03"], history.loc[:"2025-01-03"])
        self.assertEqual(result.loc[target], history.loc[target] * 1.37 + .0001)
        self.assertTrue(np.isnan(result.iloc[2]))
        pd.testing.assert_series_equal(history, original)

    def test_truncation_keeps_calendar_and_missing_past_through_t_minus_one(self):
        result = audit.prefix_variant(self.history(), pd.Timestamp("2025-01-04"), "truncated")
        self.assertEqual(result.index.tolist(), pd.date_range("2025-01-01", periods=3).tolist())
        self.assertTrue(np.isnan(result.iloc[-1]))

    def test_unknown_branch_does_not_silently_select_an_alternative(self):
        with self.assertRaises(ValueError):
            audit.prefix_variant(self.history(), pd.Timestamp("2025-01-04"), "new_branch")


class ComparisonGuardTests(unittest.TestCase):
    def test_even_tiny_prefix_change_fails_exact_criterion(self):
        dates = ["2025-01-01", "2025-01-02"]
        reference = dict(zip(dates, [.01, .02]))
        self.assertTrue(audit.exact_prefix(reference, reference.copy(), dates, np)["passed"])
        changed = dict(zip(dates, [.01 + 1e-15, .02]))
        self.assertFalse(audit.exact_prefix(reference, changed, dates, np)["passed"])

    def test_missing_and_nonfinite_forecasts_fail(self):
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                audit.exact_prefix({"d": .01}, {"d": value}, ["d"], np)
        with self.assertRaises(KeyError):
            audit.exact_prefix({"d": .01}, {}, ["d"], np)

    def test_actual_parameter_change_is_not_hidden_by_unchanged_receipt(self):
        model = SimpleNamespace(_result=SimpleNamespace(params=np.array([1., 2.])),
            receipt={"parameters": [1., 2.], "order": [2, 0, 1], "fit_cutoff": "2024-10-31"}, scale=.02)
        before = audit.parameter_record(model, np)
        model._result.params[0] += .001
        self.assertNotEqual(before, audit.parameter_record(model, np))

    def test_fit_is_unconditionally_forbidden(self):
        with self.assertRaises(RuntimeError):
            audit.forbid_fit(object(), [1., 2.], order=(2, 0, 1))


class OriginalHelperContractTests(unittest.TestCase):
    def test_walk_persists_each_forecast_before_same_day_observation(self):
        receipt = install_prediction_raw_alias(audit.ROOT, controls)
        self.assertFalse(receipt["numeric_predictions_changed"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            events = []

            class FakeFrozenModel:
                def __init__(self):
                    self.current = pd.Timestamp("2024-12-30")
                    self.origin = pd.Timestamp("2024-11-08")
                    self.previous = .01

                def forecast(self, target):
                    events.append(("forecast", target))
                    return controls.Prediction(target.isoformat(), self.previous, self.previous, False)

                def observe(self, target, value):
                    path = output / (target.strftime("%Y-%m-%d") + ".json")
                    if not path.is_file():
                        raise AssertionError("Same-day observation revealed before durable forecast")
                    events.append(("observe", target))
                    self.current, self.previous = target, value

            history = pd.Series([.02, 99.], index=pd.date_range("2024-12-31", periods=2))
            values = final._arima_walk(FakeFrozenModel(), history, "2025-01-01", output)
            self.assertEqual(values, {"2024-12-31": .01, "2025-01-01": .02})
            self.assertEqual([e[0] for e in events], ["forecast", "observe", "forecast", "observe"])
            saved = audit.read_json(output / "2025-01-01.json")
            self.assertEqual(saved["raw_prediction"], .02)
            self.assertFalse(saved["coefficients_refitted"])

    def test_naive_helpers_use_calendar_lags_with_unobserved_future_present(self):
        dates = pd.date_range("2024-12-20", periods=25)
        history = pd.Series(np.linspace(.01, .03, len(dates)), index=dates)
        origin = pd.Timestamp("2025-01-01")
        for lag in (1, 7):
            expected = final._naive_predictions(history, [origin], lag)
            for branch in ("truncated", "future_only_perturbed"):
                observed = final._naive_predictions(audit.prefix_variant(history, origin, branch), [origin], lag)
                self.assertEqual(expected, observed)

    def test_changed_bound_file_fails_and_outputs_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            audit.write_new(path, {"state": "first"})
            record = audit.binding(path)
            audit.verify(record)
            with self.assertRaises(FileExistsError):
                audit.write_new(path, {"state": "second"})
            path.write_text("changed")
            with self.assertRaises(ValueError):
                audit.verify(record)


if __name__ == "__main__":
    unittest.main()
