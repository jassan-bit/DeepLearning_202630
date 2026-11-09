"""Synthetic-only tests. No repository data, benchmark runner or model search."""
import copy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_temporal_controls as c


def synthetic():
    index = pd.date_range("2020-01-01", periods=170, freq="D")
    rng = np.random.default_rng(819)
    return pd.Series(2 + np.cumsum(rng.normal(0, .025, len(index))), index=index)


class CalendarContract(unittest.TestCase):
    def test_periods_exact_dates_and_missing_lag(self):
        y = synthetic()
        t = y.index[-1]
        for m in (1, 7, 14, 28):
            self.assertEqual(c.seasonal_naive(y, t, m).value, y.loc[t - m * c.DAY])
            missing = y.drop(t - m * c.DAY)
            with self.assertRaises(c.ControlFailure):
                c.seasonal_naive(missing, t, m)
        with self.assertRaises(ValueError):
            c.seasonal_naive(y, t, True)

    def test_calendar_not_compressed_invalid_and_zero(self):
        y = synthetic().iloc[:10]
        y.iloc[1:5] = [0, -1, np.inf, np.nan]
        result = c.labels(y.drop(y.index[7]))
        self.assertEqual(len(result), 10)
        self.assertEqual(result.iloc[1], 0)
        self.assertTrue(result.iloc[[2, 3, 4, 7]].isna().all())

    def test_sigma_requires_eight_complete_closes(self):
        prices = pd.Series(np.exp(np.arange(20) / 100), index=pd.date_range("2020-01-01", periods=20))
        quality = pd.Series(True, index=prices.index)
        y = c.sigma7(prices, quality)
        self.assertTrue(y.iloc[:7].isna().all())
        np.testing.assert_allclose(y.iloc[7:], 0, atol=1e-8)
        quality.iloc[10] = False
        y = c.sigma7(prices, quality)
        self.assertTrue(y.iloc[10:18].isna().all())
        self.assertTrue(np.isfinite(y.iloc[18]))
        y = c.sigma7(prices.drop(prices.index[10]), quality)
        self.assertTrue(y.iloc[10:18].isna().all())

    def test_sigma_population_log_returns(self):
        prices = pd.Series(np.exp(np.cumsum([0, .01, -.02, .03, .01, -.01, .04, .02])),
                           index=pd.date_range("2020-01-01", periods=8))
        y = c.sigma7(prices, pd.Series(True, index=prices.index))
        self.assertAlmostEqual(y.iloc[-1], np.std([.01, -.02, .03, .01, -.01, .04, .02], ddof=0))

    def test_common_mask_preserves_all_rows(self):
        y = synthetic()
        targets = y.index[-3:]
        y.loc[targets[1] - 28 * c.DAY] = np.nan
        neural = pd.Series(True, index=y.index)
        neural.loc[targets[2]] = False
        mask = c.common_mask(y, neural, targets)
        self.assertEqual(len(mask), 3)
        self.assertEqual(mask.eligible.tolist(), [True, False, False])
        self.assertEqual(mask.reasons.iloc[1], ("lag_28",))
        self.assertEqual(mask.reasons.iloc[2], ("neural_support",))

    def test_outputs_and_no_silent_fallback(self):
        self.assertEqual(c.output("2020-01-01", -2).value, 0)
        self.assertEqual(c.output("2020-01-01", -2).raw, -2)
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(c.ControlFailure):
                c.output("2020-01-01", value)

    def test_daily_validation(self):
        with self.assertRaises(ValueError):
            c.daily(pd.Series([1, 2], index=["2020-01-01", "2020-01-01"]))
        with self.assertRaises(ValueError):
            c.day("2020-01-01 12:00")
        self.assertEqual(c.day("2020-01-01T00:00Z"), pd.Timestamp("2020-01-01"))

    def test_training_guard_and_explicit_dependency_failure(self):
        y = synthetic()
        with self.assertRaises(c.ControlFailure) as failure:
            c.FrozenARIMA.fit(y.iloc[:10], y.index[12], (0, 1, 0), source_commit="synthetic")
        self.assertEqual(failure.exception.receipt["status"], "failed")
        with self.assertRaises(c.ControlFailure):
            c.FrozenARIMA.fit(y * 0, y.index[140], (0, 1, 0), source_commit="synthetic")
        with patch.object(c, "version", side_effect=PackageNotFoundError):
            with self.assertRaisesRegex(c.ControlFailure, "not installed"):
                c._backend(c.STATSMODELS_VERSION)
        with patch.object(c, "version", return_value="different"):
            with self.assertRaisesRegex(c.ControlFailure, "required"):
                c._backend(c.STATSMODELS_VERSION)


try:
    HAS_PINNED_BACKEND = version("statsmodels") == c.STATSMODELS_VERSION
except PackageNotFoundError:
    HAS_PINNED_BACKEND = False


@unittest.skipUnless(HAS_PINNED_BACKEND, "requires statsmodels==" + c.STATSMODELS_VERSION)
class ARIMAIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y = synthetic()
        cls.origin = cls.y.index[140]
        cls.base = c.FrozenARIMA.fit(cls.y, cls.origin, (0, 1, 0), source_commit="synthetic-test")

    def bridge(self, model, y=None):
        y = self.y if y is None else y
        for date in pd.date_range(model.current + c.DAY, model.origin - c.DAY):
            model.observe(date, y.get(date, np.nan))
        return model

    def test_fit_cutoff_and_future_invariance(self):
        changed = self.y.copy()
        changed.loc[self.origin - 7 * c.DAY:] = 99999
        other = c.FrozenARIMA.fit(changed, self.origin, (0, 1, 0), source_commit="synthetic-test")
        self.assertEqual(self.base.receipt["parameters"], other.receipt["parameters"])
        self.assertEqual(self.base.scale, other.scale)
        self.assertEqual(self.base.current, self.origin - 8 * c.DAY)
        self.assertEqual(self.base.receipt["history_sha256"], other.receipt["history_sha256"])
        original = self.bridge(copy.deepcopy(self.base))
        future = self.y.copy()
        future.loc[self.origin:] = 1e6
        perturbed = self.bridge(copy.deepcopy(other), future)
        self.assertEqual(original.forecast(self.origin), perturbed.forecast(self.origin))

    def test_state_changes_without_refit_and_random_walk_persistence(self):
        original = self.bridge(copy.deepcopy(self.base))
        other = copy.deepcopy(self.base)
        changed = self.y.copy()
        changed.loc[self.origin - c.DAY] += .5
        self.bridge(other, changed)
        a, b = original.forecast(self.origin), other.forecast(self.origin)
        self.assertAlmostEqual(a.value, self.y.loc[self.origin - c.DAY], places=7)
        self.assertAlmostEqual(b.value - a.value, .5, places=7)
        self.assertEqual(original.receipt["parameters"], other.receipt["parameters"])
        self.assertEqual(original.scale, other.scale)

    def test_missing_calendar_and_incremental_prefix_equivalence(self):
        model = copy.deepcopy(self.base)
        reference = copy.deepcopy(self.base._result)
        for i, date in enumerate(pd.date_range(model.current + c.DAY, self.origin - c.DAY)):
            value = np.nan if i in (0, 2, 6) else self.y.loc[date]
            model.observe(date, value)
            obs = pd.Series([value / model.scale], index=pd.date_range(date, periods=1))
            reference = reference.append(obs, refit=False)
            np.testing.assert_allclose(model._result.forecast(1), reference.forecast(1), atol=1e-10)
        self.assertEqual(model.current, self.origin - c.DAY)
        self.assertEqual(len(model.observations), 7)
        self.assertIsNone(model.observations[-1]["value"])
        self.assertTrue(np.isfinite(model.forecast(self.origin).value))

    def test_training_missing_dates_retained(self):
        y = self.y.drop(self.y.index[100])
        # This pinned backend can report nonconvergence on this gapped fixture.
        # Either outcome must preserve the gap and receipt, never retry/drop it.
        try:
            model = c.FrozenARIMA.fit(y, self.origin, (0, 1, 0), source_commit="synthetic-test")
            receipt = model.receipt
            self.assertEqual(model._result.nobs, self.base._result.nobs)
        except c.ControlFailure as failure:
            receipt = failure.receipt
            self.assertFalse(receipt["converged"])
            self.assertEqual(receipt["retries"], 0)
            self.assertIn("Nonconvergence", receipt["error"])
        self.assertEqual(receipt["calendar_count"], self.base.receipt["calendar_count"])
        self.assertEqual(receipt["valid_count"], self.base.receipt["valid_count"] - 1)

    def test_temporal_state_machine_and_pickle_roundtrip(self):
        model = copy.deepcopy(self.base)
        with self.assertRaises(ValueError):
            model.forecast(self.origin)
        with self.assertRaises(ValueError):
            model.observe(model.current + 2 * c.DAY, 1)
        self.bridge(model)
        with self.assertRaises(ValueError):
            model.observe(self.origin, 1)
        restored = pickle.loads(pickle.dumps(model))  # trusted in-memory test only
        self.assertEqual(model.forecast(self.origin), restored.forecast(self.origin))
        with self.assertRaises(ValueError):
            model.forecast(self.origin)
        model.observe(self.origin, self.y.loc[self.origin])
        restored.observe(self.origin, self.y.loc[self.origin])
        self.assertEqual(model.forecast(self.origin + c.DAY), restored.forecast(self.origin + c.DAY))
        self.assertEqual(model.truncation_rate, 0)

    def test_nested_boundary_is_same_fit_contract(self):
        outer = self.origin + 90 * c.DAY
        inner = outer - 90 * c.DAY
        model = c.FrozenARIMA.fit(self.y, inner, (0, 1, 0), source_commit="synthetic-test")
        self.assertEqual(model.current, outer - 98 * c.DAY)
        model.observe(outer - 97 * c.DAY, 1)
        self.assertEqual(model.current, outer - 97 * c.DAY)

    def test_stationary_constant_recipe(self):
        rng = np.random.default_rng(77)
        y = pd.Series(2 + rng.normal(0, .2, len(self.y)), index=self.y.index)
        model = c.FrozenARIMA.fit(y, self.origin, (0, 0, 0), source_commit="synthetic-test")
        self.assertEqual(model.receipt["trend"], "c")
        self.assertEqual(len(model.receipt["parameters"]), 2)
        self.bridge(model, y)
        self.assertTrue(np.isfinite(model.forecast(self.origin).raw))

    def test_nonfinite_forecast_is_not_saved_as_zero(self):
        model = self.bridge(copy.deepcopy(self.base))
        with patch.object(model._result, "forecast", return_value=np.array([np.nan])):
            with self.assertRaises(c.ControlFailure):
                model.forecast(self.origin)
        self.assertEqual(model.predictions, [])
        self.assertIsNone(model._pending)
        self.assertIsNone(model.truncation_rate)
        with self.assertRaisesRegex(c.ControlFailure, "Candidate stopped"):
            model.forecast(self.origin)
        with self.assertRaisesRegex(c.ControlFailure, "Candidate stopped"):
            model.observe(self.origin, 2)

    def test_backend_failures_terminal_without_retry_or_refit(self):
        for stage in ("forecast", "observe"):
            with self.subTest(stage=stage):
                model = self.bridge(copy.deepcopy(self.base))
                if stage == "observe":
                    model.forecast(self.origin)
                previous_predictions = copy.deepcopy(model.predictions)
                previous_observations = copy.deepcopy(model.observations)
                backend_method = "forecast" if stage == "forecast" else "extend"
                with patch.object(model._result, backend_method,
                                  side_effect=RuntimeError("backend failed")) as operation, \
                        patch.object(c, "_backend") as fit_backend:
                    with self.assertRaisesRegex(c.ControlFailure, "backend failed"):
                        if stage == "forecast":
                            model.forecast(self.origin)
                        else:
                            model.observe(self.origin, 2)
                    with self.assertRaisesRegex(c.ControlFailure, "Candidate stopped"):
                        model.forecast(self.origin)
                    with self.assertRaisesRegex(c.ControlFailure, "Candidate stopped"):
                        model.observe(self.origin, 2)
                    self.assertEqual(operation.call_count, 1)
                    fit_backend.assert_not_called()
                self.assertEqual(model.receipt["status"], "failed")
                self.assertEqual(model.receipt["failure_stage"], stage)
                self.assertEqual(model.receipt["retries"], 0)
                self.assertEqual(model.predictions, previous_predictions)
                self.assertEqual(model.observations, previous_observations)
                self.assertEqual(model.current, self.origin - c.DAY)
                restored = pickle.loads(pickle.dumps(model))
                with self.assertRaisesRegex(c.ControlFailure, "Candidate stopped"):
                    restored.forecast(self.origin)

    def test_argument_error_does_not_stop_candidate(self):
        model = self.bridge(copy.deepcopy(self.base))
        with self.assertRaises(ValueError):
            model.forecast(self.origin + c.DAY)
        model.forecast(self.origin)
        with self.assertRaises(ValueError):
            model.observe(self.origin, "not a number")
        self.assertIsNone(model._failure)
        model.observe(self.origin, 2)
        self.assertTrue(np.isfinite(model.forecast(self.origin + c.DAY).value))

    def test_failure_receipt_no_retry(self):
        with patch.object(c, "_backend", side_effect=RuntimeError("synthetic fit failure")) as backend:
            with self.assertRaises(c.ControlFailure) as failure:
                c.FrozenARIMA.fit(self.y, self.origin, (0, 1, 0), source_commit="synthetic-test")
        self.assertEqual(backend.call_count, 1)
        self.assertEqual(failure.exception.receipt["retries"], 0)
        self.assertEqual(failure.exception.receipt["status"], "failed")


if __name__ == "__main__":
    unittest.main()
