"""Synthetic-only adversarial tests; never open final data or fit a model."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1]/'scripts/audit_e2_common_availability.py'
spec = importlib.util.spec_from_file_location('availability_under_test', PATH)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.date_range('2023-01-01', periods=400, tz='UTC')
        self.x = np.tile([10., 12., 9., 11., 20.], (400, 1))
        self.y = np.ones(400)

    def masks(self, quality=None):
        return a.support_masks(self.calendar, self.x, self.y, quality)

    def test_eight_closes_and_28_lag_warmup(self):
        m = self.masks()
        self.assertEqual(np.flatnonzero(m['D7_target'])[0], 7)
        self.assertEqual(np.flatnonzero(m['lag28_target'])[0], 35)
        self.assertEqual(np.flatnonzero(m['common'])[0], 35)

    def test_missing_day_propagation_not_compressed(self):
        self.x[80] = np.nan
        m = self.masks()
        self.assertFalse(m['target_sigma7_quality'][80:88].any())
        self.assertFalse(m['D7_window'][81:88].any())
        self.assertTrue(m['target_sigma7_quality'][88])
        for lag in a.LAGS:
            self.assertFalse(m[f'lag{lag}_target'][80+lag:88+lag].any())
        self.assertFalse(m['common'][108:116].any())
        self.assertTrue(m['D7_target'][108:116].all())

    def test_bad_quality_masks_otherwise_finite_days(self):
        quality = np.ones(400, dtype=bool); quality[100] = False
        m = self.masks(quality)
        self.assertFalse(m['target_sigma7_quality'][100:108].any())
        self.assertFalse(m['common'][128:136].any())

    def test_invalid_targets_and_ohlcv(self):
        for bad in (np.nan, np.inf, -1.):
            self.y[100] = bad
            self.assertFalse(self.masks()['target_sigma7_quality'][100])
        self.y[100] = 0.
        self.assertTrue(self.masks()['target_sigma7_quality'][100])
        self.x[120, 1] = 8.
        self.assertFalse(self.masks()['D7_window'][121])
        self.assertFalse(self.masks()['target_sigma7_quality'][120:128].any())
        self.assertTrue(self.masks()['target_sigma7_quality'][128])
        self.assertFalse(self.masks(np.ones(400, dtype=bool))['target_sigma7_quality'][120])

    def test_invalid_calendars_rejected(self):
        for cal in (self.calendar.delete(80), self.calendar.tz_localize(None),
                    self.calendar[::-1], self.calendar+pd.Timedelta(hours=1)):
            with self.assertRaises(ValueError):
                a.validate_calendar(cal)

    def test_calendar_and_shapes_not_silently_aligned(self):
        with self.assertRaises(ValueError):
            a.support_masks(self.calendar, self.x[:-1], self.y)
        with self.assertRaises(ValueError):
            self.masks(np.ones(400))

    def test_future_magnitudes_cannot_change_current_mask(self):
        before = self.masks()
        self.x[300:] *= 2.; self.y[300:] *= 9.
        after = self.masks()
        for name in before:
            np.testing.assert_array_equal(before[name][:300], after[name][:300])

    def test_arima_keeps_missing_calendar_and_separate_neural_train(self):
        self.x[80] = np.nan
        m = self.masks()
        r = a.arima_support(self.calendar, m, '2024-01-01')
        self.assertEqual(r['outer_bridge']['calendar_days'], 7)
        self.assertEqual(r['inner_bridge']['calendar_days'], 7)
        self.assertEqual(r['inner_validation']['calendar_days'], 83)
        self.assertEqual(r['outer_training']['cutoff'], '2023-12-24')
        self.assertTrue(r['outer_training']['at_least_90'])
        self.assertFalse(r['convergence_certified'])
        self.assertGreater(r['outer_training']['neural_D7_training_count'],
                           int((m['common'] & (self.calendar <= pd.Timestamp('2023-12-24', tz='UTC'))).sum()))

    def test_arima_threshold_not_guaranteed_by_dates_alone(self):
        self.y[7:300] = np.nan
        r = a.arima_support(self.calendar, self.masks(), '2024-01-01')
        self.assertGreater(r['outer_training']['calendar_rows'], 90)
        self.assertFalse(r['outer_training']['at_least_90'])

    def test_missing_calendar_not_inferred_to_be_bad_market(self):
        r = a.period_report(self.calendar, self.masks(), '2025-01-01', '2025-01-03', 3)
        self.assertEqual(r['common_count'], 0)
        self.assertEqual(len(r['exclusion_causes_nonexclusive']['missing_calendar_row']), 3)
        self.assertEqual(r['exclusion_causes_nonexclusive']['target_sigma7_quality'], [])

    def test_prohibited_imports_absent(self):
        import ast
        tree = ast.parse(PATH.read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or '')
        self.assertFalse(any(m.startswith(('torch', 'statsmodels', 'dl_data', 'e2_final')) for m in modules))


if __name__ == '__main__':
    unittest.main()
