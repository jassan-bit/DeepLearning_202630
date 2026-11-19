"""Synthetic invariants for shared causal OHLCV input corruption."""
from datetime import date, timedelta
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("stress_under_test", Path(__file__).with_name("perturb.py"))
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)


def bars(n, start="2020-01-01"):
    rng = np.random.default_rng(719)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .025, n)))
    raw = np.column_stack((close*.999, close*1.02, close*.98, close,
                           np.rint(rng.lognormal(7, .5, n))))
    labels = [(date.fromisoformat(start)+timedelta(days=i)).isoformat() for i in range(n)]
    return raw, labels


class PerturbationTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.dates = bars(50)
        self.profile = stress.fit_noise_profile(self.raw[:20], self.dates[:20], "D03",
                            fit_end_declared=self.dates[19], volume_units="contracts")

    def test_shared_bars_are_identical_in_overlapping_windows(self):
        first, a = stress.perturb_window(self.raw[20:27], self.dates[20:27], self.profile)
        second, b = stress.perturb_window(self.raw[21:28], self.dates[21:28], self.profile)
        np.testing.assert_array_equal(first[1:], second[:-1])
        self.assertEqual(a["log_price_errors"][1:], b["log_price_errors"][:-1])
        self.assertEqual(a["origin_date"], self.dates[26])

    def test_future_bars_do_not_change_profile_or_prior_corruptions(self):
        before, receipt = stress.perturb_window(self.raw[20:27], self.dates[20:27], self.profile)
        changed = self.raw.copy()
        changed[30:, :4] *= 1000
        changed[30:, 4] *= 2
        profile = stress.fit_noise_profile(changed[:20], self.dates[:20], "D03",
                            fit_end_declared=self.dates[19], volume_units="contracts")
        after, repeated = stress.perturb_window(changed[20:27], self.dates[20:27], profile)
        self.assertEqual(profile, self.profile)
        np.testing.assert_array_equal(after, before)
        self.assertEqual(receipt, repeated)

    def test_three_severities_preserve_ohlc_native_units_and_zero_volume(self):
        raw = self.raw[20:27].copy(); raw[2, 4] = 0
        receipts = []
        for severity in (1, 2, 3):
            result, receipt = stress.perturb_window(raw, self.dates[20:27], self.profile, severity=severity)
            receipts.append(receipt)
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue((result[:, 1] >= np.max(result[:, [0, 2, 3]], axis=1)).all())
            self.assertTrue((result[:, 2] <= np.min(result[:, [0, 1, 3]], axis=1)).all())
            self.assertTrue((result[:, :4] > 0).all())
            self.assertEqual(result[2, 4], 0)
            np.testing.assert_array_equal(result[:, 4], np.rint(result[:, 4]))
            # Candle shape changes only by a shared positive price multiplier.
            np.testing.assert_allclose(result[:, 1]/result[:, 2], raw[:, 1]/raw[:, 2])
            self.assertFalse(receipt["targets_changed"])
        np.testing.assert_allclose(np.asarray(receipts[1]["log_price_errors"]),
                                   np.asarray(receipts[0]["log_price_errors"])*2)
        np.testing.assert_allclose(np.asarray(receipts[2]["log_price_errors"]),
                                   np.asarray(receipts[0]["log_price_errors"])*4)
        np.testing.assert_array_equal(self.raw[20:27], bars(50)[0][20:27])

    def test_rejects_test_fit_boundary_changes_tampering_and_retrospective_use(self):
        with self.assertRaisesRegex(ValueError, "FIT"):
            stress.fit_noise_profile(self.raw[:20], self.dates[:20], "D01", partition="test")
        with self.assertRaisesRegex(ValueError, "boundary"):
            stress.fit_noise_profile(self.raw[:21], self.dates[:21], "D01", fit_end_declared=self.dates[19])
        with self.assertRaisesRegex(ValueError, "digest"):
            stress.fit_noise_profile(self.raw[:20], self.dates[:20], "D01", expected_fit_dates_sha256="0"*64)
        changed = dict(self.profile); changed["price_scale"] *= 2
        with self.assertRaisesRegex(ValueError, "integrity"):
            stress.perturb_window(self.raw[20:27], self.dates[20:27], changed)
        with self.assertRaisesRegex(ValueError, "origin"):
            stress.perturb_window(self.raw[:7], self.dates[:7], self.profile)

    def test_rejects_noncausal_contracts_and_unplanned_severity(self):
        for kwargs in ({"severity":4},{"severity":True},{"family":"iid_future_returns"},{"seed":True}):
            with self.assertRaises(ValueError):
                stress.perturb_window(self.raw[20:27], self.dates[20:27], self.profile, **kwargs)
        with self.assertRaises(ValueError):
            stress.perturb_window(self.raw[20:28], self.dates[20:28], self.profile)
        with self.assertRaises(ValueError):
            stress.perturb_window(self.raw[20:27], list(reversed(self.dates[20:27])), self.profile)


if __name__ == "__main__":
    unittest.main()
