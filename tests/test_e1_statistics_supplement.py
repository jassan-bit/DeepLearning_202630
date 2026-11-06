"""Checks pairing safeguards and rank-test invariants of the new supplement."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import supplement_e1_planned_statistics as supplement


def source_fixture():
    rows = [{"model": model, "seed": seed,
             "rmse": .004 + (seed - 8100) * .000001 + offset * .0001,
             "n_dates": 60}
            for offset, model in enumerate(supplement.MODELS)
            for seed in supplement.EXPECTED_SEEDS]
    rows.append({"model": "Persistence", "seed": 0, "rmse": .0038, "n_dates": 60})
    return pd.DataFrame(rows)


class PairedInputTests(unittest.TestCase):
    def test_pairing_is_by_identifier_not_row_order(self):
        original = source_fixture()
        shuffled = original.sample(frac=1, random_state=7)
        pd.testing.assert_frame_equal(supplement.paired_panel(original),
                                      supplement.paired_panel(shuffled))

    def test_missing_partner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Every family"):
            supplement.paired_panel(source_fixture().drop(index=0))

    def test_duplicate_seed_is_rejected(self):
        frame = source_fixture()
        with self.assertRaisesRegex(ValueError, "unique"):
            supplement.paired_panel(pd.concat([frame, frame.iloc[[0]]]))

    def test_nonfinite_metric_or_wrong_calendar_is_rejected(self):
        for column, value in (("rmse", np.inf), ("n_dates", 59)):
            frame = source_fixture()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                supplement.paired_panel(frame)


class RankAnalysisTests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_restores_pair_order(self):
        np.testing.assert_allclose(supplement.holm([.04, .01, .03]), [.06, .03, .06])
        np.testing.assert_allclose(supplement.holm([0, 1, .9]), [0, 1, 1])

    def test_complete_agreement_has_unit_kendall_w_and_signed_effect(self):
        # Every block has ranks 1, 2, 3. By the closed-form no-tie Friedman
        # formula, chi-square=n*(k-1), independently of the RMSE spacings.
        t = np.arange(20, dtype=float)
        panel = pd.DataFrame({"GRU": 1 + t / 100,
                              "LSTM": 2 + t / 90,
                              "TCN": 3 + t / 80})
        result, pairs, diagnostics, coordinates = supplement.analyze(panel)
        self.assertAlmostEqual(result["chi_square"], 40.)
        self.assertAlmostEqual(result["kendall_W"], 1.)
        self.assertEqual(result["mean_ranks_lower_rmse_better"],
                         {"GRU": 1., "LSTM": 2., "TCN": 3.})
        self.assertTrue(pairs.matched_pairs_rank_biserial_a_minus_b.eq(-1).all())
        self.assertTrue(pairs.wilcoxon_statistic_min_rank_sum.eq(0).all())
        self.assertEqual(len(coordinates), 60)
        self.assertEqual(len(diagnostics), 3)


if __name__ == "__main__":
    unittest.main()
