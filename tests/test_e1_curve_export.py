"""Meaningful export checks on synthetic epoch data, never model fitting."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_e1_reproduced_curves as export


def curve(stage="inner"):
    return pd.DataFrame({"epoch": [1, 2], "stage": [stage, stage],
        "optimization_mse_scaled": [2., 1.], "train_mse_scaled_eval": [1.8, .9],
        "validation_mse_scaled_eval": [2.1, 1.1] if stage == "inner" else [np.nan, np.nan],
        "best_epoch_so_far": [1, 2], "learning_rate": [.003, .003]})


class ExportChecks(unittest.TestCase):
    def test_refit_does_not_invent_validation(self):
        value = curve("refit")
        export.validate_curve(value, "refit", 2)
        value.loc[1, "validation_mse_scaled_eval"] = 1.2
        with self.assertRaisesRegex(ValueError, "invented validation"):
            export.validate_curve(value, "refit", 2)

    def test_epochs_and_loss_values_are_checked(self):
        value = curve()
        value.loc[1, "epoch"] = 3
        with self.assertRaisesRegex(ValueError, "epoch order"):
            export.validate_curve(value, "inner", 2)
        value = curve()
        value.loc[1, "train_mse_scaled_eval"] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            export.validate_curve(value, "inner", 2)

    def test_aggregation_never_pads_stopped_runs(self):
        first = curve().assign(model="GRU", seed=8101)
        second = curve().iloc[:1].copy().assign(model="GRU", seed=8102)
        second.loc[0, "train_mse_scaled_eval"] = 4.2
        result = export.aggregate_curves(pd.concat([first, second], ignore_index=True))
        result = result.loc[result.metric.eq("train_mse_scaled_eval")].set_index("epoch")
        self.assertEqual(result.loc[1, "n_trajectories"], 2)
        self.assertAlmostEqual(result.loc[1, "mean"], 3.)
        self.assertEqual(result.loc[2, "n_trajectories"], 1)
        self.assertAlmostEqual(result.loc[2, "mean"], .9)

    def test_curve_schema_rejects_observation_columns(self):
        value = curve().assign(y_true=[.1, .2])
        with self.assertRaisesRegex(ValueError, "schema"):
            export.validate_curve(value, "inner", 2)


if __name__ == "__main__":
    unittest.main()
