import copy
import csv
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "scripts/build_e1_historical_core.py"
SPEC = importlib.util.spec_from_file_location("historical_core", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def sample_raw():
    out = io.StringIO()
    fields = ["history_days", "model", "seed", "n_dates", *module.METRICS,
              "mape_coverage", "mase_coverage", "variant"]
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    for history in module.HISTORIES:
        for model in ("LSTM", "MLP"):
            for seed in module.SEEDS:
                rmse = .001 + (seed - 4100) * .00001
                writer.writerow({"history_days": history, "model": model, "seed": str(float(seed)),
                                 "n_dates": 366, "rmse": rmse, "mse": rmse ** 2,
                                 "mae": .001, "r2": .5, "mape": 10, "mase": 1,
                                 "mape_coverage": 1, "mase_coverage": 1,
                                 "variant": "relative_window"})
    return out.getvalue().encode()


class HistoricalCoreChecks(unittest.TestCase):
    def test_float_encoded_seed_is_reconciled_but_duplicate_seed_rejected(self):
        raw = sample_raw()
        rows = module.parse_runs(raw, "cv", ("LSTM", "MLP"))
        self.assertEqual(len(rows), 80)
        self.assertEqual(rows[0]["seed"], 4101)
        with self.assertRaisesRegex(ValueError, "Duplicate or missing"):
            module.parse_runs(raw + raw.splitlines()[1] + b"\n", "cv", ("LSTM", "MLP"))

    def test_different_period_or_representation_is_rejected(self):
        for old, new in ((b",366,", b",608,"), (b"relative_window", b"absolute_log")):
            with self.assertRaises(ValueError):
                module.parse_runs(sample_raw().replace(old, new), "cv", ("LSTM", "MLP"))

    def test_metric_inconsistency_and_incomplete_coverage_are_rejected(self):
        rows = list(csv.DictReader(io.StringIO(sample_raw().decode())))
        for key, value in (("mse", "1"), ("mape_coverage", "0.9")):
            changed = copy.deepcopy(rows)
            changed[0][key] = value
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=list(changed[0]))
            writer.writeheader()
            writer.writerows(changed)
            with self.assertRaises(ValueError):
                module.parse_runs(out.getvalue().encode(), "cv", ("LSTM", "MLP"))

    def test_lstm_control_requires_exact_per_seed_equality(self):
        old = module.parse_runs(sample_raw(), "cv", ("LSTM", "MLP"))
        current = [copy.deepcopy(x) for x in old if x["model"] == "LSTM"]
        self.assertEqual(module.verify_lstm_control(old, current), 240)
        current[0]["rmse"] += 1e-15
        with self.assertRaisesRegex(ValueError, "match exactly"):
            module.verify_lstm_control(old, current)

    def test_holm_preserves_pair_order_and_monotonic_adjustment(self):
        self.assertEqual(module.holm([.04, .001, .02]), [.04, .003, .04])

    def test_changed_historical_source_is_rejected_before_compilation(self):
        with tempfile.TemporaryDirectory() as target:
            path = Path(target) / module.INPUTS["old_cv"][0]
            path.parent.mkdir(parents=True)
            path.write_bytes(b"changed historical source")
            with self.assertRaisesRegex(ValueError, "Historical source hash changed"):
                module.compile_outputs(Path(target))

    def test_saved_sources_yield_240_runs_without_forecast_vectors(self):
        result = module.compile_outputs()
        import json
        summary = json.loads(result["summary.json"])
        self.assertEqual(summary["runs"], 240)
        self.assertEqual(summary["lstm_exact_control_metric_equalities"], 480)
        self.assertEqual(summary["new_model_fits"], 0)
        self.assertEqual(summary["daily_prediction_vectors_read"], 0)
        self.assertFalse(summary["new_core_final_study_results_substituted"])
        self.assertTrue(summary["historical_core_selection_is_posthoc"])
        self.assertEqual(len(list(csv.DictReader(io.StringIO(result["metric_summary.csv"].decode())))), 144)
        self.assertEqual(summary["best_history_by_saved_cv_mean_rmse"], {"LSTM": 7, "MLP": 7, "GRU": 7})
        self.assertEqual(summary["historical_parameters_by_D"]["28"]["MLP"], 2273)


if __name__ == "__main__":
    unittest.main()
