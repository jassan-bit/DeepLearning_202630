import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/build_e1_primary_608.py"
SPEC = importlib.util.spec_from_file_location("primary_608", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)

NEURAL_MODELS = {"LSTM", "MLP", "GRU"}
METRICS = ("rmse", "mae", "mse", "r2", "mape", "mase")


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


class Primary608Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs = module.compile_outputs()
        cls.summary = json.loads(cls.outputs["summary.json"])
        cls.runs = csv_rows(cls.outputs["run_metrics.csv"])
        cls.metrics = csv_rows(cls.outputs["metric_summary.csv"])

    def test_primary_comparison_retains_the_requested_calendar_and_recipes(self):
        summary = self.summary
        self.assertEqual(summary["models"], ["LSTM", "MLP", "GRU"])
        self.assertEqual(summary["baseline"], "GRU")
        self.assertEqual(summary["R_per_neural"], 10)
        self.assertEqual(summary["n_dates"], 608)
        self.assertEqual(summary["primary_scope"], "historical_known_fixed_recipes")
        self.assertTrue(summary["HPO_not_equal"])
        self.assertTrue(summary["editorial_promotion_after_results_seen"])
        self.assertEqual(summary["new_model_fits"], 0)
        self.assertEqual(summary["new_forecasts"], 0)

    def test_exactly_ten_neural_seeds_and_one_deterministic_reference_per_phase(self):
        self.assertEqual(len(self.runs), 62)
        self.assertEqual(len(self.metrics), 48)
        for phase, count in (("cv", 366), ("test", 608)):
            phase_rows = [row for row in self.runs if row["phase"] == phase]
            self.assertEqual(len(phase_rows), 31)
            self.assertEqual({int(row["n_dates"]) for row in phase_rows}, {count})
            for model in NEURAL_MODELS:
                selected = [row for row in phase_rows if row["model"] == model]
                self.assertEqual(len(selected), 10)
                self.assertEqual({int(row["seed"]) for row in selected}, set(range(4101, 4111)))
                self.assertEqual({int(row["history_days"]) for row in selected}, {7})
                expected_parameters = {"LSTM": 5025, "MLP": 593, "GRU": 3777}[model]
                self.assertEqual({int(row["parameters"]) for row in selected}, {expected_parameters})
            reference = [row for row in phase_rows if row["model"] not in NEURAL_MODELS]
            self.assertEqual(len(reference), 1)
            self.assertEqual(reference[0]["seed"], "")
            self.assertEqual(int(reference[0]["parameters"]), 0)

    def test_persistence_has_no_seed_uncertainty(self):
        references = [row for row in self.metrics if row["model"] not in NEURAL_MODELS]
        self.assertEqual(len(references), 12)
        self.assertEqual({row["metric"] for row in references}, set(METRICS))
        for row in references:
            self.assertEqual(int(row["n_seeds"]), 0)
            self.assertEqual(int(row["n_total"]), 1)
            self.assertEqual(int(row["n_defined"]), 1)
            for key in ("sample_sd", "ci95_low", "ci95_high"):
                self.assertEqual(row[key], "")

    def test_all_neural_metrics_are_original_saved_D7_values(self):
        original = csv_rows((ROOT / "results/e1_historical_core_20261107/run_metrics.csv").read_bytes())
        expected = {(row["phase"], row["model"], row["seed"]): row
                    for row in original if int(row["history_days"]) == 7}
        actual = {(row["phase"], row["model"], row["seed"]): row
                  for row in self.runs if row["model"] in NEURAL_MODELS}
        self.assertEqual(set(actual), set(expected))
        for key, row in actual.items():
            for metric in METRICS:
                self.assertEqual(float(row[metric]), float(expected[key][metric]), (key, metric))

    def test_original_MLP_execution_is_authentic_and_not_a_new_fit(self):
        evidence = json.loads(self.outputs["execution_provenance.json"])["original_mlp"]
        self.assertEqual(evidence["scored_runs"], 40)
        self.assertEqual(evidence["D7_scored_runs"], 10)
        self.assertEqual(evidence["dates_per_run"], 608)
        self.assertTrue(evidence["all_scored_once"])
        self.assertEqual(set(evidence["D7_run_ids"]),
                         {f"D7_MLP_s{seed}" for seed in range(4101, 4111)})
        self.assertEqual(evidence["first_D7_score_at_utc"], "2026-11-04T19:44:23.161749+00:00")
        self.assertEqual(evidence["last_D7_score_at_utc"], "2026-11-04T19:44:23.359340+00:00")

    def test_changed_source_fails_closed_before_compilation(self):
        relative_path, _ = next(iter(module.INPUTS.values()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / relative_path
            path.parent.mkdir(parents=True)
            path.write_bytes(b"mutated preserved experiment")
            with self.assertRaisesRegex(ValueError, "[Hh]ash changed"):
                module.compile_outputs(root=Path(directory))


if __name__ == "__main__":
    unittest.main()
