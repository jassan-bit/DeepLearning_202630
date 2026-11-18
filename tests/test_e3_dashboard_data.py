"""Dashboard data contracts: lineage, one global mask, and honest missing fields."""
from __future__ import annotations

import csv
from datetime import date, timedelta
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from e3_dashboard import data


ROOT = Path(__file__).resolve().parents[1]


class DashboardDataTests(unittest.TestCase):
    def test_catalogue_matches_book_source_metrics_and_seed_intervals(self):
        bundle = data.load_bundle()
        rows = list(csv.DictReader(io.StringIO((ROOT / "book_e3/_static/e3_revision_bloque_d_20261117/metrics_extended.csv").read_bytes().decode())))
        expected = {r["model"]: r for r in rows}
        self.assertEqual(set(expected), set(data.MODEL_IDS))
        for model in bundle["catalogue"]:
            source = expected[model["model"]]
            for key in ("RMSE_mean", "RMSE_sample_SD", "MAE_mean", "sMAPE_percent", "R2_mean", "coverage90_percent", "mean_interval_width90"):
                self.assertEqual(model[key], None if source[key] == "" else float(source[key]))
            self.assertEqual(model["R"], int(source["R"]))
        e3 = next(r for r in bundle["catalogue"] if r["model"] == data.E3_MODEL)
        self.assertIsNone(e3["coverage90_percent"])
        self.assertIsNone(e3["mean_interval_width90"])
        self.assertEqual(e3["parameters"], 1210)

    def test_representative_curves_are_not_ensembles_and_match_evaluation_targets(self):
        bundle = data.load_bundle()
        self.assertEqual(len(bundle["predictions"]["dates"]), 608)
        for model in data.MODEL_IDS:
            curve = data.prediction_for(model)
            self.assertEqual(curve["seed"], "deterministic" if model in data.CLASSICAL else 271801)
            self.assertEqual(len(curve["predictions"]), len(curve["targets"]))
        observed = [r for r in bundle["observations"] if r["partition"] == "evaluation"]
        self.assertEqual([r["target_date"] for r in observed], bundle["predictions"]["dates"])
        self.assertEqual([r["target"] for r in observed], bundle["predictions"]["targets"])

    def test_observed_covariates_always_belong_to_the_previous_day(self):
        bundle = data.load_bundle()
        for row in bundle["observations"]:
            self.assertEqual(date.fromisoformat(row["target_date"]) - date.fromisoformat(row["feature_date"]), timedelta(days=1))
            self.assertGreaterEqual(row["target"] + 1e-12, row["known_floor"])
        self.assertEqual({p: len(data.filter_observations([p])) for p in data.PARTITIONS},
                         {"training": 655, "calibration": 53, "evaluation": 608})

    def test_inclusive_global_filter_and_empty_selection(self):
        rows = data.filter_observations(["evaluation"], "2025-01-01", "2025-01-03")
        self.assertEqual([r["target_date"] for r in rows], ["2025-01-01", "2025-01-02", "2025-01-03"])
        lower = min(r["target"] for r in rows)
        upper = max(r["target"] for r in rows)
        self.assertEqual(data.filter_observations(["evaluation"], "2025-01-01", "2025-01-03", [lower, upper]), rows)
        self.assertEqual(data.filter_observations([]), [])
        self.assertEqual(data.filter_observations(["training"], "2025-01-01", "2025-01-03"), [])

    def test_invalid_filters_and_unknown_ids_fail_explicitly(self):
        for kwargs in ({"partitions": ["test-not-a-partition"]}, {"start": "2026-01-02", "end": "2026-01-01"},
                       {"value_range": [0.2, 0.1]}, {"value_range": [float("nan"), 1]}):
            with self.assertRaises(ValueError):
                data.filter_observations(**kwargs)
        with self.assertRaises(data.DataIntegrityError):
            data.prediction_for("not-a-model")
        with self.assertRaises(data.DataIntegrityError):
            data.get_example("not-an-example")

    def test_actual_examples_have_seven_past_days_and_frozen_hashes(self):
        for identifier in ("evaluation-0", "evaluation-303", "evaluation-607"):
            example = data.get_example(identifier)
            self.assertEqual(hashlib.sha256(example["csv_text"].encode()).hexdigest(), example["csv_sha256"])
            rows = list(csv.DictReader(io.StringIO(example["csv_text"])))
            target_day = date.fromisoformat(example["target_date"])
            self.assertEqual([r["date"] for r in rows], [(target_day - timedelta(days=7-i)).isoformat() for i in range(7)])

    def test_corrupt_source_is_rejected_before_it_can_reach_figures(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "bundle"
            shutil.copytree(data.DEFAULT_BUNDLE_DIR, folder)
            path = folder / "predictions.json"
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + b" ")
            with self.assertRaisesRegex(data.DataIntegrityError, "huella"):
                data.load_bundle(folder)

    def test_temporal_semantics_are_checked_even_after_hash_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "bundle"
            shutil.copytree(data.DEFAULT_BUNDLE_DIR, folder)
            path = folder / "observations.csv"
            rows = list(csv.DictReader(io.StringIO(path.read_bytes().decode())))
            rows[0]["feature_date"] = rows[0]["target_date"]
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\r\n")
            writer.writeheader(); writer.writerows(rows)
            raw = stream.getvalue().encode(); path.write_bytes(raw)
            manifest = json.loads((folder / "manifest.json").read_bytes())
            manifest["files"]["observations.csv"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            (folder / "manifest.json").write_bytes(json.dumps(manifest).encode())
            with self.assertRaisesRegex(data.DataIntegrityError, "temporal"):
                data.load_bundle(folder)

    def test_no_seed_bootstrap_or_prediction_interval_is_invented(self):
        bundle = data.load_bundle()
        self.assertIsNone(bundle["predictions"]["models"][data.E3_MODEL]["q"])
        for model in bundle["catalogue"]:
            if model["model"] in data.CLASSICAL:
                self.assertEqual(model["R"], 1)
                self.assertIsNone(model["RMSE_sample_SD"])
                self.assertIsNone(model["RMSE_seed_ci95_lower"])
                self.assertIsNone(model["RMSE_seed_ci95_upper"])
        self.assertEqual(len(bundle["mcs"]), 17)
        self.assertEqual(len(bundle["contrasts"]), 48)
        self.assertEqual(len(bundle["metadata"]["mcs_retained"]), 10)


if __name__ == "__main__":
    unittest.main()
