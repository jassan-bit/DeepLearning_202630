"""Serving admission, exact interventions and artifact integrity, without Torch."""
import csv
from datetime import date, timedelta
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from e3_dashboard import inference as serving
from e3_dashboard.explain import occlusion_explanation

ROOT = Path(__file__).resolve().parents[1]


def csv_window(raw, target="2023-01-08"):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["date", *serving.FEATURES])
    origin = date.fromisoformat(target)
    for i, row in enumerate(raw):
        writer.writerow([(origin - timedelta(days=7-i)).isoformat(), *row])
    return output.getvalue()


def reference(model_id="MLP16"):
    meta, _ = serving._metadata(model_id)
    return np.asarray(meta["explanation_reference_raw"])


class Admission(unittest.TestCase):
    def test_valid_bars_preserve_dates_order_and_values(self):
        raw = reference()
        values, dates = serving.validate_csv(csv_window(raw), "2023-01-08")
        np.testing.assert_array_equal(values, raw)
        self.assertEqual(dates[0], "2023-01-01")
        self.assertEqual(dates[-1], "2023-01-07")

    def test_bad_inputs_fail_before_a_session_is_loaded(self):
        text = csv_window(reference())
        cases = [text.replace("2023-01-03", "2023-01-02"),
                 text.replace("2023-01-01", "2023-01-08"),
                 text.replace("2023-01-01", "2022-12-31"),
                 text.replace("volume", "Volume"),
                 "\n".join(text.splitlines()[:-1]),
                 text + text.splitlines()[1] + "\n", text + "a" * serving.MAX_UPLOAD_BYTES,
                 text.replace(text.splitlines()[1].split(",")[1], "NaN", 1)]
        serving.clear_cache()
        for case in cases:
            with self.subTest(case=case[:40]):
                with self.assertRaises(serving.InputError):
                    serving.predict_csv("MLP16", case, "2023-01-08")
                self.assertIsNone(serving._SESSION)

    def test_prices_volume_and_bar_coherence_are_enforced(self):
        raw = reference()
        for row, column, value in ((2, 3, 0), (1, 4, -1), (3, 1, 1), (4, 2, 1e100)):
            modified = raw.copy(); modified[row, column] = value
            with self.subTest(column=column):
                with self.assertRaises(serving.InputError):
                    serving.validate_csv(csv_window(modified), "2023-01-08")

    def test_future_origin_and_noncanonical_date_are_rejected(self):
        with self.assertRaises(serving.InputError):
            serving.validate_csv(csv_window(reference(), "2999-01-08"), "2999-01-08")
        with self.assertRaises(serving.InputError):
            serving.validate_csv(csv_window(reference()), "20230108")

    def test_preprocessing_and_observable_floor_against_hand_calculation(self):
        logs = np.asarray([0,.01,.03,.06,.10,.15,.21])
        raw = np.column_stack([np.exp(logs)] * 4 + [np.ones(7)*3])
        meta = {"preprocessing": {"mean": [0]*5, "std": [1]*5}}
        transformed, floor = serving.transform_window(raw, meta)
        np.testing.assert_allclose(transformed[0,:,3], logs-.21, atol=1e-7)
        np.testing.assert_allclose(transformed[0,:,4], np.log(4), atol=1e-7)
        self.assertAlmostEqual(float(floor[0]), .00025, places=14)


class Serving(unittest.TestCase):
    def tearDown(self):
        serving.clear_cache()

    def test_four_models_and_intervals_have_explicit_capabilities(self):
        self.assertEqual(len(serving.available_models()), 4)
        for model in serving.available_models():
            result = serving.predict_csv(model["id"], csv_window(reference(model["id"])), "2023-01-08")
            self.assertGreaterEqual(result["prediction"], 0)
            self.assertEqual(result["seed"], 271801)
            if model["id"].endswith(" E3"):
                self.assertIsNone(result["interval"])
                self.assertIsNone(result["nominal_coverage"])
            else:
                meta, _ = serving._metadata(model["id"])
                self.assertEqual(result["interval"], [max(0, result["prediction"]-meta["calibration_q"]),
                                                       result["prediction"]+meta["calibration_q"]])
            self.assertEqual(len(result["explanation"]["attributions"]), 7)

    def test_cache_reuses_only_current_model_and_evicts_before_next_load(self):
        serving.predict_csv("MLP16", csv_window(reference()), "2023-01-08")
        previous = serving._SESSION
        serving.predict_csv("MLP16", csv_window(reference()), "2023-01-08")
        self.assertIs(serving._SESSION, previous)
        serving.predict_csv("LSTM32", csv_window(reference("LSTM32")), "2023-01-08")
        self.assertIsNot(serving._SESSION, previous)
        self.assertEqual(serving._SESSION_ID[0], "LSTM32")

    def test_fit_range_warning_changes_for_valid_extreme_volume(self):
        raw = reference("CNN-LSTM-atencion-aditiva E3")
        original = serving.predict_csv("CNN-LSTM-atencion-aditiva E3", csv_window(raw), "2023-01-08")
        self.assertFalse(original["ood"])
        raw[:,4] *= 1e100
        changed = serving.predict_csv("CNN-LSTM-atencion-aditiva E3", csv_window(raw), "2023-01-08")
        self.assertTrue(changed["ood"])
        self.assertGreater(changed["ood_fraction"], 0)
        self.assertTrue(np.isfinite(changed["prediction"]))

    def test_corrupt_metadata_and_model_are_rejected(self):
        import shutil
        for filename in ("metadata.json", "model.onnx"):
            with tempfile.TemporaryDirectory() as folder:
                copied = Path(folder) / "artifacts"
                shutil.copytree(serving.ARTIFACT_ROOT, copied)
                path = copied / "mlp16" / filename
                raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
                serving.clear_cache()
                with patch.object(serving, "ARTIFACT_ROOT", copied):
                    with self.assertRaises(RuntimeError):
                        serving.predict_csv("MLP16", csv_window(reference()), "2023-01-08")

    def test_training_and_torch_are_not_imported_in_runtime_process(self):
        code = "from e3_dashboard.inference import available_models; import sys; assert len(available_models())==4; assert 'torch' not in sys.modules; assert not any(x.startswith('src.e3_') for x in sys.modules)"
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)

    def test_occlusion_is_the_exact_predictor_contrast_and_reference_identity(self):
        raw = reference()
        seen = []
        def fake_predict(x):
            seen.append(x.copy())
            self.assertTrue(np.all(x[:,2] <= np.minimum(x[:,0],x[:,3])))
            self.assertTrue(np.all(x[:,1] >= np.maximum(x[:,0],x[:,3])))
            return float(x[:,3].sum())
        report = occlusion_explanation(raw, raw, fake_predict, float(raw[:,3].sum()))
        self.assertEqual(len(seen),7)
        self.assertTrue(all(x["effect"] == 0 for x in report["attributions"]))
        self.assertIn("pendientes", report["fidelity_status"])


if __name__ == "__main__":
    unittest.main()
