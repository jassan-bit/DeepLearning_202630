"""Verify five shipped CPU predictors against the pinned 608-date cohort.

The historical verifier retains its four-model contract. This version checks
the exact expanded registry and reuses frozen forecasts without fitting,
selecting or calibrating any model. Torch is never imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from e3_dashboard import inference
from e3_dashboard.data import load_bundle

NPZ_SHA = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
NPZ_URL = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/af3f1223ad5899ed63016d2af2d09bae8470ab00/book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz"
EXPECTED = {"MLP16": "mlp16", "LSTM32": "lstm32", "GRU32": "gru32",
            "CNN-LSTM-atencion-aditiva E3": "cnn_lstm_attention_e3", "TSMixer": "tsmixer"}
TOLERANCE = 1e-6


def validate_registry(rows):
    """Reject missing, duplicate, extra or renamed deployed predictors."""
    ids = [row["id"] for row in rows]
    if len(ids) != len(EXPECTED) or len(set(ids)) != len(ids):
        raise ValueError("The serving V2 registry must contain five unique models.")
    actual = {row["id"]: row["slug"] for row in rows}
    if actual != EXPECTED:
        raise ValueError("The serving V2 model IDs and slugs do not match the five frozen predictors.")


def pinned_cohort(path, download=False):
    path = Path(path)
    if not path.is_file():
        if not download:
            raise ValueError("The pinned cohort is missing; use --download or supply --npz.")
        with urlopen(NPZ_URL, timeout=60) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != NPZ_SHA:
            raise ValueError("Pinned cohort download failed verification.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    if hashlib.sha256(path.read_bytes()).hexdigest() != NPZ_SHA:
        raise ValueError("Pinned cohort bytes changed.")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: np.asarray(arrays[key]) for key in
                ("evaluation_raw", "evaluation_dates", "evaluation_y_original",
                 "calibration_raw", "calibration_dates", "calibration_y_original")}


def run(path, download=False):
    validate_registry(inference._registry())
    bundle = load_bundle()
    cohort = pinned_cohort(path, download)
    raw = np.asarray(cohort["evaluation_raw"], dtype=np.float64)
    dates = [str(x) for x in cohort["evaluation_dates"]]
    y = np.asarray(cohort["evaluation_y_original"], dtype=np.float64)
    if (raw.shape != (608, 7, 5) or y.shape != (608,) or not np.isfinite(raw).all()
            or dates != bundle["predictions"]["dates"]
            or not np.array_equal(y, bundle["predictions"]["targets"])):
        raise ValueError("Pinned evaluation inputs, dates and targets do not match frozen references.")
    records = []
    calibration = None
    try:
        for model_id in EXPECTED:
            meta, folder = inference._metadata(model_id)
            if meta["model"] != model_id or meta["seed"] != 271801:
                raise ValueError("The metadata identity or representative seed changed.")
            with inference._LOCK:
                session = inference._session(model_id, meta, folder)
                outputs = np.asarray([inference._run(window, meta, session) for window in raw])
                if model_id == "CNN-LSTM-atencion-aditiva E3":
                    profile = meta.get("serving_calibration")
                    if (not profile or profile["profile"] != "posterior_absolute_residual_90_v2"
                            or profile["q_sigma7"] != .005402463770651868
                            or profile["applicable_after_target_date"] != "2024-12-30"):
                        raise ValueError("The matching E3 posterior serving calibration is missing or changed.")
                    cal_raw = np.asarray(cohort["calibration_raw"], dtype=np.float64)
                    cal_y = np.asarray(cohort["calibration_y_original"], dtype=np.float64)
                    cal_dates = [str(x) for x in cohort["calibration_dates"]]
                    if cal_raw.shape != (53, 7, 5) or cal_dates != profile["calibration_dates"]:
                        raise ValueError("E3 calibration inputs or chronology changed.")
                    cal_outputs = np.asarray([inference._run(window, meta, session) for window in cal_raw])
                    residual_error = float(np.max(np.abs(np.abs(cal_outputs-cal_y)
                                                    -np.asarray(profile["absolute_calibration_residuals_sigma7"]))))
                    if not np.isfinite(residual_error) or residual_error > TOLERANCE:
                        raise ValueError("E3 calibration residuals fail parity with the saved posterior profile.")
                    calibration = {"model":model_id,"dates":53,"q_sigma7":profile["q_sigma7"],
                                   "applicable_after_target_date":profile["applicable_after_target_date"],
                                   "maximum_absolute_residual_error_vs_saved":residual_error,
                                   "q_refitted":False,"scope":"Posterior interval adoption only; no prospective coverage guarantee"}
            frozen = np.asarray(bundle["predictions"]["models"][model_id]["predictions"], dtype=np.float64)
            maximum = float(np.max(np.abs(outputs-frozen)))
            if (frozen.shape != (608,) or not np.isfinite(outputs).all() or (outputs < 0).any()
                    or not np.isfinite(maximum) or maximum > TOLERANCE):
                raise ValueError("Serving parity failed for " + model_id)
            records.append({"model":model_id,"slug":EXPECTED[model_id],"seed":271801,"dates":608,
                            "maximum_absolute_error_vs_frozen":maximum,
                            "absolute_tolerance_original_sigma7":TOLERANCE,
                            "RMSE_served":float(np.sqrt(np.mean((outputs-y)**2))),
                            "RMSE_frozen":float(np.sqrt(np.mean((frozen-y)**2))),
                            "model_sha256":meta["model_sha256"]})
        forbidden = [name for name in sys.modules if name == "torch" or name.startswith("torch.")
                     or name == "training" or name.startswith("training.") or name.startswith("src.e3_")]
        if forbidden:
            raise ValueError("The serving verification imported training code: " + repr(forbidden))
    finally:
        inference.clear_cache()
    return {"schema_version":2,"status":"passed_five_model_CPU_parity",
            "checked_at_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
            "models":records,"npz_sha256":NPZ_SHA,"Torch_imported":False,
            "E3_posterior_calibration":calibration,"evaluation_forward_windows":5*608,
            "additional_calibration_forward_windows":53,"new_training_runs":0,
            "new_calibration_fits":0,"GPU_used":False,"public_Render_acceptance":False,
            "scope":"Serving parity for fixed seed271801 on the known608-date cohort; no new superiority test",
            "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path(__file__),ROOT/"e3_dashboard/inference.py",
                              ROOT/"artifacts/e3_dashboard_v1/manifest.json",
                              ROOT/"artifacts/e3_serving_extensions_v2/manifest.json",
                              ROOT/"artifacts/e3_serving_calibration_v2/manifest.json")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.npz, args.download)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"models":len(record["models"]),"dates_each":608,
                      "E3_calibration_dates":53,"Torch_imported":False,"parity":"verified"}))
