"""Execute supplementary14 parity/API/resource checks; no training imports.

Run inside the clean profile image. Inputs are the existing SHA-bound 655/53/608
cohort and133-predictor snapshot, mounted read-only. Print one JSON receipt.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import numpy as np


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_window(raw, target):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["date", "open", "high", "low", "close", "volume"])
    target = date.fromisoformat(str(target))
    writer.writerows([[(target - timedelta(days=7-i)).isoformat(), *r] for i, r in enumerate(raw)])
    return stream.getvalue()


def run(args):
    started = perf_counter()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    require(digest(args.cohort) == "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683", "Cohort identity changed")
    require(digest(args.predictions) == "df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655", "Historical prediction snapshot changed")
    import e3_profile14
    from e3_dashboard_v3 import runtime
    import onnxruntime as ort
    require(np.__version__ == "2.3.5" and ort.__version__ == "1.23.2", "Runtime cohort changed")
    prior = json.loads(args.predictions.read_bytes())
    models = runtime.catalogue()
    require(len(models) == 14 and not any("V3" in r["id"] for r in models), "Historical14/V3 boundary changed")
    client = e3_profile14.server.test_client()
    with np.load(args.cohort, allow_pickle=False) as data:
        raw = data["evaluation_raw"].copy()
        cal_raw = data["calibration_raw"].copy()
        dates = [str(x) for x in data["evaluation_dates"]]
        cal_dates = [str(x) for x in data["calibration_dates"]]
    require(dates == prior["dates"] and raw.shape == (608, 7, 5), "Dates/window shape changed")
    parity = []
    for row in runtime.profile()["models"]:
        if row["legacy"]:
            continue
        bound, meta = runtime.metadata(row["id"])
        session = runtime._session(bound)
        frozen = [p for p in prior["predictors"] if p["model"] == row["id"] and str(p["seed"]) == "271801"]
        require(len(frozen) == 1, "Non-unique historical predictor")
        ref = frozen[0]
        observed = [runtime._run(w, meta, bound, session) for w in raw]
        observed_cal = [runtime._run(w, meta, bound, session) for w in cal_raw]
        error = float(np.max(np.abs(np.asarray(observed) - ref["prediction_original"])))
        error_cal = float(np.max(np.abs(np.asarray(observed_cal) - ref["calibration_predictions"])))
        require(error <= 1e-6 and error_cal <= 1e-6, "New adapter parity failed: " + row["id"])
        require(meta["calibration_q"] == ref["calibrator_q"], "Frozen quantile changed")
        parity.append({"model": row["id"], "cases": 661, "evaluation_max_abs_error_sigma7": error,
                       "calibration_max_abs_error_sigma7": error_cal, "q_unchanged": meta["calibration_q"]})
    navigation = {}
    for route in ["/", "/eda", "/modelos", "/perfil-14", "/healthz", "/api/profile14/catalogue", "/api/profile14/example.csv"]:
        response = client.get(route)
        require(response.status_code == 200, "Route failed: " + route)
        navigation[route] = response.status_code
    health = client.get("/healthz").get_json()
    require(health["profile_predictors"] == 14 and health["original_UI_predictors"] == 5 and not health["V3_enabled"], "Health capability mismatch")
    api = []
    for model in models:
        payload = {"model_id": model["id"], "csv": csv_window(raw[0], dates[0]), "target_date": dates[0]}
        t = perf_counter()
        response = client.post("/api/profile14/predict", json=payload)
        elapsed = (perf_counter() - t)*1000
        require(response.status_code == 200, "API failed: " + model["id"])
        result = response.get_json()
        _, meta = runtime.metadata(model["id"])
        require(result["calibration_q"] == meta["calibration_q"], "Wrong predictor quantile")
        require(result["interval"] == [max(0.0, result["prediction"] - meta["calibration_q"]), result["prediction"] + meta["calibration_q"]], "Wrong interval units/formula")
        require(result["model_sha256"] == model["model_sha256"] and result["metadata_sha256"] == model["metadata_sha256"], "API fingerprints differ")
        require(len(result["explanation"]["attributions"]) == 7 and result["explanation_status"] == "sensitivity_not_causal", "Explanation contract failed")
        historical = client.post("/api/profile14/predict", json={"model_id": model["id"], "csv": csv_window(cal_raw[-1], cal_dates[-1]), "target_date": cal_dates[-1]}).get_json()
        require(historical["interval"] is None and historical["calibration_q"] is None, "Posterior calibration leaked into earlier target")
        ood_raw = raw[0].copy()
        ood_raw[:, 4] = 1e30
        ood = client.post("/api/profile14/predict", json={**payload, "csv": csv_window(ood_raw, dates[0])}).get_json()
        require(ood["ood"] and ood["ood_fraction"] > 0, "OOD warning missing")
        api.append({"model": model["id"], "prediction": result["prediction"], "q": result["calibration_q"],
                    "model_sha256": result["model_sha256"], "metadata_sha256": result["metadata_sha256"],
                    "HTTP_testclient_end_to_end_ms_including_explanation": elapsed,
                    "point_ms": result["latency_ms"], "explanation_ms": result["explanation"]["latency_ms"],
                    "historical_interval_suppressed": True, "OOD_warning_verified": True})
    valid = {"model_id": "MLP16", "csv": csv_window(raw[0], dates[0]), "target_date": dates[0]}
    invalid = [None, {}, {**valid, "model_id": "V3 atom_crps experimental"}, {**valid, "csv": ""},
               {**valid, "csv": "x" * 102401}, {**valid, "model_id": ["MLP16"]}, {**valid, "target_date": ["2025-01-01"]},
               {**valid, "csv": valid["csv"].replace(",99429.6,", ",nan,")}, {**valid, "extra": 1}]
    errors = []
    for index, body in enumerate(invalid):
        response = client.post("/api/profile14/predict", json=body)
        require(response.status_code == 400, "Malformed input not safely rejected: " + str(index))
        errors.append({"case": index, "status": response.status_code, "error": response.get_json()["error"]})
    require("torch" not in sys.modules, "Training framework imported")
    receipt = {"status": "supplementary_profile14_local_container_checks_passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": runtime.profile()["profile"], "profile_sha256": runtime.sha(runtime.PROFILE.read_bytes()),
        "python": platform.python_version(), "packages": {k: importlib.metadata.version(k) for k in ["numpy", "onnxruntime", "dash", "Flask", "gunicorn"]},
        "navigation": navigation, "health_observed": health, "adapter_historical_parity": parity,
        "adapter_parity_cases": 9*661, "absolute_tolerance_sigma7": 1e-6, "API_models": api, "invalid_inputs": errors,
        "process_peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "RSS_scope": "Full Flask/Dash test-client process including14-model switches,explanations,arrays; not incremental ORT RAM or publicHTTP/Render.",
        "elapsed_seconds": perf_counter()-started, "new_training_runs": 0, "new_calibration_fits": 0,
        "new_forecast_dates": 0, "ONNX_reexported": False, "GPU_used": False,
        "public_Render_acceptance": False, "AI_assistance_declared": True}
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/app"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    run(parser.parse_args())
