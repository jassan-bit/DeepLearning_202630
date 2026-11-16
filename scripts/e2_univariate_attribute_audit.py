"""Posterior, fixed35-coordinate train-only OLS screen of the real E2 inputs.

This is an attribute diagnostic, never model/feature selection. --prepare fixes
the design without fitting; --run requires separate QA for that exact plan.
The caller reserves/charges execution through e2_corrections_budget.py.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
SCI = Path("book_e2/_static/e2_scientific_closure_20261116")
BASE = ROOT / SCI / "univariate_attributes"
VERSION = "e2-univariate-attribute-audit-v1"
FEATURES = ("open", "high", "low", "close", "volume")
SIZES = {"training": 655, "calibration": 53, "evaluation": 608}
ENDPOINTS = {
    "training": ("2023-01-08", "2024-10-31"),
    "calibration": ("2024-11-08", "2024-12-30"),
    "evaluation": ("2025-01-01", "2026-08-31"),
}
INPUTS = (
    SCI / "data_audit/current_cohort_arrays.npz",
    SCI / "data_audit/cohort_array_manifest.json",
    SCI / "data_audit/cohort_rows.csv",
    SCI / "data_audit/audit_result.json",
    SCI / "data_audit/audit_specification.json",
    SCI / "data_audit/reconstructed_training_scaler.json",
    Path("experiments/e2_reduced_protocol_20261112/protocol.json"),
    Path("scripts/e2_current_data_closure.py"),
    Path("scripts/e2_development_data.py"),
    Path("scripts/e2_real_executor.py"),
    Path("scripts/e2_temporal_training.py"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def coordinate_rows():
    return [
        {"coordinate": day * 5 + channel, "lag_days": 7 - day,
         "channel": name, "name": f"{name}_lag{7-day}",
         "raw_bar_offset_days": day - 7,
         "normalization_anchor_offset_days": None if name == "volume" else -1,
         "latest_required_bar_offset_days": day - 7 if name == "volume" else -1,
         "available_strictly_before_target": True,
         "transformation": "log1p(volume), then train-only scaling" if name == "volume"
         else "log(price)-log(last-known-close), then train-only scaling"}
        for day in range(7) for channel, name in enumerate(FEATURES)
    ]


def validate_coordinates(rows):
    require(rows == coordinate_rows(), "Coordinate order/lag/availability differs from D7x5 causal schema")
    require(all(r["latest_required_bar_offset_days"] < 0 for r in rows), "Current or future input is forbidden")


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), "Use a new plan path; registered plans are immutable")
    specification = json.loads((ROOT / INPUTS[4]).read_text())
    contract = specification["contract"]
    require(contract["D"] == 7 and tuple(contract["features"]) == FEATURES
            and contract["horizon_days"] == 1 and contract["future_or_current_outcome_features_allowed"] is False,
            "Original causal data contract differs")
    records = [{"path": p.as_posix(), "bytes": (ROOT / p).stat().st_size,
                "sha256": sha(ROOT / p)} for p in INPUTS]
    plan = {
        "version": VERSION, "registered_at_utc": utc(),
        "design_role": "post-benchmark attribute diagnostic; known inherited holdout",
        "historical_preregistration_or_blindness_claimed": False,
        "script_sha256": sha(__file__), "inputs": records,
        "features": list(FEATURES), "coordinates": coordinate_rows(),
        "cohort_sizes": SIZES, "cohort_endpoints": ENDPOINTS,
        "X": "actual float32 training/calibration/evaluation_X, promoted to float64 for OLS arithmetic",
        "y": "original nonannualized population sigma7; *_y_original",
        "method": "35 separate OLS regressions with intercept, each fitted only on655training pairs; frozen coefficients score all3cohorts",
        "constant_training_coordinate": "slope0,intercept=training target mean; never fit using calibration/test values",
        "constant_target_score": "R2=null if within-cohort total squared deviation is zero; retain row and RMSE",
        "R2": "1-SSE/sum((y-cohort_mean(y))**2); negative out-of-sample values retained",
        "suspicion_rule": {"R2_at_least": .95, "cohorts": list(SIZES),
                           "action": "Report every flagged coordinate and its dated causal availability; review before approving attribute check; never remove features/retune by this screen"},
        "availability_rule": "Every coordinate uses a raw bar t-k,k1..7; relative OHLC also needs Close(t-1); learned scaler uses train cutoff2024-10-31 only",
        "runtime": {"python_major": 3, "numpy": "2.3.5"},
        "budget": {"cap_seconds": 30, "charging": "external e2_corrections_budget.py wrapper; actual scientific command walltime including failure"},
        "new_OLS_fits": 35, "new_deep_learning_fits": 0,
        "new_bootstrap_or_hypothesis_tests": 0, "HPO_or_feature_selection": False,
        "original_A2_RQ1_changed": False, "QA_required_before_run": True,
        "limitations": [
            "A high univariate R2 is a review flag, not proof of leakage; low R2 does not exclude nonlinear/multivariate leakage.",
            "One known chronology and overlapping windows; these are descriptive predictive risks, not independent tests or confirmation.",
            "Calibration/test targets enter scoring denominators and residuals only, never OLS fitting/scaling/model selection.",
            "Availability is checked against the pinned causal reconstruction, not an independent timestamp-of-publication feed.",
        ],
    }
    validate_coordinates(plan["coordinates"])
    write_json(output, plan)
    return plan


def array_sha(values):
    import numpy as np
    array = np.asarray(values)
    require(not array.dtype.hasobject, "Object arrays are forbidden")
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    header = json.dumps({"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def validate_ids(ids, cohort, expected_count=None):
    ids = list(map(str, ids))
    require(len(ids) == (SIZES[cohort] if expected_count is None else expected_count), "Unexpected cohort count")
    require(ids == sorted(set(ids)), "Date IDs must be unique and ordered")
    for value in ids:
        require(date.fromisoformat(value).isoformat() == value, "Date IDs must be canonical ISO days")
    return ids


def fit_ols(x, y):
    import numpy as np
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    require(x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) >= 2
            and np.isfinite(x).all() and np.isfinite(y).all(), "Invalid finite training pairs")
    xm, ym = float(x.mean()), float(y.mean())
    centered_x, centered_y = x - xm, y - ym
    sxx = float(np.dot(centered_x, centered_x))
    constant = sxx == 0
    slope = 0.0 if constant else float(np.dot(centered_x, centered_y) / sxx)
    intercept = ym - slope * xm
    require(np.isfinite([intercept, slope]).all(), "Nonfinite OLS coefficients")
    return {"intercept": intercept, "slope": slope, "constant_training_coordinate": constant,
            "training_N": len(x), "training_x_mean": xm, "training_y_mean": ym}


def score_ols(coefficients, x, y):
    import numpy as np
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    require(x.ndim == y.ndim == 1 and len(x) == len(y) and len(x) > 0
            and np.isfinite(x).all() and np.isfinite(y).all(), "Invalid finite scoring pairs")
    prediction = coefficients["intercept"] + coefficients["slope"] * x
    residuals = prediction - y
    sse, sst = float(np.dot(residuals, residuals)), float(np.dot(y-y.mean(), y-y.mean()))
    r2 = None if sst == 0 else 1 - sse/sst
    result = {"N": len(y), "R2": r2, "RMSE": float(np.sqrt(sse/len(y)))}
    require(np.isfinite(result["RMSE"]) and (r2 is None or np.isfinite(r2)), "Nonfinite score")
    return result


def admit(plan):
    import numpy as np
    require(plan["version"] == VERSION and plan["script_sha256"] == sha(__file__), "Registered source changed")
    validate_coordinates(plan["coordinates"])
    require(plan["features"] == list(FEATURES) and plan["cohort_sizes"] == SIZES
            and plan["suspicion_rule"] == {"R2_at_least": .95, "cohorts": list(SIZES), "action": "Report every flagged coordinate and its dated causal availability; review before approving attribute check; never remove features/retune by this screen"},
            "Registered diagnostic design differs")
    require([r["path"] for r in plan["inputs"]] == [p.as_posix() for p in INPUTS], "Input identity set differs")
    for row in plan["inputs"]:
        p = ROOT / row["path"]
        require(p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], "Pinned input changed: " + row["path"])
    require(importlib.metadata.version("numpy") == plan["runtime"]["numpy"], "Pinned NumPy version required")
    manifest = json.loads((ROOT / INPUTS[1]).read_text())
    require(manifest["sha256"] == sha(ROOT / INPUTS[0])
            and manifest["future_data_or_target_covariates_added"] is False,
            "Actual cohort identity/causality differs")
    arrays = {}
    with np.load(ROOT / INPUTS[0], allow_pickle=False) as stored:
        for cohort, count in SIZES.items():
            group = {}
            for suffix in ("X", "y_original", "dates"):
                key = cohort + "_" + suffix
                value = stored[key].copy()
                descriptor = manifest["arrays"][key]
                require(list(value.shape) == descriptor["shape"] and value.dtype.str == descriptor["dtype"]
                        and array_sha(value) == descriptor["array_sha256"], "Cohort array changed: " + key)
                group[suffix] = value
            require(group["X"].shape == (count, 7, 5) and group["y_original"].shape == (count,)
                    and np.isfinite(group["X"]).all() and np.isfinite(group["y_original"]).all(), "Invalid cohort shape/values")
            ids = validate_ids(group["dates"], cohort)
            require((ids[0], ids[-1]) == ENDPOINTS[cohort], "Cohort endpoints differ")
            arrays[cohort] = group
    sets = [set(arrays[k]["dates"].tolist()) for k in SIZES]
    require(all(not sets[a] & sets[b] for a in range(3) for b in range(a+1,3)), "Outcome IDs overlap")
    scaler = json.loads((ROOT / INPUTS[5]).read_text())
    require(scaler["train_dates"] == arrays["training"]["dates"].tolist(), "Scaler used a different training cohort")
    rows = list(csv.DictReader((ROOT / INPUTS[2]).open()))
    expected = {(cohort, str(day)) for cohort in SIZES for day in arrays[cohort]["dates"]}
    require(len(rows) == sum(SIZES.values()) and {(r["cohort"], r["outcome_day_utc"]) for r in rows} == expected,
            "Availability rows do not identify all three actual cohorts")
    for row in rows:
        target = date.fromisoformat(row["outcome_day_utc"])
        require(row["input_first_day"] == (target-timedelta(days=7)).isoformat()
                and row["input_last_day"] == (target-timedelta(days=1)).isoformat(), "Current/future bar in availability support")
    return arrays


def run(plan_path, qa_path, output):
    plan_path, qa_path, output = (Path(p).resolve() for p in (plan_path, qa_path, output))
    started = time.monotonic()
    require(not output.exists(), "Fresh result path required")
    plan, qa = json.loads(plan_path.read_text()), json.loads(qa_path.read_text())
    require(qa.get("approved_for_univariate_attribute_audit") is True
            and qa.get("plan_sha256") == sha(plan_path)
            and qa.get("script_sha256") == sha(__file__), "Independent approval for this exact source and plan required")
    arrays = admit(plan)
    results, coefficients = [], []
    threshold = plan["suspicion_rule"]["R2_at_least"]
    for coordinate in plan["coordinates"]:
        day, channel = divmod(coordinate["coordinate"], 5)
        fitted = fit_ols(arrays["training"]["X"][:, day, channel], arrays["training"]["y_original"])
        coefficients.append({**coordinate, **fitted})
        for cohort in SIZES:
            score = score_ols(fitted, arrays[cohort]["X"][:, day, channel], arrays[cohort]["y_original"])
            flag = score["R2"] is not None and score["R2"] >= threshold
            results.append({"coordinate": coordinate["coordinate"], "attribute": coordinate["name"],
                            "channel": coordinate["channel"], "lag_days": coordinate["lag_days"],
                            "cohort": cohort, **score, "suspicious_R2": flag,
                            "latest_required_bar_offset_days": coordinate["latest_required_bar_offset_days"],
                            "causal_availability_verified": True,
                            "coefficient_fit_cohort": "training655_only"})
    flagged = [r for r in results if r["suspicious_R2"]]
    summary = {"version": VERSION, "status": "REVIEW_REQUIRED" if flagged else "PASS_declared_univariate_screen",
               "plan_sha256": sha(plan_path), "QA_sha256": sha(qa_path), "script_sha256": sha(__file__),
               "coordinates": 35, "scored_rows": len(results), "OLS_fits": 35, "deep_learning_fits": 0,
               "suspicious_R2_threshold": threshold, "flagged_coordinate_cohorts": len(flagged),
               "flagged_rows": flagged, "all_availability_offsets_strictly_past": True,
               "OLS_fit_data": "training655 only", "calibration_or_test_used_for_fit": False,
               "new_hypothesis_tests_or_bootstrap": 0, "feature_or_model_selection": False,
               "original_A2_RQ1_changed": False, "limitations": plan["limitations"]}
    output.mkdir(parents=True)
    for name, rows in (("univariate_risks_105.csv", results), ("OLS_coefficients_35.csv", coefficients)):
        with (output / name).open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output / "summary.json", summary)
    write_json(output / "execution_receipt.json", {"completed_at_utc": utc(), "version": VERSION,
               "plan_sha256": sha(plan_path), "QA_sha256": sha(qa_path), "script_sha256": sha(__file__),
               "elapsed_compute_wall_seconds": time.monotonic()-started,
               "charge_scope": "Caller wrapper charges actual whole scientific command including import; do not add this inner elapsed again",
               "runtime": {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy")}})
    (output / "README.md").write_text(
        "# Pantalla univariada posterior de atributos\n\n"
        "35 regresiones OLS con intercepto ajustadas sólo en train655; 105 filas de R² y RMSE. "
        "Se conservan R² negativos y coordenadas constantes. Cal53/test608 sólo se puntúan, sin seleccionar atributos. "
        "R²≥0,95 activa revisión; un valor menor no prueba ausencia general de fuga. "
        "Los desplazamientos de disponibilidad se verifican contra las cohortes y el contrato causal fijados. "
        "No se modifican redes, escaladores, A2 o RQ1; no hay bootstrap ni pruebas nuevas.\n",
        encoding="utf-8")
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(output.iterdir())]
    write_json(output / "manifest.json", {"version": VERSION, "files": files, "inputs": plan["inputs"],
               "analysis_plan_path": plan_path.relative_to(ROOT).as_posix(), "analysis_plan_sha256": sha(plan_path),
               "QA_path": str(qa_path), "QA_sha256": sha(qa_path), "script_sha256": sha(__file__),
               "scope": "Posterior attribute diagnostic only; finite univariate check, no universal absence-of-leakage proof"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--plan", type=Path, default=BASE / "plan_v1/analysis_plan.json")
    parser.add_argument("--qa", type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "result_v1")
    args = parser.parse_args()
    if args.prepare:
        plan = prepare(args.plan)
        print(json.dumps({"plan_path": str(args.plan), "plan_sha256": sha(args.plan), "OLS_fits": 0,
                          "status": "prepared_before_attribute_calculation", "version": plan["version"]}))
    else:
        if args.qa is None:
            parser.error("--qa is required before numeric execution")
        print(json.dumps(run(args.plan, args.qa, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
