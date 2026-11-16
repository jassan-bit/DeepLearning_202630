"""Reproduce a post-hoc diagnostic of the already-used 116-date development set.

No fitting, network access, checkpoint loading, calibration/evaluation members,
statistical inference, new benchmark or novelty/superiority claim is involved.
The two prediction CSVs and their original summary are bound to a fixed GitHub
snapshot; the raw NPZ is independently admitted before its three training keys.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np


RESULTS_COMMIT = "f7826d120f3a6eb8e9a0952dd9842cd77f152f34"
DATA_COMMIT = "af3f1223ad5899ed63016d2af2d09bae8470ab00"
NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
SNAPSHOT_SHA256 = "159943471990119a0ae80be7f72c9f4b2853bf75f6343d83ace3df9921642742"
DATA_PATH = Path("book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz")
OUTPUT = Path("experiments/e3_revision_post_piloto_20261116/audit")
TRAINING_KEYS = ("training_raw", "training_y_original", "training_dates")
MODELS = ("full", "M00")
START, END = "2024-07-08", "2024-10-31"
TOLERANCE_VARIANCE = 1e-12
CHECK_ATOL, CHECK_RTOL = 1e-14, 1e-12
SCOPE = ("Exploratory post-hoc diagnostic on development validation already reused for early stopping. "
         "Projection and the floor-only reference are not new trained models, confirmatory benchmarks, "
         "or evidence of R1/R3. Original pilot metrics remain unchanged.")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validation_dates():
    first, last = date.fromisoformat(START), date.fromisoformat(END)
    return [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]


def variance_floor(closes):
    """Return sqrt(A), A=sum((r_i-mean_6)^2)/7 from seven known closes.

    For the unknown seventh return x, population variance is exactly
    A + (6/49)*(x-mean_6)^2. This is not the six-return population SD.
    """
    closes = np.asarray(closes, dtype=np.float64)
    require(closes.ndim >= 1 and closes.shape[-1] == 7, "Seven known closes are required")
    require(np.isfinite(closes).all() and (closes > 0).all(), "Known closes must be positive and finite")
    returns = np.diff(np.log(closes), axis=-1)
    centered = returns - returns.mean(axis=-1, keepdims=True)
    return np.sqrt(np.square(centered).sum(axis=-1) / 7.0)


def project_to_floor(predictions, floor):
    """Diagnostic Euclidean projection; never modifies the saved predictions."""
    predictions, floor = np.asarray(predictions, dtype=np.float64), np.asarray(floor, dtype=np.float64)
    require(predictions.shape == floor.shape and np.isfinite(predictions).all()
            and np.isfinite(floor).all() and (floor >= 0).all(), "Invalid projection inputs")
    return np.maximum(predictions, floor)


def load_validation_data(npz_path, expected_sha256=NPZ_SHA256, *, expected_dates=None):
    """Read only the three original training members, after file admission."""
    raw_bytes = Path(npz_path).read_bytes()
    require(digest(raw_bytes) == expected_sha256, "NPZ SHA-256 mismatch")
    try:
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as archive:
            raw = np.asarray(archive[TRAINING_KEYS[0]], dtype=np.float64).copy()
            targets = np.asarray(archive[TRAINING_KEYS[1]], dtype=np.float64).copy()
            dates = np.asarray(archive[TRAINING_KEYS[2]]).copy()
    except KeyError as error:
        raise ValueError("Missing training-only NPZ member") from error
    require(raw.ndim == 3 and raw.shape[0] > 0 and raw.shape[1:] == (7, 5) and targets.shape == dates.shape == (len(raw),),
            "Invalid training raw/target/date schema")
    require(np.isfinite(raw).all() and np.isfinite(targets).all() and (targets >= 0).all(),
            "Nonfinite or negative admitted values")
    text_dates = [str(value) for value in dates]
    require(all(date.fromisoformat(value).isoformat() == value for value in text_dates)
            and text_dates == sorted(set(text_dates)) and text_dates[-1] <= "2024-10-31",
            "Invalid training date identity")
    selected = (dates >= START) & (dates <= END)
    actual_dates = dates[selected].tolist()
    require(actual_dates == (validation_dates() if expected_dates is None else list(expected_dates)),
            "Development validation dates mismatch")
    return {"raw": raw[selected], "targets": targets[selected], "dates": actual_dates,
            "source_training_count": len(raw), "keys_read": list(TRAINING_KEYS)}


def load_result_inputs(directory):
    """Admit exact original text bytes; no mutable branch or network fallback."""
    directory = Path(directory)
    snapshot_bytes = (directory / "snapshot_manifest.json").read_bytes()
    require(digest(snapshot_bytes) == SNAPSHOT_SHA256, "Pinned snapshot SHA-256 mismatch")
    snapshot = json.loads(snapshot_bytes)
    payloads = {}
    for local, remote in [("summary.json", "summary.json"),
                          *[(name + "_validation_predictions.csv", name + "/validation_predictions.csv") for name in MODELS]]:
        payload = (directory / local).read_bytes()
        record = snapshot["files"][remote]
        require(len(payload) == record["bytes"] and digest(payload) == record["sha256"],
                "Pinned result size/hash mismatch: " + local)
        payloads[local] = payload
    summary = json.loads(payloads["summary.json"])
    require(summary["status"] == "completed" and not snapshot["active_session"], "Pilot snapshot is not closed")
    for field in ("data_fingerprint", "plan_sha256", "source_hashes"):
        require(summary[field] == snapshot[field], "Pilot fingerprint mismatch: " + field)
    predictions = {}
    for name in MODELS:
        reader = csv.DictReader(io.StringIO(payloads[name + "_validation_predictions.csv"].decode()))
        require(reader.fieldnames == ["date", "target_sigma7", "prediction_sigma7"], "Unexpected prediction CSV schema")
        predictions[name] = list(reader)
    return snapshot, summary, predictions, {name: digest(raw) for name, raw in payloads.items()}


def _model_metrics(pred, targets, floor, quartiles):
    error, below = pred - targets, pred < floor
    squared, absolute = np.square(error), np.abs(error)
    projected = project_to_floor(pred, floor)
    projected_error = projected - targets
    require(np.all(np.abs(projected_error) <= absolute + 1e-15), "Projection increased a pointwise error")
    total_squared, total_absolute = float(squared.sum()), float(absolute.sum())
    require(total_squared > 0 and total_absolute > 0, "Pilot error totals unexpectedly vanish")
    result = {"original_rmse": float(np.sqrt(squared.mean())), "original_mae": float(absolute.mean()),
              "bias_prediction_minus_target": float(error.mean()),
              "below_floor_count": int(below.sum()), "below_floor_fraction": float(below.mean()),
              "below_floor_squared_error_share": float(squared[below].sum() / total_squared),
              "below_floor_absolute_error_share": float(absolute[below].sum() / total_absolute),
              "below_floor_mean_deficit": float((floor[below] - pred[below]).mean()) if below.any() else None,
              "below_floor_max_deficit": float(np.maximum(floor - pred, 0).max()),
              "projected_posthoc_rmse": float(np.sqrt(np.square(projected_error).mean())),
              "projected_posthoc_mae": float(np.abs(projected_error).mean()),
              "projection_squared_error_reduction_fraction": float(1 - np.square(projected_error).sum() / total_squared),
              "projection_absolute_error_reduction_fraction": float(1 - np.abs(projected_error).sum() / total_absolute),
              "projection_pointwise_error_increase_count": 0, "target_quartiles": []}
    for quartile in range(1, 5):
        selected = quartiles == quartile
        require(selected.any(), "Empty descriptive target quartile")
        result["target_quartiles"].append({"quartile": quartile, "n": int(selected.sum()),
            "target_min": float(targets[selected].min()), "target_max": float(targets[selected].max()),
            "target_mean": float(targets[selected].mean()), "prediction_mean": float(pred[selected].mean()),
            "bias": float(error[selected].mean()), "rmse": float(np.sqrt(squared[selected].mean())),
            "mae": float(absolute[selected].mean()), "below_floor_count": int(below[selected].sum()),
            "squared_error_share": float(squared[selected].sum() / total_squared)})
    return result, projected


def generate(root):
    root = Path(root)
    snapshot, summary, predictions, input_hashes = load_result_inputs(root / OUTPUT / "inputs")
    data = load_validation_data(root / DATA_PATH)
    require(data["source_training_count"] == 655, "Unexpected original training cohort size")
    targets, dates = data["targets"], data["dates"]
    floor = variance_floor(data["raw"][:, :, 3])
    margins = np.square(targets) - np.square(floor)
    require(np.all(margins >= -TOLERANCE_VARIANCE), "Observed target violates the known-return floor")
    edges = np.quantile(targets, [0.25, 0.5, 0.75], method="linear")
    quartiles = np.searchsorted(edges, targets, side="right") + 1
    report = {"version": "e3-postpilot-development-audit-v1", "scope": SCOPE,
              "results_commit": RESULTS_COMMIT, "data_commit": DATA_COMMIT,
              "data_path": DATA_PATH.as_posix(), "npz_sha256": NPZ_SHA256,
              "snapshot_sha256": SNAPSHOT_SHA256, "input_result_sha256": input_hashes,
              "keys_read": data["keys_read"], "n": len(targets), "first_date": dates[0], "last_date": dates[-1],
              "new_training_performed": False, "calibration_scored": False, "evaluation_scored": False,
              "statistical_inference_performed": False, "R1_or_R3_claimed": False,
              "floating_comparison": {"absolute_tolerance": CHECK_ATOL, "relative_tolerance": CHECK_RTOL,
                  "strict_fields": "Hashes, identities, dates, counts and method metadata remain exact"},
              "floor_formula": "A=sum((six_known_log_returns-mean6)^2)/7; floor=sqrt(A)",
              "variance_identity": "sigma7_squared=A+(6/49)*(unknown_return-mean6)^2",
              "variance_tolerance": TOLERANCE_VARIANCE,
              "target_floor_violation_count": int((margins < -TOLERANCE_VARIANCE).sum()),
              "minimum_target_variance_minus_floor_variance": float(margins.min()),
              "quartile_probabilities": [0.25, 0.5, 0.75], "quartile_edges": edges.tolist(),
              "quartile_method": "linear empirical target quantiles; descriptive only, not preregistered",
              "floor_to_target_ratio_median": float(np.median(floor / targets)),
              "realized_A_over_target_squared_median": float(np.median(np.square(floor / targets))),
              "ratio_interpretation": "Realized pointwise proportion in this cohort, with a future target denominator; not R-squared, variance explained, or general predictability.",
              "floor_only_posthoc_reference": {"scope": "Analytic descriptive reference, not a new trained model or confirmatory benchmark",
                  "rmse": float(np.sqrt(np.square(floor - targets).mean())),
                  "mae": float(np.abs(floor - targets).mean()), "bias": float((floor - targets).mean())},
              "models": {}}
    details = []
    for name in MODELS:
        records = predictions[name]
        require([row["date"] for row in records] == dates, "Prediction date identity mismatch: " + name)
        csv_targets = np.array([float(row["target_sigma7"]) for row in records])
        pred = np.array([float(row["prediction_sigma7"]) for row in records])
        require(np.isfinite(pred).all() and (pred >= 0).all()
                and np.allclose(csv_targets, targets, rtol=0, atol=1e-15), "Prediction target/value mismatch: " + name)
        metrics, projected = _model_metrics(pred, targets, floor, quartiles)
        original = summary["models"][name]
        require(abs(metrics["original_rmse"] - original["best_validation_rmse"]) < 1e-15
                and abs(metrics["original_mae"] - original["best_validation_mae"]) < 1e-15,
                "Original pilot metrics do not reproduce: " + name)
        metrics["original_best_epoch"] = original["best_epoch"]
        report["models"][name] = metrics
        for index, day in enumerate(dates):
            error = pred[index] - targets[index]
            details.append({"model": name, "date": day, "target_sigma7": float(targets[index]),
                "original_prediction_sigma7": float(pred[index]), "floor_sigma7": float(floor[index]),
                "target_variance_minus_floor_variance": float(margins[index]),
                "below_floor": bool(pred[index] < floor[index]), "target_quartile": int(quartiles[index]),
                "error_prediction_minus_target": float(error), "absolute_error": float(abs(error)),
                "squared_error": float(error ** 2), "projected_posthoc_diagnostic": float(projected[index])})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(details[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(details)
    table = io.StringIO(newline="")
    table_writer = csv.DictWriter(table, fieldnames=["metodo", "RMSE", "MAE", "alcance"], lineterminator="\n")
    table_writer.writeheader()
    for name in MODELS:
        metrics = report["models"][name]
        table_writer.writerow({"metodo": name + " original", "RMSE": metrics["original_rmse"],
            "MAE": metrics["original_mae"], "alcance": "Desarrollo original; validación reutilizada para parada"})
    for name in MODELS:
        metrics = report["models"][name]
        table_writer.writerow({"metodo": name + " proyectado al piso", "RMSE": metrics["projected_posthoc_rmse"],
            "MAE": metrics["projected_posthoc_mae"], "alcance": "Diagnóstico post hoc; no modelo entrenado ni R3"})
    reference = report["floor_only_posthoc_reference"]
    table_writer.writerow({"metodo": "Piso solo (referencia)", "RMSE": reference["rmse"],
        "MAE": reference["mae"], "alcance": "Referencia analítica post hoc; no benchmark confirmatorio ni R3"})
    return {OUTPUT / "metrics.json": (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(),
            OUTPUT / "detail.csv": stream.getvalue().encode(), OUTPUT / "table.csv": table.getvalue().encode()}


def _compare_json(saved, generated, path=()):
    require(type(saved) is type(generated), "Saved diagnostic type differs: " + "/".join(path))
    if isinstance(generated, dict):
        require(saved.keys() == generated.keys(), "Saved diagnostic keys differ: " + "/".join(path))
        for key in generated:
            _compare_json(saved[key], generated[key], (*path, key))
    elif isinstance(generated, list):
        require(len(saved) == len(generated), "Saved diagnostic length differs: " + "/".join(path))
        for index, value in enumerate(generated):
            _compare_json(saved[index], value, (*path, str(index)))
    elif isinstance(generated, float) and path and path[0] in {
            "models", "floor_only_posthoc_reference", "quartile_edges", "floor_to_target_ratio_median",
            "realized_A_over_target_squared_median", "minimum_target_variance_minus_floor_variance"}:
        require(math.isfinite(saved) and math.isfinite(generated)
                and math.isclose(saved, generated, abs_tol=CHECK_ATOL, rel_tol=CHECK_RTOL),
                "Saved diagnostic numeric value differs: " + "/".join(path))
    else:
        require(saved == generated, "Saved diagnostic metadata/count differs: " + "/".join(path))


def compare_saved(relative, saved, generated):
    """Keep provenance strict while allowing small platform-dependent log ULPs."""
    if Path(relative).name == "metrics.json":
        _compare_json(json.loads(saved), json.loads(generated))
        return
    require(Path(relative).name in {"detail.csv", "table.csv"}, "Unknown diagnostic product")
    a = csv.DictReader(io.StringIO(saved.decode()))
    b = csv.DictReader(io.StringIO(generated.decode()))
    require(a.fieldnames == b.fieldnames, "Saved diagnostic CSV columns differ")
    saved_rows, generated_rows = list(a), list(b)
    require(len(saved_rows) == len(generated_rows), "Saved diagnostic CSV row count differs")
    numeric = ({"target_sigma7", "original_prediction_sigma7", "floor_sigma7",
                "target_variance_minus_floor_variance", "error_prediction_minus_target",
                "absolute_error", "squared_error", "projected_posthoc_diagnostic"}
               if Path(relative).name == "detail.csv" else {"RMSE", "MAE"})
    for index, (first, second) in enumerate(zip(saved_rows, generated_rows)):
        for key in b.fieldnames:
            if key in numeric:
                x, y = float(first[key]), float(second[key])
                require(math.isfinite(x) and math.isfinite(y)
                        and math.isclose(x, y, abs_tol=CHECK_ATOL, rel_tol=CHECK_RTOL),
                        f"Saved diagnostic CSV value differs: row {index + 1}, {key}")
            else:
                require(first[key] == second[key], f"Saved diagnostic CSV identity differs: row {index + 1}, {key}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="Recompute offline; strict provenance/counts and explicit floating tolerance")
    args = parser.parse_args(argv)
    outputs = generate(args.root)
    for relative, payload in outputs.items():
        path = args.root / relative
        if args.check:
            require(path.is_file(), "Saved diagnostic is missing: " + str(relative))
            compare_saved(relative, path.read_bytes(), payload)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    print("OK: post-hoc development audit reproduces; no fitting, calibration/test scoring or R1/R3 claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
