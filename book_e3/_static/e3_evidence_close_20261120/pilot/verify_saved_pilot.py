"""Independent standard-library arithmetic checks for the saved-pilot readout.

No training, inference, market download, bootstrap or hypothesis test is run.
The arithmetic deliberately computes a mean of individual RMSEs rather than
the RMSE of an ensemble-mean prediction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(root):
    source = root / "archive" / "payload"
    generated = root / "generated"
    receipt = json.loads((source / "pilot_receipt.json").read_bytes())
    output = json.loads((generated / "readout.json").read_bytes())
    for name, fingerprint in receipt["artifact_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == fingerprint, name
    predictions = read(source / "assessment_predictions.csv")
    groups = {}
    for row in predictions:
        groups.setdefault((row["name"], row["seed"]), []).append(row)
    assert len(groups) == 17 and len(predictions) == 935
    individual = {}
    errors = {}
    for identity, rows in groups.items():
        rows.sort(key=lambda r: r["date"])
        assert len(rows) == 55 and len({r["date"] for r in rows}) == 55
        error = [float(r["prediction"])-float(r["target"]) for r in rows]
        errors[identity] = error
        individual[identity] = {"RMSE": math.sqrt(math.fsum(e*e for e in error)/55),
                                "MAE": math.fsum(abs(e) for e in error)/55,
                                "bias": math.fsum(error)/55}
    saved = read(generated / "metrics_recomputed.csv")
    assert len(saved) == 7
    for row in saved:
        values = [r for (name, _), r in individual.items() if name == row["name"]]
        for key, generated_key in (("RMSE", "mean_individual_RMSE"),
                                   ("MAE", "mean_individual_MAE"),
                                   ("bias", "mean_bias_prediction_minus_target")):
            actual = math.fsum(r[key] for r in values)/len(values)
            assert math.isclose(actual, float(row[generated_key]), rel_tol=1e-12, abs_tol=1e-15), row["name"]
        assert len(values) == int(row["trajectories"])
        final = [r for r in read(source / "final_assessment_scores.csv") if r["name"] == row["name"] and r["CRPS_U"]]
        if final:
            crps = math.fsum(float(r["CRPS_U"]) for r in final)/len(final)
            assert math.isclose(crps, float(row["source_mean_CRPS_U_not_recomputed"]), rel_tol=1e-12)
        else:
            assert row["source_mean_CRPS_U_not_recomputed"] == ""
    full = next(r for r in saved if r["name"] == "atom_crps")
    best = min((r for r in saved if r["name"] != "atom_crps"), key=lambda r:float(r["mean_individual_RMSE"]))
    gain = 1.-float(full["mean_individual_RMSE"])/float(best["mean_individual_RMSE"])
    assert best["name"] == "atom_mse_only" and int(full["RMSE_rank_of_7"]) == 5
    assert math.isclose(gain, receipt["relative_gain_vs_best_all"], rel_tol=1e-12, abs_tol=1e-15)
    assert gain < 0 and receipt["continuation_heuristic_passed"] is False
    assert output["continuation_heuristic_passed"] is False
    assert len(read(generated / "residual_diagnostics.csv")) == 17
    monthly = read(generated / "monthly_diagnostics.csv")
    assert len(monthly) == 34
    for row in monthly:
        trajectory = groups[row["name"], row["seed"]]
        selected = [r for r in trajectory if r["date"].startswith(row["month"])]
        assert len(selected) == int(row["dates"]) in (24, 31)
        error = [float(r["prediction"])-float(r["target"]) for r in selected]
        rmse = math.sqrt(math.fsum(e*e for e in error)/len(error))
        assert math.isclose(rmse, float(row["RMSE"]), rel_tol=1e-12)
    correlations = read(generated / "residual_lag_correlations.csv")
    assert len(correlations) == 119
    assert all(-1 <= float(r["residual_correlation"]) <= 1 for r in correlations)
    for row in correlations:
        error = errors[row["name"], row["seed"]]
        lag = int(row["lag"])
        left, right = error[:-lag], error[lag:]
        left_mean, right_mean = math.fsum(left)/len(left), math.fsum(right)/len(right)
        numerator = math.fsum((a-left_mean)*(b-right_mean) for a, b in zip(left, right))
        denominator = math.sqrt(math.fsum((a-left_mean)**2 for a in left)*math.fsum((b-right_mean)**2 for b in right))
        assert math.isclose(numerator/denominator, float(row["residual_correlation"]), rel_tol=1e-11, abs_tol=1e-13)
    assert all(output[k] == 0 for k in ("new_real_data_fits", "new_model_inference", "new_statistical_resamples"))
    assert not any(output[k] for k in ("p_values_computed", "prospective_power_certified", "R1_accredited", "R3_accredited"))
    return {"status": "passed_independent_standard_library_arithmetic", "source_artifacts_sha256_verified": 5,
            "trajectory_metrics_verified": 17, "recipe_means_verified": 7,
            "residual_rows": 17, "monthly_rows": 34, "lag_correlation_rows": 119,
            "new_real_data_fits": 0, "new_model_inference": 0, "new_statistical_resamples": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
