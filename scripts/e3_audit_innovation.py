"""Reproduce the closed V2 development audit from pinned CSV/JSON text only.

No NPZ or checkpoint is opened, no model is imported, and no fit or inference
is performed. Regimes use the already fixed fit-target q75. Error cases are
explicitly post-hoc diagnostics and never alter the full 116-date decision.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "experiments/e3_cierre_v2_20261116/audit"
COMMIT = "1ca029e869a517cb8de704e573c78ca1e0cbad3c"
SNAPSHOT_SHA256 = "5fe65596b99b334de197f3fc08811514c5ee6ea48b8447318057fea1b7cf9be1"
MODELS = ("floor_only", "floor_plus_fit_mean", "frozen_M00_projected",
          "additive_positive_head", "innovation_mean_only", "innovation_density_mean")
NEURAL = MODELS[3:]
REFERENCE, CANDIDATE, CASE_REFERENCE = MODELS[2], MODELS[5], MODELS[3]
ULPS = 8


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, ulps=ULPS):
    require(type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected), label + ": nonfinite/non-numeric")
    tolerance = ulps * max(math.ulp(float(actual)), math.ulp(float(expected)))
    require(abs(actual - expected) <= tolerance, label + ": numerical mismatch")


def load_inputs(directory, provenance_path=None):
    directory = Path(directory)
    provenance_path = Path(provenance_path) if provenance_path else directory.parent / "source_provenance.json"
    proof = json.loads(provenance_path.read_text())
    require(proof["repository"] == "jassan-bit/DeepLearning_202630"
            and proof["results_commit"] == COMMIT and proof["prefix"] == "runs/innovacion_v2"
            and proof["snapshot_manifest_sha256"] == SNAPSHOT_SHA256,
            "Pinned source provenance mismatch")
    snapshot_bytes = (directory / "snapshot_manifest.json").read_bytes()
    require(sha(snapshot_bytes) == SNAPSHOT_SHA256, "Snapshot manifest SHA-256 mismatch")
    snapshot, payloads = json.loads(snapshot_bytes), {}
    for name, record in proof["copied_text_files"].items():
        require(Path(name).as_posix() == name and not Path(name).is_absolute()
                and ".." not in Path(name).parts and Path(name).suffix in (".json", ".csv", ".jsonl", ".sha256"),
                "Only admitted relative text paths may be opened")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "Input is missing or symlinked: " + name)
        value = path.read_bytes()
        require(record == snapshot["files"][name]
                and len(value) == record["bytes"] and sha(value) == record["sha256"],
                "Input SHA-256/size mismatch: " + name)
        payloads[name] = value
    require(len(payloads) == 23 and proof["checkpoints_loaded"] is False
            and proof["market_fits_performed"] == 0 and proof["new_market_inferences_performed"] == 0,
            "The copied text-only input inventory changed")
    docs = {name[:-5]: json.loads(value) for name, value in payloads.items() if name.endswith(".json")}
    return snapshot, proof, payloads, docs


def _csv_rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def validate_metadata(docs, snapshot, payloads, source_root=ROOT):
    summary, manifest, plan = docs["summary"], docs["manifest"], docs["plan"]
    registry, ledger, admission, state = (docs[k] for k in ("global_registry", "ledger", "data_admission", "state"))
    require(snapshot["registry"] == registry, "Snapshot registry differs")
    for key in ("run_id", "claim_id", "plan_sha256"):
        require(summary[key] == manifest[key] == registry[key] == manifest["global_claim"][key],
                "Run identity mismatch: " + key)
    require(sha(canonical(plan)) == manifest["plan_sha256"], "Canonical plan hash mismatch")
    operational_plan = Path(source_root) / "experiments/e3_innovacion_v2_20261116/plan.json"
    require(sha(canonical(json.loads(operational_plan.read_text()))) == manifest["plan_sha256"],
            "Operational plan differs from the immutable run plan")
    require(sha(canonical(admission)) == summary["data_fingerprint"] == manifest["data_fingerprint"],
            "Data admission fingerprint mismatch")
    require(summary["source_hashes"] == manifest["source_hashes"], "Summary source identities differ")
    for name, digest in manifest["source_hashes"].items():
        path = Path(source_root) / name
        require(path.is_file() and sha(path.read_bytes()) == digest, "Scientific source changed: " + name)
    require(summary["status"] == state["status"] == "completed" and manifest["initialized"] is True
            and summary["comparison_complete"] is True and summary["pilot_incomplete"] is False,
            "The run is not complete")
    require(summary["calibration_scored"] is False and summary["evaluation_scored"] is False
            and summary["statistical_tests_performed"] is False and summary["winner_declared"] is False,
            "Unexpected confirmatory or forbidden-partition claim")
    require(admission["keys_read"] == ["training_raw", "training_y_original", "training_dates"]
            and admission["forbidden_arrays_loaded"] is False and admission["rows_dropped"] == 0
            and (admission["n_fit"], admission["n_purged_or_unused"], admission["n_validation"]) == (532, 7, 116),
            "Training-only cohort admission changed")
    require(admission["npz_sha256"] == plan["data"]["sha256"] == registry["npz_sha256"], "Input identity differs")
    require(ledger["active_session"] is None and registry["active_session"] is None
            and registry["phase"] == "ready", "The final registry/ledger must be closed")
    close(ledger["charged_seconds"], registry["charged_seconds"], "Registry charge")
    close(ledger["charged_seconds"], summary["charged_seconds"], "Summary charge")
    require(ledger["limit_seconds"] == registry["additional_limit_seconds"] == summary["limit_seconds"] == 600,
            "Additional budget changed")
    prior = plan["proposed_budget"]["prior_charged_seconds"]
    close(prior, 369.4779596389991, "Sealed prior cost")
    close(summary["prior_charged_seconds"], prior, "Summary prior cost")
    close(registry["prior_charged_seconds"], prior, "Registry prior cost")
    close(summary["global_charged_seconds"], prior + ledger["charged_seconds"], "Cumulative charge")
    require(summary["global_limit_seconds"] == registry["global_limit_seconds"] == 1200
            and ledger["charged_seconds"] <= 600 and summary["global_charged_seconds"] <= 1200,
            "Execution exceeded its admitted limits")
    totals = [e["charged_total_seconds"] for e in ledger["events"] if e["kind"] == "session_closed"]
    require(totals == sorted(totals) and totals and totals[-1] == ledger["charged_seconds"],
            "Closed-session costs are inconsistent or nonmonotonic")
    require(summary["models"].keys() == set(MODELS), "All six predictors are required")
    best_epochs = {}
    for name in NEURAL:
        rows = _csv_rows(payloads[name + "/curves.csv"])
        epochs = [int(r["epoch"]) for r in rows]
        require(epochs == list(range(1, len(rows) + 1)), "Epoch sequence is incomplete: " + name)
        best = min(rows, key=lambda r: (float(r["validation_RMSE_original"]), int(r["epoch"])))
        epoch = int(best["epoch"])
        model = summary["models"][name]
        require(model["complete"] is True and state["models"][name]["complete"] is True
                and model["completed_epochs"] == state["models"][name]["epoch"] == len(rows) == 40
                and model["best_epoch"] == epoch and model["parameters"] == 1210
                and manifest["models"][name]["trainable_parameters"] == 1210,
                "Neural recipe/epoch metadata mismatch: " + name)
        close(model["RMSE"], float(best["validation_RMSE_original"]), "Best-epoch RMSE")
        close(model["MAE"], float(best["validation_MAE_original"]), "Best-epoch MAE")
        for part in ("best", "last"):
            receipt = payloads[name + "/" + part + ".pt.sha256"].decode().strip()
            require(receipt == snapshot["files"][name + "/" + part + ".pt"]["sha256"],
                    "Checkpoint receipt differs from snapshot")
        best_epochs[name] = epoch
    require(manifest["controls_sha256"] == sha(payloads["controls.json"])
            and manifest["control_predictions_sha256"] == sha(payloads["controls/validation_predictions.csv"]),
            "Control artifact identities differ")
    return best_epochs


def align_predictions(payloads, expected_dates):
    grouped = {name: [] for name in MODELS}
    for row in _csv_rows(payloads["controls/validation_predictions.csv"]):
        require(row["model"] in MODELS[:3], "Unexpected analytic/frozen control")
        grouped[row["model"]].append(row)
    for name in NEURAL:
        grouped[name] = _csv_rows(payloads[name + "/validation_predictions.csv"])
    result, reference = {}, None
    require(expected_dates == sorted(set(expected_dates)), "Admission dates must be ordered and unique")
    for name in MODELS:
        rows = sorted(grouped[name], key=lambda r: r["date"])
        require([r["date"] for r in rows] == expected_dates, "Prediction date alignment mismatch: " + name)
        converted = []
        for row in rows:
            require(date.fromisoformat(row["date"]).isoformat() == row["date"], "Invalid prediction date")
            r = {"date": row["date"]}
            for key in ("target_sigma7", "prediction_sigma7", "known_floor", "scale_U", "NLL_U", "nominal_05", "nominal_95"):
                require(key in row, "Prediction column absent: " + key)
                value = float(row[key]) if row[key] else None
                require(value is None or math.isfinite(value), "Nonfinite prediction value")
                r[key] = value
            require(all(r[k] is not None and r[k] >= 0 for k in ("target_sigma7", "prediction_sigma7", "known_floor")),
                    "Invalid sigma/floor value")
            require(r["target_sigma7"] >= r["known_floor"] - 1e-14, "Observed target violates the known floor")
            prob = name in NEURAL[1:]
            require(all((r[k] is not None) == prob for k in ("scale_U", "NLL_U", "nominal_05", "nominal_95")),
                    "Probabilistic columns disagree with the frozen recipe")
            if prob:
                require(r["scale_U"] > 0 and r["known_floor"] <= r["nominal_05"] <= r["nominal_95"],
                        "Invalid nominal interval/innovation scale")
            converted.append(r)
        shared = [(r["date"], r["target_sigma7"], r["known_floor"]) for r in converted]
        require(reference is None or shared == reference, "Targets/floors differ across predictors")
        reference, result[name] = shared, converted
    return result


def regime(target, threshold):
    return "high" if target >= threshold else "normal"


def aggregate(rows, threshold):
    n = len(rows)
    require(n > 0, "Empty diagnostic regime")
    errors = [r["prediction_sigma7"] - r["target_sigma7"] for r in rows]
    sse = math.fsum(e * e for e in errors)
    bias = math.fsum(errors) / n
    variance = math.fsum((e - bias) ** 2 for e in errors) / n
    mse = sse / n
    require(abs(mse - bias * bias - variance) <= ULPS * max(math.ulp(mse), math.ulp(variance)),
            "MSE = bias squared + population residual variance failed")
    predicted_excess = math.fsum(r["prediction_sigma7"] - r["known_floor"] for r in rows) / n
    realized_excess = math.fsum(r["target_sigma7"] - r["known_floor"] for r in rows) / n
    result = {"n": n, "RMSE": math.sqrt(mse), "MAE": math.fsum(abs(e) for e in errors) / n,
        "bias": bias, "bias_squared": bias * bias, "residual_variance_ddof0": variance,
        "MSE": mse, "SSE": sse, "underestimation_count": sum(e < 0 for e in errors),
        "underestimation_rate": sum(e < 0 for e in errors) / n,
        "below_floor_count": sum(r["prediction_sigma7"] < r["known_floor"] - 1e-14 for r in rows),
        "mean_realized_excess": realized_excess, "mean_predicted_excess": predicted_excess,
        "maximum_absolute_error": max(abs(e) for e in errors), "NLL_U": None,
        "nominal_90pct_covered_count": None, "nominal_90pct_coverage": None,
        "mean_nominal_interval_width": None, "threshold_fit_q75": threshold}
    if rows[0]["NLL_U"] is not None:
        count = sum(r["nominal_05"] <= r["target_sigma7"] <= r["nominal_95"] for r in rows)
        result.update(NLL_U=math.fsum(r["NLL_U"] for r in rows) / n,
            nominal_90pct_covered_count=count, nominal_90pct_coverage=count / n,
            mean_nominal_interval_width=math.fsum(r["nominal_95"] - r["nominal_05"] for r in rows) / n)
    return result


def _csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def analyze(models, threshold, docs, best_epochs):
    all_metrics, by_regime, per_date, decomposition = {}, [], [], []
    count = len(models[REFERENCE])
    for name in MODELS:
        all_metrics[name] = {}
        total = aggregate(models[name], threshold)["SSE"]
        for group in ("all", "normal", "high"):
            selected = [r for r in models[name] if group == "all" or regime(r["target_sigma7"], threshold) == group]
            metrics = aggregate(selected, threshold)
            metrics["SSE_share_within_model"] = metrics["SSE"] / total
            all_metrics[name][group] = metrics
            by_regime.append({"model": name, "regime": group, **metrics})
        summary_metrics = docs["summary"]["metrics"][name]
        for key in ("RMSE", "MAE", "bias", "below_floor_count", "NLL_U", "nominal_90pct_coverage"):
            actual, expected = all_metrics[name]["all"][key], summary_metrics[key]
            require(actual is None and expected is None or actual is not None and expected is not None,
                    "Metric optionality mismatch")
            if actual is not None:
                close(actual, expected, name + " summary " + key)
        for suffix, key in (("RMSE", "RMSE"), ("MAE", "MAE"), ("count", "n")):
            close(all_metrics[name]["high"][key], summary_metrics["high_volatility_" + suffix], "High regime summary")
        close(threshold, summary_metrics["high_volatility_threshold_fit_q75"], "Recorded fit q75")
        for row, ref, control in zip(models[name], models[REFERENCE], models[CASE_REFERENCE]):
            target, prediction, floor = (row[k] for k in ("target_sigma7", "prediction_sigma7", "known_floor"))
            error = prediction - target
            realized, predicted = target - floor, prediction - floor
            require(abs(error - (predicted - realized)) <= ULPS * max(math.ulp(error), math.ulp(predicted), math.ulp(realized)),
                    "Known-floor/excess error identity failed")
            per_date.append({"model": name, "date": row["date"], "realized_target_regime": regime(target, threshold),
                "threshold_fit_q75": threshold, "target_sigma7": target, "prediction_sigma7": prediction,
                "known_floor": floor, "realized_excess": realized, "predicted_excess": predicted,
                "signed_error": error, "absolute_error": abs(error), "squared_error": error * error,
                "underestimated": int(error < 0), "share_of_model_SSE": error * error / total,
                "squared_error_difference_vs_M00": error * error - (ref["prediction_sigma7"] - target) ** 2,
                "squared_error_difference_vs_additive": error * error - (control["prediction_sigma7"] - target) ** 2,
                "NLL_U": row["NLL_U"], "nominal_05": row["nominal_05"], "nominal_95": row["nominal_95"],
                "nominal_covered": int(row["nominal_05"] <= target <= row["nominal_95"]) if row["NLL_U"] is not None else None})
    for name in MODELS:
        for group in ("all", "normal", "high"):
            own, baseline = all_metrics[name][group], all_metrics[REFERENCE][group]
            diff = own["SSE"] - baseline["SSE"]
            decomposition.append({"model": name, "reference": REFERENCE, "regime": group, "n": own["n"],
                "model_SSE": own["SSE"], "reference_SSE": baseline["SSE"],
                "SSE_model_minus_reference": diff, "contribution_to_global_MSE_difference": diff / count,
                "relative_RMSE_improvement_vs_reference": (baseline["RMSE"] - own["RMSE"]) / baseline["RMSE"]})
        own, baseline = all_metrics[name], all_metrics[REFERENCE]
        split = sum(own[g]["SSE"] - baseline[g]["SSE"] for g in ("normal", "high"))
        whole = own["all"]["SSE"] - baseline["all"]["SSE"]
        require(abs(split - whole) <= ULPS * max(math.ulp(own["all"]["SSE"]), math.ulp(baseline["all"]["SSE"])),
                "Exact regime SSE decomposition failed")
    best_control = min(MODELS[:5], key=lambda n: all_metrics[n]["all"]["RMSE"])
    gain = (all_metrics[best_control]["all"]["RMSE"] - all_metrics[CANDIDATE]["all"]["RMSE"]) / all_metrics[best_control]["all"]["RMSE"]
    gate = gain >= docs["plan"]["decision"]["minimum_relative_rmse_improvement_for_continuation"]
    summary = docs["summary"]
    require(best_control == summary["best_fixed_control"] == summary["decision"]["best_control"], "Best fixed control mismatch")
    close(gain, summary["decision"]["improvement_fraction"], "Exploratory gate effect")
    require(gate == summary["decision"]["continuation_gate_passed"] == summary["exploratory_continuation_gate_passed"]
            and summary["decision"]["complete"] is True, "Exploratory gate mismatch")
    case_reference_rows = sorted([r for r in per_date if r["model"] == CASE_REFERENCE],
                                 key=lambda r: (-r["absolute_error"], r["date"]))[:10]
    cases = [{"case_rank": rank, "selection_reference": CASE_REFERENCE,
              "selection_status": "posthoc_diagnostic_not_exclusion_not_confirmation", **r}
             for rank, selected in enumerate(case_reference_rows, 1)
             for r in per_date if r["date"] == selected["date"]]
    return all_metrics, by_regime, per_date, decomposition, cases, best_control, gain, gate


def figure_svg(models, threshold):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e3_innovation_audit_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    colors = ("#777777", "#b38a19", "#8756a5", "#16834a", "#c36b18", "#bd3345")
    labels = ("Piso conocido", "Piso + incremento medio", "M00 conservado y proyectado", "Cabeza aditiva (control fijo)",
              "Innovación: sólo media", "Innovación: densidad + media")
    with matplotlib.rc_context({"svg.hashsalt": "e3-closed-v2-audit", "font.family": "DejaVu Sans",
                                "font.size": 9, "axes.grid": True, "grid.alpha": .18}):
        fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), layout="constrained")
        for name, color, label in zip(MODELS, colors, labels):
            rows = models[name]
            errors = [r["prediction_sigma7"] - r["target_sigma7"] for r in rows]
            axes[0].scatter([r["target_sigma7"] for r in rows], errors, s=12, alpha=.55,
                            color=color, label=label, linewidths=0)
            if name != REFERENCE:
                increments = [e * e - (r["prediction_sigma7"] - r["target_sigma7"]) ** 2
                              for e, r in zip(errors, models[REFERENCE])]
                accumulated = [math.fsum(increments[:i]) for i in range(1, len(increments) + 1)]
                axes[1].plot([datetime.fromisoformat(r["date"]) for r in rows], accumulated,
                             color=color, label=label, linewidth=1.5)
        axes[0].axhline(0, color="#222222", linewidth=.8)
        axes[0].axvline(threshold, color="#222222", linestyle=":", linewidth=1)
        axes[0].set(title="Errores firmados y umbral q75 fijado en ajuste", xlabel="σ₇ realizada (retornos logarítmicos diarios)",
                    ylabel="Predicción − objetivo (unidades de σ₇)")
        axes[0].legend(loc="lower left", ncol=2, fontsize=8)
        axes[1].axhline(0, color="#222222", linewidth=.8, label="M00 conservado y proyectado (referencia cero)")
        axes[1].set(title="Diferencia acumulada de SSE frente al M00 conservado y proyectado",
                    xlabel="Fecha de validación de desarrollo", ylabel="Diferencia acumulada de SSE (σ₇²)")
        axes[1].xaxis.set_major_locator(mdates.MonthLocator())
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axes[1].legend(loc="lower left", ncol=2, fontsize=8)
        fig.suptitle("Piloto v2 cerrado: diagnóstico descriptivo en las mismas 116 fechas", fontsize=12)
        fig.text(.5, -.012, "Alta: objetivo realizado ≥ q75 de ajuste; grupo no observable al predecir. Una diferencia de SSE negativa favorece al modelo.",
                 ha="center", fontsize=8)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight", metadata={"Date": None, "Creator": "e3_audit_innovation.py"})
        plt.close(fig)
    return buffer.getvalue(), matplotlib.__version__


def build_artifacts(input_dir=DEFAULT_AUDIT / "inputs", source_root=ROOT, provenance_path=None):
    snapshot, proof, payloads, docs = load_inputs(input_dir, provenance_path)
    best_epochs = validate_metadata(docs, snapshot, payloads, source_root)
    threshold = docs["data_admission"]["scaler"]["fit_target_q75"]
    models = align_predictions(payloads, docs["data_admission"]["validation_dates"])
    require(len(models[REFERENCE]) == 116, "All 116 development dates must be retained")
    metrics, by_regime, per_date, decomposition, cases, best_control, gain, gate = analyze(models, threshold, docs, best_epochs)
    require(metrics[CANDIDATE]["normal"]["n"] == 98 and metrics[CANDIDATE]["high"]["n"] == 18,
            "Pinned realized-target regime counts changed")
    svg, matplotlib_version = figure_svg(models, threshold)
    result = {"version": "e3-closed-innovation-audit-v1", "results_commit": COMMIT,
        "snapshot_manifest_sha256": SNAPSHOT_SHA256, "source_provenance_sha256": sha(canonical(proof)),
        "text_files_verified": len(payloads) + 1, "scientific_source_hashes_verified": docs["manifest"]["source_hashes"],
        "model_order": list(MODELS), "models": metrics, "best_epochs": best_epochs,
        "global_RMSE_ranking": sorted(MODELS, key=lambda n: metrics[n]["all"]["RMSE"]),
        "regime_definition": {"threshold": threshold, "origin": "inner-fit target q75 fixed before the pilot",
            "high": "realized_target_sigma7 >= threshold", "normal": "realized_target_sigma7 < threshold",
            "observable_at_inference": False, "counts": {"normal": 98, "high": 18}},
        "error_definition": "prediction_sigma7 - target_sigma7 = predicted_excess - realized_excess; excess = sigma7 - known_floor",
        "SSE_difference_sign": "model minus frozen_M00_projected; negative favors model",
        "MSE_identity": "MSE = bias^2 + population residual variance (ddof=0)",
        "case_selection": {"reference": CASE_REFERENCE, "rule": "10 largest absolute errors, descending; date ascending breaks ties",
            "status": "posthoc diagnostic; no dates excluded; all 696 date-model predictions retained",
            "dates": [r["date"] for r in cases if r["model"] == CASE_REFERENCE]},
        "decision": {"best_fixed_control": best_control, "candidate": CANDIDATE,
            "relative_RMSE_improvement": gain, "required_improvement": .02,
            "continuation_gate_passed": gate, "comparison_complete": True,
            "selection_metric_changed": False, "interpretation": "exploratory heuristic only; no winner or R3 claim"},
        "budget": {k: docs["summary"][k] for k in ("charged_seconds", "prior_charged_seconds", "global_charged_seconds",
                                                     "limit_seconds", "global_limit_seconds")},
        "limitations": ["one seed", "validation reused for early stopping and comparison", "overlapping seven-day targets are dependent",
            "R1 novelty unverified", "R3 statistical superiority not established", "nominal intervals are not certified calibration",
            "no irreducible-error claim from the algebraic floor", "posthoc error cases cannot justify changing the frozen continuation decision"],
        "operations": {"NPZ_opened": False, "checkpoints_opened": False, "new_fits": 0,
            "new_market_inferences": 0, "calibration_or_evaluation_accessed": False,
            "statistical_tests_or_bootstrap": False},
        "reproduction": {"float_comparison_ulps": ULPS, "matplotlib": matplotlib_version,
                         "SVG_comparison": "exact bytes; fixed hashsalt and no date metadata"}}
    return {"metrics.json": json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n",
        "metrics_by_regime.csv": _csv_bytes(by_regime), "per_date_errors.csv": _csv_bytes(per_date),
        "error_cases.csv": _csv_bytes(cases), "sse_decomposition.csv": _csv_bytes(decomposition), "regime_errors.svg": svg}


def _compare_json(actual, expected, context):
    if type(expected) is float:
        close(actual, expected, context)
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), context + ": JSON keys changed")
        for key in expected:
            _compare_json(actual[key], expected[key], context + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), context + ": JSON list changed")
        for index, value in enumerate(expected):
            _compare_json(actual[index], value, context + f"[{index}]")
    else:
        require(type(actual) is type(expected) and actual == expected, context + ": metadata changed")


def check_artifacts(directory, artifacts):
    """Compare all metadata exactly and numbers within eight representable steps."""
    for name, expected in artifacts.items():
        path = Path(directory) / name
        require(path.is_file(), "Audit output is absent: " + name)
        actual = path.read_bytes()
        if name.endswith(".json"):
            _compare_json(json.loads(actual), json.loads(expected), name)
        elif name.endswith(".csv"):
            a, e = _csv_rows(actual), _csv_rows(expected)
            require(len(a) == len(e), name + ": CSV row count changed")
            for i, (got, wanted) in enumerate(zip(a, e)):
                require(got.keys() == wanted.keys(), name + ": CSV columns changed")
                for key, value in wanted.items():
                    if got[key] == value:
                        continue
                    try:
                        expected_number, actual_number = float(value), float(got[key])
                    except ValueError:
                        raise ValueError(f"{name}[{i}].{key}: metadata changed") from None
                    close(actual_number, expected_number, f"{name}[{i}].{key}")
        else:
            require(actual == expected, name + ": deterministic figure bytes changed")


def check_book_exports(directory, audited_directory):
    """Keep the published downloads identical to the verified master outputs."""
    directory, audited_directory = Path(directory), Path(audited_directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    names = {"metrics.json", "metrics_by_regime.csv", "per_date_errors.csv",
             "error_cases.csv", "sse_decomposition.csv", "regime_errors.svg"}
    require(manifest["version"] == "e3-v2-closure-downloads-v1"
            and manifest["results_commit"] == COMMIT and set(manifest["files"]) == names,
            "Book export inventory/source changed")
    for name in sorted(names):
        path = directory / name
        require(path.is_file() and not path.is_symlink(), "Book export missing or symlinked: " + name)
        payload = path.read_bytes()
        require(payload == (audited_directory / name).read_bytes(), "Book export differs from audited master: " + name)
        require(manifest["files"][name] == {
            "bytes": len(payload), "sha256": sha(payload),
            "source": "experiments/e3_cierre_v2_20261116/audit/" + name},
            "Book export receipt changed: " + name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--check", action="store_true", help="Regenerate in memory and compare; do not overwrite artifacts")
    args = parser.parse_args(argv)
    try:
        artifacts = build_artifacts(args.audit_dir / "inputs", args.source_root)
        if args.check:
            check_artifacts(args.audit_dir, artifacts)
            check_book_exports(args.source_root / "book_e3/_static/e3_cierre_v2_20261116", args.audit_dir)
        else:
            for name, value in artifacts.items():
                (args.audit_dir / name).write_bytes(value)
        print(json.dumps({"status": "verified" if args.check else "generated", "artifacts": list(artifacts),
                          "new_fits": 0, "new_market_inferences": 0}))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"V2 audit failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
