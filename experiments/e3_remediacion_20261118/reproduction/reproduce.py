"""Replay frozen E3 statistical products in a separate CPU environment.

This rebuilds aggregates from saved forecasts. It does not train, infer with a
model, recalibrate, certify novelty, or impersonate an independent human.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ABS_TOL = 1e-14
REL_TOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fetch_inputs(data):
    """Explicitly acquire eleven immutable public files, rejecting changed bytes."""
    manifest = json.loads((HERE / "input_manifest.json").read_bytes())
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    for name, record in manifest["files"].items():
        destination = data / name
        if not destination.exists():
            url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + record["commit"] + "/" + record["path"]
            with urlopen(Request(url, headers={"User-Agent": "E3ScientificReplay/1"}), timeout=60) as response:
                raw = response.read(record["bytes"] + 1)
            require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
                    "Downloaded immutable input differs: " + name)
            with destination.open("xb") as stream:
                stream.write(raw)
        require(destination.is_file() and not destination.is_symlink(), "Regular input required: " + name)
        require(destination.stat().st_size == record["bytes"] and sha(destination) == record["sha256"],
                "Frozen input differs: " + name)
    return {name: {"sha256": sha(data / name), "bytes": (data / name).stat().st_size}
            for name in manifest["files"]}


def read_csv(raw, keys):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    fields = reader.fieldnames
    require(fields and len(fields) == len(set(fields)), "Missing or duplicate CSV columns")
    require(keys == ["@row"] or all(k in fields for k in keys), "CSV key column is absent")
    rows, order = {}, []
    for index, row in enumerate(reader):
        require(None not in row and all(v is not None for v in row.values()), "CSV row has a different column count")
        key = (str(index),) if keys == ["@row"] else tuple(row[k] for k in keys)
        require(all(key) and key not in rows, "Empty or duplicate CSV row key")
        rows[key] = row
        order.append(key)
    require(rows, "CSV table is empty")
    return fields, rows, order


def typed_cell(value):
    if value == "":
        return "missing", None
    if value.startswith(("[", "{")):
        def reject_constant(constant):
            raise ValueError("Nonfinite JSON numeric cell: " + constant)
        return "structured", json.loads(value, parse_constant=reject_constant)
    try:
        number = float(value)
    except ValueError:
        return "text", value
    require(math.isfinite(number), "Nonfinite CSV numeric cell")
    return "number", number


def field_unit(field, row):
    if field == "statistic":
        return "sigma7_original" if row.get("statistic_name") == "observed_gain_minus_null_boundary" else "dimensionless_test_statistic"
    if field in {"block_days", "HAC_lags"}:
        return "days"
    if field in {"R", "train_n", "calendar_n", "seed_n", "degrees_of_freedom", "exceedances", "bootstrap_repetitions", "Cohen_dz_estimable_bootstrap_draws"}:
        return "count"
    if field in {"seed", "seed_ids"}:
        return "seed_identifier"
    if "percent" in field or field in {"sMAPE (%)", "Cobertura 90% (%)"}:
        return "percent"
    if "RMSE" in field or "MAE" in field or field in {"delta", "null_boundary", "standard_error", "gain_ci95", "bonferroni_lower_bound", "mean_interval_width90", "gain_ci95_temporal", "paired_gain_sample_SD", "DE entre semillas", "Anchura media"}:
        return "sigma7_original"
    if field.startswith(("p_", "paired_Cohen", "Cohen_", "Shapiro_")) or field in {"R2", "R2_mean", "R² medio", "monte_carlo_standard_error_approx"}:
        return "dimensionless"
    return "text_or_missing"


def compare_csv(reference_raw, observed_raw, keys, *, abs_tol=ABS_TOL, rel_tol=REL_TOL):
    """Align by explicit keys, preserving null/text semantics and units per field."""
    require(math.isfinite(abs_tol) and math.isfinite(rel_tol) and abs_tol >= 0 and rel_tol >= 0, "Invalid comparison tolerance")
    fields, expected, order = read_csv(reference_raw, keys)
    other_fields, observed, other_order = read_csv(observed_raw, keys)
    require(fields == other_fields, "CSV column schema changed")
    require(set(expected) == set(observed), "CSV row support or keys changed")
    records, failures = {}, []

    def numeric(left, right, field, key, leaf, unit):
        require(unit != "text_or_missing", "Numeric field has no declared unit: " + field)
        require(type(left) in (int, float) and type(right) in (int, float) and math.isfinite(left) and math.isfinite(right),
                "A numeric leaf became nonfinite or changed type")
        delta = abs(float(left) - float(right))
        record = records.setdefault((field, unit), {"field": field, "unit": unit, "numeric_pairs": 0,
            "maximum_absolute_difference": 0.0, "location_of_maximum": None, "tolerance_violations": 0})
        record["numeric_pairs"] += 1
        if delta > record["maximum_absolute_difference"]:
            record["maximum_absolute_difference"] = delta
            record["location_of_maximum"] = {"row_key": list(key), "leaf": leaf}
        if delta > abs_tol + rel_tol * abs(float(left)):
            record["tolerance_violations"] += 1
            failures.append({"row_key": list(key), "field": field, "leaf": leaf, "unit": unit, "reason": "numeric_tolerance"})

    def structured(left, right, field, key, leaf, unit):
        if type(left) in (int, float) and type(right) in (int, float):
            numeric(left, right, field, key, leaf, unit)
        elif isinstance(left, list) and isinstance(right, list):
            require(len(left) == len(right), "Structured CSV cell length changed")
            for i, (lval, rval) in enumerate(zip(left, right, strict=True)):
                structured(lval, rval, field, key, leaf + "[" + str(i) + "]", unit)
        elif isinstance(left, dict) and isinstance(right, dict):
            require(set(left) == set(right), "Structured CSV cell keys changed")
            for name in left:
                structured(left[name], right[name], field, key, leaf + "." + name, unit)
        else:
            require(type(left) is type(right) and left == right, "Structured CSV cell value changed")

    exact_cells = missing_cells = 0
    for key in order:
        for field in fields:
            left_kind, left = typed_cell(expected[key][field])
            right_kind, right = typed_cell(observed[key][field])
            require(left_kind == right_kind, "Missing/text/numeric CSV cell type changed: " + field)
            unit = field_unit(field, expected[key])
            if left_kind == "number":
                numeric(left, right, field, key, field, unit)
            elif left_kind == "structured":
                structured(left, right, field, key, field, unit)
            else:
                require(left == right, "CSV text or missing value changed: " + field)
                exact_cells += 1
                missing_cells += int(left_kind == "missing")
    return {"status": "passed" if not failures else "failed", "rows": len(expected), "keys": list(keys),
            "columns": fields, "same_row_order": order == other_order, "bytes_identical": reference_raw == observed_raw,
            "reference_sha256": hashlib.sha256(reference_raw).hexdigest(),
            "observed_sha256": hashlib.sha256(observed_raw).hexdigest(),
            "absolute_tolerance": abs_tol, "relative_tolerance": rel_tol,
            "missing_cells_preserved": missing_cells, "exact_text_or_missing_cells": exact_cells,
            "field_discrepancies": list(records.values()), "failures": failures}


def run_replay(data, output):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), "Output directory must be absent or empty; preserve previous receipts")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"schema_version": 1, "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "automated_CPU_replay_from_frozen_predictions_not_retraining_or_independent_human_reproduction",
        "reference_source_commit": "73d8958c50a69fbea46deb828f03f89385318789",
        "model_fits": 0, "model_inferences": 0, "new_calibrations": 0,
        "statistical_analysis_reexecuted": True, "human_independent_reproduction": False,
        "R1_R2_R3_certified": False, "tables": [], "errors": [],
        "tolerance_rule": "abs(observed-reference) <= 1e-14 + 1e-12*abs(reference); comparisons grouped by field and unit",
        "limits": ["The saved 143 forecast trajectories are inputs; their training and forward passes are not repeated.",
                   "This replays the posterior analysis on a known calendar; it is not a new confirmatory experiment.",
                   "No pooled maximum mixes distinct metrics, probabilities, counts, or units.",
                   "A human external reviewer must still independently run and sign their own reproduction receipt."]}
    try:
        manifest = json.loads((HERE / "reference_manifest.json").read_bytes())
        for rel, expected in manifest["code"].items():
            require(sha(ROOT / rel) == expected, "Scientific generator changed: " + rel)
        for table in manifest["tables"]:
            require(sha(ROOT / table["reference"]) == table["reference_sha256"], "Published reference changed: " + table["name"])
        inputs = json.loads((HERE / "input_manifest.json").read_bytes())
        report["verified_inputs"] = {}
        for name, record in inputs["files"].items():
            path = Path(data) / name
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                    "Frozen input differs: " + name)
            report["verified_inputs"][name] = {"sha256": sha(path), "bytes": path.stat().st_size}
        report["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
            "libraries": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
            "Torch_available": importlib.util.find_spec("torch") is not None,
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
        require(report["environment"]["libraries"] == {"numpy": "2.3.5", "scipy": "1.17.0", "matplotlib": "3.10.8"}, "Scientific library versions differ")
        require(not report["environment"]["Torch_available"], "This aggregate-replay profile must not include a training framework")
        report["code_sha256"] = manifest["code"]
        report["replay_script_sha256"] = sha(__file__)
        report["requirements_sha256"] = sha(ROOT / "deployment/science/requirements.lock.txt")
        commands = [
            [sys.executable, "experiments/e3_estadistica_posterior_20261117/run.py", "--data", str(Path(data).resolve()), "--output", str((output / "statistics").resolve())],
            [sys.executable, "experiments/e3_revision_bloque_d_20261117/metrics_supplement.py", "--data-dir", str(Path(data).resolve()), "--output", str((output / "supplement").resolve())],
        ]
        report["commands"] = commands
        for index, command in enumerate(commands):
            with (output / ("generator_" + str(index + 1) + ".log")).open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        for table in manifest["tables"]:
            try:
                result = compare_csv((ROOT / table["reference"]).read_bytes(), (output / table["observed"]).read_bytes(), table["keys"])
            except (ValueError, OSError) as error:
                result = {"status": "failed", "error": str(error)}
            report["tables"].append({"name": table["name"], **result})
        report["systems"] = 17
        report["saved_predictors"] = 143
        report["common_dates"] = 608
        report["status"] = "passed" if all(table["status"] == "passed" for table in report["tables"]) else "failed"
    except (ValueError, OSError, subprocess.SubprocessError, importlib.metadata.PackageNotFoundError) as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    report["elapsed_seconds"] = time.perf_counter() - started
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(output / "reproduction_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download and verify eleven immutable public inputs")
    fetch.add_argument("--data", type=Path, required=True)
    run = sub.add_parser("run", help="Rebuild tables in the pinned CPU environment")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        print(json.dumps({"verified_inputs": fetch_inputs(args.data), "model_fits": 0, "model_inferences": 0}))
        return 0
    result = run_replay(args.data, args.output)
    print(json.dumps({"status": result["status"], "tables": len(result["tables"]), "model_fits": 0,
                      "model_inferences": 0, "human_independent_reproduction": False,
                      "receipt": str(args.output / "reproduction_report.json")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
