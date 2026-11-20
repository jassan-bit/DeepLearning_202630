"""Compare a human-produced CSV against fixed A/B profiles; no model execution.

Pure comparison routines are adapted from the project's frozen scientific
replay client, source73d8958c50a69fbea46deb828f03f89385318789. New wrapper adds
fixed profiles, exact calendar/order checks and immutable-reference hashes.
Prepared with declared AI assistance; never asserts a human signature.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone, date
import hashlib
import io
import json
import math
from pathlib import Path
ABS_TOL=1e-14
REL_TOL=1e-12

def require(condition, message):
    if not condition:
        raise ValueError(message)


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
    if field == 'known_A_sigma7_squared':
        return 'sigma7_squared'
    if field.endswith('_sigma7'):
        return 'daily_population_sigma7_ddof0'
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
        if unit in {'count','seed_identifier'}:
            require(float(left).is_integer() and float(right).is_integer(), 'Noninteger count or seed')
        delta = abs(float(left) - float(right))
        record = records.setdefault((field, unit), {"field": field, "unit": unit, "numeric_pairs": 0,
            "maximum_absolute_difference": 0.0, "location_of_maximum": None, "tolerance_violations": 0})
        record["numeric_pairs"] += 1
        if delta > record["maximum_absolute_difference"]:
            record["maximum_absolute_difference"] = delta
            record["location_of_maximum"] = {"row_key": list(key), "leaf": leaf}
        field_abs,field_rel=abs_tol,rel_tol
        if unit in {'count','seed_identifier'} or field in {'frozen_GPU_sigma7','accepted_ORT_sigma7'}:
            field_abs,field_rel=0.0,0.0
        elif field in {'known_A_sigma7_squared','structural_floor_sigma7'}:
            field_abs,field_rel=1e-14,1e-12
        if delta > field_abs + field_rel * abs(float(left)):
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


def run(profile_id,reference,observed,profiles_path=None):
    profiles_path=profiles_path or Path(__file__).with_name('comparison_profiles.json')
    profile=json.loads(Path(profiles_path).read_bytes())['profiles'][profile_id]
    require(reference.is_file() and not reference.is_symlink(),'Regular frozen reference required')
    require(observed.is_file() and not observed.is_symlink(),'Regular observed CSV required')
    require(reference.stat().st_size<20_000_000 and observed.stat().st_size<20_000_000,'CSV exceeds comparison budget')
    ref=reference.read_bytes();got=observed.read_bytes()
    require(hashlib.sha256(ref).hexdigest()==profile['reference_sha256'],'Frozen reference SHA256 mismatch')
    fields,rows,order=read_csv(ref,profile['keys'])
    other_fields,other,other_order=read_csv(got,profile['keys'])
    require(fields==profile['columns'] and other_fields==fields,'Predeclared schema differs')
    require(len(rows)==profile['expected_rows'] and len(other)==len(rows),'Predeclared row count differs')
    require(order==other_order,'Identifier/calendar order differs')
    if profile['level']=='B':
        dates=[key[0] for key in order]
        require(dates[0]==profile['date_first'] and dates[-1]==profile['date_last'],'Frozen calendar endpoints differ')
        for day in dates:require(date.fromisoformat(day).isoformat()==day,'Noncanonical date')
    result=compare_csv(ref,got,profile['keys'],abs_tol=profile['absolute_tolerance'],rel_tol=profile['relative_tolerance'])
    if profile['level']=='B':
        for key in other_order:
            row=other[key]
            current=float(row['current_ORT_sigma7']);floor=float(row['structural_floor_sigma7'])
            require(current>=floor,'Observed prediction violates fixed structural floor')
            require(abs(floor*floor-float(row['known_A_sigma7_squared']))<=1e-14,'Floor and known A disagree')
            for base,error in [('frozen_GPU_sigma7','absolute_error_GPU_sigma7'),('accepted_ORT_sigma7','absolute_error_accepted_ORT_sigma7')]:
                require(abs(abs(current-float(row[base]))-float(row[error]))<=1e-14,'Derived discrepancy field disagrees with prediction')
    result.update(profile=profile_id,level=profile['level'],comparison_only=True,
                  new_model_fits=0,new_model_inferences=0,human_independent_execution_claimed=False,
                  R3_accredited=False,reference_source_path=profile['reference_source_path'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--reference',required=True,type=Path)
    parser.add_argument('--observed',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Choose a new output path; prior receipt retained')
    report={'started_at_utc':datetime.now(timezone.utc).isoformat(),'profile':args.profile,'status':'failed',
            'comparison_only':True,'new_model_fits':0,'new_model_inferences':0,
            'human_independent_execution_claimed':False,'AI_assistance_declared':True}
    try:report.update(run(args.profile,args.reference,args.observed))
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
        report['error']={'type':type(error).__name__,'message':str(error)}
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'profile':args.profile,'receipt':str(args.output)},ensure_ascii=False))
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())


