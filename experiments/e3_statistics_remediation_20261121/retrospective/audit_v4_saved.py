"""Fixed retrospective V4 adequacy diagnostics. AI assistance declared.

Saved errors only: no forecaster imports, fitting, forward passes or bootstrap.
Auxiliary OLS regressions are explicitly counted and are not forecasting fits.
All inference is conditional on the saved trajectory, asymptotic and exploratory.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import scipy
from scipy.stats import chi2, norm

PLAN_SHA256 = "864f321f69783810ee306a29fe47588a6aadb7bb4dd7dc0e2cb2c1791796a80e"
KINDS = ("Ljung_Box_7", "Ljung_Box_14", "ARCH_LM_7", "Mincer_Zarnowitz_HAC_14")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                   ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    require(bool(rows), "No empty output table")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    require(p.ndim == 1 and len(p) > 0 and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(), "Invalid complete p family")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1., np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty(len(p)); out[order] = adjusted
    return out


def finite_vector(values, minimum=1):
    x = np.asarray(values, float)
    require(x.ndim == 1 and len(x) >= minimum and np.isfinite(x).all(), "Finite vector required")
    return x


def acf(errors, lag):
    e = finite_vector(errors, lag + 1)
    z = e - e.mean()
    scale = float(np.max(np.abs(z)))
    if scale == 0:
        return None
    z = z / scale
    return float(z[lag:] @ z[:-lag] / (z @ z))


def ljung_box(errors, lag):
    e = finite_vector(errors, lag + 2)
    r = [acf(e, k) for k in range(1, lag + 1)]
    if any(v is None for v in r):
        return {"status": "not_computed_constant_residual", "statistic": None, "p": None}
    n = len(e)
    q = n * (n + 2) * sum(v * v / (n - k) for k, v in enumerate(r, 1))
    return {"status": "computed_asymptotic_conditional", "statistic": float(q),
            "p": float(chi2.sf(q, lag))}


def arch_lm(errors, lag):
    e = finite_vector(errors, 2 * lag + 3)
    scale = float(np.max(np.abs(e)))
    if scale == 0:
        return {"status": "not_computed_zero_residual", "statistic": None, "p": None}
    squares = (e / scale) ** 2
    y = squares[lag:]
    X = np.column_stack([np.ones(len(y)), *[squares[lag-k:-k] for k in range(1, lag+1)]])
    if np.linalg.matrix_rank(X) != lag + 1 or float(np.var(y)) == 0:
        return {"status": "not_computed_degenerate_variance_design", "statistic": None, "p": None}
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ssr = float(np.sum((y - X @ beta) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    r2 = min(1., max(0., 1 - ssr / sst)); lm = len(y) * r2
    return {"status": "computed_asymptotic_conditional", "statistic": float(lm),
            "p": float(chi2.sf(lm, lag))}


def hac_meat(scores, lag):
    z = np.asarray(scores, float)
    require(z.ndim == 2 and len(z) > lag >= 0 and np.isfinite(z).all(), "Invalid HAC scores/lag")
    out = z.T @ z
    for k in range(1, lag + 1):
        term = z[k:].T @ z[:-k]
        out += (1 - k / (lag + 1)) * (term + term.T)
    return (out + out.T) / 2


def mz_hac(targets, predictions, lag):
    y = finite_vector(targets, lag + 4); p = finite_vector(predictions, lag + 4)
    require(y.shape == p.shape, "MZ target/prediction length mismatch")
    location = float(p.mean()); scale = float(np.std(p))
    if scale == 0:
        return {"status": "not_computed_constant_prediction", "statistic": None, "p": None}
    # Standardized coordinates protect rank from volatility-unit rescaling.
    X = np.column_stack([np.ones(len(y)), (p - location) / scale])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    if float(np.max(np.abs(residual))) <= 64 * np.finfo(float).eps * float(np.max(np.abs(y))):
        return {"status": "not_computed_numerically_exact_fit", "statistic": None, "p": None}
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ hac_meat(X * residual[:, None], lag) @ bread * len(y) / (len(y) - 2)
    eigenvalues = np.linalg.eigvalsh(cov)
    if not np.isfinite(cov).all() or eigenvalues[-1] <= 0 or eigenvalues[0] <= 64 * np.finfo(float).eps * eigenvalues[-1]:
        return {"status": "not_computed_degenerate_HAC_covariance", "statistic": None, "p": None}
    difference = beta - np.array([location, scale])
    wald = float(difference @ np.linalg.solve(cov, difference))
    require(math.isfinite(wald) and wald >= 0, "Invalid MZ Wald statistic")
    transform = np.array([[1., -location/scale], [0., 1/scale]])
    original_beta = transform @ beta
    original_cov = transform @ cov @ transform.T
    half = norm.ppf(.975) * np.sqrt(np.maximum(np.diag(original_cov), 0))
    return {"status": "computed_asymptotic_conditional", "statistic": wald,
            "p": float(chi2.sf(wald, 2)), "alpha": float(original_beta[0]),
            "beta": float(original_beta[1]), "alpha_pointwise95_low": float(original_beta[0]-half[0]),
            "alpha_pointwise95_high": float(original_beta[0]+half[0]),
            "beta_pointwise95_low": float(original_beta[1]-half[1]),
            "beta_pointwise95_high": float(original_beta[1]+half[1])}


def mean_error_interval(errors, lag):
    e = finite_vector(errors, lag + 2)
    center = float(e.mean()); z = (e-center)[:, None]
    variance = float(hac_meat(z, lag)[0, 0] / len(e) ** 2 * len(e)/(len(e)-1))
    tolerance = 64 * np.finfo(float).eps * max(float(np.sum(z*z)) / len(e)**2, np.finfo(float).tiny)
    require(variance >= -tolerance, "Negative mean-error HAC variance")
    if variance <= 0:
        return {"status": "not_computed_no_estimated_variability", "mean_error": center,
                "standard_error": None, "lower95": None, "upper95": None}
    se = math.sqrt(variance); half = float(norm.ppf(.975)) * se
    return {"status": "computed_pointwise_asymptotic_conditional", "mean_error": center,
            "standard_error": se, "lower95": center-half, "upper95": center+half}


def metrics(y, p):
    e = y - p
    return {"RMSE": float(np.sqrt(np.mean(e*e))), "MAE": float(np.mean(np.abs(e))),
            "mean_residual": float(e.mean())}


def verify_inputs(input_root, manifest):
    root = Path(input_root).resolve()
    for f in manifest["files"]:
        rel = Path(f["path"])
        require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe input manifest path")
        p = root/rel
        require(p.is_file() and not p.is_symlink(), "Input absent/symlink: " + str(rel))
        require(p.stat().st_size == f["bytes"] and sha(p) == f["sha256"], "Input changed: " + str(rel))
    return len(manifest["files"])


def load_trajectories(input_root, plan):
    rows = read_csv(Path(input_root)/"assessment_predictions.csv")
    scope = plan["scope"]
    require(len(rows) == scope["expected_predictions"], "Prediction row count changed")
    parts = {}
    for row in rows:
        require(set(row) == {"name", "seed", "date", "target", "prediction"}, "Prediction schema changed")
        key = row["name"], row["seed"]
        parts.setdefault(key, []).append(row)
    require(len(parts) == scope["expected_trajectories"], "Trajectory count changed")
    calendar = [str(dt.date.fromisoformat(scope["assessment_start"]) + dt.timedelta(days=i))
                for i in range(scope["daily_dates"])]
    require(calendar[-1] == scope["assessment_end"], "Plan calendar inconsistent")
    targets = None; result = {}
    for key in sorted(parts):
        part = sorted(parts[key], key=lambda row: row["date"])
        require([row["date"] for row in part] == calendar, "Misaligned/duplicate/missing calendar")
        y = np.array([float(row["target"]) for row in part]); p = np.array([float(row["prediction"]) for row in part])
        require(np.isfinite(y).all() and np.isfinite(p).all() and (y >= 0).all() and (p >= 0).all(), "Invalid nonnegative volatility point")
        if targets is None:
            targets = y.copy()
        else:
            require(np.array_equal(y, targets), "Targets differ across trajectories")
        result[key] = (calendar, y, p)
    names = {k[0] for k in result}
    require(names == set(scope["neural_recipes"] + scope["analytical_controls"]), "Recipe identity changed")
    for name in scope["neural_recipes"]:
        require({int(k[1]) for k in result if k[0] == name} == set(scope["neural_seeds"]), "Neural seeds changed")
    for name in scope["analytical_controls"]:
        require(sum(k[0] == name for k in result) == 1, "Analytic trajectory copied into false seed repetitions")
    return result


def describe(parts):
    trajectories=[]; points=[]; monthly=[]; diagnostic=[]; intervals=[]
    for (name, seed), (calendar, y, p) in sorted(parts.items()):
        e = y-p; centered = e-e.mean(); variance = float(np.mean(centered**2))
        skew = float(np.mean(centered**3)/variance**1.5) if variance else None
        kurtosis = float(np.mean(centered**4)/variance**2) if variance else None
        common={"dataset":"D00_CPU_exposed_development", "recipe":name,"seed":seed,
                "dates":len(e),"start":calendar[0],"end":calendar[-1]}
        trajectories.append({**common, **metrics(y,p), "sample_SD_residual":float(e.std(ddof=1)),
            "skewness":skew,"Pearson_kurtosis":kurtosis,
            "Jarque_Bera_shape_only":len(e)/6*(skew**2+(kurtosis-3)**2/4) if variance else None,
            **{"ACF_lag"+str(k):acf(e,k) for k in range(1,15)}})
        for date, value in zip(calendar,e):
            points.append({"recipe":name,"seed":seed,"date":date,"residual_target_minus_prediction":float(value),
                           "squared_residual":float(value*value),"absolute_error":float(abs(value))})
        for month in sorted({d[:7] for d in calendar}):
            indices=np.array([d.startswith(month) for d in calendar])
            selected=[d for d in calendar if d.startswith(month)]
            monthly.append({"recipe":name,"seed":seed,"calendar_month":month,"dates":int(indices.sum()),
                            "start":selected[0],"end":selected[-1],**metrics(y[indices],p[indices]),
                            "conditional_descriptive_not_independent_replicates":True})
        results = [ljung_box(e,7),ljung_box(e,14),arch_lm(e,7),mz_hac(y,p,14)]
        for kind, result in zip(KINDS, results):
            pvalue=result.pop("p")
            diagnostic.append({**common,"diagnostic":kind,**result,"raw_asymptotic_p":pvalue,
                "p_used_only_to_reserve_Holm_slot":pvalue if pvalue is not None else 1.,
                "marginal_calibration_certified":False,"R3_superiority_test":False,
                "adequacy_pass_from_non_rejection":False})
        for lag in (7,14):
            intervals.append({**common,"HAC_lag":lag,**mean_error_interval(e,lag),
                "interval_unit":"unannualized_sigma7","pointwise_not_simultaneous":True,
                "conditional_saved_weights_only":True,"calendar_uncertainty_asymptotic_only":True,
                "seed_population_uncertainty_included":False,"coverage_empirically_certified":False})
    require(len(diagnostic) == 88, "Prespecified diagnostic family must retain88slots")
    adjusted=holm([r["p_used_only_to_reserve_Holm_slot"] for r in diagnostic])
    for row,value in zip(diagnostic,adjusted):row["Holm88_asymptotic_diagnostic_p"]=float(value)
    return trajectories,points,monthly,diagnostic,intervals


def recipe_and_effects(parts, trajectories):
    records=[]; effects=[]
    names=sorted({k[0] for k in parts})
    for name in names:
        selected=[row for row in trajectories if row["recipe"]==name]
        arrays=[p[2] for k,p in sorted(parts.items()) if k[0]==name]
        y=next(p[1] for k,p in parts.items() if k[0]==name)
        records.append({"recipe":name,"trajectories":len(selected),"mean_individual_RMSE":statistics.mean(r["RMSE"] for r in selected),
            "sample_SD_individual_RMSE":statistics.stdev(r["RMSE"] for r in selected) if len(selected)>1 else None,
            "mean_individual_MAE":statistics.mean(r["MAE"] for r in selected),
            "RMSE_of_mean_prediction_different_secondary_estimand":metrics(y,np.mean(arrays,axis=0))["RMSE"],
            "independent_calendar_replications":1,"primary_population_CI_computed":False})
    candidate={r["seed"]:r for r in trajectories if r["recipe"]=="sigma_crps_v4"}
    for name in names:
        if name=="sigma_crps_v4":continue
        control={r["seed"]:r for r in trajectories if r["recipe"]==name}
        if len(control)==1:
            contrast=statistics.mean(r["RMSE"] for r in candidate.values())-next(iter(control.values()))["RMSE"]
            effects.append({"candidate":"sigma_crps_v4","control":name,"definition":"candidate mean individual RMSE minus one analytic-control RMSE",
                "paired_seed_repetitions":0,"mean_RMSE_difference":contrast,"sample_SD_RMSE_difference":None,
                "Cohen_dz_descriptive":None,"calendar_population_inference":False,"new_p_value":None})
        else:
            require(set(control)==set(candidate),"Paired effect seeds differ")
            d=[candidate[s]["RMSE"]-control[s]["RMSE"] for s in sorted(candidate)]
            mean=statistics.mean(d);sd=statistics.stdev(d)
            effects.append({"candidate":"sigma_crps_v4","control":name,"definition":"paired per-seed candidate RMSE minus control RMSE on the same exposed calendar",
                "paired_seed_repetitions":len(d),"mean_RMSE_difference":mean,"sample_SD_RMSE_difference":sd,
                "Cohen_dz_descriptive":mean/sd if sd else None,"minimum_RMSE_difference":min(d),"maximum_RMSE_difference":max(d),
                "calendar_population_inference":False,"new_p_value":None})
    return records,effects


def learning_curves(input_root, trajectories):
    records=[]; root=Path(input_root)
    for row in trajectories:
        if not row["recipe"] in ("sigma_crps_v4","u_crps_control","mse_only_control","additive_positive_head"):
            continue
        stem=root/"jobs"/("seed"+row["seed"])/row["recipe"]
        job=read_json(stem/"job.json");curve=read_json(stem/"supervised/curve.json")
        phase=job["phase"];best=int(phase["best_epoch"])
        require(job["status"]=="completed_validated_new_fit" and len(curve)==int(phase["epochs"]),"Unexpected preserved job state")
        chosen=[r for r in curve if int(r["epoch"])==best]
        require(len(chosen)==1,"Best epoch absent/duplicated")
        require(math.isclose(float(chosen[0]["validation_mse_scaled"]),float(phase["best_validation_mse_scaled"]),rel_tol=1e-12,abs_tol=1e-15),"Best validation curve disagrees with frozen receipt")
        selection=float(job["selection_RMSE"])
        records.append({"recipe":row["recipe"],"seed":row["seed"],"epochs":len(curve),"selected_epoch":best,
            "selected_last_epoch":best==len(curve),"stop_reason":phase["stop_reason"],
            "training_objective_first_epoch":float(curve[0]["train_loss"]),
            "training_objective_selected_epoch":float(chosen[0]["train_loss"]),
            "training_objective_final_epoch":float(curve[-1]["train_loss"]),
            "stop_validation_MSE_scaled_at_selected_epoch":float(chosen[0]["validation_mse_scaled"]),
            "stop_validation_RMSE_original_units_from_receipt":selection,
            "assessment_RMSE_original_units":row["RMSE"],
            "assessment_over_selected_stop_RMSE_descriptive":row["RMSE"]/selection,
            "training_objective_not_divided_by_validation_MSE":True,
            "composite_objective_comparable_to_validation_point_MSE":False,
            "saved_fit_predictions_available":False,"saved_stop_predictions_available":False,
            "formal_no_overfit_established":False,"prospective_gap_margin_available":False})
    require(len(records)==20,"Expected20preserved jobs")
    return records


def run(input_root, plan_path, anchor_path, out):
    here=Path(__file__).resolve().parent
    require(sha(plan_path)==PLAN_SHA256,"Fixed retrospective plan changed")
    plan=read_json(plan_path);anchor=read_json(anchor_path);freeze=read_json(here/"IMPLEMENTATION_FREEZE.json")
    require(anchor.get("status")=="retrospective_analysis_plan_and_code_published_before_execution"
            and anchor.get("root_verified_publication") is True
            and len(anchor.get("commit",""))==40
            and all(c in "0123456789abcdef" for c in anchor.get("commit",""))
            and anchor.get("commit") != "0"*40
            and anchor.get("plan_sha256")==PLAN_SHA256
            and anchor.get("implementation_freeze_sha256")==sha(here/"IMPLEMENTATION_FREEZE.json"),"Public analysis anchor absent or inconsistent")
    for f in freeze["files"]:
        require(sha(here/f["path"])==f["sha256"],"Frozen implementation changed: "+f["path"])
    require(sha(here/"input_manifest.json")==plan["input_manifest_sha256"],"Input manifest changed")
    manifest=read_json(here/"input_manifest.json");verified=verify_inputs(input_root,manifest)
    out=Path(out);require(not out.exists(),"New output directory required; no overwrite/rerun")
    out.mkdir(parents=True)
    start=time.monotonic();receipt={"status":"running","started_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan_sha256":PLAN_SHA256,"input_manifest_sha256":plan["input_manifest_sha256"],
        "implementation_freeze_sha256":sha(here/"IMPLEMENTATION_FREEZE.json"),"public_anchor_commit":anchor["commit"],
        "inputs_verified":verified,"new_forecasting_model_fits":0,"new_forecasting_model_forward_passes":0,
        "new_market_predictions":0,"new_bootstrap_draws":0,"new_auxiliary_OLS_regressions":0,
        "R3_accredited":False,"confirmation_enabled":False,"primary_cross_seed_R10_guard_relaxed":False,
        "conditional_saved_trajectory_diagnostics_only":True,"all_calendars_and_original_pilot_verdicts_previously_exposed":True,
        "human_execution_claimed":False,"AI_assistance_declared":True,
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}}
    write_json(out/"execution_receipt.json",receipt)
    try:
        parts=load_trajectories(input_root,plan)
        trajectories,points,monthly,diagnostics,intervals=describe(parts)
        recipes,effects=recipe_and_effects(parts,trajectories)
        curves=learning_curves(input_root,trajectories)
        outputs={"trajectory_descriptives.csv":trajectories,"residual_points.csv":points,"monthly_sensitivity.csv":monthly,
                 "diagnostics_Holm88.csv":diagnostics,"conditional_mean_error_CIs.csv":intervals,
                 "recipe_estimands.csv":recipes,"paired_seed_effects_descriptive.csv":effects,"learning_curve_inventory.csv":curves}
        for name,rows in outputs.items():write_csv(out/name,rows)
        valid=[r for r in diagnostics if r["raw_asymptotic_p"] is not None]
        summary={"scope":"new retrospective V4 CPU exposed development only; no primary superiority or adequate-model certificate",
            "source_commit":manifest["source_commit"],"trajectories":len(parts),"recipes":len(recipes),"prediction_points":len(points),
            "diagnostic_family_slots":88,"computed_diagnostic_p_values":len(valid),"unsupported_reserved_slots":88-len(valid),
            "Holm88_rejections_conditional_on_marginal_asymptotic_validity":sum(r["Holm88_asymptotic_diagnostic_p"]<.05 for r in valid),
            "by_diagnostic":[{"diagnostic":kind,"slots":22,"computed":sum(r["diagnostic"]==kind for r in valid),
                "Holm88_rejections_conditional_on_assumptions":sum(r["diagnostic"]==kind and r["Holm88_asymptotic_diagnostic_p"]<.05 for r in valid)} for kind in KINDS],
            "pointwise_asymptotic_mean_error_intervals":len(intervals),"simultaneous_coverage_claimed":False,
            "Cohen_dz_scope":"descriptive five paired seeds on one exposed calendar; no superiority p or population CI",
            "new_auxiliary_OLS_regressions":44,"new_forecaster_fits":0,"new_forecaster_forward_passes":0,
            "new_bootstrap_draws":0,"primary_cross_seed_inference_run":False,"primary_minimum_seeds_required":10,
            "neural_seed_count_observed":5,"formal_overfit_adequacy_pass":False,
            "V4_predictive_interval_coverage_or_PIT_available":False,"V4_calibrator_invented":False,
            "V3_original_NO_GO_preserved":True,"V4_original_abandon_decision_preserved":True,"R3_accredited":False}
        write_json(out/"analysis_summary.json",summary)
        require(time.monotonic()-start<=plan["runtime_budget_seconds"],"Analysis exceeded declared wall budget")
        receipt.update(status="completed_retrospective_adequacy_not_confirmation",completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-start,new_auxiliary_OLS_regressions=44,diagnostic_slots=88,
            computed_asymptotic_diagnostic_p_values=len(valid),unsupported_reserved_slots=88-len(valid),
            frozen_inputs_after_execution_unchanged=verify_inputs(input_root,manifest)==verified)
        write_json(out/"execution_receipt.json",receipt)
        members=[]
        for path in sorted(out.iterdir()):
            if path.is_file():members.append({"path":path.name,"bytes":path.stat().st_size,"sha256":sha(path)})
        write_json(out/"OUTPUT_MANIFEST.json",{"schema_version":1,"files":members,"all_outputs_fixed_after_one_execution":True})
        return summary
    except BaseException as exc:
        receipt.update(status="failed_evidence_retained",error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-start)
        write_json(out/"execution_receipt.json",receipt)
        raise


def verify_only(input_root, out):
    here=Path(__file__).resolve().parent;verified=verify_inputs(input_root,read_json(here/"input_manifest.json"))
    manifest=read_json(Path(out)/"OUTPUT_MANIFEST.json")
    for f in manifest["files"]:
        p=Path(out)/f["path"];require(p.is_file() and not p.is_symlink() and sha(p)==f["sha256"] and p.stat().st_size==f["bytes"],"Output changed")
    require(set(p.name for p in Path(out).iterdir() if p.is_file())=={f["path"] for f in manifest["files"]}|{"OUTPUT_MANIFEST.json"},"Unexpected output member")
    return {"status":"verified_frozen_retrospective_inputs_and_outputs","inputs":verified,"outputs":len(manifest["files"]),"reanalysis_performed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--plan",type=Path,default=Path(__file__).with_name("analysis_plan.json"))
    parser.add_argument("--anchor",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if args.verify_only:result=verify_only(args.input_root,args.out)
    else:
        require(args.anchor is not None,"Explicit immutable public anchor required")
        result=run(args.input_root,args.plan,args.anchor,args.out)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":main()
