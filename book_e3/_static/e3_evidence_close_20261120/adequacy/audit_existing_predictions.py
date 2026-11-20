"""Retrospective adequacy audit of saved metrics and predictions; AI assistance declared.

This module never imports model code, trains a forecaster, performs inference,
downloads data or resamples an existing analysis. Statistical auxiliary OLS
regressions use frozen historical point predictions only. They are diagnostics,
not tests of forecast superiority, prospective power or an R3 decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import gzip
import io
import json
import math
from pathlib import Path
import platform
import statistics
import tempfile

import numpy as np
import scipy
from scipy.stats import chi2

CANDIDATE = "CNN-LSTM-atencion-aditiva E3"
DATASETS = ("D00", "D01", "D02", "D03")
CONFIG = {"scope": "retrospective_documentary_and_diagnostic_only",
          "historical_residual_calendar": "2025-01-01..2026-08-31",
          "Ljung_Box_lags": [7, 14], "ARCH_LM_lag": 7,
          "Mincer_Zarnowitz_HAC_Bartlett_lag": 14,
          "Jarque_Bera": "shape statistic only; no IID p-value interpreted",
          "multiplicity": {"historical_510c3bdd":"Holm40 across10seeds4diagnostics",
                           "extended_796_candidate":"Holm160 across4datasets10seeds4diagnostics",
                           "global_Holm200_or_R3_claimed":False},
          "formal_adequacy_pass_fail": False, "R3_decision": False,
          "new_forecast_model_fits": 0, "new_forecast_model_inference": 0,
          "new_statistical_resamples": 0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    path=Path(path)
    opener=gzip.open if path.suffix==".gz" else open
    with opener(path,"rt",newline="",encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    assert rows
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")


def verify_archive(archive):
    manifest = json.loads((archive / "input_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert sha(archive / name) == expected, "Input hash mismatch: " + name
    raw=gzip.decompress((archive/"historical_e3_predictions.csv.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest()==manifest["provenance"]["historical_residual_predictions"]["sha256"]
    for record in manifest["provenance"].get("extended_candidate_residual_panels",{}).values():
        raw=gzip.decompress((archive/record["archive_file"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest()==record["extracted_uncompressed_sha256"]
    return manifest


def verify_original_source(archive,source):
    """Optional full check of12 already-published source tables and extracts."""
    manifest=verify_archive(archive)
    original=json.loads((source/"analysis/output_manifest.json").read_text())
    assert sha(source/"analysis/output_manifest.json")==manifest["provenance"]["extended_audit"]["original_analysis_output_sha256"]
    for name,record in original["files"].items():
        path=source/"analysis"/name
        assert sha(path)==record["sha256"] and path.stat().st_size==record["bytes"],name
    for name,record in manifest["extractions"].items():
        source_rows=[r for r in read_csv(source/"analysis"/record["original"])
                     if all(r[k]==v for k,v in record["filter"].items())]
        assert source_rows==read_csv(archive/name),"Source extraction differs: "+name
    print(json.dumps({"original_analysis_tables_verified":len(original["files"]),"candidate_extractions_verified":len(manifest["extractions"])}))


def gaps(metrics):
    rows = []
    for dataset in DATASETS:
        selected = [r for r in metrics if r["dataset"] == dataset and
                    r["scenario"] == "unperturbed" and r["model"] == CANDIDATE]
        by_split = {s: {r["seed"]: r for r in selected if r["split"] == s}
                    for s in ("fit", "validation", "calibration", "test")}
        seeds = sorted(by_split["fit"])
        assert len(seeds) == 10 and all(sorted(v) == seeds for v in by_split.values())
        for heldout in ("validation", "calibration", "test"):
            fit = [float(by_split["fit"][s]["RMSE"]) for s in seeds]
            out = [float(by_split[heldout][s]["RMSE"]) for s in seeds]
            ratios = [b / a for a, b in zip(fit, out)]
            paired = [b - a for a, b in zip(fit, out)]
            rows.append({"dataset": dataset, "comparison": heldout + "_versus_fit",
                         "seeds": len(seeds), "fit_dates": by_split["fit"][seeds[0]]["n"],
                         "heldout_dates": by_split[heldout][seeds[0]]["n"],
                         "mean_fit_RMSE": statistics.mean(fit), "mean_heldout_RMSE": statistics.mean(out),
                         "ratio_of_mean_RMSE": statistics.mean(out) / statistics.mean(fit),
                         "mean_seed_RMSE_ratio": statistics.mean(ratios),
                         "SD_seed_RMSE_ratio_fixed_calendar": statistics.stdev(ratios),
                         "min_seed_RMSE_ratio": min(ratios), "max_seed_RMSE_ratio": max(ratios),
                         "mean_paired_RMSE_gap": statistics.mean(paired),
                         "SD_paired_RMSE_gap_fixed_calendar": statistics.stdev(paired),
                         "mean_MSE_gap_sigma7_squared": statistics.mean(b*b-a*a for a,b in zip(fit,out)),
                         "prospective_margin_available": False, "formal_no_overfit_established": False})
    return rows


def baselines(metrics):
    rows = []
    controls = ("floor_only", "floor_plus_mean_excess", "Persistence", "SeasonalNaive", "ARIMA")
    for dataset in DATASETS:
        a = {r["model"]: r for r in metrics if r["dataset"] == dataset and
             r["scenario"] == "unperturbed" and r["split"] == "test"}
        assert len(a) == 19
        candidate = a[CANDIDATE]
        for control in controls:
            rival = a[control]
            rmse, rrmse = float(candidate["RMSE_mean_individual"]), float(rival["RMSE_mean_individual"])
            mae, rmae = float(candidate["MAE_mean_individual"]), float(rival["MAE_mean_individual"])
            rows.append({"dataset": dataset, "control": control,
                         "candidate_mean_individual_RMSE": rmse, "control_RMSE": rrmse,
                         "candidate_RMSE_gain_percent": 100*(1-rmse/rrmse),
                         "candidate_mean_individual_MAE": mae, "control_MAE": rmae,
                         "candidate_MAE_gain_percent": 100*(1-mae/rmae),
                         "calendar_dates": candidate["dates"], "new_p_value": None,
                         "superiority_or_R3_claimed": False})
    return rows


def coverage(intervals, quarters):
    rows = []
    for dataset in DATASETS:
        overall = [r for r in intervals if r["dataset"] == dataset]
        assert len(overall) == 10 and all(r["scenario"] == "unperturbed" for r in overall)
        subgroup = [r for r in quarters if r["dataset"] == dataset]
        names = sorted({r["subgroup"] for r in subgroup})
        means = {name: statistics.mean(float(r["coverage"]) for r in subgroup if r["subgroup"] == name)
                 for name in names}
        worst = min(means, key=means.get)
        rows.append({"dataset": dataset, "seeds": 10, "test_dates": overall[0]["n"],
                     "nominal_coverage": .9,
                     "mean_seed_coverage": statistics.mean(float(r["coverage"]) for r in overall),
                     "min_seed_coverage": min(float(r["coverage"]) for r in overall),
                     "max_seed_coverage": max(float(r["coverage"]) for r in overall),
                     "calendar_subgroups": len(names), "lowest_mean_coverage_subgroup": worst,
                     "lowest_quarter_mean_seed_coverage": means[worst],
                     "small_subgroup_rows_n_under50": sum(r["small_subgroup"] == "True" for r in subgroup),
                     "PIT_available_from_point_plus_interval": False,
                     "prospective_coverage_guarantee": False,
                     "subgroup_selection": "retrospective minimum across all saved calendar quarters"})
    return rows


def resource_inventory(resources):
    rows = []
    fields = ("parameters", "seconds", "optimizer_steps", "peak_cuda_bytes", "allocated_seconds",
              "deadline_overrun_seconds", "stop_reason", "unclosed_crash_compute_unknown", "FLOPs", "energy_joules")
    for scope, records in (("all_supplied_phase_measurements", resources),
                           ("candidate_supplied_phase_measurements", [r for r in resources if r["model"] == CANDIDATE])):
        for field in fields:
            present = [r[field] for r in records if field in r and r[field] != ""]
            rows.append({"scope": scope, "field": field, "measurement_rows": len(records),
                         "supplied_values": len(present), "unsupplied_values": len(records)-len(present),
                         "zero_values_are_not_missing": True,
                         "semantics": "phase rows are not job counts; blank can be unmeasured or inapplicable"})
    return rows


def autocorrelation(e, lag):
    centered = e - e.mean()
    denom = float(centered @ centered)
    return float(centered[lag:] @ centered[:-lag] / denom) if denom else 0.


def mz_hac(y, forecast, lag):
    X = np.column_stack([np.ones(len(y)), forecast])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y-X@beta
    z = X * resid[:, None]
    meat = z.T@z
    for k in range(1, lag+1):
        term = z[k:].T@z[:-k]
        meat += (1-k/(lag+1))*(term+term.T)
    bread = np.linalg.inv(X.T@X)
    covariance = bread@meat@bread * len(y)/(len(y)-2)
    difference = beta-np.array([0.,1.])
    statistic = float(difference @ np.linalg.pinv(covariance) @ difference)
    assert statistic >= 0 and np.isfinite(covariance).all()
    return float(beta[0]), float(beta[1]), statistic, float(chi2.sf(statistic,2))


def arch_lm(e, lag):
    squared = e**2
    y = squared[lag:]
    X = np.column_stack([np.ones(len(y)), *[squared[lag-k:-k] for k in range(1,lag+1)]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    ssr = float(np.sum((y-X@beta)**2)); sst=float(np.sum((y-y.mean())**2))
    r2=max(0.,1-ssr/sst) if sst else 0.
    statistic=len(y)*r2
    return statistic,float(chi2.sf(statistic,lag))


def adjust_holm(rows,key):
    order=sorted(range(len(rows)),key=lambda i:rows[i]["raw_asymptotic_p"])
    previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(rows)-rank)*rows[index]["raw_asymptotic_p"]))
        rows[index][key]=previous
    return rows


def residual_diagnostics(predictions,*,scope="historical_E3_510c3bdd_only_not_796_audit_or_V3",
                         dataset="D00_legacy",expected_dates=608,expected_start="2025-01-01",
                         expected_end="2026-08-31",adjusted_key="Holm40_asymptotic_diagnostic_p"):
    seeds=sorted({r["seed"] for r in predictions})
    assert len(seeds)==10 and len(predictions)==10*expected_dates
    rows=[];calendar=None;targets=None
    for seed in seeds:
        part=sorted([r for r in predictions if r["seed"]==seed],key=lambda r:r["date"])
        assert len(part)==expected_dates and len({r["date"] for r in part})==expected_dates
        if expected_start is not None:assert part[0]["date"]==expected_start
        if expected_end is not None:assert part[-1]["date"]==expected_end
        y=np.array([float(r["target_sigma7"]) for r in part]); p=np.array([float(r["prediction_sigma7"]) for r in part]); e=y-p
        assert np.isfinite(y).all() and np.isfinite(p).all()
        if calendar is None:calendar=[r["date"] for r in part];targets=y.copy()
        else:assert calendar==[r["date"] for r in part] and np.array_equal(targets,y)
        centered=e-e.mean(); variance=float(np.mean(centered**2))
        skew=float(np.mean(centered**3)/variance**1.5); kurt=float(np.mean(centered**4)/variance**2)
        jb=len(e)/6*(skew**2+(kurt-3)**2/4)
        common={"scope":scope,"dataset":dataset, "seed":seed,"dates":len(e),
                "start":part[0]["date"],"end":part[-1]["date"], "RMSE":float(np.sqrt(np.mean(e**2))),
                "bias_target_minus_prediction":float(e.mean()), "skewness_population_moment":skew,
                "kurtosis_Pearson_population_moment":kurt,"Jarque_Bera_shape_statistic":jb,
                "Jarque_Bera_p_not_interpreted_under_serial_dependence":True,
                "autocorrelation_lag1":autocorrelation(e,1),"autocorrelation_lag7":autocorrelation(e,7)}
        for lag in CONFIG["Ljung_Box_lags"]:
            q=len(e)*(len(e)+2)*sum(autocorrelation(e,k)**2/(len(e)-k) for k in range(1,lag+1))
            rows.append({**common,"diagnostic":"Ljung_Box","lag":lag,"statistic":q,"df":lag,
                         "raw_asymptotic_p":float(chi2.sf(q,lag)),"MZ_alpha":None,"MZ_beta":None})
        a,b,w,pvalue=mz_hac(y,p,CONFIG["Mincer_Zarnowitz_HAC_Bartlett_lag"])
        rows.append({**common,"diagnostic":"Mincer_Zarnowitz_joint_HAC","lag":14,"statistic":w,"df":2,
                     "raw_asymptotic_p":pvalue,"MZ_alpha":a,"MZ_beta":b})
        lm,pvalue=arch_lm(e,CONFIG["ARCH_LM_lag"])
        rows.append({**common,"diagnostic":"ARCH_LM","lag":7,"statistic":lm,"df":7,
                     "raw_asymptotic_p":pvalue,"MZ_alpha":None,"MZ_beta":None})
    assert len(rows)==40
    if adjusted_key is not None:adjust_holm(rows,adjusted_key)
    for row in rows:
        row["formal_adequacy_pass_fail"]=None
        row["R3_superiority_test"]=False
    return rows


def extended_residuals(archive,manifest,seeds):
    rows=[];consistency=[]
    sources=manifest["provenance"]["extended_candidate_residual_panels"]
    assert sorted(sources)==list(DATASETS)
    for dataset in DATASETS:
        source=sources[dataset]
        predictions=read_csv(archive/source["archive_file"])
        assert len(predictions)==source["extracted_rows"]
        published=[r for r in seeds if r["dataset"]==dataset and r["split"]=="test"]
        assert len(published)==10
        for record in published:
            part=[r for r in predictions if r["seed"]==record["seed"]]
            assert len(part)==int(record["n"])
            errors=np.array([float(r["target_sigma7"])-float(r["prediction_sigma7"]) for r in part])
            derived={"RMSE":float(np.sqrt(np.mean(errors**2))),"MAE":float(np.mean(np.abs(errors)))}
            difference=max(abs(derived[k]-float(record[k])) for k in derived)
            assert difference<1e-15
            consistency.append({"dataset":dataset,"seed":record["seed"],"dates":len(part),
                                "max_absolute_difference_saved_RMSE_MAE":difference})
        rows.extend(residual_diagnostics(predictions,scope="extended_796_audit_clean_test_candidate_not_V3",
                    dataset=dataset,expected_dates=int(published[0]["n"]),expected_start=None,expected_end=None,adjusted_key=None))
    assert len(rows)==160 and len(consistency)==40
    adjust_holm(rows,"Holm160_asymptotic_diagnostic_p")
    return rows,consistency


def main(archive, output):
    archive,output=Path(archive),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest=verify_archive(archive)
    metrics=read_csv(archive/"model_metrics.csv")
    seeds=read_csv(archive/"per_seed_candidate_clean.csv")
    interval=read_csv(archive/"interval_candidate_clean.csv")
    quarter=read_csv(archive/"quarter_candidate_clean.csv")
    resources=read_csv(archive/"resources_supplied.csv")
    historic=read_csv(archive/"historical_e3_predictions.csv.gz")
    extended,extended_checks=extended_residuals(archive,manifest,seeds)
    products={"descriptive_fit_heldout_gaps.csv":gaps(seeds),"trivial_baseline_comparison.csv":baselines(metrics),
              "coverage_drift_readout.csv":coverage(interval,quarter),"resource_field_inventory.csv":resource_inventory(resources),
              "historical_residual_diagnostics.csv":residual_diagnostics(historic),
              "extended_candidate_residual_diagnostics.csv":extended}
    # Independent consistency check: recompute published candidate means from
    # per-seed saved metrics without creating or averaging any prediction path.
    consistency=[]
    for dataset in DATASETS:
        for split in ("fit","validation","calibration","test"):
            rows=[r for r in seeds if r["dataset"]==dataset and r["split"]==split]
            published=next(r for r in metrics if r["dataset"]==dataset and r["split"]==split and
                           r["scenario"]=="unperturbed" and r["model"]==CANDIDATE)
            errors={k:abs(statistics.mean(float(r[k]) for r in rows)-float(published[k+"_mean_individual"]))
                    for k in ("RMSE","MAE")}
            assert max(errors.values())<1e-15
            consistency.append({"dataset":dataset,"split":split,"max_absolute_difference":max(errors.values())})
    for name,rows in products.items():write_csv(output/name,rows)
    unknown=[{key:r[key] for key in ("dataset","model","seed","kind","phase","seconds","deadline_overrun_seconds")}
             for r in resources if r["unclosed_crash_compute_unknown"]=="True"]
    resource_accounting={"phase_rows":len(resources),
                         "unique_jobs_by_dataset_model_seed_kind":len({tuple(r[k] for k in ("dataset","model","seed","kind")) for r in resources}),
                         "unclosed_crash_compute_unknown_rows":unknown,
                         "maximum_deadline_overrun_seconds_including_downtime":max(float(r["deadline_overrun_seconds"] or 0) for r in resources),
                         "maximum_overrun_is_not_GPU_active_compute":True,
                         "actual_equal_total_compute_certified":False,
                         "FLOPs_or_energy_reconstructed_from_parameters":False}
    assert resource_accounting["phase_rows"]==900 and resource_accounting["unique_jobs_by_dataset_model_seed_kind"]==796
    receipt={"status":"completed_retrospective_audit_only","configuration":CONFIG,
             "input_manifest_sha256":sha(archive/"input_manifest.json"),"source_provenance":manifest["provenance"],
             "inputs_verified":len(manifest["files_sha256"]),"published_mean_consistency_checks":consistency,
             "extended_point_prediction_consistency_checks":extended_checks,
             "output_rows":{name:len(rows) for name,rows in products.items()},
             "output_sha256":{name:sha(output/name) for name in products},"script_sha256":sha(__file__),
             "runtime":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
             "resource_accounting":resource_accounting,
             "asymptotic_diagnostic_p_value_caveat":"Stationarity/moment/HAC asymptotics are not certified by these data; Holm40 does not repair invalid marginal p-values. Shared calendar and overlapping targets prohibit interpreting ten seeds as ten independent markets.",
             "R1_accredited":False,"R3_accredited":False,"new_forecast_model_fits":0,
             "new_forecast_model_inference":0,"new_statistical_resamples":0,
             "limitations":["Fit/heldout dates and target regimes differ; RMSE ratio<=1 does not establish absence of overfit.",
                            "No prospectively chosen generalization margin/power was reconstructed.",
                            "Historical and extended residual diagnostics retain distinct weights, calendars and Holm40/Holm160 families; neither is V3 or an R3 superiority family.",
                            "No point-plus-interval data define a full predictive CDF, so PIT/CRPS are not fabricated.",
                            "No BDS or Giacomini-Rossi test is claimed completed.",
                            "FLOPs/energy and a full common-hardware Pareto frontier remain unmeasured."]}
    write_json(output/"adequacy_receipt.json",receipt)
    print(json.dumps({"status":receipt["status"],"inputs_verified":receipt["inputs_verified"],
                      "tables":len(products),"new_forecast_model_fits":0,"new_forecast_model_inference":0,
                      "new_statistical_resamples":0,"R3_accredited":False}))
    return receipt


def verify_only(archive,results):
    """Recompute in a temporary folder; compare numeric CSV values with tolerance.

    Bundled bytes must match their receipt. Recomputed floating-point results
    may differ within1e-9relative/1e-12absolute (p-values1e-15absolute).
    Source files and input hashes retain exact comparisons.
    """
    expected=json.loads((results/"adequacy_receipt.json").read_text())
    assert expected["script_sha256"]==sha(__file__),"Bundled receipt belongs to another script"
    assert expected["input_manifest_sha256"]==sha(archive/"input_manifest.json"),"Receipt input manifest binding differs"
    for name,digest in expected["output_sha256"].items():
        assert sha(results/name)==digest,"Bundled output was changed: "+name
    numeric=0;records=0
    with tempfile.TemporaryDirectory(prefix="e3-existing-adequacy-verify-") as folder:
        current=main(archive,Path(folder))
        for key in ("status","configuration","inputs_verified","input_manifest_sha256","source_provenance","output_rows","resource_accounting",
                    "R1_accredited","R3_accredited","new_forecast_model_fits","new_forecast_model_inference","new_statistical_resamples"):
            assert current[key]==expected[key],"Receipt semantic mismatch: "+key
        for name in expected["output_sha256"]:
            a,b=read_csv(results/name),read_csv(Path(folder)/name)
            assert len(a)==len(b)
            for left,right in zip(a,b):
                assert left.keys()==right.keys()
                records+=1
                for key,value in left.items():
                    try:
                        x,y=float(value),float(right[key])
                    except (ValueError,TypeError):
                        assert value==right[key],(name,key,value,right[key])
                    else:
                        tolerance=1e-15 if "_p" in key else 1e-12
                        assert math.isfinite(x) and math.isfinite(y)
                        assert math.isclose(x,y,rel_tol=1e-9,abs_tol=tolerance),(name,key,x,y)
                        numeric+=1
    print(json.dumps({"status":"verified_existing_adequacy_numeric_tolerance","records_compared":records,
                      "numeric_cells_compared":numeric,"new_forecast_model_fits":0,"new_forecast_model_inference":0,
                      "new_statistical_resamples":0}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=Path(__file__).parent/"archive")
    parser.add_argument("--output",type=Path,default=Path(__file__).parent/"results")
    parser.add_argument("--source-dir",type=Path,help="Optional original completed-audit static folder to verify12 source tables")
    parser.add_argument("--verify-only",action="store_true",help="Verify bundled bytes and recompute only in a temporary directory")
    args=parser.parse_args()
    if args.source_dir:verify_original_source(args.archive,args.source_dir)
    if args.verify_only:verify_only(args.archive,args.output)
    else:main(args.archive,args.output)
