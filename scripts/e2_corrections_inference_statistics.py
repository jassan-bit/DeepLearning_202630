"""Execute the pre-specified, post-benchmark gap and CPU-cost diagnostics.

Reads frozen-inference outputs only. Does not fit, load models, change a margin,
or replace A2/RQ1. The immutable plan predates the corrective model forwards.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import t as student_t

from e2_effects_rq2_closure import (NEURAL, TOP10, SEEDS, load_data,
    moving_block_counts, sha256, write_csv, write_json, require)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "book_e2/_static/e2_corrections_20261115/inference_statistics/statistical_plan.json"
PLAN_SHA = "714f32c6ed1f55019092ec6ef6fd4b0bff554f53d180a79e3b6ae3f7ee72e1d0"


def holm(pvalues):
    values = [1.0 if p is None else float(p) for p in pvalues]
    require(all(0 <= p <= 1 for p in values), "Invalid p-values")
    result, previous = [None] * len(values), 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        previous = max(previous, min(1.0, (len(values)-rank)*values[index]))
        if pvalues[index] is not None:
            result[index] = previous
    return result


def six_seed_test(values, boundary, *, lower=True):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (6,) and np.isfinite(values).all(), "Six finite seed differences required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    if np.all(values == values[0]) or sd == 0:
        return dict(mean=mean, SD=sd, statistic=None, p_raw=None,
                    CI95_lower=None, CI95_upper=None, status="not_estimable_zero_seed_variance")
    se = sd/math.sqrt(6)
    statistic = (mean-boundary)/se
    p = float(student_t.cdf(statistic, 5) if lower else 2*student_t.sf(abs(statistic), 5))
    radius = float(student_t.ppf(.975, 5))*se
    return dict(mean=mean, SD=sd, statistic=statistic, p_raw=p,
                CI95_lower=mean-radius, CI95_upper=mean+radius,
                status="estimable_conditional_small_sample_t")


def checked_report(directory, name, stage):
    directory = Path(directory)
    report = json.loads((directory/name).read_text())
    require(report["status"] == "passed" and report["stage"] == stage,
            "Inference/timing admission report not passed")
    for relative, expected in report["all_output_files"].items():
        require(Path(relative).name == relative, "Unsafe report path")
        path = directory/relative
        require(path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"],
                "Frozen inference output changed: " + relative)
    return report


def gap_analysis(directory, y, prediction, delta_g):
    rows = list(csv.DictReader((Path(directory)/"risks_by_predictor.csv").open()))
    indexed = {}
    for row in rows:
        key = row["model"], int(row["seed"])
        require(key not in indexed, "Duplicate predictor risk")
        require(int(row["train_count"]) == 655 and int(row["calibration_count"]) == 53,
                "Wrong risk calendars")
        require(row["state_before_sha256"] == row["state_after_sha256"]
                and int(row["batch_size"]) == 1, "Wrong frozen inference state/mode")
        indexed[key] = row
    require(set(indexed) == {(m, s) for m in NEURAL for s in SEEDS}, "Incomplete risk registry")
    gaps, consistency = [], []
    for model in NEURAL:
        train = np.array([float(indexed[model,s]["train_rmse"]) for s in SEEDS])
        cal = np.array([float(indexed[model,s]["calibration_rmse"]) for s in SEEDS])
        require(np.isfinite(train).all() and np.isfinite(cal).all(), "Nonfinite risk")
        gaps.append({"model":model, "mean_train_RMSE":float(train.mean()),
            "mean_calibration_RMSE":float(cal.mean()), "delta_g":delta_g,
            **six_seed_test(cal-train, delta_g), "n_seed_pairs":6, "df":5,
            "scope":"conditional optimization uncertainty on train655/calibration53; not iid dates or universal absence of overfit"})
        test = np.sqrt(((prediction[model]-y)**2).mean(axis=1))
        consistency.append({"model":model, "mean_test_RMSE":float(test.mean()),
            "mean_calibration_RMSE":float(cal.mean()),
            **six_seed_test(test-cal, 0, lower=False), "n_seed_pairs":6, "df":5,
            "scope":"exploratory fixed-period test-minus-calibration consistency; no equivalence or attribution to overfitting"})
    for family in (gaps, consistency):
        for row, adjusted in zip(family, holm([r["p_raw"] for r in family])):
            row["p_Holm13"] = adjusted
            row["reject_FWER05"] = adjusted is not None and adjusted <= .05
    for row in gaps:
        row["within_tolerance_conditionally"] = row["reject_FWER05"]
        row["within_tolerance_stricter_alpha005"] = row["p_Holm13"] is not None and row["p_Holm13"] <= .005
    return gaps, consistency


def latency_matrices(rows, *, blocks=100, seeds=SEEDS, candidates=TOP10):
    indexed = {}
    for row in rows:
        candidate, block, seed, role = row["candidate"], int(row["block_id"]), int(row["seed"]), row["role"]
        require(candidate in candidates and 0 <= block < blocks and seed in seeds and role in ("candidate","reference"), "Wrong timing domain")
        key = candidate, block, seed, role
        require(key not in indexed, "Duplicate timing pair role")
        require(row["model"] == (candidate if role == "candidate" else "GRU32")
                and row["baseline"] == "GRU32" and int(row["batch_size"]) == 1
                and row["device"] == "cpu", "Incomparable timing identity/backend")
        elapsed = int(row["elapsed_ns"])
        require(elapsed > 0 and int(row["end_perf_counter_ns"])-int(row["start_perf_counter_ns"]) == elapsed,
                "Invalid raw elapsed duration")
        indexed[key] = row
    require(len(indexed) == len(candidates)*blocks*len(seeds)*2, "Missing paired timings")
    a = np.empty((blocks,len(candidates),len(seeds)), dtype=float)
    b = np.empty_like(a)
    for j, candidate in enumerate(candidates):
        for block in range(blocks):
            for k, seed in enumerate(seeds):
                first, second = [indexed[candidate,block,seed,r] for r in ("candidate","reference")]
                require(first["pair_id"] == second["pair_id"] and first["input_date"] == second["input_date"]
                        and first["order"] == second["order"] and {int(first["position"]),int(second["position"])} == {0,1},
                        "Timing pair/order/input mismatch")
                a[block,j,k], b[block,j,k] = int(first["elapsed_ns"])*1e-9, int(second["elapsed_ns"])*1e-9
    return a.mean(axis=2), b.mean(axis=2)


def centered_summary(observed, samples, boundary, *, lower=False):
    samples = np.asarray(samples,dtype=float)
    valid = np.isfinite(samples)
    require(valid.all() and math.isfinite(observed), "Non-estimable component; do not impute zero")
    null = boundary+(samples-observed)
    count = np.count_nonzero(null <= observed) if lower else np.count_nonzero(null >= observed)
    ci = np.quantile(samples,[.025,.975])
    return {"effect":float(observed), "CI95_lower":float(ci[0]), "CI95_upper":float(ci[1]),
            "p_raw":float((1+count)/(len(samples)+1)), "valid_draws":len(samples),
            "invalid_draws":0, "null_tail_count":int(count)}


def rq3_analysis(latency_dir, y, prediction, plan):
    design = plan["RQ3"]
    candidates = tuple(design["candidates"])
    models = candidates+("GRU32",)
    loss = np.stack([(prediction[m]-y)**2 for m in models])
    risks = np.sqrt(loss.mean(axis=2)).mean(axis=1)
    gains = risks[-1]-risks[:-1]
    sampled = []
    for counts in moving_block_counts(608,14,20000,design["gain_seed"]):
        q = np.sqrt(np.einsum("bt,mst->bms",counts,loss,optimize=True)/608).mean(axis=2)
        sampled.append(q[:,-1,None]-q[:,:-1])
    sampled_gain = np.concatenate(sampled)
    raw = list(csv.DictReader((Path(latency_dir)/"raw_cpu_latency.csv").open()))
    candidate, baseline = latency_matrices(raw,candidates=candidates)
    difference = candidate-baseline
    costs = difference.mean(axis=0)
    rng = np.random.default_rng(design["cost_seed"])
    indices = rng.integers(0,100,(20000,100))
    sampled_cost = difference[indices].mean(axis=1)
    rows = []
    for j, model in enumerate(candidates):
        gain = centered_summary(gains[j],sampled_gain[:,j],plan["delta"])
        cost = centered_summary(costs[j],sampled_cost[:,j],0,lower=True)
        rows.append({"candidate":model,"reference":"GRU32","delta":plan["delta"],
            **{"gain_"+k:v for k,v in gain.items()}, **{"cost_"+k:v for k,v in cost.items()},
            "candidate_mean_latency_seconds":float(candidate[:,j].mean()),
            "reference_mean_latency_seconds":float(baseline[:,j].mean()),
            "p_IUT":max(gain["p_raw"],cost["p_raw"]),"blocks":100,"seeds_per_block":6,
            "scope":"posthoc CPU cost and fixed608-date gain; separate exploratory Holm10 family"})
    for row, adjusted in zip(rows,holm([r["p_IUT"] for r in rows])):
        row["p_IUT_Holm10"] = adjusted
        row["joint_improvement_established"] = adjusted <= .05
    return rows


def execute(args):
    started = datetime.now(timezone.utc).isoformat()
    require(sha256(args.plan) == PLAN_SHA,"Statistical specification changed")
    plan = json.loads(args.plan.read_text())
    for path, expected in plan["inputs"].items():
        require(sha256(ROOT/path) == expected,"Pinned analysis input changed")
    inference = checked_report(args.predictions_output,"inference_report.json","predictions")
    timing = checked_report(args.latency_output,"latency_report.json","latency")
    require(inference["spec_sha256"] == timing["spec_sha256"],"Predictions and timing belong to different plans")
    require(inference["predictors_completed"] == inference["calibration_checks_passed"] == 78
            and inference["states_unchanged"] and timing["states_unchanged"],"Frozen predictor gate failed")
    require(not args.output.exists(),"Use a fresh output directory; preserve all earlier attempts")
    args.output.mkdir(parents=True)
    y,prediction,_,_,_ = load_data()
    gaps, consistency = gap_analysis(args.predictions_output,y,prediction,plan["delta_g"])
    joint = rq3_analysis(args.latency_output,y,prediction,plan)
    write_csv(args.output/"delta_g_Holm13.csv",gaps)
    write_csv(args.output/"validation_test_Holm13.csv",consistency)
    write_csv(args.output/"RQ3_cpu_IUT_Holm10.csv",joint)
    for path, expected in plan["inputs"].items():
        require(sha256(ROOT/path) == expected,"Analysis input changed during execution")
    summary = {"version":plan["version"],"gap_tests":13,
        "within_tolerance_conditional_Holm13":sum(r["within_tolerance_conditionally"] for r in gaps),
        "within_tolerance_stricter_alpha005":sum(r["within_tolerance_stricter_alpha005"] for r in gaps),
        "period_consistency_tests":13,"period_differences_Holm13":sum(r["reject_FWER05"] for r in consistency),
        "RQ3_tests":10,"RQ3_joint_improvements_Holm10":sum(r["joint_improvement_established"] for r in joint),
        "RQ3_backend":"CPU FP32 batch1, new timing campaign; historical A100 measurements unchanged",
        "new_model_fits":0,"primary_A2_changed":False,"primary_RQ1_changed":False,
        "inferential_status":"Exploratory, conditional on fixed periods/trajectories; t small-sample and block-bootstrap assumptions not universal guarantees"}
    write_json(args.output/"summary.json",summary)
    write_json(args.output/"execution_receipt.json",{"started_at_utc":started,
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":PLAN_SHA,
        "script_sha256":sha256(__file__),"inference_report_sha256":sha256(args.predictions_output/"inference_report.json"),
        "latency_report_sha256":sha256(args.latency_output/"latency_report.json"),
        "python":platform.python_version(),"numpy":np.__version__,"scipy":__import__("scipy").__version__})
    write_json(args.output/"manifest.json",{"files":{p.name:{"sha256":sha256(p),"bytes":p.stat().st_size}
        for p in sorted(args.output.iterdir()) if p.is_file()},"scope":"posthoc frozen-inference statistics"})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=DEFAULT_PLAN)
    p.add_argument("--predictions-output",type=Path,required=True)
    p.add_argument("--latency-output",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    print(json.dumps(execute(p.parse_args()),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
