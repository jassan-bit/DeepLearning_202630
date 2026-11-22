"""Execute the pre-frozen bounded synthetic plan; no market data or fitting.

Run from package root with OPENBLAS_NUM_THREADS=1. Refuses to overwrite evidence.
For reproduction, use --output-dir /tmp/temporal-reproduction (new directory).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.statistics import analyze


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cp(count, n):
    return [float(beta.ppf(.025, count, n-count+1)) if count else 0.,
            float(beta.ppf(.975, count+1, n-count)) if count < n else 1.]


def binomial_result(count, n):
    return {"count": int(count), "n": int(n), "estimate": float(count/n) if n else None,
            "cp95": cp(count, n) if n else [0., 1.]}


def ar(rng, n, phi, dims):
    out = np.empty((n+300, dims))
    out[0] = rng.uniform(-1, 1, size=dims)
    for t in range(1, len(out)):
        out[t] = phi*out[t-1] + rng.uniform(-1, 1, size=dims)
    return out[300:]


def generate(plan, spec, scenario_index, replicate):
    n, rivals, overlap = plan["n_dates"], plan["n_rivals"], spec["overlap"]
    rng = np.random.default_rng(np.random.SeedSequence([plan["seed"], scenario_index, replicate]))
    exogenous = ar(rng, n+1, .3, 1)[:, 0]
    q0 = 4 + .3*exogenous[1:]
    shared = ar(rng, n+overlap-1, spec["phi"], 1)
    independent = ar(rng, n+overlap-1, spec["phi"], rivals)
    innovations = .3*(.6*shared + .8*independent)
    differential = np.stack([innovations[k:k+n] for k in range(overlap)]).mean(axis=0).T
    shifts = np.zeros(rivals)
    shifts[spec["null_rivals"]:] = spec["shift"]
    if spec["boundary_null"] == "practical":
        qj = q0[None, :]/(1-plan["margin"])**2 + differential
        truth_d = 4/((1-plan["margin"])**2) - 4
        truth_gain = plan["margin"]
    else:
        qj = q0[None, :] + differential + shifts[:, None]
        truth_d = float(shifts[0])
        truth_gain = 1-np.sqrt(4/(4+truth_d))
    q = np.concatenate([q0[None, :], qj])
    offsets = np.linspace(-.01, .01, plan["n_fixed_seeds"])
    prediction = np.sqrt(q[:, None, :] + offsets[None, :, None])
    z = np.column_stack([np.ones(n), exogenous[:-1]])
    return prediction, np.zeros(n), z, float(truth_d), float(truth_gain)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source_dir = Path(__file__).resolve().parent
    plan_path, freeze_path = source_dir/"plan_v1.json", source_dir/"freeze_v1.json"
    plan, freeze = json.loads(plan_path.read_text()), json.loads(freeze_path.read_text())
    if sha(plan_path) != freeze["plan_sha256"] or sha(ROOT/"engine/statistics.py") != freeze["statistics_sha256"] or sha(Path(__file__)) != freeze["runner_sha256"]:
        raise RuntimeError("Frozen source/plan identity mismatch; retain v1 and create a new version")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path, report_path = args.output_dir/"replicates_v1.jsonl", args.output_dir/"report_v1.json"
    if raw_path.exists() or report_path.exists():
        raise FileExistsError("Evidence exists; use a fresh --output-dir for reproduction")
    started_cpu, started_wall = time.process_time(), time.monotonic()
    report = {"version": plan["version"], "plan_sha256": freeze["plan_sha256"],
              "statistics_sha256": freeze["statistics_sha256"], "runner_sha256": freeze["runner_sha256"],
              "scope": plan["purpose"], "frozen_criteria": plan["acceptance_fixed"], "scenarios": []}
    with raw_path.open("x") as raw:
        for scenario_index, spec in enumerate(plan["scenarios"]):
            rows = []
            for replicate in range(plan["replicates_per_stochastic_scenario"]):
                if time.process_time()-started_cpu >= plan["cpu_budget_seconds"]:
                    break
                p, y, z, truth_d, truth_gain = generate(plan, spec, scenario_index, replicate)
                result = analyze(p, y, ["candidate"]+[f"rival_{i:02d}" for i in range(plan["n_rivals"])],
                                 bootstrap_reps=plan["bootstrap_reps"], block_length=plan["block_length"],
                                 hac_lags=plan["hac_lags"], instruments=z, margin=plan["margin"],
                                 seed=plan["seed"]+scenario_index*100000+replicate)
                comparisons = result["comparisons"]
                null_label = spec["boundary_null"]
                false_reject, false_favorable = False, False
                method_false = {m: False for m in result["family"]["methods"]}
                for c in comparisons[:spec["null_rivals"]]:
                    for method, tests in c["methods"].items():
                        test = tests[null_label]
                        reject = test["adjusted_p"] < .05
                        false_reject |= reject
                        method_false[method] |= reject
                        false_favorable |= reject and test["direction"] == "candidate_better"
                ci_d, ci_gain = comparisons[0]["ci95_mse_difference"], comparisons[0]["ci95_relative_rmse_gain"]
                row = {"scenario": spec["id"], "replicate": replicate, "false_reject_any_null": bool(false_reject),
                       "false_favorable_any_null": bool(false_favorable), "method_false_reject": method_false,
                       "mse_coverage": bool(ci_d[0] <= truth_d <= ci_d[1]),
                       "rmse_gain_coverage": bool(ci_gain[0] <= truth_gain <= ci_gain[1]),
                       "per_rival_gate": [c["provisional_statistical_gate"] for c in comparisons],
                       "joint_gate": bool(result["all_rivals_statistical_gate"]),
                       "rival0_ci_mse": ci_d, "rival0_ci_gain": ci_gain,
                       "rival0_truth_mse": truth_d, "rival0_truth_gain": truth_gain}
                rows.append(row)
                raw.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                raw.flush()
            n = len(rows)
            scenario = {"id": spec["id"], "replicates_completed": n,
                        "complete": n == plan["replicates_per_stochastic_scenario"]}
            if n:
                scenario["fwer"] = binomial_result(sum(r["false_reject_any_null"] for r in rows), n) if spec["null_rivals"] else None
                scenario["false_favorable_fwer"] = binomial_result(sum(r["false_favorable_any_null"] for r in rows), n) if spec["null_rivals"] else None
                scenario["fwer_by_method"] = {m: binomial_result(sum(r["method_false_reject"][m] for r in rows), n) for m in rows[0]["method_false_reject"]} if spec["null_rivals"] else None
                scenario["mse_coverage"] = binomial_result(sum(r["mse_coverage"] for r in rows), n)
                scenario["rmse_gain_coverage"] = binomial_result(sum(r["rmse_gain_coverage"] for r in rows), n)
                scenario["joint_power"] = binomial_result(sum(r["joint_gate"] for r in rows), n)
                scenario["per_rival_power"] = [binomial_result(sum(r["per_rival_gate"][j] for r in rows), n) for j in range(plan["n_rivals"])]
                scenario["fwer_demonstrated_in_this_dgp"] = bool(scenario["fwer"] is not None and scenario["fwer"]["cp95"][1] <= .05)
                scenario["nominal_coverage_demonstrated_in_this_dgp"] = bool(scenario["mse_coverage"]["cp95"][0] >= .95 and scenario["rmse_gain_coverage"]["cp95"][0] >= .95)
                scenario["power_target_demonstrated_in_this_large_alternative"] = bool(spec["null_rivals"] == 0 and min(p["cp95"][0] for p in scenario["per_rival_power"]) >= .9)
                scenario["failure_detected"] = bool((scenario["fwer"] is not None and scenario["fwer"]["cp95"][0] > .05) or scenario["mse_coverage"]["cp95"][1] < .95 or scenario["rmse_gain_coverage"]["cp95"][1] < .95)
            report["scenarios"].append(scenario)
            print(json.dumps({"scenario": spec["id"], "n": n, "fwer": scenario.get("fwer"), "coverage": scenario.get("mse_coverage"), "joint_power": scenario.get("joint_power")}), flush=True)
    report["cpu_seconds"] = time.process_time()-started_cpu
    report["wall_seconds"] = time.monotonic()-started_wall
    report["raw_results_sha256"] = sha(raw_path)
    report["all_scenarios_complete"] = all(s["complete"] for s in report["scenarios"])
    report["general_validation_established"] = False
    report["protocol_closure_permitted"] = False
    report["conclusion"] = "Bounded DGP-specific evidence only; uncertainty and any failures are retained. No market/R3 or general FWER/coverage/power claim."
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"cpu_seconds": report["cpu_seconds"], "wall_seconds": report["wall_seconds"], "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
