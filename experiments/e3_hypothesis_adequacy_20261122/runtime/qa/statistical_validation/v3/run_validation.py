"""Frozen V3 synthetic audit. No market data, training, or test opening."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engine.statistics import analyze

GENERATOR = ROOT/"qa/statistical_validation/runner_v1.py"
spec = importlib.util.spec_from_file_location("preserved_generator", GENERATOR)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
generate = module.generate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binomial(count, n):
    return {"count": int(count), "n": n, "estimate": count/n if n else None,
            "cp95": [float(beta.ppf(.025, count, n-count+1)) if count else 0.,
                     float(beta.ppf(.975, count+1, n-count)) if count < n else 1.]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    plan = json.loads((folder/"plan.json").read_text())
    freeze = json.loads((folder/"freeze.json").read_text())
    for relative, digest in freeze["sha256"].items():
        if sha(ROOT/relative) != digest:
            raise RuntimeError(f"Frozen identity mismatch: {relative}; retain results and version any correction")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path, report_path = args.output_dir/"replicates.jsonl", args.output_dir/"report.json"
    if raw_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite evidence; choose a new --output-dir")
    cpu, wall = time.process_time(), time.monotonic()
    report = {"version": plan["version"], "protocol_version": plan["protocol_version"],
              "plan_sha256": sha(folder/"plan.json"), "source_hashes": freeze["sha256"],
              "scope": plan["purpose"], "criteria": plan["acceptance_fixed"], "scenarios": [],
              "intervals": "Operational CI99 and diagnostic CI95 are named and assessed separately; Monte Carlo CP intervals are pointwise, not simultaneous."}
    with raw_path.open("x") as raw:
        for scenario_index, scenario in enumerate(plan["scenarios"]):
            rows = []
            for replicate in range(plan["replicates_per_stochastic_scenario"]):
                if time.process_time()-cpu >= plan["cpu_budget_seconds"]:
                    break
                p, y, z, truth_d, truth_gain = generate(plan, scenario, scenario_index, replicate)
                result = analyze(p, y, ["candidate"]+[f"rival_{i:02d}" for i in range(plan["n_rivals"])],
                                 bootstrap_reps=plan["bootstrap_reps"], block_length=plan["block_length"],
                                 hac_lags=plan["hac_lags"], instruments=z, margin=plan["margin"],
                                 seed=plan["seed"]+scenario_index*100000+replicate,
                                 reference_reps=plan["reference_reps"], reference_seed=plan["reference_seed"],
                                 bootstrap_backend=plan["bootstrap_backend"])
                comparisons = result["comparisons"]
                method_false = {m: False for m in result["family"]["methods"]}
                false_reject, false_favorable = False, False
                for c in comparisons[:scenario["null_rivals"]]:
                    for method, tests in c["methods"].items():
                        test = tests[scenario["boundary_null"]]
                        reject = test["adjusted_p"] < .05
                        false_reject |= reject
                        false_favorable |= reject and test["direction"] == "candidate_better"
                        method_false[method] |= reject
                c0 = comparisons[0]
                row = {"scenario": scenario["id"], "replicate": replicate,
                       "false_reject_any_null": bool(false_reject), "false_favorable_any_null": bool(false_favorable),
                       "method_false_reject": method_false, "per_rival_gate": [c["provisional_statistical_gate"] for c in comparisons],
                       "joint_gate": bool(result["all_rivals_statistical_gate"]), "truth_mse": truth_d, "truth_gain": truth_gain,
                       "ci95_mse": c0["ci95_mse_difference"], "ci99_mse": c0["ci99_mse_difference"],
                       "ci95_gain": c0["ci95_relative_rmse_gain"], "ci99_gain": c0["ci99_relative_rmse_gain"]}
                for label, truth in [("mse", truth_d), ("gain", truth_gain)]:
                    for confidence in [95, 99]:
                        interval = row[f"ci{confidence}_{label}"]
                        row[f"coverage_ci{confidence}_{label}"] = bool(interval is not None and interval[0] <= truth <= interval[1])
                # Preserve every p/statistic/direction/tail count compactly in a
                # declared order: rival, method in METHODS order, EPA/practical.
                tests = [t for c in comparisons for m in c["methods"].values() for t in m.values()]
                row["test_vectors"] = {key: [t.get(key) for t in tests] for key in
                                       ["statistic", "raw_p", "adjusted_p", "direction", "status", "mc_tail_count", "mc_reps", "mc_p_rule"]}
                rows.append(row)
                raw.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+"\n")
                raw.flush()
                if scenario_index == 0 and replicate == 0:
                    report["reference_identity"] = result["settings"]
                    report["bootstrap_backend"] = result["bootstrap_backend"]
            n = len(rows)
            item = {"id": scenario["id"], "n": n, "complete": n == plan["replicates_per_stochastic_scenario"]}
            if n:
                item["fwer"] = binomial(sum(r["false_reject_any_null"] for r in rows), n) if scenario["null_rivals"] else None
                item["false_favorable_fwer"] = binomial(sum(r["false_favorable_any_null"] for r in rows), n) if scenario["null_rivals"] else None
                item["fwer_by_method"] = {m: binomial(sum(r["method_false_reject"][m] for r in rows), n) for m in method_false} if scenario["null_rivals"] else None
                for label in ["mse", "gain"]:
                    for confidence in [95, 99]:
                        key = f"coverage_ci{confidence}_{label}"
                        item[key] = binomial(sum(r[key] for r in rows), n)
                item["joint_power"] = binomial(sum(r["joint_gate"] for r in rows), n)
                item["per_rival_power"] = [binomial(sum(r["per_rival_gate"][j] for r in rows), n) for j in range(plan["n_rivals"])]
                item["fwer_demonstrated"] = bool(item["fwer"] is not None and item["fwer"]["cp95"][1] <= .05)
                item["operational_ci99_coverage_demonstrated"] = all(item[f"coverage_ci99_{label}"]["cp95"][0] >= .95 for label in ["mse", "gain"])
                item["diagnostic_ci95_undercoverage_detected"] = any(item[f"coverage_ci95_{label}"]["cp95"][1] < .95 for label in ["mse", "gain"])
                item["failure_detected"] = bool((item["fwer"] is not None and item["fwer"]["cp95"][0] > .05) or any(item[f"coverage_ci99_{label}"]["cp95"][1] < .95 for label in ["mse", "gain"]))
                item["strong_alternative_power_demonstrated"] = bool(scenario["null_rivals"] == 0 and min(x["cp95"][0] for x in item["per_rival_power"]) >= .9)
            report["scenarios"].append(item)
            print(json.dumps({"scenario": item["id"], "n": n, "fwer": item.get("fwer"), "coverage99mse": item.get("coverage_ci99_mse"), "coverage99gain": item.get("coverage_ci99_gain"), "joint_power": item.get("joint_power")}), flush=True)
    complete = all(x["complete"] for x in report["scenarios"])
    failures = [x["id"] for x in report["scenarios"] if x.get("failure_detected")]
    bounded = bool(complete and not failures and all(x["operational_ci99_coverage_demonstrated"] for x in report["scenarios"]) and all(x["fwer_demonstrated"] for x in report["scenarios"][:-1]) and report["scenarios"][-1]["strong_alternative_power_demonstrated"])
    report.update(cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall,
                  all_scenarios_complete=complete, failed_scenarios=failures,
                  status="FAILED_FINITE_SAMPLE_VALIDATION" if failures else "BOUNDED_DGP_SCOPE_VALIDATED" if bounded else "INCONCLUSIVE_BOUNDED_VALIDATION",
                  admitted=bounded, conditional_scope_admitted=bounded, general_validation_established=False,
                  protocol_closure_permitted=False, raw_results_sha256=sha(raw_path))
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "cpu_seconds": report["cpu_seconds"], "wall_seconds": report["wall_seconds"], "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
