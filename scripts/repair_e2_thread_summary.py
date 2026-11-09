"""Recover a missing thread summary from immutable receipts, without fitting.

The preregistered runner completed all12 fits and then rejected integer keys
while serializing its runtime policy. This helper verifies retained sources,
registration, workers and event chains before exclusively publishing the same
summary schema with string JSON keys. No source, receipt or log is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_e2_cpu_scaling as runner


def repair(phase_dir,registration,kernels_summary,frozen_runner,error_log,recovery_receipt):
    phase_dir=Path(phase_dir).resolve()
    registration=Path(registration).resolve()
    kernels_summary=Path(kernels_summary).resolve()
    frozen_runner=Path(frozen_runner).resolve()
    error_log=Path(error_log).resolve()
    recovery_receipt=Path(recovery_receipt).resolve()
    if (phase_dir/"summary.json").exists() or recovery_receipt.exists():
        raise FileExistsError("Summary and repair receipt must both be new")
    registered=runner.load_registration(registration)
    plan=json.loads((phase_dir/"plan.json").read_text())
    old_hash=runner.digest(frozen_runner)
    if old_hash!="c8fc2ff49421684d35b422345638fbc0e95f1827e52adff8cf91e4581f93cb90":
        raise ValueError("Archived preregistered runner is not the original exact source")
    if (plan["phase"]!="threads" or plan["source_script_sha256"]!=old_hash
            or plan["registration_sha256"]!=runner.digest(registration)
            or plan["kernels_summary_sha256"]!=runner.digest(kernels_summary)):
        raise ValueError("Retained plan does not match the registered original execution")
    kernels=json.loads(kernels_summary.read_text())
    for key in ("registration_commit","registration_sha256"):
        if kernels["plan"][key]!=plan[key]:raise ValueError("Runtime preregistrations differ")
    kernel_policy=runner.select_kernel_policy(kernels["results"],
        parity_suite_passed=kernels["plan"]["parity_suite_passed"])
    if kernel_policy!=plan["selected_kernel_policy"]:
        raise ValueError("Recorded kernel policy changed")
    expected=runner.build_jobs("threads",registered,kernel_backend=kernel_policy["selected_backend"])
    if expected!=plan["jobs"]:raise ValueError("Retained thread job domain changed")
    paths=sorted(phase_dir.glob("outcome-*.json"))
    if len(paths)!=12 or [p.name for p in paths]!=["outcome-%03d.json"%i for i in range(12)]:
        raise ValueError("All12 distinct retained outcomes are required")
    tracked=[registration,kernels_summary,frozen_runner,error_log,phase_dir/"plan.json",phase_dir/"events.jsonl",*paths]
    results=[]
    original_runner_path="scripts/run_e2_cpu_scaling.py"
    for index,path in enumerate(paths):
        record=json.loads(path.read_text());worker=record.get("worker_profile") or {}
        worker_path=phase_dir/("worker-%03d"%index)/"worker_profile.json"
        worker_events=worker_path.parent/"events.jsonl"
        if (record.get("job")!=expected[index] or record.get("status")!="success"
                or worker.get("status")!="completed" or worker.get("job")!=expected[index]
                or json.loads(worker_path.read_text())!=worker):
            raise ValueError("Retained worker is incomplete or its outcome copy differs")
        if any(worker.get(key)!=plan[key] for key in ("registration_commit","registration_sha256")):
            raise ValueError("Worker registration differs from retained plan")
        before,after=worker["source_sha256_before"],worker["source_sha256_after"]
        if (before.get(original_runner_path)!=old_hash or after.get(original_runner_path)!=old_hash
                or any(after.get(name)!=sha for name,sha in before.items())):
            raise ValueError("Worker sources changed while fitting")
        for name,sha in after.items():
            source=frozen_runner if name==original_runner_path else runner.ROOT/name
            if runner.digest(source)!=sha:raise ValueError("Frozen source differs: "+name)
        events=runner.read_events(worker_events)
        if events!=record["durable_events"] or events["incomplete_tail"]:
            raise ValueError("Worker durable events differ")
        if (worker.get("sources_unchanged_during_fit") is not True
                or worker.get("optimizer_steps")!=30 or record.get("optimizer_steps")!=30
                or record.get("optimizer_steps_censored") is not False):
            raise ValueError("A worker does not retain its complete30 steps")
        tracked.extend((worker_path,worker_events))
        results.append(record)
    events=runner.read_events(phase_dir/"events.jsonl")
    if events["incomplete_tail"] or events["events"][-1].get("kind")!="phase_finished":
        raise ValueError("A complete phase event chain is required")
    last=events["events"][-1]
    if last.get("known_optimizer_steps")!=360 or last.get("all_fits_completed") is not True:
        raise ValueError("Final phase event does not confirm the12 completed fits")
    error_text=error_log.read_text()
    marker="Traceback (most recent call last):"
    trace=error_text[error_text.rfind(marker):] if marker in error_text else error_text
    if not trace.rstrip().endswith("TypeError: Pilot JSON keys must be strings"):
        raise ValueError("Repair applies only to the recorded JSON key serialization failure")
    original_hashes={str(path):runner.digest(path) for path in tracked}
    policy=runner.select_thread_policy(results)
    if policy["selected_threads"] is None:raise ValueError("Complete runtime policy is required")
    summary={"version":runner.VERSION,"phase":"threads","status":"completed_all_attempts",
        "registration_sha256":plan["registration_sha256"],"registration_commit":plan["registration_commit"],
        "all_requested_jobs_attempted":True,"all_fits_completed":True,
        "results":results,"micro_fallback_results":[],"plan":plan,
        "kernel_policy":kernel_policy,"thread_policy":policy,
        "known_optimizer_steps":360,"unknown_optimizer_step_workers":0,
        "elapsed_workers_sum_seconds":sum(row["elapsed_wall_seconds"] for row in results),
        "no_final_market_data_read":True,"scientific_HPO_or_NAS_executed":False,
        "limits":["Profiles do not constitute scientific architecture or hyperparameter selection.",
                  "A one-batch micro profile is distinct from its failed full-cohort attempt.",
                  "Stratified cost estimates and extrapolations require declared assumptions.",
                  "Sampled RSS watchdog is not a hard kernel memory limit."],"phase_events":events}
    runner.write_json_new(phase_dir/"summary.json",summary)
    loaded=runner.load_runtime_policy(kernels_summary,phase_dir/"summary.json")
    if any(runner.digest(path)!=sha for path,sha in original_hashes.items()):
        raise ValueError("A retained input changed during repair")
    receipt={"version":"e2-thread-summary-serialization-repair-v1",
        "cause":"JSON metadata policy keys were integers; exclusive writer requires strings.",
        "failure_trace":trace,"old_runner_sha256":old_hash,
        "new_runner_sha256":runner.digest(runner.__file__),
        "repair_helper_sha256":runner.digest(__file__),
        "frozen_runner_path":str(frozen_runner),"summary_path":str(phase_dir/"summary.json"),
        "summary_sha256":runner.digest(phase_dir/"summary.json"),"original_input_sha256":original_hashes,
        "all_retained_inputs_unchanged":True,"training_executed":False,"market_data_read":False,
        "completed_fits_reused":12,"optimizer_steps_reused":360,"additional_optimizer_steps":0,
        "algorithm_or_measurements_changed":False,"selected_threads":loaded["threads"],
        "registration_commit":plan["registration_commit"],"registration_sha256":plan["registration_sha256"]}
    runner.write_json_new(recovery_receipt,receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ("phase-dir","registration","kernels-summary","frozen-runner","error-log","recovery-receipt"):
        parser.add_argument("--"+option,type=Path,required=True)
    args=parser.parse_args()
    result=repair(args.phase_dir,args.registration,args.kernels_summary,args.frozen_runner,args.error_log,args.recovery_receipt)
    print(json.dumps({key:result[key] for key in ("summary_sha256","selected_threads","completed_fits_reused","additional_optimizer_steps")}))


if __name__=="__main__":main()
