"""Assemble16 conditional CPU-budget cases after all86 profiles were attempted.

This wrapper uses only the standard library and the arithmetic-only
e2_cpu_budget module. It does not import Torch, construct models, read market
observations, fit, simulate, retry workers or alter frozen source/protocol files.
The separate lower-epoch protocol is a counterfactual cost sensitivity only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

try:
    from . import e2_cpu_budget as budget
except ImportError:
    import e2_cpu_budget as budget

VERSION = "e2-cpu-budget-scenario-bundle-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT/"book_e2/_static/search_protocol_20261109/protocol.json"
R_SCENARIOS = [6, 10, 20, 30]
BATCH_ASSUMPTIONS = ("constant_step", "linear_batch")


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular immutable input file is required: "+str(path))
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _write_new(path, value):
    # Failures/censored records must still be valid portable JSON; no NaN or
    # silent overwrite is admitted in the assembled evidence.
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _shape_key(job):
    pattern = job.get("pattern")
    if pattern is None:
        pattern_id = None
    elif isinstance(pattern, dict):
        pattern_id = pattern.get("pattern_id")
        if (not isinstance(pattern_id, str)
                or pattern_id != budget._pattern_id(pattern.get("widths", []), pattern.get("dropout_pattern", []))):
            raise ValueError("Profile pattern ID differs from its widths/bits")
    else:
        raise ValueError("Profile pattern must be an explicit object or None")
    return job.get("model"), job.get("cohort"), pattern_id


def validate_scaling_summary(summary, protocol, registration, registration_sha256):
    """Require the entire registered sample, including unsuccessful attempts."""
    if (summary.get("phase") != "scaling"
            or summary.get("all_requested_jobs_attempted") is not True
            or not isinstance(summary.get("results"), list)
            or len(summary["results"]) != 86):
        raise ValueError("The final scaling summary must preserve all86 attempted primary jobs")
    if summary.get("registration_sha256") != registration_sha256:
        raise ValueError("Scaling summary is not bound to the supplied exact registration bytes")
    estimation = registration.get("budget_estimation") or {}
    if estimation.get("k_operational") != 40 or estimation.get("R_scenarios") != R_SCENARIOS:
        raise ValueError("This wrapper requires registered operative k40 and R6/10/20/30")
    if registration.get("status") != "registered_before_new_market_fits":
        raise ValueError("A draft registration cannot be used as the profile provenance")
    specs = protocol["models"]+protocol["neural_reference_profiles"]
    nas = [spec["model"] for spec in specs if spec["nas"]]
    fixed = [spec["model"] for spec in specs if not spec["nas"]]
    if len(nas) != 6 or len(fixed) != 8:
        raise ValueError("Expected six NAS families and eight fixed neural systems")
    frame = budget.profile_sampling_frame(protocol)
    reference = next(row["pattern_id"] for row in frame if row["is_reference"])
    expected = {(name, "large", sample["pattern_id"]) for name in nas for sample in frame}
    expected |= {(name, "small", reference) for name in nas}
    expected |= {(name, "large", None) for name in fixed}
    seen, ids, threads, backends = set(), set(), set(), set()
    for row in summary["results"]:
        job = row.get("job") or {}
        key = _shape_key(job)
        job_id = job.get("job_id")
        if (job.get("phase") != "scaling" or key not in expected or key in seen
                or not isinstance(job_id, str) or not job_id or job_id in ids):
            raise ValueError("Primary profile jobs must match the distinct registered86-case domain")
        seen.add(key); ids.add(job_id)
        threads.add(job.get("threads")); backends.add(job.get("backend"))
        if not isinstance(row.get("status"), str):
            raise ValueError("Every attempted primary needs its own explicit status")
    if seen != expected:
        raise ValueError("Missing registered primary profile units")
    if len(threads) != 1 or next(iter(threads)) not in (1, 2, 4) or len(backends) != 1:
        raise ValueError("The cost sample must use one common selected CPU/thread policy")
    backend = next(iter(backends))
    if backend not in ("dense", "cropped"):
        raise ValueError("Unknown CPU convolution backend")
    return {"expected_primary_attempts": 86, "provided_primary_attempts": 86,
            "primary_status_counts": dict(Counter(row["status"] for row in summary["results"])),
            "all_primary_fits_completed": all(row["status"] == "success" for row in summary["results"]),
            "separate_micro_attempts": len(summary.get("micro_fallback_results", [])),
            "common_threads": next(iter(threads)), "common_kernel_policy": backend}


def lower_epoch_protocol(protocol, *, original_sha256, registration_sha256):
    """Create a named analysis copy; never mutate or authorize another campaign."""
    if (protocol["common_hpo"]["max_epochs"] != 200
            or protocol["nas"]["phase1_epochs"] != 50
            or protocol["nas"]["phase2_epochs"] != 200):
        raise ValueError("Source maximum scenario must remain short50/full200")
    lower = deepcopy(protocol)
    lower["status"] = "counterfactual_epoch_cost_sensitivity_only_not_execution_protocol"
    lower["common_hpo"]["max_epochs"] = 100
    lower["nas"]["phase1_epochs"] = 25
    lower["nas"]["phase2_epochs"] = 100
    lower["cost_sensitivity_metadata"] = {
        "purpose": "Illustrative half-cap time sensitivity; not observed early stopping or execution authorization.",
        "source_protocol_sha256": original_sha256,
        "source_registration_sha256": registration_sha256,
        "operative_k": 40, "source_short_epochs": 50, "source_full_epochs": 200,
        "sensitivity_short_epochs": 25, "sensitivity_full_epochs": 100,
        "hardware_stage_time_caps_unchanged": True,
        "checkpoint_storage_attempt_counts_unchanged": True}
    budget.planner.validate_protocol(lower)
    return lower


def assemble(protocol, scaling, panels, inventory, registration, *, source_hashes):
    """Return an in-memory bundle; no source files or profile outcomes are changed."""
    budget.planner.validate_protocol(protocol)
    if registration.get("source_protocol_sha256") != source_hashes["protocol"]:
        raise ValueError("Registration identifies different protocol bytes")
    validation = validate_scaling_summary(scaling, protocol, registration, source_hashes["registration"])
    if not isinstance(panels, dict) or len(panels.get("origins", [])) != 11:
        raise ValueError("All eleven actual cohort panels are required")
    inventory_models = inventory.get("models") if isinstance(inventory, dict) else inventory
    if not isinstance(inventory_models, list):
        raise ValueError("Supply the existing full-space parameter inventory")
    expected_names = {spec["model"] for spec in protocol["models"]+protocol["neural_reference_profiles"]}
    names = [model.get("model") for model in inventory_models]
    if len(names) != len(expected_names) or set(names) != expected_names:
        raise ValueError("Parameter inventory must contain exactly the fourteen versioned neural systems")
    lower = lower_epoch_protocol(protocol, original_sha256=source_hashes["protocol"],
                                 registration_sha256=source_hashes["registration"])
    epoch_protocols = {"max50_200": protocol, "lower25_100": lower}
    scenarios = {}
    has_micros = bool(scaling.get("micro_fallback_results"))
    for epoch_name, scenario_protocol in epoch_protocols.items():
        for R in R_SCENARIOS:
            for assumption in BATCH_ASSUMPTIONS:
                key = f"k40_R{R}_{epoch_name}_{assumption}"
                case = budget.project_search(scenario_protocol, scaling, panels,
                                             k=40, R=R, batch_assumption=assumption,
                                             nas_batch_size=32)
                case["scenario_id"] = key
                case["primary_scaling_sha256"] = source_hashes["scaling"]
                case["epoch_scenario"] = {
                    "name": epoch_name, "short_epochs": scenario_protocol["nas"]["phase1_epochs"],
                    "full_epochs": scenario_protocol["common_hpo"]["max_epochs"],
                    "lower_scenario_is_counterfactual_only": epoch_name == "lower25_100"}
                if has_micros:
                    case["separate_micro_sensitivity_not_observed_total"] = budget.micro_fallback_sensitivity(
                        scenario_protocol, scaling, panels, k=40, R=R,
                        batch_assumption=assumption, nas_batch_size=32)
                scenarios[key] = case
    storage = {f"k40_R{R}": budget.checkpoint_storage_inventory(
        inventory_models, selection_origins=11, trials=18, hpo_seeds=2,
        finalists=40, phase2_seeds=5, final_seeds=R) for R in R_SCENARIOS}
    primary_id = "k40_R30_max50_200_constant_step"
    primary = scenarios[primary_id]
    proxy = primary.get("separate_micro_sensitivity_not_observed_total") or {}
    return {
        "version": VERSION, "budget_calculator_version": budget.VERSION,
        "status": "conditional_scenarios_assembled" if validation["all_primary_fits_completed"]
                  else "conditional_scenarios_with_preserved_primary_failures",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_scaling_sha256": source_hashes["scaling"],
        "input_sha256": deepcopy(source_hashes),
        "operative_k": 40, "R_resource_scenarios": R_SCENARIOS,
        "epoch_scenarios": ["max50_200", "lower25_100_counterfactual_only"],
        "batch_cost_assumptions_not_validated_bounds": list(BATCH_ASSUMPTIONS),
        "profile_domain_validation": validation,
        "primary_case_id": primary_id,
        "primary_case_projection": {
            "case_id": primary_id,
            "conditional_core_hours_primary_observations": primary["core_conditional_compute_hours_excluding_disk"],
            "conditional_core_hours_separate_micro_proxy": proxy.get("core_conditional_compute_hours_excluding_disk"),
            "known_family_subtotal_hours_not_total": primary["known_family_subtotal_hours_not_total"],
            "full_campaign_wall_hours": None,
            "scope": primary["scope"],
            "no_profile_failure_or_consumed_cost_was_removed": True},
        "scenario_results": scenarios,
        "reference_cohort_scaling_check": budget.reference_cohort_scaling_check(protocol, scaling),
        "panels": deepcopy(panels),
        "meta_parameter_inventory": deepcopy(inventory),
        "storage_k40_scenarios": storage,
        "sampling_frame": budget.profile_sampling_frame(protocol),
        "registration_budget_estimation": deepcopy(registration["budget_estimation"]),
        "lower_epoch_counterfactual_protocol": lower,
        "primary_failures_retained": [{"job": deepcopy(row["job"]), "status": row["status"],
            "elapsed_wall_seconds": row.get("elapsed_wall_seconds"),
            "optimizer_steps": row.get("optimizer_steps"),
            "optimizer_steps_censored": row.get("optimizer_steps_censored")}
            for row in scaling["results"] if row["status"] != "success"],
        "micro_attempt_linkage": [{"job_id": row["job"].get("job_id"),
            "parent_job_id": row["job"].get("parent_job_id"), "status": row["status"],
            "actual_train_n": (row.get("worker_profile") or {}).get("train_n"),
            "elapsed_wall_seconds": row.get("elapsed_wall_seconds")}
            for row in scaling.get("micro_fallback_results", [])],
        "execution_in_this_wrapper": {"training": False, "profiling": False,
            "model_imports": False, "market_data_read": False,
            "frozen_sources_or_protocol_modified": False, "old_weights_deleted": False},
        "limits": [
            "Core compute projections cover HPO, repeated NAS and selected refits, not full-campaign wall time.",
            "Raw profile failures and their consumed cost remain in the bound scaling evidence.",
            "Micro rates are a separate unvalidated extrapolation, not observed full-size completion or memory feasibility.",
            "One draw per stratum cannot identify a design-based within-stratum confidence interval.",
            "Batch, finalist-distribution and long-horizon timing transfers are explicit assumptions.",
            "Lower25/100 is an illustrative cost sensitivity, not observed early stopping or changed execution authorization.",
            "Checkpoint figures are prospective uncompressed tensor-payload allocations, not serialized disk bounds.",
            "Classical timing, disk I/O and remaining scientific components stay unestimated when unprofiled."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling-summary", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--parameter-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Budget scenario output directory must be new")
    values, hashes = {}, {}
    for name, path in (("protocol", args.protocol), ("scaling", args.scaling_summary),
                       ("panels", args.panels), ("parameter_inventory", args.parameter_inventory),
                       ("registration", args.registration)):
        values[name], hashes[name] = _read(path)
    hashes["budget_calculator_source"] = hashlib.sha256(Path(budget.__file__).read_bytes()).hexdigest()
    hashes["scenario_wrapper_source"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = assemble(values["protocol"], values["scaling"], values["panels"],
                      values["parameter_inventory"], values["registration"], source_hashes=hashes)
    # All domain checks and arithmetic finish before the output directory is
    # created. No partial profile or missing-case set can be packaged as final.
    args.output.mkdir(parents=True)
    scenario_files = {}
    for key, case in report["scenario_results"].items():
        filename = key+".json"
        scenario_files[filename] = _write_new(args.output/filename, case)
    report["scenario_files_sha256"] = scenario_files
    report["lower_protocol_file_sha256"] = _write_new(
        args.output/"protocol_lower_epoch_sensitivity_25_100.json",
        report["lower_epoch_counterfactual_protocol"])
    _write_new(args.output/"report.json", report)
    print(json.dumps({"status": report["status"], "report": str(args.output/"report.json"),
                      "primary_scaling_sha256": report["primary_scaling_sha256"],
                      "scenario_files": len(scenario_files),
                      "primary_case_projection": report["primary_case_projection"]},
                     ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
