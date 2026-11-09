"""Conditional CPU search/storage arithmetic; this module never trains a model.

The twelve architecture profiles contain three certainty anchors and one
uniform draw from each of nine disjoint strata of the remaining 997 designs.
Their arithmetic mean is not an estimate of the complete space's mean. We use
an inclusion-probability weighted total and retain every censored observation.
The result is conditional on stationary measured per-step costs, profiled
structural options and an explicit batch-cost assumption. It is not a measured
campaign duration or a statistical confidence interval.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

try:
    from . import e2_search_plan as planner
except ImportError:
    import e2_search_plan as planner

VERSION = "e2-cpu-budget-v1"


def _nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be a finite nonnegative number")
    return float(value)


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(name + f" must be an integer >= {minimum}")
    return value


def _pattern_id(widths, bits):
    return "w" + "-".join(map(str, widths)) + ":b" + "".join(map(str, bits))


def profile_sampling_frame(protocol):
    """Recover the exact inclusion probability of each registered pilot draw."""
    planner.validate_protocol(protocol)
    draws = planner.noise_pilot_architectures(protocol)
    if len(draws) != 12:
        raise ValueError("This estimator requires the declared 3 anchors + 9 draws")
    space = [(w, b) for w in itertools.product(protocol["nas"]["widths"], repeat=3)
             for b in itertools.product((0, 1), repeat=3)]
    anchors = [((64, 64, 64), (0, 1, 0)), ((16, 16, 16), (0, 0, 0)),
               ((256, 256, 256), (1, 1, 1))]
    remainder = [p for p in space if p not in anchors]
    result = []
    for i, draw in enumerate(draws):
        pair = (tuple(draw["widths"]), tuple(draw["dropout_pattern"]))
        if i < 3:
            if pair != anchors[i]:
                raise ValueError("Anchor order differs from the registered sampling design")
            stratum, size = None, 1
        else:
            stratum = i - 3
            left, right = len(remainder)*stratum//9, len(remainder)*(stratum+1)//9
            if pair not in remainder[left:right]:
                raise ValueError("Pilot draw does not belong to its declared stratum")
            size = right-left
        result.append({**deepcopy(draw), "role": "anchor" if i < 3 else "sample",
                       "stratum_index": stratum, "stratum_size": size,
                       "inclusion_probability": 1.0/size, "expansion_weight": size})
    if sum(r["expansion_weight"] for r in result) != 1000:
        raise ValueError("Sampling weights do not cover the 1000-design population")
    return result


def ht_total(frame, observations, *, population_size=1000):
    """Certainty-anchor + stratified Horvitz-Thompson total, without survivor bias.

    observations maps pattern_id to {status:'success', value:nonnegative}.
    Any missing/censored/failed sample makes the point estimate unavailable.
    The observed weighted subtotal is NOT a lower bound on the population.
    One draw per nontrivial stratum cannot identify within-stratum variance.
    """
    _positive_int(population_size, "population_size")
    if not isinstance(frame, list) or not frame:
        raise ValueError("A nonempty sampling frame is required")
    ids = [row["pattern_id"] for row in frame]
    if len(set(ids)) != len(ids):
        raise ValueError("Sampling frame pattern IDs must be unique")
    weights = [_positive_int(row["stratum_size"], "stratum_size") for row in frame]
    if sum(weights) != population_size:
        raise ValueError("Expansion weights do not cover the declared population")
    strata = [row["stratum_index"] for row in frame if row["role"] == "sample"]
    if len(set(strata)) != len(strata):
        raise ValueError("Exactly one sampled unit per stratum is required")
    known, incomplete = 0.0, []
    for row, weight in zip(frame, weights):
        if row["role"] == "anchor" and weight != 1:
            raise ValueError("Certainty anchors cannot be expanded")
        observation = observations.get(row["pattern_id"])
        if not isinstance(observation, dict) or observation.get("status") != "success":
            incomplete.append({"pattern_id": row["pattern_id"], "expansion_weight": weight,
                               "status": observation.get("status", "missing")
                               if isinstance(observation, dict) else "missing"})
            continue
        known += weight * _nonnegative(observation.get("value"), "sampled cost")
    return {"estimated_population_total": None if incomplete else known,
            "observed_weighted_subtotal_not_population_bound": known,
            "incomplete_samples": incomplete, "population_size": population_size,
            "within_stratum_variance_estimate": None,
            "confidence_interval": None,
            "variance_reason": "One draw per nontrivial stratum; no within-stratum variance estimator.",
            "method": "certainty anchors plus sum(stratum_size * one uniform sampled cost)"}


def batch_count(n, batch_size):
    """The singleton-tail redistribution retains ceil(N/batch) batches."""
    _positive_int(n, "training observations", 2)
    _positive_int(batch_size, "batch_size", 2)
    return (n + batch_size - 1)//batch_size


def project_phase(rate, *, train_n, validation_n, epochs, batch_size,
                  selection, batch_assumption="constant_step"):
    """Expand separated timing components; count setup/finalization only once.

    Required rate fields are derived from a successful fixed-epoch profile:
    profile_train_n, profile_validation_n, profile_batch_size, profile_epochs,
    optimizer_steps, update_loop_seconds, train_evaluation_seconds,
    validation_seconds_per_pass, setup_seconds, finalization_seconds,
    checkpoint_clone_seconds. Finalization already contains one checkpoint
    clone. Selection reserves an additional clone at every later epoch, a
    conditional worst-improvement scenario; disk serialization is unmeasured.
    """
    _positive_int(epochs, "epochs")
    if type(selection) is not bool:
        raise ValueError("selection must be explicit")
    if batch_assumption not in ("constant_step", "linear_batch"):
        raise ValueError("Unknown explicit batch-cost assumption")
    pe = _positive_int(rate["profile_epochs"], "profile_epochs")
    pb = _positive_int(rate["profile_batch_size"], "profile_batch_size", 2)
    pn = _positive_int(rate["profile_train_n"], "profile_train_n", 2)
    profile_steps = _positive_int(rate["optimizer_steps"], "optimizer_steps")
    if profile_steps != pe * batch_count(pn, pb):
        raise ValueError("Profile step count does not match fixed epochs and batch geometry")
    target_steps = epochs * batch_count(train_n, batch_size)
    factor = batch_size/pb if batch_assumption == "linear_batch" else 1.0
    loop = _nonnegative(rate["update_loop_seconds"], "update_loop_seconds") / profile_steps
    training = loop * target_steps * factor
    # Every fixed-epoch training stage also computes a train-set score.
    train_evaluation = (_nonnegative(rate["train_evaluation_seconds"], "train eval")
                        / profile_steps * target_steps * factor)
    validation = 0.0
    if selection:
        _positive_int(validation_n, "validation_n", 2)
        pvn = _positive_int(rate["profile_validation_n"], "profile validation_n", 2)
        validation = (_nonnegative(rate["validation_seconds_per_pass"], "validation pass")
                      * batch_count(validation_n, batch_size)/batch_count(pvn, pb)
                      * epochs * factor)
    setup = _nonnegative(rate["setup_seconds"], "setup")
    finalization = _nonnegative(rate["finalization_seconds"], "finalization")
    extra_checkpoint = ((epochs-1) * _nonnegative(rate["checkpoint_clone_seconds"], "clone")
                        if selection else 0.0)
    components = {"update_loop_seconds": training, "train_evaluation_seconds": train_evaluation,
                  "validation_seconds": validation, "setup_seconds_once": setup,
                  "finalization_seconds_once": finalization,
                  "additional_in_memory_checkpoint_seconds": extra_checkpoint}
    return {"conditional_compute_seconds_excluding_disk": sum(components.values()),
            "components": components, "target_optimizer_steps": target_steps,
            "batch_cost_assumption": batch_assumption,
            "batch_cost_transfer_unvalidated": batch_size != pb,
            "cohort_size_transfer_unvalidated": train_n != pn or (selection and validation_n != rate["profile_validation_n"]),
            "validation_cost_basis": rate.get("validation_cost_basis", "supplied separated validation-pass rate"),
            "disk_checkpoint_seconds": None,
            "limits": ["Measured loop rates are assumed stationary beyond the short profile.",
                       "Batch-cost scenarios are assumptions, not validated bounds.",
                       "Per-step initialization effects and future contention may change rates.",
                       "Selection reserves a checkpoint clone every epoch; no disk throughput was measured."]}


def project_trajectory(profile, *, train_n, validation_n, epochs, batch_size,
                       selection, batch_assumption="constant_step"):
    """Project all phases, retaining failures instead of dropping slow cases."""
    if profile.get("status") != "success":
        return {"status": profile.get("status", "missing"), "value": None,
                "reason": "Censored or failed worker; no successful-survivor extrapolation."}
    worker = profile.get("worker_profile") or {}
    phase_rates = worker.get("budget_phase_rates")
    if not isinstance(phase_rates, dict) or not phase_rates:
        return {"status": "timing_decomposition_missing", "value": None,
                "reason": "Stage elapsed time alone cannot separate setup from repeated epoch work."}
    phases = {name: project_phase(rate, train_n=train_n, validation_n=validation_n,
                                  epochs=epochs, batch_size=batch_size, selection=selection,
                                  batch_assumption=batch_assumption)
              for name, rate in phase_rates.items()}
    build = _nonnegative(worker.get("build_seconds", 0), "model build")
    between = _nonnegative(worker.get("between_stage_setup_seconds", 0), "between-stage setup")
    value = build+between+sum(r["conditional_compute_seconds_excluding_disk"] for r in phases.values())
    return {"status": "success", "value": value, "phases": phases,
            "model_build_seconds_once": build, "between_stage_setup_seconds_once": between,
            "disk_serialization_seconds": None,
            "structural_hpo_cost_transfer_unvalidated": True}


def allocation_caps(protocol, *, k, R):
    """Operational per-stage limits are caps, not predicted runtimes."""
    count = planner.budget(protocol, k, R)
    short_cap = _nonnegative(protocol["hardware"]["short_stage_wall_limit_seconds"], "short cap")
    full_cap = _nonnegative(protocol["hardware"]["full_stage_wall_limit_seconds"], "full cap")
    seconds = count["short_phase_total"]*short_cap + count["full_phase_total"]*full_cap
    return {"scope": count["scope"], "short_stages": count["short_phase_total"],
            "full_stages": count["full_phase_total"], "serial_stage_allocation_cap_seconds": seconds,
            "serial_stage_allocation_cap_hours": seconds/3600,
            "is_runtime_estimate": False,
            "classical_cost_included": False,
            "does_not_guarantee_completion_before_timeout": True,
            "excludes_process_launch_setup_disk_and_scientific_components": count["excluded"]}


def normalize_profile_timing(profile):
    """Adapt the fixed-epoch CPU worker's measured phase timers to cost rates.

    No observation is dropped. An absent auxiliary-validation timer is an
    explicit train-evaluation-throughput proxy, not invented measurement. Its
    basis remains in every phase rate. Unsuccessful or incomplete records keep
    their status and cannot produce a complete population estimate.
    """
    row = deepcopy(profile)
    if row.get("status") != "success":
        return row
    worker = row.get("worker_profile") or {}
    if worker.get("budget_phase_rates"):
        return row
    timings, receipts = worker.get("stage_timing"), worker.get("phase_receipts")
    if not isinstance(timings, dict) or not isinstance(receipts, dict) or not timings:
        return row
    pn = _positive_int(worker.get("train_n"), "profile training cohort", 2)
    nv = _positive_int(worker.get("validation_n"), "profile validation cohort", 2)
    batch = worker.get("batch_size", (row.get("job") or {}).get("batch_size", 32))
    _positive_int(batch, "measured batch size", 2)
    validation = worker.get("validation_by_stage_seconds") or {}
    rates = {}
    for phase, timing in timings.items():
        receipt = receipts.get(phase)
        if not isinstance(receipt, dict):
            raise ValueError("Every measured phase timer needs its own receipt")
        epochs = _positive_int(receipt.get("epochs_completed"), "observed fixed epochs")
        train_eval = _nonnegative(timing["train_evaluation_seconds"], "train-evaluation timer")
        measured = validation.get(phase)
        if measured is None:
            validation_seconds = train_eval/epochs/batch_count(pn, batch)*batch_count(nv, batch)
            basis = "train-evaluation throughput proxy; auxiliary validation not measured"
        else:
            validation_seconds = _nonnegative(measured, "measured validation timer")
            basis = "measured forward evaluation on profile validation cohort"
        rates[phase] = {
            "profile_train_n": pn, "profile_validation_n": nv,
            "profile_batch_size": batch, "profile_epochs": epochs,
            "optimizer_steps": _positive_int(receipt.get("optimizer_steps"), "observed steps"),
            "update_loop_seconds": _nonnegative(timing["update_loop_seconds"], "update-loop timer"),
            "train_evaluation_seconds": train_eval,
            "validation_seconds_per_pass": validation_seconds,
            "validation_cost_basis": basis,
            "setup_seconds": _nonnegative(timing["setup_seconds"], "setup timer"),
            "finalization_seconds": _nonnegative(timing["finalization_seconds"], "finalization timer"),
            "checkpoint_clone_seconds": _nonnegative(timing["checkpoint_clone_seconds"], "in-memory clone timer")}
    worker["budget_phase_rates"] = rates
    worker["between_stage_setup_seconds"] = _nonnegative(worker.get("fit_management_seconds", 0), "fit-management timer")
    worker["timing_normalization_limits"] = [
        "Fixed-epoch timers do not measure selected-training early stopping or every-epoch best-state copies.",
        "Auxiliary validation uses an explicitly labeled throughput proxy if not measured.",
        "Build/management and setup/finalization are counted once, not multiplied by future epochs.",
        "No checkpoint disk throughput is measured."]
    row["worker_profile"] = worker
    return row


def index_profiles(results):
    """Index each immutable job; duplicates are an error, never a best-time choice."""
    if isinstance(results, dict):
        results = results.get("results")
    if not isinstance(results, list):
        raise ValueError("Profile results must be a list or a summary with results")
    index = {}
    for row in results:
        row = normalize_profile_timing(row)
        job = row.get("job") or {}
        name, cohort = job.get("model"), job.get("cohort")
        if not isinstance(name, str) or cohort not in ("small", "large"):
            raise ValueError("Every profile needs explicit model and cohort")
        pattern = job.get("pattern")
        pattern_id = pattern.get("pattern_id") if isinstance(pattern, dict) else None
        key = (name, cohort, pattern_id)
        if key in index:
            raise ValueError("Duplicate profile job; cannot choose a fastest survivor")
        index[key] = row
    return index


def reference_cohort_scaling_check(protocol, results):
    """Report held-out small-reference residuals; do not call them validation bounds.

    The same reference is profiled at161 and619 training observations. This
    compares the large-profile stationary-step projection against the actual
    small fit. It does not validate unprofiled architectures,200 epochs,
    batch16/64, disk I/O or a statistical sampling confidence interval.
    """
    index = index_profiles(results)
    reference = _pattern_id([64]*3, [0, 1, 0])
    rows = []
    for spec in protocol["models"]:
        if not spec["nas"]:
            continue
        name = spec["model"]
        small, large = index.get((name, "small", reference)), index.get((name, "large", reference))
        if not small or not large or small.get("status") != "success" or large.get("status") != "success":
            rows.append({"model": name, "status": "not_estimable_missing_or_censored_reference_pair"})
            continue
        worker = small.get("worker_profile") or {}
        receipts = worker.get("phase_receipts") or {}
        epochs = {r.get("epochs_completed") for r in receipts.values()}
        if len(epochs) != 1:
            rows.append({"model": name, "status": "not_estimable_unequal_or_missing_phase_epochs"})
            continue
        ep = epochs.pop()
        projected = project_trajectory(large, train_n=worker["train_n"], validation_n=0,
                                       epochs=ep, batch_size=worker.get("batch_size", 32),
                                       selection=False)
        if projected["status"] != "success":
            rows.append({"model": name, "status": projected["status"]})
            continue
        actual = _nonnegative(worker["build_seconds"], "small model build")+_nonnegative(worker["fit_seconds"], "small fit")
        predicted = projected["value"]
        rows.append({"model": name, "status": "observed_reference_comparison_only",
                     "small_train_n": worker["train_n"],
                     "large_train_n": large["worker_profile"]["train_n"],
                     "fixed_epochs": ep, "observed_small_build_and_fit_seconds": actual,
                     "projected_from_large_seconds": predicted,
                     "observed_over_projected_ratio": actual/predicted if predicted > 0 else None,
                     "not_a_bound_or_full_campaign_validation": True})
    return rows


def _sum_or_missing(values):
    values = list(values)
    return None if any(v is None for v in values) else sum(values)


def _profile_cost(index, name, pattern, *, cohort="large", **projection):
    row = index.get((name, cohort, pattern))
    if row is None:
        return {"status": "missing", "value": None}
    return project_trajectory(row, **projection)


def project_search(protocol, results, panels, *, k, R, batch_assumption="constant_step",
                   nas_batch_size=32):
    """Conditional HPO/NAS/refit compute scenarios over all eleven origins.

    Does not pretend to know which finalists/architecture win. Phase2/refit
    costs for NAS use the estimated uniform-space mean (reference included
    explicitly), an unvalidated finalist-distribution assumption. The cost of
    each HPO row uses its actual batch count but profiled structural options.
    Global pilots, classical fits, I/O and excluded scientific work are listed
    separately, never silently assigned zero.
    """
    planner.validate_protocol(protocol)
    planner.budget(protocol, k, R)
    frame, profiles = profile_sampling_frame(protocol), index_profiles(results)
    origins = panels.get("origins") if isinstance(panels, dict) else panels
    if not isinstance(origins, list) or len(origins) != protocol["budget"]["selection_origins"]:
        raise ValueError("Supply every actual inner/refit panel, including final selection")
    ids = [row["origin"] for row in origins]
    if len(set(ids)) != len(ids):
        raise ValueError("Origin identities must be unique")
    for row in origins:
        for field in ("inner_train_n", "refit_train_n", "validation_n"):
            _positive_int(row[field], field, 2)
    reference_id = _pattern_id([64]*3, [0, 1, 0])
    all_models = protocol["models"]+protocol["neural_reference_profiles"]
    out = {}
    for model in all_models:
        name, nas = model["model"], model["nas"]
        reference = reference_id if nas else None
        hpo_rows = planner.generate_hpo(protocol, name)
        hpo_costs, short_costs, phase2_costs, refit_costs, missing = [], [], [], [], []
        for origin in origins:
            n, nv, nr = origin["inner_train_n"], origin["validation_n"], origin["refit_train_n"]
            for trial in hpo_rows:
                estimate = _profile_cost(profiles, name, reference, train_n=n, validation_n=nv,
                                         epochs=protocol["common_hpo"]["max_epochs"],
                                         batch_size=trial["batch_size"], selection=True,
                                         batch_assumption=batch_assumption)
                value = estimate["value"]
                hpo_costs.append(None if value is None else value*len(protocol["common_hpo"]["optimization_seeds"]))
                if estimate["status"] != "success":
                    missing.append({"origin": origin["origin"], "component": "HPO", "status": estimate["status"]})
            if nas:
                estimates = {}
                for component, epochs, selection, size in (
                        ("short", protocol["nas"]["phase1_epochs"], True, n),
                        ("full", protocol["nas"]["phase2_epochs"], True, n),
                        ("refit", protocol["nas"]["phase2_epochs"], False, nr)):
                    values = {sample["pattern_id"]: _profile_cost(
                        profiles, name, sample["pattern_id"], train_n=size, validation_n=nv,
                        epochs=epochs, batch_size=nas_batch_size, selection=selection,
                        batch_assumption=batch_assumption) for sample in frame}
                    weighted = ht_total(frame, values)
                    estimates[component] = weighted
                    if weighted["incomplete_samples"]:
                        missing.append({"origin": origin["origin"], "component": component,
                                        "incomplete_samples": weighted["incomplete_samples"]})
                total_short = estimates["short"]["estimated_population_total"]
                short_costs.append(total_short)
                total_full = estimates["full"]["estimated_population_total"]
                ref = _profile_cost(profiles, name, reference, train_n=n, validation_n=nv,
                                    epochs=protocol["nas"]["phase2_epochs"], batch_size=nas_batch_size,
                                    selection=True, batch_assumption=batch_assumption)["value"]
                # Reserve k * mean(non-reference architectures) + reference.
                # This is an explicit conditional finalist cost distribution.
                phase2_costs.append(None if total_full is None or ref is None else
                                    (k*(total_full-ref)/999+ref)*len(protocol["nas"]["phase2_seeds"]))
                total_refit = estimates["refit"]["estimated_population_total"]
                refit_costs.append(None if total_refit is None else total_refit/1000*R)
            else:
                value = _profile_cost(profiles, name, None, train_n=nr, validation_n=nv,
                                      epochs=protocol["common_hpo"]["max_epochs"],
                                      batch_size=nas_batch_size, selection=False,
                                      batch_assumption=batch_assumption)["value"]
                refit_costs.append(None if value is None else value*R)
        components = {"hpo": _sum_or_missing(hpo_costs), "nas_phase1": _sum_or_missing(short_costs),
                      "nas_phase2_conditional_finalists": _sum_or_missing(phase2_costs),
                      "selected_refits_conditional_architecture": _sum_or_missing(refit_costs)}
        out[name] = {"nas": nas, "components_seconds": components,
                     "conditional_compute_seconds_excluding_disk": _sum_or_missing(components.values()),
                     "incomplete_or_censored": missing}
    subtotal = sum(row["conditional_compute_seconds_excluding_disk"] for row in out.values()
                   if row["conditional_compute_seconds_excluding_disk"] is not None)
    total = _sum_or_missing(row["conditional_compute_seconds_excluding_disk"] for row in out.values())
    return {"version": VERSION, "status": "conditional_complete_core_projection" if total is not None
            else "incomplete_no_full_campaign_point_estimate", "k": k, "R_resource_scenario": R,
            "batch_cost_assumption": batch_assumption, "nas_selected_batch_assumed": nas_batch_size,
            "models": out, "core_conditional_compute_seconds_excluding_disk": total,
            "core_conditional_compute_hours_excluding_disk": None if total is None else total/3600,
            "known_family_subtotal_seconds_not_total": subtotal,
            "known_family_subtotal_hours_not_total": subtotal/3600,
            "full_campaign_wall_hours": None,
            "not_estimated": ["global noise/resource/power pilots", "classical fits", "disk/checkpoint serialization",
                              "controls", "calibration-specific refits", "robustness", "ablations",
                              "interpretability", "inference", "environment reproduction"],
            "scope": "HPO + repeated NAS + selected-network refits over actual inner/refit cohort counts only",
            "assumptions": ["Serial execution at the profiled thread configuration; no parallel speedup assumed.",
                            "Stationary per-step rates and known reference validation geometry.",
                            "Auxiliary validation uses measured rates when available, otherwise an explicit train-evaluation throughput proxy.",
                            "Unprofiled batches and structural HPO changes are scenario assumptions, not validated bounds.",
                            "NAS finalists/refit architecture costs equal the uniform population mean, not an observed winner.",
                            "One sample per stratum cannot provide a design-based confidence interval.",
                            "No prediction of achieved statistical power, final early-stop epochs or full-campaign duration."],
            "allocation_caps_not_forecast": allocation_caps(protocol, k=k, R=R)}


def micro_fallback_sensitivity(protocol, summary, panels, *, k, R,
                               batch_assumption="constant_step", nas_batch_size=32):
    """Separate surrogate scenario; never change the observed full-size result.

    A resource-censored primary may have its own separately registered one-batch
    micro attempt. We retain the failed primary, and produce an OPTIONAL second
    scenario by transferring that micro's per-step rates to target cohorts. The
    stratified weights then apply to surrogate costs, not fully observed units;
    this is not a design-unbiased HT estimate of the actual full-cohort cost.
    """
    if not isinstance(summary, dict) or not isinstance(summary.get("results"), list):
        raise ValueError("Micro sensitivity requires a summary with preserved primary results")
    primaries = summary["results"]
    primary_by_id = {}
    for row in primaries:
        job_id = (row.get("job") or {}).get("job_id")
        if not isinstance(job_id, str) or not job_id or job_id in primary_by_id:
            raise ValueError("Micro linkage requires distinct primary job IDs")
        primary_by_id[job_id] = row
    micros = summary.get("micro_fallback_results", [])
    if not isinstance(micros, list):
        raise ValueError("Separate micro attempts must be a list")
    replacements, linkage = {}, []
    for micro in micros:
        job = micro.get("job") or {}
        parent_id = job.get("parent_job_id")
        if parent_id not in primary_by_id or parent_id in replacements:
            raise ValueError("Micro parent is missing or has repeated replacement candidates")
        primary = primary_by_id[parent_id]
        parent_job = primary["job"]
        error = (primary.get("worker_profile") or {}).get("error_code")
        eligible = primary.get("status") in ("timeout", "memory_limit_observed") or error == "resource_limit"
        if not eligible or primary.get("status") == "success":
            raise ValueError("Only a retained primary resource failure can have a micro proxy")
        for field in ("model", "cohort", "pattern", "threads", "backend"):
            if job.get(field) != parent_job.get(field):
                raise ValueError("Micro model/cohort/architecture/runtime differs from its primary")
        if job.get("phase") != "micro_fallback" or job.get("epochs") != 1 or job.get("batch_windows") != 32:
            raise ValueError("Only the registered one-epoch32-window micro design can be projected")
        entry = {"parent_job_id": parent_id, "primary_status": primary.get("status"),
                 "primary_consumed_wall_seconds": primary.get("elapsed_wall_seconds"),
                 "micro_job_id": job.get("job_id"), "micro_status": micro.get("status"),
                 "micro_consumed_wall_seconds": micro.get("elapsed_wall_seconds"),
                 "proxy_used": False}
        if micro.get("status") == "success":
            worker = micro.get("worker_profile") or {}
            if worker.get("train_n") != 32 or worker.get("fixed_epoch_count") != 1:
                raise ValueError("Completed micro's actual training geometry must be32 windows/one epoch")
            replacement = deepcopy(micro)
            replacement["job"] = deepcopy(parent_job)
            replacement["is_surrogate_micro_extrapolation"] = True
            replacements[parent_id] = replacement
            entry.update(proxy_used=True, micro_actual_train_n=32, micro_fixed_epochs=1)
        else:
            # Keep the key occupied so another micro cannot be chosen as a
            # faster successful survivor for the same primary.
            replacements[parent_id] = None
        linkage.append(entry)
    projected_rows = [replacements.get(row["job"]["job_id"]) or deepcopy(row) for row in primaries]
    scenario = project_search(protocol, projected_rows, panels, k=k, R=R,
                              batch_assumption=batch_assumption, nas_batch_size=nas_batch_size)
    scenario["status"] = "micro_extrapolation_sensitivity_not_observed_completion"
    scenario["primary_censoring_not_resolved"] = True
    scenario["full_cohort_fit_completion_established"] = False
    scenario["weighted_surrogate_costs_not_unbiased_full_size_HT_estimate"] = True
    scenario["micro_linkage"] = linkage
    scenario["primary_results_modified"] = False
    scenario["additional_assumptions"] = [
        "Resource-failed full-size sample costs remain unobserved.",
        "One-batch one-epoch rates are transferred to larger cohorts/epoch caps without demonstrated linearity.",
        "A completed micro cannot establish full-cohort memory feasibility or completion before timeout.",
        "Weighted surrogate outputs are sensitivity arithmetic, not successful-survivor measurements or a design-based confidence interval."]
    return scenario


def checkpoint_storage_inventory(inventory, *, selection_origins, trials=18, hpo_seeds=2,
                                 finalists=20, phase2_seeds=5, final_seeds=30):
    """Model-state byte allocations from explicit architecture counts.

    inventory is a list of {model,nas,reference_state_bytes,architectures:[
      {pattern_id,state_bytes,parameter_count,auxiliary_state_bytes:[...]}]}.
    State sizes include buffers when supplied. Storage below is an uncompressed
    tensor-payload lower allocation: archive headers, metadata, RNG, logs,
    optimizer states and filesystem effects are excluded. No old file is deleted.
    NAS phase2/refit chosen architectures are unknown, so allocation ranges use
    the exact min/max of the supplied full inventory, not a sampled average.
    """
    for value, label in ((selection_origins, "origins"), (trials, "trials"),
                         (hpo_seeds, "hpo_seeds"), (finalists, "finalists"),
                         (phase2_seeds, "phase2_seeds"), (final_seeds, "final_seeds")):
        _positive_int(value, label)
    totals = {"hpo_model_states": 0, "nas_phase1_model_states": 0,
              "nas_phase2_min": 0, "nas_phase2_max": 0,
              "selected_refits_min": 0, "selected_refits_max": 0}
    models = {}
    for model in inventory:
        architectures = model["architectures"]
        if not architectures:
            raise ValueError("Every model needs parameter/state inventory")
        ids = [a["pattern_id"] for a in architectures]
        if len(set(ids)) != len(ids):
            raise ValueError("Parameter inventory must contain unique parent architecture IDs")
        sizes = []
        for row in architectures:
            size = _positive_int(row["state_bytes"], "state bytes")
            aux = [_positive_int(v, "auxiliary state bytes") for v in row.get("auxiliary_state_bytes", [])]
            sizes.append(size+sum(aux))
        reference = _positive_int(model["reference_state_bytes"], "reference state bytes")
        hpo = reference*trials*hpo_seeds*selection_origins
        nas = model["nas"]
        if type(nas) is not bool:
            raise ValueError("Explicit NAS status required")
        if nas and len(architectures) != 1000:
            raise ValueError("NAS storage requires the complete1000 architecture inventory")
        phase1 = sum(sizes)*selection_origins if nas else 0
        if nas:
            # Reference is budgeted once in addition to k unknown distinct
            # finalists, even if a future winner set might already include it.
            ref_id = model.get("reference_pattern_id", _pattern_id([64]*3, [0, 1, 0]))
            if ref_id not in ids:
                raise ValueError("NAS inventory must identify the actual reference architecture")
            ref_index = ids.index(ref_id)
            if sizes[ref_index] != reference:
                raise ValueError("Reference allocation must include its predictor and auxiliary states")
            candidates = sorted(size for i, size in enumerate(sizes) if i != ref_index)
            if finalists > len(candidates):
                raise ValueError("Finalist count must leave room for the reference allocation")
            p2min = (sum(candidates[:finalists])+reference)*phase2_seeds*selection_origins
            p2max = (sum(candidates[-finalists:])+reference)*phase2_seeds*selection_origins
        else:
            p2min = p2max = 0
        rmin = min(sizes)*selection_origins*final_seeds
        rmax = max(sizes)*selection_origins*final_seeds
        values = {"hpo_model_states": hpo, "nas_phase1_model_states": phase1,
                  "nas_phase2_min": p2min, "nas_phase2_max": p2max,
                  "selected_refits_min": rmin, "selected_refits_max": rmax}
        models[model["model"]] = values
        for key in totals:
            totals[key] += values[key]
    lower = totals["hpo_model_states"]+totals["nas_phase1_model_states"]+totals["nas_phase2_min"]+totals["selected_refits_min"]
    upper_payload = totals["hpo_model_states"]+totals["nas_phase1_model_states"]+totals["nas_phase2_max"]+totals["selected_refits_max"]
    return {"models": models, "components_payload_bytes": totals,
            "retain_all_completed_stage_checkpoints_payload_bytes_range": [lower, upper_payload],
            "retain_only_selected_refits_payload_bytes_range": [totals["selected_refits_min"], totals["selected_refits_max"]],
            "payload_range_is_not_serialized_disk_bound": True,
            "reference_addition_allocation_assumption": "Reserve k distinct non-reference finalists plus reference; actual reference inclusion may reduce this allocation.",
            "structural_options_condition": "Ranges apply only to the complete architecture inventory supplied, not unprofiled HPO structural alternatives.",
            "optimizer_states_included": False, "deletion_executed_or_authorized": False,
            "excluded": ["archive/serialization overhead", "RNG and optimizer resumption states",
                         "logs", "pilots and diagnostic checkpoints", "existing retained history"],
            "policy_note": "These are prospective retention allocations only; no historical weights are removed."}


def meta_parameter_inventory(protocol, *, fixed_state_bytes=None):
    """Count complete-space tensors on meta, never allocate CPU/GPU model weights.

    This optional function imports PyTorch only when explicitly invoked. Dropout
    bits have no tensor state, so state counts are cached by the125 width tuples
    per family and expanded to all1000 parent architecture IDs. All six new
    families use the fixed profiling structural options (TimesNet six kernels,
    top-k4), not an assumed winning HPO row. Fixed bodies use provided measured
    state sizes, or4*known parameter count as an explicitly parameter-only
    payload approximation. No forward, backward or optimizer is run.
    """
    import torch
    try:
        from .e2_nas_factory import build_nas_model
        from .e2_nas_autoencoder import SAEAutoencoderNAS
        from .e2_nas_common import NASConfig
        from .e2_training_models import FIXED_COUNTS
    except ImportError:
        from e2_nas_factory import build_nas_model
        from e2_nas_autoencoder import SAEAutoencoderNAS
        from e2_nas_common import NASConfig
        from e2_training_models import FIXED_COUNTS
    planner.validate_protocol(protocol)
    fixed_state_bytes = fixed_state_bytes or {}
    models, cache, auxiliary_cache, constructions = [], {}, {}, 0
    reference = _pattern_id([64]*3, [0, 1, 0])

    def counts(module):
        if any(value.device.type != "meta" for value in module.state_dict().values()):
            raise RuntimeError("Parameter inventory must not allocate model tensors outside meta")
        return {"parameter_count": sum(p.numel() for p in module.parameters()),
                "trainable_parameter_count": sum(p.numel() for p in module.parameters() if p.requires_grad),
                "state_bytes": sum(t.numel()*t.element_size() for t in module.state_dict().values())}

    for spec in protocol["models"]+protocol["neural_reference_profiles"]:
        name, nas = spec["model"], spec["nas"]
        if nas:
            rows = []
            for row in planner.enumerate_architectures(protocol, name):
                widths = tuple(row["widths"])
                key = (name, widths)
                if key not in cache:
                    with torch.device("meta"):
                        model = build_nas_model(name, widths, [0, 0, 0])
                        cache[key] = counts(model)
                        del model
                    constructions += 1
                aux = []
                if spec["auxiliary_phases"]:
                    auxkey = widths[:2]
                    if auxkey not in auxiliary_cache:
                        with torch.device("meta"):
                            config = NASConfig(widths, (0, 0, 0))
                            auxiliary_cache[auxkey] = [counts(SAEAutoencoderNAS(config, phase))
                                                       for phase in (1, 2)]
                        constructions += 2
                    aux = auxiliary_cache[auxkey]
                rows.append({"pattern_id": _pattern_id(row["widths"], row["dropout_pattern"]),
                             "widths": row["widths"], "dropout_pattern": row["dropout_pattern"],
                             **cache[key], "auxiliary_state_bytes": [a["state_bytes"] for a in aux],
                             "auxiliary_parameter_counts": [a["parameter_count"] for a in aux]})
            refrow = next(r for r in rows if r["pattern_id"] == reference)
            reference_bytes = refrow["state_bytes"]+sum(refrow["auxiliary_state_bytes"])
            models.append({"model": name, "nas": True, "reference_pattern_id": reference,
                           "reference_state_bytes": reference_bytes, "architectures": rows,
                           "structural_options": "fixed pilot extras; TimesNet six kernels/topk4",
                           "full_space_parameter_count_min": min(r["parameter_count"] for r in rows),
                           "full_space_parameter_count_max": max(r["parameter_count"] for r in rows),
                           "adam_fp32_weights_grad_two_moments_min_bytes": 16*min(r["trainable_parameter_count"] for r in rows),
                           "adam_fp32_weights_grad_two_moments_max_bytes": 16*max(r["trainable_parameter_count"] for r in rows)})
        else:
            parameter_count = FIXED_COUNTS[name]
            state = fixed_state_bytes.get(name, 4*parameter_count)
            _positive_int(state, "fixed state size")
            models.append({"model": name, "nas": False, "reference_state_bytes": state,
                           "architectures": [{"pattern_id": "fixed", "state_bytes": state,
                                               "parameter_count": parameter_count,
                                               "auxiliary_state_bytes": []}],
                           "state_size_basis": "measured profile" if name in fixed_state_bytes else "4*parameters only; buffers excluded"})
    return {"version": VERSION, "device": "meta", "numerical_forward_executed": False,
            "optimizer_steps": 0, "market_data_read": False, "models": models,
            "meta_constructor_calls": constructions,
            "cache_reason": "Dropout bits do not alter parameter/buffer shapes;125 width tuples expanded to1000 IDs.",
            "memory_lower_bound_limits": ["4 FP32 slots per trainable parameter only: weights, gradients and two Adam moments.",
                                          "Excludes activations, temporary/fused kernels, copied checkpoints, allocator/runtime and buffers.",
                                          "This lower bound cannot certify a model fits the memory limit."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter-inventory", type=Path)
    parser.add_argument("--include-micro-sensitivity", action="store_true")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--R", type=int, default=30)
    parser.add_argument("--nas-batch", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--batch-assumption", choices=("constant_step", "linear_batch"), default="constant_step")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Budget output must be new")
    protocol = json.loads(args.protocol.read_text())
    results = json.loads(args.profiles.read_text())
    panels = json.loads(args.panels.read_text())
    report = project_search(protocol, results, panels, k=args.k, R=args.R,
                            batch_assumption=args.batch_assumption, nas_batch_size=args.nas_batch)
    report["reference_cohort_scaling_check"] = reference_cohort_scaling_check(protocol, results)
    if args.include_micro_sensitivity:
        report["separate_micro_sensitivity_not_observed_total"] = micro_fallback_sensitivity(
            protocol, results, panels, k=args.k, R=args.R,
            batch_assumption=args.batch_assumption, nas_batch_size=args.nas_batch)
    if args.parameter_inventory is not None:
        inventory = json.loads(args.parameter_inventory.read_text())
        report["prospective_checkpoint_storage"] = checkpoint_storage_inventory(
            inventory["models"] if isinstance(inventory, dict) else inventory,
            selection_origins=protocol["budget"]["selection_origins"],
            trials=protocol["common_hpo"]["trials"],
            hpo_seeds=len(protocol["common_hpo"]["optimization_seeds"]),
            finalists=args.k, phase2_seeds=len(protocol["nas"]["phase2_seeds"]), final_seeds=args.R)
    report["input_sha256"] = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                              for key, path in (("protocol", args.protocol), ("profiles", args.profiles), ("panels", args.panels))}
    if args.parameter_inventory is not None:
        report["input_sha256"]["parameter_inventory"] = hashlib.sha256(args.parameter_inventory.read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "core_hours": report["core_conditional_compute_hours_excluding_disk"],
                      "full_campaign_wall_hours": None, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
