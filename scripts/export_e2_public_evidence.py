"""Export an explicit scientific JSON view of completed development work.

Only named fields enter this public view. Raw receipts, paths, commands, machine
metadata, environments and journals remain in their original private files.
This exporter performs no fit or simulation and never modifies its inputs.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re


VERSION = "e2-scientific-public-v1"
REG_SHA = "3b54ac44f5aed956ac98fbc31c95dfcb70c1116b6cca65a8269ff11cd843ac91"
REG_COMMIT = "d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2"
MODELS = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "CNN-1D + LSTM + atención temporal", "MLP16", "LSTM32", "GRU32")
STAGES = ("ae1", "ae2", "supervised")
TIME_FIELDS = ("build_seconds", "fit_seconds", "validation_seconds", "build_cpu_seconds",
               "fit_cpu_seconds", "validation_cpu_seconds", "total_work_seconds", "total_cpu_seconds",
               "elapsed_worker_seconds", "worker_peak_rss_bytes", "fit_management_seconds")
STAGE_TIMES = ("completed", "batch_calls", "train_evaluation_seconds", "train_evaluation_cpu_seconds",
               "checkpoint_clone_seconds", "checkpoint_clone_cpu_seconds", "state_hash_seconds",
               "state_hash_cpu_seconds", "total_seconds", "total_cpu_seconds", "setup_seconds",
               "setup_cpu_seconds", "loop_seconds", "loop_cpu_seconds", "finalization_seconds",
               "finalization_cpu_seconds", "update_loop_seconds", "update_loop_cpu_seconds")
COMPONENTS = ("hpo", "nas_phase1", "nas_phase2_conditional_finalists", "selected_refits_conditional_architecture")
FORBIDDEN = re.compile(r"/workspace(?:/|\b)|/home(?:/|\b)|/tmp(?:/|\b)|sediment:|"
                       r"ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|"
                       r"authorization\s*:|bearer\s+|-----BEGIN [A-Z ]*PRIVATE KEY|"
                       r"(?:api[_-]?key|access[_-]?token|password)\s*[:=]", re.I)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON input")
    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def audit_public(value):
    """Secondary guard; explicit field constructors below provide the whitelist."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or FORBIDDEN.search(key):
                raise ValueError("Unsafe public JSON key")
            audit_public(child)
    elif isinstance(value, list):
        for child in value:
            audit_public(child)
    elif isinstance(value, str):
        if FORBIDDEN.search(value):
            raise ValueError("Private path or credential marker in public JSON")
    elif value is not None and type(value) not in (int, float, bool):
        raise ValueError("Public JSON contains an unsupported value")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite public number")


def numeric(value):
    if value is not None and (type(value) not in (int, float, bool)
                              or isinstance(value, float) and not math.isfinite(value)):
        raise ValueError("Expected a finite scientific number, boolean or null")
    return value


def numbers(source, names):
    if source is None:
        return {}
    return {name: numeric(source[name]) for name in names if name in source}


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_:+-]+", value):
        raise ValueError("Invalid scientific category")
    return value


def model(value):
    if value not in MODELS:
        raise ValueError("Model outside the scientific registry")
    return value


def digest_value(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
        raise ValueError("Expected a SHA256 digest")
    return value


def day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T00:00:00(?:\+00:00|Z)?)?", value):
        raise ValueError("Expected a daily scientific date")
    result = value[:10]
    date.fromisoformat(result)
    return result


def pattern(value):
    if value is None:
        return None
    identity = value["pattern_id"]
    if not re.fullmatch(r"w\d+-\d+-\d+:b[01]{3}", identity):
        raise ValueError("Invalid architecture pattern identity")
    return {"pattern_id": identity, "widths": [numeric(v) for v in value["widths"]],
            "dropout_pattern": [numeric(v) for v in value["dropout_pattern"]],
            **numbers(value, ("dropout_rate", "is_reference"))}


def stage_receipt(value):
    result = numbers(value, ("best_epoch", "epochs_completed", "optimizer_steps", "elapsed_seconds",
                              "optimizer_initial_state_entries", "fixed_epochs", "scheduler_total_steps",
                              "max_observed_rss_bytes", "seed"))
    if "stop_reason" in value:
        result["stop_reason"] = token(value["stop_reason"])
    for name in ("initial_state_sha256", "selected_state_sha256"):
        if name in value:
            result[name] = digest_value(value[name])
    return result


def profile_attempt(row):
    job, worker = row["job"], row.get("worker_profile") or {}
    result = {"job_id": token(job["job_id"]), "phase": token(job["phase"]),
              "model": model(job["model"]), "cohort": token(job["cohort"]),
              "pattern": pattern(job.get("pattern")), "status": token(row["status"]),
              "error_code": token(worker["error_code"]) if worker.get("error_code") else None,
              **numbers(job, ("seed", "epochs", "batch_windows")),
              **numbers(row, ("optimizer_steps", "optimizer_steps_censored", "elapsed_wall_seconds",
                              "sampled_peak_process_group_rss_bytes", "replacement_attempt",
                              "separate_micro_profile_not_replacement")),
              "runtime": {"threads": numeric(job["threads"]), "interop_threads": 1,
                          "backend": token(job["backend"]), "device": "cpu"},
              "measurements": numbers(worker, TIME_FIELDS+("train_n", "validation_n", "parameter_count", "state_bytes")),
              "stage_timings": {name: numbers(worker.get("stage_timing", {}).get(name, {}), STAGE_TIMES)
                                for name in STAGES if name in worker.get("stage_timing", {})},
              "stages": {name: stage_receipt(worker.get("phase_receipts", {}).get(name, {}))
                         for name in STAGES if name in worker.get("phase_receipts", {})}}
    if "parent_job_id" in job:
        result["parent_job_id"] = token(job["parent_job_id"])
    return result


def costs(rows, complete):
    known = sum(numeric(row["optimizer_steps"]) for row in rows if row.get("optimizer_steps") is not None)
    unknown = sum(row.get("optimizer_steps") is None for row in rows)
    return {"attempts": len(rows), "completed_attempts": sum(row["status"] == complete for row in rows),
            "known_optimizer_steps": known, "unknown_optimizer_step_workers": unknown,
            "total_optimizer_steps": None if unknown else known,
            "elapsed_workers_sum_seconds": sum(numeric(row["elapsed_wall_seconds"]) for row in rows)}


def profiles_view(phases):
    primary, micro = [], []
    for phase, value in phases.items():
        expected = {"kernels": 8, "threads": 12, "scaling": 86}[phase]
        rows = value["results"]
        if len(rows) != expected or len({row["job"]["job_id"] for row in rows}) != expected:
            raise ValueError("A registered profile attempt is absent or duplicated")
        primary.extend(profile_attempt(row) for row in rows)
        micro.extend(profile_attempt(row) for row in value.get("micro_fallback_results", []))
    return {"version": VERSION, "primary": primary, "micro": micro,
            "primary_costs": costs(primary, "success"), "micro_costs": costs(micro, "success"),
            "scientific_HPO_or_NAS_executed": False, "final_market_values_read": False}


def variance_view(value):
    attempts, forecasts = [], []
    for name in ("TimeMixer", "CNN-1D + LSTM + atención temporal", "GRU32"):
        for phase in ("selection", "refit"):
            rows = value["models"][name][phase]
            for row in rows:
                worker = row.get("worker_profile") or {}
                result = {"model": model(name), "phase": token(row["phase"]),
                          "status": token(row["status"]), "seed": numeric(row["seed"]),
                          **numbers(row, ("optimizer_steps", "best_epoch", "elapsed_wall_seconds",
                                          "sampled_peak_process_group_rss_bytes", "replacement_attempt")),
                          "fixed_epoch_count": numeric(worker.get("fixed_epoch_count")),
                          "stage": stage_receipt(worker.get("stage_record", {})),
                          "runtime": numbers(worker, ("torch_num_threads", "torch_num_interop_threads")),
                          "checkpoint": {key: digest_value(worker.get("checkpoint", {}).get(key))
                                         for key in ("sha256", "state_sha256")},
                          "replay_sha256": digest_value(worker.get("replay_ledger_sha256")),
                          "metrics": numbers(worker.get("metrics_original_units", {}), ("n", "RMSE", "MAE", "MASE"))}
                for split in ("training_split", "validation_split"):
                    source = worker.get(split)
                    dates = source.get("dates", []) if isinstance(source, dict) else []
                    result[split] = {"n": len(dates), "start": day(dates[0]), "end": day(dates[-1])} if dates else None
                attempts.append(result)
                evaluation = worker.get("evaluation")
                if phase == "refit" and isinstance(evaluation, dict):
                    dates = [day(v) for v in evaluation["dates"]]
                    predictions = [numeric(v) for v in evaluation["predictions"]]
                    truth = [numeric(v) for v in evaluation["truths"]]
                    if len(dates) != 53 or len(predictions) != 53 or len(truth) != 53:
                        raise ValueError("Expected the complete 53-date development forecast panel")
                    forecasts.append({"model": model(name), "seed": numeric(row["seed"]),
                                      "dates": dates, "predictions": predictions, "truth": truth,
                                      "checkpoint_sha256": result["checkpoint"]["sha256"],
                                      "replay_sha256": result["replay_sha256"]})
    if len(attempts) != 30 or len(forecasts) != 15:
        raise ValueError("The complete three-model five-seed development panel is required")
    return {"version": VERSION, "attempts": attempts, "forecasts": forecasts,
            "costs": costs(attempts, "completed"),
            "fixed_epoch_counts": {model(k): numeric(v) for k, v in value["fixed_epoch_counts"].items() if k in MODELS},
            "target": "Population standard deviation of seven daily log returns, ddof0, unannualized",
            "final_R": None, "achieved_final_power": None, "scientific_HPO_or_NAS_executed": False}


def calibration_cell(value):
    output = numbers(value, ("R", "calibration_noise_spread", "null_validated", "supports_conditional_target",
                             "minimum_direction_CP_lower"))
    output["status"] = token(value["status"])
    output["calibration"] = numbers(value.get("calibration", {}),
        ("critical_abs", "draws", "tail_target", "gamma", "order_rank_1based", "numpy_order_index",
         "tail_order_k", "allowed_exceedances_in_calibration", "binomial_failure_probability_bound"))
    output["validation"] = numbers(value.get("validation", {}),
        ("draws", "hits", "rejection_frequency", "CP_upper", "gamma", "nominal_alpha", "validated", "critical_value_retuned"))
    output["alternatives"] = {direction: numbers(value.get("alternatives", {}).get(direction, {}),
        ("effect_shift", "draws", "hits", "conditional_rejection_frequency", "CP_lower", "gamma",
         "interpretation_admitted_by_null_validation")) for direction in ("plus_delta", "minus_delta")}
    output["descriptive_MDE"] = numbers(value.get("descriptive_MDE", {}),
        ("plus_effect", "minus_effect_magnitude", "null_validation_required_for_interpretation", "scientific_or_distribution_free_MDE"))
    output["draws_executed"] = numbers(value.get("draws_executed", {}), ("calibration", "validation", "alternative"))
    output["streams"] = {role: {"pool": numeric(value.get("streams", {}).get(role, {}).get("pool")),
        "calendar_entropy": [numeric(v) for v in value.get("streams", {}).get(role, {}).get("calendar_entropy", [])],
        "seed_entropy": [numeric(v) for v in value.get("streams", {}).get(role, {}).get("seed_entropy", [])]}
        for role in ("calibration", "validation", "alternative")}
    for name in ("calibration_noise_sha256", "validation_noise_sha256", "alternative_noise_sha256"):
        output[name] = digest_value(value.get(name))
    return output


def calibration_view(value):
    groups = {}
    for label in ("old", "new"):
        source = value["groups"][label]
        analysis = source.get("analysis") or {}
        pairs = []
        for pair in analysis.get("pairs", []):
            pairs.append({"models": [model(v) for v in pair["models"]],
                          "paired_seed_differences": [numeric(v) for v in pair["paired_seed_differences"]],
                          **numbers(pair, ("sigma_d", "original_mean_difference", "sigma_d_upper95_normal_assumption",
                                           "seed_residual_inflation")),
                          "status": token(pair["status"]), "cells": [calibration_cell(v) for v in pair["cells"]]})
        groups[label] = {"status": token(source["status"]), "model_names": [model(v) for v in source["model_names"]],
                         "seed_ids": [numeric(v) for v in source["seed_ids"]],
                         "dates": [day(v) for v in source["dates"]], "pairs": pairs,
                         "config": {**numbers(analysis.get("config", {}), ("calibration_draws", "validation_draws",
                             "alternative_draws", "alpha", "power_target", "family_mc_error", "calibration_pool",
                             "validation_pool", "alternative_pool", "block_length", "batch_size")),
                             "R_values": [numeric(v) for v in analysis.get("config", {}).get("R_values", [])]},
                         **numbers(analysis, ("delta", "multiplicity_family_size", "nominal_alpha_per_comparison",
                             "calibration_total_absolute_tail_target", "gamma_calibration_per_cell", "gamma_validation_per_cell",
                             "gamma_power_per_cell_direction", "observed_pair_count", "cells_per_MC_family",
                             "calibration_MC_family_confidence", "validation_MC_family_confidence", "power_MC_family_confidence",
                             "joint_three_family_confidence_claimed", "conditionally_supported_R", "resource_envelope_R")),
                         "final_R": None, "achieved_final_power": None}
    return {"version": VERSION, "groups": groups, "groups_pooled": False,
            "delta": numeric(value["delta"]), "final_R": None, "achieved_final_power": None,
            "interpretation": "Separate conditional empirical-DGP Monte Carlo families; no final-study power claim",
            "critical_rule": "Strict absolute statistic greater than independent-null critical value"}


def model_costs(source):
    return {name: {"components_seconds": numbers(source[name].get("components_seconds", {}), COMPONENTS),
                   **numbers(source[name], ("nas", "conditional_compute_seconds_excluding_disk"))}
            for name in MODELS if name in source}


def budget_view(value):
    scenarios = []
    identities = [f"k40_R{R}_{epochs}_{batch}" for epochs in ("max50_200", "lower25_100")
                  for R in (6, 10, 20, 30) for batch in ("constant_step", "linear_batch")]
    for identity in identities:
        source = value["scenario_results"][identity]
        micro = source.get("separate_micro_sensitivity_not_observed_total", {})
        scenarios.append({"scenario_id": identity, "status": token(source["status"]),
                          **numbers(source, ("k", "R_resource_scenario", "nas_selected_batch_assumed",
                              "core_conditional_compute_seconds_excluding_disk", "core_conditional_compute_hours_excluding_disk",
                              "known_family_subtotal_seconds_not_total", "known_family_subtotal_hours_not_total", "full_campaign_wall_hours")),
                          "batch_cost_assumption": token(source["batch_cost_assumption"]),
                          "epoch_scenario": {"name": token(source["epoch_scenario"]["name"]),
                              **numbers(source["epoch_scenario"], ("short_epochs", "full_epochs", "lower_scenario_is_counterfactual_only"))},
                          "models": model_costs(source["models"]),
                          "allocation_caps_not_forecast": numbers(source.get("allocation_caps_not_forecast", {}),
                              ("short_stages", "full_stages", "serial_stage_allocation_cap_seconds", "serial_stage_allocation_cap_hours",
                               "is_runtime_estimate", "classical_cost_included", "does_not_guarantee_completion_before_timeout")),
                          "separate_micro_sensitivity_not_observed_total": {
                              "status": token(micro["status"]), "models": model_costs(micro.get("models", {})),
                              **numbers(micro, ("core_conditional_compute_seconds_excluding_disk", "core_conditional_compute_hours_excluding_disk",
                                  "known_family_subtotal_hours_not_total", "full_campaign_wall_hours", "primary_censoring_not_resolved",
                                  "full_cohort_fit_completion_established", "weighted_surrogate_costs_not_unbiased_full_size_HT_estimate",
                                  "primary_results_modified"))}})
    if len(scenarios) != 16:
        raise ValueError("All sixteen resource scenarios must be retained")
    origins = [{"origin": day(v["origin"]), **numbers(v, ("inner_train_n", "validation_n", "refit_train_n")),
                **{name: day(v[name]) for name in ("inner_train_start", "inner_train_end", "validation_start",
                                                   "validation_end", "refit_train_end", "inner_cutoff", "refit_cutoff") if name in v}}
               for v in value["panels"]["origins"]]
    storage = {}
    for R in (6, 10, 20, 30):
        source = value["storage_k40_scenarios"]["k40_R"+str(R)]
        storage["k40_R"+str(R)] = {
            "models": {name: numbers(source["models"][name],
                ("hpo_model_states", "nas_phase1_model_states", "nas_phase2_min", "nas_phase2_max",
                 "selected_refits_min", "selected_refits_max")) for name in MODELS if name in source["models"]},
            "components_payload_bytes": numbers(source.get("components_payload_bytes", {}),
                ("hpo_model_states", "nas_phase1_model_states", "nas_phase2_min", "nas_phase2_max", "selected_refits_min", "selected_refits_max")),
            "retain_all_completed_stage_checkpoints_payload_bytes_range": [numeric(v) for v in source["retain_all_completed_stage_checkpoints_payload_bytes_range"]],
            "retain_only_selected_refits_payload_bytes_range": [numeric(v) for v in source["retain_only_selected_refits_payload_bytes_range"]],
            **numbers(source, ("payload_range_is_not_serialized_disk_bound", "optimizer_states_included", "deletion_executed_or_authorized"))}
    return {"version": VERSION, "operative_k": numeric(value["operative_k"]), "scenarios": scenarios,
            "origins": origins, "storage_k40_scenarios": storage,
            "stage_wall_allocation_caps_seconds": {"short": 1800, "full": 7200},
            "every_projected_stage_checked_against_wall_cap": False,
            "reference_cohort_scaling_check": [{"model": model(v["model"]),
                **numbers(v, ("small_train_n", "large_train_n", "fixed_epochs", "observed_small_build_and_fit_seconds",
                              "projected_from_large_seconds", "observed_over_projected_ratio", "not_a_bound_or_full_campaign_validation"))}
                for v in value["reference_cohort_scaling_check"]],
            "sampling_frame": [{**pattern(v), "role": token(v["role"]),
                                **numbers(v, ("stratum_index", "stratum_size", "inclusion_probability", "expansion_weight"))}
                               for v in value["sampling_frame"]],
            "assumptions": ["Serial CPU execution; stationary per-step rates are conditional assumptions.",
                "Three exact anchors and nine weighted strata; one sample per stratum gives no design-based confidence interval.",
                "Batch and early-stop epoch assumptions are sensitivity cases, not validated bounds.",
                "Finalist and selected-refit architecture costs use the uniform population mean rather than an observed winner.",
                "Unmeasured structural HPO choices can change cost and storage beyond these scenarios.",
                "Two-epoch throughput extrapolation does not validate the 50 or 200 epoch horizons or compliance of every stage with its wall cap.",
                "Microprofile extrapolation does not establish completion of the failed full cohort.",
                "Storage is prospective model payload; serialization and optimizer overhead are excluded.",
                "Classical fits and additional scientific analyses are excluded from the conditional core calculation."],
            "final_R": None, "full_campaign_wall_hours": None, "meta_inventory_published": False}


def registration_view(value):
    return {"version": VERSION, "original_registration_sha256": REG_SHA,
            "original_registration_commit": REG_COMMIT,
            "original_registration_url": "https://github.com/jassan-bit/DeepLearning_202630/blob/"+REG_COMMIT+
                "/experiments/e2_compute_optimization_20261109/registration.json",
            "hardware": numbers(value["hardware"], ("cpu_quota_vcpus", "memory_bytes", "gpu_visible", "parallel_fits", "interop_threads")),
            "profile_design": numbers(value["cpu_profile"], ("kernel_epochs", "kernel_jobs", "thread_epochs", "thread_jobs", "scaling_epochs", "scaling_jobs")),
            "development_cutoff": "2024-12-30", "target_ddof": 0, "target_logreturn_days": 7,
            "scientific_HPO_or_NAS_executed": False, "final_R": None, "resource_ceiling_R": 30}


def export_public(*, kernels, threads, scaling, variance, calibration, budget, registration, output):
    paths = {"kernels": kernels, "threads": threads, "scaling": scaling,
             "variance": variance, "calibration": calibration, "budget": budget, "registration": registration}
    values = {name: load(path) for name, path in paths.items()}
    if sha(Path(registration).read_bytes()) != REG_SHA:
        raise ValueError("Use the exact original scientific registration")
    profiles = profiles_view({name: values[name] for name in ("kernels", "threads", "scaling")})
    variance_result = variance_view(values["variance"])
    documents = {"profiles.json": profiles, "variance.json": variance_result,
                 "calibration.json": calibration_view(values["calibration"]),
                 "budget.json": budget_view(values["budget"]),
                 "registration_scientific.json": registration_view(values["registration"])}
    documents["summary.json"] = {"version": VERSION, "profiles_primary": profiles["primary_costs"],
        "profiles_micro": profiles["micro_costs"], "variance": variance_result["costs"],
        "public_forecast_panels": len(variance_result["forecasts"]), "forecast_days_per_panel": 53,
        "budget_scenarios": len(documents["budget.json"]["scenarios"]), "final_R": None,
        "achieved_final_power": None, "weights_published": False, "final_market_values_read": False,
        "scientific_HPO_or_NAS_executed": False,
        "selected_runtime": {"threads": numeric(values["scaling"]["results"][0]["job"]["threads"]),
                             "backend": token(values["scaling"]["results"][0]["job"]["backend"]), "device": "cpu"},
        "publication_scope": "Explicit scientific fields only; full private records remain unchanged.",
        "excluded": ["Absolute machine paths", "Commands and environment metadata", "Raw messages and logs",
                     "Journal contents and private receipts", "Checkpoint weights", "Full parameter inventory"]}
    encoded = {}
    for name, value in documents.items():
        audit_public(value)
        encoded[name] = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    manifest = {"version": VERSION,
        "artifacts": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in encoded.items()},
        "private_source_report_sha256": {name: sha(Path(path).read_bytes()) for name, path in paths.items()},
        "private_source_reports_published": False, "compression_or_binary_encoding_used": False}
    audit_public(manifest)
    encoded["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode()
    output = Path(output)
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise FileExistsError("Public JSON output must be a new ordinary directory")
    output.mkdir(parents=True)
    for name, raw in encoded.items():
        (output/name).write_bytes(raw)
    return {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in encoded.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kernels", "threads", "scaling", "variance", "calibration", "budget", "registration", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    result = export_public(**vars(parser.parse_args(argv)))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
