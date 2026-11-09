"""Bounded CUDA resource pilot on the hash-pinned development prefix.

This runner never launches HPO/NAS selection. It admits CUDA before opening
market files, seals a finite subset of the published pilot domain, and runs
fresh workers sequentially. Raw receipts stay local. The separate public JSON
contains only explicitly selected scientific fields, never process records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from profile_e2_training import run_isolated_profile
from run_e2_development_pilot import DurableEvents, read_events, write_json_new

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "book_e2/_static/search_protocol_20261109/protocol.json"
PROTOCOL_SHA256 = "e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe"
VERSION = "e2-gpu-pilot-v1"
REGISTRATION_VERSION = "e2-gpu-pilot-registration-v1"
NAS_NAMES = ("LSTM-autoencoder con predictor", "TimesNet", "TiDE", "TimeMixer",
             "PatchTST", "CNN-1D + LSTM + atención temporal")
FIXED_NAMES = ("FCN-1D", "ResNet-1D", "Inception-1D", "TSMixer", "iTransformer",
               "MLP16", "LSTM32", "GRU32")
MODEL_NAMES = NAS_NAMES + FIXED_NAMES
RESOURCE_CONTROLS = {"parallel_fits": 1, "cpu_threads": 2, "interop_threads": 1,
                     "host_rss_limit_bytes": 12*1024**3, "gpu_memory_fraction": .8,
                     "gpu_memory_cap_bytes": 12*1024**3,
                     "internal_wall_seconds": 300, "external_wall_seconds": 360,
                     "campaign_wall_seconds": 900, "minimum_worker_start_seconds": 5}
EXECUTION_CONTROLS = {"device": "cuda", "dtype": "float32", "amp": False,
                      "tf32": False, "deterministic": True,
                      "cublas_workspace_config": ":4096:8", "cudnn_benchmark": False,
                      "cpu_fallback": False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _hex(value, n):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{"+str(n)+"}", value) is not None


def require_cuda_environment(device="cuda:0"):
    """Admit an actual CUDA device; never select CPU as a fallback."""
    mode = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if mode != ":4096:8":
        raise RuntimeError("Set the published deterministic cuBLAS workspace before CUDA initialization")
    import torch
    from e2_cuda_training import validate_cuda_device
    selected = validate_cuda_device(device)
    if torch.version.cuda is None:
        raise RuntimeError("The published pilot requires a CUDA-built PyTorch wheel")
    properties = torch.cuda.get_device_properties(selected)
    # Driver visibility alone does not establish that this wheel can execute
    # kernels on the selected accelerator. No market data are involved here.
    probe = torch.tensor([1., 2.], dtype=torch.float32, device=selected).square().sum()
    torch.cuda.synchronize(selected)
    if float(probe.item()) != 5.:
        raise RuntimeError("The CUDA kernel preflight did not produce its exact synthetic result")
    del probe
    torch.cuda.empty_cache()
    total = int(properties.total_memory)
    limit = min(RESOURCE_CONTROLS["gpu_memory_cap_bytes"],
                int(RESOURCE_CONTROLS["gpu_memory_fraction"]*total))
    if limit < 256*1024**2:
        raise RuntimeError("The admitted CUDA memory allowance is too small for this pilot")
    return {"device": str(selected), "device_name": str(properties.name),
            "total_memory_bytes": total, "effective_memory_limit_bytes": limit,
            "torch": str(torch.__version__), "torch_cuda": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(), "cuda_available": True,
            "cuda_kernel_probe_passed": True,
            "dtype": "float32", "amp": False, "tf32": False,
            "deterministic": True, "cpu_threads": 2, "interop_threads": 1,
            "execution_policy_scope": "Requested registered settings, enforced by the backend during fit and inference; the preflight only validates device/kernel availability",
            "memory_limit_scope": "PyTorch allocator fraction and sampled peak allocated; not all driver allocations"}


def load_registration(path, expected_sha256, commit):
    """Require exact published registration bytes and immutable controls."""
    path = Path(path)
    if not _hex(expected_sha256, 64) or not _hex(commit, 40):
        raise ValueError("Pass an explicit registration SHA256 and full published commit SHA")
    if path.is_symlink() or not path.is_file() or digest(path) != expected_sha256:
        raise ValueError("GPU registration bytes differ from the explicit published SHA256")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("version") != REGISTRATION_VERSION or
            value.get("status") != "registered_before_gpu_market_fits"):
        raise ValueError("A published GPU pilot registration is required")
    if value.get("source_protocol_sha256") != PROTOCOL_SHA256 or digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("The on-disk development protocol is not the exact registered snapshot")
    profile = value.get("profile", {})
    required = {"models": list(MODEL_NAMES), "cohorts": ["small", "large"],
                "default_cohorts": ["small"], "epochs": 2, "seed": 274201,
                "optional_largest_timesnet": True}
    if any(profile.get(key) != expected for key, expected in required.items()):
        raise ValueError("GPU pilot domain or recipe differs from the registered controls")
    resources, execution = value.get("resources", {}), value.get("execution", {})
    if any(resources.get(k) != v for k, v in RESOURCE_CONTROLS.items()):
        raise ValueError("GPU resource controls differ from registration")
    if any(execution.get(k) != v for k, v in EXECUTION_CONTROLS.items()):
        raise ValueError("GPU execution precision or deterministic controls changed")
    protocol = json.loads(PROTOCOL.read_text())
    if value.get("fixed_recipe") != protocol["budget"]["noise_and_resource_pilot_fixed_nonarchitecture"]:
        raise ValueError("The fixed preselection recipe must not use known HPO winners")
    cohorts = value.get("cohorts", {})
    expected_cohorts = {"small": {"train_n": 161, "validation_n": 83,
                                  "train_end": "2023-06-25", "validation_start": "2023-07-03",
                                  "validation_end": "2023-09-23"},
                        "large": {"train_n": 619, "validation_n": 83,
                                  "train_end": "2024-09-25", "validation_start": "2024-10-03",
                                  "validation_end": "2024-12-24",
                                  "origin_calendar_placeholder": "2025-01-01",
                                  "numeric_2025_values_read": False}}
    if (set(cohorts) != set(expected_cohorts) or any(
            not isinstance(cohorts[name], dict)
            or cohorts[name].get("train_n") != fields["train_n"]
            or cohorts[name].get("validation_n") != fields["validation_n"]
            or any(key in cohorts[name] and cohorts[name][key] != expected for key, expected in fields.items())
            for name, fields in expected_cohorts.items())):
        raise ValueError("Only the two registered development cohorts are admitted")
    return value


def _selection(requested, allowed, label):
    values = list(allowed if requested is None else requested)
    if (not values or any(not isinstance(v, str) for v in values)
            or len(set(values)) != len(values) or any(v not in allowed for v in values)):
        raise ValueError("Choose a nonempty distinct subset of registered "+label)
    return [name for name in allowed if name in values]


def build_jobs(registration, models=None, cohorts=None, include_largest=False):
    """Seal at most thirty jobs; defaults are fourteen small reference bodies."""
    if type(include_largest) is not bool:
        raise ValueError("Largest-case opt-in must be an explicit boolean")
    profile = registration["profile"]
    names = _selection(models, list(MODEL_NAMES), "models")
    selected_cohorts = _selection(profile["default_cohorts"] if cohorts is None else cohorts,
                                  ["small", "large"], "cohorts")
    if include_largest and "TimesNet" not in names:
        raise ValueError("Largest TimesNet requires TimesNet in the explicit model subset")
    protocol = json.loads(PROTOCOL.read_text())
    import e2_search_plan as planner
    reference = next(row for row in planner.noise_pilot_architectures(protocol) if row["is_reference"])
    largest = next(row for row in planner.noise_pilot_architectures(protocol)
                   if row["widths"] == [256]*3 and row["dropout_pattern"] == [1]*3)
    jobs = []
    for cohort in selected_cohorts:
        for name in names:
            jobs.append({"model": name, "cohort": cohort, "case": "reference",
                         "pattern": deepcopy(reference) if name in NAS_NAMES else None,
                         "seed": profile["seed"], "epochs": profile["epochs"],
                         "batch_size": 32, "stage_count": 3 if name == NAS_NAMES[0] else 1})
        if include_largest:
            jobs.append({"model": "TimesNet", "cohort": cohort, "case": "largest",
                         "pattern": deepcopy(largest), "seed": profile["seed"],
                         "epochs": profile["epochs"], "batch_size": 32, "stage_count": 1})
    for i, job in enumerate(jobs):
        job["job_id"] = "gpu-profile-%03d" % i
    if not 1 <= len(jobs) <= 30:
        raise ValueError("A resource pilot must stay inside the finite registered domain")
    return jobs


def prepare_cohort(data_root, cohort):
    # Reuse the already tested chronological development admission unchanged.
    from run_e2_cpu_scaling import prepare_cohort as prepare
    return prepare(data_root, cohort)


def _sources():
    paths = list((ROOT/"scripts").rglob("*.py")) + [PROTOCOL]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths) if path.is_file()}


def _error_code(error):
    import torch
    if isinstance(error, (TimeoutError, MemoryError, torch.OutOfMemoryError)):
        return "resource_limit"
    receipt = getattr(error, "receipt", None)
    if isinstance(receipt, dict) and isinstance(receipt.get("error_code"), str):
        return receipt["error_code"]
    return getattr(error, "error_code", "runtime_error")


def _partial_steps(receipt):
    if not isinstance(receipt, dict):
        return None
    for key in ("optimizer_steps", "optimizer_steps_known"):
        if type(receipt.get(key)) is int and receipt[key] >= 0:
            return receipt[key]
    return None


def _failure_steps(phases, receipt, *, fit_entered, fit_completed):
    """Separate exact counts from completed-update lower bounds after errors."""
    completed = sum(p.optimizer_steps for p in phases.values())
    if not fit_entered or fit_completed:
        return {"optimizer_steps": completed, "optimizer_steps_known_lower_bound": completed,
                "partial_optimizer_update_may_have_occurred": False}
    if not isinstance(receipt, dict):
        return {"optimizer_steps": None, "optimizer_steps_known_lower_bound": completed,
                "partial_optimizer_update_may_have_occurred": True}
    prior = receipt.get("completed_phases", {})
    previous = sum(p.get("optimizer_steps", 0) for p in prior.values() if isinstance(p, dict))
    current = receipt.get("completed_optimizer_steps", _partial_steps(receipt))
    if type(current) is not int or current < 0:
        current = None
    all_stages = receipt.get("known_optimizer_steps_all_sae_phases")
    lower = all_stages if type(all_stages) is int and all_stages >= 0 else previous+(current or 0)
    uncertain = bool(receipt.get("partial_optimizer_update_may_have_occurred", current is None))
    return {"optimizer_steps": None if uncertain or current is None else lower,
            "optimizer_steps_known_lower_bound": lower,
            "partial_optimizer_update_may_have_occurred": uncertain}


def _validate_worker_spec(spec, registration):
    if spec.get("version") != VERSION or not isinstance(spec.get("job"), dict):
        raise ValueError("Malformed CUDA worker specification")
    job = spec["job"]
    candidates = build_jobs(registration, spec["models"], spec["cohorts"], spec["include_largest"])
    if job not in candidates:
        raise ValueError("CUDA worker is outside the sealed model/cohort domain")
    if spec.get("source_sha256_expected") != _sources():
        raise ValueError("Repository sources changed after the campaign plan was sealed")
    return job


def run_worker(spec):
    output = Path(spec["output"])
    events = DurableEvents(output/"events.jsonl")
    started, cpu_started = time.perf_counter(), time.process_time()
    phases, data_receipt, before, hardware = {}, None, None, None
    fit_entered, fit_completed = False, False
    job = spec["job"]
    events.append("worker_started", job=job)
    try:
        hardware = require_cuda_environment(spec["device"])
        import torch
        from e2_cuda_training import fit_supervised, fit_sae, predict_scaled
        from e2_development_models import build_pilot_model, fixed_pilot_config
        from e2_nas_common import NASConfig
        from e2_staged_training import state_sha256
        registration = load_registration(spec["registration"], spec["registration_sha256"],
                                         spec["registration_commit"])
        _validate_worker_spec(spec, registration)
        torch.set_num_threads(RESOURCE_CONTROLS["cpu_threads"])
        torch.set_num_interop_threads(RESOURCE_CONTROLS["interop_threads"])
        device = torch.device(hardware["device"])
        torch.cuda.set_per_process_memory_fraction(
            hardware["effective_memory_limit_bytes"]/hardware["total_memory_bytes"], device)
        torch.cuda.reset_peak_memory_stats(device)
        before = _sources()
        prepared, data_receipt = prepare_cohort(spec["data_root"], job["cohort"])
        train, valid = prepared.inner_train.clone(), prepared.inner_validation.clone()
        events.append("development_cohort_sealed", cohort=job["cohort"],
                      train_n=len(train.dates), validation_n=len(valid.dates),
                      source_sha256=before, provenance=data_receipt)
        config = fixed_pilot_config(job["seed"], epoch_cap=job["epochs"],
                    wallcap=RESOURCE_CONTROLS["internal_wall_seconds"],
                    memory_limit_bytes=RESOURCE_CONTROLS["host_rss_limit_bytes"])
        source_roots = spec.get("source_roots")
        torch.cuda.synchronize(device)
        at = time.perf_counter()
        model = build_pilot_model(job["model"], job["pattern"], seed=job["seed"], source_roots=source_roots)
        metadata = deepcopy(model.architecture_metadata)
        # SAE's three-stage manager constructs the actual trainable modules
        # inside fit_sae. Do not transfer this count/metadata-only body to CUDA.
        if job["model"] != NAS_NAMES[0]:
            model = model.to(device)
        torch.cuda.synchronize(device)
        build_seconds = time.perf_counter()-at
        parameter_count = sum(p.numel() for p in model.parameters())
        events.append("fixed_epoch_fit_started", parameter_count=parameter_count,
                      fixed_epochs=job["epochs"], optimizer_steps=None)
        torch.cuda.synchronize(device)
        at = time.perf_counter()
        common = {"device": str(device), "gpu_memory_limit_bytes": hardware["effective_memory_limit_bytes"],
                  "deterministic": True}
        fit_entered = True
        if job["model"] == NAS_NAMES[0]:
            del model
            result = fit_sae(NASConfig(job["pattern"]["widths"], job["pattern"]["dropout_pattern"]),
                train.x, train.y, None, None, train.scaler.target_scale, config,
                fold_id="gpu-resource-"+prepared.signature,
                fixed_epochs={p: job["epochs"] for p in ("ae1", "ae2", "supervised")}, **common)
            model, phases = result.model, result.phases
        else:
            stage = fit_supervised(model, train.x, train.y, None, None,
                                   train.scaler.target_scale, config,
                                   fixed_epochs=job["epochs"], **common)
            phases = {"supervised": stage}
        fit_completed = True
        torch.cuda.synchronize(device)
        fit_seconds = time.perf_counter()-at
        if any(p.epochs_completed != job["epochs"] or p.fixed_epochs != job["epochs"]
               or p.optimizer_initial_state_entries != 0 for p in phases.values()):
            raise ValueError("Fresh-optimizer fixed-epoch contract changed")
        # Validation is timed only after fixed fitting; it never chooses a model.
        torch.cuda.synchronize(device)
        at = time.perf_counter()
        prediction = predict_scaled(model, valid.x, device=str(device), batch_size=32)
        if prediction.shape != valid.y.shape or not torch.isfinite(prediction).all():
            raise ValueError("CUDA validation produced invalid predictions")
        torch.cuda.synchronize(device)
        validation_seconds = time.perf_counter()-at
        if state_sha256(model) != phases["supervised"].selected_state_sha256:
            raise ValueError("Validation changed the selected model state")
        after = _sources()
        if before != after:
            raise ValueError("A source file changed during the CUDA fit")
        worker = {"version": VERSION, "status": "completed", "job": job,
                  "registration_commit": spec["registration_commit"],
                  "registration_sha256": spec["registration_sha256"],
                  "source_sha256_before": before, "source_sha256_after": after,
                  "source_hashes_unchanged": True, "hardware": hardware,
                  "cohort_provenance": data_receipt, "train_n": len(train.dates),
                  "validation_n": len(valid.dates), "parameter_count": parameter_count,
                  "state_bytes": sum(t.numel()*t.element_size() for t in model.state_dict().values()),
                  "phase_receipts": {name: phase.to_dict() for name, phase in phases.items()},
                  "optimizer_steps": sum(p.optimizer_steps for p in phases.values()),
                  "optimizer_steps_known_lower_bound": sum(p.optimizer_steps for p in phases.values()),
                  "partial_optimizer_update_may_have_occurred": False,
                  "timing": {"build_seconds": build_seconds, "fit_seconds": fit_seconds,
                             "validation_seconds": validation_seconds,
                             "worker_seconds": time.perf_counter()-started,
                             "process_cpu_seconds": time.process_time()-cpu_started},
                  "cuda_metrics": {"max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                      "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                      "effective_limit_bytes": hardware["effective_memory_limit_bytes"]},
                  "model_metadata": metadata, "accuracy_used_for_selection": False,
                  "validation_labels_used_for_stopping": False,
                  "weights_saved": False, "final_market_values_read": False,
                  "scientific_HPO_or_NAS_executed": False, "final_R": None}
        write_json_new(output/"worker_profile.json", worker)
        events.append("worker_completed", optimizer_steps=worker["optimizer_steps"])
        return 0
    except Exception as error:
        receipt = getattr(error, "receipt", None)
        steps = _failure_steps(phases, receipt, fit_entered=fit_entered, fit_completed=fit_completed)
        worker = {"version": VERSION, "status": "failed", "job": job,
                  "registration_commit": spec.get("registration_commit"),
                  "registration_sha256": spec.get("registration_sha256"),
                  "error_code": _error_code(error), "error_type": type(error).__name__,
                  "message": str(error), "training_failure_receipt": receipt,
                  **steps,
                  "hardware": hardware, "cohort_provenance": data_receipt,
                  "source_sha256_before": before, "source_sha256_after": _sources(),
                  "elapsed_worker_seconds": time.perf_counter()-started,
                  "final_market_values_read": False, "replacement_attempt": False}
        write_json_new(output/"worker_profile.json", worker)
        events.append("worker_failed", error_code=worker["error_code"], **steps)
        return 1
    finally:
        events.close()


def run_campaign(data_root, registration_path, expected_registration_sha256, registration_commit,
                 output, *, device="cuda:0", models=None, cohorts=None, include_largest=False,
                 source_root_itransformer=None):
    hardware = require_cuda_environment(device)  # Before registration/data/output access.
    registration_path = Path(registration_path).resolve()
    registration = load_registration(registration_path, expected_registration_sha256, registration_commit)
    jobs = build_jobs(registration, models, cohorts, include_largest)
    if any(job["model"] == "iTransformer" for job in jobs) and source_root_itransformer is None:
        raise ValueError("Pinned iTransformer source root is required for this selected model")
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise ValueError("Development cache directory must exist")
    output = Path(output)
    if output.exists():
        raise FileExistsError("GPU pilot output must be a new directory")
    output.mkdir(parents=True)
    started = time.monotonic()
    before = _sources()
    plan = {"version": VERSION, "jobs": jobs, "registration_sha256": expected_registration_sha256,
            "registration_commit": registration_commit, "hardware": hardware,
            "source_sha256": before, "source_protocol_sha256": PROTOCOL_SHA256,
            "resources": RESOURCE_CONTROLS, "parallel_fits": 1,
            "subset_sealed_before_market_file_read": True, "scientific_HPO_or_NAS_executed": False}
    write_json_new(output/"plan.json", plan)
    results = []
    journal = DurableEvents(output/"events.jsonl")
    try:
        journal.append("pilot_plan_sealed", job_count=len(jobs), registration_sha256=expected_registration_sha256)
        for index, job in enumerate(jobs):
            remaining = RESOURCE_CONTROLS["campaign_wall_seconds"]-(time.monotonic()-started)
            if remaining < RESOURCE_CONTROLS["minimum_worker_start_seconds"]:
                row = {"job": job, "status": "not_attempted_campaign_budget", "worker_profile": None,
                       "optimizer_steps": None, "optimizer_steps_censored": False,
                       "optimizer_steps_known_lower_bound": 0,
                       "elapsed_wall_seconds": 0., "sampled_peak_process_group_rss_bytes": None,
                       "replacement_attempt": False}
                results.append(row)
                write_json_new(output/("outcome-%03d.json" % index), row)
                journal.append("worker_not_attempted", job_id=job["job_id"], reason="campaign_budget")
                continue
            spec = {"version": VERSION, "job": job, "models": sorted({j["model"] for j in jobs}, key=MODEL_NAMES.index),
                    "cohorts": [c for c in ("small", "large") if any(j["cohort"] == c for j in jobs)],
                    "include_largest": include_largest, "registration": str(registration_path),
                    "registration_sha256": expected_registration_sha256, "registration_commit": registration_commit,
                    "device": hardware["device"], "data_root": str(data_root),
                    "output": str((output/("worker-%03d" % index)).resolve()),
                    "source_sha256_expected": before,
                    "source_roots": {"iTransformer": str(Path(source_root_itransformer).resolve())}
                        if source_root_itransformer is not None else None}
            spec_path = output/("worker-%03d-spec.json" % index)
            write_json_new(spec_path, spec)
            journal.append("worker_requested", job_id=job["job_id"])
            supervised = run_isolated_profile(
                [sys.executable, str(Path(__file__).resolve()), "--worker-spec", str(spec_path.resolve())],
                Path(spec["output"]), timeout_seconds=min(RESOURCE_CONTROLS["external_wall_seconds"], remaining),
                memory_limit_bytes=RESOURCE_CONTROLS["host_rss_limit_bytes"], cwd=ROOT)
            worker = supervised.get("worker_profile")
            complete = (supervised["status"] == "success" and isinstance(worker, dict)
                        and worker.get("status") == "completed" and worker.get("job") == job)
            censored = supervised["status"] in ("timeout", "memory_limit_observed")
            status = "success" if complete else supervised["status"]
            if not complete and status == "success":
                status = "invalid_worker_result"
            steps = None if censored or not worker else worker.get("optimizer_steps")
            lower = (worker or {}).get("optimizer_steps_known_lower_bound", steps if steps is not None else 0)
            row = {"job": job, "status": status, "worker_profile": worker,
                   "optimizer_steps": steps, "optimizer_steps_censored": steps is None,
                   "optimizer_steps_known_lower_bound": lower,
                   "partial_optimizer_update_may_have_occurred": bool(censored or not worker or
                       worker.get("partial_optimizer_update_may_have_occurred", False)),
                   "elapsed_wall_seconds": supervised["elapsed_wall_seconds"],
                   "sampled_peak_process_group_rss_bytes": supervised["sampled_peak_process_group_rss_bytes"],
                   "replacement_attempt": False}
            results.append(row)
            write_json_new(output/("outcome-%03d.json" % index), row)
            journal.append("worker_finished", job_id=job["job_id"], status=status, optimizer_steps=steps)
            print(json.dumps({"job_id": job["job_id"], "model": job["model"], "status": status}), flush=True)
        after = _sources()
        if before != after:
            raise ValueError("Repository sources changed during the GPU pilot")
        summary = {"version": VERSION, "status": "completed_registered_job_list", "plan": plan,
                   "registration_sha256": expected_registration_sha256, "registration_commit": registration_commit,
                   "results": results, "known_optimizer_steps": sum(row["optimizer_steps_known_lower_bound"] for row in results),
                   "unknown_optimizer_step_workers": sum(row["optimizer_steps"] is None for row in results
                                      if row["status"] != "not_attempted_campaign_budget"),
                   "all_fits_completed": all(row["status"] == "success" for row in results),
                   "source_sha256_before": before, "source_sha256_after": after,
                   "source_hashes_unchanged": True, "scientific_HPO_or_NAS_executed": False,
                   "final_market_values_read": False, "final_R": None,
                   "full_campaign_wall_hours": None,
                   "limits": ["Timing and VRAM profiles only; no scientific HPO/NAS winner is selected.",
                              "The campaign allowance bounds supervised worker time plus parent bookkeeping, not a kernel-enforced absolute wall clock.",
                              "Resource failures and not-attempted jobs remain explicit; no replacement fit is launched.",
                              "No CPU to GPU speedup or currency cost is assumed."]}
        write_json_new(output/"summary.json", summary)
        write_json_new(output/"scientific_report.json", public_scientific_report(summary))
        journal.append("pilot_finished", all_fits_completed=summary["all_fits_completed"])
        return summary
    finally:
        journal.close()


def _number(value, *, integer=False):
    if value is None:
        return None
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("A public resource metric must be finite and nonnegative")
    if integer and type(value) is not int:
        raise ValueError("A count must be an integer")
    return value


def _scientific_job(job):
    if job.get("model") not in MODEL_NAMES or job.get("cohort") not in ("small", "large"):
        raise ValueError("Unknown public model/cohort identity")
    case = job.get("case", "reference")
    if case not in ("reference", "largest"):
        raise ValueError("Unknown public architecture case")
    identity = job.get("job_id")
    if not isinstance(identity, str) or not re.fullmatch(r"gpu-profile-[0-9]{3}", identity):
        raise ValueError("Unknown public job identity")
    result = {"job_id": identity, "model": job["model"], "cohort": job["cohort"], "case": case,
              "seed": _number(job.get("seed"), integer=True),
              "epochs": _number(job.get("epochs"), integer=True),
              "batch_size": _number(job.get("batch_size"), integer=True)}
    pattern = job.get("pattern")
    if pattern is not None:
        widths, bits = pattern.get("widths"), pattern.get("dropout_pattern")
        if (not isinstance(widths, list) or len(widths) != 3 or any(type(v) is not int or v not in (16,32,64,128,256) for v in widths)
                or not isinstance(bits, list) or len(bits) != 3 or any(type(v) is not int or v not in (0,1) for v in bits)):
            raise ValueError("Malformed scientific architecture dimensions")
        result["pattern"] = {"widths": widths[:], "dropout_pattern": bits[:], "dropout_rate": .2}
    return result


def _public_hardware(hardware):
    if not isinstance(hardware, dict):
        return None
    result = {}
    patterns = {"device": r"cuda:[0-9]+", "device_name": r"[A-Za-z0-9 ._+()\-]{1,160}",
                "torch": r"[A-Za-z0-9._+\-]{1,80}", "torch_cuda": r"[0-9.]{1,30}"}
    for key, pattern in patterns.items():
        if key in hardware:
            value = hardware[key]
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise ValueError("Invalid public CUDA hardware identifier")
            result[key] = value
    for key in ("total_memory_bytes", "effective_memory_limit_bytes", "cudnn", "cpu_threads", "interop_threads"):
        if key in hardware:
            result[key] = _number(hardware[key], integer=True)
    result.update(dtype="float32", amp=False, tf32=False, deterministic=True)
    result["execution_policy_scope"] = "Registered fit/inference policy; not a claim that the preflight validated all model kernels"
    if "cuda_kernel_probe_passed" in hardware:
        if type(hardware["cuda_kernel_probe_passed"]) is not bool:
            raise ValueError("CUDA probe status must be a boolean")
        result["cuda_kernel_probe_passed"] = hardware["cuda_kernel_probe_passed"]
    return result


def public_scientific_report(summary):
    """Positive field whitelist: raw nested objects are never copied to public JSON."""
    if not _hex(summary.get("registration_commit"), 40) or not _hex(summary.get("registration_sha256"), 64):
        raise ValueError("Public scientific results require the exact registration identifiers")
    rows = []
    statuses = {"success", "worker_failure", "timeout", "memory_limit_observed", "spawn_failure",
                "invalid_worker_profile", "invalid_worker_result", "not_attempted_campaign_budget"}
    errors = {"resource_limit", "geometry", "nonfinite", "missing_gradient", "checkpoint", "runtime_error"}
    for raw in summary.get("results", []):
        status = raw.get("status")
        if status not in statuses:
            raise ValueError("Unknown scientific terminal status")
        worker = raw.get("worker_profile") or {}
        code = worker.get("error_code")
        row = {"job": _scientific_job(raw["job"]), "status": status,
               "error_code": code if code in errors else ("runtime_error" if code else None),
               "optimizer_steps": _number(raw.get("optimizer_steps"), integer=True),
               "optimizer_steps_known_lower_bound": _number(raw.get("optimizer_steps_known_lower_bound",
                       raw.get("optimizer_steps") if raw.get("optimizer_steps") is not None else 0), integer=True),
               "elapsed_wall_seconds": _number(raw.get("elapsed_wall_seconds")),
               "sampled_peak_process_group_rss_bytes": _number(raw.get("sampled_peak_process_group_rss_bytes"), integer=True)}
        for key in ("train_n", "validation_n", "parameter_count", "state_bytes"):
            if key in worker:
                row[key] = _number(worker[key], integer=True)
        row["timing"] = {key: _number(worker.get("timing", {}).get(key))
                         for key in ("build_seconds", "fit_seconds", "validation_seconds", "worker_seconds", "process_cpu_seconds")}
        row["cuda_metrics"] = {key: _number(worker.get("cuda_metrics", {}).get(key), integer=True)
                               for key in ("max_memory_allocated_bytes", "max_memory_reserved_bytes", "effective_limit_bytes")}
        row["phases"] = {}
        for name in ("ae1", "ae2", "supervised"):
            phase = worker.get("phase_receipts", {}).get(name)
            if isinstance(phase, dict):
                row["phases"][name] = {key: _number(phase.get(key), integer=key != "elapsed_seconds")
                    for key in ("epochs_completed", "best_epoch", "optimizer_steps", "fixed_epochs", "elapsed_seconds")}
        rows.append(row)
    attempted = [row for row in rows if row["status"] != "not_attempted_campaign_budget"]
    return {"version": VERSION+"-scientific-only", "results": rows,
            "registration_commit": summary["registration_commit"],
            "registration_sha256": summary["registration_sha256"],
            "hardware": _public_hardware(summary.get("plan", {}).get("hardware")),
            "profile_attempts": len(attempted), "not_attempted": len(rows)-len(attempted),
            "costs": {"known_optimizer_steps": sum(row["optimizer_steps_known_lower_bound"] for row in attempted),
                      "unknown_optimizer_step_workers": sum(row["optimizer_steps"] is None for row in attempted),
                      "completed_attempts": sum(row["status"] == "success" for row in attempted),
                      "failed_attempts": sum(row["status"] != "success" for row in attempted),
                      "not_attempted_jobs": len(rows)-len(attempted)},
            "scientific_HPO_or_NAS_executed": False, "final_market_values_read": False,
            "final_R": None, "full_campaign_wall_hours": None,
            "publication_scope": "Selected numerical resource measurements and model identities only; raw receipts and logs remain local"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-environment", action="store_true")
    parser.add_argument("--worker-spec", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--expected-registration-sha256")
    parser.add_argument("--registration-commit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--cohorts", nargs="+", choices=("small", "large"))
    parser.add_argument("--include-largest", action="store_true")
    parser.add_argument("--source-root-itransformer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.check_environment:
        print(json.dumps(require_cuda_environment(args.device), ensure_ascii=False, allow_nan=False))
        return 0
    if args.worker_spec is not None:
        return run_worker(json.loads(args.worker_spec.read_text()))
    if any(getattr(args, name) is None for name in
           ("data_root", "registration", "expected_registration_sha256", "registration_commit", "output")):
        parser.error("A pilot requires data root, exact registration/hash/commit and a new output directory")
    value = run_campaign(args.data_root, args.registration, args.expected_registration_sha256,
                         args.registration_commit, args.output, device=args.device,
                         models=args.models, cohorts=args.cohorts, include_largest=args.include_largest,
                         source_root_itransformer=args.source_root_itransformer)
    print(json.dumps({"status": value["status"], "all_fits_completed": value["all_fits_completed"],
                      "scientific_HPO_or_NAS_executed": False, "final_R": None}))
    return 0 if value["all_fits_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
