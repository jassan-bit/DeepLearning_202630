"""Preregistered CPU thread and architecture-cost profiles on development data.

No HPO/NAS winner is selected and no scientific campaign is launched. The
timing observer calls the frozen trainer functions unchanged, then restores
their exact objects. All workers are separate sequential processes; source
files, failures and partial costs remain auditable. Profile weights are not
persisted as training checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

from profile_e2_training import run_isolated_profile, hardware_snapshot
from run_e2_development_pilot import (DurableEvents, read_events, write_json_new,
                                     _partial_steps, _failure_code)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "book_e2/_static/search_protocol_20261109/protocol.json"
VERSION = "e2-cpu-scaling-v1"
REGISTRATION_VERSION = "e2-compute-calibration-registration-v1"
THREAD_MODELS = ("TimeMixer", "TimesNet")
THREAD_COUNTS = (1, 2, 4)
THREAD_SEEDS = (272101, 272102)
SCALING_SEED = 272201
KERNEL_SEEDS = (272301, 272302)
MICRO_SEED = 272501
NAS_NAMES = ("LSTM-autoencoder con predictor", "TimesNet", "TiDE", "TimeMixer",
             "PatchTST", "CNN-1D + LSTM + atención temporal")
FIXED_NAMES = ("FCN-1D", "ResNet-1D", "Inception-1D", "TSMixer", "iTransformer",
               "MLP16", "LSTM32", "GRU32")
MEMORY_BYTES = 12 * 1024**3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def load_registration(path):
    value = json.loads(Path(path).read_text())
    expected = {"thread_counts": list(THREAD_COUNTS), "thread_models": list(THREAD_MODELS),
                "thread_epochs": 5, "thread_seeds": list(THREAD_SEEDS),
                "thread_internal_wall_seconds": 210, "thread_external_wall_seconds": 240,
                "scaling_epochs": 2, "scaling_seed": SCALING_SEED,
                "scaling_internal_wall_seconds": 300, "scaling_external_wall_seconds": 360,
                "memory_limit_bytes": MEMORY_BYTES,
                "kernel_models":["TimesNet"], "kernel_patterns":"reference_and_largest",
                "kernel_threads":4,"kernel_epochs":1,"kernel_seeds":list(KERNEL_SEEDS),
                "kernel_internal_wall_seconds":210,"kernel_external_wall_seconds":240,
                "kernel_batch_windows":32,"micro_seed":MICRO_SEED,
                "micro_epochs":1,"micro_batch_windows":32}
    if value.get("version") != REGISTRATION_VERSION:
        raise ValueError("Use the compute/calibration preregistration version")
    if value.get("status") != "registered_before_new_market_fits":
        raise ValueError("A draft registration cannot authorize market profiling")
    profile = value.get("cpu_profile", {})
    if any(profile.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("CPU profile controls differ from their registered domain")
    if value.get("source_protocol_sha256") != digest(PROTOCOL):
        raise ValueError("Profile protocol bytes differ from preregistration")
    return value


def profile_sampling_frame():
    """Three certainty anchors and nine fixed uniform-within-stratum samples."""
    import e2_search_plan as planner
    protocol = json.loads(PROTOCOL.read_text())
    rows = planner.noise_pilot_architectures(protocol)
    space = [(widths, bits) for widths in itertools.product(protocol["nas"]["widths"], repeat=3)
             for bits in itertools.product((0, 1), repeat=3)]
    anchors = {(tuple(row["widths"]), tuple(row["dropout_pattern"])) for row in rows[:3]}
    remaining = [pair for pair in space if pair not in anchors]
    if len(rows) != 12 or len(remaining) != 997:
        raise ValueError("Exactly three anchors and nine registered strata are required")
    result = []
    for index, row in enumerate(rows):
        item = {"pattern": deepcopy(row), "sampling_role": "anchor" if index < 3 else "stratum_sample",
                "stratum_index": None, "stratum_size": 1}
        if index >= 3:
            stratum = index - 3
            left, right = len(remaining)*stratum//9, len(remaining)*(stratum+1)//9
            pair = (tuple(row["widths"]), tuple(row["dropout_pattern"]))
            if pair not in remaining[left:right]:
                raise ValueError("Frozen pattern is outside its original random stratum")
            item.update(stratum_index=stratum, stratum_size=right-left)
        result.append(item)
    if sum(row["stratum_size"] for row in result) != 1000:
        raise ValueError("Sampling-frame weights do not cover all1000 architectures")
    return result


def build_jobs(phase, registration, threads=None, kernel_backend="dense"):
    profile = registration["cpu_profile"]
    frame = profile_sampling_frame()
    reference = next(item for item in frame if item["pattern"]["is_reference"])
    jobs = []
    if kernel_backend not in ("dense", "cropped"):
        raise ValueError("Unknown measured TimesNet implementation")
    if phase == "kernels":
        largest=next(item for item in frame if item["pattern"]["widths"]==[256]*3
                     and item["pattern"]["dropout_pattern"]==[1]*3)
        for replicate,seed in enumerate(KERNEL_SEEDS):
            order=("dense","cropped") if replicate==0 else ("cropped","dense")
            for item in (reference,largest):
                for backend in order:
                    jobs.append({"phase":phase,"model":"TimesNet","cohort":"small",
                        "pattern":deepcopy(item["pattern"]),"sampling":None,
                        "threads":4,"seed":seed,"epochs":1,"backend":backend,
                        "batch_windows":32})
    elif phase == "threads":
        for replicate, seed in enumerate(THREAD_SEEDS):
            order = THREAD_COUNTS if replicate == 0 else THREAD_COUNTS[::-1]
            for name in THREAD_MODELS:
                for count in order:
                    jobs.append({"phase": phase, "model": name, "cohort": "small",
                                 "pattern": deepcopy(reference["pattern"]), "sampling": None,
                                 "threads": count, "seed": seed, "epochs": profile["thread_epochs"]})
    elif phase == "scaling":
        if threads not in THREAD_COUNTS or type(threads) is not int:
            raise ValueError("Scaling requires a thread count chosen from the complete benchmark")
        for name in NAS_NAMES:
            for item in frame:
                jobs.append({"phase": phase, "model": name, "cohort": "large",
                             "pattern": deepcopy(item["pattern"]), "sampling": deepcopy(item),
                             "threads": threads, "seed": SCALING_SEED,
                             "epochs": profile["scaling_epochs"]})
        for name in NAS_NAMES:
            jobs.append({"phase": phase, "model": name, "cohort": "small",
                         "pattern": deepcopy(reference["pattern"]), "sampling": None,
                         "threads": threads, "seed": SCALING_SEED,
                         "epochs": profile["scaling_epochs"]})
        for name in FIXED_NAMES:
            jobs.append({"phase": phase, "model": name, "cohort": "large", "pattern": None,
                         "sampling": None, "threads": threads, "seed": SCALING_SEED,
                         "epochs": profile["scaling_epochs"]})
    else:
        raise ValueError("Unknown CPU profiling phase")
    for index, job in enumerate(jobs):
        job["job_id"] = phase + "-" + str(index).zfill(3)
        job.setdefault("backend",kernel_backend)
        job.setdefault("batch_windows",None)
    return jobs


def select_kernel_policy(records, *, parity_suite_passed):
    """Select crop only when both paired cases support its whole-worker speed."""
    frame=profile_sampling_frame()
    reference=next(item["pattern"]["pattern_id"] for item in frame if item["pattern"]["is_reference"])
    largest=next(item["pattern"]["pattern_id"] for item in frame
                 if item["pattern"]["widths"]==[256]*3 and item["pattern"]["dropout_pattern"]==[1]*3)
    expected=set(itertools.product((reference,largest),("dense","cropped"),KERNEL_SEEDS))
    seen, grouped, problems = set(),{p:{"dense":[],"cropped":[]} for p in (reference,largest)},[]
    dense_censored=[]
    for row in records:
        job=row["job"]
        key=(job["pattern"]["pattern_id"],job["backend"],job["seed"])
        if job.get("phase")!="kernels" or key not in expected or key in seen:
            raise ValueError("Kernel benchmark must contain exactly its eight distinct attempts")
        seen.add(key)
        status=row["status"]
        worker=row.get("worker_profile") or {}
        message=str(worker.get("message","")).lower()
        wall_censor=(status=="timeout" or (worker.get("error_code")=="resource_limit" and "wall" in message))
        completed=status=="success" and worker.get("status")=="completed"
        if not completed and not (job["backend"]=="dense" and wall_censor):
            problems.append({"key":list(key),"status":status,"error_code":worker.get("error_code")})
            continue
        value=row["elapsed_wall_seconds"]
        if type(value) not in (float,int) or not math.isfinite(value) or value<=0:
            raise ValueError("Kernel profile needs positive complete/censored wall duration")
        grouped[key[0]][key[1]].append(float(value))
        if wall_censor:dense_censored.append(list(key))
    full=seen==expected
    dense_error=any(item["key"][1]=="dense" for item in problems)
    per_case={}
    ratios=[]
    for pattern,backends in grouped.items():
        complete=all(len(backends[b])==2 for b in ("dense","cropped"))
        medians={b:statistics.median(v) if v else None for b,v in backends.items()}
        ratio=medians["cropped"]/medians["dense"] if complete else None
        per_case[pattern]={"median_worker_seconds":medians,"cropped_to_dense_ratio":ratio,
                           "baseline_duration_may_be_lower_bound":any(x[0]==pattern for x in dense_censored)}
        ratios.append(ratio)
    crop=(full and parity_suite_passed is True and not problems
          and all(ratio is not None and ratio<=.95 for ratio in ratios))
    backend=None if not full or dense_error else ("cropped" if crop else "dense")
    return {"status":"runtime_kernel_selected" if backend else "blocked_kernel_benchmark",
            "selected_backend":backend,"crop":backend=="cropped",
            "parity_suite_passed":parity_suite_passed is True,"provided_attempts":len(seen),
            "expected_attempts":8,"dense_wall_censored_cases":dense_censored,
            "other_failures":problems,"per_case":per_case,"required_wall_ratio":.95,
            "accuracy_used_for_selection":False,
            "limits":["A censored baseline duration is only a lower bound on complete duration.",
                       "No incomplete cropped attempt can justify selecting the optimization.",
                       "Runtime parity is independently tested; this benchmark does not select prediction scores."]}


def load_runtime_policy(kernels_path, threads_path):
    kernels=json.loads(Path(kernels_path).read_text())
    threads=json.loads(Path(threads_path).read_text())
    choice=select_kernel_policy(kernels["results"],parity_suite_passed=kernels["plan"]["parity_suite_passed"])
    timing=select_thread_policy(threads["results"])
    if choice["selected_backend"] is None or timing["selected_threads"] is None:
        raise ValueError("A complete admitted runtime policy is required")
    if threads["plan"]["kernels_summary_sha256"]!=digest(kernels_path):
        raise ValueError("Thread profiles refer to a different kernel benchmark")
    if any(kernels["plan"][key]!=threads["plan"][key]
           for key in ("registration_commit","registration_sha256")):
        raise ValueError("Runtime phases refer to different preregistrations")
    if any(row["job"]["backend"]!=choice["selected_backend"] for row in threads["results"]):
        raise ValueError("Thread profiles did not use the selected kernel implementation")
    return {"threads":timing["selected_threads"],"crop":choice["crop"],
            "kernel_backend":choice["selected_backend"],
            "kernels_summary_sha256":digest(kernels_path),
            "threads_summary_sha256":digest(threads_path),"kernel_policy":choice,"thread_policy":timing,
            "registration_commit":kernels["plan"]["registration_commit"],
            "registration_sha256":kernels["plan"]["registration_sha256"]}


def eligible_micro_fallback(record):
    if record["job"].get("phase")!="scaling" or record["status"]=="success":return False
    worker=record.get("worker_profile") or {}
    return record["status"] in ("timeout","memory_limit_observed") or worker.get("error_code")=="resource_limit"


def micro_job(parent):
    job=deepcopy(parent)
    job.update(phase="micro_fallback",job_id=parent["job_id"]+"-micro",
               parent_job_id=parent["job_id"],seed=MICRO_SEED,epochs=1,batch_windows=32)
    return job


def select_thread_policy(records):
    """Choose one global runtime policy from all12 paired fits, never their losses."""
    expected = set(itertools.product(THREAD_MODELS, THREAD_COUNTS, THREAD_SEEDS))
    grouped = {name: {count: [] for count in THREAD_COUNTS} for name in THREAD_MODELS}
    seen, unfinished = set(), []
    for record in records:
        job = record["job"]
        key = (job["model"], job["threads"], job["seed"])
        if job.get("phase") != "threads" or key not in expected or key in seen:
            raise ValueError("A thread benchmark cannot duplicate or replace a registered attempt")
        seen.add(key)
        worker = record.get("worker_profile") or {}
        if record["status"] != "success" or worker.get("status") != "completed":
            unfinished.append(job["job_id"])
            continue
        value = worker.get("total_work_seconds")
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("A completed benchmark needs positive finite runtime")
        grouped[job["model"]][job["threads"]].append(float(value))
    if seen != expected or unfinished:
        return {"status": "blocked_incomplete_thread_benchmark", "selected_threads": None,
                "expected_attempts": 12, "provided_attempts": len(seen),
                "missing_attempts": len(expected-seen), "unfinished_jobs": unfinished,
                "accuracy_used_for_selection": False}
    medians = {name: {count: statistics.median(values) for count, values in family.items()}
               for name, family in grouped.items()}
    normalized = {count: sum(medians[name][count]/medians[name][4] for name in THREAD_MODELS)
                  for count in THREAD_COUNTS}
    best = min(normalized.values())
    eligible = [count for count in THREAD_COUNTS if normalized[count] <= 1.05*best]
    return {"status": "complete_runtime_policy_selected", "selected_threads": min(eligible),
            "family_median_work_seconds": {name:{str(count):value for count,value in family.items()}
                                           for name,family in medians.items()},
            "normalized_runtime_scores": {str(count):value for count,value in normalized.items()},
            "tie_within_relative_fraction": .05, "tie_rule": "fewest_threads",
            "accuracy_used_for_selection": False, "completed_attempts": 12,
            "normalization_baseline_threads": 4,
            "limits": ["Two timed seeds and this server do not guarantee future throughput.",
                       "Only the two registered families choose the global CPU policy."]}


def prepare_cohort(data_root, cohort):
    """Seal161/619 training rows; appended2025 dates carry no market values."""
    import numpy as np
    import pandas as pd
    from e2_development_data import load_development
    from e2_temporal_training import DatedPanel, prepare_b
    if cohort == "small":
        panel, receipt = load_development(data_root, "2023-09-23")
        prepared = prepare_b(panel, "2023-10-01", "2023-10-01")
        expected = (161, "2023-06-25", "2023-07-03", "2023-09-23")
        placeholders = receipt["placeholder_dates"]
    elif cohort == "large":
        original, receipt = load_development(data_root, "2024-12-30")
        calendar = pd.date_range(original.calendar[0], pd.Timestamp("2025-01-01",tz="UTC"), freq="D")
        features = np.full((len(calendar), 5), np.nan)
        targets = np.full(len(calendar), np.nan)
        quality = np.zeros(len(calendar), dtype=bool)
        n = len(original.calendar)
        features[:n], targets[:n], quality[:n] = original.features, original.targets, original.quality
        panel = DatedPanel(calendar, features, targets, quality)
        prepared = prepare_b(panel, "2025-01-01", "2025-01-01")
        expected = (619, "2024-09-25", "2024-10-03", "2024-12-24")
        placeholders = calendar[n:].strftime("%Y-%m-%d").tolist()
    else:
        raise ValueError("Only the two preregistered development cohorts are admitted")
    train, valid = prepared.inner_train, prepared.inner_validation
    actual = (len(train.dates), str(train.dates[-1].date()),
              str(valid.dates[0].date()), str(valid.dates[-1].date()))
    if actual != expected or len(valid.dates) != 83 or any(prepared.outer_plan.eligible):
        raise ValueError("Profile chronology, cardinality or unavailable outer labels changed")
    provenance = {"cohort": cohort, "train_n": len(train.dates), "validation_n": len(valid.dates),
                  "train_end": actual[1], "validation_start": actual[2], "validation_end": actual[3],
                  "outer_observations_available": 0, "placeholder_dates": placeholders,
                  "numeric_2025_market_values_read": False, "data_receipt": receipt,
                  "prepared_signature": prepared.signature}
    return prepared, provenance


@contextmanager
def observe_trainer_timings():
    """Time call boundaries while preserving arguments, results, RNG and files.

    The first batch_indices call calculates the scheduler horizon; the second
    starts the first update loop. Fixed-mode train evaluation ends every epoch.
    Setup, loop and finalization partition total time; train evaluation, cloning
    and hashing are labelled subsets, not extra additive stage costs.
    """
    import e2_staged_training as staged
    names = ("_fit", "batch_indices", "_mse", "clone_state", "state_sha256")
    originals = {name: getattr(staged, name) for name in names}
    records, active = [], []

    def stamp():
        return (time.perf_counter(), time.process_time())

    def difference(end, start):
        return end[0]-start[0], end[1]-start[1]

    def timed_fit(*args, **kwargs):
        entry = {"start": stamp(), "batch_calls": 0, "loop_start": None,
                 "last_evaluation_end": None, "train_evaluation_seconds": 0.,
                 "train_evaluation_cpu_seconds": 0., "checkpoint_clone_seconds": 0.,
                 "checkpoint_clone_cpu_seconds": 0., "state_hash_seconds": 0.,
                 "state_hash_cpu_seconds": 0., "completed": False}
        active.append(entry)
        try:
            result = originals["_fit"](*args, **kwargs)
            entry["completed"] = True
            return result
        finally:
            end = stamp()
            active.pop()
            first = entry["loop_start"] or end
            last = entry["last_evaluation_end"] if entry["completed"] else end
            last = last or first
            total = difference(end, entry["start"])
            setup = difference(first, entry["start"])
            loop = difference(last, first)
            final = difference(end, last)
            record = {key: entry[key] for key in ("completed", "batch_calls", "train_evaluation_seconds",
                      "train_evaluation_cpu_seconds", "checkpoint_clone_seconds",
                      "checkpoint_clone_cpu_seconds", "state_hash_seconds", "state_hash_cpu_seconds")}
            for prefix, values in (("total", total), ("setup", setup), ("loop", loop), ("finalization", final)):
                record[prefix+"_seconds"], record[prefix+"_cpu_seconds"] = values
            record["update_loop_seconds"] = max(0., loop[0]-entry["train_evaluation_seconds"])
            record["update_loop_cpu_seconds"] = max(0., loop[1]-entry["train_evaluation_cpu_seconds"])
            records.append(record)

    def batches(*args, **kwargs):
        if active:
            entry = active[-1]
            entry["batch_calls"] += 1
            if entry["batch_calls"] == 2:
                entry["loop_start"] = stamp()
        return originals["batch_indices"](*args, **kwargs)

    def timed(name, field):
        def call(*args, **kwargs):
            start = stamp()
            try:
                return originals[name](*args, **kwargs)
            finally:
                end = stamp()
                if active:
                    entry = active[-1]
                    elapsed, cpu = difference(end, start)
                    entry[field+"_seconds"] += elapsed
                    entry[field+"_cpu_seconds"] += cpu
                    if name == "_mse":
                        entry["last_evaluation_end"] = end
        return call

    staged._fit, staged.batch_indices = timed_fit, batches
    staged._mse = timed("_mse", "train_evaluation")
    staged.clone_state = timed("clone_state", "checkpoint_clone")
    staged.state_sha256 = timed("state_sha256", "state_hash")
    try:
        yield records
    finally:
        for name, original in originals.items():
            setattr(staged, name, original)


def _source_hashes():
    scripts = (ROOT/"scripts").resolve()
    result = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(scripts) and path.is_file():
                result[str(path.relative_to(ROOT))] = digest(path)
    return dict(sorted(result.items()))


def build_profile_model(name, pattern, seed, source_roots, registration, backend="dense"):
    from e2_development_models import build_pilot_model
    model=build_pilot_model(name, pattern, seed=seed, source_roots=source_roots)
    if backend=="cropped" and name=="TimesNet":
        from e2_cpu_conv import enable_cpu_crop
        model=enable_cpu_crop(model,name=name)
    elif backend not in ("dense","cropped"):
        raise ValueError("Unknown profile implementation")
    return model


def run_worker(spec):
    output = Path(spec["output"])
    events = DurableEvents(output/"events.jsonl")
    started, cpu_started = time.perf_counter(), time.process_time()
    job = spec["job"]
    data_receipt = None
    before = None
    observed_timings = []
    events.append("worker_started", job=job, registration_sha256=spec["registration_sha256"],
                  registration_commit=spec["registration_commit"])
    try:
        import torch
        import e2_staged_training as staged
        from e2_nas_common import NASConfig
        from e2_development_models import fixed_pilot_config
        if digest(spec["registration"]) != spec["registration_sha256"]:
            raise ValueError("Registration changed after the parent sealed its plan")
        registration = load_registration(spec["registration"])
        phase=job["phase"]
        if phase=="micro_fallback":
            parent=json.loads(Path(spec["parent_outcome"]).read_text())
            if (digest(spec["parent_outcome"])!=spec["parent_outcome_sha256"]
                    or not eligible_micro_fallback(parent) or micro_job(parent["job"])!=job):
                raise ValueError("Micro profile does not match an eligible retained scaling failure")
            expected=build_jobs("scaling",registration,job["threads"],job["backend"])
            if parent["job"] not in expected:
                raise ValueError("Micro parent is outside registered scaling domain")
        else:
            expected = build_jobs(phase, registration, job["threads"] if phase == "scaling" else None,
                                  job["backend"])
            if job not in expected:
                raise ValueError("Worker is outside the exact registered profile domain")
        if phase!="kernels":
            if digest(spec["kernels_summary"])!=spec["kernels_summary_sha256"]:
                raise ValueError("Kernel summary changed after the plan was sealed")
            kernel_summary=json.loads(Path(spec["kernels_summary"]).read_text())
            kernel_choice=select_kernel_policy(kernel_summary["results"],
                parity_suite_passed=kernel_summary["plan"]["parity_suite_passed"])
            if kernel_choice["selected_backend"]!=job["backend"]:
                raise ValueError("Worker changed the measured kernel policy")
        torch.set_num_threads(job["threads"])
        torch.set_num_interop_threads(1)
        prepared, data_receipt = prepare_cohort(spec["data_root"], job["cohort"])
        train, valid = prepared.inner_train.clone(), prepared.inner_validation.clone()
        events.append("development_cohort_sealed", provenance=data_receipt,
                      train= train.record(), validation=valid.record())
        cap = 210 if phase in ("threads","kernels") else 300
        config = fixed_pilot_config(job["seed"], epoch_cap=job["epochs"],
                                   wallcap=cap, memory_limit_bytes=MEMORY_BYTES)
        before = _source_hashes()
        build_at, build_cpu_at = time.perf_counter(), time.process_time()
        model = build_profile_model(job["model"], job["pattern"], job["seed"],
                                    spec.get("source_roots"), registration,job["backend"])
        metadata = deepcopy(model.architecture_metadata)
        parameter_count = sum(p.numel() for p in model.parameters())
        build_seconds, build_cpu_seconds = time.perf_counter()-build_at, time.process_time()-build_cpu_at
        events.append("fixed_epoch_fit_started", config=config.__dict__, parameter_count=parameter_count,
                      source_sha256=before, mkldnn_enabled=torch.backends.mkldnn.enabled)
        fit_n=job["batch_windows"] or len(train.dates)
        x_train,y_train=train.x[:fit_n].clone(),train.y[:fit_n].clone()
        fit_at, fit_cpu_at = time.perf_counter(), time.process_time()
        with observe_trainer_timings() as observed_timings:
            if job["model"] == "LSTM-autoencoder con predictor":
                del model
                counts = {name:job["epochs"] for name in ("ae1", "ae2", "supervised")}
                result = staged.fit_sae(NASConfig(job["pattern"]["widths"], job["pattern"]["dropout_pattern"]),
                    x_train, y_train, None, None, train.scaler.target_scale, config,
                    fold_id="cpu-profile-"+prepared.signature, fixed_epochs=counts)
                model, phases = result.model, result.phases
                auxiliary_scale = result.auxiliary_scale
            else:
                stage = staged.fit_supervised(model, x_train, y_train, None, None,
                    train.scaler.target_scale, config, fixed_epochs=job["epochs"])
                phases = {"supervised": stage}
                auxiliary_scale = None
        fit_seconds, fit_cpu_seconds = time.perf_counter()-fit_at, time.process_time()-fit_cpu_at
        if len(observed_timings) != len(phases):
            raise ValueError("Timing observer missed an optimization stage")
        phase_timing = dict(zip(phases, observed_timings))
        phase_receipts = {name:stage.to_dict() for name,stage in phases.items()}
        if any(stage.fixed_epochs != job["epochs"] or stage.epochs_completed != job["epochs"]
               or stage.optimizer_initial_state_entries != 0 for stage in phases.values()):
            raise ValueError("Profile epochs or fresh optimizer contract changed")
        validation_at, validation_cpu_at = time.perf_counter(), time.process_time()
        # Validation loss is evaluated for its cost only; never for stopping,
        # thread choice, recipe selection or model ranking.
        staged._mse(model, valid.x, valid.y, config.batch_size)
        validation_seconds = time.perf_counter()-validation_at
        validation_cpu_seconds = time.process_time()-validation_cpu_at
        expected_hash = phases["supervised"].selected_state_sha256
        if staged.state_sha256(model) != expected_hash:
            raise ValueError("Evaluation changed the selected predictor state")
        after = _source_hashes()
        if any(after.get(path) != sha for path, sha in before.items()):
            raise ValueError("A loaded profile source changed during fitting")
        state_bytes = sum(t.numel()*t.element_size() for t in model.state_dict().values())
        phase_state_bytes = {name:sum(t.numel()*t.element_size() for t in stage.state_dict.values())
                             for name,stage in phases.items()}
        execution_optimization = None
        if job["backend"] == "cropped" and job["model"] == "TimesNet":
            from e2_cpu_conv import execution_report
            execution_optimization = execution_report(model)
        # Auxiliary validation is not directly measured in this fixed-stage
        # profile. Its train-evaluation throughput is available as an explicitly
        # conditional proxy; consumers must not invent a measured value.
        worker = {"version":VERSION, "status":"completed", "job":job,
                  "model":job["model"], "pattern":job["pattern"],
                  "registration_commit":spec["registration_commit"],
                  "registration_sha256":spec["registration_sha256"],
                  "protocol_sha256":digest(PROTOCOL), "source_sha256_before":before,
                  "source_sha256_after":after, "sources_unchanged_during_fit":True,
                  "cohort_provenance":data_receipt, "prepared_signature":prepared.signature,
                  "train_n":fit_n, "scaler_source_train_n":len(train.dates),
                  "actual_train_dates":[day.isoformat() for day in train.dates[:fit_n]],
                  "validation_n":len(valid.dates),
                  "phase_receipts":phase_receipts, "stage_timing":phase_timing,
                  "phase_state_bytes":phase_state_bytes, "parameter_count":parameter_count,
                  "state_bytes":state_bytes, "fixed_epoch_count":job["epochs"],
                  "optimizer_steps":sum(stage.optimizer_steps for stage in phases.values()),
                  "build_seconds":build_seconds, "fit_seconds":fit_seconds,
                  "validation_seconds":validation_seconds,
                  "build_cpu_seconds":build_cpu_seconds, "fit_cpu_seconds":fit_cpu_seconds,
                  "validation_cpu_seconds":validation_cpu_seconds,
                  "total_work_seconds":build_seconds+fit_seconds+validation_seconds,
                  "total_cpu_seconds":time.process_time()-cpu_started,
                  "elapsed_worker_seconds":time.perf_counter()-started,
                  "worker_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  "sae_fit_includes_auxiliary_construction":job["model"]=="LSTM-autoencoder con predictor",
                  "fit_management_seconds":max(0.,fit_seconds-sum(t["total_seconds"] for t in observed_timings)),
                  "validation_by_stage_seconds":{"supervised":validation_seconds,
                      **({"ae1":None,"ae2":None} if len(phases)==3 else {})},
                  "validation_labels_used_for_stopping":False,
                  "accuracy_used_for_thread_selection":False, "auxiliary_scale":auxiliary_scale,
                  "runtime":{"torch":str(torch.__version__), "threads":torch.get_num_threads(),
                      "interop_threads":torch.get_num_interop_threads(),
                      "mkldnn_enabled":torch.backends.mkldnn.enabled,"device":"cpu"},
                  "kernel_backend":job["backend"],
                  "execution_optimization":execution_optimization,
                  "separate_micro_profile_not_replacement":phase=="micro_fallback",
                  "model_metadata":metadata, "checkpoint_disk_io_measured":False,
                  "inference_weights_persisted":False, "HPO_or_NAS_search_executed":False,
                  "final_market_observations_read":False,
                  "limits":["Fixed epochs profile only; not selected-epoch scientific training.",
                            "Timing observation preserves frozen trainer calls but adds a few clock reads.",
                            "Stage timing subsets are not additional additive costs.",
                            "Auxiliary validation and disk checkpoint time are not directly measured.",
                            "The largest origin uses2025 calendar placeholders, not2025 numerical data."]}
        write_json_new(output/"worker_profile.json", worker)
        events.append("worker_completed", optimizer_steps=worker["optimizer_steps"],
                      stage_timing=phase_timing, source_sha256=after)
        return 0
    except Exception as error:
        receipt = getattr(error, "receipt", None)
        worker = {"version":VERSION, "status":"failed", "job":job,
                  "registration_commit":spec.get("registration_commit"),
                  "registration_sha256":spec.get("registration_sha256"),
                  "error_code":_failure_code(error), "error_type":type(error).__name__,
                  "message":str(error), "training_failure_receipt":receipt,
                  "optimizer_steps":_partial_steps(receipt), "stage_timing":observed_timings,
                  "cohort_provenance":data_receipt, "source_sha256_before":before,
                  "source_sha256_after":_source_hashes(),
                  "elapsed_worker_seconds":time.perf_counter()-started,
                  "total_cpu_seconds":time.process_time()-cpu_started,
                  "worker_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  "retry_performed":False}
        write_json_new(output/"worker_profile.json", worker)
        events.append("worker_failed", error_code=worker["error_code"],
                      optimizer_steps=worker["optimizer_steps"])
        return 1
    finally:
        events.close()


def _parity_evidence(path, registration):
    path=Path(path).resolve()
    expected=registration["cpu_profile"].get("crop_parity_evidence_sha256")
    if not isinstance(expected,str) or digest(path)!=expected:
        raise ValueError("Use the exact preregistered synthetic parity evidence")
    value=json.loads(path.read_text())
    if (value.get("passed") is not True or value.get("synthetic_only") is not True
            or value.get("market_data_read") is not False or not value.get("source_sha256")):
        raise ValueError("CPU cropping requires the complete passing synthetic parity suite")
    for name,expected_source in value["source_sha256"].items():
        source=(ROOT/name).resolve()
        if not source.is_relative_to(ROOT) or digest(source)!=expected_source:
            raise ValueError("Parity proof source changed before market profiling")
    return value


def run_campaign(phase, data_root, registration_path, registration_commit, output,
                 *, kernels_summary=None, threads_summary=None, parity_evidence=None,
                 source_root_itransformer=None):
    registration_path=Path(registration_path).resolve()
    registered=load_registration(registration_path)
    if (not isinstance(registration_commit,str) or len(registration_commit)!=40
            or any(c not in "0123456789abcdef" for c in registration_commit)):
        raise ValueError("Pass the exact published preregistration commit SHA")
    data_root=Path(data_root).resolve()
    if not data_root.is_dir():raise ValueError("Development cache directory must exist")
    kernel_choice,thread_choice,proof=None,None,None
    if phase=="kernels":
        if parity_evidence is None:raise ValueError("Kernel profiles require pinned synthetic parity evidence")
        proof=_parity_evidence(parity_evidence,registered)
        threads,backend=None,"dense"
    else:
        if kernels_summary is None:raise ValueError("Run the complete kernel benchmark first")
        kernels_summary=Path(kernels_summary).resolve()
        previous=json.loads(kernels_summary.read_text())
        if (previous["plan"]["registration_sha256"]!=digest(registration_path)
                or previous["plan"]["registration_commit"]!=registration_commit):
            raise ValueError("Kernel benchmark belongs to a different preregistration")
        kernel_choice=select_kernel_policy(previous["results"],
                            parity_suite_passed=previous["plan"]["parity_suite_passed"])
        if kernel_choice["selected_backend"] is None:raise ValueError("Kernel benchmark did not admit a runtime implementation")
        backend=kernel_choice["selected_backend"]
        threads=None
        if phase=="scaling":
            if threads_summary is None:raise ValueError("Scaling requires all12 thread attempts")
            runtime=load_runtime_policy(kernels_summary,threads_summary)
            if runtime["registration_sha256"]!=digest(registration_path):
                raise ValueError("Runtime policy registration changed")
            threads=runtime["threads"]
            thread_choice=runtime["thread_policy"]
            if source_root_itransformer is None:raise ValueError("Pinned iTransformer source directory required")
    jobs=build_jobs(phase,registered,threads,backend)
    output=Path(output)
    if output.exists():raise FileExistsError("CPU profile output must be a new directory")
    output.mkdir(parents=True)
    journal=DurableEvents(output/"events.jsonl")
    sources={"iTransformer":str(Path(source_root_itransformer).resolve())} if source_root_itransformer else None
    timeout=240 if phase in ("threads","kernels") else 360
    plan={"version":VERSION,"phase":phase,"jobs":jobs,
          "registration_sha256":digest(registration_path),"registration_commit":registration_commit,
          "source_script_sha256":digest(__file__),"source_protocol_sha256":digest(PROTOCOL),
          "hardware":hardware_snapshot(),"selected_kernel_policy":kernel_choice,
          "selected_thread_policy":thread_choice,
          "parity_suite_passed":bool(proof) if phase=="kernels" else previous["plan"]["parity_suite_passed"],
          "parity_evidence_sha256":digest(parity_evidence) if parity_evidence else None,
          "kernels_summary_sha256":digest(kernels_summary) if kernels_summary else None,
          "threads_summary_sha256":digest(threads_summary) if threads_summary else None,
          "external_worker_wall_seconds":timeout,"rss_watchdog_bytes":MEMORY_BYTES,"parallel_fits":1}
    write_json_new(output/"plan.json",plan)
    results,micros=[],[]

    def launch(job,index,*,parent_outcome=None):
        label="micro-%03d"%index if parent_outcome else "worker-%03d"%index
        worker_dir=output/label
        spec={"job":job,"registration":str(registration_path),
              "registration_sha256":plan["registration_sha256"],"registration_commit":registration_commit,
              "data_root":str(data_root),"source_roots":sources,"output":str(worker_dir.resolve()),
              "kernels_summary":str(kernels_summary) if kernels_summary else None,
              "kernels_summary_sha256":plan["kernels_summary_sha256"]}
        if parent_outcome:
            spec.update(parent_outcome=str(parent_outcome.resolve()),parent_outcome_sha256=digest(parent_outcome))
        spec_path=output/(label+"-spec.json")
        write_json_new(spec_path,spec)
        journal.append("worker_requested",job=job,spec_sha256=digest(spec_path))
        supervised=run_isolated_profile([sys.executable,str(Path(__file__).resolve()),
            "--worker-spec",str(spec_path.resolve())],worker_dir,timeout_seconds=timeout,
            memory_limit_bytes=MEMORY_BYTES,cwd=ROOT)
        worker=supervised.get("worker_profile")
        complete=(supervised["status"]=="success" and isinstance(worker,dict)
                  and worker.get("status")=="completed" and worker.get("job")==job)
        censored=supervised["status"] in ("timeout","memory_limit_observed")
        status="success" if complete else supervised["status"]
        if not complete and status=="success":status="invalid_worker_result"
        row={"job":job,"status":status,
             "optimizer_steps":None if censored or not worker else worker.get("optimizer_steps"),
             "optimizer_steps_censored":bool(censored or not worker or worker.get("optimizer_steps") is None),
             "elapsed_wall_seconds":supervised["elapsed_wall_seconds"],
             "sampled_peak_process_group_rss_bytes":supervised["sampled_peak_process_group_rss_bytes"],
             "returncode":supervised["returncode"],"worker_profile":worker,
             "durable_events":read_events(worker_dir/"events.jsonl"),"replacement_attempt":False,
             "separate_micro_profile_not_replacement":parent_outcome is not None}
        journal.append("worker_finished",job_id=job["job_id"],status=status,
                       optimizer_steps=row["optimizer_steps"],elapsed_wall_seconds=row["elapsed_wall_seconds"])
        print(json.dumps({"job_id":job["job_id"],"model":job["model"],"cohort":job["cohort"],
                          "threads":job["threads"],"backend":job["backend"],"status":status,
                          "elapsed_seconds":row["elapsed_wall_seconds"]},ensure_ascii=False),flush=True)
        return row

    try:
        journal.append("phase_registered",phase=phase,attempts=len(jobs),
                       registration_commit=registration_commit,registration_sha256=digest(registration_path))
        for index,job in enumerate(jobs):
            row=launch(job,index)
            results.append(row)
            outcome_path=output/("outcome-%03d.json"%index)
            write_json_new(outcome_path,row)
            if phase=="scaling" and eligible_micro_fallback(row):
                # A distinct seed and a distinct one-batch unit. The failed
                # full-cohort attempt remains in results with all consumed cost.
                micro=launch(micro_job(job),index,parent_outcome=outcome_path)
                micros.append(micro)
                write_json_new(output/("micro-outcome-%03d.json"%index),micro)
        if phase=="kernels":kernel_choice=select_kernel_policy(results,parity_suite_passed=True)
        if phase=="threads":thread_choice=select_thread_policy(results)
        all_rows=results+micros
        summary={"version":VERSION,"phase":phase,"status":"completed_all_attempts",
                 "registration_sha256":plan["registration_sha256"],"registration_commit":registration_commit,
                 "all_requested_jobs_attempted":len(results)==len(jobs),
                 "all_fits_completed":all(r["status"]=="success" for r in results),
                 "results":results,"micro_fallback_results":micros,"plan":plan,
                 "kernel_policy":kernel_choice,"thread_policy":thread_choice,
                 "known_optimizer_steps":sum(r["optimizer_steps"] for r in all_rows if type(r["optimizer_steps"]) is int),
                 "unknown_optimizer_step_workers":sum(r["optimizer_steps"] is None for r in all_rows),
                 "elapsed_workers_sum_seconds":sum(r["elapsed_wall_seconds"] for r in all_rows),
                 "no_final_market_data_read":True,"scientific_HPO_or_NAS_executed":False,
                 "limits":["Profiles do not constitute scientific architecture or hyperparameter selection.",
                           "A one-batch micro profile is distinct from its failed full-cohort attempt.",
                           "Stratified cost estimates and extrapolations require declared assumptions.",
                           "Sampled RSS watchdog is not a hard kernel memory limit."]}
        journal.append("phase_finished",phase=phase,all_fits_completed=summary["all_fits_completed"],
                       known_optimizer_steps=summary["known_optimizer_steps"])
        summary["phase_events"]=read_events(output/"events.jsonl")
        write_json_new(output/"summary.json",summary)
        return summary
    finally:
        journal.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("kernels","threads","scaling"))
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--registration",type=Path)
    parser.add_argument("--registration-commit")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--threads-summary",type=Path)
    parser.add_argument("--kernels-summary",type=Path)
    parser.add_argument("--parity-evidence",type=Path)
    parser.add_argument("--source-root-itransformer",type=Path)
    parser.add_argument("--worker-spec",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker_spec is not None:
        return run_worker(json.loads(args.worker_spec.read_text()))
    if any(getattr(args,name) is None for name in ("phase","data_root","registration","registration_commit","output")):
        parser.error("--phase, --data-root, --registration, --registration-commit and --output NEWDIR are required")
    result=run_campaign(args.phase,args.data_root,args.registration,args.registration_commit,args.output,
                        threads_summary=args.threads_summary,kernels_summary=args.kernels_summary,
                        parity_evidence=args.parity_evidence,source_root_itransformer=args.source_root_itransformer)
    return 0 if result["all_fits_completed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
