"""New-seed CPU-calibrated fixed-reference development variance campaign.

This is a separate preregistered campaign: no old failed attempt is replaced,
no HPO/NAS winner is selected and no final R or final-test claim is produced.
Importing the module neither loads observations nor launches a fit. Each fit
runs in a fresh sequential process with durable receipts and an outer watchdog.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import re
import resource
import sys
import time

from run_e2_reference_variance import (
    digest, json_new, median_epoch_count, safe_load_checkpoint, replay_vectors)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-optimized-preselection-reference-variance-v1"
REGISTRATION_VERSION = "e2-compute-calibration-registration-v1"
PROTOCOL_SHA256 = "e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe"
MODELS = ("TimeMixer", "CNN-1D + LSTM + atención temporal", "GRU32")
SEEDS = (273001, 273002, 273003, 273004, 273005)
PHASES = ("epoch_selection", "fresh_refit")
MEMORY_BYTES = 12 * 1024**3
OLD_REGISTRATION_COMMIT = "cf68db00f4d7c3626bdb517bf922265a3b229ea4"


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{"+str(length)+r"}", value) is not None


def registration(path, expected_sha256, commit):
    """Admit only the exact new registration and fixed development chronology."""
    path = Path(path)
    if not _hex(expected_sha256, 64) or not _hex(commit, 40) or commit == OLD_REGISTRATION_COMMIT:
        raise ValueError("An explicit new registration SHA256 and full commit are required")
    if path.is_symlink() or not path.is_file() or digest(path) != expected_sha256:
        raise ValueError("Registration bytes differ from the explicit preregistered SHA256")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("version") != REGISTRATION_VERSION:
        raise ValueError("A new compute-calibration registration is required")
    if value.get("source_protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Section-27 source protocol differs from the frozen snapshot")
    if digest(ROOT/"book_e2/_static/search_protocol_20261109/protocol.json") != PROTOCOL_SHA256:
        raise ValueError("On-disk fixed-recipe protocol differs from registration")
    expected = {
        "models": list(MODELS), "seed_pool": list(SEEDS),
        "epoch_selection_origin": "2024-10-01", "epoch_selection_train_end": "2024-06-25",
        "epoch_selection_validation": ["2024-07-03", "2024-09-23"],
        "refit_origin": "2024-11-08", "fresh_refit_training_end": "2024-10-31",
        "evaluation": ["2024-11-08", "2024-12-30"], "evaluation_days": 53,
        "max_epochs_per_stage": 200, "patience_epochs": 20,
        "internal_stage_wall_seconds": 600, "external_worker_wall_seconds": 660,
    }
    request = value.get("preselection_variance", {})
    if any(request.get(key) != wanted for key, wanted in expected.items()):
        raise ValueError("New-seed reference domain, chronology or budget differs from registration")
    if value.get("resources", {}).get("sampled_worker_group_rss_limit_bytes") != MEMORY_BYTES:
        raise ValueError("The preregistered RSS watchdog must remain12GiB")
    if not isinstance(value.get("cpu_profile"), dict):
        raise ValueError("Preregistered CPU policy rules are required")
    return value


def _validate_runtime_policy(policy):
    if (not isinstance(policy, dict) or type(policy.get("threads")) is not int
            or policy["threads"] not in (1, 2, 4) or type(policy.get("crop")) is not bool
            or policy.get("kernel_backend") != ("cropped" if policy["crop"] else "dense")
            or any(not _hex(policy.get(key), 64) for key in
                   ("kernels_summary_sha256", "threads_summary_sha256", "registration_sha256"))
            or not _hex(policy.get("registration_commit"),40)):
        raise ValueError("A complete benchmark-derived CPU runtime policy is required")
    return deepcopy(policy)


def _bind_policy_registration(policy, expected_sha256, commit):
    if policy["registration_sha256"] != expected_sha256 or policy["registration_commit"] != commit:
        raise ValueError("CPU benchmark policy belongs to a different registration")


def load_runtime_policy(kernels_path, threads_path):
    # This helper independently recomputes both decisions and their proof chain.
    from run_e2_cpu_scaling import load_runtime_policy as load
    return _validate_runtime_policy(load(kernels_path, threads_path))


def _source_hashes():
    """Hash the same on-disk Python sources before and after every worker."""
    paths = list((ROOT/"scripts").rglob("*.py"))
    paths.append(ROOT/"book_e2/_static/search_protocol_20261109/protocol.json")
    return {str(path.relative_to(ROOT)):digest(path) for path in sorted(paths) if path.is_file()}


def _validate_spec(spec):
    if (not isinstance(spec, dict) or spec.get("version") != VERSION
            or spec.get("model") not in MODELS or type(spec.get("seed")) is not int
            or spec["seed"] not in SEEDS or spec.get("phase") not in PHASES
            or not _hex(spec.get("registration_commit"), 40)
            or spec["registration_commit"] == OLD_REGISTRATION_COMMIT
            or not _hex(spec.get("expected_registration_sha256"), 64)):
        raise ValueError("Worker identity is outside the new preregistered domain")
    _validate_runtime_policy(spec.get("runtime_policy"))
    epochs = spec.get("fixed_epochs")
    if spec["phase"] == "epoch_selection" and epochs is not None:
        raise ValueError("Selection cannot receive refit epochs")
    if spec["phase"] == "fresh_refit" and (type(epochs) is not int or not 1 <= epochs <= 200):
        raise ValueError("Refits require the frozen five-seed median within1..200")


def run_worker(spec):
    """One fresh fixed-reference fit; no implicit retry or selected-weight reuse."""
    _validate_spec(spec)
    output = Path(spec["output"])
    if not output.is_dir() or output.is_symlink() or (output/"worker_profile.json").exists():
        raise ValueError("Use a new supervisor directory without a worker result")
    registered = registration(spec["registration"], spec["expected_registration_sha256"],
                              spec["registration_commit"])
    policy = load_runtime_policy(spec["kernels_summary"], spec["threads_summary"])
    _bind_policy_registration(policy,spec["expected_registration_sha256"],spec["registration_commit"])
    if policy != spec["runtime_policy"]:
        raise ValueError("Runtime summaries changed between supervisor and worker")
    request = registered["preselection_variance"]
    source_before = _source_hashes()
    started = time.monotonic()
    result = {
        "version": VERSION, "status": "started", "model": spec["model"], "seed": spec["seed"],
        "phase": spec["phase"], "registration_commit": spec["registration_commit"],
        "registration_file_sha256": spec["expected_registration_sha256"],
        "runtime_policy": policy, "source_sha256_before": source_before,
        "cpu_kernel_crop_applied": False, "kernel_crop_applicable_to_model": False,
        "preselection_reference_only": True, "hpo_executed": False, "nas_search_executed": False,
        "selected_hpo_recipe": False, "weights_reused": False, "fresh_initialization": True,
        "final_2025_2026_access": False, "replacement_attempt": False,
        "supersedes_previous_attempts": False, "optimizer_steps": 0,
    }
    progress = None
    fitting_started = False
    try:
        import torch
        from e2_development_data import load_development
        from e2_development_models import build_pilot_model, reference_pattern, fixed_pilot_config
        from e2_temporal_training import prepare_b
        from e2_staged_training import fit_supervised, state_sha256
        from e2_training_bindings import CheckpointStore, FittedNeural
        from e2_real_executor import DatedReplay, DatedJournal, run_frozen_block

        progress = DatedJournal(output/"progress")
        progress.append("worker_started", model=spec["model"], seed=spec["seed"], phase=spec["phase"],
                        registration_file_sha256=spec["expected_registration_sha256"],
                        runtime_policy=policy, optimizer_steps=0)
        if source_before != spec["source_sha256_expected"]:
            raise ValueError("Python sources differ from the supervisor's frozen source snapshot")
        torch.set_num_threads(policy["threads"])
        torch.set_num_interop_threads(1)
        result["torch_num_threads"] = torch.get_num_threads()
        result["torch_num_interop_threads"] = torch.get_num_interop_threads()
        panel, data_receipt = load_development(spec["data_root"], admitted_end="2024-12-30")
        result["data_receipt"] = data_receipt
        origin = request["epoch_selection_origin"] if spec["phase"] == "epoch_selection" else request["refit_origin"]
        end = "2024-10-08" if spec["phase"] == "epoch_selection" else request["evaluation"][1]
        prepared = prepare_b(panel, origin, end)
        train = prepared.inner_train.clone() if spec["phase"] == "epoch_selection" else prepared.refit_train.clone()
        validation = prepared.inner_validation.clone() if spec["phase"] == "epoch_selection" else None
        cutoff = request["epoch_selection_train_end"] if validation is not None else request["fresh_refit_training_end"]
        if str(train.dates[-1].date()) != cutoff:
            raise ValueError("Training support differs from the registered fit cutoff")
        if validation is not None:
            if [str(validation.dates[0].date()), str(validation.dates[-1].date())] != request["epoch_selection_validation"]:
                raise ValueError("Epoch-selection validation differs from registration")
        elif (len(prepared.outer_plan.evaluation_dates) != 53 or not all(prepared.outer_plan.eligible)):
            raise ValueError("All53 development evaluation dates require complete common support")
        result["training_split"] = train.record()
        result["validation_split"] = None if validation is None else validation.record()
        pattern = None if spec["model"] == "GRU32" else reference_pattern()
        model = build_pilot_model(spec["model"], pattern, seed=spec["seed"])
        metadata = deepcopy(model.architecture_metadata)
        metadata.update(compute_calibration_registration=spec["registration_commit"],
                        preselection_reference_recipe=True, HPO_or_NAS_winner=False,
                        runtime_policy=policy, cpu_kernel_crop_applied=False)
        model.architecture_metadata = metadata
        config = fixed_pilot_config(spec["seed"], epoch_cap=200, wallcap=600,
                                    memory_limit_bytes=MEMORY_BYTES)
        progress.append("fit_started", optimizer_steps=None, fixed_epochs=spec["fixed_epochs"],
                        training_split_signature=train.signature)
        fitting_started = True
        result["optimizer_steps"] = None
        stage = fit_supervised(model, train.x, train.y,
            None if validation is None else validation.x,
            None if validation is None else validation.y, train.scaler.target_scale,
            config, fixed_epochs=spec["fixed_epochs"])
        phase_record = stage.to_dict()
        result.update(optimizer_steps=stage.optimizer_steps, stage_record=phase_record,
                      best_epoch=stage.best_epoch,
                      fresh_optimizer=stage.optimizer_initial_state_entries == 0)
        json_new(output/"fit_receipt.json", {"optimizer_steps":stage.optimizer_steps,
                 "stage_record":phase_record,"model":spec["model"],"seed":spec["seed"],
                 "phase":spec["phase"],"registration_file_sha256":spec["expected_registration_sha256"]})
        progress.append("fit_completed", optimizer_steps=stage.optimizer_steps,
                        best_epoch=stage.best_epoch, selected_state_sha256=stage.selected_state_sha256)
        if not result["fresh_optimizer"]:
            raise RuntimeError("Optimizer state was reused")
        run_id = VERSION+":"+spec["phase"]+":"+spec["model"]+":seed"+str(spec["seed"])
        store = CheckpointStore(output/"checkpoints")
        checkpoint = store.save(run_id, {
            "version":VERSION, "run_id":run_id, "state_dict":stage.state_dict,
            "model_metadata":metadata, "scaler":train.scaler.record(),
            "phase_epochs":{"supervised":stage.best_epoch}, "phase_records":{"supervised":phase_record},
            "registration_file_sha256":spec["expected_registration_sha256"],
            "runtime_policy":policy,"training_split_signature":train.signature,
            "optimizer_state_saved":False})
        checkpoint_path = store.directory/checkpoint["file"]
        loaded = safe_load_checkpoint(checkpoint_path, checkpoint["sha256"], stage.selected_state_sha256)
        restored = build_pilot_model(spec["model"], pattern, seed=spec["seed"])
        restored.load_state_dict(loaded["state_dict"], strict=True)
        restored.architecture_metadata = metadata
        restored.eval()
        if state_sha256(restored) != stage.selected_state_sha256:
            raise RuntimeError("Restored inference state differs from fitted checkpoint")
        result["checkpoint"] = {**checkpoint,"relative_path":str(checkpoint_path.relative_to(output)),
            "safe_load_weights_only":True,"state_sha256":stage.selected_state_sha256}
        progress.append("checkpoint_verified", sha256=checkpoint["sha256"],
                        selected_state_sha256=stage.selected_state_sha256)
        if validation is not None:
            result.update(status="completed", phase_epochs={"supervised":stage.best_epoch},
                          outer_labels_used_for_epoch_selection=False)
        else:
            artifact = FittedNeural(spec["model"], spec["seed"], restored, train.scaler,
                                    checkpoint, {"supervised":phase_record})
            system = artifact.frozen_system()
            journal = DatedJournal(output/"replay")
            replay = DatedReplay(panel.calendar,panel.features,panel.targets,quality=panel.quality)
            try:
                scores = run_frozen_block(prepared.outer_plan,replay,[system],journal)
            finally:
                journal.close()
            dates = [d.isoformat() for d in prepared.outer_plan.evaluation_dates]
            vectors = replay_vectors(output/"replay/events.jsonl",system.name,dates)
            prediction_path = output/"predictions.csv"
            with prediction_path.open("x",newline="",encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("date_utc","prediction_sigma7","observed_sigma7"))
                writer.writerows(zip(vectors["dates"],vectors["predictions"],vectors["truths"]))
                handle.flush(); os.fsync(handle.fileno())
            result.update(status="completed",phase_epochs={"supervised":stage.best_epoch},
                fixed_epoch_count=spec["fixed_epochs"],evaluation=vectors,
                metrics_original_units=scores["metrics"][system.name],replay=scores,
                replay_ledger_sha256=digest(output/"replay/events.jsonl"),
                predictions_file={"file":"predictions.csv","sha256":digest(prediction_path),"rows":len(dates)},
                frozen_fingerprint_invariant=True,observation_after_durable_prediction=True,
                epoch_selection_inputs_passed_to_refit=False)
        result["model_metadata"] = metadata
    except Exception as error:
        receipt = deepcopy(getattr(error,"receipt",None))
        if isinstance(receipt,dict) and type(receipt.get("optimizer_steps")) is int:
            result["optimizer_steps"] = receipt["optimizer_steps"]
        elif fitting_started and "stage_record" not in result:
            result["optimizer_steps"] = None
        result.update(status="failed",failure={"type":type(error).__name__,"message":str(error),
                                               "stage_receipt":receipt})
    finally:
        source_after = _source_hashes()
        result.update(source_sha256_after=source_after,source_hashes_unchanged=source_after == source_before,
                      elapsed_wall_seconds=time.monotonic()-started,
                      worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if not result["source_hashes_unchanged"]:
            result.update(status="failed",source_integrity_failure="Python sources changed during the worker")
        result["optimizer_steps_known"] = type(result["optimizer_steps"]) is int
        result["limits"] = ["Separate fixed-reference preselection campaign, not HPO/NAS or a replacement.",
            "New seeds and runtime costs do not establish final R, final power or an untouched test.",
            "TimesNet kernel cropping is inapplicable to these three fixed reference systems."]
        if progress is not None:
            progress.append("worker_finished",status=result["status"],optimizer_steps=result["optimizer_steps"],
                            source_hashes_unchanged=result["source_hashes_unchanged"])
            progress.close()
        json_new(output/"worker_profile.json",result)
    return result


def _slug(name):
    return {"TimeMixer":"timemixer","CNN-1D + LSTM + atención temporal":"hybrid","GRU32":"gru32"}[name]


def _known_steps(worker, directory, spec):
    if (isinstance(worker,dict)
            and all(worker.get(key) == spec[key] for key in ("model","seed","phase"))
            and worker.get("registration_file_sha256") == spec["expected_registration_sha256"]):
        steps = worker.get("optimizer_steps")
        if type(steps) is int and steps >= 0:
            return steps
    # An outer kill after fitting may still leave a durable complete stage receipt.
    path = Path(directory)/"fit_receipt.json"
    if path.is_file() and not path.is_symlink():
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError,OSError):
            return None
        if (all(receipt.get(key) == spec[key] for key in ("model","seed","phase"))
                and receipt.get("registration_file_sha256") == spec["expected_registration_sha256"]
                and type(receipt.get("optimizer_steps")) is int and receipt["optimizer_steps"] >= 0):
            return receipt["optimizer_steps"]
    return None


def _costs(records):
    rows = [row for model in records.values() for phase in model.values() for row in phase]
    return {"fit_attempts":len(rows),
            "known_optimizer_steps":sum(row["optimizer_steps"] for row in rows if row["optimizer_steps"] is not None),
            "unknown_optimizer_step_attempts":sum(row["optimizer_steps"] is None for row in rows),
            "known_failed_optimizer_steps":sum(row["optimizer_steps"] for row in rows
                                               if row["status"] != "completed" and row["optimizer_steps"] is not None),
            "elapsed_worker_wall_seconds":sum(row["elapsed_wall_seconds"] for row in rows)}


def run_optimized_variance(data_root, registration_path, output, *, registration_commit,
                           expected_registration_sha256, kernels_summary, threads_summary):
    """Run exactly15 inner attempts, then eligible15 fresh refits, sequentially."""
    from profile_e2_training import run_isolated_profile, hardware_snapshot
    from e2_real_executor import DatedJournal
    value = registration(registration_path,expected_registration_sha256,registration_commit)
    policy = load_runtime_policy(kernels_summary,threads_summary)
    _bind_policy_registration(policy,expected_registration_sha256,registration_commit)
    source_before = _source_hashes()
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("The new campaign output directory must be new")
    root.mkdir(parents=True)
    journal = DatedJournal(root/"pilot_journal")
    plans = value["preselection_variance"]
    json_new(root/"request.json", {"version":VERSION,"registration_commit":registration_commit,
        "registration_file_sha256":expected_registration_sha256,"runtime_policy":policy,
        "data_root":str(Path(data_root).resolve()),"models":list(MODELS),"seeds":list(SEEDS),
        "resources":deepcopy(value["resources"]),"preselection_variance":deepcopy(plans),
        "hardware":hardware_snapshot(),"source_sha256_before":source_before,
        "supersedes_previous_attempts":False})
    records = {name:{"selection":[],"refit":[]} for name in MODELS}
    medians, blocked = {}, {}
    try:
        journal.append("pilot_registration",registration_commit=registration_commit,
            registration_file_sha256=expected_registration_sha256,runtime_policy=policy,
            model_count=3,seeds=list(SEEDS),HPO=False,NAS=False,replacement_attempt=False)
        for phase in PHASES:
            for name in MODELS:
                if phase == "fresh_refit" and name in blocked:
                    journal.append("model_refits_blocked",model=name,reason=blocked[name],replacement_attempt=False)
                    continue
                for seed in SEEDS:
                    label = _slug(name)+"-"+phase+"-seed"+str(seed)
                    directory = root/label
                    spec = {"version":VERSION,"model":name,"seed":seed,"phase":phase,
                        "fixed_epochs":medians.get(name) if phase == "fresh_refit" else None,
                        "registration":str(Path(registration_path).resolve()),"registration_commit":registration_commit,
                        "expected_registration_sha256":expected_registration_sha256,"runtime_policy":policy,
                        "source_sha256_expected":source_before,
                        "kernels_summary":str(Path(kernels_summary).resolve()),
                        "threads_summary":str(Path(threads_summary).resolve()),
                        "data_root":str(Path(data_root).resolve()),"output":str(directory.resolve())}
                    _validate_spec(spec)
                    spec_path = root/(label+".json"); json_new(spec_path,spec)
                    journal.append("worker_requested",model=name,seed=seed,phase=phase,
                                   spec_sha256=digest(spec_path),fixed_epochs=spec["fixed_epochs"])
                    supervised = run_isolated_profile(
                        [sys.executable,str(Path(__file__).resolve()),"--worker-spec",str(spec_path.resolve())],
                        directory,timeout_seconds=660,memory_limit_bytes=MEMORY_BYTES,cwd=ROOT)
                    worker = supervised.get("worker_profile")
                    if not isinstance(worker,dict): worker = None
                    completed = (supervised["status"] == "success" and worker is not None
                        and worker.get("status") == "completed" and worker.get("source_hashes_unchanged") is True
                        and all(worker.get(key) == spec[key] for key in ("model","seed","phase","registration_commit"))
                        and worker.get("registration_file_sha256") == expected_registration_sha256
                        and worker.get("runtime_policy") == policy
                        and worker.get("torch_num_threads") == policy["threads"]
                        and worker.get("torch_num_interop_threads") == 1)
                    record = {"model":name,"seed":seed,"phase":phase,
                        "status":"completed" if completed else "failed","supervisor_status":supervised["status"],
                        "worker_result_file":str((directory/"worker_profile.json").relative_to(root)) if worker else None,
                        "supervisor_result_file":str((directory/"result.json").relative_to(root)),
                        "elapsed_wall_seconds":supervised["elapsed_wall_seconds"],
                        "sampled_peak_process_group_rss_bytes":supervised["sampled_peak_process_group_rss_bytes"],
                        "optimizer_steps":_known_steps(worker,directory,spec),
                        "best_epoch":worker.get("best_epoch") if completed else None,
                        "worker_profile":worker,"replacement_attempt":False}
                    records[name]["selection" if phase == "epoch_selection" else "refit"].append(record)
                    json_new(root/(label+"-receipt.json"),record)
                    journal.append("worker_finished",model=name,seed=seed,phase=phase,
                        status=record["status"],supervisor_status=record["supervisor_status"],
                        optimizer_steps=record["optimizer_steps"],elapsed_wall_seconds=record["elapsed_wall_seconds"],
                        replacement_attempt=False)
                if phase == "epoch_selection":
                    try:
                        # Never use the old helper's default seed pool.
                        medians[name] = median_epoch_count(records[name]["selection"],seeds=SEEDS)
                        journal.append("epoch_count_frozen",model=name,ceil_median_supervised_epochs=medians[name],
                                       seed_count=5,recipe_or_architecture_selected=False)
                    except ValueError as error:
                        blocked[name] = str(error)
        source_after = _source_hashes()
        complete = (source_after == source_before and all(len(records[n]["refit"]) == 5
                    and all(row["status"] == "completed" for row in records[n]["refit"]) for n in MODELS))
        summary = {"version":VERSION,"status":"complete" if complete else "incomplete",
            "registration_commit":registration_commit,"registration_file_sha256":expected_registration_sha256,
            "runtime_policy":policy,"preselection_reference_only":True,"HPO_executed":False,
            "NAS_search_executed":False,"fixed_epoch_counts":medians,"blocked_refits":blocked,
            "models":records,"paired_3x5_complete":complete,"resource_costs":_costs(records),
            "replacement_attempts":0,"supersedes_previous_attempts":False,"final_2025_2026_access":False,
            "final_R_set":False,"final_power_claimed":False,"source_sha256_before":source_before,
            "source_sha256_after":source_after,"source_hashes_unchanged":source_after == source_before,
            "limits":["A model missing any of five inner fits cannot enter fresh refits.",
                      "Failures and known/unknown optimizer costs remain part of this separate campaign.",
                      "Reference variation cannot establish postselection power or final R."]}
        json_new(root/"summary.json",summary)
        journal.append("pilot_completed",status=summary["status"],paired_complete=complete,
                       resource_costs=summary["resource_costs"],final_R_set=False)
        return summary
    finally:
        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--registration",type=Path)
    parser.add_argument("--expected-registration-sha256")
    parser.add_argument("--registration-commit")
    parser.add_argument("--kernels-summary",type=Path)
    parser.add_argument("--threads-summary",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--worker-spec",type=Path)
    args = parser.parse_args()
    if args.worker_spec is not None:
        result = run_worker(json.loads(args.worker_spec.read_text(encoding="utf-8")))
    else:
        required = ("data_root","registration","expected_registration_sha256","registration_commit",
                    "kernels_summary","threads_summary","output")
        if any(getattr(args,key) is None for key in required):
            parser.error("Explicit data, registration hash/commit, both runtime summaries and new output are required")
        result = run_optimized_variance(args.data_root,args.registration,args.output,
            registration_commit=args.registration_commit,expected_registration_sha256=args.expected_registration_sha256,
            kernels_summary=args.kernels_summary,threads_summary=args.threads_summary)
        print(json.dumps({"status":result["status"],"fixed_epoch_counts":result["fixed_epoch_counts"],
                          "paired_complete":result["paired_3x5_complete"]},ensure_ascii=False))
    return 0 if result["status"] in ("complete","completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
