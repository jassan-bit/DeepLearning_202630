"""Preregistered fixed-reference preselection variance pilot, not HPO/NAS.

The parent process launches exactly the registered seed/model attempts in
isolated sequential workers. Only hash-pinned 2023--2024 development derivatives
are admitted. There is no final-data path, replacement seed or selected-weight
reuse. Calling this module does not itself authorise fitting.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-preselection-reference-variance-v1"
REGISTRATION_COMMIT = "cf68db00f4d7c3626bdb517bf922265a3b229ea4"
REGISTRATION_SHA256 = "cc10ddba82f0c59cfcd8caf70e713481ec0701e71860862ed13ead0df0bf4797"
MODELS = ("TimeMixer", "CNN-1D + LSTM + atención temporal", "GRU32")
SEEDS = (271901, 271902, 271903, 271904, 271905)
PHASES = ("epoch_selection", "fresh_refit")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_new(path, value):
    """Exclusive fsync JSON; a partial temporary file never becomes a result."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists():
        raise FileExistsError("Refusing to replace an existing pilot record")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def registration(path, commit=REGISTRATION_COMMIT):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or digest(path) != REGISTRATION_SHA256:
        raise ValueError("Use the exact preregistered pilot registration file")
    if commit != REGISTRATION_COMMIT:
        raise ValueError("Registration commit differs from the preregistered version")
    value = json.loads(path.read_text(encoding="utf-8"))
    p = value.get("preselection_variance", {})
    expected = {
        "models": list(MODELS), "seed_pool": list(SEEDS),
        "epoch_selection_origin": "2024-10-01",
        "epoch_selection_train_end": "2024-06-25",
        "epoch_selection_validation": ["2024-07-03", "2024-09-23"],
        "refit_origin": "2024-11-08", "fresh_refit_training_end": "2024-10-31",
        "evaluation": ["2024-11-08", "2024-12-30"], "evaluation_days": 53,
        "max_epochs_per_stage": 200, "patience_epochs": 20,
        "internal_stage_wall_seconds": 150, "external_worker_wall_seconds": 180,
    }
    if any(p.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("Reference-pilot domain or chronology differs from registration")
    if value["resources"]["sampled_worker_group_rss_limit_bytes"] != 12*1024**3:
        raise ValueError("Reference-pilot RSS watchdog differs from registration")
    return value


def median_epoch_count(records, seeds=SEEDS):
    """All five distinct successful attempts are mandatory; no survivor median."""
    records = list(records)
    if (len(records) != len(seeds)
            or {row.get("seed") for row in records} != set(seeds)
            or any(row.get("status") != "completed" for row in records)):
        raise ValueError("Five distinct completed selection attempts are required")
    epochs = []
    for row in records:
        value = row.get("best_epoch")
        if type(value) is not int or not 1 <= value <= 200:
            raise ValueError("Every strict-best selection epoch must be within1..200")
        epochs.append(value)
    return int(math.ceil(statistics.median(epochs)))


def safe_load_checkpoint(path, expected_sha256, expected_state_sha256):
    """Load only tensor/primitive checkpoints whose exact bytes are known."""
    import torch
    from e2_staged_training import state_sha256
    path = Path(path)
    if path.is_symlink() or not path.is_file() or digest(path) != expected_sha256:
        raise ValueError("Checkpoint bytes changed before safe load")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict)
            or any(not isinstance(v, torch.Tensor) for v in payload["state_dict"].values())):
        raise ValueError("Explicit tensor state dictionary required")
    if state_sha256(payload["state_dict"]) != expected_state_sha256:
        raise ValueError("Checkpoint tensor state differs from selected state")
    if payload.get("optimizer_state_saved") is not False:
        raise ValueError("Pilot inference checkpoints must not reuse optimizer state")
    return payload


def replay_vectors(events_path, system_name, expected_dates):
    """Extract paired values only from a completed prediction-before-observe log."""
    wanted = list(expected_dates)
    predictions, observations, pending = {}, {}, {}
    completed = False
    previous, sequence = "0"*64, 0
    for line in Path(events_path).read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        given = event.get("event_sha256")
        plain = {key:value for key,value in event.items() if key!="event_sha256"}
        actual = hashlib.sha256(json.dumps(plain,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if (event.get("sequence")!=sequence or event.get("previous_sha256")!=previous
                or given!=actual):
            raise ValueError("Pilot replay hash chain changed")
        previous, sequence = given, sequence+1
        if event["kind"] == "predictions":
            date = event["date"]
            if date in predictions or system_name not in event["systems"]:
                raise ValueError("Duplicate or incomplete prediction event")
            value = event["systems"][system_name]["value"]
            if event["eligible"] is not True or value is None or not math.isfinite(value):
                raise ValueError("The registered53 dates require complete common support")
            predictions[date] = float(value)
            pending[date] = event["event_sha256"]
        elif event["kind"] == "observation":
            date = event["date"]
            if (date not in pending or date in observations
                    or event["prediction_event_sha256"] != pending[date]):
                raise ValueError("Observation appeared without its preceding durable prediction")
            value = event["value"]
            if value is None or not math.isfinite(value):
                raise ValueError("The registered53 dates require a finite observation")
            observations[date] = float(value)
            pending.pop(date)
        elif event["kind"] == "block_completed":
            completed = True
        elif event["kind"] == "block_failed":
            raise ValueError("A failed replay cannot supply complete pilot vectors")
    if (not completed or list(predictions) != wanted or list(observations) != wanted):
        raise ValueError("Incomplete or reordered pilot replay")
    return {"dates": wanted, "predictions": [predictions[d] for d in wanted],
            "truths": [observations[d] for d in wanted]}


def _source_hashes():
    scripts = (ROOT/"scripts").resolve()
    result = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(scripts) and path.is_file():
            result[str(path.relative_to(ROOT))] = digest(path)
    return dict(sorted(result.items()))


def _validate_spec(spec):
    if (not isinstance(spec, dict) or spec.get("version") != VERSION
            or spec.get("model") not in MODELS or spec.get("seed") not in SEEDS
            or spec.get("phase") not in PHASES
            or spec.get("registration_commit") != REGISTRATION_COMMIT):
        raise ValueError("Worker identity is outside the preregistered domain")
    epochs = spec.get("fixed_epochs")
    if spec["phase"] == "epoch_selection" and epochs is not None:
        raise ValueError("Selection cannot receive a fixed refit epoch count")
    if spec["phase"] == "fresh_refit" and (type(epochs) is not int or not 1 <= epochs <= 200):
        raise ValueError("Fresh refit requires the frozen1..200 epoch count")


def run_worker(spec):
    """One selection/refit fit. The outer supervisor supplies wall/RSS enforcement."""
    _validate_spec(spec)
    output = Path(spec["output"])
    if not output.is_dir() or (output/"worker_profile.json").exists():
        raise ValueError("Use an existing new supervisor directory without worker result")
    registered = registration(spec["registration"], spec["registration_commit"])
    request = registered["preselection_variance"]
    started = time.monotonic()
    result = {"version": VERSION, "status": "started", "model": spec["model"],
              "seed": spec["seed"], "phase": spec["phase"],
              "registration_commit": REGISTRATION_COMMIT,
              "registration_file_sha256": REGISTRATION_SHA256,
              "hpo_executed": False, "nas_search_executed": False,
              "selected_hpo_recipe": False, "weights_reused": False,
              "final_2025_2026_access": False, "replacement_attempt": False,
              "optimizer_steps": 0}
    try:
        import pandas as pd
        import torch
        from e2_development_data import load_development
        from e2_development_models import (build_pilot_model, reference_pattern,
                                            fixed_pilot_config)
        from e2_temporal_training import prepare_b
        from e2_staged_training import fit_supervised, state_sha256
        from e2_training_bindings import CheckpointStore, FittedNeural
        from e2_real_executor import DatedReplay, DatedJournal, run_frozen_block

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        panel, data_receipt = load_development(spec["data_root"], admitted_end="2024-12-30")
        result["data_receipt"] = data_receipt
        origin = request["epoch_selection_origin"] if spec["phase"] == "epoch_selection" else request["refit_origin"]
        end = ("2024-10-08" if spec["phase"] == "epoch_selection" else request["evaluation"][1])
        prepared = prepare_b(panel, origin, end)
        train = prepared.inner_train.clone() if spec["phase"] == "epoch_selection" else prepared.refit_train.clone()
        validation = prepared.inner_validation.clone() if spec["phase"] == "epoch_selection" else None
        expected_train_end = request["epoch_selection_train_end"] if validation is not None else request["fresh_refit_training_end"]
        if str(train.dates[-1].date()) != expected_train_end:
            raise ValueError("Training support exceeds the registered fit cutoff")
        if validation is not None:
            actual = [str(validation.dates[0].date()), str(validation.dates[-1].date())]
            if actual != request["epoch_selection_validation"]:
                raise ValueError("Epoch selection validation differs from registration")
        elif (len(prepared.outer_plan.evaluation_dates) != request["evaluation_days"]
              or not all(prepared.outer_plan.eligible)):
            raise ValueError("All53 registered evaluation dates must have common support")

        pattern = None if spec["model"] == "GRU32" else reference_pattern()
        model = build_pilot_model(spec["model"], pattern, seed=spec["seed"])
        metadata = deepcopy(model.architecture_metadata)
        metadata.update(reference_variance_registration=REGISTRATION_COMMIT,
                        preselection_reference_recipe=True, HPO_or_NAS_winner=False)
        model.architecture_metadata = metadata
        config = fixed_pilot_config(spec["seed"], epoch_cap=request["max_epochs_per_stage"],
            wallcap=request["internal_stage_wall_seconds"],
            memory_limit_bytes=registered["resources"]["sampled_worker_group_rss_limit_bytes"])
        # No epoch-selection inputs are passed to fresh refits.
        stage = fit_supervised(model, train.x, train.y,
            None if validation is None else validation.x,
            None if validation is None else validation.y, train.scaler.target_scale,
            config, fixed_epochs=spec.get("fixed_epochs"))
        phase_record = stage.to_dict()
        result["optimizer_steps"] = stage.optimizer_steps
        result["stage_record"] = phase_record
        result["best_epoch"] = stage.best_epoch
        result["training_split"] = train.record()
        result["validation_split"] = None if validation is None else validation.record()
        result["fresh_optimizer"] = stage.optimizer_initial_state_entries == 0
        if not result["fresh_optimizer"]:
            raise RuntimeError("Optimizer state was reused")

        run_id = VERSION+":"+spec["phase"]+":"+spec["model"]+":seed"+str(spec["seed"])
        store = CheckpointStore(output/"checkpoints")
        checkpoint = store.save(run_id, {
            "version": VERSION, "run_id": run_id, "state_dict": stage.state_dict,
            "model_metadata": metadata, "scaler": train.scaler.record(),
            "phase_epochs": {"supervised": stage.best_epoch},
            "phase_records": {"supervised": phase_record},
            "registration_file_sha256": REGISTRATION_SHA256,
            "training_split_signature": train.signature, "optimizer_state_saved": False})
        checkpoint_path = store.directory/checkpoint["file"]
        loaded = safe_load_checkpoint(checkpoint_path, checkpoint["sha256"], stage.selected_state_sha256)
        restored = build_pilot_model(spec["model"], pattern, seed=spec["seed"])
        restored.load_state_dict(loaded["state_dict"], strict=True)
        restored.architecture_metadata = metadata
        restored.eval()
        if state_sha256(restored) != stage.selected_state_sha256:
            raise RuntimeError("Restored inference state differs from fitted checkpoint")
        result["checkpoint"] = {**checkpoint, "relative_path": str(checkpoint_path.relative_to(output)),
                                "safe_load_weights_only": True,
                                "state_sha256": stage.selected_state_sha256}
        if validation is not None:
            result["status"] = "completed"
            result["phase_epochs"] = {"supervised": stage.best_epoch}
            result["outer_labels_used_for_epoch_selection"] = False
        else:
            artifact = FittedNeural(spec["model"], spec["seed"], restored, train.scaler,
                                    checkpoint, {"supervised": phase_record})
            system = artifact.frozen_system()
            journal = DatedJournal(output/"replay")
            replay = DatedReplay(panel.calendar, panel.features, panel.targets, quality=panel.quality)
            try:
                scores = run_frozen_block(prepared.outer_plan, replay, [system], journal)
            finally:
                journal.close()
            expected_dates = [d.isoformat() for d in prepared.outer_plan.evaluation_dates]
            vectors = replay_vectors(output/"replay/events.jsonl", system.name, expected_dates)
            prediction_path = output/"predictions.csv"
            with prediction_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(("date_utc", "prediction_sigma7", "observed_sigma7"))
                writer.writerows(zip(vectors["dates"], vectors["predictions"], vectors["truths"]))
                handle.flush()
                os.fsync(handle.fileno())
            result.update(status="completed", phase_epochs={"supervised": stage.best_epoch},
                fixed_epoch_count=spec["fixed_epochs"], evaluation=vectors,
                metrics_original_units=scores["metrics"][system.name],
                replay=scores, replay_ledger_sha256=digest(output/"replay/events.jsonl"),
                predictions_file={"file":"predictions.csv","sha256":digest(prediction_path),
                                  "rows":len(vectors["dates"])},
                frozen_fingerprint_invariant=True, observation_after_durable_prediction=True,
                epoch_selection_inputs_passed_to_refit=False)
        result["model_metadata"] = metadata
    except Exception as error:
        receipt = deepcopy(getattr(error, "receipt", None))
        if isinstance(receipt, dict):
            result["optimizer_steps"] = receipt.get("optimizer_steps")
        result.update(status="failed", failure={"type":type(error).__name__, "message":str(error),
                                              "stage_receipt":receipt})
    finally:
        result["elapsed_wall_seconds"] = time.monotonic()-started
        result["worker_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        result["source_sha256"] = _source_hashes()
        result["limits"] = [
            "Fixed-reference preselection pilot; not selected recipes, HPO or NAS.",
            "Five-seed variation is conditional on this recipe, support and53 development dates.",
            "No final R, final power or untouched final-test claim is established.",
            "Per-stage boundary checks supplement the outer process watchdog."]
        json_new(output/"worker_profile.json", result)
    return result


def _slug(name):
    return {"TimeMixer":"timemixer", "CNN-1D + LSTM + atención temporal":"hybrid",
            "GRU32":"gru32"}[name]


def run_reference_variance(data_root, registration_path, output, *,
                           registration_commit=REGISTRATION_COMMIT):
    """Execute all preregistered attempts sequentially, without replacing failures."""
    from profile_e2_training import run_isolated_profile, hardware_snapshot
    from e2_real_executor import DatedJournal
    value = registration(registration_path, registration_commit)
    root = Path(output)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Reference-pilot output directory must be new")
    root.mkdir(parents=True)
    journal = DatedJournal(root/"pilot_journal")
    plans = value["preselection_variance"]
    checkpoint = {"version":VERSION, "registration_commit":REGISTRATION_COMMIT,
                  "registration_file_sha256":REGISTRATION_SHA256,
                  "data_root":str(Path(data_root).resolve()),
                  "models":list(MODELS), "seeds":list(SEEDS),
                  "resource_caps":deepcopy(value["resources"]),
                  "preselection_variance":deepcopy(plans), "hardware":hardware_snapshot(),
                  "source_sha256":_source_hashes()}
    json_new(root/"request.json", checkpoint)
    records, medians, blocked = {name:{"selection":[],"refit":[]} for name in MODELS}, {}, {}
    try:
        journal.append("pilot_registration", registration_commit=REGISTRATION_COMMIT,
                       registration_file_sha256=REGISTRATION_SHA256,
                       model_count=3, seeds=list(SEEDS), HPO=False, NAS=False)
        for phase in PHASES:
            for name in MODELS:
                if phase == "fresh_refit" and name in blocked:
                    journal.append("model_refits_blocked", model=name, reason=blocked[name],
                                   replacement_attempt=False)
                    continue
                for seed in SEEDS:
                    label=_slug(name)+"-"+phase+"-seed"+str(seed)
                    directory=root/label
                    spec={"version":VERSION,"model":name,"seed":seed,"phase":phase,
                          "fixed_epochs":medians.get(name) if phase=="fresh_refit" else None,
                          "registration":str(Path(registration_path).resolve()),
                          "registration_commit":REGISTRATION_COMMIT,
                          "data_root":str(Path(data_root).resolve()), "output":str(directory.resolve())}
                    spec_path=root/(label+".json");json_new(spec_path,spec)
                    journal.append("worker_requested", model=name, seed=seed, phase=phase,
                                   spec_sha256=digest(spec_path), fixed_epochs=spec["fixed_epochs"])
                    supervised=run_isolated_profile(
                        [sys.executable,str(Path(__file__).resolve()),"--worker-spec",str(spec_path.resolve())],
                        directory, timeout_seconds=plans["external_worker_wall_seconds"],
                        memory_limit_bytes=value["resources"]["sampled_worker_group_rss_limit_bytes"],
                        cwd=ROOT)
                    worker=supervised.get("worker_profile")
                    if not isinstance(worker,dict):
                        worker=None
                    completed=(supervised["status"]=="success" and isinstance(worker,dict)
                               and worker.get("status")=="completed"
                               and worker.get("model")==name and worker.get("seed")==seed
                               and worker.get("phase")==phase
                               and worker.get("registration_file_sha256")==REGISTRATION_SHA256)
                    record={"model":name,"seed":seed,"phase":phase,
                            "status":"completed" if completed else "failed",
                            "supervisor_status":supervised["status"],
                            "worker_result_file":str((directory/"worker_profile.json").relative_to(root))
                                                  if worker is not None else None,
                            "supervisor_result_file":str((directory/"result.json").relative_to(root)),
                            "elapsed_wall_seconds":supervised["elapsed_wall_seconds"],
                            "sampled_peak_process_group_rss_bytes":
                                supervised["sampled_peak_process_group_rss_bytes"],
                            "optimizer_steps":worker.get("optimizer_steps") if worker else None,
                            "best_epoch":worker.get("best_epoch") if completed else None,
                            "worker_profile":worker, "replacement_attempt":False}
                    records[name]["selection" if phase=="epoch_selection" else "refit"].append(record)
                    journal.append("worker_finished", model=name,seed=seed,phase=phase,
                        status=record["status"],supervisor_status=record["supervisor_status"],
                        optimizer_steps=record["optimizer_steps"],
                        elapsed_wall_seconds=record["elapsed_wall_seconds"],
                        replacement_attempt=False)
                if phase == "epoch_selection":
                    try:
                        medians[name]=median_epoch_count(records[name]["selection"])
                        journal.append("epoch_count_frozen",model=name,
                            ceil_median_supervised_epochs=medians[name],seed_count=5,
                            recipe_or_architecture_selected=False)
                    except ValueError as error:
                        blocked[name]=str(error)
        paired_complete=all(len(records[n]["refit"])==5
                            and all(r["status"]=="completed" for r in records[n]["refit"])
                            for n in MODELS)
        summary={"version":VERSION, "status":"complete" if paired_complete else "incomplete",
            "registration_commit":REGISTRATION_COMMIT,
            "registration_file_sha256":REGISTRATION_SHA256,
            "preselection_reference_only":True, "HPO_executed":False,
            "NAS_search_executed":False,"fixed_epoch_counts":medians,
            "blocked_refits":blocked,"models":records,"paired_3x5_complete":paired_complete,
            "replacement_attempts":0,"final_2025_2026_access":False,
            "final_R_set":False,"final_power_claimed":False,
            "source_sha256":_source_hashes(),
            "limits":["No failed seed is replaced or silently removed.",
                      "A model missing any of five selection fits has no refit stage.",
                      "Reference-only results cannot close postselection power or final R.",
                      "Worker wall/RSS censoring is preserved with unknown optimizer counts null."]}
        json_new(root/"summary.json",summary)
        journal.append("pilot_completed",status=summary["status"],paired_complete=paired_complete,
                       replacement_attempts=0,final_R_set=False)
        return summary
    finally:
        journal.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--registration",type=Path)
    parser.add_argument("--registration-commit",default=REGISTRATION_COMMIT)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--worker-spec",type=Path)
    args=parser.parse_args()
    if args.worker_spec is not None:
        result=run_worker(json.loads(args.worker_spec.read_text()))
        return 0 if result["status"]=="completed" else 1
    if args.data_root is None or args.registration is None or args.output is None:
        parser.error("--data-root, --registration and --output are required")
    result=run_reference_variance(args.data_root,args.registration,args.output,
                                  registration_commit=args.registration_commit)
    print(json.dumps({"status":result["status"],"fixed_epoch_counts":result["fixed_epoch_counts"],
                      "paired_complete":result["paired_3x5_complete"]},ensure_ascii=False))
    return 0 if result["status"]=="complete" else 1


if __name__=="__main__":
    raise SystemExit(main())

