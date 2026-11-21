"""Prospective, development-only V3 pilot. Prepared with disclosed AI assistance.

No network, market-data download, fitting or data admission occurs on import.
Only training_* NPZ members are admitted by the original hash-bound loader.
All final fits finish before this engine evaluates the development assessment.
This pilot cannot certify novelty, R1, R3 or confirmatory generalization.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import tempfile
import time

VERSION = "e3-v3-controlled-development-pilot-1"
NEURAL = ("additive_positive_head", "atom_crps", "atom_mse_only", "fixed_probs", "atom_plugin")
PARAMETERS = dict(zip(NEURAL, (1210, 1262, 1262, 1236, 1262)))
ANALYTICAL = ("floor_only", "floor_plus_mean_excess")
LEARNING_RATES = (.0003, .001, .003)
HPO_SEED = 20261117
FINAL_SEEDS = (20261118, 20261119, 20261120)
CONFIG = {"weight_decay": 0., "batch_size": 32, "max_epochs": 40,
          "patience": 8, "gradient_clip": 1., "lambda_crps": .1}
SOURCE_FILES = ("src/original_model.py", "src/e3_development.py", "src/e3_innovation_model.py", "src/e3_overlap_model_v3.py",
                "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json",
                "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json.sha256",
                "experiments/e3_evaluacion_ampliada_20261119/evaluation/phase.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        require(json.loads(path.read_text()) == value, "Immutable record changed: " + str(path))
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".partial-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path, rows):
    require(bool(rows), "Empty result table refused")
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def emit(event, **fields):
    print(canonical({"event": event, **fields}), flush=True)


def durability_barrier(directory, stage, job, binding):
    """Token-free handshake: launcher verifies a remote save, then acknowledges."""
    if directory is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    barrier_id = digest({"stage": stage, "job": job, "binding": binding})
    ack_path = directory / (barrier_id + ".json")
    emit("durability_barrier", barrier_id=barrier_id, stage=stage, job=job, binding=binding)
    started = time.monotonic()
    while not ack_path.exists():
        require(time.monotonic() - started < 600., "Remote durability acknowledgement timed out; preserve run")
        time.sleep(.2)
    ack = json.loads(ack_path.read_text())
    require(ack.get("barrier_id") == barrier_id and ack.get("remote_verified") is True,
            "Remote durability acknowledgement invalid")


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Cannot load source: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_identity(source_root, plugin_path, source_commit):
    require(bool(re.fullmatch(r"[0-9a-f]{40}", source_commit)), "Exact source commit required")
    source_root, plugin_path = Path(source_root).resolve(), Path(plugin_path).resolve()
    require(all((source_root / name).is_file() for name in SOURCE_FILES), "Frozen model/phase source missing")
    return {"source_commit": source_commit,
            "source_files": {name: sha(source_root / name) for name in SOURCE_FILES},
            "engine_sha256": sha(__file__), "plugin_sha256": sha(plugin_path)}


def cohort(device):
    import numpy as np
    import torch
    selected = torch.device(device)
    require(selected.type in ("cpu", "cuda"), "Only explicit CPU or CUDA devices admitted")
    result = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
              "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "device": str(selected), "cpu_threads": torch.get_num_threads(),
              "deterministic_algorithms": True, "allow_tf32": False}
    if selected.type == "cuda":
        require(torch.cuda.is_available(), "Requested CUDA unavailable")
        props = torch.cuda.get_device_properties(selected)
        result["GPU"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                         "capability": list(torch.cuda.get_device_capability(selected))}
    return result


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def split_development(data):
    """Strict dates and raw-support separation: 54 selection,7 purge,55 assessment."""
    import numpy as np
    require(len(data["train_X"]) == 532 and len(data["val_X"]) == 116,
            "D00 development geometry must be fit532/validation116")
    dates = np.asarray(data["val_dates"]).astype("datetime64[D]")
    require(np.all(np.diff(dates).astype(int) > 0), "Validation dates must increase")
    select, purged, assessment = np.arange(54), np.arange(54, 61), np.arange(61, 116)
    require(int((dates[assessment[0]] - dates[select[-1]]).astype(int)) >= 8,
            "Selection/assessment windows share raw support")
    return {"select": select, "purged": purged, "assessment": assessment}


def load_data(source_root, npz_path, expected_sha256):
    sys.path.insert(0, str(Path(source_root).resolve()))
    from src.e3_innovation_model import load_innovation_data
    data = load_innovation_data(npz_path, expected_sha256)
    split_development(data)
    require(data["admission"]["keys_read"] == ["training_raw", "training_y_original", "training_dates"]
            and not data["admission"]["forbidden_arrays_loaded"], "Training-only admission violated")
    return data


def prepare(output, source_root, plugin_path, npz_path, npz_sha256, source_commit, device):
    """Seal all identities before a remote publication and before any optimizer fit."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    require(not output.is_symlink(), "Symlink run directory refused")
    hardware = output / "hardware_profile.json"
    require(hardware.is_file(), "Launcher must persist hardware_profile.json before preparation")
    launcher_identity_path = output / "launcher_identity.json"
    launcher_lock_path = output / "launcher_lock.json"
    require(launcher_identity_path.is_file() and launcher_lock_path.is_file(),
            "Launcher identity and source/runtime lock must precede preparation")
    launcher_identity = json.loads(launcher_identity_path.read_text())
    require(isinstance(launcher_identity.get("master_run_id"), str) and bool(launcher_identity["master_run_id"]),
            "Launcher master_run_id required")
    identity = source_identity(source_root, plugin_path, source_commit)
    current = cohort(device)
    data = load_data(source_root, npz_path, npz_sha256)
    split = split_development(data)
    admission = {**data["admission"], "dataset": "D00_BTC", "scope": "exposed_development_only",
                 "selection_dates": data["val_dates"][split["select"]].tolist(),
                 "purged_dates": data["val_dates"][split["purged"]].tolist(),
                 "assessment_dates": data["val_dates"][split["assessment"]].tolist(),
                 "assessment_previously_exposed_calendar": True, "D01_D02_D03_used": False,
                 "calibration_arrays_loaded": False, "test_arrays_loaded": False}
    old = json.loads((output / "plan.json").read_text()) if (output / "plan.json").exists() else None
    if old:
        require(old["master_run_id"] == launcher_identity["master_run_id"], "Launcher master identity changed")
    plan = {"version": VERSION, "master_run_id": launcher_identity["master_run_id"],
            "created_utc": old["created_utc"] if old else datetime.now(timezone.utc).isoformat(),
            "scope": "development_only_not_confirmation", "identity": identity,
            "hardware_profile_sha256": sha(hardware), "cohort": current,
            "launcher_lock_sha256": sha(launcher_lock_path),
            "launcher_identity_sha256": sha(launcher_identity_path),
            "admission_sha256": digest(admission), "neural_variants": list(NEURAL),
            "expected_parameters": PARAMETERS, "analytical_controls": list(ANALYTICAL),
            "fit_rows": 532, "selection_rows": 54, "purged_rows": 7, "assessment_rows": 55,
            "hpo_learning_rates": list(LEARNING_RATES), "hpo_seed": HPO_SEED,
            "final_seeds": list(FINAL_SEEDS), "config": CONFIG, "budget_seconds_per_fit": 120.,
            "required_hpo_fits": 15, "required_final_fits": 15, "total_required_fits": 30,
            "nominal_fit_budget_seconds": 3600., "optimizer": "AdamW",
            "budget_scope": "bounded_optimizer_phase_from_immutable_marker; model_construction_and_tensor_setup_reported_separately",
            "durability_wait_consumes_immutable_fit_opportunity": True,
            "slow_before_fit_publication_policy": "expired phase fails without a validated trained checkpoint; deadline never reset",
            "selection_metric": "selection_RMSE_raw_sigma7_common_to_all_variants",
            "selection_tie_break": "lowest_learning_rate",
            "final_fit_early_stopping": "same selection54; assessment55 never used by optimizer",
            "assessment_policy": "after_all_HPO_and_final_fits_complete; no_retuning",
            "heuristic": {"minimum_relative_RMSE_gain": .02,
                          "against": "best_neural_control_and_best_all_controls",
                          "meaning": "development continuation only; not delta, significance, R1 or R3"},
            "forbidden": ["calibration53", "test608", "D01", "D02", "D03"],
            "prior_796_jobs_reused_as_new_fits": False,
            "AI_assistance": "Model control, engine and review prepared with AI assistance; author must understand and defend them."}
    write_json(output / "admission.json", admission, immutable=True)
    write_json(output / "cohort.json", current, immutable=True)
    write_json(output / "plan.json", plan, immutable=True)
    write_json(output / "state.json", {"status": "prepared_awaiting_remote_plan_publication",
               "master_run_id": plan["master_run_id"], "plan_sha256": digest(plan),
               "new_real_data_fits": 0}, immutable=not (output / "state.json").exists()) if not old else None
    emit("prepared", master_run_id=plan["master_run_id"], plan_sha256=digest(plan),
         admission_sha256=digest(admission), new_real_data_fits=0)
    return plan, data


def verify_publication(plan, receipt_path):
    receipt = json.loads(Path(receipt_path).read_text())
    for key, expected in {"plan_sha256": digest(plan), "admission_sha256": plan["admission_sha256"],
                          "source_commit": plan["identity"]["source_commit"],
                          "master_run_id": plan["master_run_id"]}.items():
        require(receipt.get(key) == expected, "Remote publication receipt mismatch: " + key)
    require(bool(re.fullmatch(r"[0-9a-f]{40}", receipt.get("git_commit", ""))),
            "Published plan commit required before real fitting")
    return receipt


def build_model(name, seed, data, plugin_path, device):
    from src.e3_innovation_model import InnovationModel
    from src.e3_overlap_model_v3 import OverlapDistributionModel
    args = {"u_scale": data["scaler"]["u_scale"], "y_scale": data["scaler"]["y_scale"], "seed": seed}
    if name == "additive_positive_head":
        model = InnovationModel(name, **args)
    elif name == "atom_plugin":
        plugin = module_from_path("e3_pilot_secondmoment_plugin", plugin_path)
        model = plugin.PluginSecondMomentModel(**args, lambda_crps=.1)
    else:
        model = OverlapDistributionModel(name, **args, lambda_crps=.1)
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS[name], "Model capacity changed: " + name)
    return model.to(device)


def _tensors(data, device):
    import torch
    keys = ("train_X", "train_A", "train_U", "train_y_original", "val_X", "val_A", "val_U", "val_y_original")
    return {key: torch.as_tensor(data[key], device=device) for key in keys}


def fit_job(name, seed, learning_rate, stage, data, split, output, phase, plugin_path, plan, device, durability_dir=None):
    import torch
    setup_started = time.perf_counter()
    seed_all(seed)
    job_id = f"{stage}/{name}/seed{seed}_lr{learning_rate:.7f}"
    directory = Path(output) / "jobs" / job_id
    model = build_model(name, seed, data, plugin_path, device)
    tensors = _tensors(data, device)
    ix = torch.as_tensor(split["select"], device=device)
    x, vx = tensors["train_X"], tensors["val_X"][ix]
    scale = data["scaler"]["y_scale"]
    config = {**plan["config"], "learning_rate": learning_rate}
    identity = {"pilot_plan_sha256": digest(plan), "name": name, "seed": seed,
                "stage": stage, "configuration": model.configuration(),
                "data_admission_sha256": plan["admission_sha256"], "cohort": plan["cohort"]}

    def loss(m, indexes):
        if indexes is None:
            return ((m(vx, tensors["val_A"][ix]) - tensors["val_y_original"][ix]) / scale).square().mean()
        if name != "additive_positive_head":
            out = m.forward_components(x[indexes], tensors["train_A"][indexes])
            return m.loss_components(out, tensors["train_y_original"][indexes], tensors["train_U"][indexes])["loss"]
        return ((m(x[indexes], tensors["train_A"][indexes]) - tensors["train_y_original"][indexes]) / scale).square().mean()

    previous = directory / "job.json"
    was_complete = previous.is_file()
    # Frozen phase preserves optimizer/RNG and the original absolute wall deadline.
    admitted = {"identity": identity, "config": config, "budget_seconds": plan["budget_seconds_per_fit"]}
    write_json(directory / "admitted_job.json", admitted, immutable=True)
    setup_seconds = time.perf_counter() - setup_started
    original_write_json = phase.write_json
    first_validated_barrier_done = False

    def phase_write_json(path, value):
        nonlocal first_validated_barrier_done
        original_write_json(path, value)
        if Path(path).name == "request.json":
            # The frozen phase has already persisted its absolute deadline. Remote
            # publication time consumes this opportunity; it never extends120s.
            durability_barrier(durability_dir, "before_fit", job_id,
                               {"request_sha256": digest(value), "plan_sha256": digest(plan)})
        elif Path(path).name == "curve.json" and not first_validated_barrier_done:
            require(bool(value) and value[-1].get("optimizer_steps", 0) > 0,
                    "First checkpoint barrier needs a validated optimizer update")
            checkpoint = Path(path).with_name("checkpoint.pt")
            durability_barrier(durability_dir, "first_validated_checkpoint", job_id,
                               {"checkpoint_sha256": sha(checkpoint), "curve_sha256": sha(path),
                                "plan_sha256": digest(plan)})
            first_validated_barrier_done = True

    phase.write_json = phase_write_json
    try:
        if (directory / "supervised" / "phase_started.json").exists() and not was_complete:
            durability_barrier(durability_dir, "before_fit", job_id,
                               {"phase_marker_sha256": sha(directory / "supervised" / "phase_started.json"),
                                "plan_sha256": digest(plan)})
        state = phase.phase(model, x, tensors["train_y_original"] / scale, vx,
                            tensors["val_y_original"][ix] / scale, directory / "supervised", config,
                            identity=identity, loss_fn=loss, budget_seconds=plan["budget_seconds_per_fit"])
    finally:
        phase.write_json = original_write_json
    record = {"job_id": job_id, "name": name, "seed": seed, "learning_rate": learning_rate,
              "stage": stage, "selection_RMSE": math.sqrt(state["best_score"]) * scale,
              "parameters": PARAMETERS[name], "phase": state["resource_receipt"],
              "construction_and_tensor_setup_seconds_this_invocation": setup_seconds,
              "plan_sha256": digest(plan)}
    # A completed cache may differ only in resource receipt read timestamp; compare substantive fit identity.
    if was_complete:
        prior = json.loads(previous.read_text())
        for key in ("job_id", "selection_RMSE", "parameters", "plan_sha256"):
            require(record[key] == prior[key], "Completed job mismatch: " + key)
    write_json(previous, record)
    emit("job_completed", job_id=job_id, reused_completed_checkpoint=was_complete,
         selection_RMSE=record["selection_RMSE"], optimizer_steps=state["steps"])
    durability_barrier(durability_dir, "after_job", job_id,
                       {"checkpoint_sha256": sha(directory / "supervised" / "checkpoint.pt"),
                        "plan_sha256": digest(plan)})
    return model, record


def select_recipe(rows):
    require(len(rows) == 3 and sorted(row["learning_rate"] for row in rows) == list(LEARNING_RATES),
            "All three declared learning rates required; incomplete HPO refused")
    return min(rows, key=lambda row: (row["selection_RMSE"], row["learning_rate"]))["learning_rate"]


def metrics(prediction, target):
    import numpy as np
    require(prediction.shape == target.shape and np.isfinite(prediction).all(), "Finite aligned predictions required")
    return {"RMSE": float(np.sqrt(np.mean((prediction - target)**2))),
            "MAE": float(np.mean(np.abs(prediction - target)))}


def evaluate_model(model, name, data, indexes, device):
    import numpy as np
    import torch
    from src.e3_overlap_model_v3 import discrete_crps
    x = torch.as_tensor(data["val_X"][indexes], device=device)
    a = torch.as_tensor(data["val_A"][indexes], device=device)
    model.eval()
    with torch.no_grad():
        out = None if name == "additive_positive_head" else model.forward_components(x, a)
        prediction = model(x, a) if out is None else out["prediction"]
        pred = prediction.cpu().numpy()
        crps = None if out is None else float(discrete_crps(out["atoms"], out["probabilities"],
              torch.as_tensor(data["val_U"][indexes], device=device)).mean())
    require(np.all(pred + 1e-12 >= np.sqrt(data["val_A"][indexes])), "Prediction violates floor")
    return pred, crps


def summarize(final_rows, analytical_rows):
    import numpy as np
    summary = []
    for name in NEURAL:
        rows = [row for row in final_rows if row["name"] == name]
        require(len(rows) == 3 and sorted(row["seed"] for row in rows) == list(FINAL_SEEDS),
                "All final seeds of every neural variant are required")
        values = [row["RMSE"] for row in rows]
        summary.append({"name": name, "seeds": 3, "mean_RMSE": float(np.mean(values)),
                        "SD_RMSE": float(np.std(values, ddof=1)),
                        "mean_MAE": float(np.mean([row["MAE"] for row in rows])),
                        "mean_CRPS_U": None if name == "additive_positive_head" else
                            float(np.mean([row["CRPS_U"] for row in rows]))})
    candidate = next(row for row in summary if row["name"] == "atom_crps")
    neural_best = min((row for row in summary if row["name"] != "atom_crps"), key=lambda row: row["mean_RMSE"])
    all_options = [(row["name"], row["mean_RMSE"]) for row in summary if row["name"] != "atom_crps"]
    all_options += [(row["name"], row["RMSE"]) for row in analytical_rows]
    all_best = min(all_options, key=lambda row: row[1])
    gain_neural = 1. - candidate["mean_RMSE"] / neural_best["mean_RMSE"]
    gain_all = 1. - candidate["mean_RMSE"] / all_best[1]
    return summary, {"candidate": "atom_crps", "best_neural_control": neural_best["name"],
                     "best_all_controls": all_best[0], "relative_gain_vs_best_neural": gain_neural,
                     "relative_gain_vs_best_all": gain_all,
                     "continuation_heuristic_passed": bool(gain_neural >= .02 and gain_all >= .02),
                     "comparison_unit": "mean_seed_RMSE_on_exposed_development_assessment55",
                     "formal_delta": None, "p_values_computed": False, "R1_accredited": False,
                     "R3_accredited": False, "confirmatory_claim_allowed": False}


def run(output, source_root, plugin_path, data, plan, device, publication_receipt, durability_dir=None):
    import numpy as np
    output = Path(output)
    require(json.loads((output / "plan.json").read_text()) == plan, "Sealed plan bytes/values changed")
    require(plan["identity"] == source_identity(source_root, plugin_path, plan["identity"]["source_commit"]),
            "Source changed after sealing")
    require(cohort(device) == plan["cohort"], "GPU/software cohort changed; preserve run and restore matching cohort")
    require(sha(output / "hardware_profile.json") == plan["hardware_profile_sha256"], "Hardware profile changed")
    require(sha(output / "launcher_lock.json") == plan["launcher_lock_sha256"], "Launcher lock changed")
    require(sha(output / "launcher_identity.json") == plan["launcher_identity_sha256"], "Launcher identity changed")
    require(digest(json.loads((output / "admission.json").read_text())) == plan["admission_sha256"], "Admission changed")
    verify_publication(plan, publication_receipt)
    split = split_development(data)
    phase = module_from_path("e3_pilot_frozen_phase", Path(source_root) / SOURCE_FILES[-1])
    started = time.time()
    write_json(output / "state.json", {"status": "running_HPO", "master_run_id": plan["master_run_id"]})
    trial_rows, choices = [], {}
    for name in NEURAL:
        rows = []
        for learning_rate in LEARNING_RATES:
            _, record = fit_job(name, HPO_SEED, learning_rate, "HPO", data, split, output, phase, plugin_path, plan, device, durability_dir)
            rows.append(record)
            trial_rows.append({key: record[key] for key in ("name", "seed", "learning_rate", "selection_RMSE", "parameters")})
        choices[name] = select_recipe(rows)
    write_csv(output / "HPO_scores.csv", trial_rows)
    selection = {"plan_sha256": digest(plan), "selected_learning_rates": choices,
                 "all_15_trials_complete": True, "assessment_used": False,
                 "selection_rule": plan["selection_tie_break"]}
    write_json(output / "closed_selection.json", selection, immutable=True)
    emit("selection_closed", selected_learning_rates=choices, assessment_evaluated=False)
    # Train every final recipe before the first assessment prediction is produced.
    final_models, final_jobs = {}, []
    for name in NEURAL:
        for seed in FINAL_SEEDS:
            model, record = fit_job(name, seed, choices[name], "final", data, split, output, phase, plugin_path, plan, device, durability_dir)
            final_models[name, seed] = model.to("cpu")
            final_jobs.append(record)
    require(len(trial_rows) + len(final_jobs) == 30, "Incomplete pilot cannot be assessed")
    write_json(output / "all_fits_complete.json", {"plan_sha256": digest(plan), "HPO_fits": 15,
               "final_fits": 15, "closed_selection_sha256": digest(selection)}, immutable=True)
    indexes = split["assessment"]
    target = data["val_y_original"][indexes]
    final_rows, predictions = [], []
    for (name, seed), model in final_models.items():
        pred, crps = evaluate_model(model.to(device), name, data, indexes, device)
        row = {"name": name, "seed": seed, "learning_rate": choices[name], **metrics(pred, target), "CRPS_U": crps}
        final_rows.append(row)
        predictions.extend({"name": name, "seed": seed, "date": str(day), "target": float(y), "prediction": float(p)}
                           for day, y, p in zip(data["val_dates"][indexes], target, pred))
        model.to("cpu")
    analytical_rows = []
    floor = np.sqrt(data["val_A"][indexes])
    for name, pred in (("floor_only", floor), ("floor_plus_mean_excess", floor + data["scaler"]["d_fit"])):
        analytical_rows.append({"name": name, "fitted_on": "fit532_only", **metrics(pred, target), "CRPS_U": None})
        predictions.extend({"name": name, "seed": "analytic", "date": str(day), "target": float(y), "prediction": float(p)}
                           for day, y, p in zip(data["val_dates"][indexes], target, pred))
    summary, verdict = summarize(final_rows, analytical_rows)
    write_csv(output / "final_assessment_scores.csv", final_rows)
    write_csv(output / "analytical_controls.csv", analytical_rows)
    write_csv(output / "recipe_summary.csv", summary)
    write_csv(output / "assessment_predictions.csv", predictions)
    write_json(output / "development_verdict.json", verdict)
    receipt = {"status": "completed_development_only", "master_run_id": plan["master_run_id"],
               "plan_sha256": digest(plan), "required_fits": 30, "completed_fits": 30,
               "market_calibration_arrays_opened": 0, "market_test_arrays_opened": 0,
               "additional_datasets_used": 0, "assessment_rows": 55,
               "elapsed_this_invocation_seconds": time.time() - started,
               "fit_resource_receipts": {path.relative_to(output).as_posix(): json.loads(path.read_text())["phase"]
                                         for path in sorted((output / "jobs").rglob("job.json"))},
               "artifact_sha256": {path.name: sha(path) for path in output.glob("*.csv")},
               **verdict}
    write_json(output / "pilot_receipt.json", receipt)
    write_json(output / "state.json", {"status": "completed_development_only", "master_run_id": plan["master_run_id"],
               "completed_fits": 30, "pilot_receipt_sha256": sha(output / "pilot_receipt.json")})
    emit("completed_development_only", receipt=str(output / "pilot_receipt.json"), **verdict)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--npz-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    sibling_plugin = Path(__file__).resolve().with_name("plugin_control.py")
    default_plugin = sibling_plugin if sibling_plugin.is_file() else Path(__file__).resolve().parents[1] / "model" / "plugin_control.py"
    parser.add_argument("--plugin", default=str(default_plugin))
    parser.add_argument("--publication-receipt")
    parser.add_argument("--durability-dir")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    plan, data = prepare(args.output, args.source_root, args.plugin, args.npz, args.npz_sha256, args.source_commit, args.device)
    if args.run:
        require(args.publication_receipt is not None, "Remote publication receipt required")
        run(args.output, args.source_root, args.plugin, data, plan, args.device, args.publication_receipt, args.durability_dir)


if __name__ == "__main__":
    main()
