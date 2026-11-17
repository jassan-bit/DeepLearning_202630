"""Resumable, bounded exploratory moment-head development on training-era data.

The admitted global claim precedes every fit. Historical training is not reset;
the reference is a sealed CSV, never a prior model or checkpoint. The immutable
v2 transaction/budget/RNG helpers are reused explicitly, without rebinding their
scientific globals. No network operation or credential handling occurs here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from src.e3_development import canonical, run_lock, runtime_info, sha256_file
from src.e3_innovation import (BudgetExhausted, _Budget, _atomic_json, _csv,
    _determinism, _event, _load_checkpoint, _read_json, _recover_best, _restore_rng,
    _rng, _save_checkpoint)
from src.e3_moment_model import (MomentModel, NEURAL_NAMES, load_moment_data,
                                 loss_components)


VERSION = "e3-moment-development-v1"
CLAIM_VERSION = "e3-moment-development-budget-v1"
CAMPAIGN = "e3-moment-pilot-20261116"
PRIOR_COMMIT = "1ca029e869a517cb8de704e573c78ca1e0cbad3c"
PRIOR_SNAPSHOT_SHA256 = "5fe65596b99b334de197f3fc08811514c5ee6ea48b8447318057fea1b7cf9be1"
PRIOR_CHARGED = 556.2126593619992
PRODUCTION_NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
ANALYTIC_NAMES = ("floor_only", "floor_plus_fit_mean", "frozen_v2_additive")
CANDIDATE = "moment_joint"
TERMINAL = {"completed", "budget_exhausted", "failed", "failed_numerical"}
REFERENCE_FILES = ("plan.json", "summary.json", "ledger.json", "global_registry.json",
    "state.json", "manifest.json", "data_admission.json",
    "additive_positive_head/validation_predictions.csv")


def source_fingerprints():
    root = Path(__file__).resolve().parents[1]
    names = ("src/original_model.py", "src/e3_development.py", "src/e3_innovation_model.py",
        "src/e3_innovation.py", "src/e3_moment_model.py", "src/e3_moment_training.py",
        "scripts/e3_run_moments.py", "scripts/e3_moment_github.py", "scripts/e3_github_storage.py")
    return {name: sha256_file(root / name) for name in names}


def validate_plan(plan):
    """Preserve the raw plan hash while enforcing the one permitted pilot."""
    if not isinstance(plan, dict):
        raise ValueError("An explicit moment pilot plan is required")
    plan = deepcopy(plan)
    recipe = plan["training"]
    for key in ("seed", "batch_size", "max_epochs_per_neural_variant", "patience"):
        if type(recipe[key]) is not int or recipe[key] < (0 if key == "seed" else 1):
            raise ValueError("Invalid training setting: " + key)
    if recipe["seed"] >= 2 ** 32 or recipe["max_epochs_per_neural_variant"] > 40:
        raise ValueError("Seed/epoch setting exceeds the admitted recipe")
    expected = {"optimizer": "AdamW", "learning_rate": .001, "weight_decay": 0,
                "gradient_clip_norm": 1}
    if any(recipe.get(k) != v for k, v in expected.items()) or recipe.get("AMP", False) or recipe.get("TF32", False):
        raise ValueError("The fixed learning recipe/precision must be preserved")
    architecture = {"family": "causal_CNN_LSTM_temporal_content_attention", "base_variant": "M00",
        "history_days": 7, "input_features": 5, "cnn_channels": 8, "kernel_size": 3,
        "lstm_hidden_size": 12, "learned_query": True, "discordance_penalty": False,
        "radial_value_saturation": False, "trainable_parameters_per_new_neural_variant": 1236}
    if any(plan["architecture"].get(k) != v for k, v in architecture.items()):
        raise ValueError("The fixed 1236-parameter moment architecture must be preserved")
    expected_controls = list(ANALYTIC_NAMES) + [n for n in NEURAL_NAMES if n != CANDIDATE]
    if [x["id"] for x in plan["controls"]] != expected_controls or plan["candidate"]["id"] != CANDIDATE:
        raise ValueError("The complete fixed candidate/control family must be preserved")
    candidate = plan["candidate"]
    controls = {row["id"]: row for row in plan["controls"]}
    if (candidate.get("lambda_m") != .1 or candidate.get("lambda_q") != .1
            or any(controls["moment_mean_only"].get(k) != 0 for k in ("lambda_m", "lambda_q"))
            or any(controls["direct_moment_aux"].get(k) != .1 for k in ("lambda_m", "lambda_q"))):
        raise ValueError("Candidate/control auxiliary weights must match their fixed recipes")
    if (plan["loss"].get("lambda_m") != .1 or plan["loss"].get("lambda_q") != .1
            or plan["decision"]["minimum_relative_rmse_improvement_for_continuation"] != .02
            or plan["decision"]["require_all_neural_recipes_completed"] is not True):
        raise ValueError("Moment scores and exploratory continuation rule are fixed")
    budget = plan["proposed_budget"]
    cap = budget["additional_aggregate_charged_cap_seconds"]
    if (isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap)
            or not 0 < cap <= 600 or budget["prior_charged_seconds"] != PRIOR_CHARGED
            or budget["prior_result_commit"] != PRIOR_COMMIT
            or budget["global_charged_execution_limit_seconds"] != 1200
            or PRIOR_CHARGED + cap > 1200):
        raise ValueError("Historical cost and global budget cannot reset or increase")
    if plan["frozen_control"]["source_commit"] != PRIOR_COMMIT:
        raise ValueError("The frozen reference source commit is fixed")
    if plan["data"]["sha256"] == PRODUCTION_NPZ_SHA256:
        fixed = {"seed": 20261116, "batch_size": 32, "max_epochs_per_neural_variant": 40, "patience": 8}
        if any(recipe[k] != v for k, v in fixed.items()):
            raise ValueError("Production training recipe is fixed")
        fixed_data = {"fit_end": "2024-06-30", "validation_start": "2024-07-08",
            "validation_end": "2024-10-31", "fit_count": 532, "validation_count": 116, "purged_count": 7}
        if any(plan["data"].get(k) != v for k, v in fixed_data.items()):
            raise ValueError("Production development cohort is fixed")
        if plan["frozen_control"]["snapshot_manifest_sha256"] != PRIOR_SNAPSHOT_SHA256:
            raise ValueError("Production reference snapshot is fixed")
    canonical(plan)
    return plan


def validate_global_claim(claim, plan, expected_sha256, run_id=None):
    if not isinstance(claim, dict):
        raise RuntimeError("Confirmed global budget claim required before any fit")
    expected = {"version": CLAIM_VERSION, "campaign_id": CAMPAIGN, "prior_commit": PRIOR_COMMIT,
        "prior_snapshot_sha256": plan["frozen_control"]["snapshot_manifest_sha256"],
        "prior_charged_seconds": PRIOR_CHARGED,
        "additional_limit_seconds": plan["proposed_budget"]["additional_aggregate_charged_cap_seconds"],
        "global_limit_seconds": 1200., "plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(),
        "npz_sha256": expected_sha256.lower()}
    if any(claim.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Global claim/plan/input/prior reference mismatch")
    for key in ("run_id", "claim_id"):
        if not isinstance(claim.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", claim[key]):
            raise RuntimeError("Invalid run/claim identity")
    if not re.fullmatch(r"[0-9a-f]{40}", str(claim.get("registry_commit", ""))):
        raise RuntimeError("A confirmed registry commit ACK is required")
    if run_id is not None and claim["run_id"] != run_id:
        raise RuntimeError("Global claim belongs to another run")
    return deepcopy(claim)


def _check_ledger(ledger, claim):
    ledger = deepcopy(ledger)
    charged = ledger.get("charged_seconds")
    if (ledger.get("limit_seconds") != claim["additional_limit_seconds"]
            or isinstance(charged, bool) or not isinstance(charged, (int, float))
            or not math.isfinite(charged) or charged < 0 or not isinstance(ledger.get("events"), list)):
        raise RuntimeError("Invalid aggregate moment ledger")
    active = ledger.get("active_session")
    if active is not None:
        if not active.get("session_id") or datetime.fromisoformat(active["last_wall_utc"]).tzinfo is None:
            raise RuntimeError("Invalid ACTIVE reservation timestamp")
    return ledger


def _load_data(npz_path, expected_sha256, plan):
    d = plan["data"]
    result = load_moment_data(npz_path, expected_sha256, fit_end=d["fit_end"],
        validation_start=d["validation_start"], validation_end=d["validation_end"])
    observed = (len(result["train_X"]), len(result["validation_X"]), result["admission"]["n_purged_or_unused"])
    if observed != (d["fit_count"], d["validation_count"], d["purged_count"]):
        raise RuntimeError("Data cohort counts disagree with the admitted plan")
    return result


def _frozen_predictions(directory, snapshot_path, data, plan, budget):
    """Verify sealed text bytes and exact date/target alignment; no deserialization."""
    directory, snapshot_path = Path(directory), Path(snapshot_path)
    config = plan["frozen_control"]
    if sha256_file(snapshot_path) != config["snapshot_manifest_sha256"]:
        raise RuntimeError("Frozen v2 snapshot SHA-256 mismatch")
    snapshot = _read_json(snapshot_path)
    registry = snapshot["registry"]
    prior_names = {"additive_positive_head", "innovation_mean_only", "innovation_density_mean"}
    if (set(snapshot["progress"]["models"]) != prior_names
            or snapshot["progress"]["state_status"] != "completed" or registry["active_session"] is not None
            or not all(x["complete"] for x in snapshot["progress"]["models"].values())
            or registry["prior_charged_seconds"] + registry["charged_seconds"] != PRIOR_CHARGED):
        raise RuntimeError("Frozen v2 snapshot is not closed or does not prove historical cost")
    for name in REFERENCE_FILES:
        budget.tick()
        path, record = directory / name, snapshot["files"][name]
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]):
            raise RuntimeError("Frozen v2 reference member byte hash/size mismatch: " + name)
    prior_manifest = _read_json(directory / "manifest.json")
    prior_admission = _read_json(directory / "data_admission.json")
    prior_summary = _read_json(directory / "summary.json")
    prior_ledger = _read_json(directory / "ledger.json")
    prior_state = _read_json(directory / "state.json")
    if (prior_summary["status"] != "completed" or prior_summary["global_charged_seconds"] != PRIOR_CHARGED
            or prior_ledger["active_session"] is not None or prior_ledger["charged_seconds"] != registry["charged_seconds"]
            or prior_state["status"] != "completed"
            or _read_json(directory / "global_registry.json") != registry):
        raise RuntimeError("Frozen v2 text does not prove its closed final state")
    if (hashlib.sha256(canonical(_read_json(directory / "plan.json"))).hexdigest() != prior_manifest["plan_sha256"]
            or hashlib.sha256(canonical(prior_admission)).hexdigest() != prior_manifest["data_fingerprint"]
            or registry.get("plan_sha256") != prior_manifest["plan_sha256"]
            or snapshot.get("scientific_identity", {}).get("data_fingerprint") != prior_manifest["data_fingerprint"]
            or registry.get("npz_sha256") != data["admission"]["npz_sha256"]):
        raise RuntimeError("Frozen v2 plan/admission provenance mismatch")
    if prior_admission["validation_dates"] != data["validation_dates"].tolist():
        raise RuntimeError("Frozen v2 validation dates differ from this development cohort")
    for key in ("mean", "std", "target_scale", "u_scale", "y_scale"):
        if not np.array_equal(np.asarray(prior_admission["scaler"][key]), np.asarray(data["scaler"][key])):
            raise RuntimeError("Frozen v2 preprocessing differs from this inner fit")
    # Admission fingerprint dictionaries include the raw transformed labels, so
    # comparing them verifies both training and validation bytes without opening
    # any additional NPZ member or a prior pickle/checkpoint.
    if prior_admission["arrays_sha256"] != data["admission"]["arrays_sha256"]:
        raise RuntimeError("Frozen v2 admitted input/label array hashes differ")
    csv_path = directory / "additive_positive_head/validation_predictions.csv"
    if config.get("predictions_sha256") != sha256_file(csv_path):
        raise RuntimeError("Frozen additive prediction SHA-256 mismatch")
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [r["date"] for r in rows] != data["validation_dates"].tolist():
        raise RuntimeError("Frozen additive prediction dates differ")
    y = np.asarray([float(r["target_sigma7"]) for r in rows], dtype=np.float64)
    predicted = np.asarray([float(r["prediction_sigma7"]) for r in rows], dtype=np.float64)
    floor = np.asarray([float(r["known_floor"]) for r in rows], dtype=np.float64)
    if not np.array_equal(y, data["validation_y_original"]):
        raise RuntimeError("Frozen additive prediction targets differ")
    if (not np.isfinite(predicted).all()
            or not np.allclose(floor, np.sqrt(data["validation_A"]), rtol=0, atol=2e-14)):
        raise RuntimeError("Frozen additive predictions/floors invalid")
    return predicted, {"source_commit": PRIOR_COMMIT,
        "snapshot_manifest_sha256": config["snapshot_manifest_sha256"],
        "predictions_sha256": sha256_file(csv_path), "parameters": 1210,
        "new_fit": False, "prior_checkpoint_loaded": False,
        "capacity_differs_from_new_models": True}


def _metrics(predicted, data, components=None):
    y, floor = data["validation_y_original"], np.sqrt(data["validation_A"])
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape != y.shape or not np.isfinite(predicted).all():
        raise RuntimeError("Invalid or nonfinite validation predictions")
    residual = predicted - y
    high = y >= np.quantile(data["train_y_original"], .75)
    result = {"RMSE": float(np.sqrt(np.mean(residual ** 2))), "MAE": float(np.mean(np.abs(residual))),
        "bias": float(residual.mean()), "high_volatility_count": int(high.sum()),
        "high_volatility_RMSE": float(np.sqrt(np.mean(residual[high] ** 2))) if high.any() else None,
        "high_volatility_MAE": float(np.mean(np.abs(residual[high]))) if high.any() else None,
        "high_volatility_threshold_fit_q75": float(np.quantile(data["train_y_original"], .75)),
        "below_floor_count": int(np.sum(predicted < floor - 1e-14)),
        "moment_m_score": None, "moment_q_score": None, "outside_estimated_mean_envelope_count": None,
        "prediction_interval_claimed": False}
    extra = components or {}
    if components is not None:
        values = {k: torch.as_tensor(extra[k], dtype=torch.float64) for k in ("prediction", "m", "q")}
        scores = loss_components(values, torch.as_tensor(y), torch.as_tensor(data["validation_U"]),
            y_scale=data["scaler"]["y_scale"], u_scale=data["scaler"]["u_scale"], lambda_m=.1, lambda_q=.1)
        result.update(moment_m_score=float(scores["m_score"]), moment_q_score=float(scores["q_score"]),
            outside_estimated_mean_envelope_count=int(np.sum((predicted < extra["L"] - 1e-14)
                                                        | (predicted > extra["H"] + 1e-14))))
    rows = [{"date": str(day), "target_sigma7": float(target), "prediction_sigma7": float(pred),
        "known_floor": float(fl), **{key: float(extra[key][i]) if key in extra else None
                                     for key in ("m", "q", "rho", "L", "H")}}
        for i, (day, target, pred, fl) in enumerate(zip(data["validation_dates"], y, predicted, floor))]
    return result, rows


def _new_checkpoint(model, optimizer, manifest, seed):
    return {"version": VERSION, "model_name": model.variant,
        "model_state": deepcopy(model.state_dict()), "optimizer_state": deepcopy(optimizer.state_dict()),
        "rng": _rng(), "dataloader_generator_state": torch.Generator().manual_seed(seed).get_state(),
        "epoch": 0, "active_permutation": None, "batch_cursor": 0,
        "epoch_loss_sum": 0., "epoch_examples": 0, "no_improvement": 0,
        "best_validation_rmse": None, "best_epoch": None, "history": [], "complete": False,
        **{k: manifest[k] for k in ("source_hashes", "plan_sha256", "data_fingerprint", "run_id", "claim_id")}}


def _state_progress(checkpoints, previous=None):
    previous_models = (previous or {}).get("models", {})
    return {name: {"epoch": checkpoints[name]["epoch"], "complete": checkpoints[name]["complete"]}
        if name in checkpoints else deepcopy(previous_models.get(name, {"epoch": 0, "complete": False}))
        for name in NEURAL_NAMES}


def _load_moment_checkpoint(path, manifest, name):
    """Bind equal-shaped state dictionaries to their exact recipe/version."""
    value = _load_checkpoint(path, manifest)
    records = [value]
    if value.get("best_snapshot"):
        records.append(value["best_snapshot"]["checkpoint"])
    for record in records:
        if record.get("version") != VERSION or record.get("model_name") != name:
            raise RuntimeError("Checkpoint variant/version mismatch")
        if any(record.get(key) != manifest[key] for key in
               ("source_hashes", "plan_sha256", "data_fingerprint", "run_id", "claim_id")):
            raise RuntimeError("Checkpoint provenance mismatch")
        _assert_finite_state(record["model_state"], record["optimizer_state"])
    return value


def _assert_finite_state(*records):
    """Check floating state before committing it, including Adam accumulators."""
    groups = {}
    def collect(value):
        if torch.is_tensor(value):
            if value.is_floating_point() or value.is_complex():
                groups.setdefault((value.device, value.dtype), []).append(value.detach().reshape(-1))
        elif isinstance(value, dict):
            for member in value.values():
                collect(member)
        elif isinstance(value, (tuple, list)):
            for member in value:
                collect(member)
    for record in records:
        collect(record)
    if any(not bool(torch.isfinite(torch.cat(tensors)).all()) for tensors in groups.values()):
        raise RuntimeError("Nonfinite model/optimizer state; last committed checkpoint preserved")


def _summary(directory, manifest, status):
    ledger = _read_json(directory / "ledger.json")
    models = {}
    for name in NEURAL_NAMES:
        path = directory / name / "last.pt"
        cp = _load_moment_checkpoint(path, manifest, name) if path.is_file() and manifest.get("initialized") else None
        snapshot = cp.get("best_snapshot") if cp else None
        models[name] = {"completed_epochs": cp["epoch"] if cp else 0,
            "complete": cp["complete"] if cp else False, "parameters": 1236,
            "best_epoch": cp["best_epoch"] if cp else None,
            "metrics": snapshot["metrics"] if snapshot else None,
            "moment_outputs_elicited_by_auxiliary_scores": name != "moment_mean_only"}
        if cp:
            models[name]["last_checkpoint_sha256"] = sha256_file(path)
    path = directory / "controls.json"
    controls = _read_json(path)["metrics"] if path.is_file() else {}
    controls_complete = set(controls) == set(ANALYTIC_NAMES) and all(
        isinstance(m, dict) and all(isinstance(m.get(k), (int, float)) and math.isfinite(m[k])
                                  for k in ("RMSE", "MAE", "bias")) for m in controls.values())
    all_metrics = {**controls, **{n: x["metrics"] for n, x in models.items() if x["metrics"]}}
    complete = all(x["complete"] for x in models.values()) and controls_complete and len(all_metrics) == 6
    best_control = improvement = gate = None
    if complete:
        fixed = list(ANALYTIC_NAMES) + [n for n in NEURAL_NAMES if n != CANDIDATE]
        best_control = min(fixed, key=lambda name: all_metrics[name]["RMSE"])
        baseline = all_metrics[best_control]["RMSE"]
        improvement = (baseline - all_metrics[CANDIDATE]["RMSE"]) / baseline if baseline else None
        gate = improvement is not None and improvement >= .02
    for name in ANALYTIC_NAMES:
        models[name] = {"completed_epochs": 0, "complete": name in controls, "best_epoch": None,
            "parameters": 1210 if name == "frozen_v2_additive" else (1 if name == "floor_plus_fit_mean" else 0),
            "metrics": controls.get(name)}
    for row in models.values():
        metric = row["metrics"] or {}
        row.update({key: metric.get(key) for key in ("RMSE", "MAE", "bias")})
    result = {"version": VERSION, "run_id": manifest["run_id"], "claim_id": manifest["claim_id"],
        "status": status, "models": models, "metrics": all_metrics,
        "comparison_complete": complete, "pilot_incomplete": not complete,
        "best_fixed_control": best_control, "relative_RMSE_improvement": improvement,
        "exploratory_continuation_gate_passed": bool(gate) if complete else False,
        "decision": {"status": "complete" if complete else "incomplete", "complete": complete,
            "best_control": best_control, "improvement_fraction": improvement,
            "continuation_gate_passed": bool(gate) if complete else False,
            "interpretation": "exploratory 2% continuation heuristic, not the confirmatory margin"},
        "winner_declared": False, "statistical_tests_performed": False,
        "charged_seconds": ledger["charged_seconds"], "limit_seconds": ledger["limit_seconds"],
        "prior_charged_seconds": PRIOR_CHARGED,
        "global_charged_seconds": PRIOR_CHARGED + ledger["charged_seconds"], "global_limit_seconds": 1200.,
        "budget_remaining_seconds": max(0., ledger["limit_seconds"] - ledger["charged_seconds"]),
        "budget_overshoot_seconds": max(0., ledger["charged_seconds"] - ledger["limit_seconds"]),
        "scope": "one-seed exposed development validation reused for stopping; no R1/R3 certification",
        "calibration_scored": False, "evaluation_scored": False, "prior_checkpoint_loaded": False,
        "source_hashes": manifest["source_hashes"], "plan_sha256": manifest["plan_sha256"],
        "data_fingerprint": manifest.get("data_fingerprint")}
    _atomic_json(directory / "summary.json", result)
    return result


def _model(name, data, seed, device):
    s = data["scaler"]
    return MomentModel(name, seed=seed, u_scale=s["u_scale"], y_scale=s["y_scale"],
        mean_v=s["mean_v"], initial_mean_excess=s["initial_mean_excess"]).to(device)


def _initialize(directory, npz_path, expected_sha256, manifest, plan, runtime, budget):
    budget.tick()
    data = _load_data(npz_path, expected_sha256, plan)
    budget.tick()
    manifest["data_fingerprint"] = hashlib.sha256(canonical(data["admission"])).hexdigest()
    _atomic_json(directory / "data_admission.json", data["admission"])
    device, seed = runtime["device_type"], plan["training"]["seed"]
    _determinism(seed)
    models = {name: _model(name, data, seed, device) for name in NEURAL_NAMES}
    budget.tick()
    frozen, reference = _frozen_predictions(manifest["frozen_v2_dir"], manifest["snapshot_manifest_path"],
                                            data, plan, budget)
    predictions = {"floor_only": np.sqrt(data["validation_A"]),
        "floor_plus_fit_mean": np.sqrt(data["validation_A"]) + data["scaler"]["d_fit"],
        "frozen_v2_additive": frozen}
    controls, rows = {}, []
    for name, predicted in predictions.items():
        budget.tick()
        controls[name], records = _metrics(predicted, data)
        rows.extend({"model": name, **record} for record in records)
    _atomic_json(directory / "controls.json", {"metrics": controls, "frozen_control": reference})
    (directory / "controls").mkdir(exist_ok=True)
    _csv(directory / "controls/validation_predictions.csv", rows)
    manifest.update(controls_sha256=sha256_file(directory / "controls.json"),
        control_predictions_sha256=sha256_file(directory / "controls/validation_predictions.csv"),
        models={name: model.configuration() for name, model in models.items()})
    for name, model in models.items():
        budget.tick()
        if sum(p.numel() for p in model.parameters()) != 1236:
            raise RuntimeError("Moment model must have 1236 active parameters")
        (directory / name).mkdir(exist_ok=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan["training"]["learning_rate"],
                                      weight_decay=plan["training"]["weight_decay"])
        _save_checkpoint(directory / name / "last.pt", _new_checkpoint(model, optimizer, manifest, seed))
    budget.tick()
    manifest["initialized"] = True
    _atomic_json(directory / "manifest.json", manifest)
    _event(directory, event="moment_initialization_completed", initialization_scope="fit_only")
    return data


def create_moment_run(npz_path, expected_sha256, run_dir, *, global_claim=None,
        initial_ledger=None, device="auto", plan_path=None, plan=None, frozen_v2_dir=None,
        snapshot_manifest_path=None, clock=time.monotonic):
    if (plan is None) == (plan_path is None):
        raise ValueError("Supply exactly one explicit plan or plan_path")
    plan = validate_plan(_read_json(plan_path) if plan_path is not None else plan)
    claim = validate_global_claim(global_claim, plan, expected_sha256)
    if plan["data"]["sha256"] != expected_sha256.lower():
        raise RuntimeError("Plan NPZ fingerprint mismatch")
    if frozen_v2_dir is None or snapshot_manifest_path is None:
        raise ValueError("Sealed v2 reference directory and snapshot manifest are required")
    if initial_ledger is None:
        raise RuntimeError("Confirmed ACTIVE ledger required before initialization")
    ledger = _check_ledger(initial_ledger, claim)
    if not ledger.get("active_session"):
        raise RuntimeError("Initialization requires the coordinator ACTIVE reservation")
    runtime = runtime_info(device)
    runtime["torch"] = str(runtime["torch"])
    if expected_sha256.lower() == PRODUCTION_NPZ_SHA256 and runtime["device_type"] != "cuda":
        raise RuntimeError("The production Colab pilot requires an assigned CUDA GPU")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"version": VERSION, "run_id": claim["run_id"], "claim_id": claim["claim_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "initialized": False,
        "global_claim": claim, "npz_sha256": expected_sha256.lower(),
        "plan_sha256": claim["plan_sha256"], "source_hashes": source_fingerprints(),
        "runtime_initial": runtime, "frozen_v2_dir": str(Path(frozen_v2_dir).resolve()),
        "snapshot_manifest_path": str(Path(snapshot_manifest_path).resolve()), "models": {}}
    with run_lock(directory):
        for name, value in (("plan.json", plan), ("manifest.json", manifest),
                            ("runtime.json", runtime), ("ledger.json", ledger)):
            _atomic_json(directory / name, value)
        budget = _Budget(directory, ledger, clock, runtime["device_type"])
        status, error = "initialized", None
        try:
            _atomic_json(directory / "state.json", {"status": "initializing", "models": _state_progress({})})
            _initialize(directory, npz_path, expected_sha256, manifest, plan, runtime, budget)
        except BudgetExhausted:
            status = "budget_exhausted"
        except BaseException as caught:
            status = "interrupted" if isinstance(caught, (KeyboardInterrupt, SystemExit)) else "failed"
            error = caught
        finally:
            budget.close(status)
            _atomic_json(directory / "state.json", {"status": status, "models": _state_progress({})})
        _summary(directory, manifest, status)
        if error:
            raise error
    return manifest


def _validate_resume(directory, npz_path, expected_sha256, claim, device):
    manifest, plan = _read_json(directory / "manifest.json"), validate_plan(_read_json(directory / "plan.json"))
    validate_global_claim(claim, plan, expected_sha256, manifest["run_id"])
    if claim["claim_id"] != manifest["claim_id"]:
        raise RuntimeError("Cannot switch campaign claims")
    if source_fingerprints() != manifest["source_hashes"]:
        raise RuntimeError("Source fingerprint changed; resume requires recorded bytes")
    if hashlib.sha256(canonical(plan)).hexdigest() != manifest["plan_sha256"]:
        raise RuntimeError("Plan fingerprint changed")
    if expected_sha256.lower() != manifest["npz_sha256"] or sha256_file(npz_path) != expected_sha256.lower():
        raise RuntimeError("NPZ fingerprint changed")
    if manifest["initialized"]:
        for name, key in (("controls.json", "controls_sha256"),
                          ("controls/validation_predictions.csv", "control_predictions_sha256")):
            if not (directory / name).is_file() or sha256_file(directory / name) != manifest[key]:
                raise RuntimeError("Frozen control metrics/predictions changed")
    runtime = runtime_info(device)
    runtime["torch"] = str(runtime["torch"])
    if expected_sha256.lower() == PRODUCTION_NPZ_SHA256 and runtime["device_type"] != "cuda":
        raise RuntimeError("The production Colab pilot requires an assigned CUDA GPU")
    old = manifest["runtime_initial"]
    if any(runtime.get(k) != old.get(k) for k in ("python", "numpy", "torch", "device_type", "cuda_build", "gpu", "cpu_threads")):
        raise RuntimeError("Exact continuation requires the original numerical runtime/device")
    return manifest, plan, runtime


def _validation(model, data, device, batch_size, budget):
    values = {name: [] for name in ("prediction", "m", "q", "rho", "L", "H")}
    model.eval()
    with torch.no_grad():
        for first in range(0, len(data["validation_X"]), batch_size):
            budget.tick()
            x = torch.from_numpy(data["validation_X"][first:first + batch_size]).to(device)
            a = torch.from_numpy(data["validation_A"][first:first + batch_size]).to(device)
            outputs = model.forward_components(x, a)
            for key in values:
                if outputs.get(key) is not None:
                    values[key].append(outputs[key].cpu().numpy())
    components = {key: np.concatenate(parts) for key, parts in values.items() if parts}
    return _metrics(components["prediction"], data, components)


def run_moment(run_dir, npz_path, expected_sha256, *, global_claim=None, resume=False,
        device="auto", stop_after_epochs=None, clock=time.monotonic, frozen_v2_dir=None,
        snapshot_manifest_path=None):
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("stop_after_epochs must be a positive integer")
    if (frozen_v2_dir is None) != (snapshot_manifest_path is None):
        raise ValueError("Supply both frozen reference paths together")
    directory = Path(run_dir)
    with run_lock(directory):
        manifest, plan, runtime = _validate_resume(directory, npz_path, expected_sha256, global_claim, device)
        previous = _read_json(directory / "state.json")
        if previous["status"] in TERMINAL:
            return _summary(directory, manifest, previous["status"])
        if previous["status"] != "initialized" and not resume:
            raise RuntimeError("Explicit resume required for an already started run")
        ledger = _check_ledger(_read_json(directory / "ledger.json"), global_claim)
        if not ledger.get("active_session"):
            raise RuntimeError("Training/resume requires a confirmed ACTIVE reservation")
        budget = _Budget(directory, ledger, clock, runtime["device_type"])
        status, error, epochs_this_call = "running", None, 0
        checkpoints, models, optimizers, generators = {}, {}, {}, {}
        try:
            budget.tick()
            if not manifest["initialized"]:
                if frozen_v2_dir is not None:
                    manifest["frozen_v2_dir"] = str(Path(frozen_v2_dir).resolve())
                    manifest["snapshot_manifest_path"] = str(Path(snapshot_manifest_path).resolve())
                    _atomic_json(directory / "manifest.json", manifest)
                data = _initialize(directory, npz_path, expected_sha256, manifest, plan, runtime, budget)
            else:
                data = _load_data(npz_path, expected_sha256, plan)
                if hashlib.sha256(canonical(data["admission"])).hexdigest() != manifest["data_fingerprint"]:
                    raise RuntimeError("Data/scaler fingerprint changed")
            budget.tick()
            device, recipe = runtime["device_type"], plan["training"]
            _determinism(recipe["seed"])
            for name in NEURAL_NAMES:
                budget.tick()
                cp = _load_moment_checkpoint(directory / name / "last.pt", manifest, name)
                recorded = previous.get("models", {}).get(name, {"epoch": 0, "complete": False})
                if cp["epoch"] < recorded["epoch"] or (recorded["complete"] and not cp["complete"]):
                    raise RuntimeError("Checkpoint progress regresses behind acknowledged state")
                _recover_best(directory, name, cp)
                model = _model(name, data, recipe["seed"], device)
                model.load_state_dict(cp["model_state"], strict=True)
                optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["learning_rate"], weight_decay=recipe["weight_decay"])
                optimizer.load_state_dict(cp["optimizer_state"])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(device)
                generator = torch.Generator()
                generator.set_state(cp["dataloader_generator_state"].cpu())
                checkpoints[name], models[name], optimizers[name], generators[name] = cp, model, optimizer, generator
            tensors = {key: torch.from_numpy(data["train_" + key]).to(device) for key in ("X", "A", "U", "y_original")}
            batch_size = recipe["batch_size"]
            while not all(cp["complete"] for cp in checkpoints.values()):
                order = sorted(NEURAL_NAMES, key=lambda name: (checkpoints[name]["epoch"], NEURAL_NAMES.index(name)))
                for name in order:
                    cp, model, optimizer, generator = checkpoints[name], models[name], optimizers[name], generators[name]
                    if cp["complete"]:
                        continue
                    budget.tick()
                    _restore_rng(cp["rng"])
                    if cp["active_permutation"] is None:
                        cp["active_permutation"] = torch.randperm(len(tensors["X"]), generator=generator)
                        cp["dataloader_generator_state"] = generator.get_state()
                    while cp["batch_cursor"] < len(tensors["X"]):
                        budget.tick()
                        indices = cp["active_permutation"][cp["batch_cursor"]:cp["batch_cursor"] + batch_size].to(device)
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        outputs = model.forward_components(tensors["X"][indices], tensors["A"][indices])
                        components = loss_components(outputs, tensors["y_original"][indices], tensors["U"][indices],
                            y_scale=data["scaler"]["y_scale"], u_scale=data["scaler"]["u_scale"],
                            lambda_m=model.lambda_m, lambda_q=model.lambda_q)
                        loss = components["loss"]
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("Nonfinite moment training loss; trial stopped")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip_norm"], error_if_nonfinite=True)
                        optimizer.step()
                        _assert_finite_state(model.state_dict(), optimizer.state_dict())
                        cp["batch_cursor"] += len(indices)
                        cp["epoch_loss_sum"] += float(loss.detach()) * len(indices)
                        cp["epoch_examples"] += len(indices)
                        cp["model_state"], cp["optimizer_state"], cp["rng"] = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), _rng()
                        _save_checkpoint(directory / name / "last.pt", cp)
                        budget.tick()
                    metrics, rows = _validation(model, data, device, batch_size, budget)
                    budget.tick()
                    cp["epoch"] += 1
                    row = {"epoch": cp["epoch"], "training_online_composite_loss": cp["epoch_loss_sum"] / cp["epoch_examples"],
                        "validation_RMSE_original": metrics["RMSE"], "validation_MAE_original": metrics["MAE"],
                        "validation_m_score": metrics["moment_m_score"], "validation_q_score": metrics["moment_q_score"]}
                    cp["history"].append(row)
                    improved = cp["best_validation_rmse"] is None or metrics["RMSE"] < cp["best_validation_rmse"]
                    cp["no_improvement"] = 0 if improved else cp["no_improvement"] + 1
                    if improved:
                        cp["best_validation_rmse"], cp["best_epoch"] = metrics["RMSE"], cp["epoch"]
                    cp["complete"] = cp["epoch"] >= recipe["max_epochs_per_neural_variant"] or cp["no_improvement"] >= recipe["patience"]
                    cp["active_permutation"], cp["batch_cursor"] = None, 0
                    cp["epoch_loss_sum"], cp["epoch_examples"], cp["rng"] = 0., 0, _rng()
                    if improved:
                        cp["best_snapshot"] = {"checkpoint": deepcopy({k: v for k, v in cp.items() if k != "best_snapshot"}),
                            "metrics": metrics, "predictions": rows}
                    _save_checkpoint(directory / name / "last.pt", cp)
                    if improved:
                        _recover_best(directory, name, cp)
                    _csv(directory / name / "curves.csv", cp["history"])
                    epochs_this_call += 1
                    _event(directory, event="epoch_completed", model=name, **row, complete=cp["complete"])
                    print(json.dumps({"event": "epoch_completed", "model": name, **row,
                        "remaining_seconds": budget.tick()}, allow_nan=False), flush=True)
                    _atomic_json(directory / "state.json", {"status": "running", "models": _state_progress(checkpoints, previous)})
                    if stop_after_epochs is not None and epochs_this_call >= stop_after_epochs:
                        status = "paused"
                        break
                if status == "paused":
                    break
            if all(cp["complete"] for cp in checkpoints.values()):
                status = "completed"
        except BudgetExhausted:
            status = "budget_exhausted"
        except BaseException as caught:
            status = "interrupted" if isinstance(caught, (KeyboardInterrupt, SystemExit)) else "failed"
            error = caught
            _event(directory, event="session_error", kind=type(caught).__name__, message=str(caught))
        finally:
            budget.close(status)
            _atomic_json(directory / "state.json", {"status": status, "models": _state_progress(checkpoints, previous)})
        result = _summary(directory, manifest, status)
        if error:
            raise error
        return result
