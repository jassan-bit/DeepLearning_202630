"""Bounded, resumable development of the three fixed innovation-head recipes.

This module performs no network operations. A confirmed global budget claim is
required before fitting a scaler, matching an initial bias, or training. Only
the training-era inner fit and development validation are used. Results remain
descriptive and cannot establish novelty, a p-value, or R3.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import time
import uuid

import numpy as np
import torch

from src.e3_development import (DevelopmentModel, canonical, run_lock,
                                runtime_info, sha256_file)
from src.e3_innovation_model import (NEURAL_NAMES, InnovationModel,
    load_innovation_data, match_initial_biases, loss_components,
    half_student_nll, nominal_interval)


VERSION = "e3-innovation-development-v2"
CLAIM_VERSION = "e3-global-development-budget-v2"
CAMPAIGN = "e3-innovation-v2-20261116"
PRIOR_COMMIT = "f7826d120f3a6eb8e9a0952dd9842cd77f152f34"
PRIOR_CHARGED = 369.4779596389991
PRODUCTION_NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
FROZEN_SNAPSHOT_SHA256 = "159943471990119a0ae80be7f72c9f4b2853bf75f6343d83ace3df9921642742"
FROZEN_BEST_SHA256 = "07d1087c5cfd8fe7c6e7f3e796d86ed8484d0179ba56a506920cd258ba26179c"
ANALYTIC_NAMES = ("floor_only", "floor_plus_fit_mean", "frozen_M00_projected")


class BudgetExhausted(RuntimeError):
    """The single admitted aggregate execution counter has no balance."""


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_bytes(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, value):
    _atomic_bytes(path, canonical(value) + b"\n")


def _csv(path, rows):
    if not rows:
        return
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode())


def _event(directory, **event):
    with (directory / "events.jsonl").open("ab") as stream:
        stream.write(canonical({"utc": _utc(), **event}) + b"\n")
        stream.flush()


def source_fingerprints():
    root = Path(__file__).resolve().parents[1]
    paths = ["src/original_model.py", "src/e3_development.py", "scripts/e3_run_development.py", "src/e3_innovation_model.py",
             "src/e3_innovation.py", "scripts/e3_run_innovation.py",
             "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json",
             "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json.sha256"]
    if any(not (root / p).is_file() for p in paths):
        raise RuntimeError("Required scientific source is absent")
    return {p: sha256_file(root / p) for p in paths}


def validate_plan(plan):
    """Validate the raw operational plan without changing its fingerprint."""
    if not isinstance(plan, dict):
        raise ValueError("An explicit innovation plan is required")
    plan = deepcopy(plan)
    t = plan["training"]
    if t["optimizer"] != "AdamW":
        raise ValueError("Only the fixed AdamW recipe is supported")
    for key in ("seed", "batch_size", "max_epochs_per_neural_variant", "patience"):
        if type(t[key]) is not int or t[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid training {key}")
    if t["seed"] >= 2 ** 32 or t["max_epochs_per_neural_variant"] > 40:
        raise ValueError("Seed/epoch setting exceeds the admitted recipe")
    if (t["learning_rate"] != .001 or t["weight_decay"] != 0 or t["gradient_clip_norm"] != 1
            or t.get("AMP", False) or t.get("TF32", False)):
        raise ValueError("The fixed learning recipe and precision must be preserved")
    if plan["architecture"]["trainable_parameters_per_new_neural_variant"] != 1210:
        raise ValueError("The unchanged M00 backbone must retain 1210 parameters")
    architecture = {"family": "causal_CNN_LSTM_temporal_content_attention", "base_variant": "M00",
        "history_days": 7, "input_features": 5, "cnn_channels": 8, "kernel_size": 3,
        "lstm_hidden_size": 12, "learned_query": True, "discordance_penalty": False,
        "radial_value_saturation": False}
    if any(plan["architecture"].get(k) != v for k, v in architecture.items()):
        raise ValueError("The fixed causal CNN-LSTM-content-attention architecture must be preserved")
    if (plan["distribution"]["degrees_of_freedom"] != 5
            or plan["distribution"]["quadrature_nodes"] != 128
            or plan["loss"]["lambda_nll"] != .1):
        raise ValueError("The fixed distribution, quadrature and loss must be preserved")
    if [x["id"] for x in plan["controls"]] != list(ANALYTIC_NAMES) + list(NEURAL_NAMES[:2]):
        raise ValueError("All five controls must remain in the frozen order")
    if plan["candidate"]["id"] != NEURAL_NAMES[2]:
        raise ValueError("Unexpected innovation candidate")
    if (plan["decision"]["minimum_relative_rmse_improvement_for_continuation"] != .02
            or plan["decision"]["require_all_neural_recipes_completed"] is not True):
        raise ValueError("The fixed exploratory continuation rule must be preserved")
    if plan["data"]["sha256"] == PRODUCTION_NPZ_SHA256:
        if any(t[k] != value for k, value in {"seed": 20261116, "batch_size": 32,
                "max_epochs_per_neural_variant": 40, "patience": 8}.items()):
            raise ValueError("The production pilot must use the frozen training recipe")
        if any(plan["data"][k] != value for k, value in {"fit_end": "2024-06-30",
                "validation_start": "2024-07-08", "validation_end": "2024-10-31",
                "fit_count": 532, "validation_count": 116, "purged_count": 7}.items()):
            raise ValueError("The production development cohort is fixed")
    b = plan["proposed_budget"]
    cap = b["additional_aggregate_charged_cap_seconds"]
    if isinstance(cap, bool) or not math.isfinite(float(cap)) or not 0 < cap <= 600:
        raise ValueError("Additional aggregate budget must be positive and <=600 seconds")
    if (b["prior_charged_seconds"] != PRIOR_CHARGED
            or b["prior_result_commit"] != PRIOR_COMMIT
            or b["global_charged_execution_limit_seconds"] != 1200
            or PRIOR_CHARGED + cap > 1200):
        raise ValueError("The sealed v1 cost and global budget must be preserved")
    canonical(plan)
    return plan


def validate_global_claim(claim, plan, expected_sha256, run_id=None):
    """Check the local ACK receipt; the coordinator separately verifies its CAS."""
    if not isinstance(claim, dict):
        raise RuntimeError("A confirmed global budget claim is required before any fit")
    expected = {"version": CLAIM_VERSION, "campaign_id": CAMPAIGN,
        "v1_commit": PRIOR_COMMIT, "prior_charged_seconds": PRIOR_CHARGED,
        "additional_limit_seconds": plan["proposed_budget"]["additional_aggregate_charged_cap_seconds"],
        "global_limit_seconds": 1200.0,
        "plan_sha256": hashlib.sha256(canonical(plan)).hexdigest(),
        "npz_sha256": expected_sha256.lower()}
    if any(claim.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Global budget claim/plan/input mismatch")
    for key in ("run_id", "claim_id"):
        if not isinstance(claim.get(key), str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", claim[key]):
            raise RuntimeError("Invalid preassigned global run/claim identity")
    if not re.fullmatch(r"[0-9a-f]{40}", str(claim.get("registry_commit", ""))):
        raise RuntimeError("A confirmed registry commit ACK is required")
    if run_id is not None and claim["run_id"] != run_id:
        raise RuntimeError("The global claim belongs to another run")
    return deepcopy(claim)


def _check_ledger(ledger, claim):
    ledger = deepcopy(ledger)
    limit, charged = ledger.get("limit_seconds"), ledger.get("charged_seconds")
    if (limit != claim["additional_limit_seconds"] or isinstance(charged, bool)
            or not isinstance(charged, (int, float)) or not math.isfinite(charged) or charged < 0):
        raise RuntimeError("Invalid aggregate innovation ledger")
    if not isinstance(ledger.get("events"), list):
        raise RuntimeError("Ledger events must be retained")
    active = ledger.get("active_session")
    if active is not None:
        stamp = datetime.fromisoformat(active["last_wall_utc"])
        if stamp.tzinfo is None or not active.get("session_id"):
            raise RuntimeError("Invalid active reservation timestamp")
    return ledger


class _Budget:
    def __init__(self, directory, ledger, clock, device):
        self.directory, self.ledger, self.clock, self.device = directory, ledger, clock, device
        self.last = float(clock())
        self.session_id = uuid.uuid4().hex
        if not math.isfinite(self.last):
            raise ValueError("Invalid monotonic clock")
        old = ledger.get("active_session")
        if old:
            gap = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(old["last_wall_utc"])).total_seconds())
            cost = min(gap, max(0.0, ledger["limit_seconds"] - ledger["charged_seconds"]))
            ledger["charged_seconds"] += cost
            ledger["events"].append({"kind": "unclosed_session_recovery", "seconds": cost,
                "previous_session": old["session_id"], "previous_last_wall_utc": old["last_wall_utc"]})
        ledger["active_session"] = {"session_id": self.session_id, "last_wall_utc": _utc()}
        _atomic_json(directory / "ledger.json", ledger)

    def tick(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        now = float(self.clock())
        if not math.isfinite(now) or now < self.last:
            raise ValueError("Clock must be finite and monotonic")
        self.ledger["charged_seconds"] += now - self.last
        self.last = now
        self.ledger["active_session"]["last_wall_utc"] = _utc()
        _atomic_json(self.directory / "ledger.json", self.ledger)
        remaining = self.ledger["limit_seconds"] - self.ledger["charged_seconds"]
        if remaining <= 0:
            raise BudgetExhausted("Aggregate innovation execution budget exhausted")
        return remaining

    def close(self, status):
        try:
            self.tick()
        except BudgetExhausted:
            pass
        self.ledger["events"].append({"kind": "session_closed", "session_id": self.session_id,
            "status": status, "charged_total_seconds": self.ledger["charged_seconds"]})
        self.ledger["active_session"] = None
        _atomic_json(self.directory / "ledger.json", self.ledger)


def _determinism(seed):
    from src.e3_development import _configure_determinism
    _configure_determinism(seed)


def _rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"algorithm": state[0],
        "keys": torch.tensor(state[1].astype(np.int64)), "position": state[2],
        "has_gauss": state[3], "cached_gaussian": state[4]},
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["algorithm"], n["keys"].cpu().numpy().astype(np.uint32),
                        n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA generator topology changed")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def _save_checkpoint(path, value):
    """Two-generation digest transaction; last.pt is the sole authoritative state."""
    path = Path(path)
    buffer = io.BytesIO()
    torch.save(value, buffer)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    receipt = path.with_suffix(path.suffix + ".sha256")
    accepted = ([sha256_file(path)] if path.exists() else []) + [digest]
    _atomic_bytes(receipt, ("\n".join(dict.fromkeys(accepted)) + "\n").encode())
    _atomic_bytes(path, payload)
    _atomic_bytes(receipt, (digest + "\n").encode())


def _load_checkpoint(path, manifest):
    path = Path(path)
    receipt = path.with_suffix(path.suffix + ".sha256")
    if not receipt.is_file() or sha256_file(path) not in receipt.read_text().splitlines():
        raise RuntimeError("Checkpoint byte hash mismatch before deserialization")
    value = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("source_hashes", "plan_sha256", "data_fingerprint", "run_id", "claim_id"):
        if value[key] != manifest[key]:
            raise RuntimeError("Checkpoint provenance mismatch")
    return value


def _load_data(npz_path, expected_sha256, plan):
    d = plan["data"]
    data = load_innovation_data(npz_path, expected_sha256, fit_end=d["fit_end"],
        validation_start=d["validation_start"], validation_end=d["validation_end"])
    counts = (len(data["train_X"]), len(data["validation_X"]), data["admission"]["n_purged_or_unused"])
    if counts != (d["fit_count"], d["validation_count"], d["purged_count"]):
        raise RuntimeError("Data cohort counts disagree with the admitted plan")
    return data


def _safe_frozen_load(path):
    # The immutable v1 checkpoint contains a NumPy MT19937 array. This narrow
    # allowlist admits that array while retaining weights_only restrictions.
    reconstruct = np._core.multiarray._reconstruct
    allowed = [reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def _frozen_predictions(directory, snapshot_path, data, plan, device, budget):
    directory, snapshot_path = Path(directory), Path(snapshot_path)
    config = plan["frozen_control"]
    expected = config["snapshot_manifest_sha256"]
    if (plan["data"]["sha256"] == PRODUCTION_NPZ_SHA256
            and (expected != FROZEN_SNAPSHOT_SHA256 or config["best_checkpoint_sha256"] != FROZEN_BEST_SHA256)):
        raise RuntimeError("The production frozen control is anchored to sealed v1 bytes")
    if sha256_file(snapshot_path) != expected or config["source_commit"] != PRIOR_COMMIT:
        raise RuntimeError("Frozen snapshot manifest/source mismatch")
    snapshot = _read_json(snapshot_path)
    if snapshot["active_session"] is not None or snapshot["state_status"] != "completed":
        raise RuntimeError("Frozen v1 control must be a completed closed snapshot")
    if snapshot["charged_seconds"] != PRIOR_CHARGED:
        raise RuntimeError("Frozen snapshot does not prove the sealed prior cost")
    names = ("M00/best.pt", "data_admission.json", "manifest.json", "plan.json")
    for name in names:
        budget.tick()
        path, record = directory / name, snapshot["files"][name]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError("Frozen snapshot member byte hash/size mismatch")
    if snapshot["files"]["M00/best.pt"]["sha256"] != config["best_checkpoint_sha256"]:
        raise RuntimeError("Frozen M00 checkpoint hash mismatch")
    admission, manifest = _read_json(directory / "data_admission.json"), _read_json(directory / "manifest.json")
    if (hashlib.sha256(canonical(admission)).hexdigest() != snapshot["data_fingerprint"]
            or manifest["data_fingerprint"] != snapshot["data_fingerprint"]
            or manifest["plan_sha256"] != snapshot["plan_sha256"]
            or hashlib.sha256(canonical(_read_json(directory / "plan.json"))).hexdigest() != snapshot["plan_sha256"]):
        raise RuntimeError("Frozen v1 data/plan provenance mismatch")
    for key in ("mean", "std", "target_scale"):
        if not np.array_equal(np.asarray(admission["scaler"][key]), np.asarray(data["scaler"][key])):
            raise RuntimeError("Frozen M00 preprocessing differs from this inner fit")
    if admission["validation_dates"] != data["validation_dates"].tolist():
        raise RuntimeError("Frozen M00 validation cohort differs")
    for name in ("train_X", "validation_X", "train_y_original", "validation_y_original"):
        value = np.ascontiguousarray(data[name])
        digest = hashlib.sha256(str(value.dtype).encode() + canonical(list(value.shape)) + value.tobytes()).hexdigest()
        if digest != admission["arrays_sha256"][name]:
            raise RuntimeError("Frozen M00 input/target array bytes differ from its sealed cohort")
    for name, digest in snapshot["source_hashes"].items():
        path = Path(__file__).resolve().parents[1] / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError("Frozen v1 implementation bytes changed")
    budget.tick()
    checkpoint = _safe_frozen_load(directory / "M00/best.pt")
    for key in ("source_hashes", "plan_sha256", "data_fingerprint"):
        if checkpoint[key] != snapshot[key]:
            raise RuntimeError("Frozen checkpoint provenance differs from sealed snapshot")
    model = DevelopmentModel("M00", _read_json(directory / "plan.json")["seed"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.requires_grad_(False).eval()
    out = []
    with torch.no_grad():
        for first in range(0, len(data["validation_X"]), plan["training"]["batch_size"]):
            budget.tick()
            x = torch.from_numpy(data["validation_X"][first:first + plan["training"]["batch_size"]]).to(device)
            out.append(model(x).cpu().numpy().astype(np.float64) * data["scaler"]["y_scale"])
    predicted = np.maximum(np.concatenate(out), np.sqrt(data["validation_A"]))
    return predicted, {"source_commit": PRIOR_COMMIT, "snapshot_manifest_sha256": expected,
                       "best_checkpoint_sha256": config["best_checkpoint_sha256"],
                       "best_epoch": checkpoint["epoch"], "parameters": 1210,
                       "weights_only": True, "new_fit": False}


def _metrics(predicted, data, scales=None):
    y, floor = data["validation_y_original"], np.sqrt(data["validation_A"])
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - y
    high = y >= np.quantile(data["train_y_original"], .75)
    if not np.isfinite(predicted).all():
        raise RuntimeError("Nonfinite validation prediction")
    result = {"RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "MAE": float(np.mean(np.abs(residual))), "bias": float(residual.mean()),
        "high_volatility_count": int(high.sum()),
        "high_volatility_RMSE": float(np.sqrt(np.mean(residual[high] ** 2))) if high.any() else None,
        "high_volatility_MAE": float(np.mean(np.abs(residual[high]))) if high.any() else None,
        "below_floor_count": int(np.sum(predicted < floor - 1e-14)),
        "high_volatility_threshold_fit_q75": float(np.quantile(data["train_y_original"], .75)),
        "NLL_U": None, "nominal_90pct_coverage": None}
    low = upper = nll = None
    if scales is not None:
        u_scale = data["scaler"]["u_scale"]
        scale_t = torch.as_tensor(scales, dtype=torch.float64)
        nll = half_student_nll(torch.from_numpy(data["validation_U"] / u_scale), scale_t / u_scale).detach().cpu().numpy()
        low_t, upper_t = nominal_interval(torch.from_numpy(data["validation_A"]), scale_t, coverage=.9)
        low, upper = low_t.detach().cpu().numpy(), upper_t.detach().cpu().numpy()
        result.update(NLL_U=float(np.mean(nll)), nominal_90pct_coverage=float(np.mean((y >= low) & (y <= upper))))
    rows = [{"date": str(day), "target_sigma7": float(target), "prediction_sigma7": float(pred),
             "known_floor": float(fl), "scale_U": float(scales[i]) if scales is not None else None,
             "NLL_U": float(nll[i]) if nll is not None else None,
             "nominal_05": float(low[i]) if low is not None else None,
             "nominal_95": float(upper[i]) if upper is not None else None}
            for i, (day, target, pred, fl) in enumerate(zip(data["validation_dates"], y, predicted, floor))]
    return result, rows


def _new_checkpoint(model, optimizer, manifest, seed):
    return {"version": VERSION, "model_name": model.variant,
        "model_state": deepcopy(model.state_dict()), "optimizer_state": deepcopy(optimizer.state_dict()),
        "rng": _rng(), "dataloader_generator_state": torch.Generator().manual_seed(seed).get_state(),
        "epoch": 0, "active_permutation": None, "batch_cursor": 0,
        "epoch_loss_sum": 0.0, "epoch_examples": 0, "no_improvement": 0,
        "best_validation_rmse": None, "best_epoch": None, "history": [], "complete": False,
        **{k: manifest[k] for k in ("source_hashes", "plan_sha256", "data_fingerprint", "run_id", "claim_id")}}


def _recover_best(directory, name, checkpoint):
    snapshot = checkpoint.get("best_snapshot")
    if not snapshot:
        if checkpoint["best_epoch"] is not None:
            raise RuntimeError("Best checkpoint recovery record missing")
        return
    # These are derived artifacts: always reproduce them from authoritative last.
    _save_checkpoint(directory / name / "best.pt", snapshot["checkpoint"])
    _csv(directory / name / "validation_predictions.csv", snapshot["predictions"])
    _csv(directory / name / "curves.csv", checkpoint["history"])


def _state_progress(checkpoints, previous=None):
    """Retain all three recipes when setup stops before loading every model.

    Setup does not advance epochs. For recipes not yet loaded, the preceding
    committed state remains authoritative; a fresh initialization starts at 0.
    """
    previous_models = (previous or {}).get("models", {})
    return {name: {"epoch": checkpoints[name]["epoch"], "complete": checkpoints[name]["complete"]}
            if name in checkpoints else deepcopy(previous_models.get(name, {"epoch": 0, "complete": False}))
            for name in NEURAL_NAMES}


def _summary(directory, manifest, status):
    ledger = _read_json(directory / "ledger.json")
    models = {}
    for name in NEURAL_NAMES:
        path = directory / name / "last.pt"
        if not path.is_file() or not manifest.get("initialized"):
            models[name] = {"completed_epochs": 0, "complete": False, "parameters": 1210,
                            "best_epoch": None, "metrics": None}
            continue
        cp = _load_checkpoint(path, manifest)
        snapshot = cp.get("best_snapshot")
        models[name] = {"completed_epochs": cp["epoch"], "complete": cp["complete"],
            "best_epoch": cp["best_epoch"], "parameters": 1210,
            "metrics": snapshot["metrics"] if snapshot else None,
            "last_checkpoint_sha256": sha256_file(path)}
    controls_path = directory / "controls.json"
    controls = _read_json(controls_path)["metrics"] if controls_path.is_file() else {}
    all_metrics = {**controls, **{n: x["metrics"] for n, x in models.items() if x["metrics"]}}
    controls_complete = set(controls) == set(ANALYTIC_NAMES)
    controls_complete = controls_complete and all(
        isinstance(m, dict) and all(isinstance(m.get(k), (int, float))
                                   and math.isfinite(m[k]) for k in ("RMSE", "MAE", "bias"))
        for m in controls.values())
    complete = all(m["complete"] for m in models.values()) and controls_complete and len(all_metrics) == 6
    best_control = improvement = gate = None
    if complete and len(all_metrics) == 6:
        best_control = min(list(ANALYTIC_NAMES) + list(NEURAL_NAMES[:2]), key=lambda n: all_metrics[n]["RMSE"])
        baseline = all_metrics[best_control]["RMSE"]
        improvement = (baseline - all_metrics[NEURAL_NAMES[2]]["RMSE"]) / baseline if baseline else None
        gate = improvement is not None and improvement >= .02
    for name in ANALYTIC_NAMES:
        models[name] = {"completed_epochs": 0, "complete": name in controls,
            "best_epoch": None, "parameters": 1210 if name == "frozen_M00_projected" else (1 if name == "floor_plus_fit_mean" else 0),
            "metrics": controls.get(name)}
    for name, row in models.items():
        metric = row["metrics"] or {}
        row.update({"RMSE": metric.get("RMSE"), "MAE": metric.get("MAE"),
                    "bias": metric.get("bias"), "coverage90": metric.get("nominal_90pct_coverage"),
                    "NLLU": metric.get("NLL_U")})
    result = {"version": VERSION, "run_id": manifest["run_id"], "claim_id": manifest["claim_id"],
        "status": status, "models": models, "metrics": all_metrics,
        "comparison_complete": complete, "pilot_incomplete": not complete,
        "best_fixed_control": best_control, "relative_RMSE_improvement": improvement,
        "exploratory_continuation_gate_passed": gate if complete else False,
        "decision": {"status": "complete" if complete else "incomplete", "complete": complete,
            "best_control": best_control, "improvement_fraction": improvement,
            "continuation_gate_passed": bool(gate) if complete else False,
            "interpretation": "exploratory 2% continuation heuristic; not a confirmatory margin"},
        "winner_declared": False, "statistical_tests_performed": False,
        "charged_seconds": ledger["charged_seconds"], "limit_seconds": ledger["limit_seconds"],
        "prior_charged_seconds": PRIOR_CHARGED,
        "global_charged_seconds": PRIOR_CHARGED + ledger["charged_seconds"],
        "global_limit_seconds": 1200.0,
        "budget_remaining_seconds": max(0.0, ledger["limit_seconds"] - ledger["charged_seconds"]),
        "budget_overshoot_seconds": max(0.0, ledger["charged_seconds"] - ledger["limit_seconds"]),
        "scope": "one-seed development validation reused for stopping; no p-values, R1 certification or R3",
        "calibration_scored": False, "evaluation_scored": False,
        "source_hashes": manifest["source_hashes"], "plan_sha256": manifest["plan_sha256"],
        "data_fingerprint": manifest.get("data_fingerprint")}
    _atomic_json(directory / "summary.json", result)
    return result


def _initialize(directory, npz_path, expected_sha256, manifest, plan, runtime, budget):
    budget.tick()
    data = _load_data(npz_path, expected_sha256, plan)
    budget.tick()
    manifest["data_fingerprint"] = hashlib.sha256(canonical(data["admission"])).hexdigest()
    _atomic_json(directory / "data_admission.json", data["admission"])
    device, seed = runtime["device_type"], plan["training"]["seed"]
    _determinism(seed)
    models = {n: InnovationModel(n, seed=seed, u_scale=data["scaler"]["u_scale"],
                               y_scale=data["scaler"]["y_scale"]).to(device) for n in NEURAL_NAMES}
    budget.tick()
    x = torch.from_numpy(data["train_X"]).to(device)
    a = torch.from_numpy(data["train_A"]).to(device)
    report = match_initial_biases(models, x, a, data["scaler"]["d_fit"],
                                  data["scaler"]["y_scale"], budget_callback=budget.tick)
    budget.tick()
    frozen, frozen_info = _frozen_predictions(manifest["frozen_v1_dir"], manifest["snapshot_manifest_path"],
                                             data, plan, device, budget)
    controls, rows = {}, []
    predictions = {"floor_only": np.sqrt(data["validation_A"]),
        "floor_plus_fit_mean": np.sqrt(data["validation_A"]) + data["scaler"]["d_fit"],
        "frozen_M00_projected": frozen}
    for name, predicted in predictions.items():
        budget.tick()
        controls[name], records = _metrics(predicted, data)
        rows.extend({"model": name, **r} for r in records)
    _atomic_json(directory / "controls.json", {"metrics": controls, "frozen_control": frozen_info})
    controls_dir = directory / "controls"
    controls_dir.mkdir(exist_ok=True)
    _csv(controls_dir / "validation_predictions.csv", rows)
    manifest["controls_sha256"] = sha256_file(directory / "controls.json")
    manifest["control_predictions_sha256"] = sha256_file(controls_dir / "validation_predictions.csv")
    manifest["initial_bias_matching"] = report
    manifest["models"] = {n: models[n].configuration() for n in NEURAL_NAMES}
    for name, model in models.items():
        budget.tick()
        if sum(p.numel() for p in model.parameters()) != 1210:
            raise RuntimeError("Innovation model no longer has 1210 parameters")
        subdir = directory / name
        subdir.mkdir(exist_ok=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan["training"]["learning_rate"],
                                      weight_decay=plan["training"]["weight_decay"])
        _save_checkpoint(subdir / "last.pt", _new_checkpoint(model, optimizer, manifest, seed))
    budget.tick()
    manifest["initialized"] = True
    _atomic_json(directory / "manifest.json", manifest)
    _event(directory, event="initial_bias_matching_completed", report=report)
    return data


def create_innovation_run(npz_path, expected_sha256, run_dir, *, global_claim=None,
        initial_ledger=None, device="auto", plan_path=None, plan=None, frozen_v1_dir=None,
        snapshot_manifest_path=None, clock=time.monotonic):
    """Initialize a billed run only after the coordinator confirms its claim."""
    if (plan is None) == (plan_path is None):
        raise ValueError("Supply exactly one explicit plan or plan_path")
    plan = validate_plan(_read_json(plan_path) if plan_path is not None else plan)
    claim = validate_global_claim(global_claim, plan, expected_sha256)
    if plan["data"]["sha256"] != expected_sha256.lower():
        raise RuntimeError("Plan NPZ fingerprint mismatch")
    if frozen_v1_dir is None or snapshot_manifest_path is None:
        raise ValueError("The sealed frozen-v1 directory and snapshot manifest are required")
    if initial_ledger is None:
        raise RuntimeError("The confirmed ACTIVE reservation ledger is required before initialization")
    ledger = _check_ledger(initial_ledger, claim)
    if not ledger.get("active_session"):
        raise RuntimeError("Initialization requires the coordinator's ACTIVE timestamp")
    runtime = runtime_info(device)
    runtime["torch"] = str(runtime["torch"])
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"version": VERSION, "run_id": claim["run_id"], "claim_id": claim["claim_id"],
        "created_at_utc": _utc(), "initialized": False, "global_claim": claim,
        "npz_sha256": expected_sha256.lower(), "plan_sha256": claim["plan_sha256"],
        "source_hashes": source_fingerprints(), "runtime_initial": runtime,
        "frozen_v1_dir": str(Path(frozen_v1_dir).resolve()),
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
            status, error = "failed", caught
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
        raise RuntimeError("Cannot switch global campaign claims")
    if source_fingerprints() != manifest["source_hashes"]:
        raise RuntimeError("Source fingerprint changed; resume requires the recorded source bytes")
    if hashlib.sha256(canonical(plan)).hexdigest() != manifest["plan_sha256"]:
        raise RuntimeError("Plan fingerprint changed")
    if expected_sha256.lower() != manifest["npz_sha256"] or sha256_file(npz_path) != expected_sha256.lower():
        raise RuntimeError("NPZ fingerprint changed")
    if manifest["initialized"]:
        for filename, key in (("controls.json", "controls_sha256"),
                              ("controls/validation_predictions.csv", "control_predictions_sha256")):
            if not (directory / filename).is_file() or sha256_file(directory / filename) != manifest[key]:
                raise RuntimeError("Fixed control metrics/predictions changed or are absent")
    runtime = runtime_info(device)
    runtime["torch"] = str(runtime["torch"])
    old = manifest["runtime_initial"]
    keys = ("python", "numpy", "torch", "device_type", "cuda_build", "gpu", "cpu_threads")
    if any(runtime.get(k) != old.get(k) for k in keys):
        raise RuntimeError("Exact continuation requires the original numerical runtime and device")
    return manifest, plan, runtime


def _validation(model, data, device, batch_size, budget):
    predictions, scales = [], []
    model.eval()
    with torch.no_grad():
        for first in range(0, len(data["validation_X"]), batch_size):
            budget.tick()
            x = torch.from_numpy(data["validation_X"][first:first + batch_size]).to(device)
            a = torch.from_numpy(data["validation_A"][first:first + batch_size]).to(device)
            out = model.forward_components(x, a)
            predictions.append(out["prediction"].cpu().numpy())
            if model.variant != NEURAL_NAMES[0]:
                scales.append(out["scale"].cpu().numpy())
    return _metrics(np.concatenate(predictions), data, np.concatenate(scales) if scales else None)


def run_innovation(run_dir, npz_path, expected_sha256, *, global_claim=None,
        resume=False, device="auto", stop_after_epochs=None, clock=time.monotonic,
        frozen_v1_dir=None, snapshot_manifest_path=None):
    """Interleave one epoch per live NN, with exact batch-cursor checkpoints."""
    if stop_after_epochs is not None and (type(stop_after_epochs) is not int or stop_after_epochs < 1):
        raise ValueError("stop_after_epochs must be a positive integer")
    if (frozen_v1_dir is None) != (snapshot_manifest_path is None):
        raise ValueError("Supply both frozen control paths together")
    directory = Path(run_dir)
    with run_lock(directory):
        manifest, plan, runtime = _validate_resume(directory, npz_path, expected_sha256, global_claim, device)
        previous = _read_json(directory / "state.json")
        if previous["status"] != "initialized" and not resume:
            raise RuntimeError("Explicit resume is required for an already started run")
        if previous["status"] == "completed":
            return _summary(directory, manifest, "completed")
        ledger = _check_ledger(_read_json(directory / "ledger.json"), global_claim)
        if ledger.get("active_session") is None:
            raise RuntimeError("Training/resume requires a confirmed ACTIVE global reservation")
        budget = _Budget(directory, ledger, clock, runtime["device_type"])
        status, error, epochs_this_call = "running", None, 0
        checkpoints, models, optimizers, generators = {}, {}, {}, {}
        try:
            budget.tick()
            if not manifest["initialized"]:
                if frozen_v1_dir is not None:
                    # Paths are local transport details. Byte-pinned baseline
                    # identity is rechecked in _frozen_predictions before use.
                    manifest["frozen_v1_dir"] = str(Path(frozen_v1_dir).resolve())
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
                cp = _load_checkpoint(directory / name / "last.pt", manifest)
                _recover_best(directory, name, cp)
                model = InnovationModel(name, seed=recipe["seed"], u_scale=data["scaler"]["u_scale"],
                                        y_scale=data["scaler"]["y_scale"]).to(device)
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
            tensors = {key: torch.from_numpy(data["train_" + key]).to(device)
                       for key in ("X", "A", "U", "y_original")}
            batch_size = recipe["batch_size"]
            while not all(cp["complete"] for cp in checkpoints.values()):
                order = sorted(NEURAL_NAMES, key=lambda n: (checkpoints[n]["epoch"], NEURAL_NAMES.index(n)))
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
                        idx = cp["active_permutation"][cp["batch_cursor"]:cp["batch_cursor"] + batch_size].to(device)
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        outputs = model.forward_components(tensors["X"][idx], tensors["A"][idx])
                        loss = loss_components(outputs, tensors["y_original"][idx], tensors["U"][idx],
                            y_scale=data["scaler"]["y_scale"], u_scale=data["scaler"]["u_scale"],
                            lambda_nll=model.lambda_nll)["loss"]
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError("Nonfinite innovation training loss")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["gradient_clip_norm"], error_if_nonfinite=True)
                        optimizer.step()
                        cp["batch_cursor"] += len(idx)
                        cp["epoch_loss_sum"] += float(loss.detach()) * len(idx)
                        cp["epoch_examples"] += len(idx)
                        cp["model_state"], cp["optimizer_state"], cp["rng"] = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), _rng()
                        _save_checkpoint(directory / name / "last.pt", cp)
                        budget.tick()
                    metrics, rows = _validation(model, data, device, batch_size, budget)
                    budget.tick()
                    cp["epoch"] += 1
                    row = {"epoch": cp["epoch"], "training_online_composite_loss": cp["epoch_loss_sum"] / cp["epoch_examples"],
                           "validation_RMSE_original": metrics["RMSE"], "validation_MAE_original": metrics["MAE"]}
                    cp["history"].append(row)
                    improved = cp["best_validation_rmse"] is None or metrics["RMSE"] < cp["best_validation_rmse"]
                    cp["no_improvement"] = 0 if improved else cp["no_improvement"] + 1
                    if improved:
                        cp["best_validation_rmse"], cp["best_epoch"] = metrics["RMSE"], cp["epoch"]
                    cp["complete"] = cp["epoch"] >= recipe["max_epochs_per_neural_variant"] or cp["no_improvement"] >= recipe["patience"]
                    cp["active_permutation"], cp["batch_cursor"] = None, 0
                    cp["epoch_loss_sum"], cp["epoch_examples"], cp["rng"] = 0.0, 0, _rng()
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
                    _atomic_json(directory / "state.json", {"status": "running", "models": {
                        n: {"epoch": c["epoch"], "complete": c["complete"]} for n, c in checkpoints.items()}})
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
            status, error = ("interrupted" if isinstance(caught, (KeyboardInterrupt, SystemExit)) else "failed"), caught
            _event(directory, event="session_error", kind=type(caught).__name__, message=str(caught))
        finally:
            budget.close(status)
            _atomic_json(directory / "state.json", {"status": status,
                "models": _state_progress(checkpoints, previous)})
        result = _summary(directory, manifest, status)
        if error:
            raise error
        return result
