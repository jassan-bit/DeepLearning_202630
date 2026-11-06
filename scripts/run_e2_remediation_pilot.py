#!/usr/bin/env python3
"""Registered, development-only D7 feasibility/variance pilot and optional OOF.

No Bitcoin evaluation period after 2024 is accepted by this driver. Fitting
decodes only the prefix ending at origin-8; scoring is a later explicit phase
requiring every scheduled checkpoint to be closed. This pilot does not select
architectures, replace HPO, or establish the power of a future final procedure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_data import CALENDAR, FEATURES, SOURCE_PATHS, make_windows, mase_scale
from dl_data_v2 import fit_relative_preprocessing, transform_relative_windows, split_for_origin
from e2_temporal_models import MODELS, TOP_MODELS, architecture_spec, build_model, count_parameters

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/e2_remediation_20261106/pilot_design.json"
SEEDS = tuple(range(660101, 660106))
PILOT_ORIGIN = "2024-11-01"
PILOT_END = "2024-12-30"
OOF_ORIGINS = tuple(f"2024-{month:02d}-01" for month in range(1, 11))
DELTA = 0.00025772842664981106
HIGH_VOL_THRESHOLD = 0.024247163097617024
E1_SCIENTIFIC_SHA256 = "1464fa79d472f6755651774989b140659ee1f21daeb30032fc650a8770f4fdc4"


def digest(path): return sha256(Path(path).read_bytes()).hexdigest()
def stamp(): return datetime.now(timezone.utc).isoformat()
def json_bytes(value): return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
def read_json(path): return json.loads(Path(path).read_bytes())
def write_json(path, value, exclusive=False):
    with Path(path).open("xb" if exclusive else "wb") as stream: stream.write(json_bytes(value))


def read_plan(path):
    p = read_json(path)
    if p.get("scope") != "P0_development_feasibility_variance_gap_not_final_benchmark":
        raise ValueError("This driver is restricted to the registered P0 development pilot")
    if tuple(p["seeds"]) != SEEDS or tuple(p["models"]) != MODELS:
        raise ValueError("The fixed paired model/seed registry differs")
    if p["input_contract"] != {"D": 7, "n": 7, "h": 1, "channels": list(FEATURES), "target_lags": False}:
        raise ValueError("The inherited D7×5, n7, h1 contract differs")
    dates = p["calendar"]
    if dates.get("pilot_origin") != PILOT_ORIGIN or dates.get("pilot_end") != PILOT_END:
        raise ValueError("The registered development pilot calendar differs")
    if tuple(dates.get("OOF_origins", [])) != OOF_ORIGINS or dates.get("max_decoded_outcome") != PILOT_END:
        raise ValueError("OOF origins or hard development date boundary differs")
    if p["margin_absolute"] != DELTA or p["known_high_volatility_threshold"] != HIGH_VOL_THRESHOLD:
        raise ValueError("The inherited absolute margin or causal regime threshold differs")
    training = p["training"]
    expected = {"optimizer": "AdamW", "learning_rate": .003, "weight_decay": .0001,
                "batch_size": 32, "max_epochs": 100, "patience": 12,
                "min_delta": 1e-5, "gradient_clip_norm": 1.0, "torch_threads": 1,
                "inner_validation_nominal_days": 90, "refit_seed_offset": 10000000}
    if any(training.get(k) != value for k, value in expected.items()):
        raise ValueError("The registered fixed P0 training recipe differs")
    if p["fresh_GRU_reference"] != {"candidate_id": "c03", "width": 16,
                                     "dropout": .1, "learning_rate": .003}:
        raise ValueError("The inherited GRU-c03 reference recipe differs")
    if set(p["development_sources"]) != set(SOURCE_PATHS) or set(p["author_sources"]) != {"TimeMixer", "controls"}:
        raise ValueError("Only the two development files and two pinned author source trees are admitted")
    if p.get("inherited_scientific_sha256") != E1_SCIENTIFIC_SHA256:
        raise ValueError("The protected scientific E1 inheritance differs")
    if not isinstance(p["OOF"]["enabled"], bool):
        raise ValueError("The OOF decision must be a fixed boolean before registration")
    dimensions = {"d_model": 16, "d_ff": 32, "encoder_layers": 2, "dropout": .1,
                  "attention_heads_if_used": 4, "activation": "gelu"}
    if p["adapters"]["shared_Top10_dimensions"] != dimensions:
        raise ValueError("The fixed pilot encoder dimensions differ")
    tm, pt = p["adapters"]["TimeMixer"], p["adapters"]["PatchTST"]
    if (tm["scales"] != [7, 3] or tm["moving_avg"] != 25 or tm["channel_independence"] != 1
            or tm["use_norm"] != 1 or pt["patch_len"] != 2 or pt["stride"] != 1):
        raise ValueError("The fixed pilot multiscale/patching recipe differs")
    return p


def load_development_prefix(root, last_outcome):
    """Decode only an admissible prefix; never parse a later outcome row.

    Full-file hashes can be verified independently without decoding their
    contents. The returned fixed 731-position calendar has NaN arrays outside
    this prefix so the unchanged E1 split/preprocessing helpers remain usable.
    """
    last = pd.Timestamp(last_outcome, tz="UTC")
    if pd.isna(last) or last != last.normalize() or last < CALENDAR[0] or last > pd.Timestamp(PILOT_END, tz="UTC"):
        raise ValueError("Only development outcomes through 2024-12-30 may be decoded")
    n = int((last - CALENDAR[0]).days) + 1
    market = pd.read_csv(Path(root) / SOURCE_PATHS[0], usecols=["market_day_utc", *FEATURES], nrows=n)
    target = pd.read_csv(Path(root) / SOURCE_PATHS[1], usecols=["market_day_utc", "sigma_7"], nrows=n)
    for frame in (market, target):
        dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("market_day_utc"), utc=True))
        if not dates.equals(CALENDAR[:n]): raise ValueError("Development prefix/calendar differs")
    observed = market.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    labels = target.sigma_7.to_numpy(dtype=np.float64)
    complete = np.isfinite(observed).all(axis=1)
    invalid = CALENDAR[:n][~complete].strftime("%Y-%m-%d").tolist()
    if invalid != (["2023-03-24"] if last >= pd.Timestamp("2023-03-24", tz="UTC") else []):
        raise ValueError("Inherited incomplete-day mask differs")
    good = observed[complete]
    if (np.isinf(observed).any() or np.isinf(labels).any() or (good[:, :4] <= 0).any()
            or (good[:, 4] < 0).any()
            or (good[:, 1] < np.maximum(good[:, 0], good[:, 3])).any()
            or (good[:, 2] > np.minimum(good[:, 0], good[:, 3])).any()):
        raise ValueError("Development OHLCV values are invalid")
    returns = pd.Series(np.log(observed[:, 3]), index=CALENDAR[:n]).diff()
    recomputed = returns.rolling(7, min_periods=7).std(ddof=0).to_numpy()
    if not np.allclose(labels, recomputed, equal_nan=True, rtol=1e-9, atol=1e-13):
        raise ValueError("Target is not the inherited population seven-return volatility")
    features = np.full((len(CALENDAR), 5), np.nan); features[:n] = observed
    targets = np.full(len(CALENDAR), np.nan); targets[:n] = labels
    windows = make_windows(features, 7)
    eligible = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(targets)
    return {"calendar": CALENDAR.copy(), "features": features, "targets": targets,
            "windows_by_history": {7: windows}, "eligible_by_history": {7: eligible},
            "persistence": np.r_[np.nan, targets[:-1]],
            "decoded_outcome_max": last.strftime("%Y-%m-%d")}


def source_root_for(model, args):
    return args.source_root if model == "TimeMixer" else args.control_source_root


def immutable_inputs(plan, plan_path, args):
    paths = ["scripts/e2_temporal_models.py", "scripts/run_e2_remediation_pilot.py",
             "scripts/execute_e2_development_pilot.py", "scripts/e2_remediation_data_audit.py",
             "requirements-paper-guide.lock.txt",
             "scripts/dl_data.py", "scripts/dl_data_v2.py", "scripts/dl_models.py",
             "scripts/dl_models_gru_lstm_tcn.py", "requirements-e2-remediation.lock.txt"]
    project = Path(args.project_root)
    reference = read_json(project / "book_e2/_static/e1_reference.json")
    keys = ("source_commit", "registered_baseline", "base_model_recipes", "recipe_architectures",
            "primary_metric", "practical_margin", "independence", "partition")
    inherited = sha256(json_bytes({key: reference[key] for key in keys})).hexdigest()
    if inherited != E1_SCIENTIFIC_SHA256:
        raise ValueError("The actual E1 scientific inheritance changed")
    for name, expected in plan["development_sources"].items():
        if name not in SOURCE_PATHS or digest(project / name) != expected:
            raise ValueError("Development source hash differs: " + name)
    external = []
    for source, receipt in plan["author_sources"].items():
        base = Path(args.source_root if source == "TimeMixer" else args.control_source_root).resolve()
        for item in receipt["files"]:
            relative = Path(item["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid author source receipt path")
            file = base / relative
            if file.is_symlink() or not file.resolve().is_relative_to(base) or digest(file) != item["sha256"]:
                raise ValueError("Pinned author source differs: " + str(relative))
            external.append({"source": source, "path": str(relative), "sha256": item["sha256"]})
    return {"plan_sha256": digest(plan_path), "development_sources": plan["development_sources"],
            "inherited_scientific_sha256": inherited,
            "code_sha256": {path: digest(ROOT / path) for path in paths}, "author_sources": external}


def environment():
    packages = {}
    for name in ("torch", "numpy", "pandas", "einops", "reformer-pytorch"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "not-installed-as-distribution"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "packages": packages,
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def private_output(output_dir):
    output = Path(output_dir).resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("Checkpoints and individual predictions must remain outside the repository")
    return output


def registration(args, inputs):
    path = private_output(args.output_dir) / "registration.json"
    if not path.is_file(): raise ValueError("Register the committed plan before any dataset fit")
    record = read_json(path)
    if record["immutable_inputs"] != inputs: raise ValueError("Registered plan, data or code changed")
    return record


def register(args, plan, inputs):
    output = private_output(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if (output / "registration.json").exists(): return registration(args, inputs)
    if any(output.glob("**/checkpoint.pt")) or any(output.glob("**/started.json")):
        raise ValueError("Existing fit attempts cannot be registered retrospectively")
    tracked = [str(Path(args.design).resolve().relative_to(ROOT)),
               "scripts/run_e2_remediation_pilot.py", "scripts/e2_temporal_models.py"]
    for path in tracked:
        subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT,
                       check=True, capture_output=True)
    if subprocess.check_output(["git", "diff", "HEAD", "--", *tracked], cwd=ROOT):
        raise ValueError("Commit the fixed pilot design and implementation before registration")
    record = {"registered_at_utc": stamp(), "source_commit": subprocess.check_output(
              ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "immutable_inputs": inputs, "dataset_fits_in_this_output_before_registration": 0,
              "scope": plan["scope"], "Bitcoin_test_period_access": False,
              "environment": environment()}
    write_json(output / "registration.json", record, exclusive=True)
    return record


def configure(seed, threads=1):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)


def fit_stage(model_name, source, x, y, seed, train_settings, x_valid=None, y_valid=None, epochs=None):
    configure(seed, train_settings["torch_threads"])
    model = build_model(model_name, source)
    generator = torch.Generator().manual_seed(seed + 17)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=train_settings["batch_size"], shuffle=True, drop_last=False,
        num_workers=0, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_settings["learning_rate"],
                                 weight_decay=train_settings["weight_decay"])
    curve = []; best_loss = float("inf"); best_epoch = None; stale = 0
    start = time.perf_counter(); maximum = epochs or train_settings["max_epochs"]
    for epoch in range(1, maximum + 1):
        model.train(); total = 0.; count = 0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True); prediction = model(bx)
            loss = (prediction - by).square().mean()
            if not torch.isfinite(loss): raise ValueError("Nonfinite development optimization loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), train_settings["gradient_clip_norm"])
            optimizer.step(); total += float(loss.detach()) * len(bx); count += len(bx)
        val = None
        if x_valid is not None:
            model.eval()
            with torch.no_grad():
                val = float((model(torch.from_numpy(x_valid)) - torch.from_numpy(y_valid)).square().mean())
            if not np.isfinite(val): raise ValueError("Nonfinite inner validation loss")
            if val < best_loss - train_settings["min_delta"]:
                best_loss = val; best_epoch = epoch; stale = 0
            else: stale += 1
        curve.append({"epoch": epoch, "training_MSE_scaled": total / count,
                      "validation_MSE_scaled": val, "elapsed_seconds": time.perf_counter() - start})
        if x_valid is not None and stale >= train_settings["patience"]: break
    selected = maximum if x_valid is None else best_epoch
    if selected is None: raise ValueError("No admissible selected epoch")
    return model, selected, curve, time.perf_counter() - start


def directory_for(output, phase, model, seed, origin):
    return Path(output) / phase / origin / model / f"seed_{seed}"


def completed(directory, inputs, expected=None):
    directory = Path(directory); record = read_json(directory / "completion.json")
    if record["plan_sha256"] != inputs["plan_sha256"]: raise ValueError("Fit belongs to another plan")
    if record.get("evaluation_predictions_computed") is not False or record.get("optimizer_fits") != 2:
        raise ValueError("Fit closure is not a prediction-free two-stage trajectory")
    if expected and any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Closed trajectory belongs to another model, seed, phase or origin")
    required = {"checkpoint.pt", "setup.json", "preprocessing.json", "split.json", "history_inner.csv", "history_refit.csv", "started.json"}
    if set(record["artifact_sha256"]) != required:
        raise ValueError("Closed trajectory has an incomplete or unexpected artifact manifest")
    for file, value in record["artifact_sha256"].items():
        if digest(directory / file) != value: raise ValueError("Closed fit artifact changed: " + file)
    return record


def fit_one(args, plan, inputs):
    registration(args, inputs)
    phase = "oof" if args.phase == "oof" else "pilot"
    origin = args.origin if phase == "oof" else PILOT_ORIGIN
    if phase == "oof" and (not plan["OOF"]["enabled"] or origin not in OOF_ORIGINS):
        raise ValueError("OOF execution is not registered or origin is outside Jan-Oct2024")
    if args.model not in MODELS or args.seed not in SEEDS: raise ValueError("Unregistered model or seed")
    path = directory_for(args.output_dir, phase, args.model, args.seed, origin)
    identity = {"model": args.model, "seed": args.seed, "phase": phase, "fit_origin": origin}
    if path.exists(): return completed(path, inputs, identity)
    path.mkdir(parents=True)
    write_json(path / "started.json", {"at_utc": stamp(), "plan_sha256": inputs["plan_sha256"]}, exclusive=True)
    last = (pd.Timestamp(origin) - pd.Timedelta(days=8)).strftime("%Y-%m-%d")
    data = load_development_prefix(args.project_root, last)
    split = split_for_origin(data, 7, origin, inner_days=90)
    inner, valid, train = (split[k] for k in ("inner_train_indices", "inner_validation_indices", "train_indices"))
    pre_inner = fit_relative_preprocessing(data, 7, inner)
    pre_full = fit_relative_preprocessing(data, 7, train)
    xi, yi = transform_relative_windows(data, 7, inner, pre_inner)
    xv, yv = transform_relative_windows(data, 7, valid, pre_inner)
    source = source_root_for(args.model, args)
    _, epoch, curve_inner, seconds_inner = fit_stage(args.model, source, xi, yi, args.seed,
                                                    plan["training"], xv, yv)
    xt, yt = transform_relative_windows(data, 7, train, pre_full)
    refit_seed = args.seed + plan["training"]["refit_seed_offset"]
    model, _, curve_refit, seconds_refit = fit_stage(args.model, source, xt, yt, refit_seed,
                                                   plan["training"], epochs=epoch)
    torch.save(model.state_dict(), path / "checkpoint.pt")
    setup = {"model": args.model, "seed": args.seed, "refit_seed": refit_seed,
             "phase": phase, "fit_origin": origin, "fit_outcome_max": last,
             "scaler_fit_outcome_max": pre_full["last_fit_outcome_day_utc"],
             "configuration_selection_outcome_max": split["split_metadata"]["inner_validation_last_day_utc"],
             "selected_epoch": epoch, "architecture": architecture_spec(model),
             "training": plan["training"], "training_samples": len(train),
             "inner_training_samples": len(inner), "inner_validation_samples": len(valid),
             "training_seconds": seconds_inner + seconds_refit,
             "optimizer_fits": 2, "pilot_or_OOF_predictions_computed": False,
             "max_decoded_outcome_before_fit_closure": data["decoded_outcome_max"],
             "mase_scale": float(mase_scale(data, train)), "plan_sha256": inputs["plan_sha256"],
             "environment": environment()}
    write_json(path / "setup.json", setup)
    write_json(path / "preprocessing.json", {"inner": pre_inner, "refit": pre_full})
    write_json(path / "split.json", {**split["split_metadata"], "train_indices": train.tolist(),
        "inner_train_indices": inner.tolist(), "inner_validation_indices": valid.tolist()})
    pd.DataFrame(curve_inner).to_csv(path / "history_inner.csv", index=False)
    pd.DataFrame(curve_refit).to_csv(path / "history_refit.csv", index=False)
    artifacts = {p.name: digest(p) for p in path.iterdir() if p.is_file()}
    record = {"completed_at_utc": stamp(), "plan_sha256": inputs["plan_sha256"],
              "model": args.model, "seed": args.seed, "phase": phase,
              "fit_origin": origin, "fit_outcome_max": last, "selected_epoch": epoch,
              "optimizer_fits": 2, "training_seconds": setup["training_seconds"],
              "evaluation_predictions_computed": False, "artifact_sha256": artifacts}
    write_json(path / "completion.json", record, exclusive=True)
    print(json.dumps({k: record[k] for k in ("model", "seed", "phase", "fit_origin", "selected_epoch", "training_seconds")}), flush=True)
    return record


def scheduled_fits(plan, phase):
    origins = OOF_ORIGINS if phase == "oof" else (PILOT_ORIGIN,)
    return [(origin, model, seed) for origin in origins for model in MODELS for seed in SEEDS]


def score(args, plan, inputs):
    registration(args, inputs)
    phase = "oof" if args.phase == "score-oof" else "pilot"
    if phase == "oof" and not plan["OOF"]["enabled"]: raise ValueError("OOF is not registered")
    jobs = scheduled_fits(plan, phase)
    receipts = [completed(directory_for(args.output_dir, phase, model, seed, origin), inputs,
                         {"model": model, "seed": seed, "phase": phase, "fit_origin": origin})
                for origin, model, seed in jobs]
    output = Path(args.output_dir) / (phase + "_scoring")
    if output.exists():
        record = read_json(output / "completion.json")
        if record.get("plan_sha256") != inputs["plan_sha256"] or record.get("phase") != phase:
            raise ValueError("Scored artifacts belong to another registered plan or phase")
        for file, value in record["artifact_sha256"].items():
            if digest(output / file) != value: raise ValueError("Scored artifact changed")
        return record
    output.mkdir()
    scoring_started = stamp()
    if any(pd.Timestamp(record["completed_at_utc"]) >= pd.Timestamp(scoring_started) for record in receipts):
        raise ValueError("Fit closure timestamps must precede scoring")
    write_json(output / "started.json", {"at_utc": scoring_started, "fit_checkpoints_closed": len(receipts),
        "all_scheduled_fit_closures_verified": True, "plan_sha256": inputs["plan_sha256"]}, exclusive=True)
    frames = []
    for origin, model_name, seed in jobs:
        folder = directory_for(args.output_dir, phase, model_name, seed, origin)
        setup = read_json(folder / "setup.json"); pre = read_json(folder / "preprocessing.json")["refit"]
        end = (pd.Timestamp(origin) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d") if phase == "oof" else PILOT_END
        data = load_development_prefix(args.project_root, end)
        dates = data["calendar"]
        idx = np.flatnonzero(data["eligible_by_history"][7] & (dates >= pd.Timestamp(origin, tz="UTC"))
                             & (dates <= pd.Timestamp(end, tz="UTC")))
        if len(idx) != (pd.Timestamp(end) - pd.Timestamp(origin)).days + 1:
            raise ValueError("A registered evaluation day is ineligible; do not compress the calendar")
        x, _ = transform_relative_windows(data, 7, idx, pre)
        model = build_model(model_name, source_root_for(model_name, args))
        model.load_state_dict(torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)); model.eval()
        torch.set_num_threads(plan["training"]["torch_threads"])
        with torch.no_grad(): scaled = model(torch.from_numpy(x)).numpy().astype(np.float64)
        prediction = scaled * pre["y_scale"]
        if not np.isfinite(prediction).all() or (prediction < 0).any(): raise ValueError("Invalid volatility prediction")
        run_id = f"{phase}_{origin}_{model_name}_s{seed}"
        frame = pd.DataFrame({"run_id": run_id, "model": model_name, "seed": seed,
            "outcome_day": dates[idx].strftime("%Y-%m-%d"), "y_true": data["targets"][idx],
            "y_pred": prediction, "persistence_pred": data["persistence"][idx],
            "mase_scale": setup["mase_scale"], "high_volatility_known": data["persistence"][idx] > HIGH_VOL_THRESHOLD,
            "fit_origin": origin, "fit_outcome_max": setup["fit_outcome_max"],
            "scaler_fit_outcome_max": setup["scaler_fit_outcome_max"],
            "configuration_selection_outcome_max": setup["configuration_selection_outcome_max"],
            "checkpoint_closed_at_utc": read_json(folder / "completion.json")["completed_at_utc"],
            "scoring_started_at_utc": scoring_started,
            "checkpoint_sha256": digest(folder / "checkpoint.pt"), "scope": "development_P0_not_final_benchmark",
            "source_stage": "P0_diagnostic", "eligible": True})
        filename = run_id + ".csv"; frame.to_csv(output / filename, index=False); frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    artifacts = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    record = {"completed_at_utc": stamp(), "phase": phase, "scheduled_trajectories": len(jobs),
              "plan_sha256": inputs["plan_sha256"], "all_fit_closures_before_scoring": True,
              "max_decoded_outcome": max(f.outcome_day.max() for f in frames),
              "rows": sum(len(f) for f in frames), "Bitcoin_test_results": 0,
              "artifact_sha256": artifacts}
    write_json(output / "completion.json", record, exclusive=True)
    return record


def analyze(args, plan, inputs):
    """Descriptive P0 variability only; no post-HPO power decision or new fits."""
    registration(args, inputs); outputs = []
    for phase in ("pilot", "oof"):
        base = Path(args.output_dir) / (phase + "_scoring")
        if not (base / "completion.json").exists(): continue
        record = read_json(base / "completion.json")
        if record.get("plan_sha256") != inputs["plan_sha256"] or record.get("phase") != phase:
            raise ValueError("Closed scoring belongs to another registered plan or phase")
        for file, value in record["artifact_sha256"].items():
            if digest(base / file) != value: raise ValueError("Closed scoring differs")
        frame = pd.read_csv(base / "predictions.csv")
        for (model, seed), group in frame.groupby(["model", "seed"]):
            error = group.y_pred.to_numpy() - group.y_true.to_numpy()
            high = group.high_volatility_known.to_numpy(dtype=bool)
            high_rmse = float(np.sqrt(np.mean(error[high] ** 2))) if high.any() else None
            rest_rmse = float(np.sqrt(np.mean(error[~high] ** 2))) if (~high).any() else None
            outputs.append({"phase": phase, "model": model, "seed": int(seed),
                "n_origins": len(group), "RMSE": float(np.sqrt(np.mean(error ** 2))),
                "MAE": float(np.mean(np.abs(error))), "MSE": float(np.mean(error ** 2)),
                "mean_signed_error": float(error.mean()), "high_count": int(high.sum()),
                "high_RMSE": high_rmse, "rest_RMSE": rest_rmse,
                "high_minus_rest_RMSE": None if high_rmse is None or rest_rmse is None else high_rmse - rest_rmse})
    if not outputs: raise ValueError("No closed development scoring exists")
    out = Path(args.output_dir) / "descriptive_analysis"; out.mkdir(exist_ok=True)
    pd.DataFrame(outputs).to_csv(out / "per_seed_metrics.csv", index=False)
    report = {"at_utc": stamp(), "scope": plan["scope"], "registered_models": list(MODELS),
              "model_seed_phase_rows": len(outputs), "post_HPO_final_power_established": False,
              "architecture_or_HPO_selection_performed": False, "Bitcoin_test_results": 0,
              "gap_hypothesis_guaranteed": False,
              "inference_status": "Descriptive pilot metrics; formal temporal bootstrap/variance analysis is a separately registered analysis"}
    write_json(out / "report.json", report)
    return report


def validate(args, plan, inputs, profile=False):
    details = []
    for name in MODELS:
        configure(20261106, 1); model = build_model(name, source_root_for(name, args)); model.eval()
        x = torch.randn(4, 7, 5)
        with torch.no_grad():
            y = model(x); solo = model(x[:1]); joined = model(x)[:1]
        if y.shape != (4,) or not torch.isfinite(y).all() or (y < 0).any(): raise ValueError("Bad synthetic adapter output")
        if not torch.allclose(solo, joined, atol=1e-6, rtol=1e-5): raise ValueError("Inference depends on other batch origins")
        detail = architecture_spec(model)
        if profile:
            model.train(); xx = torch.randn(32, 7, 5); target = torch.ones(32)
            seconds = []
            for _ in range(4):
                model.zero_grad(set_to_none=True); start = time.perf_counter()
                (model(xx) - target).square().mean().backward(); seconds.append(time.perf_counter() - start)
            detail["synthetic_forward_backward_seconds"] = seconds
            detail["synthetic_optimizer_steps"] = 0
            detail["parameter_activity_after_backward"] = count_parameters(model)
        details.append(detail)
    return {"valid": True, "scope": plan["scope"], "models": details,
            "source_files_verified": len(inputs["author_sources"]), "market_rows_decoded": 0,
            "dataset_optimizer_fits": 0, "planned_optimizer_fits_P0": 40,
            "planned_optional_optimizer_fits_OOF": 400 if plan["OOF"]["enabled"] else 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=["validate", "profile", "register", "pilot", "oof", "score", "score-oof", "analyze"])
    p.add_argument("--design", type=Path, default=PLAN)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--control-source-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", choices=MODELS); p.add_argument("--seed", type=int); p.add_argument("--origin")
    args = p.parse_args(argv)
    try:
        plan = read_plan(args.design); private_output(args.output_dir)
        inputs = immutable_inputs(plan, args.design, args)
        if args.phase in ("validate", "profile"): result = validate(args, plan, inputs, args.phase == "profile")
        elif args.phase == "register": result = register(args, plan, inputs)
        elif args.phase in ("pilot", "oof"): result = fit_one(args, plan, inputs)
        elif args.phase in ("score", "score-oof"): result = score(args, plan, inputs)
        else: result = analyze(args, plan, inputs)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False)); return 0
    except (ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
