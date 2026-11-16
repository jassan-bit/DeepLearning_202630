"""Portable E2 fixed-recipe runner; check is the default and never fits models.

The package contains relative, hash-bound inputs.  Reanalysis uses saved
forecasts; replay-checkpoints runs fitted models; train-fixed fits 78+3 models.
None of these modes is called a replay of the complete historical HPO campaign.
This runner never imports a historical budget ledger or downloads inputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

VERSION = "e2-portable-fixed-recipe-campaign-v1"
CPU_VERSIONS = {"torch": "2.8.0+cpu", "numpy": "2.3.5", "pandas": "2.2.3",
                "scipy": "1.17.0", "statsmodels": "0.14.5"}
A2_FIELDS = ("model", "RMSE", "RMSE_seed_SD", "MAE", "sMAPE", "R2_mean",
             "coverage90", "mean_interval_width90", "R", "n_common_dates")
INCOMPLETE = ["historical HPO selection", "all robustness and explanations",
              "all ablations", "all book tables and figures", "historical missing timings"]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temp = path.with_suffix(path.suffix + ".pending")
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temp.replace(path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")


def relative_path(root, name, *, exists=True):
    if not isinstance(name, str):
        raise ValueError("A relative path must be a string")
    part = PurePosixPath(name)
    if not name or "\\" in name or part.is_absolute() or ".." in part.parts or str(part) != name:
        raise ValueError("Non-normalized or absolute package path: " + name)
    path = root.joinpath(*part.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Package path escapes its root")
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)):
        raise ValueError("Symlinked package inputs are not admitted: " + name)
    if exists and not path.is_file():
        raise ValueError("Missing regular input: " + name)
    return path


def file_record(root, path):
    path = Path(path)
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def tree(root):
    return [file_record(root, p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and not p.is_symlink()]


def verify_record(root, row):
    path = relative_path(root, row["path"])
    if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
        raise ValueError("Input/output bytes changed: " + row["path"])
    return path


def load_config(path):
    path = Path(path).absolute()
    root = path.parent
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != VERSION:
        raise ValueError("Unsupported portable campaign configuration")
    if data.get("runner_sha256") != sha(__file__):
        raise ValueError("Runner differs from the package configuration")
    for row in data["files"]:
        verify_record(root, row)
    for key in ("source_root", "author_root", "recorded_run"):
        relative_path(root, data[key], exists=False)
    for key in ("fixed_recipe_plan", "daily_csv", "aggregation_receipt", "arrays_npz",
                "reference_npz", "reference_A2"):
        relative_path(root, data[key])
    if len(data["files"]) != len({r["path"] for r in data["files"]}):
        raise ValueError("Duplicate bound input path")
    return root, data


def runtime():
    versions = {}
    for name in CPU_VERSIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "os_release": platform.freedesktop_os_release() if hasattr(platform, "freedesktop_os_release") and sys.platform == "linux" else {},
            "packages": versions}


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scientific(root, config):
    source = relative_path(root, config["source_root"], exists=False)
    sys.path.insert(0, str(source / "scripts"))
    base = imported(source / "scripts/e2_reproduce_from_raw.py", "_portable_base")
    frozen = json.loads(relative_path(root, config["fixed_recipe_plan"]).read_text())
    for row in frozen["bound_sources"]:
        verify_record(source, row)
    expected = {(m, s) for m in base.NEURAL for s in base.SEEDS} | {(m, "deterministic") for m in base.CLASSICAL}
    if len(frozen["jobs"]) != 81 or {(j["model"], j["seed"]) for j in frozen["jobs"]} != expected:
        raise ValueError("The fixed schedule must contain exactly 78 neural and three classical predictors")
    compatibility = imported(source / "scripts/e2_arima_prediction_compat.py", "_portable_arima_compat")
    compatibility.install_prediction_raw_alias(source)
    return source, base, frozen


def configure(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    observed = runtime()["packages"]
    expected = dict(CPU_VERSIONS)
    if device == "cuda:0":
        expected.pop("torch")
    mismatches = {k: {"required": v, "observed": observed[k]} for k, v in expected.items() if observed[k] != v}
    if mismatches:
        raise ValueError("Install the declared runtime: " + json.dumps(mismatches))
    if device == "cuda:0" and (not torch.cuda.is_available() or torch.__version__.split("+")[0] != "2.8.0"):
        raise ValueError("CUDA mode requires Torch 2.8.0 and a usable local CUDA device")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2 if device == "cpu" else 1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def require_cpu_versions():
    observed = runtime()["packages"]
    mismatch = {k: {"required": v, "observed": observed[k]}
                for k, v in CPU_VERSIONS.items() if observed[k] != v}
    if mismatch:
        raise ValueError("Install the declared CPU analysis runtime: " + json.dumps(mismatch))


def admitted_panel(root, config, source):
    """Reconstruct from the admitted daily derivative, never pretend to open ZIPs."""
    import numpy as np
    import pandas as pd
    import e2_reduced_final_data as data
    aggregation = json.loads(relative_path(root, config["aggregation_receipt"]).read_text())
    daily = relative_path(root, config["daily_csv"])
    if sha(daily) != aggregation["daily_csv_sha256"] or aggregation["archive_count"] != 20 or aggregation["csv_rows"] != 875520:
        raise ValueError("Daily derivative is not bound to its original 20-archive aggregation receipt")
    reserve = data.reserve_known_e1_evaluation(source)
    if [(r["archive"], r["sha256"]) for r in reserve["archives"]] != [
            ("BTCUSDT-1m-" + r["month"] + ".zip", r["sha256"]) for r in aggregation["quality_by_archive"]]:
        raise ValueError("Aggregation provenance and registered raw archive identities differ")
    development, _ = data.load_development_bridge(source)
    evaluation = pd.read_csv(daily, index_col="market_day_utc")
    evaluation.index = pd.DatetimeIndex(pd.to_datetime(evaluation.index, utc=True))
    dev = pd.DataFrame(development.features, index=development.calendar, columns=data.FEATURES)
    panel = data._panel_from_frames(dev, evaluation)
    dates = pd.date_range("2025-01-01", "2026-08-31", tz="UTC")
    mask = panel.support.loc[dates, "common"].to_numpy(dtype=bool)
    labels = panel.targets_at(dates)
    with np.load(relative_path(root, config["arrays_npz"]), allow_pickle=False) as arrays:
        if not np.array_equal(arrays["evaluation_dates"].astype(str), dates.strftime("%Y-%m-%d").to_numpy()):
            raise ValueError("Daily derivative and audited array calendars differ")
        if not np.array_equal(labels, arrays["evaluation_y_original"]):
            raise ValueError("Daily-derived target values differ from the separately audited arrays")
    if mask.sum() != 608:
        raise ValueError("The admitted common support must contain all 608 evaluation dates")
    return panel, dates, mask, labels


def a2(targets, predictions, calibrators, analysis):
    import numpy as np
    targets = np.asarray(targets, dtype=np.float64)
    mask = np.ones(len(targets), dtype=bool)
    targets = targets[mask]
    rows = []
    for model in analysis.SYSTEMS:
        # Preserve the original analyze_benchmark p[:,mask] indexing/layout.
        # A different array layout can change floating point reductions by an
        # ulp, even with identical forecast and target values.
        values = np.asarray(predictions[model], dtype=np.float64)[:, mask]
        row = {"model": model, **analysis.metrics(targets, values)}
        covers, widths = [], []
        for prediction, calibrator in zip(values, calibrators[model], strict=True):
            lo, hi = analysis.prediction_intervals(prediction, calibrator)
            covers.append(float(np.mean((targets >= lo) & (targets <= hi))))
            widths.append(float(np.mean(hi - lo)))
        row.update(coverage90=float(np.mean(covers)), mean_interval_width90=float(np.mean(widths)))
        rows.append(row)
    return rows


def write_a2(directory, rows):
    with (directory / "benchmark_A2.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=A2_FIELDS)
        writer.writeheader()
        writer.writerows({k: row[k] for k in A2_FIELDS} for row in rows)
    write(directory / "benchmark_A2.json", rows)


def compare_a2(root, config, rows):
    old = {r["model"]: r for r in csv.DictReader(relative_path(root, config["reference_A2"]).open(encoding="utf-8"))}
    maximum = 0.0
    for row in rows:
        for key in A2_FIELDS[1:]:
            previous, current = old[row["model"]][key], row[key]
            if previous == "" and current is None:
                continue
            delta = abs(float(previous) - float(current))
            if not math.isfinite(delta):
                raise ValueError("Nonfinite A2 difference")
            maximum = max(maximum, delta)
    if maximum != 0.0:
        raise ValueError("A2 differs from the own CPU reference: " + repr(maximum))
    return maximum


def import_predictors(root, config, campaign, frozen):
    """Copy only verified fitted artifacts, never saved test forecasts."""
    donor = relative_path(root, config["recorded_run"], exists=False)
    for job in frozen["jobs"]:
        directory = relative_path(donor, job["directory"], exists=False)
        result = json.loads(relative_path(directory, "result.json").read_text())
        if result.get("status") != "completed" or (result["model"], result["seed"]) != (job["model"], job["seed"]):
            raise ValueError("Recorded predictor identity changed")
        for name, checksum in result["artifacts_sha256"].items():
            if sha(relative_path(directory, name)) != checksum:
                raise ValueError("A recorded fit artifact differs from its producer receipt")
        for name in ("checkpoint", "scaler", "calibrator"):
            if sha(relative_path(directory, result[name])) != result[name + "_sha256"]:
                raise ValueError("A fitted predictor/scaler/calibrator changed")
        shutil.copytree(directory, campaign / job["directory"])


def score_worker(root, config, output, *, device, compare):
    import numpy as np
    import pandas as pd
    import pickle
    torch = configure(device)
    source, base, frozen = scientific(root, config)
    import e2_reduced_final as final
    import e2_reduced_final_analysis as analysis
    from e2_temporal_training import FoldScaler
    from e2_published_pilot_training import predict_pilot
    cpu = imported(source / "scripts/e2_cpu_fixed_recipe_reproduction.py", "_portable_cpu")
    campaign = output.parent.parent / "campaign"
    if compare:
        import_predictors(root, config, campaign, frozen)
    shutil.copyfile(source / frozen["pilot_path"], campaign / "pilot_power_MDE.json")
    freeze = final.freeze_final(campaign)
    panel, dates, mask, targets = admitted_panel(root, config, source)
    predictions = {m: [] for m in final.SYSTEMS}
    calibrators = {m: [] for m in final.SYSTEMS}
    history = final._history(panel)
    maximum = 0.0
    with np.load(relative_path(root, config["reference_npz"]), allow_pickle=False) as reference:
        expected = {(str(m), str(s)): values for m, s, values in zip(
            reference["model"], reference["seed"], reference["prediction_original"], strict=True)}
    for item in freeze["models"]:
        name, seed = item["model"], item["seed"]
        checkpoint = campaign / item["checkpoint"]
        if name in final.NEURAL:
            spec = json.loads((checkpoint.parent.parent / "worker_spec.json").read_text())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            record = saved["scaler"]
            scaler = FoldScaler(tuple(record["mean"]), tuple(record["std"]), record["target_scale"],
                                tuple(pd.Timestamp(d, tz="UTC") for d in record["train_dates"]), record["training_sha256"])
            model = final._trial_builder(name, spec["selection"]["winner_trial"], seed,
                {m: str(relative_path(root, config["author_root"], exists=False) / m) for m in {r["model"] for r in frozen["author_files"]}})
            model.load_state_dict(saved["state_dict"])
            model.eval()
            x = torch.as_tensor(scaler.transform(panel.raw_windows(dates[mask])), dtype=torch.float32)
            predictor = cpu.cpu_predict if device == "cpu" else predict_pilot
            values = predictor(model, x, batch_size=1, device=device).numpy() * scaler.target_scale
            del model
        elif name == "ARIMA":
            with checkpoint.open("rb") as stream:
                model = pickle.load(stream)
            before = list(model.receipt["parameters"])
            forecasts = final._arima_walk(model, history, dates[-1], output / "ARIMA_causal_predictions")
            values = np.asarray([forecasts[d] for d in dates.strftime("%Y-%m-%d")])
            if before != model.receipt["parameters"]:
                raise ValueError("ARIMA coefficients changed during replay")
        else:
            saved = json.loads(checkpoint.read_text())
            values = final._naive_predictions(history, dates.strftime("%Y-%m-%d").tolist(), saved["lag"])
        values = np.asarray(values, dtype=np.float64)
        if compare:
            difference = float(np.max(np.abs(values - expected[(name, str(seed))])))
            maximum = max(maximum, difference)
        predictions[name].append(values.tolist())
        calibrators[name].append(json.loads((campaign / item["calibrator"]).read_text()))
        write(output / ("predictions_" + base.slug(name) + "_" + str(seed) + ".json"),
              {"model": name, "seed": seed, "dates": dates.strftime("%Y-%m-%d").tolist(), "prediction_original": values.tolist()})
    rows = a2(targets, predictions, calibrators, analysis)
    write_a2(output, rows)
    if compare and maximum != 0.0:
        raise ValueError("Checkpoint forecasts differ from their own CPU reference: " + repr(maximum))
    write(output / "prediction_panel.json", {"predictions": predictions, "calibrators": calibrators,
          "targets": targets.tolist(), "dates": dates.strftime("%Y-%m-%d").tolist(), "mask": mask.tolist()})
    write(output / "readout.json", {"scope": "fitted checkpoint inference" if compare else "new fixed-recipe predictions",
          "new_neural_forwards": 78 * 608, "new_predictor_trajectories": 81, "new_model_fits_in_this_stage": 0,
          "input_level": "previously admitted daily OHLCV and independently audited arrays, not monthly ZIP decode",
          "max_prediction_difference_own_CPU_reference": maximum if compare else None,
          "max_A2_difference_own_CPU_reference": compare_a2(root, config, rows) if compare else None,
          "raw_archives_opened": 0, "original_A100_parity_claimed": False,
          "original_full_campaign_reproduced": False})


def reanalysis_worker(root, config, output):
    require_cpu_versions()
    import numpy as np
    _, _, frozen = scientific(root, config)
    import e2_reduced_final_analysis as analysis
    reference = np.load(relative_path(root, config["reference_npz"]), allow_pickle=False)
    predictions = {m: [] for m in analysis.SYSTEMS}
    calibrators = {m: [] for m in analysis.SYSTEMS}
    donor = relative_path(root, config["recorded_run"], exists=False)
    for job in frozen["jobs"]:
        directory = donor / job["directory"]
        result = json.loads((directory / "result.json").read_text())
        matches = [(m == job["model"] and s == str(job["seed"])) for m, s in zip(reference["model"], reference["seed"], strict=True)]
        if sum(matches) != 1:
            raise ValueError("Exactly one saved trajectory is required per predictor")
        predictions[job["model"]].append(reference["prediction_original"][matches][0].tolist())
        calibrators[job["model"]].append(json.loads((directory / result["calibrator"]).read_text()))
    rows = a2(reference["observed_y_original"], predictions, calibrators, analysis)
    write_a2(output, rows)
    write(output / "prediction_panel.json", {"predictions": predictions, "calibrators": calibrators,
          "targets": reference["observed_y_original"].tolist(), "dates": reference["dates"].tolist(), "mask": [True] * 608})
    write(output / "readout.json", {"scope": "arithmetic reanalysis from saved forecasts", "new_model_fits": 0,
          "new_model_forwards": 0, "max_A2_difference_own_CPU_reference": compare_a2(root, config, rows),
          "original_full_campaign_reproduced": False})


def worker(args, root, config):
    output = args.output / "stages" / args.worker
    output.mkdir(parents=True, exist_ok=False)
    if args.worker == "reanalysis":
        reanalysis_worker(root, config, output)
    elif args.worker in ("replay-checkpoints", "score-new-fits"):
        score_worker(root, config, output, device=args.device, compare=args.worker == "replay-checkpoints")
    elif args.worker == "primary20000":
        require_cpu_versions()
        scientific(root, config)
        import e2_reduced_final_analysis as analysis
        previous = "reanalysis" if args.mode == "reanalysis" else "replay-checkpoints" if args.mode == "replay-checkpoints" else "score-new-fits"
        panel = json.loads((args.output / "stages" / previous / "prediction_panel.json").read_text())
        result = analysis.analyze_benchmark(panel["targets"], panel["predictions"], dates=panel["dates"],
                    mask=__import__("numpy").asarray(panel["mask"], dtype=bool), calibrators=panel["calibrators"], draws=20000)
        write(output / "benchmark_analysis_20000.json", result)
    else:
        source, base, frozen = scientific(root, config)
        job = next(j for j in frozen["jobs"] if j["id"] == args.worker)
        campaign = args.output / "campaign"
        campaign.mkdir(parents=True, exist_ok=True)
        local = SimpleNamespace(source_root=source, author_root=relative_path(root, config["author_root"], exists=False),
                                output=campaign, plan=args.config)
        if args.device == "cpu":
            cpu = imported(source / "scripts/e2_cpu_fixed_recipe_reproduction.py", "_portable_cpu")
            policy = {"runtime_policy": {**CPU_VERSIONS, "threads": 2, "interop_threads": 2,
                      "float32_matmul_precision": "highest", "mkldnn_enabled": True},
                      "parameter_counts": config["parameter_counts"]}
            cpu.fit_cpu(local, policy, base, frozen, job)
        else:
            base.execute_job(local, frozen, job)
        write(output / "fit_result.json", {"model": job["model"], "seed": job["seed"],
              "kind": job["kind"], "new_fixed_recipe_fit": True, "HPO_performed": False,
              "artifacts": tree(campaign / job["directory"])})


def run(args, root, config):
    if args.mode == "check":
        report = {"version": VERSION, "status": "inputs_verified", "files_verified": len(config["files"]),
                  "runtime": runtime(), "new_model_fits": 0, "new_model_forwards": 0,
                  "full_historical_campaign_reproduced": False, "remaining_separate_requirements": INCOMPLETE}
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            write(args.output / "preflight.json", report)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    if not args.execute or args.output is None or args.cap_seconds is None or not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0:
        raise ValueError("Computational modes require --execute --output NEW_DIRECTORY --cap-seconds POSITIVE_LIMIT")
    if args.mode == "replay-checkpoints" and args.device != "cpu":
        raise ValueError("The exact own-CPU checkpoint replay must use --device cpu")
    if args.output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Outputs must be outside the immutable package")
    identity = {"version": VERSION, "config_sha256": sha(args.config), "runner_sha256": sha(__file__),
                "mode": args.mode, "device": args.device, "primary20000": args.primary20000,
                "cap_seconds": args.cap_seconds, "python_executable": str(args.python),
                "new_training_requested": args.mode == "train-fixed", "models": args.models, "seeds": args.seeds}
    if args.resume:
        if json.loads((args.output / "run_identity.json").read_text()) != identity:
            raise ValueError("Resume configuration, runtime entry point, scope or cap changed")
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Fresh output required; resume accepts only hash-verified completed stages")
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / "run_identity.json", identity)
    frozen = json.loads(relative_path(root, config["fixed_recipe_plan"]).read_text())
    jobs = frozen["jobs"]
    if args.models or args.seeds:
        if args.mode != "train-fixed":
            raise ValueError("Model/seed subsets are only valid in explicit train-fixed software QA")
        if args.models and not set(args.models).issubset({j["model"] for j in jobs}):
            raise ValueError("Unknown model in training subset")
        if args.seeds and not set(args.seeds).issubset({j["seed"] for j in jobs if j["kind"] == "neural"}):
            raise ValueError("Unregistered neural seed in training subset")
        jobs = [j for j in jobs if (not args.models or j["model"] in args.models)
                and (j["kind"] == "classical" or not args.seeds or j["seed"] in args.seeds)]
    full_schedule = len(jobs) == 81
    if not jobs or (args.primary20000 and args.mode == "train-fixed" and not full_schedule):
        raise ValueError("A nonempty fit subset cannot run final81 scoring or primary inference")
    stages = ([j["id"] for j in jobs] + (["score-new-fits"] if full_schedule else [])) if args.mode == "train-fixed" else [args.mode]
    if args.primary20000:
        stages.append("primary20000")
    state = {"status": "running", "mode": args.mode, "stages": [], "charged_worker_wall_seconds": 0.0,
             "new_model_fits_completed_this_run": 0, "full_historical_campaign_reproduced": False,
             "HPO_repeated": False, "training_schedule_jobs": len(jobs) if args.mode == "train-fixed" else 0,
             "training_subset_QA": args.mode == "train-fixed" and not full_schedule,
             "final81_scoring_scheduled": args.mode == "train-fixed" and full_schedule,
             "remaining_separate_requirements": INCOMPLETE}
    args.output.joinpath("logs").mkdir(exist_ok=True)
    args.output.joinpath("receipts").mkdir(exist_ok=True)
    for stage in stages:
        receipt = args.output / "receipts" / (stage + ".json")
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            state["charged_worker_wall_seconds"] += previous["charged_worker_wall_seconds"]
            if previous["returncode"] != 0 or previous["run_identity_sha256"] != sha(args.output / "run_identity.json"):
                raise ValueError("A failed or mismatched stage is preserved; use a new output and a recovery plan")
            for row in previous["artifacts"]:
                verify_record(args.output, row)
            state["stages"].append({"id": stage, "status": "verified_completed_resume", "new_execution": False})
            continue
        remaining = args.cap_seconds - state["charged_worker_wall_seconds"]
        if remaining <= 1:
            state["status"] = "budget_exhausted_before_stage"
            break
        command = [str(args.python), "-I", str(Path(__file__).absolute()), "--config", str(args.config),
                   "--worker", stage, "--mode", args.mode, "--device", args.device, "--output", str(args.output)]
        log = args.output / "logs" / (stage + ".log")
        start, code, error = time.monotonic(), 1, None
        environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "2"
        try:
            with log.open("x") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                         cwd=args.output, env=environment, timeout=max(0.1, remaining - 1))
                code = process.returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Worker timed out; partial artifacts retained"
        except OSError as exc:
            error = str(exc)
        elapsed = time.monotonic() - start
        paths = [p for p in [args.output / "stages" / stage] if p.is_dir()]
        if args.mode == "train-fixed" and stage in {j["id"] for j in frozen["jobs"]}:
            paths.append(args.output / "campaign" / next(j["directory"] for j in frozen["jobs"] if j["id"] == stage))
        artifacts = [dict(r, path=folder.relative_to(args.output).as_posix() + "/" + r["path"]) for folder in paths for r in tree(folder)]
        record = {"stage": stage, "run_identity_sha256": sha(args.output / "run_identity.json"),
                  "returncode": code, "error": error, "charged_worker_wall_seconds": elapsed,
                  "parent_wall_seconds_added": False, "command": command, "log_sha256": sha(log),
                  "artifacts": artifacts, "runtime": runtime()}
        write(receipt, record)
        state["charged_worker_wall_seconds"] += elapsed
        state["stages"].append({"id": stage, "status": "completed" if code == 0 else "failed", "new_execution": True})
        if code:
            state["status"] = "failed_with_artifacts_retained"
            break
        if args.mode == "train-fixed" and stage in {j["id"] for j in frozen["jobs"]}:
            state["new_model_fits_completed_this_run"] += 1
        write(args.output / "status.json", state, replace=True)
    else:
        state["status"] = "completed_declared_scope_only"
    write(args.output / "status.json", state, replace=True)
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["status"] == "completed_declared_scope_only" else 2


def build_bundle(args):
    """Acquire local bytes by verification/copy, with no numerical model operation."""
    destination = args.output
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Bundle preparation requires a fresh output")
    destination.mkdir(parents=True, exist_ok=True)
    source, authors, recorded = args.source_root, args.author_root, args.recorded_run
    plan_path = args.fixed_plan
    frozen = json.loads(plan_path.read_text())
    for row in frozen["bound_sources"]:
        path = verify_record(source, row)
        target = destination / "code" / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    project = Path(__file__).resolve().parents[1]
    for name in ("e2_cpu_fixed_recipe_reproduction.py", "e2_reproduce_from_raw.py", "e2_arima_prediction_compat.py"):
        target = destination / "code/scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / "scripts" / name, target)
    for row in frozen["author_files"]:
        path = relative_path(authors, row["model"] + "/" + row["path"])
        if sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Pinned author source differs")
        target = destination / "authors" / row["model"] / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for job in frozen["jobs"]:
        directory = recorded / job["directory"]
        result = json.loads((directory / "result.json").read_text())
        for relative, checksum in result["artifacts_sha256"].items():
            if sha(relative_path(directory, relative)) != checksum:
                raise ValueError("Donor fitted artifact differs")
        for name in ("checkpoint", "scaler", "calibrator"):
            if sha(relative_path(directory, result[name])) != result[name + "_sha256"]:
                raise ValueError("Donor predictor differs")
        shutil.copytree(directory, destination / "recorded" / job["directory"], ignore=shutil.ignore_patterns("__pycache__"))
    exports = project / "book_e2/_static/e2_scientific_closure_20261116"
    copies = {"fixed_recipe_plan.json": plan_path,
              "known_E1_daily.csv": recorded / "private-known-E1-data/known_E1_daily.csv",
              "aggregation_receipt.json": exports / "data_audit/evaluation_aggregation_receipt.json",
              "current_cohort_arrays.npz": exports / "data_audit/current_cohort_arrays.npz",
              "predictions_81.npz": exports / "cpu_fixed_recipe_reproduction/result_v1/predictions_81.npz",
              "reference_A2.csv": exports / "cpu_fixed_recipe_reproduction/result_v1/analysis/benchmark_A2.csv"}
    for name, path in copies.items():
        target = destination / "inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(__file__, destination / "run.py")
    frozen_inputs = json.loads((source / "book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json").read_text())
    (destination / "requirements_cpu.txt").write_text("--extra-index-url https://download.pytorch.org/whl/cpu\n" +
        "\n".join(name + "==" + version for name, version in CPU_VERSIONS.items()) + "\neinops==0.8.1\nreformer-pytorch==1.4.4\n", encoding="utf-8")
    documentation = project / "book_e2/_static/e2_closure_followup_20261116/reproduction"
    for name in ("README.md", "portability_plan.json"):
        shutil.copyfile(documentation / name, destination / name)
    config = {"version": VERSION, "runner_sha256": sha(destination / "run.py"),
              "source_root": "code", "author_root": "authors", "recorded_run": "recorded",
              "fixed_recipe_plan": "inputs/fixed_recipe_plan.json", "daily_csv": "inputs/known_E1_daily.csv",
              "aggregation_receipt": "inputs/aggregation_receipt.json", "arrays_npz": "inputs/current_cohort_arrays.npz",
              "reference_npz": "inputs/predictions_81.npz", "reference_A2": "inputs/reference_A2.csv",
              "parameter_counts": {r["model"]: r["architecture"]["actual_parameter_count"] for r in frozen_inputs},
              "runtime_policy": {"cpu_packages": CPU_VERSIONS, "threads": 2, "interop_threads": 2,
                   "dtype": "float32", "score_batch_size": 1, "mkldnn_enabled": True},
              "entry_level": "admitted daily OHLCV and audited window arrays; monthly raw ZIPs are provenance, not runtime inputs",
              "historical_full_campaign_reproduced": False, "files": tree(destination)}
    write(destination / "portable_campaign.json", config)
    print(json.dumps({"status": "bundle_prepared_not_executed", "files_bound": len(config["files"]),
                      "output": str(destination), "new_model_fits": 0, "new_model_forwards": 0}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "portable_campaign.json")
    parser.add_argument("--mode", choices=("check", "reanalysis", "replay-checkpoints", "train-fixed"), default="check")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cap-seconds", type=float)
    parser.add_argument("--primary20000", action="store_true")
    parser.add_argument("--models", nargs="+", help="Explicit training-only subset; never described as a full81 fit")
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit registered neural seeds for training-only software QA")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--build-bundle", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--author-root", type=Path)
    parser.add_argument("--recorded-run", type=Path)
    parser.add_argument("--fixed-plan", type=Path)
    args = parser.parse_args(argv)
    for name in ("config", "output", "python", "source_root", "author_root", "recorded_run", "fixed_plan"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).absolute())
    try:
        if args.build_bundle:
            if any(getattr(args, k) is None for k in ("source_root", "author_root", "recorded_run", "fixed_plan", "output")):
                raise ValueError("Bundle preparation requires source-root, author-root, recorded-run, fixed-plan and output")
            return build_bundle(args)
        root, config = load_config(args.config)
        if args.worker:
            worker(args, root, config)
            return 0
        return run(args, root, config)
    except (OSError, ValueError, RuntimeError, ImportError, StopIteration) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
