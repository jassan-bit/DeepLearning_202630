"""Audit export inputs and export three inherited CPU references without fitting.

The export command uses the smallest registered final seed, not the best test
seed. It compares real saved weights with their TorchScript round trip on the
53 historical calibration dates. Those dates are not the 608-date A2 test.
Heavy dependencies are imported only by the explicit export subcommand.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

VERSION = "e3-export-readiness-v1"
REFERENCE_WIDTHS = {"MLP16": 16, "LSTM32": 32, "GRU32": 32}
HEX = set("0123456789abcdef")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def safe_path(root, relative):
    root = Path(root).resolve()
    candidate = root / relative
    if Path(relative).is_absolute() or not candidate.resolve().is_relative_to(root):
        raise ValueError("Artifact path escapes its root")
    if candidate.is_symlink():
        raise ValueError("Artifact symlinks are not admitted")
    return candidate


def validate_file(root, relative, expected, *, required=False):
    if not valid_sha(expected):
        raise ValueError("Expected artifact SHA-256 is missing or invalid")
    if root is None:
        return {"status": "not_checked", "sha256": expected}
    path = safe_path(root, relative)
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {"status": "missing_locally", "sha256": expected}
    if file_sha(path) != expected:
        raise ValueError("Artifact hash mismatch: " + relative)
    if path.read_bytes()[:100].startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError("LFS pointer is not an inference artifact")
    return {"status": "bytes_verified", "sha256": expected,
            "bytes": path.stat().st_size}


def select_seed(freeze, model):
    rows = [r for r in freeze["models"] if r["model"] == model]
    if not rows:
        raise ValueError("Model absent from frozen catalogue")
    seeds = [r["seed"] for r in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate frozen model/seed")
    if seeds == ["deterministic"]:
        return rows[0]
    if any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Invalid neural seed")
    return min(rows, key=lambda r: r["seed"])


def verified_export(exports_root, model, record):
    if exports_root is None:
        return None
    directory = safe_path(exports_root, model)
    report_path = directory / "roundtrip_report.json"
    if not report_path.is_file():
        return None
    report = read_json(report_path)
    metadata = read_json(directory / "metadata.json")
    artifact = directory / "model.torchscript.pt"
    if (report.get("model") != model or report.get("seed") != record["seed"]
            or report.get("checkpoint_sha256") != record["checkpoint_sha256"]
            or report.get("export_sha256") != file_sha(artifact)
            or metadata.get("inference_artifact_sha256") != report["export_sha256"]
            or metadata.get("source_checkpoint_sha256") != record["checkpoint_sha256"]
            or report.get("cohort",{}).get("role") != "historical_calibration"
            or report.get("cohort",{}).get("A2_test_608_reproduced") is not False
            or report.get("roundtrip",{}).get("within_tolerance") is not True):
        raise ValueError("Export receipt or artifact integrity mismatch for " + model)
    return {"status": "roundtrip_verified_on_historical_calibration",
            "format": "TorchScript", "artifact_sha256": report["export_sha256"],
            "artifact_bytes": artifact.stat().st_size, "roundtrip": report["roundtrip"],
            "CPU_profile": report["CPU_profile"], "A2_exported_test_metric_verified": False}


def readiness(freeze, release, *, weights_root=None, metadata_root=None, exports_root=None):
    artifacts = {r["path"]: r for r in release["artifacts"]}
    if len(artifacts) != len(release["artifacts"]):
        raise ValueError("Duplicate paths in release manifest")
    if len(set(freeze["systems"])) != len(freeze["systems"]):
        raise ValueError("Duplicate frozen systems")
    rows = []
    for model in freeze["systems"]:
        record = select_seed(freeze, model)
        checks = {}
        for field in ("checkpoint", "scaler", "calibrator"):
            artifact = artifacts.get(record[field])
            if artifact is None or artifact.get("sha256") != record[field + "_sha256"]:
                raise ValueError("Freeze/release mismatch for " + record[field])
            where = weights_root if field == "checkpoint" and artifact["storage"] == "release_asset" else metadata_root
            checks[field] = {**validate_file(where, record[field], record[field + "_sha256"]),
                             "path": record[field], "storage": artifact["storage"],
                             "release_url": artifact.get("url")}
        verified = verified_export(exports_root, model, record)
        rows.append({"model": model, "deployment_seed": record["seed"],
                     "seed_rule": "smallest registered final seed; no performance selection",
                     "artifacts": checks,
                     "export_status": verified["status"] if verified else "not_executed_by_inventory",
                     "export_evidence": verified,
                     "full_optimizer_checkpoint": "not_verified_by_selected_weights_inventory",
                     "next_step": "export-reference" if model in REFERENCE_WIDTHS else
                     "implement or verify model-specific CPU exporter and round trip"})
    return {"version": VERSION, "systems": len(rows), "frozen_predictors": len(freeze["models"]),
            "rows": rows, "inference_artifacts_created": False,
            "verified_exports_present": sum(r["export_evidence"] is not None for r in rows),
            "note": "A release locator or declared hash does not establish an export or a successful download."}


def finite_vector(values, label):
    if not values:
        raise ValueError(label + " must not be empty")
    result = [float(v) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(label + " has nonfinite values")
    return result


def roundtrip_metrics(memory, exported, truth, *, tolerance):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("A positive finite absolute tolerance is required")
    memory, exported, truth = (finite_vector(x, label) for x, label in
                               [(memory, "memory"), (exported, "export"), (truth, "truth")])
    if len(memory) != len(exported) or len(memory) != len(truth):
        raise ValueError("Round-trip cohort lengths differ")
    maximum = max(abs(a-b) for a,b in zip(memory, exported))
    rmse_memory = math.sqrt(statistics.fmean((a-b)**2 for a,b in zip(memory, truth)))
    rmse_export = math.sqrt(statistics.fmean((a-b)**2 for a,b in zip(exported, truth)))
    if maximum > tolerance:
        raise ValueError("Round-trip absolute tolerance exceeded")
    return {"n": len(truth), "absolute_tolerance_original_sigma7": tolerance,
            "maximum_absolute_prediction_error_original": maximum,
            "RMSE_memory_original": rmse_memory, "RMSE_export_original": rmse_export,
            "RMSE_export_minus_memory": rmse_export-rmse_memory,
            "within_tolerance": True}


def validate_windows(windows):
    """Serving guard retained outside a TorchScript trace (no silent reshape)."""
    if not isinstance(windows,list) or not windows:
        raise ValueError("Expected nonempty batch of standardized D7x5 windows")
    for window in windows:
        if not isinstance(window,list) or len(window)!=7:
            raise ValueError("Every window must contain exactly seven days")
        for row in window:
            if not isinstance(row,list) or len(row)!=5:
                raise ValueError("Every day must contain exactly five channels")
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in row):
                raise ValueError("Input channels must be finite numeric values")
    return windows


def predict_exported_reference(directory, windows):
    """Load verified CPU artifact; caller supplies the documented standardized input."""
    import torch
    windows=validate_windows(windows)
    directory=Path(directory)
    metadata=read_json(directory/"metadata.json")
    path=directory/"model.torchscript.pt"
    if file_sha(path)!=metadata.get("inference_artifact_sha256"):
        raise ValueError("Serving artifact hash mismatch")
    scale=float(metadata["preprocessing"]["target_scale"])
    if not math.isfinite(scale) or scale<=0:
        raise ValueError("Invalid saved output scale")
    model=torch.jit.load(str(path),map_location="cpu").eval()
    tensor=torch.tensor(windows,dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("Input overflows float32")
    with torch.inference_mode():
        values=model(tensor)*scale
    return finite_vector(values.tolist(),"exported prediction")


def calibration_cohort(market_path, target_path, scaler, dates):
    """Construct the saved D7 relative-window contract, never fit a scaler."""
    with Path(market_path).open(newline="", encoding="utf-8") as stream:
        market = {r["market_day_utc"][:10]: r for r in csv.DictReader(stream)}
    with Path(target_path).open(newline="", encoding="utf-8") as stream:
        targets = {r["market_day_utc"][:10]: r for r in csv.DictReader(stream)}
    if len(dates) != 53 or dates != [(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]:
        raise ValueError("Only the frozen 53-date historical calibration cohort is admitted")
    means, stds = finite_vector(scaler["mean"], "means"), finite_vector(scaler["std"], "stds")
    scale = float(scaler["target_scale"])
    if len(means) != 5 or len(stds) != 5 or min(stds) <= 0 or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid frozen D7 scaler")
    x, y = [], []
    for day in dates:
        current = date.fromisoformat(day)
        prior = [(current-timedelta(days=j)).isoformat() for j in range(7,0,-1)]
        anchor = float(market[prior[-1]]["close"])
        if anchor <= 0:
            raise ValueError("Nonpositive anchor close")
        window = []
        for previous in prior:
            r = market[previous]
            prices = [float(r[k]) for k in ("open", "high", "low", "close")]
            volume = float(r["volume"])
            if min(prices) <= 0 or volume < 0:
                raise ValueError("Invalid OHLCV for logarithmic input")
            raw = [math.log(p)-math.log(anchor) for p in prices]+[math.log1p(volume)]
            window.append([(v-means[j])/stds[j] for j,v in enumerate(raw)])
        target = float(targets[day]["sigma_7"])
        closes = [float(market[(current-timedelta(days=j)).isoformat()]["close"]) for j in range(7,-1,-1)]
        returns = [math.log(b/a) for a,b in zip(closes, closes[1:])]
        mean = statistics.fmean(returns)
        reconstructed = math.sqrt(statistics.fmean((v-mean)**2 for v in returns))
        if not math.isfinite(target) or abs(target-reconstructed) > 1e-12:
            raise ValueError("Public target differs from population sigma7 reconstruction")
        x.append(window); y.append(target)
    return x, y


def export_reference(args):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)  # Does not train or determine the saved predictor seed.
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        raise ValueError("A positive finite absolute tolerance is required")
    freeze, release = read_json(args.freeze), read_json(args.release_manifest)
    record = select_seed(freeze, args.model)
    selected = next(r for r in readiness(freeze, release, weights_root=args.weights_root,
                                         metadata_root=args.metadata_root)["rows"] if r["model"] == args.model)
    for field in ("checkpoint", "scaler", "calibrator"):
        if selected["artifacts"][field]["status"] != "bytes_verified":
            raise FileNotFoundError("Verified local " + field + " required")
    sys.path.insert(0, str(Path(args.sources).resolve()))
    from dl_models import SmallMLP, SingleLayerLSTM
    from dl_models_gru_lstm_tcn import SingleLayerGRU
    saved = torch.load(safe_path(args.weights_root, record["checkpoint"]), map_location="cpu", weights_only=True)
    architecture = saved["architecture"]
    if architecture.get("model") != args.model or architecture.get("initialization_seed") != record["seed"]:
        raise ValueError("Saved architecture/model/seed differs from freeze")
    receipt = read_json(args.selection_receipt)
    if receipt.get("model_id") != args.model or architecture.get("training_hpo_record") != receipt.get("winner_trial"):
        raise ValueError("Saved training recipe differs from the validation-selected recipe")
    scaler = read_json(safe_path(args.metadata_root, record["scaler"]))
    calibrator = read_json(safe_path(args.metadata_root, record["calibrator"]))
    if scaler != saved["scaler"] or calibrator.get("predictor_sha256") != record["checkpoint_sha256"]:
        raise ValueError("Scaler or calibrator is not bound to selected saved weights")
    constructors = {"MLP16": SmallMLP, "LSTM32": SingleLayerLSTM, "GRU32": SingleLayerGRU}
    model = constructors[args.model](7, REFERENCE_WIDTHS[args.model], dropout=float(architecture["dropout"]), input_features=5)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    x, truth = calibration_cohort(args.market, args.targets, scaler, calibrator["dates"])
    tensor = torch.tensor(x, dtype=torch.float32)
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite standardized input")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Choose an empty output directory; exports are immutable")
    output.mkdir(parents=True)
    exported_path = output / "model.torchscript.pt"
    with torch.inference_mode():
        # Python shape guards are traced away; the serving app must validate the
        # schema before invoking this fixed-D7, dynamic-batch artifact.
        traced = torch.jit.trace(model, tensor[:1], check_inputs=[(tensor[:3],)], strict=True)
        torch.jit.save(traced, str(exported_path))
        restored = torch.jit.load(str(exported_path), map_location="cpu").eval()
        memory = model(tensor) * scaler["target_scale"]
        recovered = restored(tensor) * scaler["target_scale"]
        metrics = roundtrip_metrics(memory.tolist(), recovered.tolist(), truth, tolerance=args.tolerance)
        single = tensor[:1]
        for _ in range(args.warmup):
            restored(single)
        timings = []
        for _ in range(args.repetitions):
            start = time.perf_counter_ns(); restored(single)
            timings.append((time.perf_counter_ns()-start)/1e6)
    timings.sort()
    try:
        import resource
        raw_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = raw_peak/(1024*1024 if sys.platform == "darwin" else 1024)
    except ImportError:
        peak = None
    parameters = sum(p.numel() for p in model.parameters())
    profile = {"device": "CPU", "torch_threads": 1, "batch_size": 1,
               "warmup_calls": args.warmup, "measured_calls": args.repetitions,
               "latency_median_ms": statistics.median(timings),
               "latency_p95_ms": timings[math.ceil(.95*len(timings))-1],
               "process_peak_RSS_MiB": peak,
               "memory_scope": "isolated export process; includes Python, torch, checkpoint and cohort; not incremental model RAM",
               "latency_scope": "TorchScript forward only, preprocessed CPU tensor; excludes IO and preprocessing",
               "parameters": parameters, "FLOPs": None,
               "FLOPs_status": "not_measured_by_this_runner"}
    environment = {"python": sys.version.split()[0], "torch": torch.__version__,
                   "system": platform.system(), "machine": platform.machine()}
    report = {"version": VERSION, "model": args.model, "seed": record["seed"],
              "seed_rule": selected["seed_rule"], "checkpoint_sha256": record["checkpoint_sha256"],
              "export_sha256": file_sha(exported_path), "export_bytes": exported_path.stat().st_size,
              "cohort": {"role": "historical_calibration", "dates": calibrator["dates"],
                         "market_sha256": file_sha(args.market), "targets_sha256": file_sha(args.targets),
                         "used_for_training_or_HPO_here": False, "A2_test_608_reproduced": False},
              "roundtrip": metrics, "CPU_profile": profile,
              "environment": environment,
              "model_sources_sha256": {name:file_sha(Path(args.sources)/name) for name in
                                       ("dl_models.py", "dl_models_gru_lstm_tcn.py")},
              "retrained": False, "selected_checkpoint_has_optimizer_state": "optimizer" in saved or "optimizer_state_dict" in saved,
              "scope": "Real saved reference weights and real historical calibration data; not confirmation of E3 or the original A2 test metric."}
    metadata = {"schema_version": VERSION, "model": args.model, "seed": record["seed"],
                "format": "TorchScript", "libraries": environment,
                "hyperparameters": receipt["winner_trial"], "fixed_phase_epochs": receipt.get("fixed_phase_epochs"),
                "preprocessing": {"name": "relative_window", "input_history_days": 7, "features": ["open", "high", "low", "close", "volume"],
                                  "price_transform": "log(P)-log(last_known_close)", "volume_transform": "log1p(V)",
                                  "channel_mean": scaler["mean"], "channel_std": scaler["std"],
                                  "training_dates_min": min(scaler["train_dates"]),
                                  "training_dates_max": max(scaler["train_dates"]),
                                  "training_dates_count": len(scaler["train_dates"]),
                                  "target_scale": scaler["target_scale"], "scaler_sha256": record["scaler_sha256"]},
                "input_schema": {"shape": [None,7,5], "dtype": "float32", "representation": "standardized relative_window OHLCV",
                                 "finite_required": True, "domain_ranges": "not estimated by export; serving OOD checks pending",
                                 "validation": "serving app must validate shape/dtype/finiteness before TorchScript invocation"},
                "output_schema": {"shape": [None], "meaning": "positive sigma7/target_scale; multiply by target_scale outside artifact"},
                "calibration": calibrator, "validation": {"winner_pooled_cv_RMSE_original": receipt.get("winner_pooled_cv_RMSE_original"),
                                                           "selection_scope": receipt.get("selection_scope")},
                "historical_calibration_roundtrip": metrics, "A2_exported_test_metric": None,
                "A2_exported_test_metric_status": "pending: no 608-date test input cohort supplied to this command",
                "inference_artifact_sha256": file_sha(exported_path), "source_checkpoint_sha256": record["checkpoint_sha256"]}
    for name,value in (("metadata.json",metadata),("roundtrip_report.json",report)):
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("inventory")
    audit.add_argument("--exports-root",type=Path)
    export = commands.add_parser("export-reference")
    for sub in (audit,export):
        sub.add_argument("--freeze",type=Path,required=True)
        sub.add_argument("--release-manifest",type=Path,required=True)
        sub.add_argument("--weights-root",type=Path)
        sub.add_argument("--metadata-root",type=Path)
        sub.add_argument("--output",type=Path,required=True)
    export.add_argument("--model",choices=tuple(REFERENCE_WIDTHS),required=True)
    export.add_argument("--selection-receipt",type=Path,required=True)
    export.add_argument("--sources",type=Path,default=Path(__file__).resolve().parent)
    export.add_argument("--market",type=Path,required=True)
    export.add_argument("--targets",type=Path,required=True)
    export.add_argument("--tolerance",type=float,default=1e-6)
    export.add_argument("--warmup",type=int,default=10)
    export.add_argument("--repetitions",type=int,default=100)
    args = parser.parse_args()
    if args.command == "inventory":
        result = readiness(read_json(args.freeze),read_json(args.release_manifest),
                           weights_root=args.weights_root,metadata_root=args.metadata_root,
                           exports_root=args.exports_root)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({"systems":result["systems"],"frozen_predictors":result["frozen_predictors"]}))
    else:
        if args.warmup < 0 or args.repetitions < 20:
            parser.error("Require nonnegative warmup and at least20 measured calls")
        result = export_reference(args)
        print(json.dumps({"model":result["model"],"roundtrip":result["roundtrip"],"CPU_profile":result["CPU_profile"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
