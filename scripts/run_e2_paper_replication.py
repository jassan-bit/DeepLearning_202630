#!/usr/bin/env python3
"""Registered ETTh1 reproduction and a separate, small trained-model pilot.

The author backbones are imported from separately acquired, hash-checked source
trees. No author code or benchmark CSV is bundled here. Dataset fitting, scoring
and the pilot are explicit phases; a public-data numerical reproduction is not
the Bitcoin benchmark or a new confirmatory Bitcoin test.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from e2_paper_data import fit_scaler, load_etth1, make_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "experiments/e2_paper_guide_20261106/design.json"
EXPECTED_DATA_SHA = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
ALLOWED_MODELS = {"TimeMixer", "iTransformer", "PatchTST"}
DEFAULT_MODEL_ARGS = {
    "task_name": "long_term_forecast", "seq_len": 96, "label_len": 0,
    "pred_len": 96, "enc_in": 7, "dec_in": 7, "c_out": 7,
    "e_layers": 2, "d_layers": 1, "d_model": 16, "d_ff": 32,
    "down_sampling_layers": 3, "down_sampling_window": 2,
    "down_sampling_method": "avg", "dropout": 0.1, "embed": "timeF",
    "freq": "h", "channel_independence": 1, "decomp_method": "moving_avg",
    "moving_avg": 25, "use_norm": 1, "use_future_temporal_feature": 0,
    "top_k": 5, "n_heads": 4, "factor": 3, "activation": "gelu",
    "output_attention": False,
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, exclusive=False):
    with Path(path).open("xb" if exclusive else "wb") as stream:
        stream.write(json_bytes(value))


def now():
    return datetime.now(timezone.utc).isoformat()


def runtime():
    versions = {}
    for name in ("torch", "numpy", "pandas", "scipy", "einops", "reformer-pytorch"):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return {"python": sys.version, "packages": versions, "device": "cpu",
            "torch_threads": torch.get_num_threads(), "CUDA_used": False}


def read_design(path):
    plan = read_json(path)
    block = plan.get("replication", {})
    if block.get("model", "TimeMixer") != "TimeMixer":
        raise ValueError("This registered primary reproduction must use TimeMixer")
    seeds = block.get("seeds", [2019, 2020, 2021, 2022, 2023])
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(type(s) is not int for s in seeds):
        raise ValueError("The primary reproduction requires five distinct integer seeds")
    expected = plan.get("dataset", {}).get("sha256", EXPECTED_DATA_SHA)
    if expected != EXPECTED_DATA_SHA:
        raise ValueError("The original ETTh1 dataset identity differs")
    registered_data = plan.get("dataset", {})
    pinned_data = {"global_scaler_fit_rows": [0, 8640], "scaler_ddof": 0,
                   "split_source_rows": {"train": [0, 8640],
                                         "validation_labels": [8640, 11520],
                                         "test_labels": [11520, 14400]},
                   "nominal_windows": {"train": 8449, "val": 2785, "test": 2785},
                   "author_processed_windows": {"train": 8448, "val": 2688, "test": 2688}}
    for field, required in pinned_data.items():
        if field in registered_data and registered_data[field] != required:
            raise ValueError("Registered dataset boundary differs: " + field)
    return plan


def section(plan, phase):
    return plan["pilot" if phase == "pilot" else "replication"]


def model_arguments(plan, phase="replication", model_name="TimeMixer"):
    block = section(plan, phase)
    result = dict(DEFAULT_MODEL_ARGS)
    if phase == "pilot":
        result.update(block.get("model_args", {}))
        result.update(block.get("model_args_by_name", {}).get(model_name, {}))
    else:
        result.update(block.get("model_args", {}))
    if (result["seq_len"], result["pred_len"], result["enc_in"], result["c_out"]) != (96, 96, 7, 7):
        raise ValueError("The registered external experiment uses L96, H96 and seven channels")
    if phase != "pilot":
        pinned = ("seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out",
                  "e_layers", "d_model", "d_ff", "down_sampling_layers",
                  "down_sampling_window", "down_sampling_method", "dropout", "embed", "freq",
                  "channel_independence", "decomp_method", "moving_avg", "use_norm",
                  "use_future_temporal_feature")
        if any(result[key] != DEFAULT_MODEL_ARGS[key] for key in pinned):
            raise ValueError("Primary architecture no longer matches the fixed author preset")
    return result


def training_arguments(plan, phase="replication"):
    defaults = ({"batch_size": 128, "epochs": 2, "learning_rate": .001,
                 "threads": 4, "patience": 2, "drop_last": True,
                 "shuffle_validation": False, "scheduler": "constant"}
                if phase == "pilot" else
                {"batch_size": 128, "epochs": 10, "patience": 10,
                 "learning_rate": .01, "pct_start": .2, "threads": 4,
                 "drop_last": True, "shuffle_validation": True,
                 "scheduler": "OneCycleLR"})
    defaults.update(section(plan, phase).get("training", {}))
    for key in ("batch_size", "epochs", "threads", "patience"):
        if type(defaults[key]) is not int or defaults[key] < 1:
            raise ValueError("Invalid registered training setting: " + key)
    if phase != "pilot" and (defaults["batch_size"], defaults["epochs"], defaults["patience"],
                             defaults["learning_rate"], defaults["drop_last"]) != (128, 10, 10, .01, True):
        raise ValueError("Primary reproduction no longer matches the fixed author preset")
    if phase != "pilot" and (defaults.get("scheduler"), defaults.get("pct_start"),
                             defaults.get("optimizer", "Adam")) != ("OneCycleLR", .2, "Adam"):
        raise ValueError("Primary scheduler or optimizer differs from the fixed author preset")
    return defaults


def build_model(args, source_path, model_name="TimeMixer"):
    """Instantiate an unchanged author backbone; independent CLI jobs are safest.

    Clearing only author package names also permits read-only analysis of
    multiple separately acquired source trees in the same process. Existing
    instantiated classes retain their original function/module references.
    """
    if model_name not in ALLOWED_MODELS:
        raise ValueError("Unregistered model name")
    source_path = Path(source_path).resolve()
    if not (source_path / "models" / (model_name + ".py")).is_file():
        raise ValueError("Author model file not found: " + model_name)
    for name in list(sys.modules):
        if name in {"models", "layers", "utils", "data_provider"} or name.startswith(
                ("models.", "layers.", "utils.", "data_provider.")):
            del sys.modules[name]
    sys.path.insert(0, str(source_path))
    try:
        module = importlib.import_module("models." + model_name)
        values = vars(args) if isinstance(args, SimpleNamespace) else dict(args)
        return module.Model(SimpleNamespace(**values)).float()
    finally:
        sys.path.remove(str(source_path))


def source_receipts(plan, source_root, control_source_root=None):
    receipts = []
    specs = plan.get("sources", {})
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Registered source receipts are required before training")
    for key, spec in specs.items():
        base = Path(source_root if key in {"TimeMixer", "primary"} else
                    (control_source_root or source_root)).resolve()
        files = spec.get("files", spec.get("source_receipts", []))
        if not files:
            raise ValueError("No registered files for source " + key)
        for item in files:
            relative = item.get("path", item.get("repository_path"))
            if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Invalid source receipt path")
            path = base / relative
            if path.is_symlink() or not path.resolve().is_relative_to(base):
                raise ValueError("A source receipt escapes its root")
            actual = digest(path)
            if actual != item["sha256"]:
                raise ValueError("Source hash differs: " + key + "/" + relative)
            receipts.append({"source": key, "path": relative, "sha256": actual})
    return receipts


def registration_payload(design_path, plan, data_path, source_root, control_source_root=None):
    return {"design_sha256": digest(design_path), "data_sha256": digest(data_path),
            "runner_sha256": digest(__file__),
            "data_helper_sha256": digest(Path(__file__).with_name("e2_paper_data.py")),
            "sources": source_receipts(plan, source_root, control_source_root)}


def verify_registration(output_dir, expected):
    path = Path(output_dir) / "registration.json"
    if not path.is_file():
        raise ValueError("Run --phase register after committing the design and driver")
    registered = read_json(path)
    if registered.get("immutable_inputs") != expected:
        raise ValueError("Registered design, driver, dataset or author source has changed")
    return registered


def register(args, plan, inputs):
    tracked = [str(Path(args.design).resolve().relative_to(ROOT)),
               "scripts/run_e2_paper_replication.py", "scripts/e2_paper_data.py"]
    for path in tracked:
        subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT,
                       check=True, capture_output=True)
    if subprocess.check_output(["git", "diff", "HEAD", "--", *tracked], cwd=ROOT):
        raise ValueError("Commit the design and driver before registering dataset fits")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "registration.json"
    if path.exists():
        verify_registration(out, inputs)
        return read_json(path)
    result = {"registered_at_utc": now(), "source_commit": commit,
              "immutable_inputs": inputs, "dataset_fits_before_registration": 0,
              "scope": "External ETTh1 reproduction/pilot, not the Bitcoin benchmark"}
    write_json(path, result, exclusive=True)
    return result


def set_seed(seed, threads):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def make_loader(dataset, batch_size, seed, shuffle=False, drop_last=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=0, generator=generator)


def forward_batch(model, batch, pred_len=96):
    x, y, xmark, ymark = (v.float() for v in batch)
    # The TimeMixer preset downsamples and supplies no decoder input. Both
    # selected attention controls ignore the decoder values as well.
    pred = model(x, xmark, None, ymark)
    if isinstance(pred, tuple): pred = pred[0]
    target = y[:, -pred_len:, :]
    pred = pred[:, -pred_len:, :]
    if pred.shape != target.shape or not torch.isfinite(pred).all():
        raise ValueError("Nonfinite forecast or incorrect output shape")
    return pred, target


def predict(model, loader, pred_len=96):
    """Predict a supplied, explicitly selected cohort; never chooses a split."""
    model.eval(); predictions = []; targets = []
    with torch.no_grad():
        for batch in loader:
            p, y = forward_batch(model, batch, pred_len)
            predictions.append(p.cpu().numpy()); targets.append(y.cpu().numpy())
    if not predictions:
        raise ValueError("The loader produced no complete batches")
    return {"predictions": np.concatenate(predictions), "targets": np.concatenate(targets)}


def validation_loss(model, loader):
    model.eval(); values = []
    with torch.no_grad():
        for batch in loader:
            pred, target = forward_batch(model, batch)
            values.append(float(torch.mean((pred - target).square())))
    if not values: raise ValueError("Empty validation loader")
    return float(np.mean(values))


def run_directory(output_dir, model_name, seed, phase="replication"):
    return Path(output_dir) / ("pilot" if phase == "pilot" else "replication") / model_name / f"seed_{seed}"


def check_completed(directory, design_sha256, checkpoint=True):
    path = Path(directory) / "completed.json"
    if not path.is_file(): raise ValueError("A required fit is not complete: " + str(directory))
    receipt = read_json(path)
    if receipt.get("design_sha256") != design_sha256:
        raise ValueError("Completed fit belongs to another design")
    if checkpoint and digest(Path(directory) / "checkpoint.pth") != receipt["checkpoint_sha256"]:
        raise ValueError("A frozen checkpoint has changed")
    for filename, key in (("setup.json", "setup_sha256"), ("scaler.json", "scaler_sha256")):
        if key in receipt and digest(Path(directory) / filename) != receipt[key]:
            raise ValueError("Frozen fit metadata has changed: " + filename)
    if "validation_predictions_sha256" in receipt and digest(Path(directory) / "validation.npz") != receipt["validation_predictions_sha256"]:
        raise ValueError("Frozen pilot validation predictions have changed")
    return receipt


def fit_one(args, plan, inputs, pilot=False):
    phase = "pilot" if pilot else "replication"
    model_name = args.model if pilot else "TimeMixer"
    allowed = section(plan, phase).get("seeds", plan["replication"].get("seeds", [2019,2020,2021,2022,2023]))
    if args.seed not in allowed: raise ValueError("The seed is not in the registered plan")
    if pilot and model_name not in section(plan, phase).get("models", list(ALLOWED_MODELS)):
        raise ValueError("The pilot model is not registered")
    verify_registration(args.output_dir, inputs)
    directory = run_directory(args.output_dir, model_name, args.seed, phase)
    if directory.exists():
        return check_completed(directory, inputs["design_sha256"])
    directory.mkdir(parents=True)
    write_json(directory / "started.json", {"at_utc": now(), "immutable_inputs": inputs,
               "phase": phase, "model": model_name, "seed": args.seed}, exclusive=True)
    train_cfg = training_arguments(plan, phase); model_args = model_arguments(plan, phase, model_name)
    set_seed(args.seed, train_cfg["threads"])
    data = load_etth1(args.data, EXPECTED_DATA_SHA)
    scaler = fit_scaler(data["values"])
    train = make_split(data["values"], data["dates"], "train", scaler=scaler)
    if pilot:
        n = int(plan["pilot"].get("train_windows", 2048))
        if n < train_cfg["batch_size"] or n > len(train): raise ValueError("Invalid pilot subset")
        train = Subset(train, range(len(train) - n, len(train)))
    train_loader = make_loader(train, train_cfg["batch_size"], args.seed + 100001,
                               shuffle=True, drop_last=True)
    valid = None; valid_loader = None
    if not pilot:
        valid = make_split(data["values"], data["dates"], "val", scaler=scaler)
        valid_loader = make_loader(valid, train_cfg["batch_size"], args.seed + 200001,
                                  shuffle=train_cfg["shuffle_validation"], drop_last=True)
    source = args.source_root if model_name == "TimeMixer" else args.control_source_root
    if not source: raise ValueError("A control source root is required")
    model = build_model(model_args, source, model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_cfg["learning_rate"],
                 epochs=train_cfg["epochs"], steps_per_epoch=len(train_loader),
                 pct_start=train_cfg.get("pct_start", .2)) if not pilot else None)
    setup = {"model_name": model_name, "model_args": model_args, "training": train_cfg,
             "phase": phase, "seed": args.seed, "design_sha256": inputs["design_sha256"],
             "loader_seeds": {"training": args.seed + 100001, "validation": args.seed + 200001},
             "loader_workers": 0, "runtime": runtime(), "scaler": scaler,
             "author_driver_deviations": ["CPU/current locked framework instead of original historical environment",
               "Independent loader generators; num_workers0; no per-epoch test evaluation",
               "Five registered new seeds, not recovery of unpublished author repetitions"],
             "n_train_windows": len(train), "n_train_batches": len(train_loader),
             "parameters": sum(p.numel() for p in model.parameters())}
    write_json(directory / "setup.json", setup, exclusive=True)
    write_json(directory / "scaler.json", scaler, exclusive=True)
    curves = []; best = math.inf; best_epoch = None; stale = 0; beginning = time.perf_counter()
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train(); losses = []; epoch_start = time.perf_counter()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred, target = forward_batch(model, batch)
            loss = torch.mean((pred - target).square())
            if not torch.isfinite(loss): raise ValueError("Nonfinite training loss")
            loss.backward(); optimizer.step()
            if scheduler is not None: scheduler.step()
            losses.append(float(loss.detach()))
        train_seconds = time.perf_counter() - epoch_start
        val = validation_loss(model, valid_loader) if valid_loader is not None else None
        if pilot or val <= best:
            if val is not None: best = val
            best_epoch = epoch; stale = 0
            torch.save(model.state_dict(), directory / "checkpoint.pth")
        else: stale += 1
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": val,
                  "training_seconds": train_seconds, "elapsed_seconds": time.perf_counter() - beginning,
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        curves.append(record)
        write_json(directory / "curves.json", curves)
        print(json.dumps({"model": model_name, "phase": phase, "seed": args.seed, **record}), flush=True)
        if not pilot and stale >= train_cfg["patience"]: break
    if not pilot and not math.isfinite(best): raise ValueError("No finite selected validation loss")
    pilot_predictions_hash = None
    if pilot:
        # Fixed-epoch pilot: validation does not select weights or any setting.
        # Keep every validation origin; this is distinct from the author's
        # drop-last replication cohort used only by the primary score phase.
        valid = make_split(data["values"], data["dates"], "val", scaler=scaler)
        valid_loader = make_loader(valid, train_cfg["batch_size"], args.seed + 200001,
                                   shuffle=False, drop_last=False)
        result = predict(model, valid_loader)
        np.savez_compressed(directory / "validation.npz", **result,
                            origins=np.asarray(valid.forecast_origins))
        pilot_predictions_hash = digest(directory / "validation.npz")
    receipt = {"completed_at_utc": now(), "phase": phase, "model": model_name,
               "seed": args.seed, "design_sha256": inputs["design_sha256"],
               "checkpoint_sha256": digest(directory / "checkpoint.pth"),
               "setup_sha256": digest(directory / "setup.json"),
               "scaler_sha256": digest(directory / "scaler.json"),
               "best_epoch": best_epoch, "best_validation_mse": None if pilot else best,
               "epochs_completed": len(curves), "optimizer_steps": len(curves) * len(train_loader),
               "training_elapsed_seconds": time.perf_counter() - beginning,
               "test_predictions_computed": False, "Bitcoin_model_fits": 0}
    if pilot:
        receipt["validation_predictions_sha256"] = pilot_predictions_hash
        receipt["validation_origins"] = len(valid)
        receipt["validation_used_for_weight_selection"] = False
    write_json(directory / "completed.json", receipt, exclusive=True)
    return receipt


def load_fitted_model(run_dir, source_path):
    directory = Path(run_dir); setup = read_json(directory / "setup.json")
    check_completed(directory, setup["design_sha256"])
    model = build_model(setup["model_args"], source_path, setup["model_name"])
    model.load_state_dict(torch.load(directory / "checkpoint.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model, setup


def score(args, plan, inputs):
    verify_registration(args.output_dir, inputs)
    seeds = plan["replication"].get("seeds", [2019,2020,2021,2022,2023])
    # Check every frozen fit BEFORE constructing an evaluation split or target.
    directories = [run_directory(args.output_dir, "TimeMixer", seed) for seed in seeds]
    receipts = [check_completed(d, inputs["design_sha256"]) for d in directories]
    output = Path(args.output_dir) / "replication_scores"
    if output.exists():
        existing = read_json(output / "completed.json")
        if existing.get("design_sha256") != inputs["design_sha256"]:
            raise ValueError("Existing scores use another design")
        for row in existing["rows"]:
            path = output / f"seed_{row['seed']}_test.npz"
            if digest(path) != row["prediction_npz_sha256"]:
                raise ValueError("Frozen scored predictions have changed")
        if "metrics_csv_sha256" in existing and digest(output / "metrics.csv") != existing["metrics_csv_sha256"]:
            raise ValueError("Frozen score table has changed")
        return existing
    output.mkdir()
    write_json(output / "scoring_started.json", {"at_utc": now(), "design_sha256": inputs["design_sha256"],
               "frozen_checkpoint_hashes": [r["checkpoint_sha256"] for r in receipts]}, exclusive=True)
    data = load_etth1(args.data, EXPECTED_DATA_SHA); rows = []
    for seed, directory in zip(seeds, directories):
        model, setup = load_fitted_model(directory, args.source_root)
        torch.set_num_threads(setup["training"]["threads"])
        test = make_split(data["values"], data["dates"], "test", scaler=setup["scaler"])
        loader = make_loader(test, 128, seed + 300001, shuffle=False, drop_last=True)
        result = predict(model, loader)
        pred = result["predictions"]; target = result["targets"]
        np.savez_compressed(output / f"seed_{seed}_test.npz", predictions=pred, targets=target,
                            origins=np.asarray(test.forecast_origins[:len(pred)]))
        error = pred - target
        rows.append({"seed": seed, "MSE": float(np.mean(error ** 2)),
                     "MAE": float(np.mean(np.abs(error))), "nominal_origins": len(test),
                     "scored_origins": len(pred), "omitted_origins": len(test) - len(pred),
                     "prediction_npz_sha256": digest(output / f"seed_{seed}_test.npz")})
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    final = {"completed_at_utc": now(), "design_sha256": inputs["design_sha256"],
             "rows": rows, "source_scale": "ETTh1 train-standardized, all7channels, H96",
             "metrics_csv_sha256": digest(output / "metrics.csv"),
             "scoring_once_after_all_registered_checkpoints_frozen": True,
             "Bitcoin_results": 0}
    write_json(output / "completed.json", final, exclusive=True)
    return final


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=["validate", "register", "fit", "score", "pilot"])
    p.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--control-source-root", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="TimeMixer", choices=sorted(ALLOWED_MODELS))
    p.add_argument("--seed", type=int)
    args = p.parse_args(argv)
    try:
        plan = read_design(args.design)
        model_arguments(plan)
        training_arguments(plan)
        inputs = registration_payload(args.design, plan, args.data, args.source_root, args.control_source_root)
        if inputs["data_sha256"] != EXPECTED_DATA_SHA: raise ValueError("Dataset bytes differ")
        if args.phase == "validate":
            data = load_etth1(args.data, EXPECTED_DATA_SHA); scaler = fit_scaler(data["values"])
            result = {"valid": True, "dataset_sha256": inputs["data_sha256"],
                      "windows": {s: len(make_split(data["values"], data["dates"], s, scaler=scaler))
                                  for s in ("train", "val", "test")},
                      "dataset_fits": 0, "Bitcoin_results": 0}
        elif args.phase == "register": result = register(args, plan, inputs)
        elif args.phase == "score": result = score(args, plan, inputs)
        else:
            if args.seed is None: raise ValueError("A registered seed is required")
            result = fit_one(args, plan, inputs, pilot=args.phase == "pilot")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
