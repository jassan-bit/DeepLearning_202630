"""Claim-bound CUDA fits and delayed evaluation for the posterior E2 extension.

This adapter calls the unchanged D02 production fitter. It never changes the
historical six-seed campaign and never owns or resets a compute budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

VERSION = "e2-seed-extension-v1"
NEURAL = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "MLP16", "LSTM32", "GRU32")
NEW_SEEDS = tuple(range(271807, 271811))
TRAIN_SIGNATURE = "7077ed2c3da26013c018727b32efc168efba478c7397e81cfd4302195b825ff3"
NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
SCALER_SHA256 = "245eaf5deeb2c236a51c495b2225b5a7a6f351e2abd2bd1941ad2d3eb65b2959"
FROZEN_SHA256 = "da004139cfc4663ce1d9a13c88a422df8865b53f6448fbdf015da0244f9f51cf"
REGISTER_SHA256 = "e0c13b18cc20b161fc1f7abe8ff8ae3b1a42d5b6ea07916e45e098d99c66c072"
SOURCE_BINDING_SHA256 = "b2e7063e11e3f5fff5a00722412c88666e59cb375bdc26b1a34f146790e9e8ac"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def immutable_json(path: Path, value: Any) -> None:
    path = Path(path)
    payload = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Existing immutable artifact differs: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def safe_job_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,96}", value):
        raise ValueError("A bounded, path-safe job ID is required")
    return value


def verify_file(path: Path, expected: str) -> Path:
    path = Path(path).absolute()
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("A regular immutable input is required: " + str(path))
    if sha(path) != expected:
        raise ValueError("Input SHA-256 differs: " + str(path))
    return path


def package_paths(bundle_root: Path) -> tuple[Path, Path]:
    package = Path(bundle_root).resolve()
    if not (package / "code/scripts/e2_reduced_fit.py").is_file():
        raise ValueError("--bundle-root must name the extracted D02 package")
    return package, package / "code"


def original_module(code_root: Path, name: str):
    scripts = Path(code_root).resolve() / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != scripts / (name + ".py"):
        raise ValueError("An original scientific module was shadowed: " + name)
    return module


def source_inventory(bundle_root: Path) -> tuple[dict, dict, dict]:
    package, code = package_paths(bundle_root)
    source = read(package.parents[1] / "source_manifest.json")
    binding = {k: source[k] for k in ("source_zip_sha256", "source_commit", "source_repository_path", "members")}
    if hashlib.sha256(canonical(binding)).hexdigest() != SOURCE_BINDING_SHA256:
        raise ValueError("The complete source/author manifest differs from the pinned D02 archive")
    selected = set()
    for row in source["members"]:
        parts = Path(row["path"])
        if parts.is_absolute() or ".." in parts.parts or parts.parts[0] != "D02":
            raise ValueError("Unsafe source manifest member")
        path = verify_file(package.parent / parts, row["sha256"])
        if path.stat().st_size != row["bytes"]:
            raise ValueError("Source member byte count differs")
        selected.add(path)
    if {p.absolute() for p in package.rglob("*") if p.is_file()} != selected:
        raise ValueError("Unregistered files were added to the source-only bundle")
    standalone = original_module(code, "e2_standalone_code")
    report, recipes, roots = standalone.inventory(package, code)
    if report["author_files_verified"] != 76 or len(recipes) != 13:
        raise ValueError("Exactly thirteen recipes and 76 author files are required")
    verify_file(code / standalone.FROZEN_INPUTS, FROZEN_SHA256)
    verify_file(code / standalone.REGISTER, REGISTER_SHA256)
    return report, recipes, roots


def recipe_epochs(code_root: Path, recipes: dict, model: str) -> tuple[dict, dict]:
    import csv
    standalone = original_module(code_root, "e2_standalone_code")
    path = verify_file(code_root / standalone.REGISTER, REGISTER_SHA256)
    with path.open(newline="", encoding="utf-8") as stream:
        selected = [r for r in csv.DictReader(stream) if r["model"] == model]
    if len(selected) != 1:
        raise ValueError("One frozen epoch register row is required per model")
    row, trial = selected[0], recipes[model]["trial"]
    if row["trial_id"] != trial["trial_id"] or row["recipe_id"] != trial["recipe_id"]:
        raise ValueError("Winner trial and epoch register differ")
    phases = ("ae1", "ae2", "supervised") if model == NEURAL[3] else ("supervised",)
    epochs = {phase: int(row["fixed_" + phase + "_epochs"]) for phase in phases}
    if any(not 1 <= v <= 200 for v in epochs.values()):
        raise ValueError("A frozen phase count is invalid")
    return trial, epochs


def input_paths(inputs: Path) -> tuple[Path, Path]:
    folder = Path(inputs)
    return (verify_file(folder / "current_cohort_arrays.npz", NPZ_SHA256),
            verify_file(folder / "reconstructed_training_scaler.json", SCALER_SHA256))


def prepared_training(code_root: Path, arrays_path: Path, scaler_record: dict):
    """Materialize training members only; no calibration or test array access."""
    import numpy as np
    import torch
    temporal = original_module(code_root, "e2_temporal_training")
    dated = original_module(code_root, "e2_real_executor")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        x = torch.from_numpy(arrays["training_X"].copy())
        y = torch.from_numpy(arrays["training_y_scaled"].copy())
        original = tuple(float(v) for v in arrays["training_y_original"])
        dates = tuple(dated.dated_index(arrays["training_dates"].tolist()))
    if x.dtype != torch.float32 or y.dtype != torch.float32 or tuple(x.shape) != (655, 7, 5):
        raise ValueError("Frozen training tensors must remain 655 FP32 D7 windows")
    scaler = temporal.FoldScaler(tuple(scaler_record["mean"]), tuple(scaler_record["std"]),
        scaler_record["target_scale"], tuple(dated.dated_index(scaler_record["train_dates"])),
        scaler_record["training_sha256"])
    name = "reduced_final_train_through_2024-10-31"
    signature = dated.canonical_hash({"name": name, "dates": dated.date_strings(dates),
        "x": temporal._tensor_hash(x), "y": temporal._tensor_hash(y),
        "y_original": list(original), "scaler": scaler.record()})
    if signature != TRAIN_SIGNATURE or scaler.record() != scaler_record:
        raise ValueError("Training-only scaler or all-78 historical training signature differs")
    split = temporal.PreparedSplit(name, x, y, original, dates, scaler, signature)
    split.validate()
    return split


def cohort(arrays_path: Path, phase: str):
    """Labels never enter the original prediction callable."""
    import numpy as np
    import torch
    if phase not in ("calibration", "evaluation"):
        raise ValueError("Unknown cohort")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        x = torch.from_numpy(arrays[phase + "_X"].copy())
        y = arrays[phase + "_y_original"].copy()
        dates = arrays[phase + "_dates"].tolist()
    n = 53 if phase == "calibration" else 608
    if x.dtype != torch.float32 or tuple(x.shape) != (n, 7, 5) or y.shape != (n,):
        raise ValueError("Frozen cohort geometry changed")
    return x, y, dates


def runtime_identity(device: str, bundle_root: Path) -> dict:
    if device != "cuda:0":
        raise ValueError("Production workers require cuda:0")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before worker startup")
    import platform
    import importlib.metadata
    import torch
    if torch.__version__ != "2.8.0+cu126" or not torch.cuda.is_available():
        raise ValueError("The verified Torch 2.8.0+cu126 CUDA runtime is required")
    if tuple(sys.version_info[:3]) != (3, 12, 12):
        raise ValueError("The registered Python 3.12.12 runtime is required")
    expected = {}
    lock = Path(bundle_root) / "requirements_cuda.txt"
    for line in lock.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--index-url ", "--extra-index-url ")):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if not match:
            raise ValueError("Unexpected original runtime lock syntax")
        expected[match[1]] = match[2]
    if len(expected) != 43:
        raise ValueError("The exact original 43-package runtime lock is required")
    actual_packages = {name: importlib.metadata.version(name) for name in expected}
    if actual_packages != expected:
        raise ValueError("Actual worker packages differ from the pinned original runtime")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    props = torch.cuda.get_device_properties(0)
    return {"device": device, "device_name": props.name,
        "total_memory_bytes": props.total_memory, "python": platform.python_version(),
        # TorchVersion is a str subclass; the frozen artifact writer accepts
        # exact JSON builtin types, so normalize at this adapter boundary.
        "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "cpu_threads": 2, "interop_threads": 1, "precision": "float32",
        "amp": False, "tf32": False, "cudnn_tf32": False, "deterministic": True,
        "packages": actual_packages}


def claim_context(run_dir: Path, claim_path: Path, job_id: str, phase: str) -> tuple[dict, dict, dict]:
    """Read-only coordinator admission; the parent alone charges worker wall time."""
    manifest = read(run_dir / "run_manifest.json")
    plan = read(run_dir / "plan.json")
    if (hashlib.sha256(canonical(plan)).hexdigest() != manifest.get("plan_fingerprint")
            or plan.get("jobs") != manifest.get("jobs")):
        raise ValueError("The immutable plan fingerprint or job grid changed")
    hashes = manifest.get("code_hashes", {})
    if not hashes or hashes.get("worker.py") != sha(Path(__file__)):
        raise ValueError("The worker is not bound to the registered code manifest")
    for name, expected in hashes.items():
        parts = Path(name)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("Unsafe registered code path")
        verify_file(Path(__file__).parent / parts, expected)
    claim = read(claim_path)
    fields = ("run_id", "plan_fingerprint")
    if claim.get("version") != VERSION or any(claim.get(k) != manifest.get(k) for k in fields):
        raise ValueError("Claim version, run or plan differs")
    if claim.get("job_id") != job_id or claim.get("phase") != phase:
        raise ValueError("Claim is for a different job or phase")
    if not isinstance(claim.get("reservation_id"), str) or not claim["reservation_id"]:
        raise ValueError("An explicit coordinator reservation is required")
    cap = claim.get("cap_seconds")
    phase_cap = {"fit": 1800, "evaluate": 300, "aggregate": 120}.get(phase, 0)
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or not 0 < cap <= phase_cap:
        raise ValueError("A positive bounded coordinator reservation is required")
    jobs = manifest.get("jobs", [])
    expected = {(m, s) for m in NEURAL for s in NEW_SEEDS}
    if len(jobs) != 52 or {(j.get("model"), j.get("seed")) for j in jobs} != expected:
        raise ValueError("Only the exact thirteen-by-four new-seed schedule is admitted")
    selected = [j for j in jobs if j.get("id") == job_id]
    if len(selected) != 1:
        raise ValueError("One registered job ID is required")
    ledger_path = run_dir / "ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines() if line]
    active = {}
    previous = "0" * 64
    for index, row in enumerate(rows):
        value = {k: v for k, v in row.items() if k != "sha256"}
        if (row.get("sha256") != hashlib.sha256(canonical(value)).hexdigest()
                or row.get("sequence") != index or row.get("previous_sha256") != previous):
            raise ValueError("Coordinator ledger hash chain differs")
        previous = row["sha256"]
        reservation = row.get("reservation_id")
        if row.get("event") == "reserve":
            active[reservation] = row
        elif row.get("event") in ("finish", "charge"):
            active.pop(reservation, None)
    reservation = active.get(claim["reservation_id"])
    if len(active) != 1 or not reservation or any(reservation.get(k) != claim.get(k) for k in
        ("run_id", "plan_fingerprint", "job_id", "phase", "cap_seconds")):
        raise ValueError("Claim has no matching active coordinator ledger reservation")
    if claim.get("ledger_event_sha256") != reservation["sha256"]:
        raise ValueError("Claim does not bind the active reservation bytes")
    if not re.fullmatch(r"[0-9a-f]{40}", claim.get("ack_commit", "")):
        raise ValueError("A remote-acknowledged reservation commit is required")
    return manifest, claim, selected[0]


def verify_artifacts(directory: Path, artifacts: dict) -> None:
    if not artifacts:
        raise ValueError("A complete immutable artifact map is required")
    for relative, expected in artifacts.items():
        parts = Path(relative)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("Artifact path escapes its job")
        verify_file(directory / relative, expected)


def completed_fit(run_dir: Path, job_id: str, *, run_id: str | None = None,
                  plan_fingerprint: str | None = None) -> dict | None:
    path = Path(run_dir) / "jobs" / safe_job_id(job_id) / "fit_receipt.json"
    if not path.is_file():
        return None
    value = read(path)
    if value.get("version") != VERSION or value.get("status") != "completed":
        raise ValueError("A prior fit requires reconciliation")
    if run_id is not None and value.get("run_id") != run_id:
        raise ValueError("Completed fit belongs to another run")
    if plan_fingerprint is not None and value.get("plan_fingerprint") != plan_fingerprint:
        raise ValueError("Completed fit belongs to another plan")
    verify_artifacts(path.parent, value["artifacts_sha256"])
    for key in ("checkpoint", "scaler", "calibrator", "calibration_predictions"):
        verify_file(path.parent / value[key], value[key + "_sha256"])
    return value


def load_predictor(code_root: Path, recipes: dict, roots: dict, model: str, seed: int,
                   checkpoint: Path, scaler: dict):
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("scaler") != scaler:
        raise ValueError("Checkpoint scaler differs from the original training-only scaler")
    original = original_module(code_root, "e2_reduced_fit")
    predictor = original._builder(model, recipes[model]["trial"], seed, roots)
    predictor.load_state_dict(saved["state_dict"], strict=True)
    count = sum(p.numel() for p in predictor.parameters())
    if count != recipes[model]["architecture"]["actual_parameter_count"]:
        raise ValueError("Frozen predictor parameter count differs")
    return predictor.eval()


def fit_job(bundle_root: Path, inputs: Path, run_dir: Path, job: dict, claim: dict,
            manifest: dict, runtime: dict) -> dict:
    """One production new-seed fit/calibration; never admits evaluation arrays."""
    job_id = safe_job_id(job["id"])
    if job.get("model") not in NEURAL or job.get("seed") not in NEW_SEEDS:
        raise ValueError("Only an additional E2 seed can be fitted")
    if (run_dir / "all_fits_remote_ack.json").exists() or any((run_dir / "jobs").glob("*/evaluation.json")):
        raise ValueError("Fitting is closed permanently after the extension freeze/evaluation")
    directory = run_dir / "jobs" / job_id
    previous = completed_fit(run_dir, job_id, run_id=manifest["run_id"],
        plan_fingerprint=manifest["plan_fingerprint"])
    if previous is not None:
        if (previous.get("model"), previous.get("seed")) != (job["model"], job["seed"]):
            raise ValueError("Completed fit has a different model or seed")
        return previous
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Partial fit/calibration is nonresumable; preserve it for reconciliation")
    _, code = package_paths(bundle_root)
    inventory, recipes, roots = source_inventory(bundle_root)
    arrays, scaler_path = input_paths(inputs)
    scaler_record = read(scaler_path)
    train = prepared_training(code, arrays, scaler_record)
    trial, epochs = recipe_epochs(code, recipes, job["model"])
    if (job.get("trial") != trial or job.get("fixed_epochs") != epochs
            or job.get("parameter_count") != recipes[job["model"]]["architecture"]["actual_parameter_count"]):
        raise ValueError("The registered job differs from the frozen trial, phases or parameter count")
    fitter = original_module(code, "e2_reduced_fit")
    source = {"scope": "posterior E2 four-seed extension", "run_id": manifest["run_id"],
        "job_id": job_id, "plan_fingerprint": manifest["plan_fingerprint"],
        "reservation_id": claim["reservation_id"], "claim_sha256": hashlib.sha256(canonical(claim)).hexdigest(),
        "scientific_source_commit": inventory["scientific_source_commit"],
        "constructor_sha256": original_module(code, "e2_standalone_code").CONSTRUCTOR_SHA256,
        "arrays_sha256": NPZ_SHA256, "frozen_recipe_sha256": FROZEN_SHA256,
        "epoch_register_sha256": REGISTER_SHA256, "author_files_verified": 76}
    # Check the unchanged writer's strict type contract before the fitter can
    # publish request.json or create a phase attempt.
    original_module(code, "e2_amended_pilot_artifacts")._canonical({
        "source": source, "resources": runtime,
        "data": {"train": train.record(), "validation": None}})
    # The parent separately enforces its remaining whole-worker reservation.
    phase_limit = 1800.0
    result = fitter.fit_reduced_job(job["model"], trial, job["seed"], train, None,
        directory / "fit", fixed_epochs=epochs, source_roots=roots,
        device="cuda:0", source_identity=source, runtime_identity=runtime,
        fit_wall_seconds=phase_limit, allow_cpu_for_tests=False)
    checkpoint = directory / "fit" / result["selected_checkpoint"]["path"]
    verify_file(checkpoint, result["selected_checkpoint"]["sha256"])
    predictor = load_predictor(code, recipes, roots, job["model"], job["seed"], checkpoint, scaler_record)
    x, target, dates = cohort(arrays, "calibration")
    production = original_module(code, "e2_published_pilot_training")
    prediction = production.predict_pilot(predictor, x, batch_size=1, device="cuda:0").numpy() * train.scaler.target_scale
    analysis = original_module(code, "e2_reduced_final_analysis")
    calibrator = analysis.residual_calibrator(prediction, target, dates=dates, predictor_sha256=sha(checkpoint))
    immutable_json(directory / "scaler.json", scaler_record)
    immutable_json(directory / "calibrator.json", calibrator)
    immutable_json(directory / "calibration_predictions.json", {"model": job["model"],
        "seed": job["seed"], "dates": dates, "prediction_original": prediction.tolist(),
        "target_original": target.tolist(), "checkpoint_sha256": sha(checkpoint),
        "same_predictor_as_final": True, "used_for_HPO": False})
    artifacts = {p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob("*")
        if p.is_file() and not p.name.endswith((".lock", ".log", ".pending", ".tmp"))}
    receipt = {"version": VERSION, "status": "completed", "phase": "fit", "job_id": job_id,
        "model": job["model"], "seed": job["seed"], "run_id": manifest["run_id"],
        "plan_fingerprint": manifest["plan_fingerprint"], "claim_sha256": hashlib.sha256(canonical(claim)).hexdigest(),
        "reservation_id": claim["reservation_id"], "training_examples": 655,
        "calibration_examples": 53, "training_signature": TRAIN_SIGNATURE,
        "fixed_phase_epochs": epochs, "trial": trial,
        "checkpoint": checkpoint.relative_to(directory).as_posix(), "checkpoint_sha256": sha(checkpoint),
        "scaler": "scaler.json", "scaler_sha256": sha(directory / "scaler.json"),
        "calibrator": "calibrator.json", "calibrator_sha256": sha(directory / "calibrator.json"),
        "calibration_predictions": "calibration_predictions.json",
        "calibration_predictions_sha256": sha(directory / "calibration_predictions.json"),
        "runtime_identity": runtime, "source_identity": source, "artifacts_sha256": artifacts,
        "artifacts": [{"path": (directory / p).relative_to(run_dir).as_posix(),
                       "sha256": checksum, "bytes": (directory / p).stat().st_size}
                      for p, checksum in sorted(artifacts.items())],
        "evaluation_arrays_materialized": False, "HPO_repeated": False,
        "old_weights_reused": False, "optimizer_resume": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    immutable_json(directory / "fit_receipt.json", receipt)
    return receipt


def freeze_gate(run_dir: Path, manifest: dict) -> dict:
    path = run_dir / "all_fits_remote_ack.json"
    gate = read(path)
    if gate.get("version") != VERSION or any(gate.get(k) != manifest.get(k) for k in ("run_id", "plan_fingerprint")):
        raise ValueError("The acknowledged extension freeze belongs to another run/plan")
    rows = gate.get("fits", [])
    indexed = {r.get("job_id"): r for r in rows}
    if len(rows) != 52 or set(indexed) != {j["id"] for j in manifest["jobs"]}:
        raise ValueError("All 52 fit/calibration artifacts need remote acknowledgement before evaluation")
    for job in manifest["jobs"]:
        receipt = completed_fit(run_dir, job["id"], run_id=manifest["run_id"],
            plan_fingerprint=manifest["plan_fingerprint"])
        if receipt is None:
            raise ValueError("A frozen fit is missing")
        if ((receipt.get("model"), receipt.get("seed")) != (job["model"], job["seed"])
                or receipt.get("trial") != job.get("trial")
                or receipt.get("fixed_phase_epochs") != job.get("fixed_epochs")):
            raise ValueError("A frozen fit differs from its registered coordinate or recipe")
        row = indexed[job["id"]]
        if row.get("fit_receipt_sha256") != sha(run_dir / "jobs" / job["id"] / "fit_receipt.json"):
            raise ValueError("An acknowledged fit receipt changed")
        ack_path = run_dir / "remote_acks" / (job["id"] + ".json")
        verify_file(ack_path, row.get("remote_ack_sha256", ""))
        ack = read(ack_path)
        if (ack.get("fit_receipt_sha256") != row["fit_receipt_sha256"]
                or ack.get("job_id") != job["id"] or ack.get("status") != "acknowledged"
                or not re.fullmatch(r"[0-9a-f]{40}", ack.get("ack_commit", ""))):
            raise ValueError("Remote acknowledgement does not bind this frozen fit")
    return gate


def evaluate_job(bundle_root: Path, inputs: Path, run_dir: Path, job: dict,
                 claim: dict, manifest: dict, runtime: dict) -> dict:
    gate = freeze_gate(run_dir, manifest)  # Before any evaluation member access.
    job_id = safe_job_id(job["id"])
    receipt = completed_fit(run_dir, job_id, run_id=manifest["run_id"],
        plan_fingerprint=manifest["plan_fingerprint"])
    if receipt is None:
        raise ValueError("A completed acknowledged predictor is required")
    destination = run_dir / "jobs" / job_id / "evaluation.json"
    if destination.exists():
        existing = read(destination)
        if existing.get("checkpoint_sha256") != receipt["checkpoint_sha256"] or existing.get("plan_fingerprint") != manifest["plan_fingerprint"]:
            raise ValueError("Completed evaluation predictor/plan differs")
        return existing
    _, code = package_paths(bundle_root)
    _, recipes, roots = source_inventory(bundle_root)
    arrays, scaler_path = input_paths(inputs)
    directory = run_dir / "jobs" / job_id
    scaler = read(scaler_path)
    if read(directory / receipt["scaler"]) != scaler:
        raise ValueError("Frozen predictor scaler changed")
    predictor = load_predictor(code, recipes, roots, job["model"], job["seed"],
        directory / receipt["checkpoint"], scaler)
    x, target, dates = cohort(arrays, "evaluation")
    production = original_module(code, "e2_published_pilot_training")
    prediction = production.predict_pilot(predictor, x, batch_size=1, device="cuda:0").numpy() * scaler["target_scale"]
    value = {"version": VERSION, "status": "completed", "phase": "evaluate", "job_id": job_id,
        "model": job["model"], "seed": job["seed"], "run_id": manifest["run_id"],
        "plan_fingerprint": manifest["plan_fingerprint"], "reservation_id": claim["reservation_id"],
        "claim_sha256": hashlib.sha256(canonical(claim)).hexdigest(),
        "fit_receipt_sha256": sha(directory / "fit_receipt.json"),
        "checkpoint_sha256": receipt["checkpoint_sha256"], "scaler_sha256": receipt["scaler_sha256"],
        "calibrator_sha256": receipt["calibrator_sha256"],
        "all_fits_remote_ack_sha256": sha(run_dir / "all_fits_remote_ack.json"),
        "dates": dates, "prediction_original": prediction.tolist(), "target_original": target.tolist(),
        "n_dates": 608, "batch_size": 1, "inverse_scaling_count": 1,
        "retrospective_known_calendar": True, "used_for_HPO": False,
        "runtime_identity": runtime, "new_fit_performed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    prediction_path = directory / "test_predictions.json"
    immutable_json(prediction_path, value)
    value["artifacts"] = [{"path": prediction_path.relative_to(run_dir).as_posix(),
        "sha256": sha(prediction_path), "bytes": prediction_path.stat().st_size}]
    immutable_json(destination, value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "evaluate", "aggregate"), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--claim-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if not sys.dont_write_bytecode or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ValueError("Launch workers with Python -B and PYTHONDONTWRITEBYTECODE=1")
    if args.phase == "aggregate":
        raise ValueError("Aggregation is owned by the separately claim-bound aggregate adapter")
    safe_job_id(args.job_id)
    manifest, claim, job = claim_context(args.run_dir, args.claim_path, args.job_id, args.phase)
    runtime = runtime_identity(args.device, args.bundle_root)
    function = fit_job if args.phase == "fit" else evaluate_job
    result = function(args.bundle_root, args.inputs, args.run_dir, job, claim, manifest, runtime)
    print(json.dumps({"status": result["status"], "phase": args.phase,
        "job_id": args.job_id, "new_seed": job["seed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("E2 extension worker refused: " + str(error), file=sys.stderr)
        raise SystemExit(2)
