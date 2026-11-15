"""Portable entry point for the D02 source ZIP.

Layout: PACKAGE/{run.py,authors/<model>/<path>,code/{scripts,experiments,
book_e2/_static,results/eda_development}}. Original scientific files remain
unchanged. The default executes real CPU forwards on artificial inputs only.
Inventory/verification uses only Python's standard library. A local CUDA refit
is opt-in and uses the original durable fitting API, not a campaign controller.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys

MODEL_ORDER = (
    "FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
    "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
    "MLP16", "LSTM32", "GRU32",
)
AUTHOR_MODELS = frozenset(("TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST", "iTransformer"))
LICENSES = {"TSMixer": "Apache-2.0", "TiDE": "Apache-2.0", "TimeMixer": "Apache-2.0",
            "TimesNet": "MIT", "PatchTST": "Apache-2.0", "iTransformer": "MIT"}
SCIENTIFIC_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
FROZEN_INPUTS = "book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json"
FROZEN_SHA256 = "da004139cfc4663ce1d9a13c88a422df8865b53f6448fbdf015da0244f9f51cf"
DESIGN = "experiments/e2_completion_20261106/design.json"
REGISTER = "book_e2/_static/e2_data_training_uncertainty_current_v1/model_training_register.csv"
REGISTER_SHA256 = "e0c13b18cc20b161fc1f7abe8ff8ae3b1a42d5b6ea07916e45e098d99c66c072"
METADATA_SHA256 = {
    "experiments/e2_completion_20261106/design.json": "58f54626d0765ee7a97dfd3cbe2dec932c74a28d2f47b5bda519fb0f5777a7f0",
    "experiments/e2_published_architectures_20261112/architecture_policy.json": "a1c9440046af34c2b70bef9c05229dff2c6d863aa06ae191ed02c2de6d8bf0f2",
    "book_e2/_static/search_protocol_20261109/protocol.json": "5195337af13f4193da16bf4499c8396586c621d4ab084a5b296a297f2141edeb",
    "book_e2/_static/current_implementation_20261108/source_design.json": "ce0a76304186fabf922f8e1351ea2e9d44db49f1714f44cd1fe2df04baeb3dbe",
}
CONSTRUCTOR_SHA256 = {
    "scripts/e2_reduced_fit.py": "64e83d0a2a39920ae470252b8eb81c8f78429817fed978bd22f64866fdfb8f5f",
    "scripts/e2_published_registry.py": "0704784aa0bb2887908d12ad12345094c8f6a707918aa695dbc11cda88092fec",
    "scripts/e2_training_models.py": "46be0644e0ed9d966163b17f171092e81b6a82dd734efc7954d6104f5ee5360b",
    "scripts/e2_recipe_models.py": "47380cc87e2d4b4221ee4fc97da3286990470b3304dd1d9ef990ace51c338b75",
    "scripts/e2_published_periodic.py": "a0ead3e4c394cb7d832e50e2821e0c73a3bb09652ed5e977a87f119941517af9",
    "scripts/e2_published_patchtst.py": "99aae30754cf33fe12883b4af2f4f957cfc32ec9f4bcc2eb71dfb34b39974d5f",
    "scripts/e2_published_sae.py": "0436d73f9724bb065dc3642f18218cb35cda5e9abfa4de5090f8c5d69eb6c1aa",
    "scripts/e2_published_tide.py": "5ba7a01a30c52793b566a44525e4f5ccfaf0254e9c6cdd2667c56a34319c747a",
    "scripts/e2_final_models_mlp.py": "beeada2f6754f0ca3ff1304ec090a7259a774cc7ea611fa912428cf9210f6c68",
    "scripts/e2_final_models_attention.py": "a3f1a2a7534142f505fbf36bddf92e0abf4f76d89ac1a1fb61874559e5aafa07",
    "scripts/dl_models.py": "e24110b57aeee201e1e9ab53519eb6b05967927ac54102fe23d3908c3fc57d69",
    "scripts/dl_models_gru_lstm_tcn.py": "c11af1229fec43dd7da01a21e349dc2b109d9d61d977f6df3d01cd7da736b35d",
}


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
        raise ValueError("Unsafe bundled relative path: " + relative)
    path = root.joinpath(*parts.parts)
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Missing regular bundled file: " + str(path))
    return path


def layout(package_root):
    root = Path(package_root).resolve()
    code = root / "code"
    if not (code / "scripts/e2_reduced_fit.py").is_file():
        raise ValueError("Expected extracted D02 package with code/scripts and authors directories; use --package-root PATH")
    return root, code


def fetch_sources(root, code):
    """Explicit optional public HTTPS acquisition, never invoked by default."""
    sys.path.insert(0, str(code / "scripts"))
    import download_e2_completion_sources as acquisition
    design = json.loads(checked_path(code, DESIGN).read_text(encoding="utf-8"))
    design["source_locations"] = {k: v for k, v in design["source_locations"].items() if k in AUTHOR_MODELS}
    design["author_sources"] = [r for r in design["author_sources"] if r["model"] in AUTHOR_MODELS]
    acquisition.acquire(design, root / "authors")


def inventory(root, code):
    """Verify original constructors, frozen recipes and every required author file."""
    for relative, expected in METADATA_SHA256.items():
        if file_sha(checked_path(code, relative)) != expected:
            raise ValueError("Original constructor metadata changed: " + relative)
    for relative, expected in CONSTRUCTOR_SHA256.items():
        if file_sha(checked_path(code, relative)) != expected:
            raise ValueError("Original scientific constructor changed: " + relative)
    frozen = checked_path(code, FROZEN_INPUTS)
    if file_sha(frozen) != FROZEN_SHA256:
        raise ValueError("Frozen recipe input JSON changed")
    entries = json.loads(frozen.read_text(encoding="utf-8"))
    rows = {r["model"]: r for r in entries}
    if len(entries) != 13 or set(rows) != set(MODEL_ORDER):
        raise ValueError("Exactly thirteen frozen neural recipes are required")
    design = json.loads(checked_path(code, DESIGN).read_text(encoding="utf-8"))
    author_rows = [r for r in design["author_sources"] if r["model"] in AUTHOR_MODELS]
    if len(author_rows) != 76 or {r["model"] for r in author_rows} != AUTHOR_MODELS:
        raise ValueError("The six declared author subsets must contain exactly 76 files")
    counts = {}
    for row in author_rows:
        relative = row["model"] + "/" + row["path"]
        path = checked_path(root / "authors", relative)
        if path.stat().st_size != row["bytes"] or file_sha(path) != row["sha256"]:
            raise ValueError("Pinned author bytes differ: " + relative)
        counts[row["model"]] = counts.get(row["model"], 0) + 1
    for name in AUTHOR_MODELS:
        if not any(r["model"] == name and r["path"] == "LICENSE" for r in author_rows):
            raise ValueError("Pinned author license missing: " + name)
    roots = {name: str((root / "authors" / name).resolve()) for name in AUTHOR_MODELS}
    report = {
        "version": "e2-standalone-code-v1", "status": "verified",
        "scientific_source_commit": SCIENTIFIC_COMMIT,
        "original_constructor_files_verified": len(CONSTRUCTOR_SHA256),
        "original_constructor_metadata_files_verified": len(METADATA_SHA256),
        "author_files_verified": len(author_rows),
        "author_subsets": [{"model": n, "files": counts[n], "license": LICENSES[n]}
                           for n in sorted(AUTHOR_MODELS)],
        "models": [{"model": n, "recipe_id": rows[n]["trial"]["recipe_id"],
                    "trial_id": rows[n]["trial"]["trial_id"],
                    "parameters": rows[n]["architecture"]["actual_parameter_count"]}
                   for n in MODEL_ORDER],
        "scope": {"market_data_read": False, "trained_weights_loaded": False,
                  "new_model_fits": 0, "predictive_accuracy_evaluated": False,
                  "absolute_historical_result_paths_used": False},
    }
    return report, rows, roots


def smoke_cpu(code, rows, roots, chosen):
    import torch
    sys.path.insert(0, str(code / "scripts"))
    from e2_reduced_fit import _builder
    torch.set_num_threads(2)
    fixture = torch.linspace(-0.5, 0.5, 35, dtype=torch.float32).reshape(1, 7, 5)
    results = []
    for name in chosen:
        row = rows[name]
        model = _builder(name, row["trial"], 271801, roots).cpu().eval()
        count = sum(p.numel() for p in model.parameters())
        if count != row["architecture"]["actual_parameter_count"]:
            raise ValueError("Frozen parameter count changed: " + name)
        with torch.inference_mode():
            prediction = model(fixture.clone())
        if tuple(prediction.shape) != (1,) or not bool(torch.isfinite(prediction).all()) or bool((prediction < 0).any()):
            raise ValueError("Real model did not return one finite nonnegative scalar: " + name)
        results.append({"model": name, "parameters": count, "output_shape": list(prediction.shape),
                        "finite_nonnegative": True, "real_forward_executed": True})
        del model
    return {"status": "passed", "device": "cpu", "torch": torch.__version__,
            "python": platform.python_version(), "threads": 2,
            "fixture": "fixed linspace tensor [1,7,5], not market observations",
            "models": results, "optimizer_created": False, "training_performed": False,
            "trained_checkpoint_loaded": False, "scientific_benchmark_reproduced": False}


def train_local_cuda(root, code, rows, roots, args):
    """Run one real local refit through the unchanged production CUDA API."""
    if args.model is None or args.output is None:
        raise ValueError("--train requires --model NAME and --output NEW_DIRECTORY")
    if args.device != "cuda:0":
        raise ValueError("Original production refitting requires cuda:0. CPU is available for --smoke-cpu; its test-only fit path is not relabelled as real-data production training.")
    if os.name != "posix":
        raise ValueError("Original durable refitting requires POSIX file locks. Use Linux/WSL2 with CUDA for --train; inventory and CPU forwards do not use that fitting path.")
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Local refit requires a new output directory, retaining any earlier attempt")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    sys.path.insert(0, str(code / "scripts"))
    from e2_cuda_training import validate_cuda_device
    from e2_development_data import load_development
    from e2_reduced_final import final_training_split, _calibration_split
    from e2_reduced_fit import fit_reduced_job, write_json
    validate_cuda_device(args.device)
    torch.set_num_threads(2)
    register_path = checked_path(code, REGISTER)
    if file_sha(register_path) != REGISTER_SHA256:
        raise ValueError("Exact winner/phase-epoch register changed")
    with register_path.open(encoding="utf-8", newline="") as stream:
        register = list(csv.DictReader(stream))
    registered = [r for r in register if r["model"] == args.model]
    if len(registered) != 1 or registered[0]["trial_id"] != rows[args.model]["trial"]["trial_id"]:
        raise ValueError("Exact winner and phase-epoch register required")
    selected = registered[0]
    epochs = {phase: int(selected[field]) for phase, field in (
        ("ae1", "fixed_ae1_epochs"), ("ae2", "fixed_ae2_epochs"),
        ("supervised", "fixed_supervised_epochs")) if selected[field] != ""}
    panel, admission = load_development(code, "2024-12-30")
    train = final_training_split(panel)
    calibration = _calibration_split(panel, train.scaler)
    if len(train.x) != 655 or len(calibration.x) != 53:
        raise ValueError("Frozen final train/calibration support changed")
    props = torch.cuda.get_device_properties(0)
    source_identity = {"scope": "local standalone reproduction, not original campaign",
        "scientific_commit": SCIENTIFIC_COMMIT, "constructor_sha256": CONSTRUCTOR_SHA256,
        "frozen_recipe_sha256": FROZEN_SHA256, "author_files_verified": 76}
    runtime = {"device": "cuda:0", "device_name": props.name,
        "total_memory_bytes": props.total_memory, "torch": torch.__version__,
        "cuda": torch.version.cuda, "python": platform.python_version(), "cpu_threads": 2}
    # This is the original refitting implementation. No campaign controller,
    # historical budget ledger, HPO, or 2025+ observations are accessed.
    result = fit_reduced_job(args.model, rows[args.model]["trial"], args.seed, train, None,
        output / "fit", fixed_epochs=epochs, source_roots=roots, device="cuda:0",
        source_identity=source_identity, runtime_identity=runtime,
        fit_wall_seconds=args.phase_wall_seconds, allow_cpu_for_tests=False)
    report = {"status": result["status"], "model": args.model, "seed": args.seed,
        "output": str(output), "phase_epochs": epochs, "train_examples": 655,
        "calibration_examples_held_out": 53, "calibration_used_for_fitting": False,
        "calibration_predictions_produced": False, "test_data_read": False,
        "local_training_performed": True, "original_campaign_modified": False,
        "original_weights_reproduced_or_compared": False,
        "checkpoint": str(output / "fit" / result["selected_checkpoint"]["path"])}
    write_json(output / "development_admission.json", admission)
    write_json(output / "standalone_refit.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inventory", dest="mode", action="store_const", const="inventory")
    modes.add_argument("--verify-offline", dest="mode", action="store_const", const="inventory")
    modes.add_argument("--smoke-cpu", dest="mode", action="store_const", const="smoke")
    modes.add_argument("--train", dest="mode", action="store_const", const="train")
    parser.set_defaults(mode="smoke")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--fetch-sources", action="store_true", help="Explicit optional public HTTPS acquisition of missing pinned sources")
    parser.add_argument("--model", choices=MODEL_ORDER, help="One model; default smoke checks all thirteen")
    parser.add_argument("--seed", type=int, default=271801)
    parser.add_argument("--device", default="cuda:0", help="Only used for opt-in production refitting")
    parser.add_argument("--phase-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, help="Fresh report file for smoke/inventory, or fresh directory for --train")
    args = parser.parse_args(argv)
    try:
        if args.seed < 0:
            raise ValueError("Seed must be nonnegative")
        root, code = layout(args.package_root)
        if args.fetch_sources:
            fetch_sources(root, code)
        verified, rows, roots = inventory(root, code)
        if args.mode == "train":
            report = train_local_cuda(root, code, rows, roots, args)
        else:
            report = verified
            if args.mode == "smoke":
                report["smoke_cpu"] = smoke_cpu(code, rows, roots, (args.model,) if args.model else MODEL_ORDER)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
