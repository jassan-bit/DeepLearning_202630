"""Synthetic 17-system Colab/CPU preflight; never launches a real-data campaign.

No market file, pretrained weight, optimizer step, HPO or test metric is used.
The 13 E2 recipes use the immutable D02 author/code package. The historical E3
profile is InnovationModel(additive_positive_head), not a different candidate.
Every worker has its own Python process and a retained receipt, even on failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PACKAGE_COMMIT = "c4a4a1e68e9844cdfb81de0075177ced0cd96c07"
PACKAGE_PATH = "book_e2/_static/entregables_independientes_v2_20261115/02_codigo_modelos.zip"
PACKAGE_SHA256 = "5423c85bf220729c826e47a1c37d79a49ce2ba8b0e8f764c47ab9c0796b4bcae"
PACKAGE_URL = f"https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/{PACKAGE_COMMIT}/{PACKAGE_PATH}"
NEURAL = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor", "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer", "MLP16", "LSTM32", "GRU32")
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
CANDIDATE = "CNN-LSTM-atencion-aditiva E3"
FAMILY = (*NEURAL, *CLASSICAL, CANDIDATE)
CANDIDATE_PROFILE = {"class": "src.e3_innovation_model.InnovationModel", "variant": "additive_positive_head", "parameters": 1210,
    "synthetic_only_configuration": {"u_scale": 0.01, "y_scale": 0.02, "seed": 271801},
    "future_real_scaler_must_be_train_only": True}
APPROVED_DEPENDENCIES = {"numpy": "2.3.5", "scipy": "1.17.0", "pandas": "2.2.3", "statsmodels": "0.14.5", "einops": "0.8.1"}
LOCKED_SOURCE_FILES = {"src/original_model.py", "src/e3_development.py", "src/e3_innovation_model.py",
    "scripts/e2_standalone_code.py", "scripts/e2_temporal_controls.py",
    "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json",
    "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification/quadrature_k128.json.sha256"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def source_gate(root=ROOT):
    lock = json.loads((HERE / "colab_source_lock.json").read_text())
    if not isinstance(lock.get("files"), dict) or set(lock["files"]) != LOCKED_SOURCE_FILES:
        return {"passed": False, "files": [], "error": "The exact seven reviewed source files must be bound"}
    checks = []
    for relative, expected in lock["files"].items():
        path = Path(root) / relative
        passed = path.is_file() and not path.is_symlink() and sha(path) == expected
        checks.append({"path": relative, "expected_sha256": expected, "verified": passed})
    return {"passed": all(r["verified"] for r in checks), "reviewed_source_commit": lock["reviewed_source_commit"],
        "files": checks, "lock_sha256": sha(HERE / "colab_source_lock.json")}


def environment(device):
    result = {"python": platform.python_version(), "platform": platform.platform(), "requested_device": device,
        "packages": {}, "approved": True, "blockers": []}
    for name, expected in APPROVED_DEPENDENCIES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        result["packages"][name] = actual
        if actual != expected:
            result["blockers"].append(f"{name}: instalado {actual}; perfil revisado {expected}")
    try:
        import torch
        result.update(torch=torch.__version__, cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
            cudnn_version=torch.backends.cudnn.version(), cpu_count=os.cpu_count())
        approved_torch = "2.8.0+cu126" if device.startswith("cuda") else "2.8.0+cpu"
        if torch.__version__ != approved_torch:
            result["blockers"].append(f"Torch {torch.__version__}; perfil revisado para {device}: {approved_torch}. No se sustituye automáticamente el runtime CUDA.")
        if device.startswith("cuda"):
            if not torch.cuda.is_available() or torch.version.cuda != "12.6":
                result["blockers"].append("GPU CUDA 12.6 disponible requerida para este perfil; la ejecución GPU todavía no se verificó.")
            else:
                prop = torch.cuda.get_device_properties(0)
                result["GPU"] = {"name": prop.name, "total_memory_bytes": prop.total_memory,
                    "capability": list(torch.cuda.get_device_capability(0)), "device_count": torch.cuda.device_count()}
                try:
                    result["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True, timeout=10).strip()
                except (OSError, subprocess.SubprocessError) as exc:
                    result["nvidia_smi_diagnostic"] = str(exc)
        if not sys.version_info[:2] == (3, 12):
            result["blockers"].append("Perfil revisado para Python 3.12; no se acredita compatibilidad de otra versión.")
    except ImportError as exc:
        result["blockers"].append("Torch no disponible: " + str(exc))
    result["approved"] = not result["blockers"]
    return result


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Extraction requires a new destination; retain prior files")
    with zipfile.ZipFile(archive) as source:
        if sum(x.file_size for x in source.infolist()) > 100_000_000:
            raise ValueError("Unexpected package expansion")
        for member in source.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.filename or ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Unsafe package member")
        destination.mkdir(parents=True)
        source.extractall(destination)


def acquire_package(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "02_codigo_modelos.zip"
    with urllib.request.urlopen(PACKAGE_URL, timeout=60) as response, archive.open("xb") as stream:
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > 10_000_000:
                raise ValueError("Unexpected source package size")
            stream.write(chunk)
    if archive.stat().st_size != 7839993 or sha(archive) != PACKAGE_SHA256:
        raise ValueError("Published D02 archive bytes differ; retain the failed download")
    safe_extract(archive, destination / "extracted")
    candidates = list((destination / "extracted").rglob("code/scripts/e2_standalone_code.py"))
    if len(candidates) != 1:
        raise ValueError("Exactly one D02 code tree required")
    return candidates[0].parents[2]


def portable_inventory(package):
    path = Path(package) / "code/scripts/e2_standalone_code.py"
    # Inventory code itself is bound by the reviewed repository's matching bytes.
    if sha(path) != sha(ROOT / "scripts/e2_standalone_code.py"):
        raise ValueError("D02 inventory entrypoint differs from reviewed source")
    spec = importlib.util.spec_from_file_location("e3_preflight_d02", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report, rows, roots = module.inventory(Path(package).resolve(), (Path(package) / "code").resolve())
    if tuple(rows) != NEURAL and set(rows) != set(NEURAL):
        raise ValueError("Exact thirteen-source family required")
    return report, rows, roots


def check_vector(output, batch, torch):
    if tuple(output.shape) != (batch,) or not bool(torch.isfinite(output).all()) or bool((output < 0).any()):
        raise ValueError("Expected one finite nonnegative sigma7 prediction per artificial window")


def neural_worker(name, package, device, output):
    import torch
    report, rows, roots = portable_inventory(package)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(Path(package) / "code/scripts"))
    from e2_reduced_fit import _builder
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(271801)
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(271801)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    # Explicit deterministic profile; no optimizer or mixed precision is used.
    torch.use_deterministic_algorithms(True)
    fixture = torch.linspace(-.5, .5, 4 * 7 * 5, dtype=torch.float32, device=device).reshape(4, 7, 5)
    A = torch.linspace(.00001, .00004, 4, dtype=torch.float64, device=device)
    if name == CANDIDATE:
        from src.e3_innovation_model import InnovationModel
        def build():
            return InnovationModel("additive_positive_head", u_scale=.01, y_scale=.02, seed=271801).to(device)
        expected, configuration = 1210, CANDIDATE_PROFILE
        def forward(model):
            return model(fixture, A)
    else:
        row = rows[name]
        def build():
            return _builder(name, row["trial"], 271801, roots).to(device)
        expected = row["architecture"]["actual_parameter_count"]
        configuration = {"trial": row["trial"], "frozen_architecture": row["architecture"]}
        def forward(model):
            return model(fixture)
    started = time.perf_counter()
    model = build()
    count = sum(p.numel() for p in model.parameters())
    if count != expected:
        raise ValueError(f"Frozen parameter count differs for {name}: {count} != {expected}")
    model.train()
    prediction = forward(model)
    check_vector(prediction, 4, torch)
    prediction.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        raise ValueError("Missing or nonfinite synthetic gradient")
    model.eval()
    with torch.inference_mode():
        before = forward(model).detach().cpu()
    checkpoint = output / "synthetic_state.pt"
    torch.save({"state_dict": model.state_dict(), "model": name, "configuration": configuration,
                "synthetic_untrained": True}, checkpoint)
    restored = build()
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved["model"] != name or not saved["synthetic_untrained"]:
        raise ValueError("Checkpoint identity differs")
    restored.load_state_dict(saved["state_dict"], strict=True)
    restored.eval()
    with torch.inference_mode():
        after = forward(restored).detach().cpu()
    if not torch.equal(before, after):
        raise ValueError("Same-device checkpoint round trip changed synthetic output")
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return {"model": name, "status": "passed", "device": device, "parameters": count,
        "fixture_shape": [4, 7, 5], "output_shape": list(after.shape), "finite_nonnegative": True,
        "input_contract": "artificial normalized OHLCV features; not raw market bars",
        "output_units": "sigma7_original_synthetic_scale" if name == CANDIDATE else "sigma7_divided_by_training_target_scale; not compared across models",
        "synthetic_backward_executed": True, "finite_gradient_tensors": len(grads),
        "optimizer_steps": 0, "synthetic_only": True, "real_data_model_fits": 0,
        "checkpoint_sha256": sha(checkpoint), "checkpoint_reload_identical": True,
        "elapsed_seconds": time.perf_counter() - started,
        "GPU_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None,
        "configuration": configuration, "author_files_verified": report["author_files_verified"]}


def classical_worker(name, output):
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    import e2_temporal_controls as controls
    dates = pd.date_range("2000-01-01", periods=160, freq="D")
    axis = np.arange(160, dtype=float)
    values = .02 + .001 * np.sin(axis / 5) + .0004 * np.cos(axis / 9) + .0001 * np.sin(axis * 1.7)
    history = pd.Series(values[:159], index=dates[:159])
    target = dates[159]
    if name == "ARIMA":
        model = controls.FrozenARIMA.fit(history, target, (2, 0, 1), source_commit="synthetic_preflight_only_not_market_fit")
        while model.current < target - controls.DAY:
            when = model.current + controls.DAY
            model.observe(when, float(history.loc[when]))
        try:
            model.observe(target, .99)
        except ValueError:
            causal_guard = True
        else:
            raise ValueError("ARIMA accepted target observation before forecast")
        prediction = model.forecast(target)
        checkpoint = output / "synthetic_arima.pkl"
        model._result.save(checkpoint)
        from statsmodels.tsa.arima.model import ARIMAResults
        restored = ARIMAResults.load(checkpoint)
        roundtrip = max(0., float(np.asarray(restored.forecast(1)).item()) * model.scale)
        if roundtrip != prediction.value:
            raise ValueError("Synthetic ARIMA checkpoint round trip differs")
        extra = {"synthetic_ARIMA_fits": 1, "forecast_before_observe_guard_passed": causal_guard,
                 "parameter_fits_on_market_data": 0, "checkpoint_sha256": sha(checkpoint), "checkpoint_reload_identical": True,
                 "order": [2, 0, 1], "forecast_origin": target.isoformat(), "fit_receipt": model.receipt}
    else:
        period = 1 if name == "Persistence" else 7
        prediction = controls.seasonal_naive(history, target, period)
        expected = float(history.loc[target - period * controls.DAY])
        if prediction.value != expected:
            raise ValueError("Naive did not use its exact past lag")
        checkpoint = output / "synthetic_lag_state.json"
        write_new(checkpoint, {"period": period, "target": target.isoformat(), "source": (target - period * controls.DAY).isoformat(), "value": expected})
        restored = json.loads(checkpoint.read_text())
        if restored["value"] != prediction.value:
            raise ValueError("Naive state receipt round trip differs")
        extra = {"synthetic_ARIMA_fits": 0, "parameter_fits_on_market_data": 0,
            "causal_exact_lag_verified": True, "checkpoint_sha256": sha(checkpoint), "checkpoint_reload_identical": True}
    if not math.isfinite(prediction.value) or prediction.value < 0:
        raise ValueError("Invalid synthetic classical prediction")
    return {"model": name, "status": "passed", "device": "cpu_classical", "fixture": "artificial positive dated volatility history; no market data",
        "output_shape": [1], "finite_nonnegative": True, "synthetic_backward_executed": False,
        "optimizer_steps": 0, "real_data_model_fits": 0, "synthetic_only": True, **extra}


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Explicit UTC timezone required")
    return parsed.astimezone(timezone.utc)


def campaign_gate(manifest=None, root=ROOT):
    """Fail closed on metadata, identities and verifiable future freeze ordering.

    This returns a readiness receipt only. It never dispatches training and a
    passing technical gate would not establish novelty or scientific superiority.
    """
    blockers = []
    if manifest is not None and not isinstance(manifest, dict):
        return {"allowed": False, "blockers": ["El manifiesto final debe ser un objeto JSON."],
                "real_training_dispatched": False, "R1_R3_verified": False}
    if manifest is None:
        return {"allowed": False, "blockers": ["No hay manifiesto final de campaña ni tag anotado de protocolo.",
            "Falta vincular al manifiesto final la admisión de los tres datasets; una admisión parcial no habilita la campaña.",
            "Faltan tratamiento común de 17 sistemas, presupuesto y potencia final de la batería."],
            "real_training_dispatched": False, "R1_R3_verified": False}
    root = Path(root).resolve()
    try:
        if manifest.get("candidate", {}).get("class") != CANDIDATE_PROFILE["class"] or manifest.get("candidate", {}).get("variant") != "additive_positive_head" or manifest.get("candidate", {}).get("parameters") != 1210:
            blockers.append("Identidad del candidato distinta del perfil histórico aditivo de 1210 parámetros.")
        if manifest.get("family") != list(FAMILY):
            blockers.append("La familia debe conservar exactamente los 17 sistemas declarados.")
        datasets = manifest.get("datasets", [])
        if not isinstance(datasets, list) or not all(isinstance(item, dict) for item in datasets):
            raise ValueError("Datasets must be a list of admission objects")
        if len(datasets) != 3 or len({x.get("id") for x in datasets}) != 3:
            blockers.append("Se requieren exactamente tres datasets adicionales distintos y admitidos.")
        for item in datasets:
            if item.get("status") != "admitted" or item.get("target") != "population_sigma7_log_close_returns_ddof0_unannualized" or item.get("horizon") != 1 or item.get("clock") not in ("UTC_calendar_day", "exchange_session_explicitly_homologated"):
                blockers.append("Dataset sin admisión o sin contrato explícito sigma7/h1: " + str(item.get("id")))
            for key in ("source_manifest", "license_evidence", "technical_admission", "exposure_log"):
                bound = item.get(key, {})
                p = (root / bound.get("path", "")).resolve()
                if not p.is_relative_to(root) or not p.is_file() or not bound.get("sha256") or sha(p) != bound["sha256"]:
                    blockers.append("Evidencia verificable faltante: " + str(item.get("id")) + "/" + key)
        for key in ("candidate_source_configuration", "common_treatment", "budget_17_systems", "power_individual_and_joint", "robustness_and_calibration"):
            bound = manifest.get(key, {})
            p = (root / bound.get("path", "")).resolve()
            if not p.is_relative_to(root) or not p.is_file() or not bound.get("sha256") or sha(p) != bound["sha256"]:
                blockers.append("Decisión final sin archivo/hash: " + key)
        # Final scientific decisions need typed content as well as a sealed hash.
        for key, required in {
            "common_treatment": {"same_information_sets_all_17": True, "train_only_scalers": True,
                "selection_before_evaluation": True, "calibration_policy_frozen": True},
            "budget_17_systems": {"all_17_systems_covered": True, "same_resource_and_search_policy": True,
                "failed_and_aborted_jobs_charged": True, "old_ledgers_reset": False},
            "power_individual_and_joint": {"prospective_for_this_campaign": True,
                "observed_effect_used_as_planning_alternative": False, "rivals": 16,
                "procedures_per_rival": 3, "full_battery_dimensioned": True},
        }.items():
            path = (root / manifest.get(key, {}).get("path", "")).resolve()
            if path.is_relative_to(root) and path.is_file():
                decision = json.loads(path.read_text())
                if any(decision.get(field) != value for field, value in required.items()):
                    blockers.append("Decisión final incompleta o incompatible: " + key)
                if key == "power_individual_and_joint" and (not isinstance(decision.get("minimum_individual_power"), (int, float)) or not .9 <= decision["minimum_individual_power"] <= 1 or not isinstance(decision.get("joint_power"), (int, float)) or not .9 <= decision["joint_power"] <= 1):
                    blockers.append("Potencia final individual y conjunta >=0.90 no verificada para la batería.")
        frozen = manifest.get("protocol_freeze", {})
        tag, path = frozen.get("annotated_git_tag", ""), frozen.get("manifest_path_in_tag", "")
        if not tag or not path or git("cat-file", "-t", "refs/tags/" + tag, root=root) != "tag":
            blockers.append("Falta tag Git anotado del protocolo final.")
        else:
            sealed = json.loads(git("show", f"refs/tags/{tag}:{path}", root=root))
            if sealed != manifest:
                blockers.append("El manifiesto no coincide con los bytes sellados por el tag.")
            tag_time = timestamp(git("for-each-ref", "--format=%(taggerdate:iso-strict)", "refs/tags/" + tag, root=root))
            declared = timestamp(frozen["frozen_at_utc"])
            first = timestamp(manifest["planned_first_real_fit_or_evaluation_utc"])
            now = datetime.now(timezone.utc)
            if tag_time > declared or not declared < first:
                blockers.append("Tag/congelamiento debe preceder al primer ajuste o evaluación real; no se acepta preregistro retroactivo.")
            if declared > now or first <= now:
                blockers.append("El congelamiento debe existir ahora y el primer ajuste/evaluación planificado debe seguir en el futuro.")
            if manifest.get("real_fits_or_evaluation_already_started") is not False:
                blockers.append("La campaña ya comenzó o no declara el estado previo a las salidas reales.")
    except (KeyError, ValueError, TypeError, AttributeError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        blockers.append("No se pudo verificar el protocolo final: " + str(exc))
    return {"allowed": not blockers, "blockers": blockers, "real_training_dispatched": False, "R1_R3_verified": False}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_gate()
    runtime = environment(args.device)
    try:
        commit = git("rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    gate = campaign_gate()
    result = {"schema_version": 1, "scope": "synthetic_preflight_not_confirmation", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_HEAD": commit, "preflight_script_sha256": sha(__file__), "source_gate": source,
        "environment": runtime, "candidate_profile": CANDIDATE_PROFILE, "family": list(FAMILY), "models": [],
        "campaign_guard": gate, "new_real_data_fits": 0, "new_market_predictions": 0, "new_hypothesis_tests": 0,
        "GPU_execution_verified": False, "R1_R3_verified": False}
    write_new(args.output / "hardware_and_source_receipt.json", result)
    if not source["passed"] or not runtime["approved"]:
        result["status"] = "blocked_before_synthetic_models"
        write_new(args.output / "preflight_receipt.json", result)
        print(json.dumps({"status": result["status"], "blockers": runtime["blockers"], "source_gate_passed": source["passed"], "receipt": str(args.output / "preflight_receipt.json")}, ensure_ascii=False))
        return 2
    try:
        package = args.package_root if args.package_root is not None else acquire_package(args.output / "source_package")
        inventory, _, _ = portable_inventory(package)
        result["source_package"] = {"url": PACKAGE_URL, "archive_sha256": PACKAGE_SHA256, "inventory": inventory}
    except Exception as exc:
        result.update(status="blocked_at_source_package", source_package_error=type(exc).__name__ + ": " + str(exc))
        write_new(args.output / "preflight_receipt.json", result)
        print(json.dumps({"status": result["status"], "error": result["source_package_error"]}, ensure_ascii=False))
        return 2
    for index, name in enumerate(FAMILY):
        destination = args.output / f"model_{index + 1:02d}"
        destination.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--device", args.device,
                   "--package-root", str(package), "--output", str(destination)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        started = time.perf_counter()
        try:
            with (destination / "worker.log").open("x") as log:
                proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=180)
            receipt_path = destination / "receipt.json"
            item = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {"model": name, "status": "failed", "error": "No worker receipt; inspect retained worker.log", "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            item = {"model": name, "status": "failed", "error": "Synthetic worker exceeded 180 seconds; partial files retained"}
        item["worker_wall_seconds"] = time.perf_counter() - started
        result["models"].append(item)
        print(f"{index + 1}/17 {name}: {item['status']}", flush=True)
    passed = all(r["status"] == "passed" for r in result["models"])
    neural_names = {*NEURAL, CANDIDATE}
    neural_receipts = [r for r in result["models"] if r["model"] in neural_names]
    gpu_models_verified = (passed and args.device.startswith("cuda")
        and len(neural_receipts) == len(neural_names)
        and {r["model"] for r in neural_receipts} == neural_names
        and all(r.get("device") == args.device and r.get("synthetic_backward_executed") is True
                and r.get("checkpoint_reload_identical") is True for r in neural_receipts))
    result.update(status="passed_synthetic_only" if passed else "failed_synthetic_contracts",
        GPU_execution_verified=gpu_models_verified,
        synthetic_neural_backward_count=sum(bool(r.get("synthetic_backward_executed")) for r in result["models"]),
        synthetic_ARIMA_fit_count=sum(r.get("synthetic_ARIMA_fits", 0) for r in result["models"]))
    write_new(args.output / "preflight_receipt.json", result)
    print(json.dumps({"status": result["status"], "GPU_execution_verified": result["GPU_execution_verified"], "real_campaign_allowed": False, "receipt": str(args.output / "preflight_receipt.json")}, ensure_ascii=False))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=FAMILY)
    args = parser.parse_args()
    if args.worker:
        try:
            value = classical_worker(args.worker, args.output) if args.worker in CLASSICAL else neural_worker(args.worker, args.package_root, args.device, args.output)
        except Exception as exc:
            value = {"model": args.worker, "status": "failed", "error": type(exc).__name__ + ": " + str(exc), "real_data_model_fits": 0}
            import traceback
            traceback.print_exc()
        write_new(args.output / "receipt.json", value)
        return 0 if value["status"] == "passed" else 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
