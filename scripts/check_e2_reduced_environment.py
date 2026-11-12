"""Read-only runtime/budget diagnostics for the reduced E2 registration.

No market loader, optimizer, trainer, reservation, retry, installation or budget
extension is provided. A tiny CUDA tensor calculation checks the selected
runtime only; it cannot validate all model kernels or authorize training.
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
import subprocess
import sys

VERSION = "e2-reduced-environment-check-v1"
EXPECTED = {"python_major_minor": [3, 12], "torch": "2.8.0+cu126", "torch_cuda": "12.6"}
CONTROLLERS = {"run_e2_cuda_search.py", "run_e2_cuda_recovery.py", "e2_guide_schedule.py",
    "run_e2_optimized_selection.py", "run_e2_optimized_selection_v2.py", "run_e2_amended_pilots.py"}
_AUTO = object()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("A finite nonnegative budget value is required")
    return float(value)


def _checked(value):
    if type(value) is not dict:
        raise ValueError("An explicit hash-bound budget record is required")
    plain = {key: val for key, val in value.items() if key != "sha256"}
    if value.get("sha256") != digest(plain):
        raise ValueError("Budget record SHA256 changed")
    return plain


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate diagnostic record key")
        value[key] = item
    return value


def _regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("A regular existing diagnostic input is required")
    return path


def controller_pids(proc_root=Path("/proc")):
    """Inspect argv boundaries, not shell strings containing script names."""
    found = []
    for directory in Path(proc_root).iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            words = (directory / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(word and not any(c.isspace() for c in word) and Path(word).name in CONTROLLERS
               for word in words[1:]):
            found.append(int(directory.name))
    return sorted(found)


def policy_compatible(runtime):
    return (runtime.get("python_major_minor") == EXPECTED["python_major_minor"]
        and runtime.get("torch") == EXPECTED["torch"]
        and runtime.get("torch_cuda") == EXPECTED["torch_cuda"]
        and runtime.get("cuda_available") is True
        and runtime.get("cuda_kernel_probe_passed") is True)


def physical_gpu_probe():
    """Visible driver devices, separately from the selected Torch build."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if result.returncode:
            return {"status": "nvidia_driver_probe_failed", "devices": []}
        devices = []
        for row in result.stdout.splitlines():
            name, memory = row.rsplit(",", 1)
            devices.append({"name": name.strip(), "memory_mib": int(memory.strip())})
        return {"status": "driver_devices_visible" if devices else "no_driver_devices", "devices": devices}
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return {"status": "nvidia_smi_unavailable_or_unparseable", "devices": []}


def runtime_flags(torch):
    """Observe current flags separately from the temporary probe context."""
    def call(name):
        function = getattr(torch, name, None)
        return function() if callable(function) else None
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    matmul = getattr(getattr(backends, "cuda", None), "matmul", None)
    cudnn_version = getattr(cudnn, "version", None)
    dtype = call("get_default_dtype")
    return {"cudnn": cudnn_version() if callable(cudnn_version) else None,
        "cpu_threads": call("get_num_threads"), "interop_threads": call("get_num_interop_threads"),
        "dtype": str(dtype).removeprefix("torch.") if dtype is not None else None,
        "amp": call("is_autocast_enabled"), "tf32": getattr(matmul, "allow_tf32", None),
        "cudnn_tf32": getattr(cudnn, "allow_tf32", None),
        "deterministic": call("are_deterministic_algorithms_enabled"),
        "flags_scope": "Observed before the temporary CUDA probe; no training policy applied here"}


def cuda_probe(torch):
    """FP32 deterministic matmul only, with no gradient or optimizer steps."""
    old = {"deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "precision": torch.get_float32_matmul_precision(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic}
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            x = torch.arange(32, device="cuda:0", dtype=torch.float32).reshape(4, 8)
            y = x @ x.T
            torch.cuda.synchronize(0)
            return {"passed": y.device.type == "cuda" and bool(torch.isfinite(y).all()),
                "device": str(y.device), "dtype": str(y.dtype), "shape": list(y.shape),
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "amp": False, "tf32": False, "optimizer_steps": 0,
                "cpu_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                "scope": "Tiny CUDA matrix probe; not all model-kernel acceptance"}
    finally:
        torch.use_deterministic_algorithms(old["deterministic"], warn_only=old["warn_only"])
        torch.backends.cuda.matmul.allow_tf32 = old["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = old["cudnn_tf32"]
        torch.backends.cudnn.benchmark = old["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = old["cudnn_deterministic"]
        torch.set_float32_matmul_precision(old["precision"])


def inspect_runtime(torch_module=_AUTO, *, proc_root=Path("/proc")):
    # This is a short-lived diagnostic process; set cuBLAS before CUDA startup.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report = {"python": sys.version.split()[0], "python_major_minor": list(sys.version_info[:2]),
        "python_executable": sys.executable, "torch": None, "torch_cuda": None,
        "cuda_available": False, "cuda_kernel_probe_passed": False,
        "device": None, "device_name": None, "cuda_device_count": 0, "cuda_total_memory_bytes": None,
        "total_memory_bytes": None, "effective_memory_limit_bytes": None,
        "other_controller_pids": controller_pids(proc_root), "expected_policy": EXPECTED,
        "production_policy_compatible": False, "optimizer_steps": 0,
        "market_training_started": False, "CPU_training_fallback_performed": False,
        "all_model_kernels_validated": False, "physical_gpu_probe": physical_gpu_probe()}
    try:
        torch = importlib.import_module("torch") if torch_module is _AUTO else torch_module
        if torch is None:
            raise ImportError("Torch is not installed in this interpreter")
        report.update(torch=str(torch.__version__), torch_cuda=torch.version.cuda,
                      cuda_available=bool(torch.cuda.is_available()))
        report.update(runtime_flags(torch))
        if report["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            report.update(device="cuda:0", device_name=str(properties.name), cuda_device_count=torch.cuda.device_count(),
                          cuda_total_memory_bytes=int(properties.total_memory), total_memory_bytes=int(properties.total_memory),
                          effective_memory_limit_bytes=int(properties.total_memory * .8),
                          effective_memory_limit_scope="Registered80%-VRAM allocator threshold, not an enforced hardware quota")
            report["probe"] = cuda_probe(torch)
            report["cuda_kernel_probe_passed"] = report["probe"]["passed"]
        report["production_policy_compatible"] = policy_compatible(report)
        report["status"] = ("cuda_runtime_policy_verified" if report["production_policy_compatible"] else
                            "cuda_visible_policy_mismatch" if report["cuda_available"] else "no_cuda_runtime")
    except Exception as error:
        report.update(status="runtime_check_failed", reason=type(error).__name__ + ": " + str(error))
    return report


def inspect_budget(evidence_path=None, *, ledger=None):
    """Read a reconciled snapshot and optional local ledger without any writes.

    The snapshot is not silently refreshed/reanchored. Remote freshness is an
    external receipt; an unknown local attempt retains its full reservation.
    """
    if evidence_path is None:
        return {"status": "budget_evidence_pending", "training_authorized": False,
                "fresh_remote_heads_checked_here": False, "budget_reset": False}
    path = _regular(evidence_path)
    evidence = json.loads(path.read_text(), object_pairs_hook=_unique_pairs)
    value = _checked(evidence)
    cap, charged = _finite(value.get("authorized_combined_seconds")), _finite(value.get("historical_charged_seconds"))
    remaining = _finite(value.get("remaining_seconds_before_pilots"))
    if (value.get("version") != "e2-amended-pilot-budget-v1"
            or value.get("status") != "reconciled_before_new_pilot_reservations"
            or value.get("budget_reset") is not False
            or _finite(value.get("authorized_combined_hours")) != 64
            or cap != 64 * 3600 or not math.isclose(remaining, max(0, cap-charged), abs_tol=1e-6)):
        raise ValueError("Use the reconciled shared64-hour allowance without resetting past charges")
    charges, active, attempts, progress = 0., {}, set(), {}
    count, previous = 0, "0" * 64
    ledger_path = None
    if ledger is not None:
        ledger_path = Path(ledger)
        if ledger_path.is_dir():
            ledger_path = ledger_path / "pilot_budget.jsonl"
        raw = _regular(ledger_path).read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("A truncated local ledger cannot release a reservation")
        for index, line in enumerate(raw.splitlines()):
            record = json.loads(line, object_pairs_hook=_unique_pairs)
            event = _checked(record)
            if (event.get("version") != "e2-amended-pilot-budget-v1" or event.get("sequence") != index
                    or event.get("previous_sha256") != previous
                    or event.get("reconciliation_sha256") != evidence["sha256"]):
                raise ValueError("Local budget chain or historical anchor changed")
            kind, identity = event.get("kind"), event.get("attempt_id")
            if kind == "reserved":
                amount = _finite(event.get("reservation_seconds"))
                if (not isinstance(identity, str) or not identity or identity in attempts or active
                        or amount <= 0 or amount > max(0, remaining-charges-sum(active.values()))+1e-9):
                    raise ValueError("Invalid or duplicate local reservation")
                attempts.add(identity); active[identity] = amount
            elif kind == "charged":
                if (identity not in active or event.get("reservation_seconds") != active[identity]
                        or event.get("status") not in ("completed", "failed", "paused_clean")):
                    raise ValueError("Local charge lacks its reservation")
                charges += _finite(event.get("elapsed_seconds")); del active[identity]
            elif kind == "overhead":
                charges += _finite(event.get("elapsed_seconds"))
            elif kind == "progress":
                elapsed, steps = _finite(event.get("elapsed_seconds_lower_bound")), event.get("optimizer_steps_lower_bound")
                before = progress.get(identity, (0., 0))
                if identity not in active or type(steps) is not int or steps < before[1] or elapsed < before[0]:
                    raise ValueError("Invalid local progress lower bound")
                progress[identity] = (elapsed, steps)
            else:
                raise ValueError("Unknown local budget event")
            previous, count = record["sha256"], index+1
    return {"status": "reconciliation_required" if active else
                "snapshot_and_supplied_ledger_verified" if ledger_path is not None else "budget_snapshot_verified",
        "evidence_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reconciliation_sha256": evidence["sha256"], "historical_heads": value.get("historical_heads"),
        "authorized_total_hours": 64, "historical_charged_seconds": charged,
        "local_pilot_charged_seconds": charges, "unclosed_reservations": active,
        "remaining_seconds": max(0, remaining-charges-sum(active.values())),
        "ledger_records_checked": count, "ledger_scope": None if ledger_path is None else str(ledger_path),
        "other_mounts_or_private_ledgers_checked": False, "fresh_remote_heads_checked_here": False,
        "budget_reset": False, "training_authorized": False}


def build_report(*, budget_evidence=None, ledger=None, torch_module=_AUTO, proc_root=Path("/proc")):
    report = {"version": VERSION, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Runtime/budget diagnostics only; not source/data/cache or model-kernel acceptance",
        "runtime": inspect_runtime(torch_module, proc_root=proc_root),
        "training_authorized": False, "market_training_started": False, "scientific_HPO_or_NAS_executed": False,
        "final_market_values_read": False, "budget_extended": False, "data_and_cache_audit_performed_here": False}
    try:
        report["budget"] = inspect_budget(budget_evidence, ledger=ledger)
    except Exception as error:
        report["budget"] = {"status": "budget_check_failed", "reason": type(error).__name__ + ": " + str(error),
                            "training_authorized": False, "budget_reset": False}
    report["sha256"] = digest(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New exclusive JSON diagnostic file")
    parser.add_argument("--budget-evidence", type=Path)
    parser.add_argument("--ledger", type=Path, help="Existing canonical local pilot ledger; no lock/write")
    args = parser.parse_args(argv)
    report = build_report(budget_evidence=args.budget_evidence, ledger=args.ledger)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    runtime_ok = report["runtime"]["production_policy_compatible"] and not report["runtime"]["other_controller_pids"]
    budget_ok = report["budget"]["status"] in ("snapshot_and_supplied_ledger_verified", "budget_snapshot_verified")
    return 0 if runtime_ok and budget_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
