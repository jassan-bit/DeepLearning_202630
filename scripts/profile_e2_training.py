"""Bounded subprocess profiles of synthetic E2 training stages.

Only internally generated artificial windows are accepted by the worker CLI.
This measures a small fixture, not a full HPO/NAS campaign or market benchmark.
RSS sampling is a watchdog, not a kernel-enforced memory allocation limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "book_e2/_static/search_protocol_20261109/protocol.json"
VERSION = "e2-synthetic-training-profile-v1"


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + " must be positive and finite")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_text(path: str):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def hardware_snapshot():
    cpu_max = _read_text("/sys/fs/cgroup/cpu.max")
    quota = None
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0] != "max":
            quota = int(parts[0]) / int(parts[1])
    memory_max = _read_text("/sys/fs/cgroup/memory.max")
    cpu_model = None
    for line in (_read_text("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    return {
        "cpu_model": cpu_model,
        "visible_logical_cpus": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": cpu_max,
        "cgroup_cpu_quota_vcpus": quota,
        "cgroup_memory_limit_bytes": int(memory_max) if memory_max and memory_max != "max" else None,
        "gpu_device_nodes": sorted(str(p) for p in Path("/dev").glob("nvidia[0-9]*")),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def process_group_rss_bytes(process_group: int) -> int:
    """Observed RSS sum for live members of one spawned process group.

    Shared pages may be counted more than once. A short allocation can escape
    sampling. Worker code also reports its own OS peak RSS after success.
    """
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            tail = stat[stat.rfind(")") + 2:].split()
            # tail begins with state, ppid, pgrp, session.
            if tail[0] == "Z" or int(tail[2]) != process_group:
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


def _stop_process_group(process: subprocess.Popen, *, grace_seconds: float = 1.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=max(1., grace_seconds))


def run_isolated_profile(command, output_dir, *, timeout_seconds=120.,
                         memory_limit_bytes=12 * 1024**3, poll_interval=.05,
                         cwd=None, environment=None):
    """Run one explicit argv list, preserving stdout/stderr and failure metadata.

    This generic supervisor does not certify what an arbitrary command reads.
    The public profiling CLI below launches only its fixed synthetic worker.
    """
    if not isinstance(command, (list, tuple)) or not command or any(not isinstance(x, str) or not x for x in command):
        raise ValueError("An explicit nonempty argv list is required; shell execution is not supported")
    _positive(timeout_seconds, "timeout_seconds")
    _positive(poll_interval, "poll_interval")
    if type(memory_limit_bytes) is not int or memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be a positive integer")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Profile output directory must be new")
    output.mkdir(parents=True)
    started = time.monotonic()
    command_bytes = json.dumps(list(command), ensure_ascii=False).encode()
    record = {
        "version": VERSION, "status": "running",
        "executable": Path(command[0]).name,
        "command_sha256": hashlib.sha256(command_bytes).hexdigest(),
        "argument_count": len(command) - 1,
        "timeout_seconds": float(timeout_seconds),
        "rss_watchdog_limit_bytes": memory_limit_bytes,
        "rss_watchdog_is_hard_memory_limit": False,
        "poll_interval_seconds": float(poll_interval),
        "supervisor_reads_market_files": False,
        "hardware": hardware_snapshot(),
    }
    _atomic_json(output / "request.json", record)
    peak = 0
    returncode = None
    process = None
    status = "spawn_failure"
    failure = None
    worker_profile = None
    try:
        with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(list(command), stdout=stdout, stderr=stderr,
                                       cwd=cwd, env=environment, start_new_session=True)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                peak = max(peak, process_group_rss_bytes(process.pid))
                if elapsed > timeout_seconds:
                    status = "timeout"
                    _stop_process_group(process)
                    break
                if peak > memory_limit_bytes:
                    status = "memory_limit_observed"
                    _stop_process_group(process)
                    break
                time.sleep(poll_interval)
            returncode = process.wait()
            if status not in ("timeout", "memory_limit_observed"):
                status = "success" if returncode == 0 else "worker_failure"
            possible = output / "worker_profile.json"
            if possible.is_file():
                try:
                    worker_profile = json.loads(possible.read_text())
                except (ValueError, OSError) as error:
                    failure = {"type": type(error).__name__, "message": str(error)}
                    status = "invalid_worker_profile"
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        if process is not None:
            _stop_process_group(process)
            returncode = process.returncode
    finally:
        if process is not None and process.poll() is None:
            _stop_process_group(process)
    result = {
        **record, "status": status, "returncode": returncode,
        "elapsed_wall_seconds": time.monotonic() - started,
        "sampled_peak_process_group_rss_bytes": peak,
        "worker_profile": worker_profile, "failure": failure,
        "limits": [
            "Sampled RSS is a watchdog, not a hard allocation guarantee.",
            "An arbitrary supervised argv is not a data-access sandbox.",
            "No total-campaign hours are inferred from this result.",
        ],
    }
    _atomic_json(output / "result.json", result)
    return result


def _profile_models(protocol):
    return [m["model"] for m in protocol["models"]] + [
        m["model"] for m in protocol["neural_reference_profiles"]]


def _stage_record(stage):
    return {
        "best_epoch": stage.best_epoch,
        "epochs_completed": stage.epochs_completed,
        "optimizer_steps": stage.optimizer_steps,
        "elapsed_seconds": stage.elapsed_seconds,
        "state_bytes": sum(t.numel() * t.element_size() for t in stage.state_dict.values()),
        "initial_state_sha256": stage.initial_state_sha256,
        "selected_state_sha256": stage.selected_state_sha256,
        "optimizer_initial_state_entries": stage.optimizer_initial_state_entries,
        "stop_reason": stage.stop_reason,
        "selection_metric": stage.selection_metric,
        "scheduler_total_steps": stage.scheduler_total_steps,
        "max_observed_rss_bytes": stage.max_observed_rss_bytes,
        "seed": stage.seed,
        "logs": list(stage.logs),
    }


def _loaded_repository_source_hashes():
    result = {}
    scripts = (ROOT / "scripts").resolve()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(scripts) and path.is_file():
            result[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def run_synthetic_worker(spec):
    """Actual bounded optimizer steps on artificial windows only."""
    if spec.get("synthetic_only") is not True:
        raise ValueError("Only an explicit synthetic profile is supported")
    import torch
    import e2_search_plan as planner
    from e2_training_models import build_train_model
    from e2_staged_training import TrainConfig, fit_supervised, fit_sae

    protocol = json.loads(Path(spec["protocol"]).read_text())
    planner.validate_protocol(protocol)
    name = spec["model"]
    if name not in _profile_models(protocol):
        raise ValueError("Model is outside the versioned fourteen-network registry")
    if type(spec["epochs"]) is not int or not 1 <= spec["epochs"] <= 3:
        raise ValueError("Synthetic profiling permits one to three epochs only")
    if spec["train_windows"] not in (16, 32) or spec["validation_windows"] not in (8, 16):
        raise ValueError("Only bounded synthetic fixture sizes are supported")
    if spec["threads"] not in (1, 2, 4):
        raise ValueError("Use one, two or four CPU threads")
    torch.set_num_threads(spec["threads"])
    torch.set_num_interop_threads(1)
    generator = torch.Generator().manual_seed(spec["fixture_seed"])
    train_x = torch.randn(spec["train_windows"], 7, 5, generator=generator)
    validation_x = torch.randn(spec["validation_windows"], 7, 5, generator=generator)
    mean = train_x.mean(dim=(0, 1), keepdim=True)
    deviation = train_x.std(dim=(0, 1), unbiased=False, keepdim=True)
    deviation = torch.where(deviation == 0, torch.ones_like(deviation), deviation)
    train_x = (train_x - mean) / deviation
    validation_x = (validation_x - mean) / deviation
    train_y = .01 + .02 * torch.rand(spec["train_windows"], generator=generator)
    validation_y = .01 + .02 * torch.rand(spec["validation_windows"], generator=generator)
    target_scale = float(train_y.mean())
    train_y = train_y / target_scale
    validation_y = validation_y / target_scale
    trial = planner.generate_hpo(protocol, name)[0]
    model_spec = next(m for m in protocol["models"] + protocol["neural_reference_profiles"]
                      if m["model"] == name)
    architecture = None
    largest = spec.get("largest_timesnet", False)
    if largest and name != "TimesNet":
        raise ValueError("The optional largest-width case belongs only to TimesNet")
    if model_spec["nas"]:
        architectures = planner.enumerate_architectures(protocol, name)
        if largest:
            trial = next(t for t in planner.generate_hpo(protocol, name) if t["num_kernels"] == 6)
            architecture = next(a for a in architectures if a["widths"] == [256, 256, 256]
                                and a["dropout_pattern"] == [1, 1, 1])
        else:
            architecture = next(a for a in architectures if a["is_reference"])
    started = time.perf_counter()
    model = build_train_model(name, trial, architecture=architecture,
                              source_roots=spec.get("source_roots"), seed=spec["training_seed"])
    source_hashes = _loaded_repository_source_hashes()
    architecture_metadata = getattr(model, "architecture_metadata", {})
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adam_fp32_lower_bound_bytes = 4 * 4 * trainable_parameter_count
    if adam_fp32_lower_bound_bytes > .8 * spec["memory_limit_bytes"]:
        raise MemoryError("Weights+gradients+two FP32 Adam moments already approach the RSS watchdog limit")
    config = TrainConfig(
        learning_rate=trial["learning_rate"], weight_decay=trial["weight_decay"],
        batch_size=trial["batch_size"], max_epochs=spec["epochs"], patience=20,
        warmup_fraction=trial.get("warmup_fraction", 0.), seed=spec["training_seed"],
        gradient_clip=1., wall_limit_seconds=spec["timeout_seconds"],
        memory_limit_bytes=spec["memory_limit_bytes"],
    )
    if name == "LSTM-autoencoder con predictor":
        result = fit_sae(model.config, train_x, train_y,
                         validation_x, validation_y, target_scale, config,
                         fold_id="artificial-profile")
        phases = {phase: _stage_record(stage) for phase, stage in result.phases.items()}
        auxiliary_scale = result.auxiliary_scale
    else:
        # The optional extreme uses one complete optimizer step over its sixteen
        # synthetic windows, with no validation or checkpoint selection.
        if largest:
            if spec["train_windows"] != 16 or trial["batch_size"] < 16:
                raise ValueError("Largest TimesNet profile must remain a one-batch fixture")
            result = fit_supervised(model, train_x, train_y, None, None,
                                    target_scale, config, fixed_epochs=1)
        else:
            result = fit_supervised(model, train_x, train_y,
                                    validation_x, validation_y,
                                    target_scale, config)
        phases = {"supervised": _stage_record(result)}
        auxiliary_scale = None
    total_steps = sum(record["optimizer_steps"] for record in phases.values())
    final_source_hashes = _loaded_repository_source_hashes()
    if any(final_source_hashes.get(path) != sha for path, sha in source_hashes.items()):
        raise RuntimeError("A loaded source file changed during the isolated profile")
    return {
        "version": VERSION, "status": "success", "synthetic_only": True,
        "market_data_read": False, "hpo_search_executed": False,
        "nas_search_executed": False, "model": name,
        "architecture": architecture, "registered_trial": trial,
        "architecture_metadata": architecture_metadata,
        "largest_timesnet": largest, "profile_epochs_cap": spec["epochs"],
        "fixture": {"seed": spec["fixture_seed"], "train_windows": spec["train_windows"],
                    "validation_windows": 0 if largest else spec["validation_windows"],
                    "shape": [7, 5], "train_only_input_scaling": True,
                    "target_scale": target_scale},
        "training_seed": spec["training_seed"], "phases": phases,
        "training_seed_is_profile_override": True,
        "source_sha256_at_model_construction": source_hashes,
        "source_sha256_at_completion": final_source_hashes,
        "loaded_sources_unchanged_during_profile": True,
        "auxiliary_scale": auxiliary_scale, "optimizer_steps": total_steps,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "adam_fp32_weights_grad_moments_lower_bound_bytes": adam_fp32_lower_bound_bytes,
        "elapsed_build_and_fit_seconds": time.perf_counter() - started,
        "worker_os_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "runtime": {"torch": str(torch.__version__), "python": platform.python_version(),
                    "threads": torch.get_num_threads(), "device": "cpu"},
        "hardware": hardware_snapshot(),
        "limits": [
            "Artificial short fixture; not a full-data or 200-epoch training cost.",
            "Registered structural and training row is used, but epoch cap is explicitly shortened for profiling.",
            "Losses are synthetic mechanism outputs, not predictive or scientific results.",
            "Peak RSS includes construction, training, checkpoint copies and imported runtime in this worker.",
            "This does not close the complete-campaign compute or storage budget.",
        ],
    }


def profile_models(protocol_path, output, names, *, epochs=2, threads=4,
                   timeout_seconds=120., memory_limit_bytes=12 * 1024**3,
                   train_windows=16, validation_windows=8,
                   source_roots=None, include_largest_timesnet=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Profile campaign output directory must be new")
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    valid_names = _profile_models(protocol)
    if not names or len(set(names)) != len(names) or any(name not in valid_names for name in names):
        raise ValueError("Choose distinct supported neural models")
    if type(epochs) is not int or not 1 <= epochs <= 3:
        raise ValueError("Use one to three explicitly short profiling epochs")
    _positive(timeout_seconds, "timeout_seconds")
    if timeout_seconds > 300:
        raise ValueError("This bounded synthetic profiler allows at most300 seconds per worker")
    if type(memory_limit_bytes) is not int or not 0 < memory_limit_bytes <= 12 * 1024**3:
        raise ValueError("Choose a watchdog between zero and12GiB")
    if threads not in (1, 2, 4) or train_windows not in (16, 32) or validation_windows not in (8, 16):
        raise ValueError("Unsupported bounded fixture or thread count")
    output.mkdir(parents=True)
    specs = []
    for index, name in enumerate(names):
        specs.append({"name": name, "largest": False, "index": index})
    if include_largest_timesnet:
        specs.append({"name": "TimesNet", "largest": True, "index": len(specs)})
    results = []
    for item in specs:
        index, name, largest = item["index"], item["name"], item["largest"]
        job = output / f"profile-{index:02d}"
        spec = {
            "synthetic_only": True, "protocol": str(protocol_path), "model": name,
            "epochs": 1 if largest else epochs, "threads": threads,
            "train_windows": 16 if largest else train_windows,
            "validation_windows": validation_windows,
            "fixture_seed": 291101, "training_seed": 291201,
            "timeout_seconds": float(timeout_seconds),
            "memory_limit_bytes": memory_limit_bytes,
            "source_roots": source_roots or {}, "largest_timesnet": largest,
        }
        spec_path = output / f"spec-{index:02d}.json"
        _atomic_json(spec_path, spec)
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--worker-spec", str(spec_path.resolve()),
                   "--worker-output", str((job / "worker_profile.json").resolve())]
        result = run_isolated_profile(command, job, timeout_seconds=timeout_seconds,
                                      memory_limit_bytes=memory_limit_bytes, cwd=ROOT)
        worker = result["worker_profile"]
        if result["status"] == "success":
            if (not isinstance(worker, dict) or worker.get("synthetic_only") is not True
                    or worker.get("market_data_read") is not False):
                result["status"] = "invalid_synthetic_scope"
                _atomic_json(job / "result.json", result)
        results.append({"model": name, "largest_timesnet": largest,
                        "status": result["status"],
                        "elapsed_wall_seconds": result["elapsed_wall_seconds"],
                        "sampled_peak_process_group_rss_bytes": result["sampled_peak_process_group_rss_bytes"],
                        "worker_profile": worker,
                        "returncode": result["returncode"]})
        print(json.dumps({"model": name, "largest_timesnet": largest,
                          "status": result["status"]}, ensure_ascii=False), flush=True)
    summary = {
        "version": VERSION,
        "status": "complete" if all(r["status"] == "success" for r in results) else "partial_or_failed",
        "synthetic_only": True, "market_data_read": False,
        "full_market_campaign_executed": False,
        "hours_for_full_campaign": None, "currency_cost_estimate": None,
        "protocol_file_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "profile_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "requested_models": names, "results": results,
        "hardware": hardware_snapshot(),
        "limits": [
            "Every failure/timeout is retained; no silent model omission or replacement.",
            "RSS watchdog is sampled and cannot guarantee allocation prevention.",
            "Small synthetic optimizer profiles do not estimate the full campaign without a validated scaling study.",
            "External pinned source roots, when used, contain author code, not market data.",
        ],
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", action="append")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=120.)
    parser.add_argument("--memory-limit-gib", type=float, default=12.)
    parser.add_argument("--train-windows", type=int, default=16)
    parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--source-root-itransformer", type=Path)
    parser.add_argument("--include-largest-timesnet", action="store_true")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec is not None:
        if args.worker_output is None:
            parser.error("Worker output is required")
        try:
            result = run_synthetic_worker(json.loads(args.worker_spec.read_text()))
        except Exception as error:
            receipt = getattr(error, "receipt", None)
            _atomic_json(args.worker_output, {
                "version": VERSION, "status": "failed", "synthetic_only": True,
                "market_data_read": False,
                "failure": {"type": type(error).__name__, "message": str(error)},
                "training_failure_receipt": receipt,
            })
            raise
        else:
            _atomic_json(args.worker_output, result)
        return 0
    if args.worker_output is not None:
        parser.error("Worker output is internal to a worker spec")
    if args.output is None:
        parser.error("--output NEWDIR is required")
    protocol = json.loads(args.protocol.read_text())
    roots = {"iTransformer": str(args.source_root_itransformer.resolve())} if args.source_root_itransformer else {}
    summary = profile_models(
        args.protocol, args.output, args.model or _profile_models(protocol),
        epochs=args.epochs, threads=args.threads,
        timeout_seconds=args.timeout_seconds,
        memory_limit_bytes=int(args.memory_limit_gib * 1024**3),
        train_windows=args.train_windows, validation_windows=args.validation_windows,
        source_roots=roots, include_largest_timesnet=args.include_largest_timesnet,
    )
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

