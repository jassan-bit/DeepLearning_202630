"""Package registered CPU measurements without changing any historical receipt.

This command imports only the standard library. It performs no fit, simulation,
checkpoint load, or checkpoint deletion. Compressed summaries retain their input
bytes; ledger exports retain UTF-8 text, line endings, byte counts and SHA256.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from copy import deepcopy
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


VERSION = "e2-compute-evidence-v1"
REGISTRATION_COMMIT = "d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2"
REGISTRATION_SHA256 = "3b54ac44f5aed956ac98fbc31c95dfcb70c1116b6cca65a8269ff11cd843ac91"
PARITY_SHA256 = "52de23de9ce3fea87c28faadf8b9157f626bd3cc48c7b88fcf89902a3acd6209"
AMENDMENT_COMMIT = "c768b51fdf0077d3cf0d52e79e3db824d3453c61"
AMENDMENT_SHA256 = "141a0ed37b891818145f05bb7ea6452ca4c362c5bd6f43626f878805c58997ff"
THREAD_REPAIR_SHA256 = "9555184caa3ddece0db67adce4f091bf5f753525d3ef786d2f7032fc70d98f87"
SCALING_LAUNCH_SHA256 = "cfd950b74a9bc8f581ba0711a1474bad31a052a6de864b4deabd1b1b0f5175cf"
AMENDED_RUNNER_SHA256 = "192763746eb618a4208942d365a14d42910240cd362b1c565c85951dc402a2a5"
VARIANCE_VERSION = "e2-optimized-preselection-reference-variance-v1"
PROTOCOL_SOURCE_PATH = "book_e2/_static/search_protocol_20261109/protocol.json"
PROTOCOL_SOURCE_SHA256 = "e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe"
ROOT = Path(__file__).resolve().parents[1]
MODELS = ("TimeMixer", "CNN-1D + LSTM + atención temporal", "GRU32")
SEEDS = tuple(range(273001, 273006))
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".log", ".txt"}
HISTORICAL_SOURCES = {
    ("scripts/run_e2_cpu_scaling.py", "c8fc2ff49421684d35b422345638fbc0e95f1827e52adff8cf91e4581f93cb90"):
        "experiments/e2_compute_optimization_20261109/frozen_sources/run_e2_cpu_scaling_preregistered.py",
}
DEVELOPMENT_SOURCES = {
    "results/eda_development/daily_market_development.csv":
        "bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171",
    "results/eda_development/daily_volatility_development.csv":
        "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(raw):
    def bad_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(raw, parse_constant=bad_constant, object_pairs_hook=_unique_object)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def _steps(value):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("Optimizer steps must be a nonnegative integer or null")
    return value


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Worker elapsed seconds must be finite and nonnegative")
    return float(value)


def cost_inventory(rows, *, success_status):
    known, failed_known, elapsed, unknown = 0, 0, 0., 0
    statuses = Counter()
    for row in rows:
        status = row["status"]
        if not isinstance(status, str):
            raise ValueError("Every attempt must retain its status")
        steps = _steps(row.get("optimizer_steps"))
        if status == success_status and steps is None:
            raise ValueError("Completed workers cannot have an unknown step count")
        if steps is None:
            unknown += 1
        else:
            known += steps
            if status != success_status:
                failed_known += steps
        elapsed += _seconds(row["elapsed_wall_seconds"])
        statuses[status] += 1
    return {"attempts": len(rows), "completed_attempts": statuses[success_status],
            "failed_attempts": len(rows)-statuses[success_status],
            "status_counts": dict(sorted(statuses.items())),
            "known_optimizer_steps": known, "known_failed_optimizer_steps": failed_known,
            "unknown_optimizer_step_workers": unknown,
            "total_optimizer_steps": None if unknown else known,
            "elapsed_workers_sum_seconds": elapsed}


def verify_chain(raw):
    """Verify complete records; retain and disclose a killed worker's partial tail."""
    lines = raw.splitlines(keepends=True)
    incomplete = bool(lines and not lines[-1].endswith(b"\n"))
    if incomplete:
        lines = lines[:-1]
    previous = "0" * 64
    style = None
    for sequence, line in enumerate(lines):
        row = strict_json(line)
        field = "event_sha256" if "event_sha256" in row else "sha256"
        if style is not None and style != field:
            raise ValueError("A journal changed its hash convention")
        style = field
        claimed = row.pop(field)
        if row.get("sequence") != sequence or row.get("previous_sha256") != previous:
            raise ValueError("A durable event sequence or predecessor changed")
        options = {"sort_keys": True, "allow_nan": False}
        if field == "sha256":
            options.update(ensure_ascii=False, separators=(",", ":"))
        if digest(json.dumps(row, **options).encode()) != claimed:
            raise ValueError("A durable event digest changed")
        previous = claimed
    return {"events_verified": len(lines), "incomplete_tail": incomplete,
            "last_complete_event_sha256": previous, "hash_field": style}


def _safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("Use a normalized relative evidence path")
    return path


def _has_symlink(path):
    return any(parent.is_symlink() for parent in (path, *path.parents))


def check_sources(value, source_root, observed=None):
    """Check source bytes and the executed worker's source-snapshot contract.

    CPU workers lazily import architecture modules during construction. Their
    invariant preserves every previously loaded source; newly loaded sources
    are separately checked against exact current or admitted archived bytes.
    The variance worker declares a fixed source set and retains strict equality.
    """
    observations = 0
    if observed is None:
        observed = {}
    if isinstance(value, list):
        return sum(check_sources(item, source_root, observed) for item in value)
    if not isinstance(value, dict):
        return 0
    before, after = value.get("source_sha256_before"), value.get("source_sha256_after")
    if isinstance(before, dict) and isinstance(after, dict):
        fixed_set = value.get("version") == VARIANCE_VERSION or "source_hashes_unchanged" in value
        if fixed_set and before != after:
            raise ValueError("A fixed-set variance source snapshot changed")
        if any(after.get(name) != expected for name, expected in before.items()):
            raise ValueError("A previously loaded source changed or disappeared during the worker")
    for key, child in value.items():
        if key in ("source_sha256_before", "source_sha256_after", "source_sha256"):
            if not isinstance(child, dict):
                raise ValueError("A source snapshot must be a mapping")
            for name, expected in child.items():
                if name in DEVELOPMENT_SOURCES:
                    if expected != DEVELOPMENT_SOURCES[name]:
                        raise ValueError("A recorded development data source changed")
                    # Inputs are already pinned by the preregistered loaders;
                    # packaging never opens market CSVs or external cloud paths.
                    continue
                relative = _safe_relative(name)
                current_path = Path(source_root).joinpath(*relative.parts)
                is_pinned_protocol = name == PROTOCOL_SOURCE_PATH and expected == PROTOCOL_SOURCE_SHA256
                historical = HISTORICAL_SOURCES.get((name, expected))
                path = Path(source_root)/historical if historical and digest(current_path.read_bytes()) != expected else current_path
                if ((path.suffix != ".py" and not is_pinned_protocol)
                        or _has_symlink(path) or not path.is_file()):
                    raise ValueError("A recorded source is absent or outside the Python/pinned-protocol allowlist: " + name)
                if not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or digest(path.read_bytes()) != expected:
                    raise ValueError("Recorded source changed after fitting: " + name)
                versions = observed.setdefault(name, {})
                versions[expected] = path.relative_to(source_root).as_posix()
                observations += 1
        else:
            observations += check_sources(child, source_root, observed)
    return observations


def _bind(value, reg_sha, commit, *, variance=False):
    field = "registration_file_sha256" if variance else "registration_sha256"
    if value.get(field) != reg_sha or value.get("registration_commit") != commit:
        raise ValueError("Evidence belongs to another preregistration")


def validate_phase(value, phase, registration, reg_sha, commit):
    _bind(value, reg_sha, commit)
    plan = value["plan"]
    _bind(plan, reg_sha, commit)
    count_key = {"kernels": "kernel_jobs", "threads": "thread_jobs", "scaling": "scaling_jobs"}[phase]
    count = registration["cpu_profile"][count_key]
    if (value.get("phase") != phase or value.get("status") != "completed_all_attempts"
            or value.get("all_requested_jobs_attempted") is not True
            or value.get("no_final_market_data_read") is not True
            or value.get("scientific_HPO_or_NAS_executed") is not False
            or plan.get("source_protocol_sha256") != registration["source_protocol_sha256"]):
        raise ValueError("The complete registered resource phase is required")
    rows = value["results"]
    expected_ids = {phase + "-" + str(i).zfill(3) for i in range(count)}
    if (len(rows) != count or len(plan["jobs"]) != count
            or {row["job"]["job_id"] for row in rows} != expected_ids
            or [row["job"] for row in rows] != plan["jobs"]):
        raise ValueError("Missing, duplicated or changed registered resource jobs")
    for row in rows:
        if row.get("replacement_attempt") is not False:
            raise ValueError("A resource attempt was replaced")
        if row["status"] == "success":
            worker = row.get("worker_profile") or {}
            if (worker.get("sources_unchanged_during_fit") is not True
                    or worker.get("job") != row["job"]
                    or worker.get("final_market_observations_read") is not False):
                raise ValueError("Completed resource receipt lacks source/data integrity")
    micros = value.get("micro_fallback_results", [])
    if micros and phase != "scaling":
        raise ValueError("Only the scaling phase admits distinct microprofiles")
    parents = {row["job"]["job_id"]: row for row in rows}
    seen = set()
    for row in micros:
        job = row["job"]
        parent_id = job.get("parent_job_id")
        parent = parents.get(parent_id)
        if parent is None or parent_id in seen:
            raise ValueError("A microprofile has an absent or duplicated parent")
        seen.add(parent_id)
        eligible = (parent["status"] in ("timeout", "memory_limit_observed")
                    or (parent.get("worker_profile") or {}).get("error_code") == "resource_limit")
        expected = deepcopy(parent["job"])
        expected.update(phase="micro_fallback", job_id=parent_id+"-micro", parent_job_id=parent_id,
                        seed=registration["cpu_profile"]["micro_seed"], epochs=1, batch_windows=32)
        if (parent["status"] == "success" or not eligible or job != expected
                or row.get("replacement_attempt") is not False
                or row.get("separate_micro_profile_not_replacement") is not True):
            raise ValueError("A microprofile replaced or misrepresented its original failure")
    primary_cost = cost_inventory(rows, success_status="success")
    micro_cost = cost_inventory(micros, success_status="success")
    all_cost = cost_inventory(rows+micros, success_status="success")
    if (value.get("known_optimizer_steps") != all_cost["known_optimizer_steps"]
            or value.get("unknown_optimizer_step_workers") != all_cost["unknown_optimizer_step_workers"]
            or not math.isclose(value["elapsed_workers_sum_seconds"], all_cost["elapsed_workers_sum_seconds"],
                                rel_tol=1e-12, abs_tol=1e-7)):
        raise ValueError("Resource cost totals omit or change an attempt")
    if value.get("all_fits_completed") is not None and value["all_fits_completed"] != (primary_cost["failed_attempts"] == 0):
        raise ValueError("Resource completion flag contradicts individual attempts")
    return {"primary": primary_cost, "separate_microprofiles": micro_cost, "combined": all_cost,
            "kernel_policy": value.get("kernel_policy"), "thread_policy": value.get("thread_policy")}


def validate_variance(value, reg_sha, commit):
    _bind(value, reg_sha, commit, variance=True)
    if (value.get("supersedes_previous_attempts") is not False or value.get("replacement_attempts") != 0
            or value.get("final_2025_2026_access") is not False or value.get("final_R_set") is not False
            or value.get("final_power_claimed") is not False or value.get("HPO_executed") is not False
            or value.get("NAS_search_executed") is not False or set(value["models"]) != set(MODELS)):
        raise ValueError("Variance evidence changed its registered scope")
    rows, blocked = [], 0
    for name in MODELS:
        model = value["models"][name]
        selection, refits = model["selection"], model["refit"]
        for group, expected_phase in ((selection, "epoch_selection"), (refits, "fresh_refit")):
            if (len(group) not in (0, 5) or len({row["seed"] for row in group}) != len(group)
                    or any(row["model"] != name or row["phase"] != expected_phase
                           or row.get("replacement_attempt") is not False for row in group)):
                raise ValueError("Variance attempt identity, phase or multiplicity changed")
            if group and {row["seed"] for row in group} != set(SEEDS):
                raise ValueError("Variance evidence discarded or replaced a registered seed")
        if len(selection) != 5:
            raise ValueError("All fifteen inner variance attempts must be retained")
        can_refit = all(row["status"] == "completed" for row in selection)
        if can_refit != (len(refits) == 5):
            raise ValueError("Fresh refits must require all five inner completions")
        if not can_refit:
            blocked += 5
            if name not in value["blocked_refits"]:
                raise ValueError("Missing reason for blocked fresh refits")
        rows.extend(selection+refits)
    inventory = cost_inventory(rows, success_status="completed")
    declared = value["resource_costs"]
    for field, key in (("fit_attempts", "attempts"), ("known_optimizer_steps", "known_optimizer_steps"),
                       ("unknown_optimizer_step_attempts", "unknown_optimizer_step_workers"),
                       ("known_failed_optimizer_steps", "known_failed_optimizer_steps")):
        if declared[field] != inventory[key]:
            raise ValueError("Variance cost totals changed or omitted attempts")
    if not math.isclose(declared["elapsed_worker_wall_seconds"], inventory["elapsed_workers_sum_seconds"],
                        rel_tol=1e-12, abs_tol=1e-7):
        raise ValueError("Variance elapsed totals differ from retained workers")
    inventory.update(blocked_refits_not_attempted=blocked,
                     paired_3x5_complete=value["paired_3x5_complete"],
                     fixed_epoch_counts=value["fixed_epoch_counts"])
    return inventory


def collect_ledgers(roots):
    """Export exact UTF-8 bytes; never traverse symlinks or checkpoint folders."""
    files, chain_count, incomplete = {}, 0, 0
    for label, directory in roots.items():
        _safe_relative(label)
        directory = Path(directory)
        if _has_symlink(directory) or not directory.is_dir():
            raise ValueError("Evidence roots must be ordinary directories")
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if path.is_symlink():
                raise ValueError("Symlinks are not permitted in evidence trees")
            if "checkpoints" in relative.parts or path.suffix not in TEXT_SUFFIXES or not path.is_file():
                continue
            raw = path.read_bytes()
            if len(raw) > 128 * 1024**2:
                raise ValueError("Individual text receipt exceeds128MiB")
            item = {"bytes": len(raw), "sha256": digest(raw)}
            try:
                item.update(encoding="utf-8", text=raw.decode("utf-8"))
            except UnicodeDecodeError:
                # A killed text write may end inside a multibyte character.
                # Preserve the bytes, including that tail, rather than repair it.
                item.update(encoding="base64", raw_base64=base64.b64encode(raw).decode("ascii"))
            if path.suffix == ".json":
                try:
                    strict_json(raw)
                    item["json_parse_status"] = "complete"
                except (ValueError, UnicodeDecodeError):
                    item["json_parse_status"] = "unparseable_bytes_retained"
            if path.suffix == ".jsonl" and path.name == "events.jsonl":
                item["event_chain"] = verify_chain(raw)
                chain_count += 1
                incomplete += int(item["event_chain"]["incomplete_tail"])
            files[label+"/"+relative.as_posix()] = item
    return {"version": VERSION, "files": files, "event_chains_verified": chain_count,
            "incomplete_event_tails_retained": incomplete,
            "export_semantics": "UTF8 text.encode('utf-8') or decoded raw_base64 reconstructs the exact original file bytes, including incomplete tails.",
            "path_semantics": "Export keys are logical download paths; absolute cloud paths inside historical receipts are unchanged.",
            "checkpoint_payloads_included": False}


def _artifact(raw, *, compressed=False):
    receipt = {"bytes": len(raw), "sha256": digest(raw)}
    if compressed:
        decoded = gzip.decompress(raw)
        receipt.update(uncompressed_bytes=len(decoded), uncompressed_sha256=digest(decoded),
                       gzip_mtime=0)
    return receipt


def validate_amendment(amendment_raw, repair_raw, launch_raw, *, amendment_commit, kernel_sha, thread_sha):
    if (amendment_commit != AMENDMENT_COMMIT or digest(amendment_raw) != AMENDMENT_SHA256
            or digest(repair_raw) != THREAD_REPAIR_SHA256 or digest(launch_raw) != SCALING_LAUNCH_SHA256):
        raise ValueError("Use the exact published serialization amendment and launch/repair receipts")
    amendment, repair, launch = map(strict_json, (amendment_raw, repair_raw, launch_raw))
    _bind(repair, REGISTRATION_SHA256, REGISTRATION_COMMIT)
    _bind(launch, REGISTRATION_SHA256, REGISTRATION_COMMIT)
    if (amendment.get("original_registration_commit") != REGISTRATION_COMMIT
            or amendment.get("original_registration_sha256") != REGISTRATION_SHA256
            or amendment.get("new_runner_sha256") != AMENDED_RUNNER_SHA256
            or amendment.get("algorithm_or_measurements_changed") is not False
            or amendment.get("new_market_fits_executed_for_repair") != 0
            or amendment.get("additional_optimizer_steps") != 0
            or amendment.get("reconstructed_summary_sha256") != thread_sha
            or repair.get("all_retained_inputs_unchanged") is not True
            or repair.get("training_executed") is not False
            or repair.get("additional_optimizer_steps") != 0
            or repair.get("summary_sha256") != thread_sha
            or launch.get("amendment_commit") != AMENDMENT_COMMIT
            or launch.get("amendment_sha256") != AMENDMENT_SHA256
            or launch.get("thread_summary_repair_sha256") != THREAD_REPAIR_SHA256
            or launch.get("runner_sha256") != AMENDED_RUNNER_SHA256
            or launch.get("kernel_summary_sha256") != kernel_sha
            or launch.get("thread_summary_sha256") != thread_sha
            or launch.get("scaling_fits_started_before_this_receipt") is not False):
        raise ValueError("Serialization repair or pre-fit launch provenance changed")
    return launch


def verify_package(directory):
    directory = Path(directory)
    manifest = strict_json((directory/"manifest.json").read_bytes())
    for name, expected in manifest["artifacts"].items():
        relative = _safe_relative(name)
        path = directory.joinpath(*relative.parts)
        if _has_symlink(path) or not path.is_file():
            raise ValueError("Published artifact is absent or unsafe")
        if _artifact(path.read_bytes(), compressed=name.endswith(".gz")) != expected:
            raise ValueError("Published artifact byte or decoded digest differs: " + name)
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != set(manifest["artifacts"]) | {"manifest.json"}:
        raise ValueError("Publication directory contains unexpected or missing files")
    return manifest


def build_package(*, kernels, threads, scaling, variance, calibration, budget,
                  registration, parity, output, source_root=ROOT,
                  registration_commit=REGISTRATION_COMMIT, source_amendment=None,
                  amendment_commit=None, thread_repair=None, scaling_launch=None):
    paths = {key: Path(path) for key, path in (("kernels", kernels), ("threads", threads),
             ("scaling", scaling), ("variance", variance), ("calibration", calibration), ("budget", budget))}
    raw_inputs = {key: path.read_bytes() for key, path in paths.items()}
    values = {key: strict_json(raw) for key, raw in raw_inputs.items()}
    reg_raw, parity_raw = Path(registration).read_bytes(), Path(parity).read_bytes()
    if (digest(reg_raw) != REGISTRATION_SHA256 or digest(parity_raw) != PARITY_SHA256
            or registration_commit != REGISTRATION_COMMIT):
        raise ValueError("Only the exact published preregistration/parity bytes are admitted")
    registered = strict_json(reg_raw)
    protocol = Path(source_root)/registered["source_protocol_snapshot"]
    if digest(protocol.read_bytes()) != registered["source_protocol_sha256"]:
        raise ValueError("Published protocol snapshot no longer matches registration")
    phase_costs = {phase: validate_phase(values[phase], phase, registered, REGISTRATION_SHA256,
                                       registration_commit) for phase in ("kernels", "threads", "scaling")}
    kernel_sha, thread_sha = digest(raw_inputs["kernels"]), digest(raw_inputs["threads"])
    for phase in ("threads", "scaling"):
        if values[phase]["plan"].get("kernels_summary_sha256") != kernel_sha:
            raise ValueError("A resource phase changed its preceding kernel summary")
    if values["scaling"]["plan"].get("threads_summary_sha256") != thread_sha:
        raise ValueError("Scaling changed its preceding thread summary")
    runtime = values["variance"]["runtime_policy"]
    if (runtime.get("kernels_summary_sha256") != kernel_sha
            or runtime.get("threads_summary_sha256") != thread_sha
            or runtime.get("registration_sha256") != REGISTRATION_SHA256):
        raise ValueError("Variance changed the independently measured runtime policy")
    variance_costs = validate_variance(values["variance"], REGISTRATION_SHA256, registration_commit)
    # The caller's independent role streams and report provenance are retained
    # verbatim. No downstream packager may close the final scientific gate.
    for key in ("calibration", "budget"):
        report = values[key]
        if not isinstance(report, dict):
            raise ValueError(key + " report must be an explicit JSON object")
        if report.get("final_R") is not None or report.get("achieved_final_power") is not None:
            raise ValueError("Resource/calibration reports cannot set final R or achieved final power")
    calibration_report = values["calibration"]
    _bind(calibration_report, REGISTRATION_SHA256, registration_commit)
    if (calibration_report.get("new_summary_sha256") != digest(raw_inputs["variance"])
            or calibration_report.get("groups_pooled") is not False
            or calibration_report.get("model_fits_executed") != 0
            or calibration_report.get("final_R_set") is not False):
        raise ValueError("Calibration caller is not bound to the separate registered variance panel")
    if values["budget"].get("primary_scaling_sha256") != digest(raw_inputs["scaling"]):
        raise ValueError("Budget wrapper must pin the exact retained primary scaling summary")
    observed_sources = {}
    checked_sources = sum(check_sources(value, source_root, observed_sources) for value in values.values())
    checked_sources += check_sources(strict_json(parity_raw), source_root, observed_sources)
    amended_sources = any(len(versions) > 1 for versions in observed_sources.values())
    amendment_raw = None
    repair_raw, launch_raw = None, None
    if amended_sources:
        if source_amendment is None or thread_repair is None or scaling_launch is None:
            raise ValueError("Multiple executed source versions require the explicit published amendment")
        amendment_raw = Path(source_amendment).read_bytes()
        repair_raw, launch_raw = Path(thread_repair).read_bytes(), Path(scaling_launch).read_bytes()
        launch = validate_amendment(amendment_raw, repair_raw, launch_raw,
                                   amendment_commit=amendment_commit, kernel_sha=kernel_sha, thread_sha=thread_sha)
        events = values["scaling"]["phase_events"]["events"]
        start = next(event["utc"] for event in events if event["kind"] == "worker_requested")
        if datetime.fromisoformat(start) < datetime.fromisoformat(launch["created_utc"]):
            raise ValueError("The amended scaling fits preceded their launch receipt")
        if values["scaling"]["plan"]["source_script_sha256"] != AMENDED_RUNNER_SHA256:
            raise ValueError("Scaling used a runner outside the published serialization amendment")
    ledgers = collect_ledgers({phase: paths[phase].parent for phase in
                              ("kernels", "threads", "scaling", "variance")})
    payload = {key+"_full.json.gz": gzip.compress(raw, mtime=0) for key, raw in raw_inputs.items()}
    payload["ledgers.json.gz"] = gzip.compress(json_bytes(ledgers), mtime=0)
    payload["registration.json"], payload["cpu_conv_parity.json"] = reg_raw, parity_raw
    if amendment_raw is not None:
        payload["source_amendment.json"] = amendment_raw
        payload["thread_summary_repair.json"] = repair_raw
        payload["scaling_launch_receipt.json"] = launch_raw
    checkpoint_count = sum(row["status"] == "completed" and bool((row.get("worker_profile") or {}).get("checkpoint"))
                           for model in values["variance"]["models"].values()
                           for phase in model.values() for row in phase)
    summary = {"version": VERSION, "registration_commit": registration_commit,
               "registration_sha256": REGISTRATION_SHA256,
               "source_protocol_sha256": registered["source_protocol_sha256"],
               "remote_provider": "Codex Cloud", "device": "cpu", "gpu_visible": False,
               "profile_costs_by_phase": phase_costs, "separate_variance_costs": variance_costs,
               "variance_checkpoint_receipts": checkpoint_count,
               "recorded_source_hash_observations_checked": checked_sources,
               "ledger_file_count": len(ledgers["files"]),
               "event_chains_verified": ledgers["event_chains_verified"],
               "incomplete_event_tails_retained": ledgers["incomplete_event_tails_retained"],
               "input_sha256": {key: digest(raw) for key, raw in raw_inputs.items()},
               "final_R": None, "achieved_final_power": None, "full_HPO_or_NAS_executed": False,
               "final_2025_2026_values_read": False, "weights_published": False,
               "historical_failures_superseded": False,
               "limits": ["Known costs are partial whenever any worker has unknown steps; null is not zero.",
                          "Microprofiles remain separate from failed full-cohort profiles.",
                          "This export checks receipt bytes, event chains and recorded sources, without loading checkpoint tensors.",
                          "CPU source checks preserve every previously loaded module; lazy imports may add modules whose recorded bytes are also verified. Variance source sets must match exactly.",
                          "Absolute cloud paths inside receipts are historical evidence; download keys are logical paths.",
                          "Conditional budget and empirical Monte Carlo planning do not establish final-study power or total campaign duration."]}
    payload["summary.json"] = json_bytes(summary)
    manifest = {"version": VERSION, "registration_commit": registration_commit,
                "registration_sha256": REGISTRATION_SHA256,
                "recorded_source_sha256": {name: next(iter(versions)) for name, versions in sorted(observed_sources.items())
                                           if len(versions) == 1},
                "recorded_source_versions": [{"source_path": name, "sha256": sha, "verification_path": path}
                                             for name, versions in sorted(observed_sources.items())
                                             for sha, path in sorted(versions.items())],
                "source_amendment_commit": amendment_commit if amended_sources else None,
                "source_amendment_sha256": digest(amendment_raw) if amendment_raw else None,
                "source_verification_contract": {
                    "cpu_profiles": "Every before-snapshot path remains present and unchanged in after; new lazy-import paths are admitted only after exact byte-hash verification.",
                    "variance": "The explicitly fixed before/after source sets must be identical.",
                    "all_recorded_source_versions": "Python modules and the exact registered protocol JSON are verified against the current file or explicitly admitted archived source version.",
                },
                "postfit_packager_sha256": digest(Path(__file__).read_bytes()),
                "artifacts": {name: _artifact(raw, compressed=name.endswith(".gz"))
                              for name, raw in sorted(payload.items())},
                "manifest_sha256_is_external": True, "weights_included": False}
    output = Path(output)
    if output.exists() or _has_symlink(output):
        raise FileExistsError("Evidence output directory must be new")
    output.mkdir(parents=True)
    for name, raw in payload.items():
        with (output/name).open("xb") as stream:
            stream.write(raw)
    with (output/"manifest.json").open("xb") as stream:
        stream.write(json_bytes(manifest))
    verify_package(output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("kernels", "threads", "scaling", "variance", "calibration", "budget",
                 "registration", "parity", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--registration-commit", default=REGISTRATION_COMMIT)
    parser.add_argument("--source-amendment", type=Path)
    parser.add_argument("--amendment-commit")
    parser.add_argument("--thread-repair", type=Path)
    parser.add_argument("--scaling-launch", type=Path)
    args = parser.parse_args(argv)
    result = build_package(**vars(args))
    print(json.dumps({"status": "packaged_and_verified", "output": str(args.output),
                      "profile_costs_by_phase": result["profile_costs_by_phase"],
                      "separate_variance_costs": result["separate_variance_costs"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
