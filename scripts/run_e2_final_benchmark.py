"""Registered, resumable E2 benchmark with causal nested HPO and sealed phases.

No phase is inferred from the presence of a directory.  Checkpoints, selection
decisions and phase receipts are content-addressed.  ``plan`` is the default;
it reads metadata only.  Market labels beyond development may be decoded only
by the explicit ``score`` phase, after *all* scheduled models have closed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import resource
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PROCESS_STARTED_WALL = time.perf_counter()
DEFAULT_DESIGN = ROOT / "experiments/e2_completion_20261106/design.json"
TOP10 = ("TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST", "Informer",
         "Autoformer", "iTransformer", "Crossformer", "S-Mamba")
TUNABLE_NEURAL = TOP10 + ("GRU-E2", "LSTM-E2", "TCN-E2")
TUNABLE = TUNABLE_NEURAL + ("Ridge", "ARIMA")
ANCHORS = ("GRU/D7-c03", "LSTM/D7-c02", "TCN/D7-c02")
NAIVES = ("Persistence", "SeasonalNaive")
CATALOGUE = TOP10 + ("GRU-E2", "LSTM-E2", "TCN-E2") + ANCHORS + ("Ridge", "Persistence", "SeasonalNaive", "ARIMA")
DETERMINISTIC = frozenset(("Ridge", "ARIMA", *NAIVES))
NEGATIVE_CONTROL_MODELS = tuple(m for m in CATALOGUE if m not in ("ARIMA", *NAIVES))
NEGATIVE_CONTROL_SEED = 690101
HPO_SEEDS = (2026110611, 2026110612)
P1_MODELS = ("TimeMixer", "iTransformer", "PatchTST", "GRU/D7-c03", "GRU-E2")
P1_SEEDS = tuple(range(660201, 660211))
P1_ORIGIN, P1_END = date(2024, 11, 8), date(2024, 12, 30)
A_SEEDS = tuple(range(680101, 680106))
OUTER_ORIGINS = tuple(date(2024, month, 1) for month in range(1, 11))
FEATURES = ("open", "high", "low", "close", "volume")
DELTA = 0.00025772842664981106
HIGH_VOLATILITY = 0.024247163097617024
E1_SCIENTIFIC_SHA256 = "1464fa79d472f6755651774989b140659ee1f21daeb30032fc650a8770f4fdc4"
DAY = timedelta(days=1)
PHASES = ("plan", "preflight", "register", "hpo", "p1", "power", "final-fit",
          "calibration", "cost", "oof-errors", "ablation", "score", "analyze", "all")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def json_bytes(value):
    return (json.dumps(clean(value), ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode()


def hash_json(value):
    return sha256(json_bytes(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb" if exclusive else "wb") as stream:
        stream.write(json_bytes(value))


def csv_write(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise ValueError("A published or private result table cannot silently be empty")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("Result table rows have different schemas")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(clean(v), ensure_ascii=False)
                                 if isinstance(v, (dict, list, tuple)) else clean(v))
                             for k, v in row.items()})


def csv_read(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return [{k: (None if v == "" else v) for k, v in row.items()} for row in csv.DictReader(stream)]


def day(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Naive datetimes are not UTC calendar dates")
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def calendar(start, end):
    start, end = day(start), day(end)
    if end < start:
        raise ValueError("Calendar end precedes start")
    return tuple(start + i * DAY for i in range((end - start).days + 1))


def local(root, name):
    root = Path(root).resolve()
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Use a repository-relative metadata/source path")
    result = (root / p).resolve()
    if not result.is_relative_to(root) or (root / p).is_symlink():
        raise ValueError("Source path leaves the registered checkout")
    return result


def private_output(path, root=ROOT):
    output, root = Path(path).resolve(), Path(root).resolve()
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Experimental outputs must be outside the repository and its parents")
    return output


def model_slug(model):
    if model not in CATALOGUE:
        raise ValueError("Unknown benchmark model")
    return model.replace("/", "_")


def inner_blocks(origin):
    """Three contiguous 30-day inner blocks, ending at outer origin minus8."""
    origin = day(origin)
    return tuple({"origin": origin - i * DAY, "end": origin - (i - 29) * DAY}
                 for i in (97, 67, 37))


def is_neural(model):
    return model not in DETERMINISTIC


def seeds_for(model, seeds):
    return tuple(seeds) if is_neural(model) else ("deterministic",)


def fixed_candidate(model):
    if model not in ANCHORS + NAIVES:
        raise ValueError("Only an unchanged anchor or a naive rule has a fixed candidate")
    dropout = .1 if model.startswith("GRU/") else 0.0
    return {"model": model, "candidate_id": model_slug(model) + "-fixed",
            "parameters": ({"learning_rate": .003, "weight_decay": .0001,
                            "batch_size": 32, "dropout": dropout}
                           if model in ANCHORS else {"lag": 1 if model == "Persistence" else 7})}


def shuffled_control_candidate(candidate):
    """Keep the selected architecture/HPO parameters, under a distinct fit ID."""
    value = json.loads(json.dumps(candidate))
    value["candidate_id"] += "-shuffled-labels-control"
    return value


def read_design(path=DEFAULT_DESIGN, root=ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("The completion design must be a committed file inside this checkout")
    design = read_json(path)
    if not isinstance(design.get("protocol_id"), str) or not design["protocol_id"]:
        raise ValueError("A nonempty versioned protocol_id is required")
    if design.get("input_contract") != {"D": 7, "n": 7, "h": 1,
                                         "channels": list(FEATURES), "target_lags": False}:
        raise ValueError("The inherited D7×5, sigma7 and h1 contract must remain explicit")
    if tuple(design.get("catalogue", ())) != CATALOGUE:
        raise ValueError("The full declared twenty-configuration catalogue is required")
    hpo_path = local(root, design["hpo_config"])
    prereg_path = local(root, design["preregistration"])
    registry_path = local(root, design["candidate_registry"])
    hpo, prereg, candidates = map(read_json, (hpo_path, prereg_path, registry_path))
    if tuple(hpo["catalogue"]) != CATALOGUE or set(hpo["fairness"]["applies_to"]) != set(TUNABLE):
        raise ValueError("HPO must cover every one of the fifteen tunable recipes")
    if tuple(hpo["selection"]["seed_ids"]) != HPO_SEEDS:
        raise ValueError("The two predeclared paired HPO seeds differ")
    expected_selection = {"outer_folds": 10, "inner_folds": 3, "inner_block_days": 30,
                          "latest_training_target_offset_days": -8}
    if any(hpo["selection"].get(k) != v for k, v in expected_selection.items()):
        raise ValueError("Nested HPO calendar or embargo differs from the declared procedure")
    expected_optim = {"optimizer": "AdamW for DL only", "maximum_epochs": 100,
                      "patience": 12, "min_delta": 0.0, "gradient_clip_norm": 1.0,
                      "threads": 1, "inter_op_threads": 1,
                      "maximum_optimizer_fit_seconds": 900}
    if any(hpo["optimization"].get(k) != v for k, v in expected_optim.items()):
        raise ValueError("The registered maximum epochs/stopping/CPU controls differ")
    if prereg["inherited_contract"]["absolute_delta"] != DELTA:
        raise ValueError("The inherited practical RMSE margin changed")
    if prereg["power"]["maximum_R"] != 200 or prereg["power"]["target"] != .8:
        raise ValueError("Power target/cap must remain .80/200 with honest insufficiency reporting")
    if tuple(prereg["pilots"]["P1"]["seed_ids"]) != P1_SEEDS:
        raise ValueError("P1 fresh-seed registry differs")
    validate_candidates(candidates, hpo)
    if hpo.get("source_reference_candidates"):
        inherited = read_json(local(root, "book_e2/_static/remediation_protocol/candidate_registry.json"))
        old = {r["candidate_id"]: r for r in inherited}
        for row in candidates:
            if row.get("source_reference_point") is not True:
                previous = old.get(row["candidate_id"])
                if previous is None or row["parameters"] != previous["parameters"]:
                    raise ValueError("The amendment must preserve the original first eleven random points and all classical points")
    if set(design.get("development_sources", ())) != {
        "results/eda_development/daily_market_development.csv",
        "results/eda_development/daily_volatility_development.csv"}:
        raise ValueError("Exactly the two inherited development derivatives must be identified")
    return design, hpo, prereg, candidates


def validate_candidates(candidates, hpo):
    if not isinstance(candidates, list) or len(candidates) != 180:
        raise ValueError("Exactly 180 pre-generated candidates are required")
    seen = set()
    for model in TUNABLE:
        selected = [r for r in candidates if r.get("model") == model]
        if len(selected) != 12:
            raise ValueError("Every tunable model requires twelve explicit candidates")
        for row in selected:
            identity = row.get("candidate_id")
            if not isinstance(identity, str) or identity in seen or "/" in identity or ".." in identity:
                raise ValueError("Candidate identities must be unique and safe filenames")
            seen.add(identity)
            values = row.get("parameters", {})
            if model in TUNABLE_NEURAL:
                if set(values) != {"learning_rate", "weight_decay", "dropout", "batch_size"}:
                    raise ValueError("Each neural candidate must account for LR/WD/dropout/batch")
                source = hpo.get("source_reference_candidates", {}).get(model)
                is_source = row.get("source_reference_point") is True
                if is_source:
                    if (not isinstance(source, dict) or identity != source.get("candidate_id")
                            or not identity.endswith("-c12") or values != source.get("parameters")
                            or values["weight_decay"] != .0001 or values["batch_size"] != 32
                            or not source.get("source")):
                        raise ValueError("Only the exact hash-bound declared c12 source point may leave the random HPO ranges")
                elif (not 1e-4 <= values["learning_rate"] <= .003
                        or not 1e-6 <= values["weight_decay"] <= .01
                        or not 0 <= values["dropout"] <= .3
                        or values["batch_size"] not in (16, 32, 64)):
                    raise ValueError("Candidate lies outside the registered HPO axes")
                if source and identity == source.get("candidate_id") and not is_source:
                    raise ValueError("A declared source reference candidate must identify that role explicitly")
            elif model == "Ridge":
                if set(values) != {"alpha"} or not math.isfinite(values["alpha"]) or values["alpha"] <= 0:
                    raise ValueError("Ridge requires one positive alpha per candidate")
            else:
                p, d, q = values.get("order", (-1, -1, -1))
                if p not in (0, 1, 2) or d not in (0, 1) or q not in (0, 1):
                    raise ValueError("ARIMA candidates must be the complete registered order grid")
                if values.get("trend") != ("c" if d == 0 else "n"):
                    raise ValueError("ARIMA trend must be estimator-appropriate and fixed")
            if row.get("assigned_compute_seconds") != 172800:
                raise ValueError("Each tunable recipe has the same assigned172800 CPU-second cap")
    arima = [tuple(r["parameters"]["order"]) for r in candidates if r["model"] == "ARIMA"]
    if set(arima) != {(p, d, q) for p in (0, 1, 2) for d in (0, 1) for q in (0, 1)}:
        raise ValueError("The ARIMA twelve-candidate grid is not exhaustive")
    if len(seen) != 180:
        raise ValueError("The candidate registry has an unknown model or duplicate")
    source_points = hpo.get("source_reference_candidates")
    if source_points is not None:
        if set(source_points) != set(TUNABLE_NEURAL):
            raise ValueError("The amended source-reference allocation applies uniformly to all thirteen neural recipes")
        for model in TUNABLE_NEURAL:
            if sum(row.get("source_reference_point") is True for row in candidates if row["model"] == model) != 1:
                raise ValueError("Exactly one of the twelve neural candidates is the registered source reference point")


def planned_inventory():
    inner = sorted({block["origin"] for origin in OUTER_ORIGINS for block in inner_blocks(origin)})
    union = sorted(set(inner) | set(OUTER_ORIGINS))
    neural = len(TUNABLE_NEURAL) * 12 * len(HPO_SEEDS) * len(union)
    classical = 2 * 12 * len(union)
    anchor = len(ANCHORS) * len(HPO_SEEDS) * len(OUTER_ORIGINS)
    return {"catalogue_size": 20, "tunable_recipes": 15, "candidates_per_tunable": 12,
            "candidate_count": 180, "outer_origins": [d.isoformat() for d in OUTER_ORIGINS],
            "nested_inner_calendar": [{"outer_origin": origin.isoformat(), **clean(block)}
                                       for origin in OUTER_ORIGINS for block in inner_blocks(origin)],
            "unique_inner_origins": [d.isoformat() for d in inner],
            "unique_inner_plus_finalsearch_origins": [d.isoformat() for d in union],
            "cache_unique_HPO_neural_trajectories": neural,
            "cache_unique_HPO_classical_trajectories": classical,
            "fixed_anchor_outer_trajectories": anchor,
            "HPO_optimizer_fit_upper_bound_with_exact_caching": 2 * neural,
            "anchor_optimizer_fits": 2 * anchor,
            "label_shuffle_control_models": len(NEGATIVE_CONTROL_MODELS),
            "label_shuffle_control_optimizer_fits": 2 * (len(NEGATIVE_CONTROL_MODELS)-1),
            "label_shuffle_control_Ridge_estimator_fits": 1,
            "HPO_estimator_fit_upper_bound_with_exact_caching": classical,
            "assigned_CPU_seconds_per_tunable": 172800,
            "all_tunable_assigned_CPU_seconds": 15 * 172800,
            "maximum_seconds_per_optimizer_fit": 900,
            "counts_scope": "Prospective maximum scheduled work, not observed fits or runtime",
            "neural_same_seed_recipes_reused_only_by_exact_input_cache_key": True,
            "nested_outer_selection_is_closed_before_final_recipe_search": True}


def source_roots(path):
    result = read_json(path) if path else {}
    if not isinstance(result, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          for k, v in result.items()):
        raise ValueError("Source-roots JSON must map model names to private author checkout roots")
    return {k: str(Path(v).resolve()) for k, v in result.items()}


def environment():
    packages = {}
    for name in ("numpy", "pandas", "torch", "scipy", "statsmodels", "einops"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "system": platform.platform(),
            "packages": packages, "threads_per_fit": 1, "neural_precision": "FP32",
            "classical_precision": "float64", "CPU_affinity_count": len(os.sched_getaffinity(0)),
            "torch_deterministic_algorithms_set_in_worker": True}


def collect_inputs(design_path, design, roots, root=ROOT):
    root = Path(root).resolve()
    own = {Path(design_path).resolve().relative_to(root).as_posix(),
           design["hpo_config"], design["preregistration"], design["candidate_registry"],
           design["architecture_review"], "scripts/run_e2_final_benchmark.py",
           "scripts/dl_data.py", "scripts/dl_data_v2.py", "scripts/e2_temporal_baselines.py",
           "scripts/run_e2_remediation_pilot.py", "scripts/planned_study_data.py"}
    own.update(design.get("source_files", []))
    own.add("book_e2/_static/remediation_protocol/candidate_registry.json")
    own.add(design.get("error_taxonomy", "book_e2/_static/remediation_error_results/training_taxonomy.json"))
    for name in ("e2_final_models.py", "e2_final_models_mlp.py", "e2_final_models_attention.py",
                 "e2_final_statistics.py", "e2_final_test_data.py", "e2_final_uncertainty.py",
                 "e2_final_diagnostics.py", "e2_compute_profile.py", "e2_predictability_reference.py"):
        own.add("scripts/" + name)
    own.add("scripts/e2_leakage_controls.py")
    own.add("scripts/e2_hpo_report.py")
    source = {name: digest(local(root, name)) for name in sorted(own)}
    development = {}
    for name, expected in design["development_sources"].items():
        actual = digest(local(root, name))
        if actual != expected:
            raise ValueError("The inherited development source hash differs: " + name)
        development[name] = actual
    authors = []
    for record in design.get("author_sources", []):
        base = Path(roots[record["model"]]).resolve()
        p = local(base, record["path"])
        actual = digest(p)
        if actual != record["sha256"]:
            raise ValueError("Pinned author implementation changed: " + record["model"])
        authors.append({"model": record["model"], "path": record["path"], "sha256": actual})
    if not authors:
        raise ValueError("Author sources must be hash-bound, not inferred from a mutable path")
    return {"source_files": source, "development_sources": development,
            "author_sources": authors, "inherited_scientific_sha256": E1_SCIENTIFIC_SHA256}


def git_identity(root=ROOT):
    root = Path(root)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Register the completion protocol from a clean committed checkout")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def verify_architecture_review(design, root=ROOT):
    review = read_json(local(root, design["architecture_review"]))
    models = review.get("models", {})
    if isinstance(models, list):
        models = {row.get("model", row.get("id")): row for row in models}
    if review.get("status") != "passed" or set(TOP10) - set(models):
        raise ValueError("Every final Top10 adaptation needs an independent architectural review")
    unresolved = [m for m in TOP10 if models[m].get("status") != "passed"]
    if unresolved:
        raise ValueError("Final architectural review not passed: " + ", ".join(unresolved))
    reviewed = review.get("reviewed_files", {})
    if not {"scripts/e2_final_models_mlp.py", "scripts/e2_final_models_attention.py"}.issubset(reviewed):
        raise ValueError("Architectural review must identify the exact two reviewed final-adapter sources")
    for name, expected in reviewed.items():
        if digest(local(root, name)) != expected:
            raise ValueError("Architectural review is stale for the current implementation: " + name)
    return review


def phase_receipt(path, *, expected_inputs=None, required_status="passed"):
    path = Path(path)
    receipt = read_json(path)
    if receipt.get("status") != required_status:
        raise ValueError("Required phase has not passed: " + str(path))
    if expected_inputs is not None and receipt.get("registration_inputs_sha256") != expected_inputs:
        raise ValueError("Phase belongs to a different registered input identity")
    levels = receipt.get("artifact_root_levels_above_phase", 0)
    if type(levels) is not int or not 0 <= levels <= 3:
        raise ValueError("Invalid phase artifact-root depth")
    artifact_root = path.parent
    for _ in range(levels):
        artifact_root = artifact_root.parent
    if levels and not (artifact_root/"registration.json").is_file():
        raise ValueError("Cross-phase artifacts must remain inside their registered output")
    for name, expected in receipt.get("artifacts", {}).items():
        p = local(artifact_root, name)
        if not p.is_file() or digest(p) != expected:
            raise ValueError("Phase artifact changed after closure: " + name)
    return receipt


def close_phase(path, inputs_hash, *, artifacts=(), **metadata):
    path = Path(path)
    artifacts = list(map(Path, artifacts))
    artifact_root = Path(os.path.commonpath([str(path.parent), *(str(p.parent) for p in artifacts)]))
    levels = len(path.parent.relative_to(artifact_root).parts)
    if levels and not (artifact_root/"registration.json").is_file():
        raise ValueError("Cross-phase closure must stay inside the registered experimental output")
    value = {"status": "passed", "completed_at_utc": stamp(),
             "registration_inputs_sha256": inputs_hash,
             "artifact_root_levels_above_phase": levels,
             "artifacts": {p.relative_to(artifact_root).as_posix(): digest(p) for p in artifacts},
             **metadata}
    write_json(path, value)
    return value


def register(args, context):
    output = context["output"]
    record_path = output / "registration.json"
    inputs_hash = hash_json(context["inputs"])
    if record_path.exists():
        existing = read_json(record_path)
        if existing["immutable_inputs"] != context["inputs"]:
            raise ValueError("Registered source/data/architecture identity changed; amendment and new output required")
        return existing
    if any(output.rglob("started.json")) or any(output.rglob("checkpoint.pt")) or any(output.rglob("model.npz")):
        raise ValueError("Cannot register after a dataset fit already exists in this output")
    commit = git_identity(context["root"])
    for name, checksum in context["inputs"]["source_files"].items():
        saved = subprocess.check_output(["git", "show", commit + ":" + name], cwd=context["root"])
        if sha256(saved).hexdigest() != checksum:
            raise ValueError("A required source is not part of the registered Git commit: " + name)
    verify_architecture_review(context["design"], context["root"])
    preflight = phase_receipt(output / "preflight/completion.json", expected_inputs=inputs_hash)
    if preflight.get("all_twenty_configurations_preflighted") is not True:
        raise ValueError("Complete twenty-configuration preflight is required before registration")
    from planned_study_data import reserve_metadata
    from e2_final_test_data import reserve_final_state_metadata
    reserved = reserve_metadata(context["root"], args.raw_root)
    state = reserve_final_state_metadata(context["root"], args.raw_root)
    write_json(output / "test_reservation.json", reserved)
    write_json(output / "state_history_reservation.json", state)
    record = {"registered_at_utc": stamp(), "source_commit": commit,
              "protocol_id": context["design"]["protocol_id"], "immutable_inputs": context["inputs"],
              "registration_inputs_sha256": inputs_hash, "dataset_fits_before_registration": 0,
              "known_historical_test": True, "new_blind_test_claimed": False,
              "test_prices_or_labels_decoded": False, "environment": environment(),
              "preflight_receipt_sha256": digest(output / "preflight/completion.json"),
              "test_reservation_sha256": digest(output / "test_reservation.json"),
              "state_history_reservation_sha256": digest(output / "state_history_reservation.json")}
    write_json(record_path, record)
    return record


def check_registration(context):
    record = read_json(context["output"] / "registration.json")
    if record["immutable_inputs"] != context["inputs"]:
        raise ValueError("Inputs no longer match the preregistered completion source/data")
    if record["registration_inputs_sha256"] != hash_json(context["inputs"]):
        raise ValueError("Registration digest is inconsistent")
    observed = environment()
    if any(record["environment"].get(key) != observed.get(key)
           for key in ("python", "system", "packages")):
        raise ValueError("The registered runtime or package versions changed; amend prospectively instead of mixing environments")
    preflight_path = context["output"]/"preflight/completion.json"
    if digest(preflight_path) != record["preflight_receipt_sha256"]:
        raise ValueError("The preregistered model/ablation preflight receipt changed")
    phase_receipt(preflight_path, expected_inputs=record["registration_inputs_sha256"])
    for name, key in [("test_reservation.json", "test_reservation_sha256"),
                      ("state_history_reservation.json", "state_history_reservation_sha256")]:
        if digest(context["output"] / name) != record[key]:
            raise ValueError("Reserved data identity changed after registration")
    return record


def configure(seed):
    import numpy as np
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)


@contextmanager
def wallclock_cap(seconds):
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("Hard per-fit wallclock caps require Unix interval timers")
    def timeout_handler(signum, frame):
        raise TimeoutError("Registered optimizer-fit wallclock limit reached")
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def development(last_outcome, root=ROOT):
    from run_e2_remediation_pilot import load_development_prefix
    return load_development_prefix(root, day(last_outcome).isoformat())


def split_indices(data, origin):
    """Generic 2023/2024 inner origin with the unchanged seven-day embargo."""
    import numpy as np
    dates = tuple(day(x) for x in data["calendar"])
    origin = day(origin)
    if origin < date(2023, 7, 1) or origin > date(2025, 1, 1):
        raise ValueError("A training origin lies outside the registered development procedure")
    eligible = np.asarray(data["eligible_by_history"][7], dtype=bool)
    train_max, valid_start = origin - 8 * DAY, origin - 90 * DAY
    inner_max = origin - 98 * DAY
    train = np.array([i for i, d in enumerate(dates) if eligible[i] and d <= train_max], dtype=np.int64)
    inner = np.array([i for i, d in enumerate(dates) if eligible[i] and d <= inner_max], dtype=np.int64)
    valid = np.array([i for i, d in enumerate(dates) if eligible[i] and valid_start <= d <= train_max], dtype=np.int64)
    if min(len(train), len(inner), len(valid)) == 0:
        raise ValueError("Training, early-stop training and validation must all be nonempty")
    if set(inner) & set(valid):
        raise ValueError("Inner training and validation share outcomes")
    return {"train": train, "inner_train": inner, "inner_validation": valid,
            "metadata": {"origin": origin.isoformat(), "fit_outcome_max": train_max.isoformat(),
                         "inner_fit_outcome_max": inner_max.isoformat(),
                         "inner_validation_first": dates[valid.min()].isoformat(),
                         "inner_validation_last": dates[valid.max()].isoformat(),
                         "train_count": len(train), "inner_train_count": len(inner),
                         "inner_validation_count": len(valid), "embargo_days": 7,
                         "train_indices_sha256": sha256(train.astype("<i8").tobytes()).hexdigest(),
                         "inner_indices_sha256": sha256(inner.astype("<i8").tobytes()).hexdigest(),
                         "validation_indices_sha256": sha256(valid.astype("<i8").tobytes()).hexdigest()}}


def evaluation_indices(data, start, end):
    import numpy as np
    dates = tuple(day(x) for x in data["calendar"])
    start, end = day(start), day(end)
    selected = np.array([i for i, d in enumerate(dates) if start <= d <= end], dtype=np.int64)
    if len(selected) != len(calendar(start, end)):
        raise ValueError("Evaluation calendar cannot be compressed")
    if not np.asarray(data["eligible_by_history"][7], bool)[selected].all():
        raise ValueError("Ineligible evaluation date: freeze and audit the full common cohort, never silently drop it")
    return selected


def preprocess_x(data, indices, preprocessing):
    """Apply a frozen development scaler to either development or reserve windows."""
    import numpy as np
    from dl_data_v2 import relative_features
    raw = np.asarray(data["windows_by_history"][7])[indices]
    x = ((relative_features(raw) - np.asarray(preprocessing["x_mean"])) /
         np.asarray(preprocessing["x_scale"])).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Frozen preprocessing produced nonfinite inputs")
    return x


def metrics(truth, prediction):
    import numpy as np
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.shape != prediction.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Metrics require aligned nonempty daily vectors")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all() or min(truth.min(), prediction.min()) < 0:
        raise ValueError("Original-scale volatility outcomes and forecasts must be finite and nonnegative")
    err = prediction - truth
    denominator = float(((truth - truth.mean())**2).sum())
    nz = truth != 0
    return {"RMSE": float(np.sqrt(np.mean(err**2))), "MSE": float(np.mean(err**2)),
            "MAE": float(np.mean(np.abs(err))),
            "R2": float(1 - np.sum(err**2)/denominator) if denominator else None,
            "MAPE_percent": float(100*np.mean(np.abs(err[nz]/truth[nz]))) if nz.any() else None,
            "MAPE_coverage": float(nz.mean()), "n_origins": len(truth)}


def optimizer_groups(model, weight_decay):
    ordinary, exempt = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (exempt if getattr(parameter, "_no_weight_decay", False) else ordinary).append(parameter)
    groups = []
    if ordinary:
        groups.append({"params": ordinary, "weight_decay": float(weight_decay)})
    if exempt:
        groups.append({"params": exempt, "weight_decay": 0.0})
    return groups, {"ordinary_parameters": sum(p.numel() for p in ordinary),
                    "source_no_weight_decay_parameters": sum(p.numel() for p in exempt),
                    "policy": "Honor explicit source _no_weight_decay flags; candidate WD for every other trainable parameter"}


def build_neural(model, candidate, roots):
    from e2_final_models import build_model
    return build_model(model, candidate=candidate.get("parameters", candidate), source_roots=roots)


def train_neural(model_name, candidate, roots, x, y, seed, optimization, *,
                 x_validation=None, y_validation=None, epochs=None, min_delta=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    configure(seed)
    model = build_neural(model_name, candidate, roots)
    model._e2_fit_started = True
    params = candidate["parameters"]
    groups, group_receipt = optimizer_groups(model, params["weight_decay"])
    optimizer = torch.optim.AdamW(groups, lr=params["learning_rate"])
    generator = torch.Generator().manual_seed(int(seed) + 17)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=int(params["batch_size"]), shuffle=True, drop_last=False,
                        num_workers=0, generator=generator)
    max_epochs = int(epochs or optimization["maximum_epochs"])
    patience = int(optimization["patience"])
    min_delta = float(optimization["min_delta"] if min_delta is None else min_delta)
    curve, selected, best, stale, best_state = [], None, float("inf"), 0, None
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    with wallclock_cap(optimization["maximum_optimizer_fit_seconds"]):
        for epoch in range(1, max_epochs + 1):
            model.train()
            total, count = 0.0, 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                forecast = model(batch_x)
                if forecast.shape != batch_y.shape:
                    raise ValueError("Neural model must produce one scalar per input window")
                loss = (forecast - batch_y).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss; the attempt is retained as a failure")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip_norm"])
                optimizer.step()
                total += float(loss.detach()) * len(batch_y)
                count += len(batch_y)
            validation = None
            if x_validation is not None:
                model.eval()
                with torch.no_grad():
                    residual = model(torch.from_numpy(x_validation)) - torch.from_numpy(y_validation)
                    validation = float(residual.square().mean())
                if not math.isfinite(validation):
                    raise FloatingPointError("Nonfinite inner-validation loss")
                if validation < best - min_delta:
                    best, selected, stale = validation, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
            curve.append({"epoch": epoch, "online_training_MSE_scaled": total/count,
                          "validation_MSE_scaled": validation,
                          "elapsed_seconds": time.perf_counter()-started_wall})
            if x_validation is not None and stale >= patience:
                break
    if x_validation is None:
        selected = max_epochs
    elif selected is None or best_state is None:
        raise RuntimeError("No finite early-stop epoch exists")
    else:
        model.load_state_dict(best_state)
    model.eval()
    training_inference_started = stamp()
    with torch.no_grad():
        training_prediction = model(torch.from_numpy(x)).numpy().astype(float)
        validation_prediction = (model(torch.from_numpy(x_validation)).numpy().astype(float)
                                 if x_validation is not None else None)
    if not np.isfinite(training_prediction).all() or np.any(training_prediction < 0):
        raise FloatingPointError("The fitted neural predictor is not finite/nonnegative")
    return {"model": model, "selected_epoch": selected, "curve": curve,
            "wall_seconds": time.perf_counter()-started_wall,
            "CPU_seconds": time.process_time()-started_cpu,
            "training_prediction_scaled": training_prediction,
            "validation_prediction_scaled": validation_prediction,
            "training_inference_started_at_utc": training_inference_started,
            "optimizer_groups": group_receipt}


def ridge_fit(x, y, alpha):
    """Float64 primal ridge with an unpenalized intercept, no sklearn dependency."""
    import numpy as np
    x, y = np.asarray(x, float).reshape(len(x), -1), np.asarray(y, float)
    mean_x, mean_y = x.mean(axis=0), float(y.mean())
    u, singular, vt = np.linalg.svd(x-mean_x, full_matrices=False)
    coef = vt.T @ ((singular/(singular**2+float(alpha))) * (u.T @ (y-mean_y)))
    return {"coef": coef, "intercept": mean_y-float(mean_x @ coef), "alpha": float(alpha)}


def _checkpoint_artifacts(directory, names):
    return {name: digest(Path(directory)/name) for name in names}


def verify_job(directory, inputs_hash, identity=None):
    directory = Path(directory)
    record = read_json(directory / "completion.json")
    if record.get("registration_inputs_sha256") != inputs_hash:
        raise ValueError("Cached fit belongs to another registered protocol")
    if identity is not None and record.get("identity") != identity:
        raise ValueError("Cached candidate/seed/origin/support identity changed")
    if record.get("status") not in ("passed", "failed"):
        raise ValueError("A fit directory is not a terminal recorded attempt")
    pruning = None
    pruning_path = directory/"checkpoint_pruned_after_development_scoring.json"
    if pruning_path.exists():
        pruning = read_json(pruning_path)
        output = directory.parents[4]
        score_receipt = local(output, pruning["scoring_receipt_relative_to_output"])
        if (pruning.get("original_fit_receipt_sha256") != digest(directory/"completion.json")
                or pruning.get("registration_inputs_sha256") != inputs_hash
                or digest(score_receipt) != pruning["closed_scoring_receipt_sha256"]):
            raise ValueError("Checkpoint pruning lacks its unchanged original fit and closed scoring receipts")
        phase_receipt(score_receipt, expected_inputs=inputs_hash)
    for name, expected in record.get("artifacts", {}).items():
        path = local(directory, name)
        if (not path.exists() and pruning is not None and name in ("checkpoint.pt", "model.pickle")
                and pruning.get("pruned_artifact_sha256", {}).get(name) == expected):
            continue
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Frozen fit artifact changed: " + name)
    return record


def job_identity(context, model, candidate, seed, origin):
    origin = day(origin)
    parameters = candidate["parameters"]
    return {"model": model, "candidate_id": candidate["candidate_id"],
            "candidate_sha256": hash_json(candidate), "seed": seed,
            "origin": origin.isoformat(), "fit_outcome_max": (origin-8*DAY).isoformat(),
            "development_sources": context["inputs"]["development_sources"],
            "registration_inputs_sha256": hash_json(context["inputs"]),
            "preprocessing": "D7relativeOHLCV/train-onlychannelmoments/ymeanpositive",
            "parameters": parameters}


def job_folder(output, model, candidate, seed, origin):
    return Path(output)/"fits"/model_slug(model)/candidate["candidate_id"]/day(origin).isoformat()/f"seed_{seed}"


def validate_requested_fit(context, model, candidate, seed, origin, budget_scope):
    """A worker cannot invent a candidate, phase, seed or calendar from its payload."""
    if model not in CATALOGUE or candidate.get("model") != model:
        raise ValueError("Worker candidate model differs from the registered catalogue")
    identity = candidate.get("candidate_id", "")
    if not identity or "/" in identity or ".." in identity:
        raise ValueError("Unsafe candidate identity")
    origin = day(origin)
    if budget_scope == "ablation_development":
        schedule = read_json(context["output"]/"ablation/schedule.json")
        if (schedule["model"] != model or candidate not in schedule["variants"].values()
                or seed not in schedule["seeds"] or origin != P1_ORIGIN):
            raise ValueError("Ablation worker differs from its frozen development-selected schedule")
        return
    if budget_scope == "negative_controls":
        recipes = read_json(context["output"]/"hpo/primary_recipes.json")
        expected = shuffled_control_candidate(recipes[model]["candidate"])
        if (model not in NEGATIVE_CONTROL_MODELS or candidate != expected
                or seed not in seeds_for(model, (NEGATIVE_CONTROL_SEED,))
                or origin != date(2025, 1, 1)):
            raise ValueError("Label-shuffle control differs from its frozen primary recipe or fresh seed")
        return
    allowed = context["candidates"] if model in TUNABLE else [fixed_candidate(model)]
    if candidate not in allowed:
        raise ValueError("The worker candidate is not one of the immutable pre-generated/fixed recipes")
    if budget_scope == "HPO":
        origins = {b["origin"] for o in OUTER_ORIGINS for b in inner_blocks(o)} | set(OUTER_ORIGINS)
        valid = model in TUNABLE and origin in origins and seed in seeds_for(model, HPO_SEEDS)
    elif budget_scope == "fixed_anchor_development":
        valid = model in ANCHORS+NAIVES and origin in OUTER_ORIGINS and seed in seeds_for(model, HPO_SEEDS)
    elif budget_scope == "P1_fits":
        valid = model in P1_MODELS and origin == P1_ORIGIN and seed in P1_SEEDS
    elif budget_scope in ("primary_final", "branch_A"):
        R = int(read_json(context["output"]/"power/selection.json")["selected_R"])
        phase_seeds = A_SEEDS if budget_scope == "branch_A" else tuple(context["prereg"]["power"]["final_seed_ids_at_cap"][:R])
        recipes = read_json(context["output"]/("hpo/branch_A_recipes.json" if budget_scope == "branch_A" else "hpo/primary_recipes.json"))
        valid = (origin == (P1_ORIGIN if budget_scope == "branch_A" else date(2025, 1, 1))
                 and seed in seeds_for(model, phase_seeds) and candidate == recipes[model]["candidate"])
    else:
        valid = False
    if not valid:
        raise ValueError("Worker seed, phase or origin is outside the registered fit procedure")


def fit_trial(context, model, candidate, seed, origin, *, budget_scope=None):
    """Fit a two-stage neural or one-stage classical trajectory, without scoring."""
    import numpy as np
    import torch
    from dl_data_v2 import fit_relative_preprocessing, transform_relative_windows
    from dl_data import mase_scale
    check_registration(context)
    validate_requested_fit(context, model, candidate, seed, origin, budget_scope)
    identity = job_identity(context, model, candidate, seed, origin)
    inputs_hash = identity["registration_inputs_sha256"]
    folder = job_folder(context["output"], model, candidate, seed, origin)
    if folder.exists():
        if not (folder/"completion.json").exists():
            raise RuntimeError("Interrupted fit requires an infrastructure audit; no silent optimizer restart")
        return verify_job(folder, inputs_hash, identity)
    if (context["output"]/"final_scoring/started.json").exists():
        raise ValueError("No new model fit is permitted after the single final scoring phase starts")
    folder.mkdir(parents=True)
    write_json(folder/"started.json", {"started_at_utc": stamp(), "identity": identity})
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    try:
        data = development(day(origin)-8*DAY, context["root"])
        split = split_indices(data, origin)
        preprocessing = fit_relative_preprocessing(data, 7, split["train"])
        xt, yt = transform_relative_windows(data, 7, split["train"], preprocessing)
        label_permutations = {}
        if budget_scope == "negative_controls":
            from e2_leakage_controls import shuffled_training_targets
            yt, label_permutations["outer"] = shuffled_training_targets(
                yt, [day(data["calendar"][i]).isoformat() for i in split["train"]],
                seed=NEGATIVE_CONTROL_SEED, block_days=7)
        explanation_baseline = xt.astype(np.float64).mean(axis=0)
        preprocessing.update({"explanation_baseline_standardized_by_cell": explanation_baseline.tolist(),
                              "explanation_baseline_sha256": hash_json(explanation_baseline),
                              "explanation_baseline_training_index_sha256": preprocessing["fit_outcome_indices_sha256"],
                              "explanation_baseline_source": "mean of the35 standardized feature cells across admitted training windows only; no test or calibration fitting"})
        from e2_final_diagnostics import fit_explanation_baseline, preprocessing_fingerprint
        scaler_fingerprint = preprocessing_fingerprint(preprocessing)
        preprocessing["scaler_sha256"] = scaler_fingerprint
        preprocessing["explanation_baseline_receipt"] = fit_explanation_baseline(
            xt, training_outcome_dates=[day(data["calendar"][i]).isoformat() for i in split["train"]],
            maximum_training_outcome=identity["fit_outcome_max"], scaler_sha256=scaler_fingerprint)
        metadata = {"model": model, "candidate": candidate, "seed": seed,
                    "fit_origin": day(origin).isoformat(), "fit_outcome_max": identity["fit_outcome_max"],
                    "scaler_fit_outcome_max": preprocessing["last_fit_outcome_day_utc"],
                    "configuration_selection_outcome_max": "2024-10-31" if model in ANCHORS else identity["fit_outcome_max"],
                    "configuration_selection_kind": "inherited_fixed_E1_recipe" if model in ANCHORS else "registered_candidate_or_causal_rule",
                    "historical_selection_exposure_disclosed": model in ANCHORS,
                    "epoch_selection_outcome_max": identity["fit_outcome_max"],
                    "E2_new_configuration_selection": model not in ANCHORS,
                    "n_train": len(split["train"]), "mase_scale": float(mase_scale(data, split["train"])),
                    "test_data_decoded": False, "preprocessing_sha256": hash_json(preprocessing)}
        if budget_scope == "negative_controls":
            metadata.update({"negative_control": True, "label_permutation": label_permutations,
                             "original_training_target_mean": float(np.mean(data["targets"][split["train"]])),
                             "control_training_labels_are_permuted": True,
                             "future_evaluation_labels_are_not_permuted": True,
                             "reported_original_label_training_metrics_scope": "descriptive evaluation of the label-permuted predictor, not its optimization objective"})
        names = ["started.json", "setup.json", "preprocessing.json", "split.json"]
        if model in NAIVES:
            write_json(folder/"rule.json", {"lag": candidate["parameters"]["lag"],
                                           "causal_calendar_rule": True,
                                           "coefficient_estimation": False})
            metadata.update({"optimizer_fits": 0, "estimator_fits": 0,
                             "parameters_total": 0, "parameters_trainable": 0,
                             "selected_epoch": None, "deterministic_rule": True})
            names.append("rule.json")
        elif is_neural(model):
            inner_pre = fit_relative_preprocessing(data, 7, split["inner_train"])
            xi, yi = transform_relative_windows(data, 7, split["inner_train"], inner_pre)
            xv, yv = transform_relative_windows(data, 7, split["inner_validation"], inner_pre)
            if budget_scope == "negative_controls":
                yi, label_permutations["inner_train"] = shuffled_training_targets(
                    yi, [day(data["calendar"][i]).isoformat() for i in split["inner_train"]],
                    seed=690118, block_days=7)
                yv, label_permutations["inner_validation"] = shuffled_training_targets(
                    yv, [day(data["calendar"][i]).isoformat() for i in split["inner_validation"]],
                    seed=690132, block_days=7)
            min_delta = 1e-5 if model in ANCHORS else 0.0
            inner = train_neural(model, candidate, context["roots"], xi, yi, int(seed),
                                 context["hpo"]["optimization"], x_validation=xv,
                                 y_validation=yv, min_delta=min_delta)
            csv_write(folder/"history_inner.csv", inner["curve"])
            inner_scale = inner_pre["y_scale"]
            state_hash = sha256()
            for name, tensor in sorted(inner["model"].state_dict().items()):
                state_hash.update(json_bytes({"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)}))
                state_hash.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            metadata["selected_inner_state_sha256"] = state_hash.hexdigest()
            metadata["inner_checkpoint_stored"] = False
            metadata["inner_train_metrics"] = metrics(data["targets"][split["inner_train"]], inner["training_prediction_scaled"]*inner_scale)
            metadata["inner_validation_metrics"] = metrics(data["targets"][split["inner_validation"]], inner["validation_prediction_scaled"]*inner_scale)
            np.savez_compressed(folder/"inner_evaluation.npz",
                                train_indices=split["inner_train"], validation_indices=split["inner_validation"],
                                train_truth=data["targets"][split["inner_train"]],
                                train_prediction=inner["training_prediction_scaled"]*inner_scale,
                                validation_truth=data["targets"][split["inner_validation"]],
                                validation_prediction=inner["validation_prediction_scaled"]*inner_scale)
            metadata["inference_mode_validation_minus_train_RMSE"] = (
                metadata["inner_validation_metrics"]["RMSE"]-metadata["inner_train_metrics"]["RMSE"])
            selected_epoch = inner["selected_epoch"]
            del inner["model"]
            refit = train_neural(model, candidate, context["roots"], xt, yt,
                                 int(seed)+10000000, context["hpo"]["optimization"],
                                 epochs=selected_epoch, min_delta=min_delta)
            torch.save(refit["model"].state_dict(), folder/"checkpoint.pt")
            csv_write(folder/"history_refit.csv", refit["curve"])
            metadata.update({"selected_epoch": selected_epoch, "refit_seed": int(seed)+10000000,
                             "optimizer_fits": 2, "estimator_fits": 0,
                             "optimizer_groups": refit["optimizer_groups"],
                             "training_inference_started_at_utc": refit["training_inference_started_at_utc"],
                             "architecture": getattr(refit["model"], "architecture_metadata", {}),
                             "parameters_total": sum(p.numel() for p in refit["model"].parameters()),
                             "parameters_trainable": sum(p.numel() for p in refit["model"].parameters() if p.requires_grad),
                             "full_training_metrics": metrics(data["targets"][split["train"]],
                                                              refit["training_prediction_scaled"]*preprocessing["y_scale"]),
                             "inner_preprocessing": inner_pre})
            names += ["inner_evaluation.npz", "checkpoint.pt", "history_inner.csv", "history_refit.csv"]
        elif model == "Ridge":
            fitted = ridge_fit(xt, yt, candidate["parameters"]["alpha"])
            np.savez(folder/"model.npz", **fitted)
            metadata.update({"optimizer_fits": 0, "estimator_fits": 1, "parameters_total": 36,
                             "parameters_trainable": 36, "selected_epoch": None,
                             "classical_dtype": "float64", "point_link": "max(0,raw)"})
            names.append("model.npz")
        elif model == "ARIMA":
            from e2_temporal_baselines import FixedParameterARIMA, FitWindow, ArimaConfig
            history = {day(d): float(y) if np.isfinite(y) else None for d, y in
                       zip(data["calendar"], data["targets"]) if day(d) <= day(origin)-8*DAY}
            config = ArimaConfig(tuple(candidate["parameters"]["order"]))
            with wallclock_cap(context["hpo"]["optimization"]["maximum_optimizer_fit_seconds"]):
                estimator = FixedParameterARIMA(history, FitWindow(day(data["calendar"][0]),
                                                   day(origin)-8*DAY, day(origin)), config)
            (folder/"model.pickle").write_bytes(pickle.dumps(estimator, protocol=5))
            metadata.update({"optimizer_fits": 0, "estimator_fits": 1,
                             "parameters_total": len(estimator._result.params),
                             "parameters_trainable": len(estimator._result.params),
                             "selected_epoch": None, "classical_dtype": "float64",
                             "arima_receipt": estimator.receipt(), "point_link": "max(0,raw)"})
            metadata["arima_training_sigma_mean"] = estimator.target_scale
            names.append("model.pickle")
        else:
            raise ValueError("Naive rules are sealed as rules, never counted as fitted estimators")
        if budget_scope in ("primary_final", "branch_A"):
            if is_neural(model):
                train_prediction = refit["training_prediction_scaled"]*preprocessing["y_scale"]
            else:
                metadata["training_inference_started_at_utc"] = stamp()
                if model == "Ridge":
                    train_prediction = np.maximum(0., xt.astype(float).reshape(len(xt), -1)@fitted["coef"]+fitted["intercept"])*preprocessing["y_scale"]
                elif model == "ARIMA":
                    fitted_values = np.asarray(estimator._result.fittedvalues, float)*estimator.target_scale
                    by_day = dict(zip(estimator.training_dates, fitted_values))
                    train_prediction = np.maximum(0., np.asarray([by_day[day(data["calendar"][i])] for i in split["train"]]))
                else:
                    from e2_temporal_baselines import naive_forecast
                    history = {day(d): float(y) if np.isfinite(y) else None for d, y in zip(data["calendar"], data["targets"])}
                    values = [naive_forecast(history, day(data["calendar"][i]), lag=candidate["parameters"]["lag"])
                              for i in split["train"]]
                    train_prediction = np.asarray([f.value if f.eligible else np.nan for f in values], float)
            np.savez_compressed(folder/"training_evaluation.npz", train_indices=split["train"],
                                train_truth=data["targets"][split["train"]], train_prediction=train_prediction)
            names.append("training_evaluation.npz")
            metadata["training_evaluation_uses_same_refit_checkpoint"] = True
            metadata["training_evaluation_does_not_refit_or_use_future_labels"] = True
        metadata["CPU_seconds"] = time.process_time()-started_cpu
        metadata["wall_seconds"] = time.perf_counter()-started_wall
        metadata["peak_RSS_bytes_process"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
        write_json(folder/"setup.json", metadata)
        write_json(folder/"preprocessing.json", preprocessing)
        write_json(folder/"split.json", {**split["metadata"], "train_indices": split["train"],
                                         "inner_train_indices": split["inner_train"],
                                         "inner_validation_indices": split["inner_validation"]})
        record = {"status": "passed", "completed_at_utc": stamp(), "identity": identity,
                  "budget_scope": budget_scope,
                  "registration_inputs_sha256": inputs_hash,
                  "CPU_seconds": metadata["CPU_seconds"], "wall_seconds": metadata["wall_seconds"],
                  "optimizer_fits": metadata["optimizer_fits"], "estimator_fits": metadata["estimator_fits"],
                  "evaluation_predictions_computed": False,
                  "artifacts": _checkpoint_artifacts(folder, names)}
    except Exception as error:
        # The failure is terminal, with no new seed or alternative candidate substituted.
        record = {"status": "failed", "completed_at_utc": stamp(), "identity": identity,
                  "budget_scope": budget_scope,
                  "registration_inputs_sha256": inputs_hash,
                  "CPU_seconds": time.process_time()-started_cpu,
                  "wall_seconds": time.perf_counter()-started_wall,
                  "error_type": type(error).__name__, "error": str(error),
                  "optimizer_fits": None, "estimator_fits": None,
                  "evaluation_predictions_computed": False,
                  "artifacts": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}}
    write_json(folder/"completion.json", record)
    return record




def prediction_from_fit(context, folder, data, indices, *, history=None):
    """Consume one hash-verified, already-closed model; never update coefficients."""
    import numpy as np
    import torch
    folder = Path(folder)
    record = verify_job(folder, hash_json(context["inputs"]))
    if record["status"] != "passed":
        raise ValueError("A failed candidate cannot produce a substitute forecast")
    setup, preprocessing = read_json(folder/"setup.json"), read_json(folder/"preprocessing.json")
    model = setup["model"]
    raw, elapsed = [], []
    if is_neural(model):
        predictor = build_neural(model, setup["candidate"], context["roots"])
        predictor.load_state_dict(torch.load(folder/"checkpoint.pt", map_location="cpu", weights_only=True))
        predictor._e2_fit_started = True
        predictor.eval()
        torch.set_num_threads(1)
        x = preprocess_x(data, indices, preprocessing)
        start = time.perf_counter()
        with torch.no_grad():
            # Evaluation is deliberately pointwise, avoiding a batch-selected
            # temporal mechanism changing its result when the cohort changes.
            output = [float(predictor(torch.from_numpy(x[i:i+1]))[0]) for i in range(len(x))]
        raw = np.asarray(output, float)*preprocessing["y_scale"]
        elapsed = [1000*(time.perf_counter()-start)/len(x)]*len(x)
    elif model == "Ridge":
        x = preprocess_x(data, indices, preprocessing).astype(float).reshape(len(indices), -1)
        with np.load(folder/"model.npz", allow_pickle=False) as saved:
            coef, intercept = saved["coef"], float(saved["intercept"])
        for row in x:
            start = time.perf_counter()
            raw.append(float(row@coef+intercept)*preprocessing["y_scale"])
            elapsed.append(1000*(time.perf_counter()-start))
        raw = np.asarray(raw)
    else:
        from e2_temporal_baselines import naive_forecast, benchmark_nonnegative
        dates = [day(data["calendar"][i]) for i in indices]
        if history is None:
            history = {day(d): (float(y) if np.isfinite(y) else None)
                       for d, y in zip(data["calendar"], data["targets"])}
        if model == "ARIMA":
            # Only this private, locally generated, hash-bound pickle is loaded.
            estimator = pickle.loads((folder/"model.pickle").read_bytes())
            accessor = lambda d: benchmark_nonnegative(estimator.predict(history, d))
        else:
            accessor = lambda d: naive_forecast(history, d, lag=setup["candidate"]["parameters"]["lag"])
        for d in dates:
            start = time.perf_counter()
            result = accessor(d)
            elapsed.append(1000*(time.perf_counter()-start))
            if not result.eligible or result.value is None:
                raise ValueError("The frozen common cohort contains an unavailable endogenous reference")
            raw.append(float(result.raw_value if result.raw_value is not None else result.value))
        raw = np.asarray(raw)
    values = np.maximum(0., raw) if model in ("Ridge", "ARIMA") else np.asarray(raw, float)
    if values.shape != (len(indices),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("A nonfinite/negative output is a retained failure, not a removed date")
    return values, np.asarray(raw), elapsed


def scored_rows(context, model, candidate, seed, origin, start, end, *, stage, selection_outcome_max=None):
    """Score development after this exact fit is frozen; no test path is admitted."""
    start, end, origin = day(start), day(end), day(origin)
    if model in ANCHORS:
        selection_outcome_max = date(2024, 10, 31)
    if end > date(2024, 12, 30) or start < origin:
        raise ValueError("Development scoring accepts only an admissible post-fit development calendar")
    folder = job_folder(context["output"], model, candidate, seed, origin)
    fit = verify_job(folder, hash_json(context["inputs"]), job_identity(context, model, candidate, seed, origin))
    if fit["status"] != "passed":
        return None
    key = hash_json({"fit_sha256": digest(folder/"completion.json"), "start": start, "end": end,
                     "stage": stage, "configuration_selection_outcome_max": selection_outcome_max})
    directory = context["output"] / "development_scores" / key
    if directory.exists():
        receipt = phase_receipt(directory/"completion.json", expected_inputs=hash_json(context["inputs"]))
        prune_development_fit(context, folder, directory/"completion.json")
        return csv_read(directory/"predictions.csv")
    pruning_path = folder/"checkpoint_pruned_after_development_scoring.json"
    if pruning_path.exists():
        pruning = read_json(pruning_path)
        original = local(context["output"], pruning["scoring_receipt_relative_to_output"])
        phase_receipt(original, expected_inputs=hash_json(context["inputs"]))
        cached = csv_read(original.parent/"predictions.csv")
        wanted = [d.isoformat() for d in calendar(start, end)]
        actual = [r["outcome_day"] for r in cached]
        if actual != wanted:
            raise RuntimeError("Pruned fit has no matching immutable prediction calendar; a new registered fit would be required")
        expected_selection = day(selection_outcome_max or (origin-8*DAY)).isoformat()
        if any(r["configuration_selection_outcome_max"] != expected_selection for r in cached):
            raise ValueError("Cached forecasts cannot be relabeled with a different selection information set")
        rows = [{**r, "source_stage": stage} for r in cached]
        directory.mkdir(parents=True)
        csv_write(directory/"predictions.csv", rows)
        write_json(directory/"metrics.json", read_json(original.parent/"metrics.json"))
        close_phase(directory/"completion.json", hash_json(context["inputs"]),
                    artifacts=[directory/"predictions.csv", directory/"metrics.json"],
                    fit_completion_sha256=digest(folder/"completion.json"), scope="development_only",
                    reused_closed_predictions=True, original_scoring_receipt_sha256=digest(original),
                    model_refitting_or_new_inference_performed=False)
        return rows
    directory.mkdir(parents=True)
    cpu_started = time.process_time()
    scoring_started = stamp()
    if fit["completed_at_utc"] >= scoring_started:
        raise ValueError("Checkpoint closure must precede outcome decoding/scoring")
    data = development(end, context["root"])
    indices = evaluation_indices(data, start, end)
    prediction, raw, latency = prediction_from_fit(context, folder, data, indices)
    setup = read_json(folder/"setup.json")
    checkpoint = "checkpoint.pt" if is_neural(model) else ("model.npz" if model == "Ridge" else
                 "model.pickle" if model == "ARIMA" else "rule.json")
    rows = []
    for j, i in enumerate(indices):
        origin_day = day(data["calendar"][i])
        rows.append({"model": model, "seed": seed, "candidate_id": candidate["candidate_id"],
                     "outcome_day": origin_day.isoformat(), "y_true": float(data["targets"][i]),
                     "y_pred": float(prediction[j]), "raw_y_pred": float(raw[j]),
                     "persistence_pred": float(data["persistence"][i]),
                     "high_volatility_known": bool(data["persistence"][i] > HIGH_VOLATILITY),
                     "mase_scale": setup["mase_scale"], "eligible": True,
                     "fit_origin": origin.isoformat(), "fit_outcome_max": setup["fit_outcome_max"],
                     "scaler_fit_outcome_max": setup["scaler_fit_outcome_max"],
                     "configuration_selection_outcome_max": day(selection_outcome_max or (origin-8*DAY)).isoformat(),
                     "historical_selection_exposure_disclosed": model in ANCHORS,
                     "configuration_selection_kind": "inherited_fixed_E1_recipe" if model in ANCHORS else "registered_nested_selection_or_rule",
                     "E2_new_configuration_selection": model not in ANCHORS,
                     "unbiased_nested_selection_claimed": model not in ANCHORS,
                     "epoch_selection_outcome_max": (origin-8*DAY).isoformat(),
                     "source_stage": stage,
                     "run_id": hash_json({"fit": fit["identity"]}),
                     "fit_directory": folder.relative_to(context["output"]).as_posix(),
                     "checkpoint_sha256": fit["artifacts"][checkpoint],
                     "scaler_sha256": fit["artifacts"]["preprocessing.json"],
                     "config_sha256": fit["artifacts"]["setup.json"],
                     "checkpoint_closed_at_utc": fit["completed_at_utc"],
                     "scoring_started_at_utc": scoring_started,
                     "development_latency_ms_per_window": float(latency[j])})
    csv_write(directory/"predictions.csv", rows)
    summary = metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
    summary["development_latency_ms_per_window"] = sum(latency)/len(latency)
    summary["parameters_total"] = setup["parameters_total"]
    write_json(directory/"metrics.json", summary)
    close_phase(directory/"completion.json", hash_json(context["inputs"]),
                artifacts=[directory/"predictions.csv", directory/"metrics.json"],
                fit_completion_sha256=digest(folder/"completion.json"), scope="development_only")
    if stage in ("E2_HPO_inner", "E2_final_recipe_search", "E2_nested_development") and model in TUNABLE:
        charge = context["output"]/"CPU_ledger"/model_slug(model)/(key+".json")
        if not charge.exists():
            write_json(charge, {"activity": stage, "CPU_seconds": time.process_time()-cpu_started,
                                "budget_scope": "HPO", "registration_inputs_sha256": hash_json(context["inputs"])})
    prune_development_fit(context, folder, directory/"completion.json")
    return rows


def prune_development_fit(context, folder, score_receipt):
    """Discard only replaceable HPO/OOF weights after immutable forecasts close.

    Configuration, curves, exact scalar errors, scalers, index receipts and
    checkpoint hashes remain.  P1, primary-final and branch-A predictors never
    pass this scope gate.  The original fit receipt is not rewritten.
    """
    folder, score_receipt = Path(folder), Path(score_receipt)
    record = verify_job(folder, hash_json(context["inputs"]))
    if (record.get("budget_scope") not in ("HPO", "fixed_anchor_development", "ablation_development")
            or (record.get("budget_scope") == "ablation_development" and int(record["identity"]["seed"]) == 670101)):
        return
    marker = folder/"checkpoint_pruned_after_development_scoring.json"
    if marker.exists():
        previous = read_json(marker)
        for name, expected in previous["pruned_artifact_sha256"].items():
            path = folder/name
            if path.exists():
                if digest(path) != expected:
                    raise ValueError("A pending-to-prune checkpoint changed")
                path.unlink()
        return
    phase_receipt(score_receipt, expected_inputs=hash_json(context["inputs"]))
    names = [name for name in ("checkpoint.pt", "model.pickle") if (folder/name).is_file()]
    if not names:
        return
    receipt = {"pruned_at_utc": stamp(), "registration_inputs_sha256": hash_json(context["inputs"]),
               "original_fit_receipt_sha256": digest(folder/"completion.json"),
               "scoring_receipt_relative_to_output": score_receipt.relative_to(context["output"]).as_posix(),
               "closed_scoring_receipt_sha256": digest(score_receipt),
               "pruned_artifact_sha256": {name: record["artifacts"][name] for name in names},
               "pruned_bytes": sum((folder/name).stat().st_size for name in names),
               "scope": "Only already-scored development HPO/OOF weights; original receipt, configuration, curves, scaler, supports and predictions retained",
               "P1_or_final_or_branch_A_checkpoint_pruned": False}
    write_json(marker, receipt)
    for name in names:
        (folder/name).unlink()


def score_candidate_rows(rows):
    """Mean of seed-wise RMSE over concatenated dates, never monthly RMSE mean."""
    import numpy as np
    groups = {}
    for row in rows:
        identity = str(row["seed"])
        groups.setdefault(identity, []).append(row)
    if not groups:
        raise ValueError("A candidate without complete predictions cannot be ranked")
    dates = None
    scores, costs = [], []
    for seed, records in groups.items():
        records = sorted(records, key=lambda row: row["outcome_day"])
        actual = [row["outcome_day"] for row in records]
        if len(actual) != len(set(actual)):
            raise ValueError("Candidate validation dates must not be duplicated")
        if dates is not None and dates != actual:
            raise ValueError("All paired seeds must use exactly the same candidate dates")
        dates = actual
        scores.append(metrics([float(r["y_true"]) for r in records],
                              [float(r["y_pred"]) for r in records])["RMSE"])
        costs.extend(float(r["development_latency_ms_per_window"]) for r in records)
    return {"mean_seed_RMSE": float(np.mean(scores)), "seed_RMSE": scores,
            "unique_origins": len(dates), "dates_sha256": hash_json(dates),
            "latency_ms": float(np.mean(costs))}


def choose_candidate(results):
    valid = [r for r in results if r.get("status") == "passed"]
    if not valid:
        raise RuntimeError("No fully evaluated registered candidate exists; no substitute is selected")
    if len({r["dates_sha256"] for r in valid}) != 1:
        raise ValueError("Candidate ranking requires the same full eligible validation cohort")
    return min(valid, key=lambda r: (r["mean_seed_RMSE"], r["latency_ms"],
                                     r["parameters_total"], r["candidate_id"]))


class TrialRunner:
    """One fresh process per actual fit, bounded parallelism and CPU accounting."""
    def __init__(self, context, args):
        self.context, self.args = context, args
        self.logs = context["output"] / "worker_logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.cap = 172800.0
        self.reservation_per_job = 2*context["hpo"]["optimization"]["maximum_optimizer_fit_seconds"] + 180.0

    def consumed(self, model):
        total = 0.0
        for p in (self.context["output"]/"fits"/model_slug(model)).glob("**/completion.json"):
            r = read_json(p)
            if r.get("budget_scope") == "HPO":
                total += float(r.get("CPU_seconds", 0))
        for p in (self.context["output"]/"CPU_ledger"/model_slug(model)).glob("*.json"):
            r = read_json(p)
            if r.get("budget_scope") == "HPO":
                total += float(r["CPU_seconds"])
        return total

    def _run(self, job, budget_scope):
        model, candidate, seed, origin = job
        folder = job_folder(self.context["output"], model, candidate, seed, origin)
        if folder.exists():
            return {"cached": True, "record": verify_job(folder, hash_json(self.context["inputs"]),
                                                           job_identity(self.context, model, candidate, seed, origin))}
        payload = {"model": model, "candidate": candidate, "seed": seed,
                   "origin": day(origin).isoformat(), "budget_scope": budget_scope}
        identity = hash_json(payload)
        jobs = self.context["output"] / "worker_jobs"
        jobs.mkdir(exist_ok=True)
        job_file = jobs / (identity + ".json")
        if not job_file.exists():
            write_json(job_file, payload)
        command = [sys.executable, str(Path(__file__).resolve()), "--phase", "worker",
                   "--design", str(self.args.design), "--output-dir", str(self.context["output"]),
                   "--job-json", str(job_file)]
        if self.args.source_roots_json:
            command += ["--source-roots-json", str(self.args.source_roots_json)]
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        with (self.logs/(identity+".log")).open("ab") as log:
            result = subprocess.run(command, cwd=self.context["root"], env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=self.reservation_per_job+60)
        if not (folder/"completion.json").exists():
            raise RuntimeError("Worker interruption left no terminal receipt; audit before resuming: " + identity)
        record = verify_job(folder, hash_json(self.context["inputs"]), job_identity(self.context, model, candidate, seed, origin))
        if result.returncode != 0 and record["status"] == "passed":
            raise RuntimeError("Worker exit/receipt inconsistency")
        return {"cached": False, "record": record}

    def run(self, jobs, *, budget_scope="HPO"):
        jobs = list(jobs)
        results = []
        # Caching is verified before scheduling and never bypasses input identity.
        pending = []
        for job in jobs:
            model, candidate, seed, origin = job
            folder = job_folder(self.context["output"], model, candidate, seed, origin)
            if folder.exists():
                results.append(self._run(job, budget_scope))
            else:
                pending.append(job)
        for index in range(0, len(pending), self.args.workers):
            batch = pending[index:index+self.args.workers]
            reserved = {}
            accepted = []
            for job in batch:
                model = job[0]
                if budget_scope == "HPO" and self.consumed(model) + reserved.get(model, 0) + self.reservation_per_job > self.cap:
                    results.append({"cached": False, "record": {"status": "budget_not_attempted",
                                    "model": model, "candidate_id": job[1]["candidate_id"],
                                    "origin": day(job[3]).isoformat(), "seed": job[2],
                                    "reason": "Remaining common CPU budget is below the reserved worst-case job cost",
                                    "registered_cap_CPU_seconds": self.cap}})
                else:
                    accepted.append(job)
                    reserved[model] = reserved.get(model, 0) + self.reservation_per_job
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                futures = [pool.submit(self._run, job, budget_scope) for job in accepted]
                for future in as_completed(futures):
                    results.append(future.result())
            if budget_scope == "HPO":
                for model in reserved:
                    if self.consumed(model) > self.cap:
                        raise RuntimeError("The common CPU cap was exceeded; stop and retain the accounting")
            write_json(self.context["output"]/"progress.json", {
                "updated_at_utc": stamp(), "budget_scope": budget_scope,
                "batch_jobs_requested": len(jobs), "batch_results_observed": len(results),
                "cache_hits": sum(r["cached"] for r in results),
                "actual_new_terminal_attempts": sum(not r["cached"] and r["record"].get("status") in ("passed", "failed") for r in results),
                "status_counts": {status: sum(r["record"].get("status") == status for r in results)
                                  for status in ("passed", "failed", "budget_not_attempted")},
                "scope": "Mutable batch heartbeat; not cumulative benchmark or fit-completion evidence"}, exclusive=False)
        return results


def worker(args, context):
    payload = read_json(args.job_json)
    record = fit_trial(context, payload["model"], payload["candidate"], payload["seed"], payload["origin"],
                       budget_scope=payload["budget_scope"])
    path = job_folder(context["output"], payload["model"], payload["candidate"], payload["seed"], payload["origin"])/"completion.json"
    # Scope is administrative cost accounting, fixed before this worker starts.
    if record.get("worker_accounting_finalized") is not True:
        record["fit_CPU_seconds"] = record["CPU_seconds"]
        record["CPU_seconds"] = time.process_time()
        record["wall_seconds_including_process_setup"] = time.perf_counter()-PROCESS_STARTED_WALL
        record["worker_accounting_finalized"] = True
        write_json(path, record, exclusive=False)
    elif record["budget_scope"] != payload["budget_scope"]:
        # Exact reuse is permitted but the original charged computation is not
        # silently transferred between budgets or charged twice.
        record = {**record, "reused_original_budget_scope": record["budget_scope"]}
    return record


def month_end(origin):
    origin = day(origin)
    following = date(origin.year+1, 1, 1) if origin.month == 12 else date(origin.year, origin.month+1, 1)
    return following-DAY


def candidate_for(context, model, candidate_id):
    if model in ANCHORS + NAIVES:
        fixed = fixed_candidate(model)
        if fixed["candidate_id"] != candidate_id:
            raise ValueError("A fixed reference was assigned a tunable recipe")
        return fixed
    found = [c for c in context["candidates"] if c["model"] == model and c["candidate_id"] == candidate_id]
    if len(found) != 1:
        raise ValueError("Selected candidate is absent or ambiguous in the registered registry")
    return found[0]


def hpo_model(context, args, model, *, nested_only=False):
    check_registration(context)
    inputs_hash = hash_json(context["inputs"])
    base = context["output"]/"hpo"/model_slug(model)
    base.mkdir(parents=True, exist_ok=True)
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    runner = TrialRunner(context, args)
    candidates = [c for c in context["candidates"] if c["model"] == model]
    nested_rows = []
    for outer in OUTER_ORIGINS:
        directory = base/"nested"/outer.isoformat()
        directory.mkdir(parents=True, exist_ok=True)
        selection_path = directory/"selection.json"
        blocks = inner_blocks(outer)
        if selection_path.exists():
            choice = read_json(selection_path)
            if choice["registration_inputs_sha256"] != inputs_hash:
                raise ValueError("Outer selection belongs to another input identity")
        else:
            jobs = [(model, c, seed, block["origin"]) for c in candidates
                    for block in blocks for seed in seeds_for(model, HPO_SEEDS)]
            attempts = runner.run(jobs)
            if any(a["record"]["status"] == "budget_not_attempted" for a in attempts):
                write_json(directory/"budget_exhaustion.json", {"at_utc": stamp(), "attempts": attempts}, exclusive=False)
                raise RuntimeError("Common HPO budget exhausted; do not present an incomplete search as equivalent")
            rankings = []
            for candidate in candidates:
                prediction_rows = []
                failed = []
                parameters_total = None
                for block in blocks:
                    for seed in seeds_for(model, HPO_SEEDS):
                        folder = job_folder(context["output"], model, candidate, seed, block["origin"])
                        receipt = verify_job(folder, inputs_hash)
                        if receipt["status"] != "passed":
                            failed.append({"origin": block["origin"].isoformat(), "seed": seed,
                                           "error": receipt.get("error")})
                            continue
                        result = scored_rows(context, model, candidate, seed, block["origin"],
                                             block["origin"], block["end"], stage="E2_HPO_inner",
                                             selection_outcome_max=block["origin"]-8*DAY)
                        prediction_rows.extend(result)
                        parameters_total = read_json(folder/"setup.json")["parameters_total"]
                row = {"candidate_id": candidate["candidate_id"], "status": "failed" if failed else "passed",
                       "failures": failed, "parameters_total": parameters_total}
                if not failed:
                    row.update(score_candidate_rows(prediction_rows))
                rankings.append(row)
            winner = choose_candidate(rankings)
            choice = {"selected_at_utc": stamp(), "registration_inputs_sha256": inputs_hash,
                      "outer_origin": outer.isoformat(), "selection_outcome_max": (outer-8*DAY).isoformat(),
                      "inner_blocks": blocks, "ranking": rankings, "winner": winner["candidate_id"],
                      "outer_predictions_computed_before_this_selection": False}
            write_json(selection_path, choice)
        candidate = candidate_for(context, model, choice["winner"])
        attempts = runner.run([(model, candidate, seed, outer) for seed in seeds_for(model, HPO_SEEDS)])
        if any(a["record"]["status"] != "passed" for a in attempts):
            raise RuntimeError("Selected outer candidate failed; record the failure without substituting another winner")
        for seed in seeds_for(model, HPO_SEEDS):
            nested_rows.extend(scored_rows(context, model, candidate, seed, outer, outer, month_end(outer),
                                           stage="E2_nested_development",
                                           selection_outcome_max=outer-8*DAY))
        if not (directory/"completion.json").exists():
            close_phase(directory/"completion.json", inputs_hash, artifacts=[selection_path],
                        outer_scored_after_choice=True, winner=choice["winner"])
    nested_file = base/"nested_oof_predictions.csv"
    if not nested_file.exists():
        csv_write(nested_file, nested_rows)
    # Seal the complete nested outer procedure before the extra all-candidate
    # development search can inspect these same months for a unique final recipe.
    nested_closed = base/"nested_completion.json"
    if not nested_closed.exists():
        close_phase(nested_closed, inputs_hash, artifacts=[nested_file],
                    outer_choices_immutable=True, future_recipe_search_started=False)
    else:
        phase_receipt(nested_closed, expected_inputs=inputs_hash)
    if nested_only:
        return phase_receipt(nested_closed, expected_inputs=inputs_hash)
    final_rows = []
    final_ranking, branch_ranking = [], []
    for candidate in candidates:
        all_rows, failed, parameters_total = [], [], None
        attempts = runner.run([(model, candidate, seed, origin) for origin in OUTER_ORIGINS
                               for seed in seeds_for(model, HPO_SEEDS)])
        if any(a["record"]["status"] == "budget_not_attempted" for a in attempts):
            raise RuntimeError("Common HPO budget exhausted in final-recipe search")
        for origin in OUTER_ORIGINS:
            for seed in seeds_for(model, HPO_SEEDS):
                folder = job_folder(context["output"], model, candidate, seed, origin)
                receipt = verify_job(folder, inputs_hash)
                if receipt["status"] != "passed":
                    failed.append({"origin": origin.isoformat(), "seed": seed, "error": receipt.get("error")})
                    continue
                all_rows.extend(scored_rows(context, model, candidate, seed, origin, origin, month_end(origin),
                                            stage="E2_final_recipe_search",
                                            selection_outcome_max=origin-8*DAY))
                parameters_total = read_json(folder/"setup.json")["parameters_total"]
        final_rows.extend(all_rows)
        row = {"candidate_id": candidate["candidate_id"], "status": "failed" if failed else "passed",
               "failures": failed, "parameters_total": parameters_total}
        branch = dict(row)
        if not failed:
            row.update(score_candidate_rows(all_rows))
            branch.update(score_candidate_rows(all_rows))
        final_ranking.append(row)
        branch_ranking.append(branch)
    primary, branch = choose_candidate(final_ranking), choose_candidate(branch_ranking)
    recipe_path, branch_path = base/"primary_recipe.json", base/"branch_A_recipe.json"
    if not recipe_path.exists():
        write_json(recipe_path, {"model": model, "candidate": candidate_for(context, model, primary["candidate_id"]),
                                "selection_outcome_max": "2024-10-31", "selected_at_utc": stamp(),
                                "ranking": final_ranking, "nested_choice_update_permitted": False})
    if not branch_path.exists():
        write_json(branch_path, {"model": model, "candidate": candidate_for(context, model, branch["candidate_id"]),
                                "selection_outcome_max": "2024-10-31", "selected_at_utc": stamp(),
                                "ranking": branch_ranking, "same_primary_hpo_recipe": True,
                                "branch": "secondary_frozen_calibration_A53",
                                "independence_amendment": "Fit/selection throughOct31; embargoNov1–7; calibrationNov8–Dec30,53days"})
    final_file = base/"final_recipe_search_predictions.csv"
    if not final_file.exists():
        csv_write(final_file, final_rows)
    from e2_hpo_report import build_hpo_report, write_hpo_report
    report_dir = base/"hpo_report"
    report = build_hpo_report(context["candidates"], final_ranking, model=model)
    report_files = [report_dir/"hpo_report.json", report_dir/"hpo_curves.svg"]
    if any(p.exists() for p in report_files):
        if not all(p.is_file() for p in report_files) or read_json(report_files[0]) != report:
            raise ValueError("An interrupted or changed HPO report requires an infrastructure audit")
    else:
        report_files = write_hpo_report(report, report_dir)
    return close_phase(base/"completion.json", inputs_hash,
                       artifacts=[nested_file, nested_closed, final_file, recipe_path, branch_path, *report_files],
                       model=model, all_twelve_candidates_attempted=True,
                       final_candidate=primary["candidate_id"], branch_A_candidate=branch["candidate_id"],
                       realized_CPU_seconds=runner.consumed(model),
                       failed_candidate_count=sum(r["status"] != "passed" for r in final_ranking),
                       all_candidates_finite=all(r["status"] == "passed" for r in final_ranking),
                       scope="Nested development model-selection procedure plus separately frozen final recipes")


def preflight(context, args):
    """Exercise the exact final factories on synthetic windows, no dataset fits."""
    import numpy as np
    import torch
    inputs_hash = hash_json(context["inputs"])
    folder = context["output"]/"preflight"
    if (folder/"completion.json").exists():
        return phase_receipt(folder/"completion.json", expected_inputs=inputs_hash)
    folder.mkdir(parents=True, exist_ok=True)
    verify_architecture_review(context["design"], context["root"])
    configure(202611066)
    synthetic = torch.from_numpy(np.random.default_rng(202611066).normal(size=(4, 7, 5)).astype("float32"))
    results, all_ablations = [], {}
    for model in CATALOGUE:
        start = time.perf_counter()
        if model in DETERMINISTIC:
            row = {"model": model, "status": "passed", "synthetic_check": "configuration sanity; causal helper unit tests are separately recorded evidence; no market fit",
                   "optimizer_fits": 0, "parameters_total": None}
        else:
            candidate = (next(c for c in context["candidates"] if c["model"] == model)
                         if model in TUNABLE_NEURAL else fixed_candidate(model))
            predictor = build_neural(model, candidate, context["roots"])
            predictor.train()
            output = predictor(synthetic)
            if tuple(output.shape) != (4,) or not torch.isfinite(output).all() or (output < 0).any():
                raise ValueError("Final factory failed synthetic output contract: " + model)
            output.square().mean().backward()
            active = sum(p.numel() for p in predictor.parameters() if p.grad is not None)
            if active == 0 or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in predictor.parameters()):
                raise ValueError("Final factory backward is absent/nonfinite: " + model)
            predictor.eval()
            with torch.no_grad():
                batch = predictor(synthetic)
                points = torch.cat([predictor(synthetic[i:i+1]) for i in range(4)])
            if not torch.allclose(batch, points, atol=2e-6, rtol=2e-5):
                raise ValueError("Final predictions change with evaluation batch companions: " + model)
            row = {"model": model, "status": "passed", "synthetic_output_shape": list(output.shape),
                   "backward_finite": True, "evaluation_batch_invariant": True,
                   "parameters_total": sum(p.numel() for p in predictor.parameters()),
                   "parameters_trainable": sum(p.numel() for p in predictor.parameters() if p.requires_grad),
                   "parameters_with_gradient": active,
                   "state_dict_storage_bytes": sum(t.numel()*t.element_size() for t in predictor.state_dict().values()),
                   "architecture": getattr(predictor, "architecture_metadata", {}),
                   "synthetic_backward_wall_seconds": time.perf_counter()-start,
                   "dataset_optimizer_fits": 0}
            if model in TOP10:
                from e2_final_diagnostics import ablation_definitions
                all_ablations[model] = ablation_definitions(predictor)
            del predictor
        row["wall_seconds"] = time.perf_counter()-start
        results.append(row)
    write_json(folder/"models.json", results)
    write_json(folder/"ablation_definitions.json", all_ablations)
    write_json(folder/"environment.json", environment())
    inventory = planned_inventory()
    sum_state_bytes = sum(r.get("state_dict_storage_bytes", 0) for r in results)
    inventory["all_sixteen_neural_state_bytes_per_paired_seed"] = sum_state_bytes
    inventory["branch_A_five_member_state_bytes"] = 5*sum_state_bytes
    inventory["primary_plus_A_state_bytes_by_R"] = {str(R): (R+5)*sum_state_bytes
                                                    for R in context["prereg"]["power"]["candidate_R"]}
    inventory["primary_A_controls_state_bytes_by_R"] = {str(R): (R+6)*sum_state_bytes
                                                        for R in context["prereg"]["power"]["candidate_R"]}
    inventory["checkpoint_storage_estimate_scope"] = "FP32 tensors/buffers only; filesystem/checkpoint headers and retained diagnostics add overhead. Prunable HPO weights excluded after their scores close."
    write_json(folder/"prospective_compute_inventory.json", inventory)
    # Reading source CSV hashes is not decoding future labels.  The structural
    # dataset-support audit is performed by its already registered helper.
    from e2_remediation_data_audit import audit as support_audit
    support = {"inherited_P0_support_audit": support_audit(context["root"]),
               "final_procedure_support_audit": final_support_audit(context)}
    write_json(folder/"data_support.json", support)
    return close_phase(folder/"completion.json", inputs_hash,
                       artifacts=[folder/"models.json", folder/"environment.json",
                                  folder/"prospective_compute_inventory.json", folder/"data_support.json",
                                  folder/"ablation_definitions.json"],
                       all_twenty_configurations_preflighted=True, dataset_fits=0,
                       test_prices_or_labels_decoded=False,
                       all_models_have_source_review=True,
                       scope="Synthetic final-factory and development support preflight; not benchmark results")


def fixed_outer(context, args):
    inputs_hash = hash_json(context["inputs"])
    folder = context["output"]/"hpo"/"fixed_outer_references"
    if (folder/"completion.json").exists():
        phase_receipt(folder/"completion.json", expected_inputs=inputs_hash)
        return csv_read(folder/"nested_oof_predictions.csv")
    folder.mkdir(parents=True, exist_ok=True)
    runner, all_rows = TrialRunner(context, args), []
    for model in ANCHORS + NAIVES:
        candidate = fixed_candidate(model)
        attempts = runner.run([(model, candidate, seed, origin) for origin in OUTER_ORIGINS
                               for seed in seeds_for(model, HPO_SEEDS)], budget_scope="fixed_anchor_development")
        if any(a["record"]["status"] != "passed" for a in attempts):
            raise RuntimeError("An unchanged reference failed; no alternative is silently substituted")
        for origin in OUTER_ORIGINS:
            for seed in seeds_for(model, HPO_SEEDS):
                all_rows.extend(scored_rows(context, model, candidate, seed, origin, origin, month_end(origin),
                                             stage="E2_nested_development",
                                             selection_outcome_max=origin-8*DAY))
    csv_write(folder/"nested_oof_predictions.csv", all_rows)
    close_phase(folder/"completion.json", inputs_hash, artifacts=[folder/"nested_oof_predictions.csv"],
                fixed_recipes_not_HPO=True, references=list(ANCHORS+NAIVES))
    return all_rows


def validate_development_panel(rows, *, models=CATALOGUE, seeds=HPO_SEEDS):
    expected_dates = [d.isoformat() for d in calendar("2024-01-01", "2024-10-31")]
    model_rows, truth, monthly = {}, {}, {}
    for row in rows:
        model, seed, d = row["model"], str(row["seed"]), row["outcome_day"]
        if model not in models or seed not in {str(v) for v in seeds_for(model, seeds)} or d not in expected_dates:
            raise ValueError("Development panel includes an undeclared model, seed or date")
        if row.get("source_stage") != "E2_nested_development":
            raise ValueError("P0, paper replicas and final-recipe in-sample selections cannot replace nested OOF")
        if row.get("eligible") not in (True, "True", "true", 1, "1"):
            raise ValueError("The audited common development calendar must be retained for every model")
        key = model, seed, d
        if key in model_rows:
            raise ValueError("Development panel duplicates a model/seed/date")
        cutoff = day(d).replace(day=1)-8*DAY
        if any(day(row[k]) > cutoff for k in ("fit_outcome_max", "scaler_fit_outcome_max")):
            raise ValueError("Development fold used information beyond its registered outer-origin embargo")
        if model in ANCHORS:
            if (row.get("candidate_id") != fixed_candidate(model)["candidate_id"]
                    or day(row["configuration_selection_outcome_max"]) != date(2024, 10, 31)
                    or row.get("historical_selection_exposure_disclosed") not in (True, "True", "true", 1, "1")
                    or row.get("configuration_selection_kind") != "inherited_fixed_E1_recipe"
                    or row.get("E2_new_configuration_selection") not in (False, "False", "false", 0, "0")
                    or row.get("unbiased_nested_selection_claimed") not in (False, "False", "false", 0, "0")
                    or day(row["epoch_selection_outcome_max"]) > cutoff):
                raise ValueError("Inherited anchors must disclose their real October31 historical recipe-selection exposure")
        elif day(row["configuration_selection_outcome_max"]) > cutoff:
            raise ValueError("A new model selection used future outer-fold information")
        actual = float(row["y_true"])
        forecast = float(row["y_pred"])
        if not math.isfinite(actual) or not math.isfinite(forecast) or min(actual, forecast) < 0:
            raise ValueError("Development volatility values must be finite and nonnegative")
        if d in truth and truth[d] != actual:
            raise ValueError("Ground truth is not identical across models/seeds")
        truth[d] = actual
        model_rows[key] = row
        month_key = model, seed, d[:7]
        frozen = tuple(row.get(k) for k in ("run_id", "checkpoint_sha256", "fit_outcome_max",
                                            "scaler_fit_outcome_max", "configuration_selection_outcome_max"))
        if month_key in monthly and monthly[month_key] != frozen:
            raise ValueError("A monthly frozen outer predictor cannot change within the month")
        monthly[month_key] = frozen
    expected_count = sum(len(seeds_for(model, seeds))*len(expected_dates) for model in models)
    if len(model_rows) != expected_count:
        raise ValueError("Every declared model/seed must retain all305 eligible calendar origins")
    return {"models": list(models), "unique_dates_per_model": 305, "rows": expected_count,
            "stochastic_seeds": list(seeds), "deterministic_references_stored_once": True,
            "monthly_frozen_predictors": True, "all_common_eligible_dates": True,
            "inherited_anchors_historically_exposed": [m for m in models if m in ANCHORS],
            "unbiased_nested_recipe_selection_claimed_for_legacy_anchors": False}


def run_hpo(context, args):
    check_registration(context)
    inputs_hash = hash_json(context["inputs"])
    output = context["output"]
    selected = tuple(args.models or TUNABLE)
    if not set(selected).issubset(TUNABLE) or len(selected) != len(set(selected)):
        raise ValueError("HPO model subset must contain unique registered tunables")
    for model in selected:
        hpo_model(context, args, model, nested_only=True)
    nested_paths = [output/"hpo"/model_slug(m)/"nested_completion.json" for m in TUNABLE]
    if not all(p.exists() for p in nested_paths):
        return {"status": "partial", "completed_nested_models": [m for m, p in zip(TUNABLE, nested_paths) if p.exists()],
                "final_recipe_search_started": False, "whole_HPO_complete": False}
    fixed_rows = fixed_outer(context, args)
    all_rows = [r for m in TUNABLE for r in csv_read(output/"hpo"/model_slug(m)/"nested_oof_predictions.csv")]+fixed_rows
    validation = validate_development_panel(all_rows)
    global_nested = output/"hpo"/"nested_completion.json"
    if not global_nested.exists():
        all_file = output/"hpo"/"nested_oof_predictions.csv"
        csv_write(all_file, all_rows)
        close_phase(global_nested, inputs_hash, artifacts=[all_file, *nested_paths,
                    output/"hpo/fixed_outer_references/completion.json"],
                    validation=validation, all_twenty_outer_procedures_closed_before_final_recipe_search=True)
    else:
        phase_receipt(global_nested, expected_inputs=inputs_hash)
    for model in selected:
        hpo_model(context, args, model)
    done = [output/"hpo"/model_slug(m)/"completion.json" for m in TUNABLE]
    if not all(p.exists() for p in done):
        return {"status": "partial", "final_recipe_models": [m for m, p in zip(TUNABLE, done) if p.exists()],
                "whole_HPO_complete": False}
    if (output/"hpo/completion.json").exists():
        return phase_receipt(output/"hpo/completion.json", expected_inputs=inputs_hash)
    primary, branch = {}, {}
    for model in CATALOGUE:
        if model in TUNABLE:
            directory = output/"hpo"/model_slug(model)
            primary[model] = read_json(directory/"primary_recipe.json")
            branch[model] = read_json(directory/"branch_A_recipe.json")
        else:
            primary[model] = {"model": model, "candidate": fixed_candidate(model),
                              "selection_outcome_max": "2024-10-31", "fixed_reference": True,
                              "configuration_origin": "Inherited E1 recipe selected on Jan–Oct2024" if model in ANCHORS else "Prespecified causal lag rule, no fitted selection",
                              "historical_E1_selection_outcome_max": "2024-10-31" if model in ANCHORS else None}
            branch[model] = dict(primary[model])
    primary_path, branch_path = output/"hpo/primary_recipes.json", output/"hpo/branch_A_recipes.json"
    write_json(primary_path, primary)
    write_json(branch_path, branch)
    costs = {m: {"actual_CPU_seconds": TrialRunner(context, args).consumed(m),
                 "assigned_CPU_seconds": 172800,
                 "all_candidates_finite": read_json(output/"hpo"/model_slug(m)/"completion.json")["all_candidates_finite"]}
             for m in TUNABLE}
    write_json(output/"hpo/cost_accounting.json", costs)
    scores = {model: score_candidate_rows([r for r in all_rows if r["model"] == model])["mean_seed_RMSE"]
              for model in CATALOGUE}
    references = [model for model in CATALOGUE if model not in TOP10]
    efficiency = {"best_model": min(scores, key=lambda m: (scores[m], m)),
                  "best_top10": min(TOP10, key=lambda m: (scores[m], m)),
                  "top3_development": sorted(scores, key=lambda m: (scores[m], m))[:3],
                  "best_baseline": min(references, key=lambda m: (scores[m], m)),
                  "outcome_max": "2024-10-31", "source": "complete20model_common_development",
                  "frozen_before_test": True, "primary_RQ_baselines_changed": False,
                  "historical_E1_anchors_selection_exposed_through": "2024-10-31",
                  "best_reference_selection_scope": "common development RMSE including historically selected E1 anchors; no unbiased nested claim for these three",
                  "scores": scores, "tie_break": "lexicographic model identity after exact RMSE tie"}
    write_json(output/"hpo/efficiency_selection.json", efficiency)
    return close_phase(output/"hpo/completion.json", inputs_hash,
                       artifacts=[primary_path, branch_path, global_nested, *done, output/"hpo/cost_accounting.json",
                                  output/"hpo/efficiency_selection.json"],
                       tunable_models=15, assigned_trials_each=12, catalogue_size=20,
                       all_registered_candidates_attempted=True,
                       primary_recipe_outcome_max="2024-10-31", branch_A_recipe_outcome_max="2024-10-31",
                       nested_outcomes_replaced_by_final_search=False, test_decoded=False)


def freeze_fit_group(context, args, phase, recipes, models, seeds, origin):
    """Close every declared member before *any* outcome scoring in that group."""
    inputs_hash = hash_json(context["inputs"])
    base = context["output"]/phase
    base.mkdir(parents=True, exist_ok=True)
    expected = [{"model": model, "candidate": recipes[model]["candidate"], "seed": seed,
                 "origin": day(origin).isoformat()}
                for model in models for seed in seeds_for(model, seeds)]
    schedule = base/"schedule.json"
    specification = {"models": list(models), "randomized_seeds": list(seeds), "origin": day(origin).isoformat(),
                     "jobs": expected, "recipes_sha256": hash_json(recipes),
                     "registration_inputs_sha256": inputs_hash}
    if schedule.exists():
        if read_json(schedule) != specification:
            raise ValueError("A frozen phase cannot change its member/recipe/seed schedule")
    else:
        write_json(schedule, specification)
    if (base/"completion.json").exists():
        phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
        return read_json(base/"fit_manifest.json")
    runner = TrialRunner(context, args)
    attempts = runner.run([(r["model"], r["candidate"], r["seed"], r["origin"]) for r in expected], budget_scope=phase)
    if any(a["record"]["status"] != "passed" for a in attempts):
        write_json(base/"failed_attempts.json", attempts, exclusive=False)
        raise RuntimeError("A required final/pilot member failed; retain it and do not substitute another seed")
    members = []
    for row in expected:
        folder = job_folder(context["output"], row["model"], row["candidate"], row["seed"], row["origin"])
        fit = verify_job(folder, inputs_hash, job_identity(context, row["model"], row["candidate"], row["seed"], row["origin"]))
        if (folder/"checkpoint_pruned_after_development_scoring.json").exists():
            raise ValueError("Final/P1/A checkpoints cannot be replaced by already-scored HPO predictions")
        checkpoint = ("checkpoint.pt" if is_neural(row["model"]) else "model.npz" if row["model"] == "Ridge"
                      else "model.pickle" if row["model"] == "ARIMA" else "rule.json")
        members.append({"model": row["model"], "seed": row["seed"], "candidate": row["candidate"],
                        "fit_directory": folder.relative_to(context["output"]).as_posix(),
                        "run_id": hash_json({"fit": fit["identity"]}),
                        "checkpoint_file": checkpoint, "checkpoint_sha256": fit["artifacts"][checkpoint],
                        "scaler_sha256": fit["artifacts"]["preprocessing.json"],
                        "config_sha256": fit["artifacts"]["setup.json"],
                        "completion_sha256": digest(folder/"completion.json"),
                        "checkpoint_closed_at_utc": fit["completed_at_utc"],
                        "fit_origin": row["origin"], "fit_outcome_max": fit["identity"]["fit_outcome_max"],
                        "configuration_selection_outcome_max": recipes[row["model"]]["selection_outcome_max"]})
    manifest = {"phase": phase, "origin": day(origin).isoformat(), "models": list(models),
                "stochastic_seeds": list(seeds), "members": members,
                "all_weights_closed_at_utc": max(r["checkpoint_closed_at_utc"] for r in members),
                "checkpoint_count": len(members), "registered_after_fit": False,
                "registration_inputs_sha256": inputs_hash,
                "optimizer_fits_completed": sum(2 for r in members if is_neural(r["model"])),
                "deterministic_estimator_fits_completed": sum(verify_job(local(context["output"], r["fit_directory"]), inputs_hash).get("estimator_fits", 0)
                                                             for r in members if r["model"] in ("Ridge", "ARIMA")),
                "deterministic_comparison_estimators": sum(1 for r in members if r["model"] in ("Ridge", "ARIMA")),
                "naive_rules_counted_as_fits": False}
    write_json(base/"fit_manifest.json", manifest)
    close_phase(base/"completion.json", inputs_hash, artifacts=[schedule, base/"fit_manifest.json"],
                models=list(models), checkpoint_count=len(members), all_closed_before_scoring=True)
    return manifest


def verify_fit_manifest(context, path, *, expected_models, expected_seeds, expected_origin):
    manifest = read_json(path)
    if (tuple(manifest["models"]) != tuple(expected_models)
            or tuple(manifest["stochastic_seeds"]) != tuple(expected_seeds)
            or day(manifest["origin"]) != day(expected_origin)
            or manifest["registration_inputs_sha256"] != hash_json(context["inputs"])):
        raise ValueError("Fit manifest model/seed/calendar/input contract differs")
    expected = {(model, str(seed)) for model in expected_models for seed in seeds_for(model, expected_seeds)}
    actual = {(r["model"], str(r["seed"])) for r in manifest["members"]}
    if actual != expected or len(actual) != len(manifest["members"]):
        raise ValueError("A fit manifest omits or duplicates a required member")
    for row in manifest["members"]:
        folder = local(context["output"], row["fit_directory"])
        fit = verify_job(folder, hash_json(context["inputs"]))
        if (fit["status"] != "passed" or digest(folder/"completion.json") != row["completion_sha256"]
                or not (folder/row["checkpoint_file"]).is_file()
                or digest(folder/row["checkpoint_file"]) != row["checkpoint_sha256"]):
            raise ValueError("A required frozen predictor is absent, failed or changed")
    return manifest


def run_p1(context, args):
    inputs_hash = hash_json(context["inputs"])
    phase_receipt(context["output"]/"hpo/completion.json", expected_inputs=inputs_hash)
    recipes = read_json(context["output"]/"hpo/primary_recipes.json")
    base = context["output"]/"P1"
    manifest = freeze_fit_group(context, args, "P1_fits", recipes, P1_MODELS, P1_SEEDS, P1_ORIGIN)
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    scoring_started = stamp()
    if manifest["all_weights_closed_at_utc"] >= scoring_started:
        raise ValueError("All P1 members must close before any P1 outcomes are decoded")
    all_rows = []
    for row in manifest["members"]:
        all_rows.extend(scored_rows(context, row["model"], row["candidate"], row["seed"], P1_ORIGIN,
                                    P1_ORIGIN, P1_END, stage="E2_P1_dimensioning",
                                    selection_outcome_max=recipes[row["model"]]["selection_outcome_max"]))
    if any(day(r["fit_outcome_max"]) > date(2024, 10, 31) or
           day(r["configuration_selection_outcome_max"]) > date(2024, 10, 31) for r in all_rows):
        raise ValueError("P1 must purge November1–7 after the October31 selection cutoff")
    csv_write(base/"predictions.csv", all_rows)
    return close_phase(base/"completion.json", inputs_hash,
                       artifacts=[base/"predictions.csv", context["output"]/"P1_fits/fit_manifest.json"],
                       P1_origin=P1_ORIGIN.isoformat(), P1_end=P1_END.isoformat(),
                       unique_daily_origins=53, actual_seeds_per_model=10, models=list(P1_MODELS),
                       all_P1_weights_closed_before_first_scoring=True,
                       scoring_started_at_utc=scoring_started,
                       maximum_selection_outcome="2024-10-31", purged_dates=["2024-11-01", "2024-11-07"],
                       final_test_calendar_unchanged=True, test_decoded=False)


def forecast_callable(context, member):
    """A steady-state batch1 forward on one fixed admitted development window."""
    import numpy as np
    import torch
    folder = local(context["output"], member["fit_directory"])
    setup, pre = read_json(folder/"setup.json"), read_json(folder/"preprocessing.json")
    model = member["model"]
    data = development(setup["fit_outcome_max"], context["root"])
    split = split_indices(data, setup["fit_origin"])
    index = split["train"][-1:]
    x = preprocess_x(data, index, pre)
    if is_neural(model):
        predictor = build_neural(model, setup["candidate"], context["roots"])
        predictor.load_state_dict(torch.load(folder/"checkpoint.pt", map_location="cpu", weights_only=True))
        predictor._e2_fit_started = True
        predictor.eval()
        tensor = torch.from_numpy(x)
        return lambda: float(predictor(tensor)[0])*pre["y_scale"], predictor, tensor, {
            "input_kind": "same last admitted development D7 window, frozen scaler",
            "complete_scalar_forward": True, "neural_FP32": True}
    if model == "Ridge":
        with np.load(folder/"model.npz", allow_pickle=False) as saved:
            coef, intercept = saved["coef"].copy(), float(saved["intercept"])
        vector = x.astype(float).ravel()
        return lambda: max(0., float(vector@coef+intercept))*pre["y_scale"], None, None, {
            "input_kind": "same admitted D7 window transformed with frozen scaler", "classical_dtype": "float64"}
    if model == "ARIMA":
        estimator = pickle.loads((folder/"model.pickle").read_bytes())
        last = float(data["targets"][index[0]]) / estimator.target_scale
        # Functional extend returns a new filtered result: fixed coefficients,
        # one observed scalar state update and the complete one-step forecast.
        return lambda: max(0., float(np.asarray(estimator._result.extend([last]).forecast(1))[0]) * estimator.target_scale), None, None, {
            "input_kind": "one steady-state conditional update using last admitted label as a fixed benchmark input",
            "parameters_reestimated": False, "state_mutated_across_repetitions": False,
            "one_past_label_update_included": True, "classical_dtype": "float64"}
    from e2_temporal_baselines import naive_forecast
    history = {day(d): (float(y) if np.isfinite(y) else None) for d, y in zip(data["calendar"], data["targets"])}
    cost_origin = day(data["calendar"][index[0]])+DAY
    lag = 1 if model == "Persistence" else 7
    return lambda: float(naive_forecast(history, cost_origin, lag=lag).value), None, None, {
        "input_kind": "complete causal calendar lookup/accessor; no coefficient estimation", "lag": lag}


def profile_flops(predictor, tensor):
    """Report profiler support honestly; FFT/scan/pointwise ops are not made zero."""
    import torch
    if predictor is None:
        return {"profiled_supported_FLOPs": None, "FLOPs_scope": "Classical estimator: Torch profiler not applicable"}
    try:
        from e2_compute_profile import profile_model
        result = profile_model(predictor, tensor)
        return {"profiled_supported_FLOPs": result["counted_flops_per_sample"],
                "FLOPs_per_window": (result["estimated_flops_per_sample"]+1
                                      if result["estimated_flops_per_sample"] is not None else None),
                "FLOPs_scope": result["coverage"], "FLOPs_convention": result["convention"],
                "unsupported_floating_operations": result["unsupported_floating_ops"],
                "final_original_unit_conversion_FLOPs": 1,
                "profiled_arithmetic_is_not_hardware_instruction_count": True}
    except ImportError:
        pass
    except Exception as error:
        return {"profiled_supported_FLOPs": None, "FLOPs_per_window": None,
                "FLOPs_scope": "not_estimable", "reason": str(error)}
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True,
                                    record_shapes=True) as profile:
            with torch.inference_mode():
                predictor(tensor)
        events = list(profile.key_averages())
        total = sum(int(e.flops or 0) for e in events)
        omitted = sorted({e.key for e in events if e.key.startswith("aten::") and not e.flops})
        return {"profiled_supported_FLOPs": total or None, "FLOPs_per_window": None,
                "FLOPs_scope": "Partial profiler-supported subtotal, not a hardware lower bound; omitted FFT/selective-scan/normalization/pointwise operations are not claimed zero",
                "unaccounted_operator_names": omitted,
                "analytical_estimate_from_architecture": getattr(predictor, "architecture_metadata", {}).get("analytical_FLOPs")}
    except Exception as error:
        return {"profiled_supported_FLOPs": None, "FLOPs_scope": "not_estimable", "reason": str(error)}


def run_cost(context, args, *, phase="primary_final"):
    import numpy as np
    import torch
    inputs_hash = hash_json(context["inputs"])
    if phase == "P1":
        models, seeds, origin, manifest_path = P1_MODELS, P1_SEEDS, P1_ORIGIN, context["output"]/"P1_fits/fit_manifest.json"
        phase_receipt(context["output"]/"P1/completion.json", expected_inputs=inputs_hash)
    else:
        power = read_json(context["output"]/"power/selection.json")
        R = int(power["selected_R"])
        models, seeds, origin, manifest_path = CATALOGUE, tuple(context["prereg"]["power"]["final_seed_ids_at_cap"][:R]), date(2025, 1, 1), context["output"]/"primary_final/fit_manifest.json"
    manifest = verify_fit_manifest(context, manifest_path, expected_models=models,
                                   expected_seeds=seeds, expected_origin=origin)
    base = context["output"]/"costs"/phase
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    configure(202611065)
    functions, metadata, throughputs = {}, [], {}
    hpo_costs = read_json(context["output"]/"hpo/cost_accounting.json")
    # The first predeclared member is fixed before any latency measurements.
    for model in models:
        member = next(m for m in manifest["members"] if m["model"] == model)
        function, predictor, tensor, spec = forecast_callable(context, member)
        functions[model] = function
        with torch.inference_mode():
            for _ in range(256):
                value = function()
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite fixed-input latency forward")
        setup = read_json(local(context["output"], member["fit_directory"])/"setup.json")
        profile = profile_flops(predictor, tensor)
        throughput = None
        if predictor is not None:
            from e2_compute_profile import measure_throughput
            throughput = measure_throughput(predictor, tensor.repeat(32, 1, 1), warmup=32,
                                             blocks=31, calls_per_block=32, threads=1)
            throughputs[model] = throughput
        if model == "Ridge":
            profile.update(FLOPs_per_window=72, FLOPs_scope="Analytical MAC2 convention:35MACs+intercept+original-unit scaling; preprocessing excluded")
        elif model in NAIVES:
            profile.update(FLOPs_per_window=0, FLOPs_scope="Zero floating arithmetic for an already-available lag lookup; memory/indexing/instructions are not FLOPs")
        member_set = [m for m in manifest["members"] if m["model"] == model]
        setups = [read_json(local(context["output"], m["fit_directory"])/"setup.json") for m in member_set]
        metadata.append({"model": model, "checkpoint_sha256": member["checkpoint_sha256"],
                         "parameters_total": setup["parameters_total"],
                         "parameters_trainable": setup["parameters_trainable"],
                         "training_process_peak_RSS_bytes": setup["peak_RSS_bytes_process"],
                         "training_process_peak_RSS_scope": "Complete isolated fit process, including runtime, source imports, data and model; not model-only allocation",
                         "parameters": setup["parameters_total"],
                         "peak_memory_bytes": max(s["peak_RSS_bytes_process"] for s in setups),
                         "training_seconds": sum(s["wall_seconds"] for s in setups)/len(setups),
                         "training_seconds_scope": "Mean complete two-stage trajectory wall clock, or one deterministic estimator/rule; repeated deterministic seeds are not invented",
                         "hpo_seconds": hpo_costs.get(model, {}).get("actual_CPU_seconds", 0. if model not in TUNABLE else None),
                         "hpo_seconds_kind": "actual_CPU_seconds",
                         "throughput_samples_sec": None if throughput is None else throughput["throughput_samples_sec"],
                         "throughput_scope": "Batch32 model-kernel outputs before original-unit scalar decode; preprocessing and serialization excluded; secondary throughput, not the batch1 RQ3 latency",
                         **spec, **profile})
    rng, rows = np.random.default_rng(202611065), []
    with torch.inference_mode():
        for block in range(61):
            order = list(rng.permutation(list(models)))
            for position, model in enumerate(order):
                started = time.perf_counter_ns()
                for _ in range(256):
                    value = functions[model]()
                elapsed = (time.perf_counter_ns()-started)/1e6/256
                info = next(r for r in metadata if r["model"] == model)
                rows.append({"model": model, "block": block, "latency_ms": elapsed,
                             "phase": phase, "warmup_calls": 256, "calls_per_block": 256,
                             "interleaved_order_position": position,
                             "parameters": info["parameters"], "FLOPs_per_window": info.get("FLOPs_per_window"),
                             "peak_memory_bytes": info["peak_memory_bytes"], "training_seconds": info["training_seconds"],
                             "hpo_seconds": info["hpo_seconds"], "hpo_seconds_kind": info["hpo_seconds_kind"],
                             "throughput_samples_sec": info["throughput_samples_sec"],
                             "source_checkpoint_sha256": next(m for m in manifest["members"] if m["model"] == model)["checkpoint_sha256"]})
    csv_write(base/"latency_blocks.csv", rows)
    write_json(base/"model_cost_metadata.json", metadata)
    write_json(base/"throughput_batches.json", throughputs)
    write_json(base/"environment.json", environment())
    return close_phase(base/"completion.json", inputs_hash,
                       artifacts=[base/"latency_blocks.csv", base/"model_cost_metadata.json", base/"environment.json", base/"throughput_batches.json"],
                       models=list(models), paired_timing_blocks=61, batch_size=1, CPU_threads=1,
                       predictive_calendar_observations_created=0, test_prices_or_labels_decoded=False)


def run_power(context, args):
    from e2_final_statistics import power_selection
    inputs_hash = hash_json(context["inputs"])
    phase_receipt(context["output"]/"P1/completion.json", expected_inputs=inputs_hash)
    run_cost(context, args, phase="P1")
    base = context["output"]/"power"
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    pilot_path, cost_path = context["output"]/"P1/predictions.csv", context["output"]/"costs/P1/latency_blocks.csv"
    result = power_selection(csv_read(pilot_path), csv_read(cost_path), context["prereg"])
    R = result.get("selected_R")
    if type(R) is not int or R not in context["prereg"]["power"]["candidate_R"] or R > 200:
        raise ValueError("Power simulation did not select a concrete predeclared R within the cap")
    result = {**result, "P1_prediction_sha256": digest(pilot_path), "P1_costs_sha256": digest(cost_path),
              "P1_calendar_days": 53, "projected_final_calendar_days": 60,
              "P1_is_not_60_new_observations": True, "selected_at_utc": stamp()}
    write_json(base/"selection.json", result)
    return close_phase(base/"completion.json", inputs_hash, artifacts=[base/"selection.json"],
                       selected_R=R, final_model_weights_created=False,
                       power_scope="Registered simulation after frozen HPO; uncertainty and unmet targets retained")


def run_final_fits(context, args):
    inputs_hash = hash_json(context["inputs"])
    phase_receipt(context["output"]/"hpo/completion.json", expected_inputs=inputs_hash)
    phase_receipt(context["output"]/"power/completion.json", expected_inputs=inputs_hash)
    R = int(read_json(context["output"]/"power/selection.json")["selected_R"])
    inventory = read_json(context["output"]/"preflight/prospective_compute_inventory.json")
    required = int(inventory["primary_A_controls_state_bytes_by_R"][str(R)]*1.15)
    # Already closed final members need not be allocated again after a resume.
    # Checkpoints live under fits/, not beside the group manifests.  Count only
    # actual primary/A neural members already retained after an interrupted run.
    retained = 0
    for model in CATALOGUE:
        if not is_neural(model):
            continue
        for scope, origin, phase_seeds in (("primary_final", "2025-01-01", context["prereg"]["power"]["final_seed_ids_at_cap"][:R]),
                                           ("branch_A", P1_ORIGIN, A_SEEDS),
                                           ("negative_controls", "2025-01-01", (NEGATIVE_CONTROL_SEED,))):
            recipes_path = context["output"]/"hpo"/("branch_A_recipes.json" if scope == "branch_A" else "primary_recipes.json")
            recipe = read_json(recipes_path)[model]["candidate"]
            if scope == "negative_controls":
                recipe = shuffled_control_candidate(recipe)
            for seed in phase_seeds:
                checkpoint = job_folder(context["output"], model, recipe, seed, origin)/"checkpoint.pt"
                if checkpoint.is_file():
                    retained += checkpoint.stat().st_size
    free = shutil.disk_usage(context["output"]).free
    if max(0, required-retained) > free:
        raise RuntimeError(f"Insufficient disk for every frozen primary/A member: need {required-retained} more bytes, have {free}; no silent R reduction or early test scoring")
    seeds = tuple(context["prereg"]["power"]["final_seed_ids_at_cap"][:R])
    if len(seeds) != R or len(set(seeds)) != R:
        raise ValueError("The final paired seed registry is incomplete")
    primary_recipes = read_json(context["output"]/"hpo/primary_recipes.json")
    A_recipes = read_json(context["output"]/"hpo/branch_A_recipes.json")
    if any(day(A_recipes[m]["selection_outcome_max"]) > date(2024, 10, 31) for m in CATALOGUE):
        raise ValueError("Branch A53 selection must stop by October31 before the seven-day purge")
    primary = freeze_fit_group(context, args, "primary_final", primary_recipes, CATALOGUE, seeds, "2025-01-01")
    branch = freeze_fit_group(context, args, "branch_A", A_recipes, CATALOGUE, A_SEEDS, P1_ORIGIN)
    control_recipes = {m: {**primary_recipes[m], "candidate": shuffled_control_candidate(primary_recipes[m]["candidate"])}
                       for m in NEGATIVE_CONTROL_MODELS}
    controls = freeze_fit_group(context, args, "negative_controls", control_recipes,
                                NEGATIVE_CONTROL_MODELS, (NEGATIVE_CONTROL_SEED,), "2025-01-01")
    base = context["output"]/"all_final_weights"
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    return close_phase(base/"completion.json", inputs_hash,
                       artifacts=[context["output"]/"primary_final/fit_manifest.json",
                                  context["output"]/"branch_A/fit_manifest.json",
                                  context["output"]/"negative_controls/fit_manifest.json"],
                       all_primary_and_A_weights_closed_at_utc=max(primary["all_weights_closed_at_utc"], branch["all_weights_closed_at_utc"]),
                       all_negative_control_weights_closed_at_utc=controls["all_weights_closed_at_utc"],
                       all_weights_including_controls_closed_at_utc=max(primary["all_weights_closed_at_utc"], branch["all_weights_closed_at_utc"], controls["all_weights_closed_at_utc"]),
                       primary_R=R, branch_A_randomized_members=5,
                       negative_control_models=17, negative_control_seed=NEGATIVE_CONTROL_SEED,
                       primary_models=20, branch_A_models=20, test_decoded=False,
                       calibration_outcomes_scored=False)


def descriptor(context, path):
    path = Path(path)
    return {"path": path.relative_to(context["output"]).as_posix(), "sha256": digest(path)}


def make_calibration_receipts(context, manifest, scoring_started):
    receipts = {}
    for model in CATALOGUE:
        members = [r for r in manifest["members"] if r["model"] == model]
        seeds = [str(r["seed"]) for r in members]
        closed = max(r["checkpoint_closed_at_utc"] for r in members)
        record = {"branch": "secondary_frozen_calibration_A53", "model": model,
                  "fit_origin": P1_ORIGIN.isoformat(), "fit_outcome_max": "2024-10-31",
                  "scaler_fit_outcome_max": "2024-10-31",
                  "configuration_selection_outcome_max": "2024-10-31",
                  "branch_selection_outcome_max": "2024-10-31",
                  "primary_fit_replaced": False,
                  "point_predictor_kind": "mean_frozen_seed_members" if is_neural(model) else "deterministic_frozen_predictor",
                  "ensemble_seeds": seeds,
                  "member_checkpoint_sha256": {str(m["seed"]): m["checkpoint_sha256"] for m in members},
                  "member_run_ids": {str(m["seed"]): m["run_id"] for m in members},
                  "member_scaler_sha256": {str(m["seed"]): m["scaler_sha256"] for m in members},
                  "member_config_sha256": {str(m["seed"]): m["config_sha256"] for m in members},
                  "member_completion_sha256": {str(m["seed"]): m["completion_sha256"] for m in members},
                  "member_checkpoint_closed_at_utc": {str(m["seed"]): m["checkpoint_closed_at_utc"] for m in members},
                  "checkpoint_closed_at_utc": closed,
                  "calibration_scoring_started_at_utc": scoring_started,
                  "target_scale": "original_unannualized_daily_volatility",
                  "refit_during_calibration": False, "calibration_used_for_weight_selection": False,
                  "test_used_for_selection": False, "calibration_calendar_days": 53,
                  "calibration_start": P1_ORIGIN.isoformat(), "calibration_end": P1_END.isoformat(),
                  "all_twenty_models_predeclared_before_calibration": True,
                  "inherited_E1_recipe_selection_disclosed": model in ANCHORS,
                  "member_files": {}}
        for member in members:
            folder = member["fit_directory"]
            record["member_files"][str(member["seed"])] = {
                "checkpoint": folder+"/"+member["checkpoint_file"],
                "scaler": folder+"/preprocessing.json", "config": folder+"/setup.json",
                "completion": folder+"/completion.json"}
        receipts[model] = record
    return receipts


def calibration(context, args):
    import numpy as np
    from e2_final_uncertainty import fit_calibration_cohort, build_subgroup_receipt
    inputs_hash = hash_json(context["inputs"])
    gate = phase_receipt(context["output"]/"all_final_weights/completion.json", expected_inputs=inputs_hash)
    R = int(read_json(context["output"]/"power/selection.json")["selected_R"])
    primary_seeds = tuple(context["prereg"]["power"]["final_seed_ids_at_cap"][:R])
    primary_path, A_path = context["output"]/"primary_final/fit_manifest.json", context["output"]/"branch_A/fit_manifest.json"
    verify_fit_manifest(context, primary_path, expected_models=CATALOGUE,
                        expected_seeds=primary_seeds, expected_origin="2025-01-01")
    verify_fit_manifest(context, context["output"]/"negative_controls/fit_manifest.json",
                        expected_models=NEGATIVE_CONTROL_MODELS,
                        expected_seeds=(NEGATIVE_CONTROL_SEED,), expected_origin="2025-01-01")
    branch = verify_fit_manifest(context, A_path, expected_models=CATALOGUE,
                                 expected_seeds=A_SEEDS, expected_origin=P1_ORIGIN)
    base = context["output"]/"calibration"
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    subgroup_path = base/"subgroup_receipt.json"
    if not subgroup_path.exists():
        training = development("2023-12-24", context["root"])
        selected = np.flatnonzero(np.isfinite(training["targets"]))
        subgroup = build_subgroup_receipt(training["targets"][selected],
                    [day(training["calendar"][i]).isoformat() for i in selected],
                    source_sha256=context["inputs"]["development_sources"]["results/eda_development/daily_volatility_development.csv"])
        write_json(subgroup_path, subgroup)
    marker = base/"started.json"
    if not marker.exists():
        write_json(marker, {"scoring_started_at_utc": stamp(),
                            "all_primary_and_A_weights_closed_at_utc": gate["all_primary_and_A_weights_closed_at_utc"],
                            "registration_inputs_sha256": inputs_hash})
    begun = read_json(marker)
    if begun["registration_inputs_sha256"] != inputs_hash:
        raise ValueError("Calibration restart belongs to another immutable model cohort")
    scoring_started = begun["scoring_started_at_utc"]
    if gate["all_primary_and_A_weights_closed_at_utc"] >= scoring_started:
        raise ValueError("Every primary and A weight must close before calibration outcomes are decoded")
    if gate["all_weights_including_controls_closed_at_utc"] >= scoring_started:
        raise ValueError("Every negative-control weight must also close before any calibration score")
    rows = []
    for member in branch["members"]:
        rows.extend(scored_rows(context, member["model"], member["candidate"], member["seed"],
                                P1_ORIGIN, P1_ORIGIN, P1_END, stage="E2_branch_A_calibration",
                                selection_outcome_max="2024-10-31"))
    pred_path = base/"predictions.csv"
    if not pred_path.exists():
        csv_write(pred_path, rows)
    receipts = make_calibration_receipts(context, branch, scoring_started)
    receipt_path = base/"receipts.json"
    if not receipt_path.exists():
        write_json(receipt_path, receipts)
    elif read_json(receipt_path) != clean(receipts):
        raise ValueError("Frozen calibration predictor receipts changed")
    cohort_fit_receipt = {
        "all_primary_and_A_weights_closed_at_utc": gate["all_primary_and_A_weights_closed_at_utc"],
        "calibration_scoring_started_at_utc": scoring_started,
        "primary_fit_manifest": descriptor(context, primary_path),
        "branch_A_fit_manifest": descriptor(context, A_path),
        "calibration_predictions": descriptor(context, pred_path),
        "subgroup_receipt": descriptor(context, subgroup_path)}
    if not (base/"cohort_fit_receipt.json").exists():
        write_json(base/"cohort_fit_receipt.json", cohort_fit_receipt)
    elif read_json(base/"cohort_fit_receipt.json") != cohort_fit_receipt:
        raise ValueError("A resumed calibration cohort changed its frozen artifacts")
    calibrated = fit_calibration_cohort(rows, receipts, artifact_root=context["output"],
                                        cohort_fit_receipt=cohort_fit_receipt)
    if not (base/"frozen_calibrators.json").exists():
        write_json(base/"frozen_calibrators.json", calibrated)
    return close_phase(base/"completion.json", inputs_hash,
                       artifacts=[pred_path, receipt_path, marker, base/"cohort_fit_receipt.json",
                                  base/"frozen_calibrators.json", subgroup_path],
                       all_twenty_models_calibrated=True, calibration_days=53,
                       calibration_scoring_started_at_utc=scoring_started,
                       all_primary_and_A_weights_closed_before_calibration=True,
                       future_E2_test_prices_or_labels_decoded=False)


def oof_fit_records(context, rows):
    records = {}
    for row in rows:
        if row["run_id"] in records:
            continue
        folder = local(context["output"], row["fit_directory"])
        verify_job(folder, hash_json(context["inputs"]))
        key = hash_json({"fit_sha256": digest(folder/"completion.json"),
                         "start": day(row["fit_origin"]), "end": month_end(row["fit_origin"]),
                         "stage": "E2_nested_development",
                         "configuration_selection_outcome_max": day(row["configuration_selection_outcome_max"])})
        score_folder = context["output"]/"development_scores"/key
        phase_receipt(score_folder/"completion.json", expected_inputs=hash_json(context["inputs"]))
        records[row["run_id"]] = {"folder": row["fit_directory"],
                                   "scoring_receipt": (score_folder/"completion.json").relative_to(context["output"]).as_posix(),
                                   "predictions_file": (score_folder/"predictions.csv").relative_to(context["output"]).as_posix()}
    return records


def run_oof_errors(context, args):
    from e2_final_uncertainty import analyze_final_oof
    inputs_hash = hash_json(context["inputs"])
    phase_receipt(context["output"]/"hpo/completion.json", expected_inputs=inputs_hash)
    base = context["output"]/"oof_errors"
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    rows = csv_read(context["output"]/"hpo/nested_oof_predictions.csv")
    validate_development_panel(rows)
    taxonomy = read_json(local(context["root"], context["design"].get("error_taxonomy", "book_e2/_static/remediation_error_results/training_taxonomy.json")))
    result = analyze_final_oof(rows, taxonomy, models=list(CATALOGUE), seeds=list(HPO_SEEDS),
                               deterministic_models=list(DETERMINISTIC), artifact_root=context["output"],
                               fit_records=oof_fit_records(context, rows),
                               registration_inputs_sha256=inputs_hash,
                               top3=read_json(context["output"]/"hpo/efficiency_selection.json")["top3_development"])
    for name, value in result.items():
        if name not in ("summary", "private_audit_cases", "private_frontier"):
            raise ValueError("Unrecognized OOF export: do not expose individual errors implicitly")
        write_json(base/(name+".json"), value)
    return close_phase(base/"completion.json", inputs_hash,
                       artifacts=list(base.glob("*.json")), models=list(CATALOGUE),
                       unique_daily_origins_per_model=305, unique_audit_sample_per_model=100,
                       source_stage="E2_nested_development", P0_relabelled=False,
                       individual_cases_remain_private=True, test_decoded=False)


def training_rows(context, fit_manifest, *, source_stage, split=None):
    """Expose saved refit errors on a common structural training calendar.

    The seasonal naive needs Y[t-7], which is unavailable for the first seven
    otherwise admissible D7 outcomes. These positions stay explicit but blank
    for every model's training diagnostic; fitted model supports are unchanged.
    An invalid prediction on an otherwise available common date is a failure.
    """
    import numpy as np
    from dl_data import CALENDAR
    maximum = max(day(m["fit_outcome_max"]) for m in fit_manifest["members"])
    data = development(maximum, context["root"])
    target = np.asarray(data["targets"], float)
    admissible = np.asarray(data["eligible_by_history"][7], bool).copy()
    for lag in (1, 7):
        available = np.zeros(len(target), bool)
        available[lag:] = np.isfinite(target[:-lag])
        admissible &= available
    rows = []
    for member in fit_manifest["members"]:
        folder = local(context["output"], member["fit_directory"])
        fit = verify_job(folder, hash_json(context["inputs"]))
        setup = read_json(folder/"setup.json")
        if "training_evaluation.npz" not in fit["artifacts"]:
            raise ValueError("The final refit lacks its own sealed inference-mode training predictions")
        with np.load(folder/"training_evaluation.npz", allow_pickle=False) as saved:
            values = {int(i): (float(y), float(p)) for i, y, p in
                      zip(saved["train_indices"], saved["train_truth"], saved["train_prediction"])}
        for i, timestamp in enumerate(CALENDAR):
            d = day(timestamp)
            if d > day(member["fit_outcome_max"]):
                break
            pair = values.get(i)
            eligible = bool(i < len(admissible) and admissible[i])
            if eligible and (pair is None or not np.isfinite(pair).all()):
                raise ValueError("A common structurally available training date lacks a finite saved predictor value")
            row = {"model": member["model"], "seed": member["seed"], "outcome_day": d.isoformat(),
                   "y_true": pair[0] if eligible else None, "y_pred": pair[1] if eligible else None,
                   "eligible": bool(eligible), "source_stage": source_stage,
                   "fit_origin": member["fit_origin"], "fit_outcome_max": member["fit_outcome_max"],
                   "scaler_fit_outcome_max": setup["scaler_fit_outcome_max"],
                   "configuration_selection_outcome_max": member["configuration_selection_outcome_max"],
                   "run_id": member["run_id"], "checkpoint_sha256": member["checkpoint_sha256"],
                   "scaler_sha256": member["scaler_sha256"], "config_sha256": member["config_sha256"],
                   "checkpoint_closed_at_utc": member["checkpoint_closed_at_utc"],
                   "scoring_started_at_utc": setup["training_inference_started_at_utc"]}
            if split is not None:
                row["split"] = split
            rows.append(row)
    return rows


def run_A_overfit(context):
    from e2_final_statistics import analyze_A53_overfit
    inputs_hash = hash_json(context["inputs"])
    phase_receipt(context["output"]/"calibration/completion.json", expected_inputs=inputs_hash)
    base = context["output"]/"A53_overfit"
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    manifest = read_json(context["output"]/"branch_A/fit_manifest.json")
    train = training_rows(context, manifest, source_stage="E2_A53_train_validation", split="train")
    valid = [{**row, "source_stage": "E2_A53_train_validation", "split": "validation"}
             for row in csv_read(context["output"]/"calibration/predictions.csv")]
    # Normalize schemas: calibration has extra loss/cost context fields, which
    # are not part of the formal train/validation contrast.
    fields = tuple(train[0])
    valid = [{key: row.get(key) for key in fields} for row in valid]
    csv_write(base/"train_validation_predictions.csv", train+valid)
    result = analyze_A53_overfit(train+valid)
    write_json(base/"summary.json", result)
    return close_phase(base/"completion.json", inputs_hash,
                       artifacts=[base/"train_validation_predictions.csv", base/"summary.json"],
                       scope="A53 same-predictor inference-mode training-versus-purged-validation diagnostics; primary Dec24 refit not substituted",
                       randomized_members=5, deterministic_trajectories=1, validation_dates=53)


def final_member_rows(context, member, data, indices, history, *, stage, scoring_started):
    folder = local(context["output"], member["fit_directory"])
    setup = read_json(folder/"setup.json")
    pred, raw, elapsed = prediction_from_fit(context, folder, data, indices, history=history)
    return [{"model": member["model"], "seed": member["seed"],
             "candidate_id": member["candidate"]["candidate_id"],
             "outcome_day": day(data["calendar"][i]).isoformat(),
             "y_true": float(data["targets"][i]), "y_pred": float(pred[j]), "raw_y_pred": float(raw[j]),
             "persistence_pred": float(data["persistence"][i]),
             "high_volatility_known": bool(data["persistence"][i] > HIGH_VOLATILITY),
             "mase_scale": setup["mase_scale"], "eligible": True,
             "fit_origin": member["fit_origin"], "fit_outcome_max": member["fit_outcome_max"],
             "scaler_fit_outcome_max": setup["scaler_fit_outcome_max"],
             "configuration_selection_outcome_max": member["configuration_selection_outcome_max"],
             "source_stage": stage, "run_id": member["run_id"],
             "fit_directory": member["fit_directory"], "checkpoint_sha256": member["checkpoint_sha256"],
             "scaler_sha256": member["scaler_sha256"], "config_sha256": member["config_sha256"],
             "checkpoint_closed_at_utc": member["checkpoint_closed_at_utc"],
             "scoring_started_at_utc": scoring_started,
             "development_latency_ms_per_window": float(elapsed[j])}
            for j, i in enumerate(indices)]


def score_final(context, args):
    """The ONLY phase that decodes prices/labels in the inherited known test."""
    import numpy as np
    from planned_study_data import load_reserved_data
    from e2_final_test_data import load_final_state_history
    inputs_hash = hash_json(context["inputs"])
    check_registration(context)
    frozen = phase_receipt(context["output"]/"all_final_weights/completion.json", expected_inputs=inputs_hash)
    calibrated = phase_receipt(context["output"]/"calibration/completion.json", expected_inputs=inputs_hash)
    phase_receipt(context["output"]/"ablation/completion.json", expected_inputs=inputs_hash)
    run_cost(context, args, phase="primary_final")
    R = int(read_json(context["output"]/"power/selection.json")["selected_R"])
    seeds = tuple(context["prereg"]["power"]["final_seed_ids_at_cap"][:R])
    primary_path, A_path = context["output"]/"primary_final/fit_manifest.json", context["output"]/"branch_A/fit_manifest.json"
    primary = verify_fit_manifest(context, primary_path, expected_models=CATALOGUE, expected_seeds=seeds, expected_origin="2025-01-01")
    branch = verify_fit_manifest(context, A_path, expected_models=CATALOGUE, expected_seeds=A_SEEDS, expected_origin=P1_ORIGIN)
    controls_path = context["output"]/"negative_controls/fit_manifest.json"
    controls = verify_fit_manifest(context, controls_path, expected_models=NEGATIVE_CONTROL_MODELS,
                                   expected_seeds=(NEGATIVE_CONTROL_SEED,), expected_origin="2025-01-01")
    base = context["output"]/"final_scoring"
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    signature = {"registration_inputs_sha256": inputs_hash,
                 "primary_fit_manifest_sha256": digest(primary_path),
                 "A_fit_manifest_sha256": digest(A_path),
                 "negative_controls_fit_manifest_sha256": digest(controls_path),
                 "calibration_completion_sha256": digest(context["output"]/"calibration/completion.json"),
                 "single_scoring_phase": True}
    marker = base/"started.json"
    if marker.exists():
        begun = read_json(marker)
        if any(begun.get(k) != v for k, v in signature.items()):
            raise ValueError("The one E2 scoring phase cannot resume with changed weights, calibration or inputs")
    else:
        begun = {**signature, "evaluation_scoring_started_at_utc": stamp(),
                 "all_primary_and_A_weights_closed_at_utc": frozen["all_primary_and_A_weights_closed_at_utc"],
                 "calibration_closed_at_utc": calibrated["completed_at_utc"],
                 "prior_E2_scoring_phases": 0, "historical_E1_test_already_known": True}
        write_json(marker, begun)
    started = begun["evaluation_scoring_started_at_utc"]
    if not frozen["all_primary_and_A_weights_closed_at_utc"] < calibrated["completed_at_utc"] < started:
        raise ValueError("All weights and every calibrator must close before the first E2 test decode")
    if not frozen["all_weights_including_controls_closed_at_utc"] < calibrated["completed_at_utc"]:
        raise ValueError("Label-shuffle control weights must close before calibration and test decoding")
    configure(202611067)
    # No caller above this point opens any2025/2026 market payload.
    data, reserve_audit = load_reserved_data(context["root"], args.raw_root,
                                            read_json(context["output"]/"test_reservation.json"), base/"reserved")
    state = load_final_state_history(context["root"], args.raw_root,
                                     read_json(context["output"]/"state_history_reservation.json"), base/"historical_state")
    history_data = development("2024-12-24", context["root"])
    history = {day(d): float(y) if np.isfinite(y) else None
               for d, y in zip(history_data["calendar"], history_data["targets"]) if day(d) <= date(2024, 12, 24)}
    history.update(state["history"])
    for d, y in zip(data["calendar"], data["targets"]):
        d = day(d)
        if d in history and history[d] is not None and np.isfinite(y):
            if not math.isclose(float(history[d]), float(y), rel_tol=1e-9, abs_tol=1e-13):
                raise ValueError("Continuous state-history and reserved-context target reconstruction disagree")
        if d >= date(2026, 9, 1):
            history[d] = float(y) if np.isfinite(y) else None
        # Do not overwrite genuine August1–7 sigma7 values from the continuous
        # state history with the reserve loader's artificial context warm-up NaN.
    indices = np.asarray(data["test_indices"], int)
    if [day(data["calendar"][i]) for i in indices] != list(calendar("2026-09-01", "2026-10-30")):
        raise ValueError("The inherited sixty-day common test cohort changed")
    for label, manifest, stage in (("primary", primary, "E2_final_primary"),
                                   ("A", branch, "E2_branch_A_evaluation"),
                                   ("negative_controls", controls, "E2_label_shuffle_negative_control")):
        directory = base/label
        directory.mkdir(exist_ok=True)
        combined = []
        for member in manifest["members"]:
            file = directory/(member["run_id"]+".csv")
            closed = directory/(member["run_id"]+".completion.json")
            if file.exists() or closed.exists():
                if not file.exists() or not closed.exists():
                    raise RuntimeError("Interrupted individual scoring artifact requires an infrastructure audit, not silent replacement")
                phase_receipt(closed, expected_inputs=inputs_hash)
                rows = csv_read(file)
            else:
                rows = final_member_rows(context, member, data, indices, history, stage=stage, scoring_started=started)
                csv_write(file, rows)
                close_phase(closed, inputs_hash, artifacts=[file], checkpoint_sha256=member["checkpoint_sha256"],
                            single_phase_id=hash_json(signature), source_stage=stage)
            combined.extend(rows)
        aggregate = base/(label+"_predictions.csv")
        if not aggregate.exists():
            csv_write(aggregate, combined)
    cohort = {**read_json(context["output"]/"calibration/cohort_fit_receipt.json"),
              "calibration_closed_at_utc": calibrated["completed_at_utc"],
              "evaluation_scoring_started_at_utc": started,
              "calibrators_file": descriptor(context, context["output"]/"calibration/frozen_calibrators.json"),
              "evaluation_predictions": descriptor(context, base/"A_predictions.csv"),
              "primary_predictions": descriptor(context, base/"primary_predictions.csv")}
    if not (base/"cohort_receipt.json").exists():
        write_json(base/"cohort_receipt.json", cohort)
    # Diagnostics are predeclared fixed functions and first-five-member subsets,
    # evaluated within this same scoring phase. They never alter any checkpoint.
    auxiliary = run_auxiliary_diagnostics(context, args, data, indices, history, primary, begun)
    return close_phase(base/"completion.json", inputs_hash,
                       artifacts=[marker, base/"primary_predictions.csv", base/"A_predictions.csv", base/"cohort_receipt.json",
                                  base/"negative_controls_predictions.csv",
                                  base/"reserved/raw_source_audit.json", base/"historical_state/state_history_receipt.json",
                                  auxiliary],
                       one_E2_scoring_phase=True, primary_randomized_members=R,
                       negative_control_models=17, negative_control_test_labels_permuted=False,
                       A_randomized_members=5, models=20, evaluation_days=60,
                       all_checkpoints_and_calibrators_closed_before_first_decode=True,
                       test_period_previously_known_from_E1=True, training_or_HPO_after_scoring=False)


def analyze(context, args):
    from e2_final_statistics import analyze_final
    from e2_final_uncertainty import evaluate_final_uncertainty
    inputs_hash = hash_json(context["inputs"])
    phase_receipt(context["output"]/"final_scoring/completion.json", expected_inputs=inputs_hash)
    base = context["output"]/"analysis"
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    primary_path = context["output"]/"final_scoring/primary_predictions.csv"
    primary = csv_read(primary_path)
    costs = csv_read(context["output"]/"costs/primary_final/latency_blocks.csv")
    efficiency = read_json(context["output"]/"hpo/efficiency_selection.json")
    prim_train = training_rows(context, read_json(context["output"]/"primary_final/fit_manifest.json"),
                               source_stage="E2_primary_train_inference")
    result = analyze_final(primary, [], costs, context["prereg"], output=base/"primary",
                           primary_training_panel=prim_train, efficiency_selection=efficiency)
    write_json(base/"primary_summary.json", result)
    run_A_overfit(context)
    run_oof_errors(context, args)
    write_json(base/"predictability_reference.json", run_predictability_reference(context, primary, efficiency))
    write_json(base/"leakage_controls.json", run_leakage_diagnostics(context))
    uncertainty = evaluate_final_uncertainty(
        csv_read(context["output"]/"calibration/predictions.csv"),
        csv_read(context["output"]/"final_scoring/A_predictions.csv"),
        read_json(context["output"]/"calibration/receipts.json"),
        primary_rows=primary, subgroup_receipt=read_json(context["output"]/"calibration/subgroup_receipt.json"),
        top3=efficiency["top3_development"], artifact_root=context["output"],
        cohort_receipt=read_json(context["output"]/"final_scoring/cohort_receipt.json"),
        frozen_calibrators=read_json(context["output"]/"calibration/frozen_calibrators.json"))
    private_keys = {"summary", "private_calibrators", "private_evaluation_frontier"}
    for name, value in uncertainty.items():
        if name not in private_keys:
            raise ValueError("Unexpected uncertainty output; preserve the aggregate/private boundary explicitly")
        write_json(base/("uncertainty_"+name+".json"), value)
    return close_phase(base/"completion.json", inputs_hash,
                       artifacts=list(base.glob("*.json")), source_scoring_sha256=digest(context["output"]/"final_scoring/completion.json"),
                       models=20, supplementary_A53_and_OOF_analysis=True,
                       individual_forecasts_error_ledgers_and_checkpoints_remain_private=True,
                       new_benchmark_model_fits=0, diagnostic_domain_classifiers_fitted=True,
                       test_scores_recomputed=False)


def run_leakage_diagnostics(context):
    """Analyze frozen shuffle predictors and fixed post-freeze shift diagnostics.

    Domain classifiers are diagnostic-only models. Their registered blocked-CV
    fits never replace, tune or update any benchmark or negative-control weight.
    """
    import numpy as np
    from e2_leakage_controls import analyze_shuffled_predictions, adversarial_validation
    inputs_hash = hash_json(context["inputs"])
    closed = phase_receipt(context["output"]/"all_final_weights/completion.json", expected_inputs=inputs_hash)
    calibrated = phase_receipt(context["output"]/"calibration/completion.json", expected_inputs=inputs_hash)
    scoring = read_json(context["output"]/"final_scoring/started.json")
    manifest = verify_fit_manifest(context, context["output"]/"negative_controls/fit_manifest.json",
                                    expected_models=NEGATIVE_CONTROL_MODELS,
                                    expected_seeds=(NEGATIVE_CONTROL_SEED,), expected_origin="2025-01-01")
    receipts, means = {}, {}
    for member in manifest["members"]:
        setup = read_json(local(context["output"], member["fit_directory"])/"setup.json")
        model = member["model"]
        receipts[model] = {"model": model, "seed": member["seed"],
                           "fit_origin": member["fit_origin"], "fit_outcome_max": member["fit_outcome_max"],
                           "configuration_selection_outcome_max": member["configuration_selection_outcome_max"],
                           "label_permutation": setup["label_permutation"],
                           "checkpoint_sha256": member["checkpoint_sha256"],
                           "checkpoint_closed_at_utc": member["checkpoint_closed_at_utc"],
                           "registration_inputs_sha256": inputs_hash}
        means[model] = setup["original_training_target_mean"]
    receipt = {"control_weights_closed_at_utc": manifest["all_weights_closed_at_utc"],
               "all_primary_and_A_weights_closed_at_utc": closed["all_primary_and_A_weights_closed_at_utc"],
               "all_weights_including_controls_closed_at_utc": closed["all_weights_including_controls_closed_at_utc"],
               "calibration_closed_at_utc": calibrated["completed_at_utc"],
               "evaluation_scoring_started_at_utc": scoring["evaluation_scoring_started_at_utc"],
               "checkpoint_count": 17, "registration_inputs_sha256": inputs_hash}
    shuffle = analyze_shuffled_predictions(csv_read(context["output"]/"final_scoring/negative_controls_predictions.csv"),
                training_mean_by_model=means, control_receipts=receipts, phase_receipt=receipt)
    training = development("2024-12-24", context["root"])
    train_indices = split_indices(training, "2025-01-01")["train"]
    primary = read_json(context["output"]/"primary_final/fit_manifest.json")
    pre = read_json(local(context["output"], primary["members"][0]["fit_directory"])/"preprocessing.json")
    train_x = preprocess_x(training, train_indices, pre)
    from dl_data_v2 import relative_features
    with np.load(context["output"]/"final_diagnostics/final_context.npz", allow_pickle=False) as context_data:
        raw, test_days = context_data["raw_windows"].copy(), context_data["origins"].tolist()
    test_x = ((relative_features(raw)-np.asarray(pre["x_mean"])) / np.asarray(pre["x_scale"])).astype(np.float32)
    shift = adversarial_validation(train_x, [day(training["calendar"][i]).isoformat() for i in train_indices],
                                    test_x, test_days, repetitions=2000, seed=690201)
    return {"label_shuffle": shuffle, "adversarial_domain_validation": shift,
            "phase_order": receipt, "benchmark_weights_fitted_by_analyzer": 0,
            "domain_classifier_fits_are_diagnostic_only": True}


def final_support_audit(context):
    """Audit actual support for every distinct new training origin, on development only."""
    import numpy as np
    data = development("2024-12-30", context["root"])
    origins = sorted({b["origin"] for o in OUTER_ORIGINS for b in inner_blocks(o)}
                     | set(OUTER_ORIGINS) | {P1_ORIGIN, date(2025, 1, 1)})
    results = []
    for origin in origins:
        split = split_indices(data, origin)
        dates = [day(d) for d in data["calendar"]]
        for left, right in (("inner_train", "inner_validation"),):
            if set(split[left]) & set(split[right]):
                raise ValueError("Final support audit detected shared training and stopping targets")
            if dates[max(split[left])]+8*DAY > dates[min(split[right])]:
                raise ValueError("Final support audit detected an insufficient target embargo")
        if dates[max(split["train"])]+8*DAY > origin:
            raise ValueError("A new fit contains an outcome after its causal training cutoff")
        # Overlapping rolling windows are expected. Exact full-window duplicates
        # across the inner split are separately checked, without deduplicating days.
        raw = np.asarray(data["windows_by_history"][7])
        left_hashes = {sha256(np.asarray(raw[i], dtype="<f8").tobytes()).hexdigest() for i in split["inner_train"]}
        duplicate_count = sum(sha256(np.asarray(raw[i], dtype="<f8").tobytes()).hexdigest() in left_hashes
                              for i in split["inner_validation"])
        if duplicate_count:
            raise ValueError("Exact OHLCV windows cross an inner split; resolve before model fitting")
        results.append({**split["metadata"], "exact_cross_split_duplicate_windows": duplicate_count,
                        "scaler_fitting_indices_equal_final_training_indices": True,
                        "outcome_calendar_compressed": False})
    return {"status": "passed", "distinct_origins": len(origins), "supports": results,
            "development_calendar_days": len(data["calendar"]), "test_payloads_decoded": 0,
            "scope": "Actual indices and full raw-window duplication audit of all nested, monthly, P1/A53 and primary-refit supports"}


def run_ablation(context, args):
    """Matched, fresh-init control plus three lesions on purged development53."""
    from e2_final_statistics import compare_matched_ablation_errors
    inputs_hash = hash_json(context["inputs"])
    global_weights = phase_receipt(context["output"]/"all_final_weights/completion.json", expected_inputs=inputs_hash)
    base = context["output"]/"ablation"
    if (base/"completion.json").exists():
        return phase_receipt(base/"completion.json", expected_inputs=inputs_hash)
    base.mkdir(parents=True, exist_ok=True)
    efficiency = read_json(context["output"]/"hpo/efficiency_selection.json")
    model = efficiency["best_top10"]
    definitions = read_json(context["output"]/"preflight/ablation_definitions.json")[model]
    if len(definitions) != 3 or model not in TOP10:
        raise ValueError("Ablation requires the prospectively defined three lesions of the development-selected Top10 model")
    recipe = read_json(context["output"]/"hpo/primary_recipes.json")[model]
    R = int(read_json(context["output"]/"power/selection.json")["selected_R"])
    seeds = tuple(context["prereg"]["power"]["final_seed_ids_at_cap"][:R])
    variants = {}
    for variant in ("control", *definitions):
        candidate = json.loads(json.dumps(recipe["candidate"]))
        candidate["candidate_id"] += "-ablation-"+variant
        if variant != "control":
            candidate["parameters"]["ablation_id"] = variant
        variants[variant] = candidate
    specification = {"model": model, "seeds": list(seeds), "variants": variants,
                     "definitions_sha256": hash_json(definitions), "origin": P1_ORIGIN,
                     "validation_end": P1_END, "registration_inputs_sha256": inputs_hash,
                     "selection_outcome_max": recipe["selection_outcome_max"],
                     "test_used": False, "same_final_R": True}
    schedule = base/"schedule.json"
    if schedule.exists():
        saved_schedule = read_json(schedule)
        if {k: v for k, v in saved_schedule.items() if k != "ablation_schedule_closed_at_utc"} != clean(specification):
            raise ValueError("The preregistered ablation model, lesions, recipe or seeds changed")
    else:
        saved_schedule = {**specification, "ablation_schedule_closed_at_utc": stamp()}
        write_json(schedule, saved_schedule)
    schedule_hash = digest(schedule)
    schedule_closed = saved_schedule["ablation_schedule_closed_at_utc"]
    global_closed = global_weights["all_primary_and_A_weights_closed_at_utc"]
    runner = TrialRunner(context, args)
    combined = []
    for seed in seeds:
        for variant, candidate in variants.items():
            attempts = runner.run([(model, candidate, seed, P1_ORIGIN)], budget_scope="ablation_development")
            if attempts[0]["record"]["status"] != "passed":
                raise RuntimeError("Matched ablation member failed; no lesion or seed substitution is permitted")
            rows = scored_rows(context, model, candidate, seed, P1_ORIGIN, P1_ORIGIN, P1_END,
                               stage="E2_ablation_control_2024", selection_outcome_max=recipe["selection_outcome_max"])
            folder = job_folder(context["output"], model, candidate, seed, P1_ORIGIN)
            fit_started = read_json(folder/"started.json")["started_at_utc"]
            if not schedule_closed < fit_started or any(not global_closed < row["scoring_started_at_utc"] for row in rows):
                raise ValueError("Ablation schedule and primary/A closure must precede the fixed matched development experiment")
            combined.extend({**row, "variant": variant, "base_model": model,
                             "ablation_schedule_sha256": schedule_hash,
                             "ablation_schedule_closed_at_utc": schedule_closed,
                             "ablation_fit_started_at_utc": fit_started,
                             "all_primary_A_weights_closed_at_utc": global_closed} for row in rows)
    panel = base/"predictions.csv"
    if not panel.exists():
        csv_write(panel, combined)
    result = compare_matched_ablation_errors(combined)
    if not (base/"summary.json").exists():
        write_json(base/"summary.json", result)
    return close_phase(base/"completion.json", inputs_hash, artifacts=[schedule, panel, base/"summary.json"],
                       model=model, variants=4, paired_randomized_seeds=R, development_days=53,
                       all_primary_and_A_weights_closed_before_ablation_scoring=True,
                       test_payloads_opened=0, first_seed_all_variant_checkpoints_retained=True,
                       inference_scope="Separate exploratory three-contrast two-sided practical-effect Holm family; same R, not separately power-certified")


def run_auxiliary_diagnostics(context, args, data, indices, history, primary, begun):
    """Frozen first-five diagnostics inside the one E2 scoring phase."""
    import numpy as np
    import torch
    from e2_final_diagnostics import (evaluate_final_diagnostics, fit_perturbation_reference,
                                      torch_raw_predictor, preprocessing_fingerprint)
    inputs_hash = hash_json(context["inputs"])
    base = context["output"]/"final_diagnostics"
    completion = base/"completion.json"
    if completion.exists():
        phase_receipt(completion, expected_inputs=inputs_hash)
        return completion
    if base.exists() and any(base.iterdir()):
        raise RuntimeError("Interrupted diagnostics need an infrastructure audit before replay; no silent replacement of case outputs")
    diagnostic_seeds = tuple(range(670101, 670106))
    entries = {}
    training = development("2024-12-24", context["root"])
    admitted = split_indices(training, "2025-01-01")["train"]
    reference = fit_perturbation_reference(np.asarray(training["windows_by_history"][7])[admitted],
        training_outcome_dates=[day(training["calendar"][i]).isoformat() for i in admitted],
        maximum_training_outcome="2024-12-24")
    for name in CATALOGUE:
        members = [r for r in primary["members"] if r["model"] == name
                   and (not is_neural(name) or int(r["seed"]) in diagnostic_seeds)]
        if len(members) != (5 if is_neural(name) else 1):
            raise ValueError("The fixed diagnostic first-five/one trajectory cohort is incomplete")
        if is_neural(name):
            loaded, callbacks = [], []
            for member in sorted(members, key=lambda r: int(r["seed"])):
                folder = local(context["output"], member["fit_directory"])
                verify_job(folder, inputs_hash)
                pre = read_json(folder/"preprocessing.json")
                model = build_neural(name, member["candidate"], context["roots"])
                model.load_state_dict(torch.load(folder/"checkpoint.pt", map_location="cpu", weights_only=True))
                model._e2_fit_started = True
                model.eval()
                callbacks.append(torch_raw_predictor(model, pre, batch_size=1))
                loaded.append({"model": model, "preprocessing": pre, "seed": int(member["seed"]),
                               "scaler_sha256": preprocessing_fingerprint(pre),
                               "explanation_baseline_receipt": pre["explanation_baseline_receipt"]})
            def predict_raw(raw, callbacks=tuple(callbacks)):
                return np.stack([callback(raw) for callback in callbacks])
            entries[name] = {"input_kind": "OHLCV", "predict_raw": predict_raw,
                             "seed_ids": diagnostic_seeds, "members": loaded,
                             "fit_outcome_max": "2024-12-24"}
            continue
        member = members[0]
        folder = local(context["output"], member["fit_directory"])
        verify_job(folder, inputs_hash)
        pre = read_json(folder/"preprocessing.json")
        if name == "Ridge":
            from dl_data_v2 import relative_features
            with np.load(folder/"model.npz", allow_pickle=False) as saved:
                coefficient, intercept = saved["coef"].copy(), float(saved["intercept"])
            def normalized(x, coefficient=coefficient, intercept=intercept, scale=pre["y_scale"]):
                x = np.asarray(x, dtype=float).reshape(len(x), -1)
                return np.maximum(0., (x@coefficient+intercept)*scale)
            def raw_predict(x, pre=pre, normalized=normalized):
                standardized = ((relative_features(np.asarray(x)) - np.asarray(pre["x_mean"])) /
                                np.asarray(pre["x_scale"])).astype(np.float32)
                return normalized(standardized)[None, :]
            entries[name] = {"input_kind": "OHLCV", "seed_ids": ["deterministic"],
                             "predict_raw": raw_predict, "predict_normalized": normalized,
                             "preprocessing": pre, "scaler_sha256": preprocessing_fingerprint(pre),
                             "explanation_baseline_receipt": pre["explanation_baseline_receipt"],
                             "linear_coefficients": coefficient*pre["y_scale"],
                             "linear_intercept": intercept*pre["y_scale"], "fit_outcome_max": "2024-12-24"}
        else:
            values = [float(training["targets"][i]) for i in admitted]
            if name == "ARIMA":
                frozen_bytes = (folder/"model.pickle").read_bytes()
                def predict_history(past, origin, frozen_bytes=frozen_bytes):
                    from e2_temporal_baselines import benchmark_nonnegative
                    mapping = {day(d): float(v) if np.isfinite(v) else None
                               for d, v in zip(past["dates"], past["values"])}
                    if any(d >= day(origin) for d in mapping):
                        raise ValueError("ARIMA diagnostic callback accepts strictly past target history")
                    result = benchmark_nonnegative(pickle.loads(frozen_bytes).predict(mapping, day(origin)))
                    if not result.eligible:
                        raise ValueError("Perturbed ARIMA history has no causal forecast")
                    return float(result.value)
                lag = None
            else:
                lag = 1 if name == "Persistence" else 7
                def predict_history(past, origin, lag=lag):
                    mapping = {day(d): float(v) for d, v in zip(past["dates"], past["values"])}
                    if any(d >= day(origin) for d in mapping):
                        raise ValueError("Naive diagnostic callback accepts strictly past target history")
                    return float(mapping[day(origin)-lag*DAY])
            entries[name] = {"input_kind": "endogenous", "seed_ids": ["deterministic"],
                             "predict_from_history": predict_history, "fit_outcome_max": "2024-12-24",
                             "training_sigma_mean": float(np.mean(values)), "lag": lag}
    ordered_history = sorted(history)
    history_source = {"dates": [d.isoformat() for d in ordered_history],
                      "values": np.asarray([np.nan if history[d] is None else history[d] for d in ordered_history], float)}
    raw = np.asarray(data["windows_by_history"][7])[indices]
    truth = np.asarray(data["targets"])[indices]
    origins = [day(data["calendar"][i]).isoformat() for i in indices]
    closure = {"all_model_weights_closed": True, "catalogue": list(CATALOGUE),
               "primary_fit_manifest_sha256": digest(context["output"]/"primary_final/fit_manifest.json"),
               "A_fit_manifest_sha256": digest(context["output"]/"branch_A/fit_manifest.json"), **begun}
    efficiency = read_json(context["output"]/"hpo/efficiency_selection.json")
    evaluate_final_diagnostics(entries, raw, truth, origins,
        best3_development=efficiency["top3_development"], reference=reference,
        history_source=history_source, private_output=base, closure_receipt=closure,
        eligible=np.ones(len(indices), bool), seed_ids=diagnostic_seeds, repetitions=2000, block_length=7)
    np.savez_compressed(base/"final_context.npz", raw_windows=raw, truth=truth,
                        origins=np.asarray(origins, dtype="U10"))
    return_path = base/"completion.json"
    close_phase(return_path, inputs_hash, artifacts=list(base.iterdir()),
                all20_robustness_and3_development_selected_explanations=True,
                no_fits=True, first_five_seed_subset=True, inputs_after_global_checkpoint_closure=True)
    return return_path


def run_predictability_reference(context, primary, efficiency):
    import numpy as np
    from e2_predictability_reference import InnovationReference, predictability_summary
    training = development("2023-12-24", context["root"])
    take = [i for i, d in enumerate(training["calendar"]) if day(d) <= date(2023, 12, 24)]
    law = InnovationReference.from_training_daily([day(training["calendar"][i]) for i in take],
                                                 np.asarray(training["features"])[take, 3])
    with np.load(context["output"]/"final_diagnostics/final_context.npz", allow_pickle=False) as data:
        raw, truth, dates = data["raw_windows"].copy(), data["truth"].copy(), data["origins"].tolist()
    best, baseline = efficiency["best_model"], efficiency["best_baseline"]
    def predictions(name):
        records = [r for r in primary if r["model"] == name]
        seed_ids = sorted({str(r["seed"]) for r in records}, key=lambda s: -1 if s == "deterministic" else int(s))
        arrays = []
        for seed in seed_ids:
            rows = sorted([r for r in records if str(r["seed"]) == seed], key=lambda r: r["outcome_day"])
            if [r["outcome_day"] for r in rows] != dates:
                raise ValueError("Predictability reference needs the unchanged complete primary evaluation calendar")
            arrays.append([float(r["y_pred"]) for r in rows])
        return np.asarray(arrays), seed_ids
    b, bs = predictions(baseline)
    m, ms = predictions(best)
    paired = ms if len(ms) > 1 else bs if len(bs) > 1 else None
    return {str(block): predictability_summary(law, dates, raw[:, :, 3], truth, b, m,
              baseline_name=baseline, best_name=best, best_selection_scope="frozen_development_selection",
              seed_ids=paired, repetitions=2000, block_length=block, seed=202611066)
            for block in (7, 14, 28)}


def create_context(args):
    design, hpo, prereg, candidates = read_design(args.design)
    roots = source_roots(args.source_roots_json)
    output = private_output(args.output_dir)
    return {"root": ROOT, "output": output, "design": design, "hpo": hpo,
            "prereg": prereg, "candidates": candidates, "roots": roots,
            "inputs": collect_inputs(args.design, design, roots)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES+("worker",), default="plan")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/e2_completion_20261106/final_run"))
    parser.add_argument("--source-roots-json", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models", nargs="*", choices=TUNABLE)
    parser.add_argument("--job-json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("Use one to four fresh CPU fit workers, each with one numerical thread")
    if args.phase == "plan":
        result = planned_inventory()
        result.update({"plan_only": True, "dataset_payloads_opened": 0,
                       "design_path": str(args.design.resolve()), "design_exists": args.design.is_file(),
                       "source_roots_schema": {"model_name": "absolute private author-checkout path"}})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    context = create_context(args)
    context["output"].mkdir(parents=True, exist_ok=True)
    if args.phase in ("register", "score", "all") and args.raw_root is None:
        parser.error("--raw-root is required for metadata reservation or the explicit final scoring phase")
    if args.phase not in ("preflight", "register", "all"):
        check_registration(context)
    actions = {"preflight": preflight, "register": lambda c, a: register(a, c), "hpo": run_hpo,
               "p1": run_p1, "power": run_power, "final-fit": run_final_fits,
               "calibration": calibration, "cost": run_cost, "oof-errors": run_oof_errors,
               "ablation": run_ablation, "score": score_final, "analyze": analyze, "worker": worker}
    if args.phase == "worker" and args.job_json is None:
        parser.error("An internal worker requires its content-addressed job payload")
    if args.phase == "all":
        if args.models:
            parser.error("The all phase cannot silently restrict the fifteen-model HPO catalogue")
        result = None
        for phase in ("preflight", "register", "hpo", "oof-errors", "p1", "power", "final-fit",
                      "calibration", "cost", "ablation", "score", "analyze"):
            print(json.dumps({"phase_started": phase, "at_utc": stamp()}), flush=True)
            result = actions[phase](context, args)
    else:
        result = actions[args.phase](context, args)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
