"""Registered new GRU/LSTM/MLP execution on the known historical E1 calendar.

No phase implicitly downloads or scores reserve data. Publish the design/source
before register and all new fits; publish pilot-derived planning before final
fits. All learned artifacts and daily vectors stay outside source checkouts.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = "e1_core_mlp_extension_20261107"
DESIGN = f"experiments/{STUDY}/design.json"
RUNNER = "scripts/run_e1_core_mlp_extension.py"
MODELS = ("GRU", "LSTM", "MLP")
PINNED = {"torch": "2.15.0+cpu", "numpy": "2.5.4", "pandas": "3.1.0",
          "scipy": "1.18.1", "statsmodels": "0.15.0", "matplotlib": "3.11.2"}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode(value):
    import run_gru_lstm_tcn_planned as old
    return (json.dumps(old.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def write(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb" if exclusive else "wb") as stream:
        stream.write(encode(value))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def safe_private(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT):
        raise ValueError("Private outputs must remain outside the source checkout")
    if any((parent / ".git").exists() and parent not in {Path("/workspace"), Path("/tmp")}
           for parent in (path, *path.parents)):
        raise ValueError("Private outputs cannot be placed inside another Git checkout")
    return path


def runtime():
    import torch
    return {"at_utc": stamp(), "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: importlib.metadata.version(name) for name in PINNED},
        "torch_config": torch.__config__.show(), "device": "cpu",
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "environment": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")},
        "actual_threads": torch.get_num_threads(), "actual_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "float32_matmul_precision": torch.get_float32_matmul_precision()}


def pins():
    value = runtime()
    if value["python"] != "3.12.14" or value["versions"] != PINNED:
        raise ValueError("The exact preregistered Python and six package versions are required")
    return value


def grid(widths):
    return [{"candidate_id": f"c{index:02d}", "width": width, "learning_rate": rate, "dropout": dropout}
            for index, (width, rate, dropout) in enumerate(itertools.product(widths, (.001, .003), (0., .1)))]


def prepare():
    """Write a prospective source/design freeze; no model or optimizer exists."""
    original = read(ROOT / "experiments/gru_lstm_tcn_planned_20261105/design.json")
    plan = json.loads(json.dumps(original))
    plan.update(study_id=STUDY, created_at_utc=stamp(), models=list(MODELS),
        scope="New coherent exploratory historical extension, registered before its new fits; September–October2026 was already evaluated historically and is not a new confirmatory holdout",
        historical_results_unchanged=True, historical_weights_unchanged=True,
        reuse="Import immutable GRU/LSTM/SmallMLP classes and fit routines only; rerun all three families' HPO/pilot/final from scratch; do not mix earlier metrics or predictions",
        complementary_model="TCN: retained in the separate original study; not rerun or added to the new primary family")
    plan["search"]["candidates"] = {"GRU": grid((16, 32)), "LSTM": grid((16, 32)), "MLP": grid((32, 128))}
    plan["search"]["tie_break"] = "Within the1e-12 score tolerance: lower parameter count, candidate_id, then family name for the across-family baseline"
    plan["search"]["baseline_selection"] = "Best selected-family CV score; freeze before the new pilot; never reselect using the known historical reserve"
    plan["seed_identifier_reuse"] = "Reuse the registered numeric seed IDs for reproducibility, but execute all three families anew; never pool these trials with the old study as additional independent benchmark samples"
    plan["definitive"]["minimum_runs"] = 465
    plan["definitive"]["R_rule"] = "max(465, full pilot-derived required R across the three core pairs); no hidden cap; hard failure if required R exceeds10000"
    stats = plan["statistics"]
    stats["primary_pairs"] = [["GRU", "LSTM"], ["GRU", "MLP"], ["LSTM", "MLP"]]
    stats["bootstrap_seed"], stats["temporal_bootstrap_seed"] = 20261107, 20261108
    stats["temporal_secondary"]["pairs"] = stats["primary_pairs"] + [[name, "Persistence"] for name in MODELS]
    plan["reserve"].update(
        prestudy_access="This calendar has already been scored in the original project study; development, prior scores and literature were known before this extension",
        allocation="Reuse of a known historical evaluation calendar for a new exploratory model extension",
        interpretation="Exploratory historical comparison; neither a new confirmatory holdout nor a live future forecast",
        read_policy="Only an explicitly invoked score phase after all new weights/scalers and the published dimensional plan are verified; exactly the32 originally hashed archives including August context")
    plan["chronology"]["public_pilot_derived_plan_before_final_fits_required"] = True
    plan["mlp_architecture_justification"] = {
        "class": "Immutable dl_models.SmallMLP: ordered flatten(7*5), one Dense hidden layer, ReLU, dropout, Dense1, Softplus",
        "widths": [32, 128], "parameters_by_width": {"32": 1185, "128": 4737},
        "capacity_rationale": "Bracket the GRU/LSTM search parameter scale with the same eight candidate/two seed/ten month effort; parameter counts need not be equal",
        "crypto_anchors": "Garcia-Medina/Aguayo-Moreno shallow MLP and Dudek feedforward crypto comparator; their target/input/activation/selected-width differences remain explicit",
        "relu_anchor": "Zhang et al., Volatility Forecasting with Machine Learning and Intraday Commonality, AppendixB one/two/three-hidden-layer ReLU feedforward variants; equity evidence is transfer evidence",
        "project_adaptations": "D7 OHLCV relative windows, widths32/128, common optimizer/early stopping, positive Softplus head and positive multiplicative target scale; not an exact paper replication",
        "evidence": "references/literature_e1/free_model_expansion_20261107/readings.json"}
    plan["execution"] = {"workers": 3, "start_method": "spawn", "python": "3.12.14", "versions": PINNED,
        "environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "0"},
        "no_fits_at_preparation": True, "failure_policy": "Stop on technical failure; preserve all output, do not substitute seeds or silently change the design"}
    old = read(ROOT / "results/gru_lstm_tcn_planned_20261105/source_manifest.json")
    sources = {**old["registered_scientific_source_sha256"], **old["canonical_public_records_sha256"]}
    for name, expected in sources.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Historical frozen source changed: " + name)
    names = (RUNNER, "scripts/e1_core_models_mlp.py", "scripts/e1_core_mlp_statistics.py",
        "tests/test_e1_core_mlp_extension.py", "tests/test_e1_core_mlp_statistics.py",
        "scripts/export_e1_core_mlp_extension.py", "tests/test_e1_core_mlp_export.py",
        f"experiments/{STUDY}/README.md", "requirements-e1-curves.lock.txt",
        "data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json",
        plan["mlp_architecture_justification"]["evidence"])
    sources.update({name: digest(ROOT / name) for name in names})
    plan["sources_sha256"], plan["runtime_observed_at_preparation"] = sources, pins()
    plan["new_fits_at_preparation"] = 0
    write(ROOT / DESIGN, plan)
    return {"design": DESIGN, "new_fits": 0, "planned_hpo_fit_stages": 960, "planned_pilot_fit_stages": 60}


def design():
    plan = read(ROOT / DESIGN)
    if (plan["models"] != list(MODELS) or plan["history_days"] != 7
            or plan["target_returns_days"] != 7 or plan["forecast_days"] != 1):
        raise ValueError("The exact registered core and D7/n7/h1 are required")
    expected = {"GRU": grid((16, 32)), "LSTM": grid((16, 32)), "MLP": grid((32, 128))}
    if plan["search"]["candidates"] != expected or plan["search"]["seeds"] != [6101, 6102]:
        raise ValueError("The registered equal eight-candidate/two-seed grid changed")
    if plan["pilot"]["seeds"] != list(range(7101, 7111)) or plan["definitive"]["minimum_runs"] != 465:
        raise ValueError("Pilot seeds and the definitive floor must remain registered")
    if (plan["search"]["months"] != [f"2024-{month:02d}" for month in range(1, 11)]
            or plan["search"]["expected_new_fits"] != 960
            or plan["pilot"]["origin"] != "2024-11-01"
            or plan["pilot"]["evaluation_end_inclusive"] != "2024-12-30"):
        raise ValueError("The registered fixed HPO and pilot calendars are required")
    for name, expected_hash in plan["sources_sha256"].items():
        if digest(ROOT / name) != expected_hash:
            raise ValueError("Registered source/input changed: " + name)
    pins()
    return plan


def publication(path, revision, *, file_hash=None):
    receipt = read(path)
    if receipt.get("sha", receipt.get("source_commit")) != revision:
        raise ValueError("Remote publication receipt has the wrong commit")
    observed = datetime.fromisoformat(receipt["observed_at_utc"].replace("Z", "+00:00"))
    if observed.tzinfo is None or observed > datetime.now(timezone.utc):
        raise ValueError("A dated prior remote observation is required")
    if file_hash is not None and receipt.get("file_sha256") != file_hash:
        raise ValueError("The remote plan receipt must verify the exact published file SHA256")
    return receipt


def register(output, revision, receipt_path):
    plan = design()
    if revision != git("rev-parse", "HEAD") or git("status", "--porcelain"):
        raise ValueError("Register only from the clean, already-published exact source commit")
    for name in (DESIGN, *plan["sources_sha256"]):
        if subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError("All registered sources must be committed unchanged")
    evidence = publication(receipt_path, revision)
    output = safe_private(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("New registration requires an empty private output root")
    from planned_study_data import reserve_metadata
    reservation = reserve_metadata(ROOT)  # Metadata only: never reads archive payloads.
    record = {"study_id": STUDY, "registered_at_utc": stamp(), "source_commit": revision,
        "design_sha256": digest(ROOT / DESIGN), "design": plan, "sources_sha256": plan["sources_sha256"],
        "publication_verification": evidence, "runtime": runtime(), "workers": 3,
        "settings": {"training": plan["training"]}, "reservation": reservation,
        "new_fits_before_registration": 0, "reserve_price_payloads_read": 0,
        "scope": plan["scope"]}
    record["registration_signature"] = hashlib.sha256(encode(record)).hexdigest()
    write(output / "registration.json", record)
    return {"registered_at_utc": record["registered_at_utc"], "new_fits": 0, "source_commit": revision}


def registered(output):
    output = safe_private(output)
    record = read(output / "registration.json")
    unsigned = {key: value for key, value in record.items() if key != "registration_signature"}
    if hashlib.sha256(encode(unsigned)).hexdigest() != record["registration_signature"]:
        raise ValueError("Registration signature changed")
    if record["design_sha256"] != digest(ROOT / DESIGN) or record["design"] != design():
        raise ValueError("The registered design changed")
    if record["source_commit"] != git("rev-parse", "HEAD") or git("status", "--porcelain"):
        raise ValueError("Execute from the clean registered immutable source checkout")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("Launch the runner with PYTHONHASHSEED=0")
    return output, record


def artifacts(directory, output):
    return {path.relative_to(output).as_posix(): digest(path) for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def verify_artifacts(output, mapping):
    import run_gru_lstm_tcn_planned as old
    return old.verify_mapping(output, mapping)


def tagged(job, record):
    return {**job, "study_id": STUDY, "execution_id": record["registration_signature"]}


def worker_init(root, output, record, phase):
    import torch
    import run_gru_lstm_tcn_planned as old
    import e1_core_models_mlp as models
    torch.backends.mkldnn.enabled = True
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    old.worker_init(root, output, record)
    old._WORKER["models"] = models
    old._WORKER["fitting"].build_model = models.build_model
    observed = runtime()
    if observed["actual_threads"] != 1 or observed["actual_interop_threads"] != 1 or not observed["deterministic_algorithms"]:
        raise ValueError("The worker numerical controls differ from the design")
    write(Path(output) / "worker_runtime" / f"{phase}_{os.getpid()}.json",
          {"phase": phase, "pid": os.getpid(), "start_method": mp.get_start_method(), "runtime": observed})


def tag_predictions(directory, record):
    """Add provenance to private outputs without parsing or changing decimals."""
    path = Path(directory) / "predictions.csv"
    if not path.is_file():
        return
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        names, rows = list(reader.fieldnames), list(reader)
    for key, value in {"study_id": STUDY, "execution_id": record["registration_signature"]}.items():
        if key in names and any(row[key] != value for row in rows):
            raise ValueError("Mixed provenance in a private prediction file")
        if key not in names:
            names.append(key)
        for row in rows:
            row[key] = value
    temporary = path.with_suffix(".tagging.tmp")
    with temporary.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    completion = read(Path(directory) / "completion.json")
    completion["artifact_sha256"]["predictions.csv"] = digest(path)
    completion["provenance_tagging_complete"] = True
    write(Path(directory) / "completion.json", completion, exclusive=False)


def hpo_job(job):
    import run_gru_lstm_tcn_planned as old
    result = old.hpo_job(job)
    state = old._WORKER
    directory = state["out"] / "hpo" / "runs" / f"{job['model']}_{job['candidate_id']}_s{job['trajectory_seed']}"
    for month in range(1, 11):
        tag_predictions(directory / f"month_{month:02d}", state["registration"])
    return result


def fixed_job(job):
    import run_gru_lstm_tcn_planned as old
    result = old.fixed_job(job)
    state = old._WORKER
    tag_predictions(state["out"] / job["phase"] / f"{job['model']}_s{job['trajectory_seed']}", state["registration"])
    return result


def execute(output, record, jobs, function, phase):
    if (output / "score" / "raw_access_started.json").exists():
        raise ValueError("All fitting is closed after reserve payload access starts")
    marker = output / f"{phase}_started.json"
    if not marker.exists():
        write(marker, {"started_at_utc": stamp(), "registration_signature": record["registration_signature"],
            "phase": phase, "planned_trajectories": len(jobs),
            "planned_fit_stages": len(jobs) * (20 if phase == "hpo" else 2)})
    rows, start = [], time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("spawn"),
        initializer=worker_init, initargs=(str(ROOT), str(output), record, phase))
    pending = []
    try:
        pending = [pool.submit(function, job) for job in jobs]
        for future in as_completed(pending):
            rows.append(future.result())
            print(json.dumps({"phase": phase, "completed": len(rows), "total": len(jobs),
                              "wall_seconds": round(time.perf_counter() - start, 2)}), flush=True)
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return rows


def selected(output, record):
    selection = read(output / "hpo" / "selection.json")
    if selection["registration_signature"] != record["registration_signature"]:
        raise ValueError("Selected recipes belong to a different execution")
    verify_artifacts(output, selection["artifact_sha256"])
    return selection


def select_candidates(rows, candidates):
    frame = pd.DataFrame(rows)
    if (not {"study_id", "execution_id"} <= set(frame)
            or set(frame.study_id) != {STUDY} or frame.execution_id.isna().any()
            or frame.execution_id.nunique() != 1):
        raise ValueError("All HPO rows must come from the same new execution")
    if set(frame.model) != set(MODELS) or frame.duplicated(["model", "candidate_id", "trajectory_seed"]).any():
        raise ValueError("The complete unique three-family tuning panel is required")
    summaries, recipes = [], {}
    for candidate in candidates:
        group = frame[(frame.model == candidate["model"]) & (frame.candidate_id == candidate["candidate_id"])]
        if len(group) != 2 or set(group.trajectory_seed) != {6101, 6102} or not np.isfinite(group.rmse).all():
            raise ValueError("Each candidate requires both registered finite seed trajectories")
        summaries.append({**candidate, "mean_rmse": float(group.rmse.mean()),
            "sample_sd": float(group.rmse.std(ddof=1)), "parameters": int(group.parameters.iloc[0])})
    if len(frame) != 48:
        raise ValueError("The core search must complete all48 trajectories")
    for family in MODELS:
        family_rows = [row for row in summaries if row["model"] == family]
        minimum = min(row["mean_rmse"] for row in family_rows)
        recipes[family] = min((row for row in family_rows if row["mean_rmse"] <= minimum + 1e-12),
                             key=lambda row: (row["parameters"], row["candidate_id"]))
    minimum = min(row["mean_rmse"] for row in recipes.values())
    baseline = min((row for row in recipes.values() if row["mean_rmse"] <= minimum + 1e-12),
                   key=lambda row: (row["parameters"], row["candidate_id"], row["model"]))["model"]
    return recipes, baseline, summaries


def hpo(output, record):
    if (output / "hpo" / "selection.json").exists():
        return selected(output, record)
    plan = record["design"]
    candidates = [{**candidate, "model": family, "history_days": 7}
                  for family in MODELS for candidate in plan["search"]["candidates"][family]]
    jobs = [tagged({**candidate, "phase": "hpo", "trajectory_seed": seed, "seed": seed}, record)
            for candidate in candidates for seed in plan["search"]["seeds"]]
    rows = execute(output, record, jobs, hpo_job, "hpo")
    recipes, baseline, summaries = select_candidates(rows, candidates)
    pd.DataFrame(rows).sort_values(["model", "candidate_id", "trajectory_seed"]).to_csv(output / "hpo" / "run_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "hpo" / "candidate_summary.csv", index=False)
    result = {"selected_at_utc": stamp(), "study_id": STUDY, "registration_signature": record["registration_signature"],
        "recipes": recipes, "baseline_model": baseline, "trajectories": 48, "fit_stages": 960,
        "reserve_price_payloads_read": 0, "artifact_sha256": artifacts(output / "hpo", output)}
    write(output / "hpo" / "selection.json", result)
    return result


def pilot_complete(output, record):
    complete = read(output / "pilot" / "complete.json")
    if (complete["registration_signature"] != record["registration_signature"]
            or complete["selection_sha256"] != digest(output / "hpo" / "selection.json")
            or complete["trajectories"] != 30):
        raise ValueError("Pilot completion is inconsistent with the frozen new HPO")
    verify_artifacts(output, complete["artifact_sha256"])
    return complete


def pilot(output, record):
    recipes = selected(output, record)["recipes"]
    if (output / "pilot" / "complete.json").exists():
        return pilot_complete(output, record)
    jobs = [tagged({**recipes[family], "phase": "pilot", "trajectory_seed": seed, "seed": seed}, record)
            for family in MODELS for seed in record["design"]["pilot"]["seeds"]]
    rows = execute(output, record, jobs, fixed_job, "pilot")
    frame = pd.DataFrame(rows).sort_values(["model", "seed"])
    frame["fit_origin_day_utc"] = "2024-11-01"
    frame["evaluation_start"], frame["evaluation_end_inclusive"] = "2024-11-01", "2024-12-30"
    frame.to_csv(output / "pilot" / "metrics.csv", index=False)
    result = {"completed_at_utc": stamp(), "registration_signature": record["registration_signature"],
        "selection_sha256": digest(output / "hpo" / "selection.json"), "trajectories": 30, "fit_stages": 60,
        "reserve_price_payloads_read": 0, "artifact_sha256": artifacts(output / "pilot", output)}
    write(output / "pilot" / "complete.json", result)
    return result


def validate_plan(output, record):
    from e1_core_mlp_statistics import plan_sample_size
    pilot_complete(output, record)
    plan = read(output / "plan.json")
    if (plan["registration_signature"] != record["registration_signature"]
            or plan["pilot_complete_sha256"] != digest(output / "pilot" / "complete.json")
            or plan["selection_sha256"] != digest(output / "hpo" / "selection.json")):
        raise ValueError("The dimensional plan is not bound to the complete new pilot")
    recalculated = plan_sample_size(pd.read_csv(output / "pilot" / "metrics.csv"), record["design"])
    for key, value in recalculated.items():
        if encode(plan[key]) != encode(value):
            raise ValueError("The dimensional plan differs from its registered calculation: " + key)
    if plan["R"] < 465 or plan["R"] > 10000 or plan["seeds"] != list(range(8101, 8101 + plan["R"])):
        raise ValueError("The declared definitive floor, bound and exact seed grid are required")
    return plan


def dimensional_plan(output, record):
    from e1_core_mlp_statistics import plan_sample_size
    selected(output, record)
    pilot_complete(output, record)
    if (output / "plan.json").exists():
        return validate_plan(output, record)
    result = plan_sample_size(pd.read_csv(output / "pilot" / "metrics.csv"), record["design"])
    result.update(study_id=STUDY, execution_id=record["registration_signature"], planned_at_utc=stamp(),
        registration_signature=record["registration_signature"],
        pilot_complete_sha256=digest(output / "pilot" / "complete.json"),
        selection_sha256=digest(output / "hpo" / "selection.json"), reserve_price_payloads_read=0)
    write(output / "plan.json", result)
    return validate_plan(output, record)


def register_plan(output, record, published_file, revision, receipt_path):
    plan = validate_plan(output, record)
    if (output / "final_started.json").exists():
        raise ValueError("The dimensional plan must be published before the first definitive fit")
    if Path(published_file).read_bytes() != (output / "plan.json").read_bytes():
        raise ValueError("The public dimensional plan must be identical to the private calculation")
    evidence = publication(receipt_path, revision, file_hash=digest(output / "plan.json"))
    result = {"registered_at_utc": stamp(), "plan_sha256": digest(output / "plan.json"),
        "published_plan_commit": revision, "publication_verification": evidence,
        "registration_signature": record["registration_signature"], "R": plan["R"],
        "new_definitive_fits_before_plan_publication": 0}
    result["receipt_signature"] = hashlib.sha256(encode(result)).hexdigest()
    write(output / "plan_publication.json", result)
    return result


def published_plan(output, record):
    plan = validate_plan(output, record)
    receipt = read(output / "plan_publication.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_signature"}
    if hashlib.sha256(encode(unsigned)).hexdigest() != receipt.get("receipt_signature"):
        raise ValueError("The published dimensional plan receipt signature changed")
    if receipt["plan_sha256"] != digest(output / "plan.json") or receipt["registration_signature"] != record["registration_signature"]:
        raise ValueError("The published dimensional plan receipt changed")
    return plan


def fit_final(output, record):
    plan = published_plan(output, record)
    if (output / "final_models_manifest.json").exists():
        return validate_final(output, record)
    recipes = selected(output, record)["recipes"]
    if not (output / "final_started.json").exists():
        write(output / "final_started.json", {"at_utc": stamp(), "planned_trajectories": 3 * plan["R"],
            "registration_signature": record["registration_signature"], "plan_sha256": digest(output / "plan.json"),
            "plan_publication_sha256": digest(output / "plan_publication.json")})
    jobs = [tagged({**recipes[family], "phase": "final", "trajectory_seed": seed, "seed": seed}, record)
            for family in MODELS for seed in plan["seeds"]]
    rows = execute(output, record, jobs, fixed_job, "final")
    pd.DataFrame(rows).sort_values(["model", "seed"]).to_csv(output / "final" / "metrics.csv", index=False)
    result = {"all_weights_frozen_at_utc": stamp(), "registration_signature": record["registration_signature"],
        "plan_sha256": digest(output / "plan.json"), "selection_sha256": digest(output / "hpo" / "selection.json"),
        "models": len(rows), "R": plan["R"], "fit_stages": 2 * len(rows), "jobs": rows,
        "reserve_price_payloads_read": 0, "artifact_sha256": artifacts(output / "final", output)}
    write(output / "final_models_manifest.json", result)
    return validate_final(output, record)


def validate_final(output, record):
    plan = published_plan(output, record)
    manifest = read(output / "final_models_manifest.json")
    if (manifest["registration_signature"] != record["registration_signature"]
            or manifest["plan_sha256"] != digest(output / "plan.json")
            or manifest["selection_sha256"] != digest(output / "hpo" / "selection.json")):
        raise ValueError("All final artifacts must belong to the closed new execution")
    expected = {(family, seed) for family in MODELS for seed in plan["seeds"]}
    actual = {(job["model"], job["seed"]) for job in manifest["jobs"]}
    if expected != actual or manifest["models"] != len(expected) or len(manifest["jobs"]) != len(expected):
        raise ValueError("Every registered family and definitive seed must be frozen before reserve access")
    verify_artifacts(output, manifest["artifact_sha256"])
    return manifest


def score(output, record, raw_root):
    manifest = validate_final(output, record)  # Before any reserved archive bytes.
    complete_path = output / "score" / "complete.json"
    if complete_path.exists():
        complete = read(complete_path)
        verify_artifacts(output, complete["artifact_sha256"])
        return complete
    import torch
    import e1_core_models_mlp as models
    from planned_study_data import load_reserved_data, reserve_metadata
    from final_fit_filipe_multihistory import apply_frozen_test_preprocessing
    import run_gru_lstm_tcn_planned as old
    from train_dl_benchmark import metrics
    # Metadata/stat only first. The original loader verifies all32 original SHA256
    # payloads and the full60-date cohort before a single model is evaluated.
    reserve_metadata(ROOT, raw_root)
    torch.backends.mkldnn.enabled = True
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    score_dir = output / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    if not (score_dir / "runtime.json").exists():
        write(score_dir / "runtime.json", runtime())
    gate = score_dir / "frozen_access_gate.json"
    if not gate.exists():
        write(gate, {"checked_at_utc": stamp(), "all_final_models_verified": len(manifest["jobs"]),
            "final_models_manifest_sha256": digest(output / "final_models_manifest.json"),
            "plan_sha256": digest(output / "plan.json"), "known_historical_exploratory_calendar": True})
    data, audit = load_reserved_data(ROOT, raw_root, record["reservation"], score_dir)
    indices = np.asarray(data["test_indices"], dtype=int)
    dates = data["calendar"][indices].strftime("%Y-%m-%d").tolist()
    if dates != pd.date_range("2026-09-01", "2026-10-30", tz="UTC").strftime("%Y-%m-%d").tolist():
        raise ValueError("The exact known60-date historical calendar is required")
    frames, rows = [], []
    for job in sorted(manifest["jobs"], key=lambda row: (row["model"], row["seed"])):
        name = f"{job['model']}_s{job['seed']}"
        directory, model_dir = score_dir / "runs" / name, output / "final" / name
        existing = old.cached_completion(directory, record["registration_signature"])
        if existing is None:
            old.prepare_directory(directory)
            checkpoint = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
            pre = read(model_dir / "preprocessing.json")["refit"]
            model = models.build_model(job["model"], 7, job["width"], job["dropout"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            x = apply_frozen_test_preprocessing(data, 7, indices, pre)
            scaled, seconds = old.infer_once(model, x)
            predictions = scaled.astype(float) * pre["y_scale"]
            frame = pd.DataFrame({"model": job["model"], "seed": job["seed"], "outcome_day": dates,
                "y_true": data["targets"][indices], "y_pred": predictions,
                "persistence_pred": data["persistence"][indices], "mase_scale": job["mase_scale"],
                "study_id": STUDY, "execution_id": record["registration_signature"]})
            frame.to_csv(directory / "predictions.csv", index=False)
            existing = tagged({"model": job["model"], "seed": job["seed"], "inference_seconds": seconds,
                "n_dates": len(indices), "model_sha256": digest(model_dir / "model.pt"),
                **metrics(frame.y_true, frame.y_pred, job["mase_scale"])}, record)
            write(directory / "completion.json", {"completed_at_utc": stamp(),
                "registration_signature": record["registration_signature"], "record": existing,
                "artifact_sha256": {"predictions.csv": digest(directory / "predictions.csv")}})
        frames.append(pd.read_csv(directory / "predictions.csv"))
        rows.append(existing)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(score_dir / "predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(score_dir / "run_metrics.csv", index=False)
    write(score_dir / "data_audit.json", audit)
    result = {"completed_at_utc": stamp(), "registration_signature": record["registration_signature"],
        "final_models_manifest_sha256": digest(output / "final_models_manifest.json"),
        "trajectories": len(rows), "dates": 60, "prediction_rows": len(predictions),
        "scoring_passes_per_checkpoint": 1, "historical_exploratory_not_confirmatory": True,
        "artifact_sha256": artifacts(score_dir, output)}
    write(complete_path, result)
    return result


def analyze(output, record):
    from e1_core_mlp_statistics import analyze_predictions
    complete = read(output / "score" / "complete.json")
    verify_artifacts(output, complete["artifact_sha256"])
    plan = validate_plan(output, record)
    selected_recipe = selected(output, record)
    analysis_design = {**record["design"], "baseline_model": selected_recipe["baseline_model"]}
    result = analyze_predictions(pd.read_csv(output / "score" / "predictions.csv"), analysis_design, plan)
    destination = output / "analysis"
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Preserve existing analysis; do not silently overwrite it")
    destination.mkdir(parents=True, exist_ok=True)
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(destination / f"{key}.csv", index=False)
        else:
            write(destination / f"{key}.json", value)
    manifest = {"analyzed_at_utc": stamp(), "registration_signature": record["registration_signature"],
        "score_complete_sha256": digest(output / "score" / "complete.json"),
        "artifact_sha256": artifacts(destination, output)}
    write(destination / "manifest.json", manifest)
    return {"analysis": str(destination), "files": len(result)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "register", "hpo", "pilot", "plan", "register-plan", "fit", "freeze", "score", "analyze"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--published-commit")
    parser.add_argument("--publication-receipt", type=Path)
    parser.add_argument("--published-plan-file", type=Path)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare()
    elif args.phase == "register":
        if not args.output_dir or not args.published_commit or not args.publication_receipt:
            parser.error("register requires --output-dir --published-commit --publication-receipt")
        result = register(args.output_dir, args.published_commit, args.publication_receipt)
    else:
        if not args.output_dir:
            parser.error("--output-dir is required")
        output, record = registered(args.output_dir)
        if args.phase == "register-plan":
            if not args.published_plan_file or not args.published_commit or not args.publication_receipt:
                parser.error("register-plan requires --published-plan-file --published-commit --publication-receipt")
            result = register_plan(output, record, args.published_plan_file, args.published_commit, args.publication_receipt)
        elif args.phase == "score":
            if not args.raw_root:
                parser.error("score requires the explicitly provided --raw-root with all32 original archives")
            result = score(output, record, args.raw_root)
        else:
            handlers = {"hpo": hpo, "pilot": pilot, "plan": dimensional_plan,
                        "fit": fit_final, "freeze": validate_final, "analyze": analyze}
            result = handlers[args.phase](output, record)
    # Never print full plans or private learned values in terminal output.
    if isinstance(result, dict) and "artifact_sha256" in result:
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {"jobs", "recipes", "artifact_sha256"}}))
    else:
        print(encode(result).decode())


if __name__ == "__main__":
    main()
