"""Portable final-fit and score stages for the public LSTM/GRU/TCN adapter.

This is a publication adapter written after the recorded experiment. It creates
new reproduction outputs and never claims its own source hash was used by the
archived fits. The fixed historical test is already observed and exploratory.
No stage downloads data, updates a book, publishes, or modifies training code.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePath
import sys
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

HISTORIES = (7, 14, 21, 28)
FAMILIES = ("LSTM", "GRU", "TCN")
TEST_MONTHS = tuple(pd.date_range("2025-01-01", "2026-08-01", freq="MS").strftime("%Y-%m"))
TEST_DATES = tuple(pd.date_range("2025-01-01", "2026-08-31").strftime("%Y-%m-%d"))
RESERVE_RELATIVE = "data/manifests/test_raw_reserve_20250101_20260831.json"
_WORKER = None


def _api():
    return importlib.import_module("run_gru_lstm_tcn")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe(base, name):
    base, relative = Path(base).resolve(), Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact names must be relative to their declared root")
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError("Artifact path escapes its declared root")
    return candidate


def _relative(base, path):
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("This source must remain inside --project-root for portable restoration") from exc


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(_clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _read(path):
    return json.loads(Path(path).read_text())


def _verify(mapping, base):
    for name, expected in mapping.items():
        if _digest(_safe(base, name)) != expected:
            raise ValueError("A signed artifact changed: " + name)


def run_name(job):
    return f"D{int(job['history_days'])}_{job['model']}_s{int(job['seed'])}"


def check_jobs(jobs, plan):
    if tuple(plan["histories"]) != HISTORIES or set(plan["models"]) != set(FAMILIES):
        raise ValueError("The public benchmark requires four histories and LSTM/GRU/TCN")
    expected = {(d, m, int(s)) for d in HISTORIES for m in FAMILIES for s in plan["seeds"]}
    observed = {(int(j["history_days"]), j["model"], int(j["seed"])) for j in jobs}
    if observed != expected or len(jobs) != len(expected):
        raise ValueError("Every registered model/history/seed job is required exactly once")
    for job in jobs:
        epoch = job["epochs"]
        recipe = plan["models"][job["model"]]
        if (isinstance(epoch, bool) or int(epoch) != epoch
                or not 1 <= epoch <= plan["training"]["max_epochs"]
                or job["width"] != recipe["width"]
                or job["learning_rate"] != recipe["learning_rate"]):
            raise ValueError("A final job differs from its closed training recipe")


def choose_setting(rows, tolerance=1e-12):
    rows = list(rows)
    if not rows or any(not np.isfinite(row["cv_rmse_mean"]) for row in rows):
        raise ValueError("Only finite CV scores can select a history")
    minimum = min(row["cv_rmse_mean"] for row in rows)
    ties = [row for row in rows if abs(row["cv_rmse_mean"] - minimum) <= tolerance]
    return min(ties, key=lambda row: (row["history_days"], row["parameters"], row["model"]))


def validate_reservation(reserve):
    records = reserve.get("archives", [])
    labels = [str(record["archive"]).removeprefix("BTCUSDT-1m-").removesuffix(".zip") for record in records]
    if labels != list(TEST_MONTHS) or reserve.get("calendar_days") != 608:
        raise ValueError("Only the twenty January2025--August2026 archives are admitted")
    for record in records:
        relative = PurePath(record["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Archive names must remain relative to the supplied raw root")
        expected = record.get("sha256", "")
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("Every archived month needs its recorded SHA256")
    return records


def declared_test_source(root, *, test_daily_csv=None, test_audit_json=None):
    """Read metadata only; this deliberately never reads the daily CSV or ZIPs."""
    root = Path(root).resolve()
    if bool(test_daily_csv) != bool(test_audit_json):
        raise ValueError("Cached mode requires both --test-daily-csv and --test-audit-json")
    if test_daily_csv:
        daily = Path(test_daily_csv)
        audit = Path(test_audit_json)
        daily = (root / daily).resolve() if not daily.is_absolute() else daily.resolve()
        audit = (root / audit).resolve() if not audit.is_absolute() else audit.resolve()
        # Pin a declared hash, not a payload hash. The payload is read after weights close.
        metadata = _read(audit)
        expected = metadata.get("daily_market_sha256", "")
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("Cached audit metadata must declare daily_market_sha256")
        quality = metadata.get("quality_by_month")
        if quality is not None and [row["month"] for row in quality] != list(TEST_MONTHS):
            raise ValueError("Cached metadata includes months outside the fixed test period")
        return {"mode": "cached_daily", "daily_csv": _relative(root, daily),
            "audit_json": _relative(root, audit), "audit_sha256": _digest(audit),
            "expected_daily_sha256": expected,
            "calendar_start": TEST_DATES[0], "calendar_end": TEST_DATES[-1]}
    reserve_path = _safe(root, RESERVE_RELATIVE)
    validate_reservation(_read(reserve_path))
    return {"mode": "reserved_archives", "reservation_json": RESERVE_RELATIVE,
        "reservation_sha256": _digest(reserve_path),
        "calendar_start": TEST_DATES[0], "calendar_end": TEST_DATES[-1],
        "raw_root_role": "Explicit score-time location of pre-downloaded archives; no downloader"}


def close_configuration(root, out, *, test_daily_csv=None, test_audit_json=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    final = out / "final_evaluation"
    if (final / "freeze.json").exists() or (final / "protocol.json").exists():
        raise FileExistsError("A final configuration closure must never be overwritten")
    plan, context, _ = _api().load_registered(root, out, require_cv_complete=True)
    runs = pd.read_csv(out / "results/runs.csv")
    predictions = pd.read_csv(out / "results/predictions.csv")
    expected_dates = pd.date_range("2024-01-01", "2024-12-31").strftime("%Y-%m-%d").tolist()
    jobs, scores, epochs_hashes = [], [], {}
    for history in HISTORIES:
        for family in FAMILIES:
            rmses = []
            for seed in plan["seeds"]:
                matching = runs.loc[runs.history_days.eq(history) & runs.model.eq(family) & runs.seed.eq(seed)]
                frame = predictions.loc[predictions.history_days.eq(history) & predictions.model.eq(family)
                                        & predictions.seed.eq(seed)].sort_values("outcome_day")
                if (len(matching) != 1 or frame.outcome_day.tolist() != expected_dates
                        or frame.run_id.nunique() != 1 or not np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all()):
                    raise ValueError("A final job lacks its complete common-CV trajectory")
                row = matching.iloc[0]
                rmse = float(np.sqrt(np.mean((frame.y_true.to_numpy() - frame.y_pred.to_numpy()) ** 2)))
                if not np.isclose(rmse, row.rmse, rtol=1e-10, atol=1e-14):
                    raise ValueError("CV score does not match its stored predictions")
                rmses.append(rmse)
                run_id = str(row.run_id)
                if Path(run_id).name != run_id or run_id != str(frame.run_id.iloc[0]):
                    raise ValueError("Unsafe or mismatched CV run identity")
                configuration = _safe(out, f"results/runs/{run_id}/2024-12/configuration.json")
                config = _read(configuration)
                recipe = plan["models"][family]
                if (config["model"] != family or config["history_days"] != history
                        or config["trajectory_seed"] != seed or config["width"] != recipe["width"]
                        or config["learning_rate"] != recipe["learning_rate"]):
                    raise ValueError("December epoch source differs from the registered recipe")
                relative = configuration.relative_to(out).as_posix()
                epochs_hashes[relative] = _digest(configuration)
                jobs.append({"model": family, "history_days": history, "seed": int(seed),
                    "width": recipe["width"], "learning_rate": recipe["learning_rate"],
                    "epochs": config["best_epoch"], "parameters": int(row.parameters),
                    "epoch_source_configuration": relative})
            scores.append({"model": family, "history_days": history, "parameters": int(row.parameters),
                "cv_rmse_mean": float(np.mean(rmses)), "cv_rmse_sample_sd": float(np.std(rmses, ddof=1))})
    check_jobs(jobs, plan)
    selected = {family: choose_setting(row for row in scores if row["model"] == family) for family in FAMILIES}
    test_source = declared_test_source(root, test_daily_csv=test_daily_csv, test_audit_json=test_audit_json)
    final_plan = {**plan, "phase": "final_development_refit_then_observed_test_exploration",
        "registered_final_at_utc": _now(), "test_source": test_source,
        "selected_by_cv": selected, "all_settings_reported": True,
        "test_already_observed": True, "test_selects_configuration": False,
        "publication_adapter_after_archived_fits": True,
        "seed_budget_limitation": "Ten fixed seeds; no prospective power-sized pilot and no fresh HPO",
        "final_epoch_rule": "Per-job December2024 inner-CV best epoch, fixed before final fits",
        "test_start": TEST_DATES[0], "test_end_inclusive": TEST_DATES[-1],
        "september_october2026_assigned": False, "automatic_downloads": False}
    _write(final / "protocol.json", final_plan, exclusive=True)
    code = dict(plan["code_sha256"])
    code[_relative(root, Path(__file__))] = _digest(__file__)
    for name in ("scripts/final_fit_filipe_multihistory.py", "scripts/load_filipe_reserved_test.py"):
        code[name] = _digest(_safe(root, name))
    evidence = {name: _digest(out / name) for name in (
        "protocol.json", "results/runs.csv", "results/predictions.csv",
        "results/fold_metrics.csv", "results/execution_summary.json")}
    freeze = {"schema_version": 1, "closed_at_utc": _now(),
        "scope": "Exploratory reproduction on the already observed January2025--August2026 test",
        "protocol_sha256": _digest(final / "protocol.json"), "core_protocol_sha256": _digest(out / "protocol.json"),
        "code_sha256": code, "source_cv_sha256": evidence,
        "epoch_source_sha256": epochs_hashes, "final_jobs": jobs,
        "selection": {"by_model": selected, "baseline": choose_setting(scores), "all_settings": scores,
            "rule": "Minimum mean pooled CV RMSE; numerical ties smaller D, parameters, model name"},
        "test_source": test_source, "test_already_observed": True,
        "test_selection_allowed": False, "automatic_publication_allowed": False}
    _write(final / "freeze.json", freeze, exclusive=True)
    print(json.dumps({"status": "final_configuration_closed", "jobs": len(jobs),
                      "selected_by_cv": selected, "test_payloads_opened": 0}), flush=True)
    return freeze


def validate_final_configuration(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    # The portable runner checks registered source hashes and locked runtime.
    # Metadata-only validation is permitted here; no test payload is opened.
    _api().load_registered(root, out)
    final = out / "final_evaluation"
    freeze, plan = _read(final / "freeze.json"), _read(final / "protocol.json")
    if (_digest(final / "protocol.json") != freeze["protocol_sha256"]
            or _digest(out / "protocol.json") != freeze["core_protocol_sha256"]
            or freeze.get("test_already_observed") is not True
            or freeze.get("test_selection_allowed") is not False
            or freeze.get("automatic_publication_allowed") is not False
            or plan.get("test_start") != TEST_DATES[0] or plan.get("test_end_inclusive") != TEST_DATES[-1]
            or freeze["test_source"] != plan["test_source"]):
        raise ValueError("Final closure identity or exploratory test scope changed")
    _verify(freeze["code_sha256"], root)
    _verify(plan["sources"], root)
    _verify(freeze["source_cv_sha256"], out)
    _verify(freeze["epoch_source_sha256"], out)
    check_jobs(freeze["final_jobs"], plan)
    source = freeze["test_source"]
    if source["mode"] == "cached_daily":
        if _digest(_safe(root, source["audit_json"])) != source["audit_sha256"]:
            raise ValueError("Cached test audit metadata changed after closure")
    elif source["mode"] == "reserved_archives":
        if _digest(_safe(root, source["reservation_json"])) != source["reservation_sha256"]:
            raise ValueError("Archive reservation metadata changed after closure")
    else:
        raise ValueError("Unsupported declared test source")
    return freeze, plan


def validate_final_models(root, out):
    out = Path(out).resolve()
    freeze, plan = validate_final_configuration(root, out)
    final = out / "final_evaluation"
    manifest = _read(final / "final_models_manifest.json")
    if manifest["freeze_sha256"] != _digest(final / "freeze.json"):
        raise ValueError("Final checkpoints belong to another configuration closure")
    check_jobs(manifest["jobs"], plan)
    expected = {(j["history_days"], j["model"], j["seed"]) for j in freeze["final_jobs"]}
    actual = {(j["history_days"], j["model"], j["seed"]) for j in manifest["jobs"]}
    if expected != actual:
        raise ValueError("Every closed job must have a frozen final checkpoint")
    frozen_by_id = {(j["history_days"], j["model"], j["seed"]): j for j in freeze["final_jobs"]}
    for job in manifest["jobs"]:
        frozen = frozen_by_id[(job["history_days"], job["model"], job["seed"])]
        if any(job[key] != frozen[key] for key in ("epochs", "width", "learning_rate", "parameters")):
            raise ValueError("A final checkpoint record differs from the closed job")
    required_artifacts = {f"final_fit/{run_name(job)}/{name}"
        for job in freeze["final_jobs"]
        for name in ("model.pt", "preprocessing.json", "configuration.json", "completion.json")}
    if not isinstance(manifest.get("artifact_sha256"), dict) or not required_artifacts <= set(manifest["artifact_sha256"]):
        raise ValueError("The weight manifest lacks a required checkpoint/scaler/configuration/completion artifact")
    _verify(manifest["artifact_sha256"], final)
    return freeze, plan, manifest


def _initialize_fit(root, out, plan, freeze):
    global _WORKER
    root, out = Path(root), Path(out)
    fitting, _, models = _api().configure(root, out)
    final_helper = importlib.import_module("final_fit_filipe_multihistory")
    final_helper.ROOT = root
    final_helper.OUT = out / "final_evaluation"
    final_helper.architecture_spec = models.architecture_spec
    final_helper.build_model = models.build_model
    final_helper.init_worker(plan, freeze, _digest(final_helper.OUT / "freeze.json"))
    _WORKER = final_helper


def _fit_job(job):
    return _WORKER.fit_job(job)


def fit_final_models(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    final = out / "final_evaluation"
    freeze, plan = validate_final_configuration(root, out)
    if (final / "test/access_started.json").exists() or (final / "test/predictions.csv").exists():
        raise RuntimeError("Final fitting is closed after this reproduction starts reading the test")
    jobs = freeze["final_jobs"]
    records = []
    with ProcessPoolExecutor(max_workers=int(plan["workers"]), mp_context=mp.get_context("spawn"),
            initializer=_initialize_fit, initargs=(str(root), str(out), plan, freeze)) as pool:
        for future in as_completed([pool.submit(_fit_job, job) for job in jobs]):
            row = future.result()
            records.append(row)
            print(json.dumps({"final_models_completed": len(records), "total": len(jobs),
                              "run_id": row["run_id"]}), flush=True)
    records.sort(key=lambda row: (row["history_days"], row["model"], row["seed"]))
    pd.DataFrame(records).to_csv(final / "final_fit_summary.csv", index=False)
    artifact_map = {path.relative_to(final).as_posix(): _digest(path)
                    for path in (final / "final_fit").rglob("*") if path.is_file()}
    manifest = {"completed_at_utc": _now(), "freeze_sha256": _digest(final / "freeze.json"),
        "jobs": records, "artifact_sha256": artifact_map, "completed_project_fits": len(records),
        "test_payloads_opened": 0,
        "scope": "New portable reproduction fits; all configurations and weights frozen before test access"}
    _write(final / "final_models_manifest.json", manifest)
    validate_final_models(root, out)
    return manifest


def _test_data(root, out, *, raw_root=None):
    """The first payload read is below the complete configuration/weight gate."""
    root, out = Path(root).resolve(), Path(out).resolve()
    freeze, _, manifest = validate_final_models(root, out)
    final, source = out / "final_evaluation", freeze["test_source"]
    test = final / "test"
    daily_path, audit_path = test / "daily_market_test.csv", test / "test_source_audit.json"
    test.mkdir(parents=True, exist_ok=True)
    if daily_path.exists() != audit_path.exists():
        raise RuntimeError("Partial extraction must be retained and audited before repeating payload reads")
    if daily_path.exists():
        audit = _read(audit_path)
        if (audit["freeze_sha256"] != _digest(final / "freeze.json")
                or _digest(daily_path) != audit["daily_market_sha256"]
                or (source["mode"] == "cached_daily"
                    and audit["daily_market_sha256"] != source["expected_daily_sha256"])):
            raise ValueError("Saved test derivative changed")
        daily = pd.read_csv(daily_path, index_col="market_day_utc")
        daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index, utc=True))
    else:
        if (test / "access_started.json").exists():
            raise RuntimeError("An interrupted extraction must be audited; do not silently reopen payloads")
        if source["mode"] == "reserved_archives" and raw_root is None:
            raise ValueError("Supply --raw-root with the already downloaded reserved archives; no automatic download occurs")
        _write(test / "access_started.json", {"at_utc": _now(),
            "freeze_sha256": _digest(final / "freeze.json"),
            "final_models_manifest_sha256": _digest(final / "final_models_manifest.json"),
            "source_mode": source["mode"], "test_already_observed": True}, exclusive=True)
        loader = importlib.import_module("load_filipe_reserved_test")
        qualities, archives_opened = [], 0
        if source["mode"] == "cached_daily":
            payload = _safe(root, source["daily_csv"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != source["expected_daily_sha256"]:
                raise ValueError("Cached test payload differs from the metadata identity frozen before fitting")
            daily_path.write_bytes(payload)
            daily = pd.read_csv(daily_path, index_col="market_day_utc")
            daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index, utc=True))
        else:
            records = validate_reservation(_read(_safe(root, source["reservation_json"])))
            pieces = []
            for month, record in zip(TEST_MONTHS, records):
                archive = loader.safe_archive_path(Path(raw_root), record["relative_path"])
                part, quality = loader.aggregate_month_payload(archive.read_bytes(), month,
                                                                expected_sha256=record["sha256"])
                pieces.append(part)
                qualities.append(quality)
                archives_opened += 1
            daily = pd.concat(pieces)
            daily.to_csv(daily_path)
        expected_calendar = pd.DatetimeIndex(pd.to_datetime(TEST_DATES, utc=True))
        if not daily.index.equals(expected_calendar):
            raise ValueError("Test derivative must retain exactly January2025--August2026; no later dates")
        audit = {"completed_at_utc": _now(), "freeze_sha256": _digest(final / "freeze.json"),
            "final_models_manifest_sha256": _digest(final / "final_models_manifest.json"),
            "daily_market_sha256": _digest(daily_path), "source_mode": source["mode"],
            "archives_opened": archives_opened, "quality_by_month": qualities,
            "test_already_observed": True, "september_october2026_opened": False,
            "model_refitting_after_access": False}
        _write(audit_path, audit, exclusive=True)
    development = pd.read_csv(root / "results/eda_development/daily_market_development.csv", index_col="market_day_utc")
    development.index = pd.DatetimeIndex(pd.to_datetime(development.index, utc=True))
    loader = importlib.import_module("load_filipe_reserved_test")
    data, eligibility, cohort = loader.build_combined_data(development, daily)
    if cohort["n_common_outcomes"] != 608:
        raise ValueError("This recorded-data reproduction requires the same 608-date common cohort")
    cohort.update({"freeze_sha256": _digest(final / "freeze.json"),
        "test_source_audit_sha256": _digest(audit_path), "test_already_observed": True})
    cohort_path = test / "cohort_manifest.json"
    if cohort_path.exists():
        if _read(cohort_path) != cohort:
            raise ValueError("Common test cohort changed during resumption")
    else:
        _write(cohort_path, cohort, exclusive=True)
    eligibility.to_csv(test / "eligibility.csv", index=False)
    return data, cohort, audit


def score_final_models(root, out, *, raw_root=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    _, plan, manifest = validate_final_models(root, out)
    fitting, _, models = _api().configure(root, out)
    final_helper = importlib.import_module("final_fit_filipe_multihistory")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data, cohort, audit = _test_data(root, out, raw_root=raw_root)
    final, indices = out / "final_evaluation", data["test_indices"]
    records, prediction_parts, monthly_parts = [], [], []
    for job in manifest["jobs"]:
        name, history = run_name(job), int(job["history_days"])
        source = final / "final_fit" / name
        destination = final / "test/runs" / name
        completion = destination / "completion.json"
        if completion.exists():
            saved = _read(completion)
            if (saved["freeze_sha256"] != _digest(final / "freeze.json")
                    or saved["model_sha256"] != _digest(source / "model.pt")
                    or saved["cohort_sha256"] != _digest(final / "test/cohort_manifest.json")):
                raise ValueError("A saved score belongs to a different checkpoint/cohort/freeze")
            _verify(saved["artifact_sha256"], destination)
            frame = pd.read_csv(destination / "predictions.csv")
            per_month = pd.read_csv(destination / "fold_metrics.csv")
            record = saved["record"]
        else:
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError("A partial score needs audit before repeating any checkpoint forward call")
            destination.mkdir(parents=True, exist_ok=True)
            preprocessing = _read(source / "preprocessing.json")
            checkpoint = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
            model = models.build_model(job["model"], history, job["width"], checkpoint["dropout"])
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            x = final_helper.apply_frozen_test_preprocessing(data, history, indices, preprocessing)
            started = time.perf_counter()
            with torch.inference_mode():
                scaled = model(torch.from_numpy(x)).cpu().numpy()
            seconds = time.perf_counter() - started
            predictions = scaled.astype(float) * preprocessing["y_scale"]
            dates = data["calendar"][indices]
            frame = pd.DataFrame({"variant": "relative_window", "run_id": name,
                "history_days": history, "model": job["model"], "seed": job["seed"], "horizon": 1,
                "fold_id": dates.strftime("%Y-%m"), "outcome_day": dates.strftime("%Y-%m-%d"),
                "prediction_origin_utc": [value.isoformat() for value in dates],
                "target_available_at_utc": [(value + pd.Timedelta(days=1)).isoformat() for value in dates],
                "input_first_day": data["calendar"][indices - history].strftime("%Y-%m-%d"),
                "input_last_day": data["calendar"][indices - 1].strftime("%Y-%m-%d"),
                "y_true": data["targets"][indices], "y_pred": predictions,
                "persistence_pred": data["persistence"][indices], "mase_train_scale": job["mase_train_scale"]})
            frame.to_csv(destination / "predictions.csv", index=False)
            record = {"variant": "relative_window", "run_id": name, "phase": "observed_test_exploratory",
                "history_days": history, "model": job["model"], "seed": job["seed"],
                "n_dates": len(frame), "parameters": job["parameters"], "training_seconds": job["training_seconds"],
                "inference_seconds": seconds, "checkpoint_forward_calls": 1,
                **fitting.metrics(frame.y_true, frame.y_pred, job["mase_train_scale"])}
            per_month = pd.DataFrame([{"variant": "relative_window", "history_days": history,
                "model": job["model"], "seed": job["seed"], "fold_id": month, "n_dates": len(part),
                "model_update": "none; reporting interval of frozen weights",
                **fitting.metrics(part.y_true, part.y_pred, job["mase_train_scale"])}
                for month, part in frame.groupby("fold_id", sort=True)])
            per_month.to_csv(destination / "fold_metrics.csv", index=False)
            artifacts = {path.name: _digest(path) for path in destination.iterdir() if path.is_file()}
            _write(completion, {"completed_at_utc": _now(), "freeze_sha256": _digest(final / "freeze.json"),
                "model_sha256": _digest(source / "model.pt"), "cohort_sha256": _digest(final / "test/cohort_manifest.json"),
                "record": record, "artifact_sha256": artifacts})
        records.append(record)
        prediction_parts.append(frame)
        monthly_parts.append(per_month)
        print(json.dumps({"test_runs_scored": len(records), "total": len(manifest["jobs"]), "run_id": name}), flush=True)
    pd.concat(prediction_parts, ignore_index=True).to_csv(final / "test/predictions.csv", index=False)
    pd.concat(monthly_parts, ignore_index=True).to_csv(final / "test/fold_metrics.csv", index=False)
    pd.DataFrame(records).to_csv(final / "test/runs.csv", index=False)
    summary = {"completed_at_utc": _now(), "freeze_sha256": _digest(final / "freeze.json"),
        "final_models_manifest_sha256": _digest(final / "final_models_manifest.json"),
        "model_runs": len(records), "histories": list(HISTORIES), "seeds": len(plan["seeds"]),
        "common_prediction_dates": len(indices), "predictions": sum(len(part) for part in prediction_parts),
        "calendar_days": 608, "monthly_reporting_blocks": 20, "test_period_model_refits": 0,
        "actual_checkpoint_forward_calls": sum(row["checkpoint_forward_calls"] for row in records),
        "raw_archives_opened": audit["archives_opened"], "test_used_for_configuration_selection": False,
        "test_already_observed": True,
        "scope": "New reproduction on a previously observed test; exploratory, not independent temporal confirmation"}
    _write(final / "test/execution_summary.json", summary)
    return summary


def run_final_phase(project_root, output_dir, phase, *, test_daily_csv=None,
                    test_audit_json=None, raw_root=None):
    """Explicit phase entry point; importing this module performs no data work."""
    if phase == "close":
        return close_configuration(project_root, output_dir,
            test_daily_csv=test_daily_csv, test_audit_json=test_audit_json)
    if phase == "fit":
        return fit_final_models(project_root, output_dir)
    if phase == "score":
        return score_final_models(project_root, output_dir, raw_root=raw_root)
    raise ValueError("Final workflow phase must be close, fit or score")
