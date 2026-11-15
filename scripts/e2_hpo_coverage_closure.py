"""Recover and reconcile existing reduced-HPO receipts; no fits or new trials."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "book_e2/_static/e2_corrections_20261115/hpo"
COMMIT = "6a8000022d2f374731faacc8a5c66f9b9d46b465"
CLOSURE = "runs/reduced-selection-closure-64d1024aefb76ab5239c8db7"
HEAD_PATH = "control/github-primary-v1/head.json"
MANIFEST_SHA = "555994b92de664f3b298942929640cc5ba36ee4e197b718eed7bf13176b83705"
VERSION = "e2-hpo-coverage-closure-v1"
FOLDS = ("2024-01-01", "2024-07-01", "2024-10-01")
PARAMETERS = ("learning_rate", "weight_decay", "batch_size", "dropout", "warmup_fraction")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    return digest(Path(path).read_bytes())


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def recover(output):
    source = output / "sources"
    require(not (source / "recovery_manifest.json").exists(), "Recovery exists; retain immutable inputs")
    source.mkdir(parents=True, exist_ok=True)
    manifest_raw = git_bytes(CLOSURE + "/manifest.json")
    require(digest(manifest_raw) == MANIFEST_SHA, "Closure manifest changed")
    manifest = json.loads(manifest_raw)
    (source / "closure_manifest.json").write_bytes(manifest_raw)
    for record in manifest["artifacts"]:
        raw = git_bytes(record["remote_path"])
        require(digest(raw) == record["sha256"] and len(raw) == record["bytes"], "Closure source mismatch")
        name = PurePosixPath(record["path"])
        require(not name.is_absolute() and ".." not in name.parts, "Unsafe archive path")
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    head_raw = git_bytes(HEAD_PATH)
    (source / "control_head.json").write_bytes(head_raw)
    head = json.loads(head_raw)
    wanted = {name: record for name, record in head["files"].items()
              if (name.startswith("jobs/") and name.rsplit("/", 1)[-1] in
                  ("selection_row.json", "request.json", "result.json", "job_result.json", "failure_result.json", "recovery_disposition.json",
                   "cached_source_record.json", "cache_admission.json"))
              or name.startswith("reconciliation/") or name == "shared_budget/pilot_budget.jsonl"}
    recovered = {}
    process = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for archive in reversed(head["metadata_deltas"]):
            process.stdin.write((archive["git_blob_sha"] + "\n").encode()); process.stdin.flush()
            header = process.stdout.readline().decode().strip().split()
            require(len(header) == 3 and header[1] == "blob", "Missing local Git blob")
            raw = process.stdout.read(int(header[2])); require(process.stdout.read(1) == b"\n", "Invalid batch framing")
            require(digest(raw) == archive["sha256"] and len(raw) == archive["bytes"], "Delta archive hash mismatch")
            with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
                for name in zipped.namelist():
                    if name not in wanted or name in recovered:
                        continue
                    value = zipped.read(name); record = wanted[name]
                    if digest(value) != record["sha256"] or len(value) != record["bytes"]:
                        continue  # This immutable version was superseded in a later delta.
                    relative = PurePosixPath(name)
                    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe delta path")
                    target = source / "metadata" / relative
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(value)
                    recovered[name] = {**record, "archive_path": archive["path"],
                                       "archive_sha256": archive["sha256"],
                                       "archive_git_blob_sha": archive["git_blob_sha"]}
    finally:
        process.stdin.close(); process.wait()
    require(set(recovered) == set(wanted), "Some metadata bound by the closure head is unavailable")
    for name, path in {
        "configurations_registered.json": ROOT / "experiments/e2_reduced_protocol_20261112/configurations.json",
        "configurations_TiDE_R0_v2.json": ROOT / "experiments/e2_tide_published_r0_20261112/configurations.json",
        "historical_trials.csv": ROOT / "experiments/e2_delivery_20261112/trials.csv",
        "phase_resources.csv": ROOT / "book_e2/_static/e2_efficiency_closure_20261113/phase_resources.csv",
    }.items():
        (source / name).write_bytes(path.read_bytes())
    files = {p.relative_to(source).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha(p)}
             for p in source.rglob("*") if p.is_file()}
    record = {"version": VERSION, "recovered_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": COMMIT, "source_commit_url": f"https://github.com/jassan-bit/DeepLearning_202630/tree/{COMMIT}",
              "closure_manifest_sha256": MANIFEST_SHA, "control_head_sha256": digest(head_raw),
              "metadata_deltas_checked": len(head["metadata_deltas"]),
              "recovered_bound_metadata_files": len(recovered), "metadata_provenance": recovered,
              "files": files, "new_fits": 0, "network_used": False}
    write_json(source / "recovery_manifest.json", record)
    return {"recovered_files": len(files), "bound_metadata_files": len(recovered), "source_commit": COMMIT}


def verify_sources(output):
    source = output / "sources"
    record = json.loads((source / "recovery_manifest.json").read_text())
    for name, expected in record["files"].items():
        path = source / name
        require(path.stat().st_size == expected["bytes"] and file_sha(path) == expected["sha256"],
                f"Recovered input changed: {name}")
    return record


def recover_dispositions(output):
    """Append omitted terminal receipts before analysis; retain initial recovery."""
    require(not (output / "analysis_plan.json").exists(), "Cannot change registered sources")
    source = output / "sources"
    verify_sources(output)
    head = json.loads((source / "control_head.json").read_text())
    wanted = {name: record for name, record in head["files"].items()
              if name.endswith(("failure_result.json", "recovery_disposition.json"))}
    recovered = {}
    for archive in reversed(head["metadata_deltas"]):
        raw = subprocess.check_output(["git", "cat-file", "blob", archive["git_blob_sha"]], cwd=ROOT)
        require(digest(raw) == archive["sha256"], "Disposition archive changed")
        with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
            for name in zipped.namelist():
                if name not in wanted or name in recovered:
                    continue
                value = zipped.read(name)
                if digest(value) != wanted[name]["sha256"]:
                    continue
                target = source / "metadata" / name
                if target.exists():
                    require(target.read_bytes() == value, "Existing terminal receipt changed")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(value)
                recovered[name] = {**wanted[name], "archive_path": archive["path"], "archive_sha256": archive["sha256"]}
        if set(recovered) == set(wanted): break
    require(set(recovered) == set(wanted), "Terminal receipt inaccessible")
    record = {"version": VERSION, "source_commit": COMMIT, "files": recovered}
    write_json(source / "terminal_receipts_provenance.json", record)
    return {"terminal_receipts": len(recovered)}


def register_plan(output):
    recovery = verify_sources(output)
    value = {
        "version": VERSION, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "posthoc_documentary_and_descriptive_HPO_reconciliation_no_new_fits",
        "script_sha256": file_sha(__file__), "source_recovery_manifest_sha256": file_sha(output / "sources/recovery_manifest.json"),
        "terminal_receipts_provenance_sha256": file_sha(output / "sources/terminal_receipts_provenance.json"),
        "source_commit": COMMIT, "expected_opportunities": 234, "expected_completed": 231,
        "expected_known_failures": 2, "expected_unknown_outcomes": 1,
        "folds": list(FOLDS), "selection_metric": "sqrt(sum(83*fold_RMSE^2)/249)",
        "catalogue": "13current neural recipes, including TIDE-D7-PUBLISHED-R0-v2; no NAS",
        "configuration_metric": "only complete3fold configurations eligible; no imputation for missing/failed/unknown outcomes",
        "cost_policy": "separate actual phase elapsed, campaign budget charges, conservative unknown allocation, and original cached-fit charges; no double-counted grand total",
        "importance_policy": "no isolated causal importance or fANOVA estimated from six covarying tuples; publish performance curves and exact parameter-level coverage",
        "new_fits": 0, "new_model_inference": False, "A2_unchanged": True,
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     "numpy": importlib.metadata.version("numpy"),
                     "matplotlib": importlib.metadata.version("matplotlib")},
        "recovered_source_files": len(recovery["files"]),
        "previous_registration_guard_failure_sha256": file_sha(output / "analysis_plan_initial_guard_failure.json")
            if (output / "analysis_plan_initial_guard_failure.json").exists() else None,
    }
    write_json(output / "analysis_plan.json", value)
    return value


def pooled_rmse(values, weights):
    require(len(values) == len(weights) and all(w > 0 for w in weights), "Invalid fold weights")
    return math.sqrt(sum(w * value**2 for value, w in zip(values, weights)) / sum(weights))


def configurations(source):
    values = json.loads((source / "configurations_registered.json").read_text())
    tide = json.loads((source / "configurations_TiDE_R0_v2.json").read_text())
    values["TiDE"] = tide["TiDE"]
    require(len(values) == 13 and all(len(x) == 6 for x in values.values()), "Requires13recipes and6slots")
    require(all(x["trial"]["recipe_id"] == "TIDE-D7-PUBLISHED-R0-v2" for x in values["TiDE"]), "Retired TiDE body entered")
    return values


def verify_ledger(events):
    previous = "0" * 64
    for i, event in enumerate(events):
        require(event["sequence"] == i and event["previous_sha256"] == previous, "Budget chronology chain changed")
        unsigned = {k: v for k, v in event.items() if k != "sha256"}
        canonical = json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        require(digest(canonical) == event["sha256"], "Budget event hash invalid")
        previous = event["sha256"]
    return previous


def read_opportunities(source, configs):
    ledger_path = source / "metadata/shared_budget/pilot_budget.jsonl"
    events = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    verify_ledger(events)
    charges = {r["attempt_id"]: r for r in events if r["kind"] == "charged"}
    rows = []
    for path in sorted((source / "metadata/jobs").glob("*/selection_row.json")):
        row = json.loads(path.read_text())
        model, slot, fold = row["model_id"], row["configuration_id"], row["fold_origin"]
        entry = next(c for c in configs[model] if c["configuration_id"] == slot)
        require(row["configuration_sha256"] == entry["configuration_sha256"], "Configuration binding changed")
        require(row["seed"] == 271201 and fold in FOLDS, "Wrong HPO seed/calendar")
        folder = path.parent; attempt = folder.name
        result_path = folder / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        failure_path = folder / "failure_result.json"
        failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
        cache = bool(row.get("resource_use", {}).get("cache_reuse", result.get("cache_reuse", False)))
        unknown = row.get("failure_code") == "expired_runtime_outcome_unknown" or row.get("resource_use", {}).get("outcome_unknown", False)
        status = "outcome_unknown" if unknown else ("completed" if row["status"] == "completed" else "known_failed")
        metric = row.get("rmse_original")
        require((metric is not None) == (status == "completed"), "Missing or fabricated performance outcome")
        request_path = folder / "request.json"
        request = json.loads(request_path.read_text()) if request_path.exists() else {}
        cached_path = folder / "cached_source_record.json"
        cached = json.loads(cached_path.read_text()) if cached_path.exists() else {}
        phases = result.get("phase_records", failure.get("phase_records", {}))
        phase_seconds = [r["elapsed_seconds"] for r in phases.values() if r.get("elapsed_seconds") is not None]
        measured = sum(phase_seconds) if phase_seconds and len(phase_seconds) == len(phases) else None
        charge = charges.get(attempt)
        basis = (charge or {}).get("resource_use", {}).get("accounting_basis", "measured_closed_campaign_charge" if charge else "not_present")
        rows.append({
            "model": model, "configuration_id": slot, "fold_origin": fold, "seed": row["seed"],
            "parent_trial_index": entry["parent_trial_index"], "trial_id": row["parent_trial_id"],
            "configuration_sha256": row["configuration_sha256"], "status": status,
            "source_status": row["status"], "RMSE_original": metric,
            "validation_count": row["validation_count"] if status == "completed" else None,
            "admitted_validation_count": row.get("validation_count"),
            "best_epoch": row.get("best_epoch"), "phase_best_epochs": row.get("phase_best_epochs"),
            "cached_outcome_reused": cache, "new_training_receipt_present": bool(result.get("training_performed", False)),
            "optimizer_steps_reported": row.get("resource_use", {}).get("optimizer_steps"),
            "measured_phase_elapsed_seconds": measured,
            "phase_names": list(phases),
            "campaign_budget_charge_seconds": charge.get("elapsed_seconds") if charge else None,
            "campaign_charge_basis": basis, "campaign_charge_event_sha256": charge.get("sha256") if charge else None,
            "original_cached_fit_charge_seconds": cached.get("charged_wall_seconds"),
            "original_cached_fit_callback_seconds": cached.get("callback_wall_seconds"),
            "current_device": request.get("resources", {}).get("device_name"),
            "failure_code": row.get("failure_code"), "reason": row.get("reason"),
            "terminal_failure_receipt_path": failure_path.relative_to(source).as_posix() if failure else None,
            "terminal_failure_receipt_sha256": file_sha(failure_path) if failure else None,
            "failed_worker_wall_seconds": failure.get("supervision", {}).get("charged_wall_seconds"),
            "attempt_id": attempt, "selection_row_path": path.relative_to(source).as_posix(),
            "selection_row_sha256": file_sha(path),
            "result_path": result_path.relative_to(source).as_posix() if result_path.exists() else None,
            "result_sha256": file_sha(result_path) if result_path.exists() else None,
            "cached_source_path": cached_path.relative_to(source).as_posix() if cached else None,
            "cached_source_sha256": file_sha(cached_path) if cached else None,
            "budget_ledger_sha256": file_sha(ledger_path),
        })
    expected = {(model, row["configuration_id"], fold) for model, values in configs.items() for row in values for fold in FOLDS}
    observed = {(r["model"], r["configuration_id"], r["fold_origin"]) for r in rows}
    require(len(rows) == len(observed) == 234 and observed == expected, "Logical opportunity registry incomplete or duplicated")
    require(Counter(r["status"] for r in rows) == {"completed": 231, "known_failed": 2, "outcome_unknown": 1}, "Disposition reconciliation differs")
    return rows


def summarize_configs(source, configs, opportunities):
    records = []
    for model, values in configs.items():
        selection_path = source / "selections" / (model.replace(" ", "-") + ".json")
        selection = json.loads(selection_path.read_text())
        for entry in values:
            slot = entry["configuration_id"]
            folds = [r for r in opportunities if r["model"] == model and r["configuration_id"] == slot]
            valid = [r for r in folds if r["status"] == "completed"]
            eligible = len(valid) == 3
            point = pooled_rmse([r["RMSE_original"] for r in valid], [r["validation_count"] for r in valid]) if eligible else None
            recorded = next(r for r in selection["candidate_ledger"] if r["configuration_id"] == slot)
            require(eligible == (recorded["status"] == "completed"), "Selection eligibility differs from fold receipts")
            if eligible:
                require(math.isclose(point, recorded["pooled_cv_RMSE_original"], rel_tol=1e-12, abs_tol=1e-14), "Pooled score differs from original")
            trial = entry["trial"]
            records.append({"model": model, "configuration_id": slot, "parent_trial_index": entry["parent_trial_index"],
                            "trial_id": trial["trial_id"], "recipe_id": trial["recipe_id"],
                            **{name: trial.get(name, 0 if name == "warmup_fraction" else None) for name in PARAMETERS},
                            "completed_folds": len(valid), "known_failed_folds": sum(r["status"] == "known_failed" for r in folds),
                            "unknown_outcome_folds": sum(r["status"] == "outcome_unknown" for r in folds),
                            "eligible_for_selection": eligible, "pooled_cv_RMSE": point,
                            "recorded_pooled_cv_RMSE": recorded.get("pooled_cv_RMSE_original"),
                            "recorded_winner": slot == selection["winner_configuration_id"],
                            "fixed_phase_epochs_if_complete": recorded.get("fixed_phase_epochs"),
                            "selection_source_path": selection_path.relative_to(source).as_posix(),
                            "selection_source_sha256": file_sha(selection_path)})
        complete = [r for r in records if r["model"] == model and r["eligible_for_selection"]]
        calculated = min(complete, key=lambda r: (r["pooled_cv_RMSE"], r["configuration_id"]))
        require(calculated["configuration_id"] == selection["winner_configuration_id"], "Recorded winner changed")
    return records


def parameter_coverage(config_rows, opportunities):
    summaries, levels = [], []
    for model in dict.fromkeys(r["model"] for r in config_rows):
        configs = [r for r in config_rows if r["model"] == model]
        winner = next(r for r in configs if r["recorded_winner"])
        for parameter in PARAMETERS:
            values = sorted({r[parameter] for r in configs if r[parameter] is not None})
            require(values, "An applicable training parameter is missing")
            summaries.append({"model": model, "parameter": parameter, "applicable": True,
                              "observed_registered_values": values, "distinct_registered_values": len(values),
                              "variation": "varied" if len(values) > 1 else "fixed",
                              "winner_value": winner[parameter], "winner_configuration_id": winner["configuration_id"],
                              "winner_parent_index": winner["parent_trial_index"],
                              "winner_value_sorted_level_position": values.index(winner[parameter])+1,
                              "position_is_value_order_not_performance_rank": True,
                              "six_joint_tuples_not_full_factorial": True,
                              "warmup_missing_key_means_registered_default_zero": parameter == "warmup_fraction"})
            for value in values:
                slots = [r["configuration_id"] for r in configs if r[parameter] == value]
                attempts = [r for r in opportunities if r["model"] == model and r["configuration_id"] in slots]
                levels.append({"model": model, "parameter": parameter, "value": value,
                               "configuration_ids": slots, "registered_configuration_count": len(slots),
                               "planned_fold_opportunities": len(attempts),
                               "completed_fold_results": sum(r["status"] == "completed" for r in attempts),
                               "known_failures": sum(r["status"] == "known_failed" for r in attempts),
                               "unknown_outcomes": sum(r["status"] == "outcome_unknown" for r in attempts),
                               "complete_configurations": sum(r["eligible_for_selection"] for r in configs if r["configuration_id"] in slots),
                               "contains_recorded_winner": winner["configuration_id"] in slots})
    return summaries, levels


def figures(output, rows, opportunities, coverage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)
    colors = ("#2878b5", "#d47a12", "#9b4a9d")
    for ax, model in zip(axes.flat, models):
        selected = [r for r in rows if r["model"] == model]
        for f, color in zip(FOLDS, colors):
            scores = [next(r for r in opportunities if r["model"] == model and r["configuration_id"] == c["configuration_id"] and r["fold_origin"] == f)["RMSE_original"] for c in selected]
            ax.plot(range(1, 7), [x if x is not None else np.nan for x in scores], ".:", color=color, alpha=.75, label=f)
        scores = [r["pooled_cv_RMSE"] if r["pooled_cv_RMSE"] is not None else np.nan for r in selected]
        ax.plot(range(1, 7), scores, "o-", color="black", lw=1.6, label="RMSE concatenada")
        for i, row in enumerate(selected, 1):
            if row["recorded_winner"]:
                ax.plot(i, row["pooled_cv_RMSE"], "*", color="#128540", markersize=13)
            if not row["eligible_for_selection"]:
                ax.axvspan(i-.23, i+.23, color="#bd3333", alpha=.12)
        ax.set_title("SAE" if "autoencoder" in model else model)
        ax.set_xticks(range(1, 7), ["C%02d" % i for i in range(1, 7)])
        ax.set_ylabel("RMSE interna, σ₇")
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(models):]: ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle("HPO registrado: seis tuplas y tres folds por red\nEstrella: ganador conservado; sombreado: configuración incompleta. Ejes verticales por modelo.")
    fig.savefig(output / "performance_by_configuration.svg")
    fig.savefig(output / "performance_by_configuration.png", dpi=160)
    plt.close(fig)
    matrix = np.array([[next(r["distinct_registered_values"] for r in coverage if r["model"] == model and r["parameter"] == parameter) for parameter in PARAMETERS] for model in models])
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    im = ax.imshow(matrix, vmin=1, vmax=6, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(models)), ["SAE" if "autoencoder" in m else m for m in models])
    ax.set_xticks(range(5), ["Learning rate", "Weight decay", "Batch", "Dropout", "Warmup"])
    for i in range(len(models)):
        for j in range(5): ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > 3 else "black")
    ax.set_title("Número de valores distintos en las seis tuplas\n1 = fijo; dimensiones covarían. No representa importancia ni cobertura factorial.")
    fig.colorbar(im, ax=ax, label="Valores distintos registrados", ticks=range(1, 7))
    fig.savefig(output / "parameter_coverage.svg")
    fig.savefig(output / "parameter_coverage.png", dpi=160)
    plt.close(fig)


def execute(output):
    plan = json.loads((output / "analysis_plan.json").read_text())
    require(plan["script_sha256"] == file_sha(__file__), "Program bytes differ from plan")
    require(plan["source_recovery_manifest_sha256"] == file_sha(output / "sources/recovery_manifest.json"), "Recovery registration changed")
    require(plan["terminal_receipts_provenance_sha256"] == file_sha(output / "sources/terminal_receipts_provenance.json"), "Terminal sources changed")
    terminal = json.loads((output / "sources/terminal_receipts_provenance.json").read_text())
    for name, record in terminal["files"].items():
        require(file_sha(output / "sources/metadata" / name) == record["sha256"], "Terminal receipt bytes changed")
    verify_sources(output)
    started = datetime.now(timezone.utc).isoformat()
    source = output / "sources"
    configs = configurations(source)
    opportunities = read_opportunities(source, configs)
    config_rows = summarize_configs(source, configs, opportunities)
    coverage, levels = parameter_coverage(config_rows, opportunities)
    write_csv(output / "opportunities_234.csv", opportunities)
    write_csv(output / "configurations_78.csv", config_rows)
    write_csv(output / "parameter_coverage_65.csv", coverage)
    write_csv(output / "parameter_value_outcomes.csv", levels)
    summary_rows = []
    for model in configs:
        rows = [r for r in opportunities if r["model"] == model]
        selected = next(r for r in config_rows if r["model"] == model and r["recorded_winner"])
        summary_rows.append({"model": model, "planned": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
                             "known_failed": sum(r["status"] == "known_failed" for r in rows),
                             "unknown": sum(r["status"] == "outcome_unknown" for r in rows),
                             "reused_outcomes": sum(r["cached_outcome_reused"] for r in rows),
                             "measured_phase_time_receipts": sum(r["measured_phase_elapsed_seconds"] is not None for r in rows),
                             "known_training_phase_seconds": sum(r["measured_phase_elapsed_seconds"] for r in rows if r["measured_phase_elapsed_seconds"] is not None),
                             "campaign_charge_receipts": sum(r["campaign_budget_charge_seconds"] is not None for r in rows),
                             "campaign_charge_seconds": sum(r["campaign_budget_charge_seconds"] for r in rows if r["campaign_budget_charge_seconds"] is not None),
                             "original_cached_fit_charge_receipts": sum(r["original_cached_fit_charge_seconds"] is not None for r in rows),
                             "winner_configuration_id": selected["configuration_id"], "winner_pooled_cv_RMSE": selected["pooled_cv_RMSE"]})
    write_csv(output / "model_reconciliation_13.csv", summary_rows)
    figures(output, config_rows, opportunities, coverage)
    summary = {"version": VERSION, "opportunities": 234, "completed": 231, "known_failed": 2, "outcome_unknown": 1,
               "configuration_count": len(config_rows), "eligible_complete_configurations": sum(r["eligible_for_selection"] for r in config_rows),
               "recorded_winners_reproduced": 13, "reused_outcomes": sum(r["cached_outcome_reused"] for r in opportunities),
               "original_cached_fit_cost_receipts": sum(r["original_cached_fit_charge_seconds"] is not None for r in opportunities),
               "measured_phase_time_receipts": sum(r["measured_phase_elapsed_seconds"] is not None for r in opportunities),
               "campaign_budget_charge_receipts": sum(r["campaign_budget_charge_seconds"] is not None for r in opportunities),
               "known_training_phase_seconds": sum(r["measured_phase_elapsed_seconds"] for r in opportunities if r["measured_phase_elapsed_seconds"] is not None),
               "campaign_budget_charge_seconds": sum(r["campaign_budget_charge_seconds"] for r in opportunities if r["campaign_budget_charge_seconds"] is not None),
               "unknown_allocation_is_not_measured_training_time": True,
               "cost_categories_not_added_into_one_total": True, "new_fits": 0, "new_model_inference": False,
               "NAS_executed": False, "TiDE_recipe": "TIDE-D7-PUBLISHED-R0-v2",
               "isolated_hyperparameter_importance_identified": False,
               "limitations": ["Only six covarying configurations and one optimization seed per fold; parameter effects are confounded.",
                               "The same validation folds selected the winner; performance curves do not estimate selection uncertainty or external generalization.",
                               "The lost T4 session retains unknown scientific outcome and conservative budget charge, never reconstructed metrics or optimizer steps.",
                               "Failed configurations retain their available fold scores but are ineligible for pooled selection.",
                               "Recorded applicable dimensions may remain fixed; this inventory does not close missing search breadth or family-specific minima.",
                               "Current campaign charges, original cached-fit cost and phase elapsed have different scopes and must not be added indiscriminately."]}
    write_json(output / "summary.json", summary)
    verify_sources(output)
    write_json(output / "execution_receipt.json", {"version": VERSION, "started_at_utc": started,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": file_sha(output / "analysis_plan.json"),
                "script_sha256": file_sha(__file__), "source_recovery_manifest_sha256": file_sha(source / "recovery_manifest.json"),
                "input_bytes_reverified": True, "new_fits": 0, "new_model_inference": False})
    (output / "README.md").write_text("""# Cobertura y recibos del HPO ejecutado

Se recuperan fuentes públicas ya existentes del cierre de selección, sus hashes,
recibos por oportunidad y diario de presupuesto. No se realizan ajustes nuevos,
no se cambian ganadores, no se ejecuta NAS y TiDE conserva su cuerpo publicado R0-v2.

`opportunities_234.csv` distingue resultado válido, fallo conocido y desenlace
desconocido. Mantiene métricas disponibles de configuraciones incompletas sin
inventar las que faltan. `configurations_78.csv` recalcula la RMSE concatenada de
los tres folds (83 fechas por fold) y verifica los trece ganadores registrados.

`parameter_coverage_65.csv` y `parameter_value_outcomes.csv` identifican valores,
variación/fijación, posición del ganador y éxito por nivel. Las figuras muestran
rendimiento por configuración y cobertura observada. Seis tuplas que cambian
varias dimensiones a la vez no identifican importancia aislada, causal o fANOVA:
no se publica una falsa importancia para cerrar esa casilla.

Los costes se separan: duración de fases medida, cargo del diario de esta campaña
y coste histórico del ajuste cacheado. Un resultado reutilizado no es un nuevo
entrenamiento. El cargo conservador de la sesión T4 perdida no es tiempo de
entrenamiento medido. Los campos desconocidos se mantienen vacíos, sin asignar cero.
Las sumas conocidas no se presentan como una atribución integral de todo el
HPO histórico ni como coste GPU activo; no se suman categorías que pueden solaparse.

Las fuentes originales y su procedencia están en `sources/` y su manifiesto.
Reproducción con Git y Python/Matplotlib instalados, en una carpeta nueva:

```sh
python scripts/e2_hpo_coverage_closure.py --output /tmp/e2-hpo-repeat --recover-only
python scripts/e2_hpo_coverage_closure.py --output /tmp/e2-hpo-repeat --recover-dispositions
python scripts/e2_hpo_coverage_closure.py --output /tmp/e2-hpo-repeat --register-plan
python scripts/e2_hpo_coverage_closure.py --output /tmp/e2-hpo-repeat --execute
```

La recuperación lee objetos Git del commit fijado; no consulta resultados nuevos.
La fase de cálculo requiere los mismos bytes registrados y no sobrescribe salidas.
""", encoding="utf-8")
    files = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": file_sha(p)}
             for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"}
    write_json(output / "manifest.json", {"version": VERSION, "files": files, "new_fits": 0})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--recover-only", action="store_true")
    action.add_argument("--recover-dispositions", action="store_true")
    action.add_argument("--register-plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = (recover(args.output) if args.recover_only else recover_dispositions(args.output)
              if args.recover_dispositions else register_plan(args.output) if args.register_plan else execute(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
