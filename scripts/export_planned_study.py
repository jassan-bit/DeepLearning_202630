"""Export only public aggregates from the completed, separately planned D7 study.

This post-execution publishing adapter never trains, loads a checkpoint, decodes
an archive, calculates a target or performs model inference.  Its source hash is
not attributed retrospectively to the registered scientific implementation.
Saved individual predictions are read only for the explicitly fixed-seed PNG;
their vectors, raw observations, learned scalers and weights are never exported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "gru_lstm_tcn_planned_20261105"
DEFAULT_EXPERIMENT = Path("/workspace/e1_new_study_20261105/run")
MODELS = ("GRU", "LSTM", "TCN")
ANALYSIS_TABLES = ("metric_summary", "run_metrics", "pairwise_rmse", "dm_pairs",
                   "temporal_bootstrap", "diagnostics")
PRIVATE_KEYS = {
    "y_true", "y_pred", "prediction", "predictions", "residual", "residuals",
    "x_mean", "x_scale", "x_std", "y_mean", "y_scale", "y_std", "mase_scale",
    "mase_train_scale", "preprocessing", "scalers", "scaler", "state_dict",
    "train_indices", "test_indices", "inner_train_indices", "inner_validation_indices",
    "eligible_indices", "raw_root", "environment", "environ", "env",
}
PRIVATE_NAMES = {
    "predictions.csv", "pilot_predictions.csv", "holdout_predictions.csv",
    "daily_market_reserved.csv", "daily_market_development.csv", "daily_market_test.csv",
    "daily_volatility_development.csv", "daily_volatility_test.csv", "preprocessing.json",
    "history_inner.csv", "history_refit.csv", "split.json",
}
SAFE_SUFFIXES = {".json", ".csv", ".md", ".png"}
CATALOG_FIELDS = (
    "model", "candidate_id", "phase", "trajectory_seed", "seed", "width", "learning_rate",
    "dropout", "parameters", "selected_epoch", "train_samples", "inner_train_samples",
    "inner_validation_samples", "training_seconds", "inference_seconds", "evaluation_days",
    "n_dates", "rmse", "mae", "mse", "r2", "mape", "mase",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def public_metadata(value):
    """Keep configuration/hash provenance while removing learned/data vectors."""
    if isinstance(value, dict):
        return {str(k): public_metadata(v) for k, v in value.items()
                if str(k).lower() not in PRIVATE_KEYS}
    if isinstance(value, (list, tuple)):
        return [public_metadata(v) for v in value]
    return clean(value)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


def _safe(base, relative):
    base, relative = Path(base).resolve(), Path(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("A source artifact must use a safe relative path")
    result = (base / relative).resolve()
    if not result.is_relative_to(base) or result.is_symlink():
        raise ValueError("Artifact escapes its declared source")
    return result


def verify_mapping(base, mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("A nonempty signed artifact mapping is required")
    for name, expected in mapping.items():
        if digest(_safe(base, name)) != expected:
            raise ValueError("A signed source artifact changed: " + name)


def utc(value):
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Every chronology timestamp needs an explicit timezone")
    return stamp.astimezone(timezone.utc)


def chronology(registration, selection, pilot, plan, final, access, score, analysis):
    """Check actual recorded order; never infer a preregistration from filenames."""
    times = {
        "registered_before_hpo_at_utc": registration["registered_at_utc"],
        "recipes_selected_before_pilot_at_utc": selection["selected_at_utc"],
        "pilot_completed_at_utc": pilot["completed_at_utc"],
        "dimensioned_before_definitive_at_utc": plan["planned_at_utc"],
        "all_weights_frozen_at_utc": final["all_weights_frozen_at_utc"],
        "first_reserve_price_read_started_at_utc": access["at_utc"],
        "scoring_completed_at_utc": score["completed_at_utc"],
        "analysis_completed_at_utc": analysis["analyzed_at_utc"],
    }
    values = [utc(v) for v in times.values()]
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("The recorded prospective stage order is not strictly increasing")
    signature = registration["registration_signature"]
    for label, record in (("selection", selection), ("pilot", pilot), ("plan", plan),
                          ("final", final), ("score", score), ("analysis", analysis)):
        if record.get("registration_signature") != signature:
            raise ValueError("A stage belongs to another registration: " + label)
    if (registration.get("reserve_price_payloads_read") != 0
            or selection.get("reserve_price_payloads_read") != 0
            or pilot.get("reserve_price_payloads_read") != 0
            or final.get("reserve_price_payloads_read") != 0):
        raise ValueError("A pre-score stage reports reserve price access")
    return {
        "timestamps": times,
        "strict_stage_order_verified": True,
        "registration_signature": signature,
        "prior_access_scope": registration.get("holdout_scope"),
        "previous_benchmark_stays_exploratory": True,
        "new_reserve_is_historical_not_live": True,
        "new_reserve_is_another_market": False,
        "temporal_power_claimed": False,
    }


def verify_closed_planning(closed, attestation, registration, selection, plan,
                           source_sha256, first_final_completion_at):
    """Validate the public closure of recipes/R against the actual private plan.

    A completion timestamp is labeled as completion, not invented as fit start.
    The external publication receipt separately records that no definitive fit
    had begun when the public closure was verified.
    """
    count = int(plan["R"])
    if (closed["registration_signature"] != registration["registration_signature"]
            or closed["registration_sha256"] != source_sha256["registration"]
            or closed["hpo_selection_sha256"] != source_sha256["selection"]
            or closed["pilot_complete_sha256"] != source_sha256["pilot"]
            or closed["dimensioning_sha256"] != source_sha256["plan"]
            or closed["dimensioning"] != plan
            or closed["selected_recipes"] != selection["recipes"]
            or closed["baseline_model_selected_by_hpo"] != selection["baseline_model"]
            or closed["preregistration_commit"] != registration["source_commit"]
            or closed["planned_fresh_final_models"] != 3 * count
            or closed["planned_final_fit_stages"] != 6 * count
            or closed["planned_reserve_prediction_rows"] != 3 * count * 60
            or closed["new_reserve_payloads_read"] != 0
            or not utc(plan["planned_at_utc"]) < utc(closed["closed_before_definitive_fit_at_utc"])
                < utc(first_final_completion_at)):
        raise ValueError("The public recipe/dimensioning closure does not match this definitive study")
    result = {
        "closed_planning_sha256": source_sha256["closed_planning"],
        "closed_before_definitive_fit_at_utc": closed["closed_before_definitive_fit_at_utc"],
        "private_dimensioning_sha256_matches_public_closure": True,
        "registered_R_matches_public_closure": count,
        "first_definitive_job_completed_at_utc": first_final_completion_at,
        "completion_time_is_not_claimed_as_fit_start": True,
    }
    if attestation is not None:
        if (attestation["public_planning_sha256"] != source_sha256["closed_planning"]
                or attestation["planned_R_per_family"] != count
                or attestation["definitive_fits_started_before_publication"] != 0
                or attestation["new_reserve_payload_accessed"] is not False
                or not utc(closed["closed_before_definitive_fit_at_utc"])
                    < utc(attestation["public_planning_verified_at_utc"])
                    < utc(first_final_completion_at)):
            raise ValueError("Public dimensioning verification does not precede the definitive model completions")
        result.update({
            "public_planning_commit": attestation["public_planning_commit"],
            "public_planning_url": attestation["public_planning_url"],
            "public_planning_verified_at_utc": attestation["public_planning_verified_at_utc"],
            "public_receipt_records_zero_definitive_fits_started_at_verification": True,
        })
    return result


def public_catalog(frame):
    allowed = [field for field in CATALOG_FIELDS if field in frame.columns]
    if not {"model", "trajectory_seed"}.issubset(frame.columns):
        if not {"model", "seed"}.issubset(frame.columns):
            raise ValueError("A trajectory catalog needs model and seed identity")
    result = frame[allowed].copy()
    if "trajectory_seed" in result:
        result = result.rename(columns={"trajectory_seed": "seed"}) if "seed" not in result else result.drop(columns="trajectory_seed")
    return result


def cost_summary(phases):
    """Summarize observed wall time; missing timings are never replaced by zero."""
    rows = []
    for phase, frame in phases.items():
        for model in MODELS:
            group = frame.loc[frame.model.eq(model)]
            if not len(group):
                raise ValueError("Every cost phase needs the three registered models")
            row = {"phase": phase, "model": model, "runs": len(group),
                   "scope": "Observed CPU wall time; same registered workers/threads; not a repeated latency benchmark"}
            for column in ("training_seconds", "inference_seconds"):
                values = pd.to_numeric(group[column], errors="coerce").dropna() if column in group else pd.Series(dtype=float)
                row[column + "_n_defined"] = len(values)
                row[column + "_total"] = float(values.sum()) if len(values) else np.nan
                row[column + "_mean"] = float(values.mean()) if len(values) else np.nan
                row[column + "_sample_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def validate_payloads(directory):
    """Reject raw-vector names/fields and any unapproved binary format."""
    directory = Path(directory)
    for path in directory.iterdir():
        if (not path.is_file() or path.is_symlink() or path.name in PRIVATE_NAMES
                or path.suffix.lower() not in SAFE_SUFFIXES):
            raise ValueError("Refusing a nonpublic export member: " + path.name)
        if path.suffix == ".csv":
            fields = set(pd.read_csv(path, nrows=0).columns)
            if fields & PRIVATE_KEYS:
                raise ValueError("Individual values or scalers in a public CSV: " + path.name)
        if path.suffix == ".json":
            def inspect(value):
                if isinstance(value, dict):
                    bad = set(map(str.lower, value)) & PRIVATE_KEYS
                    if bad:
                        raise ValueError("Private JSON fields in " + path.name + ": " + str(sorted(bad)))
                    for item in value.values():
                        inspect(item)
                elif isinstance(value, list):
                    for item in value:
                        inspect(item)
            inspect(read_json(path))


def publish_identical(staging, destinations):
    """Copy one allowlisted snapshot; refuse silently replacing other evidence."""
    staging = Path(staging)
    source = {path.name: path.read_bytes() for path in staging.iterdir()}
    for destination in map(Path, destinations):
        if destination.exists() and any(destination.iterdir()):
            current = {path.name: path.read_bytes() for path in destination.iterdir() if path.is_file()}
            if current != source or len(current) != len(list(destination.iterdir())):
                raise ValueError("Never overwrite a different published evidence snapshot: " + str(destination))
        destination.mkdir(parents=True, exist_ok=True)
        for name, payload in source.items():
            (destination / name).write_bytes(payload)
        if {p.name: p.read_bytes() for p in destination.iterdir()} != source:
            raise ValueError("Public evidence mirror differs after copying")


def _plot(staging, candidates, pilot, metrics, predictions, fixed_seed):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = {"GRU": "#3366cc", "LSTM": "#e67e22", "TCN": "#138d75", "Persistence": "#616a6b"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for axis, model in zip(axes, MODELS):
        group = candidates.loc[candidates.model.eq(model)].sort_values("candidate_id")
        axis.errorbar(group.candidate_id, group.mean_rmse, yerr=group.sample_sd,
                      fmt="o", capsize=3, color=colors[model])
        axis.set(title=model, xlabel="Candidato registrado")
        axis.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("RMSE CV; media ± DE de dos semillas")
    fig.suptitle("Mismo grid: ocho candidatos y dos semillas por familia")
    fig.tight_layout()
    fig.savefig(staging / "hpo_candidates.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 3.5))
    for number, model in enumerate(MODELS):
        group = pilot.loc[pilot.model.eq(model)].sort_values("seed")
        axis.scatter(np.full(len(group), number), group.rmse, color=colors[model], alpha=.8)
    axis.set(xticks=range(3), xticklabels=MODELS, ylabel="RMSE por semilla",
             title="Piloto: diez semillas por receta; 60 fechas de desarrollo")
    fig.tight_layout()
    fig.savefig(staging / "pilot_rmse.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 3.5))
    for number, model in enumerate((*MODELS, "Persistence")):
        row = metrics.loc[metrics.model.eq(model) & metrics.metric.eq("rmse")].iloc[0]
        mean = float(row["mean"])
        if model == "Persistence":
            axis.scatter([number], [mean], marker="D", color=colors[model])
        else:
            axis.errorbar(number, mean,
                yerr=np.array([[mean-float(row.ci95_lower)], [float(row.ci95_upper)-mean]]),
                fmt="o", color=colors[model], capsize=5)
    axis.set(xticks=range(4), xticklabels=(*MODELS, "Persistencia"), ylabel="RMSE",
             title="Reserva histórica: IC95 % entre semillas, condicional al calendario")
    fig.tight_layout()
    fig.savefig(staging / "holdout_rmse.png", dpi=160)
    plt.close(fig)

    # The fixed seed is declared before the new outcome read, not chosen by error.
    fig, axis = plt.subplots(figsize=(11, 4))
    sample = predictions.loc[predictions.model.eq(MODELS[0]) & predictions.seed.eq(fixed_seed)].sort_values("outcome_day")
    if len(sample) != 60:
        raise ValueError("The fixed-seed illustration needs exactly the registered sixty dates")
    axis.plot(pd.to_datetime(sample.outcome_day), sample.y_true, color="black", lw=1.8, label="Observado")
    for model in MODELS:
        sample = predictions.loc[predictions.model.eq(model) & predictions.seed.eq(fixed_seed)].sort_values("outcome_day")
        if len(sample) != 60:
            raise ValueError("Incomplete fixed-seed illustration")
        axis.plot(pd.to_datetime(sample.outcome_day), sample.y_pred, color=colors[model], alpha=.85, label=model)
    axis.set(ylabel="σ₇, sin anualizar", title=f"Ilustración fija: semilla {fixed_seed}; no representa un ensemble")
    axis.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(staging / "predictions_fixed_seed.png", dpi=160)
    plt.close(fig)


def export_study(experiment=DEFAULT_EXPERIMENT, project_root=ROOT):
    root, experiment = Path(project_root).resolve(), Path(experiment).resolve()
    if experiment.is_relative_to(root):
        raise ValueError("The completed private experiment must remain outside the public checkout")
    required = ("public_preregistration.json", "registration.json", "reservation.json", "design_original.json", "hpo/selection.json",
        "hpo/candidate_summary.csv", "hpo/run_metrics.csv", "pilot/complete.json", "pilot/metrics.csv",
        "plan.json", "final_models_manifest.json", "final/metrics.csv", "score/raw_access_started.json",
        "score/frozen_access_gate.json", "score/complete.json", "score/data_audit.json", "score/run_metrics.csv",
        "score/predictions.csv", "analysis/manifest.json", "analysis/summary.json",
        *[f"analysis/{name}.csv" for name in ANALYSIS_TABLES])
    for name in required:
        if not _safe(experiment, name).is_file():
            raise FileNotFoundError("Complete the registered study before exporting: " + name)
    records = {name: read_json(experiment / name) for name in required if name.endswith(".json")}
    registration = records["registration.json"]
    design = records["design_original.json"]
    if design != registration["design"] or design != read_json(root / "experiments" / ID / "design.json"):
        raise ValueError("The original design differs from its registered public source")
    verify_mapping(root, registration["sources_sha256"])
    public_registration = records["public_preregistration.json"]
    if (public_registration["public_code_commit"] != registration["source_commit"]
            or public_registration["public_design_sha256"] != digest(experiment / "design_original.json")
            or public_registration["new_real_model_fits_before_verification"] != 0
            or public_registration["new_reserve_outcomes_evaluated_before_verification"] != 0
            or utc(public_registration["public_design_verified_at_utc"]) >= utc(registration["registered_at_utc"])):
        raise ValueError("The public preregistration record does not precede this registered experiment")
    selection, pilot, plan, final, score, analysis = [records[name] for name in
        ("hpo/selection.json", "pilot/complete.json", "plan.json", "final_models_manifest.json",
         "score/complete.json", "analysis/manifest.json")]
    for record in (selection, pilot, final, score, analysis):
        verify_mapping(experiment, record["artifact_sha256"])
    if (plan["hpo_selection_sha256"] != digest(experiment / "hpo/selection.json")
            or plan["pilot_complete_sha256"] != digest(experiment / "pilot/complete.json")
            or final["plan_sha256"] != digest(experiment / "plan.json")
            or score["frozen_models_manifest_sha256"] != digest(experiment / "final_models_manifest.json")
            or analysis["score_complete_sha256"] != digest(experiment / "score/complete.json")):
        raise ValueError("A linked stage digest no longer matches its predecessor")
    times = chronology(registration, selection, pilot, plan, final,
                       records["score/raw_access_started.json"], score, analysis)
    times["public_design_verified_before_registration_at_utc"] = public_registration["public_design_verified_at_utc"]
    times["public_preregistration_commit"] = public_registration["public_code_commit"]
    times["public_preregistration_url"] = public_registration["commit_url"]
    canonical = root / "experiments" / ID
    if digest(canonical / "public_preregistration.json") != digest(experiment / "public_preregistration.json"):
        raise ValueError("Canonical and private public-preregistration receipts differ")
    closed = read_json(canonical / "closed_planning.json")
    final_completions = [read_json(experiment / "final" /
        f"{job['model']}_s{job['trajectory_seed']}" / "completion.json")["completed_at_utc"]
        for job in final["jobs"]]
    if not final_completions:
        raise ValueError("Every definitive job needs its recorded completion")
    first_final_completion = min(final_completions, key=utc)
    if utc(first_final_completion) > utc(final["all_weights_frozen_at_utc"]):
        raise ValueError("Definitive completion cannot follow its all-weights freeze")
    public_dimensioning_path = experiment / "public_dimensioning.json"
    public_dimensioning = read_json(public_dimensioning_path) if public_dimensioning_path.is_file() else None
    planning_closure = verify_closed_planning(closed, public_dimensioning, registration, selection, plan, {
        "registration": digest(experiment / "registration.json"),
        "selection": digest(experiment / "hpo/selection.json"),
        "pilot": digest(experiment / "pilot/complete.json"),
        "plan": digest(experiment / "plan.json"),
        "closed_planning": digest(canonical / "closed_planning.json"),
    }, first_final_completion)
    times["public_dimensioning_closure"] = planning_closure
    count = int(plan["R"])
    gate = records["score/frozen_access_gate.json"]
    if (gate["registration_sha256"] != digest(experiment / "registration.json")
            or gate["plan_sha256"] != digest(experiment / "plan.json")
            or gate["final_models_manifest_sha256"] != digest(experiment / "final_models_manifest.json")
            or gate["all_final_models_verified"] != 3 * count
            or not utc(final["all_weights_frozen_at_utc"]) < utc(gate["checked_at_utc"])
                < utc(records["score/raw_access_started.json"]["at_utc"])):
        raise ValueError("The pre-access gate does not identify all frozen models before the first payload read")
    times["frozen_access_gate_verified_at_utc"] = gate["checked_at_utc"]
    times["frozen_access_gate_links_and_counts_verified"] = True
    if (list(design["models"]) != list(MODELS) or len(plan["seeds"]) != count
            or count < design["statistics"]["min_runs"] or final["models"] != 3*count
            or score["trajectories"] != 3*count or score["prediction_rows"] != 3*count*60
            or score["dates"] != 60 or score["scoring_passes_per_checkpoint"] != 1
            or selection["candidate_evaluations_per_family"] != 8
            or selection["tuning_runs_per_family"] != 16 or selection["new_fits_per_family"] != 320
            or pilot["runs"] != 30 or pilot["fits"] != 60):
        raise ValueError("The actual stage counts differ from the registered equal-budget study")
    candidates = pd.read_csv(experiment / "hpo/candidate_summary.csv")
    pilot_metrics = public_catalog(pd.read_csv(experiment / "pilot/metrics.csv"))
    final_config = public_catalog(pd.read_csv(experiment / "final/metrics.csv"))
    hpo_runs = public_catalog(pd.read_csv(experiment / "hpo/run_metrics.csv"))
    holdout_costs = public_catalog(pd.read_csv(experiment / "score/run_metrics.csv"))
    tables = {name: pd.read_csv(experiment / "analysis" / (name + ".csv")) for name in ANALYSIS_TABLES}
    if (len(candidates) != 24 or len(pilot_metrics) != 30 or len(final_config) != 3*count
            or len(tables["metric_summary"]) != 24 or len(tables["pairwise_rmse"]) != 3
            or len(tables["dm_pairs"]) != 6 or len(tables["temporal_bootstrap"]) != 18):
        raise ValueError("The aggregate export does not have all registered models/comparisons")
    baseline = selection["baseline_model"]
    analysis_summary = records["analysis/summary.json"]
    if analysis_summary["baseline_selected_before_reserve"] != baseline:
        raise ValueError("The reported baseline differs from its pre-reserve selection")
    source_manifest = {
        "scope": "Recorded SHA256 identity of private source evidence; payloads are not public exports",
        "file_sha256": {name: digest(experiment / name) for name in required},
        "registered_scientific_source_sha256": registration["sources_sha256"],
        "canonical_public_records_sha256": {
            str((canonical / name).relative_to(root)): digest(canonical / name)
            for name in ("public_preregistration.json", "closed_planning.json")},
        "public_dimensioning_receipt_sha256": digest(public_dimensioning_path) if public_dimensioning is not None else None,
        "export_adapter_sha256": digest(__file__),
        "export_adapter_is_post_execution": True,
        "scientific_implementation_is_not_retroactively_replaced": True,
    }
    summary = {
        "models": list(MODELS), "study_id": ID, "export_complete": True,
        "history_days": 7, "target_returns_days": 7, "forecast_days": 1,
        "baseline_model": baseline, "baseline_selection_uses_reserve": False,
        "reserve": {"start": design["reserve"]["start"], "end": design["reserve"]["end"], "days": 60},
        "definitive_runs_per_family": count, "definitive_seeds": plan["seeds"],
        "hpo_candidate_evaluations_per_family": 8, "hpo_tuning_seeds": design["search"]["seeds"],
        "hpo_months": 10, "hpo_dates_per_trajectory": 305, "hpo_new_fits": 960,
        "pilot_runs_per_family": 10, "pilot_new_fits": 60,
        "final_models": 3*count, "final_new_fits": 6*count,
        "reserve_prediction_rows": 3*count*60,
        "planned_delta": plan["delta"], "planning_power_target": plan["target_power"],
        "planning_sd_upper_simultaneous_confidence": plan["sd_upper_simultaneous_confidence"],
        "temporal_power_claimed": False, "not_live_forecasts": True,
        "earlier_test_remains_exploratory": True,
        "chronology_verified": True, "registration_signature": registration["registration_signature"],
        "fixed_seed_illustration": int(design["definitive"]["seed_start"]),
        "analysis_scope": analysis_summary,
    }
    runtime = {key: registration[key] for key in ("runtime", "workers") if key in registration}
    with tempfile.TemporaryDirectory(prefix="planned-study-public-") as tmp:
        staging = Path(tmp)
        documents = {
            "design.json": design, "public_preregistration.json": public_registration,
            "closed_planning.json": closed,
            "registration.json": registration, "reservation.json": records["reservation.json"],
            "selected_recipes.json": selection, "dimensioning.json": plan, "chronology.json": times,
            "source_manifest.json": source_manifest, "runtime.json": runtime,
            "data_quality_audit.json": records["score/data_audit.json"],
            "final_freeze_provenance.json": final, "summary.json": summary,
        }
        if public_dimensioning is not None:
            documents["public_dimensioning.json"] = public_dimensioning
        for name, document in documents.items():
            write_json(staging / name, public_metadata(document))
        pd.DataFrame([{**row, "model": model} for model, rows in design["search"]["candidates"].items()
                      for row in rows]).to_csv(staging / "hpo_candidates.csv", index=False)
        candidates.to_csv(staging / "hpo_summary.csv", index=False)
        hpo_runs.to_csv(staging / "hpo_run_metrics.csv", index=False)
        pilot_metrics.to_csv(staging / "pilot_metrics.csv", index=False)
        final_config.to_csv(staging / "final_configurations.csv", index=False)
        holdout_costs.to_csv(staging / "holdout_costs.csv", index=False)
        cost_summary({"hpo": hpo_runs, "pilot": pilot_metrics, "final": final_config,
                      "holdout": holdout_costs}).to_csv(staging / "cost_summary.csv", index=False)
        for name, frame in tables.items():
            frame.to_csv(staging / (name + ".csv"), index=False)
        predictions = pd.read_csv(experiment / "score/predictions.csv")
        _plot(staging, candidates, pilot_metrics, tables["metric_summary"], predictions,
              summary["fixed_seed_illustration"])
        lines = ["# Fase principal D7 planificada: GRU–LSTM–TCN", "",
            "Esta fase se registró antes de su HPO, piloto, dimensionamiento y evaluación sobre una reserva histórica adicional. "
            "El primer benchmark conserva sus resultados y alcance exploratorio. La reserva nueva no constituye otro mercado ni predicción en vivo.", "",
            "| Elemento | Evidencia ejecutada |", "|---|---|",
            "| HPO | Ocho candidatos y dos semillas por familia; diez meses; 960 ajustes nuevos |",
            "| Piloto | Diez semillas nuevas por receta; 60 días de desarrollo; 60 ajustes nuevos |",
            f"| Dimensionamiento | R={count} por familia; potencia marginal objetivo0,80 bajo diferencias normales y DE conservadora |",
            f"| Ajuste definitivo | {3*count} modelos nuevos; {6*count} ajustes inner/refit; pesos congelados antes del acceso de precios |",
            f"| Reserva | 01/09–30/10/2026; 60 fechas; {3*count*60} predicciones |",
            f"| Baseline fijado con HPO | {baseline}/D7; no se reelige con los resultados de reserva |", "",
            "`chronology.json` verifica el orden real de registros. `dimensioning.json` conserva DE, incertidumbre, "
            "potencia de planificación y sensibilidades. Esa potencia corresponde a aleatoriedad de entrenamiento condicionada al calendario; no acredita potencia temporal80 %.", "",
            "Las tres comparaciones primarias usan RMSE pareada por semilla y Holm3. DM/HAC6/HLN y el bootstrap temporal "
            "de bloques7, con sensibilidades3/14, son análisis secundarios con seis pares fijos. Significación frente a cero "
            "y una mejora práctica superior al5 % son afirmaciones distintas.", "",
            "Los archivos públicos contienen sólo agregados, metadatos y figuras. `predictions_fixed_seed.png` "
            f"usa la semilla {summary['fixed_seed_illustration']} fijada antes del test; no se entrega su CSV individual. "
            "No se publican brutos, pesos, escaladores ni vectores diarios. La adquisición había inspeccionado "
            "timestamps/esquema de los archivos; no se afirma que los ZIP nunca se hubieran abierto.", "",
            "El adaptador de exportación es posterior a la ejecución; las firmas del código científico registrado permanecen separadas.", ""]
        (staging / "README.md").write_text("\n".join(lines))
        validate_payloads(staging)
        write_json(staging / "public_allowlist_manifest.json", {
            "study_id": ID, "file_sha256": {p.name: digest(p) for p in sorted(staging.iterdir())},
            "scope": "Aggregate results, configuration/hash metadata and declared fixed-seed PNG only",
            "no_raw_vectors_weights_or_scalers": True,
            "export_complete": True,
        })
        validate_payloads(staging)
        publish_identical(staging, (root / "results" / ID, root / "book/_static" / ID))
    print(json.dumps({"public_output": str(root / "results" / ID), "files": len(list((root / "results" / ID).iterdir())),
                      "R": count, "chronology": "PASS", "new_fits_by_export": 0, "new_forecasts_by_export": 0}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    export_study(args.experiment, args.project_root)


if __name__ == "__main__":
    main()
