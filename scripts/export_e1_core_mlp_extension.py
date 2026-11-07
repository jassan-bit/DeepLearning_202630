"""Export closed public aggregates from the new GRU/LSTM/MLP execution.

The private run must finish tuning, pilot, published planning, final fitting,
scoring and analysis. This command never fits, scores, loads weights or opens
market archives. Only loss curves, metric aggregates and projected provenance
can enter its explicit public allowlist.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = "e1_core_mlp_extension_20261107"
DESIGN = f"experiments/{STUDY}/design.json"
MODELS = ("GRU", "LSTM", "MLP")
METRICS = ("rmse", "mae", "mse", "r2", "mape", "mase")
ANALYSIS_NAMES = ("metric_summary", "run_metrics", "pairwise_rmse", "dm_pairs",
                  "temporal_bootstrap", "diagnostics")
CURVE_COLUMNS = {"epoch", "stage", "optimization_mse_scaled", "train_mse_scaled_eval",
                 "validation_mse_scaled_eval", "best_epoch_so_far", "learning_rate"}
LOSS_COLUMNS = ("optimization_mse_scaled", "train_mse_scaled_eval", "validation_mse_scaled_eval")
PUBLIC_FILES = frozenset({
    "README.md", "summary.json", "registration.json", "source_manifest.json",
    "selection.json", "plan.json", "plan_publication.json", "chronology.json", "runtime.json",
    "hpo_summary.csv", "hpo_run_metrics.csv", "pilot_metrics.csv", "final_fit_summary.csv",
    "architecture_summary.csv", "learning_curves_by_run.csv", "learning_curve_summary.csv",
    "learning_curves.png", "metric_summary.csv", "run_metrics.csv", "pairwise_rmse.csv",
    "dm_pairs.csv", "temporal_bootstrap.csv", "diagnostics.csv", "analysis_summary.json",
})
PRIVATE_FIELDS = {"y_true", "y_pred", "persistence_pred", "outcome_day", "observed", "prediction",
                  "predictions", "state_dict", "weights", "scaler", "scalers", "preprocessing",
                  "mase_scale", "mase_train_scale", "y_scale", "x_mean", "x_scale", "raw_prices"}
JOB_FIELDS = ("study_id", "execution_id", "model", "seed", "trajectory_seed", "candidate_id", "width",
              "learning_rate", "dropout", "history_days", "parameters", "selected_epoch",
              "train_samples", "inner_train_samples", "inner_validation_samples", "training_seconds",
              "inference_seconds", "fit_origin_day_utc", "evaluation_days", "n_dates", *METRICS)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def moment(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed > datetime.now(timezone.utc):
        raise ValueError("Observed chronology needs an offset and must not be in the future")
    return parsed


def regular(path, boundary):
    path, boundary = Path(path), Path(boundary).resolve()
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(boundary):
        raise ValueError("Only regular files inside the declared input boundary are allowed")
    return path


def verify_mapping(private, mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("A nonempty completion artifact mapping is required")
    for name, expected in mapping.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or rel.as_posix() != name:
            raise ValueError("Unsafe artifact path")
        if sha(regular(private / name, private)) != expected:
            raise ValueError("Closed private artifact changed: " + name)


def identity(frame, signature):
    if not {"study_id", "execution_id"} <= set(frame):
        raise ValueError("Every new run needs its own study and execution identity")
    if not frame.study_id.eq(STUDY).all() or not frame.execution_id.eq(signature).all():
        raise ValueError("Mixed historical or unrelated execution rows are forbidden")


def panel(frame, seeds, *, seed_column="seed", persistence=False):
    required = {"model", seed_column}
    if not required <= set(frame):
        raise ValueError("Missing family/seed panel columns")
    neural = frame.loc[frame.model.isin(MODELS)]
    allowed = set(MODELS) | ({"Persistence"} if persistence else set())
    if set(frame.model) != allowed or neural.duplicated(["model", seed_column]).any():
        raise ValueError("Expected the exact new core family panel; TCN belongs to another study")
    expected = {(model, int(seed)) for model in MODELS for seed in seeds}
    actual = {(str(row.model), int(getattr(row, seed_column))) for row in neural.itertuples()}
    if actual != expected or len(neural) != len(expected):
        raise ValueError("Incomplete or duplicated new core family/seed panel")
    if persistence and len(frame.loc[frame.model.eq("Persistence")]) != 1:
        raise ValueError("Persistence is one deterministic reference")


def public_frame(frame):
    if set(frame) & PRIVATE_FIELDS or any("prediction" in str(name).lower() for name in frame):
        raise ValueError("Observations, daily forecasts, weights and scalers cannot be exported")
    return frame


def projected_runs(frame):
    names = [name for name in JOB_FIELDS if name in frame]
    return public_frame(frame.loc[:, names].copy())


def projected_record(value, private_path):
    """Keep observational receipts, not arbitrary API responses/contact fields.

    The execution signature remains its identity. If projected, it signs the
    original private record, whose file hash is given separately; it is never
    represented as a signature of these edited public bytes.
    """
    result = json.loads(json.dumps(value))
    if "publication_verification" in result:
        allowed = {"sha", "source_commit", "observed_at_utc", "url", "commit_url", "method",
                   "verification_method", "observer", "file_sha256", "limitation"}
        receipt = result["publication_verification"]
        result["publication_verification"] = {key: receipt[key] for key in sorted(allowed & set(receipt))}
        if result["publication_verification"] != receipt:
            result["public_projection"] = True
            result["original_private_record_sha256"] = sha(private_path)
            result["signature_scope"] = "Execution identity/signature belongs to the original private record, not this public projection"
    return result


def validate_curve(curve, stage, selected):
    if set(curve) != CURVE_COLUMNS or not len(curve) or not curve.stage.eq(stage).all():
        raise ValueError("Unexpected loss curve schema or fitting stage")
    if curve.epoch.tolist() != list(range(1, len(curve) + 1)) or not 1 <= selected <= len(curve) <= 100:
        raise ValueError("Invalid recorded loss epoch sequence")
    for name in LOSS_COLUMNS[:2]:
        if not np.isfinite(curve[name]).all() or curve[name].lt(0).any():
            raise ValueError("Training curves require finite nonnegative observed losses")
    if stage == "inner":
        if (not np.isfinite(curve.validation_mse_scaled_eval).all()
                or curve.validation_mse_scaled_eval.lt(0).any()
                or int(curve.best_epoch_so_far.iloc[-1]) != selected):
            raise ValueError("Invalid observed internal validation or selected epoch")
    elif len(curve) != selected or not curve.validation_mse_scaled_eval.isna().all():
        raise ValueError("Final refit has no reserve validation losses")


def verify_chronology(record, selection, pilot, plan, publication, final_start, final, gate, score, analysis, workers):
    source_receipt = record["publication_verification"]
    if source_receipt.get("sha", source_receipt.get("source_commit")) != record["source_commit"]:
        raise ValueError("The source publication receipt has another commit")
    if (publication["publication_verification"].get("file_sha256") != publication["plan_sha256"]
            or publication["publication_verification"].get("sha", publication["publication_verification"].get("source_commit")) != publication["published_plan_commit"]):
        raise ValueError("The planning receipt does not bind the exact published plan")
    points = {
        "source_publication_observed": source_receipt["observed_at_utc"],
        "registration": record["registered_at_utc"],
        "hpo_first_worker": min(x["runtime"]["at_utc"] for x in workers if x["phase"] == "hpo"),
        "selection": selection["selected_at_utc"],
        "pilot_first_worker": min(x["runtime"]["at_utc"] for x in workers if x["phase"] == "pilot"),
        "pilot_complete": pilot["completed_at_utc"],
        "planning": plan["planned_at_utc"],
        "plan_publication_observed": publication["publication_verification"]["observed_at_utc"],
        "plan_publication_registration": publication["registered_at_utc"],
        "final_started": final_start["at_utc"],
        "final_first_worker": min(x["runtime"]["at_utc"] for x in workers if x["phase"] == "final"),
        "weights_frozen": final["all_weights_frozen_at_utc"],
        "score_gate": gate["checked_at_utc"],
        "score_complete": score["completed_at_utc"],
        "analysis": analysis["analyzed_at_utc"],
    }
    ordered = [moment(value) for value in points.values()]
    if any(a > b for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Publication, pilot, planning, frozen weights and scoring chronology is inconsistent")
    for phase, lower, upper in (("hpo", "registration", "selection"),
            ("pilot", "selection", "pilot_complete"), ("final", "final_started", "weights_frozen")):
        times = [moment(row["runtime"]["at_utc"]) for row in workers if row["phase"] == phase]
        if len(times) != 3 or any(not moment(points[lower]) <= value <= moment(points[upper]) for value in times):
            raise ValueError("Each registered phase requires three workers inside its recorded chronology")
    return points


def verify(source, private):
    source, private = Path(source).resolve(), Path(private).resolve()
    if source != ROOT.resolve() or private.is_relative_to(source):
        raise ValueError("Use the immutable exporter source and an external private run")
    record = read(regular(private / "registration.json", private))
    unsigned = {key: value for key, value in record.items() if key != "registration_signature"}
    signature = hashlib.sha256(encode(unsigned)).hexdigest()
    if signature != record["registration_signature"] or record["study_id"] != STUDY:
        raise ValueError("The new study registration changed")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    if revision != record["source_commit"] or dirty:
        raise ValueError("Export from the clean registered source commit")
    design = read(source / DESIGN)
    if sha(source / DESIGN) != record["design_sha256"] or design != record["design"]:
        raise ValueError("The frozen design is inconsistent")
    if design["models"] != list(MODELS) or design["history_days"] != 7:
        raise ValueError("The design must contain exactly GRU/LSTM/MLP and D7")
    for name, expected in record["sources_sha256"].items():
        if sha(regular(source / name, source)) != expected:
            raise ValueError("A registered scientific source changed: " + name)
    paths = {"selection": "hpo/selection.json", "pilot": "pilot/complete.json", "plan": "plan.json",
             "publication": "plan_publication.json", "final_start": "final_started.json",
             "final": "final_models_manifest.json", "gate": "score/frozen_access_gate.json",
             "score": "score/complete.json", "analysis": "analysis/manifest.json"}
    records = {key: read(regular(private / name, private)) for key, name in paths.items()}
    for key in ("selection", "pilot", "plan", "publication", "final_start", "final", "score", "analysis"):
        if records[key].get("registration_signature") != signature:
            raise ValueError("A completion record belongs to another execution: " + key)
    for key in ("selection", "pilot", "final", "score", "analysis"):
        verify_mapping(private, records[key]["artifact_sha256"])
    selection, pilot, plan, publication, start, final, gate, score, analysis = (records[name] for name in paths)
    public_unsigned = {key: value for key, value in publication.items() if key != "receipt_signature"}
    if hashlib.sha256(encode(public_unsigned)).hexdigest() != publication.get("receipt_signature"):
        raise ValueError("The observed planning-publication receipt changed")
    if (selection["trajectories"] != 48 or selection["fit_stages"] != 960
            or set(selection["recipes"]) != set(MODELS) or selection["baseline_model"] not in MODELS
            or pilot["trajectories"] != 30 or pilot["fit_stages"] != 60):
        raise ValueError("Tuning and pilot panels must be fully complete for the three new families")
    r = int(plan["R"])
    if r < 465 or r != int(plan["R_final"]) or r < int(plan["R_required"]) or r > 10000:
        raise ValueError("The complete pilot-derived plan and minimum465 are required")
    if plan["seeds"] != list(range(8101, 8101 + r)):
        raise ValueError("Unexpected new definitive seed range")
    checks = ((pilot, "selection_sha256", paths["selection"]), (plan, "pilot_complete_sha256", paths["pilot"]),
              (plan, "selection_sha256", paths["selection"]), (publication, "plan_sha256", paths["plan"]),
              (start, "plan_sha256", paths["plan"]), (start, "plan_publication_sha256", paths["publication"]),
              (final, "plan_sha256", paths["plan"]), (final, "selection_sha256", paths["selection"]),
              (gate, "plan_sha256", paths["plan"]), (gate, "final_models_manifest_sha256", paths["final"]),
              (score, "final_models_manifest_sha256", paths["final"]), (analysis, "score_complete_sha256", paths["score"]))
    if any(row[key] != sha(private / name) for row, key, name in checks):
        raise ValueError("A stage is not bound to the preceding frozen stage")
    if (record["new_fits_before_registration"] != 0 or publication["new_definitive_fits_before_plan_publication"] != 0
            or any(row.get("reserve_price_payloads_read") != 0 for row in (record, selection, pilot, plan, final))):
        raise ValueError("Reserve access or fitting preceded the registered gates")
    if (final["models"] != 3*r or final["fit_stages"] != 6*r or final["R"] != r
            or start["planned_trajectories"] != 3*r or gate["all_final_models_verified"] != 3*r
            or score["trajectories"] != 3*r or score["dates"] != 60 or score["prediction_rows"] != 180*r
            or score["scoring_passes_per_checkpoint"] != 1 or not score["historical_exploratory_not_confirmatory"]):
        raise ValueError("Require a closed complete fit and exactly one known60-date scoring pass")
    jobs = pd.DataFrame(final["jobs"])
    identity(jobs, signature)
    panel(jobs, plan["seeds"])
    workers = [read(path) for path in sorted((private / "worker_runtime").glob("*.json"))]
    if set(row["phase"] for row in workers) != {"hpo", "pilot", "final"}:
        raise ValueError("All three independent worker stages need observed runtime records")
    for worker in workers:
        actual = worker["runtime"]
        if (worker["start_method"] != "spawn" or actual["actual_threads"] != 1
                or actual["actual_interop_threads"] != 1 or not actual["deterministic_algorithms"]):
            raise ValueError("Observed workers differ from the registered numerical controls")
    chronology = verify_chronology(record, selection, pilot, plan, publication, start, final, gate, score, analysis, workers)
    return record, records, jobs, workers, chronology


def destination(path, source, private):
    if Path(path).is_symlink():
        raise ValueError("A public destination cannot be a symlink")
    path = Path(path).resolve()
    if (path.is_relative_to(source) or path.is_relative_to(private) or private.is_relative_to(path)
            or source.is_relative_to(path) or path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise ValueError("Public output must be a new empty external directory, separate from private data")
    return path


def aggregate_curves(curves):
    rows = []
    for (model, stage, epoch), group in curves.groupby(["model", "stage", "epoch"], sort=True):
        for name in LOSS_COLUMNS:
            values = group[name].dropna().to_numpy(float)
            if len(values):
                rows.append({"model": model, "stage": stage, "epoch": int(epoch), "metric": name,
                    "n_available": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
                    "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75))})
    return pd.DataFrame(rows)


def plot_curves(aggregated, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e1-core-mlp-export-plots"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for index, model in enumerate(MODELS):
        for column, stage in enumerate(("inner", "refit")):
            axis = axes[index, column]
            for metric, label, color in (("train_mse_scaled_eval", "Entrenamiento (evaluación)", "#2764a5"),
                    ("validation_mse_scaled_eval", "Validación interna", "#df7922")):
                rows = aggregated[(aggregated.model == model) & (aggregated.stage == stage) & (aggregated.metric == metric)]
                if rows.empty:
                    continue
                axis.plot(rows.epoch, rows["mean"], label=label, color=color)
                axis.fill_between(rows.epoch, rows.q25, rows.q75, alpha=.18, color=color)
            axis.set(title=f"{model} · {'Selección interna' if stage == 'inner' else 'Reajuste'}",
                     xlabel="Época", ylabel="MSE escalada de desarrollo")
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle("Nuevas recetas GRU–LSTM–MLP: media e intervalo intercuartílico; sin validación de reserva")
    fig.savefig(output / "learning_curves.png", dpi=150)
    plt.close(fig)


def export(source, private, output, mirror=None):
    source, private = Path(source).resolve(), Path(private).resolve()
    output = destination(output, source, private)
    mirror = destination(mirror, source, private) if mirror is not None else None
    if mirror is not None and (mirror == output or mirror.is_relative_to(output) or output.is_relative_to(mirror)):
        raise ValueError("Public output and mirror must be separate")
    record, records, jobs, workers, chronology = verify(source, private)
    signature, plan = record["registration_signature"], records["plan"]
    frames = {}
    for name in ANALYSIS_NAMES:
        frames[name] = public_frame(pd.read_csv(private / "analysis" / (name + ".csv")))
    identity(frames["run_metrics"], signature)
    panel(frames["run_metrics"], plan["seeds"], persistence=True)
    summary = read(private / "analysis" / "summary.json")
    if (summary["study_id"] != STUDY or summary["execution_id"] != signature or summary["R_final"] != plan["R"]
            or summary["models"] != list(MODELS) or not summary["known_reserve_exploratory"]
            or summary["fresh_confirmatory_holdout"] or summary["selection_using_holdout"]
            or summary["dates_per_trajectory"] != 60 or summary["baseline_selected_before_reserve"] != records["selection"]["baseline_model"]):
        raise ValueError("The public analysis must preserve the new known-reserve scope and frozen baseline")
    hpo_runs = pd.read_csv(private / "hpo" / "run_metrics.csv")
    identity(hpo_runs, signature)
    hpo_summary = public_frame(pd.read_csv(private / "hpo" / "candidate_summary.csv"))
    if len(hpo_runs) != 48 or len(hpo_summary) != 24 or set(hpo_summary.model) != set(MODELS):
        raise ValueError("Missing new tuning runs or candidate summaries")
    for model in MODELS:
        rows = hpo_runs.loc[hpo_runs.model.eq(model)]
        if (set(rows.candidate_id) != {f"c{x:02d}" for x in range(8)}
                or rows.duplicated(["candidate_id", "trajectory_seed"]).any()
                or any(set(group.trajectory_seed) != {6101, 6102} for _, group in rows.groupby("candidate_id"))):
            raise ValueError("Every new tuning candidate requires both registered seeds")
        candidates = hpo_summary.loc[hpo_summary.model.eq(model)]
        if len(candidates) != 8 or candidates.candidate_id.duplicated().any() or set(candidates.candidate_id) != {f"c{x:02d}" for x in range(8)}:
            raise ValueError("Candidate summaries require eight distinct configurations per family")
    pilot = pd.read_csv(private / "pilot" / "metrics.csv")
    identity(pilot, signature)
    panel(pilot, range(7101, 7111))
    fit = pd.read_csv(private / "final" / "metrics.csv")
    identity(fit, signature)
    panel(fit, plan["seeds"])
    curves = []
    for row in jobs.itertuples():
        directory = private / "final" / f"{row.model}_s{int(row.seed)}"
        for stage in ("inner", "refit"):
            frame = pd.read_csv(regular(directory / f"history_{stage}.csv", private))
            validate_curve(frame, stage, int(row.selected_epoch))
            frame.insert(0, "seed", int(row.seed))
            frame.insert(0, "model", row.model)
            curves.append(frame)
    curves = pd.concat(curves, ignore_index=True)
    aggregated = aggregate_curves(curves)
    architecture = projected_runs(jobs).drop(columns=[x for x in ("seed", "trajectory_seed", "training_seconds", "inference_seconds", "selected_epoch") if x in jobs]).drop_duplicates()
    # One recipe per family; varying epochs or seeds are not new architectures.
    recipe_columns = [name for name in ("model", "candidate_id", "width", "learning_rate", "dropout", "history_days", "parameters") if name in architecture]
    architecture = architecture.loc[:, recipe_columns].drop_duplicates().sort_values("model")
    if len(architecture) != 3:
        raise ValueError("Every family must retain one development-frozen final recipe")
    for row in architecture.itertuples():
        width = int(row.width)
        expected = ((3 if row.model == "GRU" else 4)*width*(5+width+2)+width+1
                    if row.model != "MLP" else width*(7*5+2)+1)
        recipe = records["selection"]["recipes"][row.model]
        if int(row.parameters) != expected or any(getattr(row, name) != recipe[name] for name in
                ("candidate_id", "width", "learning_rate", "dropout", "parameters")):
            raise ValueError("Final architecture differs from its development-frozen recipe or parameter count")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e1-core-mlp-public-", dir=output.parent) as temp:
        stage = Path(temp)
        for name, frame in {**frames, "hpo_summary": hpo_summary, "hpo_run_metrics": projected_runs(hpo_runs),
                "pilot_metrics": projected_runs(pilot), "final_fit_summary": projected_runs(fit),
                "architecture_summary": architecture, "learning_curves_by_run": curves,
                "learning_curve_summary": aggregated}.items():
            public_frame(frame).to_csv(stage / (name + ".csv"), index=False)
        metadata = {"registration.json": projected_record(record, private / "registration.json"),
            "selection.json": records["selection"], "plan.json": plan,
            "plan_publication.json": projected_record(records["publication"], private / "plan_publication.json"), "chronology.json": chronology,
            "runtime.json": {"registration": record["runtime"], "workers": workers}, "analysis_summary.json": summary,
            "source_manifest.json": {"source_commit": record["source_commit"], "design_sha256": record["design_sha256"],
                "sources_sha256": record["sources_sha256"], "private_completion_record_sha256": {
                    key: sha(private / name) for key, name in {"hpo_selection": "hpo/selection.json", "pilot": "pilot/complete.json",
                    "plan": "plan.json", "plan_publication": "plan_publication.json", "final": "final_models_manifest.json",
                    "score": "score/complete.json", "analysis": "analysis/manifest.json"}.items()}}}
        for name, value in metadata.items():
            (stage / name).write_bytes(encode(value))
        report = {**summary, "exported_at_utc": stamp(), "source_commit": record["source_commit"],
            "hpo_trajectories": 48, "hpo_fit_stages": 960, "pilot_trajectories": 30, "pilot_fit_stages": 60,
            "definitive_trajectories": len(jobs), "definitive_fit_stages": 2*len(jobs), "curve_rows": len(curves),
            "core_models": list(MODELS), "complementary_TCN_included_in_new_core": False,
            "source_publication_precedes_new_fits": True, "plan_publication_precedes_definitive_fits": True,
            "scoring_after_all_new_weights_frozen": True, "original_scientific_results_modified": False,
            "exporter_model_fits": 0, "exporter_checkpoint_forecasts": 0, "exporter_raw_market_archive_reads": 0,
            "private_payloads_exported": False,
            "curve_scope": "Observed development2023–2024 losses. Refit has no reserve validation; late epochs have fewer surviving runs.",
            "registration_reservation_note": "Original acquisition/reservation metadata are inherited byte-for-byte; this new execution explicitly evaluates an already observed historical calendar."}
        (stage / "summary.json").write_bytes(encode(report))
        (stage / "README.md").write_text(
            "# Nueva comparación histórica GRU–LSTM–MLP\n\n"
            "Se publican tres familias ejecutadas desde cero bajo el mismo diseño: HPO, piloto, planificación, ajuste y evaluación. "
            "El baseline procede de la validación de desarrollo y quedó congelado antes del piloto. TCN se conserva en el estudio complementario anterior.\n\n"
            "La reserva de septiembre–octubre de2026 ya era conocida. Esta ejecución es exploratoria y no recupera independencia confirmatoria. "
            "Las semillas comparten60fechas: no son mercados ni calendarios independientes. El R calculado condiciona la incertidumbre al calendario.\n\n"
            "Los CSV de curvas contienen sólo pérdidas de desarrollo. La selección interna muestra entrenamiento y validación; el reajuste sólo entrenamiento. "
            "La figura usa media e intervalo intercuartílico. Las épocas tardías reúnen menos trayectorias, según n_available.\n\n"
            "No se redistribuyen precios, objetivos o pronósticos diarios, pesos, valores de escaladores o archivos brutos. "
            "Los hashes verifican el cierre de los artefactos privados sin hacerlos públicos. La publicación observada precede a los ajustes y "
            "el plan derivado del piloto se publicó antes del ajuste definitivo. Los resultados originales permanecen separados.\n", encoding="utf-8")
        plot_curves(aggregated, stage)
        if {path.name for path in stage.iterdir()} != PUBLIC_FILES:
            raise ValueError("Public output differs from the closed allowlist")
        mapping = {name: {"sha256": sha(stage / name), "bytes": (stage / name).stat().st_size} for name in sorted(PUBLIC_FILES)}
        manifest = {"publication_schema": "e1_public_metadata_allowlist_v1", "approved_public_files": sorted(PUBLIC_FILES),
                    "files": mapping, "source_commit": record["source_commit"], "private_payloads_exported": False}
        (stage / "manifest.json").write_bytes(encode(manifest))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, output, dirs_exist_ok=True)
    if mirror is not None:
        shutil.copytree(output, mirror, dirs_exist_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path)
    args = parser.parse_args()
    result = export(args.source_root, args.private_dir, args.output_dir, args.mirror_dir)
    print(json.dumps({key: result[key] for key in ("source_commit", "R_final", "definitive_trajectories", "curve_rows", "known_reserve_exploratory")}, indent=2))


if __name__ == "__main__":
    main()
