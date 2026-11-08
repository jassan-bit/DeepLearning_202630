"""Export closed public aggregates from the new GRU/LSTM/MLP execution.

The private run must finish tuning, pilot, published planning, final fitting,
scoring and analysis. This command never fits, scores, loads weights or opens
market archives. Only loss curves, metric aggregates and projected provenance
can enter its explicit public allowlist.
"""
from __future__ import annotations

import argparse
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
STUDY = "e1_608_remediation_20261107"
DESIGN = f"experiments/{STUDY}/design.json"
MODELS = ("GRU", "LSTM", "MLP")
METRICS = ("rmse", "mae", "mse", "r2", "mape", "mase")
ANALYSIS_NAMES = ("metric_summary", "run_metrics", "pairwise_rmse", "dm_pairs",
                  "temporal_bootstrap", "diagnostics", "friedman_rmse", "wilcoxon_rmse", "qq_data")
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
    "friedman_rmse.csv", "wilcoxon_rmse.csv", "qq_data.csv", "qq_rmse_differences.png",
    "first_seed_learning_curves.csv", "first_seed_learning_curves.png", "run_costs.csv",
    "cost_summary.csv", "pilot_summary.csv",
})
PRIVATE_FIELDS = {"y_true", "y_pred", "persistence_pred", "outcome_day", "observed", "prediction",
                  "predictions", "state_dict", "weights", "scaler", "scalers", "preprocessing",
                  "mase_scale", "mase_train_scale", "y_scale", "x_mean", "x_scale", "raw_prices"}
JOB_FIELDS = ("study_id", "execution_id", "model", "seed", "trajectory_seed", "candidate_id", "width",
              "learning_rate", "dropout", "history_days", "parameters", "selected_epoch",
              "train_samples", "inner_train_samples", "inner_validation_samples", "training_seconds",
              "inference_seconds", "fit_origin_day_utc", "evaluation_days", "n_dates", *METRICS)
JOB_FIELDS += ("source_signature", "source_commit", "design_sha256")
# Metadata columns are closed independently of file names. Unrecognized columns
# never become public merely because their names do not resemble a private key.
ANALYSIS_FIELDS = frozenset(JOB_FIELDS) | frozenset("""
metric n_total n_defined mean sample_sd ci95_lower ci95_upper stochastic_training ci_method scope
model_a model_b n_paired mean_a mean_b difference_a_minus_b difference_sample_sd
ci_family_lower ci_family_upper statistic degrees_of_freedom p_value_unadjusted status cohen_d_z
effect_ci95_lower effect_ci95_upper effect_bootstrap_n effect_bootstrap_defined
gain_a_over_b_absolute gain_a_over_b_relative_percent_point absolute_planning_delta
gain_a_over_b_family_ci_lower gain_a_over_b_family_ci_upper
a_improvement_family_ci_exceeds_absolute_delta b_improvement_family_ci_exceeds_absolute_delta
shapiro_W shapiro_p normality_diagnostic_changes_test direction p_value_holm3 reject_holm3
multiplicity_family_size n_obs n horizon hac_lags mean_loss_difference long_run_variance
variance_hac hln_correction p_value p_value_holm6 reject_holm6 limitations loss_estimand
baseline_selected_before_reserve block_length role draws bootstrap_seed
relative_gain_a_over_b_percent relative_gain_ci95_lower relative_gain_ci95_upper
relative_gain_family_lower relative_gain_family_upper a_relative_gain_family_lower_exceeds_5percent
relative_gain_b_over_a_percent relative_gain_b_over_a_family_lower relative_gain_b_over_a_family_upper
b_relative_gain_family_lower_exceeds_5percent estimand n_residuals diagnostic lag epsilon model_df
p_value_holm_diagnostic_family reject_holm kendall_W n_models test zero_method method
order_index theoretical_normal_quantile ordered_difference_a_minus_b ordered_standardized_difference
MDE_at_actual_R_point_sd
hac_kernel mean_squared_loss_a mean_squared_loss_b mean_loss_a_minus_b
variance_of_mean_loss_difference dm_raw_statistic hln_correction_factor dm_hln_statistic
continuity_correction null_assumption
mape_defined_dates mde_scope
""".split())


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


def bound_file(private, mapping, name):
    """An aggregate must be covered by its immutable phase completion hash."""
    if name not in mapping:
        raise ValueError("Required public input is absent from the closed artifact mapping: " + name)
    return regular(private / name, private)


def identity(frame, signature, source_signature=None):
    if not {"study_id", "execution_id"} <= set(frame):
        raise ValueError("Every new run needs its own study and execution identity")
    if not frame.study_id.eq(STUDY).all() or not frame.execution_id.eq(signature).all():
        raise ValueError("Mixed historical or unrelated execution rows are forbidden")
    if source_signature is not None and ("source_signature" not in frame
            or not frame.source_signature.eq(source_signature).all()):
        raise ValueError("Mixed or missing registered scientific source identity")


def panel(frame, seeds, *, seed_column="seed", persistence=False):
    required = {"model", seed_column}
    if not required <= set(frame):
        raise ValueError("Missing family/seed panel columns")
    neural = frame.loc[frame.model.isin(MODELS)]
    values = pd.to_numeric(neural[seed_column], errors="coerce")
    if not np.isfinite(values).all() or not (values % 1).eq(0).all():
        raise ValueError("Registered seed identifiers must be finite integers")
    allowed = set(MODELS) | ({"Persistence"} if persistence else set())
    if set(frame.model) != allowed or neural.duplicated(["model", seed_column]).any():
        raise ValueError("Expected the exact new core family panel; TCN belongs to another study")
    expected = {(model, int(seed)) for model in MODELS for seed in seeds}
    actual = {(str(row.model), int(getattr(row, seed_column))) for row in neural.itertuples()}
    if actual != expected or len(neural) != len(expected):
        raise ValueError("Incomplete or duplicated new core family/seed panel")
    if persistence and len(frame.loc[frame.model.eq("Persistence")]) != 1:
        raise ValueError("Persistence is one deterministic reference")


def public_frame(frame, allowed=None):
    if (set(frame) & PRIVATE_FIELDS
            or any(any(part in str(name).lower() for part in ("prediction", "scaler", "weights", "raw_price", "y_true", "y_pred")) for name in frame)):
        raise ValueError("Observations, daily forecasts, weights and scalers cannot be exported")
    if allowed is not None and not set(frame) <= set(allowed):
        raise ValueError("Public columns differ from the closed metadata allowlist: " + str(sorted(set(frame) - set(allowed))))
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
                   "verification_method", "observer", "verified_by", "file_sha256", "limitation"}
        receipt = result["publication_verification"]
        result["publication_verification"] = {key: receipt[key] for key in sorted(allowed & set(receipt))}
        if result["publication_verification"] != receipt:
            result["public_projection"] = True
            result["original_private_record_sha256"] = sha(private_path)
            result["signature_scope"] = "Execution identity/signature belongs to the original private record, not this public projection"
    return result


def no_private_metadata(value):
    """Reject numeric model/data payloads even when nested inside a JSON record."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in PRIVATE_FIELDS:
                raise ValueError("Private payload field in public metadata: " + key)
            no_private_metadata(child)
    elif isinstance(value, list):
        for child in value:
            no_private_metadata(child)
    return value


def verify_support(score, r):
    if (score["trajectories"] != 3*r or score["dates"] != 608
            or score["prediction_rows"] != 1824*r
            or score["scoring_passes_per_checkpoint"] != 1
            or score["historical_exploratory_not_confirmatory"] is not True):
        raise ValueError("Require exactly one complete known608-date scoring pass for every new checkpoint")


def verify_timeline(points):
    ordered = [moment(value) for value in points.values()]
    if any(a > b for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Publication, pilot, planning, frozen weights and scoring chronology is inconsistent")
    return points


def verify_plan(plan):
    r = plan["R"]
    if (isinstance(r, bool) or int(r) != r or r != max(30, plan["R_required"])
            or r != plan["R_final"] or not 30 <= r <= 10000
            or plan["seeds"] != list(range(18101, 18101 + int(r)))):
        raise ValueError("The pilot-derived plan requires R=max(30,R_required) and the exact new definitive seeds")
    return int(r)


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


def verify_chronology(record, selection, pilot, plan, publication, final_start, final, gate, score, analysis, workers, raw_access=None):
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
        **({"first_raw_price_access": raw_access["at_utc"]} if raw_access is not None else {}),
        "score_complete": score["completed_at_utc"],
        "analysis": analysis["analyzed_at_utc"],
    }
    verify_timeline(points)
    for phase, lower, upper in (("hpo", "registration", "selection"),
            ("pilot", "selection", "pilot_complete"), ("final", "final_started", "weights_frozen")):
        times = [moment(row["runtime"]["at_utc"]) for row in workers if row["phase"] == phase]
        if len(times) < 3 or any(not moment(points[lower]) <= value <= moment(points[upper]) for value in times):
            raise ValueError("Each registered phase requires at least three workers inside its recorded chronology")
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
    source_signature = hashlib.sha256(encode({"sources_sha256": record["sources_sha256"],
        "design_sha256": record["design_sha256"], "source_commit": record["source_commit"]})).hexdigest()
    if record.get("source_signature") != source_signature:
        raise ValueError("Registered source identity does not bind its scientific source mapping")
    public_unsigned = {key: value for key, value in publication.items() if key != "receipt_signature"}
    if hashlib.sha256(encode(public_unsigned)).hexdigest() != publication.get("receipt_signature"):
        raise ValueError("The observed planning-publication receipt changed")
    if (selection["trajectories"] != 24 or selection["fit_stages"] != 480
            or set(selection["recipes"]) != set(MODELS) or selection["baseline_model"] not in MODELS
            or pilot["trajectories"] != 30 or pilot["fit_stages"] != 60):
        raise ValueError("Tuning and pilot panels must be fully complete for the three new families")
    r = verify_plan(plan)
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
            or score["trajectories"] != 3*r):
        raise ValueError("Require a closed complete fit and exactly one known608-date scoring pass")
    verify_support(score, r)
    jobs = pd.DataFrame(final["jobs"])
    identity(jobs, signature, source_signature)
    panel(jobs, plan["seeds"])
    workers = [read(path) for path in sorted((private / "worker_runtime").glob("*.json"))]
    if set(row["phase"] for row in workers) != {"hpo", "pilot", "final"}:
        raise ValueError("All three independent worker stages need observed runtime records")
    for worker in workers:
        actual = worker["runtime"]
        if (worker["start_method"] != "spawn" or actual["actual_threads"] != 1
                or actual["actual_interop_threads"] != 1 or not actual["deterministic_algorithms"]):
            raise ValueError("Observed workers differ from the registered numerical controls")
    raw = read(bound_file(private, score["artifact_sha256"], "score/raw_access_started.json"))
    bound_file(private, score["artifact_sha256"], paths["gate"])
    for access in (gate, raw):
        if (access["registration_signature"] != signature or access["source_signature"] != source_signature
                or access["all_final_models_verified"] != 3*r
                or access["final_models_manifest_sha256"] != sha(private / paths["final"])):
            raise ValueError("The first raw-price access must be bound to every frozen final model")
    chronology = verify_chronology(record, selection, pilot, plan, publication, start, final, gate, score, analysis, workers, raw)
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


def plot_curves(aggregated, output, filename="learning_curves.png", first_seed=False):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e1-608-remediation-export-plots"))
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
    fig.suptitle("GRU–LSTM–MLP: " + ("semilla predefinida18101" if first_seed else "media e intervalo intercuartílico") + "; pérdidas de desarrollo")
    fig.savefig(output / filename, dpi=150)
    plt.close(fig)


def verify_qq(qq, runs, seeds):
    from itertools import combinations
    from scipy import stats
    columns = {"model_a", "model_b", "n_paired", "order_index", "theoretical_normal_quantile",
               "ordered_difference_a_minus_b", "ordered_standardized_difference"}
    if set(qq) != columns or len(qq) != 3*len(seeds):
        raise ValueError("QQ data require all three registered pairs and all observed final seeds")
    pivot = runs.loc[runs.model.isin(MODELS)].pivot(index="seed", columns="model", values="rmse").sort_index()
    if set(zip(qq.model_a, qq.model_b)) != set(combinations(MODELS, 2)):
        raise ValueError("QQ pairs differ from registration")
    for a, b in combinations(MODELS, 2):
        rows = qq.loc[qq.model_a.eq(a) & qq.model_b.eq(b)].sort_values("order_index")
        observed = (pivot[a] - pivot[b]).to_numpy(float)
        theoretical = stats.probplot(observed, dist="norm", fit=False)[0]
        if (len(rows) != len(seeds) or not rows.n_paired.eq(len(seeds)).all()
                or not np.allclose(rows.theoretical_normal_quantile, theoretical, rtol=1e-10, atol=1e-14)
                or not np.allclose(rows.ordered_difference_a_minus_b, np.sort(observed), rtol=1e-10, atol=1e-14)):
            raise ValueError("QQ curves must use the observed paired RMSE differences")


def plot_qq(qq, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, ((a, b), rows) in zip(axes, qq.groupby(["model_a", "model_b"], sort=False)):
        axis.scatter(rows.theoretical_normal_quantile, rows.ordered_difference_a_minus_b, s=12)
        axis.set(title=f"{a} − {b}", xlabel="Cuantil normal teórico", ylabel="Diferencia observada de RMSE")
        axis.grid(alpha=.2)
    fig.suptitle("QQ de las tres diferencias por semilla; diagnóstico predefinido, sin selección de pruebas")
    fig.savefig(output / "qq_rmse_differences.png", dpi=150)
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
        filename = "analysis/" + name + ".csv"
        frames[name] = public_frame(pd.read_csv(bound_file(private, records["analysis"]["artifact_sha256"], filename)), ANALYSIS_FIELDS)
    identity(frames["run_metrics"], signature, record["source_signature"])
    panel(frames["run_metrics"], plan["seeds"], persistence=True)
    if (not set(METRICS) <= set(frames["run_metrics"])
            or not frames["run_metrics"].n_dates.eq(608).all()):
        raise ValueError("All six public metrics require complete608-date trajectories")
    verify_qq(frames["qq_data"], frames["run_metrics"], plan["seeds"])
    summary = read(bound_file(private, records["analysis"]["artifact_sha256"], "analysis/summary.json"))
    if (summary["study_id"] != STUDY or summary["execution_id"] != signature or summary["R_final"] != plan["R"]
            or summary["models"] != list(MODELS) or not summary["known_reserve_exploratory"]
            or summary["fresh_confirmatory_holdout"] or summary["selection_using_holdout"]
            or summary["dates_per_trajectory"] != 608 or summary["baseline_selected_before_reserve"] != records["selection"]["baseline_model"]):
        raise ValueError("The public analysis must preserve the new known-reserve scope and frozen baseline")
    hpo_runs = pd.read_csv(bound_file(private, records["selection"]["artifact_sha256"], "hpo/run_metrics.csv"))
    identity(hpo_runs, signature, record["source_signature"])
    hpo_summary = public_frame(pd.read_csv(bound_file(private, records["selection"]["artifact_sha256"], "hpo/candidate_summary.csv")),
        set(JOB_FIELDS) | {"cv_rmse_mean", "cv_rmse_sd", "trajectories", "rank", "mean_rmse", "sample_sd", "sample_sd_rmse", "sd_rmse"})
    if len(hpo_runs) != 24 or len(hpo_summary) != 12 or set(hpo_summary.model) != set(MODELS):
        raise ValueError("Missing new tuning runs or candidate summaries")
    for model in MODELS:
        rows = hpo_runs.loc[hpo_runs.model.eq(model)]
        if (set(rows.candidate_id) != {f"c{x:02d}" for x in range(4)}
                or rows.duplicated(["candidate_id", "trajectory_seed"]).any()
                or any(set(group.trajectory_seed) != {16101, 16102} for _, group in rows.groupby("candidate_id"))):
            raise ValueError("Every new tuning candidate requires both registered seeds")
        candidates = hpo_summary.loc[hpo_summary.model.eq(model)]
        if len(candidates) != 4 or candidates.candidate_id.duplicated().any() or set(candidates.candidate_id) != {f"c{x:02d}" for x in range(4)}:
            raise ValueError("Candidate summaries require four distinct configurations per family")
    pilot = pd.read_csv(bound_file(private, records["pilot"]["artifact_sha256"], "pilot/metrics.csv"))
    identity(pilot, signature, record["source_signature"])
    panel(pilot, range(17101, 17111))
    pilot_summary = pd.DataFrame(plan["pilot_summary"])
    if set(pilot_summary.model) != set(MODELS) or len(pilot_summary) != 3 or not pilot_summary.n_pilot.eq(10).all():
        raise ValueError("The published plan requires all three model pilot summaries")
    for row in pilot_summary.itertuples():
        values = pilot.loc[pilot.model.eq(row.model), "rmse"].to_numpy(float)
        mean, sd = values.mean(), values.std(ddof=1)
        if not np.allclose([row.mean_rmse, row.sample_sd_rmse, row.coefficient_of_variation], [mean, sd, sd / mean], rtol=1e-10, atol=1e-14):
            raise ValueError("Pilot descriptive mean, SD and CV differ from the observed new pilot")
    fit = pd.read_csv(bound_file(private, records["final"]["artifact_sha256"], "final/metrics.csv"))
    identity(fit, signature, record["source_signature"])
    panel(fit, plan["seeds"])
    costs = pd.read_csv(bound_file(private, records["score"]["artifact_sha256"], "score/run_metrics.csv"))
    identity(costs, signature, record["source_signature"])
    panel(costs, plan["seeds"])
    costs = costs.loc[:, ["model", "seed", "inference_seconds", "n_dates"]].merge(
        jobs.loc[:, ["model", "seed", "parameters", "training_seconds"]], on=["model", "seed"], validate="one_to_one")
    if not np.isfinite(costs[["inference_seconds", "training_seconds", "parameters"]]).all().all() or costs[["inference_seconds", "training_seconds"]].lt(0).any().any() or not costs.n_dates.eq(608).all():
        raise ValueError("Per-seed training and full608-date inference costs must be observed and finite")
    cost_rows = []
    for model, part in costs.groupby("model", sort=True):
        if part.parameters.nunique() != 1:
            raise ValueError("Cost summaries require one fixed recipe per family")
        for metric in ("training_seconds", "inference_seconds"):
            cost_rows.append({"model": model, "metric": metric, "n_runs": len(part),
                "mean": float(part[metric].mean()), "sample_sd": float(part[metric].std(ddof=1)),
                "parameters": int(part.parameters.iloc[0]), "inference_dates_per_run": 608,
                "device": "cpu", "workers": 3, "threads_per_worker": 1})
    cost_summary = pd.DataFrame(cost_rows)
    curves = []
    for row in jobs.itertuples():
        directory = private / "final" / f"{row.model}_s{int(row.seed)}"
        for stage in ("inner", "refit"):
            filename = (directory / f"history_{stage}.csv").relative_to(private).as_posix()
            frame = pd.read_csv(bound_file(private, records["final"]["artifact_sha256"], filename))
            validate_curve(frame, stage, int(row.selected_epoch))
            frame.insert(0, "seed", int(row.seed))
            frame.insert(0, "model", row.model)
            curves.append(frame)
    curves = pd.concat(curves, ignore_index=True)
    aggregated = aggregate_curves(curves)
    first_seed = curves.loc[curves.seed.eq(18101)].copy()
    architecture = projected_runs(jobs).drop(columns=[x for x in ("seed", "trajectory_seed", "training_seconds", "inference_seconds", "selected_epoch") if x in jobs]).drop_duplicates()
    # One recipe per family; varying epochs or seeds are not new architectures.
    recipe_columns = [name for name in ("model", "candidate_id", "width", "learning_rate", "dropout", "history_days", "parameters") if name in architecture]
    architecture = architecture.loc[:, recipe_columns].drop_duplicates().sort_values("model")
    if len(architecture) != 3:
        raise ValueError("Every family must retain one development-frozen final recipe")
    for row in architecture.itertuples():
        width = int(row.width)
        if width != {"GRU": 32, "LSTM": 32, "MLP": 16}[row.model]:
            raise ValueError("Registered prototype widths must be GRU32/LSTM32/MLP16")
        expected = ((3 if row.model == "GRU" else 4)*width*(5+width+2)+width+1
                    if row.model != "MLP" else width*(7*5+2)+1)
        recipe = records["selection"]["recipes"][row.model]
        if int(row.parameters) != expected or any(getattr(row, name) != recipe[name] for name in
                ("candidate_id", "width", "learning_rate", "dropout", "parameters")):
            raise ValueError("Final architecture differs from its development-frozen recipe or parameter count")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e1-608-remediation-public-", dir=output.parent) as temp:
        stage = Path(temp)
        for name, frame in {**frames, "hpo_summary": hpo_summary, "hpo_run_metrics": projected_runs(hpo_runs),
                "pilot_metrics": projected_runs(pilot), "final_fit_summary": projected_runs(fit),
                "architecture_summary": architecture, "learning_curves_by_run": curves,
                "learning_curve_summary": aggregated, "first_seed_learning_curves": first_seed,
                "run_costs": costs, "cost_summary": cost_summary, "pilot_summary": pilot_summary}.items():
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
            (stage / name).write_bytes(encode(no_private_metadata(value)))
        report = {**summary, "exported_at_utc": stamp(), "source_commit": record["source_commit"],
            "hpo_trajectories": 24, "hpo_blocks": 240, "hpo_fit_stages": 480, "pilot_trajectories": 30, "pilot_fit_stages": 60,
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
            "La reserva de enero de2025–agosto de2026 ya era conocida. Esta ejecución es exploratoria y no recupera independencia confirmatoria. "
            "Las semillas comparten608fechas: no son mercados ni calendarios independientes. El R calculado condiciona la incertidumbre al calendario.\n\n"
            "La potencia planificada corresponde a una diferencia absoluta δ bajo diferencias normales y una varianza piloto transferible. "
            "La prueba primaria contrasta diferencia cero; no constituye potencia alcanzada ni potencia temporal. "
            "Friedman, Wilcoxon y QQ son diagnósticos predefinidos y no sustituyen la familia primaria.\n\n"
            "Los CSV de curvas contienen sólo pérdidas de desarrollo. La selección interna muestra entrenamiento y validación; el reajuste sólo entrenamiento. "
            "La figura usa media e intervalo intercuartílico. Las épocas tardías reúnen menos trayectorias, según n_available.\n\n"
            "No se redistribuyen precios, objetivos o pronósticos diarios, pesos, valores de escaladores o archivos brutos. "
            "Los hashes verifican el cierre de los artefactos privados sin hacerlos públicos. La publicación observada precede a los ajustes y "
            "el plan derivado del piloto se publicó antes del ajuste definitivo. Los resultados originales permanecen separados.\n", encoding="utf-8")
        plot_curves(aggregated, stage)
        plot_curves(aggregate_curves(first_seed), stage, "first_seed_learning_curves.png", first_seed=True)
        plot_qq(frames["qq_data"], stage)
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
