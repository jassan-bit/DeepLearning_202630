"""Export only loss curves and provenance from the new final-recipe replay.

Weights, scalers, observations and predictions never enter the public output.
Training uses the separately published immutable reproduction runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e1-reproduced-curve-plots"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "e1-reproduced-curve-cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("GRU", "LSTM", "TCN")
SEEDS = tuple(range(8101, 8566))
PROTOCOL = "experiments/e1_curve_reproduction_20261106/protocol.json"
ORIGINAL = "results/gru_lstm_tcn_planned_20261105"
SCIENCE = ("history_inner.csv", "history_refit.csv", "configuration.json",
           "preprocessing.json", "split.json", "model.pt")
METRICS = ("optimization_mse_scaled", "train_mse_scaled_eval", "validation_mse_scaled_eval")
PUBLICATION_RECEIPT = {
    "source_commit": "fe41b8ad1eaa1425aad900f6c3cd966843d600c7",
    "commit_url": "https://github.com/jassan-bit/DeepLearning_202630/commit/fe41b8ad1eaa1425aad900f6c3cd966843d600c7",
    "github_committer_date": "2026-11-07T01:55:58Z",
    "verification_method": "Parent agent retrieved this immutable commit with the connected GitHub GET tool before invoking the private register command",
    "observed_before_registration": True,
    "limitation": "This receipt records observed ordering. A committer date alone is not an independent timestamp proving publication time; no HTTP response timestamp is invented.",
}
PUBLIC_FILES = frozenset({"README.md", "summary.json", "reproduction_registration.json",
    "execution.json", "runtime.json", "fit_summary.csv", "learning_curves_by_run.csv",
    "learning_curve_summary.csv", "epoch_summary.csv", "artifact_hash_comparison.csv",
    "learning_curves_final_recipes.png", "learning_curves_seed8101.png"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def verify_registration(source, private):
    record = read(private / "registration.json")
    unsigned = {k: v for k, v in record.items() if k != "registration_signature"}
    if hashlib.sha256(encode(unsigned)).hexdigest() != record["registration_signature"]:
        raise ValueError("The new reproduction registration changed")
    if sha(source / PROTOCOL) != record["protocol_sha256"]:
        raise ValueError("Registered source protocol differs")
    if read(source / PROTOCOL) != record["protocol"]:
        raise ValueError("Registered protocol payload differs")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != record["source_commit"]:
        raise ValueError("Use the immutable reproduction source checkout")
    for name, expected in record["protocol"]["sources_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError("Registered source changed: " + name)
    finish = read(private / "fit_finished.json")
    if (finish["registration_signature"] != record["registration_signature"]
            or finish["models"] != 1395 or finish["new_fit_stages"] != 2790
            or finish["reserve_price_payloads_read"] != 0 or finish["forecasts_generated"] != 0
            or sha(private / "fit_metrics.csv") != finish["fit_metrics_sha256"]):
        raise ValueError("A complete development-only replay is required")
    return record, finish


def validate_curve(curve, stage, selected_epoch):
    required = {"epoch", "stage", *METRICS, "best_epoch_so_far", "learning_rate"}
    if set(curve) != required or not len(curve):
        raise ValueError("Unexpected loss-curve schema or empty curve")
    if (not curve.stage.eq(stage).all()
            or curve.epoch.tolist() != list(range(1, len(curve) + 1))
            or not 1 <= selected_epoch <= len(curve) <= 100):
        raise ValueError("Invalid stage, epoch order or selected epoch")
    for name in METRICS[:2]:
        if not np.isfinite(curve[name]).all() or curve[name].lt(0).any():
            raise ValueError("Training losses must be finite nonnegative observed values")
    if stage == "inner":
        if (not np.isfinite(curve.validation_mse_scaled_eval).all()
                or curve.validation_mse_scaled_eval.lt(0).any()
                or int(curve.best_epoch_so_far.iloc[-1]) != selected_epoch):
            raise ValueError("Invalid observed inner validation curve")
    elif len(curve) != selected_epoch or not curve.validation_mse_scaled_eval.isna().all():
        raise ValueError("Refit has exactly the chosen duration and no invented validation losses")


def aggregate_curves(curves):
    rows = []
    for (model, stage, epoch), frame in curves.groupby(["model", "stage", "epoch"], sort=True):
        for metric in METRICS:
            values = frame[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            rows.append({"model": model, "stage": stage, "epoch": int(epoch), "metric": metric,
                "n_trajectories": int(len(values)), "mean": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "median": float(np.median(values)), "q25": float(np.quantile(values, .25)),
                "q75": float(np.quantile(values, .75)), "minimum": float(values.min()),
                "maximum": float(values.max())})
    return pd.DataFrame(rows)


def figures(curves, aggregated, destination):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = {"train_mse_scaled_eval": "#1673a4", "validation_mse_scaled_eval": "#c65d32"}
    for column, model in enumerate(MODELS):
        for row, stage in enumerate(("inner", "refit")):
            axis = axes[row, column]
            for metric, color in colors.items():
                frame = aggregated.loc[aggregated.model.eq(model) & aggregated.stage.eq(stage)
                                       & aggregated.metric.eq(metric)].sort_values("epoch")
                if frame.empty:
                    continue
                axis.plot(frame.epoch, frame["mean"], color=color,
                    label="Train" if metric.startswith("train") else "Validación interna")
                axis.fill_between(frame.epoch, frame.q25, frame.q75, color=color, alpha=.18)
            axis.set(title=f"{model} · {'selección interna' if stage == 'inner' else 'reajuste final'}",
                xlabel="Época", ylabel="MSE en la escala del ajuste")
            counts = aggregated.loc[aggregated.model.eq(model) & aggregated.stage.eq(stage)
                & aggregated.metric.eq("train_mse_scaled_eval")].sort_values("epoch")
            if len(counts):
                axis.text(.98, .97, f"n por época: {int(counts.n_trajectories.iloc[0])} → {int(counts.n_trajectories.iloc[-1])}",
                    transform=axis.transAxes, ha="right", va="top", fontsize=8)
            axis.grid(alpha=.22)
            axis.legend(fontsize=8)
    fig.suptitle("Modelo final D7 · curvas de una reproducción nueva\n465 semillas por red; media y banda intercuartil", fontsize=13)
    fig.savefig(destination / "learning_curves_final_recipes.png", dpi=150,
                metadata={"Software": "E1 development curve reproduction export"})
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for column, model in enumerate(MODELS):
        for row, stage in enumerate(("inner", "refit")):
            axis = axes[row, column]
            frame = curves.loc[curves.model.eq(model) & curves.seed.eq(8101) & curves.stage.eq(stage)]
            for metric, color in colors.items():
                available = frame.loc[frame[metric].notna()]
                if len(available):
                    axis.plot(available.epoch, available[metric], marker="." if len(available) < 4 else None,
                        color=color, label="Train" if metric.startswith("train") else "Validación interna")
            if stage == "inner":
                selected = int(frame.best_epoch_so_far.iloc[-1])
                axis.axvline(selected, color="#777777", linestyle="--", linewidth=.9,
                             label=f"Época elegida: {selected}")
            axis.set(title=f"{model} · {'selección interna' if stage == 'inner' else 'reajuste final'}",
                xlabel="Época", ylabel="MSE en la escala del ajuste")
            axis.grid(alpha=.22)
            axis.legend(fontsize=8)
    fig.suptitle("Reproducción nueva · semilla 8101 predefinida antes de ajustar", fontsize=13)
    fig.savefig(destination / "learning_curves_seed8101.png", dpi=150,
                metadata={"Software": "E1 development curve reproduction export"})
    plt.close(fig)


def export(source, private, output, mirror=None):
    source, private, output = map(lambda p: Path(p).resolve(), (source, private, output))
    if output == private or output.is_relative_to(private) or private.is_relative_to(output):
        raise ValueError("Public output must be separate from private evidence")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Public output must be new or empty")
    registration, finish = verify_registration(source, private)
    original = read(source / ORIGINAL / "final_freeze_provenance.json")
    historic_jobs = {(r["model"], r["trajectory_seed"]): r for r in original["jobs"]}
    expected_hashes = original["artifact_sha256"]
    fit_rows, curve_frames, comparison = [], [], []
    for model in MODELS:
        for seed in SEEDS:
            directory = private / "final" / f"{model}_s{seed}"
            complete = read(directory / "completion.json")
            if complete["registration_signature"] != registration["registration_signature"]:
                raise ValueError("Trajectory belongs to another reproduction")
            for name, expected in complete["artifact_sha256"].items():
                if Path(name).name != name or sha(directory / name) != expected:
                    raise ValueError("Private completed artifact is invalid: " + name)
            record = complete["record"]
            if record["model"] != model or record["trajectory_seed"] != seed:
                raise ValueError("Trajectory identity differs from planned panel")
            selected = int(record["selected_epoch"])
            one_fit = {key: record[key] for key in ("model", "candidate_id", "width", "learning_rate",
                "dropout", "parameters", "train_samples", "inner_train_samples", "inner_validation_samples",
                "selected_epoch", "training_seconds")}
            one_fit.update(seed=seed, original_selected_epoch=int(historic_jobs[model, seed]["selected_epoch"]))
            one_fit["selected_epoch_equal_original"] = selected == one_fit["original_selected_epoch"]
            for stage in ("inner", "refit"):
                curve = pd.read_csv(directory / f"history_{stage}.csv")
                validate_curve(curve, stage, selected)
                one_fit[f"{stage}_epochs"] = len(curve)
                curve.insert(0, "seed", seed)
                curve.insert(0, "model", model)
                curve_frames.append(curve)
            fit_rows.append(one_fit)
            for name in SCIENCE:
                relative = f"final/{model}_s{seed}/{name}"
                expected = expected_hashes[relative]
                actual = sha(directory / name)
                comparison.append({"model": model, "seed": seed, "artifact": name,
                    "historical_expected_sha256": expected, "reproduced_sha256": actual,
                    "byte_identical": actual == expected})
    curves, fits, comparison = pd.concat(curve_frames, ignore_index=True), pd.DataFrame(fit_rows), pd.DataFrame(comparison)
    aggregated = aggregate_curves(curves)
    epochs = []
    for model, frame in fits.groupby("model"):
        epochs.append({"model": model, "runs": len(frame),
            "selected_epoch_median": float(frame.selected_epoch.median()),
            "selected_epoch_min": int(frame.selected_epoch.min()), "selected_epoch_max": int(frame.selected_epoch.max()),
            "inner_epochs_median": float(frame.inner_epochs.median()),
            "inner_stopped_before_100": int(frame.inner_epochs.lt(100).sum()),
            "selected_epochs_equal_original": int(frame.selected_epoch_equal_original.sum()),
            "training_seconds_sum": float(frame.training_seconds.sum())})
    curve_comparison = comparison.loc[comparison.artifact.isin(("history_inner.csv", "history_refit.csv"))]
    summary = {"schema_version": 1, "purpose": "New development-only reproduction of final D7 learning curves",
        "source_commit": registration["source_commit"], "registered_at_utc": registration["registered_at_utc"],
        "started_at_utc": read(private / "fit_started.json")["started_at_utc"],
        "finished_at_utc": finish["finished_at_utc"], "models": list(MODELS), "seeds_per_family": 465,
        "new_trajectories": 1395, "new_fit_stages": 2790, "curve_rows": len(curves),
        "same_seed_replay_of_historical_recipes": True,
        "additional_independent_benchmark_samples": 0,
        "representative_seed": 8101, "reserve_price_payloads_read": 0, "forecasts_generated": 0,
        "forecast_counter_scope": "No scoring/inference trajectories or dated forecasts are emitted; forward passes used to evaluate train/inner-validation losses are intrinsic to fitting.",
        "historical_curve_hash_comparisons": len(curve_comparison),
        "historical_curve_hash_matches": int(curve_comparison.byte_identical.sum()),
        "other_scientific_artifact_hash_comparisons": len(comparison) - len(curve_comparison),
        "all_scientific_artifact_hash_matches": int(comparison.byte_identical.sum()),
        "all_scientific_artifact_hash_comparisons": len(comparison),
        "selected_epochs_equal_original": int(fits.selected_epoch_equal_original.sum()),
        "matches_by_artifact": {name: {"compared": len(frame), "matched": int(frame.byte_identical.sum())}
            for name, frame in comparison.groupby("artifact")},
        "original_private_histories_recovered": False,
        "publication_verification": PUBLICATION_RECEIPT if registration["source_commit"] == PUBLICATION_RECEIPT["source_commit"] else {
            "verified_by_this_receipt": False, "source_commit": registration["source_commit"]},
        "scope": "A replay after the original evaluation, not a new holdout benchmark or original-log recovery",
        "interpretation": "Report all historical hash mismatches. Equality supports identical reproduction under the recorded manifests; it does not create temporal independence.",
        "aggregation": "Per-epoch means and IQRs include only trajectories that reached that epoch; inner/refit target scales differ and should not be directly compared.",
        "original_results_baseline_and_margin_changed": False}
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("fit_summary.csv", fits), ("learning_curves_by_run.csv", curves),
            ("learning_curve_summary.csv", aggregated), ("epoch_summary.csv", pd.DataFrame(epochs)),
            ("artifact_hash_comparison.csv", comparison)):
        frame.to_csv(output / name, index=False)
    for name, document in (("summary.json", summary), ("reproduction_registration.json", registration),
            ("runtime.json", registration["runtime"]),
            ("execution.json", {"started": read(private / "fit_started.json"), "finished": finish})):
        (output / name).write_bytes(encode(document))
    figures(curves, aggregated, output)
    (output / "README.md").write_text(
        "# Curvas reproducidas de las recetas finales D7\n\n"
        "Se ejecutaron de nuevo las tres recetas congeladas, con 465 semillas por familia, "
        "sólo desarrollo 2023–2024 y código histórico intacto. Esta es una reproducción posterior; "
        "no se recuperaron los logs privados originales. Se publicaron protocolo, código y semillas "
        "antes de esta nueva ejecución.\n\n"
        f"Fuente inmutable publicada: {registration['published_source_url']}.\n\n"
        "Instalación del entorno con el paquete CPU fijado en el lock:\n\n"
        "```bash\npython -m pip install -r requirements-e1-curves.lock.txt "
        "--extra-index-url https://download.pytorch.org/whl/cpu\n```\n\n"
        f"La comparación con el manifiesto histórico encuentra {summary['historical_curve_hash_matches']} "
        f"de {summary['historical_curve_hash_comparisons']} CSV de curvas idénticos byte a byte. "
        "La tabla de hashes conserva todas las diferencias y las coincidencias; no se sustituyen "
        "semillas ni recetas según ese resultado.\n\n"
        "Las figuras agregadas muestran media y rango intercuartil. Cada época incluye sólo "
        "trayectorias que la alcanzaron; consultar n_trajectories. Las bandas describen variación "
        "por inicialización sobre el mismo calendario y no son IC de generalización temporal. "
        "La semilla 8101 se fijó para la figura individual antes del ajuste.\n\n"
        "La selección interna tiene pérdidas train/validación; el reajuste conserva sólo train "
        "durante el número cerrado de épocas. La ausencia de validación en el reajuste es "
        "intencional. La escala del objetivo se ajusta por separado con los datos permitidos de "
        "cada etapa; no comparar directamente los valores entre inner y refit.\n\n"
        "Este producto no consulta la reserva de 2026, no genera pronósticos nuevos y no cambia "
        "el baseline, δ o los resultados originales. Pesos, observaciones, predicciones y valores "
        "de escaladores quedan fuera de la publicación.\n", encoding="utf-8")
    if {p.name for p in output.iterdir()} != PUBLIC_FILES:
        raise ValueError("Unexpected public export member")
    manifest = {"schema_version": 1, "source_commit": registration["source_commit"],
        "protocol_sha256": registration["protocol_sha256"],
        "exporter_sha256": sha(Path(__file__)),
        "inputs_sha256": {"registration.json": sha(private / "registration.json"),
            "fit_started.json": sha(private / "fit_started.json"), "fit_finished.json": sha(private / "fit_finished.json"),
            "fit_metrics.csv": sha(private / "fit_metrics.csv")},
        "files": {name: {"bytes": (output / name).stat().st_size, "sha256": sha(output / name)}
                  for name in sorted(PUBLIC_FILES)}}
    (output / "manifest.json").write_bytes(encode(manifest))
    if mirror is not None:
        mirror = Path(mirror).resolve()
        if mirror.exists() and any(mirror.iterdir()):
            raise ValueError("Mirror must be new or empty")
        mirror.mkdir(parents=True, exist_ok=True)
        for path in output.iterdir():
            shutil.copyfile(path, mirror / path.name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path)
    args = parser.parse_args()
    result = export(args.source_root, args.private_dir, args.output_dir, args.mirror_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
