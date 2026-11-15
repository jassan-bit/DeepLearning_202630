#!/usr/bin/env python3
"""Draw saved benchmark epoch histories, without fitting or executing models.

The immutable audit archive supplies every displayed loss. Existing outputs are
accepted only when byte-identical; a different output requires a new directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import textwrap
import zipfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e2-learning-curves-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "e2-learning-curves-xdg"))
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-saved-benchmark-learning-curves-v1"
SOURCES = {
    "book_e2/_static/e2_f2a_audit_20261113/training_evidence.zip":
        "daaf2cfcf050a37dbc992260011cfbb919a5d577c14f9003aaf6cf1c8be1900d",
    "book_e2/_static/e2_benchmark_final_20261113/final_freeze.json":
        "c0e9e2589dca9fa03b341bb72f87d8b541d33b4721715622daeae4438dbd60b4",
    "experiments/e2_reduced_protocol_20261112/protocol.json":
        "867cf6837158c1353a1aa86ad9c51fc6fc260948ffe02dce8e779d3c329fb1cc",
    "book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv":
        "879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1",
}
ARCHIVE, FREEZE, PROTOCOL, A2 = tuple(SOURCES)
FOLDS = ("2024-01-01", "2024-07-01", "2024-10-01")
SEEDS = tuple(range(271801, 271807))
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
SAE = "LSTM-autoencoder con predictor"
BLUE, ORANGE = "#1673ac", "#cf681e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def records(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def key(row):
    return tuple(row.get(field, "") for field in ("stage", "model", "seed", "fold_origin", "trial_id"))


def number(row, field):
    value = float(row[field])
    require(math.isfinite(value) and value >= 0, "Invalid saved nonnegative risk: " + field)
    return value


def histories(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[key(row) + (row["phase"],)].append(row)
    for ident, history in grouped.items():
        history.sort(key=lambda row: int(row["epoch"]))
        require([int(row["epoch"]) for row in history] == list(range(1, len(history) + 1)),
                "Missing, duplicated or noncontiguous epochs")
        best = int(history[0]["best_epoch"])
        require(sum(row["selected_epoch"] == "True" for row in history) == 1
                and history[best - 1]["selected_epoch"] == "True", "Selected epoch differs")
        require(all(int(row["epochs_completed"]) == len(history)
                    and int(row["best_epoch"]) == best for row in history), "Epoch metadata differs")
        if ident[0] == "final":
            require(best == len(history) and all(int(row["fixed_epochs"]) == len(history)
                                               for row in history), "Final epochs must be fixed and terminal")
    return dict(grouped)


def title(model):
    return "\n".join(textwrap.wrap(model, 28, break_long_words=False))


def save_figure(fig):
    outputs = {}
    for extension in ("png", "svg"):
        buf = io.BytesIO()
        metadata = {"Software": "Matplotlib"} if extension == "png" else {"Date": None, "Creator": "Matplotlib"}
        fig.savefig(buf, format=extension, dpi=140, metadata=metadata)
        raw = buf.getvalue()
        if extension == "svg":
            raw = ("\n".join(line.rstrip() for line in raw.decode().splitlines()) + "\n").encode()
        outputs[extension] = raw
    plt.close(fig)
    return outputs


def base_grid(heading, subheading):
    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    fig.subplots_adjust(left=.065, right=.985, bottom=.095, top=.83, wspace=.33, hspace=.62)
    fig.suptitle(heading + "\n" + subheading, fontsize=15, y=.98)
    for ax in axes.flat:
        ax.grid(alpha=.16, linewidth=.5)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("Época", fontsize=8)
        ax.set_ylabel("RMSE σ₇", fontsize=8)
    for ax in list(axes.flat)[13:]:
        ax.axis("off")
    return fig, list(axes.flat)


def bound(ax, values):
    maximum = max(float(np.max(v)) for v in values)
    ax.set_ylim(0, maximum * 1.07 if maximum else 1)
    ax.margins(x=.02)


def build(root=ROOT):
    raw = {}
    for name, expected in SOURCES.items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), "Missing regular source: " + name)
        raw[name] = path.read_bytes()
        require(sha(raw[name]) == expected, "Pinned source changed: " + name)
    freeze, protocol = json.loads(raw[FREEZE]), json.loads(raw[PROTOCOL])
    systems = protocol["systems"]
    neural = [model for model in systems if model not in CLASSICAL]
    require(systems == freeze["systems"] and len(systems) == 16 and len(neural) == 13,
            "Expected the same sixteen-system catalogue and thirteen neural recipes")
    require({row["model"] for row in records(raw[A2])} == set(systems), "A2 catalogue changed")
    require(freeze["final_seed_ids"] == list(SEEDS), "Registered final seed IDs changed")
    frozen = {(row["model"], str(row["seed"])): row for row in freeze["models"]}
    require(len(frozen) == 81, "Expected81 unique frozen predictors")
    with zipfile.ZipFile(io.BytesIO(raw[ARCHIVE])) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == 4 and set(names) == {
            "training_curves.csv", "auxiliary_curves.csv", "fit_overfit_summary.csv", "source_evidence.json"},
            "Unexpected archive members")
        members = {name: archive.read(name) for name in names}
    source = json.loads(members["source_evidence.json"])
    require(source["freeze_sha256"] == SOURCES[FREEZE], "Training archive freeze binding differs")
    for name, description in source["outputs"].items():
        require(len(members[name]) == description["bytes"] and sha(members[name]) == description["sha256"],
                "Member bytes or hash changed: " + name)
    fit_rows = records(members["fit_overfit_summary.csv"])
    supervised_rows = records(members["training_curves.csv"])
    auxiliary_rows = records(members["auxiliary_curves.csv"])
    require(len(fit_rows) == 117 and len(supervised_rows) == 6440 and len(auxiliary_rows) == 3250,
            "Unexpected fitting or epoch coverage")
    require(Counter(row["stage"] for row in supervised_rows) == {"final": 4356, "hpo": 2084}
            and Counter(row["stage"] for row in auxiliary_rows) == {"final": 2094, "hpo": 1156},
            "Original epoch counts changed")
    fits = {key(row): row for row in fit_rows}
    require(len(fits) == 117 and Counter(row["stage"] for row in fit_rows) == {"final": 78, "hpo": 39},
            "Expected78 final and39 selected HPO fit records")
    supervised, auxiliary = histories(supervised_rows), histories(auxiliary_rows)
    require(len(supervised) == 107 and len(auxiliary) == 18, "Expected107 supervised and18 auxiliary histories")
    for history in list(supervised.values()) + list(auxiliary.values()):
        for row in history:
            fit = fits[key(row)]
            require(row["selected_checkpoint_sha256"] == fit["selected_checkpoint_sha256"]
                    and row["train_signature"] == fit["train_signature"]
                    and row["validation_signature"] == fit["validation_signature"],
                    "Epoch and fit metadata bindings differ")
            number(row, "train_mse_scaled")
            train_name = "train_rmse_original" if row["phase"] == "supervised" else "train_reconstruction_rmse"
            val_name = "validation_rmse_original" if row["phase"] == "supervised" else "validation_reconstruction_rmse"
            number(row, train_name)
            require(bool(row[val_name]) == (row["stage"] == "hpo"), "Unexpected validation availability")
            if row[val_name]:
                number(row, val_name)
            if row["phase"] != "supervised":
                require(math.isclose(number(row, train_name), math.sqrt(number(row, "train_mse_scaled")),
                                     rel_tol=1e-9, abs_tol=1e-12), "Auxiliary training-unit mismatch")
    by_role = {(row["stage"], row["model"], row["seed"], row["fold_origin"]): ident
               for ident, row in fits.items()}
    require(len(by_role) == 117, "Duplicate model/stage/seed/origin")
    coverage = []
    for ident, fit in fits.items():
        stage, model, seed, fold, _ = ident
        require(model in neural and (stage == "final" and seed in {str(s) for s in SEEDS} and fold == ""
                                    or stage == "hpo" and seed == "271201" and fold in FOLDS), "Unregistered fit role")
        history = supervised.get(ident + ("supervised",), [])
        available = fit["curve_availability"] == "complete_original_epoch_logs"
        require(bool(history) == available, "Fit and epoch availability differ")
        if stage == "final":
            require(fit["selected_checkpoint_sha256"] == frozen[(model, seed)]["checkpoint_sha256"],
                    "Final history differs from frozen checkpoint")
        coverage.append({"stage": stage, "model": model, "seed": seed, "fold_origin": fold,
                         "trial_id": fit["trial_id"], "phase": "supervised",
                         "curve_availability": fit["curve_availability"], "recorded_epochs": len(history),
                         "best_epoch_metadata": fit["best_epoch"], "fixed_epochs": fit["fixed_epochs"],
                         "train_values": len(history), "validation_values": len(history) if stage == "hpo" else 0,
                         "selected_checkpoint_sha256": fit["selected_checkpoint_sha256"],
                         "figure": "final_supervised.png" if stage == "final" else "hpo_" + fold + ".png",
                         "reason": "Final fixed epochs; validation intentionally absent" if stage == "final" else
                                   "Saved HPO epoch train/validation risks" if available else
                                   "Historical cache preserves fit outcome but not epoch losses; no curve reconstructed"})
    missing = [row for row in coverage if row["stage"] == "hpo" and row["recorded_epochs"] == 0]
    require(Counter(row["fold_origin"] for row in missing) == {FOLDS[0]: 8, FOLDS[1]: 2},
            "Expected ten unavailable HPO cached histories")
    for model in CLASSICAL:
        coverage.append({"stage": "final", "model": model, "seed": "deterministic", "fold_origin": "",
                         "trial_id": "", "phase": "not_applicable", "curve_availability": "not_applicable_neural_epochs",
                         "recorded_epochs": 0, "best_epoch_metadata": "", "fixed_epochs": "",
                         "train_values": 0, "validation_values": 0,
                         "selected_checkpoint_sha256": frozen[(model, "deterministic")]["checkpoint_sha256"],
                         "figure": "", "reason": "Classical comparator has no saved neural epoch training/validation history"})
    coverage.sort(key=lambda row: (systems.index(row["model"]), row["stage"], row["fold_origin"], row["seed"]))
    require(len(coverage) == 120, "Coverage must contain117 neural fit records plus3 classical comparators")
    palette = [plt.get_cmap("Blues")(value) for value in np.linspace(.42, .98, 6)]
    seed_colors = dict(zip(SEEDS, palette))
    outputs = {}
    with plt.rc_context({"svg.hashsalt": VERSION, "font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "path.simplify": False}):
        fig, axes = base_grid("Benchmark Bitcoin · curvas finales guardadas",
                              "13 redes · seis semillas individuales · épocas fijadas previamente")
        for ax, model in zip(axes, neural):
            values = []
            for seed in SEEDS:
                ident = by_role[("final", model, str(seed), "")]
                history = supervised[ident + ("supervised",)]
                y = np.asarray([number(row, "train_rmse_original") for row in history])
                values.append(y)
                ax.plot(range(1, len(y) + 1), y, color=seed_colors[seed], lw=1, alpha=.83)
            require(len({len(y) for y in values}) == 1, "Final frozen epoch count must match across seeds")
            ax.set_title(title(model) + " · " + str(len(values[0])) + " épocas", fontsize=9)
            bound(ax, values)
        fig.legend([Line2D([], [], color=seed_colors[s], lw=1.6) for s in SEEDS],
                   [str(s) for s in SEEDS], title="Semilla final · riesgo de entrenamiento",
                   loc="upper center", bbox_to_anchor=(.5, .91), ncol=6, frameon=False, fontsize=8)
        fig.text(.5, .035, "RMSE de σ₇ en unidades originales; escala vertical propia por panel.\n"
                 "Los ajustes finales no generan una curva de validación; no se añade ni se infiere.",
                 ha="center", va="center", fontsize=8)
        for ext, content in save_figure(fig).items():
            outputs["final_supervised." + ext] = content
        for fold in FOLDS:
            available_count = sum(row["fold_origin"] == fold and row["recorded_epochs"] > 0 for row in coverage)
            fig, axes = base_grid("Benchmark Bitcoin · selección HPO · origen " + fold,
                                  str(available_count) + "/13 curvas disponibles · semilla 271201")
            for ax, model in zip(axes, neural):
                ident = by_role[("hpo", model, "271201", fold)]
                fit, history = fits[ident], supervised.get(ident + ("supervised",), [])
                ax.set_title(title(model), fontsize=9)
                if not history:
                    ax.set_facecolor("#f0f2f4");ax.set_xticks([]);ax.set_yticks([])
                    ax.text(.5, .57, "Curva por época no disponible\nen la caché histórica", transform=ax.transAxes,
                            ha="center", va="center", fontsize=8.5, color="#4d5964")
                    ax.text(.5, .26, "Mejor época registrada: " + fit["best_epoch"], transform=ax.transAxes,
                            ha="center", fontsize=7.5, color="#4d5964")
                    continue
                x = [int(row["epoch"]) for row in history]
                train = np.asarray([number(row, "train_rmse_original") for row in history])
                val = np.asarray([number(row, "validation_rmse_original") for row in history])
                best = int(history[0]["best_epoch"])
                ax.plot(x, train, color=BLUE, lw=1.1);ax.plot(x, val, color=ORANGE, lw=1.1)
                ax.axvline(best, color="#737b80", ls=":", lw=.8)
                ax.plot(best, val[best - 1], "o", color=ORANGE, ms=3)
                bound(ax, [train, val])
            fig.legend([Line2D([], [], color=BLUE), Line2D([], [], color=ORANGE),
                        Line2D([], [], color="#737b80", ls=":", marker="o", markersize=3)],
                       ["Entrenamiento", "Validación", "Mejor época HPO"], loc="upper center",
                       bbox_to_anchor=(.5, .91), ncol=3, frameon=False)
            fig.text(.5, .035, "RMSE de σ₇ en unidades originales; ejes propios por panel y curvas completas.\n"
                     "Cada origen usa una sola semilla; las pérdidas ausentes de caché no se reconstruyen.",
                     ha="center", va="center", fontsize=8)
            for ext, content in save_figure(fig).items():
                outputs["hpo_" + fold + "." + ext] = content
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.subplots_adjust(left=.07, right=.985, bottom=.13, top=.76, wspace=.35, hspace=.55)
        fig.suptitle("SAE · fases auxiliares guardadas\nReconstrucción en escala auxiliar; no es el objetivo σ₇", fontsize=15, y=.985)
        aux_mapping = []
        for ri, phase in enumerate(("ae1", "ae2")):
            for ci, fold in enumerate(FOLDS + ("final",)):
                ax = axes[ri, ci];ys = []
                ax.grid(alpha=.16, linewidth=.5);ax.set_xlabel("Época", fontsize=8)
                ax.set_ylabel("RMSE de reconstrucción", fontsize=8);ax.tick_params(labelsize=8)
                ax.set_title(phase.upper() + " · " + (fold if fold != "final" else "ajuste final, seis semillas"), fontsize=9)
                for seed in (SEEDS if fold == "final" else (271201,)):
                    ident = by_role[("final" if fold == "final" else "hpo", SAE, str(seed), "" if fold == "final" else fold)]
                    history = auxiliary[ident + (phase,)]
                    train = np.asarray([number(row, "train_reconstruction_rmse") for row in history]);ys.append(train)
                    ax.plot(range(1, len(train) + 1), train, color=seed_colors[seed] if fold == "final" else BLUE, lw=1)
                    if fold != "final":
                        val = np.asarray([number(row, "validation_reconstruction_rmse") for row in history]);ys.append(val)
                        best = int(history[0]["best_epoch"])
                        ax.plot(range(1, len(val) + 1), val, color=ORANGE, lw=1)
                        ax.axvline(best, color="#737b80", ls=":", lw=.7)
                        ax.plot(best, val[best - 1], "o", ms=3, color=ORANGE)
                    aux_mapping.append({"stage": ident[0], "fold_origin": ident[3], "seed": seed,
                                        "phase": phase, "recorded_epochs": len(history),
                                        "validation_values": 0 if fold == "final" else len(history)})
                bound(ax, ys)
        fig.legend([Line2D([], [], color=BLUE), Line2D([], [], color=ORANGE)] +
                   [Line2D([], [], color=seed_colors[s]) for s in SEEDS],
                   ["Entrenamiento HPO", "Validación HPO"] + ["Final " + str(s) for s in SEEDS],
                   loc="upper center", bbox_to_anchor=(.5, .89), ncol=4, frameon=False, fontsize=8)
        fig.text(.5, .045, "AE1 y AE2 reconstruyen las entradas escaladas, con entradas internas distintas; se muestran por separado.\n"
                 "Ejes propios; historias completas. La validación no existe en las fases finales de épocas fijas.",
                 ha="center", va="center", fontsize=8)
        for ext, content in save_figure(fig).items():
            outputs["sae_auxiliary." + ext] = content
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(coverage[0]), lineterminator="\n")
    writer.writeheader();writer.writerows(coverage)
    outputs["coverage.csv"] = buf.getvalue().encode()
    report = {
        "version": VERSION, "scope": "posthoc_visualization_of_immutable_saved_epoch_losses_no_new_model_result",
        "input_sha256": SOURCES, "archive_members": {
            name: {"bytes": len(content), "sha256": sha(content)} for name, content in members.items()},
        "script_sha256": sha(Path(__file__).read_bytes()),
        "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "catalogue_order": systems, "neural_models": 13, "classical_models": list(CLASSICAL),
        "final_seed_ids": list(SEEDS), "HPO_seed": 271201, "HPO_fold_origins": list(FOLDS),
        "coverage": {"final_supervised_fits_with_curves": 78, "selected_HPO_fits": 39,
                     "selected_HPO_fits_with_curves": 29, "selected_HPO_cached_fits_without_curves": 10,
                     "supervised_epoch_rows": 6440, "final_supervised_epoch_rows": 4356,
                     "HPO_supervised_epoch_rows": 2084, "auxiliary_epoch_rows": 3250,
                     "auxiliary_histories": 18, "coverage_csv_rows": len(coverage),
                     "coverage_csv_scope": "117supervised fit metadata records plus3classical not-applicable records; auxiliary histories separately in thisJSON",
                     "missing_HPO": [{"model": row["model"], "fold_origin": row["fold_origin"],
                                      "best_epoch_metadata": row["best_epoch_metadata"]} for row in missing]},
        "auxiliary_history_coverage": aux_mapping,
        "metrics": {"supervised": "Saved epoch-boundary full-training and HPO validation RMSE in original sigma7 units; not ETTh1 standardized MSE",
                    "auxiliary": "Saved RMSE reconstruction of scaled original inputs, separate ae1/ae2 panels; not sigma7",
                    "risk_evaluation": "Original durable fitting histories, epoch-boundary model.eval/no_grad; no new risk computed here",
                    "final_validation": "Intentionally absent under frozen fixed-epoch protocol; no calibration/test curve substituted",
                    "aggregation": "None. Every recorded history is shown individually; no fold or seed averaging, smoothing or uncertainty band.",
                    "selection_marker": "Saved HPO best epoch only; final epochs were previously fixed"},
        "drawing": {"per_panel_y_scale": True, "y_axis_starts_at_zero": True,
                    "complete_epoch_range": True, "all_available_individual_histories_drawn": True,
                    "selected_favourable_epoch_subsets": False, "SVG_Date_omitted": True,
                    "stable_svg_hashsalt": VERSION},
        "access": {"training_metadata_archive_opened": 1, "raw_market_archives_opened": 0,
                   "checkpoint_weights_loaded": 0, "model_fits": 0, "model_inference_calls": 0,
                   "predictive_scores_computed": 0, "hypothesis_tests_computed": 0,
                   "saved_source_artifacts_modified": False},
        "limits": ["Epoch risks use the configured batch; saved final forecasts used batch1. They are not guaranteed numerically identical, especially for batch-dependent period selection.",
                   "Expanding HPO folds are not independent repetitions and are not averaged. The HPO curves concern the chosen recipe, not all234opportunities.",
                   "Final curves show training only. Calibration and test were not new epoch-selection validation sets.",
                   "A visual train/validation gap is not the formal absence-of-overfit test. Historical audit delta_g=null is not interpreted as proof that no prior margin existed.",
                   "No unavailable cache curve is reconstructed. Classical comparators have no comparable neural epoch history.",
                   "Local y scales improve readability; absolute panel heights do not rank model performance. Auxiliary reconstruction phases have a different target/scale from supervised risk."],
        "output_sha256": {name: sha(content) for name, content in outputs.items()},
    }
    outputs["learning_curves.json"] = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/e2-saved-learning-curves-v1"))
    args = parser.parse_args(argv)
    require(not args.output.is_symlink(), "Output cannot be a symlink")
    output = args.output.resolve();products = build()
    for name, content in products.items():
        target = output / name
        require(not target.is_symlink(), "Output file cannot be a symlink")
        require(not target.exists() or target.is_file() and target.read_bytes() == content,
                "Refusing changed existing output; choose a new directory: " + name)
    output.mkdir(parents=True, exist_ok=True)
    for name, content in products.items():
        path = output / name
        if not path.exists():
            path.write_bytes(content)
    print(json.dumps({"version": VERSION, "figures": 5, "products": len(products),
                      "final_curves": 78, "HPO_curves": 29, "HPO_missing": 10,
                      "auxiliary_histories": 18, "new_model_fits": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
