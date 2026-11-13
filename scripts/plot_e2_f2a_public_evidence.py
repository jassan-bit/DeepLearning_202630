"""Reproduce F2a figures from saved derived evidence; never loads a model."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIXED_SEED = 271801
SHORT_NAMES = {"LSTM-autoencoder con predictor": "LSTM-AE + predictor"}
FIGURE_CAPTIONS = {
    "coverage_by_system": (
        "Cobertura de los intervalos congelados del 90%, con la semilla neural 271801 "
        "fijada por identidad y la trayectoria propia de cada control determinista. "
        "Las barras son intervalos temporales nominales del 95% obtenidos mediante "
        "bootstrap circular de bloques de 14 días y 2.000 réplicas, condicionados a "
        "las trayectorias congeladas; no corrigen simultaneidad. La línea señala el "
        "90% nominal. Esta figura no selecciona la mejor semilla."
    ),
    "temporal_rmse": (
        "Media de RMSE por trimestre entre seis semillas para las redes y una "
        "trayectoria para cada control. Las celdas están en unidades de sigma7 por "
        "10^-3 y conservan las fechas originales. 2026T3 es parcial: julio y agosto. "
        "La comparación describe el calendario ya observado; no corresponde a "
        "nuevos orígenes ni establece estabilidad temporal."
    ),
    "fit_calibration_test": (
        "Media de RMSE entre seis semillas de los mismos predictores neuronales "
        "finales en ajuste, calibración y evaluación. Las ventanas son distintas: "
        "ajuste hasta 2024-10-31, calibración de 53 fechas y evaluación de 608 fechas. "
        "El riesgo de ajuste procede de evaluaciones por época con lotes de 16, 32 "
        "o 64; las predicciones persistidas de calibración y evaluación usan lote 1. "
        "Las diferencias no identifican por sí solas sobreajuste ni su ausencia."
    ),
    "residual_acf": (
        "Autocorrelación descriptiva de los residuos y-p hasta el rezago 28, con la "
        "semilla neural fija 271801 y cada control determinista. Se conserva el "
        "calendario de 608 fechas. No se dibujan bandas de independencia: los "
        "residuos tienen dependencia temporal y el diagnóstico no prueba causalidad."
    ),
}


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _short(model):
    return SHORT_NAMES.get(model, model)


def _save(fig, destination, stem):
    destination = Path(destination)
    for extension in ("svg", "png"):
        metadata = ({"Date": None, "Creator": "e2-f2a-audit-v1"}
                    if extension == "svg" else {"Software": "e2-f2a-audit-v1"})
        fig.savefig(destination / (stem + "." + extension), dpi=180,
                    facecolor="white", bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "axes.labelsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "svg.hashsalt": "e2-f2a-audit-v1",
        "figure.dpi": 120, "savefig.dpi": 180})


def _coverage(records, models, destination):
    by_model = {}
    for row in records:
        if row["seed"] in (str(FIXED_SEED), "deterministic"):
            if row["model"] in by_model:
                raise ValueError("Duplicate fixed-seed coverage record")
            by_model[row["model"]] = row
    if set(by_model) != set(models):
        raise ValueError("Coverage requires every fixed-seed system")
    centers = np.asarray([float(by_model[m]["coverage90"]) * 100 for m in models])
    bounds = np.asarray([json.loads(by_model[m]["coverage90_CI95_temporal"]) for m in models]) * 100
    fig, ax = plt.subplots(figsize=(10, 8.2))
    y = np.arange(len(models))
    neural = np.asarray([by_model[m]["seed"] != "deterministic" for m in models])
    for flag, color, label in ((True, "#2166ac", "Red: semilla 271801"),
                               (False, "#b35806", "Control determinista")):
        selected = neural == flag
        ax.errorbar(centers[selected], y[selected],
            xerr=np.vstack((centers[selected] - bounds[selected, 0], bounds[selected, 1] - centers[selected])),
            fmt="o", color=color, markersize=5, elinewidth=1.6, capsize=3, label=label)
    ax.axvline(90, color="#555555", linestyle="--", linewidth=1.3, label="Cobertura nominal: 90%")
    ax.set_yticks(y, [_short(m) for m in models])
    ax.invert_yaxis()
    ax.set_xlim(min(60, float(bounds[:, 0].min()) - 2), 101)
    ax.set_xlabel("Cobertura observada (%) e intervalo temporal nominal del 95%")
    fig.suptitle("Cobertura de los intervalos congelados", x=.27, y=.985,
                 ha="left", fontsize=13)
    ax.grid(axis="x", color="#e5e5e5", linewidth=.7)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(.265, .95), ncol=1, frameon=False, fontsize=9)
    fig.text(.1, .015, "608 fechas; bloques circulares de 14 días; 2.000 réplicas.\n"
             "Intervalos condicionados a las trayectorias, sin corrección simultánea.", fontsize=8.5)
    fig.subplots_adjust(left=.27, bottom=.12, top=.83)
    _save(fig, destination, "coverage_by_system")


def _temporal(records, models, destination):
    quarters = sorted({row["quarter"] for row in records})
    if len(quarters) != 7:
        raise ValueError("Exactly seven observed calendar quarters are required")
    lookup = {(r["model"], r["quarter"]): float(r["RMSE"]) * 1000 for r in records}
    matrix = np.asarray([[lookup[m, q] for q in quarters] for m in models])
    labels = [q.replace("Q", "T") + ("*" if q == "2026Q3" else "") for q in quarters]
    fig, ax = plt.subplots(figsize=(11.2, 8.3))
    plot = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(7), labels)
    ax.set_yticks(range(len(models)), [_short(m) for m in models])
    ax.set_title("RMSE en siete trimestres del calendario observado", loc="left", pad=16)
    midpoint = float((matrix.min() + matrix.max()) / 2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8.2,
                    color="white" if matrix[i, j] > midpoint else "#222222")
    colorbar = fig.colorbar(plot, ax=ax, fraction=.04, pad=.035)
    colorbar.set_label("RMSE de σ7 (×10⁻³)")
    ax.tick_params(length=0)
    fig.text(.10, .025, "* 2026T3 parcial: 2026-07-01 a 2026-08-31. "
             "Media de seis semillas para redes; R=1 para controles.\n"
             "Descripción retrospectiva; no son nuevos orígenes ni una prueba de estabilidad.", fontsize=8.5)
    fig.subplots_adjust(left=.25, right=.91, bottom=.12, top=.91)
    _save(fig, destination, "temporal_rmse")


def _fit(records, models, destination):
    networks = [m for m in models if any(r["model"] == m and r["stage"] == "final" for r in records)]
    if len(networks) != 13:
        raise ValueError("Exactly thirteen final neural systems are required")
    fields = ("selected_train_rmse_original", "calibration_rmse_original", "final_rmse_original")
    matrix = []
    for model in networks:
        rows = [r for r in records if r["model"] == model and r["stage"] == "final"]
        if len(rows) != 6 or {int(r["seed"]) for r in rows} != set(range(271801, 271807)):
            raise ValueError("Every final neural system requires all six frozen seeds")
        matrix.append([float(np.mean([float(r[field]) for r in rows])) * 1000 for field in fields])
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8.6, 8.2))
    plot = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(3), ["Ajuste\nhasta 2024-10-31", "Calibración\n53 fechas", "Evaluación\n608 fechas"])
    ax.set_yticks(range(13), [_short(m) for m in networks])
    ax.set_title("RMSE medio por etapa de los predictores finales", loc="left", pad=18)
    midpoint = float((matrix.min() + matrix.max()) / 2)
    for i in range(13):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > midpoint else "#222222")
    colorbar = fig.colorbar(plot, ax=ax, fraction=.045, pad=.04)
    colorbar.set_label("RMSE de σ7 (×10⁻³); media de R=6")
    ax.tick_params(length=0)
    fig.text(.09, .02, "Ventanas distintas; ajuste evaluado por época con lotes 16/32/64; "
             "calibración y evaluación con lote 1.\n"
             "Las diferencias no establecen por sí solas sobreajuste ni ausencia de sobreajuste.", fontsize=8.2)
    fig.subplots_adjust(left=.29, right=.88, bottom=.15, top=.91)
    _save(fig, destination, "fit_calibration_test")


def _residual_acf(bundle, models, destination):
    targets = np.asarray(bundle["y_true"], dtype=float)
    selected = {p["model"]: p for p in bundle["predictors"]
                if p["seed"] in (FIXED_SEED, "deterministic")}
    if set(selected) != set(models):
        raise ValueError("Residual ACF requires every fixed-seed system")
    fig, axes = plt.subplots(4, 4, figsize=(12, 9.7), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models):
        residual = targets - np.asarray(selected[model]["prediction_original"], dtype=float)
        centered = residual - residual.mean()
        denominator = float(np.dot(centered, centered))
        values = [float(np.dot(centered[lag:], centered[:-lag])) / denominator if denominator else 0.
                  for lag in range(1, 29)]
        ax.vlines(range(1, 29), 0, values, color="#2166ac", linewidth=1.2)
        ax.axhline(0, color="#777777", linewidth=.7)
        ax.set_title(_short(model), fontsize=10.2, loc="left", pad=5)
        ax.set_xlim(0, 29)
        ax.set_ylim(-1, 1)
        ax.set_xticks([0, 7, 14, 21, 28])
        ax.grid(axis="y", color="#eeeeee", linewidth=.6)
    for ax in axes[-1]:
        ax.set_xlabel("Rezago (días)")
    for ax in axes[:, 0]:
        ax.set_ylabel("ACF residual")
    fig.suptitle("Dependencia residual: semilla neural fija 271801 y controles deterministas",
                 fontsize=14, x=.08, ha="left")
    fig.text(.08, .02, "608 fechas; residuo y − p; rezagos 1–28. "
             "Sin bandas de independencia ni interpretación causal.", fontsize=9)
    fig.subplots_adjust(left=.08, right=.97, bottom=.095, top=.91, hspace=.43, wspace=.20)
    _save(fig, destination, "residual_acf")


def generate_figures(input_path, statistics_dir, training_dir, destination):
    """Create eight deterministic image files from actual saved evidence only."""
    bundle = json.loads(Path(input_path).read_bytes())
    models = list(dict.fromkeys(p["model"] for p in bundle["predictors"]))
    if len(models) != 16:
        raise ValueError("The frozen sixteen-system benchmark is required")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    _style()
    _coverage(_read_csv(Path(statistics_dir) / "coverage_intervals.csv"), models, destination)
    _temporal(_read_csv(Path(statistics_dir) / "quarterly_metrics.csv"), models, destination)
    _fit(_read_csv(Path(training_dir) / "fit_overfit_summary.csv"), models, destination)
    _residual_acf(bundle, models, destination)
    return {"files": [stem + "." + ext for stem in FIGURE_CAPTIONS for ext in ("svg", "png")],
            "captions": dict(FIGURE_CAPTIONS), "fixed_neural_seed": FIXED_SEED}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    result = generate_figures(args.input, args.statistics, args.training, args.destination)
    print(json.dumps({"files": result["files"], "fixed_neural_seed": FIXED_SEED}))
