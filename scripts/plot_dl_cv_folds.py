"""Plot the actual admitted outer CV indices, without fitting or raw reads."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/dl-cv-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dl-cv-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/dl_r2_cv_20261104"


def main():
    folds = json.loads((RESULTS / "folds.json").read_text())["folds"]
    execution = json.loads((RESULTS / "execution_summary.json").read_text())
    if execution["actual_cv_folds"] != len(folds) or execution["fold_model_segments"] != 720:
        raise ValueError("Expected the completed twelve-block experiment")
    calendar = pd.date_range("2023-01-01", "2024-12-31")
    matrix = np.zeros((len(folds), len(calendar)), dtype=int)
    for row, f in enumerate(folds):
        matrix[row, f["train_indices"]] = 1
        matrix[row, f["outer_training_last_allowed_index"]+1:f["fit_origin_index"]] = 2
        matrix[row, f["test_indices"]] = 3
    colors = ["#ffffff", "#3d6fa6", "#a7afb9", "#d9852d"]
    figure, axis = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    first = mdates.date2num(calendar[0].to_pydatetime())
    last = mdates.date2num((calendar[-1]+pd.Timedelta(days=1)).to_pydatetime())
    axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=ListedColormap(colors),
                vmin=0, vmax=3, extent=[first, last, len(folds)-.5, -.5])
    axis.set_yticks(np.arange(len(folds)), [f["fold_id"] for f in folds])
    axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    axis.set_xlabel("Calendario UTC; se conservan los huecos excluidos")
    axis.set_ylabel("Fold mensual realmente ajustado")
    axis.set_title("Forward chaining de timeseries-cv: entrenamiento creciente y validación mensual\n"
                   "Pesos fijos dentro del mes; predicción diaria h=1; embargo de 7 días")
    axis.legend(handles=[Patch(color=colors[1], label="Entrenamiento permitido (refit)"),
        Patch(color=colors[2], label="Embargo"), Patch(color=colors[3], label="Validación 2024")],
        loc="upper left", bbox_to_anchor=(0, -.12), ncol=3)
    figure.text(.5, -.035, "Índices: timeseries-cv 0.1.5 + filtros explícitos | Datos: Binance Vision BTCUSDT | Procesamiento: Jassan Arteta",
                ha="center", fontsize=9)
    target = RESULTS / "analysis/figures/cv_folds_actual.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
