"""Draw the actual monthly Filipe calendar from signed index metadata only."""
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e1-cv-calendar-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/e1-cv-calendar-cache")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "book/_static/e1_cv_calendar_20261104"
HISTORIES = (7, 14, 21, 28)


def day(value):
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def interval(start, finish):
    """Translate inclusive day labels to a left/right plotting interval."""
    left = mdates.date2num(day(start))
    right = mdates.date2num(day(finish) + timedelta(days=1))
    return left, right-left


def main():
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "svg.hashsalt": "e1-filipe-calendar-20261104"})
    fig = plt.figure(figsize=(14, 11.5))
    grid = fig.add_gridspec(3, 2, height_ratios=(.48, 1, 1), hspace=.40, wspace=.22)
    overview = fig.add_subplot(grid[0, :])
    colors = {"train": "#30638e", "gap": "#e49b0f", "cv": "#2d936c", "test": "#7651a2"}
    periods = [("2023-01-01", "2023-12-31", "Historia inicial de ajuste · 365 días", "train"),
               ("2024-01-01", "2024-12-31", "CV forward · 366 días", "cv"),
               ("2025-01-01", "2026-08-31", "Test de pesos finales congelados · 608 días", "test")]
    for start, finish, label, key in periods:
        left, width = interval(start, finish)
        overview.broken_barh([(left, width)], (.3, .6), facecolors=colors[key])
        overview.text(left+width/2, .6, label, ha="center", va="center", color="white", fontsize=9)
    overview.set_xlim(day("2023-01-01"), day("2026-09-01"))
    overview.set_ylim(0, 1.3)
    overview.set_yticks([])
    overview.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    overview.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y", tz=timezone.utc))
    overview.set_title("Calendario asignado UTC · porcentajes descriptivos de duración, no proporciones impuestas por Filipe", fontsize=11)
    for side in ("left", "right", "top"):
        overview.spines[side].set_visible(False)

    for number, history in enumerate(HISTORIES):
        ax = fig.add_subplot(grid[1+number//2, number%2])
        source = (ROOT / "results/dl_r2_cv_20261104/folds.json" if history == 7 else
                  ROOT / f"results/filipe_multihistory_20261104/D{history}/folds.json")
        manifest = json.loads(source.read_text())
        folds = manifest["folds"]
        if len(folds) != 12 or manifest["history_days"] != history:
            raise ValueError("Expected twelve signed monthly blocks for each D")
        for row, fold in enumerate(folds):
            origin = day(fold["fit_origin_day_utc"])
            last_train = day(fold["train"]["last_day"])
            if (origin-last_train).days != history+1:
                raise ValueError("The plotted train/validation gap differs from the signed D purge")
            ax.broken_barh([interval(fold["train"]["first_day"], fold["train"]["last_day"])],
                           (row-.32, .64), facecolors=colors["train"])
            gap_start = origin-timedelta(days=history)
            ax.broken_barh([(mdates.date2num(gap_start), history)], (row-.32, .64), facecolors=colors["gap"])
            ax.broken_barh([interval(fold["test"]["first_day"], fold["test"]["last_day"])],
                           (row-.32, .64), facecolors=colors["cv"])
        ax.set_yticks(range(12), [fold["fold_id"] for fold in folds])
        ax.set_ylim(11.7, -.7)
        ax.set_xlim(day("2023-01-01"), day("2025-01-01"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y", tz=timezone.utc))
        ax.grid(axis="x", alpha=.18)
        ax.set_title(f"D={history} · doce reajustes mensuales por modelo y semilla", fontsize=10)
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)

    legend = [Patch(facecolor=colors[k], label=label) for k, label in
              [("train", "Periodo del entrenamiento mensual admitido"), ("gap", "Etiquetas purgadas: D días"),
               ("cv", "Validación mensual: pronósticos diarios h=1"), ("test", "Test: mismos pesos finales, sin reajustes")]]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .034), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("CV de Filipe realmente ejecutada · cuatro entradas, objetivo σ7 y horizonte de un día", fontsize=14, y=.98)
    fig.text(.5, .016, "Cada fila es un bloque mensual; las diez semillas y tres redes comparten esos cortes. Las barras delimitan periodos: los huecos y sus ventanas se excluyen.\nEl embargo excluye etiquetas del ajuste; el OHLCV de esos días ya conocidos puede formar la entrada. Septiembre–octubre de 2026 sigue fuera del protocolo.",
             ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(top=.93, bottom=.11)
    DEST.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST / "cv_calendar.svg", metadata={"Date": None})
    svg_path = DEST / "cv_calendar.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines())+"\n")
    fig.savefig("/tmp/e1_cv_calendar_20261104.png", dpi=110)
    plt.close(fig)
    print(json.dumps({"svg": str((DEST / "cv_calendar.svg").relative_to(ROOT)), "histories": list(HISTORIES),
                      "monthly_blocks_drawn": 48, "raw_market_reads": 0, "new_model_fits": 0}))


if __name__ == "__main__":
    main()
