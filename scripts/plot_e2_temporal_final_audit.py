"""Render saved GR diagnostic paths; performs no model fit or inference."""
import argparse
import csv
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def plot_saved_paths(output):
    output = Path(output)
    with (output/"relative_fluctuation_paths.csv").open() as stream:
        paths = list(csv.DictReader(stream))
    models = list(dict.fromkeys(row["model"] for row in paths))
    if len(models) != 10:
        raise ValueError("Ten frozen candidate profiles are required")
    fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharex=True)
    for ax, model in zip(axes.flat, models):
        for baseline, color in (("GRU32", "#0072B2"), ("Persistence", "#D55E00")):
            selected = [row for row in paths if row["model"] == model and row["baseline"] == baseline]
            if len(selected) != 427:
                raise ValueError("Exactly 427 windows per contrast required")
            ax.plot([date.fromisoformat(row["window_end"]) for row in selected],
                    [float(row["GR_EPA_statistic"]) for row in selected],
                    color=color, label=baseline, lw=1.2)
        ax.axhline(0, color="#999999", lw=.7)
        for threshold in (-3.012, 3.012):
            ax.axhline(threshold, color="#555555", ls="--", lw=.7)
        label = model if len(model) < 25 else "LSTM-autoencoder + predictor"
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.supylabel("GR: suma local de pérdida candidato − referencia / (√182 × σ HAC14)", fontsize=10)
    fig.supxlabel("Fin de ventana móvil de 182 días; mismos predictores congelados", fontsize=10, y=.035)
    fig.suptitle("Fluctuación relativa sobre 608 fechas conocidas desde E1", fontsize=14)
    fig.text(.5, .012, "Líneas ±3,012: crítico GR original al 5 % SIN ajustar. Decisiones: bootstrap 20.000 y Holm40.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[.02, .065, 1, .975])
    fig.savefig(output/"GR_fluctuation_paths.svg")
    fig.savefig(output/"GR_fluctuation_paths.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    plot_saved_paths(parser.parse_args().output)
