"""Static scientific figures, preserving uncertainty units in titles/labels."""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/e3-statistics-matplotlib")


def make_figures(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "e3-stats-20261117", "font.size": 10})
    original = "CNN-LSTM-atencion-aditiva E3"
    rows = sorted(result["metrics"], key=lambda row: row["RMSE_mean"])
    labels = ["CNN-LSTM atención + salida aditiva (E3)" if r["model"] == original else r["model"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, row in enumerate(rows):
        color = "#176b47" if row["model"] == original else "#355c88"
        if row["RMSE_sample_SD"] is None:
            ax.plot(row["RMSE_mean"], i, "D", color=color)
        else:
            ax.errorbar(row["RMSE_mean"], i, xerr=row["RMSE_sample_SD"], fmt="o", capsize=3, color=color)
    ax.set_yticks(range(len(rows)), labels); ax.invert_yaxis()
    ax.set_xlabel("Media de RMSE individual (unidades originales de σ₇)")
    ax.set_title("608 fechas conocidas: punto y DE muestral entre semillas\nRedes: R=10; clásicos: R=1, sin barra de dispersión")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"ranking_rmse.svg", metadata={"Date": None}); plt.close(fig)
    effects = sorted(result["effects"], key=lambda row: row["gain_RMSE"])
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, row in enumerate(effects):
        lo, hi = row["gain_ci95_temporal"] or [np.nan, np.nan]
        ax.hlines(i, lo, hi, color="#355c88", linewidth=2)
        ax.plot(row["gain_RMSE"], i, "o", color="#176b47")
    delta = result["primary_contrasts"][0]["delta"]
    ax.axvline(0, color="black", linewidth=.8)
    ax.axvline(delta, color="#b14421", linestyle="--", label=f"Margen heredado δ={delta:.8f}")
    ax.set_yticks(range(len(effects)), [r["rival"] for r in effects]); ax.invert_yaxis()
    ax.set_xlabel("Ganancia: RMSE rival − RMSE E3 (unidades originales de σ₇)")
    ax.set_title("Ganancias e IC95% temporales marginales\nBootstrap circular pareado, bloque 14; trayectorias ajustadas fijas")
    ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"gains_ci.svg", metadata={"Date": None}); plt.close(fig)
