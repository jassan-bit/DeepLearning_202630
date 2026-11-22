"""Create a scientific figure from a completed, unchanged validation report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text())
    if report["status"] != "completed":
        raise ValueError("A complete report is required; incomplete results are not plotted as final")
    rows = report["scenarios"]
    labels = ["Nulo iid", "Nulo AR(0,5)", "Frontera principal", "Fronteras 2/3",
              "Dependencia negativa", "Cancelación condicional", "Cambio de varianza",
              "Inversión de orden", "Control favorable"]
    if len(rows) != len(labels):
        raise ValueError("This figure documents exactly the nine frozen scenarios")
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1.1]})
    null_rows = [(i, row) for i, row in enumerate(rows) if row["kind"] == "null"]
    for i, row in null_rows:
        point = row["global_false_acceptance_rate"]
        upper = row["global_type_I_upper_simultaneous_MC"]
        axes[0].plot([point, upper], [i, i], color="#17638c", linewidth=2)
        axes[0].plot(point, i, "o", color="#17638c", markersize=5)
        axes[0].plot(upper, i, "|", color="#17638c", markersize=11)
    axes[0].axvline(.05, color="#a1272f", linestyle="--", linewidth=1.3)
    axes[0].set(yticks=np.arange(8), yticklabels=labels[:8],
                xlabel="Frecuencia y límite superior MC", title="Aceptación conjunta falsa")
    axes[0].invert_yaxis()
    axes[0].set_xlim(-.002, max(.06, max(row["global_type_I_upper_simultaneous_MC"]
                                         for _, row in null_rows) * 1.08))
    for offset, key, color, marker, legend in (
        (-.13, "CI99_rmse_difference", "#17638c", "o", "IC99: diferencia de RMSE"),
        (.13, "CI99_relative_rmse_gain", "#a35a00", "s", "IC99: ganancia relativa"),
    ):
        values = [row["coverage"][key]["lower_simultaneous_MC"] for row in rows]
        axes[1].scatter(values, np.arange(9) + offset, c=color, marker=marker,
                        label=legend, s=30)
    axes[1].axvline(.95, color="#a1272f", linestyle="--", linewidth=1.3)
    axes[1].set(yticks=np.arange(9), yticklabels=labels,
                xlabel="Límite inferior MC de cobertura", title="Intervalos nominales del 99 %")
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].set_xlim(min(.90, min(row["coverage"][key]["lower_simultaneous_MC"]
        for row in rows for key in ("CI99_rmse_difference", "CI99_relative_rmse_gain")) - .01), 1.002)
    for ax in axes:
        ax.grid(axis="x", color="#dedede", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Validación sintética: media de RMSE y regla principal + 2/3", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(args.report, args.output)
