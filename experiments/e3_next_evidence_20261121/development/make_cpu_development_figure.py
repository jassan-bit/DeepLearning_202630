"""Render one publication figure from the completed frozen development CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SOURCE_SHA = "f933b7b3e99e89170f38f29eea045219e9d9d3b591b2632a1baddf3605e06987"
ORDER = ("sigma_crps_v4", "u_crps_control", "mse_only_control", "additive_positive_head",
         "floor_only", "floor_plus_mean_excess")
LABELS = ("V4 · CRPS de σ", "Control V3 · CRPS de U", "Control MSE-only", "Control aditivo",
          "Control sólo piso", "Control piso + exceso medio")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.summary) != SOURCE_SHA or args.output.exists():
        raise ValueError("Pinned completed summary and fresh figure directory required")
    with args.summary.open(newline="") as stream:
        rows = {row["name"]: row for row in csv.DictReader(stream)}
    if set(rows) != set(ORDER):
        raise ValueError("Six declared recipes required")
    means = [1000 * float(rows[name]["mean_RMSE"]) for name in ORDER]
    sd = [1000 * float(rows[name]["sample_SD_RMSE"]) if rows[name]["sample_SD_RMSE"] else 0 for name in ORDER]
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "v4-cpu-development-20261121"})
    fig, axis = plt.subplots(figsize=(10.2, 6.2), constrained_layout=False)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.79, bottom=0.29)
    colors = ["#B66042"] + ["#537D91"] * 5
    axis.barh(range(6), means, color=colors, edgecolor="none", height=0.62)
    # Deterministic controls have no estimated seed SD: do not draw zero-length
    # error caps for them, which could visually suggest an estimated zero SD.
    axis.errorbar(means[:4], range(4), xerr=sd[:4], fmt="none", ecolor="#173D4F",
                  capsize=3, elinewidth=1.1)
    axis.set_yticks(range(6), LABELS, fontsize=9); axis.invert_yaxis()
    axis.set_xlim(0, 3.2); axis.set_xlabel("RMSE individual media × 10⁻³ (σ₇ sin anualizar)", fontsize=9)
    axis.xaxis.grid(True, alpha=0.18); axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    for index, mean in enumerate(means):
        axis.text(mean + sd[index] + 0.045, index, f"{mean:.3f}", va="center", fontsize=9)
    fig.text(0.05, 0.94, "Piloto CPU de desarrollo · 20 ajustes nuevos", fontsize=15, weight="bold", color="#173D4F")
    fig.text(0.05, 0.89, "55 fechas ya expuestas · cinco semillas por receta neuronal · dos controles analíticos", fontsize=9)
    fig.legend(handles=[Patch(color=colors[0], label="V4 evaluada"), Patch(color=colors[1], label="Controles existentes")],
               loc="upper left", bbox_to_anchor=(0.29, 0.865), frameon=False, ncol=2, fontsize=8)
    fig.text(0.05, 0.18, "V4 tuvo 1,443 % más RMSE que MSE-only; no superó la regla previa del 2 % de reducción.", fontsize=9, color="#8B3A29")
    fig.text(0.05, 0.13, "Decisión: abandonar esta receta. No se promueve automáticamente un control.", fontsize=9, color="#8B3A29")
    fig.text(0.05, 0.075, "Barras: media de RMSE individual. Bigotes: DE entre cinco semillas, no intervalos de confianza.", fontsize=8)
    fig.text(0.05, 0.04, "Analíticos: una trayectoria; sin DE estimada. Evidencia de desarrollo, sin confirmación o R3 acreditada.", fontsize=8)
    args.output.mkdir(parents=True)
    targets = [args.output / "v4_cpu_development_rmse.png", args.output / "v4_cpu_development_rmse.svg"]
    fig.savefig(targets[0], dpi=180, facecolor="white")
    fig.savefig(targets[1], facecolor="white", metadata={"Date": None})
    plt.close(fig)
    report = {"kind": "one_posthoc_publication_figure_from_frozen_completed_pilot", "matplotlib": matplotlib.__version__,
              "source_recipe_summary_sha256": SOURCE_SHA, "error_bars": "sample_SD_across5_paired_seeds_not_CI",
              "analytics": "one deterministic trajectory each; no fake seed replication",
              "new_model_fits_or_forwards": 0, "R3_accredited": False,
              "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)} for path in targets],
              "AI_assistance_declared": True}
    (args.output / "figure_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
