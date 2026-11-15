#!/usr/bin/env python3
"""Present nine published calibration summaries and nine existing temporal CIs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e2-calibration-communication-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-saved-calibration-communication-v1"
SOURCES = {
    "book_e2/_static/e2_conditional_calibration_20261114/model_metrics.csv":
        "463bfe0466764330cf35d91cd44f0ca637684cf1ce00e83c73286aac8e0b1d22",
    "book_e2/_static/e2_conditional_calibration_20261114/conditional_coverage_CI.csv":
        "6349ec7fb84ce5b8397036702d4c0d0659b9d7fc33e61b8add3213a665bf2844",
}
MODELS = (("LSTM-autoencoder con predictor", "SAE"), ("GRU32", "GRU32"), ("ARIMA", "ARIMA"))
POLICIES = (("original_frozen", "Original congelado", "#285a8e", "o"),
            ("rolling53_fixed_alpha", "Móvil 53, α=0,10", "#b5672a", "s"),
            ("ACI53_gamma005", "ACI53, γ=0,005", "#21856b", "D"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    tables = []
    for name, expected in SOURCES.items():
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), "Regular input required: " + name)
        raw = path.read_bytes()
        require(sha(raw) == expected, "Pinned input changed: " + name)
        tables.append(list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))))
    metrics, intervals = tables
    readout = []
    for model, label in MODELS:
        for policy, policy_label, _, _ in POLICIES:
            selected = [row for row in metrics if row["model"] == model and row["policy"] == policy and row["group"] == "global"]
            selected_ci = [row for row in intervals if row["model"] == model and row["policy"] == policy
                           and row["group"] == "global" and row["block_days"] == "14"]
            require(len(selected) == len(selected_ci) == 1, "Exactly one matching published summary and CI required")
            value, ci = selected[0], selected_ci[0]
            require(value["coverage"] == ci["coverage"] and value["calendar_days_with_membership"] == "608"
                    and ci["draws"] == "10000" and ci["undefined_bootstrap_draws"] == "0"
                    and ci["CI_status"] == "marginal_temporal_percentile"
                    and ci["resampling_scope"] == "full608calendar_joint_methods_and_seeds_fixed"
                    and value["width_status"] == "finite", "Published uncertainty scope differs")
            require(value["R"] == ("1" if model == "ARIMA" else "6"), "Published predictor count differs")
            require(0 <= float(ci["CI95_low"]) <= float(value["coverage"]) <= float(ci["CI95_high"]) <= 1,
                    "Coverage outside recorded interval")
            readout.append({"model": model, "model_label": label, "policy": policy, "policy_label": policy_label,
                            **{name: value[name] for name in ("coverage", "mean_width", "R")},
                            **{name: ci[name] for name in ("CI95_low", "CI95_high", "draws", "block_days", "CI_status", "resampling_scope")},
                            "width_interval_available": False})
    with plt.rc_context({"svg.hashsalt": VERSION, "font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.3))
        fig.subplots_adjust(left=.075, right=.975, top=.745, bottom=.25, wspace=.28)
        fig.suptitle("Calibración temporal · cobertura y anchura del Top 3 observado", fontsize=16, y=.975)
        fig.text(.5, .925, "SAE y GRU32: seis predictores fijos · ARIMA: una trayectoria · mismo calendario de 608 fechas", ha="center", fontsize=10)
        for index, (policy, label, color, marker) in enumerate(POLICIES):
            for model_index, (model, _) in enumerate(MODELS):
                row = next(row for row in readout if row["model"] == model and row["policy"] == policy)
                at = model_index + (index - 1) * .21
                axes[0].vlines(at, 100 * float(row["CI95_low"]), 100 * float(row["CI95_high"]), color=color, lw=1.6)
                axes[0].plot(at, 100 * float(row["coverage"]), marker=marker, color=color, ms=6)
                axes[1].plot(at, float(row["mean_width"]), marker=marker, color=color, ms=6)
        axes[0].axhline(90, color="#394b57", linestyle="--", lw=1)
        axes[0].set_ylim(75, 100);axes[1].set_ylim(0, .020)
        axes[0].set_ylabel("Cobertura (%) e IC95 % temporal marginal")
        axes[1].set_ylabel("Anchura media en unidades originales de σ₇")
        axes[0].set_title("Cobertura: intervalos ya publicados", fontsize=11)
        axes[1].set_title("Anchura: estimaciones puntuales; IC no disponible", fontsize=11)
        for ax in axes:
            ax.set_xticks(range(3), [label for _, label in MODELS]);ax.set_xlim(-.5, 2.5)
            ax.grid(axis="y", alpha=.2);ax.set_axisbelow(True)
        fig.legend([Line2D([], [], color=color, marker=marker, linestyle="", ms=6) for _, _, color, marker in POLICIES],
                   [label for _, label, _, _ in POLICIES], loc="upper center", bbox_to_anchor=(.5, .88), ncol=3, frameon=False)
        fig.text(.5, .15, "IC95 % temporales marginales guardados: B=10.000, bloques de 14 días y semillas fijas; no se promedian IC individuales.", ha="center", fontsize=9)
        fig.text(.5, .105, "La anchura no tiene IC publicado: los puntos no acreditan equivalencia ni superioridad estadística entre políticas.", ha="center", fontsize=9)
        fig.text(.5, .06, "Sólo presentación de resultados existentes; no se recalculan coberturas, anchuras, intervalos ni pruebas.", ha="center", fontsize=9, color="#415563")
        outputs = {}
        for extension in ("png", "svg"):
            buf = io.BytesIO()
            metadata = {"Software": "Matplotlib"} if extension == "png" else {"Creator": "Matplotlib", "Date": None}
            fig.savefig(buf, format=extension, dpi=150, metadata=metadata)
            raw = buf.getvalue()
            if extension == "svg":
                raw = ("\n".join(line.rstrip() for line in raw.decode().splitlines()) + "\n").encode()
            outputs["top3_coverage_width." + extension] = raw
        plt.close(fig)
    report = {"version": VERSION, "scope": "presentation_only_of_published_aggregate_results",
              "input_sha256": SOURCES, "script_sha256": sha(Path(__file__).read_bytes()),
              "software": {"matplotlib": matplotlib.__version__}, "published_rows_copied_exactly": readout,
              "selection": {"models": [model for model, _ in MODELS], "policies": [policy for policy, _, _, _ in POLICIES],
                            "group": "global", "coverage_CI_block_days": "14", "row_count": 9},
              "uncertainty": {"coverage": "Existing marginal temporal percentile CI; 10000 joint calendar-block draws, 14-day blocks, fixed seeds and realized adaptive paths; not simultaneous and not a new run of ACI feedback.",
                              "width": "Only published point means available. No CI, zero error bar, SD or equivalence claim is invented.",
                              "individual_CIs_averaged": False, "CI_endpoints_recomputed": False},
              "operations": {"source_CSV_tables_read": 2, "display_only_coverage_fraction_to_percent": True,
                             "new_model_fits": 0, "new_model_inference": 0, "new_latency_measurements": 0,
                             "new_predictive_scores": 0, "new_bootstrap_resamples": 0, "new_statistical_tests": 0,
                             "published_sources_modified": False},
              "drawing": {"SVG_Date_omitted": True, "stable_svg_hashsalt": VERSION, "width_error_bars": False},
              "output_sha256": {name: sha(raw) for name, raw in outputs.items()}}
    outputs["calibration_plot_readout.json"] = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/e2-calibration-communication-v1"))
    args = parser.parse_args()
    require(not args.output.is_symlink(), "Output directory cannot be a symlink")
    output = args.output.resolve();products = build()
    for name, raw in products.items():
        target = output / name
        require(not target.is_symlink() and (not target.exists() or target.is_file() and target.read_bytes() == raw),
                "Refusing changed existing output; choose a fresh directory: " + name)
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in products.items():
        target = output / name
        if not target.exists():
            target.write_bytes(raw)
    print(json.dumps({"version": VERSION, "copied_summary_rows": 9, "copied_existing_CIs": 9, "products": len(products)}))


if __name__ == "__main__":
    main()
