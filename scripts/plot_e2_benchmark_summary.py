#!/usr/bin/env python3
"""Present published benchmark ranks, Holm decisions and GPU latency points.

This script only reads four pinned result tables. It does not load market data,
weights or predictions and performs no fitting, inference or statistical test.
Existing outputs must be byte-identical; otherwise choose a fresh directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e2-summary-plots-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "e2-summary-plots-xdg"))
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-published-benchmark-summary-plots-v1"
SOURCES = {
    "book_e2/_static/e2_f2a_s8_20261113/conditional_seed_ranks.csv":
        "40229ce75e8449c2829740efbd3d62b70f1900c9cbb48f42ce4fdcba34612c4b",
    "book_e2/_static/e2_f2a_s8_20261113/conditional_wilcoxon_holm78.csv":
        "1c2c399eb925636ced5db6b7ea3c953bee72286d653e7b1be32e096f4b7007ed",
    "book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv":
        "879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1",
    "book_e2/_static/e2_posthoc_gpu_20261113/inference_profiles_by_model.csv":
        "9a517bdc60282fa6d2362477f2ec5799451cd0b74d96c88ab188b2ff66a1fd1f",
}
RANKS, HOLM, A2, PROFILES = tuple(SOURCES)
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
SAE = "LSTM-autoencoder con predictor"
TEAL, BLUE, GREY = "#087f82", "#176ba0", "#7c8c98"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def records(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def number(row, name):
    value = float(row[name])
    require(math.isfinite(value) and value > 0, "Invalid positive published value: " + name)
    return value


def save_figure(fig):
    products = {}
    for extension in ("png", "svg"):
        buf = io.BytesIO()
        metadata = {"Software": "Matplotlib"} if extension == "png" else {"Creator": "Matplotlib", "Date": None}
        fig.savefig(buf, format=extension, dpi=150, metadata=metadata)
        raw = buf.getvalue()
        if extension == "svg":
            raw = ("\n".join(line.rstrip() for line in raw.decode().splitlines()) + "\n").encode()
        products[extension] = raw
    plt.close(fig)
    return products


def draw_rank_groups(rank_rows):
    """Draw already published mean ranks and the already published decisions."""
    ordered = sorted(enumerate(rank_rows), key=lambda item: (number(item[1], "mean_rank"), item[0]))
    rows = [row for _, row in ordered]
    fig = plt.figure(figsize=(13, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1);ax.set_ylim(0, 1);ax.axis("off")
    ax.text(.5, .985, "Rangos medios publicados y agrupación de no rechazo", ha="center", va="top", fontsize=16)
    ax.text(.5, .925, "13 redes · Wilcoxon exacto bilateral · ajuste de Holm de 78 comparaciones", ha="center", fontsize=11)
    x_rank = lambda rank: .34 + .38 * (rank - 1) / 12
    axis_y = .81
    ax.plot([x_rank(1), x_rank(13)], [axis_y, axis_y], color="#263b49", lw=1)
    for rank in range(1, 14):
        xpos = x_rank(rank)
        ax.plot([xpos, xpos], [axis_y - .008, axis_y + .008], color="#263b49", lw=.9)
        ax.text(xpos, axis_y + .02, str(rank), ha="center", fontsize=10)
    ax.text(.16, .81, "Rango menor = menor RMSE", ha="center", fontsize=9, color="#415563")
    # The source contains the complete 78-pair family, with every adjusted p=1.
    # This single bar reports that decision pattern, not a common CD threshold.
    group_min, group_max = min(number(row, "mean_rank") for row in rows), max(number(row, "mean_rank") for row in rows)
    ax.plot([x_rank(group_min), x_rank(group_max)], [.868, .868], color=TEAL, lw=5, solid_capstyle="butt")
    ax.text(.53, .89, "Un grupo de no rechazo: todos los p de Holm = 1", ha="center", fontsize=10, color=TEAL)
    for index, row in enumerate(rows):
        left = index < 7
        row_number = index if left else index - 7
        ypos = .70 - .075 * row_number
        xpos = x_rank(number(row, "mean_rank"))
        edge = .30 if left else .76
        ax.plot([xpos, xpos, edge], [axis_y - .012, ypos, ypos], color=GREY, lw=.85, zorder=1)
        ax.plot(xpos, axis_y, "o", color=BLUE, ms=3.5, zorder=3)
        label = row["model"] + "  ·  " + f"{number(row, 'mean_rank'):.2f}".replace(".", ",")
        ax.text(edge - .01 if left else edge + .01, ypos, label,
                ha="right" if left else "left", va="center", fontsize=10)
    ax.text(.5, .13, "Seis vectores de RMSE por semilla sobre una misma cronología de 608 fechas: alcance condicional.",
            ha="center", fontsize=9)
    ax.text(.5, .095, "Con seis pares, el mínimo p bilateral exacto es 0,03125; no alcanza el primer umbral de Holm para 78 comparaciones.",
            ha="center", fontsize=9)
    ax.text(.5, .06, "El no rechazo no demuestra equivalencia. No se ejecuta Nemenyi ni se atribuye una distancia crítica numérica común.",
            ha="center", fontsize=9, color="#415563")
    return save_figure(fig)


def draw_pareto(points):
    fig, ax = plt.subplots(figsize=(12.8, 7.8))
    fig.subplots_adjust(left=.095, right=.955, bottom=.225, top=.82)
    fig.suptitle("Benchmark Bitcoin · RMSE frente a latencia GPU publicada", fontsize=17, y=.975)
    fig.text(.5, .925, "Frontera puntual observada entre 13 redes; ambos objetivos se minimizan", ha="center", fontsize=11)
    frontier = sorted((point for point in points if point["observed_nondominated"]),
                      key=lambda point: number(point, "mean_latency_gpu_median_ms"))
    ax.plot([number(point, "mean_latency_gpu_median_ms") for point in frontier],
            [number(point, "RMSE") for point in frontier], color=TEAL, lw=1.8, zorder=2)
    for point in points:
        nondominated = point["observed_nondominated"]
        ax.scatter(number(point, "mean_latency_gpu_median_ms"), number(point, "RMSE"),
                   s=72 if nondominated else 48, marker="D" if nondominated else "o",
                   c=TEAL if nondominated else GREY, edgecolors="white", linewidths=.8, zorder=4)
    # Fixed display offsets keep every full model name visible without changing data.
    offsets = {
        "MLP16": (-18, -33), "GRU32": (-43, 20), "LSTM32": (10, 14), SAE: (15, 18),
        "FCN-1D": (10, 13), "TiDE": (-8, 17), "iTransformer": (-32, 22),
        "ResNet-1D": (-60, -23), "TSMixer": (13, 16), "Inception-1D": (-37, -42),
        "PatchTST": (10, 0), "TimeMixer": (13, -21), "TimesNet": (7, 12),
    }
    for point in points:
        model = point["model"]
        label = "LSTM-autoencoder\ncon predictor" if model == SAE else model
        ax.annotate(label, (number(point, "mean_latency_gpu_median_ms"), number(point, "RMSE")),
                    xytext=offsets[model], textcoords="offset points", fontsize=9.5,
                    color="#155d60" if point["observed_nondominated"] else "#283e4b",
                    arrowprops={"arrowstyle": "-", "color": "#9eabb3", "lw": .65},
                    ha="left", va="center", zorder=5)
    ax.set_xscale("log")
    ax.set_xticks([.3, .5, 1, 2, 5, 10])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(.255, 12)
    ax.set_ylim(.0031, .0109)
    ax.set_xlabel("Media de seis medianas de latencia GPU (ms, escala logarítmica)", fontsize=11, labelpad=12)
    ax.set_ylabel("RMSE de σ₇ en unidades originales", fontsize=11, labelpad=10)
    ax.grid(alpha=.2, linewidth=.6)
    ax.legend(handles=[Line2D([], [], color=TEAL, marker="D", markersize=7, label="No dominado observado"),
                       Line2D([], [], color=GREY, marker="o", linestyle="", markersize=7, label="Dominado observado")],
              loc="upper left", frameon=False, fontsize=10)
    fig.text(.5, .135, "Latencia: eventos CUDA con entrada residente; excluye transferencia y preprocesamiento. A100 según el resumen del expediente.",
             ha="center", fontsize=9)
    fig.text(.5, .10, "Se copian los RMSE de A2 y la media de seis medianas registradas: no se miden latencias nuevas ni se agrupan muestras individuales.",
             ha="center", fontsize=9)
    fig.text(.5, .065, "Dominancia determinista de estos 13 puntos, sin inferencia sobre la frontera. Tres comparadores clásicos sin perfil GPU comparable: excluidos.",
             ha="center", fontsize=9, color="#415563")
    return save_figure(fig)


def build(root=ROOT):
    tables = {}
    for name, expected in SOURCES.items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), "Missing regular input: " + name)
        raw = path.read_bytes()
        require(sha(raw) == expected, "Pinned input changed: " + name)
        tables[name] = records(raw)
    ranks, pairs, a2, profiles = (tables[name] for name in SOURCES)
    rank_models = {row["model"] for row in ranks}
    require(len(ranks) == len(rank_models) == 13, "Expected 13 unique mean-rank records")
    require(len(a2) == 16 and {row["model"] for row in a2} == rank_models | set(CLASSICAL),
            "Published A2 catalogue differs")
    require(len(profiles) == 13 and {row["model"] for row in profiles} == rank_models,
            "GPU profile catalogue differs")
    require(len(pairs) == 78, "Expected 78 published pairwise decisions")
    pair_keys, degree = set(), Counter()
    for pair in pairs:
        first, second = pair["model_A"], pair["model_B"]
        require(first in rank_models and second in rank_models and first != second, "Unregistered pair")
        pair_keys.add(frozenset((first, second)));degree.update((first, second))
        require(pair["status"] == "exact" and pair["p_Holm78"] == "1.0"
                and pair["reject_Holm05"] == "False", "Published Holm pattern differs")
        number(pair, "p_raw")
    require(len(pair_keys) == 78 and set(degree.values()) == {12}, "Pair family is not complete and unique")
    require(min(number(row, "p_raw") for row in pairs) == .03125, "Expected exact six-pair lower bound")
    for row in ranks:
        require(1 <= number(row, "mean_rank") <= 13, "Mean rank outside published catalogue")
    a2_by_model = {row["model"]: row for row in a2}
    profiles_by_model = {row["model"]: row for row in profiles}
    models = [row["model"] for row in a2 if row["model"] in rank_models]
    points = []
    for model in models:
        score, profile = a2_by_model[model], profiles_by_model[model]
        require(score["R"] == "6" and score["n_common_dates"] == "608", "Unexpected A2 scope")
        require(profile["R_available"] == profile["R_planned"] == profile["n_observed_latency_gpu_median_ms"] == "6"
                and profile["all_six_seeds_available"] == "True", "Six recorded latency medians required")
        require(profile["aggregation_scope"] == "mean_of_recorded_seed_level_metrics_not_pooled_latency_samples"
                and json.loads(profile["recorded_latency_scopes"]) == ["CUDA_event_model_forward_resident_input_excludes_transfer_preprocessing"],
                "Latency measurement scope differs")
        number(score, "RMSE");number(profile, "mean_latency_gpu_median_ms")
        points.append({"model": model, "RMSE": score["RMSE"],
                       "mean_latency_gpu_median_ms": profile["mean_latency_gpu_median_ms"],
                       "observed_nondominated": None, "dominated_by_models": [], "R": "6",
                       "RMSE_source": A2, "latency_source": PROFILES,
                       "latency_aggregation": "mean_of_six_recorded_seed_medians"})
    for point in points:
        x, y = number(point, "mean_latency_gpu_median_ms"), number(point, "RMSE")
        dominators = [other["model"] for other in points if other is not point
                      and number(other, "mean_latency_gpu_median_ms") <= x and number(other, "RMSE") <= y
                      and (number(other, "mean_latency_gpu_median_ms") < x or number(other, "RMSE") < y)]
        point["observed_nondominated"] = not dominators
        point["dominated_by_models"] = dominators
    frontier = sorted((point for point in points if point["observed_nondominated"]),
                      key=lambda point: number(point, "mean_latency_gpu_median_ms"))
    require([point["model"] for point in frontier] == ["MLP16", "GRU32", SAE], "Pinned point frontier differs")
    outputs = {}
    with plt.rc_context({"svg.hashsalt": VERSION, "font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "path.simplify": False}):
        for name, products in (("conditional_rank_groups", draw_rank_groups(ranks)), ("latency_pareto", draw_pareto(points))):
            for extension, raw in products.items():
                outputs[name + "." + extension] = raw
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(points[0]), lineterminator="\n")
    writer.writeheader()
    for point in points:
        writer.writerow({**point, "dominated_by_models": json.dumps(point["dominated_by_models"], ensure_ascii=False)})
    outputs["latency_pareto.csv"] = buf.getvalue().encode()
    report = {
        "version": VERSION, "scope": "new_presentation_of_pinned_published_aggregates_no_new_experiment",
        "input_sha256": SOURCES, "script_sha256": sha(Path(__file__).read_bytes()),
        "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "conditional_rank_groups": {
            "models": 13, "seed_vectors": 6, "common_dates_per_vector": 608,
            "mean_ranks_copied_from": RANKS, "pair_decisions_copied_from": HOLM,
            "published_pair_count": 78, "published_all_p_Holm78": 1.0, "published_reject_count": 0,
            "nonrejection_group": models, "single_numeric_critical_difference": None,
            "Nemenyi_executed": False, "new_pairwise_tests": 0,
            "interpretation": "One group displays the complete published Holm nonrejection pattern, not equivalence or a common critical-distance threshold.",
            "limits": "Six seed-level vectors share one chronology; conditional algorithmic variability does not create six independent temporal test datasets. Exact bilateral p minimum for six nonzero pairs is 0.03125, above Holm's first threshold for 78 comparisons.",
        },
        "latency_pareto": {
            "models": 13, "source_RMSE_field": "RMSE", "source_latency_field": "mean_latency_gpu_median_ms",
            "source_values_copied_exactly_as_csv_strings": True,
            "latency_scope": "Mean of six saved seed-level medians of CUDA-event model-forward resident-input latency; transfer and preprocessing excluded. Not a pooled latency distribution or end-to-end latency.",
            "hardware_scope": "NVIDIA A100-SXM4-80GB from verified root summary; individual profile receipts do not record device name.",
            "dominance_rule": "A point is dominated when another published point is no worse in both objectives and strictly better in at least one. Both RMSE and latency are minimized.",
            "frontier_in_increasing_latency_order": [point["model"] for point in frontier],
            "classical_comparators_excluded": list(CLASSICAL), "classical_exclusion_reason": "No comparable published GPU profile; missing latency is not zero.",
            "limits": "Observed point frontier only; no confidence region or statistical superiority claim about the frontier. Original RMSE and latency aggregation scopes are retained.",
        },
        "operations": {"source_result_tables_read": 4, "published_numeric_strings_copied": True,
                       "new_operations": ["deterministic pointwise dominance comparisons", "figure layout and rendering", "hashes and coverage checks"],
                       "market_data_archives_opened": 0, "prediction_arrays_loaded": 0, "checkpoint_weights_loaded": 0,
                       "model_fits": 0, "model_inference_calls": 0, "predictive_scores_computed": 0,
                       "latency_measurements": 0, "bootstrap_resamples": 0, "statistical_tests_computed": 0,
                       "published_source_artifacts_modified": False},
        "drawing": {"SVG_Date_omitted": True, "stable_svg_hashsalt": VERSION, "latency_log_x_axis": True,
                    "rank_values_recomputed": False, "new_error_bars_or_bands": False},
        "output_sha256": {name: sha(raw) for name, raw in outputs.items()},
    }
    outputs["summary_plots.json"] = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/e2-benchmark-summary-plots-v1"))
    args = parser.parse_args(argv)
    require(not args.output.is_symlink(), "Output directory cannot be a symlink")
    output = args.output.resolve();products = build()
    for name, raw in products.items():
        target = output / name
        require(not target.is_symlink(), "Output file cannot be a symlink")
        require(not target.exists() or target.is_file() and target.read_bytes() == raw,
                "Refusing changed existing output; choose a new directory: " + name)
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in products.items():
        target = output / name
        if not target.exists():
            target.write_bytes(raw)
    print(json.dumps({"version": VERSION, "products": len(products), "models": 13,
                      "published_Holm_decisions": 78, "new_model_fits": 0,
                      "new_statistical_tests": 0, "frontier": ["MLP16", "GRU32", SAE]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
