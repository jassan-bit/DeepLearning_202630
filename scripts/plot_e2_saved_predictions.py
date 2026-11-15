#!/usr/bin/env python3
"""Plot every frozen E2 prediction trajectory; no model or score is executed.

Run from a complete checkout with the versions recorded in the output JSON.
An existing output is accepted only when it is byte-identical.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e2-saved-predictions-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "e2-saved-predictions-xdg"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-saved-prediction-trajectories-v1"
SOURCES = {
    "book_e2/_static/e2_final_audit_inputs_20261114/e2_f2a_input.json":
        "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187",
    "book_e2/_static/e2_benchmark_final_20261113/final_freeze.json":
        "c0e9e2589dca9fa03b341bb72f87d8b541d33b4721715622daeae4438dbd60b4",
    "experiments/e2_reduced_protocol_20261112/protocol.json":
        "867cf6837158c1353a1aa86ad9c51fc6fc260948ffe02dce8e779d3c329fb1cc",
}
CACHE, FREEZE, PROTOCOL = tuple(SOURCES)
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(row):
    return row["model"], str(row["seed"])


def read_sources(root):
    result = {}
    for relative, expected in SOURCES.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "Missing regular source: " + relative)
        raw = path.read_bytes()
        require(digest(raw) == expected, "Pinned source changed: " + relative)
        result[relative] = json.loads(raw)
    return result


def build(root=ROOT):
    source = read_sources(root)
    cache, freeze, protocol = source[CACHE], source[FREEZE], source[PROTOCOL]
    expected_dates = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608)]
    require(expected_dates[-1] == "2026-08-31" and cache["dates"] == expected_dates,
            "Expected every UTC outcome date in the frozen 608-day calendar")
    mask = np.asarray(cache["mask"])
    require(mask.shape == (608,) and mask.dtype.kind == "b" and mask.all(),
            "Expected the complete common boolean mask")
    target = np.asarray(cache["y_true"], dtype=float)
    require(target.shape == (608,) and np.isfinite(target).all() and (target > 0).all(),
            "Expected 608 finite positive observed sigma7 targets")
    systems = protocol["systems"]
    require(len(systems) == len(set(systems)) == 16 and systems == freeze["systems"],
            "System order must match protocol and freeze, not a score ranking")
    seeds = freeze["final_seed_ids"]
    require(seeds == list(range(271801, 271807)), "The six registered neural seeds changed")
    predictors = cache["predictors"]
    frozen = {identity(row): row for row in freeze["models"]}
    require(len(predictors) == len(frozen) == 81, "Expected 81 unique frozen predictors")
    require(len({identity(row) for row in predictors}) == 81
            and {identity(row) for row in predictors} == set(frozen),
            "Saved predictor IDs differ from the freeze")
    grouped = {model: [] for model in systems}
    mapping = []
    for row in predictors:
        key = identity(row)
        require(row["checkpoint_sha256"] == frozen[key]["checkpoint_sha256"],
                "Saved checkpoint identity differs from the freeze")
        values = np.asarray(row["prediction_original"], dtype=float)
        require(values.shape == (608,) and np.isfinite(values).all(),
                "Invalid saved prediction vector")
        grouped[row["model"]].append((row["seed"], values))
        mapping.append({"model": row["model"], "seed": row["seed"],
                        "checkpoint_sha256": row["checkpoint_sha256"], "prediction_count": 608})
    for model in systems:
        members = grouped[model]
        expected = ["deterministic"] if model in CLASSICAL else seeds
        require({str(seed) for seed, _ in members} == {str(seed) for seed in expected}
                and len(members) == len(expected), "Unexpected predictor multiplicity: " + model)
        members.sort(key=lambda member: str(member[0]))
    require(Counter(len(members) for members in grouped.values()) == {6: 13, 1: 3},
            "Expected thirteen neural and three deterministic systems")

    # All 81 original trajectories are drawn individually. No prediction is
    # averaged, smoothed, fitted, ranked or converted into a predictive score.
    dates = [date.fromisoformat(value) for value in expected_dates]
    all_values = [target] + [values for members in grouped.values() for _, values in members]
    lower = min(0., min(float(values.min()) for values in all_values))
    upper = max(float(values.max()) for values in all_values)
    padding = .04 * (upper - lower)
    ticks = [date(2025, 1, 1), date(2025, 7, 1), date(2026, 1, 1), date(2026, 8, 1)]
    with plt.rc_context({"svg.hashsalt": VERSION, "font.family": "DejaVu Sans",
                         "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "path.simplify": False}):
        fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
        for ax, model in zip(axes.flat, systems):
            for _, values in grouped[model]:
                ax.plot(dates, values, color="#2276a2", alpha=.33 if len(grouped[model]) == 6 else .75,
                        linewidth=.68, zorder=2)
            ax.plot(dates, target, color="#242424", linewidth=.75, alpha=.95, zorder=3)
            title = model.replace("LSTM-autoencoder con predictor", "LSTM-autoencoder\ncon predictor")
            ax.set_title(title + " · R=" + str(len(grouped[model])), fontsize=10, pad=5)
            ax.set_xlim(dates[0], dates[-1])
            ax.set_ylim(lower - padding if lower < 0 else 0, upper + padding)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(alpha=.15, linewidth=.5)
        fig.suptitle("Predicciones congeladas y objetivo observado\n"
                     "16 sistemas · 608 fechas comunes · escala original de σ₇", fontsize=16, y=.985)
        fig.legend(handles=[Line2D([0], [0], color="#242424", linewidth=1.2,
                                  label="Objetivo observado"),
                            Line2D([0], [0], color="#2276a2", linewidth=1.2,
                                   label="Predicciones individuales (6 semillas o 1 determinista)")],
                   loc="upper center", bbox_to_anchor=(.5, .93), ncol=2, frameon=False, fontsize=10)
        fig.supxlabel("Fecha del resultado · calendario diario UTC", fontsize=11, y=.056)
        fig.supylabel("σ₇ · unidades de retorno logarítmico, sin anualizar", fontsize=11, x=.012)
        fig.text(.5, .018, "Cada trazo azul es un predictor individual: no se promedian ni seleccionan semillas.\n"
                 "Visualización descriptiva posterior del periodo heredado de E1; sin nuevas inferencias ni puntuaciones.",
                 ha="center", fontsize=9)
        fig.subplots_adjust(left=.055, right=.99, bottom=.105, top=.865, hspace=.35, wspace=.10)
        svg, png = io.BytesIO(), io.BytesIO()
        fig.savefig(svg, format="svg", metadata={"Date": None, "Creator": VERSION})
        fig.savefig(png, format="png", dpi=120, metadata={"Software": VERSION})
        plt.close(fig)
    svg_bytes = ("\n".join(line.rstrip() for line in svg.getvalue().decode().splitlines()) + "\n").encode()
    outputs = {"prediction_trajectories.svg": svg_bytes, "prediction_trajectories.png": png.getvalue()}
    metadata = {
        "version": VERSION,
        "scope": "posthoc_saved_vector_visualization_no_retuning",
        "input_sha256": SOURCES,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "calendar": {"first": expected_dates[0], "last": expected_dates[-1],
                     "dates": 608, "common_mask_true": 608, "timezone": "UTC"},
        "systems_in_protocol_order": systems,
        "systems": 16, "neural_systems": 13, "deterministic_systems": 3,
        "frozen_predictors": 81, "neural_seed_ids": seeds,
        "plotted_target_count_per_panel": 608,
        "plotted_prediction_points": 608 * 81,
        "predictor_mapping": mapping,
        "drawing": {"panels": [4, 4], "shared_axes": True,
                    "target": "unchanged observed sigma7 in each panel, black",
                    "predictions": "every individual saved predictor, blue; six neural seeds or one deterministic trajectory",
                    "global_y_extent_uses": "all targets and predictions, no clipping of extremes",
                    "full_calendar": True, "smoothing": False,
                    "ensemble_or_seed_mean": False, "seed_selection": False,
                    "score_based_system_order": False, "SVG_Date_omitted": True,
                    "SVG_hashsalt": VERSION},
        "access": {"cached_observed_targets_read": True, "cached_predictions_read": True,
                   "checkpoint_identity_metadata_read": True, "checkpoint_weights_loaded": False,
                   "raw_archives_opened": 0, "new_model_fits": 0,
                   "new_inference_calls": 0, "predictive_scoring": False,
                   "new_inferential_tests": 0, "hyperparameters_changed": False},
        "limits": ["The already known E1 evaluation calendar is not a new prospective confirmation.",
                   "A visual overlap is not a test of fit, equivalence, calibration or absence of overfitting.",
                   "The six neural trajectories are initialization variants on one shared calendar, not independent markets.",
                   "This figure does not supply variance-component inference or a formal delta_g test."],
        "output_sha256": {name: digest(raw) for name, raw in outputs.items()},
    }
    outputs["prediction_trajectories.json"] = (
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/e2-saved-prediction-trajectories"))
    args = parser.parse_args(argv)
    require(not args.output.is_symlink(), "Output directory cannot be a symlink")
    output = args.output.resolve()
    outputs = build()
    for name, raw in outputs.items():
        path = output / name
        require(not path.is_symlink(), "Output file cannot be a symlink")
        require(not path.exists() or path.is_file() and path.read_bytes() == raw,
                "Refusing to overwrite changed output; choose a new directory: " + name)
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in outputs.items():
        path = output / name
        if not path.exists():
            path.write_bytes(raw)
    print(json.dumps({"version": VERSION, "systems": 16, "predictors": 81,
                      "calendar_dates": 608, "new_model_fits": 0,
                      "new_inferences": 0, "output_files": list(outputs)}))


if __name__ == "__main__":
    main()
